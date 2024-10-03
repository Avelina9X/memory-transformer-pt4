"""
Module containing the training loop components for training LSWTransformer models.
"""

from dataclasses import dataclass
import time
import gc
import os
import typing

import tqdm
import numpy as np
from transformers import PreTrainedTokenizerBase
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer # type: ignore
from torch.optim import AdamW
from torcheval import metrics
from torcheval.metrics.toolkit import sync_and_compute

from constants import PARAMETERS_AS_BUCKET_VIEW, TORCH_COMPILE_OPTIONS
from model.configuration import LSWTConfigTraining, LSWTConfigTrainingDPH, LSWTConfigTrainingSteer
from model.modeling import DPHOutput, LSWTForCausalLM, LSWTForDPH

from optimizer.laprop import LaProp
from optimizer.ortho import Ortho
from .data_instruct.task_loader import DPHMultiTaskLoader, SteerTaskLoader
from .data import PileDataset
from .losses import DPOLoss, DPHLoss, KLPairsLoss, MLELoss, ORPOLoss, SimCTGLoss, AccuracyMetric

PILE_PATH_PATTERN = os.environ[ 'PILE_PATH_PATTERN' ]
PILE_SHARDS = int( os.environ[ 'PILE_SHARDS' ] )


class Trainer(): # pylint: disable=R0902
    """ Base class for continuous cached online pre-training.
    """

    def __init__(
        self,
        train_config: LSWTConfigTraining,
        model: LSWTForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        dataset: str | None
    ):
        """ Creates a Trainer instance for the entire pretraining pipeline.

        The constructor takes care of the following setup steps:
        - instantiating the optimizer specified in `train_config`
        - loading the pretraining dataset specified in the `dataset` argument
        - instantiating the loss function specified in `train_config`
        - instantiating the accuracy metric
        - setting up FP16 gradient scaling
        - setting up running mean accumulators for loss and accuracy
        - creating the KV cache for CPU offloading

        Args:
            train_config (LSWTConfigTraining): configuration object for the training pipeline.
            model (LSWTForCausalLM): the initialised model
            tokenizer (PreTrainedTokenizerBase): the initialised tokenizer
            dataset (str): dataset for pretraining
        """

        self.train_config = train_config
        self.model = model
        self.tokenizer = tokenizer

        self.optimizer = self._load_optimizer()
        self.optimizer_scaler = torch.cuda.amp.GradScaler() # type: ignore

        self.orthogonalize = self._load_ortho()

        self.batch_groups = train_config.batch_size // train_config.batch_size_step

        self.past_key_values_list = self._load_cache()
        self.data_loader_train = self._load_dataset( dataset )
        self.loss_function = self._load_loss_function()

        self.acc_function = AccuracyMetric( model.config.vocab_size, -100 )

        self.metrics = {
            'loss': metrics.Mean().to( 'cuda' ),
            'acc': metrics.Mean().to( 'cuda' ),
        }

        self.sequence_counter = metrics.Sum().to( 'cuda' )

        self.optimizer_step = 0


    """ ========================================================================
        Internal Utility functions
        ======================================================================== """

    def _bar_format( self, iter_n, iter_total, elapsed, epoch, loss, acc ) -> str:
        return tqdm.tqdm.format_meter(
            n=iter_n,
            total=iter_total,
            elapsed=elapsed,
            ncols=80,
            unit='it',
            bar_format='{desc}: {percentage:.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]',
            postfix=f'loss={loss:.3f}, acc={acc:.3f}',
            prefix=f'Epoch {epoch}',
        )

    def _load_cache( self ):
        batch_groups = self.train_config.batch_size // self.train_config.batch_size_step
        return [ None for _ in range( batch_groups ) ]

    def _load_optimizer( self ) -> torch.optim.Optimizer:
        params = self.model.get_param_groups( self.train_config.opt_decay_mask )

        if self.train_config.optimizer == 'AdamW':
            assert not self.train_config.opt_decay_init, 'Decay init not implemented'
            return AdamW(
                params=params,
                lr=0.0,
                betas=( self.train_config.opt_beta_1, self.train_config.opt_beta_2 ),
                eps=self.train_config.opt_eps,
                weight_decay=( self.train_config.opt_weight_decay ),
                fused=True,
            )

        if self.train_config.optimizer == 'LaProp':
            return LaProp(
                params=params,
                lr=0.0,
                betas=( self.train_config.opt_beta_1, self.train_config.opt_beta_2 ),
                eps=self.train_config.opt_eps,
                weight_decay=( self.train_config.opt_weight_decay ),
                decay_init=self.train_config.opt_decay_init,
            )

        raise ValueError( 'Invalid optimizer' )

    def _load_ortho( self ):
        params = [
            p for name, p in self.model.named_parameters()
            if any( i in name for i in self.train_config.ortho_params ) and p.requires_grad
        ]
        
        if len( params ) == 0:
            return None

        return Ortho( params, self.train_config.ortho_beta, self.train_config.ortho_norm_p )

    def _load_loss_function( self ) -> torch.nn.Module:
        if self.train_config.loss_objective == 'MLE':
            return MLELoss(
                self.model.config.vocab_size,
                self.model.config.pad_token_id
            )

        if self.train_config.loss_objective == 'SimCTG':
            return SimCTGLoss(
                self.train_config.loss_sim_margin,
                self.model.config.vocab_size,
                self.model.config.pad_token_id
            )

        raise ValueError( 'Invalid loss function' )

    def _load_dataset( self, dataset ):

        if dataset == 'pile':
            return PileDataset(
                tokenizer=self.tokenizer,
                seq_length=self.train_config.length_sequence,
                batch_size=self.train_config.batch_size,
                dir_pattern=PILE_PATH_PATTERN,
                pile_shards=list( range( PILE_SHARDS ) )
            ).as_data_loader()

        if dataset is None:
            return None

        raise ValueError( 'Invalid dataset choice' )


    """ ========================================================================
        Utility functions
        ======================================================================== """

    def reset_metrics( self ) -> dict[ str, float ]:
        """ Computes the end of epoch metrics and resets them for the next epoch.

        Returns:
            dict[ str, float ]: dictionary of { metric_name : computed_mean }
        """

        stats = {}
        for name, metric in self.metrics.items():
            stats[name] = float( metric.compute() )
            metric.reset()
        return stats

    def get_schedule( self ) -> float:
        """ Returns the learning rate percentage based on the Chinchilla schedule.

        Note, you must manually scale by the max learning rate.

        Returns:
            float: LR ratio in range [0.0, 1.0]
        """

        warmup_ratio = min( self.optimizer_step / self.train_config.lr_warmup_steps, 1.0 )

        tokens_seen = self.optimizer_step * self.train_config.batch_size * self.train_config.length_sequence
        warmup_tokens = self.train_config.lr_warmup_steps * self.train_config.batch_size * self.train_config.length_sequence

        cooldown_ratio = min( max( tokens_seen - warmup_tokens, 0.0 ) / ( self.train_config.lr_cooldown_tokens - warmup_tokens ), 1.0 )
        cooldown_ratio = np.cos( cooldown_ratio * np.pi ) * 0.5 + 0.5
        cooldown_ratio = cooldown_ratio * ( 1.0 - self.train_config.lr_cooldown_ratio ) + self.train_config.lr_cooldown_ratio

        return min( warmup_ratio, cooldown_ratio )

    def get_sequence_count( self ) -> int:
        return int( self.sequence_counter.compute() )

    def get_total_epochs( self ) -> int:
        """ Compute the total number of epochs based on the number of total tokens.

        Returns:
            int: total epochs
        """

        tokens_per_epoch = self.train_config.batch_size * self.train_config.length_sequence * self.train_config.batches_per_epoch
        return int( np.ceil( self.train_config.lr_cooldown_tokens / tokens_per_epoch ) )


    """ ========================================================================
        Forward Pass
        ======================================================================== """

    def forward_pass( self, tokens, past_key_values, cache_length ):

        torch._inductor.cudagraph_mark_step_begin() # type: ignore # pylint: disable=W0212

        outputs = self.model(
            input_ids=tokens,
            past_key_values=past_key_values,
            use_cache=cache_length > 0,
            max_key_values=cache_length,
        )

        past_key_values = outputs.past_key_values
        logits = outputs.logits
        last_hidden_state = outputs.hidden_states[-1]

        del outputs

        return logits, past_key_values, last_hidden_state


    """ ========================================================================
        Training Functions
        ======================================================================== """

    @torch.compile( **TORCH_COMPILE_OPTIONS )
    def train_sub_step( self, tokens_x, tokens_y, past_key_values ):
        with torch.autocast( device_type='cuda', dtype=torch.bfloat16 if self.model.config.use_bfloat16 else torch.float16 ): # type: ignore
            logits, past_key_values, hidden_states = self.forward_pass( tokens_x, past_key_values, self.train_config.length_cache )

            y_pred = logits
            y_true = tokens_y

            mle_loss, aux_loss = self.loss_function( hidden_states, y_pred, tokens_x, y_true )
            accu_loss = ( mle_loss + aux_loss ) / self.batch_groups
        self.optimizer_scaler.scale( accu_loss ).backward() # type: ignore

        logits.detach()
        hidden_states.detach()
        accu_loss.detach()
        y_pred.detach()
        aux_loss.detach()
        accu_loss.detach()

        accuracy = self.acc_function( logits, y_true ).detach()

        return mle_loss.detach(), accuracy, past_key_values

    def train_optim_step( self ):
        self.optimizer_step += 1

        reg_loss = self.orthogonalize.compute_loss() if self.orthogonalize else None
        if reg_loss is not None:
            self.optimizer_scaler.scale( reg_loss ).backward()

        for p_group in self.optimizer.param_groups:
            p_group[ 'lr' ] = self.get_schedule() * self.train_config.lr_max

        if self.train_config.opt_max_grad_norm > 0.0:
            self.optimizer_scaler.unscale_( self.optimizer )
            torch.nn.utils.clip_grad_norm_( self.model.parameters(), self.train_config.opt_max_grad_norm ) # type: ignore

        self.optimizer_scaler.step( self.optimizer )
        self.optimizer_scaler.update()
        self.optimizer.zero_grad()


    def train_batch_step( self, batch ):
        self.model.train()

        tokens_xs, tokens_ys = batch

        self.sequence_counter.update( ( tokens_xs == self.tokenizer.bos_token_id ).sum() )

        tokens_xs = torch.split( tokens_xs.to( device='cuda', non_blocking=True ), self.train_config.batch_size_step )
        tokens_ys = torch.split( tokens_ys.to( device='cuda', non_blocking=True ), self.train_config.batch_size_step )

        for idx in range( self.batch_groups ):
            past_key_values = self.model.cache_to( self.past_key_values_list[idx], 'cuda', non_blocking=True )
            self.past_key_values_list[idx] = None # type: ignore

            loss, accuracy, past_key_values = self.train_sub_step( tokens_xs[idx], tokens_ys[idx], past_key_values )

            past_key_values = self.model.cache_to( past_key_values, 'cpu', non_blocking=True )
            self.past_key_values_list[idx] = past_key_values # type: ignore

            self.metrics[ 'loss' ].update( loss )
            self.metrics[ 'acc' ].update( accuracy )

        self.train_optim_step()

    def train_epoch( self, iterator, epoch ):
        start_time = time.time()

        for batch in range( self.train_config.batches_per_epoch ):
            self.train_batch_step( next( iterator ) )

            bar = self._bar_format(
                iter_n=batch + 1,
                iter_total=self.train_config.batches_per_epoch,
                elapsed=time.time() - start_time,
                epoch=epoch,
                loss=self.metrics[ 'loss' ].compute(),
                acc=self.metrics[ 'acc' ].compute(),
            )

            print( '\r' + bar, end='', flush=True )
        print()

        torch.cuda.empty_cache()
        gc.collect()

        return self.reset_metrics()

class TrainerDDP( Trainer ):
    """ Distributed training class for continuous cached online pre-training.
    """

    def __init__(
        self,
        train_config: LSWTConfigTraining,
        model: LSWTForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        dataset: str,
        ddp_rank: int,
        ddp_world_size: int,
        dataset_shards: int = 30,
    ):
        """ Creates a Distributed Trainer instance for the entire pretraining pipeline.

        The constructor takes care of the following setup steps:
        - instantiating the optimizer specified in `train_config` with the Zero policy
        - loading the pretraining dataset specified in the `dataset` argument with DDP sharding
        - instantiating the loss function specified in `train_config`
        - instantiating the accuracy metric
        - setting up FP16 gradient scaling
        - setting up running mean accumulators for loss and accuracy
        - creating the KV cache for CPU offloading

        Args:
            train_config (LSWTConfigTraining): configuration object for the training pipeline.
            model (LSWTForCausalLM): the initialised model
            tokenizer (PreTrainedTokenizerBase): the initialised tokenizer
            dataset (str): dataset for pretraining
            ddp_rank (int): the rank of the current process
            ddp_world_size (int): the total number of processes in the group
            dataset_shards (int): the number of dataset shards, defaults to 30, but 24 recommended
        """

        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.dataset_shards = dataset_shards

        super().__init__( train_config, model, tokenizer, dataset )

        # Modify batch_groups for DDP
        self.batch_groups = train_config.batch_size // ( train_config.batch_size_step * self.ddp_world_size )


    """ ========================================================================
        Overridden Internal Utility functions
        ======================================================================== """

    def _load_optimizer( self ) -> torch.optim.Optimizer:
        params = self.model.get_param_groups( self.train_config.opt_decay_mask )

        if self.train_config.optimizer == 'LaProp':
            return ZeroRedundancyOptimizer(
                params=params,
                optimizer_class=LaProp,
                parameters_as_bucket_view=PARAMETERS_AS_BUCKET_VIEW,
                lr=0.0,
                betas=( self.train_config.opt_beta_1, self.train_config.opt_beta_2 ),
                eps=self.train_config.opt_eps,
                weight_decay=( self.train_config.opt_weight_decay ),
                decay_init=self.train_config.opt_decay_init,
            )

        raise ValueError( 'Invalid optimizer' )

    def _load_cache( self ):
        batch_groups = self.train_config.batch_size // ( self.train_config.batch_size_step * self.ddp_world_size )
        return [ None for _ in range( batch_groups ) ]

    def _load_dataset( self, dataset ):
        if dataset == 'pile':
            return PileDataset(
                tokenizer=self.tokenizer,
                seq_length=self.train_config.length_sequence,
                batch_size=self.train_config.batch_size // self.ddp_world_size,
                dir_pattern=PILE_PATH_PATTERN,
                pile_shards=list( range( self.ddp_rank, self.dataset_shards, self.ddp_world_size ) )
            ).as_data_loader()

        if dataset is None:
            return None

        raise ValueError( 'Invalid dataset choice' )


    """ ========================================================================
        Overridden Utility functions
        ======================================================================== """

    def reset_metrics( self ) -> dict[ str, float ]:
        stats = {}
        for name, metric in self.metrics.items():
            # Syncronise and compute metrics from all devices
            stats[name] = float( sync_and_compute( metric ) )

            # Ensure all devices wait for barrier before resetting
            dist.barrier()
            metric.reset()
        return stats

    def get_sequence_count( self ) -> int:
        return int( sync_and_compute( self.sequence_counter ) )


    """ ========================================================================
        Overridden Training Functions
        ======================================================================== """

    def train_batch_step( self, batch ):
        self.model.train()

        tokens_xs, tokens_ys = batch

        self.sequence_counter.update( ( tokens_xs == self.tokenizer.bos_token_id ).sum() )

        tokens_xs = torch.split( tokens_xs.to( device='cuda', non_blocking=True ), self.train_config.batch_size_step )
        tokens_ys = torch.split( tokens_ys.to( device='cuda', non_blocking=True ), self.train_config.batch_size_step )

        for idx in range( self.batch_groups ):
            self.model.require_backward_grad_sync = ( idx == self.batch_groups - 1 ) # type: ignore

            past_key_values = self.model.cache_to( self.past_key_values_list[idx], 'cuda', non_blocking=True )
            self.past_key_values_list[idx] = None # type: ignore

            loss, accuracy, past_key_values = self.train_sub_step( tokens_xs[idx], tokens_ys[idx], past_key_values )

            past_key_values = self.model.cache_to( past_key_values, 'cpu', non_blocking=True )
            self.past_key_values_list[idx] = past_key_values # type: ignore

            self.metrics[ 'loss' ].update( loss )
            self.metrics[ 'acc' ].update( accuracy )

        self.train_optim_step()

        if self.optimizer_step <= 3:
            torch.cuda.empty_cache()

    def train_epoch( self, iterator, epoch ):
        # dist.barrier()

        start_time = time.time()

        for batch in range( self.train_config.batches_per_epoch ):
            self.train_batch_step( next( iterator ) )

            bar = self._bar_format(
                iter_n=batch + 1,
                iter_total=self.train_config.batches_per_epoch,
                elapsed=time.time() - start_time,
                epoch=epoch,
                loss=float( sync_and_compute( self.metrics[ 'loss' ] ) ),
                acc=float( sync_and_compute( self.metrics[ 'acc' ] ) ),
            )

            if self.ddp_rank == 0:
                print( '\r' + bar, end='', flush=True )
        if self.ddp_rank == 0:
            print()

        torch.cuda.empty_cache()
        gc.collect()

        return self.reset_metrics()


class DPHTrainer():
    def __init__(
        self,
        train_config: LSWTConfigTraining,
        dph_config: LSWTConfigTrainingDPH,
        model_ref: LSWTForCausalLM,
        model_dph: LSWTForDPH,
        reward_head_key: str,
        tokenizer: PreTrainedTokenizerBase,
        dataset: DPHMultiTaskLoader,
    ):
        self.train_config = train_config
        self.dph_config = dph_config

        self.model_ref = model_ref
        self.model_dph = model_dph
        self.reward_head_key = reward_head_key

        self.tokenizer = tokenizer

        self.optimizer = self._load_optimizer()
        self.optimizer_scaler = torch.cuda.amp.GradScaler() # type: ignore

        self.orthogonalize = self._load_ortho()

        self.batch_groups = train_config.batch_size // train_config.batch_size_step

        self.dpo_loss = DPOLoss(
            beta=dph_config.dpo_beta,
            label_smoothing=dph_config.dpo_epsilon,
            average_logprobs=dph_config.dpo_average_logprobs
        )

        self.orpo_loss = ORPOLoss(
            alpha_orpo=dph_config.orpo_alpha_orpo,
            alpha_mle=dph_config.orpo_alpha_mle,
            vocab_size=model_dph.config.vocab_size,
        )

        self.dph_loss = DPHLoss(
            label_smoothing=dph_config.dph_epsilon,
            contrastive=dph_config.dph_contrastive,
            penalty=dph_config.dph_penalty,
        )

        self.kl_loss = KLPairsLoss(
            pn_ratio=dph_config.kl_pn_ratio,
            penalty=dph_config.kl_penalty,
        )

        self.dataset = dataset

        self.metrics = {
            'loss_dph': metrics.Mean().to( 'cuda' ),

            'dph/chosen': metrics.Mean().to( 'cuda' ),
            'dph/rejected': metrics.Mean().to( 'cuda' ),
            'dph/accuracy': metrics.Mean().to( 'cuda' ),
            'dph/margin': metrics.Mean().to( 'cuda' ),
        }

        if dph_config.dpo_enabled:
            self.metrics.update( {
                'loss_dpo': metrics.Mean().to( 'cuda' ),

                'dpo/chosen': metrics.Mean().to( 'cuda' ),
                'dpo/rejected': metrics.Mean().to( 'cuda' ),
                'dpo/accuracy': metrics.Mean().to( 'cuda' ),
                'dpo/margin': metrics.Mean().to( 'cuda' ),
            } )

        if dph_config.orpo_enabled:
            self.metrics.update( {
                'loss_orpo': metrics.Mean().to( 'cuda' ),

                'orpo/pos_mean': metrics.Mean().to( 'cuda' ),
                'orpo/neg_mean': metrics.Mean().to( 'cuda' ),
                'orpo/log_odds_ratio': metrics.Mean().to( 'cuda' ),
                'orpo/log_odds': metrics.Mean().to( 'cuda' ),
                'orpo/accuracy': metrics.Mean().to( 'cuda' ),
            } )

        if dph_config.kl_enabled:
            self.metrics.update( {
                'loss_kl': metrics.Mean().to( 'cuda' ),

                'kl/pos': metrics.Mean().to( 'cuda' ),
                'kl/neg': metrics.Mean().to( 'cuda' ),
            } )

        self.optimizer_step = 0


    """ ========================================================================
        Internal Utility functions
        ======================================================================== """

    def _bar_format( self, iter_n, iter_total, elapsed, epoch ) -> str:
        postfix = 'dph={0:.3f}, dph_acc={1:.3f}'.format(
            self.metrics[ 'loss_dph' ].compute(),
            self.metrics[ 'dph/accuracy' ].compute(),
        )

        if self.dph_config.dpo_enabled:
            postfix += ' | dpo={0:.3f}, dpo_acc={1:.3f}'.format(
                self.metrics[ 'loss_dpo' ].compute(),
                self.metrics[ 'dpo/accuracy' ].compute(),
            )

        if self.dph_config.orpo_enabled:
            postfix += ' | orpo={0:.3f}, orpo_acc={1:.3f}'.format(
                self.metrics[ 'loss_orpo' ].compute(),
                self.metrics[ 'orpo/accuracy' ].compute(),
            )

        if self.dph_config.kl_enabled:
            postfix += ' | kl={0:.3f}'.format(
                self.metrics[ 'loss_kl' ].compute(),
            )

        return tqdm.tqdm.format_meter(
            n=iter_n,
            total=iter_total,
            elapsed=elapsed,
            ncols=100,
            unit='it',
            bar_format='{desc}: {percentage:.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]',
            postfix=postfix,
            prefix=f'Epoch {epoch}',
        )

    def _load_optimizer( self ) -> torch.optim.Optimizer:
        params = self.model_dph.get_param_groups(
            self.train_config.opt_decay_mask,
            self.dph_config.dph_decay_mask,
            self.dph_config.dph_decay_init,
            self.dph_config.dph_weight_decay,
            self.dph_config.dph_lr_multiplier,
        )

        if self.train_config.optimizer == 'LaProp':
            return LaProp(
                params=params,
                lr=0.0,
                betas=( self.train_config.opt_beta_1, self.train_config.opt_beta_2 ),
                eps=self.train_config.opt_eps,
                weight_decay=( self.train_config.opt_weight_decay ),
                decay_init=self.train_config.opt_decay_init,
            )

        raise ValueError( 'Invalid optimizer' )

    def _load_ortho( self ):
        params = [
            p for name, p in self.model_dph.named_parameters()
            if any( i in name for i in self.train_config.ortho_params ) and p.requires_grad
        ]
        
        if len( params ) == 0:
            return None

        return Ortho( params, self.train_config.ortho_beta, self.train_config.ortho_norm_p )


    """ ========================================================================
        Utility functions
        ======================================================================== """

    def reset_metrics( self ) -> dict[ str, float ]:
        """ Computes the end of epoch metrics and resets them for the next epoch.

        Returns:
            dict[ str, float ]: dictionary of { metric_name : computed_mean }
        """

        stats = {}
        for name, metric in self.metrics.items():
            stats[name] = float( metric.compute() )
            metric.reset()
        return stats

    def get_schedule( self ) -> float:
        """ Returns the learning rate percentage based on the Chinchilla schedule.

        Note, you must manually scale by the max learning rate.

        Returns:
            float: LR ratio in range [0.0, 1.0]
        """

        warmup_ratio = min( self.optimizer_step / self.train_config.lr_warmup_steps, 1.0 )

        tokens_seen = self.optimizer_step * self.train_config.batch_size * self.train_config.length_sequence
        warmup_tokens = self.train_config.lr_warmup_steps * self.train_config.batch_size * self.train_config.length_sequence

        cooldown_ratio = min( max( tokens_seen - warmup_tokens, 0.0 ) / ( self.train_config.lr_cooldown_tokens - warmup_tokens ), 1.0 )
        cooldown_ratio = np.cos( cooldown_ratio * np.pi ) * 0.5 + 0.5
        cooldown_ratio = cooldown_ratio * ( 1.0 - self.train_config.lr_cooldown_ratio ) + self.train_config.lr_cooldown_ratio

        return min( warmup_ratio, cooldown_ratio )

    def get_total_epochs( self ) -> int:
        """ Compute the total number of epochs based on the number of total tokens.

        Returns:
            int: total epochs
        """

        tokens_per_epoch = self.train_config.batch_size * self.train_config.length_sequence * self.train_config.batches_per_epoch
        return int( np.ceil( self.train_config.lr_cooldown_tokens / tokens_per_epoch ) )


    """ ========================================================================
        Forward Pass
        ======================================================================== """

    @dataclass
    class ForwardPassOutputs:
        policy_pos_logits: torch.Tensor
        policy_neg_logits: torch.Tensor
        reference_pos_logits: torch.Tensor | None
        reference_neg_logits: torch.Tensor | None
        reward_pos_logits: torch.Tensor
        reward_neg_logits: torch.Tensor

    def forward_pass( self, pos_tokens: torch.LongTensor, neg_tokens: torch.LongTensor ) -> ForwardPassOutputs:

        # Mark start of forward pass (may not be needed as we aren't using graphs)
        torch._inductor.cudagraph_mark_step_begin() # type: ignore # pylint: disable=W0212

        # Combine the positive and negative tokens
        tokens_combined = typing.cast( torch.LongTensor, torch.cat( [ pos_tokens, neg_tokens ], dim=0 ) )

        # Compute outputs for positive and negative sequences
        dph_outputs = self.model_dph(
            input_ids=tokens_combined,
            past_key_values=None,
            use_cache=False,
        )

        # Assert that there is a CLS token ID set
        assert self.tokenizer.sep_token_id is not None
        assert self.tokenizer.cls_token_id is not None

        # Get the logits and states
        dph_logits = dph_outputs.logits
        dph_states = self.model_dph.pooler.aggregate_states( dph_outputs.hidden_states, tokens_combined, self.tokenizer.sep_token_id, self.tokenizer.cls_token_id, return_all=False )

        # Chunk the logits and states into positive and negative respectively
        assert isinstance( dph_states, torch.Tensor )
        dph_pos_logits, dph_neg_logits = dph_logits.chunk( 2, dim=0 )
        # dph_pos_states, dph_neg_states = dph_states.chunk( 2, dim=0 )


        # Compute the positive and negative rewards (honestly could be batched and then chunked)
        # pos_rewards = self.model_dph.pooler.forward( dph_pos_states, False, False )
        # neg_rewards = self.model_dph.pooler.forward( dph_neg_states, False, False )

        both_rewards: DPHOutput = self.model_dph.pooler( dph_states, output_embeddings=False, return_final=True )
        pos_rewards, neg_rewards = both_rewards.rewards[ self.reward_head_key ].chunk( 2, dim=0 )

        with torch.no_grad():
            # Compute reference logits if we need a reference model (e.g. for DPO or KL)
            if self.dph_config.requires_reference_model:
                ref_outputs = self.model_ref(
                    input_ids=tokens_combined,
                    past_key_values=None,
                    use_cache=False,
                )

                # Get reference logits and chunk into positive and negative
                ref_logits = ref_outputs.logits
                ref_pos_logits, ref_neg_logits = ref_logits.chunk( 2, dim=0 )
            else:
                ref_pos_logits = None
                ref_neg_logits = None

        # Return output structure
        return self.ForwardPassOutputs(
            policy_pos_logits=dph_pos_logits,
            policy_neg_logits=dph_neg_logits,
            reference_pos_logits=ref_pos_logits,
            reference_neg_logits=ref_neg_logits,
            reward_pos_logits=pos_rewards,
            reward_neg_logits=neg_rewards,
        )


    """ ========================================================================
        Training Functions
        ======================================================================== """

    @torch.compile( **TORCH_COMPILE_OPTIONS )
    def train_sub_step( self, pos_tokens, pos_target, neg_tokens, neg_target ):

        # Set autocast context # TODO: support bf16 in addition to fp16
        with torch.autocast( device_type='cuda', dtype=torch.bfloat16 if self.model_dph.config.use_bfloat16 else torch.float16 ):
            # Perform forward pass to get all relevant outputs
            outputs = self.forward_pass( pos_tokens, neg_tokens )

            # We always perform DPH loss no matter what
            dph_loss, dph_metrics = self.dph_loss(
                pos_logits=outputs.reward_pos_logits,
                neg_logits=outputs.reward_neg_logits
            )

            # If DPO is enabled compute loss from policy+reference models
            if self.dph_config.dpo_enabled:
                dpo_loss, dpo_metrics = self.dpo_loss(
                    policy_pos_logits=outputs.policy_pos_logits,
                    policy_neg_logits=outputs.policy_neg_logits,
                    reference_pos_logits=outputs.reference_pos_logits,
                    reference_neg_logits=outputs.reference_neg_logits,
                    pos_labels=pos_target,
                    neg_labels=neg_target,
                )
            else:
                dpo_loss = torch.zeros_like( dph_loss )
                dpo_metrics = {}

            # If ORPO is enabled compute loss from policy model only
            if self.dph_config.orpo_enabled:
                orpo_loss, orpo_metrics = self.orpo_loss(
                    policy_pos_logits=outputs.policy_pos_logits,
                    policy_neg_logits=outputs.policy_neg_logits,
                    pos_labels=pos_target,
                    neg_labels=neg_target,
                )
            else:
                orpo_loss = torch.zeros_like( dph_loss )
                orpo_metrics = {}

            # If KL is enabled compute loss from policy+reference models
            if self.dph_config.kl_enabled:
                kl_loss, kl_metrics = self.kl_loss(
                    policy_pos_logits=outputs.policy_pos_logits,
                    policy_neg_logits=outputs.policy_neg_logits,
                    reference_pos_logits=outputs.reference_pos_logits,
                    reference_neg_logits=outputs.reference_neg_logits,
                    pos_labels=pos_target,
                    neg_labels=neg_target,
                )
            else:
                kl_loss = torch.zeros_like( dph_loss )
                kl_metrics = {}

            # Compute weighted sum of losses and divide by accumulation count
            accu_loss = (
                dph_loss * self.dph_config.dph_weight +
                dpo_loss * self.dph_config.dpo_weight +
                orpo_loss * self.dph_config.orpo_weight +
                kl_loss * self.dph_config.kl_weight
            ) / self.batch_groups

        # Scaled backwards pass
        self.optimizer_scaler.scale( accu_loss ).backward()

        # Return { losses }, { metrics }
        return {
            'dph': dph_loss.detach(),
            'dpo': dpo_loss.detach(),
            'orpo': orpo_loss.detach(),
            'kl': kl_loss.detach(),
        }, {
            **dph_metrics,
            **dpo_metrics,
            **orpo_metrics,
            **kl_metrics,
        }

    def train_optim_step( self ):

        # Increment optimizer step
        self.optimizer_step += 1

        reg_loss = self.orthogonalize.compute_loss() if self.orthogonalize else None
        if reg_loss is not None:
            self.optimizer_scaler.scale( reg_loss ).backward()

        # For all parameter groups apply LR schedule
        for p_group in self.optimizer.param_groups:
            p_group[ 'lr' ] = self.get_schedule() * self.train_config.lr_max * p_group.get( 'lr_multiplier', 1.0 )


        # If gradient norm clipping is enabled perform scaling and clipping
        if self.train_config.opt_max_grad_norm > 0.0:
            self.optimizer_scaler.unscale_( self.optimizer )
            
            if self.dph_config.opt_split_norm:
                torch.nn.utils.clip_grad_norm_( self.model_dph.parameters_split( False ), self.train_config.opt_max_grad_norm ) # type: ignore
                torch.nn.utils.clip_grad_norm_( self.model_dph.parameters_split( True ), self.train_config.opt_max_grad_norm ) # type: ignore
            else:
                torch.nn.utils.clip_grad_norm_( self.model_dph.parameters(), self.train_config.opt_max_grad_norm ) # type: ignore

        # Perform optimizer update
        self.optimizer_scaler.step( self.optimizer )
        self.optimizer_scaler.update()
        self.optimizer.zero_grad()


    def train_batch_step( self, batch ):

        # Set reference model to eval state if present
        if self.dph_config.requires_reference_model:
            self.model_ref.eval()

        # Set policy model to train state
        self.model_dph.train()

        # Unpack batch
        pos_tokens, pos_target, neg_tokens, neg_target = batch

        # Get chosen tokens and targets and split into groups
        pos_tokens = torch.split( pos_tokens.to( device='cuda', non_blocking=True ), self.train_config.batch_size_step )
        pos_target = torch.split( pos_target.to( device='cuda', non_blocking=True ), self.train_config.batch_size_step )

        # Get rejected tokens and targets and split into groups
        neg_tokens = torch.split( neg_tokens.to( device='cuda', non_blocking=True ), self.train_config.batch_size_step )
        neg_target = torch.split( neg_target.to( device='cuda', non_blocking=True ), self.train_config.batch_size_step )

        # Iterate through all groups
        for idx in range( self.batch_groups ):
            # Perform forward pass sub step on group
            losses, metrics_dict = self.train_sub_step(
                pos_tokens=pos_tokens[idx],
                pos_target=pos_target[idx],
                neg_tokens=neg_tokens[idx],
                neg_target=neg_target[idx]
            )

            # Update DPO loss if enabled
            if self.dph_config.dpo_enabled:
                self.metrics[ 'loss_dpo' ].update( losses[ 'dpo' ] )

            # Update ORPO loss if enabled
            if self.dph_config.orpo_enabled:
                self.metrics[ 'loss_orpo' ].update( losses[ 'orpo' ] )

            # Update KL loss if enabled
            if self.dph_config.kl_enabled:
                self.metrics[ 'loss_kl' ].update( losses[ 'kl' ] )

            # Update DPH loss
            self.metrics[ 'loss_dph' ].update( losses[ 'dph' ] )

            # Update all other metrics
            for name, value in metrics_dict.items():
                self.metrics[ name ].update( value )

        # Perform optimizer update
        self.train_optim_step()

    def train_epoch( self, iterator, epoch ):

        # Get time of epoch start
        start_time = time.time()

        # For all batches in epoch perform step and update progress bar
        for batch in range( self.train_config.batches_per_epoch ):
            self.train_batch_step( next( iterator ) )

            bar_string = self._bar_format(
                iter_n=batch + 1,
                iter_total=self.train_config.batches_per_epoch,
                elapsed=time.time() - start_time,
                epoch=epoch
            )

            print( '\r' + bar_string, end='', flush=True )
        print()

        # Clear the cache for validation loop
        torch.cuda.empty_cache()
        gc.collect()

        # Get and reset metrics
        return self.reset_metrics()


class DPHTrainerDDP( DPHTrainer ):
    def __init__(
        self,
        train_config: LSWTConfigTraining,
        dph_config: LSWTConfigTrainingDPH,
        model_ref: LSWTForCausalLM,
        model_dph: LSWTForDPH,
        reward_head_key: str,
        tokenizer: PreTrainedTokenizerBase,
        dataset: DPHMultiTaskLoader,
        ddp_rank: int,
        ddp_world_size: int,
    ):
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size

        super().__init__(
            train_config,
            dph_config,
            model_ref,
            model_dph,
            reward_head_key,
            tokenizer,
            dataset,
        )

        # Modify batch_groups for DDP
        self.batch_groups = train_config.batch_size // ( train_config.batch_size_step * self.ddp_world_size )

    """ ========================================================================
        Overridden Internal Utility functions
        ======================================================================== """

    def _bar_format( self, iter_n, iter_total, elapsed, epoch ) -> str:
        postfix = 'dph={0:.3f}, dph_acc={1:.3f}'.format(
            sync_and_compute( self.metrics[ 'loss_dph' ] ),
            sync_and_compute( self.metrics[ 'dph/accuracy' ] ),
        )

        if self.dph_config.dpo_enabled:
            postfix += ' | dpo={0:.3f}, dpo_acc={1:.3f}'.format(
                sync_and_compute( self.metrics[ 'loss_dpo' ] ),
                sync_and_compute( self.metrics[ 'dpo/accuracy' ] ),
            )

        if self.dph_config.orpo_enabled:
            postfix += ' | orpo={0:.3f}, orpo_acc={1:.3f}'.format(
                sync_and_compute( self.metrics[ 'loss_orpo' ] ),
                sync_and_compute( self.metrics[ 'orpo/accuracy' ] ),
            )

        if self.dph_config.kl_enabled:
            postfix += ' | kl={0:.3f}'.format(
                sync_and_compute( self.metrics[ 'loss_kl' ] ),
            )

        return tqdm.tqdm.format_meter(
            n=iter_n,
            total=iter_total,
            elapsed=elapsed,
            ncols=100,
            unit='it',
            bar_format='{desc}: {percentage:.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]',
            postfix=postfix,
            prefix=f'Epoch {epoch}',
        )

    def _load_optimizer( self ) -> torch.optim.Optimizer:
        params = self.model_dph.get_param_groups(
            self.train_config.opt_decay_mask,
            self.dph_config.dph_decay_mask,
            self.dph_config.dph_decay_init,
            self.dph_config.dph_weight_decay,
            self.dph_config.dph_lr_multiplier,
        )

        if self.train_config.optimizer == 'LaProp':
            return ZeroRedundancyOptimizer(
                params=params,
                optimizer_class=LaProp,
                parameters_as_bucket_view=PARAMETERS_AS_BUCKET_VIEW,
                lr=0.0,
                betas=( self.train_config.opt_beta_1, self.train_config.opt_beta_2 ),
                eps=self.train_config.opt_eps,
                weight_decay=( self.train_config.opt_weight_decay ),
                decay_init=self.train_config.opt_decay_init,
            )

        raise ValueError( 'Invalid optimizer' )

    """ ========================================================================
        Overridden Utility functions
        ======================================================================== """

    def reset_metrics( self ) -> dict[ str, float ]:
        stats = {}
        for name, metric in self.metrics.items():
            # Syncronise and compute metrics from all devices
            stats[name] = float( sync_and_compute( metric ) )

            # Ensure all devices wait for barrier before resetting
            dist.barrier()
            metric.reset()
        return stats

    """ ========================================================================
        Overridden Training Functions
        ======================================================================== """

    def train_batch_step( self, batch ):

        # Set reference model to eval state if present
        if self.dph_config.requires_reference_model:
            self.model_ref.eval()

        # Set policy model to train state
        self.model_dph.train()

        # Unpack batch
        pos_tokens, pos_target, neg_tokens, neg_target = batch

        # Get chosen tokens and targets and split into groups
        pos_tokens = torch.split( pos_tokens.to( device='cuda', non_blocking=True ), self.train_config.batch_size_step )
        pos_target = torch.split( pos_target.to( device='cuda', non_blocking=True ), self.train_config.batch_size_step )

        # Get rejected tokens and targets and split into groups
        neg_tokens = torch.split( neg_tokens.to( device='cuda', non_blocking=True ), self.train_config.batch_size_step )
        neg_target = torch.split( neg_target.to( device='cuda', non_blocking=True ), self.train_config.batch_size_step )

        # Iterate through all groups
        for idx in range( self.batch_groups ):
            self.model_dph.require_backward_grad_sync = ( idx == self.batch_groups - 1 ) # type: ignore

            # Perform forward pass sub step on group
            losses, metrics_dict = self.train_sub_step(
                pos_tokens=pos_tokens[idx],
                pos_target=pos_target[idx],
                neg_tokens=neg_tokens[idx],
                neg_target=neg_target[idx]
            )

            # Update DPO loss if enabled
            if self.dph_config.dpo_enabled:
                self.metrics[ 'loss_dpo' ].update( losses[ 'dpo' ] )

            # Update ORPO loss if enabled
            if self.dph_config.orpo_enabled:
                self.metrics[ 'loss_orpo' ].update( losses[ 'orpo' ] )

            # Update KL loss if enabled
            if self.dph_config.kl_enabled:
                self.metrics[ 'loss_kl' ].update( losses[ 'kl' ] )

            # Update DPH loss
            self.metrics[ 'loss_dph' ].update( losses[ 'dph' ] )

            # Update all other metrics
            for name, value in metrics_dict.items():
                self.metrics[ name ].update( value )

        # Perform optimizer update
        self.train_optim_step()

        if self.optimizer_step <= 3:
            torch.cuda.empty_cache()

    def train_epoch( self, iterator, epoch ):

        # Get time of epoch start
        start_time = time.time()

        # For all batches in epoch perform step and update progress bar
        for batch in range( self.train_config.batches_per_epoch ):
            self.train_batch_step( next( iterator ) )

            bar_string = self._bar_format(
                iter_n=batch + 1,
                iter_total=self.train_config.batches_per_epoch,
                elapsed=time.time() - start_time,
                epoch=epoch
            )

            if self.ddp_rank == 0:
                print( '\r' + bar_string, end='', flush=True )
        if self.ddp_rank == 0:
            print()

        # Clear the cache for validation loop
        torch.cuda.empty_cache()
        gc.collect()

        # Get and reset metrics
        return self.reset_metrics()

class SteerTrainer():
    def __init__(
        self,
        train_config: LSWTConfigTraining,
        steer_config: LSWTConfigTrainingSteer,
        model_ref: LSWTForCausalLM,
        model_dph: LSWTForDPH,
        task_loader: SteerTaskLoader,
    ):
        self.train_config = train_config
        self.steer_config = steer_config

        self.model_ref = model_ref
        self.model_dph = model_dph

        self.task_loader = task_loader
        self.formatter = task_loader.formatter
        self.tokenizer = self.formatter.tokenizer

        self.label_keys = self.task_loader.labels

        self.optimizer = self._load_optimizer()
        self.optimizer_scaler = torch.cuda.amp.GradScaler() # type: ignore
        self.optimizer_step = 0

        self.orthogonalize = self._load_ortho()

        self.batch_groups = train_config.batch_size // train_config.batch_size_step

        self.metrics = {
            'loss_reward': metrics.Mean().to( 'cuda' )
        }

        for key in self.label_keys:
            self.metrics[ f'reward/{key}' ] = metrics.Mean().to( 'cuda' )

        if self.steer_config.sae_enabled:
            self.metrics[ 'loss_sae' ] = metrics.Mean().to( 'cuda' )
            self.metrics[ 'sae/l0' ] = metrics.Mean().to( 'cuda' )
            self.metrics[ 'sae/l1' ] = metrics.Mean().to( 'cuda' )
            self.metrics[ 'sae/l2' ] = metrics.Mean().to( 'cuda' )

        if self.steer_config.kl_enabled:
            self.metrics[ 'loss_kl' ] = metrics.Mean().to( 'cuda' )
            self.metrics[ 'kl/div' ] = metrics.Mean().to( 'cuda' )


    """ ========================================================================
        Internal Utility functions
        ======================================================================== """

    def _bar_format( self, iter_n, iter_total, elapsed, epoch ) -> str:
        postfix = 'reward={0:.3f}'.format(
            self.metrics[ 'loss_reward' ].compute()
        )

        if self.steer_config.sae_enabled:
            postfix += ' | saeL0={0:.1f}, saeL1={1:.3f}, saeL2={2:.3f}'.format(
                self.metrics[ 'sae/l0' ].compute(),
                self.metrics[ 'sae/l1' ].compute(),
                self.metrics[ 'sae/l2' ].compute(),
            )

        if self.steer_config.kl_enabled:
            postfix += ' | kl={0:.3f}'.format(
                self.metrics[ 'loss_kl' ].compute(),
            )

        return tqdm.tqdm.format_meter(
            n=iter_n,
            total=iter_total,
            elapsed=elapsed,
            ncols=100,
            unit='it',
            bar_format='{desc}: {percentage:.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]',
            postfix=postfix,
            prefix=f'Epoch {epoch}',
        )

    def _load_optimizer( self ) -> torch.optim.Optimizer:
        params = self.model_dph.get_param_groups(
            self.train_config.opt_decay_mask,
            self.steer_config.dph_decay_mask,
            self.steer_config.dph_decay_init,
            self.steer_config.dph_weight_decay,
            self.steer_config.dph_lr_multiplier,
        )

        if self.train_config.optimizer == 'LaProp':
            return LaProp(
                params=params,
                lr=0.0,
                betas=( self.train_config.opt_beta_1, self.train_config.opt_beta_2 ),
                eps=self.train_config.opt_eps,
                weight_decay=( self.train_config.opt_weight_decay ),
                decay_init=self.train_config.opt_decay_init,
            )

        raise ValueError( 'Invalid optimizer' )

    def _load_ortho( self ):
        params = [
            p for name, p in self.model_dph.named_parameters()
            if any( i in name for i in self.train_config.ortho_params ) and p.requires_grad
        ]
        
        if len( params ) == 0:
            return None
        
        return Ortho( params, self.train_config.ortho_beta, self.train_config.ortho_norm_p )

    """ ========================================================================
        Utility functions
        ======================================================================== """

    def reset_metrics( self ) -> dict[ str, float ]:
        """ Computes the end of epoch metrics and resets them for the next epoch.

        Returns:
            dict[ str, float ]: dictionary of { metric_name : computed_mean }
        """

        stats = {}
        for name, metric in self.metrics.items():
            stats[name] = float( metric.compute() )
            metric.reset()
        return stats

    def get_schedule( self ) -> float:
        """ Returns the learning rate percentage based on the Chinchilla schedule.

        Note, you must manually scale by the max learning rate.

        Returns:
            float: LR ratio in range [0.0, 1.0]
        """

        warmup_ratio = min( self.optimizer_step / self.train_config.lr_warmup_steps, 1.0 )

        tokens_seen = self.optimizer_step * self.train_config.batch_size * self.train_config.length_sequence
        warmup_tokens = self.train_config.lr_warmup_steps * self.train_config.batch_size * self.train_config.length_sequence

        cooldown_ratio = min( max( tokens_seen - warmup_tokens, 0.0 ) / ( self.train_config.lr_cooldown_tokens - warmup_tokens ), 1.0 )
        cooldown_ratio = np.cos( cooldown_ratio * np.pi ) * 0.5 + 0.5
        cooldown_ratio = cooldown_ratio * ( 1.0 - self.train_config.lr_cooldown_ratio ) + self.train_config.lr_cooldown_ratio

        return min( warmup_ratio, cooldown_ratio )

    def get_total_epochs( self ) -> int:
        """ Compute the total number of epochs based on the number of total tokens.

        Returns:
            int: total epochs
        """

        tokens_per_epoch = self.train_config.batch_size * self.train_config.length_sequence * self.train_config.batches_per_epoch
        return int( np.ceil( self.train_config.lr_cooldown_tokens / tokens_per_epoch ) )


    """ ========================================================================
        Forward Pass
        ======================================================================== """

    @dataclass
    class ForwardPassOutputs:
        policy_logits: torch.Tensor
        reference_logits: torch.Tensor | None
        dph_outputs: DPHOutput

    def forward_pass( self, tokens: torch.Tensor, selected_idx: torch.Tensor ) -> ForwardPassOutputs:

        # Mark start of forward pass (may not be needed as we aren't using graphs)
        torch._inductor.cudagraph_mark_step_begin() # type: ignore # pylint: disable=W0212

        # Compute outputs for positive and negative sequences
        dph_model_outputs = self.model_dph(
            input_ids=tokens,
            past_key_values=None,
            use_cache=False,
        )

        # Assert that there is a CLS token ID set
        assert self.tokenizer.sep_token_id is not None
        assert self.tokenizer.cls_token_id is not None

        # Get the logits and states
        dph_logits = dph_model_outputs.logits
        dph_states = self.model_dph.pooler.aggregate_states(
            dph_model_outputs.hidden_states,
            tokens,
            self.tokenizer.sep_token_id,
            self.tokenizer.cls_token_id,
            return_all=True
        )

        assert isinstance( dph_states, dict )
        states = dph_states[ 'pooled_states' ]

        # Get the individual states selected for this batch
        selected_states = states.gather( -2, selected_idx[ ..., None ].repeat( 1, 1, states.shape[-1] ) )

        # Compute the rewards and optionally the SAE outputs
        dph_outputs = self.model_dph.pooler( selected_states, output_latent_states=False, compute_sae_loss=self.steer_config.sae_enabled )

        with torch.no_grad():
            # Compute reference logits if we need a reference model (e.g. for DPO or KL)
            if self.steer_config.requires_reference_model:
                ref_outputs = self.model_ref(
                    input_ids=tokens,
                    past_key_values=None,
                    use_cache=False,
                )

                # Get reference logits and chunk into positive and negative
                ref_logits = ref_outputs.logits
            else:
                ref_logits = None

        return self.ForwardPassOutputs(
            policy_logits=dph_logits,
            reference_logits=ref_logits,
            dph_outputs=dph_outputs
        )

    """ ========================================================================
        Training Functions
        ======================================================================== """

    @torch.compile( **TORCH_COMPILE_OPTIONS )
    def train_sub_step( self, tokens: torch.Tensor, targets: torch.Tensor, selected_idx: torch.Tensor, selected_weights: torch.Tensor, y_true: torch.Tensor ):

        # Set autocast context
        with torch.autocast( device_type='cuda', dtype=torch.bfloat16 if self.model_dph.config.use_bfloat16 else torch.float16 ):
            # Perform forward pass to get all relevant outputs
            outputs = self.forward_pass( tokens, selected_idx )

            # Create the y_pred, weight and y_true tensors
            labelled_rewards = torch.cat( [ outputs.dph_outputs.rewards[key] for key in self.label_keys ], dim=-1 ) # [ Batch, Seq, Reward ]
            reward_weights = selected_weights[ ..., None ] # [ Batch, Seq, 1 ]
            true_rewards = y_true[ :, None, : ] # [ Batch, 1, Rewards ]

            # Compute the per attribute and overall reward losses
            reward_losses = ( ( labelled_rewards.float() - true_rewards ).square() * reward_weights ).sum( -2 ).mean( 0 )
            reward_loss = reward_losses.mean()

            # Compute the reward metrics
            reward_metrics = {
                f'reward/{key}': reward_losses[i].detach() for i, key in enumerate( self.label_keys )
            }

            # If SAE is enabled compute those losses and metrics
            if self.steer_config.sae_enabled:
                assert outputs.dph_outputs.aux_loss

                l1_loss = outputs.dph_outputs.aux_loss.l1_penalty
                l2_loss = outputs.dph_outputs.aux_loss.reconstruction_loss
                l0_loss = outputs.dph_outputs.aux_loss.sparsity

                sae_loss = (
                    l1_loss * self.steer_config.sae_l1_coef +
                    l2_loss * self.steer_config.sae_l2_coef
                )

                sae_metrics = {
                    'sae/l0': l0_loss.detach(),
                    'sae/l1': l1_loss.detach(),
                    'sae/l2': l2_loss.detach(),
                }
            else:
                sae_loss = torch.zeros_like( reward_loss )
                sae_metrics = {}


            # If KL is enabled compute KL
            if self.steer_config.kl_enabled:
                assert outputs.reference_logits is not None

                pol_logp = F.log_softmax( outputs.policy_logits, -1, dtype=torch.float32 )
                ref_logp = F.log_softmax( outputs.reference_logits, -1, dtype=torch.float32 )

                mask = targets != -100

                kl_div = F.kl_div( pol_logp, ref_logp, reduction='none', log_target=True ).sum( -1 )
                kl_div = ( kl_div * mask ).sum( -1 )
                kl_div = kl_div.mean()

                kl_loss = kl_div * self.steer_config.kl_penalty
                kl_metrics = {
                    'kl/div': kl_div.detach()
                }
            else:
                kl_loss = torch.zeros_like( reward_loss )
                kl_metrics = {}

            # Compute weighted sum of losses and divide by accumulation count
            accu_loss = (
                self.steer_config.dph_weight * reward_loss +
                self.steer_config.sae_weight * sae_loss +
                self.steer_config.kl_weight * kl_loss
            ) / self.batch_groups

        # Scaled backwards pass
        self.optimizer_scaler.scale( accu_loss ).backward()

        return {
            'loss_reward': reward_loss.detach(),
            'loss_sae': sae_loss.detach(),
            'loss_kl': kl_loss.detach(),
        }, {
            **reward_metrics,
            **sae_metrics,
            **kl_metrics
        }

    def train_optim_step( self ):

        # Increment optimizer step
        self.optimizer_step += 1

        # Compute ortho regularisation loss
        reg_loss = self.orthogonalize.compute_loss() if self.orthogonalize else None

        # If there are any ortho weights, do the backward pass
        if reg_loss is not None:
            self.optimizer_scaler.scale( reg_loss ).backward()

        # For all parameter groups apply LR schedule
        for p_group in self.optimizer.param_groups:
            p_group[ 'lr' ] = self.get_schedule() * self.train_config.lr_max * p_group.get( 'lr_multiplier', 1.0 )

        # If gradient norm clipping is enabled perform scaling and clipping
        if self.train_config.opt_max_grad_norm > 0.0:
            self.optimizer_scaler.unscale_( self.optimizer )
            
            if self.steer_config.opt_split_norm:
                torch.nn.utils.clip_grad_norm_( self.model_dph.parameters_split( False ), self.train_config.opt_max_grad_norm ) # type: ignore
                torch.nn.utils.clip_grad_norm_( self.model_dph.parameters_split( True ), self.train_config.opt_max_grad_norm ) # type: ignore
            else:
                torch.nn.utils.clip_grad_norm_( self.model_dph.parameters(), self.train_config.opt_max_grad_norm ) # type: ignore

        # Perform optimizer update
        self.optimizer_scaler.step( self.optimizer )
        self.optimizer_scaler.update()
        self.optimizer.zero_grad()

    def train_batch_step( self, batch ):

        # Set reference model to eval state if present
        if self.steer_config.requires_reference_model:
            self.model_ref.eval()

        # Set policy model to train state
        self.model_dph.train()

        # Unpack batch TODO: there has to be a better way to do this, right?
        tokens, targets, sel_idx, sel_wht, labels = batch

        # Split into microbatches
        tokens = torch.split( tokens.to( device='cuda', non_blocking=True ), self.train_config.batch_size_step )
        targets = torch.split( targets.to( device='cuda', non_blocking=True ), self.train_config.batch_size_step )
        sel_idx = torch.split( sel_idx.to( device='cuda', non_blocking=True ), self.train_config.batch_size_step )
        sel_wht = torch.split( sel_wht.to( device='cuda', non_blocking=True ), self.train_config.batch_size_step )
        labels = torch.split( labels.to( device='cuda', non_blocking=True ), self.train_config.batch_size_step )

        # Iterate through all microbatches
        for idx in range( self.batch_groups ):
            # Perform forward pass sub step
            losses, metrics_dict = self.train_sub_step(
                tokens=tokens[idx],
                targets=targets[idx],
                selected_idx=sel_idx[idx],
                selected_weights=sel_wht[idx],
                y_true=labels[idx],
            )

            # Update losses
            self.metrics[ 'loss_reward' ].update( losses[ 'loss_reward' ] )

            if self.steer_config.sae_enabled:
                self.metrics[ 'loss_sae' ].update( losses[ 'loss_sae' ] )

            if self.steer_config.kl_enabled:
                self.metrics[ 'loss_kl' ].update( losses[ 'loss_kl' ] )

            # Update metrics
            for key, value in metrics_dict.items():
                self.metrics[ key ].update( value )

        # Perform optimizer update
        self.train_optim_step()

    def train_epoch( self, iterator, epoch ):

        # Get time of epoch start
        start_time = time.time()

        # For all batches in epoch perform step and update progress bar
        for batch in range( self.train_config.batches_per_epoch ):
            self.train_batch_step( next( iterator ) )

            bar_string = self._bar_format(
                iter_n=batch + 1,
                iter_total=self.train_config.batches_per_epoch,
                elapsed=time.time() - start_time,
                epoch=epoch
            )

            print( '\r' + bar_string, end='', flush=True )
        print()

        # Clear the cache for validation loop
        torch.cuda.empty_cache()
        gc.collect()

        # Get and reset metrics
        return self.reset_metrics()


class SteerTrainerDDP( SteerTrainer ):
    def __init__(
        self,
        train_config: LSWTConfigTraining,
        steer_config: LSWTConfigTrainingSteer,
        model_ref: LSWTForCausalLM,
        model_dph: LSWTForDPH,
        task_loader: SteerTaskLoader,
        ddp_rank: int,
        ddp_world_size: int,
    ):
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size

        super().__init__(
            train_config,
            steer_config,
            model_ref,
            model_dph,
            task_loader,
        )

        # Modify batch_groups for DDP
        self.batch_groups = train_config.batch_size // ( train_config.batch_size_step * self.ddp_world_size )

    """ ========================================================================
        Overridden Utility functions
        ======================================================================== """

    def _bar_format( self, iter_n, iter_total, elapsed, epoch ) -> str:
        postfix = 'reward={0:.3f}'.format(
            sync_and_compute( self.metrics[ 'loss_reward' ] )
        )

        if self.steer_config.sae_enabled:
            postfix += ' | saeL0={0:.1f}, saeL1={1:.3f}, saeL2={2:.3f}'.format(
                sync_and_compute( self.metrics[ 'sae/l0' ] ),
                sync_and_compute( self.metrics[ 'sae/l1' ] ),
                sync_and_compute( self.metrics[ 'sae/l2' ] ),
            )

        if self.steer_config.kl_enabled:
            postfix += ' | kl={0:.3f}'.format(
                sync_and_compute( self.metrics[ 'loss_kl' ] ),
            )

        return tqdm.tqdm.format_meter(
            n=iter_n,
            total=iter_total,
            elapsed=elapsed,
            ncols=100,
            unit='it',
            bar_format='{desc}: {percentage:.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]',
            postfix=postfix,
            prefix=f'Epoch {epoch}',
        )

    def _load_optimizer( self ) -> torch.optim.Optimizer:
        params = self.model_dph.get_param_groups(
            self.train_config.opt_decay_mask,
            self.steer_config.dph_decay_mask,
            self.steer_config.dph_decay_init,
            self.steer_config.dph_weight_decay,
            self.steer_config.dph_lr_multiplier,
        )

        if self.train_config.optimizer == 'LaProp':
            return ZeroRedundancyOptimizer(
                params=params,
                optimizer_class=LaProp,
                parameters_as_bucket_view=PARAMETERS_AS_BUCKET_VIEW,
                lr=0.0,
                betas=( self.train_config.opt_beta_1, self.train_config.opt_beta_2 ),
                eps=self.train_config.opt_eps,
                weight_decay=( self.train_config.opt_weight_decay ),
                decay_init=self.train_config.opt_decay_init,
            )

        raise ValueError( 'Invalid optimizer' )

    """ ========================================================================
        Utility functions
        ======================================================================== """

    def reset_metrics( self ) -> dict[ str, float ]:
        stats = {}
        for name, metric in self.metrics.items():
            # Syncronise and compute metrics from all devices
            stats[name] = float( sync_and_compute( metric ) )

            # Ensure all devices wait for barrier before resetting
            dist.barrier()
            metric.reset()
        return stats

    """ ========================================================================
        Training Functions
        ======================================================================== """

    def train_batch_step( self, batch ):

        # Set reference model to eval state if present
        if self.steer_config.requires_reference_model:
            self.model_ref.eval()

        # Set policy model to train state
        self.model_dph.train()

        # Unpack batch TODO: there has to be a better way to do this, right?
        tokens, targets, sel_idx, sel_wht, labels = batch

        # Split into microbatches
        tokens = torch.split( tokens.to( device='cuda', non_blocking=True ), self.train_config.batch_size_step )
        targets = torch.split( targets.to( device='cuda', non_blocking=True ), self.train_config.batch_size_step )
        sel_idx = torch.split( sel_idx.to( device='cuda', non_blocking=True ), self.train_config.batch_size_step )
        sel_wht = torch.split( sel_wht.to( device='cuda', non_blocking=True ), self.train_config.batch_size_step )
        labels = torch.split( labels.to( device='cuda', non_blocking=True ), self.train_config.batch_size_step )

        # Iterate through all microbatches
        for idx in range( self.batch_groups ):
            self.model_dph.require_backward_grad_sync = ( idx == self.batch_groups - 1 ) # type: ignore

            # Perform forward pass sub step
            losses, metrics_dict = self.train_sub_step(
                tokens=tokens[idx],
                targets=targets[idx],
                selected_idx=sel_idx[idx],
                selected_weights=sel_wht[idx],
                y_true=labels[idx],
            )

            # Update losses
            self.metrics[ 'loss_reward' ].update( losses[ 'loss_reward' ] )

            if self.steer_config.sae_enabled:
                self.metrics[ 'loss_sae' ].update( losses[ 'loss_sae' ] )

            if self.steer_config.kl_enabled:
                self.metrics[ 'loss_kl' ].update( losses[ 'loss_kl' ] )

            # Update metrics
            for key, value in metrics_dict.items():
                self.metrics[ key ].update( value )

        # Perform optimizer update
        self.train_optim_step()

        if self.optimizer_step <= 3:
            torch.cuda.empty_cache()

    def train_epoch( self, iterator, epoch ):

        # Get time of epoch start
        start_time = time.time()

        # For all batches in epoch perform step and update progress bar
        for batch in range( self.train_config.batches_per_epoch ):
            self.train_batch_step( next( iterator ) )

            if np.isnan( sync_and_compute( self.metrics[ 'loss_reward' ] ).item() ):
                if self.ddp_rank == 0: print( 'Metrics:' )
                for key, value in self.metrics.items():
                    val = sync_and_compute( value )
                    if self.ddp_rank == 0: print( f'{key}: {val}' )
                if self.ddp_rank == 0: print()

                if self.ddp_rank == 0: print( 'Params:' )
                with torch.no_grad():
                    for name, param in self.model_dph.named_parameters():
                        if param.requires_grad:
                            nanp = torch.isnan( param ) / param.numel()
                            infp = torch.isinf( param ) / param.numel()

                            if self.ddp_rank == 0: print( f'{name}: nan={nanp}% inf={infp}%' )
                if self.ddp_rank == 0: exit( 1 )

            bar_string = self._bar_format(
                iter_n=batch + 1,
                iter_total=self.train_config.batches_per_epoch,
                elapsed=time.time() - start_time,
                epoch=epoch
            )

            if self.ddp_rank == 0:
                print( '\r' + bar_string, end='', flush=True )
        if self.ddp_rank == 0:
            print()

        # Clear the cache for validation loop
        torch.cuda.empty_cache()
        gc.collect()

        # Get and reset metrics
        return self.reset_metrics()

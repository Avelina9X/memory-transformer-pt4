"""
Module containing the training loop components for training LSWTransformer models.
"""

from dataclasses import dataclass
import time
import gc
import os

import tqdm
import numpy as np
from transformers import PreTrainedTokenizerBase
import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer # type: ignore
from torch.optim import AdamW
from torcheval import metrics
from torcheval.metrics.toolkit import sync_and_compute

from constants import TORCH_COMPILE_OPTIONS
from model.configuration import LSWTConfigTraining, LSWTConfigTrainingDPH
from model.modeling import LSWTForCausalLM, LSWTForDPH

from optimizer.minato import Minato
from optimizer.laprop import LaProp
from .data_instruct.task_loader import DPHMultiTaskLoader
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

        self.batch_groups = train_config.batch_size // train_config.batch_size_step

        self.past_key_values_list = self._load_cache()
        self.data_loader_train = self._load_dataset( dataset )
        self.loss_function = self._load_loss_function()

        self.acc_function = AccuracyMetric( model.config.vocab_size, -100 )

        self.metrics = {
            'loss': metrics.Mean().to( 'cuda' ),
            'acc': metrics.Mean().to( 'cuda' ),
        }

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
        if self.train_config.optimizer == 'Minato':
            assert not self.train_config.opt_decay_init, 'Decay init not implemented'
            return Minato(
                params=params,
                lr=0.0,
                betas=( self.train_config.opt_beta_1, self.train_config.opt_beta_2 ),
                eps=self.train_config.opt_eps,
                rho=self.train_config.opt_rho,
                weight_decay=( self.train_config.opt_weight_decay )
            )

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
        with torch.autocast( device_type='cuda', dtype=torch.float16 ): # type: ignore
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
                lr=0.0,
                betas=( self.train_config.opt_beta_1, self.train_config.opt_beta_2 ),
                eps=self.train_config.opt_eps,
                weight_decay=( self.train_config.opt_weight_decay ),
                decay_init=self.train_config.opt_decay_init,
            )

        if self.train_config.optimizer == 'Minato':
            assert not self.train_config.opt_decay_init, 'Decay init not implemented'
            return ZeroRedundancyOptimizer(
                params=params,
                optimizer_class=Minato,
                lr=0.0,
                betas=( self.train_config.opt_beta_1, self.train_config.opt_beta_2 ),
                rho=self.train_config.opt_rho,
                eps=self.train_config.opt_eps,
                weight_decay=( self.train_config.opt_weight_decay ),
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


    """ ========================================================================
        Overridden Training Functions
        ======================================================================== """

    def train_batch_step( self, batch ):
        self.model.train()

        tokens_xs, tokens_ys = batch

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
            self.dph_config.dph_decay_init,
            self.dph_config.dph_weight_decay,
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
        tokens_combined = torch.cat( [ pos_tokens, neg_tokens ], dim=0 )

        # Compute outputs for positive and negative sequences
        dph_outputs = self.model_dph(
            input_ids=tokens_combined,
            past_key_values=None,
            use_cache=False,
        )

        # Get the logits and states
        dph_logits = dph_outputs.logits
        dph_states = dph_outputs.hidden_states[self.model_dph.config.reward_select_layer]

        # Chunk the logits and states into positive and negative respectively
        dph_pos_logits, dph_neg_logits = dph_logits.chunk( 2, dim=0 )
        dph_pos_states, dph_neg_states = dph_states.chunk( 2, dim=0 )

        # Assert that there is a CLS token ID set
        assert self.tokenizer.cls_token_id is not None
        
        # Compute the positive and negative rewards (honestly could be batched and then chunked)
        pos_rewards = self.model_dph.compute_final_rewards( dph_pos_states, pos_tokens, self.tokenizer.cls_token_id )
        neg_rewards = self.model_dph.compute_final_rewards( dph_neg_states, neg_tokens, self.tokenizer.cls_token_id )

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
            reward_pos_logits=pos_rewards.rewards[ self.reward_head_key ],
            reward_neg_logits=neg_rewards.rewards[ self.reward_head_key ],
        )


    """ ========================================================================
        Training Functions
        ======================================================================== """

    @torch.compile( **TORCH_COMPILE_OPTIONS )
    def train_sub_step( self, pos_tokens, pos_target, neg_tokens, neg_target ):
        
        # Set autocast context # TODO: support bf16 in addition to fp16
        with torch.autocast( device_type='cuda', dtype=torch.float16 ):
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

        # For all parameter groups apply LR schedule
        for p_group in self.optimizer.param_groups:
            p_group[ 'lr' ] = self.get_schedule() * self.train_config.lr_max

        # If gradient norm clipping is enabled perform scaling and clipping
        if self.train_config.opt_max_grad_norm > 0.0:
            self.optimizer_scaler.unscale_( self.optimizer )
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

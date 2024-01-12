"""
Module containing the training loop components for training LSWTransformer models.
"""

import time

import tqdm
import numpy as np
from transformers import PreTrainedTokenizerBase
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torcheval import metrics
from torcheval.metrics.toolkit import sync_and_compute

from constants import TORCH_COMPILE_OPTIONS, HF_CACHE_DIR
from model.configuration import LSWTConfigTraining
from model.modeling import LSWTForCausalLM

from optimizer.minato import Minato
from optimizer.laprop import LaProp
from .data import PileDataset, OpenOrcaDataset
from .losses import MLELoss, SimCTGLoss, AccuracyMetric


class Trainer():
    def __init__( self, train_config: LSWTConfigTraining, model: LSWTForCausalLM, tokenizer: PreTrainedTokenizerBase, dataset: str ):
        self.train_config = train_config
        self.model = model
        self.tokenizer = tokenizer

        self.optimizer = self._load_optimizer()
        self.optimizer_scaler = torch.cuda.amp.GradScaler() # type: ignore

        self.batch_groups = train_config.batch_size // train_config.batch_size_step
        self.accum_groups = train_config.batch_size // train_config.batch_size_step

        self.past_key_values_list = [ None for _ in range( self.batch_groups ) ]

        self.data_loader_train = self._load_dataset( dataset )

        self.loss_function = self._load_loss_function()

        self.acc_function = AccuracyMetric( model.config.vocab_size, model.config.pad_token_id )

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

    def _load_optimizer( self ) -> torch.optim.Optimizer:        
        if self.train_config.optimizer == 'Minato':
            return Minato(
                params=self.model.get_param_groups(),
                lr=0.0,
                beta=self.train_config.opt_beta_1,
                weight_decay=( self.train_config.opt_weight_decay )
            )

        if self.train_config.optimizer == 'AdamW':
            return AdamW(
                params=self.model.get_param_groups(),
                lr=0.0,
                betas=( self.train_config.opt_beta_1, self.train_config.opt_beta_2 ),
                eps=self.train_config.opt_eps,
                weight_decay=( self.train_config.opt_weight_decay ),
                fused=True,
            )
        
        if self.train_config.optimizer == 'LaProp':
            return LaProp(
                params=self.model.get_param_groups(),
                lr=0.0,
                betas=( self.train_config.opt_beta_1, self.train_config.opt_beta_2 ),
                eps=self.train_config.opt_eps,
                weight_decay=( self.train_config.opt_weight_decay ),
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
                batch_size=self.train_config.batch_size
            ).as_data_loader()
        
        if dataset == 'openorca':
            return OpenOrcaDataset(
                tokenizer=self.tokenizer,
                seq_length=self.train_config.length_sequence,
                batch_size=self.train_config.batch_size,
                cache_dir=HF_CACHE_DIR,
            ).as_data_loader()
        
        raise ValueError( 'Invalid dataset choice' )


    """ ========================================================================
        Utility functions
        ======================================================================== """

    def reset_metrics( self ) -> dict[ str, float ]:
        stats = {}
        for name, metric in self.metrics.items():
            stats[name] = float( metric.compute() ) # TODO: torcheval.metrics.toolkit.sync_and_compute
            metric.reset()
        return stats

    def get_schedule( self ) -> float:
        warmup_ratio = min( self.optimizer_step / self.train_config.lr_warmup_steps, 1.0 )

        tokens_seen = self.optimizer_step * self.train_config.batch_size * self.train_config.length_sequence
        warmup_tokens = self.train_config.lr_warmup_steps * self.train_config.batch_size * self.train_config.length_sequence

        cooldown_ratio = min( max( tokens_seen - warmup_tokens, 0.0 ) / ( self.train_config.lr_cooldown_tokens - warmup_tokens ), 1.0 )
        cooldown_ratio = np.cos( cooldown_ratio * np.pi ) * 0.5 + 0.5
        cooldown_ratio = cooldown_ratio * ( 1.0 - self.train_config.lr_cooldown_ratio ) + self.train_config.lr_cooldown_ratio

        return min( warmup_ratio, cooldown_ratio )

    def get_total_epochs( self ) -> int:
        tokens_per_epoch = self.train_config.batch_size * self.train_config.length_sequence * self.train_config.batches_per_epoch
        return int( np.ceil( self.train_config.lr_cooldown_tokens / tokens_per_epoch ) )


    """ ========================================================================
        Forward Pass
        ======================================================================== """

    def forward_pass( self, tokens, past_key_values, cache_length ):

        torch._inductor.cudagraph_mark_step_begin()

        outputs = self.model(
            input_ids=tokens,
            past_key_values=past_key_values,
            use_cache=cache_length > 0
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
            accu_loss = ( mle_loss + aux_loss ) / self.accum_groups
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
            past_key_values = self.model.cache_to( self.past_key_values_list[idx], 'cuda' )
            self.past_key_values_list[idx] = None # type: ignore
            
            loss, accuracy, past_key_values = self.train_sub_step( tokens_xs[idx], tokens_ys[idx], past_key_values )
            
            past_key_values = self.model.cache_to( past_key_values, 'cpu', trim=self.train_config.length_cache )
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

        # torch.cuda.empty_cache()
        # gc.collect()

        return self.reset_metrics()

class TrainerDDP( Trainer ):
    def __init__(
        self,
        train_config: LSWTConfigTraining,
        model: LSWTForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        dataset: str,
        ddp_rank: int,
        ddp_world_size: int
    ):
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        
        self.train_config = train_config
        self.model = model
        self.tokenizer = tokenizer

        self.optimizer = self._load_optimizer()
        self.optimizer_scaler = torch.cuda.amp.GradScaler() # type: ignore

        self.batch_groups = train_config.batch_size // ( train_config.batch_size_step * self.ddp_world_size )
        self.accum_groups = train_config.batch_size // train_config.batch_size_step

        self.past_key_values_list = [ None for _ in range( self.batch_groups ) ]

        self.data_loader_train = self._load_dataset( dataset )

        self.loss_function = self._load_loss_function()

        self.acc_function = AccuracyMetric( model.config.vocab_size, model.config.pad_token_id )

        self.metrics = {
            'loss': metrics.Mean().to( 'cuda' ),
            'acc': metrics.Mean().to( 'cuda' ),
        }

        self.optimizer_step = 0
    
    def _load_dataset( self, dataset ):
        if dataset == 'pile':
            return PileDataset(
                tokenizer=self.tokenizer,
                seq_length=self.train_config.length_sequence,
                batch_size=self.train_config.batch_size,
                pile_shards=list( range( self.ddp_rank, 30, self.ddp_world_size ) )
            ).as_data_loader()
        
        raise ValueError( 'Invalid dataset choice' )
    
    def reset_metrics( self ) -> dict[ str, float ]:
        stats = {}
        for name, metric in self.metrics.items():
            # Syncronise and compute metrics from all devices
            stats[name] = float( sync_and_compute( metric ) )
            
            # Ensure all devices wait for barrier before resetting
            dist.barrier()
            metric.reset()
        return stats
    
    def train_epoch( self, iterator, epoch ):
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

        # torch.cuda.empty_cache()
        # gc.collect()

        return self.reset_metrics()
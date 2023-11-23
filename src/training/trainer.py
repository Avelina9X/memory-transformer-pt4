from transformers import PreTrainedTokenizerBase

from model.configuration import LSWTConfigTraining, LSWTConfig
from model.modeling import LSWTForCausalLM

from .sophia import SophiaG
from .data import PileDataset
from .losses import MLELoss, SimCTGLoss, AccuracyMetric

from torch.optim import AdamW
import torch

from torcheval import metrics

import numpy as np

import gc
import tqdm
import time
import rich

from typing import Dict

from .compile_options import TORCH_COMPILE_OPTIONS

def _bar_format( iter_n, iter_total, elapsed, epoch, loss, acc ):
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

def _load_optimizer( train_config: LSWTConfigTraining, model: LSWTForCausalLM ):
    if train_config.optimizer == 'SophiaG':
        return SophiaG(
            params=model.get_param_groups(),
            lr=0.0,
            betas=( train_config.opt_beta_1, train_config.opt_beta_2 ),
            rho=( train_config.opt_rho ),
            weight_decay=( train_config.opt_weight_decay )
        )
    elif train_config.optimizer == 'AdamW':
        return AdamW(
            params=model.get_param_groups(),
            lr=0.0,
            betas=( train_config.opt_beta_1, train_config.opt_beta_2 ),
            eps=train_config.opt_eps,
            weight_decay=( train_config.opt_weight_decay ),
        )

def _load_loss_function( train_config: LSWTConfigTraining, model_config: LSWTConfig ):
    if train_config.loss_objective == 'MLE':
        return MLELoss( model_config.vocab_size, model_config.pad_token_id )
    elif train_config.loss_objective == 'SimCTG':
        return SimCTGLoss( train_config.loss_sim_margin, model_config.vocab_size, model_config.pad_token_id )
    else:
        raise ValueError( 'Invalid loss function' )

class Trainer():
    def __init__( self, train_config: LSWTConfigTraining, model: LSWTForCausalLM, tokenizer: PreTrainedTokenizerBase, **kwargs ):
        self.train_config = train_config
        self.model = model
        self.tokenizer = tokenizer
        
        self.optimizer = _load_optimizer( train_config, model )
        self.optimizer_scaler = torch.cuda.amp.GradScaler()
        
        self.batch_groups = train_config.batch_size // train_config.batch_size_step
        
        self.past_key_values_list = [ None for _ in range( self.batch_groups ) ]
        
        self.data_loader_train = PileDataset(
            tokenizer=tokenizer,
            seq_length=train_config.length_sequence,
            batch_size=train_config.batch_size
        ).as_data_loader()
        
        self.loss_function = _load_loss_function( train_config, model.config )
        
        self.acc_function = AccuracyMetric( model.config.vocab_size, model.config.pad_token_id )
        
        self.metrics = {
            'loss': metrics.Mean().to( 'cuda' ),
            'acc': metrics.Mean().to( 'cuda' ),
        }
        
        self.optimizer_step = 0
    
    
    """ ========================================================================
        Utility functions
        ======================================================================== """
    
    def reset_metrics( self ) -> Dict[ str, float ]:
        stats = {}
        for name, metric in self.metrics.items():
            stats[name] = float( metric.compute() )
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
        past_key_values = self.model.cache_to( past_key_values, 'cuda' )
        
        torch._inductor.cudagraph_mark_step_begin()
        
        outputs = self.model(
            input_ids=tokens,
            past_key_values=past_key_values,
            use_cache=cache_length > 0
        )
        
        past_key_values = self.model.cache_to( outputs.past_key_values, 'cpu', trim=cache_length )
        logits = outputs.logits
        last_hidden_state = outputs.hidden_states[-1]
        
        del outputs
        
        return logits, past_key_values, last_hidden_state
    
    
    """ ========================================================================
        Training Functions
        ======================================================================== """
    
    @torch.compile( **TORCH_COMPILE_OPTIONS )
    def train_sub_step( self, tokens_x, tokens_y, past_key_values ):
        tokens_x = tokens_x.to( device='cuda', non_blocking=True )
        with torch.autocast( device_type='cuda', dtype=torch.float16 ):
            logits, past_key_values, hidden_states = self.forward_pass( tokens_x, past_key_values, self.train_config.length_cache )
            
            y_pred = logits
            y_true = tokens_y.to( 'cuda', non_blocking=True )
            
            mle_loss, aux_loss = self.loss_function( hidden_states, y_pred, tokens_x, y_true )
            accu_loss = ( mle_loss + aux_loss ) / self.batch_groups
        self.optimizer_scaler.scale( accu_loss ).backward()
        
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
            torch.nn.utils.clip_grad_norm_( self.model.parameters(), self.train_config.opt_max_grad_norm )
            
        self.optimizer_scaler.step( self.optimizer )
        self.optimizer_scaler.update()
        self.optimizer.zero_grad()
    
    def train_batch_step( self, batch ):
        self.model.train()
        
        tokens_xs, tokens_ys = batch
        
        tokens_xs = torch.split( tokens_xs, self.train_config.batch_size_step )
        tokens_ys = torch.split( tokens_ys, self.train_config.batch_size_step )
        
        for idx in range( self.batch_groups ):
            loss, accuracy, past_key_values = self.train_sub_step( tokens_xs[idx], tokens_ys[idx], self.past_key_values_list[idx] )
            self.past_key_values_list[idx] = None
            self.past_key_values_list[idx] = past_key_values
            
            self.metrics[ 'loss' ].update( loss )
            self.metrics[ 'acc' ].update( accuracy )
        
        self.train_optim_step()
    
    def train_epoch( self, iterator, epoch ):
        start_time = time.time()
        
        for batch in range( self.train_config.batches_per_epoch ):
            self.train_batch_step( next( iterator ) )
            
            bar = _bar_format(
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
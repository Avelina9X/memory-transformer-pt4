""" Main pretraining pipeline module. Pretrain's on The Pile """

import datetime
import os
import typing
import train_utils

import rich
import wandb
import torch
import torch.distributed as dist

from transformers import AutoTokenizer
import numpy as np

from training.trainer import Trainer, TrainerDDP
from training.eval import Eval, EvalAlpaca
from training.data import load_pile_uncopyrighted, load_wikitext, load_alpaca

from model.configuration import LSWTConfigTraining, LSWTConfig
from model.modeling import LSWTForCausalLM
from model.embedding_loader import embedding_loader

from constants import HF_CACHE_DIR, WANDB_PROJECT_NAME

WANDB_MODE = 'online'

def ddp_setup( rank, world_size ):
    os.environ[ 'MASTER_ADDR' ] = 'localhost'
    os.environ[ 'MASTER_PORT' ] = '12355'
    
    dist.init_process_group( 'nccl', rank=rank, world_size=world_size, timeout=datetime.timedelta( hours=6 ) )

def ddp_cleanup():
    dist.barrier()
    dist.destroy_process_group()
    
class MyDDP( torch.nn.parallel.DistributedDataParallel ):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def train(
    rank: int = 0,
    world_size: int = 1,
    config: dict | None = None,
    model_config: LSWTConfig | None = None,
    train_config: LSWTConfigTraining | None = None,
    wandb_mode: str | None = None,
    tags: list[str] | None = None
):
    """ Pretraining function.

    Args:
        config (dict | None, optional): Optional WandB style config. Defaults to None.
        model_config (LSWTConfig | None, optional): Optional model config. Defaults to None.
        train_config (LSWTConfigTraining | None, optional): Optional training config. Defaults to None.
        wandb_mode (str | None, optional): Optional wandb mode. Defaults to None.
        tags (list[str] | None, optional): Tags to add to wandb run. Defaults to None.
    """
    
    torch.backends.cuda.matmul.allow_tf32 = True # type: ignore # pylint: disable=W0212
    torch.backends.cudnn.allow_tf32 = True # type: ignore # pylint: disable=W0212
    torch._dynamo.config.cache_size_limit = 1024 * 1024 # type: ignore # pylint: disable=W0212
    # torch._dynamo.config.accumulated_cache_size_limit = 1024 * 1024 # type: ignore # pylint: disable=W0212
    
    # Setup ddp if world size is greater than 1
    if world_size > 1:
        ddp_setup( rank, world_size )
    
    # Set cuda device to rank
    torch.cuda.set_device( rank )
    
    wandb_mode = wandb_mode or WANDB_MODE

    # If on first machine init wandb
    if rank == 0:
        wandb.init(
            project=WANDB_PROJECT_NAME,
            group='pretraining',
            mode=wandb_mode,
            config=config,
            tags=tags,
        ) # type: ignore

    # Get validation and test datasets
    dataset_wikitext = load_wikitext( HF_CACHE_DIR )
    dataset_pile_uncopyrighted = load_pile_uncopyrighted( HF_CACHE_DIR )

    # Get and update model configs
    model_config = model_config or LSWTConfig()
    train_config = train_config or LSWTConfigTraining()
    if rank == 0:
        train_utils.modify_dicts( wandb.config, model_config, train_config )
    else:
        train_utils.modify_dicts( config, model_config, train_config ) # type: ignore

    # Load model and embeddings
    parent_embeddings = embedding_loader( model_config, cache_dir=HF_CACHE_DIR )
    model = LSWTForCausalLM( model_config, parent_embeddings ).cuda()        

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained( model_config.parent_embeddings, use_fast=True, cache_dir=HF_CACHE_DIR )

    # Instantiate trainer/evaluator
    if world_size == 1:
        trainer = Trainer( train_config, model, tokenizer, 'pile' )
    else:
        model = MyDDP( model, device_ids=[ rank ] )
        trainer = TrainerDDP( train_config, model, tokenizer, 'pile', rank, world_size, 24 ) # type: ignore
    evaluator = Eval( model, tokenizer ) # type: ignore

    # If on first machine print model and update wandb
    if rank == 0:
        # Print data
        rich.print( trainer.train_config )
        rich.print( trainer.model.config )

        # Compute params
        params_total = sum( p.numel() for p in model.parameters() )
        params_trainable = sum( p.numel() for p in model.parameters() if p.requires_grad )
        params_non_trainable = sum( p.numel() for p in model.parameters() if not p.requires_grad )

        # Print parametes
        print( '\nParameter Count:' )
        rich.print( f'total         = {params_total}' )
        rich.print( f'trainable     = {params_trainable}' )
        rich.print( f'non trainable = {params_non_trainable}' )
        print()

        # Update dict
        wandb.config.update( {
            **model_config.to_wandb_dict(),
            **train_config.to_wandb_dict(),
            'params.total': params_total,
            'params.trainable': params_trainable,
            'params.non_trainable': params_non_trainable,
        } )

    # Create training iterator
    iterator = iter( trainer.data_loader_train )

    # Train loop
    for i in range( trainer.get_total_epochs() ):
        train_metrics = trainer.train_epoch( iterator, i + 1 )
        
        # If on first machine, do validation loop and log metrics
        if rank == 0:
            valid_metrics_wikitext = evaluator.eval_epoch( dataset_wikitext, 'page', train_config.length_sequence )

            train_log = {
                'train/ppl': np.exp( train_metrics[ 'loss' ] ),
                'train/loss': train_metrics[ 'loss' ],
                'train/acc': train_metrics[ 'acc' ],
            }

            valid_log = {
                'validation/wikitext/ppl': np.exp( valid_metrics_wikitext[ 'loss' ] ),
                'validation/wikitext/loss': valid_metrics_wikitext[ 'loss' ],
                'validation/wikitext/acc': valid_metrics_wikitext[ 'acc' ],
            }

            stats_log = {
                'stats/n_tokens': trainer.optimizer_step * trainer.train_config.batch_size * trainer.train_config.length_sequence,
                'stats/n_batches': trainer.optimizer_step,
                'stats/n_epochs': i + 1,
                'stats/learning_rate': trainer.get_schedule() * trainer.train_config.lr_max,
            }

            wandb.log( {
                **train_log,
                **valid_log,
                **stats_log,
            } )

    # If on first machine, run testing, save model and log
    if rank == 0:
        test_metrics = evaluator.eval_epoch( dataset_pile_uncopyrighted, 'text', train_config.length_sequence )

        wandb.log( {
            'test/pile-uncopyrighted/ppl': np.exp( test_metrics[ 'loss' ] ),
            'test/pile-uncopyrighted/loss': test_metrics[ 'loss' ],
            'test/pile-uncopyrighted/acc': test_metrics[ 'acc' ],
        } )

        train_utils.save_model( model, log_wandb=( wandb_mode == 'online' ) ) # type: ignore
        
        wandb.finish()
    
    # Cleanup ddp if world size is greater than 1
    if world_size > 1:
        ddp_cleanup()

def finetune(
    config: dict | None = None,
    wandb_mode: str | None = None
):
    wandb_mode = wandb_mode or WANDB_MODE

    with wandb.init(
        project=WANDB_PROJECT_NAME,
        group='finetuning',
        mode=wandb_mode,
        config=config
    ): # type: ignore
        
        # Get pretrained run name and checkpoint directory
        pretrained_run_name = wandb.config[ 'finetune.pretrained_run_name' ]
        pretrained_run_dir = f'./checkpoints/{pretrained_run_name}'
        
        # Get pretrained model artifact
        pretrained_artifact = train_utils.get_model_artifact( pretrained_run_name )
        wandb.run.use_artifact( pretrained_artifact ) # type: ignore
        
        # Get and update model configs
        model_config = typing.cast( LSWTConfig, LSWTConfig.from_pretrained( pretrained_run_dir, torch_dtype=None ) )
        train_config = LSWTConfigTraining()
        train_utils.modify_dicts( wandb.config, model_config, train_config )
        
        # Load model and correct casting
        model = typing.cast( LSWTForCausalLM, LSWTForCausalLM.from_pretrained( pretrained_run_dir, **model_config.to_dict() ) ).cuda()
        train_utils.set_backbone_trainable( model, wandb.config[ 'finetune.trainable_backbone' ] )
        
        # Load tokenizer and add new segment tokens
        tokenizer = AutoTokenizer.from_pretrained( model_config.parent_embeddings, use_fast=True, cache_dir=HF_CACHE_DIR )
        train_utils.add_special_tokens( tokenizer )
        
        # Instantiate trainer for finetuning
        trainer = Trainer( train_config, model, tokenizer, wandb.config[ 'finetune.dataset' ] )
        evaluator = EvalAlpaca( model, tokenizer )

        # Print data
        rich.print( trainer.train_config )
        rich.print( trainer.model.config )

        # Compute params
        params_total = sum( p.numel() for p in model.parameters() )
        params_trainable = sum( p.numel() for p in model.parameters() if p.requires_grad )
        params_non_trainable = sum( p.numel() for p in model.parameters() if not p.requires_grad )
        
        # Print parametes
        print( '\nParameter Count:' )
        rich.print( f'total         = {params_total}' )
        rich.print( f'trainable     = {params_trainable}' )
        rich.print( f'non trainable = {params_non_trainable}' )
        print()
        
        # Update dict
        wandb.config.update( {
            **model_config.to_wandb_dict(),
            **train_config.to_wandb_dict(),
            'params.total': params_total,
            'params.trainable': params_trainable,
            'params.non_trainable': params_non_trainable,
        } )
        
        # Create training iterator
        iterator = iter( trainer.data_loader_train )
        dataset_alpaca = load_alpaca( HF_CACHE_DIR ).shard( 10, 0 ) # type: ignore # pylint: disable=W0212

        # Train loop
        for i in range( trainer.get_total_epochs() ):
            train_metrics = trainer.train_epoch( iterator, i + 1 )
            valid_metrics = evaluator.eval_epoch( dataset_alpaca, None, train_config.length_sequence // 2 )
            rich.print( valid_metrics )
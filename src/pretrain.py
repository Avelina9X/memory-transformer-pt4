""" Main pretraining pipeline module. Pretrain's on The Pile """

import copy
import os
import argparse

import rich
import wandb
from wcmatch import glob

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from transformers import AutoTokenizer

from training.trainer import Trainer, TrainerDDP
from training.eval import Eval
from training.data import load_pile_uncopyrighted, load_wikitext

from model.configuration import LSWTConfigTraining, LSWTConfig
from model.modeling import LSWTForCausalLM
from model.embedding_loader import embedding_loader

from constants import HF_CACHE_DIR, WANDB_PROJECT_NAME
import train_utils

def ddp_setup( rank: int, world_size: int ):
    """ Sets up the DDP process group and sets CUDA rank.
    Note this does not support multi-machine DDP, only multi-device DDP.

    Args:
        rank (int): current device rank
        world_size (int): global world size
    """
    torch.cuda.set_device( rank )

    os.environ[ 'MASTER_ADDR' ] = 'localhost'
    os.environ[ 'MASTER_PORT' ] = '12355'

    dist.init_process_group( 'nccl', rank=rank, world_size=world_size )

def ddp_cleanup():
    """ Shuts down the DDP process group. """
    dist.destroy_process_group()

class MyDDP( torch.nn.parallel.DistributedDataParallel ):
    """ Custom DDP wrapper. Defers method and attribute accesses to underlying module. """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def train(
    rank: int = 0,
    world_size: int = 1,
    config: dict | None = None,
    wandb_mode: str | None = None,
    wandb_tags: list[str] | None = None,
    wandb_run_name: str | None = None
):
    """ Pretraining function.

    Args:
        rank (int, optional): The DDP process rank. Defaults to 0.
        world_size (int, optional): The DDP world size. When 1 DDP is disabled. Defulats to 1.
        config (dict | None, optional): Optional WandB style config. Defaults to None.
        wandb_mode (str | None, optional): Optional wandb mode. Defaults to None.
        wandb_tags (list[str] | None, optional): Tags to add to wandb run. Defaults to None.
        wandb_run_name (str | None, optional): Optional wandb run name. Defaults to None.
    """

    # Set manual seed for reproducibility
    torch.manual_seed( 0 )

    # Set some performance flags
    torch.backends.cuda.matmul.allow_tf32 = True # type: ignore # pylint: disable=W0212
    torch.backends.cudnn.allow_tf32 = True # type: ignore # pylint: disable=W0212
    torch._dynamo.config.cache_size_limit = 1024 * 1024 # type: ignore # pylint: disable=W0212

    # Setup ddp if world size is greater than 1
    if world_size > 1:
        ddp_setup( rank, world_size )

    # If on first machine init wandb
    if rank == 0:
        wandb.init(
            project=WANDB_PROJECT_NAME,
            group='pretraining',
            mode=wandb_mode,
            config=config,
            tags=wandb_tags,
            name=wandb_run_name,
            settings=wandb.Settings( _disable_stats=True )
        ) # type: ignore

    # Get validation and test datasets
    dataset_wikitext = load_wikitext( HF_CACHE_DIR )
    dataset_pile_uncopyrighted = load_pile_uncopyrighted( HF_CACHE_DIR )

    # Get and update model configs
    model_config = LSWTConfig()
    train_config = LSWTConfigTraining()

    # Update original config
    if rank == 0:
        train_utils.modify_dicts( wandb.config, model_config, train_config )
    else:
        train_utils.modify_dicts( config, model_config, train_config ) # type: ignore

    # Because of RNG stuff, we need to init the model without registers. Create config without registers
    init_model_config = copy.deepcopy( model_config )
    init_model_config.n_registers = 0

    # Create model as normal, but with reigsters disabled
    parent_embeddings = embedding_loader( model_config, cache_dir=HF_CACHE_DIR )
    init_model = LSWTForCausalLM( init_model_config, parent_embeddings )

    # Create model as normal, but with register enabled and copy contents
    model = LSWTForCausalLM( model_config, parent_embeddings )
    state_dict_diff = model.load_state_dict( init_model.state_dict(), strict=False )
    rich.print( state_dict_diff )

    # Place model on CUDA
    model = model.cuda() # type: ignore

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

            train_log = train_utils.compute_metric_dict( train_metrics, 'train' )
            valid_log = train_utils.compute_metric_dict( valid_metrics_wikitext, 'validation/wikitext' )
            stats_log = train_utils.compute_stats_dict( trainer, i )

            wandb.log( {
                **train_log,
                **valid_log,
                **stats_log,
            } )

    # If on first machine, run testing, save model and log
    if rank == 0:
        test_metrics = evaluator.eval_epoch( dataset_pile_uncopyrighted, 'text', train_config.length_sequence )
        test_log = train_utils.compute_metric_dict( test_metrics, 'test/pile-uncopyrighted' )

        wandb.log( test_log )

        train_utils.save_model( model, log_wandb=( wandb_mode == 'online' ) ) # type: ignore

        wandb.finish()

    # Cleanup ddp if world size is greater than 1
    if world_size > 1:
        ddp_cleanup()

def run():
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '-c',
        '--config',
        type=lambda x: glob.glob( x, flags=glob.BRACE ),
        required=True
    )

    argparser.add_argument(
        '-w',
        '--wmode',
        default='disabled',
        choices=[ 'online', 'offline', 'disabled' ]
    )

    arguments = argparser.parse_args()

    config = train_utils.parse_yaml_config( arguments.config )

    if torch.cuda.device_count() == 1:
        train( config=config, wandb_mode=arguments.wmode, wandb_tags=[ 'rerope_tests' ], wandb_run_name=config[ 'meta.run_name' ] )
    else:
        mp.spawn( # type: ignore
            fn=train,
            args=(
                torch.cuda.device_count(),
                config,
                arguments.wmode,
                [ 'rerope_tests', 'ddp' ],
                config[ 'meta.run_name' ]
            ),
            nprocs=torch.cuda.device_count(),
            join=True,
        )

if __name__ == '__main__':
    run()

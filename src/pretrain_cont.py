""" Continuous pretraining module """

import datetime
import os
import typing

import rich
import wandb
import torch
import torch.distributed as dist

from transformers import AutoTokenizer
import numpy as np

from training.trainer import Trainer
from training.eval import Eval
from training.data import load_pile_uncopyrighted, load_wikitext, HFDatasetConfig

from model.configuration import LSWTConfigTraining, LSWTConfig
from model.modeling import LSWTForCausalLM
from model.embedding_loader import embedding_loader

from constants import HF_CACHE_DIR, WANDB_PROJECT_NAME
import train_utils

WANDB_MODE = 'online'

def train(
    config: dict | None = None,
    wandb_mode: str | None = None,
    tags: list[str] | None = None
):
    """ Pretraining function.

    Args:
        config (dict | None, optional): Optional WandB style config. Defaults to None.
        wandb_mode (str | None, optional): Optional wandb mode. Defaults to None.
        tags (list[str] | None, optional): Tags to add to wandb run. Defaults to None.
    """
    
    torch.backends.cuda.matmul.allow_tf32 = True # type: ignore # pylint: disable=W0212
    torch.backends.cudnn.allow_tf32 = True # type: ignore # pylint: disable=W0212
    torch._dynamo.config.cache_size_limit = 1024 * 1024 # type: ignore # pylint: disable=W0212
    # torch._dynamo.config.accumulated_cache_size_limit = 1024 * 1024 # type: ignore # pylint: disable=W0212

    pretrained_run_name = config[ 'finetune.pretrained_run_name' ]
    pretrained_run_dir = f'./checkpoints/{pretrained_run_name}'
    new_seq_len = config[ 'train.length_sequence' ]

    new_run_name = f'{pretrained_run_name}_{new_seq_len}_gov'
    
    wandb_mode = wandb_mode or WANDB_MODE

    # If on first machine init wandb
    wandb.init(
        project=WANDB_PROJECT_NAME,
        group='pretraining',
        mode=wandb_mode,
        config=config,
        tags=tags,
        name=new_run_name
    ) # type: ignore

    # Get validation and test datasets
    dataset_wikitext = load_wikitext( HF_CACHE_DIR )
    dataset_pile_uncopyrighted = load_pile_uncopyrighted( HF_CACHE_DIR )

    # Get pretrained model artifact
    pretrained_artifact = train_utils.get_model_artifact( pretrained_run_name )
    wandb.run.use_artifact( pretrained_artifact ) # type: ignore
    
    # Get and update model configs
    model_config = typing.cast( LSWTConfig, LSWTConfig.from_pretrained( pretrained_run_dir, torch_dtype=None ) )
    train_config = LSWTConfigTraining()
    train_utils.modify_dicts( wandb.config, model_config, train_config )

    # Load model and correct casting
    model = typing.cast( LSWTForCausalLM, LSWTForCausalLM.from_pretrained( pretrained_run_dir, **model_config.to_dict() ) ).cuda()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained( model_config.parent_embeddings, use_fast=True, cache_dir=HF_CACHE_DIR )

    # PG-19 config
    # dataset_config = HFDatasetConfig(
    #     'pg19',
    #     None,
    #     'train',
    #     'text',
    #     HF_CACHE_DIR
    # )

    # Gov config
    dataset_config = HFDatasetConfig(
        'tau/scrolls',
        'gov_report',
        'train',
        'input',
        HF_CACHE_DIR
    )

    # Instantiate trainer/evaluator
    trainer = Trainer( train_config, model, tokenizer, dataset_config )
    evaluator = Eval( model, tokenizer ) # type: ignore

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
    test_metrics = evaluator.eval_epoch( dataset_pile_uncopyrighted, 'text', train_config.length_sequence )

    wandb.log( {
        'test/pile-uncopyrighted/ppl': np.exp( test_metrics[ 'loss' ] ),
        'test/pile-uncopyrighted/loss': test_metrics[ 'loss' ],
        'test/pile-uncopyrighted/acc': test_metrics[ 'acc' ],
    } )

    train_utils.save_model( model, log_wandb=( wandb_mode == 'online' ) ) # type: ignore
    
    wandb.finish()


if __name__ == '__main__':
    ROPE_2K4K = 'rare-violet-77'
    REROPE_2K4K = 'rich-bee-76'

    finetune_config = {
            'model.trainable_embeddings': False,

            'train.batch_size': 240,
            'train.batch_size_step': 16,
            'train.batches_per_epoch': 64,
            
            'train.length_sequence': 4096,
            'train.length_cache': 4096,
            
            'train.loss_objective': 'MLE',
            
            'train.optimizer': 'LaProp',
            'train.lr_max': 6e-5,
            'train.opt_weight_decay': 0.1,
            'train.opt_max_grad_norm': 1.0,
            'train.opt_eps': 1e-8,
            'train.opt_beta_1': 0.9,
            'train.opt_beta_2': 0.95,

            'train.lr_cooldown_tokens': 5_000_000_000,

            'finetune.pretrained_run_name': REROPE_2K4K,
        }
    
    train(
        config=finetune_config,
        wandb_mode='online',
        tags=[ 'rerope_tests', 'cont_pretrain' ]
    )
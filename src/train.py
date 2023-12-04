""" Main pretraining pipeline module """

import os
import pathlib

import rich
import wandb

from transformers import AutoTokenizer
import numpy as np

from training.trainer import Trainer
from training.eval import Eval
from training.data import load_pile_uncopyrighted, load_wikitext

from model.configuration import LSWTConfigTraining, LSWTConfig
from model.modeling import LSWTForCausalLM
from model.embedding_loader import embedding_loader

from constants import HF_CACHE_DIR, WANDB_PROJECT_NAME


os.environ[ 'TOKENIZERS_PARALLELISM' ] = 'true'

WANDB_MODE = 'online'

def find_and_extract( source, prefix ):
    idx = source.find( prefix + '.' )
    if idx != 0:
        return None
    return source[ len( prefix ) + 1 : ]

def modify_dicts( config: dict, model_config: LSWTConfig, train_config: LSWTConfigTraining ):
    for key, value in config.items():
        model_key = find_and_extract( key, 'model' )
        train_key = find_and_extract( key, 'train' )

        if model_key is not None:
            getattr( model_config, model_key )
            setattr( model_config, model_key, value )

        if train_key is not None:
            getattr( train_config, train_key )
            setattr( train_config, train_key, value )

def get_checkpoint_path( name: str | None=None ):
    name = name or wandb.run.name or wandb.run.id # type: ignore

    root_dir = pathlib.Path().cwd().joinpath( 'checkpoints', name )
    config_dir = root_dir.joinpath( 'config.json' )
    model_dir = root_dir.joinpath( 'model.safetensors' )
    return root_dir, config_dir, model_dir

def save_model( model: LSWTForCausalLM, log_wandb: bool=False ):
    root_dir, config_dir, model_dir = get_checkpoint_path()

    model.half().save_pretrained( root_dir, safe_serialization=True )

    if log_wandb:
        model_name = model.config.model_type

        model_artifact = wandb.Artifact( name=model_name, type="model" )
        model_artifact.add_file( model_dir ) # type: ignore
        model_artifact.add_file( config_dir ) # type: ignore

        wandb.run.log_artifact( model_artifact ) # type: ignore


def train(
    config: dict | None = None,
    model_config: LSWTConfig | None = None,
    train_config: LSWTConfigTraining | None = None,
    wandb_mode: str | None = None
):

    wandb_mode = wandb_mode or WANDB_MODE

    with wandb.init(
        project=WANDB_PROJECT_NAME,
        group='pretraining',
        mode=wandb_mode,
        config=config
    ): # type: ignore

        # Get validation and test datasets
        dataset_wikitext = load_wikitext( HF_CACHE_DIR )
        dataset_pile_uncopyrighted = load_pile_uncopyrighted( HF_CACHE_DIR )

        # Get and update model configs
        model_config = model_config or LSWTConfig()
        train_config = train_config or LSWTConfigTraining()
        modify_dicts( wandb.config, model_config, train_config )

        # Load model and embeddings
        parent_embeddings = embedding_loader( model_config, cache_dir=HF_CACHE_DIR )
        model = LSWTForCausalLM( model_config, parent_embeddings ).cuda()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained( model_config.parent_embeddings, use_fast=True, cache_dir=HF_CACHE_DIR )

        # Instantiate trainer/evaluator
        trainer = Trainer( train_config, model, tokenizer )
        evaluator = Eval( model, tokenizer )

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

        test_metrics = evaluator.eval_epoch( dataset_pile_uncopyrighted, 'text', train_config.length_sequence )

        wandb.log( {
            'test/pile-uncopyrighted/ppl': np.exp( test_metrics[ 'loss' ] ),
            'test/pile-uncopyrighted/loss': test_metrics[ 'loss' ],
            'test/pile-uncopyrighted/acc': test_metrics[ 'acc' ],
        } )

        save_model( model, log_wandb=( wandb_mode == 'online' ) )

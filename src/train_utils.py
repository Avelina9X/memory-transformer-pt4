""" Utilities and helper functions for training and pretraining """

import pathlib
import yaml

import wandb
import numpy as np

from model.configuration import LSWTConfigTraining, LSWTConfig
from model.modeling import LSWTForCausalLM
from training.trainer import Trainer
from constants import WANDB_PROJECT_NAME

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

def add_special_tokens( tokenizer ):
    tokenizer.add_tokens( [ '<seg_start>', '<seg_end>' ], special_tokens=True )
    tokenizer.sep_token = '<seg_start>'
    tokenizer.cls_token = '<seg_end>'

def get_model_artifact( run_name: str ):
    pretrain_run = wandb.Api().runs(
        path=WANDB_PROJECT_NAME,
        filters={ "display_name": run_name }
    )[0]

    return [
        artifact
        for artifact in pretrain_run.logged_artifacts()
        if artifact.type == 'model'
    ][0]

def set_backbone_trainable( model: LSWTForCausalLM, trainable: bool ):
    model.model.blocks.requires_grad_( trainable )
    if not trainable:
        model.model.blocks = model.model.blocks.half()

def compute_metric_dict( inputs: dict, name: str ):
    return {
        f'{name}/ppl': np.exp( inputs[ 'loss' ] ),
        f'{name}/loss': inputs[ 'loss' ],
        f'{name}/acc': inputs[ 'acc' ],
    }

def compute_stats_dict( trainer: Trainer, i: int ):
    return {
        'stats/n_tokens': trainer.optimizer_step * trainer.train_config.batch_size * trainer.train_config.length_sequence,
        'stats/n_batches': trainer.optimizer_step,
        'stats/n_epochs': i + 1,
        'stats/learning_rate': trainer.get_schedule() * trainer.train_config.lr_max,
    }

def parse_yaml_config( files: list[str] ):
    def unpack_dict( d ):
        return {
            f'{outer_k}.{inner_k}' : inner_v
            for outer_k, outer_v in d.items()
            for inner_k, inner_v in outer_v.items()
        }
    
    config = {}
    
    for file in files:
        with open( file, 'r', encoding='utf-8' ) as f:
            obj = unpack_dict( yaml.load( f, yaml.FullLoader ) )
            config.update( obj )
    
    return config
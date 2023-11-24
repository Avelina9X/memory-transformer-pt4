import os
HF_CACHE_DIR = os.environ[ 'HF_CACHE_DIR' ]
os.environ[ 'TOKENIZERS_PARALLELISM' ] = 'true'

from training.trainer import Trainer
from training.eval import Eval
from training.data import load_pile_uncopyrighted, load_wikitext

from model.configuration import LSWTConfigTraining, LSWTConfig
from model.modeling import LSWTForCausalLM
from model.embedding_loader import embedding_loader

from transformers import AutoTokenizer
from datasets import load_dataset

import torch
import rich
import wandb

import numpy as np

import warnings
import argparse
import logging
import pathlib

from typing import Optional

WANDB_MODE = 'online'

def find_and_extract( source, prefix ):
    idx = source.find( prefix + '.' )
    if idx != 0:
        return None 
    else:
        return source[ len( prefix ) + 1 : ]

def modify_dicts( config: dict, model_config: LSWTConfig, train_config: LSWTConfigTraining ):
    for key, value in config.items():
        model_key = find_and_extract( key, 'model' )
        train_key = find_and_extract( key, 'train' )
        
        if model_key is not None:
            getattr( model_config, model_key )
            model_config.__setattr__( model_key, value )
        
        if train_key is not None:
            getattr( train_config, train_key )
            train_config.__setattr__( train_key, value )

def get_checkpoint_path( name: Optional[str]=None ):
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
        config: Optional[dict]=None,
        model_config: Optional[LSWTConfig]=None,
        train_config: Optional[LSWTConfigTraining]=None,
        wandb_mode: Optional[str]=None
    ):
    
    wandb_mode = wandb_mode or WANDB_MODE
    
    with wandb.init(
        project='memory-transformer-pt4',
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
        params_total = sum( [ p.numel() for p in model.parameters() ] )
        params_trainable = sum( [ p.numel() for p in model.parameters() if p.requires_grad ] )
        params_non_trainable = sum( [ p.numel() for p in model.parameters() if not p.requires_grad ] )
        
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
            
            

if __name__ == '__main__':       
    wandb.login( key=os.environ[ 'WANDB_API_KEY' ] )
    
    parser = argparse.ArgumentParser()
    parser.add_argument( '--sweep-id', type=str, default=None )
    parser.add_argument( '--sweep-count', type=int, default=None )
    arguments = parser.parse_args()
    
    torch._dynamo.config.cache_size_limit = 256 # type: ignore
    
    if arguments.sweep_id is not None:
        # Disable warnings
        warnings.simplefilter( 'ignore' )
        torch._logging.set_logs( dynamo=logging.FATAL ) # type: ignore
        torch._logging.set_logs( inductor=logging.FATAL ) # type: ignore
        torch._logging.set_logs( dynamic=logging.FATAL ) # type: ignore
    
        # Run agent
        wandb.agent(
            sweep_id=arguments.sweep_id,
            function=train,
            project='memory-transformer-pt4',
            count=arguments.sweep_count,
        )
        
    else:
        custom_config = {            
            'model.trainable_embeddings': True,
            'train.loss_objective': 'SimCTG',
            'train.batches_per_epoch': 32,
        }
        
        train( config=custom_config, wandb_mode='offline' )
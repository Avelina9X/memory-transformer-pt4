""" Main finetuning pipeline module. Finetunes's on Open Orca """

import typing

import rich
import wandb

from transformers import AutoTokenizer

from training.trainer import Trainer
from training.eval import EvalAlpaca
from training.data import load_alpaca

from model.configuration import LSWTConfigTraining, LSWTConfig
from model.modeling import LSWTForCausalLM

from constants import HF_CACHE_DIR, WANDB_PROJECT_NAME
import train_utils

WANDB_MODE = 'online'

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
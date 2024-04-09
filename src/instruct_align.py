from collections.abc import Sequence
import os
import argparse
import typing

import shortuuid
import rich
import wandb
from wcmatch import glob

import torch

import evaluate
import datasets
import transformers
from transformers import AutoTokenizer

from training.trainer import DPHTrainer

from training.data_instruct.task_base import BaseChoiceInstructDataset, BaseInstructDataset
from training.data_instruct.tasks import DIRECTORY_ALL
from training.data_instruct.formatter import InstructionFormatter
from training.data_instruct.task_loader import DPHMultiTaskLoader
from training.data_instruct.batcher import DPHChoiceInstructionBatcher

from model.configuration import LSWTConfigTraining, LSWTConfig, LSWTConfigTrainingDPH
from model.modeling import LSWTForCausalLM, LSWTForDPH

from constants import HF_CACHE_DIR, WANDB_API_KEY, WANDB_PROJECT_NAME
import train_utils
from instruct_tune import create_validation_zeroshot_tasks, log_full_config, log_stats

def create_align_tasks( dph_mix: list[str] ) -> list[BaseInstructDataset]:
    dph_list = [ i.split( '/' ) for i in dph_mix ]
    return [ DIRECTORY_ALL[task[0]][task[1]]( HF_CACHE_DIR ) for task in dph_list ]

def evaluate_zero_shot_task( task: BaseChoiceInstructDataset, batcher: DPHChoiceInstructionBatcher ) -> tuple[ str, dict[ str, float ] ]:
    task_ds = task.get_validation_docs()
    assert task_ds is not None
    val_metrics_both = batcher.evaluate_dataset( task, task_ds, False, False )
    val_metrics_log = val_metrics_both[ 'log' ]
    val_metrics_dph = val_metrics_both[ 'dph' ]

    val_metrics = {}
    val_metrics.update( { f'log_{name}': metric for name, metric in val_metrics_log.items() } )
    val_metrics.update( { f'dph_{name}': metric for name, metric in val_metrics_dph.items() } )

    task_name = f'{task.task_name}/{task.task_subset}'
    curr_metrics = f'{task_name}={val_metrics}'
    rich.print( curr_metrics )

    return (
        curr_metrics,
        train_utils.compute_validation_metric_dict( val_metrics, task_name )
    )

def aggregate_gpt4all_score( metrics: dict[ str, float ], prefix: str ) -> dict[ str, float ]:
    macro_scores = [
        metrics[ f'Hellaswag/no_choice/{prefix}_accuracy' ],
        metrics[ f'obqa/main/{prefix}_accuracy' ],
        metrics[ f'winogrande/no_choice/{prefix}_accuracy' ],
        metrics[ f'arc/ARC-{prefix}_Easy/accuracy' ],
        metrics[ f'arc/ARC-{prefix}_Challenge/accuracy' ],
        metrics[ f'super_glue/boolq/{prefix}_accuracy' ],
        metrics[ f'piqa/no_choice/{prefix}_accuracy' ],
    ]
    return { f'validation/{prefix}_gpt4all': 100 * sum( macro_scores ) / len( macro_scores ) }

def aggregate_glue_score( metrics: dict[ str, float ], prefix: str ) -> dict[ str, float ]:
    macro_scores = [
        metrics[ f'GLUE/cola/{prefix}_matthews_correlation' ],
        ( metrics[ f'GLUE/mnli_matched/{prefix}_accuracy' ] + metrics[ f'GLUE/mnli_mismatched/{prefix}_accuracy' ] ) / 2,
        ( metrics[ f'GLUE/mrpc/{prefix}_accuracy' ] + metrics[ f'GLUE/mrpc/{prefix}_f1' ] ) / 2,
        metrics[ f'GLUE/qnli/{prefix}_accuracy' ],
        ( metrics[ f'GLUE/qqp/{prefix}_accuracy' ] + metrics[ f'GLUE/qqp/{prefix}_f1' ] ) / 2,
        metrics[ f'GLUE/rte/{prefix}_accuracy' ],
        metrics[ f'GLUE/sst2/{prefix}_accuracy' ],
        ( metrics[ f'GLUE/stsb/{prefix}_pearson' ] + metrics[ f'GLUE/stsb/{prefix}_spearmanr' ] ) / 2,
    ]

    macro_scores_wnli = [
        *macro_scores,
        metrics[ f'GLUE/wnli/{prefix}_accuracy' ],
    ]

    return {
        f'validation/{prefix}_glue_all': 100 * sum( macro_scores_wnli ) / len( macro_scores_wnli ),
        f'validation/{prefix}_glue_no_wnli': 100 * sum( macro_scores ) / len( macro_scores ),
    }

def aggregate_race_score( metrics: dict[ str, float ], prefix: str ) -> dict[ str, float ]:
    macro_scores = [
        metrics[ f'race/middle/{prefix}_accuracy' ] * 1.44,
        metrics[ f'race/high/{prefix}_accuracy' ] * 3.45,
    ]
    return {
        f'validation/{prefix}_race_avg': 100 * sum( macro_scores ) / ( 1.44 + 3.45 )
    }

def instruct_align(
    config: dict,
    wandb_mode: str | None = None,
    wandb_tags: list[str] | None = None,
    wandb_run_name: str | None = None
):
    # Log in to wandb
    wandb.login( key=WANDB_API_KEY )

    # Set some performance flags
    torch.backends.cuda.matmul.allow_tf32 = True # type: ignore # pylint: disable=W0212
    torch.backends.cudnn.allow_tf32 = True # type: ignore # pylint: disable=W0212
    torch._dynamo.config.cache_size_limit = 1024 * 1024 * 1024 # type: ignore # pylint: disable=W0212

    if not __debug__:
        transformers.utils.logging.disable_progress_bar()
        evaluate.utils.logging.disable_progress_bar()
        datasets.utils.logging.disable_progress_bar()
        torch._inductor.select_algorithm.PRINT_AUTOTUNE = False # type: ignore # pylint: disable=W0212

    # Get pretrained run name and checkpoint directory
    pretrained_run_name = config[ 'finetune.checkpoint' ]
    pretrained_run_dir = f'./checkpoints/{pretrained_run_name}'
    output_dir = f'./checkpoints/{wandb_run_name}'

    # Grab configs
    model_config = typing.cast( LSWTConfig, LSWTConfig.from_pretrained( pretrained_run_dir, torch_dtype=None ) )
    train_config = LSWTConfigTraining()
    dph_config = LSWTConfigTrainingDPH()
    train_utils.modify_dicts( config, model_config, train_config, dph_config )

    # Get reward head name
    assert isinstance( model_config.reward_heads, Sequence )
    assert len( model_config.reward_heads ) == 1
    reward_head_name = model_config.reward_heads[0]

    # Load reference model and set trainable
    ref_model = typing.cast( LSWTForCausalLM, LSWTForCausalLM.from_pretrained( pretrained_run_dir, **model_config.to_dict() ) )
    ref_model.requires_grad_( False )
    ref_model = typing.cast( LSWTForCausalLM, ref_model.half().eval().cuda() ) # type: ignore # pylance is confused

    # Load DPH model and set trainable
    dph_model = typing.cast( LSWTForDPH, LSWTForDPH.from_pretrained( pretrained_run_dir, **model_config.to_dict() ) )
    train_utils.set_backbone_trainable( dph_model, config[ 'finetune.trainable_backbone' ] )
    dph_model = typing.cast( LSWTForDPH, dph_model.cuda() ) # type: ignore # pylance is confused

    # Mask out parameters
    if 'finetune.frozen_params' in config:
        frozen_list = train_utils.set_training_mask( dph_model, config[ 'finetune.frozen_params' ] )
        rich.print( 'Frozen params:' )
        rich.print( frozen_list )
        print()

    # Load tokenizer and add new segment tokens
    tokenizer = AutoTokenizer.from_pretrained( model_config.parent_embeddings, use_fast=True, cache_dir=HF_CACHE_DIR )
    train_utils.add_special_tokens( tokenizer )

    # Create task mixes
    train_tasks = create_align_tasks( config[ 'finetune.dph_mix' ] )
    validation_zeroshot_tasks = create_validation_zeroshot_tasks()

    # Instantiate instruct helpers
    formatter = InstructionFormatter( tokenizer )
    batcher = DPHChoiceInstructionBatcher( dph_model, formatter, 'mean', reward_head_name )

    # Create dataset
    task_loader = DPHMultiTaskLoader(
        task_list=train_tasks,
        formatter=formatter,
        seq_length=train_config.length_sequence,
        batch_size=train_config.batch_size,
        mask_type='train',
    )

    # Instantiate trainer for alignment
    trainer = DPHTrainer(
        train_config,
        dph_config,
        ref_model,
        dph_model,
        reward_head_name,
        tokenizer,
        task_loader
    )

    # Print out our configs
    rich.print( trainer.train_config )
    rich.print( trainer.model_dph.config )

    # Compute params
    params_total = sum( p.numel() for p in dph_model.parameters() )
    params_trainable = sum( p.numel() for p in dph_model.parameters() if p.requires_grad )
    params_non_trainable = sum( p.numel() for p in dph_model.parameters() if not p.requires_grad )

    # Print parametes
    print( '\nParameter Count:' )
    rich.print( f'total         = {params_total}' )
    rich.print( f'trainable     = {params_trainable}' )
    rich.print( f'non trainable = {params_non_trainable}' )
    print()

    # Update the base config
    config.update( {
        **model_config.to_wandb_dict(),
        **train_config.to_wandb_dict(),
        'params.total': params_total,
        'params.trainable': params_trainable,
        'params.non_trainable': params_non_trainable,
    } )

    # Log base config
    if wandb_mode != 'disabled':
        log_full_config( output_dir, config )

    # Create iterator
    iterator = iter( task_loader.as_data_loader() )

    # Initialise WandB
    wandb.init(
        project=WANDB_PROJECT_NAME,
        group='alignment',
        mode=wandb_mode,
        config=config,
        tags=wandb_tags,
        name=wandb_run_name,
        settings=wandb.Settings( _disable_stats=True )
    )

    input_artifact = train_utils.get_model_artifact( pretrained_run_name )
    wandb.use_artifact( input_artifact )

    # Train loop
    for i in range( trainer.get_total_epochs() ):
        # Train for an epoch and get metrics
        train_metrics = trainer.train_epoch( iterator, i + 1 )

        # Create empty validation metrics list
        validation_lines = []
        validation_dict = {}

        # If validation flag is set (or it's the last epoch) run validation
        if config[ 'meta.validate' ] or i + 1 == trainer.get_total_epochs():
            for task in validation_zeroshot_tasks:
                curr_line, curr_dict = evaluate_zero_shot_task( task, batcher )
                validation_lines.append( curr_line )
                validation_dict.update( **curr_dict )
            torch.cuda.empty_cache()

        # If we're not in debug mode log the metrics etc to the output dir
        if wandb_mode != 'disabled':
            log_stats( output_dir, train_metrics, validation_lines, trainer.optimizer_step )

        # Compute the running stats log
        stats_log = train_utils.compute_stats_dict( trainer, i ) # type: ignore

        # Log to WandB
        wandb.log( {
            **{ f'train/{name}': metric for name, metric in train_metrics.items() },
            **validation_dict,
            **stats_log,
            **aggregate_gpt4all_score( validation_dict, 'dph' ),
            **aggregate_glue_score( validation_dict, 'dph' ),
            **aggregate_race_score( validation_dict, 'dph' ),
            **aggregate_gpt4all_score( validation_dict, 'log' ),
            **aggregate_glue_score( validation_dict, 'log' ),
            **aggregate_race_score( validation_dict, 'log' ),
        } )

    # Cast the model to half, save the model and tokenizer
    dph_model.half().save_pretrained( output_dir )
    tokenizer.save_pretrained( output_dir )

    # Create artifact for the model
    model_artifact = wandb.Artifact( name=dph_model.config.model_type, type="model" )
    model_artifact.add_dir( output_dir )

    # Link the model artificat (if we're even a real run)
    assert wandb.run is not None
    wandb.run.log_artifact( model_artifact )

    # Aaand we're done!
    wandb.finish()

def run():
    argparser = argparse.ArgumentParser()

    # YAML config file(s) argument
    argparser.add_argument(
        '-c',
        '--config',
        type=lambda x: glob.glob( x, flags=glob.BRACE ),
        required=True
    )

    # WandB mode argument
    argparser.add_argument(
        '-w',
        '--wmode',
        default='disabled',
        choices=[ 'online', 'offline', 'disabled' ]
    )

    # Additional parameter(s) argument
    argparser.add_argument(
        '--params',
        type=train_utils.parse_options
    )

    # Parse the command line args
    arguments = argparser.parse_args()

    # Parse the YAML file(s)
    config = train_utils.parse_yaml_config( arguments.config )

    # If params are passed, update config
    if arguments.params is not None:
        config.update( arguments.params )

    # Add a UUID to run name
    config[ 'meta.run_name' ] += f'_{shortuuid.uuid()[:4]}'

    # If we have a UUID collision throw an error. TODO: maybe try and fix the collision instead?
    if os.path.exists( f"./checkpoints/{config['meta.run_name']}" ):
        raise ValueError( f"Cannot create run '{config['meta.run_name']}' because it already exists!" )

    # Check we're using a valid finetune mode
    if config[ 'finetune.mode' ] not in [ 'dph' ]:
        raise ValueError( "finetune.mode must be 'dph'" )

    # Add the finetune mode to the tags list
    tags = [ f"finetune_{config[ 'finetune.mode' ]}" ]

    # If we have other tags, add them to the list
    if 'meta.tags' in config:
        tags += config[ 'meta.tags' ]

    # Print the config to stdout
    rich.print( config )

    # Launch program with our settings
    instruct_align(
        config=config,
        wandb_mode=arguments.wmode,
        wandb_tags=tags,
        wandb_run_name=config[ 'meta.run_name' ],
    )

if __name__ == '__main__':
    run()

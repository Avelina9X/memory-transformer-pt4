""" Module for SFT optimization. """

import json
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

from training.trainer import Trainer

from training.data_instruct.task_base import BaseChoiceInstructDataset
from training.data_instruct.tasks import DIRECTORY_ALL, DIRECTORY_CHOICE
from training.data_instruct.formatter import InstructionFormatter
from training.data_instruct.task_loader import TaskList, ParallelMixedTaskLoader
from training.data_instruct.batcher import ChoiceInstructionBatcher

from model.configuration import LSWTConfigTraining, LSWTConfig
from model.modeling import LSWTForCausalLM

from constants import HF_CACHE_DIR, WANDB_API_KEY, WANDB_PROJECT_NAME
import train_utils

def log_full_config( output_dir: str, config: dict ):
    os.makedirs( output_dir, mode=0o777, exist_ok=True )

    json_file = os.path.join( output_dir, 'run_config.json' )
    json_str = json.dumps( config, indent=2 )

    with open( json_file, 'w', encoding="utf-8" ) as f:
        f.write( json_str + '\n' )

def log_stats( output_dir: str, train_metrics: dict, valid_list: list, step: int ):
    os.makedirs( output_dir, mode=0o777, exist_ok=True )

    log_file = os.path.join( output_dir, 'outputs.log' )

    with open( log_file, 'a', encoding="utf-8" ) as f:
        f.write( f'Step={step}\n' )
        f.write( f'Train={train_metrics}\n' )

        for line in valid_list:
            f.write( f'{line}\n' )

        f.write( '\n' )

def create_train_tasks( sft_mix: list ) -> TaskList:
    sft_mix = [
        ( i[0].split( '/' ), i[1], i[2] )
        for i in sft_mix
    ]

    return [
        ( DIRECTORY_ALL[task[0]][task[1]]( HF_CACHE_DIR ), fewshot_n, fewshot_allsys )
        for task, fewshot_n, fewshot_allsys in sft_mix
    ]

def create_validation_zeroshot_tasks() -> list[BaseChoiceInstructDataset]:
    return [
        # Hellaswag multiple choice
        DIRECTORY_CHOICE[ 'hellaswag' ][ 'choice' ]( HF_CACHE_DIR ),

        # GPT 4 ALL
        DIRECTORY_CHOICE[ 'hellaswag' ][ 'no_choice' ]( HF_CACHE_DIR ),
        DIRECTORY_CHOICE[ 'obqa' ][ 'main' ]( HF_CACHE_DIR ),
        DIRECTORY_CHOICE[ 'winogrande' ][ 'no_choice' ]( HF_CACHE_DIR ),
        DIRECTORY_CHOICE[ 'arc' ][ 'challenge' ]( HF_CACHE_DIR ),
        DIRECTORY_CHOICE[ 'arc' ][ 'easy' ]( HF_CACHE_DIR ),
        DIRECTORY_CHOICE[ 'super_glue' ][ 'boolq' ]( HF_CACHE_DIR ),
        DIRECTORY_CHOICE[ 'piqa' ][ 'no_choice' ]( HF_CACHE_DIR ),

        # Glue Benchmark
        DIRECTORY_CHOICE[ 'glue' ][ 'cola' ]( HF_CACHE_DIR ),
        DIRECTORY_CHOICE[ 'glue' ][ 'mnli_matched' ]( HF_CACHE_DIR ),
        DIRECTORY_CHOICE[ 'glue' ][ 'mnli_mismatched' ]( HF_CACHE_DIR ),
        DIRECTORY_CHOICE[ 'glue' ][ 'mrpc' ]( HF_CACHE_DIR ),
        DIRECTORY_CHOICE[ 'glue' ][ 'qnli' ]( HF_CACHE_DIR ),
        DIRECTORY_CHOICE[ 'glue' ][ 'qqp' ]( HF_CACHE_DIR ),
        DIRECTORY_CHOICE[ 'glue' ][ 'rte' ]( HF_CACHE_DIR ),
        DIRECTORY_CHOICE[ 'glue' ][ 'sst2' ]( HF_CACHE_DIR ),
        DIRECTORY_CHOICE[ 'glue' ][ 'stsb' ]( HF_CACHE_DIR ),
        DIRECTORY_CHOICE[ 'glue' ][ 'wnli' ]( HF_CACHE_DIR ),

        # Race multiple choice
        DIRECTORY_CHOICE[ 'race' ][ 'middle' ]( HF_CACHE_DIR ),
        DIRECTORY_CHOICE[ 'race' ][ 'high' ]( HF_CACHE_DIR ),
    ]

def create_validation_fewshot_tasks() -> list[BaseChoiceInstructDataset]:
    return [
        DIRECTORY_CHOICE[ 'mmlu' ][ 'all' ]( HF_CACHE_DIR )
    ]

def aggregate_gpt4all_score( metrics: dict[ str, float ] ) -> dict[ str, float ]:
    macro_scores = [
        metrics[ 'Hellaswag/no_choice/accuracy' ],
        metrics[ 'obqa/main/accuracy' ],
        metrics[ 'winogrande/no_choice/accuracy' ],
        metrics[ 'arc/ARC-Easy/accuracy' ],
        metrics[ 'arc/ARC-Challenge/accuracy' ],
        metrics[ 'super_glue/boolq/accuracy' ],
        metrics[ 'piqa/no_choice/accuracy' ],
    ]
    return { 'validation/gpt4all': 100 * sum( macro_scores ) / len( macro_scores ) }

def aggregate_glue_score( metrics: dict[ str, float ] ) -> dict[ str, float ]:
    macro_scores = [
        metrics[ 'GLUE/cola/matthews_correlation' ],
        ( metrics[ 'GLUE/mnli_matched/accuracy' ] + metrics[ 'GLUE/mnli_mismatched/accuracy' ] ) / 2,
        ( metrics[ 'GLUE/mrpc/accuracy' ] + metrics[ 'GLUE/mrpc/f1' ] ) / 2,
        metrics[ 'GLUE/qnli/accuracy' ],
        ( metrics[ 'GLUE/qqp/accuracy' ] + metrics[ 'GLUE/qqp/f1' ] ) / 2,
        metrics[ 'GLUE/rte/accuracy' ],
        metrics[ 'GLUE/sst2/accuracy' ],
        ( metrics[ 'GLUE/stsb/pearson' ] + metrics[ 'GLUE/stsb/spearmanr' ] ) / 2,
    ]

    macro_scores_wnli = [
        *macro_scores,
        metrics[ 'GLUE/wnli/accuracy' ],
    ]

    return {
        'validation/glue_all': 100 * sum( macro_scores_wnli ) / len( macro_scores_wnli ),
        'validation/glue_no_wnli': 100 * sum( macro_scores ) / len( macro_scores ),
    }

def aggregate_race_score( metrics: dict[ str, float ] ) -> dict[ str, float ]:
    macro_scores = [
        metrics[ 'race/middle/accuracy' ] * 1.44,
        metrics[ 'race/high/accuracy' ] * 3.45,
    ]
    return {
        'validation/race_avg': 100 * sum( macro_scores ) / ( 1.44 + 3.45 )
    }

def aggregate_mmlu_score( metrics: dict[ str, float ] ) -> dict[ str, float ]:
    macro_scores = [
        metrics[ 'MMLU/all/ZS/accuracy' ],
        metrics[ 'MMLU/all/FS/accuracy' ],
    ]
    return { 'validation/mmlu_avg': 100 * sum( macro_scores ) / len( macro_scores ) }

def evaluate_zero_shot_task( task: BaseChoiceInstructDataset, batcher: ChoiceInstructionBatcher ) -> tuple[ str, dict[ str, float ] ]:
    task_ds = task.get_validation_docs()
    assert task_ds is not None
    val_metrics = batcher.evaluate_dataset( task, task_ds, False, False )

    task_name = f'{task.task_name}/{task.task_subset}'
    curr_metrics = f'{task_name}={val_metrics}'
    rich.print( curr_metrics )

    return (
        curr_metrics,
        train_utils.compute_validation_metric_dict( val_metrics, task_name )
    )

def instruct_tune(
    config: dict,
    wandb_mode: str | None = None,
    wandb_tags: list[str] | None = None,
    wandb_run_name: str | None = None
):
    """ Runs the DPH optimization pipeline.

    Args:
        config (dict): Aggregated config of all sub-dicts (meta, model, train, finetune)
        wandb_mode (str | None): WandB run mode. Defaults to None.
        wandb_tags (list[str] | None): WandB run tags. Defaults to None.
        wandb_run_name (str | None): WandB run name. Defaults to None.
    """

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
    train_utils.modify_dicts( config, model_config, train_config )

    # Load model and set trainable
    model = typing.cast( LSWTForCausalLM, LSWTForCausalLM.from_pretrained( pretrained_run_dir, **model_config.to_dict() ) )
    train_utils.set_backbone_trainable( model, config[ 'finetune.trainable_backbone' ] )
    model = typing.cast( LSWTForCausalLM, model.cuda() ) # type: ignore # pylance is confused

    # Mask out parameters
    if 'finetune.frozen_params' in config:
        frozen_list = train_utils.set_training_mask( model, config[ 'finetune.frozen_params' ] )
        rich.print( 'Frozen params:' )
        rich.print( frozen_list )
        print()

    # Load tokenizer and add new segment tokens
    tokenizer = AutoTokenizer.from_pretrained( model_config.parent_embeddings, use_fast=True, cache_dir=HF_CACHE_DIR )
    train_utils.add_special_tokens( tokenizer )

    # Create task mixes
    train_tasks = create_train_tasks( config[ 'finetune.sft_mix' ] )
    validation_zeroshot_tasks = create_validation_zeroshot_tasks()

    # Instantiate instruct helpers
    formatter = InstructionFormatter( tokenizer )
    batcher = ChoiceInstructionBatcher( model, formatter, 'mean' )

    # Get mask type for this training variant
    mask_type = config.get( 'finetune.mask_override', {
        'vocab': 'train',
        'sft': 'train',
    }[ config[ 'finetune.mode' ] ] )

    # Create dataset
    # TODO: add support for multitask vs mixedtask training
    task_loader = ParallelMixedTaskLoader(
        task_list=train_tasks,
        formatter=formatter,
        seq_length=train_config.length_sequence,
        batch_size=train_config.batch_size,
        mask_type=mask_type,
    )

    # Instantiate trainer for finetuning
    trainer = Trainer( train_config, model, tokenizer, None )

    # Print out our configs
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
        group='finetuning',
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

            validation_dict.update( {
                **aggregate_gpt4all_score( validation_dict ),
                **aggregate_glue_score( validation_dict ),
                **aggregate_race_score( validation_dict ),
            } )

        if wandb_mode != 'disabled':
            log_stats( output_dir, train_metrics, validation_lines, trainer.optimizer_step )

        train_log = train_utils.compute_metric_dict( train_metrics, 'train' )
        stats_log = train_utils.compute_stats_dict( trainer, i )

        wandb.log( {
            **train_log,
            **stats_log,
            **validation_dict,
        } )


    model.half().save_pretrained( output_dir )
    tokenizer.save_pretrained( output_dir )

    model_artifact = wandb.Artifact( name=model.config.model_type, type="model" )
    model_artifact.add_dir( output_dir )

    assert wandb.run is not None
    wandb.run.log_artifact( model_artifact )

    wandb.finish()

def run():
    """ Runs the SFT optimization pipeline using command-line arguments.

    TODO: add support for passing commands via method call.

    Raises:
        ValueError: Thrown when a run name collision occurs (local only)
        ValueError: Thrown when an invalid finetuning mode is passed
    """

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
        type=train_utils.parse_options,
        help='Key value pairs to overwrite config parameters. Uses format `<key>:<value>,<key>:<value>,...`'
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
    if config[ 'finetune.mode' ] not in [ 'vocab', 'sft' ]:
        raise ValueError( "finetune.mode must be 'vocab' or 'sft'" )

    # Add the finetune mode to the tags list
    tags = [ f"finetune_{config[ 'finetune.mode' ]}" ]

    # If we have other tags, add them to the list
    if 'meta.tags' in config:
        tags += config[ 'meta.tags' ]

    # Print the config to stdout
    rich.print( config )

    # Launch program with our settings
    instruct_tune(
        config=config,
        wandb_mode=arguments.wmode,
        wandb_tags=tags,
        wandb_run_name=config[ 'meta.run_name' ],
    )

if __name__ == '__main__':
    run()

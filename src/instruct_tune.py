""" Module for SFT optimization. """

from datetime import timedelta
import os
import typing
import math

import shortuuid
import rich
import wandb

import binpacking as bp

import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
from torch.multiprocessing.queue import Queue as QueueType

import evaluate
import datasets
import transformers
from transformers import AutoTokenizer

from training.trainer import Trainer, TrainerDDP

from training.data_instruct.task_base import BaseChoiceInstructDataset
from training.data_instruct.tasks import DIRECTORY_CHOICE
from training.data_instruct.formatter import InstructionFormatter
from training.data_instruct.task_loader import ParallelMixedTaskLoader
from training.data_instruct.batcher import ChoiceInstructionBatcher

from model.configuration import LSWTConfigTraining, LSWTConfig
from model.modeling import LSWTForCausalLM

from constants import GRADIENTS_AS_BUCKET_VIEW, HF_CACHE_DIR, WANDB_API_KEY, WANDB_PROJECT_NAME
import train_utils
from train_utils import ddp_cleanup, ddp_setup, DDPModelWrapper

def create_validation_zeroshot_tasks( n_bins: int ) -> list[list[BaseChoiceInstructDataset]]:
    tasks = [
        # Auxiliary datasets
        DIRECTORY_CHOICE[ 'super_glue' ][ 'copa' ]( HF_CACHE_DIR ),
        DIRECTORY_CHOICE[ 'sciq' ][ 'no_choice' ]( HF_CACHE_DIR ),
        DIRECTORY_CHOICE[ 'truthful_qa' ][ 'mc1' ]( HF_CACHE_DIR ),
        DIRECTORY_CHOICE[ 'mmlu' ][ 'all' ]( HF_CACHE_DIR ),
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
    
    tasks_dict = { i: len( task.get_validation_docs() or [] ) for i, task in enumerate( tasks ) }
    
    idx_bins = bp.to_constant_bin_number( tasks_dict, n_bins )
    
    sorted_bins = []
    
    for curr_bin in idx_bins:
        assert isinstance( bin, dict )
        
        curr_tasks = [ tasks[idx] for idx in curr_bin.keys() ]
        curr_tasks.sort( key=lambda x: len( x.get_validation_docs() or [] ) ) # `or []` is a hack for the linter
        sorted_bins.append( curr_tasks )
    
    sorted_bins.sort( key=lambda x: sum( len( i.get_validation_docs() or [] ) for i in x ) )
    
    return sorted_bins

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


def evaluate_zero_shot_task(
    task: BaseChoiceInstructDataset,
    batcher: ChoiceInstructionBatcher,
    zero_nan=False,
) -> tuple[ str, dict[ str, float ] ]:
    """ Evaluates a task and returns the logp metrics.

    Args:
        task (BaseChoiceInstructDataset): Task to evaluate. Must have a validation split.
        batcher (DPHChoiceInstructionBatcher): Batcher used to evaluate tasks.
        zero_nan (bool): When true returns zero for NaN metrics. Defaults to False.

    Returns:
        log line (str): string suitable for file logging.
        metrics (dict[str, float]): dict suitable for WandB logging.
    """
    
    task_ds = task.get_validation_docs()
    assert task_ds is not None
    val_metrics = batcher.evaluate_dataset( task, task_ds, False, False )
    
    if zero_nan:
        for key in val_metrics:
            value = val_metrics[key]
            val_metrics[key] = 0.0 if math.isnan( value ) else value

    task_name = f'{task.task_name}/{task.task_subset}'
    curr_metrics = f'{task_name}={val_metrics}'
    rich.print( curr_metrics )

    return (
        curr_metrics,
        train_utils.compute_validation_metric_dict( val_metrics, task_name )
    )

def instruct_tune(
    rank: int = 0,
    world_size: int = 1,
    config: dict | None = None,
    wandb_mode: str | None = None,
    wandb_tags: list[str] | None = None,
    wandb_run_name: str | None = None,
    validation_queue: QueueType | None = None,
):
    """ Runs the DPH optimization pipeline.

    Args:
        rank (int, optional): The DDP process rank. Defaults to 0.
        world_size (int, optional): The DDP world size. When 1 DDP is disabled. Defulats to 1.
        config (dict | None): Aggregated config of all sub-dicts (meta, model, train, finetune)
        wandb_mode (str | None): WandB run mode. Defaults to None.
        wandb_tags (list[str] | None): WandB run tags. Defaults to None.
        wandb_run_name (str | None): WandB run name. Defaults to None.
        validation_queue (Queue | None): Queue used to pass validation results back to rank 0. Defaults to None
    """
    
    # Ensure config is not none
    assert config
    
    # Ensure DDP stuff is/isn't there if DDP is/isn't enabled
    if world_size == 1:
        assert validation_queue is None
        assert rank == 0
    else:
        assert validation_queue
    
    # Make sure rank isn't larger than world size
    assert rank < world_size

    # Log in to wandb
    wandb.require( 'core' )
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
    
    # Setup ddp if world size is greater than 1
    if world_size > 1:
        ddp_setup( rank, world_size, timedelta( hours=1.0 ) )

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
    
    # Set generation config
    model.generation_config = train_utils.create_generation_config( tokenizer )

    # Create task mixes
    train_tasks = train_utils.create_train_tasks( config[ 'finetune.sft_mix' ] )
    
    validation_zeroshot_tasks = create_validation_zeroshot_tasks( world_size )
    
    if rank == 0:
        validation_prompts = train_utils.create_validation_prompts( tokenizer )

    # Instantiate instruct helpers
    train_formatter = InstructionFormatter( tokenizer, 0 )
    
    validation_formatter = InstructionFormatter( tokenizer, None )
    batcher = ChoiceInstructionBatcher( model, validation_formatter, 'mean' )

    # Get mask type for this training variant
    mask_type = config.get( 'finetune.mask_override', {
        'vocab': 'train',
        'sft': 'train',
    }[ config[ 'finetune.mode' ] ] )

    # Create dataset
    # TODO: add support for multitask vs mixedtask training
    task_loader = ParallelMixedTaskLoader(
        task_list=train_tasks,
        formatter=train_formatter,
        seq_length=train_config.length_sequence,
        batch_size=train_config.batch_size // world_size,
        mask_type=mask_type,
        micro_batch_size=train_config.batch_size // world_size,
        task_elbow=config.get( 'finetune.sft_mix_elbow', None ),
    )

    # Instantiate trainer for finetuning
    if world_size == 1:
        trainer = Trainer( train_config, model, tokenizer, None )
    else:
        model = DDPModelWrapper( model, device_ids=[ rank ], gradient_as_bucket_view=GRADIENTS_AS_BUCKET_VIEW )
        trainer = TrainerDDP( train_config, model, tokenizer, None, rank, world_size ) # type: ignore

    # Print out our configs
    if rank == 0:
        rich.print( trainer.train_config )
        rich.print( trainer.model.config )

    # Compute params
    params_total = sum( p.numel() for p in model.parameters() )
    params_trainable = sum( p.numel() for p in model.parameters() if p.requires_grad )
    params_non_trainable = sum( p.numel() for p in model.parameters() if not p.requires_grad )

    # Print parametes
    if rank == 0:
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
    if wandb_mode != 'disabled' and rank == 0:
        train_utils.log_full_config( output_dir, config )

    # Create iterator
    iterator = iter( task_loader.as_data_loader() )

    # Initialise WandB
    if rank == 0:
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
        true_sample_count = trainer.get_sequence_count()

        
        # Create empty validation metrics list
        validation_lines = []
        validation_dict = {}
        validation_prompt_dict = {}
        
        validate_freq = config.get( 'meta.validate_freq', 1 )
        should_validate = config[ 'meta.validate' ] and ( i % validate_freq == validate_freq - 1 )

        # If validation flag is set (or it's the last epoch) run validation
        if should_validate or i + 1 == trainer.get_total_epochs():
            for task in validation_zeroshot_tasks[ rank ]:
                curr_line, curr_dict = evaluate_zero_shot_task( task, batcher, True )
                validation_lines.append( curr_line )
                validation_dict.update( **curr_dict )
            torch.cuda.empty_cache()
            
            if rank == 0:
                validation_prompt_table = train_utils.perform_prompt_validation(
                    validation_prompts,
                    tokenizer,
                    model,
                )
                
                validation_prompt_dict[ 'validation_generations' ] = validation_prompt_table
            
            if validation_queue:
                if rank > 0:
                    validation_queue.put( ( validation_lines, validation_dict ) )
                else:
                    for _ in range( world_size - 1):
                        received = validation_queue.get()
                        
                        validation_lines += received[0]
                        validation_dict.update( **received[1] )

            if rank == 0:
                validation_dict.update( {
                    **aggregate_gpt4all_score( validation_dict ),
                    **aggregate_glue_score( validation_dict ),
                    **aggregate_race_score( validation_dict ),
                } )

        if rank == 0:
            if wandb_mode != 'disabled':
                train_utils.log_stats( output_dir, train_metrics, validation_lines, trainer.optimizer_step )

            train_log = train_utils.compute_metric_dict( train_metrics, 'train' )
            stats_log = train_utils.compute_stats_dict( trainer, i, true_sample_count )
            
            wandb.log( {
                **validation_prompt_dict,
                **train_log,
                **stats_log,
                **validation_dict,
            } )


    if rank == 0:
        model.half().save_pretrained( output_dir )
        tokenizer.save_pretrained( output_dir )

        model_artifact = wandb.Artifact( name=model.config.model_type, type="model" )
        model_artifact.add_dir( output_dir )

        assert wandb.run is not None
        wandb.run.log_artifact( model_artifact )

        wandb.finish()
    
    # Cleanup ddp if world size is greater than 1
    if world_size > 1:
        ddp_cleanup()

def run():
    """ Runs the SFT optimization pipeline using command-line arguments.

    TODO: add support for passing commands via method call.

    Raises:
        ValueError: Thrown when a run name collision occurs (local only)
        ValueError: Thrown when an invalid finetuning mode is passed
    """

    arguments, config = train_utils.parse_cmd_args()

    # Add a UUID to run name
    config[ 'meta.run_name' ] += f'_{shortuuid.uuid()[:4]}'

    # If we have a UUID collision throw an error. TODO: maybe try and fix the collision instead?
    if os.path.exists( f"./checkpoints/{config['meta.run_name']}" ):
        raise ValueError( f"Cannot create run '{config['meta.run_name']}' because it already exists!" )

    # Check we're using a valid finetune mode
    if config[ 'finetune.mode' ] not in [ 'vocab', 'sft' ]:
        raise ValueError( "finetune.mode must be 'vocab' or 'sft'" )

    # Add the finetune mode to the tags list
    tags = [ f"finetune_{config[ 'finetune.mode' ]}", torch.cuda.get_device_name(), *( arguments.tags or [] ) ]

    # If we have other tags, add them to the list
    if 'meta.tags' in config:
        tags += config[ 'meta.tags' ]

    # Print the config to stdout
    rich.print( config )
    
    device_count = torch.cuda.device_count()

    # Launch program with our settings
    if device_count == 1:
        instruct_tune(
            config=config,
            wandb_mode=arguments.wmode,
            wandb_tags=tags,
            wandb_run_name=config[ 'meta.run_name' ],
        )
    else:
        mp.set_start_method( 'spawn' )
        mp.spawn( # type: ignore
            fn=instruct_tune,
            args=(
                device_count,
                config,
                arguments.wmode,
                tags,
                config[ 'meta.run_name' ],
                Queue( device_count - 1 )
            ),
            nprocs=device_count,
            join=True,
        )

if __name__ == '__main__':
    run()

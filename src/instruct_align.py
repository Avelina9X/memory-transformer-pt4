""" Module for performing Direct Preference Head optimization. """

from datetime import timedelta
from collections.abc import Sequence
import os
import typing
import math

import shortuuid
import rich
import wandb

import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
from torch.multiprocessing.queue import Queue as QueueType

import evaluate
import datasets
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from training.trainer import DPHTrainer, DPHTrainerDDP

from training.data_instruct.task_base import BaseChoiceInstructDataset

from training.data_instruct.formatter import InstructionFormatter
from training.data_instruct.task_loader import DPHMultiTaskLoader
from training.data_instruct.batcher import DPHChoiceInstructionBatcher

from evaluation import evaluate as evaluate_fn

from model.configuration import LSWTConfigTraining, LSWTConfig, LSWTConfigTrainingDPH
from model.modeling import LSWTForCausalLM, LSWTForDPH, WrappedLSWTForDPH

from constants import GRADIENTS_AS_BUCKET_VIEW, HF_CACHE_DIR, WANDB_API_KEY, WANDB_PROJECT_NAME
import train_utils
from train_utils import ddp_cleanup, ddp_setup, DDPModelWrapper
from instruct_tune import create_validation_zeroshot_tasks

def evaluate_zero_shot_task(
    task: BaseChoiceInstructDataset,
    batcher: DPHChoiceInstructionBatcher,
    zero_nan=False,
    max_batch_size=8,
) -> tuple[str, dict[str, float]]:
    """ Evaluates a task and returns the logp and dph metrics.

    Args:
        task (BaseChoiceInstructDataset): Task to evaluate. Must have a validation split.
        batcher (DPHChoiceInstructionBatcher): Batcher used to evaluate tasks.
        zero_nan (bool): When true returns zero for NaN metrics. Defaults to False.

    Returns:
        log line (str): string suitable for file logging.
        metrics (dict[str, float]): dict suitable for WandB logging.
    """

    # Get the task dataset and assert it's not None
    task_ds = task.get_validation_docs()
    assert task_ds is not None

    # Get batch size estimate
    estimate_size = len( task.create_unlabelled_message_list( task_ds[0] ) )
    batch_count = max( 1, math.floor( max_batch_size / estimate_size ) )

    # Perform evaluation and aggregate into the logprob and dph metrics
    val_metrics_both = batcher.evaluate_dataset_batched( task, task_ds, False, False, batch_count )
    val_metrics_log = val_metrics_both[ 'log' ]
    val_metrics_dph = val_metrics_both[ 'dph' ]

    # Create an empty dict and populate with anotated versions of the base metrics
    val_metrics = {}
    val_metrics.update( { f'log_{name}': metric for name, metric in val_metrics_log.items() } )
    val_metrics.update( { f'dph_{name}': metric for name, metric in val_metrics_dph.items() } )

    if zero_nan:
        for key in val_metrics:
            value = val_metrics[key]
            val_metrics[key] = 0.0 if math.isnan( value ) else value

    # Format metric for logging and print
    task_name = f'{task.task_name}/{task.task_subset}'
    curr_metrics = f'{task_name}={val_metrics}'
    rich.print( curr_metrics )

    # Return logged string and annotated dict
    return (
        curr_metrics,
        train_utils.compute_validation_metric_dict( val_metrics, task_name )
    )

def aggregate_gpt4all_score(
    metrics: dict[str, float],
    prefix: str
) -> dict[str, float]:
    """ Computes the aggreagted GPT4All scores.

    Args:
        metrics (dict[str, float]): list of all validation metrics to aggregate from.
        prefix (str): prefix to select the metric type ('log' or 'dph')

    Returns:
        dict[str, float]: dict suitable for WandB logging.
    """

    # Create list of scores, selected by prefix
    macro_scores = [
        metrics[ f'Hellaswag/no_choice/{prefix}_accuracy' ],
        metrics[ f'obqa/main/{prefix}_accuracy' ],
        metrics[ f'winogrande/no_choice/{prefix}_accuracy' ],
        metrics[ f'arc/ARC-Easy/{prefix}_accuracy' ],
        metrics[ f'arc/ARC-Challenge/{prefix}_accuracy' ],
        metrics[ f'super_glue/boolq/{prefix}_accuracy' ],
        metrics[ f'piqa/no_choice/{prefix}_accuracy' ],
    ]

    # Compute aggregated score with specified prefix
    return { f'validation/{prefix}_gpt4all': 100 * sum( macro_scores ) / len( macro_scores ) }

def aggregate_glue_score(
    metrics: dict[str, float],
    prefix: str
) -> dict[str, float]:
    """ Computes the aggregated GLUE scores (both with and without WNLI).

    Args:
        metrics (dict[str, float]): list of all validation metrics to aggregate from.
        prefix (str): prefix to select the metric type ('log' or 'dph')

    Returns:
        dict[str, float]: dict suitable for WandB logging.
    """

    # Create list of scores, aggreagted by task and selected by prefix
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

    # Bolt on WNLI for extra calc
    macro_scores_wnli = [
        *macro_scores,
        metrics[ f'GLUE/wnli/{prefix}_accuracy' ],
    ]

    # Compute aggregated scores for both varients with specified prefix
    return {
        f'validation/{prefix}_glue_all': 100 * sum( macro_scores_wnli ) / len( macro_scores_wnli ),
        f'validation/{prefix}_glue_no_wnli': 100 * sum( macro_scores ) / len( macro_scores ),
    }

def aggregate_race_score(
    metrics: dict[str, float],
    prefix: str
) -> dict[str, float]:
    """ Computes the weighted RACE score average.

    Args:
        metrics (dict[str, float]): list of all validation metrics to aggregate from.
        prefix (str): prefix to select the metric type ('log' or 'dph')

    Returns:
        dict[str, float]: dict suitable for WandB logging.
    """

    # Create list of scores, weighted by dataset size and selected by prefix
    macro_scores = [
        metrics[ f'race/middle/{prefix}_accuracy' ] * 1.44,
        metrics[ f'race/high/{prefix}_accuracy' ] * 3.45,
    ]

    # Compute weighted aggregation with specified prefix
    return {
        f'validation/{prefix}_race_avg': 100 * sum( macro_scores ) / ( 1.44 + 3.45 )
    }

def instruct_align(
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
        config (dict): Aggregated config of all sub-dicts (meta, dph, model, train, finetune)
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

    output_dir = f'./checkpoints/{wandb_run_name}'

    validation_batch_size = config[ 'meta.validation_batch' ]

    if config.get( 'finetune.wrapped_model', None ) is None:
        # Get pretrained run name and checkpoint directory
        pretrained_run_name = config[ 'finetune.checkpoint' ]
        pretrained_run_dir = f'./checkpoints/{pretrained_run_name}'

        # Grab configs
        model_config = typing.cast( LSWTConfig, LSWTConfig.from_pretrained( pretrained_run_dir, torch_dtype=None ) )
        train_config = LSWTConfigTraining()
        dph_config = LSWTConfigTrainingDPH()
        train_utils.modify_dicts( config, model_config, train_config, dph_config )

        # Get reward head name
        assert model_config.pooler_config
        assert isinstance( model_config.pooler_config.reward_heads, Sequence )
        assert len( model_config.pooler_config.reward_heads ) > 0
        reward_head_name = model_config.pooler_config.reward_heads[0]

        # Load DPH model and set trainable
        dph_model = typing.cast( LSWTForDPH, LSWTForDPH.from_pretrained( pretrained_run_dir, config=model_config ) )
        train_utils.set_backbone_trainable( dph_model, config[ 'finetune.trainable_backbone' ] )
        dph_model = typing.cast( LSWTForDPH, dph_model.cuda() ) # type: ignore # pylance is confused

        # Mask out parameters
        if 'finetune.frozen_params' in config:
            frozen_list = train_utils.set_training_mask( dph_model, config[ 'finetune.frozen_params' ] )

            if rank == 0:
                if __debug__:
                    rich.print( 'Frozen params:' )
                    rich.print( frozen_list )
                    print()
                else:
                    rich.print( f'Frozen param count: {len(frozen_list)}' )
                    print()

        # Load reference model and set trainable
        if dph_config.requires_reference_model:
            ref_model = typing.cast( LSWTForCausalLM, LSWTForCausalLM.from_pretrained( pretrained_run_dir, **model_config.to_dict() ) )
            ref_model.requires_grad_( False )
            ref_model = typing.cast( LSWTForCausalLM, ref_model.half().eval().cuda() ) # type: ignore # pylance is confused
        else:
            ref_model = dph_model

        # Load tokenizer and add new segment tokens
        tokenizer = AutoTokenizer.from_pretrained( model_config.parent_embeddings, use_fast=True, cache_dir=HF_CACHE_DIR )
        train_utils.add_special_tokens( tokenizer )

        # Set generation config
        dph_model.generation_config = train_utils.create_generation_config( tokenizer )

    else:
        # Get the wrapped model name
        wrapped_model_name = config[ 'finetune.wrapped_model' ]
        pretrained_run_name = None

        # Get the wrapped model and config
        source_config = AutoConfig.from_pretrained( wrapped_model_name )
        source_model = AutoModelForCausalLM.from_pretrained(
            wrapped_model_name,
            cache_dir=HF_CACHE_DIR,
            torch_dtype=None,
            _attn_implementation='flash_attention_2',
            output_hidden_states=True,
        ).cuda().eval()

        # Create fake model config from the wrapped config
        model_config = LSWTConfig(
            n_layers=0,
            n_heads=1,

            d_model=source_config.hidden_size,
            d_ffn=source_config.intermediate_size,
        )

        # Create train and dph configs and update them
        train_config = LSWTConfigTraining()
        dph_config = LSWTConfigTrainingDPH()
        train_utils.modify_dicts( config, model_config, train_config, dph_config )

        # Get reward head name
        assert model_config.pooler_config
        assert isinstance( model_config.pooler_config.reward_heads, Sequence )
        assert len( model_config.pooler_config.reward_heads ) > 0
        reward_head_name = model_config.pooler_config.reward_heads[0]

        # Instantiate wrapped model with the inner model
        dph_model = typing.cast( WrappedLSWTForDPH, WrappedLSWTForDPH( model_config, source_model ).cuda() ) # type: ignore
        dph_model.config.use_bfloat16 = source_config.torch_dtype == torch.bfloat16


        # Mask out parameters
        if 'finetune.frozen_params' in config:
            frozen_list = train_utils.set_training_mask( dph_model, config[ 'finetune.frozen_params' ] )

            if rank == 0:
                if __debug__:
                    rich.print( 'Frozen params:' )
                    rich.print( frozen_list )
                    print()
                else:
                    rich.print( f'Frozen param count: {len(frozen_list)}' )
                    print()

        tokenizer = AutoTokenizer.from_pretrained( wrapped_model_name, cache_dir=HF_CACHE_DIR )

        if not tokenizer.cls_token_id:
            tokenizer.cls_token_id = tokenizer.get_vocab()[ '<|im_end|>' ]

        if not tokenizer.sep_token_id:
            tokenizer.sep_token_id = tokenizer.get_vocab()[ '<|im_start|>' ]

        if not tokenizer.bos_token_id:
            tokenizer.bos_token_id = tokenizer.eos_token_id

        # Set generation config
        dph_model.generation_config = train_utils.create_generation_config( tokenizer )

        # Load reference model and set trainable
        if dph_config.requires_reference_model:
            ref_model = AutoModelForCausalLM.from_pretrained(
                wrapped_model_name,
                cache_dir=HF_CACHE_DIR,
                torch_dtype=source_config.torch_dtype,
                _attn_implementation='flash_attention_2',
                output_hidden_states=False,
            ).cuda().to( dtype=torch.bfloat16 if dph_model.config.use_bfloat16 else torch.float16 ).eval().requires_grad_( False )
        else:
            ref_model = dph_model
        ref_model.config.use_bfloat16 = dph_model.config.use_bfloat16

        model_config.pooler_config.prefix_sizes[ 'system' ] = len( tokenizer.encode( '<|im_start|>system\n', add_special_tokens=False ) )
        model_config.pooler_config.prefix_sizes[ 'user' ] = len( tokenizer.encode( '<|im_start|>user\n', add_special_tokens=False ) )
        model_config.pooler_config.prefix_sizes[ 'assistant' ] = len( tokenizer.encode( '<|im_start|>user\n', add_special_tokens=False ) )

    # Create task mixes
    train_tasks = train_utils.create_train_tasks( config[ 'finetune.dph_mix' ] )

    # Get the validation tasks
    validation_zeroshot_tasks = create_validation_zeroshot_tasks( world_size )

    # If we're on rank zero we get validation prompts
    if rank == 0 and config.get( 'meta.prompt_validate', False ):
        validation_prompts = train_utils.create_validation_prompts( tokenizer )
    else:
        validation_prompts = None

    # Instantiate instruct helpers
    train_formatter = InstructionFormatter( tokenizer, 0 )

    # Get helpers for validation
    validation_formatter = InstructionFormatter( tokenizer, None )
    batcher = DPHChoiceInstructionBatcher( dph_model, validation_formatter, reward_head_name, 'mean' )

    # Get mask type for this training variant
    mask_type = config.get( 'finetune.mask_override', {
        'dph': 'train',
    }[ config[ 'finetune.mode' ] ] )

    # Create dataset
    task_loader = DPHMultiTaskLoader(
        task_list=train_tasks,
        formatter=train_formatter,
        seq_length=train_config.length_sequence,
        batch_size=train_config.batch_size // world_size,
        mask_type=mask_type,
        task_elbow=config.get( 'finetune.dph_mix_elbow', None )
    )

    # Instantiate trainer for alignment
    if world_size == 1:
        trainer = DPHTrainer(
            train_config,
            dph_config,
            ref_model,
            dph_model,
            reward_head_name,
            tokenizer,
            task_loader
        )
    else:
        dph_model = DDPModelWrapper( dph_model, device_ids=[ rank ], gradient_as_bucket_view=GRADIENTS_AS_BUCKET_VIEW )
        trainer = DPHTrainerDDP(
            train_config,
            dph_config,
            ref_model,
            dph_model, # type: ignore
            reward_head_name,
            tokenizer,
            task_loader,
            rank,
            world_size
        )

    # Print out our configs
    if rank == 0:
        rich.print( 'LSWTConfigTraining =', trainer.train_config.__dict__ )
        rich.print( 'LSWTConfigTrainingDPH =', trainer.dph_config.__dict__ )
        rich.print( 'LSWTConfig =', trainer.model_dph.config.to_diff_dict() )

    # Compute params
    params_total = sum( p.numel() for p in dph_model.parameters() )
    params_trainable = sum( p.numel() for p in dph_model.parameters() if p.requires_grad )
    params_non_trainable = sum( p.numel() for p in dph_model.parameters() if not p.requires_grad )

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
        **dph_config.to_wandb_dict(),
        'params.total': params_total,
        'params.trainable': params_trainable,
        'params.non_trainable': params_non_trainable,
    } )

    # Log base config
    if wandb_mode != 'disabled' and rank == 0:
        train_utils.log_full_config( output_dir, config )

    # Create iterator
    iterator = iter( task_loader.as_data_loader() )

    # Initialise WandB on rank zero
    if rank == 0:
        wandb.init(
            project=WANDB_PROJECT_NAME,
            group='alignment',
            mode=wandb_mode,
            config=config,
            tags=wandb_tags,
            name=wandb_run_name,
            settings=wandb.Settings( _disable_stats=True )
        )

        # If we're not using a wrapped model we link the artifact
        if config.get( 'finetune.wrapped_model', None ) is None and pretrained_run_name is not None:
            input_artifact = train_utils.get_model_artifact( pretrained_run_name )
            wandb.use_artifact( input_artifact )

    # Train loop
    for i in range( trainer.get_total_epochs() ):
        # Train for an epoch and get metrics
        train_metrics = trainer.train_epoch( iterator, i + 1 )

        # Create empty validation metrics list
        validation_lines = []
        validation_dict = {}
        validation_prompt_dict = {}

        validate_freq = config.get( 'meta.validate_freq', 1 )
        should_validate = config[ 'meta.validate' ] and ( i % validate_freq == validate_freq - 1 )

        # If validation flag is set (or it's the last epoch) run validation
        if should_validate or i + 1 == trainer.get_total_epochs():
            for task in validation_zeroshot_tasks[ rank ]:
                curr_line, curr_dict = evaluate_zero_shot_task( task, batcher, zero_nan=True, max_batch_size=validation_batch_size )
                validation_lines.append( curr_line )
                validation_dict.update( **curr_dict )
            torch.cuda.empty_cache()

            # If rank is zero and prompt validate is enabled do prompt validation
            if validation_prompts is not None:
                validation_prompt_table = train_utils.perform_prompt_validation(
                    validation_prompts,
                    tokenizer,
                    dph_model,
                )

                validation_prompt_dict[ 'validation_generations' ] = validation_prompt_table

            # If there's a validation queue, gather all metrics
            if validation_queue:
                # If rank is 1 or above send metrics to rank 0
                if rank > 0:
                    validation_queue.put( ( validation_lines, validation_dict ) )

                # Otherwise gather all metrics to rank zero
                else:
                    for _ in range( world_size - 1):
                        received = validation_queue.get()

                        validation_lines += received[0]
                        validation_dict.update( **received[1] )

            # If rank is zero update the validation dict with log and dph scores
            if rank == 0:
                validation_dict.update( {
                    **aggregate_gpt4all_score( validation_dict, 'dph' ),
                    **aggregate_glue_score( validation_dict, 'dph' ),
                    **aggregate_race_score( validation_dict, 'dph' ),
                    **aggregate_gpt4all_score( validation_dict, 'log' ),
                    **aggregate_glue_score( validation_dict, 'log' ),
                    **aggregate_race_score( validation_dict, 'log' ),
                } )

        # If rank is zero log all stats and metrics
        if rank == 0:
            # If we're not in debug mode log the metrics etc to the output dir
            if wandb_mode != 'disabled':
                train_utils.log_stats( output_dir, train_metrics, validation_lines, trainer.optimizer_step )

            # Compute the running stats log
            stats_log = train_utils.compute_stats_dict( trainer, i, None ) # type: ignore

            pooler_log = train_utils.compute_pooler_stats_dict( dph_model ) # type: ignore

            # Log to WandB
            wandb.log( {
                **pooler_log,
                **validation_prompt_dict,
                **{ f'train/{name}': metric for name, metric in train_metrics.items() },
                **stats_log,
                **validation_dict,
            } )

    if rank == 0:
        if config.get( 'finetune.wrapped_model', None ) is None:
            # Cast the model to half, save the model and tokenizer
            dph_model.half().save_pretrained( output_dir )
            tokenizer.save_pretrained( output_dir )

            # Create artifact for the model
            model_artifact = wandb.Artifact( name=dph_model.config.model_type, type="model" )
            model_artifact.add_dir( output_dir )

            # Link the model artificat (if we're even a real run)
            assert wandb.run is not None
            wandb.run.log_artifact( model_artifact )

            if config[ 'meta.evaluate' ]:
                evaluate_fn( output_dir, 'all' )

        # Aaand we're done!
        wandb.finish()

    # Cleanup ddp if world size is greater than 1
    if world_size > 1:
        ddp_cleanup()

def run():
    """ Runs the DPH optimization pipeline using command-line arguments.

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
    if config[ 'finetune.mode' ] not in [ 'dph' ]:
        raise ValueError( "finetune.mode must be 'dph'" )

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
        instruct_align(
            config=config,
            wandb_mode=arguments.wmode,
            wandb_tags=tags,
            wandb_run_name=config[ 'meta.run_name' ],
        )
    else:
        mp.set_start_method( 'spawn' )
        mp.spawn( # type: ignore
            fn=instruct_align,
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

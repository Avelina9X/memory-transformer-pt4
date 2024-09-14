""" Module for learning attributes for Direct Preference Head models """

from datetime import timedelta
from collections.abc import Sequence
import os
import typing

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
from transformers import AutoTokenizer

from training.data_instruct.tasks import DIRECTORY_STEER
from training.trainer import SteerTrainer, SteerTrainerDDP

from training.data_instruct.formatter import SteerInstructionFormatter
from training.data_instruct.task_loader import SteerTaskLoader
from training.data_instruct.batcher import SteerInstructionBatcher

from evaluation import evaluate as evaluate_fn

from model.configuration import LSWTConfigTraining, LSWTConfig, LSWTConfigTrainingSteer
from model.modeling import LSWTForCausalLM, LSWTForDPH

from constants import GRADIENTS_AS_BUCKET_VIEW, HF_CACHE_DIR, WANDB_API_KEY, WANDB_PROJECT_NAME
import train_utils
from train_utils import ddp_cleanup, ddp_setup, DDPModelWrapper

def instruct_steer(
    rank: int = 0,
    world_size: int = 1,
    config: dict | None = None,
    wandb_mode: str | None = None,
    wandb_tags: list[str] | None = None,
    wandb_run_name: str | None = None,
    validation_queue: QueueType | None = None,
):
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
    
    # Get pretrained run name and checkpoint directory
    pretrained_run_name = config[ 'finetune.checkpoint' ]
    pretrained_run_dir = f'./checkpoints/{pretrained_run_name}'

    # Grab configs
    model_config = typing.cast( LSWTConfig, LSWTConfig.from_pretrained( pretrained_run_dir, torch_dtype=None ) )
    train_config = LSWTConfigTraining()
    steer_config = LSWTConfigTrainingSteer()
    train_utils.modify_dicts( config, model_config, train_config, steer_config=steer_config )

    # Get reward head name
    assert model_config.pooler_config
    assert isinstance( model_config.pooler_config.reward_heads, Sequence )
    assert len( model_config.pooler_config.reward_heads ) > 0
    reward_heads = model_config.pooler_config.reward_heads
    label_keys = steer_config.label_keys
    
    if not set( label_keys ).issubset( set( reward_heads ) ):
        raise ValueError( 'label_keys contains heads not included in the model\'s reward_heads' )

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
    if steer_config.requires_reference_model:
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
    
    # Create training task
    train_task_name, train_task_subset = config[ 'finetune.steer_task' ].split( '/' )
    train_task = DIRECTORY_STEER[train_task_name][train_task_subset]( HF_CACHE_DIR )
    
    validation_task = DIRECTORY_STEER[ 'HelpSteer' ][ '2' ]( HF_CACHE_DIR )
    
    train_formatter = SteerInstructionFormatter(
        tokenizer,
        0,
        min_trainable_tokens=steer_config.num_probes,
        max_total_tokens=train_config.length_sequence
    )
    
    validation_formatter = SteerInstructionFormatter( tokenizer, None )
    validation_batcher = SteerInstructionBatcher(
        dph_model,
        validation_formatter,
        label_keys=validation_task.get_available_labels()
    )
    
    # Get mask type for this training variant
    mask_type = config.get( 'finetune.mask_override', {
        'steer': 'train',
    }[ config[ 'finetune.mode' ] ] )
    
    task_loader = SteerTaskLoader(
        task=train_task,
        formatter=train_formatter,
        batch_size=train_config.batch_size // world_size,
        mask_type=mask_type,
        num_probes=steer_config.num_probes,
        labels=list( label_keys ),
    )
    
    if world_size == 1:
        trainer = SteerTrainer(
            train_config,
            steer_config,
            ref_model,
            dph_model,
            task_loader
        )
    else:
        dph_model = DDPModelWrapper( dph_model, device_ids=[ rank ], gradient_as_bucket_view=GRADIENTS_AS_BUCKET_VIEW )
        trainer = SteerTrainerDDP(
            train_config,
            steer_config,
            ref_model,
            dph_model, # type: ignore
            task_loader,
            rank,
            world_size
        )
        
    
    # Print out our configs
    if rank == 0:
        rich.print( 'LSWTConfigTraining =', trainer.train_config.__dict__ )
        rich.print( 'LSWTConfigTrainingSteer =', trainer.steer_config.__dict__ )
        rich.print( 'LSWTConfig =', trainer.model_dph.config.to_diff_dict() )
    
    # Compute params
    params_total = sum( p.numel() for p in dph_model.parameters() )
    params_trainable = sum( p.numel() for p in dph_model.parameters() if p.requires_grad )
    params_non_trainable = sum( p.numel() for p in dph_model.parameters() if not p.requires_grad )

    # Print parameters
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
        **steer_config.to_wandb_dict(),
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
            group='steer',
            mode=wandb_mode,
            config=config,
            tags=wandb_tags,
            name=wandb_run_name,
            settings=wandb.Settings( _disable_stats=True )
        )
        
        if config.get( 'finetune.wrapped_model', None ) is None:
            input_artifact = train_utils.get_model_artifact( pretrained_run_name )
            wandb.use_artifact( input_artifact )
    
    for i in range( trainer.get_total_epochs() ):
        # Train for an epoch and get metrics
        train_metrics = trainer.train_epoch( iterator, i + 1 )
        
        validation_lines = []
        validation_dict = {}
        
        validate_freq = config.get( 'meta.validate_freq', 1 )
        should_validate = config[ 'meta.validate' ] and ( i % validate_freq == validate_freq - 1 )
        
        # If validation flag is set (or it's the last epoch) run validation
        if should_validate or i + 1 == trainer.get_total_epochs():
            if rank == 0:
                task_ds = validation_task.get_validation_docs()
                assert task_ds is not None
                
                val_metrics_all = validation_batcher.evaluate_dataset( validation_task, task_ds )
                
                task_name = f'{validation_task.task_name}/{validation_task.task_subset}'
                
                for label_name in validation_task.get_available_labels():
                    full_name = f'{task_name}/{label_name}'
                    curr_dict = val_metrics_all[ label_name ]
                    curr_line = f'{full_name}={curr_dict}'
                    
                    validation_lines.append( curr_line )
                    rich.print( curr_line )
                    
                    validation_dict.update(
                        **train_utils.compute_validation_metric_dict( curr_dict, full_name )
                    )
            
        if rank == 0:
            # If we're not in debug mode log the metrics etc to the output dir
            if wandb_mode != 'disabled':
                train_utils.log_stats( output_dir, train_metrics, validation_lines, trainer.optimizer_step )
            
            # Compute the running stats log
            stats_log = train_utils.compute_stats_dict( trainer, i, None ) # type: ignore
            
            pooler_log = train_utils.compute_pooler_stats_dict( dph_model ) # type: ignore
            
            wandb.log( {
                **pooler_log,
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
    """ Runs the Steer optimization pipeline using command-line arguments.

    TODO: add support for passing commands via method call.

    Raises:
        ValueError: Thrown when a run name collision occurs (local only)
        ValueError: Thrown when an invalid finetuning mode is passed
    """

    arguments, config = train_utils.parse_cmd_args()
    
    print( config )

    # Add a UUID to run name
    config[ 'meta.run_name' ] += f'_{shortuuid.uuid()[:4]}'

    # If we have a UUID collision throw an error. TODO: maybe try and fix the collision instead?
    if os.path.exists( f"./checkpoints/{config['meta.run_name']}" ):
        raise ValueError( f"Cannot create run '{config['meta.run_name']}' because it already exists!" )

    # Check we're using a valid finetune mode
    if config[ 'finetune.mode' ] not in [ 'steer' ]:
        raise ValueError( "finetune.mode must be 'steer'" )

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
        instruct_steer(
            config=config,
            wandb_mode=arguments.wmode,
            wandb_tags=tags,
            wandb_run_name=config[ 'meta.run_name' ],
        )
    else:
        mp.set_start_method( 'spawn' )
        mp.spawn( # type: ignore
            fn=instruct_steer,
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

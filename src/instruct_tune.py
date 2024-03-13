import copy
import os
import argparse
import typing

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
from training.data_instruct.tasks import mmlu, race, glue, alpaca
from training.data_instruct.formatter import InstructionFormatter
from training.data_instruct.task_loader import TaskList, ParallelMixedTaskLoader
from training.data_instruct.batcher import ChoiceInstructionBatcher

from model.configuration import LSWTConfigTraining, LSWTConfig
from model.modeling import LSWTForCausalLM

from constants import HF_CACHE_DIR
import train_utils

def create_train_tasks( fewshot_n=5 ) -> TaskList:
    return [
        ( alpaca.AlpacaInstructDataset( cache_dir=HF_CACHE_DIR ), fewshot_n, True ),
        ( race.RaceInstructDataset( cache_dir=HF_CACHE_DIR, split='middle' ), fewshot_n, False ),
        ( race.RaceInstructDataset( cache_dir=HF_CACHE_DIR, split='high' ), fewshot_n, False ),
        ( mmlu.MMLUInstructDataset( cache_dir=HF_CACHE_DIR ), fewshot_n, False ),
        ( glue.GlueColaInstructDataset( cache_dir=HF_CACHE_DIR ), fewshot_n, False ),
        ( glue.GlueMNLIInstructDataset( cache_dir=HF_CACHE_DIR ), fewshot_n, False ),
        ( glue.GlueMRPCInstructDataset( cache_dir=HF_CACHE_DIR ), fewshot_n, False ),
        ( glue.GlueQNLIInstructDataset( cache_dir=HF_CACHE_DIR ), fewshot_n, False ),
        ( glue.GlueQQPInstructDataset( cache_dir=HF_CACHE_DIR ), fewshot_n, False ),
        ( glue.GlueRTEInstructDataset( cache_dir=HF_CACHE_DIR ), fewshot_n, False ),
        ( glue.GlueSST2InstructDataset( cache_dir=HF_CACHE_DIR ), fewshot_n, False ),
        ( glue.GlueWNLIInstructDataset( cache_dir=HF_CACHE_DIR ), fewshot_n, False ),
    ]

def create_validation_zeroshot_tasks() -> list[BaseChoiceInstructDataset]:
    return [
        glue.GlueColaInstructDataset( cache_dir=HF_CACHE_DIR ),
        glue.GlueMNLIMatchedInstructDataset( cache_dir=HF_CACHE_DIR ),
        glue.GlueMNLIMismatchedInstructDataset( cache_dir=HF_CACHE_DIR ),
        glue.GlueMRPCInstructDataset( cache_dir=HF_CACHE_DIR ),
        glue.GlueQNLIInstructDataset( cache_dir=HF_CACHE_DIR ),
        glue.GlueQQPInstructDataset( cache_dir=HF_CACHE_DIR ),
        glue.GlueRTEInstructDataset( cache_dir=HF_CACHE_DIR ),
        glue.GlueSST2InstructDataset( cache_dir=HF_CACHE_DIR ),
        glue.GlueWNLIInstructDataset( cache_dir=HF_CACHE_DIR ),
        race.RaceInstructDataset( 'middle', cache_dir=HF_CACHE_DIR ),
        race.RaceInstructDataset( 'high', cache_dir=HF_CACHE_DIR ),
    ]

def create_validation_fewshot_tasks() -> list[BaseChoiceInstructDataset]:
    return [
        mmlu.MMLUInstructDataset( cache_dir=HF_CACHE_DIR ),
    ]

def instruct_tune(
    config: dict,
    wandb_mode: str | None = None,
    wandb_tags: list[str] | None = None,
    wandb_run_name: str | None = None
):
    # Set some performance flags
    torch.backends.cuda.matmul.allow_tf32 = True # type: ignore # pylint: disable=W0212
    torch.backends.cudnn.allow_tf32 = True # type: ignore # pylint: disable=W0212
    torch._dynamo.config.cache_size_limit = 1024 * 1024 * 1024 # type: ignore # pylint: disable=W0212

    if not __debug__:
        transformers.utils.logging.disable_progress_bar()
        evaluate.utils.logging.disable_progress_bar()
        datasets.utils.logging.disable_progress_bar()

    # Get pretrained run name and checkpoint directory
    pretrained_run_name = config[ 'finetune.checkpoint' ]
    pretrained_run_dir = f'./checkpoints/{pretrained_run_name}'

    # Grab configs
    model_config = typing.cast( LSWTConfig, LSWTConfig.from_pretrained( pretrained_run_dir, torch_dtype=None ) )
    train_config = LSWTConfigTraining()
    train_utils.modify_dicts( config, model_config, train_config )

    # Load model and set trainable
    model = typing.cast( LSWTForCausalLM, LSWTForCausalLM.from_pretrained( pretrained_run_dir, **model_config.to_dict() ) )
    train_utils.set_backbone_trainable( model, config[ 'finetune.trainable_backbone' ] )
    model = typing.cast( LSWTForCausalLM, model.cuda() ) # type: ignore # pylance is confused

    # Load tokenizer and add new segment tokens
    tokenizer = AutoTokenizer.from_pretrained( model_config.parent_embeddings, use_fast=True, cache_dir=HF_CACHE_DIR )
    train_utils.add_special_tokens( tokenizer )

    # Instantiate trainer for finetuning
    trainer = Trainer( train_config, model, tokenizer, None )

    # Create task mixes
    train_tasks = create_train_tasks()
    validation_zeroshot_tasks = create_validation_zeroshot_tasks()
    validation_fewshot_tasks = create_validation_fewshot_tasks()

    # Instantiate instruct helpers
    formatter = InstructionFormatter( tokenizer )
    batcher = ChoiceInstructionBatcher( model, formatter, 'mean' )

    # Create dataset
    mask_type = {
        'vocab': 'train',
        'sft': 'train',
        'sft_dpo': 'test',
    }[ config[ 'finetune.mode' ] ]

    task_loader = ParallelMixedTaskLoader(
        task_list=train_tasks,
        formatter=formatter,
        seq_length=train_config.length_sequence,
        batch_size=train_config.batch_size,
        mask_type=mask_type,
    )

    rich.print( trainer.train_config )
    rich.print( trainer.model.config )

    # TODO: add support for multitask vs mixedtask training

    # Create iterator
    iterator = iter( task_loader.as_data_loader() )

    for i in range( trainer.get_total_epochs() ):
        train_metrics = trainer.train_epoch( iterator, i + 1 )

        for task in validation_zeroshot_tasks:
            task_ds = task.get_validation_docs()
            assert task_ds is not None

            val_metrics = batcher.evaluate_dataset(
                task=task,
                dataset=task_ds,
                fewshot=False,
                fewshot_allsys=False
            )

            rich.print( f'{task.task_name}/{task.task_subset}={val_metrics}' )

        for task in validation_fewshot_tasks:
            task_ds = task.get_validation_docs()
            assert task_ds is not None

            val_zs_metrics = batcher.evaluate_dataset(
                task=task,
                dataset=task_ds,
                fewshot=False,
                fewshot_allsys=False
            )

            rich.print( f'{task.task_name}/{task.task_subset}/ZS={val_zs_metrics}' )

            val_fs_metrics = batcher.evaluate_dataset(
                task=task,
                dataset=task_ds,
                fewshot=True,
                fewshot_allsys=False
            )

            rich.print( f'{task.task_name}/{task.task_subset}/FS={val_fs_metrics}' )

    model.half().save_pretrained( f'./checkpoints/{wandb_run_name}' )

def run():
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '-c',
        '--config',
        type=lambda x: glob.glob( x, flags=glob.BRACE ),
        required=True
    )

    argparser.add_argument(
        '-w',
        '--wmode',
        default='disabled',
        choices=[ 'online', 'offline', 'disabled' ]
    )

    arguments = argparser.parse_args()

    config = train_utils.parse_yaml_config( arguments.config )

    rich.print( config )

    if config[ 'finetune.mode' ] not in [ 'vocab', 'sft', 'dpo_sft', 'dpo' ]:
        raise ValueError( "finetune.mode must be 'vocab', 'sft', 'dpo_sft' or 'dpo'" )

    assert arguments.wmode == 'disabled'

    instruct_tune(
        config=config,
        wandb_mode=arguments.wmode,
        wandb_tags=[ f"fineutne_{config[ 'finetune.mode' ]}" ],
        wandb_run_name=config[ 'meta.run_name' ],
    )

if __name__ == '__main__':
    run()

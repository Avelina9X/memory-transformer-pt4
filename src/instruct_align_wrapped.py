from collections.abc import Sequence
import os
import argparse
import typing

import rich
import wandb

import torch

import evaluate
import datasets
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from training.trainer import DPHTrainer

from training.data_instruct.task_base import BaseChoiceInstructDataset, BaseInstructDataset
from training.data_instruct.tasks import DIRECTORY_ALL
from training.data_instruct.formatter import InstructionFormatter
from training.data_instruct.task_loader import DPHMultiTaskLoader
from training.data_instruct.batcher import DPHChoiceInstructionBatcher

from model.configuration import LSWTConfigTraining, LSWTConfig, LSWTConfigTrainingDPH
from model.modeling import LSWTForCausalLM, LSWTForDPH

from constants import HF_CACHE_DIR

from instruct_align import (
    create_align_tasks,
    evaluate_zero_shot_task,
    aggregate_gpt4all_score,
    aggregate_glue_score,
    aggregate_race_score,

    create_validation_zeroshot_tasks,
    log_stats
)

def instruct_align( model_name: str ):
    # Load the source model
    source_tokenizer = AutoTokenizer.from_pretrained( model_name, cache_dir=HF_CACHE_DIR )

    if not source_tokenizer.cls_token_id:
        source_tokenizer.cls_token_id = source_tokenizer.get_vocab()[ '<|im_end|>' ]

    if not source_tokenizer.bos_token_id:
        source_tokenizer.bos_token_id = source_tokenizer.eos_token_id

    source_config = AutoConfig.from_pretrained( model_name )
    source_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=HF_CACHE_DIR,
        torch_dtype=source_config.torch_dtype,
        device_map='cuda',
        output_hidden_states=True,

    ).eval().requires_grad_( False )

    # Create the wrapped model config
    wrapped_config = LSWTConfig(
        reward_pooler='swiglu_bert',
        reward_dropout=0.1,
        reward_heads=[ 'reward_head' ],

        n_layers=0,
        n_heads=1,

        d_model=source_config.hidden_size,
        d_ffn=source_config.intermediate_size,
    )

    # Create monkey-patch class
    class WrappedForDPH( LSWTForDPH ):
        def forward( self, *args, **kwargs ):
            return source_model( *args, **kwargs )

    # Wrap the model
    wrapped_model = WrappedForDPH( wrapped_config ).cuda()

    # Create train config
    train_config = LSWTConfigTraining(
        batch_size=64,
        batch_size_step=8,
        batches_per_epoch=512,
        length_sequence=2048,
        length_cache=2048,
        lr_cooldown_tokens=1000000000,
        lr_warmup_steps=200,
        lr_max=3.0e-6,
        opt_max_grad_norm=1.0,

        opt_weight_decay=0.5,
        opt_decay_init=True,
        opt_decay_mask=[],
    )

    # Create DPH config
    dph_config = LSWTConfigTrainingDPH(
        dpo_enabled=False,

        dph_contrastive=False,
        dph_epsilon=0.1,
        dph_weight=1.0,
        dph_decay_init=False,
        dph_weight_decay=0.1,
    )

    # Create the DPH mix
    dph_mix = [
        'glue/cola',
        'glue/mnli',
        'glue/mrpc',
        'glue/qnli',
        'glue/qqp',
        'glue/rte',
        'glue/sst2',
        'glue/stsb',
        'mmlu/all',
        'race/all',
        'hellaswag/choice',
        'hellaswag/no_choice',
        'squad/v2',
        'obqa/main',
        'winogrande/no_choice',
        'arc/challenge',
        'arc/easy',
        'super_glue/boolq',
        'piqa/no_choice',
        'ultrafeedback/binarized',
        'orca/orca_dpo_pairs',
    ]

    # Create task mixes
    train_tasks = create_align_tasks( dph_mix )
    validation_zeroshot_tasks = create_validation_zeroshot_tasks()

    # Instantiate instruct helpers
    formatter = InstructionFormatter( source_tokenizer )
    batcher = DPHChoiceInstructionBatcher( wrapped_model, formatter, 'reward_head', 'mean' )

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
        wrapped_model,
        wrapped_model,
        'reward_head',
        source_tokenizer,
        task_loader
    )

    output_dir = f'./checkpoints/{model_name}'

    # Create iterator
    iterator = iter( task_loader.as_data_loader() )

    for i in range( trainer.get_total_epochs() ):
        # Train for an epoch and get metrics
        train_metrics = trainer.train_epoch( iterator, i + 1 )

        # Create empty validation metrics list
        validation_lines = []
        validation_dict = {}

        # If validation flag is set (or it's the last epoch) run validation
        if True:
            for task in validation_zeroshot_tasks:
                curr_line, curr_dict = evaluate_zero_shot_task( task, batcher, True )
                validation_lines.append( curr_line )
                validation_dict.update( **curr_dict )
            torch.cuda.empty_cache()

            score_dict = {
                **aggregate_gpt4all_score( validation_dict, 'dph' ),
                **aggregate_glue_score( validation_dict, 'dph' ),
                **aggregate_race_score( validation_dict, 'dph' ),
                **aggregate_gpt4all_score( validation_dict, 'log' ),
                **aggregate_glue_score( validation_dict, 'log' ),
                **aggregate_race_score( validation_dict, 'log' ),
            }

            validation_lines += [
                f'{key}={value}' for key, value in score_dict.items()
            ]

        log_stats( output_dir, train_metrics, validation_lines, trainer.optimizer_step )

if __name__ == '__main__':
    instruct_align( 'Qwen/Qwen1.5-0.5B' )
    torch.cuda.empty_cache()

    instruct_align( 'Qwen/Qwen1.5-0.5B-Chat' )
    torch.cuda.empty_cache()

    instruct_align( 'Qwen/Qwen1.5-1.8B' )
    torch.cuda.empty_cache()

    instruct_align( 'Qwen/Qwen1.5-1.8B-Chat' )
    torch.cuda.empty_cache()

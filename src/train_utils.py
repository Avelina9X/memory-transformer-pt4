""" Utilities and helper functions for training and pretraining """

from collections.abc import Sequence, Mapping
from datetime import timedelta
import json
import os
import pathlib
import yaml
import argparse

import wandb
from wcmatch import glob
import numpy as np
import torch
import torch.distributed as dist

from transformers import PreTrainedTokenizerBase, GenerationConfig
from datasets import concatenate_datasets

from model.configuration import LSWTConfigTraining, LSWTConfig, LSWTConfigTrainingDPH, LSWTPoolerConfig
from model.modeling import LSWTForCausalLM, LSWTForDPH
from training.trainer import Trainer
from training.data import load_awesome_prompts, load_savvas_prompts
from training.data_instruct.tasks import DIRECTORY_ALL
from training.data_instruct.task_loader import TaskList
from constants import WANDB_PROJECT_NAME, HF_CACHE_DIR

def find_and_extract( source: str, prefix: str ) -> str | None:
    """ Given a wandb config string and target prefix, return suffix

    Args:
        source (str): wandb style string in format `"prefx.suffix"`
        prefix (str): prefix to match

    Returns:
        str | None: suffix if found, else None
    """
    idx = source.find( prefix + '.' )
    if idx != 0:
        return None
    return source[ len( prefix ) + 1 : ]


def modify_dicts( config: dict, model_config: LSWTConfig, train_config: LSWTConfigTraining, dph_config: LSWTConfigTrainingDPH | None = None ):
    """ Updates Model and Training config given a wandb dict

    Args:
        config (dict): wandb dict source
        model_config (LSWTConfig): model config destination
        train_config (LSWTConfigTraining): train config destination
        dph_config (LSWTConfigTrainingDPH | None): dph config destination
    """
    for key, value in config.items():
        model_key = find_and_extract( key, 'model' )
        train_key = find_and_extract( key, 'train' )
        dph_key = find_and_extract( key, 'dph' )

        if model_key is not None:
            if model_key != 'pooler_config':
                getattr( model_config, model_key )
                setattr( model_config, model_key, value )
            else:
                assert isinstance( value, dict )
                
                if model_config.pooler_config:
                    for dk, dv in value.items():
                        getattr( model_config.pooler_config, dk )
                        setattr( model_config.pooler_config, dk, dv )
                else:
                    setattr( model_config, 'pooler_config', LSWTPoolerConfig( **value ) )

        if train_key is not None:
            getattr( train_config, train_key )
            setattr( train_config, train_key, value )

        if dph_key is not None and dph_config is not None:
            getattr( dph_config, dph_key )
            setattr( dph_config, dph_key, value )


def get_checkpoint_path( name: str | None=None ) -> tuple[ pathlib.Path, pathlib.Path, pathlib.Path ]:
    """ Returns the directory, config path and weights path for saving a model.
    If no model name is given it will infer from the run name.
    If the run has no name it will infer from the run id.

    Args:
        name (str | None): Desired name of model, or None if infered from wanbd. Defaults to None.

    Returns:
        root_dir (Path): Path of enclosing directory
        config_dir (Path): Path of `config.json`
        model_dir (Path): Path of `model.safetensors`
    """
    name = name or wandb.run.name or wandb.run.id # type: ignore

    root_dir = pathlib.Path().cwd().joinpath( 'checkpoints', name )
    config_dir = root_dir.joinpath( 'config.json' )
    model_dir = root_dir.joinpath( 'model.safetensors' )
    return root_dir, config_dir, model_dir


def save_model( model: LSWTForCausalLM, log_wandb: bool=False ):
    """ Saves the model to disk and uploads as an artifact to wandb

    Args:
        model (LSWTForCausalLM): The model to save
        log_wandb (bool, optional): Sets if the model should be uploaded. Defaults to False.
    """
    root_dir, config_dir, model_dir = get_checkpoint_path()

    model.half().save_pretrained( root_dir, safe_serialization=True )

    if log_wandb:
        model_name = model.config.model_type

        model_artifact = wandb.Artifact( name=model_name, type="model" )
        model_artifact.add_file( model_dir ) # type: ignore
        model_artifact.add_file( config_dir ) # type: ignore

        wandb.run.log_artifact( model_artifact ) # type: ignore


def add_special_tokens( tokenizer: PreTrainedTokenizerBase ):
    """ Adds `<|im_start|>` and `<|im_end|>` to the tokenizer vocabulary

    Args:
        tokenizer (PreTrainedTokenizerBase): Target tokenizer to update
    """
    tokenizer.add_tokens( [ '<|im_start|>', '<|im_end|>' ], special_tokens=True )
    tokenizer.sep_token = '<|im_start|>'
    tokenizer.cls_token = '<|im_end|>'
    
    tokenizer.chat_template = (
        "{% for message in messages %}"
            "{% if loop.first %}"
                "{{bos_token}}"
            "{% endif %}"
            "{% if loop.first and messages[0]['role'] != 'system' %}"
                "{{ '<|im_start|>system\nYou are a conversational AI assistant. Write a response that appropriately completes the request.<|im_end|>\n' }}"
            "{% endif %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
    )

def get_model_artifact( run_name: str ):
    """ TODO: update code for multiple runs with same name """
    pretrain_run = wandb.Api().runs(
        path=WANDB_PROJECT_NAME,
        filters={ "display_name": run_name }
    )[0]

    return [
        artifact
        for artifact in pretrain_run.logged_artifacts()
        if artifact.type == 'model'
    ][0]


@torch.no_grad
def set_backbone_trainable( model: LSWTForCausalLM, trainable: bool ):
    """ Sets the trainable flag of decoder layers for a model.
    Note that if layers are set as non-trainable their weights will be cast to FP16.
    TODO: enable support for BF16

    Args:
        model (LSWTForCausalLM): The target model to update
        trainable (bool): the trainable flag
    """
    model.model.blocks.requires_grad_( trainable )
    if not trainable:
        model.model.blocks = model.model.blocks.half()


@torch.no_grad
def set_training_mask( model: torch.nn.Module, mask_patterns: Sequence[str] ) -> list[str]:
    """ Freezes parameters that match any mask pattern.
    If any parameter name contains a substring in the `mask_patterns` it will be frozen.

    Args:
        model (torch.nn.Module): The model to freeze parameters from.
        mask_patterns (Sequence[str]): Sequence of string masks.

    Returns:
        list[str]: list of frozen parameters
    """

    for name, p in model.named_parameters():
        if any( pattern in name for pattern in mask_patterns ):
            p.requires_grad_( False )

    return [ name for name, p in model.named_parameters() if not p.requires_grad ]


def compute_metric_dict( inputs: dict[str, float], name: str ) -> dict[str, float]:
    """ Given a dict of `loss` and `acc` keys, returns a wandb log dict of metrics

    Args:
        inputs (dict[str, float]): Dict contraining metrics
        name (str): Metric group name

    Returns:
        dict[str, float]: wandb log dict of metrics
    """
    return {
        f'{name}/ppl': np.exp( inputs[ 'loss' ] ),
        f'{name}/loss': inputs[ 'loss' ],
        f'{name}/acc': inputs[ 'acc' ],
    }

def compute_validation_metric_dict( inputs: dict[str, float], name: str ) -> dict[str, float]:
    """ Given a dict of arbitrary keys, returns a wandb log dict of metrics

    Args:
        inputs (dict[str, float]): Dict contraining metrics
        name (str): Metric group name

    Returns:
        dict[str, float]: wandb log dict of metrics
    """
    return {
        f'{name}/{key}' : value
        for key, value in inputs.items()
    }


def compute_stats_dict( trainer: Trainer, i: int, true_samples: int | None ) -> dict[str, float | int]:
    """ Given the trainer and current iteration returns a wandb log dict of stats

    Args:
        trainer (Trainer): Trainer used for training the model.
        i (int): Current iteration index. Should be zero indexed.
        true_samples (int | None): Number of true samples seen. If None, uses the number of batch samples.

    Returns:
        dict[str, float | int]: wandb log dict of stats
    """
    return {
        'stats/n_tokens': trainer.optimizer_step * trainer.train_config.batch_size * trainer.train_config.length_sequence,
        'stats/n_batches': trainer.optimizer_step,
        'stats/n_samples': trainer.optimizer_step * trainer.train_config.batch_size,
        'stats/n_epochs': i + 1,
        'stats/learning_rate': trainer.get_schedule() * trainer.train_config.lr_max,
        'stats/true_samples': true_samples or trainer.optimizer_step * trainer.train_config.batch_size,
    }

def compute_pooler_stats_dict( model: LSWTForDPH ):
    stats_dict = {}
    
    pooler_config: LSWTPoolerConfig = model.config.pooler_config
    
    if pooler_config.layer_pooling == 'weighted_sum':
        assert not isinstance( pooler_config.layer_select, int )
        assert model.pooler.layer_weighting is not None
        
        layer_weights = list( model.pooler.layer_weighting.detach().softmax( 0 ).mean( [ 1, 2, 3 ] ).cpu().numpy() )
        
        for i, layer_idx in enumerate( pooler_config.layer_select ):
            stats_dict[ f'pooler/layer_weights/{layer_idx}' ] = layer_weights[i]
        
    if pooler_config.token_pooling == 'ema':
        assert model.pooler.ema_beta is not None
        assert model.pooler.ema_weight is not None
        
        sigmoids = torch.sigmoid( model.pooler.ema_beta + model.pooler.ema_weight ).flatten()
        sigmoids = list( sigmoids.detach().cpu().numpy() )
        
        stats_dict[ 'pooler/ema/betas' ] = wandb.Histogram( sigmoids )
    
    if pooler_config.token_pooling_rotation:
        assert model.pooler.token_rotate is not None
        
        with torch.no_grad():
            Q = model.pooler.token_rotate.weight
            stats_dict[ 'pooler/rotate/ortho' ] = torch.dist( Q @ Q.T, torch.eye( Q.size( 0 ) ).to( device=Q.device ) ).item()
    
    return stats_dict
        

def parse_cmd_args() -> tuple[argparse.Namespace, dict]:
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
        type=parse_options,
        help='Key value pairs to overwrite config parameters. Uses format `<key>:<value>,<key>:<value>,...`'
    )
    
    # Additional tag(s) argument
    argparser.add_argument(
        '-t',
        '--tags',
        type=lambda s: s.split( ',' ),
        help='Comma seperated list of tags to add to the WandB run.'
    )

    # Parse the command line args
    arguments = argparser.parse_args()

    # Parse the YAML file(s)
    config = parse_yaml_config( arguments.config )

    # If params are passed, update config
    if arguments.params is not None:
        config.update( arguments.params )
    
    return arguments, config


def parse_yaml_config( files: list[str] ) -> dict:
    """ Given a list of YAML files, parses and returns a wandb config dict.
    Note that dict keys will be overriden by the most recent file in the list.

    Args:
        files (list[str]): List of YAML file paths

    Returns:
        dict: wandb config dict
    """
    def unpack_dict( d ):
        return {
            f'{outer_k}.{inner_k}' : inner_v
            for outer_k, outer_v in d.items()
            for inner_k, inner_v in outer_v.items()
        }
    
    def nested_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping):
                d[k] = nested_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    config = {}

    for file in files:
        if not os.path.isfile( file ):
            raise ValueError( f'Config file {file} does not exist!' )

        with open( file, 'r', encoding='utf-8' ) as f:
            obj = unpack_dict( yaml.load( f, yaml.FullLoader ) )
            config = nested_update( config, obj )

    return config

def parse_options( options: str ) -> dict:
    """ Parses strings in the format "<key>:<value>,<key>:<value>,.."

    Keys are stripped of whitespace, and values are parsed as YAML literals.

    Args:
        options (str): String of comma seperated KV pairs, delimited by colons.

    Returns:
        dict: The parsed dict.
    """
    if not options:
        return {}

    option_list = [ tuple( i.split( ':' ) ) for i in options.split( ',' ) ]
    return {
        key.strip(): yaml.load( value.strip(), yaml.FullLoader )
        for key, value in option_list
    }

def log_full_config( output_dir: str, config: dict ):
    """ Writes the full run config as JSON

    Args:
        output_dir (str): directory to create the `run_config.json` file
        config (dict): WandB style dictionary containing all config data
    """
    os.makedirs( output_dir, mode=0o777, exist_ok=True )

    json_file = os.path.join( output_dir, 'run_config.json' )
    json_str = json.dumps( config, indent=2 )

    with open( json_file, 'w', encoding="utf-8" ) as f:
        f.write( json_str + '\n' )

def log_stats( output_dir: str, train_metrics: dict, valid_list: list, step: int ):
    """ Creates log file and appends training and validation metrics

    Args:
        output_dir (str): directory of create the `outputs.log` file
        train_metrics (dict): train metrics dictionary
        valid_list (list): list of validation metric strings
        step (int): current optimizer step
    """
    os.makedirs( output_dir, mode=0o777, exist_ok=True )

    log_file = os.path.join( output_dir, 'outputs.log' )

    with open( log_file, 'a', encoding="utf-8" ) as f:
        f.write( f'Step={step}\n' )
        f.write( f'Train={train_metrics}\n' )

        for line in valid_list:
            f.write( f'{line}\n' )

        f.write( '\n' )

def ddp_setup( rank: int, world_size: int, timeout: timedelta | None = None ):
    """ Sets up the DDP process group and sets CUDA rank.
    Note this does not support multi-machine DDP, only multi-device DDP.

    Args:
        rank (int): current device rank
        world_size (int): global world size
        timeout (timedelta | None): timeout for waiting workers (should be large due to validation)
    """
    torch.cuda.set_device( rank )

    os.environ[ 'MASTER_ADDR' ] = 'localhost'
    os.environ[ 'MASTER_PORT' ] = '12355'

    dist.init_process_group( 'nccl', rank=rank, world_size=world_size, timeout=timeout )

def ddp_cleanup():
    """ Shuts down the DDP process group. """
    dist.destroy_process_group()

class DDPModelWrapper( torch.nn.parallel.DistributedDataParallel ):
    """ Custom DDP wrapper. Defers method and attribute accesses to underlying module. """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def create_train_tasks( sft_mix: dict[str, float] ) -> TaskList:
    """ Creates a list of tasks for finetuning from the sft mix list

    Args:
        sft_mix (dict[str, float]): List of SFT tasks in the correct format (TODO)

    Returns:
        TaskList: List of instantiated tasks.
    """
    sft_list = [
        ( key.split( '/' ), value )
        for key, value in sft_mix.items()
    ]

    return [
        ( DIRECTORY_ALL[task[0]][task[1]]( HF_CACHE_DIR ), weight )
        for task, weight in sft_list
    ]
    
def create_generation_config( tokenizer: PreTrainedTokenizerBase ) -> GenerationConfig:
    """ Creates the default geneartion config with Typical sampling for LSWT.

    Args:
        tokenizer (PreTrainedTokenizerBase): The tokenizer in use so we can extract the CLS and EOS tokens.

    Returns:
        GenerationConfig: The generation config.
    """
    
    config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=[
            tokenizer.cls_token_id,
            tokenizer.eos_token_id,
        ],
        
        do_sample=True,
        typical_p=0.95,
    )
    
    return config

def create_validation_prompts( tokenizer: PreTrainedTokenizerBase ): # TODO: refactor?
    def tokenize_msg( doc: dict ):
        return {
            'tokens': tokenizer.apply_chat_template(
                conversation=[ { 'role': 'user', 'content': doc[ 'prompt' ] } ],
                add_generation_prompt=True,
                tokenize=True,
            )
        }
        
    return concatenate_datasets( [
        load_awesome_prompts( HF_CACHE_DIR ), # type: ignore
        load_savvas_prompts()
    ] ).map( tokenize_msg )

def perform_prompt_validation( ds, tokenizer: PreTrainedTokenizerBase, model ): # TODO: refactor?
    with torch.inference_mode():
        model.eval()
        
        columns = [ 'Title', 'Prompt', 'Completion' ]
        data = []
        
        for doc in ds:
            title = doc[ 'act' ]
            prompt = doc[ 'prompt' ]
            tokens = doc[ 'tokens' ]
            
            with torch.autocast( device_type='cuda', dtype=torch.float16 ): # type: ignore
                response = tokenizer.decode(
                    model.generate(
                        inputs=torch.LongTensor( [ tokens ] ).cuda(),
                        use_cache=True,
                        max_new_tokens=256,
                    )[0, len( tokens ) : ].cpu(),
                    skip_special_tokens=True
                )
            
            data.append( [ title, prompt, response ] )
    
    torch.cuda.empty_cache()
    
    return wandb.Table( columns=columns, data=data )
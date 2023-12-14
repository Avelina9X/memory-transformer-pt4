"""
The LSWTransformer constants package.
Contains helpful constants used by other packages.

Contains:
    - WANDB_API_KEY: the WandB api key, loaded from an ENVAR
    - HF_API_KEY: the HF api key, loaded from an ENVAR
    - HF_CACHE_DIR: the directory for HF models and datasets, loaded from an ENVAR
    - TORCH_COMPILE_OPTIONS: options dict for `torch.compile`
"""

import os

WANDB_API_KEY = os.environ[ 'WANDB_API_KEY' ]
WANDB_PROJECT_NAME = os.environ[ 'WANDB_PROJECT_NAME' ]

HF_API_KEY = os.environ[ 'HF_API_KEY' ]
HF_CACHE_DIR = os.environ[ 'HF_CACHE_DIR' ]

TORCH_COMPILE_OPTIONS = {
    'options': {
        'triton.cudagraphs': False,
        'triton.cudagraph_trees': True,
        'triton.autotune_pointwise': True,
        'triton.autotune_cublasLt': True,
        'shape_padding': True,

        'max_autotune': True,
        'max_autotune_pointwise': True,
        'max_autotune_gemm': True,

        'allow_buffer_reuse': True,
        'epilogue_fusion': True,
    },
    'fullgraph': False,
}

OPTIMIZER_KWARGS = { # pylint: disable=R6101
    'AdamW' : {
        'train.optimizer': 'AdamW',
        'train.opt_weight_decay': 0.1,
        'train.opt_eps': 1e-5,
        'train.opt_beta_1': 0.9,
        'train.opt_beta_2': 0.95,
        'train.opt_rho': -1,
    },
    
    'Minato' : {
        'train.optimizer': 'Minato',
        'train.opt_weight_decay': 0.2,
        'train.opt_eps': -1,
        'train.opt_beta_1': 0.9,
        'train.opt_beta_2': -1,
        'train.opt_rho': 0.05,
    },
    
    'SophiaH' : {
        'train.optimizer': 'SophiaH',
        'train.opt_weight_decay': 0.2,
        'train.opt_eps': -1,
        'train.opt_beta_1': 0.9,
        'train.opt_beta_2': 0.95,
        'train.opt_rho': 0.05,
    }
}
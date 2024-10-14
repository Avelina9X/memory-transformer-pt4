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
    'disable': False,
    'dynamic': None,
}
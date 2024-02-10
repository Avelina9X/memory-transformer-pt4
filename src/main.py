""" Main module """

import os
# os.environ[ 'TORCH_LOGS' ] = '+dynamo'
# os.environ[ 'TORCHDYNAMO_VERBOSE' ] = '1'
# os.environ[ 'TORCH_LOGS' ] = 'recompiles'
# os.environ[ 'TORCHDYNAMO_REPORT_GUARD_FAILURES' ] = '1'

import argparse
import warnings
import logging

import torch
import torch.multiprocessing as mp

import wandb

from constants import WANDB_API_KEY, WANDB_PROJECT_NAME

from pretrain import train

if __name__ == '__main__':
    wandb.login( key=WANDB_API_KEY )

    parser = argparse.ArgumentParser()
    parser.add_argument( '--sweep-id', type=str, default=None )
    parser.add_argument( '--sweep-count', type=int, default=None )
    arguments = parser.parse_args()

    if arguments.sweep_id is not None:
        # Disable warnings
        warnings.simplefilter( 'ignore' )
        torch._logging.set_logs( dynamo=logging.FATAL ) # type: ignore # pylint: disable=W0212
        torch._logging.set_logs( inductor=logging.FATAL ) # type: ignore # pylint: disable=W0212
        torch._logging.set_logs( dynamic=logging.FATAL ) # type: ignore # pylint: disable=W0212

        # Run agent
        wandb.agent(
            sweep_id=arguments.sweep_id,
            function=train,
            project=WANDB_PROJECT_NAME,
            count=arguments.sweep_count,
        )

    else:
        
        # torch._logging.set_logs( dynamo=logging.ERROR ) # type: ignore # pylint: disable=W0212
        # torch._logging.set_logs( inductor=logging.ERROR ) # type: ignore # pylint: disable=W0212
        # torch._logging.set_logs( dynamic=logging.ERROR ) # type: ignore # pylint: disable=W0212
        
        REROPE_SCALE = 1
        
        wandb_mode = 'online'
        custom_config = {
            'model.trainable_embeddings': True,
            'model.rope_reversed': True,
            'model.d_model': 1536,
            'model.d_ffn': 4096,
            'model.n_heads': 24,
            'model.n_layers': 18,
            
            'train.batch_size': 480 * REROPE_SCALE,
            'train.batch_size_step': 2 * REROPE_SCALE,
            
            'train.length_sequence': 2048 // REROPE_SCALE,
            'train.length_cache': 2048 // REROPE_SCALE,
            
            'train.loss_objective': 'MLE',
            # 'train.loss_objective': 'SimCTG',
            
            # 'train.optimizer': 'Minato',
            # 'train.opt_weight_decay': 0.2,
            # 'train.opt_max_grad_norm': 1.0,
            # 'train.opt_beta_1': 0.9,
            
            # 'train.optimizer': 'AdamW',
            # 'train.opt_weight_decay': 0.1,
            # 'train.opt_max_grad_norm': 1.0,
            # 'train.opt_eps': 1e-5,
            # 'train.opt_beta_1': 0.9,
            # 'train.opt_beta_2': 0.95,
            
            'train.optimizer': 'LaProp',
            'train.opt_weight_decay': 0.1,
            'train.opt_max_grad_norm': 1.0,
            'train.opt_eps': 1e-8,
            'train.opt_beta_1': 0.9,
            'train.opt_beta_2': 0.95,
            
        }
        if torch.cuda.device_count() == 1:
            train( config=custom_config, wandb_mode=wandb_mode, tags=[ 'rerope_tests' ] )
        else:
            mp.spawn( # type: ignore
                fn=train,
                args=(
                    torch.cuda.device_count(),
                    custom_config,
                    None,
                    None,
                    wandb_mode,
                    [ 'rerope_tests', 'ddp' ]
                ),
                nprocs=torch.cuda.device_count(),
                join=True,
            )

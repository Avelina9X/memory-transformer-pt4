""" Main module """

import argparse
import warnings
import logging

import torch
import wandb

from constants import WANDB_API_KEY, WANDB_PROJECT_NAME

from pretrain import train

if __name__ == '__main__':
    wandb.login( key=WANDB_API_KEY )

    parser = argparse.ArgumentParser()
    parser.add_argument( '--sweep-id', type=str, default=None )
    parser.add_argument( '--sweep-count', type=int, default=None )
    arguments = parser.parse_args()

    torch._dynamo.config.cache_size_limit = 256 # type: ignore # pylint: disable=W0212

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
        custom_config = {
            'model.trainable_embeddings': True,
            'model.rope_reversed': True,
            
            'train.batch_size': 480,
            'train.batch_size_step': 6,
            
            'train.length_sequence': 2048,
            'train.length_cache': 2048,
            
            'train.loss_objective': 'MLE',
            
            'train.optimizer': 'AdamW',
            'train.opt_weight_decay': 0.1,
            'train.opt_max_grad_norm': 1.0,
            'train.opt_eps': 1e-5,
            'train.opt_beta_1': 0.9,
            'train.opt_beta_2': 0.95,
            
        }

        train( config=custom_config, wandb_mode='online' )

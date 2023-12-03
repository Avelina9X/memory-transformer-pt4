import argparse
import warnings

import torch
import wandb

from constants import WANDB_API_KEY, WANDB_PROJECT_NAME

from train import train

if __name__ == '__main__':
    wandb.login( key=WANDB_API_KEY )

    parser = argparse.ArgumentParser()
    parser.add_argument( '--sweep-id', type=str, default=None )
    parser.add_argument( '--sweep-count', type=int, default=None )
    arguments = parser.parse_args()

    torch._dynamo.config.cache_size_limit = 256 # type: ignore

    if arguments.sweep_id is not None:
        # Disable warnings
        warnings.simplefilter( 'ignore' )
        torch._logging.set_logs( dynamo=logging.FATAL ) # type: ignore
        torch._logging.set_logs( inductor=logging.FATAL ) # type: ignore
        torch._logging.set_logs( dynamic=logging.FATAL ) # type: ignore

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
            'train.loss_objective': 'SimCTG',
            'train.batches_per_epoch': 32,
        }

        train( config=custom_config, wandb_mode='offline' )
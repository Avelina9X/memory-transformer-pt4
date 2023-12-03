import os
import wandb

from constants import WANDB_API_KEY, WANDB_PROJECT_NAME

if __name__ == '__main__':
    wandb.login( key=WANDB_API_KEY )

    sweep_config = {
        'name': 'simctg-small-sweep',
        'metric': {
            'name': 'test/pile-uncopyrighted/loss',
            'goal': 'minimize',
        },
        'method': 'grid',
        'parameters': {
            'model.trainable_embeddings': { 'values': [ False, True ] },
            'train.loss_sim_margin': { 'values': [ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 ] },

            'train.batch_size_step': { 'value': 12 },
            'train.loss_objective': { 'value': 'SimCTG' },
            'train.batches_per_epoch': { 'value': 512 },
            'train.length_sequence': { 'value': 1024 },
            'train.length_cache': { 'value': 1024 },
            'train.lr_cooldown_tokens': { 'value': 5_000_000_000 },
        }
    }

    print( wandb.sweep( sweep_config, project=WANDB_PROJECT_NAME ) )
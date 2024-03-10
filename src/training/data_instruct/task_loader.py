from collections.abc import Iterable, Generator
import gc
import random

import torch
from torch.utils.data import IterableDataset, DataLoader

from .task_base import BaseInstructDataset
from .formatter import InstructionFormatter


def generate_forever( iterator: Iterable ) -> Generator[dict, None, None]:
    while True:
        for i in iterator:
            assert isinstance( i, dict )
            yield i
        gc.collect()

class TaskLoader( IterableDataset ):
    def __init__(
        self,
        task: BaseInstructDataset,
        formatter: InstructionFormatter,
        fewshot_count: int,
        fewshot_allsys: bool,
        seq_length: int,
        sub_batch_size: int,
        shuffle_seed: int | None,
        mask_type: str,
    ):
        """ Creates a task loader for a single task.

        Args:
            task (BaseInstructDataset): Task to load
            formatter (InstructionFormatter): Formatter to use
            fewshot_count (int): Number of examples per sequence
            fewshot_allsys (bool): If all message groups should contain a system message
            seq_length (int): Max tokens per sequence in batch
            sub_batch_size (int): Number of sequences per batch
            shuffle_seed (int | None): Dataset shuffling seed, or None for shuffling disabled
            mask_type (str): Must be 'all', 'train' or 'test'

        Raises:
            ValueError: If task does not have training docs
            ValueError: If `fewshow_count` is less than 1
            ValueError: If `mask_type` is not 'all', 'train' or 'test'
        """

        if not task.has_training_docs:
            raise ValueError( f'Task {task.group_name}:{task.task_name} has no training set.' )

        if fewshot_count < 1:
            raise ValueError( 'Fewshot count must be at least 1' )

        if mask_type not in [ 'all', 'train', 'test' ]:
            raise ValueError( 'mask_type must be `all`, `train` or `test`' )

        self.task = task
        self.formatter = formatter
        self.fewshot_count = fewshot_count
        self.fewshot_allsys = fewshot_allsys
        self.seq_length = seq_length
        self.sub_batch_size = sub_batch_size
        self.mask_type = mask_type

        dataset = self.task.get_training_docs()
        assert dataset is not None

        if shuffle_seed is not None:
            dataset = dataset.shuffle( shuffle_seed, keep_in_memory=True )

        self._dataset_shards = [
            dataset.shard( sub_batch_size, i, keep_in_memory=True )
            for i in range( sub_batch_size )
        ]

    def messages_generator( self, shard_idx: int ):
        iterator = iter( generate_forever( self._dataset_shards[shard_idx] ) )

        while True:
            message_list = [
                random.choice( self.task.create_target_message_list( next( iterator ) ) )
                for _ in range( self.fewshot_count )
            ]

            token_dict = self.formatter.tokenize_chat_training( message_list, self.fewshot_allsys )

            tokens = token_dict[ 'tokens' ]
            targets = token_dict[ 'targets' ]
            train_mask = token_dict[ 'train_mask' ]
            test_mask = token_dict[ 'test_mask' ]

            match self.mask_type:
                case 'all':
                    yield tokens, targets

                case 'train':
                    yield tokens, [ t if m else -100 for t, m in zip( targets, train_mask ) ]

                case 'test':
                    yield tokens, [ t if m else -100 for t, m in zip( targets, test_mask ) ]

    def token_generator( self, shard_idx: int ):
        for messages in self.messages_generator( shard_idx ):
            for x, y in zip( messages[0], messages[1] ):
                yield x, y

    def sub_batch_generator( self ):
        def reset():
            count = 0
            xs = []
            ys = []
            for _ in range( self.sub_batch_size ):
                xs.append( [] )
                ys.append( [] )

            return count, xs, ys

        count, xs, ys = reset()

        generators = [ iter( self.token_generator( i ) ) for i in range( self.sub_batch_size ) ]

        try:
            while True:
                for g_idx, generator in enumerate( generators ):
                    x, y = next( generator )
                    xs[ g_idx ].append( x )
                    ys[ g_idx ].append( y )
                count += 1

                if count == self.seq_length:
                    yield ( torch.LongTensor( xs ), torch.LongTensor( ys ) )

                    count, xs, ys = reset()
        except StopIteration:
            return

    def __iter__( self ):
        return iter( self.sub_batch_generator() )

    def as_data_loader( self ):
        return DataLoader(
            self,
            num_workers=1,
            batch_size=None,
            prefetch_factor=4,
        )

    def __getitem__( self, index ):
        raise NotImplementedError( "This dataset does not support random access using __getitem__" )

class MultiTaskLoader( IterableDataset ):
    def __init__(
        self,
        task_list: list[tuple[BaseInstructDataset, int, bool]],
        formatter: InstructionFormatter,
        seq_length: int,
        batch_size: int,
        shuffle_seed: int | None,
        mask_type: str,
    ):
        sub_batch_size = batch_size // len( task_list )

        if batch_size % len( task_list ) != 0:
            raise ValueError( 'batch size must be divisible by task count' )

        self.tasks = [
            TaskLoader(
                task=task,
                formatter=formatter,
                fewshot_count=fewshot_count,
                fewshot_allsys=fewshot_allsys,
                seq_length=seq_length,
                sub_batch_size=sub_batch_size,
                shuffle_seed=shuffle_seed,
                mask_type=mask_type
            )
            for task, fewshot_count, fewshot_allsys
            in task_list
        ]

    def __iter__( self ):
        gen = [
            iter( i.as_data_loader() ) for i in self.tasks
        ]

        try:
            while True:
                test_next = [ next( i ) for i in gen ]
                test_next_x = torch.cat( [ i[0] for i in test_next ] )
                test_next_y = torch.cat( [ i[1] for i in test_next ] )

                yield test_next_x, test_next_y
        except StopIteration:
            return

    def as_data_loader( self ):
        return DataLoader(
            self,
            num_workers=0,
            batch_size=None,
            pin_memory=True,
            pin_memory_device='cuda',
        )

    def __getitem__( self, index ):
        raise NotImplementedError( "This dataset does not support random access using __getitem__" )

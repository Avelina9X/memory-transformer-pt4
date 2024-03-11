from collections.abc import Iterable, Generator
import gc
import random
from typing import TypeAlias

import torch
from torch.utils.data import IterableDataset, DataLoader

from .task_base import BaseInstructDataset, MessageList
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
        shuffle_seed: int | bool,
        mask_type: str,
        shard_mode: str = 'shuffle',
    ):
        """ Creates a task loader for a single task.

        Args:
            task (BaseInstructDataset): Task to load
            formatter (InstructionFormatter): Formatter to use
            fewshot_count (int): Number of examples per sequence
            fewshot_allsys (bool): If all message groups should contain a system message
            seq_length (int): Max tokens per sequence in batch
            sub_batch_size (int): Number of sequences per batch
            shuffle_seed (int | bool): Dataset pre-shuffling seed, or True for random seed, or False for no shuffle.
            mask_type (str): Must be 'all', 'train' or 'test'
            shard_mode (str): 'split' uses exclusive sharding, 'shuffle' shards by shuffling. Defaults to 'shuffle'.

        Raises:
            ValueError: If task does not have training docs
            ValueError: If `fewshow_count` is less than 1
            ValueError: If `mask_type` is not 'all', 'train' or 'test'
            ValueError: If `shard_mode` is not 'split' or 'shuffle'
        """

        if not task.has_training_docs:
            raise ValueError( f'Task {task.group_name}:{task.task_name} has no training set.' )

        if fewshot_count < 1:
            raise ValueError( 'Fewshot count must be at least 1' )

        if mask_type not in [ 'all', 'train', 'test' ]:
            raise ValueError( 'mask_type must be `all`, `train` or `test`' )

        if shard_mode not in [ 'split', 'shuffle' ]:
            raise ValueError( 'shard_mode must be `split` or `shuffle`' )

        self.task = task
        self.formatter = formatter
        self.fewshot_count = fewshot_count
        self.fewshot_allsys = fewshot_allsys
        self.seq_length = seq_length
        self.sub_batch_size = sub_batch_size
        self.mask_type = mask_type
        self.shard_mode = shard_mode

        dataset = self.task.get_training_docs()
        assert dataset is not None

        if shuffle_seed is True:
            dataset = dataset.shuffle( None, keep_in_memory=True )
        elif shuffle_seed is not False:
            dataset = dataset.shuffle( shuffle_seed, keep_in_memory=True )

        match shard_mode:
            case 'split':
                self._dataset_shards = [
                    dataset.shard( sub_batch_size, i, keep_in_memory=True )
                    for i in range( sub_batch_size )
                ]
            case 'shuffle':
                self._dataset_shards = [
                    dataset.shuffle( i, keep_in_memory=True )
                    for i in range( sub_batch_size )
                ]

    def message_list_generator( self, shard_idx: int ) -> Generator[list[MessageList], None, None]:
        for elem in generate_forever( self._dataset_shards[shard_idx] ):
            yield [
                random.choice( self.task.create_target_message_list( elem ) )
                for _ in range( self.fewshot_count )
            ]

    def message_tokens_generator( self, shard_idx: int ) -> Generator[tuple[list[int], list[int]], None, None]:
        for message_list in self.message_list_generator( shard_idx ):
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

    def token_generator( self, shard_idx: int ) -> Generator[tuple[int, int], None, None]:
        for messages in self.message_tokens_generator( shard_idx ):
            for x, y in zip( messages[0], messages[1] ):
                yield x, y

    def sub_batch_generator( self ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        def reset():
            count, xs, ys = 0, [], []
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

TaskList: TypeAlias = list[tuple[BaseInstructDataset, int, bool]]

class MultiTaskLoader( IterableDataset ):
    def __init__(
        self,
        task_list: TaskList,
        formatter: InstructionFormatter,
        seq_length: int,
        batch_size: int,
        shuffle_seed: int | bool,
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

class MixedTaskLoader( IterableDataset ):
    def __init__(
        self,
        task_list: TaskList,
        formatter: InstructionFormatter,
        seq_length: int,
        batch_size: int,
        shuffle_seed: int | bool,
        mask_type: str,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.tasks = [
            TaskLoader(
                task=task,
                formatter=formatter,
                fewshot_count=fewshot_count,
                fewshot_allsys=fewshot_allsys,
                seq_length=seq_length,
                sub_batch_size=batch_size,
                shuffle_seed=shuffle_seed,
                mask_type=mask_type
            )
            for task, fewshot_count, fewshot_allsys
            in task_list
        ]

    def message_tokens_generator( self, shard_idx: int ) -> Generator[tuple[list[int], list[int]], None, None]:
        generators = [
            iter( task.message_tokens_generator( shard_idx ) )
            for task in self.tasks
        ]

        while True:
            yield next( random.choice( generators ) )

    def token_generator( self, shard_idx: int ) -> Generator[tuple[int, int], None, None]:
        for messages in self.message_tokens_generator( shard_idx ):
            for x, y in zip( messages[0], messages[1] ):
                yield x, y

    def sub_batch_generator( self ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        def reset():
            count, xs, ys = 0, [], []
            for _ in range( self.batch_size ):
                xs.append( [] )
                ys.append( [] )
            return count, xs, ys
        count, xs, ys = reset()
        generators = [ iter( self.token_generator( i ) ) for i in range( self.batch_size ) ]
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

class ParallelMixedTaskLoader( IterableDataset ):
    def __init__(
        self,
        task_list: TaskList,
        formatter: InstructionFormatter,
        seq_length: int,
        batch_size: int,
        mask_type: str,
    ):

        self.mixed_tasks = [
            MixedTaskLoader(
                task_list=task_list,
                formatter=formatter,
                seq_length=seq_length,
                batch_size=1,
                shuffle_seed=i,
                mask_type=mask_type,
            ) for i in range( batch_size )
        ]

    def __iter__( self ):
        gen = [
            iter( i.as_data_loader() ) for i in self.mixed_tasks
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

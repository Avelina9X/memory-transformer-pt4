from collections.abc import Iterable, Generator
import gc
import random
from typing import TypeAlias

import torch
from torch.utils.data import IterableDataset, DataLoader

from .task_base import BaseInstructDataset, BaseSteerInstructDataset, MessageList
from .formatter import InstructionFormatter, SteerInstructionFormatter


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
        seq_length: int,
        sub_batch_size: int,
        shuffle_seed: int | bool,
        mask_type: str,
        max_tokens: int | None = None,
        shard_mode: str = 'shuffle',
        sample_weight: float = 1.0,
    ):
        """ Creates a task loader for a single task.

        Args:
            task (BaseInstructDataset): Task to load
            formatter (InstructionFormatter): Formatter to use
            seq_length (int): Max tokens per sequence in batch
            sub_batch_size (int): Number of sequences per batch
            shuffle_seed (int | bool): Dataset pre-shuffling seed, or True for random seed, or False for no shuffle.
            mask_type (str): Must be 'all', 'train' or 'test'
            max_tokens (int | None): Max number of tokens in a conversation before dropping. Defaults to None.
            shard_mode (str): 'split' uses exclusive sharding, 'shuffle' shards by shuffling. Defaults to 'shuffle'.

        Raises:
            ValueError: If task does not have training docs
            ValueError: If `fewshow_count` is less than 1
            ValueError: If `mask_type` is not 'all', 'train' or 'test'
            ValueError: If `shard_mode` is not 'split' or 'shuffle'
        """

        if not task.has_training_docs:
            raise ValueError( f'Task {task.task_name}:{task.task_subset} has no training set.' )

        if mask_type not in [ 'all', 'train', 'test' ]:
            raise ValueError( 'mask_type must be `all`, `train` or `test`' )

        if shard_mode not in [ 'split', 'shuffle' ]:
            raise ValueError( 'shard_mode must be `split` or `shuffle`' )

        self.task = task
        self.formatter = formatter
        self.seq_length = seq_length
        self.sub_batch_size = sub_batch_size
        self.mask_type = mask_type
        self.max_tokens = max_tokens
        self.shard_mode = shard_mode
        

        dataset = self.task.get_training_docs()
        assert dataset is not None
        
        self.num_samples = len( dataset )
        self.sample_weight = sample_weight

        if shuffle_seed is True:
            dataset = dataset.shuffle( None )
        elif shuffle_seed is not False:
            dataset = dataset.shuffle( shuffle_seed )

        match shard_mode:
            case 'split':
                self._dataset_shards = [
                    dataset.shard( sub_batch_size, i )
                    for i in range( sub_batch_size )
                ]
            case 'shuffle':
                self._dataset_shards = [
                    dataset.shuffle( i )
                    for i in range( sub_batch_size )
                ]

    def message_list_generator( self, shard_idx: int ) -> Generator[MessageList, None, None]:
        iterator = iter( generate_forever( self._dataset_shards[shard_idx] ) )
        while True:
            yield random.choice( self.task.create_target_message_list( next( iterator ) ) )

    def message_tokens_generator( self, shard_idx: int ) -> Generator[tuple[list[int], list[int]], None, None]:
        for message_list in self.message_list_generator( shard_idx ):
            token_dict = self.formatter.tokenize_chat_training( message_list )

            tokens = token_dict[ 'tokens' ]
            targets = token_dict[ 'targets' ]
            train_mask = token_dict[ 'train_mask' ]
            test_mask = token_dict[ 'test_mask' ]

            if self.max_tokens:
                if len( tokens ) > self.max_tokens:
                    continue

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

TaskList: TypeAlias = list[tuple[BaseInstructDataset, float]]

class MultiTaskLoader( IterableDataset ):
    def __init__(
        self,
        task_list: TaskList,
        formatter: InstructionFormatter,
        seq_length: int,
        batch_size: int,
        shuffle_seed: int | bool,
        mask_type: str,
        max_tokens: int | None = None,
    ):
        sub_batch_size = batch_size // len( task_list )

        if batch_size % len( task_list ) != 0:
            raise ValueError( 'batch size must be divisible by task count' )

        self.tasks = [
            TaskLoader(
                task=task,
                formatter=formatter,
                seq_length=seq_length,
                sub_batch_size=sub_batch_size,
                shuffle_seed=shuffle_seed,
                mask_type=mask_type,
                max_tokens=max_tokens,
                sample_weight=weight,
            )
            for task, weight
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
        max_tokens: int | None = None,
        task_elbow: int | None = None,
    ):
        """ Creates a task loader for multiple tasks.
        
        The tasks are unweighted if `task_elbow` is None. Otherwise weights the probability of task
        being selected for training by the number of samples, clamped to the value of `task_elbow`.

        Args:
            task_list (TaskList): Task List of SFT tasks.
            formatter (InstructionFormatter): Formatter to use.
            seq_length (int): Max tokens per sequence in batch.
            batch_size (int): Number of sequences per batch.
            shuffle_seed (int | bool): Dataset pre-shuffling seed, or True for random seed, or False for no shuffle.
            mask_type (str): Must be 'all', 'train' or 'test'.
            max_tokens (int | None, optional): Max number of tokens in a conversation before dropping. Defaults to None.
            task_elbow (int | None, optional): Number of samples in a task to clip and weight by. Defaults to None.
        """
        
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.task_elbow = task_elbow

        self.tasks = [
            TaskLoader(
                task=task,
                formatter=formatter,
                seq_length=seq_length,
                sub_batch_size=batch_size,
                shuffle_seed=shuffle_seed,
                mask_type=mask_type,
                max_tokens=max_tokens,
                sample_weight=weight,
            )
            for task, weight
            in task_list
        ]

    def message_tokens_generator( self, shard_idx: int ) -> Generator[tuple[list[int], list[int]], None, None]:
        generators = [
            iter( task.message_tokens_generator( shard_idx ) )
            for task in self.tasks
        ]

        if self.task_elbow is None:
            while True:
                yield next( random.choice( generators ) )
        else:
            probs = [ min( task.num_samples, self.task_elbow ) * task.sample_weight for task in self.tasks ]
            
            while True:
                yield next( random.choices( generators, probs )[0] )

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
        max_tokens: int | None = None,
        micro_batch_size: int = 1,
        task_elbow: int | None = None
    ):
        """ Creates a task loader for multiple tasks with optional worker sharding.
        
        The tasks are unweighted if `task_elbow` is None. Otherwise weights the probability of task
        being selected for training by the number of samples, clamped to the value of `task_elbow`.
        
        When `micro_batch_size` is 1 will spawn `batch_size` number of workers.
        When `micro_batch_size == batch_size` only 1 worker will be spawned.

        Args:
            task_list (TaskList): Task List of SFT tasks.
            formatter (InstructionFormatter): Formatter to use.
            seq_length (int): Max tokens per sequence in batch.
            batch_size (int): Number of sequences per batch.
            mask_type (str): Must be 'all', 'train' or 'test'.
            max_tokens (int | None, optional): Max number of tokens in a conversation before dropping. Defaults to None.
            micro_batch_size (int, optional): Spawns `batch_size/micro_batch_size` workers. Defaults to 1.
            task_elbow (int | None, optional): Number of samples in a task to clip and weight by. Defaults to None.
        """
        
        assert batch_size % micro_batch_size == 0
        
        self.mixed_tasks = [
            MixedTaskLoader(
                task_list=task_list,
                formatter=formatter,
                seq_length=seq_length,
                batch_size=micro_batch_size,
                shuffle_seed=True,
                mask_type=mask_type,
                max_tokens=max_tokens,
                task_elbow=task_elbow,
            ) for _ in range( batch_size // micro_batch_size )
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

class DPHMultiTaskLoader( IterableDataset ):
    def __init__(
        self,
        task_list: TaskList,
        formatter: InstructionFormatter,
        seq_length: int,
        batch_size: int,
        mask_type: str,
        task_elbow: int | None = None,
    ):
        """ Creates a task loader for multiple DPH tasks.
        
        The tasks are unweighted if `task_elbow` is None. Otherwise weights the probability of task
        being selected for training by the number of samples, clamped to the value of `task_elbow`.
        
        Note: all tasks must provide both target and distractor messages to be used for DPH training.

        Args:
            task_list (TaskList): List of Tasks.
            formatter (InstructionFormatter): Formatter to use.
            seq_length (int): Max tokens per sequence in batch.
            batch_size (int): Number of pairs per batch.
            mask_type (str): Must be 'all', 'train' or 'test'.
            task_elbow (int | None, optional): Number of samples in a task to clip and weight by. Defaults to None.

        Raises:
            ValueError: raised when `mask_type` isn't one of `all`, `train` or `test`.
        """
        
        self.task_list = [ ( task, task.get_training_docs().shuffle(), weight ) for task, weight in task_list ] # type: ignore

        self.formatter = formatter
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.mask_type = mask_type
        
        self.task_elbow = task_elbow

        if mask_type not in [ 'all', 'train', 'test' ]:
            raise ValueError( 'mask_type must be `all`, `train` or `test`' )

    def pad_doc_single( self, line: dict ):
        curr_len = len( line[ 'tokens' ] )
        pad_len = self.seq_length - curr_len

        if pad_len < 0:
            raise ValueError( 'Sequence too long!' )

        pad_token_id = self.formatter.tokenizer.pad_token_id

        tokens = line[ 'tokens' ] + [ pad_token_id ] * pad_len
        targets = line[ 'targets' ] + [ -100 ] * pad_len
        train_mask = line[ 'train_mask' ] + [ False ] * pad_len
        test_mask = line[ 'test_mask' ] + [ False ] * pad_len

        match self.mask_type:
            case 'all':
                return tokens, targets
            case 'train':
                if not any( train_mask ):
                    raise ValueError( 'Train mask is empty!' )
                return tokens, [ t if m else -100 for t, m in zip( targets, train_mask ) ]
            case 'test':
                if not any( test_mask ):
                    raise ValueError( 'Test mask is empty!' )
                return tokens, [ t if m else -100 for t, m in zip( targets, test_mask ) ]
            case _:
                assert False, 'We should not be here!'

    def pad_doc_pair( self, task: BaseInstructDataset, doc: dict ):
        pos_list = task.create_target_message_list( doc )
        neg_list = task.create_distractor_message_list( doc )

        pos_candidate = random.choice( pos_list )
        neg_candidate = random.choice( neg_list )
        
        if pos_candidate[-1].role != 'assistant' or neg_candidate[-1].role != 'assistant':
            raise ValueError( 'Last message must be an assistant message!' )
        
        if len( pos_candidate ) != len( neg_candidate ):
            raise ValueError( 'Both examples must be of equal message count!' )

        pos_messages = self.formatter.tokenize_chat( pos_candidate )
        neg_messages = self.formatter.tokenize_chat( neg_candidate )

        pos_tokens, pos_targets = self.pad_doc_single( pos_messages )
        neg_tokens, neg_targets = self.pad_doc_single( neg_messages )

        return (
            torch.LongTensor( pos_tokens ),
            torch.LongTensor( pos_targets ),
            torch.LongTensor( neg_tokens ),
            torch.LongTensor( neg_targets )
        )

    def example_generator( self ):
        iterators = [ ( task, iter( generate_forever( ds ) ) ) for task, ds, _ in self.task_list ]
        probs = [ min( len( ds ), self.task_elbow or 1 ) * weight for _, ds, weight in self.task_list ]

        while True:
            task, ds = random.choices( iterators, probs )[0]
            doc = next( ds )

            try:
                yield self.pad_doc_pair( task, doc )
            except ValueError:
                continue

    def batch_generator( self ):
        iterator = iter( self.example_generator() )

        try:
            while True:
                pos_x_list = []
                pos_y_list = []
                neg_x_list = []
                neg_y_list = []
                for _ in range( self.batch_size ):
                    pos_tokens, pos_targets, neg_tokens, neg_targets = next( iterator )
                    pos_x_list.append( pos_tokens )
                    pos_y_list.append( pos_targets )
                    neg_x_list.append( neg_tokens )
                    neg_y_list.append( neg_targets )
                yield (
                    torch.stack( pos_x_list ),
                    torch.stack( pos_y_list ),
                    torch.stack( neg_x_list ),
                    torch.stack( neg_y_list ),
                )
        except StopIteration:
            return

    def __iter__( self ):
        return iter( self.batch_generator() )

    def as_data_loader( self ):
        return DataLoader(
            self,
            num_workers=1,
            batch_size=None,
            prefetch_factor=4,
            pin_memory=True,
            pin_memory_device='cuda',
        )

    def __getitem__( self, index ):
        raise NotImplementedError( "This dataset does not support random access using __getitem__" )


class SteerTaskLoader( IterableDataset ):
    def __init__(
        self,
        task: BaseSteerInstructDataset,
        formatter: SteerInstructionFormatter,
        batch_size: int,
        mask_type: str,
        num_probes: int,
        labels: list[str]
    ):
        
        self.task = task
        self.task_docs = task.get_training_docs().shuffle() # type: ignore

        self.formatter = formatter
        self.batch_size = batch_size
        self.mask_type = 'train'
        
        self.labels = labels
        
        self.num_probes = num_probes
        self.seq_length = self.formatter.max_total_tokens
        
        if num_probes > self.formatter.min_trainable_tokens:
            raise ValueError( 'num_probes must not be larger than the formatter\'s min_trainable_tokens!' )

        if mask_type not in [ 'all', 'train', 'test' ]:
            raise ValueError( 'mask_type must be `all`, `train` or `test`' )

    def pad_doc( self, line: dict ):
        curr_len = len( line[ 'tokens' ] )
        pad_len = self.seq_length - curr_len

        if pad_len < 0:
            raise ValueError( 'Sequence too long!' )

        pad_token_id = self.formatter.tokenizer.pad_token_id

        tokens = line[ 'tokens' ] + [ pad_token_id ] * pad_len
        targets = line[ 'targets' ] + [ -100 ] * pad_len
        train_mask = line[ 'train_mask' ] + [ False ] * pad_len
        test_mask = line[ 'test_mask' ] + [ False ] * pad_len
        segment_pos = line[ 'segment_pos' ] * [ 0 ] * pad_len

        match self.mask_type:
            case 'all':
                return tokens, targets, segment_pos
            case 'train':
                if not any( train_mask ):
                    raise ValueError( 'Train mask is empty!' )
                return tokens, [ t if m else -100 for t, m in zip( targets, train_mask ) ], segment_pos
            case 'test':
                if not any( test_mask ):
                    raise ValueError( 'Test mask is empty!' )
                return tokens, [ t if m else -100 for t, m in zip( targets, test_mask ) ], segment_pos
            case _:
                assert False, 'We should not be here!'

    def get_sample( self, doc: dict ):
        convos = self.task.create_target_message_list( doc )

        candidate = random.choice( convos )
        
        if candidate[-1].role != 'assistant':
            raise ValueError( 'Last message must be an assistant message!' )

        messages = self.formatter.tokenize_chat( candidate )

        tokens, targets, segment_pos = self.pad_doc( messages )
        labels = self.task.get_labels( doc, self.labels )
        
        segment_pos = torch.tensor( segment_pos, dtype=torch.long )
        final_pos, final_pos_idx = segment_pos.max( dim=-1, keepdims=True ) #type: ignore
        
        trimmed_pos = segment_pos * ( segment_pos != final_pos )
        
        selected_pos_idx = torch.multinomial( trimmed_pos, num_samples=self.num_probes - 1 )
        selected_pos_idx = torch.sort( selected_pos_idx, dim=-1 )[0]
        selected_pos_idx = torch.cat( [ selected_pos_idx, final_pos_idx ], dim=-1 )
        
        selected_pos_pos = segment_pos.gather( -1, selected_pos_idx )
        
        selected_pos_weight = selected_pos_pos / selected_pos_pos.sum()

        return (
            torch.LongTensor( tokens ),
            torch.LongTensor( targets ),
            torch.LongTensor( selected_pos_idx ),
            torch.LongTensor( selected_pos_weight ),
            torch.tensor( labels )
        )

    def example_generator( self ):
        iterator = iter( self.task_docs )

        while True:
            doc = next( iterator )
            assert isinstance( doc, dict )
            try:
                yield self.get_sample( doc )
            except ValueError:
                continue

    def batch_generator( self ):
        iterator = iter( self.example_generator() )

        try:
            while True:
                tokens_list = []
                targets_list = []
                idx_list = []
                weight_list = []
                labels_list = []
                for _ in range( self.batch_size ):
                    tokens, targets, selected_pos_idx, selected_pos_weight, labels = next( iterator )
                    tokens_list.append( tokens )
                    targets_list.append( targets )
                    idx_list.append( selected_pos_idx )
                    weight_list.append( selected_pos_weight )
                    labels_list.append( labels )
                yield (
                    torch.stack( tokens_list ),
                    torch.stack( targets_list ),
                    torch.stack( idx_list ),
                    torch.stack( weight_list ),
                    torch.stack( labels_list ),
                )
        except StopIteration:
            return

    def __iter__( self ):
        return iter( self.batch_generator() )

    def as_data_loader( self ):
        return DataLoader(
            self,
            num_workers=1,
            batch_size=None,
            prefetch_factor=4,
            pin_memory=True,
            pin_memory_device='cuda',
        )

    def __getitem__( self, index ):
        raise NotImplementedError( "This dataset does not support random access using __getitem__" )
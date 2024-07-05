"""
Module containing iterable datasets used to train and test LSWTransformer models.
"""

import json
from json import JSONDecodeError

import torch
from torch.utils.data import IterableDataset, DataLoader

from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

class PileShardDataset( IterableDataset ):
    """ Iterable Dataset for a single Pile shard.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        seq_length: int,
        shards_per_file: int,
        file_idx: int,
        dir_pattern: str,
    ):
        """
        Creates an iterable dataset for a single shard of the pile.

        Args:
            tokenizer (PreTrainedTokenizerBase): tokenizer to encode text.
            seq_length (int): desired sequence length.
            shards_per_file (int): number of shards to split iteration over.
            file_idx (int): id of the pile shard.
            dir_pattern (str): python format string for pile directory
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.shards_per_file = shards_per_file
        self.file_idx = file_idx
        self.dir_pattern = dir_pattern

    @classmethod
    def tokenize_line( cls, line: str, tokenizer: PreTrainedTokenizerBase ):
        tokens = tokenizer.encode( line, add_special_tokens=False )
        tokens_x = [ tokenizer.bos_token_id ] + tokens
        tokens_y = tokens + [ tokenizer.eos_token_id ]

        for x, y in zip( tokens_x, tokens_y ):
            yield ( x, y )

    @classmethod
    def line_parser( cls, path: str, shard_num: int, shard_id: int ):
        with open( path, 'rt', encoding="utf-8", buffering=1 ) as file:
            for line_num, line in enumerate( file ):
                if ( line_num % shard_num ) == shard_id:
                    try:
                        text = json.loads( line )[ 'text' ]
                        yield text
                    except JSONDecodeError:
                        pass

    @classmethod
    def line_token_generator( cls, path: str, tokenizer: PreTrainedTokenizerBase, shard_num: int, shard_id: int ):
        for line in cls.line_parser( path, shard_num, shard_id ):
            for x, y in cls.tokenize_line( line, tokenizer ):
                yield x, y

    @classmethod
    def sequence_generator( cls, path: str, tokenizer: PreTrainedTokenizerBase, shard_num: int, seq_length: int ):
        def reset():
            count = 0
            xs = []
            ys = []
            for _ in range( shard_num ):
                xs.append( [] )
                ys.append( [] )

            return count, xs, ys

        count, xs, ys = reset()

        generators = [ iter( cls.line_token_generator( path, tokenizer, shard_num, i ) ) for i in range( shard_num ) ]

        try:
            while True:
                for g_idx, generator in enumerate( generators ):
                    x, y = next( generator )
                    xs[ g_idx ].append( x )
                    ys[ g_idx ].append( y )
                count += 1

                if count == seq_length:
                    yield ( torch.LongTensor( xs ), torch.LongTensor( ys ) )

                    count, xs, ys = reset()
        except StopIteration:
            return

    def __iter__( self ):
        return iter( self.sequence_generator(
            path=self.dir_pattern.format( self.file_idx ),
            tokenizer=self.tokenizer,
            shard_num=self.shards_per_file,
            seq_length=self.seq_length,
        ) )

    def as_data_loader( self ):
        return DataLoader(
            self,
            num_workers=1,
            batch_size=None,
            prefetch_factor=2,
        )

    def __getitem__( self, index ):
        raise NotImplementedError( "This dataset does not support random access using __getitem__" )

class PileDataset( IterableDataset ):
    """ Iterable Dataset for a multiple Pile shards.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        seq_length: int,
        batch_size: int,
        dir_pattern: str,
        pile_shards: list[int] | None=None
    ):
        """
        Creates an iterable dataset for multiple shards over the pile.

        Args:
            tokenizer (PreTrainedTokenizerBase): tokenizer to encode text.
            seq_length (int): desired sequence length.
            batch_size (int): desired local batch size.
            pile_shards (Optional[List[int]], optional): List of shard IDs to use, when None uses all 30.
            dir_pattern (str): python format string for pile directory.
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.dir_pattern = dir_pattern
        self.pile_shards = pile_shards or list( range( 30 ) )

        if batch_size % len( self.pile_shards ) != 0:
            raise ValueError( 'batch size must be divisible by pile shard count' )

        self.shards_per_file = batch_size // len( self.pile_shards )

    def __iter__( self ):
        gen = [
            iter(
                PileShardDataset(
                    self.tokenizer,
                    self.seq_length,
                    self.shards_per_file,
                    i,
                    self.dir_pattern
                ).as_data_loader()
            ) for i in self.pile_shards
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

def load_wikitext( cache_dir ):
    return load_dataset(
        'EleutherAI/wikitext_document_level',
        name='wikitext-103-raw-v1',
        split='validation',
        cache_dir=cache_dir
    )

def load_pile_uncopyrighted( cache_dir ):
    return load_dataset(
        'json',
        data_files='https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main/val.jsonl.zst',
        split='train',
        cache_dir=cache_dir,
    )

def load_awesome_prompts( cache_dir ):
    return load_dataset(
        'fka/awesome-chatgpt-prompts',
        split='train',
        cache_dir=cache_dir,
    )
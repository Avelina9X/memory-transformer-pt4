from typing import List, Optional
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
import torch
import json
from json import JSONDecodeError

from torch.utils.data import IterableDataset, DataLoader

_PILE_DIR_JSONL = '/data/lhk3/the_pile/{:02d}.jsonl'


class PileShardDataset( IterableDataset ):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        seq_length: int,
        shards_per_file: int,
        file_idx: int,
    ):
        """
        Creates an iterable dataset for a single shard of the pile.

        Args:
            tokenizer (PreTrainedTokenizerBase): tokenizer to encode text.
            seq_length (int): desired sequence length.
            shards_per_file (int): number of shards to split iteration over.
            file_idx (int): id of the pile shard.
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.shards_per_file = shards_per_file
        self.file_idx = file_idx

    @classmethod
    def tokenize_line( cls, line: str, tokenizer: PreTrainedTokenizerBase ):
        tokens = tokenizer.encode( line, add_special_tokens=False )
        tokens_x = [ tokenizer.bos_token_id ] + tokens
        tokens_y = tokens + [ tokenizer.eos_token_id ]

        for x, y in zip( tokens_x, tokens_y ):
            yield ( x, y )

    @classmethod
    def line_parser( cls, path: str, shard_num: int, shard_id: int ):
        with open( path, 'rt', buffering=1 ) as file:
            for line_num, line in enumerate( file ):
                if ( line_num % shard_num ) == shard_id:
                    try:
                        text = json.loads( line )[ 'text' ]
                        yield text
                    except Exception as e:
                        if isinstance( e, JSONDecodeError ):
                            print( 'fuck' )
                            pass
                        else:
                            raise e

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
            path=_PILE_DIR_JSONL.format( self.file_idx ),
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

class PileDataset( IterableDataset ):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        seq_length: int,
        batch_size: int,
        pile_shards: Optional[List[int]]=None
    ):
        """
        Creates an iterable dataset for multiple shards over the pile.

        Args:
            tokenizer (PreTrainedTokenizerBase): tokenizer to encode text.
            seq_length (int): desired sequence length.
            batch_size (int): desired local batch size.
            pile_shards (Optional[List[int]], optional): List of shard IDs to use, when None uses all 30.
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.pile_shards = pile_shards or list( range( 30 ) )

        assert batch_size % len( self.pile_shards ) == 0, 'batch size must be divisible by pile shard count'
        self.shards_per_file = batch_size // len( self.pile_shards )

    def __iter__( self ):
        gen = [
            iter(
                PileShardDataset(
                    self.tokenizer,
                    self.seq_length,
                    self.shards_per_file,
                    i
                ).as_data_loader()
            ) for i in self.pile_shards
        ]

        while True:
            test_next = [ next( i ) for i in gen ]
            test_next_x = torch.cat( [ i[0] for i in test_next ] )
            test_next_y = torch.cat( [ i[1] for i in test_next ] )

            yield test_next_x, test_next_y

    def as_data_loader( self ):
        return DataLoader(
            self,
            num_workers=0,
            batch_size=None,
            # pin_memory=True,
            # pin_memory_device='cuda',
        )

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

def load_proofpile2( cache_dir ):
    return load_dataset(
        'EleutherAI/proof-pile-2',
        'default',
        revision='25a8c858a775f8d8d4798061c21b8091393d5980',
        cache_dir=cache_dir,
    )
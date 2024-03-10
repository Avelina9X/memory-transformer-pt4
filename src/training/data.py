"""
Module containing iterable datasets used to train and test LSWTransformer models.
"""

import json
from json import JSONDecodeError

from dataclasses import dataclass

import torch
from torch.utils.data import IterableDataset, DataLoader

from datasets import load_dataset, Dataset as HFDataset
from transformers import PreTrainedTokenizerBase


# _PILE_DIR_JSONL = '/data/lhk3/the_pile/{:02d}.jsonl'


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

@dataclass
class HFDatasetConfig:
    dataset_name: str
    dataset_sub_name: str
    dataset_split: str
    dataset_key: str
    cache_dir: str

class HFShardDataset( IterableDataset ):
    """ Iterable dataset for one or more shards of a HF dataset.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        seq_length: int,
        shard_idx: list[int],
        shard_max: int,
        dataset_config: HFDatasetConfig,
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.shard_idx = shard_idx
        self.shard_max = shard_max
        self.dataset_config = dataset_config

    @classmethod
    def tokenize_line( cls, line: str, tokenizer: PreTrainedTokenizerBase ):
        tokens = tokenizer.encode( line, add_special_tokens=False )
        tokens_x = [ tokenizer.bos_token_id ] + tokens
        tokens_y = tokens + [ tokenizer.eos_token_id ]

        for x, y in zip( tokens_x, tokens_y ):
            yield ( x, y )

    @classmethod
    def line_parser( cls, df_conf: HFDatasetConfig, shard_idx: int, shard_max: int ):
        dataset = load_dataset( df_conf.dataset_name, name=df_conf.dataset_sub_name, split=df_conf.dataset_split, cache_dir=df_conf.cache_dir, trust_remote_code=True )
        assert isinstance( dataset, HFDataset )
        dataset = dataset.shard( shard_max, shard_idx )
        for line in iter( dataset ):
            assert isinstance( line, dict )
            yield line[ df_conf.dataset_key ]

    @classmethod
    def line_token_generator( cls, df_conf: HFDatasetConfig, shard_idx: int, shard_max: int, tokenizer: PreTrainedTokenizerBase ):
        for line in cls.line_parser( df_conf, shard_idx, shard_max ):
            for x, y in cls.tokenize_line( line, tokenizer ):
                yield x, y

    @classmethod
    def sequence_generator( cls, df_conf: HFDatasetConfig, tokenizer: PreTrainedTokenizerBase, shard_idxs: list[int], shard_max: int, seq_length: int ):
        def reset():
            count = 0
            xs = []
            ys = []
            for _ in shard_idxs:
                xs.append( [] )
                ys.append( [] )

            return count, xs, ys

        count, xs, ys = reset()

        generators = [ iter( cls.line_token_generator( df_conf, i, shard_max, tokenizer ) ) for i in shard_idxs ]

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
            df_conf=self.dataset_config,
            tokenizer=self.tokenizer,
            shard_idxs=self.shard_idx,
            shard_max=self.shard_max,
            seq_length=self.seq_length,
        ) )

    def as_data_loader( self ):
        return DataLoader(
            self,
            num_workers=1,
            batch_size=None,
            prefetch_factor=4,
        )

    def __getitem__( self, index ):
        raise NotImplementedError( "This dataset does not support random access using __getitem__" )

class HFBatchDataset( IterableDataset ):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        seq_length: int,
        batch_size: int,
        dataset_config: HFDatasetConfig,
        num_proc: int,
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.dataset_config = dataset_config
        self.num_proc = num_proc

        if ( batch_size % self.num_proc ) != 0:
            raise ValueError( 'batch size must be divisible by num_proc' )

    def __iter__( self ):
        gen = [
            iter(
                HFShardDataset(
                    tokenizer=self.tokenizer,
                    seq_length=self.seq_length,
                    shard_idx=list( range( i, self.batch_size, self.num_proc ) ),
                    shard_max=self.batch_size,
                    dataset_config=self.dataset_config
                ).as_data_loader()
            ) for i in range( self.num_proc )
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


class OpenOrcaDataset( IterableDataset ):

    prompt_format: dict[str, str] = {
        'prompt': '### System Prompt:\n{}',
        'question': '### Instruction:\n{}',
        'response': '### Response:\n{}',
    }

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        seq_length: int,
        batch_size: int,
        cache_dir: str,
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.cache_dir = cache_dir

    @classmethod
    def tokenize_segment( cls, segment_type: str, segment_text: str, tokenizer: PreTrainedTokenizerBase ):
        return [
            tokenizer.sep_token_id,
            *tokenizer.encode( cls.prompt_format[segment_type].format( segment_text ), add_special_tokens=False ),
            tokenizer.cls_token_id,
        ]

    @classmethod
    def tokenize_line( cls, prompt: str, question: str, response: str, tokenizer: PreTrainedTokenizerBase ):
        p_tokens = cls.tokenize_segment( 'prompt', prompt, tokenizer )
        q_tokens = cls.tokenize_segment( 'question', question, tokenizer )
        r_tokens = cls.tokenize_segment( 'response', response, tokenizer )

        if prompt == '':
            tokens = q_tokens + r_tokens
            mask_len = len( q_tokens )
        else:
            tokens = p_tokens + q_tokens + r_tokens
            mask_len = len( p_tokens ) + len( q_tokens )

        tokens_x = [ tokenizer.bos_token_id ] + tokens
        tokens_y = tokens + [ tokenizer.eos_token_id ]

        tokens_y = [ tokenizer.pad_token_id if i < mask_len else token for i, token in enumerate( tokens_y ) ]

        for x, y in zip( tokens_x, tokens_y ):
            yield ( x, y )

    @classmethod
    def line_token_generator( cls, dataset: HFDataset, tokenizer: PreTrainedTokenizerBase, shard_num: int, shard_id: int ):
        while True:
            for line in dataset.shard( shard_num, shard_id ):
                p_str = line[ 'system_prompt' ] # type: ignore
                q_str = line[ 'question' ] # type: ignore
                r_str = line[ 'response' ] # type: ignore
                for x, y in cls.tokenize_line( p_str, q_str, r_str, tokenizer ):
                    yield x, y

    @classmethod
    def sequence_generator( cls, dataset: HFDataset, tokenizer: PreTrainedTokenizerBase, batch_size: int, seq_length: int ):
        def reset():
            count = 0
            xs = []
            ys = []
            for _ in range( batch_size ):
                xs.append( [] )
                ys.append( [] )

            return count, xs, ys

        count, xs, ys = reset()

        generators = [ iter( cls.line_token_generator( dataset, tokenizer, batch_size, i ) ) for i in range( batch_size ) ]

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
            dataset=load_openorca( self.cache_dir ), # type: ignore
            tokenizer=self.tokenizer,
            batch_size=self.batch_size,
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


def load_wikitext( cache_dir ):
    return load_dataset(
        'EleutherAI/wikitext_document_level',
        name='wikitext-103-raw-v1',
        split='validation',
        cache_dir=cache_dir
    )

def load_lambada( cache_dir ):
    return load_dataset(
        'EleutherAI/lambada_openai',
        name='en',
        split='test',
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

def load_openorca( cache_dir ):
    return load_dataset(
        'Open-Orca/OpenOrca',
        split='train',
        cache_dir=cache_dir
    )

def load_slimorca( cache_dir ):
    return load_dataset(
        'Open-Orca/SlimOrca',
        split='train',
        cache_dir=cache_dir
    )

def load_alpaca( cache_dir ):
    return load_dataset(
        'yahma/alpaca-cleaned',
        split='train',
        cache_dir=cache_dir
    )

def load_falcon_100k( cache_dir ):
    return load_dataset(
        'BEE-spoke-data/falcon-refinedweb-100k_en-long',
        split='train',
        cache_dir=cache_dir
    )

def load_pg19( cache_dir, split ):
    return load_dataset(
        'pg19',
        split=split,
        cache_dir=cache_dir
    )

def load_gov_reports( cache_dir, split ):
    return load_dataset(
        'tau/scrolls',
        'gov_report',
        split=split,
        cache_dir=cache_dir
    )

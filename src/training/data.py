from datasets import load_dataset
from transformers import PreTrainedTokenizerBase, AutoTokenizer
import torch
import json
from json import JSONDecodeError

from torch.utils.data import IterableDataset, DataLoader

_PILE_DIR_JSONL = '/data/lhk3/the_pile/{:02d}.jsonl'
_PILE_TEST_FILE = _PILE_DIR_JSONL.format( 0 )


class PileShardDataset( IterableDataset ):
    def __init__( self, tokenizer, seq_length, shards_per_file, file_idx ):
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
    def __init__( self, tokenizer, seq_length, batch_size ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.batch_size = batch_size
        
        assert batch_size % 30 == 0, 'batch size must be divisible by pile shard count (30)'
        self.shards_per_file = batch_size // 30
    
    def __iter__( self ):
        gen = [ iter( PileShardDataset( self.tokenizer, self.seq_length, self.shards_per_file, i ).as_data_loader() ) for i in range( 30 ) ]
        
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
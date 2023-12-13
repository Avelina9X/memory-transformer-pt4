"""
Module containing the evaluation loop components for training LSWTransformer models.
"""

import gc

import torch
from torcheval import metrics
from transformers import PreTrainedTokenizerBase

from model.modeling import LSWTForCausalLM
from constants import TORCH_COMPILE_OPTIONS

from .losses import MLELoss, AccuracyMetric
from .data import OpenOrcaDataset


class Eval():
    def __init__( self, model: LSWTForCausalLM, tokenizer: PreTrainedTokenizerBase ):
        self.model = model
        self.tokenizer = tokenizer

        self.loss_function = MLELoss( self.model.config.vocab_size, -100 )
        self.acc_function = AccuracyMetric( self.model.config.vocab_size, -100 )

        self.metrics = {
            'loss': metrics.Mean().to( 'cuda' ),
            'acc': metrics.Mean().to( 'cuda' ),
        }


    """ ========================================================================
        Utility functions
        ======================================================================== """

    def reset_metrics( self ) -> dict[ str, float ]:
        stats = {}
        for name, metric in self.metrics.items():
            stats[name] = float( metric.compute() )
            metric.reset()
        return stats


    def num_true( self, tokens_y, pad_token_id=-100 ):
        return ( tokens_y != pad_token_id ).sum()


    """ ========================================================================
        Forward Pass
        ======================================================================== """

    @torch.compile( **TORCH_COMPILE_OPTIONS )
    def forward_pass( self, tokens, past_key_values ):
        outputs = self.model(
            input_ids=tokens,
            past_key_values=past_key_values,
            use_cache=True,
        )

        past_key_values = outputs.past_key_values
        logits = outputs.logits

        del outputs

        return logits, past_key_values


    """ ========================================================================
        Eval Functions
        ======================================================================== """

    def eval_sub_step( self, tokens_x: torch.Tensor, tokens_y: torch.Tensor, chunk_size: int ):
        loss = 0.0
        accuracy = 0.0
        total_num_true = 0

        with torch.autocast( device_type='cuda', dtype=torch.float16 ): # type: ignore
            with torch.no_grad():
                tokens_xs = torch.split( tokens_x, chunk_size, dim=1 )
                tokens_ys = torch.split( tokens_y, chunk_size, dim=1 )
                past_key_values = None

                for tx, ty in zip( tokens_xs, tokens_ys ):
                    logits, past_key_values = self.forward_pass( tx, past_key_values )
                    past_key_values = self.model.cache_to( past_key_values, 'cuda', trim=chunk_size )

                    num_true = self.num_true( ty )

                    if num_true > 0:
                        total_num_true += num_true
                        loss += self.loss_function( None, logits, None, ty )[0] * num_true
                        accuracy += self.acc_function( logits, ty ) * num_true

                loss /= total_num_true
                accuracy /= total_num_true

            return loss, accuracy

    def eval_step( self, sequence: str, chunk_size: int ):
        self.model.eval()

        tokens = self.tokenizer.encode( sequence, add_special_tokens=False )
        tokens_x = [ self.model.config.bos_token_id ] + tokens
        tokens_y = tokens + [ self.model.config.eos_token_id ]

        pad_amt = chunk_size - ( len( tokens_x ) % chunk_size )

        tokens_x = tokens_x + pad_amt * [ self.model.config.pad_token_id ]
        tokens_y = tokens_y + pad_amt * [ -100 ]

        tokens_x = torch.LongTensor( [ tokens_x ] ).cuda()
        tokens_y = torch.LongTensor( [ tokens_y ] ).cuda()

        loss, accuracy = self.eval_sub_step( tokens_x, tokens_y, chunk_size )

        self.metrics[ 'loss' ].update( loss ) # type: ignore
        self.metrics[ 'acc' ].update( accuracy ) # type: ignore

    def eval_epoch( self, iterator, iterator_key: str, chunk_size: int ):
        for row in iterator:
            self.eval_step( row[ iterator_key ], chunk_size )

        torch.cuda.empty_cache()
        gc.collect()

        return self.reset_metrics()

class EvalAlpaca( Eval ):
    
    def eval_step( self, sequence: dict[str, str], chunk_size: int ):
        self.model.eval()

        results = OpenOrcaDataset.tokenize_line(
            'You are an AI assistant. You will be given a task. You must generate a detailed and long answer.',
            sequence[ 'instruction' ] + ( '\n### Input:\n' + sequence[ 'input' ] ) if sequence[ 'input' ] else '',
            sequence[ 'output' ],
            self.tokenizer
        )
        
        tokens_x, tokens_y = list( zip( *results ) )
        
        tokens_x = list( tokens_x )
        tokens_y = list( tokens_y )

        pad_amt = chunk_size - ( len( tokens_x ) % chunk_size )

        tokens_x = tokens_x + pad_amt * [ self.model.config.pad_token_id ]
        tokens_y = tokens_y + pad_amt * [ -100 ]
        
        tokens_x = torch.LongTensor( [ tokens_x ] ).cuda()
        tokens_y = torch.LongTensor( [ tokens_y ] ).cuda()
        
        tokens_y = torch.where( tokens_y == self.tokenizer.pad_token_id, -100, tokens_y )

        loss, accuracy = self.eval_sub_step( tokens_x, tokens_y, chunk_size )

        self.metrics[ 'loss' ].update( loss ) # type: ignore
        self.metrics[ 'acc' ].update( accuracy ) # type: ignore
    
    def eval_epoch( self, iterator, iterator_key: str | None, chunk_size: int ):
        for row in iterator:
            if row[ 'input' ] == '':
                self.eval_step( row, chunk_size )

        torch.cuda.empty_cache()
        gc.collect()

        return self.reset_metrics()
from transformers import PreTrainedTokenizerBase

from model.modeling import LSWTForCausalLM

from .losses import MLELoss, AccuracyMetric

import torch

from torcheval import metrics

import gc

from typing import Dict

from constants import TORCH_COMPILE_OPTIONS

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

    def reset_metrics( self ) -> Dict[ str, float ]:
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

    def eval_sub_step( self, tokens_x: torch.Tensor, tokens_y: torch.Tensor, chunk_size: int, chunk_compute: bool=True ):
        loss = 0.0
        accuracy = 0.0
        total_num_true = 0

        with torch.autocast( device_type='cuda', dtype=torch.float16 ): # type: ignore
            with torch.no_grad():
                tokens_xs = torch.split( tokens_x, chunk_size, dim=1 )
                tokens_ys = torch.split( tokens_y, chunk_size, dim=1 )
                past_key_values = None
                logits_list = []

                for idx in range( len( tokens_xs ) ):
                    logits, past_key_values = self.forward_pass( tokens_xs[idx], past_key_values )
                    past_key_values = self.model.cache_to( past_key_values, 'cuda', trim=chunk_size )

                    if not chunk_compute:
                        logits_list.append( logits )
                    else:
                        num_true = self.num_true( tokens_ys[idx] )

                        if num_true > 0:
                            total_num_true += num_true
                            loss += self.loss_function( None, logits, None, tokens_ys[idx] )[0] * num_true
                            accuracy += self.acc_function( logits, tokens_ys[idx] ) * num_true

                if not chunk_compute:
                    logits = torch.cat( logits_list, dim=1 )

                    loss += self.loss_function( None, logits, None, tokens_y )[0]
                    accuracy += self.acc_function( logits, tokens_y )
                else:
                    loss /= total_num_true
                    accuracy /= total_num_true

            return loss, accuracy

    def eval_step( self, sequence: str, chunk_size: int, chunk_compute: bool=True ):
        self.model.eval()

        tokens = self.tokenizer.encode( sequence, add_special_tokens=False )
        tokens_x = [ self.model.config.bos_token_id ] + tokens
        tokens_y = tokens + [ self.model.config.eos_token_id ]

        pad_amt = chunk_size - ( len( tokens_x ) % chunk_size )

        tokens_x = tokens_x + pad_amt * [ self.model.config.pad_token_id ]
        tokens_y = tokens_y + pad_amt * [ -100 ]

        tokens_x = torch.LongTensor( [ tokens_x ] ).cuda()
        tokens_y = torch.LongTensor( [ tokens_y ] ).cuda()

        loss, accuracy = self.eval_sub_step( tokens_x, tokens_y, chunk_size, chunk_compute )

        self.metrics[ 'loss' ].update( loss ) # type: ignore
        self.metrics[ 'acc' ].update( accuracy ) # type: ignore

    def eval_epoch( self, iterator, iterator_key: str, chunk_size: int, chunk_compute: bool=True ):
        for row in iterator:
            self.eval_step( row[ iterator_key ], chunk_size, chunk_compute )

        torch.cuda.empty_cache()
        gc.collect()

        return self.reset_metrics()
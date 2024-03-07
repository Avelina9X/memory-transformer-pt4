import itertools
from transformers import PreTrainedTokenizerBase

class InstructionFormatter():
    def __init__( self, tokenizer: PreTrainedTokenizerBase ):
        self.tokenizer = tokenizer
    
    def _tokenize( self, text: str ) -> list[int]:
        output = self.tokenizer( text, add_special_tokens=False )[ 'input_ids' ]
        assert isinstance( output, list )
        return output
    
    def apply_chat_template_line( self, message: dict ):
        role = message[ 'role' ]
        content = message[ 'content' ]
        complete = message[ 'complete' ]
        
        # TODO: assertions for role and completion
        
        trainable: bool = role == 'assistant'
        
        prefix = self._tokenize( f'<|im_start|>{role}\n' )
        content = self._tokenize( content )
        suffix = self._tokenize( '<|im_end|>\n' if complete else '' )
        
        prefix_train_mask = [ False for _ in prefix ]
        content_train_mask = [ trainable for _ in content ]
        suffix_train_mask = [ trainable for _ in suffix ]
        
        prefix_test_mask = [ False for _ in prefix ]
        content_test_mask = [ trainable for _ in content ]
        suffix_test_mask = [ False for _ in suffix ]
        
        return {
            'tokens': prefix + content + suffix,
            'train_mask': prefix_train_mask + content_train_mask + suffix_train_mask,
            'test_mask': prefix_test_mask + content_test_mask + suffix_test_mask,
        }
    
    def apply_chat_template( self, conversation: list[dict] ):
        lines = [ self.apply_chat_template_line( line ) for line in conversation ]
        
        tokens = [ line[ 'tokens' ] for line in lines ]
        train_mask = [ line[ 'train_mask' ] for line in lines ]
        test_mask = [ line[ 'test_mask' ] for line in lines ]
        
        return {
            'tokens': list( itertools.chain( *tokens ) ),
            'train_mask': list( itertools.chain( *train_mask ) ),
            'test_mask': list( itertools.chain( *test_mask ) ),
        }
    
    def tokenize_chat( self, conversation: list[dict] ):
        # TODO: assert final message is complete

        outputs = self.apply_chat_template( conversation )
        
        return {
            'tokens': [ self.tokenizer.eos_token_id ] + outputs[ 'tokens' ],
            'targets': outputs[ 'tokens' ] + [ self.tokenizer.eos_token_id ],
            'train_mask': outputs[ 'train_mask' ] + [ False ],
            'test_mask': outputs[ 'test_mask' ] + [ False ],
        }
    
    def tokenize_generation( self, conversation: list[dict] ):
        # TODO: assert final message is incomplete

        outputs = self.apply_chat_template( conversation )
        
        return {
            'tokens': [ self.tokenizer.eos_token_id ] + outputs[ 'tokens' ],
            'targets': None,
            'train_mask': None,
            'test_mask': None,
        }
import functools
import itertools
from transformers import PreTrainedTokenizerBase

from .task_base import Message, MessageList

class InstructionFormatter():
    def __init__( self, tokenizer: PreTrainedTokenizerBase, max_cache_size=4096 ):
        self.tokenizer = tokenizer

        self.tokenize = functools.lru_cache( max_cache_size )( self._tokenize )

    def _tokenize( self, text: str ) -> list[int]:
        output = self.tokenizer( text, add_special_tokens=False )[ 'input_ids' ]
        assert isinstance( output, list )
        return output

    def _apply_chat_template_line( self, message: Message ):
        role = message.role
        content = message.content
        complete = message.complete

        # TODO: assertions for role and completion

        trainable: bool = role == 'assistant'

        prefix = self.tokenize( f'<|im_start|>{role}\n' )
        content = self.tokenize( content )
        suffix = self.tokenize( '<|im_end|>\n' if complete else '' )

        prefix_train_mask = [ False for _ in prefix ]
        content_train_mask = [ trainable for _ in content ]
        suffix_train_mask = [ trainable and i == 0 for i, _ in enumerate( suffix ) ]

        prefix_test_mask = [ False for _ in prefix ]
        content_test_mask = [ trainable for _ in content ]
        suffix_test_mask = [ False for _ in suffix ]

        return {
            'tokens': prefix + content + suffix,
            'train_mask': prefix_train_mask + content_train_mask + suffix_train_mask,
            'test_mask': prefix_test_mask + content_test_mask + suffix_test_mask,
        }

    def _apply_chat_template( self, conversation: MessageList ):
        lines = [ self._apply_chat_template_line( line ) for line in conversation ]

        tokens = [ line[ 'tokens' ] for line in lines ]
        train_mask = [ line[ 'train_mask' ] for line in lines ]
        test_mask = [ line[ 'test_mask' ] for line in lines ]

        return {
            'tokens': list( itertools.chain( *tokens ) ),
            'train_mask': list( itertools.chain( *train_mask ) ),
            'test_mask': list( itertools.chain( *test_mask ) ),
        }

    def _remove_system_msgs( self, msgs: MessageList ):
        return [ msg for msg in msgs if msg.role != 'system' ]

    def tokenize_chat( self, conversation: MessageList ) -> dict:
        """ Tokenizes a message list for zero-shot evaluation.

        Args:
            conversation (MessageList): Target message to evaluate.

        Returns:
            dict: dictionary of tokens and masks
        """

        outputs = self._apply_chat_template( conversation )

        return {
            'tokens': [ self.tokenizer.eos_token_id ] + outputs[ 'tokens' ],
            'targets': outputs[ 'tokens' ] + [ self.tokenizer.eos_token_id ],
            'train_mask': outputs[ 'train_mask' ] + [ False ],
            'test_mask': outputs[ 'test_mask' ] + [ False ],
        }

    def tokenize_chat_fewshot(
        self,
        target_conversation: MessageList,
        fewshot_list: list[MessageList],
        fewshow_allsys: bool
    ) -> dict:
        """ Tokenizes a message list for few-shot evaluation.

        Only the final assistant message is enabled in the mask.

        Args:
            target_conversation (MessageList): Target message to evaluate.
            fewshot_list (list[MessageList]): Example few shot messages.
            fewshow_allsys (bool): When True all messages have a system message. When False only the first system message is included.

        Returns:
            dict: dictionary of tokens and masks
        """
        head = fewshot_list[0]
        body = list( itertools.chain( *fewshot_list[ 1 : ] ) )
        tail = target_conversation

        if not fewshow_allsys:
            body = self._remove_system_msgs( body )
            tail = self._remove_system_msgs( tail )

        f_outputs = self._apply_chat_template( head + body )
        t_outputs = self._apply_chat_template( tail )

        f_outputs[ 'train_mask' ] = [ False for _ in f_outputs[ 'train_mask' ] ]
        f_outputs[ 'test_mask' ] = [ False for _ in f_outputs[ 'test_mask' ] ]

        return {
            'tokens': [ self.tokenizer.eos_token_id ] + f_outputs[ 'tokens' ] + t_outputs[ 'tokens' ],
            'targets': f_outputs[ 'tokens' ] + t_outputs[ 'tokens' ] + [ self.tokenizer.eos_token_id ],
            'train_mask': f_outputs[ 'train_mask' ] + t_outputs[ 'train_mask' ] + [ False ],
            'test_mask': f_outputs[ 'test_mask' ] + t_outputs[ 'test_mask' ] + [ False ],
        }

    def tokenize_chat_training(
        self,
        target_list: list[MessageList],
        fewshow_allsys: bool
    ) -> dict:
        """ Tokenizes a message list for zero-shot or few-shot.

        All assistant message are enabled in the mask.

        Args:
            target_list (list[MessageList]): List of target messages.
            fewshow_allsys (bool): When True all messages have a system message. When False only the first system message is included.

        Returns:
            dict: dictionary of tokens and masks
        """
        head = target_list[0]
        body = list( itertools.chain( *target_list[ 1 : ] ) )

        if not fewshow_allsys:
            body = self._remove_system_msgs( body )

        outputs = self._apply_chat_template( head + body )

        return {
            'tokens': [ self.tokenizer.eos_token_id ] + outputs[ 'tokens' ],
            'targets': outputs[ 'tokens' ] + [ self.tokenizer.eos_token_id ],
            'train_mask': outputs[ 'train_mask' ] + [ False ],
            'test_mask': outputs[ 'test_mask' ] + [ False ],
        }


    # def tokenize_generation( self, conversation: MessageList ):
    #     # TODO: assert final message is incomplete
    #     outputs = self._apply_chat_template( conversation )
    #     return {
    #         'tokens': [ self.tokenizer.eos_token_id ] + outputs[ 'tokens' ],
    #         'targets': None,
    #         'train_mask': None,
    #         'test_mask': None,
    #     }

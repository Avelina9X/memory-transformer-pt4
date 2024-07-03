from collections.abc import Callable, Mapping
from datasets import DatasetDict, Dataset, load_dataset

from ..task_base import BaseInstructDataset, InstructionDatasetTask, Message, MessageList
from ..task_utils import phrase_filter

class InfinityInstructDataset( BaseInstructDataset ):
    def download( self, cache_dir: str ) -> DatasetDict:
        dataset = load_dataset( 'BAAI/Infinity-Instruct', cache_dir=cache_dir )
        assert isinstance( dataset, DatasetDict )
        dataset = dataset.filter( lambda x: phrase_filter( x, 'value', 'conversations' ) )
        return dataset

    @property
    def task_type( self ) -> InstructionDatasetTask:
        return InstructionDatasetTask.INSTRUCT_CLOSED

    @property
    def task_name( self ) -> str:
        return 'baai'

    @property
    def task_subset( self ) -> str:
        return 'Infinity-Instruct'

    def get_training_docs( self ) -> Dataset:
        return self.dataset[ 'train' ]

    def get_validation_docs( self ) -> None:
        return None

    def get_test_docs( self ) -> None:
        return None

    def get_fewshot_docs( self ) -> None:
        return None

    def format_system_message( self, doc: dict ) -> Message:
        prompt = (
            'You are a helpful AI assistant. '
            'Write a response that appropriately completes the request.'
        )
        
        return Message(
            role='system',
            content=prompt,
            complete=True,
        )

    def format_user_message( self, doc: dict ) -> Message:
        raise NotImplementedError( 'This does not use the message factory' )

    def format_target_messages( self, doc: dict ) -> list[Message]:
        raise NotImplementedError( 'This does not use the message factory' )

    def format_distractor_messages( self, doc: dict ) -> list[Message]:
        return []

    def format_unlabelled_messages( self, doc: dict ) -> list[Message]:
        return []

    def create_unlabelled_message_target( self, doc: dict ) -> None:
        return None
    
    def create_target_message_list( self, doc: dict ) -> list[MessageList]:
        role_map = {
            'system': 'system',
            'human': 'user',
            'gpt': 'assistant'
        }
        
        first = doc[ 'conversations' ][0]
        
        # If first message is from a human add our own system message
        if first[ 'from' ] == 'human':
            message_list = [ self.format_system_message( doc ) ]

            for sub_doc in doc[ 'conversations' ]:
                message_list.append(
                    Message(
                        role=role_map[ sub_doc[ 'from' ] ],
                        content=sub_doc[ 'value' ],
                        complete=True,
                    )
                )

            return [ message_list ]
        
        # If first message is from GPT turn it into a system message
        elif first[ 'from' ] == 'gpt':
            message_list = [
                Message(
                    role='system',
                    content=first[ 'value' ],
                    complete=True,
                )
            ]
            
            for sub_doc in doc[ 'conversations' ][ 1 : ]:
                message_list.append(
                    Message(
                        role=role_map[ sub_doc[ 'from' ] ],
                        content=sub_doc[ 'value' ],
                        complete=True,
                    )
                )
            
            return [ message_list ]
        
        raise ValueError( 'How did we get here?' )

    def compute_metric( self, predictions=None, references=None ) -> dict:
        # TODO: add warning for using compute
        return {}

DIRECTORY: Mapping[str, Callable[[str], BaseInstructDataset]] = {
    'Infinity-Instruct': InfinityInstructDataset
}
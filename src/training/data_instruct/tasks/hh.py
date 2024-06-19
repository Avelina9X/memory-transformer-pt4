from collections.abc import Callable, Mapping
from datasets import DatasetDict, Dataset, load_dataset

from ..task_base import BaseInstructDataset, InstructionDatasetTask, Message, MessageList

class HHRLHFInstructDataset( BaseInstructDataset ):
    def download( self, cache_dir: str ) -> DatasetDict:
        dataset = load_dataset( 'trl-internal-testing/hh-rlhf-trl-style', cache_dir=cache_dir )
        assert isinstance( dataset, DatasetDict )
        return dataset

    @property
    def task_type( self ) -> InstructionDatasetTask:
        return InstructionDatasetTask.CONVERSATIONAL

    @property
    def task_name( self ) -> str:
        return 'hh'

    @property
    def task_subset( self ) -> str:
        return 'rlhf'

    def get_training_docs( self ) -> Dataset:
        return self.dataset[ 'train' ]

    def get_validation_docs( self ) -> None:
        return None

    def get_test_docs( self ) -> Dataset:
        return self.dataset[ 'test' ]

    def get_fewshot_docs( self ) -> None:
        return None

    def format_system_message( self, doc: dict ) -> Message:
        prompt = (
            'You are a conversational AI assistant. '
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
        raise NotImplementedError( 'This does not use the message factory' )

    def create_distractor_message_list( self, doc: dict ) -> list[MessageList]:
        message_list = [ self.format_system_message( doc ) ]

        for sub_doc in doc[ 'rejected' ]:
            message_list.append(
				Message(
					role=sub_doc[ 'role' ],
					content=sub_doc[ 'content' ],
					complete=True,
				)
			)

        return [ message_list ]
    
    def create_target_message_list( self, doc: dict ) -> list[MessageList]:
        message_list = [ self.format_system_message( doc ) ]

        for sub_doc in doc[ 'chosen' ]:
            message_list.append(
				Message(
					role=sub_doc[ 'role' ],
					content=sub_doc[ 'content' ],
					complete=True,
				)
			)

        return [ message_list ]

    def format_unlabelled_messages( self, doc: dict ) -> list[Message]:
        return []

    def create_unlabelled_message_target( self, doc: dict ) -> None:
        return None

    def compute_metric( self, predictions=None, references=None ) -> dict:
        # TODO: add warning for using compute
        return {}


DIRECTORY: Mapping[str, Callable[[str], BaseInstructDataset]] = {
    'rlhf': HHRLHFInstructDataset,
}

def main():
    # pylint: disable=W0611
    # pylint: disable=C0415
    import os
    import rich

    cache_dir = os.environ[ 'HF_CACHE_DIR' ]

    rich.print( HHRLHFInstructDataset( cache_dir ) )

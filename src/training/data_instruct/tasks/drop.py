from collections.abc import Callable, Mapping
from datasets import DatasetDict, Dataset, load_dataset

from ..task_base import BaseInstructDataset, InstructionDatasetTask, Message, MessageList

class DropInstructDataset( BaseInstructDataset ):
    def download( self, cache_dir: str ) -> DatasetDict:
        dataset = load_dataset( 'ucinlp/drop', cache_dir=cache_dir )
        assert isinstance( dataset, DatasetDict )
        return dataset

    @property
    def task_name( self ) -> str:
        return 'drop'

    @property
    def task_subset( self ) -> str:
        return 'main'

    @property
    def task_type( self ) -> InstructionDatasetTask:
        return InstructionDatasetTask.SPAN_EXTRACTION

    def get_training_docs( self ) -> Dataset:
        return self.dataset[ 'train' ]

    def get_validation_docs( self ) -> Dataset:
        return self.dataset[ 'validation' ]

    def get_test_docs( self ) -> None:
        return None

    def get_fewshot_docs( self ) -> None:
        return None

    def format_system_message( self, doc: dict ) -> Message:
        prompt = (
            'Below is a question, paired with a background context. '
            'Respond with text from the background that correctly answers the question. '
            'If there are multiple answers seperate each answer with a newline.'
        )

        return Message(
            role='system',
            content=prompt,
            complete=True
        )

    def format_user_message( self, doc: dict ) -> Message:
        prompt = (
            f'Background: {doc["passage"]}\n'
            f'\n'
            f'Question: {doc["question"].strip()}\n'
            f'\n'
            f'Answer:'
        )

        return Message(
            role='user',
            content=prompt,
            complete=True,
        )

    def format_unlabelled_messages( self, doc: dict ) -> list[Message]:
        return []

    def create_unlabelled_message_target( self, doc: dict ) -> None:
        return None

    def create_unlabelled_message_list(self, doc: dict) -> list[MessageList]:
        return self.create_target_message_list( doc ) + self.create_distractor_message_list( doc )

    def format_target_messages( self, doc: dict ) -> list[Message]:
        return [
            Message(
                role='assistant',
                content='\n'.join( doc[ 'answers_spans' ][ 'spans' ] ),
                complete=True,
            )
        ]

    def format_distractor_messages( self, doc: dict ) -> list[Message]:
        return []

    def compute_metric( self, predictions=None, references=None ) -> dict:
        # TODO: add squad style eval
        return {}


DIRECTORY: Mapping[str, Callable[[str], DropInstructDataset]] = {
    'main': DropInstructDataset,
}


def main():
    # pylint: disable=W0611
    # pylint: disable=C0415
    import os
    import rich

    cache_dir = os.environ[ 'HF_CACHE_DIR' ]

    rich.print( DropInstructDataset( cache_dir ) )

from collections.abc import Callable, Mapping
from datasets import DatasetDict, Dataset, load_dataset

from ..task_base import BaseInstructDataset, InstructionDatasetTask, Message

class GSM8KInstructDataset( BaseInstructDataset ):
    def __init__( self, split: str, cache_dir: str ):
        self.split = split
        super().__init__( cache_dir )

    def download( self, cache_dir: str ) -> DatasetDict:
        dataset = load_dataset( 'gsm8k', self.split, cache_dir=cache_dir )
        assert isinstance( dataset, DatasetDict )
        return dataset

    @property
    def task_type( self ) -> InstructionDatasetTask:
        return InstructionDatasetTask.INSTRUCT_CLOSED

    @property
    def task_name( self ) -> str:
        return 'gsm8k'

    @property
    def task_subset( self ) -> str:
        return self.split

    def get_training_docs( self ) -> Dataset:
        return self.dataset[ 'train' ]

    def get_validation_docs( self ) -> None:
        return None

    def get_test_docs( self ) -> None:
        return None

    def get_fewshot_docs( self ) -> None:
        return None

    def format_system_message( self, doc: dict ) -> Message:
        if self.split == 'main':
            prompt = (
                'Below is an instruction that describes a math problem. '
                'Think about the problem and then respond with the answer after "####".'
            )
        else:
            prompt = (
                'Below is an instruction that describes a math problem. '
                'Think about the problem in a socratic maner and then respond with the answer after "####".'
            )

        return Message(
            role='system',
            content=prompt,
            complete=True,
        )

    def format_user_message( self, doc: dict ) -> Message:
        return Message(
            role='user',
            content=doc['question'],
            complete=True,
        )

    def format_target_messages( self, doc: dict ) -> list[Message]:
        return [ Message(
            role='assistant',
            content=doc['answer'],
            complete=True,
        ) ]

    def format_distractor_messages( self, doc: dict ) -> list[Message]:
        return []

    def format_unlabelled_messages( self, doc: dict ) -> list[Message]:
        return []

    def create_unlabelled_message_target( self, doc: dict ) -> None:
        return None

    def compute_metric( self, predictions=None, references=None ) -> dict:
        # TODO: add warning for using compute
        return {}


DIRECTORY: Mapping[str, Callable[[str], BaseInstructDataset]] = {
    'main': lambda cache_dir: GSM8KInstructDataset( cache_dir=cache_dir, split='main' ),
    'socratic': lambda cache_dir: GSM8KInstructDataset( cache_dir=cache_dir, split='socratic' ),
}

def main():
    # pylint: disable=W0611
    # pylint: disable=C0415
    import os
    import rich

    cache_dir = os.environ[ 'HF_CACHE_DIR' ]

    rich.print( GSM8KInstructDataset( 'main', cache_dir ) )

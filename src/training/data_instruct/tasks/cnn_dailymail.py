from collections.abc import Callable, Mapping
from datasets import DatasetDict, Dataset, load_dataset
from evaluate import load as load_metric

from ..task_base import BaseInstructDataset, InstructionDatasetTask, Message

class CNNDailymailInstructDataset( BaseInstructDataset ):
    def __init__( self, cache_dir: str ):
        self.metric = load_metric( 'rouge' )
        super().__init__( cache_dir )

    def download( self, cache_dir: str ) -> DatasetDict:
        dataset = load_dataset( 'cnn_dailymail', '3.0.0', cache_dir=cache_dir )
        assert isinstance( dataset, DatasetDict )
        return dataset

    @property
    def task_type( self ) -> InstructionDatasetTask:
        return InstructionDatasetTask.SUMMARIZATION

    @property
    def task_name( self ) -> str:
        return 'cnn_dailymail'

    @property
    def task_subset( self ) -> str:
        return '3.0.0'

    def get_training_docs( self ) -> Dataset:
        return self.dataset[ 'train' ]

    def get_validation_docs( self ) -> Dataset:
        return self.dataset[ 'validation' ]

    def get_test_docs( self ) -> Dataset:
        return self.dataset[ 'test' ]

    def get_fewshot_docs( self ) -> None:
        return None

    def format_user_message( self, doc: dict ) -> Message:
        prompt = (
            f'Article: {doc["article"]}\n'
            f'\n'
            f'Summary:'
        )

        return Message(
            role='user',
            content=prompt,
            complete=True,
        )

    def format_target_messages( self, doc: dict ) -> list[Message]:
        return [ Message(
            role='assistant',
            content=doc['highlights'],
            complete=True,
        ) ]

    def format_distractor_messages( self, doc: dict ) -> list[Message]:
        return []

    def format_unlabelled_messages( self, doc: dict ) -> list[Message]:
        return []

    def create_unlabelled_message_target( self, doc: dict ) -> None:
        return None

    def compute_metric( self, predictions=None, references=None ) -> dict:
        metric = self.metric.compute( predictions=predictions, references=references )
        assert metric is not None
        return metric


DIRECTORY: Mapping[str, Callable[[str], BaseInstructDataset]] = {
    '3.0.0': CNNDailymailInstructDataset,
}


def main():
    # pylint: disable=W0611
    # pylint: disable=C0415
    import os
    import rich

    cache_dir = os.environ[ 'HF_CACHE_DIR' ]

    rich.print( CNNDailymailInstructDataset( cache_dir ) )

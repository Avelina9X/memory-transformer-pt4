from collections.abc import Callable, Mapping
from datasets import DatasetDict, Dataset, load_dataset
from evaluate import load as load_metric

from ..task_base import BaseChoiceInstructDataset, InstructionDatasetTask, Message

class WinograndeInstructDataset( BaseChoiceInstructDataset ):
    def __init__( self, cache_dir: str ):
        self.metric = load_metric( 'accuracy' )
        super().__init__( cache_dir )

    def download( self, cache_dir: str ) -> DatasetDict:
        dataset = load_dataset( 'winogrande', 'winogrande_xl', cache_dir=cache_dir )
        assert isinstance( dataset, DatasetDict )
        return dataset

    @property
    def task_type( self ) -> InstructionDatasetTask:
        return InstructionDatasetTask.INSTRUCT_CLOSED

    @property
    def task_description( self ) -> str:
        return ''

    @property
    def task_name( self ) -> str:
        return 'winogrande'

    @property
    def task_subset( self ) -> str:
        return 'no_choice'

    def get_training_docs( self ) -> Dataset:
        return self.dataset[ 'train' ]

    def get_validation_docs( self ) -> Dataset:
        return self.dataset[ 'validation' ]

    def get_test_docs( self ) -> Dataset:
        return self.dataset[ 'test' ]

    def get_fewshot_docs( self ) -> None:
        return None

    def format_user_message( self, doc: dict ) -> Message:
        blank_idx = doc[ 'sentence' ].index( '_' )
        prompt = (
            f"Continue the following sentence:\n"
            f"\"{doc['sentence'][:blank_idx].strip()}\""
        )

        return Message(
            role='user',
            content=prompt,
            complete=True,
        )

    def _format_single_target( self, doc: dict ) -> Message:
        target_idx = self.create_unlabelled_message_target( doc )
        prefix = [ doc[ 'option1' ], doc[ 'option2' ] ][ target_idx ]

        blank_idx = doc[ 'sentence' ].index( '_' ) + 1
        suffix = doc[ 'sentence' ][ blank_idx : ].strip()

        return Message(
            role='assistant',
            content=f'{prefix} {suffix}',
            complete=True,
        )

    def _get_choices( self, doc: dict ) -> list:
        return [ '1', '2' ]

    def _get_label_key( self ) -> str:
        return 'answer'

    def create_unlabelled_message_target( self, doc: dict ) -> int:
        return int( doc[ 'answer' ] ) - 1

    def compute_metric( self, predictions=None, references=None ) -> dict:
        metric = self.metric.compute( predictions=predictions, references=references )
        assert metric is not None
        return metric


DIRECTORY: Mapping[str, Callable[[str], BaseChoiceInstructDataset]] = {
    'no_choice': WinograndeInstructDataset,
}


def main():
    # pylint: disable=W0611
    # pylint: disable=C0415
    import os

    cache_dir = os.environ[ 'HF_CACHE_DIR' ]

    WinograndeInstructDataset( cache_dir )

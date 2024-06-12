from collections.abc import Callable, Mapping
from datasets import DatasetDict, Dataset, load_dataset, DownloadConfig
from evaluate import load as load_metric

from ..task_base import BaseChoiceInstructDataset, InstructionDatasetTask, Message

class SciQNoChoiceInstructDataset( BaseChoiceInstructDataset ):
    def __init__( self, cache_dir: str ):
        self.metric = load_metric( 'accuracy', download_config=DownloadConfig( cache_dir=cache_dir ) )
        super().__init__( cache_dir )

    def download( self, cache_dir: str ) -> DatasetDict:
        dataset = load_dataset( 'allenai/sciq', cache_dir=cache_dir )
        dataset = dataset.map( lambda _: { 'label': 3 } )
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
        return 'sciq'

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
        
        if len( doc['support'] ) > 0:
            prompt = (
                f"Background: {doc['support'].lstrip()}\n"
                f"\n"
                f"Question: {doc['question']}\n"
                f"\n"
                f"Answer:"
            )
        else:
            prompt = (
                f"Question: {doc['question']}\n"
                f"\n"
                f"Answer:"
            )

        return Message(
            role='user',
            content=prompt,
            complete=True,
        )

    def _format_single_target( self, doc: dict ) -> Message:
        completions = [
            doc['distractor1'],
            doc['distractor2'],
            doc['distractor3'],
            doc['correct_answer'],
        ]
        return Message(
            role='assistant',
            content=completions[ doc[ 'label' ] ],
            complete=True,
        )

    def _get_choices( self, doc: dict ) -> list:
        return [ 0, 1, 2, 3 ]

    def _get_label_key( self ) -> str:
        return 'label'

    def create_unlabelled_message_target( self, doc: dict ) -> int:
        return doc[ 'label' ]

    def compute_metric( self, predictions=None, references=None ) -> dict:
        metric = self.metric.compute( predictions=predictions, references=references )
        assert metric is not None
        return metric


DIRECTORY: Mapping[str, Callable[[str], BaseChoiceInstructDataset]] = {
    'no_choice': SciQNoChoiceInstructDataset,
}


def main():
    # pylint: disable=W0611
    # pylint: disable=C0415
    import os

    cache_dir = os.environ[ 'HF_CACHE_DIR' ]

    SciQNoChoiceInstructDataset( cache_dir )

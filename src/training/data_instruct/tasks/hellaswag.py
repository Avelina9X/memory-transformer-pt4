from collections.abc import Callable, Mapping
from datasets import DatasetDict, Dataset, load_dataset
from evaluate import load as load_metric

from ..task_base import BaseChoiceInstructDataset, InstructionDatasetTask, Message

class HellaswagChoiceInstructDataset( BaseChoiceInstructDataset ):
    def __init__( self, cache_dir: str ):
        self.metric = load_metric( 'accuracy' )
        super().__init__( cache_dir )

    def download( self, cache_dir: str ) -> DatasetDict:
        dataset = load_dataset( 'Rowan/hellaswag', cache_dir=cache_dir, trust_remote_code=True )
        assert isinstance( dataset, DatasetDict )
        return dataset

    @property
    def task_type( self ) -> InstructionDatasetTask:
        return InstructionDatasetTask.MULTIPLE_CHOICE_OPEN

    @property
    def task_description( self ) -> str:
        return 'HellaSwag: Can a Machine Really Finish Your Sentence? is a new dataset for commonsense NLI. A paper was published at ACL2019.'

    @property
    def task_name( self ) -> str:
        return 'Hellaswag'

    @property
    def task_subset( self ) -> str:
        return 'choice'

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
            f"What is the most appropriate completion for the following text?\n"
            f"\"{doc['ctx']}\"\n"
            f"\n"
            f"Choices:\n"
            f"A. {doc['endings'][0]}\n"
            f"B. {doc['endings'][1]}\n"
            f"C. {doc['endings'][2]}\n"
            f"D. {doc['endings'][3]}\n"
            f"\n"
            f"Answer:"
        )

        return Message(
            role='user',
            content=prompt,
            complete=True,
        )

    def _format_single_target( self, doc: dict ) -> Message:
        idx = int(doc['label'])
        prompt = (
            f"{[ 'A', 'B', 'C', 'D' ][idx]}. {doc['endings'][idx]}"
        )

        return Message(
            role='assistant',
            content=prompt,
            complete=True,
        )

    def _get_choices( self, doc: dict ) -> list:
        return [ '0', '1', '2', '3' ]

    def _get_label_key( self ) -> str:
        return 'label'

    def create_unlabelled_message_target( self, doc: dict ) -> int:
        return int( doc[ 'label' ] )

    def compute_metric( self, predictions=None, references=None ) -> dict:
        metric = self.metric.compute( predictions=predictions, references=references )
        assert metric is not None
        return metric


class HellaswagNoChoiceInstructDataset( BaseChoiceInstructDataset ):
    def __init__( self, cache_dir: str ):
        self.metric = load_metric( 'accuracy' )
        super().__init__( cache_dir )

    def download( self, cache_dir: str ) -> DatasetDict:
        dataset = load_dataset( 'Rowan/hellaswag', cache_dir=cache_dir )
        assert isinstance( dataset, DatasetDict )
        return dataset

    @property
    def task_type( self ) -> InstructionDatasetTask:
        return InstructionDatasetTask.MULTIPLE_CHOICE_CLOSED

    @property
    def task_description( self ) -> str:
        return 'HellaSwag: Can a Machine Really Finish Your Sentence? is a new dataset for commonsense NLI. A paper was published at ACL2019.'

    @property
    def task_name( self ) -> str:
        return 'Hellaswag'

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
        prompt = (
            f"Continue the following sentence:\n"
            f"\"{doc['ctx']}\""
        )

        return Message(
            role='user',
            content=prompt,
            complete=True,
        )

    def _format_single_target( self, doc: dict ) -> Message:
        return Message(
            role='assistant',
            content=doc[ 'endings' ][ int( doc[ 'label' ] ) ],
            complete=True,
        )

    def _get_choices( self, doc: dict ) -> list:
        return [ '0', '1', '2', '3' ]

    def _get_label_key( self ) -> str:
        return 'label'

    def create_unlabelled_message_target( self, doc: dict ) -> int:
        return int( doc[ 'label' ] )

    def compute_metric( self, predictions=None, references=None ) -> dict:
        metric = self.metric.compute( predictions=predictions, references=references )
        assert metric is not None
        return metric


DIRECTORY: Mapping[str, Callable[[str], BaseChoiceInstructDataset]] = {
    'choice': HellaswagChoiceInstructDataset,
    'no_choice': HellaswagNoChoiceInstructDataset,
}


def main():
    # pylint: disable=W0611
    # pylint: disable=C0415
    import os

    cache_dir = os.environ[ 'HF_CACHE_DIR' ]

    HellaswagChoiceInstructDataset( cache_dir )
    HellaswagNoChoiceInstructDataset( cache_dir )

# OBQA - https://huggingface.co/datasets/allenai/openbookqa

# ARC-e, ARC-c - https://huggingface.co/datasets/allenai/ai2_arc

from collections.abc import Callable, Mapping
from datasets import DatasetDict, Dataset, load_dataset
from evaluate import load as load_metric

from ..task_base import BaseChoiceInstructDataset, InstructionDatasetTask, Message

class OBQAInstructDataset( BaseChoiceInstructDataset ):
    def __init__( self, cache_dir: str ):
        self.metric = load_metric( 'accuracy' )
        super().__init__( cache_dir )


    def download( self, cache_dir: str ) -> DatasetDict:
        dataset = load_dataset( 'allenai/openbookqa', 'main', cache_dir=cache_dir )
        assert isinstance( dataset, DatasetDict )
        return dataset

    @property
    def task_type( self ) -> InstructionDatasetTask:
        return InstructionDatasetTask.MULTIPLE_CHOICE_CLOSED

    @property
    def task_description( self ) -> str:
        return ''

    @property
    def task_name( self ) -> str:
        return 'obqa'

    @property
    def task_subset( self ) -> str:
        return 'main'

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
            f"Question: {doc['question_stem']}\n"
            f"\n"
            f"Choices:\n"
            f"{doc['choices']['label'][0]}. {doc['choices']['text'][0]}\n"
            f"{doc['choices']['label'][1]}. {doc['choices']['text'][1]}\n"
            f"{doc['choices']['label'][2]}. {doc['choices']['text'][2]}\n"
            f"{doc['choices']['label'][3]}. {doc['choices']['text'][3]}\n"
            f"\n"
            f"Answer:"
        )

        return Message(
            role='user',
            content=prompt,
            complete=True,
        )

    def _format_single_target( self, doc: dict ) -> Message:
        idx = self.create_unlabelled_message_target( doc )
        prompt = doc['choices']['label'][idx] + '. ' + doc['choices']['text'][idx]
        return Message(
            role='assistant',
            content=prompt,
            complete=True,
        )

    def _get_choices( self, doc: dict ) -> list:
        return doc['choices']['label']

    def _get_label_key( self ) -> str:
        return 'answerKey'

    def create_unlabelled_message_target( self, doc: dict ) -> int:
        return self._get_choices( doc ).index( self._get_label_key() )

    def compute_metric( self, predictions=None, references=None ) -> dict:
        metric = self.metric.compute( predictions=predictions, references=references )
        assert metric is not None
        return metric

DIRECTORY: Mapping[str, Callable[[str], BaseChoiceInstructDataset]] = {
    'main': OBQAInstructDataset
}


def main():
    # pylint: disable=W0611
    # pylint: disable=C0415
    import os
    import rich

    cache_dir = os.environ[ 'HF_CACHE_DIR' ]

    rich.print( DIRECTORY[ 'main' ]( cache_dir ), end='\n\n' )

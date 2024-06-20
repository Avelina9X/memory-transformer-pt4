from collections.abc import Callable, Mapping
import json
from datasets import DatasetDict, Dataset, load_dataset

from ..task_base import BaseInstructDataset, InstructionDatasetTask, Message

class UltrafeedbackInstructDataset( BaseInstructDataset ):
    def download( self, cache_dir: str ) -> DatasetDict:
        dataset = load_dataset( 'argilla/ultrafeedback-binarized-preferences-cleaned', cache_dir=cache_dir )
        assert isinstance( dataset, DatasetDict )
        return dataset.filter( lambda x: json.dumps( x[ 'rejected' ], sort_keys=True ) != json.dumps( x[ 'chosen' ], sort_keys=True ) )

    @property
    def task_type( self ) -> InstructionDatasetTask:
        return InstructionDatasetTask.INSTRUCT_CLOSED

    @property
    def task_name( self ) -> str:
        return 'ultrafeedback'

    @property
    def task_subset( self ) -> str:
        return 'binarized'

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
            'Below is an instruction that describes a task. '
            'Write a response that appropriately completes the request.'
        )

        return Message(
            role='system',
            content=prompt,
            complete=True,
        )

    def format_user_message( self, doc: dict ) -> Message:
        return Message(
            role='user',
            content=doc['prompt'],
            complete=True,
        )

    def format_target_messages( self, doc: dict ) -> list[Message]:
        return [ Message(
            role='assistant',
            content=doc['chosen'][0]['content'],
            complete=True,
        ) ]

    def format_distractor_messages( self, doc: dict ) -> list[Message]:
        return [ Message(
            role='assistant',
            content=doc['rejected'][0]['content'],
            complete=True,
        ) ]

    def format_unlabelled_messages( self, doc: dict ) -> list[Message]:
        return []

    def create_unlabelled_message_target( self, doc: dict ) -> None:
        return None

    def compute_metric( self, predictions=None, references=None ) -> dict:
        # TODO: add warning for using compute
        return {}


DIRECTORY: Mapping[str, Callable[[str], BaseInstructDataset]] = {
    'binarized': UltrafeedbackInstructDataset,
}

def main():
    # pylint: disable=W0611
    # pylint: disable=C0415
    import os
    import rich

    cache_dir = os.environ[ 'HF_CACHE_DIR' ]

    rich.print( UltrafeedbackInstructDataset( cache_dir ) )

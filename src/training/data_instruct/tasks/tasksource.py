from collections.abc import Callable, Mapping
from datasets import DatasetDict, Dataset, load_dataset

from ..task_base import BaseInstructDataset, InstructionDatasetTask, Message

class TasksourceInstructDataset( BaseInstructDataset ):
    def download( self, cache_dir: str ) -> DatasetDict:
        dataset = load_dataset( 'tasksource/tasksource-instruct-v0', cache_dir=cache_dir )
        assert isinstance( dataset, DatasetDict )
        return dataset
    
    @property
    def task_type( self ) -> InstructionDatasetTask:
        return InstructionDatasetTask.INSTRUCT_OPEN
    
    @property
    def task_name( self ) -> str:
        return 'tasksource'
    
    @property
    def task_subset( self ) -> str:
        return 'instruct-v0'

    def get_training_docs( self ) -> Dataset | None:
        return self.dataset[ 'train' ]

    def get_validation_docs( self ) -> Dataset | None:
        return self.dataset[ 'validation' ]

    def get_test_docs( self ) -> Dataset | None:
        return self.dataset[ 'test' ]

    def get_fewshot_docs( self ) -> Dataset | None:
        return None
    
    def format_user_message( self, doc: dict ) -> Message:
        return Message(
            role='user',
            content=doc['inputs'],
            complete=True,
        )

    def format_target_messages( self, doc: dict ) -> list[Message]:
        return [ Message(
            role='assistant',
            content=doc['targets'],
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


class TasksourcePairsInstructDataset( BaseInstructDataset ):
    def download( self, cache_dir: str ) -> DatasetDict:
        dataset = load_dataset( 'tasksource/tasksource_dpo_pairs', cache_dir=cache_dir )
        assert isinstance( dataset, DatasetDict )
        return dataset
    
    @property
    def task_type( self ) -> InstructionDatasetTask:
        return InstructionDatasetTask.INSTRUCT_OPEN
    
    @property
    def task_name( self ) -> str:
        return 'tasksource'
    
    @property
    def task_subset( self ) -> str:
        return 'dpo_pairs'

    def get_training_docs( self ) -> Dataset | None:
        return self.dataset[ 'train' ]

    def get_validation_docs( self ) -> Dataset | None:
        return self.dataset[ 'validation' ]

    def get_test_docs( self ) -> Dataset | None:
        return self.dataset[ 'test' ]

    def get_fewshot_docs( self ) -> Dataset | None:
        return None
    
    def format_user_message( self, doc: dict ) -> Message:
        return Message(
            role='user',
            content=doc['prompt'],
            complete=True,
        )

    def format_target_messages( self, doc: dict ) -> list[Message]:
        return [ Message(
            role='assistant',
            content=doc['chosen'],
            complete=True,
        ) ]

    def format_distractor_messages( self, doc: dict ) -> list[Message]:
        return [ Message(
            role='assistant',
            content=doc['rejected'],
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
    'instruct-v0': TasksourceInstructDataset,
    'dpo_pairs': TasksourcePairsInstructDataset,
}
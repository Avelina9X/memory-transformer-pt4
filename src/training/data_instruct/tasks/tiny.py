from collections.abc import Callable, Mapping
from datasets import DatasetDict, Dataset, load_dataset

from ..task_base import BaseInstructDataset, InstructionDatasetTask, Message

class TinyStoriesInstructDataset( BaseInstructDataset ):
    def __init__( self, cache_dir: str, split: str ):
        super().__init__( cache_dir )

        self.split = split

        if split not in [ 'instruct', 'summary' ]:
            raise ValueError( 'Split must be either `instruct` or `summary`' )

    def download( self, cache_dir: str ) -> DatasetDict:
        dataset = load_dataset( 'skeskinen/TinyStories-GPT4', cache_dir=cache_dir )
        assert isinstance( dataset, DatasetDict )
        return dataset

    @property
    def task_type( self ) -> InstructionDatasetTask:
        return InstructionDatasetTask.INSTRUCT_CLOSED

    @property
    def task_name( self ) -> str:
        return 'tinystories-gpt4'

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

    def format_user_message( self, doc: dict ) -> Message:
        match self.split:
            case 'instruct':
                prompt = doc['prompt']

            case 'summary':
                prompt = (
                    f"Based on the following summary write a short story (3-5 paragraphs) "
                    f"which only uses very simple words that a 3 year old child would understand."
                    f"\n\n"
                    f"Summary: {doc['summary']}\n"
                    f"\n"
                    f"Story:"
                )

            case _:
                assert False

        return Message(
            role='user',
            content=prompt,
            complete=True,
        )

    def format_target_messages( self, doc: dict ) -> list[Message]:
        return [ Message(
            role='assistant',
            content=doc['story'],
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
    'instruct': lambda cache_dir: TinyStoriesInstructDataset( cache_dir=cache_dir, split='instruct' ),
    'summary': lambda cache_dir: TinyStoriesInstructDataset( cache_dir=cache_dir, split='summary' ),
}

def main():
    # pylint: disable=W0611
    # pylint: disable=C0415
    import os
    import rich

    cache_dir = os.environ[ 'HF_CACHE_DIR' ]

    rich.print( TinyStoriesInstructDataset( cache_dir, split='instruct' ) )
    rich.print( TinyStoriesInstructDataset( cache_dir, split='summary' ) )

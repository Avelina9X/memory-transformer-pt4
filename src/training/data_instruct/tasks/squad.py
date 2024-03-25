import functools
from collections.abc import Callable, Mapping
from datasets import DatasetDict, Dataset, load_dataset
from evaluate import load as load_metric
import spacy

from ..task_base import BaseInstructDataset, InstructionDatasetTask, Message

class SquadBaseInstructDataset( BaseInstructDataset ):
    @property
    def task_name( self ) -> str:
        return 'squad'

    def get_training_docs( self ) -> Dataset:
        return self.dataset[ 'train' ]

    def get_validation_docs( self ) -> Dataset:
        return self.dataset[ 'validation' ]

    def get_test_docs( self ) -> None:
        return None

    def get_fewshot_docs( self ) -> None:
        return None

    def format_user_message( self, doc: dict ) -> Message:
        prompt = (
            f'Title: {doc["title"]}\n'
            f'\n'
            f'Background: {doc["context"]}\n'
            f'\n'
            f'Question: {doc["question"]}\n'
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


class SquadV1InstructDataset( SquadBaseInstructDataset ):
    def __init__( self, cache_dir: str ):
        self.metric = load_metric( 'squad' )
        super().__init__( cache_dir )

    def download( self, cache_dir: str ) -> DatasetDict:
        dataset = load_dataset( 'rajpurkar/squad', cache_dir=cache_dir )
        assert isinstance( dataset, DatasetDict )
        return dataset

    @property
    def task_type( self ) -> InstructionDatasetTask:
        return InstructionDatasetTask.SPAN_EXTRACTION

    @property
    def task_subset( self ) -> str:
        return 'v1.1'

    def format_target_messages( self, doc: dict ) -> list[Message]:
        return [
            Message(
                role='assistant',
                content=answer,
                complete=True,
            )
            for answer in doc[ 'answers' ][ 'text' ]
        ]

    def format_distractor_messages( self, doc: dict ) -> list[Message]:
        return []

    def compute_metric( self, predictions=None, references=None ) -> dict:
        metric = self.metric.compute( predictions=predictions, references=references )
        assert metric is not None
        return metric


class SquadV2InstructDataset( SquadBaseInstructDataset ):
    def __init__( self, cache_dir: str ):
        self.metric = load_metric( 'squad_v2' )
        self._nlp = spacy.load( 'en_core_web_md' )
        super().__init__( cache_dir )

        self.get_noun_set = functools.cache( self._get_noun_set )

    def _get_noun_set( self, context: str ) -> set[str]:
        document = self._nlp( context )
        return set( chunk.text for chunk in document.noun_chunks )

    def download( self, cache_dir: str ) -> DatasetDict:
        dataset = load_dataset( 'rajpurkar/squad_v2', cache_dir=cache_dir )
        assert isinstance( dataset, DatasetDict )
        return dataset

    @property
    def task_type( self ) -> InstructionDatasetTask:
        return InstructionDatasetTask.SPAN_EXTRACTION_V2

    @property
    def task_subset( self ) -> str:
        return 'v2'

    def format_target_messages( self, doc: dict ) -> list[Message]:
        if len( doc[ 'answers' ][ 'text' ] ) > 0:
            return [
                Message(
                    role='assistant',
                    content=answer,
                    complete=True,
                )
                for answer in doc[ 'answers' ][ 'text' ]
            ]
        else:
            return [
                Message(
                    role='assistant',
                    content='unanswerable',
                    complete=True,
                )
            ]

    def format_distractor_messages( self, doc: dict ) -> list[Message]:
        if len( doc[ 'answers' ][ 'text' ] ) > 0:
            return [
                Message(
                    role='assistant',
                    content='unanswerable',
                    complete=True,
                )
            ]
        else:
            chunk_set = self.get_noun_set( doc["context"] )
            return [
                Message(
                    role='assistant',
                    content=chunk,
                    complete=True,
                )
                for chunk in chunk_set
            ]

    def compute_metric( self, predictions=None, references=None ) -> dict:
        metric = self.metric.compute( predictions=predictions, references=references )
        assert metric is not None
        return metric


DIRECTORY: Mapping[str, Callable[[str], SquadBaseInstructDataset]] = {
    'v1.1': SquadV1InstructDataset,
    'v2': SquadV2InstructDataset,
}


def main():
    # pylint: disable=W0611
    # pylint: disable=C0415
    import os
    import rich

    cache_dir = os.environ[ 'HF_CACHE_DIR' ]

    rich.print( SquadV1InstructDataset( cache_dir ) )
    rich.print( SquadV2InstructDataset( cache_dir ) )

    squad_v2 = SquadV2InstructDataset( cache_dir )

    dataset = squad_v2.get_training_docs()

    rich.print( squad_v2.format_user_message( dataset[0] ) )
    rich.print( squad_v2.format_target_messages( dataset[0] ) )
    rich.print( squad_v2.format_distractor_messages( dataset[0] ) )
    rich.print()
    rich.print( squad_v2.format_user_message( dataset[-10] ) )
    rich.print( squad_v2.format_target_messages( dataset[-10] ) )
    rich.print( squad_v2.format_distractor_messages( dataset[-10] ) )
    rich.print()

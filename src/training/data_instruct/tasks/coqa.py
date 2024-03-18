from collections.abc import Callable, Mapping
from itertools import count
from datasets import DatasetDict, Dataset, load_dataset

from ..task_base import BaseInstructDataset, InstructionDatasetTask, Message, MessageList

class COQAInstructDataset( BaseInstructDataset ):

    def download( self, cache_dir: str ) -> DatasetDict:
        dataset = load_dataset( 'stanfordnlp/coqa', cache_dir=cache_dir )
        assert isinstance( dataset, DatasetDict )
        return dataset

    @property
    def task_type( self ) -> InstructionDatasetTask:
        return InstructionDatasetTask.CONVERSATIONAL

    @property
    def task_name( self ) -> str:
        return 'coqa'

    @property
    def task_subset( self ) -> str:
        return 'all'

    def get_training_docs( self ) -> Dataset:
        return self.dataset[ 'train' ]

    def get_validation_docs( self ) -> Dataset:
        return self.dataset[ 'validation' ]

    def get_test_docs( self ) -> None:
        return None

    def get_fewshot_docs( self ) -> None:
        return None


    def format_system_message( self, doc: dict ) -> Message:
        prompt = (
            'Below is a background context and multiple questions. '
            'Respond with text from the background that correctly answers each question. '
            'If a question cannot be answered using the background context respond with "unknown".'
        )

        return Message(
            role='system',
            content=prompt,
            complete=True
        )

    def format_user_message( self, doc: dict ) -> Message:
        prompt = ''
        if doc[ 'idx' ] == 0:
            prompt += f"Background: {doc['story']}\n\n"
        prompt += f"{doc['question']}"

        return Message(
            role='user',
            content=prompt,
            complete=True,
        )

    def format_target_messages( self, doc: dict ) -> MessageList:
        return [ Message(
            role='assistant',
            content=doc['answer'],
            complete=True,
        ) ]

    def format_distractor_messages( self, doc: dict ) -> list[Message]:
        raise NotImplementedError( 'This does not use the message factory' )

    def format_unlabelled_messages( self, doc: dict ) -> list[Message]:
        return []

    def create_target_message_list( self, doc: dict ) -> list[MessageList]:
        message_list = [ self.format_system_message( doc ) ]

        doc_iterator = zip(
            doc[ 'questions' ],
            doc[ 'answers' ][ 'input_text' ],
            count()
        )

        for question, answer, idx in doc_iterator:
            message_list.append(
                self.format_user_message( {
                    'story': doc[ 'story' ],
                    'question': question,
                    'idx': idx
                } )
            )

            message_list.append(
                self.format_target_messages( {
                    'answer': answer
                } )[0]
            )

        return [ message_list ]

    def create_unlabelled_message_target( self, doc: dict ) -> None:
        return None

    def compute_metric( self, predictions=None, references=None ) -> dict:
        # TODO: add warning for using compute
        return {}

DIRECTORY: Mapping[str, Callable[[str], BaseInstructDataset]] = {
    'all': COQAInstructDataset
}

def main():
    # pylint: disable=W0611
    # pylint: disable=C0415
    import os
    import rich

    cache_dir = os.environ[ 'HF_CACHE_DIR' ]

    ds = COQAInstructDataset( cache_dir )

    rich.print( ds.create_target_message_list( ds.get_training_docs()[0] ) )

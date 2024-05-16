from collections.abc import Callable, Mapping
from datasets import DatasetDict, Dataset, load_dataset, DownloadConfig
from evaluate import load as load_metric

from ..task_base import BaseChoiceInstructDataset, InstructionDatasetTask, Message, MessageList

class GlueBaseInstructDataset( BaseChoiceInstructDataset ):
    def __init__( self, cache_dir: str ):
        self.metric = load_metric( 'glue', self.task_subset, download_config=DownloadConfig( cache_dir=cache_dir ) )
        super().__init__( cache_dir )

    def download( self, cache_dir: str ) -> DatasetDict:
        dataset = load_dataset( 'glue', self.task_subset, cache_dir=cache_dir )
        assert isinstance( dataset, DatasetDict )
        return dataset

    @property
    def task_type( self ) -> InstructionDatasetTask:
        return InstructionDatasetTask.INSTRUCT_OPEN

    @property
    def task_name( self ) -> str:
        return 'GLUE'

    def get_training_docs( self ) -> Dataset | None:
        return self.dataset[ 'train' ]

    def get_validation_docs( self ) -> Dataset | None:
        return self.dataset[ 'validation' ]

    def get_test_docs( self ) -> Dataset | None:
        return self.dataset[ 'test' ]

    def get_fewshot_docs( self ) -> Dataset | None:
        return None

    def create_unlabelled_message_target( self, doc: dict ) -> int | float | None:
        return doc['label']

    def _get_label_key( self ) -> str:
        return 'label'

    def compute_metric( self, predictions=None, references=None ) -> dict:
        metric = self.metric.compute( predictions=predictions, references=references )
        assert metric is not None
        return metric


class GlueColaInstructDataset( GlueBaseInstructDataset ):

    @property
    def task_description( self ) -> str:
        return 'Corpus of Linguistic Acceptability. The task is to determine the grammatical correctness of a sentence.'

    @property
    def task_subset( self ) -> str:
        return 'cola'

    def format_user_message( self, doc: dict ) -> Message:
        prompt = (
            f'Given the following sentence, answer the question with "yes" or "no".\n'
            f'\n'
            f'Sentence: {doc["sentence"]}\n'
            f'\n'
            f'Question: Does this sentence make sense?\n'
            f'\n'
            f'Answer:'
        )

        return Message(
            role='user',
            content=prompt,
            complete=True,
        )

    def _format_single_target( self, doc: dict ) -> Message:
        return Message(
            role='assistant',
            content=[ 'no', 'yes' ][ doc['label'] ],
            complete=True,
        )

    def _get_choices( self, doc: dict ) -> list:
        return [ 0, 1 ]


class GlueMNLIInstructDataset( GlueBaseInstructDataset ):

    @property
    def task_description( self ) -> str:
        return 'Multi-Genre Natural Language Inference Corpus. The task is to predict whether a premise entails a hypothesis.'

    @property
    def task_subset( self ) -> str:
        return 'mnli'

    def get_training_docs( self ) -> Dataset | None:
        return self.dataset[ 'train' ]

    def get_validation_docs( self ) -> Dataset | None:
        return None

    def get_test_docs( self ) -> Dataset | None:
        return None

    def format_user_message( self, doc: dict ) -> Message:
        prompt = (
            f'Given a premise statement and a hypothesis statement, '
            f'respond with "True" if the premise entails the hypothesis, '
            f'respond with "False" if the premise contradicts the hypothesis, '
            f'or respond with "Neither" if the statements are neurtral.\n'
            f'\n'
            f'Premise: {doc["premise"]}\n'
            f'\n'
            f'Hypothesis: {doc["hypothesis"]}\n'
            f'\n'
            f'Question: True, False or Neither?\n'
            f'\n'
            f'Answer:'
        )

        return Message(
            role='user',
            content=prompt,
            complete=True,
        )

    def _format_single_target( self, doc: dict ) -> Message:
        return Message(
            role='assistant',
            content=[ 'True', 'Neither', 'False' ][ doc['label'] ],
            complete=True,
        )

    def _get_choices( self, doc: dict ) -> list:
        return [ 0, 1, 2 ]

class GlueMNLIMatchedInstructDataset( GlueMNLIInstructDataset ):

    @property
    def task_subset( self ) -> str:
        return 'mnli_matched'

    def get_training_docs( self ) -> None:
        return None

    def get_validation_docs( self ) -> Dataset:
        return self.dataset[ 'validation' ]

    def get_test_docs( self ) -> Dataset:
        return self.dataset[ 'test' ]

class GlueMNLIMismatchedInstructDataset( GlueMNLIInstructDataset ):

    @property
    def task_subset( self ) -> str:
        return 'mnli_mismatched'

    def get_training_docs( self ) -> None:
        return None

    def get_validation_docs( self ) -> Dataset:
        return self.dataset[ 'validation' ]

    def get_test_docs( self ) -> Dataset:
        return self.dataset[ 'test' ]

class GlueAXInstructDataset( GlueMNLIInstructDataset ):
    def __init__( self, cache_dir: str ):
        self.metric = None
        super( BaseChoiceInstructDataset, self ).__init__( cache_dir )

    @property
    def task_subset( self ) -> str:
        return 'ax'

    def get_training_docs( self ) -> Dataset | None:
        return None

    def get_validation_docs( self ) -> Dataset | None:
        return None

    def get_test_docs( self ) -> Dataset | None:
        return self.dataset[ 'test' ]


class GlueMRPCInstructDataset( GlueBaseInstructDataset ):

    @property
    def task_description( self ) -> str:
        return 'Microsoft Research Paraphrase Corpus. The task is to determine the semantic equivilence of 2 sentences.'

    @property
    def task_subset( self ) -> str:
        return 'mrpc'

    def format_user_message( self, doc: dict ) -> Message:
        prompt = (
            f'Given the following sentences, answer the question with "yes" or "no".\n'
            f'\n'
            f'Sentence 1: {doc["sentence1"]}\n'
            f'\n'
            f'Sentence 2: {doc["sentence2"]}\n'
            f'\n'
            f'Question: Do both sentences mean the same thing?\n'
            f'\n'
            f'Answer:'
        )

        return Message(
            role='user',
            content=prompt,
            complete=True,
        )

    def _format_single_target( self, doc: dict ) -> Message:
        return Message(
            role='assistant',
            content=[ 'no', 'yes' ][ doc['label'] ],
            complete=True,
        )

    def _get_choices( self, doc: dict ) -> list:
        return [ 0, 1 ]


class GlueQNLIInstructDataset( GlueBaseInstructDataset ):

    @property
    def task_description( self ) -> str:
        return 'Stanford Question Answering Dataset. The task is to determine if a given sentence answers a question.'

    @property
    def task_subset( self ) -> str:
        return 'qnli'

    def format_user_message( self, doc: dict ) -> Message:
        prompt = (
            f'Given the following sentences, answer the question with "yes" or "no".\n'
            f'\n'
            f'Sentence 1: {doc["question"]}\n'
            f'\n'
            f'Sentence 2: {doc["sentence"]}\n'
            f'\n'
            f'Question: Does Sentence 2 correctly answer Sentence 1?\n'
            f'\n'
            f'Answer:'
        )

        return Message(
            role='user',
            content=prompt,
            complete=True,
        )

    def _format_single_target( self, doc: dict ) -> Message:
        return Message(
            role='assistant',
            content=[ 'yes', 'no' ][ doc['label'] ],
            complete=True,
        )

    def _get_choices( self, doc: dict ) -> list:
        return [ 0, 1 ]


class GlueQQPInstructDataset( GlueBaseInstructDataset ):

    @property
    def task_description( self ) -> str:
        return 'Quora Question Pairs2 dataset. The task is to determine if 2 questions are equivilent.'

    @property
    def task_subset( self ) -> str:
        return 'qqp'

    def format_user_message( self, doc: dict ) -> Message:
        prompt = (
            f'Given the following sentences, answer the question with "yes" or "no".\n'
            f'\n'
            f'Sentence 1: {doc["question1"]}\n'
            f'\n'
            f'Sentence 2: {doc["question2"]}\n'
            f'\n'
            f'Question: Do both sentences ask the same question?\n'
            f'\n'
            f'Answer:'
        )

        return Message(
            role='user',
            content=prompt,
            complete=True,
        )

    def _format_single_target( self, doc: dict ) -> Message:
        return Message(
            role='assistant',
            content=[ 'no', 'yes' ][ doc['label'] ],
            complete=True,
        )

    def _get_choices( self, doc: dict ) -> list:
        return [ 0, 1 ]


class GlueRTEInstructDataset( GlueBaseInstructDataset ):

    @property
    def task_description( self ) -> str:
        return 'Recognizing Textual Entailment datasets. The task is to determine if 2 sentences have the same meaning.'

    @property
    def task_subset( self ) -> str:
        return 'rte'

    def format_user_message( self, doc: dict ) -> Message:
        prompt = (
            f'Given the following sentences, answer the question with "yes" or "no".\n'
            f'\n'
            f'Sentence 1: {doc["sentence1"]}\n'
            f'\n'
            f'Sentence 2: {doc["sentence2"]}\n'
            f'\n'
            f'Question: Do both sentences mean the same thing?\n'
            f'\n'
            f'Answer:'
        )

        return Message(
            role='user',
            content=prompt,
            complete=True,
        )

    def _format_single_target( self, doc: dict ) -> Message:
        return Message(
            role='assistant',
            content=[ 'yes', 'no' ][ doc['label'] ],
            complete=True,
        )

    def _get_choices( self, doc: dict ) -> list:
        return [ 0, 1 ]


class GlueSST2InstructDataset( GlueBaseInstructDataset ):

    @property
    def task_description( self ) -> str:
        return 'Stanford Sentiment Treebank. The task is to determine the sentiment of a given sentence.'

    @property
    def task_subset( self ) -> str:
        return 'sst2'

    def format_user_message( self, doc: dict ) -> Message:
        prompt = (
            f'Given the following sentence, answer the question with "positive" or "negative".\n'
            f'\n'
            f'Sentence: {doc["sentence"]}\n'
            f'\n'
            f'Question: Is this sentence positive or negative?\n'
            f'\n'
            f'Answer:'
        )

        return Message(
            role='user',
            content=prompt,
            complete=True,
        )

    def _format_single_target( self, doc: dict ) -> Message:
        return Message(
            role='assistant',
            content=[ 'negative', 'positive' ][ doc['label'] ],
            complete=True,
        )

    def _get_choices( self, doc: dict ) -> list:
        return [ 0, 1 ]


class GlueSTSBInstructDataset( GlueBaseInstructDataset ):
    @property
    def task_description( self ) -> str:
        return 'The Semantic Textual Similarity Benchmark. Each pair is human-annotated with a similarity score from 1 to 5.'

    @property
    def task_subset( self ) -> str:
        return 'stsb'

    def format_user_message( self, doc: dict ) -> Message:
        prompt = (
            f'Given the following sentences, answer the question with a number between 0 and 5.\n'
            f'\n'
            f'Sentence 1: {doc["sentence1"]}\n'
            f'\n'
            f'Sentence 2: {doc["sentence2"]}\n'
            f'\n'
            f'Question: On a scale of 0 to 5 how similar are Sentence 1 and Sentence 2?\n'
            f'\n'
            f'Answer:'
        )

        return Message(
            role='user',
            content=prompt,
            complete=True,
        )

    def _format_single_target( self, doc: dict ) -> Message:
        return Message(
            role='assistant',
            content=f"{doc['label']}",
            complete=True,
        )

    def _get_choices( self, doc: dict ) -> list:
        return [ 0, 1, 2, 3, 4, 5 ]

    def format_unlabelled_messages( self, doc: dict ) -> MessageList:
        return [
            self._format_single_target( dict( doc, **{ self._get_label_key() : i } ) )
            for i in self._get_choices( doc )
        ]

    def format_target_messages( self, doc: dict ) -> MessageList:
        target_idx = self.create_unlabelled_message_target( doc )
        assert target_idx is not None

        return [
            msg
            for i, msg in enumerate( self.format_unlabelled_messages( doc ) )
            if i == round( target_idx )
        ]

    def format_distractor_messages( self, doc: dict ) -> MessageList:
        target_idx = self.create_unlabelled_message_target( doc )
        assert target_idx is not None

        return [
            msg
            for i, msg in enumerate( self.format_unlabelled_messages( doc ) )
            if i != round( target_idx )
        ]


class GlueWNLIInstructDataset( GlueBaseInstructDataset ):

    @property
    def task_description( self ) -> str:
        return 'Winograd Schema Challenge. The task is to determine if a second sentence is true given the first.'

    @property
    def task_subset( self ) -> str:
        return 'wnli'

    def format_user_message( self, doc: dict ) -> Message:
        prompt = (
            f'Given the following sentences, answer the question with "yes" or "no".\n'
            f'\n'
            f'Sentence 1: {doc["sentence1"]}\n'
            f'\n'
            f'Sentence 2: {doc["sentence2"]}\n'
            f'\n'
            f'Question: Based on the information in Sentence 1, can we concluded that Sentence 2 is true?\n'
            f'\n'
            f'Answer:'
        )

        return Message(
            role='user',
            content=prompt,
            complete=True,
        )

    def _format_single_target( self, doc: dict ) -> Message:
        return Message(
            role='assistant',
            content=[ 'no', 'yes' ][ doc['label'] ],
            complete=True,
        )

    def _get_choices( self, doc: dict ) -> list:
        return [ 0, 1 ]


DIRECTORY: Mapping[str, Callable[[str], GlueBaseInstructDataset]] = {
    'ax': GlueAXInstructDataset,
    'cola': GlueColaInstructDataset,
    'mnli': GlueMNLIInstructDataset,
    'mnli_matched': GlueMNLIMatchedInstructDataset,
    'mnli_mismatched': GlueMNLIMismatchedInstructDataset,
    'mrpc': GlueMRPCInstructDataset,
    'qnli': GlueQNLIInstructDataset,
    'qqp': GlueQQPInstructDataset,
    'rte': GlueRTEInstructDataset,
    'sst2': GlueSST2InstructDataset,
    'stsb': GlueSTSBInstructDataset,
    'wnli': GlueWNLIInstructDataset,
}


def main():
    # pylint: disable=W0611
    # pylint: disable=C0415
    import os
    import rich

    cache_dir = os.environ[ 'HF_CACHE_DIR' ]

    rich.print( GlueColaInstructDataset( cache_dir ), end='\n\n' )
    rich.print( GlueMNLIInstructDataset( cache_dir ), end='\n\n' )
    rich.print( GlueMNLIMatchedInstructDataset( cache_dir ), end='\n\n' )
    rich.print( GlueMNLIMismatchedInstructDataset( cache_dir ), end='\n\n' )
    rich.print( GlueMRPCInstructDataset( cache_dir ), end='\n\n' )
    rich.print( GlueQNLIInstructDataset( cache_dir ), end='\n\n' )
    rich.print( GlueQQPInstructDataset( cache_dir ), end='\n\n' )
    rich.print( GlueRTEInstructDataset( cache_dir ), end='\n\n' )
    rich.print( GlueSST2InstructDataset( cache_dir ), end='\n\n' )
    rich.print( GlueSTSBInstructDataset( cache_dir ), end='\n\n' )
    rich.print( GlueWNLIInstructDataset( cache_dir ), end='\n\n' )

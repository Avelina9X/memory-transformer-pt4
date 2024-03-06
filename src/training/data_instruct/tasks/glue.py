from datasets import DatasetDict, Dataset, load_dataset

from ..task_base import BaseInstructDataset, InstructionDatasetTask

class GlueBaseInstructDataset( BaseInstructDataset ):
    
    def download( self, cache_dir: str ) -> DatasetDict:
        return load_dataset( 'glue', self.task_name, cache_dir=cache_dir ) # type: ignore
    
    @property
    def task_type( self ) -> InstructionDatasetTask:
        return InstructionDatasetTask.INSTRUCT_OPEN
    
    @property
    def group_name( self ) -> str | None: return 'GLUE'
    
    def get_training_docs( self ) -> Dataset:
        return self.dataset[ 'train' ]
    
    def get_validation_docs( self ) -> Dataset:
        return self.dataset[ 'validation' ]
    
    def get_test_docs( self ) -> Dataset:
        return self.dataset[ 'test' ]
    
    def get_fewshot_docs( self ) -> None:
        return None
    
    def create_unlabelled_message_targets( self, doc: dict ) -> int | None:
        return None if doc['label'] < 0 else doc['label']


class GlueColaInstructDataset( GlueBaseInstructDataset ):
    
    @property
    def task_description( self ) -> str:
        return 'Corpus of Linguistic Acceptability. The task is to determine the grammatical correctness of a sentence.'
    
    @property
    def task_name( self ) -> str:
        return 'cola'
    
    def format_user_message( self, doc: dict ) -> dict:
        prompt = (
            f'Given the following sentence, answer the question with "yes" or "no".\n'
            f'\n'
            f'Sentence: {doc["sentence"]}\n'
            f'\n'
            f'Question: Does this sentence make sense?\n'
            f'\n'
            f'Answer:'
        )
        
        return {
            'role': 'user',
            'content': prompt,
            'complete': True,
        }
    
    def _format_single_target( self, doc: dict ) -> dict:
        return {
            'role': 'assistant',
            'content': [ 'no', 'yes' ][ doc['label'] ],
            'complete': True,
        }
    
    def format_target_messages( self, doc: dict ) -> list[dict]:
        return [
            self._format_single_target( doc )
        ]
    
    def format_distractor_messages( self, doc: dict ) -> list[dict]:      
        return [
            self._format_single_target( dict( doc, label=i ) )
            for i in [ 0, 1 ]
            if i != doc['label']
        ]
    
    def format_unlabelled_messages( self, doc: dict ) -> list[dict]:
        return [
            self._format_single_target( dict( doc, label=i ) )
            for i in [ 0, 1 ]
        ]


class GlueMNLIInstructDataset( GlueBaseInstructDataset ):
    
    @property
    def task_description( self ) -> str:
        return 'Multi-Genre Natural Language Inference Corpus. The task is to predict whether a premise entails a hypothesis.'
    
    @property
    def task_name( self ) -> str:
        return 'mnli'
    
    def get_training_docs( self ) -> Dataset:
        return self.dataset[ 'train' ]
    
    def get_validation_docs( self ) -> None:
        return None
    
    def get_test_docs( self ) -> None:
        return None
    
    def format_user_message( self, doc: dict ) -> dict:
        prompt = (
            f'Given a premise statement and a hypothesis statment, '
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
        
        return {
            'role': 'user',
            'content': prompt,
            'complete': True,
        }
    
    def _format_single_target( self, doc: dict ) -> dict:
        return {
            'role': 'assistant',
            'content': [  'True', 'Neither', 'False' ][ doc['label'] ],
            'complete': True,
        }
    
    def format_target_messages( self, doc: dict ) -> list[dict]:
        return [
            self._format_single_target( doc )
        ]
    
    def format_distractor_messages( self, doc: dict ) -> list[dict]:      
        return [
            self._format_single_target( dict( doc, label=i ) )
            for i in [ 0, 1, 2 ]
            if i != doc['label']
        ]
    
    def format_unlabelled_messages( self, doc: dict ) -> list[dict]:
        return [
            self._format_single_target( dict( doc, label=i ) )
            for i in [ 0, 1, 2 ]
        ]

class GlueMNLIMatchedInstructDataset( GlueMNLIInstructDataset ):
    
    @property
    def task_name( self ) -> str:
        return 'mnli_matched'
    
    def get_training_docs( self ) -> None:
        return None
    
    def get_validation_docs( self ) -> Dataset:
        return self.dataset[ 'validation' ]
    
    def get_test_docs( self ) -> Dataset:
        return self.dataset[ 'test' ]

class GlueMNLIMismatchedInstructDataset( GlueMNLIInstructDataset ):
    
    @property
    def task_name( self ) -> str:
        return 'mnli_mismatched'
    
    def get_training_docs( self ) -> None:
        return None
    
    def get_validation_docs( self ) -> Dataset:
        return self.dataset[ 'validation' ]
    
    def get_test_docs( self ) -> Dataset:
        return self.dataset[ 'test' ]


class GlueMRPCInstructDataset( GlueBaseInstructDataset ):
    
    @property
    def task_description( self ) -> str:
        return 'Microsoft Research Paraphrase Corpus. The task is to determine the semantic equivilence of 2 sentences.'
    
    @property
    def task_name( self ) -> str:
        return 'mrpc'
    
    def format_user_message( self, doc: dict ) -> dict:
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
        
        return {
            'role': 'user',
            'content': prompt,
            'complete': True,
        }
    
    def _format_single_target( self, doc: dict ) -> dict:
        return {
            'role': 'assistant',
            'content': [ 'no', 'yes' ][ doc['label'] ],
            'complete': True,
        }
    
    def format_target_messages( self, doc: dict ) -> list[dict]:
        return [
            self._format_single_target( doc )
        ]
    
    def format_distractor_messages( self, doc: dict ) -> list[dict]:      
        return [
            self._format_single_target( dict( doc, label=i ) )
            for i in [ 0, 1 ]
            if i != doc['label']
        ]
    
    def format_unlabelled_messages( self, doc: dict ) -> list[dict]:
        return [
            self._format_single_target( dict( doc, label=i ) )
            for i in [ 0, 1 ]
        ]


class GlueQNLIInstructDataset( GlueBaseInstructDataset ):
    
    @property
    def task_description( self ) -> str:
        return 'Stanford Question Answering Dataset. The task is to determine if a given sentence answers a question.'
    
    @property
    def task_name( self ) -> str:
        return 'qnli'

    def format_user_message( self, doc: dict ) -> dict:
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
        
        return {
            'role': 'user',
            'content': prompt,
            'complete': True,
        }
    
    def _format_single_target( self, doc: dict ) -> dict:
        return {
            'role': 'assistant',
            'content': [ 'yes', 'no' ][ doc['label'] ],
            'complete': True,
        }
    
    def format_target_messages( self, doc: dict ) -> list[dict]:
        return [
            self._format_single_target( doc )
        ]
    
    def format_distractor_messages( self, doc: dict ) -> list[dict]:      
        return [
            self._format_single_target( dict( doc, label=i ) )
            for i in [ 0, 1 ]
            if i != doc['label']
        ]
    
    def format_unlabelled_messages( self, doc: dict ) -> list[dict]:
        return [
            self._format_single_target( dict( doc, label=i ) )
            for i in [ 0, 1 ]
        ]


class GlueQQPInstructDataset( GlueBaseInstructDataset ):
    
    @property
    def task_description( self ) -> str:
        return 'Quora Question Pairs2 dataset. The task is to determine if 2 questions are equivilent.'
    
    @property
    def task_name( self ) -> str:
        return 'qqp'
    
    def format_user_message( self, doc: dict ) -> dict:
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
        
        return {
            'role': 'user',
            'content': prompt,
            'complete': True,
        }
    
    def _format_single_target( self, doc: dict ) -> dict:
        return {
            'role': 'assistant',
            'content': [ 'no', 'yes' ][ doc['label'] ],
            'complete': True,
        }
    
    def format_target_messages( self, doc: dict ) -> list[dict]:
        return [
            self._format_single_target( doc )
        ]
    
    def format_distractor_messages( self, doc: dict ) -> list[dict]:      
        return [
            self._format_single_target( dict( doc, label=i ) )
            for i in [ 0, 1 ]
            if i != doc['label']
        ]
    
    def format_unlabelled_messages( self, doc: dict ) -> list[dict]:
        return [
            self._format_single_target( dict( doc, label=i ) )
            for i in [ 0, 1 ]
        ]


class GlueRTEInstructDataset( GlueBaseInstructDataset ):
    
    @property
    def task_description( self ) -> str:
        return 'Recognizing Textual Entailment datasets. The task is to determine if 2 sentences have the same meaning.'
    
    @property
    def task_name( self ) -> str:
        return 'rte'
    
    def format_user_message( self, doc: dict ) -> dict:
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
        
        return {
            'role': 'user',
            'content': prompt,
            'complete': True,
        }
    
    def _format_single_target( self, doc: dict ) -> dict:
        return {
            'role': 'assistant',
            'content': [ 'yes', 'no' ][ doc['label'] ],
            'complete': True,
        }
    
    def format_target_messages( self, doc: dict ) -> list[dict]:
        return [
            self._format_single_target( doc )
        ]
    
    def format_distractor_messages( self, doc: dict ) -> list[dict]:      
        return [
            self._format_single_target( dict( doc, label=i ) )
            for i in [ 0, 1 ]
            if i != doc['label']
        ]
    
    def format_unlabelled_messages( self, doc: dict ) -> list[dict]:
        return [
            self._format_single_target( dict( doc, label=i ) )
            for i in [ 0, 1 ]
        ]


class GlueSST2InstructDataset( GlueBaseInstructDataset ):
    
    @property
    def task_description( self ) -> str:
        return 'Stanford Sentiment Treebank. The task is to determine the sentiment of a given sentence.'
    
    @property
    def task_name( self ) -> str:
        return 'sst2'
    
    def format_user_message( self, doc: dict ) -> dict:
        prompt = (
            f'Given the following sentence, answer the question with "positive" or "negative".\n'
            f'\n'
            f'Sentence: {doc["sentence"]}\n'
            f'\n'
            f'Question: Is this sentence positive or negative?\n'
            f'\n'
            f'Answer:'
        )
        
        return {
            'role': 'user',
            'content': prompt,
            'complete': True,
        }
    
    def _format_single_target( self, doc: dict ) -> dict:
        return {
            'role': 'assistant',
            'content': [ 'negative', 'positive' ][ doc['label'] ],
            'complete': True,
        }
    
    def format_target_messages( self, doc: dict ) -> list[dict]:
        return [
            self._format_single_target( doc )
        ]
    
    def format_distractor_messages( self, doc: dict ) -> list[dict]:      
        return [
            self._format_single_target( dict( doc, label=i ) )
            for i in [ 0, 1 ]
            if i != doc['label']
        ]
    
    def format_unlabelled_messages( self, doc: dict ) -> list[dict]:
        return [
            self._format_single_target( dict( doc, label=i ) )
            for i in [ 0, 1 ]
        ]


class GlueWNLIInstructDataset( GlueBaseInstructDataset ):
    
    @property
    def task_description( self ) -> str:
        return 'Winograd Schema Challenge. The task is to determine if a second sentence is true given the first.'
    
    @property
    def task_name( self ) -> str:
        return 'wnli'
    
    def format_user_message( self, doc: dict ) -> dict:
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
        
        return {
            'role': 'user',
            'content': prompt,
            'complete': True,
        }
    
    def _format_single_target( self, doc: dict ) -> dict:
        return {
            'role': 'assistant',
            'content': [ 'no', 'yes' ][ doc['label'] ],
            'complete': True,
        }
    
    def format_target_messages( self, doc: dict ) -> list[dict]:
        return [
            self._format_single_target( doc )
        ]
    
    def format_distractor_messages( self, doc: dict ) -> list[dict]:      
        return [
            self._format_single_target( dict( doc, label=i ) )
            for i in [ 0, 1 ]
            if i != doc['label']
        ]
    
    def format_unlabelled_messages( self, doc: dict ) -> list[dict]:
        return [
            self._format_single_target( dict( doc, label=i ) )
            for i in [ 0, 1 ]
        ]

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
    rich.print( GlueWNLIInstructDataset( cache_dir ), end='\n\n' )
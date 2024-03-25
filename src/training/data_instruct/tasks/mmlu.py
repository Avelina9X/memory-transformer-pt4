from collections.abc import Callable, Mapping
from datasets import DatasetDict, Dataset, load_dataset, DownloadConfig
from evaluate import load as load_metric

from ..task_base import BaseChoiceInstructDataset, InstructionDatasetTask, Message, MessageList

class MMLUInstructDataset( BaseChoiceInstructDataset ):
    def __init__( self, cache_dir: str ):
        self.metric = load_metric( 'accuracy', download_config=DownloadConfig( cache_dir=cache_dir ) )
        self._fewshot_cache: dict[str, Dataset] = {}
        super().__init__( cache_dir )

    def download( self, cache_dir: str ) -> DatasetDict:
        dataset = load_dataset( 'cais/mmlu', 'all', cache_dir=cache_dir, revision='7a00892cd331d78a88c8c869d0224a5cdd149848', trust_remote_code=True )
        assert isinstance( dataset, DatasetDict )
        return dataset

    @property
    def task_type( self ) -> InstructionDatasetTask:
        return InstructionDatasetTask.MULTIPLE_CHOICE_CLOSED

    @property
    def task_description( self ) -> str:
        return 'MMLU is a massive multitask test consisting of multiple-choice questions covering 57 task subjects.'

    @property
    def task_name( self ) -> str:
        return 'MMLU'

    @property
    def task_subset( self ) -> str:
        return 'all'


    def get_training_docs( self ) -> Dataset:
        return self.dataset[ 'auxiliary_train' ]

    def get_validation_docs( self ) -> Dataset:
        return self.dataset[ 'validation' ]

    def get_test_docs( self ) -> Dataset:
        return self.dataset[ 'test' ]

    def get_fewshot_docs( self ) -> Dataset:
        return self.dataset[ 'dev' ]


    def format_user_message( self, doc: dict ) -> Message:
        prompt = (
            f"Question: {doc['question']}\n"
            f"\n"
            f"Choices:\n"
            f"A. {doc['choices'][0]}\n"
            f"B. {doc['choices'][1]}\n"
            f"C. {doc['choices'][2]}\n"
            f"D. {doc['choices'][3]}\n"
            f"\n"
            f"Answer:"
        )

        return Message(
            role='user',
            content=prompt,
            complete=True,
        )

    def _format_single_target( self, doc: dict ) -> Message:
        option = [ 'A', 'B', 'C', 'D' ][ doc[ 'answer' ] ]
        choice = doc[ 'choices' ][ doc[ 'answer' ] ]
        prompt = f'{option}. {choice}'
        return Message(
            role='assistant',
            content=prompt,
            complete=True,
        )

    def _get_choices( self, doc: dict ) -> list:
        return [ 0, 1, 2, 3 ]

    def _get_label_key( self ) -> str:
        return 'answer'

    def create_unlabelled_message_target( self, doc: dict ) -> int:
        return doc['answer']

    def create_fewshot_message_list( self, doc: dict ) -> list[MessageList]:
        # Get subject of document
        subject = doc[ 'subject' ]

        # Ensure subject isn't empty
        if subject == '':
            raise ValueError( 'Subject must be defined for fewshot generation.' )

        # If subject isn't in cache, create the cache entry
        if subject not in self._fewshot_cache:
            fewshot_docs = self.get_fewshot_docs().filter( lambda x: x[ 'subject' ] == subject )
            self._fewshot_cache[subject] = fewshot_docs

        # Otherwise retrieve from cache
        else:
            fewshot_docs = self._fewshot_cache[subject]

        # Construct fewshot message list
        return [
            self.create_target_message_list( f_doc )[0] # type: ignore # will always return dicts
            for f_doc in fewshot_docs
        ]

    def compute_metric( self, predictions=None, references=None ) -> dict:
        metric = self.metric.compute( predictions=predictions, references=references )
        assert metric is not None
        return metric

DIRECTORY: Mapping[str, Callable[[str], BaseChoiceInstructDataset]] = {
    'all': MMLUInstructDataset,
}

def main():
    # pylint: disable=W0611
    # pylint: disable=C0415
    import os
    import rich

    from transformers import AutoTokenizer
    from ..formatter import InstructionFormatter

    cache_dir = os.environ[ 'HF_CACHE_DIR' ]

    tokenizer = AutoTokenizer.from_pretrained( 'facebook/opt-125m', cache_dir=cache_dir )
    tokenizer.add_tokens( [ '<|im_start|>', '<|im_end|>' ], special_tokens=True )

    mmlu_ids = MMLUInstructDataset( cache_dir=cache_dir )
    formatter = InstructionFormatter( tokenizer )

    doc_input = mmlu_ids.create_target_message_list( mmlu_ids.get_fewshot_docs()[20] )[0]
    # doc_input = mmlu_ids.create_generation_messages( mmlu_ids.get_fewshot_docs()[20] )
    doc_formatted = formatter.tokenize_chat( doc_input )

    print( doc_formatted[ 'tokens' ] )

    doc_formatted_full = doc_formatted[ 'targets' ]
    doc_formatted_train = [ i for i, m in zip( doc_formatted[ 'targets' ], doc_formatted[ 'train_mask' ] ) if m ]
    doc_formatted_test = [ i for i, m in zip( doc_formatted[ 'targets' ], doc_formatted[ 'test_mask' ] ) if m ]

    print( '-' * 80 )
    print( f'Full doc:\n{tokenizer.decode(doc_formatted_full)}' )
    print( '-' * 80 )
    print( f'Train doc:\n{tokenizer.decode(doc_formatted_train)}' )
    print( '-' * 80 )
    print( f'Test doc:\n{tokenizer.decode(doc_formatted_test)}' )
    print( '-' * 80 )

    rich.print( mmlu_ids )

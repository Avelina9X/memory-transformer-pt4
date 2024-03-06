from datasets import DatasetDict, Dataset, load_dataset

from ..task_base import BaseInstructDataset, InstructionDatasetTask

def _format_single_target( doc: dict ) -> dict:
    prompt = doc['answer'] + '. ' + doc['options'][ ord( doc['answer'] ) - ord( 'A' ) ]
    return {
        'role': 'assistant',
        'content': prompt,
        'complete': True,
    }

class RaceInstructDataset( BaseInstructDataset ):
    def __init__( self, split: str, cache_dir: str ):
        self.split = split
        super().__init__( cache_dir )
    
    def download( self, cache_dir: str ) -> DatasetDict:
        return load_dataset( 'race', self.split, cache_dir=cache_dir ) # type: ignore
    
    @property
    def task_type( self ) -> InstructionDatasetTask:
        return InstructionDatasetTask.MULTIPLE_CHOICE_OPEN
    
    @property
    def task_description( self ) -> str:
        return 'RACE is a large-scale reading comprehension dataset with more than 28,000 passages and nearly 100,000 questions.'
    
    @property
    def group_name( self ) -> str | None:
        return 'race'
    
    @property
    def task_name( self ) -> str:
        return self.split
    
    def get_training_docs( self ) -> Dataset:
        return self.dataset[ 'train' ]
    
    def get_validation_docs( self ) -> Dataset:
        return self.dataset[ 'validation' ]
    
    def get_test_docs( self ) -> Dataset:
        return self.dataset[ 'test' ]
    
    def get_fewshot_docs( self ) -> None:
        return None

    
    def format_user_message( self, doc: dict ) -> dict:
        prompt = (
            f"Background: {doc['article']}\n"
            f"\n"
            f"Question: {doc['question']}\n"
            f"\n"
            f"Choices:\n"
            f"A. {doc['options'][0]}\n"
            f"B. {doc['options'][1]}\n"
            f"C. {doc['options'][2]}\n"
            f"D. {doc['options'][3]}\n"
            f"\n"
            f"Answer:"
        )
        
        return {
            'role': 'user',
            'content': prompt,
            'complete': True,
        }
    
    def format_target_messages( self, doc: dict ) -> list[dict]:
        return [
            _format_single_target( doc )
        ]
    
    def format_distractor_messages( self, doc: dict ) -> list[dict]:      
        return [
            _format_single_target( dict( doc, answer=i ) )
            for i in [ 'A', 'B', 'C', 'D' ]
            if i != doc['answer']
        ]
    
    def format_unlabelled_messages( self, doc: dict ) -> list[dict]:
        return [
            _format_single_target( dict( doc, answer=i ) )
            for i in [ 'A', 'B', 'C', 'D' ]
        ]
    
    def create_unlabelled_message_targets( self, doc: dict ) -> int:
        return ord( doc['answer'] ) - ord( 'A' )
    


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
    
    race_ids = RaceInstructDataset( cache_dir=cache_dir, split='all' )
    formatter = InstructionFormatter( tokenizer )
    
    doc_input = race_ids.create_target_message_list( race_ids.get_training_docs()[20] )[0]
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
    
    rich.print( race_ids )
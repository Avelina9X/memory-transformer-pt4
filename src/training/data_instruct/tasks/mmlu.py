from datasets import DatasetDict, Dataset, load_dataset

from ..task_base import BaseInstructDataset, InstructionDatasetTask

def _format_single_target( doc: dict ) -> dict:
    option = [ 'A', 'B', 'C', 'D' ][ doc[ 'answer' ] ]
    choice = doc[ 'choices' ][ doc[ 'answer' ] ]
    prompt = f'{option}. {choice}'
    return {
        'role': 'assistant',
        'content': prompt,
        'complete': True,
    }

class MMLUInstructDataset( BaseInstructDataset ):
    def download( self, cache_dir: str ) -> DatasetDict:
        return load_dataset( 'cais/mmlu', 'all', cache_dir=cache_dir ) # type: ignore
    
    @property
    def task_type( self ) -> InstructionDatasetTask:
        return InstructionDatasetTask.MULTIPLE_CHOICE_CLOSED
    
    @property
    def task_description( self ) -> str:
        return 'MMLU is a massive multitask test consisting of multiple-choice questions covering 57 task subjects.'
    
    @property
    def group_name( self ) -> str | None:
        return None
    
    @property
    def task_name( self ) -> str:
        return 'MMLU'
    
    
    def get_training_docs( self ) -> Dataset:
        return self.dataset[ 'auxiliary_train' ]
    
    def get_validation_docs( self ) -> Dataset:
        return self.dataset[ 'validation' ]
    
    def get_test_docs( self ) -> Dataset:
        return self.dataset[ 'test' ]
    
    def get_fewshot_docs( self ) -> Dataset:
        return self.dataset[ 'dev' ]
        
    
    def format_user_message( self, doc: dict ) -> dict:
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
            for i in range( 4 )
            if i != doc['answer']
        ]
        
    def format_unlabelled_messages( self, doc: dict ) -> list[dict]:
        return [
            _format_single_target( dict( doc, answer=i ) )
            for i in range( 4 )
        ]
    
    def create_unlabelled_message_targets( self, doc: dict ) -> int:
        return doc['answer']
    


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
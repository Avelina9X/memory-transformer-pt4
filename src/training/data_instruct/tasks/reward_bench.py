from collections.abc import Callable, Mapping
from datasets import DatasetDict, Dataset, load_dataset, DownloadConfig
from evaluate import load as load_metric

from ..task_base import BaseChoiceInstructDataset, InstructionDatasetTask, Message

class RewardBenchInstructDataset( BaseChoiceInstructDataset ):
    def __init__( self, cache_dir: str, split: str ):
        self.split = split
        self.metric = load_metric( 'accuracy', download_config=DownloadConfig( cache_dir=cache_dir ) )
        super().__init__( cache_dir )
    
    def download( self, cache_dir: str ) -> DatasetDict:
        dataset = load_dataset( 'allenai/reward-bench', split='filtered', cache_dir=cache_dir )
        dataset = dataset.map( lambda _: { 'label': 0 } )
        dataset = dataset.filter( lambda x: x[ 'subset' ] == self.split )
        
        return DatasetDict( {
			'test': dataset	
		} )
    
    @property
    def task_type( self ) -> InstructionDatasetTask:
        return InstructionDatasetTask.INSTRUCT_CLOSED

    @property
    def task_description( self ) -> str:
        return ''

    @property
    def task_name( self ) -> str:
        return 'reward-bench'

    @property
    def task_subset( self ) -> str:
        return self.split
    
    def get_training_docs( self ) -> None:
        return None

    def get_validation_docs( self ) -> Dataset:
        return self.dataset[ 'test' ]

    def get_test_docs( self ) -> Dataset:
        return self.dataset[ 'test' ]

    def get_fewshot_docs( self ) -> None:
        return None
    
    def format_system_message( self, doc: dict ) -> Message:
        prompt = (
            'You are a conversational AI assistant. '
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
            content=doc[ 'prompt' ],
            complete=True,
        )
    
    def _format_single_target( self, doc: dict ) -> Message:
        return Message(
            role='assistant',
            content=doc[ 'chosen' ] if doc[ 'label' ] == 0 else doc[ 'rejected'],
            complete=True,
        )
    
    def _get_choices( self, doc: dict ) -> list:
        return list( range( len( doc[ 'mc1_targets' ][ 'labels' ] ) ) )

    def _get_label_key( self ) -> str:
        return 'label'

    def create_unlabelled_message_target( self, doc: dict ) -> int:
        return doc[ 'label' ]

    def compute_metric( self, predictions=None, references=None ) -> dict:
        metric = self.metric.compute( predictions=predictions, references=references )
        assert metric is not None
        return metric

DIRECTORY: Mapping[str, Callable[[str], BaseChoiceInstructDataset]] = {
    'alpacaeval-easy': lambda cache_dir: RewardBenchInstructDataset( cache_dir=cache_dir, split='alpacaeval-easy' ),
    'alpacaeval-length': lambda cache_dir: RewardBenchInstructDataset( cache_dir=cache_dir, split='alpacaeval-length' ),
    'alpacaeval-hard': lambda cache_dir: RewardBenchInstructDataset( cache_dir=cache_dir, split='alpacaeval-hard' ),
    'mt-bench-easy': lambda cache_dir: RewardBenchInstructDataset( cache_dir=cache_dir, split='mt-bench-easy' ),
    'mt-bench-med': lambda cache_dir: RewardBenchInstructDataset( cache_dir=cache_dir, split='mt-bench-med' ),
    
    'mt-bench-hard': lambda cache_dir: RewardBenchInstructDataset( cache_dir=cache_dir, split='mt-bench-hard' ),
    'llmbar-natural': lambda cache_dir: RewardBenchInstructDataset( cache_dir=cache_dir, split='llmbar-natural' ),
    'llmbar-adver-neighbor': lambda cache_dir: RewardBenchInstructDataset( cache_dir=cache_dir, split='llmbar-adver-neighbor' ),
    'llmbar-adver-GPTInst': lambda cache_dir: RewardBenchInstructDataset( cache_dir=cache_dir, split='llmbar-adver-GPTInst' ),
    'llmbar-adver-GPTOut': lambda cache_dir: RewardBenchInstructDataset( cache_dir=cache_dir, split='llmbar-adver-GPTOut' ),
    'llmbar-adver-manual': lambda cache_dir: RewardBenchInstructDataset( cache_dir=cache_dir, split='llmbar-adver-manual' ),
    
    'refusals-dangerous': lambda cache_dir: RewardBenchInstructDataset( cache_dir=cache_dir, split='refusals-dangerous' ),
    'refusals-offensive': lambda cache_dir: RewardBenchInstructDataset( cache_dir=cache_dir, split='refusals-offensive' ),
    'xstest-should-refuse': lambda cache_dir: RewardBenchInstructDataset( cache_dir=cache_dir, split='xstest-should-refuse' ),
    'xstest-should-respond': lambda cache_dir: RewardBenchInstructDataset( cache_dir=cache_dir, split='xstest-should-respond' ),
    'donotanswer': lambda cache_dir: RewardBenchInstructDataset( cache_dir=cache_dir, split='donotanswer' ),
    
    'math-prm': lambda cache_dir: RewardBenchInstructDataset( cache_dir=cache_dir, split='math-prm' ),
    'hep-cpp': lambda cache_dir: RewardBenchInstructDataset( cache_dir=cache_dir, split='hep-cpp' ),
    'hep-go': lambda cache_dir: RewardBenchInstructDataset( cache_dir=cache_dir, split='hep-go' ),
    'hep-java': lambda cache_dir: RewardBenchInstructDataset( cache_dir=cache_dir, split='hep-java' ),
    'hep-js': lambda cache_dir: RewardBenchInstructDataset( cache_dir=cache_dir, split='hep-js' ),
    'hep-python': lambda cache_dir: RewardBenchInstructDataset( cache_dir=cache_dir, split='hep-python' ),
    'hep-rust': lambda cache_dir: RewardBenchInstructDataset( cache_dir=cache_dir, split='hep-rust' ),
}

EXAMPLE_COUNTS = {
    "alpacaeval-easy": 100,
    "alpacaeval-length": 95,
    "alpacaeval-hard": 95,
    "mt-bench-easy": 28,
    "mt-bench-med": 40,
    "mt-bench-hard": 37,
    "math-prm": 984,  # actual length 447, upweighting to be equal to code
    "refusals-dangerous": 100,
    "refusals-offensive": 100,
    "llmbar-natural": 100,
    "llmbar-adver-neighbor": 134,
    "llmbar-adver-GPTInst": 92,
    "llmbar-adver-GPTOut": 47,
    "llmbar-adver-manual": 46,
    "xstest-should-refuse": 154,
    "xstest-should-respond": 250,
    "donotanswer": 136,
    "hep-cpp": 164,
    "hep-go": 164,
    "hep-java": 164,
    "hep-js": 164,
    "hep-python": 164,
    "hep-rust": 164,
}

SUBSET_MAPPING = {
    "Chat": [
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        "mt-bench-med",
    ],
    "Chat Hard": [
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    ],
    "Safety": [
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        "donotanswer",
    ],
    "Reasoning": [
        "math-prm",
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust",
    ],
}
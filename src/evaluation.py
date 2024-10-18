""" Module for benchmarking models. """

from collections.abc import Callable
import math
import os
import typing
import argparse

import torch
import tqdm
from transformers import AutoTokenizer

from model.configuration import LSWTConfig
from model.modeling import LSWTForCausalLM, LSWTForDPH
from training.data_instruct.batcher import ChoiceInstructionBatcher, DPHChoiceInstructionBatcher
from training.data_instruct.task_base import BaseChoiceInstructDataset
from training.data_instruct.formatter import InstructionFormatter
from training.data_instruct.tasks import DIRECTORY_CHOICE

from constants import HF_CACHE_DIR


def evaluate_glue(
    batcher: ChoiceInstructionBatcher | DPHChoiceInstructionBatcher,
    eval_dir: str,
    is_dph: bool,
    max_batch_size: int,
):
    """ Evaluates on the GLUE dataset.

    Honestly this can probably be done a LOT better.
    GLUE is just a little weird since we need to upload to a test server.
    TODO: auto zip the files?

    Args:
        batcher (ChoiceInstructionBatcher | DPHChoiceInstructionBatcher): The batcher to be used for eval.
        eval_dir (str): Directory to save results to (we'll create a subdir for glue)
        is_dph (bool): If we're doing DPH or not. We can figure this out from the batcher class, but lets just be sure.
    """

    # Sanity check
    assert is_dph == isinstance( batcher, DPHChoiceInstructionBatcher )

    # Create subdir paths
    eval_dir_log = os.path.join( eval_dir, 'glue_log' )
    eval_dir_dph = os.path.join( eval_dir, 'glue_dph' ) if is_dph else None

    # Create directories
    os.makedirs( eval_dir_log, mode=0o777, exist_ok=True )
    if eval_dir_dph:
        os.makedirs( eval_dir_dph, mode=0o777, exist_ok=True )

    # Create a map of all tasks
    tasks_map = [
        ( 'AX.tsv', DIRECTORY_CHOICE[ 'glue' ][ 'ax' ]( HF_CACHE_DIR ) ),
        ( 'CoLA.tsv', DIRECTORY_CHOICE[ 'glue' ][ 'cola' ]( HF_CACHE_DIR ) ),
        ( 'MNLI-m.tsv', DIRECTORY_CHOICE[ 'glue' ][ 'mnli_matched' ]( HF_CACHE_DIR ) ),
        ( 'MNLI-mm.tsv', DIRECTORY_CHOICE[ 'glue' ][ 'mnli_mismatched' ]( HF_CACHE_DIR ) ),
        ( 'MRPC.tsv', DIRECTORY_CHOICE[ 'glue' ][ 'mrpc' ]( HF_CACHE_DIR ) ),
        ( 'QNLI.tsv', DIRECTORY_CHOICE[ 'glue' ][ 'qnli' ]( HF_CACHE_DIR ) ),
        ( 'QQP.tsv', DIRECTORY_CHOICE[ 'glue' ][ 'qqp' ]( HF_CACHE_DIR ) ),
        ( 'RTE.tsv', DIRECTORY_CHOICE[ 'glue' ][ 'rte' ]( HF_CACHE_DIR ) ),
        ( 'SST-2.tsv', DIRECTORY_CHOICE[ 'glue' ][ 'sst2' ]( HF_CACHE_DIR ) ),
        ( 'STS-B.tsv', DIRECTORY_CHOICE[ 'glue' ][ 'stsb' ]( HF_CACHE_DIR ) ),
        ( 'WNLI.tsv', DIRECTORY_CHOICE[ 'glue' ][ 'wnli' ]( HF_CACHE_DIR ) ),
    ]

    # Iterate through all tasks
    for out_file, task in tqdm.tqdm( tasks_map, smoothing=0.0 ):
        # Create task paths
        task_dir_log = os.path.join( eval_dir_log, out_file )
        task_dir_dph = os.path.join( eval_dir_dph, out_file ) if eval_dir_dph else None

        # Get the test split (we're evalaute GLUE using the test split ONLY)
        test_ds = task.get_test_docs()
        assert test_ds

        # Get the features to decide if we're a regression or classifcation task
        label_features = test_ds.features[ 'label' ]

        # Create the label2id lambdas TODO: don't do this. there's a better way
        if hasattr( label_features, 'names' ):
            if out_file in [ 'AX.tsv', 'MNLI-m.tsv', 'MNLI-mm.tsv', 'RTE.tsv', 'QNLI.tsv' ]:
                label2id = label_features.names.__getitem__
            else:
                label2id = int
        elif hasattr( label_features, 'dtype' ):
            label2id = float
        else:
            raise ValueError( 'Somehow the features are neither multiple choice nor floating point' )

        # Create empty list for all results
        results_list = []

        # Iterate through dataset and get predictions
        for line in tqdm.tqdm( test_ds, smoothing=0.0 ):
            assert isinstance( line, dict )
            result = batcher.evaluate_document( task, line, False, False )
            result[ 'id' ] = line[ 'idx' ]
            results_list.append( result )

        # If we don't have dph write only to the log dir
        if task_dir_dph is None:
            with open( task_dir_log, 'w', encoding='utf-8' ) as f:
                # Headers
                f.write( 'index\tprediction\n' )

                # Predictions
                for result in results_list:
                    f.write( f'{result["id"]}\t{label2id(result["predictions"].item())}\n' )

        # If we do have dph, write to both dph and log dirs
        else:
            with open( task_dir_log, 'w', encoding='utf-8' ) as f_log, open( task_dir_dph, 'w', encoding='utf-8' ) as f_dph:
                # Headers
                f_log.write( 'index\tprediction\n' )
                f_dph.write( 'index\tprediction\n' )

                # Predictions
                for result in results_list:
                    f_log.write( f'{result["id"]}\t{label2id(result["log_predictions"].item())}\n' )
                    f_dph.write( f'{result["id"]}\t{label2id(result["dph_predictions"].item())}\n' )



def weighted_score( metrics: dict[str, float], weights: dict[str, float] ) -> float:
    """ Computes the weighted average score of all sub tasks.

    Args:
        metrics (dict[str, float]): Metric dict of task:accuracy
        weights (dict[str, float]): DIct of task:weight

    Returns:
        float: Weighted average score
    """
    accumulated_acc = 0.0
    accumulated_weight = 0.0

    for key in metrics:
        accumulated_acc += metrics[key] * weights[key]
        accumulated_weight += weights[key]

    return accumulated_acc / accumulated_weight


def evaluate_generic(
    batcher: ChoiceInstructionBatcher | DPHChoiceInstructionBatcher,
    eval_dir: str,
    is_dph: bool,
    eval_filename: str,
    task_map: list[tuple[str, bool, BaseChoiceInstructDataset, float]],
    max_batch_size: int,
):
    """ Evaluates on an arbitrary weighted task list.

    For unweighted aggregations simply pass 1.0 for all task weights.
    Otherwise pass the sample count for a weighted average.

    Args:
        batcher (ChoiceInstructionBatcher | DPHChoiceInstructionBatcher): The batcher to be used for eval.
        eval_dir (str): Directory to save results to.
        is_dph (bool): If we're doing DPH or not. We can figure this out from the batcher class, but lets just be sure.
        eval_filename (str): File within the eval dir to save files to.
        task_map (list[tuple[str, bool, BaseChoiceInstructDataset, float]]): TODO: better docstring
    """
    # Sanity check
    assert is_dph == isinstance( batcher, DPHChoiceInstructionBatcher )

    # Create path of results files
    results_path = os.path.join( eval_dir, eval_filename )

    # Create metric dicts
    metrics_log: dict[str, float] = {}
    metrics_dph: dict[str, float] | None = {} if is_dph else None
    metrics_weights: dict[str, float] = {}

    # Iterate through tasks
    for task_name, use_test, task, weight in tqdm.tqdm( task_map, smoothing=0.0 ):
        # Get the dataset type according to LM Eval Harness
        test_ds = task.get_test_docs() if use_test else task.get_validation_docs()
        assert test_ds
        
        # Get batch size estimate
        estimate_size = len( task.create_unlabelled_message_list( test_ds[0] ) )
        batch_count = max( 1, math.floor( max_batch_size / estimate_size ) )

        # Compute the results
        results = batcher.evaluate_dataset_batched( task, test_ds, False, False, batch_count )

        # Add the weighting
        metrics_weights[ task_name ] = weight

        # Add log or log and dph results to metric dicts
        if metrics_dph is None:
            metrics_log[ task_name ] = results[ 'accuracy' ] * 100 # type: ignore
        else:
            metrics_log[ task_name ] = results[ 'log' ][ 'accuracy' ] * 100
            metrics_dph[ task_name ] = results[ 'dph' ][ 'accuracy' ] * 100

    # Compute score averages
    score_log = weighted_score( metrics_log, metrics_weights )
    score_dph = weighted_score( metrics_dph, metrics_weights ) if metrics_dph else None

    # Open the file and write to it as a tsv
    with open( results_path, 'w', encoding='utf-8' ) as f:
        # Headers
        task_name_tsv = '\t'.join( metrics_log.keys() )
        f.write( f'type\t{task_name_tsv}\tavg\n' )

        # Log prob metrics
        metrics_log_tsv = '\t'.join( [ str( i ) for i in metrics_log.values() ] )
        f.write( f'log\t{metrics_log_tsv}\t{score_log}\n')

        # Dph metrics (if exist)
        if metrics_dph:
            metrics_dph_tsv = '\t'.join( [ str( i ) for i in metrics_dph.values() ] )
            f.write( f'dph\t{metrics_dph_tsv}\t{score_dph}\n')


def evaluate_gpt4all(
    batcher: ChoiceInstructionBatcher | DPHChoiceInstructionBatcher,
    eval_dir: str,
    is_dph: bool,
    max_batch_size: int,
):
    """ Evaluates on the GPT4All tasks.

    Average score is unweighted macro average of accuracies.
    Following LM Evaluation Harness only `OpenBookQA` and `ARC` use the test splits,
    while all other tasks use the validation splits.

    Args:
        batcher (ChoiceInstructionBatcher | DPHChoiceInstructionBatcher): The batcher to be used for eval.
        eval_dir (str): Directory to save results to (we'll create a tsv file in here).
        is_dph (bool): If we're doing DPH or not. We can figure this out from the batcher class, but lets just be sure.
    """

    # Create a map of all tasts
    task_map: list[tuple[str, bool, BaseChoiceInstructDataset, float]] = [
        ( 'HellaSwag', False, DIRECTORY_CHOICE[ 'hellaswag' ][ 'no_choice' ]( HF_CACHE_DIR ), 1.0 ),
        ( 'Obqa', True, DIRECTORY_CHOICE[ 'obqa' ][ 'main' ]( HF_CACHE_DIR ), 1.0 ),
        ( 'WinoGrande', False, DIRECTORY_CHOICE[ 'winogrande' ][ 'no_choice' ]( HF_CACHE_DIR ), 1.0 ),
        ( 'ARC_c', True, DIRECTORY_CHOICE[ 'arc' ][ 'challenge' ]( HF_CACHE_DIR ), 1.0 ),
        ( 'ARC_e', True, DIRECTORY_CHOICE[ 'arc' ][ 'easy' ]( HF_CACHE_DIR ), 1.0 ),
        ( 'boolq', False, DIRECTORY_CHOICE[ 'super_glue' ][ 'boolq' ]( HF_CACHE_DIR ), 1.0 ),
        ( 'piqa', False, DIRECTORY_CHOICE[ 'piqa' ][ 'no_choice' ]( HF_CACHE_DIR ), 1.0 ),
    ]

    evaluate_generic( batcher, eval_dir, is_dph, 'gpt4all.tsv', task_map, max_batch_size )

def evaluate_race(
    batcher: ChoiceInstructionBatcher | DPHChoiceInstructionBatcher,
    eval_dir: str,
    is_dph: bool,
    max_batch_size: int,
):
    """ Evaluates RACE middle and high on the test splits.

    Computes the average score as the weighted average of both subsets (m=1436, h=3498)

    Args:
        batcher (ChoiceInstructionBatcher | DPHChoiceInstructionBatcher): The batcher to be used for eval.
        eval_dir (str): Directory to save results to (we'll create a tsv file in here)
        is_dph (bool): If we're doing DPH or not. We can figure this out from the batcher class, but lets just be sure.
    """

    # Create a map of all tasts
    task_map: list[tuple[str, bool, BaseChoiceInstructDataset, float]] = [
        ( 'RACE-m', True, DIRECTORY_CHOICE[ 'race' ][ 'middle' ]( HF_CACHE_DIR ), 1436 ),
        ( 'RACE-h', True, DIRECTORY_CHOICE[ 'race' ][ 'high' ]( HF_CACHE_DIR ), 3498 ),
    ]

    evaluate_generic( batcher, eval_dir, is_dph, 'race.tsv', task_map, max_batch_size )

def evaluate_reward_bench(
    batcher: ChoiceInstructionBatcher | DPHChoiceInstructionBatcher,
    eval_dir: str,
    is_dph: bool,
    max_batch_size: int,
):
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
    
    eval_dir = os.path.join( eval_dir, 'reward_bench' )
    os.makedirs( eval_dir, mode=0o777, exist_ok=True )
    
    # Create a map of all tasts
    task_map_chat: list[tuple[str, bool, BaseChoiceInstructDataset, float]] = [
        ( name, True, DIRECTORY_CHOICE[ 'reward_bench' ][ name ]( HF_CACHE_DIR ), EXAMPLE_COUNTS[ name ] )
        for name in SUBSET_MAPPING[ 'Chat' ]
    ]
    
    task_map_chat_hard: list[tuple[str, bool, BaseChoiceInstructDataset, float]] = [
        ( name, True, DIRECTORY_CHOICE[ 'reward_bench' ][ name ]( HF_CACHE_DIR ), EXAMPLE_COUNTS[ name ] )
        for name in SUBSET_MAPPING[ 'Chat Hard' ]
    ]
    
    task_map_safety: list[tuple[str, bool, BaseChoiceInstructDataset, float]] = [
        ( name, True, DIRECTORY_CHOICE[ 'reward_bench' ][ name ]( HF_CACHE_DIR ), EXAMPLE_COUNTS[ name ] )
        for name in SUBSET_MAPPING[ 'Safety' ]
    ]
    
    task_map_reasoning: list[tuple[str, bool, BaseChoiceInstructDataset, float]] = [
        ( name, True, DIRECTORY_CHOICE[ 'reward_bench' ][ name ]( HF_CACHE_DIR ), EXAMPLE_COUNTS[ name ] )
        for name in SUBSET_MAPPING[ 'Reasoning' ]
    ]

    evaluate_generic( batcher, eval_dir, is_dph, 'chat.tsv', task_map_chat, max_batch_size )
    evaluate_generic( batcher, eval_dir, is_dph, 'chat-hard.tsv', task_map_chat_hard, max_batch_size )
    evaluate_generic( batcher, eval_dir, is_dph, 'safety.tsv', task_map_safety, max_batch_size )
    evaluate_generic( batcher, eval_dir, is_dph, 'reasoning.tsv', task_map_reasoning, max_batch_size )

BENCHMARK_MAP: dict[str, Callable[[ChoiceInstructionBatcher | DPHChoiceInstructionBatcher, str, bool, int], None]] = {
    'glue': evaluate_glue,
    'gpt4all': evaluate_gpt4all,
    'race': evaluate_race,
    'reward_bench': evaluate_reward_bench,
}

def evaluate( model_dir: str, benchmark: str, batch_size: int ):
    """ Evaluates a model using a specific benchmark

    Args:
        model_dir (str): directory containing model weights, config and tokenizer
        benchmark (str): string identifier for benchmark
    """

    # Create directory inside model_dir with evaluation results
    eval_dir = os.path.join( model_dir, 'benchmarks' )
    os.makedirs( eval_dir, mode=0o777, exist_ok=True )

    # Grab model config
    model_config = typing.cast( LSWTConfig, LSWTConfig.from_pretrained( model_dir, torch_dtype=None ) )
    architecture = model_config.architectures[0]

    # Parse model and batcher classes
    match architecture:
        case 'LSWTForCausalLM':
            model_cls = LSWTForCausalLM
            is_dph = False
        case 'LSWTForDPH':
            model_cls = LSWTForDPH
            is_dph = True
        case _:
            raise ValueError( f'Model has an invalid architecture: {architecture}' )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained( model_dir, use_fast=True )

    # Build model
    model = model_cls.from_pretrained( model_dir, torch_dtype=torch.float16, device_map='cuda' )
    assert isinstance( model, model_cls )

    # Create formatter and batcher
    formatter = InstructionFormatter( tokenizer )
    if is_dph:
        assert model_config.reward_heads
        batcher = DPHChoiceInstructionBatcher( model, formatter, model_config.reward_heads[0], 'mean' )
    else:
        batcher = ChoiceInstructionBatcher( model, formatter, 'mean' )

    match benchmark:
        case 'all':
            for b in BENCHMARK_MAP.values():
                b( batcher, eval_dir, is_dph, batch_size )
        case _:
            BENCHMARK_MAP[benchmark]( batcher, eval_dir, is_dph, batch_size )


def run():
    """ Runs the Evaluation benchmark on a given model and benchmark.
    """

    argparser = argparse.ArgumentParser()

    # Directory of the desired model
    argparser.add_argument(
        '--dir',
        type=str,
        required=True, # TODO: enable parsing from URL in the future,
        help='Path to the model directory.'
    )

    # Benchmark
    argparser.add_argument(
        '--benchmark',
        type=str.lower,
        required=True,
        choices=[ 'all', *BENCHMARK_MAP.keys() ],
        help='Name of benchmark to perform',
    )
    
    # Benchmark
    argparser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Approximate maximum batch size to use during evaluation.',
    )

    # Parse the command line args
    arguments = argparser.parse_args()

    # Run evaluation
    evaluate( arguments.dir, arguments.benchmark, arguments.batch_size )

if __name__ == '__main__':
    run()

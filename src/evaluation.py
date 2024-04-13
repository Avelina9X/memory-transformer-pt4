

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
    is_dph: bool
):
    """ Evaluates on the GLUE dataset. Honestly this can probably be done a LOT better.
    GLUE is just a little weird since we need to upload to a test server.

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
    for out_file, task in tqdm.tqdm( tasks_map ):
        # Create task paths
        task_dir_log = os.path.join( eval_dir_log, out_file )
        task_dir_dph = os.path.join( eval_dir_dph, out_file ) if eval_dir_dph else None

        # Get the test split (we're evalaute GLUE using the test split ONLY)
        test_ds = task.get_test_docs()
        assert test_ds

        # Get the features to decide if we're a regression or classifcation task
        label_features = test_ds.features[ 'label' ]

        # Create the label2id lambda TODO: don't do this. there's a better way
        if hasattr( label_features, 'names' ):
            label2id = label_features.names.__getitem__
        elif hasattr( label_features, 'dtype' ):
            label2id = float
        else:
            raise ValueError( 'Somehow the features are neither multiple choice nor floating point' )

        results_list = []

        for line in test_ds:
            result = batcher.evaluate_document( task, line, False, False )
            result[ 'id' ] = line[ 'idx' ]
            results_list.append( result )

        if task_dir_dph is None:
            with open( task_dir_log, 'w', encoding='utf-8' ) as f:
                f.write( 'index\tprediction\n' )

                for result in results_list:
                    f.write( f'{result["id"]}\t{label2id(result["predictions"].item())}\n' )
        else:
            with open( task_dir_log, 'w', encoding='utf-8' ) as f_log, open( task_dir_dph, 'w', encoding='utf-8' ) as f_dph:
                f_log.write( 'index\tprediction\n' )
                f_dph.write( 'index\tprediction\n' )

                for result in results_list:
                    f_log.write( f'{result["id"]}\t{label2id(result["log_predictions"].item())}\n' )
                    f_dph.write( f'{result["id"]}\t{label2id(result["dph_predictions"].item())}\n' )

def weighted_score( metrics: dict[str, float], weights: dict[str, float] ):
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
):
    """ Evaluates on an arbitrary weighted task list.
    For unweighted aggregations simply pass 1.0 for all task weights.

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
    for task_name, use_test, task, weight in tqdm.tqdm( task_map ):
        # Get the dataset type according to LM Eval Harness
        test_ds = task.get_test_docs() if use_test else task.get_validation_docs()
        assert test_ds

        # Compute the results
        results = batcher.evaluate_dataset( task, test_ds, False, False )

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
    is_dph: bool
):
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

    evaluate_generic( batcher, eval_dir, is_dph, 'gpt4all.tsv', task_map )

def evaluate_race(
    batcher: ChoiceInstructionBatcher | DPHChoiceInstructionBatcher,
    eval_dir: str,
    is_dph: bool
):
    # Create a map of all tasts
    task_map: list[tuple[str, bool, BaseChoiceInstructDataset, float]] = [
        ( 'RACE-m', True, DIRECTORY_CHOICE[ 'race' ][ 'middle' ]( HF_CACHE_DIR ), 1436 ),
        ( 'RACE-h', True, DIRECTORY_CHOICE[ 'race' ][ 'high' ]( HF_CACHE_DIR ), 3498 ),
    ]

    evaluate_generic( batcher, eval_dir, is_dph, 'race.tsv', task_map )

def evaluate( model_dir: str, benchmark: str ):

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
            batch_cls = ChoiceInstructionBatcher
            is_dph = False
        case 'LSWTForDPH':
            model_cls = LSWTForDPH
            batch_cls = DPHChoiceInstructionBatcher
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
    batcher = batch_cls( model, formatter, 'mean' )

    match benchmark:
        case 'glue':
            evaluate_glue( batcher, eval_dir, is_dph )
        case 'gpt4all':
            evaluate_gpt4all( batcher, eval_dir, is_dph )
        case 'race':
            evaluate_race( batcher, eval_dir, is_dph )


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
        choices=[ 'glue', 'gpt4all', 'race' ],
        help='Name of benchmark to perform',
    )

    # Parse the command line args
    arguments = argparser.parse_args()

    # Run evaluation
    evaluate( arguments.dir, arguments.benchmark )

if __name__ == '__main__':
    run()

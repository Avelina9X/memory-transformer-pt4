import math
import itertools
from dataclasses import dataclass
from typing import cast
from collections.abc import Sequence, Iterable
from scipy import stats
import numpy as np

import torch
from transformers import PreTrainedModel

from .formatter import InstructionFormatter, SteerInstructionFormatter
from .task_base import BaseChoiceInstructDataset, BaseSteerInstructDataset


def iter_n( iterable: Iterable, n: int ):
    sentinel = object()
    for chunk in itertools.zip_longest( *[ iter( iterable ) ] * n, fillvalue=sentinel ):
        yield [ elem for elem in chunk if elem is not sentinel ]


class BaseInstructionBatcher:
    def __init__(
        self,
        model: PreTrainedModel,
        formatter: InstructionFormatter,
        aggregation: str = 'mean',
        pad_rounding: int = 16,
    ):
        """ Creates an Instruction Batcher for evaluating instruction datasets.

        Args:
            model (PreTrainedModel): Causal Model to evaluate
            formatter (InstructionFormatter): formatter object
            aggregation (str): Desired type of logprob aggregation, 'mean' or 'sum'. Defaults to 'mean'.
            pad_rounding (int): The multiple for which batches will be padded to. Defaults to 16.
        """
        self.formatter = formatter
        self.model = model
        self.aggregation = aggregation
        self.pad_rounding = pad_rounding

        if aggregation not in [ 'mean', 'sum' ]:
            raise ValueError( f'Aggregation must be `mean` or `sum` but got `{aggregation}`' )

    def aggregate( self, logprobs: torch.Tensor, mask: torch.Tensor ) -> torch.Tensor:
        """ Aggregates log probabilities based on a mask and reduction type.

        Args:
            logprobs (torch.Tensor): Logprob tensor of shape `[ Batch x SeqLen ]`
            mask (torch.Tensor): Mask tensor of shape `[ Batch x SeqLen ]`

        Returns:
            torch.Tensor: The aggergated log probability of shape `[ Batch ]`
        """
        match self.aggregation:
            case 'mean':
                return logprobs.sum( -1 ) / mask.sum( -1 )
            case 'sum':
                return logprobs.sum( -1 )
            case _:
                assert False, "It shouldn't be possible to get here!"

@dataclass
class PreparedChoiceBatch:
    tokens: torch.LongTensor
    targets: torch.LongTensor
    train_mask: torch.BoolTensor
    test_mask: torch.BoolTensor
    correct_index: int | float

class ChoiceInstructionBatcher( BaseInstructionBatcher ):
    def prepare_batch(
        self,
        task: BaseChoiceInstructDataset,
        target_doc: dict,
        device: str | torch.device,
        fewshot: bool,
        fewshot_allsys: bool,
    ) -> PreparedChoiceBatch:
        """ Creates a prepared choice batch from a task and document

        Args:
            task (BaseChoiceInstructDataset): The task used to parse documents
            target_doc (dict): The target document to parse
            device (str | torch.device): Where to place token and mask tensors
            fewshot (bool): If fewshot testing should be enabled.
            fewshot_allsys (bool): If all message groups should contain a system message
        Returns:
            PreparedChoiceBatch: Batch of tokens, targets and masks, all moved to device.
        """
        completions = task.create_unlabelled_message_list( target_doc )
        correct = task.create_unlabelled_message_target( target_doc )

        if correct is None:
            raise ValueError( 'Message targets must be defined!' )

        if fewshot is False:
            tokenized_completions = [ self.formatter.tokenize_chat( msgs ) for msgs in completions ]
        else:
            fewshot_list = task.create_fewshot_message_list( target_doc )
            tokenized_completions = [ self.formatter.tokenize_chat_fewshot( msgs, fewshot_list, fewshot_allsys ) for msgs in completions ]

        max_length = max( len( line[ 'tokens' ] ) for line in tokenized_completions )
        max_length = math.ceil( max_length / self.pad_rounding ) * self.pad_rounding

        pad_token_id = self.formatter.tokenizer.pad_token_id

        tokens_list = []
        targets_list = []
        train_mask_list = []
        test_mask_list = []

        for line in tokenized_completions:
            curr_len = len( line[ 'tokens' ] )
            pad_len = max_length - curr_len

            tokens = line[ 'tokens' ] + [ pad_token_id ] * pad_len
            targets = line[ 'targets' ] + [ pad_token_id ] * pad_len
            train_mask = line[ 'train_mask' ] + [ False ] * pad_len
            test_mask = line[ 'test_mask' ] + [ False ] * pad_len

            tokens_list.append( torch.LongTensor( tokens ).to( device=device, non_blocking=True ) )
            targets_list.append( torch.LongTensor( targets ).to( device=device, non_blocking=True ) )
            train_mask_list.append( torch.BoolTensor( train_mask ).to( device=device, non_blocking=True ) )
            test_mask_list.append( torch.BoolTensor( test_mask ).to( device=device, non_blocking=True ) )

        return PreparedChoiceBatch(
            tokens=cast( torch.LongTensor, torch.stack( tokens_list ) ),
            targets=cast( torch.LongTensor, torch.stack( targets_list ) ),
            train_mask=cast( torch.BoolTensor, torch.stack( train_mask_list ) ),
            test_mask=cast( torch.BoolTensor, torch.stack( test_mask_list ) ),
            correct_index=correct
        )

    def prepare_batches(
        self,
        task: BaseChoiceInstructDataset,
        target_docs: list[dict],
        device: str | torch.device,
        fewshot: bool,
        fewshot_allsys: bool,
    ) -> list[PreparedChoiceBatch]:
        """ Creates a list of prepared choice batches from a task and list of documents

        Args:
            task (BaseChoiceInstructDataset): The task used to parse documents
            target_docs (list[dict]): List of target documents to batch together.
            device (str | torch.device): Where to place token and mask tensors
            fewshot (bool): If fewshot testing should be enabled.
            fewshot_allsys (bool): If all message groups should contain a system message
        Returns:
            list[PreparedChoiceBatch]: List of batches of tokens, targets and masks, all moved to device.
        """
        tokenized_completions_list = []
        correct_list = []
        batch_list = []

        for target_doc in target_docs:
            completions = task.create_unlabelled_message_list( target_doc )
            correct = task.create_unlabelled_message_target( target_doc )

            if correct is None:
                raise ValueError( 'Message targets must be defined!' )

            if fewshot is False:
                tokenized_completions = [ self.formatter.tokenize_chat( msgs ) for msgs in completions ]
            else:
                fewshot_list = task.create_fewshot_message_list( target_doc )
                tokenized_completions = [ self.formatter.tokenize_chat_fewshot( msgs, fewshot_list, fewshot_allsys ) for msgs in completions ]

            tokenized_completions_list.append( tokenized_completions )
            correct_list.append( correct )

        max_length = max( len( line[ 'tokens' ] ) for tokenized_completions in tokenized_completions_list for line in tokenized_completions )
        max_length = math.ceil( max_length / self.pad_rounding ) * self.pad_rounding

        pad_token_id = self.formatter.tokenizer.pad_token_id

        for tokenized_completions, correct in zip( tokenized_completions_list, correct_list ):
            tokens_list = []
            targets_list = []
            train_mask_list = []
            test_mask_list = []

            for line in tokenized_completions:
                curr_len = len( line[ 'tokens' ] )
                pad_len = max_length - curr_len

                tokens = line[ 'tokens' ] + [ pad_token_id ] * pad_len
                targets = line[ 'targets' ] + [ pad_token_id ] * pad_len
                train_mask = line[ 'train_mask' ] + [ False ] * pad_len
                test_mask = line[ 'test_mask' ] + [ False ] * pad_len

                tokens_list.append( torch.LongTensor( tokens ).to( device=device, non_blocking=True ) )
                targets_list.append( torch.LongTensor( targets ).to( device=device, non_blocking=True ) )
                train_mask_list.append( torch.BoolTensor( train_mask ).to( device=device, non_blocking=True ) )
                test_mask_list.append( torch.BoolTensor( test_mask ).to( device=device, non_blocking=True ) )

            batch_list.append( PreparedChoiceBatch(
                tokens=cast( torch.LongTensor, torch.stack( tokens_list ) ),
                targets=cast( torch.LongTensor, torch.stack( targets_list ) ),
                train_mask=cast( torch.BoolTensor, torch.stack( train_mask_list ) ),
                test_mask=cast( torch.BoolTensor, torch.stack( test_mask_list ) ),
                correct_index=correct
            ) )

        return batch_list

    def compute_batch(
        self,
        prepared_batch: PreparedChoiceBatch,
        logits: torch.Tensor
    ) -> dict:

        # Grab easy references to elements of the batch
        targets = prepared_batch.targets.unsqueeze( -1 )
        test_mask = prepared_batch.test_mask
        correct_index = prepared_batch.correct_index

        # Compute per token log probabilites and gather by target token
        logprobs = logits.log_softmax( -1, torch.float32 ).gather( -1, targets ).squeeze( -1 )

        # Mask out all other tokens
        masked_logprobs = logprobs * test_mask

        # Aggregate log probs for each candidate in batch
        agg_logprobs = self.aggregate( masked_logprobs, test_mask )

        # Get the top index
        predicted_index = agg_logprobs.argmax().detach()

        return {
            'references': correct_index,
            'predictions': predicted_index,
        }

    def evaluate_document(
        self,
        task: BaseChoiceInstructDataset,
        doc: dict,
        fewshot: bool = False,
        fewshot_allsys: bool = True,
    ) -> dict:
        with torch.inference_mode():
            self.model.eval()
            device = self.model.get_input_embeddings().weight.device
            prepared_batch = self.prepare_batch( task, doc, device, fewshot, fewshot_allsys )

            with torch.autocast( device_type='cuda', dtype=torch.bfloat16 if self.model.config.use_bfloat16 else torch.float16 ):
                logits = self.model( prepared_batch.tokens ).logits

            results = self.compute_batch( prepared_batch, logits )
        return results
    
    def evaluate_document_batched(
        self,
        task: BaseChoiceInstructDataset,
        docs: list[dict],
        fewshot: bool = False,
        fewshot_allsys: bool = True,
    ) -> list[dict]:
        with torch.inference_mode():
            self.model.eval()
            device = self.model.get_input_embeddings().weight.device
            prepared_batches = self.prepare_batches( task, docs, device, fewshot, fewshot_allsys )
            
            tokens = torch.cat( [ i.tokens for i in prepared_batches ], dim=0 )
            tokens_shapes = ( [ i.tokens.shape[0] for i in prepared_batches ] )
            out_dicts = []
            
            with torch.autocast( device_type='cuda', dtype=torch.bfloat16 if self.model.config.use_bfloat16 else torch.float16 ):
                outputs = self.model( tokens )

                logits: torch.Tensor = outputs.logits

                logits_arr = logits.split( tokens_shapes, dim=0 )

            for idx in range( len( docs ) ):
                log_results = self.compute_batch( prepared_batches[idx], logits_arr[idx] )

                out_dicts.append( {
                    'references': log_results[ 'references' ],
                    'predictions': log_results[ 'predictions' ],
                } )
        
        return out_dicts

    def evaluate_dataset(
        self,
        task: BaseChoiceInstructDataset,
        dataset: Iterable,
        fewshot: bool = False,
        fewshot_allsys: bool = True,
    ):
        correct_list = []
        answer_list = []

        for line in dataset:
            results = self.evaluate_document( task, line, fewshot, fewshot_allsys )
            correct_list.append( results[ 'references' ] )
            answer_list.append( results[ 'predictions' ] )

        answer_list = [ i.item() for i in answer_list ]

        return task.compute_metric( references=correct_list, predictions=answer_list )

    def evaluate_dataset_batched(
        self,
        task: BaseChoiceInstructDataset,
        dataset: Iterable,
        fewshot: bool = False,
        fewshot_allsys: bool = True,
        batch_size: int | None = None,
    ):
        if batch_size is None:
            return self.evaluate_dataset( task, dataset, fewshot, fewshot_allsys )

        correct_list = []
        answer_list = []

        for line in iter_n( dataset, batch_size ):
            results = self.evaluate_document_batched( task, line, fewshot, fewshot_allsys )
            for result in results:
                correct_list.append( result[ 'references' ] )
                answer_list.append( result[ 'predictions' ] )

        answer_list = [ i.item() for i in answer_list ]

        return task.compute_metric( references=correct_list, predictions=answer_list )



class DPHChoiceInstructionBatcher( ChoiceInstructionBatcher ):
    def __init__(
        self,
        model: PreTrainedModel,
        formatter: InstructionFormatter,
        reward_head_key: str,
        aggregation: str = 'mean',
        pad_rounding: int = 16
    ):
        super().__init__( model, formatter, aggregation, pad_rounding )
        self.reward_head_key = reward_head_key

    def compute_batch_dph(
        self,
        rewards: torch.Tensor
    ) -> dict:
        predicted_index = rewards.squeeze( -1 ).argmax().detach()
        return {
            'predictions': predicted_index,
        }

    def evaluate_document(
        self,
        task: BaseChoiceInstructDataset,
        doc: dict,
        fewshot: bool = False,
        fewshot_allsys: bool = True,
    ) -> dict:
        with torch.inference_mode():
            self.model.eval()
            device = self.model.get_input_embeddings().weight.device
            prepared_batch = self.prepare_batch( task, doc, device, fewshot, fewshot_allsys )

            with torch.autocast( device_type='cuda', dtype=torch.bfloat16 if self.model.config.use_bfloat16 else torch.float16 ):
                outputs = self.model( prepared_batch.tokens )

                logits = outputs.logits
                states = outputs.hidden_states
                rewards = self.model.compute_final_rewards( states, prepared_batch.tokens )[ self.reward_head_key ]

            log_results = self.compute_batch( prepared_batch, logits )
            dph_results = self.compute_batch_dph( rewards )

        return {
            'references': log_results[ 'references' ],
            'log_predictions': log_results[ 'predictions' ],
            'dph_predictions': dph_results[ 'predictions' ],
        }

    def evaluate_document_batched(
        self,
        task: BaseChoiceInstructDataset,
        docs: list[dict],
        fewshot: bool = False,
        fewshot_allsys: bool = True,
    ) -> list[dict]:
        with torch.inference_mode():
            self.model.eval()
            device = self.model.get_input_embeddings().weight.device
            prepared_batches = self.prepare_batches( task, docs, device, fewshot, fewshot_allsys )

            tokens = torch.cat( [ i.tokens for i in prepared_batches ], dim=0 )
            tokens_shapes = ( [ i.tokens.shape[0] for i in prepared_batches ] )
            out_dicts = []

            with torch.autocast( device_type='cuda', dtype=torch.bfloat16 if self.model.config.use_bfloat16 else torch.float16 ):
                outputs = self.model( tokens )

                logits: torch.Tensor = outputs.logits
                states = outputs.hidden_states
                rewards: torch.Tensor = self.model.compute_final_rewards( states, tokens )[ self.reward_head_key ]

                logits_arr = logits.split( tokens_shapes, dim=0 )
                rewards_arr = rewards.split( tokens_shapes, dim=0 )

            for idx in range( len( docs ) ):
                log_results = self.compute_batch( prepared_batches[idx], logits_arr[idx] )
                dph_results = self.compute_batch_dph( rewards_arr[idx] )

                out_dicts.append( {
                    'references': log_results[ 'references' ],
                    'log_predictions': log_results[ 'predictions' ],
                    'dph_predictions': dph_results[ 'predictions' ],
                } )

        return out_dicts

    def evaluate_dataset(
        self,
        task: BaseChoiceInstructDataset,
        dataset: Iterable,
        fewshot: bool = False,
        fewshot_allsys: bool = True,
    ):
        correct_list = []
        log_answer_list = []
        dph_answer_list = []

        for line in dataset:
            results = self.evaluate_document( task, line, fewshot, fewshot_allsys )
            correct_list.append( results[ 'references' ] )
            log_answer_list.append( results[ 'log_predictions' ] )
            dph_answer_list.append( results[ 'dph_predictions' ] )

        log_answer_list = [ i.item() for i in log_answer_list ]
        dph_answer_list = [ i.item() for i in dph_answer_list ]

        return {
            'log': task.compute_metric( references=correct_list, predictions=log_answer_list ),
            'dph': task.compute_metric( references=correct_list, predictions=dph_answer_list ),
        }

    def evaluate_dataset_batched(
        self,
        task: BaseChoiceInstructDataset,
        dataset: Iterable,
        fewshot: bool = False,
        fewshot_allsys: bool = True,
        batch_size: int | None = None,
    ):
        if batch_size is None:
            return self.evaluate_dataset( task, dataset, fewshot, fewshot_allsys )

        correct_list = []
        log_answer_list = []
        dph_answer_list = []

        for line in iter_n( dataset, batch_size ):
            results = self.evaluate_document_batched( task, line, fewshot, fewshot_allsys )
            for result in results:
                correct_list.append( result[ 'references' ] )
                log_answer_list.append( result[ 'log_predictions' ] )
                dph_answer_list.append( result[ 'dph_predictions' ] )

        log_answer_list = [ i.item() for i in log_answer_list ]
        dph_answer_list = [ i.item() for i in dph_answer_list ]

        return {
            'log': task.compute_metric( references=correct_list, predictions=log_answer_list ),
            'dph': task.compute_metric( references=correct_list, predictions=dph_answer_list ),
        }

class SteerInstructionBatcher( BaseInstructionBatcher ):
    def __init__(
        self,
        model: PreTrainedModel,
        formatter: SteerInstructionFormatter,
        aggregation: str = 'mean',
        pad_rounding: int = 16,
        label_keys: Sequence[str] = (),
    ):
        super().__init__( model, formatter, aggregation, pad_rounding )

        self.label_keys = list( label_keys )

    def prepare_batch(
        self,
        task: BaseSteerInstructDataset,
        target_doc: dict,
        device: str | torch.device,
    ):
        messages_list = task.create_target_message_list( target_doc )
        assert len( messages_list ) == 1
        messages = messages_list[0]

        correct = task.get_labels( target_doc, self.label_keys )

        tokenized_messages = self.formatter.tokenize_chat( messages )

        tokens = tokenized_messages[ 'tokens' ]

        pad_token_id = self.formatter.tokenizer.pad_token_id

        pad_len = math.ceil( len( tokens ) / self.pad_rounding ) * self.pad_rounding - len( tokens )

        tokens = tokens + [ pad_token_id ] * pad_len

        return torch.LongTensor( [ tokens ] ).to( device=device, non_blocking=True ), correct

    def compute_batch( self, rewards: dict[str, torch.Tensor] ):
        return [
            rewards[key].squeeze().detach() for key in self.label_keys
        ]

    def evaluate_document(
        self,
        task: BaseSteerInstructDataset,
        doc: dict,
    ) -> dict:
        with torch.inference_mode():
            self.model.eval()
            device = self.model.get_input_embeddings().weight.device
            tokens, labels = self.prepare_batch( task, doc, device )

            with torch.autocast( device_type='cuda', dtype=torch.bfloat16 if self.model.config.use_bfloat16 else torch.float16 ):
                outputs = self.model( tokens )

                states = outputs.hidden_states
                rewards = self.model.compute_final_rewards( states, tokens )
            reward_list = self.compute_batch( rewards )

        return {
            'y_true': labels,
            'y_pred': reward_list
        }

    def evaluate_dataset(
        self,
        task: BaseSteerInstructDataset,
        dataset: Iterable,
    ):
        true_list = []
        pred_list = []

        for line in dataset:
            results = self.evaluate_document( task, line )
            true_list.append( results[ 'y_true' ] )
            pred_list.append( results[ 'y_pred' ] )

        pred_list = [ [ i.item() for i in preds ] for preds in pred_list ]

        true_array = np.array( true_list )
        pred_array = np.array( pred_list )

        metrics = {}

        for i, key in enumerate( self.label_keys ):
            curr_true = true_array[ :, i ]
            curr_pred = pred_array[ :, i ]

            pearsonr = stats.pearsonr( curr_true, curr_pred )
            spearmanr = stats.spearmanr( curr_true, curr_pred )

            metrics[key] = {
                'pearsonr': pearsonr.statistic, # type: ignore
                'spearmanr': spearmanr.statistic, # type: ignore
                # 'pearsonr_pvalue': pearsonr.pvalue, # type: ignore
                # 'spearmanr_pvalue': spearmanr.pvalue, # type: ignore
            }

        return metrics

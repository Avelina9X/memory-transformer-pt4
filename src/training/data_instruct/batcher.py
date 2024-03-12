import math
from dataclasses import dataclass
from typing import cast
from collections.abc import Iterable

import torch
from transformers import PreTrainedModel

from .formatter import InstructionFormatter
from .task_base import BaseChoiceInstructDataset

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
    correct_index: int

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

        if not isinstance( correct, int ):
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
        predicted_index = agg_logprobs.argmax().item()

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

            with torch.autocast( device_type='cuda', dtype=torch.float16 ):
                logits = self.model( prepared_batch.tokens ).logits

            results = self.compute_batch( prepared_batch, logits )
        return results

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

        return task.compute_metric( references=correct_list, predictions=answer_list )

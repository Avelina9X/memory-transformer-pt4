from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import PreTrainedModel
from datasets import Dataset

from .formatter import InstructionFormatter
from .task_base import BaseChoiceInstructDataset

class BaseInstructionBatcher:
    def __init__(
        self,
        model: PreTrainedModel,
        formatter: InstructionFormatter,
        aggregation: str = 'mean'
    ):
        """ Creates an Instruction Batcher for evaluating instruction datasets.

        Args:
            model (PreTrainedModel): Causal Model to evaluate
            formatter (InstructionFormatter): formatter object
            aggregation (str, optional): Desired type of logprob aggregation, 'mean' or 'sum'. Defaults to 'mean'.
        """
        self.formatter = formatter
        self.model = model
        self.aggregation = aggregation

        assert self.aggregation in [ 'mean', 'sum' ], "Aggregation must be one of 'mean' or 'sum'"
    
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
    def prepare_batch( self, task: BaseChoiceInstructDataset, doc: dict, device: str | torch.device ) -> PreparedChoiceBatch:
        completions = task.create_unlabelled_message_list( doc )
        correct = task.create_unlabelled_message_target( doc )

        tokenized_completions = [ self.formatter.tokenize_chat( msgs ) for msgs in completions ]
        max_length = max( len( line[ 'tokens' ] ) for line in tokenized_completions )
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

            tokens_list.append( torch.LongTensor( tokens ).to( device=device ) )
            targets_list.append( torch.LongTensor( targets ).to( device=device ) )
            train_mask_list.append( torch.BoolTensor( train_mask ).to( device=device ) )
            test_mask_list.append( torch.BoolTensor( test_mask ).to( device=device ) )

        return PreparedChoiceBatch(
            tokens=torch.stack( tokens_list ),
            targets=torch.stack( targets_list ),
            train_mask=torch.stack( train_mask_list ),
            test_mask=torch.stack( test_mask_list ),
            correct_index=correct
        )
    
    def compute_batch( self, prepared_batch: PreparedChoiceBatch, logits: torch.Tensor ) -> dict:

        # Grab easy references to elements of the batch
        targets = prepared_batch.targets.unsqueeze( -1 )
        test_mask = prepared_batch.test_mask
        correct_index = prepared_batch.correct_index

        # Compute per token log probabilites and gather by target token
        logprobs = logits.log_softmax( -1 ).gather( -1, targets ).squeeze( -1 )

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
    
    @torch.inference_mode
    def evaluate_document( self, task: BaseChoiceInstructDataset, doc: dict ) -> dict:
        device = self.model.get_input_embeddings().weight.device
        prepared_batch = self.prepare_batch( task, doc, device )
        logits = self.model( prepared_batch.tokens ).logits
        results = self.compute_batch( prepared_batch, logits )

        return results
    
    def evaluate_dataset( self, task: BaseChoiceInstructDataset, dataset: Iterable ):
        correct_list = []
        answer_list = []

        for line in dataset:
            results = self.evaluate_document( task, line )
            correct_list.append( results[ 'references' ] )
            answer_list.append( results[ 'predictions' ] )
        
        return task.compute_metric( references=correct_list, predictions=answer_list )
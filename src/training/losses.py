"""
Module containing loss functions and metrics.

TODO: create base loss (and metrics?) class.
TODO: move to different package?
"""

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

""" ========================================================================
    Loss classes
    ======================================================================== """

class MLELoss( nn.Module ):
    """ Wrapper for sparse cross entropy loss with logits.

    TODO: create base loss class that MLELoss inherts from.
    """

    def __init__( self, vocab_size, pad_token_id ):
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

        self.train_fct = CrossEntropyLoss( ignore_index=-100 )

    def forward( self, last_hidden_states, logits, input_ids, labels ):
        # pylint: disable=W0613
        mle_loss = self.train_fct(logits.float().view(-1, self.vocab_size), labels.view(-1))

        return mle_loss, mle_loss * 0.0

class SimCTGLoss( nn.Module ):
    """ Wrapper for SimCTG, i.e. contrastive loss + MLE loss.

    TODO: create base loss class that MLELoss inherts from

    @inproceedings{su2022a,
        title={A Contrastive Framework for Neural Text Generation},
        author={Yixuan Su and Tian Lan and Yan Wang and Dani Yogatama and Lingpeng Kong and Nigel Collier},
        booktitle={Advances in Neural Information Processing Systems},
        editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
        year={2022},
        url={https://openreview.net/forum?id=V88BafmH9Pj}
    }
    """

    def __init__( self, margin, vocab_size, pad_token_id, compute_device: str | torch.device='cuda' ):
        super().__init__()
        '''
           margin: predefined margin to push similarity score away
           vocab_size: the vocabulary size of the tokenizer
           pad_token_id: indicating which tokens are padding token
        '''
        self.margin = margin
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

        self.train_fct = CrossEntropyLoss( ignore_index=-100 )

        self.compute_device = compute_device

    # the part for contrastive loss
    def build_mask_matrix( self, input_mask: torch.Tensor ) -> torch.Tensor:
        """
        Builds the contrastive loss mask.

        Args:
            input_mask (torch.Tensor): Mask where zeros are masked out

        Returns:
            torch.Tensor: Mask matrix where the diagonal and pad entries are masked out
        """
        seq_len = input_mask.shape[-1]
        base_mask = 1.0 - torch.eye( seq_len, seq_len, device=input_mask.device )
        base_mask = base_mask[ None, ... ]

        input_mask = input_mask[ ..., None ]

        return base_mask * input_mask * input_mask.mT


    def contrastive_loss(self, score_matrix, input_ids):
        '''
           score_matrix: bsz x seqlen x seqlen
           input_ids: bsz x seqlen
        '''
        bsz, seqlen, _ = score_matrix.size()
        gold_score = torch.diagonal(score_matrix, offset=0, dim1=1, dim2=2) # bsz x seqlen
        gold_score = torch.unsqueeze(gold_score, -1)
        assert gold_score.size() == torch.Size([bsz, seqlen, 1])
        difference_matrix = gold_score - score_matrix
        assert difference_matrix.size() == torch.Size([bsz, seqlen, seqlen])
        loss_matrix = self.margin - difference_matrix # bsz x seqlen x seqlen
        loss_matrix = torch.nn.functional.relu(loss_matrix)

        ### input mask
        input_mask = torch.ones_like( input_ids ).type( torch.float16 ) # type: ignore
        if loss_matrix.is_cuda:
            input_mask = input_mask.cuda(loss_matrix.get_device())
        input_mask = input_mask.masked_fill(input_ids.eq(self.pad_token_id), 0.0)

        if loss_matrix.is_cuda:
            input_mask = input_mask.cuda(loss_matrix.get_device())

        loss_mask = self.build_mask_matrix( input_mask )
        if score_matrix.is_cuda:
            loss_mask = loss_mask.cuda(score_matrix.get_device())
        masked_loss_matrix = loss_matrix * loss_mask

        loss_matrix = torch.sum(masked_loss_matrix, dim = -1)
        assert loss_matrix.size() == input_ids.size()
        loss_matrix = loss_matrix * input_mask
        cl_loss = torch.sum(loss_matrix) / torch.sum(loss_mask)
        return cl_loss

    def forward(self, last_hidden_states, logits, input_ids, labels):
        '''
            last_hidden_states: bsz x seqlen x embed_dim
            logits: bsz x seqlen x vocab_size
            input_ids: bsz x seqlen
            labels: bsz x seqlen
        '''
        bsz, seqlen = input_ids.size()
        assert labels.size() == input_ids.size()
        assert logits.size() == torch.Size([bsz, seqlen, self.vocab_size])
        # compute mle loss
        mle_loss = self.train_fct(logits.float().view(-1, self.vocab_size), labels.view(-1))

        # compute cl loss
        norm_rep = last_hidden_states / last_hidden_states.norm(dim=2, keepdim=True)
        cosine_scores = torch.matmul(norm_rep, norm_rep.transpose(1,2))
        assert cosine_scores.size() == torch.Size([bsz, seqlen, seqlen])
        cl_loss = self.contrastive_loss(cosine_scores, input_ids).float()
        return mle_loss, cl_loss


class DPOLoss( nn.Module ):
    """ Implements the Direct Preference Optimization objective function.
    """

    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        average_logprobs: bool = False,
    ):
        """ Instantiates the DPO Loss module.

        Slightly based off TRL's DPOTrainer module, but only slightly.

        Args:
            beta (float, optional): The beta factor in DPO loss. Higher beta means less divergence from the reference policy. Defaults to 0.1.
            label_smoothing (float, optional): The robust DPO label smoothing parameter from the [cDPO](https://ericmitchell.ai/cdpo.pdf) report that should be between 0 and 0.5. Defaults to 0.0.
            average_logprobs (bool, optional): Determines if logprobs should be aggregated by averaging rather than sum. Defaults to False.
        """

        super().__init__()

        self.beta = beta
        self.label_smoothing = label_smoothing
        self.average_logprobs = average_logprobs

    def get_logprobs( self, logits: torch.Tensor, targets: torch.LongTensor ) -> torch.Tensor:
        logprobs = logits.log_softmax( -1, torch.float32 ).gather( -1, targets.unsqueeze( -1 ) ).squeeze( -1 )
        mask = targets != -100
        masked_logprobs = logprobs * mask

        if self.average_logprobs:
            return masked_logprobs.sum( -1 ) / mask.sum( -1 )
        else:
            return masked_logprobs.sum( -1 )

    def forward(
        self,
        *,
        policy_pos_logits: torch.Tensor,
        policy_neg_logits: torch.Tensor,
        reference_pos_logits: torch.Tensor,
        reference_neg_logits: torch.Tensor,
        pos_labels: torch.LongTensor,
        neg_labels: torch.LongTensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """ Compute the DPO loss and returns additional auxilary metrics

        Args:
            policy_pos_logits (torch.Tensor): positive/chosen logits from policy model
            policy_neg_logits (torch.Tensor): negative/rejected logits from policy model
            reference_pos_logits (torch.Tensor): positive/chosen logits from reference model
            reference_neg_logits (torch.Tensor): negative/rejected logits from reference model
            pos_labels (torch.LongTensor): positive/chosen input labels
            neg_labels (torch.LongTensor): negative/rejected input labels

        Returns:
            loss (torch.Tensor): average DPO loss with respect to inputs
            metrics (dict[str, torch.Tensor]]): detached metrics for DPO loss
        """

        # Get aggregated log probs
        policy_pos_logp = self.get_logprobs( policy_pos_logits, pos_labels )
        policy_neg_logp = self.get_logprobs( policy_neg_logits, neg_labels )
        reference_pos_logp = self.get_logprobs( reference_pos_logits, pos_labels )
        reference_neg_logp = self.get_logprobs( reference_neg_logits, neg_labels )

        # Compute the ratios in logspace
        pi_logratios = policy_pos_logp - policy_neg_logp
        ref_logratios = reference_pos_logp - reference_neg_logp

        # Get the logits for DPO
        logits = pi_logratios - ref_logratios

        # Compute DPO on logits
        losses = (
            - F.logsigmoid( self.beta * logits ) * ( 1.0 - self.label_smoothing ) # pylint: disable=E1102
            - F.logsigmoid( - self.beta * logits ) * ( self.label_smoothing ) # pylint: disable=E1102
        )

        # Get reward metrics
        pos_rewards = self.beta * ( policy_pos_logp - reference_pos_logp ).detach()
        neg_rewards = self.beta * ( policy_neg_logp - reference_neg_logp ).detach()
        reward_margins = pos_rewards - neg_rewards
        reward_accuracy = ( pos_rewards > neg_rewards ).float()

        # Grab metric dict and detach
        metrics = {}
        metrics[ 'dpo/chosen' ] = pos_rewards.mean().detach()
        metrics[ 'dpo/rejected' ] = neg_rewards.mean().detach()
        metrics[ 'dpo/accuracy' ] = reward_accuracy.mean().detach()
        metrics[ 'dpo/margin' ] = reward_margins.mean().detach()

        return losses.mean(), metrics


class DPHLoss( nn.Module ):
    """ Implements the Direct Preference Head objective function.

    DPH differs from DPO in that an auxilary classification head is used rather than sequence logprobs.
    And unlike DPO, DPH does not use a reference model, and instead relies on label smoothing to avoid overfitting.
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        contrastive: bool = False,
    ):
        """ Instantiates the DPH Loss module.

        Args:
            label_smoothing (float, optional): Label smoothing coeficient, should be between 0.0 and 0.5. Defaults to 0.0.
            contrastive (bool, optional): When true computes the loss using logit deltas, but is otherwise identical. Defaults to False.
        """

        super().__init__()

        self.label_smoothing = label_smoothing
        self.contrastive = contrastive

    def forward(
        self,
        *,
        pos_logits: torch.Tensor,
        neg_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """ Compute the DPH loss and returns additional auxilary metrics

        Args:
            pos_logits (torch.Tensor): positive/chosen logits from the preference head
            neg_logits (torch.Tensor): negitive/rejected logits from the preference head

        Returns:
            loss (torch.Tensor): average DPO loss with respect to inputs
            metrics (dict[str, torch.Tensor]]): detached metrics for DPO loss
        """

        # Cast to float for stability
        pos_logits = pos_logits.float()
        neg_logits = neg_logits.float()

        # Compute the contrastive loss
        con_logits = pos_logits - neg_logits
        con_loss = (
            - F.logsigmoid( con_logits ) * ( 1.0 - self.label_smoothing ) # pylint: disable=E1102
            - F.logsigmoid( - con_logits ) * ( self.label_smoothing ) # pylint: disable=E1102
        ).mean()

        # Compute the individual losses
        pos_loss = F.binary_cross_entropy_with_logits( pos_logits, torch.ones_like( pos_logits ) - self.label_smoothing, reduction='mean' )
        neg_loss = F.binary_cross_entropy_with_logits( neg_logits, torch.zeros_like( neg_logits ) + self.label_smoothing, reduction='mean' )
        sep_loss = pos_loss + neg_loss

        # Select the desired loss
        loss = con_loss if self.contrastive else sep_loss

        # Get accuracy of preference head
        accuracy = ( pos_logits > neg_logits ).float()

        # Grab metrics and detach
        metrics = {}
        metrics[ 'dph/chosen' ] = pos_loss.detach()
        metrics[ 'dph/rejected' ] = neg_loss.detach()
        metrics[ 'dph/accuracy' ] = accuracy.mean().detach()
        metrics[ 'dph/margin' ] = con_logits.mean().detach()

        return loss, metrics


class ORPOLoss( nn.Module ):
    """ Implements the Odds Ratio Preference Optimization algorithm.
    
    @misc{hong2024orpo,
        title={ORPO: Monolithic Preference Optimization without Reference Model}, 
        author={Jiwoo Hong and Noah Lee and James Thorne},
        year={2024},
        eprint={2403.07691},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }
    """
    
    def __init__(
        self,
        alpha_orpo: float,
        alpha_mle: float,
        vocab_size: int,
    ):
        """ Instantiates the DPO Loss module.

        Args:
            alpha_orpo (float): Strength of the odds ratio loss component.
            alpha_mle (float): Strength of the MLE loss component.
            vocab_size (int): Size of vocabulary.
        """
        
        super().__init__()
        
        self.alpha_orpo = alpha_orpo
        self.alpha_mle = alpha_mle
        self.vocab_size = vocab_size
        
        self.mle_fct = CrossEntropyLoss( ignore_index=-100 )
    
    def get_logprobs( self, logits: torch.Tensor, targets: torch.LongTensor ) -> torch.Tensor:
        logprobs = logits.log_softmax( -1, torch.float32 ).gather( -1, targets.unsqueeze( -1 ) ).squeeze( -1 )
        mask = targets != -100
        masked_logprobs = logprobs * mask
        
        return masked_logprobs.sum( -1 ) / mask.sum( -1 )
    
    def forward(
        self,
        *,
        policy_pos_logits: torch.Tensor,
        policy_neg_logits: torch.Tensor,
        pos_labels: torch.LongTensor,
        neg_labels: torch.LongTensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """ Compute the ORPO loss and returns additional auxilary metrics

        Args:
            policy_pos_logits (torch.Tensor): positive/chosen logits from policy model
            policy_neg_logits (torch.Tensor): negative/rejected logits from policy model
            pos_labels (torch.LongTensor): positive/chosen input labels
            neg_labels (torch.LongTensor): negative/rejected input labels

        Returns:
            loss (torch.Tensor): average DPO loss with respect to inputs
            metrics (dict[str, torch.Tensor]]): detached metrics for DPO loss
        """        
        
        # Calculate MLE loss
        mle_loss = self.train_fct(
            policy_pos_logits.float().view( -1, self.vocab_size ),
            pos_labels.view( -1 )
        )
        
        # Get aggregated log probs
        policy_pos_logp = self.get_logprobs( policy_pos_logits, pos_labels )
        policy_neg_logp = self.get_logprobs( policy_neg_logits, neg_labels )
        
        # Calculate log odds
        log_odds_n = policy_pos_logp - policy_neg_logp
        log_odds_d = torch.log1p( - torch.exp( policy_pos_logp ) ) - torch.log1p( - torch.exp( policy_neg_logp ) )
        log_odds = log_odds_n - log_odds_d
        
        ratio = F.logsigmoid( log_odds ).mean() # pylint: disable=E1102
        
        # Compute final loss
        loss = mle_loss * self.alpha_mle - ratio * self.alpha_orpo
        
        metrics = {}
        metrics[ 'orpo/pos_mean' ] = torch.mean( policy_pos_logp ).item()
        metrics[ 'orpo/neg_mean' ] = torch.mean( policy_neg_logp ).item()
        metrics[ 'orpo/log_odds_ratio' ] = torch.mean( ratio ).item()
        metrics[ 'orpo/log_odds' ] = torch.mean( log_odds ).item()
        
        return loss, metrics


""" ========================================================================
    Metric classes
    ======================================================================== """

class AccuracyMetric( nn.Module ):
    """ Wrapper for sparse multi-class accuracy.

    TODO: inhert from a base class. Decide if should be base 'loss' or base 'metric'.
    TODO: move to seperate metrics module?
    """

    def __init__( self, vocab_size, pad_token_id ):
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

    def forward( self, logits: torch.Tensor, labels: torch.LongTensor ):
        tp = ( logits.argmax( dim=-1 ) == labels ).sum().float()
        valid_len = ( labels != self.pad_token_id ).sum().float()

        return tp / valid_len

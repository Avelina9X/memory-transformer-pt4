"""
Module containing loss functions and metrics.

TODO: create base loss (and metrics?) class.
TODO: move to different package?
"""

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

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

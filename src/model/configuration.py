# pylint: disable=R0902,R0913,R0914

"""
Configuration classes for the LSWTransformer

Contains:
    - LSWTConfig: configuration class for model architecture
    - LSWTConfigTraining: configuration class for training settings
"""

from collections.abc import Sequence
import json
from typing import Any, Literal

from transformers import PretrainedConfig


class LSWTPoolerConfig( PretrainedConfig ):
    model_type = "lsw_transformer_pooler"
    
    def __init__(
        self,
        
        reward_heads: Sequence[str] | None = None,
        reward_head_bias: bool = False,
        
        embedding_size: int | None = None,
        
        embedding_dropout=0.0,
        layer_dropout=0.0,
        
        layer_pooling: Literal['layer', 'mean', 'weighted_sum'] = 'layer',
        layer_pooling_norm: Literal['pre', 'post', 'both', None] = None,
        
        token_pooling: Literal['cls', 'max', 'mean', 'sgpt', 'ema'] = 'cls',
        token_pooling_norm: Literal['pre', 'post', 'both', None] = None,
        token_pooling_ema_beta: float | None = None,
        token_pooling_ema_beta_learnable: Literal['global', 'activation', None] = None,
        token_pooling_rotation: bool = False,
        
        pooler_function: Literal['identity', 'projection'] = 'identity',
        pooler_activation: str | None = None,
        pooler_activation_gated=False,
        
        layer_select: int | Sequence[int] = -1,
        
        prefix_sizes: dict[str, int] | None = None,
        
        **kwargs
    ):
        
        super().__init__( **kwargs )
        
        self.reward_heads = reward_heads
        self.reward_head_bias = reward_head_bias
        
        self.embedding_size = embedding_size
        
        self.embedding_dropout = embedding_dropout
        self.layer_dropout = layer_dropout
        
        self.pooler_function = pooler_function
        self.pooler_activation = pooler_activation
        self.pooler_activation_gated = pooler_activation_gated
        
        self.layer_select = layer_select
        
        self.layer_pooling = layer_pooling
        self.layer_pooling_norm = layer_pooling_norm
        
        self.token_pooling = token_pooling
        self.token_pooling_norm = token_pooling_norm
        self.token_pooling_ema_beta = token_pooling_ema_beta
        self.token_pooling_ema_beta_learnable = token_pooling_ema_beta_learnable
        self.token_pooling_rotation = token_pooling_rotation
        
        if prefix_sizes is None:
            prefix_sizes = {
                'system': 3,
                'user': 3,
                'assistant': 4,
            }
        
        self.prefix_sizes = prefix_sizes
        
        if ( layer_pooling == 'layer' ) ^ ( isinstance( layer_select, int ) ):
            raise ValueError( 'layer_select must be an int when layer_pooling is `layer`' )
                
        if pooler_function == 'identity':
            if self.embedding_size is not None:
                raise ValueError( 'embedding_size must be unset for pooler_function=`identity`' )
            
            if self.pooler_activation is not None:
                raise ValueError( 'pooler_activation must be unset for pooler_function=`identity`' )
        
        elif pooler_function in [ 'projection', 'sae' ]:
            if self.embedding_size is None:
                raise ValueError( f'embedding_size must be set for pooler_function=`{pooler_function}`' )
            
            if self.pooler_activation is None:
                raise ValueError( f'pooler_activation must be set for pooler_function=`{pooler_function}`' )
        
        else:
            raise ValueError( 'Invalid pooler_function type' )
        
        if ( token_pooling == 'ema' ) ^ ( isinstance( token_pooling_ema_beta, float ) ):
            raise ValueError( 'token_pooling_ema_beta must be a float if and only if token_pooling=`ema`' )
        
        if ( token_pooling != 'ema' ) and token_pooling_ema_beta_learnable:
            raise ValueError( 'token_pooling_ema_beta_learnable can only be set when token_pooling=`ema`' )
        
        if ( token_pooling_ema_beta_learnable != 'activation' ) and token_pooling_rotation:
            raise ValueError( 'token_pooling_rotation can only be true if token_pooling_ema_beta_learnable=`activation`' )

class LSWTConfig( PretrainedConfig ):
    """
    Configuration class for the LSWTransformer architecture.

    Class attributes:
        - model_type: the model type prefix
    """

    model_type = "lsw_transformer"
    keys_to_ignore_at_inference = [ "past_key_values" ]

    attribute_map = {
        'hidden_size': 'd_model',
        'num_attention_heads': 'n_heads',
        'num_hidden_layers': 'n_layers',
        'intermediate_size': 'd_ffn',
        'max_position_embeddings': 'rope_positions',
    }

    def __init__(
        self,

        vocab_size=50272,
        trainable_embeddings=True,

        d_vocab=768,
        d_model=768,
        d_ffn=3072,
        n_heads=12,
        n_layers=12,

        n_registers=0,

        hidden_act='silu',
        gated_ffn=True,
        gated_att=False,

        qk_norm=False,

        enable_bias=True,
        init_std=0.02,

        rope_base_freq=500000,
        rope_reversed=True,
        rope_positions=None,

        rope_dynamic=False,
        rope_ntk_scale=1.0,
        rope_yarn_a=0.07,
        rope_yarn_b=1.0,

        dropout_att_mat=0.0,
        dropout_att_out=0.0,
        dropout_ffn_int=0.0,
        dropout_ffn_out=0.0,
        dropout_layers=0.0,

        pad_token_id=1,
        bos_token_id=2,
        eos_token_id=2,

        use_cache=True,

        recompute_kv=False,

        parent_embeddings='facebook/opt-125m',
        
        pooler_config: dict | LSWTPoolerConfig | None = None,

        **kwargs,
    ):
        """LSW Transformer Configuration

        Args:
            vocab_size (int): Number of elements in the embedding matrix. Defaults to 50272 (as in OPT).
            trainable_embeddings (bool): Weather the embeddings matrix should be trainable. If False will autocast the matrix to FP16. Defaults to False.

            d_vocab (int): Size of the vocab embedding matrix. Defaults to 768.
            d_model (int): Size of the backbone embedding states. Defaults to 768.
            d_ffn (int): Size of the FFN intermediate *after* gating (i.e. will be double before gating when using SwiGLU). Defaults to 3072.
            n_heads (int): Number of QKV heads in the attention matrix, which also determines head dimension. Defaults to 12.
            n_layers (int): Number of transformer blocks. Defaults to 12.

            n_registers (int): Number of attention sink registers. Set to zero to disable. Defaults to 16.

            hidden_act (str): FFN activation function. Defaults to silu.
            gated_ffn (bool): Weather to use SwiGLU in the FFN. When disabled uses GELU. Defaults to True.
            gated_att (bool): Enables head gating (currently not implemented). Defaults to False. TODO: implement attention gating.

            qk_norm (bool): Apply per-head RMS norm to the queries and keys. Defaults to False.

            enable_bias (bool): Enables bias terms in all backbone projections. Embedding projections never use bias. Defaults to True.
            init_std (float): Normal std for initialisation. Defaults to 0.02.

            rope_base_freq (int): Base frequency for RoPE. Defaults to 500000 (RoPE ABF).
            rope_reversed (bool): Reverses RoPE order (i.e. ReRoPE). Defaults to True.
            rope_positions (int): Number of positions used for training. Defaults to None.

            rope_dynamic (bool): Enables dynamic NTK and YaRN. Defaults to False.
            rope_ntk_scale (float): NTK-Aware Scaling factor. Defaults to 1.0.
            rope_yarn_a (float): YaRN temperature a scale. Zero disables YaRN. Defaults to 0.0.
            rope_yarn_b (float): YaRN temperature b scale. Defaults to 1.0.

            dropout_att_mat (float): Attention matrix dropout. Defaults to 0.0.
            dropout_att_out (float): Attention layer output dropout. Defaults to 0.0.
            dropout_ffn_int (float): FFN layer intermediate dropout. Defaults to 0.0.
            dropout_ffn_out (float): FFN layer output dropout. Defaults to 0.0.
            dropout_layers (float): Drop path/stochastic depth dropout. (Note, scales residual up during training, unlike original implementation which scales residual down during inference). Defaults to 0.0.

            pad_token_id (int): Default PAD token ID. Defaults to 1 (as in OPT).
            bos_token_id (int): Default BOS token ID. Defaults to 2 (as in OPT).
            eos_token_id (int): Default EOS token ID. Defaults to 2 (as in OPT).

            use_cache (bool): Weather KV cache should be enabled. Defaults to True.
            recompute_kv (bool): Recompute keys and values during training for additional gradients. Defaults to False.

            parent_embeddings (str): Parent embeddings and tokenizer vocab. Defaults to 'facebook/opt-125m'.

            pooler_config (dict | LSWTPoolerConfig | None): Pooler config for DPH and beyond. Defaults to None.
        """
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        # Embedding matrix size
        self.vocab_size = vocab_size
        self.trainable_embeddings = trainable_embeddings

        # Backbone shapes
        self.d_vocab = d_vocab
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Auxilary attention sinking registers
        self.n_registers = n_registers
        
        self.hidden_act = hidden_act

        # Gating settings
        self.gated_ffn = gated_ffn
        self.gated_att = gated_att

        # RMS norm for qk
        self.qk_norm = qk_norm

        # Linear layer settings
        self.init_std = init_std
        self.enable_bias = enable_bias

        # Positional embedding settings
        self.rope_base_freq = rope_base_freq
        self.rope_reversed = rope_reversed
        self.rope_positions = rope_positions

        # Context window extension settings
        self.rope_dynamic = rope_dynamic
        self.rope_ntk_scale = rope_ntk_scale
        self.rope_yarn_a = rope_yarn_a
        self.rope_yarn_b = rope_yarn_b

        # Dropout settings
        self.dropout_att_mat = dropout_att_mat
        self.dropout_att_out = dropout_att_out
        self.dropout_ffn_int = dropout_ffn_int
        self.dropout_ffn_out = dropout_ffn_out
        self.dropout_layers = dropout_layers

        # Enable cache
        self.use_cache = use_cache
        self.recompute_kv = recompute_kv

        # Parent embeddings
        self.parent_embeddings = parent_embeddings
        
        if isinstance( pooler_config, dict ):
            self.pooler_config = LSWTPoolerConfig( **pooler_config )
        elif pooler_config is None:
            self.pooler_config = LSWTPoolerConfig()
        else:
            self.pooler_config = pooler_config

        # Assertions
        if d_model % n_heads != 0:
            raise ValueError( 'd_model must be divisible by n_heads' )

    def to_wandb_dict( self, prefix='model' ) -> dict[str, Any]:
        """ Serializes this instance to WandB style dict.

        Args:
            prefix (str, optional): Prefix for all attributes. Defaults to 'model'.

        Returns:
            dict[str, Any]: Dict of all attributes that make up this configuration instance
        """
        return { f'{prefix}.{key}': value for key, value in self.to_diff_dict().items() }


class LSWTConfigTraining():
    """
    Configuration class for traing the LSWTransformer.
    """

    def __init__(
        self,

        batch_size=480,
        batch_size_step=6,
        batches_per_epoch=256,

        length_sequence=2048,
        length_cache=2048,

        lr_max=1e-3,
        lr_warmup_steps=2000,
        lr_cooldown_tokens=30_000_000_000,
        lr_cooldown_ratio=0.1,

        optimizer: Literal['LaProp'] = 'LaProp',
        opt_beta_1=0.9,
        opt_beta_2=0.95,
        opt_eps=1e-8,
        opt_weight_decay=0.1,
        opt_decay_init=False,
        opt_decay_mask: Sequence[str] = ( 'norm', 'bias', 'embedding.weight' ),
        opt_max_grad_norm=1.0,
        opt_rho=0.1,

        loss_objective: Literal['MLE', 'SimCTG'] = 'MLE',
        loss_sim_margin=0.5,
        
        ortho_params: Sequence[str] = ( 'token_rotate.weight', ),
        ortho_beta=0.01,
    ):
        """LSW Transformer Training Configuration

        Args:
            batch_size (int): Global batch size. Defaults to 480.
            batch_size_step (int): Sub-batch size for gradient accumulation. Adjust for maximum throughput. Defaults to 6.
            batches_per_epoch (int): Number of batches to accumulate metrics over. Defaults to 256.

            length_sequence (int): Input context window length. Defaults to 2048.
            length_cache (int): KV cache length. Defaults to 2048.

            lr_max (float): Maximum learning rate in the schedule. Defaults to 6e-4.
            lr_warmup_steps (int): Warmup batches from zero to lr_max. Defaults to 2000.
            lr_cooldown_tokens (int): Number of tokens that cosine decay should end at. Defaults to 30000000000.
            lr_cooldown_ratio (float): Minimum LR at end of cooldown. Defaults to 0.1.

            optimizer (str): Choice of optimizer. Currently supports 'AdamW' and 'SophiaG'. Defaults to 'SophiaG'.
            opt_beta_1 (float): Beta 1 for adaptive optimizers. Defaults to 0.9.
            opt_beta_2 (float): Beta 2 for adaptive optimizers. Defaults to 0.95.
            opt_eps (float): Epsilon factor for Adam optimizers. Defaults to 1e-9.
            opt_weight_decay (float): Weight decay (note that Adam requires half the WD of Sophia). Defaults to 0.2.
            opt_decay_init (bool): Enables Prior Regularization (arxiv:0907.1815). Defaults to False.
            opt_decay_mask (list[str]): list of string patterns which disable weight decay.
            opt_max_grad_norm (float): Max norm for gradient clipping. Set to zero to disable. Defaults to 1.0.
            opt_rho (float): Rho factor for Sophia optimizers. Defaults to 0.05.

            loss_objective (str): Loss objective, supports MLE (standard loss) and SimCTG (MLE + contrastive). Defaults to 'MLE'.
            loss_sim_margin (float): Loss margin for SimCTG. Defaults to 0.5.
        """

        # Batch size settings
        self.batch_size = batch_size
        self.batch_size_step = batch_size_step
        self.batches_per_epoch = batches_per_epoch

        # Working and Short/Long sequence length settings
        self.length_sequence = length_sequence
        self.length_cache = length_cache

        # Learning rate settings
        self.lr_max = lr_max
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_cooldown_tokens = lr_cooldown_tokens
        self.lr_cooldown_ratio = lr_cooldown_ratio

        # Optimizer settings
        self.optimizer = optimizer
        self.opt_beta_1 = opt_beta_1
        self.opt_beta_2 = opt_beta_2
        self.opt_eps = opt_eps
        self.opt_weight_decay = opt_weight_decay
        self.opt_decay_init = opt_decay_init
        self.opt_decay_mask = opt_decay_mask
        self.opt_max_grad_norm = opt_max_grad_norm
        self.opt_rho = opt_rho

        # Learning objective
        self.loss_objective=loss_objective
        self.loss_sim_margin=loss_sim_margin
        
        # Orthogonalisation
        self.ortho_params = ortho_params
        self.ortho_beta = ortho_beta

        # Assertions
        if batch_size % batch_size_step != 0:
            raise ValueError( 'batch_size must be divisible by batch_size_step' )
        if loss_objective not in [ 'MLE', 'SimCTG' ]:
            raise ValueError( 'loss_objective must be "MLE" or "SimCTG"' )

    def to_json_string( self ) -> str:
        """ Serializes this instance to a JSON string.

        Returns:
            str: JSON string of the attributes that make up this configuration instance.
        """
        return json.dumps( self.__dict__, indent=2 )

    def to_wandb_dict( self, prefix='train' ) -> dict[str, Any]:
        """ Serializes this instance to WandB style dict.

        Args:
            prefix (str, optional): Prefix for all attributes. Defaults to 'train'.

        Returns:
            dict[str, Any]: Dict of all attributes that make up this configuration instance
        """
        return { f'{prefix}.{key}': value for key, value in self.__dict__.items() }

    def __repr__( self ):
        return f'{self.__class__.__name__} {self.to_json_string()}'

class LSWTConfigTrainingDPH():
    def __init__(
        self,

        dpo_enabled=False,
        dpo_beta=0.1,
        dpo_epsilon=0.1,
        dpo_average_logprobs=False,
        dpo_weight=1.0,
        
        orpo_enabled=False,
        orpo_alpha_orpo=0.25,
        orpo_alpha_mle=1.0,
        orpo_weight=1.0,
        
        kl_enabled=False,
        kl_pn_ratio=0.5,
        kl_penalty=0.2,
        kl_weight=1.0,

        dph_contrastive=False,
        dph_epsilon=0.1,
        dph_penalty=0.1,
        dph_weight=1.0,

        dph_decay_init=False,
        dph_weight_decay=0.1,
        dph_lr_multiplier=1.0,
        dph_decay_mask: Sequence[str] = ( 'norm', 'bias', 'ema', 'layer_weighting', 'token_rotate' ),
    ):
        """ LSW Transformer config class

        Args:
            dpo_enabeld (bool): If DPO should be enabled. Defaults to False.
            dpo_beta (float, optional): Beta parameter for DPO objective. Defaults to 0.1.
            dpo_epsilon (float, optional): Label smoothing parameter for DPO objective. Defaults to 0.1.
            dpo_average_logprobs (bool, optional): When True uses average logprobs instead of sum. Defaults to False.
            dpo_weight (float, optional): Loss strength of DPO. Defaults to 1.0.
            
            orpo_enabled (bool, optional): If ORPO should be enabled. Defaults to True.
            orpo_alpha_orpo (float, optional): ORPO weigh coefficient. Defaults to 0.25.
            orpo_alpha_mle (float, optional): MLE weight coefficient. Defaults to 1.0.
            orpo_weight (float, optional): Loss strength for combined ORPO+MLE. Defaults to 1.0.
            
            kl_enabled (bool, optional): If KLPairsLoss should be enabled. Defaults to false.
            kl_pn_ratio (float, optional): Postive to negative penalty ratio. 1=only positive penalty, 0=only negative penalty, 0.5=balanced penalty. Defaults to 0.5.
            kl_penalty (float, optional): The KL Penalty to apply *after* balancing postive and negative examples. Defaults to 0.2.
            kl_weight (float, optional): Loss strength of KL. Defaults to 1.0.

            dph_contrastive (bool, optional): When true uses ConDPH instead of SepDPH. Defaults to False.
            dph_epsilon (float, optional): Label smoothing parameter for DPH objective. Defaults to 0.1.
            dph_penalty (float, optional): L2 penalty coefficient for DPH. Defaults to 0.1.
            dph_weight (float, optional): Loss strength of DPH. Defaults to 1.0.
            
            dph_decay_init (bool, optional): When true swaps weight decay with prior regularization. Defaults to False.
            dph_weight_decay (float, optional): The weight decay (or prior regularization) coefficient. Defaults to 0.1.
        """

        self.dpo_enabled = dpo_enabled
        self.dpo_beta = dpo_beta
        self.dpo_epsilon = dpo_epsilon
        self.dpo_average_logprobs = dpo_average_logprobs
        self.dpo_weight = dpo_weight
        
        self.orpo_enabled = orpo_enabled
        self.orpo_alpha_orpo = orpo_alpha_orpo
        self.orpo_alpha_mle = orpo_alpha_mle
        self.orpo_weight = orpo_weight
        
        self.kl_enabled = kl_enabled
        self.kl_pn_ratio = kl_pn_ratio
        self.kl_penalty = kl_penalty
        self.kl_weight = kl_weight

        self.dph_contrastive = dph_contrastive
        self.dph_epsilon = dph_epsilon
        self.dph_penalty = dph_penalty
        self.dph_weight = dph_weight

        self.dph_decay_init = dph_decay_init
        self.dph_weight_decay = dph_weight_decay
        self.dph_lr_multiplier = dph_lr_multiplier
        self.dph_decay_mask = dph_decay_mask

        # DPO Assertions
        if self.dpo_beta < 0:
            raise ValueError( f'DPO beta must be at least 0.0, but received {self.dpo_beta}' )
        
        if not ( 0.0 <= self.dpo_epsilon <= 0.5 ):
            raise ValueError( f'DPO epsilon must be in range [0.0,0.5], but received {self.dpo_epsilon}' )
        
        if self.dpo_weight < 0:
            raise ValueError( f'DPO weight must be at least 0.0, but received {self.dpo_weight}' )


        # ORPO Assertions
        if self.orpo_alpha_mle < 0:
            raise ValueError( f'ORPO MLE alpha must be at least 0.0, but received {self.orpo_alpha_mle}' )
        
        if self.orpo_alpha_orpo < 0:
            raise ValueError( f'ORPO Odds alpha must be at least 0.0, but received {self.orpo_alpha_orpo}' )
        
        if self.orpo_weight < 0:
            raise ValueError( f'ORPO weight must be at least 0.0, but received {self.orpo_weight}' )


        # KL Assertions
        if not ( 0.0 <= self.kl_pn_ratio <= 1.0 ):
            raise ValueError( f'KL p-n ration must be in range [0.0,1.0], but received {self.kl_pn_ratio}' )
        
        if self.kl_penalty < 0:
            raise ValueError( f'KL penalty must be at least 0.0, but receive {self.kl_penalty}' )
        
        if self.kl_weight < 0:
            raise ValueError( f'KL weight must be at least 0.0, but received {self.kl_weight}' )


        # DPH Assertions
        if not ( 0.0 <= self.dph_epsilon <= 0.5 ):
            raise ValueError( f'DPH epsilon must be in range [0.0,0.5], but received {self.dph_epsilon}' )
        
        if self.dph_penalty < 0:
            raise ValueError( f'DPH L2 penalty must be at least 0.0, but receive {self.dph_penalty}' )
        
        if self.dph_weight < 0:
            raise ValueError( f'DPH weight must be at least 0.0, but received {self.dph_weight}' )


    def to_json_string( self ) -> str:
        """ Serializes this instance to a JSON string.

        Returns:
            str: JSON string of the attributes that make up this configuration instance.
        """
        return json.dumps( self.__dict__, indent=2 )

    def to_wandb_dict( self, prefix='dph' ) -> dict[str, Any]:
        """ Serializes this instance to WandB style dict.

        Args:
            prefix (str, optional): Prefix for all attributes. Defaults to 'train'.

        Returns:
            dict[str, Any]: Dict of all attributes that make up this configuration instance
        """
        return { f'{prefix}.{key}': value for key, value in self.__dict__.items() }

    def __repr__( self ):
        return f'{self.__class__.__name__} {self.to_json_string()}'
    
    @property
    def requires_reference_model( self ):
        return self.dpo_enabled or self.kl_enabled

from transformers import PretrainedConfig
import json

class LSWTConfig( PretrainedConfig ):
    model_type = "lsw_transformer"
    
    def __init__(
        self,
        
        vocab_size=50272,
        trainable_embeddings=False,
        
        d_vocab=768,
        d_model=768,
        d_ffn=3072,
        n_heads=12,
        n_layers=12,
        
        n_registers=16,
        
        gated_ffn=True,
        gated_att=False,
        
        enable_bias=True,
        init_std=0.02,
        
        rope_base_freq=500000,
        rope_reversed=True,
        rope_xpos_scale=512,
        rope_xpos_enabled=False,
        
        dropout_att_mat=0.0,
        dropout_att_out=0.0,
        dropout_ffn_int=0.0,
        dropout_ffn_out=0.0,
        dropout_layers=0.0,
        
        pad_token_id=1,
        bos_token_id=2,
        eos_token_id=2,
        
        use_cache=True,
        
        parent_embeddings='facebook/opt-125m',
        
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
            
            gated_ffn (bool): Weather to use SwiGLU in the FFN. When disabled uses GELU. Defaults to True.
            gated_att (bool): Enables head gating (currently not implemented). Defaults to False. TODO: implement attention gating.
            
            enable_bias (bool): Enables bias terms in all backbone projections. Embedding projections never use bias. Defaults to True.
            init_std (float): Normal std for initialisation. Defaults to 0.02.
            
            rope_base_freq (int): Base frequency for RoPE. Defaults to 500000 (RoPE ABF).
            rope_reversed (bool): Reverses RoPE order (i.e. ReRoPE). Defaults to True.
            rope_xpos_scale (int): XPOS decay factor for RoPE. Defaults to 512.
            rope_xpos_enabled (bool): Enables XPOS decay for RoPE. Defaults to False.
            
            dropout_att_mat (float): Attention matrix dropout. Defaults to 0.0.
            dropout_att_out (float): Attention layer output dropout. Defaults to 0.0.
            dropout_ffn_int (float): FFN layer intermediate dropout. Defaults to 0.0.
            dropout_ffn_out (float): FFN layer output dropout. Defaults to 0.0.
            dropout_layers (float): Drop path/stochastic depth dropout. (Note, scales residual up during training, unlike original implementation which scales residual down during inference). Defaults to 0.0.
            
            pad_token_id (int): Default PAD token ID. Defaults to 1 (as in OPT).
            bos_token_id (int): Default BOS token ID. Defaults to 2 (as in OPT).
            eos_token_id (int): Default EOS token ID. Defaults to 2 (as in OPT).
            
            use_cache (bool): Weather KV cache should be enabled. Defaults to True.
            
            parent_embeddings (str): Parent embeddings and tokenizer vocab. Defaults to 'facebook/opt-125m'.
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

        # Gating settings
        self.gated_ffn = gated_ffn
        self.gated_att = gated_att

        # Linear layer settings
        self.init_std = init_std
        self.enable_bias = enable_bias
        
        # Positional embedding settings
        self.rope_base_freq = rope_base_freq
        self.rope_reversed = rope_reversed
        self.rope_xpos_scale = rope_xpos_scale
        self.rope_xpos_enabled = rope_xpos_enabled

        # Dropout settings
        self.dropout_att_mat = dropout_att_mat
        self.dropout_att_out = dropout_att_out
        self.dropout_ffn_int = dropout_ffn_int
        self.dropout_ffn_out = dropout_ffn_out
        self.dropout_layers = dropout_layers
        
        # Enable cache
        self.use_cache = use_cache
        
        # Parent embeddings
        self.parent_embeddings = parent_embeddings
        
        # Assertions
        assert d_model % n_heads == 0, 'd_model must be divisible by n_heads'
    
    def to_wandb_dict( self, prefix='model' ):        
        return { f'{prefix}.{key}': value for key, value in self.to_diff_dict().items() }

class LSWTConfigTraining():
    def __init__(
        self,
        
        batch_size=480,
        batch_size_step=6,
        batches_per_epoch=256,
        
        length_sequence=2048,
        length_cache=2048,
        
        lr_max=6e-4,
        lr_warmup_steps=2000,
        lr_cooldown_tokens=30_000_000_000,
        lr_cooldown_ratio=0.1,
        
        optimizer='SophiaG',
        opt_beta_1=0.9,
        opt_beta_2=0.95,
        opt_eps=1e-9,
        opt_weight_decay=0.2,
        opt_max_grad_norm=1.0,
        opt_rho=0.05,
        
        loss_objective='SimCTG',
        loss_sim_margin=0.5,
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
            opt_max_grad_norm (float): Max norm for gradient clipping. Set to zero to disable. Defaults to 1.0.
            opt_rho (float): Rho factor for Sophia optimizers. Defaults to 0.05.
            
            loss_objective (str): Loss objective, supports MLE (standard loss) and SimCTG (MLE + contrastive). Defaults to 'SimCTG'.
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
        self.opt_max_grad_norm = opt_max_grad_norm
        self.opt_rho = opt_rho
        
        # Learning objective
        self.loss_objective=loss_objective
        self.loss_sim_margin=loss_sim_margin
        
        # Assertions
        assert batch_size % batch_size_step == 0, 'batch_size must be divisible by batch_size_step'
        assert loss_objective in [ 'MLE', 'SimCTG' ], 'loss_objective must be "MLE" or "SimCTG"'
    
    def to_json_string( self ):
        return json.dumps( self.__dict__, indent=2 )
        
    def __repr__( self ):
        return f'{self.__class__.__name__} {self.to_json_string()}'
    
    def to_wandb_dict( self, prefix='train' ):
        return { f'{prefix}.{key}': value for key, value in self.__dict__.items() }
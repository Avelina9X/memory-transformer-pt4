"""
Module containing all the layers required for the LSWTransformer architecture.
"""

import torch
from flash_attn import flash_attn_func # type: ignore # pylint: disable=E0401

from .configuration import LSWTConfig

@torch._dynamo.disable # type: ignore # pylint: disable=W0212
def flash_attention( query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, dropout: float ):
    return flash_attn_func( query, key, value, dropout_p=dropout, causal=True )

class DropPath( torch.nn.Module ):
    """ DropPath layer.
    
    Stochastically drops a residual block during training.
    
    Note: during training output gain is increased when dropout is greater than zero,
    this mimics typical dropout behaviour.
    """
    
    def __init__( self, drop_prob: float, scale=True ):
        super().__init__()

        self.drop_prob = drop_prob
        self.scale = scale

    def forward( self, skip: torch.Tensor, residual: torch.Tensor ):
        keep_prob = 1.0 - self.drop_prob

        if not self.training or self.drop_prob == 0.0:
            return skip + residual

        new_shape = ( residual.shape[0], ) + ( 1, ) * ( residual.ndim - 1 )
        residual = residual * residual.new_empty( new_shape ).bernoulli_( keep_prob )
        return skip + residual / keep_prob

class SharedEmbeddings( torch.nn.Module ):
    """ Shared embedding layer.
    
    This implements tied embeddings using one layer instead of sharing params with a linear layer.
    """
    
    def __init__( self, vocab_size: int, d_vocab: int ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_vocab = d_vocab

        self.embedding = torch.nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.d_vocab,
        )

        self.modes = {
            'embed': self.forward_embed,
            'linear': self.forward_linear,
            'softmax': self.forward_softmax,
        }

    def forward_embed( self, x ):
        return self.embedding( x )

    def forward_linear( self, x ):
        return torch.einsum( 'Vd,...d->...V', self.embedding.weight, x )

    def forward_softmax( self, x ):
        return torch.softmax( self.forward_linear( x ), dim=-1 )

    def forward( self, x, mode ):
        return self.modes[ mode ]( x )

class RotaryEmbedding( torch.nn.Module ):
    """ Rotary Embedding layer.
    
    Creates the RoPE embeddings with support for ABF, XPos (experimental), and ReRoPE (reversal).
    """
    
    def __init__(self, dim, scale_base = 512, use_xpos = True, base_freq=10000, reverse=False ):
        super().__init__()
        self.reverse = reverse

        inv_freq = 1.0 / (base_freq ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False )

        self.use_xpos = use_xpos
        self.scale_base = scale_base
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale, persistent=False )

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device = device).type_as(self.inv_freq) # type: ignore
        if self.reverse: t = t.flip( dims=[ 0 ] )

        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)

        if not self.use_xpos:
            return freqs, torch.ones(1, device = device)

        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** power[ :, None ]
        scale = torch.cat((scale, scale), dim = -1)

        if self.reverse: scale = scale ** -1.0

        return freqs, scale

def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def _apply_rotary_pos_emb( pos, t, scale = 1.):
    return (t * pos.cos() * scale).type( t.dtype ) + ( _rotate_half(t) * pos.sin() * scale).type( t.dtype )

def apply_rope( query, key, rope_pos, rope_scale, reverse ):
    q_length = query.shape[-2]
    k_length = key.shape[-2]

    # Get rotary embeddings for queries and keys
    rope_pos_q = rope_pos[ -q_length : ]
    rope_pos_k = rope_pos[ -k_length : ]
    rope_scale_q = ( rope_scale[ -q_length : ] ) if reverse else 1.0
    rope_scale_k = ( rope_scale[ -k_length : ] ** -1.0 ) if reverse else 1.0

    query = _apply_rotary_pos_emb( rope_pos_q, query, rope_scale_q )
    key = _apply_rotary_pos_emb( rope_pos_k, key, rope_scale_k )

    return query, key


class LSWTAttention( torch.nn.Module ):
    """ Multi Head Attention layer with all the features of the LSWT.
    """
    
    def __init__( self, config: LSWTConfig ):
        super().__init__()

        self.config = config

        self.key_dim = config.d_model // config.n_heads

        if config.gated_att:
            raise ValueError( 'gated_att not yet implemented' )

        if config.n_registers > 0:
            register_shape = ( 1, config.n_heads, config.n_registers, self.key_dim )

            self.registers_k = torch.nn.Parameter( torch.empty( *register_shape ), requires_grad=True )
            self.registers_v = torch.nn.Parameter( torch.empty( *register_shape ), requires_grad=True )

            self.registers_k.data.normal_( mean=0.0, std=self.key_dim ** -0.5 )
            self.registers_v.data.normal_( mean=0.0, std=self.key_dim ** -0.5 )

        self.proj_q = torch.nn.Linear( config.d_model, config.d_model, bias=config.enable_bias )
        self.proj_k = torch.nn.Linear( config.d_model, config.d_model, bias=config.enable_bias )
        self.proj_v = torch.nn.Linear( config.d_model, config.d_model, bias=config.enable_bias )
        self.proj_o = torch.nn.Linear( config.d_model, config.d_model, bias=config.enable_bias )

        self.out_dropout = torch.nn.Dropout( config.dropout_att_out )

    @property
    def att_dropout_p( self ):
        return self.config.dropout_att_mat if self.training else 0.0

    def split_heads( self, x ):
        B, S, _ = x.size()
        return x.view( B, S, self.config.n_heads, self.key_dim )

    def merge_heads( self, x ):
        B, S, _, _ = x.size()
        return x.contiguous().view( B, S, self.config.d_model )

    def forward( self, embeddings, past_key_values, rope_pos, rope_scale ):

        # Project qkv and split into heads
        q = self.split_heads( self.proj_q( embeddings ) ).permute( 0, 2, 1, 3 )
        k = self.split_heads( self.proj_k( embeddings ) ).permute( 0, 2, 1, 3 )
        v = self.split_heads( self.proj_v( embeddings ) ).permute( 0, 2, 1, 3 )
        
        # TODO: RMS norm?

        # Append past keys and values
        if past_key_values:
            k = torch.cat( [ past_key_values[0], k ], dim=-2 )
            v = torch.cat( [ past_key_values[1], v ], dim=-2 )

        # Save new past keys and values
        past_keys = k
        past_values = v

        # Apply rotary embeddings
        q, k = apply_rope( q, k, rope_pos, rope_scale, self.config.rope_xpos_enabled )

        # If we are using registers prepend the keys and values
        if self.config.n_registers > 0:
            batch_size = k.shape[0]

            r_k = self.registers_k.to( dtype=k.dtype ).repeat( batch_size, 1, 1, 1 )
            r_v = self.registers_v.to( dtype=v.dtype ).repeat( batch_size, 1, 1, 1 )
            
            # TODO: RMS norm?

            k = torch.cat( [ r_k, k ], dim=-2 )
            v = torch.cat( [ r_v, v ], dim=-2 )

        # Permute back to normal dimension order
        q = q.permute( 0, 2, 1, 3 )
        k = k.permute( 0, 2, 1, 3 )
        v = v.permute( 0, 2, 1, 3 )

        # Do attention
        a = flash_attention( q, k, v, self.att_dropout_p )
        a = self.merge_heads( a )

        # Apply dropout
        o = self.out_dropout( self.proj_o( a ) )

        return o, ( past_keys, past_values )

class SwiGLU( torch.nn.Module ):
    """ SwiGLU activation
    """
    
    def forward( self, x ):
        x, g = x.chunk( 2, dim=-1 )
        return x * torch.nn.functional.silu( g )

class LSWTFeedForward( torch.nn.Module ):
    """ Feedforward MLP with SwiGLU support
    """
    
    def __init__( self, config: LSWTConfig ):
        super().__init__()

        self.config = config

        self.fc1 = torch.nn.Linear( config.d_model, config.d_ffn * ( 2 if config.gated_ffn else 1 ) )
        self.fc2 = torch.nn.Linear( config.d_ffn, config.d_model )
        self.act = SwiGLU() if config.gated_ffn else torch.nn.GELU()

        self.drop_int = torch.nn.Dropout( config.dropout_ffn_int )
        self.drop_out = torch.nn.Dropout( config.dropout_ffn_out )

    def forward( self, x ):
        x = self.fc1( x )
        x = self.act( x )
        x = self.drop_int( x )
        x = self.fc2( x )
        x = self.drop_out( x )

        return x

class LSWTBlock( torch.nn.Module ):
    """ Transformer block for the LSWT.
    """
    
    def __init__( self, config: LSWTConfig ):
        super().__init__()

        self.config = config

        self.att_norm = torch.nn.LayerNorm( config.d_model )
        self.att = LSWTAttention( config )

        self.ffn_norm = torch.nn.LayerNorm( config.d_model )
        self.ffn = LSWTFeedForward( config )

        if config.dropout_layers > 0.0:
            raise ValueError( 'layer drop not yet implemented' )

        self.drop_path = DropPath( config.dropout_layers )

    def forward( self, embeddings, past_key_values, rope_pos, rope_scale ):

        x, past_key_values = self.att(
            self.att_norm( embeddings ),
            past_key_values,
            rope_pos,
            rope_scale
        )

        embeddings = self.drop_path( embeddings, x )

        x = self.ffn( self.ffn_norm( embeddings ) )

        embeddings = self.drop_path( embeddings, x )

        return embeddings, past_key_values

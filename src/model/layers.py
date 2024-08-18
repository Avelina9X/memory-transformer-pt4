"""
Module containing all the layers required for the LSWTransformer architecture.
"""

import torch
from torch.cuda.amp import custom_bwd, custom_fwd

from transformers.activations import ACT2FN

from .configuration import LSWTConfig

try:
    from flash_attn import flash_attn_func # type: ignore # pylint: disable=E0401

    @torch._dynamo.disable # type: ignore # pylint: disable=W0212
    def flash_attention( query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, dropout: float ):
        return flash_attn_func( query, key, value, dropout_p=dropout, causal=True )
    
    attention_func = flash_attention
    
except ModuleNotFoundError:
    def sdpa_attention( query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, dropout: float ):
        assert dropout == 0.0, 'Dropout is not enabled when using fallback SDPA'
        
        # Get full query and key shapes
        q_shape = query.shape
        k_shape = key.shape
        
        # Get length of sequences
        q_len = q_shape[1]
        k_len = k_shape[1]
        
        # Build triagonal causal mask
        mask = torch.ones( q_len, k_len, dtype=query.dtype ).tril( k_len - q_len ).log()
        mask = mask[ None, None, :, : ]
        
        # Get scaling factor
        scale = q_shape[2] ** -0.5
        
        # Compute dot product
        dpa = torch.einsum( 'bqhd,bkhd->bhqk', query, key * scale ) + mask
        
        # Compute softmax
        dps = torch.softmax( dpa, dim=-1 )
        
        # Compute out vector
        out = torch.einsum( 'bkhd,bhqk->bqhd', value, dps )
        
        return out
    
    attention_func = sdpa_attention
    

class ProLU(torch.autograd.Function):
    STE: torch.autograd.Function
    ReLU: torch.autograd.Function

    @staticmethod
    @custom_fwd
    def forward(ctx, m, b): # pylint: disable=W0221
        gate = (m + b > 0) & (m > 0)
        ctx.save_for_backward(m, gate)
        return torch.where(gate, m, 0)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output): # pylint: disable=W0221
        raise NotImplementedError(
            "This method should be overridden by a subclass of ProLU to provide a backward implementation."
        )

class ProLU_ReLU(ProLU): # pylint: disable=W0223
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        _, gate = ctx.saved_tensors
        gated_grad = torch.where(gate, grad_output, 0)
        grad_m, grad_b = gated_grad.clone(), gated_grad.clone()
        return grad_m, grad_b, None

class ProLU_STE(ProLU): # pylint: disable=W0223
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        m, gate = ctx.saved_tensors
        gated_grad = torch.where(gate, grad_output, 0)
        grad_b = gated_grad * m
        grad_m = gated_grad + grad_b.clone()
        return grad_m, grad_b, None

ProLU.STE = ProLU_STE # type: ignore
ProLU.ReLU = ProLU_ReLU # type: ignore

def prolu_ste( m: torch.Tensor, b: torch.Tensor ) -> torch.Tensor:
    out = ProLU_STE.apply( m, b )
    assert out
    return out

def prolu_relu( m: torch.Tensor, b: torch.Tensor ) -> torch.Tensor:
    out = ProLU_ReLU.apply( m, b )
    assert out
    return out


def complex_log( float_input: torch.Tensor, eps=1e-6 ) -> torch.Tensor:
    eps = float_input.new_tensor( eps )
    real = float_input.abs().maximum( eps ).log()
    imag = ( float_input < 0 ).to( float_input.dtype ) * torch.pi
    return torch.complex( real, imag )

@torch._dynamo.disable # type: ignore # pylint: disable=W0212
def complex_scan( segment_mask, token_selected_states, log_beta, segment_pos ):
    numer = torch.where( segment_mask, complex_log( token_selected_states ) - log_beta * segment_pos, -1e9 ).logcumsumexp( -2 )
    denom = torch.where( segment_mask, - log_beta * segment_pos, -1e9 ).logcumsumexp( -2 )
    return ( numer - denom ).exp().real * segment_mask


class _RotateLayerInner( torch.nn.Module ):
    def __init__( self, features ):
        super().__init__()
        
        self.weight = torch.nn.Parameter( torch.empty( features, features ), requires_grad=True )
        torch.nn.init.orthogonal_( self.weight )

    def forward( self, x ):
        return torch.matmul( x, self.weight )

class RotateLayer( torch.nn.Module ):
    def __init__( self, features ):
        super().__init__()

        rotate_layer = _RotateLayerInner( features )
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal( rotate_layer )
    
    def forward( self, x ):
        return self.rotate_layer( x )
    
    def forwardT( self, x ):
        return torch.matmul( x, self.rotate_layer.weight.T )


class RMSHeadNorm( torch.nn.Module ):
    """ RMS Norm layer for query and keys heads.
    """
    
    def __init__( self, d_key: int, n_heads: int, eps: float = 1e-5, learn_scale: bool = True ):
        super().__init__()
        
        self.d_key = d_key
        self.n_heads = n_heads
        self.eps = eps
        
        if learn_scale:
            self.weight = torch.nn.Parameter( torch.empty( size=( n_heads, d_key ) ), requires_grad=True )
            self.weight.data.fill_( 1.0 )
        else:
            self.register_parameter( 'weight', None )
    
    def forward( self, x: torch.Tensor ):
        # Compute RMS per head
        norm = x.square().mean( -1, keepdim=True ).sqrt() + self.eps
        
        # Normalize RMS
        x = x / norm
        
        # Apply scaling if exists
        if self.weight is not None:
            weight = self.weight.to( dtype=x.dtype )
            x = torch.einsum( 'bnld,nd->bnld', x, weight )
        
        return x

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
    
    Creates the RoPE embeddings with support for ABF, and ReRoPE (reversal).
    """
    
    def __init__( self, config: LSWTConfig, ntk_power=2.0 ):
        super().__init__()
        self.config = config
        self.ntk_power = ntk_power

    def forward( self, embeddings, past_key_values ):
        # Get total sequence length including past
        seq_len = embeddings.shape[-2] + ( past_key_values[0][0].shape[-2] if past_key_values is not None else 0 )
        device = embeddings.device
        
        dim = self.config.d_model // self.config.n_heads
        rope_base_freq = self.config.rope_base_freq
        
        if not self.config.rope_dynamic:
            yarn_scale = torch.tensor( 1.0, device=device )
            ntk_scale = self.config.rope_ntk_scale
        else:
            length_scale = max( seq_len / self.config.rope_positions, 1.0 )
            ntk_scale = self.config.rope_ntk_scale * length_scale ** self.ntk_power
            
            a = self.config.rope_yarn_a
            b = self.config.rope_yarn_b
            yarn_scale = torch.tensor( a * torch.math.log( length_scale ) + b, device=device ) # type: ignore
        
        t = torch.arange( seq_len, device=device ).float()
        
        base_freq = rope_base_freq * ntk_scale ** ( dim / ( dim - 2 ) )
        inv_freq = 1.0 / ( base_freq ** ( torch.arange( 0, dim, 2, device=device ).float() / dim ) )
        
        if self.config.rope_reversed:
            t = t.flip( dims=[ 0 ] )

        freqs = torch.einsum( 'i,j->ij', t, inv_freq )
        freqs = torch.cat( ( freqs, freqs ), dim=-1 )

        return freqs, yarn_scale

def _rotate_half( x ):
    x1, x2 = x.chunk( 2, dim=-1 )
    return torch.cat( ( -x2, x1 ), dim=-1 )

def _apply_rotary_pos_emb( pos, t ):
    return ( t * pos.cos() ).type( t.dtype ) + ( _rotate_half( t ) * pos.sin() ).type( t.dtype )

def apply_rope( query, key, rope_pos, yarn_scale ):
    q_length = query.shape[-2]
    k_length = key.shape[-2]

    # Get rotary embeddings for queries and keys
    rope_pos_q = rope_pos[ -q_length : ]
    rope_pos_k = rope_pos[ -k_length : ]

    query = _apply_rotary_pos_emb( rope_pos_q, query ) * yarn_scale.type( query.dtype )
    key = _apply_rotary_pos_emb( rope_pos_k, key ) * yarn_scale.type( key.dtype )

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
        
        if config.qk_norm:
            self.q_norm = RMSHeadNorm( self.key_dim, config.n_heads )
            self.k_norm = RMSHeadNorm( self.key_dim, config.n_heads )

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

    def compute_qkv( self, embeddings, past_key_values ):
        use_kv_cache = ( not self.training ) or ( not self.config.recompute_kv ) 

        # Project qkv and split into heads
        q = self.split_heads( self.proj_q( embeddings ) ).permute( 0, 2, 1, 3 )
        
        # If we're using a kv cache project first
        if use_kv_cache:
            k = self.split_heads( self.proj_k( embeddings ) ).permute( 0, 2, 1, 3 )
            v = self.split_heads( self.proj_v( embeddings ) ).permute( 0, 2, 1, 3 )
            
            # The concat past keys
            if past_key_values:
                k = torch.cat( [ past_key_values[0], k ], dim=-2 )
                v = torch.cat( [ past_key_values[1], v ], dim=-2 )
            
            # Finally save cache
            memory = ( k, v )
        
        # If we're using a state cache, concat first
        else:
            if past_key_values:
                embeddings = torch.cat( [ past_key_values[0][ :, 0, :, : ], embeddings ], dim=-2 )
            
            # The project
            k = self.split_heads( self.proj_k( embeddings ) ).permute( 0, 2, 1, 3 )
            v = self.split_heads( self.proj_v( embeddings ) ).permute( 0, 2, 1, 3 )
            
            # Finally save states
            memory = ( embeddings[ :, None, :, : ], )
        
        return q, k, v, memory

    def forward( self, embeddings, past_key_values, rope_pos, rope_scale ):
        
        q, k, v, memory = self.compute_qkv( embeddings, past_key_values )
        
        # Apply RMS norm
        if self.config.qk_norm:
            q = self.q_norm( q )
            k = self.k_norm( k )

        # Apply rotary embeddings
        q, k = apply_rope( q, k, rope_pos, rope_scale )

        # If we are using registers prepend the keys and values
        if self.config.n_registers > 0:
            batch_size = k.shape[0]

            r_k = ( self.registers_k * rope_scale ).to( dtype=k.dtype ).repeat( batch_size, 1, 1, 1 )
            r_v = ( self.registers_v * rope_scale ).to( dtype=v.dtype ).repeat( batch_size, 1, 1, 1 )

            k = torch.cat( [ r_k, k ], dim=-2 )
            v = torch.cat( [ r_v, v ], dim=-2 )

        # Permute back to normal dimension order
        q = q.permute( 0, 2, 1, 3 )
        k = k.permute( 0, 2, 1, 3 )
        v = v.permute( 0, 2, 1, 3 )

        # Do attention
        a = attention_func( q, k, v, self.att_dropout_p )
        a = self.merge_heads( a )

        # Apply dropout
        o = self.out_dropout( self.proj_o( a ) )

        return o, memory


class ActGLU( torch.nn.Module ):
    def __init__( self, activation: str ):
        super().__init__()
        
        self.activation = ACT2FN[activation]
    
    def forward( self, x ):
        x, g = x.chunk( 2, dim=-1 )
        return x * self.activation( g )

class LSWTFeedForward( torch.nn.Module ):
    """ Feedforward MLP with SwiGLU support
    """
    
    def __init__( self, config: LSWTConfig ):
        super().__init__()

        self.config = config

        self.fc1 = torch.nn.Linear( config.d_model, config.d_ffn * ( 2 if config.gated_ffn else 1 ) )
        self.fc2 = torch.nn.Linear( config.d_ffn, config.d_model )
        self.act = ActGLU( config.hidden_act ) if config.gated_ffn else ACT2FN[config.hidden_act]

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

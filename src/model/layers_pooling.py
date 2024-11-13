from operator import itemgetter
from typing import Literal
import math

import torch
import torch.nn.functional as F

from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention

from .configuration import LSWTConfig, LSWTPoolerConfig
from .layers import LSWTFeedForward

def _group_cumsum( input: torch.Tensor, reset_mask: torch.Tensor ) -> torch.Tensor:
    # Compute cumsum along sequence dim
    input_sum = input.cumsum( 1 )

    # Subtract input to get the exclusive cumsum
    input_prime = input_sum - input

    # Compute indicies of masked locations and scatter into unmasked locations
    indicies = torch.arange( input.shape[1], device=input.device ) * reset_mask
    indicies = indicies.cummax( 1 )[0].unsqueeze( -1 )

    # Subtract the gathered values to reconstruct the group cumsums
    return input_sum - input_prime.take_along_dim( indicies, 1 )

def group_cumsum( embeddings: torch.Tensor, reset_mask: torch.Tensor ) -> torch.Tensor:
    # Cast to float
    input = embeddings.float()

    # Compute cumsum in fp32
    output = _group_cumsum( input, reset_mask )

    # Cast back to input type and return
    return output.to( embeddings.dtype )

def group_cummean( embeddings: torch.Tensor, reset_mask: torch.Tensor ) -> torch.Tensor:
    # Cast to float
    input = embeddings.float()

    # Create ones and compute group cumsum of segments
    ones = torch.ones_like( reset_mask, dtype=torch.float ).unsqueeze( -1 )
    scale = _group_cumsum( ones, reset_mask )

    # Compute cumsum in fp32 and rescale
    output = _group_cumsum( input, reset_mask ) / scale

    return output.to( embeddings.dtype )

def group_cummean_weighted( embeddings: torch.Tensor, reset_mask: torch.Tensor ) -> torch.Tensor:
    # Cast to float
    input = embeddings.float()

    # Create ones and compute group cumsum of segments
    ones = torch.ones_like( reset_mask, dtype=torch.float ).unsqueeze( -1 )
    scale = _group_cumsum( ones, reset_mask )

    # Compute scaled cumsum in fp32 and rescale
    output = _group_cumsum( input * scale, reset_mask ) / scale

    return output.to( embeddings.dtype )

def group_cumsum_noop( embeddings: torch.Tensor, reset_mask: torch.Tensor ) -> torch.Tensor:
    return embeddings


class AdaBaseLayer( torch.nn.Module ):
    ...

class AdaLinear( AdaBaseLayer ):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        repeats: int = 0,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.repeats = repeats
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        # Create base layer
        self.linear = torch.nn.Linear( in_features, out_features, use_bias )

        # Create dropout layer
        self.dropout = torch.nn.Dropout( lora_dropout )

        # Compute scale amount
        self.scale = 0 if r == 0 else lora_alpha / math.sqrt( r )

        # If rank > 0 create LoRA lists
        if r > 0 and repeats > 0:
            self.lora_A_list = torch.nn.ParameterList()
            self.lora_B_list = torch.nn.ParameterList()

            for _ in range( repeats ):
                lora_A = torch.nn.Parameter( torch.randn( r, in_features ) / r, True )
                lora_B = torch.nn.Parameter( torch.zeros( out_features, r ), True )

                self.lora_A_list.append( lora_A )
                self.lora_B_list.append( lora_B )
        else:
            self.lora_A_list = None
            self.lora_B_list = None

        # If bias is enabled create bias list
        if use_bias and repeats > 0:
            self.bias_list = torch.nn.ParameterList()

            for _ in range( repeats ):
                bias = torch.nn.Parameter( torch.zeros( out_features ), True )
                self.bias_list.append( bias )
        else:
            self.bias_list = None

    def forward( self, inputs: torch.Tensor, index: int ):
        # Compute base value
        result = self.linear( inputs )

        if self.lora_A_list is not None and self.lora_B_list is not None:
            # Get current LoRA params for this index
            lora_A = self.lora_A_list[ index ]
            lora_B = self.lora_B_list[ index ]

            # Compute delta
            d = self.dropout( inputs )
            d = F.linear( d, lora_A ) # pylint: disable=E1102
            d = F.linear( d, lora_B ) * self.scale # pylint: disable=E1102

            # Add delta to result
            result = result + d

        if self.bias_list:
            # Get current bias for this index
            b = self.bias_list[ index ]

            # Add bias to result
            result = result + b

        return result

class AdaRegister( AdaBaseLayer ):
    def __init__(
        self,
        n_heads: int,
        d_key: int,
        repeats: int = 0
    ):
        super().__init__()

        self.n_heads = n_heads
        self.d_key = d_key

        self.register = torch.nn.Parameter( torch.zeros( 1, n_heads, 1, d_key ), True )

        if repeats > 0:
            self.bias_list = torch.nn.ParameterList()

            for _ in range( repeats ):
                bias = torch.nn.Parameter( torch.zeros( 1, n_heads, 1, d_key ), True )
                self.bias_list.append( bias )
        else:
            self.bias_list = None


    def forward( self, inputs: torch.Tensor, index: int ):
        # Compute base value
        result = self.register

        if self.bias_list:
            # Get current bias for this index
            b = self.bias_list[ index ]

            # Add bias to result
            result = result + b

        return result.repeat( inputs.shape[0], 1, inputs.shape[1], 1 )


class AdaBaseModule( torch.nn.Module ):
    ...

    def forward( self, inputs: torch.Tensor, index: int ):
        raise NotImplementedError()

class AdaPoolAttention( AdaBaseModule ):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_key: int,
        repeats: int = 0,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_key = d_key

        self.repeats = repeats
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        self.key_scale = self.d_key ** -0.5

        self.q_regs = AdaRegister( n_heads, d_key, repeats )
        self.k_proj = AdaLinear( d_model, d_model, True, repeats, r, lora_alpha, lora_dropout )
        self.v_proj = AdaLinear( d_model, d_model, True, repeats, r, lora_alpha, lora_dropout )
        self.o_proj = AdaLinear( d_model, d_model, True, repeats, r, lora_alpha, lora_dropout )

    def forward( self, inputs: torch.Tensor, index: int, mask: torch.Tensor ):
        # Compute qkv projections
        q = self.q_regs( inputs, index )
        k = self.k_proj( inputs, index ).unflatten( -1, [ self.n_heads, self.d_key ] )
        v = self.v_proj( inputs, index ).unflatten( -1, [ self.n_heads, self.d_key ] )

        o = scaled_dot_product_attention( # pylint: disable=E1102
            q,
            k.transpose( 1, 2 ),
            v.transpose( 1, 2 ),
            attn_mask=mask,
            scale=self.key_scale,
        ).transpose( 2, 1 ).flatten( start_dim=2 )

        # Perform output projection and return
        return self.o_proj( o, index )

class AdaAttention( AdaBaseModule ):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_key: int,
        repeats: int = 0,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_key = d_key

        self.repeats = repeats
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        self.key_scale = self.d_key ** -0.5

        self.q_proj = AdaLinear( d_model, d_model, True, repeats, r, lora_alpha, lora_dropout )
        self.k_proj = AdaLinear( d_model, d_model, True, repeats, r, lora_alpha, lora_dropout )
        self.v_proj = AdaLinear( d_model, d_model, True, repeats, r, lora_alpha, lora_dropout )
        self.o_proj = AdaLinear( d_model, d_model, True, repeats, r, lora_alpha, lora_dropout )

    def forward( self, inputs: torch.Tensor, index: int, mask: torch.Tensor ):
        # Compute qkv projections
        q = self.q_proj( inputs, index ).unflatten( -1, [ self.n_heads, self.d_key ] )
        k = self.k_proj( inputs, index ).unflatten( -1, [ self.n_heads, self.d_key ] )
        v = self.v_proj( inputs, index ).unflatten( -1, [ self.n_heads, self.d_key ] )

        o = scaled_dot_product_attention( # pylint: disable=E1102
            q.transpose( 1, 2 ),
            k.transpose( 1, 2 ),
            v.transpose( 1, 2 ),
            attn_mask=mask,
            scale=self.key_scale,
        ).transpose( 2, 1 ).flatten( start_dim=2 )

        # Perform output projection and return
        return self.o_proj( o, index )

class AdaFeedForward( AdaBaseModule ):
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        repeats: int = 0,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ffn = d_ffn

        self.repeats = repeats
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        self.up_proj = AdaLinear( d_model, d_ffn, True, repeats, r, lora_alpha, lora_dropout )
        self.gate_proj = AdaLinear( d_model, d_ffn, True, repeats, r, lora_alpha, lora_dropout )
        self.down_proj = AdaLinear( d_ffn, d_model, True, repeats, r, lora_alpha, lora_dropout )

    def forward( self, inputs: torch.Tensor, index: int ):
        up = self.up_proj( inputs, index )
        gate = self.gate_proj( inputs, index )

        x = up * F.silu( gate )

        return self.down_proj( x, index )


class LSWTLayerPoolerSingle( torch.nn.Module ):
    def __init__( self, pooler_config: LSWTPoolerConfig ):
        super().__init__()
        
        assert pooler_config.layer_pooling == 'layer'
        assert isinstance( pooler_config.layer_pooling_select, int )
        
        self.layer_idx = pooler_config.layer_pooling_select
    
    def forward( self, hidden_states: tuple[torch.Tensor] ) -> torch.Tensor:
        return hidden_states[self.layer_idx]


class LSWTLayerPoolerWeighted( torch.nn.Module ):
    def __init__( self, pooler_config: LSWTPoolerConfig ):
        super().__init__()
        
        assert pooler_config.layer_pooling == 'weighted_sum'
        assert not isinstance( pooler_config.layer_pooling_select, int )
        
        self.layer_idx = pooler_config.layer_pooling_select
                
        self.dropout = torch.nn.Dropout( p=pooler_config.layer_pooling_dropout, inplace=True )

        self.layer_weighting = torch.nn.Parameter( torch.ones( len( self.layer_idx ), 1, 1, 1 ), requires_grad=True )
    
    def forward( self, hidden_states: tuple[torch.Tensor] ) -> torch.Tensor:
        # states = torch.stack( itemgetter( *self.layer_idx )( hidden_states ) )
        
        states = torch.stack(
            [ state for i, state in enumerate( hidden_states ) if i in self.layer_idx ]
        ) # TODO: check for negative references
            
        drop_mask = states.new_ones( [ states.size( 0 ), states.size( 1 ), 1, 1 ] )
        drop_mask = self.dropout( drop_mask )
        
        return ( self.layer_weighting.softmax( 0 ) * drop_mask * states ).sum( 0 )


class LSWTTokenPoolerCLS( torch.nn.Module ):
    def __init__( self, pooler_config: LSWTPoolerConfig ):
        super().__init__()
        
        assert pooler_config.token_pooling == 'cls'
        
        self.cls_token_id = int( pooler_config.token_pooling_config[ 'cls_token_id' ] )
    
    def forward( self, layer_states: torch.Tensor, input_ids: torch.Tensor ) -> torch.Tensor:       
        return layer_states


class LSWTTokenPoolerAttention( torch.nn.Module ):
    def __init__( self, pooler_config: LSWTPoolerConfig, base_config: LSWTConfig ):
        super().__init__()
        
        assert pooler_config.token_pooling == 'attn'
        
        self.pooler_config = pooler_config
        self.base_config = base_config
        
        self.sep_token_id = int( pooler_config.token_pooling_config[ 'sep_token_id' ] )
        self.cls_token_id = int( pooler_config.token_pooling_config[ 'cls_token_id' ] )
        self.new_token_id = int( pooler_config.token_pooling_config[ 'new_token_id' ] )
        self.pad_token_id = int( pooler_config.token_pooling_config[ 'pad_token_id' ] )
        
        self.self_type = str( pooler_config.token_pooling_config[ 'self_type' ] )

        self.n_repeats = int( pooler_config.token_pooling_config[ 'n_repeats' ] )
        self.lora_r = int( pooler_config.token_pooling_config[ 'lora_r' ] )
        self.lora_alpha = float( pooler_config.token_pooling_config[ 'lora_alpha' ] )
        self.lora_dropout = float( pooler_config.token_pooling_config[ 'lora_dropout' ] )
        
        self.d_model = base_config.d_model
        self.n_heads = base_config.n_heads
        self.d_key = self.d_model // self.n_heads
        self.d_ffn = base_config.d_ffn
        
        self.alibi_slope = float( pooler_config.token_pooling_config[ 'alibi_slope' ] )
        
        head_scale = torch.exp2( -( ( torch.arange( self.n_heads ) + 1.0 ) * self.alibi_slope / self.n_heads ) )
        self.head_scale = torch.nn.Parameter( head_scale, requires_grad=False )
        
        match self.self_type:
            case 'self':
                self.self_layer = AdaAttention( self.d_model, self.n_heads, self.d_key, self.n_repeats, self.lora_r, self.lora_alpha, self.lora_dropout )
            case 'pool':
                self.self_layer = AdaPoolAttention( self.d_model, self.n_heads, self.d_key, self.n_repeats, self.lora_r, self.lora_alpha, self.lora_dropout )
            case _:
                raise ValueError( f'Invalid `self_type` in pooler: {self.self_type}' )
        
        self.cross_layer = AdaAttention( self.d_model, self.n_heads, self.d_key, self.n_repeats, self.lora_r, self.lora_alpha, self.lora_dropout )
        self.ffn_layer = AdaFeedForward( self.d_model, self.d_ffn, self.n_repeats, self.lora_r, self.lora_alpha, self.lora_dropout )
        
        self.self_norm = torch.nn.ModuleList()
        self.cross_norm = torch.nn.ModuleList()
        self.ffn_norm = torch.nn.ModuleList()
        
        for _ in range( self.n_repeats ):
            self.self_norm.append( torch.nn.LayerNorm( self.d_model ) )
            self.cross_norm.append( torch.nn.LayerNorm( self.d_model ) )
            self.ffn_norm.append( torch.nn.LayerNorm( self.d_model ) )
    
    def compute_bias_mask_self( self, segment_ids: torch.Tensor, segment_mask: torch.Tensor ) -> torch.Tensor:
        seq_len = segment_ids.shape[-1]
        seq_ids = torch.arange( seq_len, dtype=torch.float32, device=segment_ids.device )
        
        seg_mask = segment_ids[ :, :, None ] == segment_ids[ :, None, : ]
        val_mask = segment_mask[ :, :, None ] * segment_mask[ :, None, : ]
        diagonal = torch.eye( seq_len, dtype=torch.bool, device=segment_ids.device )[ None, :, : ]
        
        bias = seq_ids[ None, None, : ] - seq_ids[ None, :, None ]
        mask = ( seg_mask * val_mask + diagonal ).tril()
        
        bias_mask = bias.where( mask, float( '-inf' ) )
        
        bias_mask = bias_mask[ :, None, :, : ] * self.head_scale[ None, :, None, None ]
        
        return bias_mask
        
    def compute_bias_mask_cross( self, segment_ids: torch.Tensor, class_mask: torch.Tensor ) -> torch.Tensor:
        seq_len = segment_ids.shape[-1]
        
        cls_mask = class_mask[ :, None, : ]
        diagonal = torch.eye( seq_len, dtype=torch.bool, device=segment_ids.device )[ None, :, : ]
        
        bias = ( segment_ids[ :, None, : ] - segment_ids[ :, :, None ] ).float()
        mask = ( cls_mask + diagonal ).tril()
        
        bias_mask = bias.where( mask, float( '-inf' ) )
        
        bias_mask = bias_mask[ :, None, :, : ] * self.head_scale[ None, :, None, None ]
        
        return bias_mask
    
    def forward( self, states: torch.Tensor, input_ids: torch.Tensor ) -> torch.Tensor:
        class_mask = input_ids == self.cls_token_id
        segment_mask = input_ids != self.pad_token_id
        segment_ids = ( input_ids == self.sep_token_id ).int().cumsum( -1 )
        
        self_mask = self.compute_bias_mask_self( segment_ids, segment_mask )
        cross_mask = self.compute_bias_mask_cross( segment_ids, class_mask )
        
        for i in range( self.n_repeats ):
            states = states + self.self_layer( self.self_norm[i]( states ), i, self_mask )
            states = states + self.cross_layer( self.cross_norm[i]( states ), i, cross_mask )
            states = states + self.ffn_layer( self.ffn_norm[i]( states ), i )
        
        return states


class LSWTEmbeddingPooler( torch.nn.Module ):
    def __init__( self, pooler_config: LSWTPoolerConfig ):
        super().__init__()
        
        self.cls_token_id = int( pooler_config.token_pooling_config[ 'cls_token_id' ] )
        
        match pooler_config.embedding_pooling:
            case 'sgpt':
                self.func = group_cummean_weighted
            case None:
                self.func = group_cumsum_noop
            case _:
                raise ValueError( f'`{pooler_config.embedding_pooling}` is not a valid value for pooler_config.embedding_pooling' )
        
    def forward( self, embeddings: torch.Tensor, input_ids: torch.Tensor, return_final: bool ) -> torch.Tensor:
        
        reset_mask = input_ids == self.cls_token_id
        
        embeddings = self.func( embeddings, reset_mask )
        
        # If return final, select the final CLS token
        if return_final:
            batch_ids = torch.arange( input_ids.shape[0], device=embeddings.device )
            seq_ids = torch.arange( input_ids.shape[1], device=embeddings.device )
            end_idx = torch.where( reset_mask, seq_ids, -1 ).max( -1 )[0]
            
            return embeddings[ batch_ids, end_idx ]
        else:
            return embeddings
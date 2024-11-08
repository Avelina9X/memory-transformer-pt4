from operator import itemgetter
from typing import Literal

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


class _AttentionBase( torch.nn.Module, ):
    def __init__( self, d_model: int, n_heads: int, d_key: int, alibi_slope: float, head_gate: bool ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_key = d_key
        self.alibi_slope = alibi_slope
        self.head_gate = head_gate
        
        self.key_scale = self.d_key ** -0.5
        
        head_scale = torch.exp2( -( ( torch.arange( n_heads ) + 1.0 ) * alibi_slope / n_heads ) )
        self.head_scale = torch.nn.Parameter( head_scale, requires_grad=False )
    
    def mask_type( self ) -> Literal['self', 'cross']:
        raise NotImplementedError()
    
    def forward( self, states: torch.Tensor, bias_mask: torch.Tensor ) -> torch.Tensor:
        raise NotImplementedError()


class _AttentionSelf( _AttentionBase ):
    def __init__( self, d_model: int, n_heads: int, d_key: int, alibi_slope: float, head_gate: bool ):
        super().__init__( d_model, n_heads, d_key, alibi_slope, head_gate )
        
        self.g_proj = torch.nn.Linear( d_model, n_heads, bias=True ) if head_gate else None
        self.q_proj = torch.nn.Linear( d_model, d_model, bias=True )
        self.k_proj = torch.nn.Linear( d_model, d_model, bias=True )
        self.v_proj = torch.nn.Linear( d_model, d_model, bias=True )
        self.o_proj = torch.nn.Linear( d_model, d_model, bias=True )
    
    def mask_type( self ) -> Literal['self', 'cross']:
        return 'self'
    
    def forward( self, states: torch.Tensor, bias_mask: torch.Tensor ) -> torch.Tensor:
        # Scale mask by heads
        bias_mask = bias_mask[ :, None, :, : ] * self.head_scale[ None, :, None, None ]
        
        # Compute qkv projections
        q = self.q_proj( states ).unflatten( -1, [ self.n_heads, self.d_key ] )
        k = self.k_proj( states ).unflatten( -1, [ self.n_heads, self.d_key ] )
        v = self.v_proj( states ).unflatten( -1, [ self.n_heads, self.d_key ] )
        
        o = scaled_dot_product_attention( # pylint: disable=E1102
            q.transpose( 1, 2 ),
            k.transpose( 1, 2 ),
            v.transpose( 1, 2 ),
            attn_mask=bias_mask,
            scale=self.key_scale,
        ).transpose( 2, 1 )
        
        # Apply gate if present
        if self.g_proj:
            g = self.g_proj( states )
            o = o * g.sigmoid().unsqueeze( -1 )
        
        # Perform output projection and return
        return self.o_proj( o.flatten( start_dim=2 ) )

class _AttentionPool( _AttentionBase ):
    def __init__( self, d_model: int, n_heads: int, d_key: int, alibi_slope: float, head_gate: bool ):
        super().__init__( d_model, n_heads, d_key, alibi_slope, head_gate )
        
        self.g_proj = torch.nn.Linear( d_model, n_heads, bias=True ) if head_gate else None
        self.q_regs = torch.nn.Parameter( torch.zeros( 1, n_heads, 1, d_key ), requires_grad=True )
        self.k_proj = torch.nn.Linear( d_model, d_model, bias=True )
        self.v_proj = torch.nn.Linear( d_model, d_model, bias=True )
        self.o_proj = torch.nn.Linear( d_model, d_model, bias=True )
    
    def mask_type( self ) -> Literal['self', 'cross']:
        return 'self'
    
    def forward( self, states: torch.Tensor, bias_mask: torch.Tensor ) -> torch.Tensor:
        # Scale mask by heads
        bias_mask = bias_mask[ :, None, :, : ] * self.head_scale[ None, :, None, None ]
        
        # Compute qkv projections
        q = self.q_regs.repeat( states.shape[0], 1, states.shape[1], 1 )
        k = self.k_proj( states ).unflatten( -1, [ self.n_heads, self.d_key ] )
        v = self.v_proj( states ).unflatten( -1, [ self.n_heads, self.d_key ] )
        
        o = scaled_dot_product_attention( # pylint: disable=E1102
            q,
            k.transpose( 1, 2 ),
            v.transpose( 1, 2 ),
            attn_mask=bias_mask,
            scale=self.key_scale,
        ).transpose( 2, 1 )
        
        # Apply gate if present
        if self.g_proj:
            g = self.g_proj( states )
            o = o * g.sigmoid().unsqueeze( -1 )
        
        # Perform output projection and return
        return self.o_proj( o.flatten( start_dim=2 ) )


class _AttentionPoolLinear( _AttentionBase ):
    def __init__( self, d_model: int, n_heads: int, d_key: int, alibi_slope: float, head_gate: bool ):
        super().__init__( d_model, n_heads, d_key, alibi_slope, head_gate )
        
        self.g_proj = torch.nn.Linear( d_model, n_heads, bias=True ) if head_gate else None
        self.a_proj = torch.nn.Linear( d_model, n_heads, bias=False )
        self.v_proj = torch.nn.Linear( d_model, d_model, bias=True )
        self.o_proj = torch.nn.Linear( d_model, d_model, bias=True )
    
    def mask_type( self ) -> Literal['self', 'cross']:
        return 'self'
    
    def forward( self, states: torch.Tensor, bias_mask: torch.Tensor ) -> torch.Tensor:
        # Scale mask by heads
        bias_mask = bias_mask[ :, None, :, : ] * self.head_scale[ None, :, None, None ]
        
        # Compute v projection
        v = self.v_proj( states ).unflatten( -1, [ self.n_heads, self.d_key ] )
        
        # Calculate linear attention matrix
        a = self.a_proj( states ).permute( 0, 2, 1 )[ :, :, None, : ] + bias_mask
        a = a.softmax( -1 )
        
        # Aggregate values
        o = torch.einsum( 'bhqk,bkhd->bqhd', a, v )
        
        # Apply gate if present
        if self.g_proj:
            g = self.g_proj( states )
            o = o * g.sigmoid().unsqueeze( -1 )
        
        # Perform output projection and return
        return self.o_proj( o.flatten( start_dim=2 ) )


class _AttentionCross( _AttentionBase ):
    def __init__( self, d_model: int, n_heads: int, d_key: int, alibi_slope: float, head_gate: bool ):
        super().__init__( d_model, n_heads, d_key, alibi_slope, head_gate )
        
        self.g_proj = torch.nn.Linear( d_model, n_heads, bias=True ) if head_gate else None
        self.q_proj = torch.nn.Linear( d_model, d_model, bias=True )
        self.k_proj = torch.nn.Linear( d_model, d_model, bias=True )
        self.v_proj = torch.nn.Linear( d_model, d_model, bias=True )
        self.o_proj = torch.nn.Linear( d_model, d_model, bias=True )
    
    def mask_type( self ) -> Literal['self', 'cross']:
        return 'cross'
    
    def forward( self, states: torch.Tensor, bias_mask: torch.Tensor ) -> torch.Tensor:
        # Scale mask by heads
        bias_mask = bias_mask[ :, None, :, : ] * self.head_scale[ None, :, None, None ]
        
        # Compute qkv projections
        q = self.q_proj( states ).unflatten( -1, [ self.n_heads, self.d_key ] )
        k = self.k_proj( states ).unflatten( -1, [ self.n_heads, self.d_key ] )
        v = self.v_proj( states ).unflatten( -1, [ self.n_heads, self.d_key ] )
        
        o = scaled_dot_product_attention( # pylint: disable=E1102
            q.transpose( 1, 2 ),
            k.transpose( 1, 2 ),
            v.transpose( 1, 2 ),
            attn_mask=bias_mask,
            scale=self.key_scale,
        ).transpose( 2, 1 )
        
        # Apply gate if present
        if self.g_proj:
            g = self.g_proj( states )
            o = o * g.sigmoid().unsqueeze( -1 )
        
        # Perform output projection and return
        return self.o_proj( o.flatten( start_dim=2 ) )


class _FFN( LSWTFeedForward ):
    def mask_type( self ) -> Literal['ffn']:
        return 'ffn'
    
    def forward( self, states: torch.Tensor, bias_mask=None ) -> torch.Tensor:
        return super().forward( states )


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

        self.layer_weighting = torch.nn.Parameter( torch.zeros( len( self.layer_idx ), 1, 1, 1 ), requires_grad=True )
    
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
        
        self.layer_names = list( pooler_config.token_pooling_config[ 'attn_layers' ] )
        self.n_repeats = int( pooler_config.token_pooling_config[ 'n_repeats' ] )
        self.include_prefix = bool( pooler_config.token_pooling_config[ 'include_prefix' ] )
        
        self.d_model = base_config.d_model
        self.n_heads = base_config.n_heads
        self.d_key = self.d_model // self.n_heads
        
        self.alibi_slope = float( pooler_config.token_pooling_config[ 'alibi_slope' ] )
        self.head_gate = bool( pooler_config.token_pooling_config[ 'head_gate' ] )
        
        norm = str( pooler_config.token_pooling_config[ 'norm' ] )
        pre_norm = norm in [ 'pre', 'both' ]
        post_norm = norm in [ 'post', 'both' ]
        
        self.unskip = bool( pooler_config.token_pooling_config[ 'unskip' ] )
        
        self.attn_layers = torch.nn.ModuleList()
        self.norm_layers_pre = torch.nn.ModuleList()
        self.norm_layers_post = torch.nn.ModuleList()
        
        self.n_layers_attn = len( self.layer_names )
        self.n_layers_total = len( self.layer_names ) * self.n_repeats
        
        for attn_type in self.layer_names:
            self.attn_layers.append( self._attention_factory( attn_type ) )
        
        for _ in range( self.n_layers_total ):
            self.norm_layers_pre.append( torch.nn.LayerNorm( self.d_model ) if pre_norm else torch.nn.Identity() )
            self.norm_layers_post.append( torch.nn.LayerNorm( self.d_model ) if post_norm else torch.nn.Identity() )
        
    
    def _attention_factory(
        self,
        attn_type: Literal['self', 'pool', 'pool_linear', 'cross', 'ffn'],
    ) -> _AttentionBase | torch.nn.Module:
        match attn_type:
            case 'self':
                cls = _AttentionSelf
            case 'pool':
                cls = _AttentionPool
            case 'pool_linear':
                cls = _AttentionPoolLinear
            case 'cross':
                cls = _AttentionCross
            case 'ffn':
                return _FFN( self.base_config )
        return cls(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_key=self.d_key,
            alibi_slope=self.alibi_slope,
            head_gate=self.head_gate,
        )
    
    @torch._dynamo.disable # type: ignore # pylint: disable=W0212
    def compute_segments( self, tokens: torch.Tensor ):
        # Tensors of current state and id for each sequnce in the batch
        current_state = torch.zeros( [ tokens.shape[0] ], dtype=torch.int, device=tokens.device )
        current_seg_id = torch.zeros( [ tokens.shape[0] ], dtype=torch.int, device=tokens.device )

        # Tensors of all states and ids for each element of each sequence in the batch
        states = torch.zeros_like( tokens, dtype=torch.int )
        seg_ids = torch.zeros_like( tokens, dtype=torch.int )

        # Get idxs of start, end and newline
        im_start_arr = tokens == self.sep_token_id
        im_end_arr = tokens == self.cls_token_id 
        newline_arr = tokens == self.new_token_id

        # Loop over all tokens
        for i in range( tokens.shape[-1] ):
            # If token is <|im_start|>
            im_start = im_start_arr[ :, i ]

            # If token is <|im_end|>
            im_end = im_end_arr[ :, i ]

            # If token is \n # TODO: check for multiple types of newline perhaps?
            newline = newline_arr[ :, i ]

            # 4->0 if \n
            current_state.masked_fill_( ( current_state == 4 ).logical_and( newline ), 0 )

            # 3->4 if im_end
            current_state.masked_fill_( ( current_state == 3 ).logical_and( im_end ), 4 )

            # 2->3 if anything
            current_state.masked_fill_( ( current_state == 2 ), 3 )

            # 1->2 if \n
            current_state.masked_fill_( ( current_state == 1 ).logical_and( newline ), 2 )

            # 0->1 if im_start
            current_state.masked_fill_( ( current_state == 0 ).logical_and( im_start ), 1 )

            # If im_start is seen increment seg_id
            current_seg_id += ( im_start ).int()

            states[ :, i ] = current_state
            seg_ids[ :, i ] = current_seg_id
        
        segment_mask = torch.isin( states, torch.tensor( [ 1, 2, 3, 4 ] if self.include_prefix else [ 3, 4 ], device=states.device ) )
        class_mask = im_end_arr
        
        return states, seg_ids, segment_mask, class_mask
    
    def compute_bias_mask_self( self, segment_ids: torch.Tensor, segment_mask: torch.Tensor ) -> torch.Tensor:
        seq_len = segment_ids.shape[-1]
        seq_ids = torch.arange( seq_len, dtype=torch.float32, device=segment_ids.device )
        
        seg_mask = segment_ids[ :, :, None ] == segment_ids[ :, None, : ]
        val_mask = segment_mask[ :, :, None ] * segment_mask[ :, None, : ]
        diagonal = torch.eye( seq_len, dtype=torch.bool, device=segment_ids.device )[ None, :, : ]
        
        bias = seq_ids[ None, None, : ] - seq_ids[ None, :, None ]
        mask = ( seg_mask * val_mask + diagonal ).tril()
        
        return bias.where( mask, float( '-inf' ) )
        
    def compute_bias_mask_cross( self, segment_ids: torch.Tensor, class_mask: torch.Tensor ) -> torch.Tensor:
        seq_len = segment_ids.shape[-1]
        
        cls_mask = class_mask[ :, None, : ]
        diagonal = torch.eye( seq_len, dtype=torch.bool, device=segment_ids.device )[ None, :, : ]
        
        bias = ( segment_ids[ :, None, : ] - segment_ids[ :, :, None ] ).float()
        mask = ( cls_mask + diagonal ).tril()
        
        return bias.where( mask, float( '-inf' ) )
    
    def forward( self, states: torch.Tensor, input_ids: torch.Tensor ) -> torch.Tensor:
        if self.include_prefix:
            class_mask = input_ids == self.cls_token_id
            segment_mask = input_ids != self.pad_token_id
            segment_ids = ( input_ids == self.sep_token_id ).int().cumsum( -1 )
        else:
            state_ids, segment_ids, segment_mask, class_mask = self.compute_segments( input_ids )
        
        bias_mask_map = {
            'self': self.compute_bias_mask_self( segment_ids, segment_mask ),
            'cross': self.compute_bias_mask_cross( segment_ids, class_mask ),
            'ffn': None,
        }
        
        skip = states
        
        for i in range( self.n_layers_total ):
            attn_layer = self.attn_layers[i % self.n_layers_attn]
            norm_layer_pre = self.norm_layers_pre[i]
            norm_layer_post = self.norm_layers_post[i]
            
            assert isinstance( attn_layer, torch.nn.Module )
            assert isinstance( norm_layer_pre, torch.nn.Module )
            assert isinstance( norm_layer_post, torch.nn.Module )
            
            # Pre-norm the states
            normed_state = norm_layer_pre( states )
            
            # Get the mask type
            mask_type = attn_layer.mask_type()
            
            # Select the correct bias mask
            bias_mask = bias_mask_map[ mask_type ]
            
            # Get the residual stream
            residual_states = attn_layer( normed_state, bias_mask )
            
            # Post norm the residual
            normed_residual_states = norm_layer_post( residual_states )
            
            # Add residual to skip connection
            states = states + normed_residual_states
        
        if self.unskip:
            states = states - skip
        
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
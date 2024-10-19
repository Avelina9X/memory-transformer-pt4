from operator import itemgetter
from typing import Literal

import torch
import torch.nn.functional as F

from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention

from .configuration import LSWTConfig, LSWTPoolerConfig

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
        
        # # Calculate attention matrix
        # a = torch.einsum( 'bqhd,bkhd->bhqk', q, k ) * self.key_scale + bias_mask
        # a = a.softmax( -1 )
        
        # # Aggregate values
        # o = torch.einsum( 'bhqk,bkhd->bqhd', a, v )
        
        o = scaled_dot_product_attention(
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
        
        # # Calculate attention matrix
        # a = torch.einsum( 'bqhd,bkhd->bhqk', q, k ) * self.key_scale + bias_mask
        # a = a.softmax( -1 )
        
        # # Aggregate values
        # o = torch.einsum( 'bhqk,bkhd->bqhd', a, v )
        
        o = scaled_dot_product_attention(
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
    
    def forward( self, layer_states: torch.Tensor, input_ids: torch.Tensor, return_final: bool ) -> torch.Tensor:
        assert return_final, 'Can only return final using CLS pooler'
        
        batch_ids = torch.arange( input_ids.shape[0], device=layer_states.device )
        seq_ids = torch.arange( input_ids.shape[1], device=layer_states.device )
        end_idx = torch.where( input_ids == self.cls_token_id, seq_ids, -1 ).max( -1 )[0]
        
        return layer_states[ batch_ids, end_idx ]

class LSWTTokenPoolerAttention( torch.nn.Module ):
    def __init__( self, pooler_config: LSWTPoolerConfig, base_config: LSWTConfig ):
        super().__init__()
        
        assert pooler_config.token_pooling == 'attn'
        
        self.sep_token_id = int( pooler_config.token_pooling_config[ 'sep_token_id' ] )
        self.cls_token_id = int( pooler_config.token_pooling_config[ 'cls_token_id' ] )
        self.new_token_id = int( pooler_config.token_pooling_config[ 'new_token_id' ] )
        self.pad_token_id = int( pooler_config.token_pooling_config[ 'pad_token_id' ] )
        
        self.layer_names = list( pooler_config.token_pooling_config[ 'attn_layers' ] )
        self.include_prefix = bool( pooler_config.token_pooling_config[ 'include_prefix' ] )
        
        self.d_model = base_config.d_model
        self.n_heads = base_config.n_heads
        self.d_key = self.d_model // self.n_heads
        
        self.alibi_slope = float( pooler_config.token_pooling_config[ 'alibi_slope' ] )
        self.head_gate = bool( pooler_config.token_pooling_config[ 'head_gate' ] )
        
        self.attn_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        
        for attn_type in self.layer_names:
            self.norm_layers.append( torch.nn.LayerNorm( self.d_model ) )
            self.attn_layers.append( self._attention_factory( attn_type ) )
    
    def _attention_factory(
        self,
        attn_type: Literal['self', 'pool', 'cross'],
    ) -> _AttentionBase:
        match attn_type:
            case 'self':
                cls = _AttentionSelf
            case 'pool':
                cls = _AttentionPool
            case 'cross':
                cls = _AttentionCross
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
    
    def forward( self, states: torch.Tensor, input_ids: torch.Tensor, return_final: bool ) -> torch.Tensor:
        if self.include_prefix:
            class_mask = input_ids == self.cls_token_id
            segment_mask = input_ids != self.pad_token_id
            segment_ids = ( input_ids == self.sep_token_id ).int().cumsum( -1 )
        else:
            state_ids, segment_ids, segment_mask, class_mask = self.compute_segments( input_ids )
        
        bias_mask_map = {
            'self': self.compute_bias_mask_self( segment_ids, segment_mask ),
            'cross': self.compute_bias_mask_cross( segment_ids, class_mask ),
        }
        
        for attn_layer, norm_layer in zip( self.attn_layers, self.norm_layers ):
            assert isinstance( attn_layer, _AttentionBase )
            assert isinstance( norm_layer, torch.nn.LayerNorm )
            
            # Pre-norm the states
            normed_state = norm_layer( states )
            
            # Get the mask type
            mask_type = attn_layer.mask_type()
            
            # Select the correct bias mask
            bias_mask = bias_mask_map[ mask_type ]
            
            # Get the residual stream
            residual_states = attn_layer( normed_state, bias_mask )
            
            # Add residual to skip connection
            states = states + residual_states
        
        # Return final cls states if that's what we want
        if return_final:
            batch_ids = torch.arange( input_ids.shape[0], device=states.device )
            seq_ids = torch.arange( input_ids.shape[1], device=states.device )
            end_idx = torch.where( class_mask, seq_ids, -1 ).max( -1 )[0]
            return states[ batch_ids, end_idx ]
        
        # Otherwise return all states
        else:
            return states
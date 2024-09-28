from operator import itemgetter

import torch
import torch.nn.functional as F

from .configuration import LSWTConfig, LSWTPoolerConfig


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
        states = torch.stack( itemgetter( *self.layer_idx )( hidden_states ) )
            
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
        
        self.cls_token_id = int( pooler_config.token_pooling_config[ 'cls_token_id' ] )
        self.sep_token_id = int( pooler_config.token_pooling_config[ 'sep_token_id' ] )
        self.new_token_id = int( pooler_config.token_pooling_config[ 'new_token_id' ] )
        
        self.layers = list( pooler_config.token_pooling_config[ 'attn_layers' ] )
        
        self.d_model = base_config.d_model
        self.n_heads = base_config.n_heads
        self.d_key = self.d_model // self.n_heads
        
    
    def forward( self, layer_states: torch.Tensor, input_ids: torch.Tensor, return_final: bool ) -> torch.Tensor:
        ...
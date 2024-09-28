from operator import itemgetter

import torch
import torch.nn.functional as F

from .configuration import LSWTPoolerConfig


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

        self.layer_weighting = torch.nn.Parameter( torch.zeros( self.num_layers, 1, 1, 1 ), requires_grad=True )
        self.layer_weighting.data.fill_( 0.0 )
    
    def forward( self, hidden_states: tuple[torch.Tensor] ) -> torch.Tensor:       
        states = torch.stack( itemgetter( *self.layer_idx )( hidden_states ) )
            
        drop_mask = states.new_ones( [ states.size( 0 ), states.size( 1 ), 1, 1 ] )
        drop_mask = self.dropout( drop_mask )
        
        return ( self.layer_weighting.softmax( 0 ) * drop_mask * states ).sum( 0 )

class LSWTTokenPoolerCLS( torch.nn.Module ):
    ...
    
    def forward( self, layer_states, input_ids, return_final ):
        assert return_final, 'Can only return final using CLS pooler'
        ...

class LSWTTokenPoolerAttention( torch.nn.Module ):
    ...
    
    def forward( self, layer_states, input_ids, return_final ):
        ...
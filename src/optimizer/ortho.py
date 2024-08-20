
import torch

class Ortho( torch.optim.Optimizer ):
    def __init__(
        self,
        params,
        beta=0.01,
    ):
        defaults = dict(
            beta=beta
        )
        
        super().__init__( params, defaults )
    
    def step( self, closure=None ):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group[ 'params' ]:
                beta = group[ 'beta' ]
                
                p.data.copy_( ( 1 + beta ) * p.data - beta * p.data.mm( p.data.transpose( 0, 1 ).mm( p.data ) ) )
        
        return loss
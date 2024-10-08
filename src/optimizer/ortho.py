
import torch

class Ortho( torch.optim.Optimizer ):
    def __init__(
        self,
        params,
        beta=1.0,
        norm_p: float | str = 2.0,
    ):
        defaults = dict(
            beta=beta,
            norm_p=norm_p,
        )
        
        super().__init__( params, defaults )
    
    # @torch.no_grad()
    # def step( self, closure=None ):
    #     loss = None
        
    #     for group in self.param_groups:
    #         for p in group[ 'params' ]:
    #             beta = group[ 'beta' ]
                
    #             # p.data.copy_( ( 1 + beta ) * p.data - beta * p.data.mm( p.data.transpose( 0, 1 ).mm( p.data ) ) )
                
    #             W = p.data
                
    #             N = W.shape[0]
    #             WWT = torch.matmul(W, W.T)
    #             I = torch.eye(N, device=W.device)
                
    #             diff = WWT - I
    #             dist = torch.dist(WWT, I)
                
    #             grad = diff / (dist + 1e-8)  # Adding a small epsilon to avoid division by zero
    #             grad = torch.matmul(grad + grad.T, W)
                
    #             p.grad.data.add_( grad, alpha=beta )
        
    #     return loss
    
    def compute_loss( self ):
        loss = None
        
        for group in self.param_groups:
            for p in group[ 'params' ]:
                beta = group[ 'beta' ]
                norm_p = group[ 'norm_p' ]
                
                I = torch.eye( p.shape[1], device=p.device, dtype=p.dtype )
                
                l = torch.norm( torch.matmul( p.T, p ) - I, p=norm_p ) * beta
                
                if loss is None:
                    loss = l
                else:
                    loss += l
        
        return loss
        
        
        
        
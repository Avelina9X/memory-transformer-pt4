"""Spectra module, contains the normalizer class.
"""

import torch

class Spectra( torch.optim.Optimizer ):
    """Spectra, the Spectral Normalizer.
    
    Spectra is not an optimizer. You should use a normal optimizer first, and then call
    `Spectra.step()` after calling `zero_grad(set_to_none=True)` on the main optimizer. 
            
    Based on Apple's ml-sigma-reparam: https://github.com/apple/ml-sigma-reparam
    but with an additional `zeta` term that controls in the influence of normalization.
    
    Setting zeta to zero disables normalization, and setting to one is full spectral norm.
    
    Currently only supports linear layers (2D weight matrices).
    Support for ND convolutions is planned.
    """
    def __init__( self, params, zeta=0.5 ):
        """Implements the Spectra algorithm.
        This is not an optimizer, `step()` should be called after gradients are zeroed.
        
        If Spectra has no effect on training try raising zeta.
        If model collapses try decreasing zeta.

        Args:
            params (Iterable): iterable of parameters to optimize or dicts defining param groups.
            zeta (float, optional): spectral norm ratio, 0 disables normalization. Defaults to 0.5.
        """
        
        if not 0.0 <= zeta <= 1.0:
            raise ValueError( f"Invalid zeta: {zeta}" )
        
        defaults = { 'zeta': zeta }
        super().__init__( params, defaults )

    @torch.no_grad()
    def step( self, closure=None ):
        if closure is not None:
            raise ValueError( 'Spectra is not an optimizer so should use a closure' )

        for group in self.param_groups:
            zeta = group['zeta']
            
            for p in group['params']:
                if p.grad is not None:
                    raise RuntimeError( 'Gradients should all be None!' )

                # Retrieve optimizer states
                state = self.state[p]

                # State initialization
                if len( state ) == 0:
                    # Set steps to zero... not necessary, but here to uphold convention
                    state['step'] = 0

                    # Calculate `u` parameter with svd
                    _, _, vh = torch.linalg.svd( p, full_matrices=False ) # pylint: disable=E1102
                    state['u'] = vh[0]

                # Increment step
                state['step'] += 1

                # Get references to param and `u` state
                weight = p
                u = state['u']

                # Compute `u` and `v`
                v = weight.mv( state['u'] )
                v = torch.nn.functional.normalize( v, dim=0 )
                u = weight.T.mv( v )
                u = torch.nn.functional.normalize( u, dim=0 )

                # Update `u`
                state['u'].data.copy_( u )

                # Compute sigma and spectral update
                sigma = torch.einsum( 'c,cd,d->', v, weight, u )
                spectral_p = p / sigma

                # Apply spectral norm to p
                p.mul_( 1.0 - zeta ).add_( spectral_p, alpha=zeta )

        return 0
        
"""Minato Module, contains optimizer class.
"""

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from constants import TORCH_COMPILE_OPTIONS

ParamsT = Iterable[torch.Tensor] | Iterable[dict[str, Any]]


class Minato(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-4,
        beta: float = 0.9,
        weight_decay: float = 2e-1,
        *,
        maximize: bool = False,
        capturable: bool = False
    ):
        """Implements the Minato algorithm.

        Args:
            params (Iterable): iterable of parameters to optimize or dicts defining parameter groups.
            lr (float, optional): learning rate. Defaults to 1e-4.
            beta (float, optional): ema coefficient. Defaults to 0.9.
            weight_decay (float, optional): weight decay coefficient, coupled to learning rate. Defaults to 2e-1.
            maximize (bool, optional): maximize the objective with respect to the params, instead of minimizing. Defaults to False.
            capturable (bool, optional): whether this instance is safe to capture in a CUDA graph. Passing True can impair ungraphed performance, so if you don't intend to graph capture this instance, leave it False. Defaults to False.
        """
        
        if lr < 0.0:
            raise ValueError( f"Invalid learning rate: {lr}" )

        if not 0.0 <= beta < 1.0:
            raise ValueError( f"Invalid beta parameter: {beta}" )

        if weight_decay < 0.0:
            raise ValueError( f"Invalid weight_decay value: {weight_decay}" )

        defaults = {
            'lr': lr,
            'beta': beta,
            'weight_decay': weight_decay,
            'maximize': maximize,
            'capturable': capturable
        }

        super().__init__( params, defaults )

    def __setstate__( self, state ):
        super().__setstate__( state )
        for group in self.param_groups:
            group.setdefault( 'maximize', False )
            group.setdefault( 'capturable', False )
        state_values = list( self.state.values() )
        step_is_tensor = ( len( state_values ) != 0 ) and torch.is_tensor( state_values[0]['step'] )
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor( float( s['step'] ) )


    @torch.no_grad()
    def step( self, closure=None ):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            state_steps = []
            beta = group['beta']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append( p )

                if p.grad.is_sparse:
                    raise RuntimeError( 'Hero does not support sparse gradients' )
                grads.append( p.grad )
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = torch.zeros( ( 1, ), dtype=torch.float, device=p.device ) if self.defaults['capturable'] else torch.tensor( 0.0 )
                    state['exp_avg'] = torch.zeros_like( p, memory_format=torch.preserve_format )

                exp_avgs.append( state['exp_avg'] )
                state_steps.append( state['step'] )

            minato(
                params_with_grad,
                grads,
                exp_avgs,
                state_steps,
                beta=beta,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                maximize=group['maximize'],
                capturable=group['capturable']
            )

        return loss

def minato(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    state_steps: list[Tensor],
    capturable: bool = False,
    *,
    beta: float,
    lr: float,
    weight_decay: float,
    maximize: bool
):

    if not all( isinstance( t, torch.Tensor ) for t in state_steps ):
        raise RuntimeError( "API has changed, `state_steps` argument must contain a list of singleton tensors" )

    func = _single_tensor_minato

    func(
        params,
        grads,
        exp_avgs,
        state_steps,
        beta=beta,
        lr=lr,
        weight_decay=weight_decay,
        maximize=maximize,
        capturable=capturable
    )

@torch.compile( **TORCH_COMPILE_OPTIONS )
def _single_tensor_minato(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    state_steps: list[Tensor],
    *,
    beta: float,
    lr: float,
    weight_decay: float,
    maximize: bool,
    capturable: bool
):

    for i, param in enumerate( params ):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        step_t = state_steps[i]

        if capturable:
            assert param.is_cuda and step_t.is_cuda

        if torch.is_complex( param ):
            grad = torch.view_as_real( grad )
            exp_avg = torch.view_as_real( exp_avg )
            param = torch.view_as_real( param )

        # update step
        step_t += 1

        # Perform stepweight decay
        param.mul_( 1.0 - lr * weight_decay )

        # Decay the first and second moment running average coefficient
        exp_avg.mul_( beta ).add_( grad, alpha=1 - beta )

        if capturable:
            step = step_t
            step_size = lr
            step_size_neg = step_size.neg() # type: ignore

            ratio = ( exp_avg.abs() / 1e-15 ).clamp( max=1.0 )
            param.addcmul_( exp_avg.sign(), ratio, value=step_size_neg )
        else:
            step = step_t.item()
            step_size_neg = - lr

            ratio = ( exp_avg.abs() / 1e-15 ).clamp( max=1.0 )
            param.addcmul_( exp_avg.sign(), ratio, value=step_size_neg )
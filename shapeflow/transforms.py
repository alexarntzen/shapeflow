"""Module with invertible transforms that return (y(x), dy/dx)

Implements:
    Invertible residual networks
    Continuous normalizing flow transformation
"""
import contextlib
import io
from typing import Union

import torch
import torch.nn as nn
import normflow as nf
from torchdyn.models import CNF, hutch_trace, autograd_trace
from torchdyn.core import NeuralODE
from torchdyn.nn import Augmenter
from collections.abc import Callable
from torch.distributions import MultivariateNormal


class NDETransform(nn.Module):
    """Transformations defined using the flow of a nural differential equation (NDE)

    Note that backwards computation using adjoint for torchdyn and reverse time,
    so we need to make this an inverse model  z = T(x)
    """

    def __init__(
        self,
        shape: torch.Tensor,
        get_net: Callable[..., nn.Module],
        trace_estimator: str = "autograd",
        t_span: torch.Tensor = None,
        sensitivity: str = "adjoint",
        verbose: bool = False,
        solver: Union[str, nn.Module] = "dopri5",
        atol: float = 1e-3,
        rtol: float = 1e-3,
        atol_adjoint: float = 1e-6,
        rtol_adjoint: float = 1e-6,
        **model_kwargs: object,
    ):
        """
        Args:
            shape: Shape of events tensor. Events must be vectors due to NDE solver
            get_net: returns the vector field form **model parameters. Can be both
                autonomous (x)and time dependet (x, t)
            trace_estimator: the  of trance estimator
            t_span: time interval of vector field to integrate
            sensitivity: method for backpropagation. Integration of adjoint or
                backpropagation.
            solver: type of ODE solver to use
            atol: tolerance for step size control of ODE solver
            rtol: relative tolerance for step size control of ODE solver
            atol_adjoint: tolerance for step size control of ODE solver of adjoint
             equation
            rtol_adjoint:  tolerance for step size control of ODE solver of adjoint
            equation
            **model_kwargs: parameters for get_net
        """
        super().__init__()

        input_features = shape[0]

        self.net = get_net(
            input_dimension=input_features,
            output_dimension=input_features,
            **model_kwargs,
        )

        # trace with noise
        noise_dist = MultivariateNormal(
            torch.zeros(input_features), torch.eye(input_features)
        )
        if trace_estimator == "hutch":
            self.trace_estimator = hutch_trace
        elif trace_estimator == "autograd":
            self.trace_estimator = autograd_trace
        else:
            raise ValueError(f"No trace estimator called '{trace_estimator}'.")

        # construct vector field of coupled system for x(t) and log_prob(x(t))
        cnf = CNF(
            net=self.net, trace_estimator=self.trace_estimator, noise_dist=noise_dist
        )
        # construct ode system for the vector field above
        with contextlib.redirect_stdout(io.StringIO()) as f:
            self.nde = NeuralODE(
                cnf,
                sensitivity=sensitivity,
                solver=solver,
                atol=atol,
                rtol=rtol,
                atol_adjoint=atol_adjoint,
                rtol_adjoint=rtol_adjoint,
                return_t_eval=False,
            )
        if verbose:
            print(f.getvalue())

        # create times for forward and inverse calculation of T
        self.t_span = torch.linspace(0, 1, 2) if t_span is None else t_span
        self.t_span_reverse = torch.flip(self.t_span, dims=[0])

        # stores log_det_abs_dx_dy (dy_dx) as first parameter
        self.model = nn.Sequential(Augmenter(augment_idx=1, augment_dims=1), self.nde)

    def forward(self, x, keepdim=False):
        """Solve ode system forwards in time with initial conditions (x,0)

        Args:
            y: initial condition for ODE in the space (x(t_1)
            keepdim: if not to squeeze the probability difference

        Returns:
            tuple:  (x(t_1), -odesolve(tr(Df(x(t),t),t_0,t_1))
        """

        # forward computation uses normal time
        self.model[1].t_span = self.t_span

        # return x(t_1)), integrate(tr(Df(x(t)), t_0, t_1)
        xtrJ = self.model(x)[1]
        y, log_abs_det_dx_dy = (
            xtrJ[:, 1:],
            xtrJ[:, 0:1],
        )

        if keepdim:
            return y, -log_abs_det_dx_dy
        else:
            return y, -log_abs_det_dx_dy[:, 0]

    def inverse(self, y, keepdim=False) -> tuple[torch.tensor, torch.Tensor]:
        """Solve ode system backwards in time with initial conditions (y,0)

        Args:
            y: initial condition for ODE in the space (x(t_1)
            keepdim: if not to squeeze the probability difference

        Returns:
            tuple:  (x(t_0), -odesolve(tr(Df(x(t),t),t_1,t_0))
        """
        # invert t_span for backward integration
        self.model[1].t_span = self.t_span_reverse

        # adjoint for backward time does not work for torchdyn yet. (26.06.22)
        # if adjoint, turn of gradient computation
        wrong_grad = self.model[1].vf.sensitivity == "adjoint"
        with torch.set_grad_enabled(not wrong_grad):
            # returns sol = [start, end]
            sol = self.model(y)[1]
            x, log_abs_det_dy_dx = (
                sol[:, 1:],
                sol[:, 0:1],
            )
            if keepdim:
                return x, -log_abs_det_dy_dx
            else:
                return x, -log_abs_det_dy_dx[:, 0]

    def __str__(self):
        return "NDETransform"


def get_residual_transform(
    shape: torch.Size,
    hidden_features: int,
    hidden_layers: int = None,
    kernel_size: int = None,
    CNN: bool = False,
    n_exact_terms: int = 2,
    n_samples: int = 1,
    reduce_memory: bool = True,
    reverse: bool = True,
) -> nf.flows.Residual:
    """

    Construct an invertible residual neural network.
    Invulnerability constrained by Lipschitz constant scaling
    Inverse computed by fixed point iteration.
    Jacobian series estimator implemented.

    Args:
        shape: Shape of input event tensor
        hidden_features: dimensions in hidden layers
        hidden_layers:  number of hidden layers
        kernel_size: for convolution layer
        CNN: If linear operation is convolution. Then blocks need to
        n_exact_terms: number of terms always included in the power series
        n_samples: number of samples used to estimate power series (hutch trace)
        reduce_memory:
        reverse: if to do fixed point iteration on forward computation
    Returns:
        nf.flows.Residual: the invertible model

    """
    latent_size = shape[0]
    if CNN:
        if kernel_size is None:
            kernel_size = hidden_features
        assert kernel_size % 2 == 1, f"kernel size must be odd but is {kernel_size}"
        # construct network that is invertible
        net = nf.nets.LipschitzCNN(
            channels=[1] * (hidden_layers + 1),
            kernel_size=[kernel_size] * (hidden_layers),
            init_zeros=True,
            lipschitz_const=0.9,
        )

    else:
        # construct network that is invertible
        net = nf.nets.LipschitzMLP(
            [latent_size] + [hidden_features] * (hidden_layers - 1) + [latent_size],
            init_zeros=True,
            lipschitz_const=0.9,
        )

    # add Jacobian and fixed point iteration to model
    transform = nf.flows.Residual(
        net,
        n_exact_terms=n_exact_terms,
        n_samples=n_samples,
        reduce_memory=reduce_memory,
        reverse=reverse,
    )
    return transform

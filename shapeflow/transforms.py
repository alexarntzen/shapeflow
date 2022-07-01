import contextlib
import io
import torch
import torch.nn as nn
import normflow as nf
from torchdyn.models import CNF, hutch_trace, autograd_trace
from torchdyn.core import NeuralODE
from torchdyn.nn import Augmenter
from collections.abc import Callable
from torch.distributions import MultivariateNormal


class NDETransform(nn.Module):
    """Note that backwards computation using adjoint for torchdyn and reverse time
    so we need to make this an iverse model  z = T(x)"""

    def __init__(
        self,
        shape,
        get_net: Callable[..., nn.Module],
        trace_estimator: str = "autograd",
        t_span=None,
        sensitivity="adjoint",
        verbose: bool = False,
        solver="dopri5",
        atol=1e-3,
        rtol=1e-3,
        atol_adjoint=1e-6,
        rtol_adjoint=1e-6,
        **model_kwargs,
    ):
        super().__init__()
        self._cached_log_abs_det_dy_dx = None
        input_features = shape[0]

        self.net = get_net(
            input_dimension=input_features,
            output_dimension=input_features,
            **model_kwargs,
        )
        # self.context_size = context_size

        # trace with noise
        noise_dist = MultivariateNormal(
            torch.zeros(input_features), torch.eye(input_features)
        )
        self.trace_estimator = (
            hutch_trace if trace_estimator == "hutch_trance" else autograd_trace
        )

        cnf = CNF(
            net=self.net, trace_estimator=self.trace_estimator, noise_dist=noise_dist
        )
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

        self.t_span = torch.linspace(0, 1, 2) if t_span is None else t_span
        self.t_span_reverse = torch.flip(self.t_span, dims=[0])

        # stores log_det_abs_dx_dy (dy_dx) as first parameter
        self.model = nn.Sequential(Augmenter(augment_idx=1, augment_dims=1), self.nde)

    def forward(self, x, keepdim=False):

        self.model[1].t_span = self.t_span

        # returns sol = [start, end]

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

        # invert t_span for backward integration
        self.model[1].t_span = self.t_span_reverse

        # TODO: Fix this wrong backward for backward time
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
    n_exact_terms=2,
    n_samples=1,
    reduce_memory=True,
    reverse=True,
):
    latent_size = shape[0]
    if CNN:

        if kernel_size is None:
            kernel_size = hidden_features
        assert kernel_size % 2 == 1, f"kernel size must be odd but is {kernel_size}"
        net = nf.nets.LipschitzCNN(
            channels=[1] * (hidden_layers + 1),
            kernel_size=[kernel_size] * (hidden_layers),
            init_zeros=True,
            lipschitz_const=0.9,
        )

    else:
        net = nf.nets.LipschitzMLP(
            [latent_size] + [hidden_features] * (hidden_layers - 1) + [latent_size],
            init_zeros=True,
            lipschitz_const=0.9,
        )
    transform = nf.flows.Residual(
        net,
        n_exact_terms=n_exact_terms,
        n_samples=n_samples,
        reduce_memory=reduce_memory,
        reverse=reverse,
    )
    return transform

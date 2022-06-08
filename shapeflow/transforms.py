from abc import ABC
from typing import Type
import warnings
import torch
import torch.nn as nn
import normflow as nf
import torch.distributions as dist
from torch.distributions import constraints
from torchdyn.models import CNF, hutch_trace, autograd_trace
from torchdyn.core import NeuralODE
from torchdyn.nn import Augmenter
from collections.abc import Callable
import nflows.transforms
import nflows.flows
from torch.distributions import MultivariateNormal

#
# class ConditionalModule(nn.Module):
#     def __init__(self, context_size, get_net:Callable[..., nn.Module], **module_kwargs):
#         super(ConditionalModule, self).__init__()
#
#         self.models_c = nn.ModuleList([get_net(**module_kwargs) for _ in range(context_size)])
#         self.context_size = context_size
#         self.context = None
#
#
#     def condition(self, context:int ):
#         self.context = context
#         return self
#
#     def forward(self, x, context:int=None):
#         if context is None and self.context is not None:
#             context = self.context
#         return self.models_c[context](x)


class NDETransform(nn.Module):
    def __init__(
        self,
        shape,
        get_net: Callable[..., nn.Module],
        trace_estimator: str = "hutch_trance",
        t_span=None,
        **model_kwargs,
    ):
        super().__init__()
        self._cached_log_abs_det_dy_dx = None
        input_features = shape[0]

        # for multiple flows in one module
        # if context_size>1:
        #     self.net = ConditionalModule(context_size=context_size, get_net=get_net, input_dimension=input_features, output_dimension=input_features, **model_kwargs)
        # else:

        self.net = get_net(input_features, input_features, **model_kwargs)
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
        self.nde = NeuralODE(
            cnf,
            solver="dopri5",
            sensitivity="adjoint",
            atol=1e-4,
            rtol=1e-4,
            return_t_eval=False,
        )

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


def get_transform_nflow(
    Transform: Type, shape: torch.Size, **kwargs
) -> nflows.transforms.Transform:
    assert len(shape) == 1
    transform = Transform(features=shape[0], **kwargs)
    if isinstance(transform, nflows.flows.Flow):
        transform = transform._transform

    if isinstance(transform, nflows.transforms.Transform):
        return transform
    else:
        RuntimeError("Failed to create Transform")

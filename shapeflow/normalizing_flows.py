"""
Cost functions associated with the SRV form. || q - sqrt(ksi_dx)r circ ksi||_{L^2}
"""
import warnings
from typing import Optional, Tuple, Sequence

import torch
import torch.nn as nn
import torch.distributions as dist
import torch.distributions.constraints as constraints
import flowtorch.bijectors as bij

l2_loss = nn.MSELoss()


class WrapModel(bij.Bijector):
    """Wrapper for nn.Module to work as bij.Bijector.

    nn.Module must have .forward and .inverse on the form described below

    Many models are not written in be compatile with torch.distributions.
    Thus we implement a simple wrapper class to make it compatible with flowtorch.
    """

    # "Default" event shape is to operate on vectors
    domain = constraints.real_vector
    codomain = constraints.real_vector

    def __init__(
        self,
        model: nn.Module,
        *,
        shape: torch.Size,
        context_shape: Optional[torch.Size] = None,
    ) -> None:
        super().__init__(params_fn=None, shape=shape, context_shape=context_shape)
        # update domain shapes to account for matrix (2) and scalar (0)  input
        self.domain = constraints.independent(constraints.real, len(shape))
        self.codomain = constraints.independent(constraints.real, len(shape))

        self.model = model
        self.parameters = model.parameters
        self._model_inverse_func = self.model.inverse
        self._model_forward_func = self.model.forward

    def _forward(
        self,
        x: torch.Tensor,
        params: Optional[Sequence[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # assumes model.forward(x) = T(x), log | det DT(x) |
        return self._model_forward_func(x)

    def _inverse(
        self,
        y: torch.Tensor,
        params: Optional[Sequence[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # assumes model.inverse(y) = T_inv(y), log | det DT_inv(y) |
        # log_abs_det is

        x, log_abs_det_jac_inv = self._model_inverse_func(y)

        # returns   T_inv(y), log | det DT(T_inv(y)) |
        return x, -log_abs_det_jac_inv

    def _log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        params: Optional[Sequence[torch.Tensor]],
    ) -> torch.Tensor:
        warnings.warn("Computing _log_abs_det_jacobian from values and not from cache.")
        if x is not None:
            _y, log_det_jac = self._model_forward_func(x)
            return log_det_jac
        elif y is not None:
            _x, log_det_jac_inv = self._model_inverse_func(y)
            return -log_det_jac_inv
        else:
            raise RuntimeError


class WrapInverseModel(WrapModel):
    """This class wraps the genius case when T = model.inverse and T_inv = model.forward

    This is usually the case since during training.
    Since the inverse operation is used the most.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        shape: torch.Size,
        context_shape: Optional[torch.Size] = None,
    ) -> None:
        super().__init__(model=model, shape=shape, context_shape=context_shape)

        self._model_inverse_func = self.model.forward
        self._model_forward_func = self.model.inverse


class ModuleBijector(WrapModel):
    def __int__(
        self,
        model: nn.Module,
        *,
        shape: torch.Size,
        context_shape: Optional[torch.Size] = None,
        **kwargs,
    ):
        super().__init__(shape=shape, context_shape=context_shape, model=model)


def torch_monte_carlo_dkl_loss(model: dist.Distribution, x_train, y_train=None):
    d_kl_est = -model.log_prob(x_train).mean()
    return d_kl_est

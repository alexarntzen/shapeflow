"""Module implements functions that wrap transforms into flowtorch bijectors

Using the flowtorch package has a few advantages:
    Classes have methods that match pytorch.distributions
    Lazy loading of bijectos until a event shape of data is given
    Forward method does not return Jacobian
    Separate method for returning Jacobian and calculating log probability

Disadvantages:
    Api of Bijector is strange due to its use of python metaclass.

Therefore, this module implements a wrapper that returns a flowtorch distribution
if you have a module that returns, (y log_det_dy_dx)  with input x

"""
import warnings
from collections.abc import Callable
from typing import Optional, Sequence, Tuple, Iterator

import torch.nn as nn
import torch.distributions.constraints as constraints
import torch.distributions.transformed_distribution
import flowtorch as ft
import flowtorch.parameters as ftparams
import flowtorch.bijectors as ftbij


class WrapModel(ftbij.Bijector):
    """Wrapper for nn.Module to work as bij.Bijector.

    nn.Module must have .forward and .inverse on the form described below

    Many models are not written in be compatible with torch.distributions.
    This class is a simple wrapper to make modules compatible with flowtorch.
    """

    # "Default" event shape is to operate on vectors
    domain = constraints.real_vector
    codomain = constraints.real_vector

    def __init__(
        self,
        params_fn: Optional[ft.Lazy] = None,
        *,
        shape: torch.Size,
        context_shape: Optional[torch.Size] = None,
    ) -> None:

        if params_fn is None:
            # find a good default here
            params_fn = LazyModule()
        super().__init__(params_fn=params_fn, shape=shape, context_shape=context_shape)
        # update domain shapes to account for matrix (2) and scalar (0)  input
        self.domain = constraints.independent(constraints.real, len(shape))
        self.codomain = constraints.independent(constraints.real, len(shape))

        # this is the transform that is being wrapped
        self.model: nn.Module = self._params_fn.transform

        self._model_forward_func = self.model.forward
        self._model_inverse_func = self.model.inverse

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        for param in self.model.parameters(recurse=recurse):
            yield param

    def _forward(
        self,
        x: torch.Tensor,
        params: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return T(x),  DT(x)"""

        return self._model_forward_func(x)

    def _inverse(
        self,
        y: torch.Tensor,
        params: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return x,  DT(x)  where x = T_inv(y)"""

        # assumes model.inverse(y) = T_inv(y), log | det DT_inv(y) |
        # log_abs_det is

        x, log_abs_det_dx_dy = self._model_inverse_func(y)

        # returns   T_inv(y), log | det DT(T_inv(y)) |
        return x, -log_abs_det_dx_dy

    def _log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        params: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Returns the Jacobian of the transformation if x or y is known"""
        warnings.warn("Computing _log_abs_det_jacobian from values and not from cache.")
        if params is not None:
            context = params[0]
            if x is not None:
                _y, log_det_jac = self._model_forward_func(x, context=context)
                return log_det_jac
            elif y is not None:
                _x, log_det_jac_inv = self._model_inverse_func(y, context=context)
                return -log_det_jac_inv
            else:
                raise RuntimeError
        else:
            if x is not None:
                _y, log_det_jac = self._model_forward_func(x)
                return log_det_jac
            elif y is not None:
                _x, log_det_jac_inv = self._model_inverse_func(y)
                return -log_det_jac_inv
            else:
                raise RuntimeError

    def param_shapes(self, shape: torch.Size) -> Sequence[torch.Size]:
        """Returns shape of parameters"""
        for param in self.parameters():
            yield param.shape


class WrapInverseModel(WrapModel):
    """Used when the given transformation (model) is used in the normalizing direction

    In this case our transport T is given by:
    T = model.inverse
    T_inv = model.forward

    This is usually the case, since we need T_inv the most when training
    """

    def __init__(
        self,
        params_fn: Optional[ft.Lazy] = None,
        *,
        shape: torch.Size,
        context_shape: Optional[torch.Size] = None,
    ):
        """Similar to WrapModel, but forward and inverse are reversed

        assume model.inverse returns x, log_det_dx_dy
        assume model.forward returns y, log_det_dy_dx
        """
        super().__init__(
            params_fn=params_fn,
            shape=shape,
            context_shape=context_shape,
            # **model_kwargs["model_kwargs"],
        )

        self._model_inverse_func = self.model.forward
        self._model_forward_func = self.model.inverse


class LazyModule(ftparams.Parameters):
    """Specify the event shape of a transformation lazily

    If you have a function get_transform that needs the shape of events,
    then this class delays the initialization until the shape of events is specified.
    """

    def __init__(
        self,
        param_shapes: Sequence[torch.Size],
        input_shape: torch.Size,
        context_shape: Optional[torch.Size],
        get_transform: Callable[..., nn.Module] = None,
        **model_kwargs,
    ) -> None:
        """Delay initialization until input_shape is given

        Args:
            param_shapes: Not used
            input_shape: Shape of events
            context_shape: not used
            get_transform: returns transform using parameters model_kwargs.
            **model_kwargs: parameters for get_transform
        """
        model_kwargs = model_kwargs["model_kwargs"]
        if get_transform is None:
            get_transform = model_kwargs["Transform"]
        super().__init__(param_shapes, input_shape, context_shape)
        self.transform = get_transform(shape=input_shape, **model_kwargs)

    def _forward(
        self,
        x: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> Optional[Sequence[torch.Tensor]]:
        # so that params = context
        # I have no idea why this is needed
        return (context,)

import warnings
from typing import Optional, Tuple, Sequence, Union, List, Iterator
from collections.abc import Callable

import flowtorch.bijectors
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import torch.distributions as dist
import torch.distributions.constraints as constraints
import torch.distributions.transformed_distribution
import flowtorch as ft
import flowtorch.bijectors as bij
import flowtorch.distributions as ftdist
from flowtorch.parameters.base import Parameters
import deepthermal.validation

l2_loss = nn.MSELoss()


def monte_carlo_dkl_loss(
    model: dist.Distribution,
    data: Union[List, TensorDataset],
    loss_func: callable = None,
) -> torch.Tensor:
    # model == flow
    # load samples from unknown
    (x_train,) = data[:]
    d_kl_est = -model.log_prob(x_train).mean()
    return d_kl_est

def get_monte_carlo_dkl_loss_conditioned(epsilon:float=0.1):
    def monte_carlo_dkl_loss_conditioned(
        model: nn.ModuleList,
        data: Union[List, TensorDataset],
        loss_func: callable = None,
    ) -> torch.Tensor:
        # model == flow
        # load samples from unknown
        x_train, p_old, priors = data[:]
        c_max = p_old.shape[-1]  # model.bijector.module.context_size
        # epsilon = 0.001
        # # print(model.bijector._context_shape)
        #
        log_p = torch.zeros((len(p_old), c_max))
        c_max = len(model)
        for k in range(c_max):
            log_p[:, k] = model[k].log_prob(x_train)

        p = torch.exp(log_p)
        p = torch.nan_to_num(p)

        sum_p = torch.sum(p*priors, dim=1, keepdim=True)
        p_new  = p*priors/sum_p
        d_kl_est = - (p_new* log_p).mean()


        return d_kl_est
    return monte_carlo_dkl_loss_conditioned

def get_post_epoch_update_p(epsilon:float=0.1):
    @torch.no_grad()
    def post_epoch_update_p(model:nn.Module, data:torch.utils.data.Dataset):
        x_train, p_old = data[:]
        c_max = p_old.shape[-1]  # model.bijector.module.context_size
        log_p = torch.zeros((len(p_old), c_max))

        for k in range(c_max):
            log_p[:,k] = model[k].log_prob(x_train)

        p = torch.exp(log_p)
        p = torch.nan_to_num(p)
        sum_p = torch.sum(p, dim=1, keepdim=True)
        p_dir = p / sum_p
        p_new = p_old + epsilon*(p_dir - p_old)

        p_old[:] = p_new[:]

    return post_epoch_update_p

def get_flow(
    get_transform: Callable[..., nn.Module],
    base_dist: dist.Distribution,
    inverse_model: bool = True,
    flowtorch: bool = True,
    num_flows: int = 1,
    **transform_kwargs,
) -> Union[nn.ModuleList,dist.TransformedDistribution, ftdist.Flow]:
    """get the flow if you have a get get_transform"""
    if num_flows > 1:
        return nn.ModuleList(
            [
                get_flow(
                    get_transform=get_transform,
                    base_dist=base_dist,
                    inverse_model=inverse_model,
                    flowtorch=flowtorch,
                    num_flows=1,
                    **transform_kwargs,
                )
                for _ in range(num_flows)
            ]
        )
    else:
        if flowtorch:
            bijector = get_bijector(
                get_transform=get_transform,
                inverse_model=inverse_model,
                **transform_kwargs,
            )
            model = ftdist.Flow(base_dist=base_dist, bijector=bijector)
            return model
        else:
            # alternative code structure
            transform = get_transform(**transform_kwargs)
            if inverse_model:
                transform = transform.inv
            model = dist.TransformedDistribution(
                base_distribution=base_dist, transforms=transform
            )
            return model


class WrapModel(bij.Bijector):
    """Wrapper for nn.Module to work as bij.Bijector.

    nn.Module must have .forward and .inverse on the form described below

    Many models are not written in be compatible with torch.distributions.
    Thus we implement a simple wrapper class to make it compatible with flowtorch.
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
        "Return T(x),  DT(x)"

        return self._model_forward_func(x)

    def _inverse(
        self,
        y: torch.Tensor,
        params: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        "Return x,  DT(x)  where x = T_inv(y)"
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
        """Returns"""
        return (shape,)


class WrapInverseModel(WrapModel):
    """This class wraps the genius case when T = model.inverse and T_inv = model.forward

    This is usually the case during training.
    Since the inverse operation is used the most.
    """

    def __init__(
        self,
        params_fn: Optional[ft.Lazy] = None,
        *,
        shape: torch.Size,
        context_shape: Optional[torch.Size] = None,
        # **model_kwargs,
    ) -> None:
        super().__init__(
            params_fn=params_fn,
            shape=shape,
            context_shape=context_shape,
            # **model_kwargs["model_kwargs"],
        )

        self._model_inverse_func = self.model.forward
        self._model_forward_func = self.model.inverse


class LazyModule(Parameters):
    def __init__(
        self,
        param_shapes: Sequence[torch.Size],
        input_shape: torch.Size,
        context_shape: Optional[torch.Size],
        get_transform: Callable[..., nn.Module] = None,
        **model_kwargs,
    ) -> None:
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
        return (context,)


def get_bijector(
    get_transform: callable = None,
    inverse_model: bool = True,
    compose: bool = False,
    **transform_kwargs,
) -> Union[ft.bijectors.Bijector, ft.Lazy]:
    """
    returns wrapped module which has a forward method which returns T(x), logDT(x)

    if composted **transform_kwargs must only contain lists
    """
    wrapp = WrapInverseModel if inverse_model else WrapModel
    if compose:
        transform_kwargs_iter = deepthermal.validation.create_subdictionary_iterator(
            transform_kwargs, product=False
        )
        bijectors = []
        for transform_kwargs_i in transform_kwargs_iter:
            bijectors.append(
                wrapp(
                    params_fn=LazyModule(
                        get_transform=get_transform, **transform_kwargs_i
                    )
                )
            )
        bijector = ft.bijectors.Compose(bijectors=bijectors)
    else:
        bijector = wrapp(
            params_fn=LazyModule(get_transform=get_transform, **transform_kwargs)
        )
    return bijector

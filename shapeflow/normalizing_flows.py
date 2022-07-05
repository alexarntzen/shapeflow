"""Module implementing loss functions and functions constructing flows

monte_carlo_dkl_loss: loss function, mean(log_prob(x_i)), estimate  variable part of
KL-divergence
get_monte_carlo_conditional_dkl_loss: loss function mean(p(c^j|x_i)log_prob(x_i|c^j))

get_flow: returns a flowtorch Flow object
get_bijector: returns a lazy loading module (flowtorch bijector)
"""
from typing import Union, List
from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import TensorDataset, Dataset
import torch.distributions as dist
import torch.distributions.transformed_distribution
import extratorch as etorch
import flowtorch as ft
import flowtorch.distributions as ftdist
import flowtorch.bijectors as ftbij

from shapeflow.flowtorch_wrapper import WrapModel, WrapInverseModel, LazyModule

l2_loss = nn.MSELoss()


def monte_carlo_dkl_loss(
    model: dist.Distribution, data: Union[List, TensorDataset], **kwargs
) -> Tensor:
    """Get Monte Carlo estimate of KL-divergence using samples from left distribution
    D_KL(P_left|| P_right)

    Only returns the variable part (i.e. cross entropy).

    Args:
        model: Model with .log_prob implemented. This is the right density
        data: Samples from the left distribution.
        **kwargs: Extra parameters given to loss functions

    Returns:
        Tensor: Computed loss of KL-divergence
    """
    # load samples from unknown
    x_train = data[:][0]

    # compute monte carlo estimate of KL-divergence
    d_kl_est = -model.log_prob(x_train).mean()
    return d_kl_est


def get_cluster_log(model: List[dist.Distribution], data: Dataset, **kwargs):
    """Log performance of clustering"""
    return {"Conditional entropy": cluster_performance(model=model, data=data).item()}


def cluster_performance(model: List[dist.Distribution], data: Dataset) -> Tensor:
    """Estimate clustering performance with conditional entropy

    Args:
        model: list of models with .log_prob, p(x|c) for all c
        data: dataset of (samples x, _ , priors)
    """
    x, _, prior = data[:]

    p = posterior(model=model, x=x, prior=prior)

    # since we do not know q we calculate estimate using conditional entropy instead
    return conditional_cross_entropy(p=p)


def conditional_cross_entropy(p: Tensor, q: Tensor = None) -> Tensor:
    """
    Compute the conditional cross entropy of p(x|c) and q(x|c).
    We use Monte Carlo estimate:
        mean(q(c|x)log(p(c|x)) over all c and samples x
    If q is not given then we estimate p≈q, and return conditional entropy

    Args:
        p: hat p(C|X) with shape (len(x), len(c))
        q: p(C|X) with shape (len(x), len(c))

    Returns:
        Tensor: Conditional cross entropy
    """
    if q is None:
        # use maximal estimate
        q = p
    return -torch.mean(q * torch.log(p))


def get_monte_carlo_conditional_dkl_loss(
    epsilon: float = 1, no_grad_posterior: bool = True
) -> Callable[List[dist.Distribution], Dataset, bool, ..., Tensor]:
    """
    Get function estimating KL-divergence with conditional model

    Args:
        epsilon: rate of update to posteriors
        no_grad_posterior: keep p(c|x) constant, (exclude it from the autodiff

    Returns:
        Callable: function estimating  KL-divergence with conditional model F
    """

    def monte_carlo_conditional_dkl_loss(
        model: List[dist.Distribution],
        data: Union[List, TensorDataset],
        log: bool = False,
        **kwargs,
    ) -> Tensor:
        """Get Monte Carlo estimate of KL-divergence using samples from left
        distribution
        D_KL(P_left|| P_right)

        Only returns the variable part (i.e. cross entropy),

        Using Monte Carlo estimate : mean(p(c|x)log(p(x|c))) over all c, x

        Args:
            model: list of estimated p(x|c) for each c
            data: dataset consisting of (sample x, estimated posterior, prior)
            log: if this is just the logging step
            **kwargs:

        Returns:
            Tensor: Computed estimate of variable part of KL-divergence
        """

        with torch.set_grad_enabled(not log):
            # assumes data on this form
            # use notation for each row since priors are used for each row
            # x, p(c_k|x), p(c_k)
            x_train, posterior, prior = data[:]

            c_max = len(model)
            log_x_cond_c = torch.zeros((len(posterior), c_max))

            for k in range(c_max):
                log_x_cond_c[..., k] = model[k].log_prob(x_train)

            with torch.set_grad_enabled(not no_grad_posterior):
                # calculate new posterior
                # p(x|c) = exp(log(p(x|c))
                x_cond_c = torch.nan_to_num(torch.exp(log_x_cond_c))

                # p(x) = sum p(x|c_i) * p(x) , (i is last dim)
                marginal_x = torch.sum(x_cond_c * prior, dim=-1, keepdim=True)

                # p(x|c) = p(x|c) * p(c) / p(x)
                new_posterior_ = x_cond_c * prior / marginal_x

                # gradual update with epsilon
                new_posterior = posterior * (1 - epsilon) + new_posterior_ * epsilon

                # if not logging, update posteriors
                if not log:
                    posterior[:] = new_posterior

        # after update
        conditional_dkl = -(new_posterior * log_x_cond_c).mean()

        return conditional_dkl

    return monte_carlo_conditional_dkl_loss


# not efficient implementation
def get_update_posterior(
    epsilon: float = 0.1,
) -> Callable[List[dist.Distribution], Dataset, None]:
    """Get function updating posterior

    Args:
        epsilon: rate of update to posteriors

    Returns:
        Callable:function updating posterior
    """

    @torch.no_grad()
    def update_posterior(model: List[dist.Distribution], data: Dataset):
        """Update posterior using the posterior function"""
        # assumes data on this form
        # use notation for each row since priors are used for each row
        # x, p(c_k|x), p(c_k)
        x_train, posterior, prior = data[:]

        new_posterior = posterior(model=model, x=x_train, prior=prior)

        posterior[:] = posterior + epsilon * (new_posterior - posterior)

    return update_posterior


def posterior(
    model: List[dist.Distribution], x: Tensor, prior: Tensor = None
) -> Tensor:
    """Compute posterior p(c|x) given prior p(c) and p(x|c)

    Args:
        model: list of modules with .log_prob implemented. Computes log(p(x|c))
        x: Observations
        prior: Tensor of p(c) for each observation x

    Returns:
        Tensor: posterior probability p(c|x) for each x, and c
    """

    if prior is None:
        # if no prior choose the prior with maximal entropy
        # any constant prior will give the same result
        prior = 1

    c_max = len(model)

    # log p(x|c)
    log_x_cond_c = torch.zeros((len(x), c_max))

    for k in range(c_max):
        log_x_cond_c[..., k] = model[k].log_prob(x)

    # p(x|c) = exp(log(p(x|c))
    cond_x_c = torch.nan_to_num(torch.exp(log_x_cond_c))

    # p(x) = sum p(x|c_k) * p(x) , (k is last dim)
    marginal_x = torch.sum(cond_x_c * prior, dim=-1, keepdim=True)

    # p(x|c) = p(x|c) * p(c) / p(x)
    posterior = cond_x_c * prior / marginal_x

    return posterior


def get_flow(
    get_transform: Callable[..., nn.Module],
    base_dist: dist.Distribution,
    inverse_model: bool = True,
    flowtorch: bool = True,
    num_flows: int = 1,
    **transform_kwargs,
) -> Union[nn.ModuleList, dist.TransformedDistribution, ftdist.Flow]:
    """Get the flow if you have a get get_transform method and its parameters.

    A way to construct flowtorch Flows without implementing a flowtorch bijector
    directly. A better approach my be to make a pull request to flowtorch

    Args:
        get_transform: Returns the transform used in the flow
        base_dist: base distribution (usually normal distribution)
        inverse_model: Use the forward method of the transformation in the
        normalizing direction
        flowtorch: use the flowtorch package to define flows using transform
        num_flows: number of flows in model. If more than one a list of Flows is
        returned
        **transform_kwargs:

    Returns:
        object: returns the flow model or list of flow models
    """
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
            # alternative code structure that does not work with pytorch yet
            transform = get_transform(**transform_kwargs)
            if inverse_model:
                transform = transform.inv
            model = dist.TransformedDistribution(
                base_distribution=base_dist, transforms=transform
            )
            return model


def get_bijector(
    get_transform: callable = None,
    inverse_model: bool = True,
    compose: bool = False,
    **transform_kwargs,
) -> Union[ft.bijectors.Bijector, ft.Lazy]:
    """
    wrap a transform into a flowtorch bijector

    If you have a method that returns a transform  which has a forward method which
    returns T(x), logDT(x), then this function returns a flowtorch bijector of that
     transform.

    if composted **transform_kwargs must only contain lists
    """
    wrapp = WrapInverseModel if inverse_model else WrapModel
    if compose:
        # create an iterator for each layer to compose
        transform_kwargs_iter = etorch.create_subdictionary_iterator(
            transform_kwargs, product=False
        )
        bijectors = []

        # for each iterator make a flowtorch bijector
        for transform_kwargs_i in transform_kwargs_iter:
            bijectors.append(
                wrapp(
                    params_fn=LazyModule(
                        get_transform=get_transform, **transform_kwargs_i
                    )
                )
            )

        # compose bijectors
        bijector = ftbij.Compose(bijectors=bijectors)
    else:

        # wrap the transform, and return a flowtorch bijector
        bijector = wrapp(
            params_fn=LazyModule(get_transform=get_transform, **transform_kwargs)
        )
    return bijector

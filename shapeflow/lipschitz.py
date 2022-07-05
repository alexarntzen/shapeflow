"""Module implements function needed to scale Lipchitz networks"""
from collections.abc import Callable

import torch.nn as nn
from residual_flows.layers.base import InducedNormLinear, InducedNormConv2d


def get_post_step_lipchitz(n_iterations: int = 5) -> Callable[nn.Module, ..., None]:
    """
    Args:
        n_iterations:  iterations for estimation operator norm
    """

    def post_step_lipchitz(model: nn.Module, **kwargs) -> None:
        """Scales all the Lipschitz networks in model

        Args:
            model: Module containing the Lipchitz networks to rescale
        """
        if isinstance(model, nn.ModuleList):
            for m in model:
                post_step_lipchitz(m, **kwargs)
        for m in model.modules():
            if isinstance(m, InducedNormConv2d) or isinstance(m, InducedNormLinear):
                # compute_weight with update=True scales the network appropriately
                m.compute_weight(update=True, n_iterations=n_iterations)

    return post_step_lipchitz

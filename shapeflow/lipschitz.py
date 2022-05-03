import torch.nn as nn
from residual_flows.layers.base import InducedNormLinear, InducedNormConv2d


def get_post_step_lipchitz(n_iterations: int = 5) -> callable:
    """
    iterations: iterations for estimation operator norm
    """

    def post_step_lipchitz(model: nn.Module, **kwargs):
        for m in model.modules():
            if isinstance(m, InducedNormConv2d) or isinstance(m, InducedNormLinear):
                m.compute_weight(update=True, n_iterations=n_iterations)

    return post_step_lipchitz

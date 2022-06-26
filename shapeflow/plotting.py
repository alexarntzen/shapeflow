"""plot predicted data with decision border"""
from os.path import join
from typing import List, Union
import itertools
import matplotlib.pyplot as plt
import torch
from flowtorch.distributions import Flow


@torch.no_grad()
def plot_2d_cluster(
    model: List[Flow],
    num_samples=100,
    grid_shape: Union[int, tuple] = 100,
    x_lim: tuple = (-1, 1),
    y_lim: tuple = (-1, 1),
    plot_name="vis_model",
    path_figures="../figures",
    **kwargs,
):
    assert len(model) <= 9, "can only plot max 10 clusters"

    if isinstance(grid_shape, int):
        grid_shape = (grid_shape, grid_shape)

    markers = itertools.cycle(("x", ".", "o", "*", "+"))

    fig, ax = plt.subplots(1, frameon=False)
    x_line = torch.linspace(*x_lim, grid_shape[0])
    y_line = torch.linspace(*y_lim, grid_shape[1])

    X, Y = torch.meshgrid(x_line, y_line, indexing="ij")

    grid = torch.stack((X.ravel(), Y.ravel()), dim=1)
    log_prob = torch.zeros((len(model),) + grid_shape)

    for k, model_k in enumerate(model):
        log_prob[k] = model_k.log_prob(grid).reshape(grid_shape)

    Z = len(model) - 1 - torch.argmax(log_prob, dim=0)

    ax.contourf(X, Y, Z)

    for k, model_k in enumerate(model):

        samples = model_k.sample([num_samples])
        ax.scatter(samples[:, 0], samples[:, 1], marker=next(markers))

    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    ax.set_aspect("equal", "box")
    ax.set(xlim=x_lim, ylim=y_lim)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(
        join(
            path_figures,
            f"{plot_name}.pdf",
        ),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)

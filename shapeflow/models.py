from typing import Union
import torch
import torch.nn as nn
from extratorch.models import FFNN


class CNN2D(FFNN):
    def __init__(
        self, kernel_size: Union[int, tuple] = (5, 5), internal_shape=None, **kwargs
    ):
        super(CNN2D, self).__init__(**kwargs)
        # if input dim = output dim must be odd
        self.kernel_size = kernel_size

        # internal_shape means vector inputs
        if internal_shape is not None:
            assert len(internal_shape) == 2

        self.internal_shape = internal_shape

        if isinstance(self.kernel_size, tuple):
            padding = [ks // 2 for ks in self.kernel_size]
        else:
            padding = self.kernel_size // 2

        self.hidden_layers = nn.ModuleList(
            [
                nn.Conv2d(1, 1, kernel_size=self.kernel_size, padding=padding)
                for _ in range(self.n_hidden_layers - 1)
            ]
        )

    def forward(self, x):
        orig_shape = x.shape
        x_internal = self.reshape_vector_matrix(x)
        for layer in self.hidden_layers:
            x_internal = self.activation_func(layer(x_internal))

        # TODO: output layer and input layer
        x_final = self.reshape_matrix_vector(x_matrix=x_internal, orig_shape=orig_shape)
        return x_final

    # so to accept both vector and matrix inputs
    def reshape_vector_matrix(self, x_vector):
        if self.internal_shape is None:
            # assume input is matrix
            if len(x_vector.shape) == 1:
                raise ValueError
            elif len(x_vector.shape) == 2:
                return x_vector.reshape((1, 1) + x_vector.shape)
            elif len(x_vector.shape) == 3:
                return x_vector.unsqueeze(-3)
            else:
                return x_vector

        if self.internal_shape is not None:
            # assume input is tensor
            if len(x_vector.shape) == 1:
                raise x_vector.reshape((1, 1) + self.internal_shape)
            elif len(x_vector.shape) == 2:
                # batch of vectors
                return x_vector.reshape((x_vector.shape[0], 1) + self.internal_shape)
            elif x_vector.shape[-2:] == self.internal_shape:
                if len(x_vector.shape) == 3:
                    # it was a matrix all along
                    return x_vector.unsqueeze(-3)
                elif len(x_vector) == 4:
                    # it was correct shape all along?
                    return x_vector
                else:
                    raise ValueError
            else:
                raise ValueError

    def reshape_matrix_vector(self, x_matrix: torch.tensor, orig_shape: tuple):
        return x_matrix.reshape(orig_shape)

    def __str__(self):
        return "CNN"

import unittest

import torch
import torch.distributions.constraints as constraints
import flowtorch.distributions as ftdist
import shapeflow
import nflows.transforms as transforms


class TestFlowTorchWrapper(unittest.TestCase):
    def test_nflow(self):
        # Set up model
        num_layers = 5
        dims = 2

        transforms_list = []
        for _ in range(num_layers):
            transforms_list.append(transforms.ReversePermutation(features=dims))
            transforms_list.append(
                transforms.MaskedAffineAutoregressiveTransform(
                    features=dims, hidden_features=4
                )
            )

        base_dist = torch.distributions.MultivariateNormal(
            torch.zeros(dims), torch.eye(dims)
        )
        transform = transforms.CompositeTransform(transforms_list)
        bijector = shapeflow.WrapInverseModel(model=transform)

        flow = ftdist.Flow(bijector=bijector, base_dist=base_dist)
        samples = flow.sample([10])
        print(samples.shape)

    def test_shape(self):
        base_dist = torch.distributions.Normal(0, 1)
        print(base_dist.event_shape)
        domain = constraints.independent(constraints.real, len(base_dist.event_shape))
        print(domain.event_dim)


if __name__ == "__main__":
    unittest.main()

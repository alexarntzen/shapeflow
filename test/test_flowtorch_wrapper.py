import unittest

import torch
import torch.distributions.constraints as constraints
import flowtorch.distributions as ftdist
import shapeflow as sf
import nflows.flows


class TestFlowTorchWrapper(unittest.TestCase):
    def test_nflow(self):
        # Set up model
        num_layers = 5
        dims = 2
        hidden_features = 4

        base_dist = torch.distributions.MultivariateNormal(
            torch.zeros(dims), torch.eye(dims)
        )

        # transform = transforms.CompositeTransform(transforms_list)
        bijector = sf.WrapInverseModel(
            get_transform=sf.utils.get_transform_nflow,
            Transform=nflows.flows.MaskedAutoregressiveFlow,
            num_layers=num_layers,
            hidden_features=hidden_features,
            num_blocks_per_layer=2,
        )

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

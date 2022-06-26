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
        bijector = sf.nf.WrapInverseModel(
            params_fn=sf.normalizing_flows.LazyModule(
                get_transform=sf.transforms.get_transform_nflow,
                Transform=nflows.flows.MaskedAutoregressiveFlow,
                num_layers=num_layers,
                hidden_features=hidden_features,
                num_blocks_per_layer=2,
            )
        )

        flow = ftdist.Flow(bijector=bijector, base_dist=base_dist)
        samples = flow.sample([10])
        print(samples.shape)

    def test_normflow(self):
        # Set up model
        num_layers = 2
        dims = 2
        hidden_features = [4] * num_layers
        hidden_layers = [5] * num_layers

        base_dist = torch.distributions.MultivariateNormal(
            torch.zeros(dims), torch.eye(dims)
        )

        # transform = transforms.CompositeTransform(transforms_list)
        composted_bij = sf.nf.get_bijector(
            get_transform=sf.transforms.get_residual_transform,
            compose=True,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
        )
        flow = ftdist.Flow(bijector=composted_bij, base_dist=base_dist)
        samples = flow.sample([10])
        print(samples.shape)

    def test_normflow_2(self):
        # Set up model
        num_layers = 2
        dims = 2
        hidden_features = [4] * num_layers
        hidden_layers = [5] * num_layers

        base_dist = torch.distributions.MultivariateNormal(
            torch.zeros(dims), torch.eye(dims)
        )

        flow = sf.nf.get_flow(
            get_transform=sf.transforms.get_residual_transform,
            base_dist=base_dist,
            compose=True,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
        )
        samples = flow.sample([10])
        print(samples.shape)

    def test_shape(self):
        base_dist = torch.distributions.Normal(0, 1)
        print(base_dist.event_shape)
        domain = constraints.independent(constraints.real, len(base_dist.event_shape))
        print(domain.event_dim)


if __name__ == "__main__":
    unittest.main()

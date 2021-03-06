"""Test that flowtorch wrapper is implemented correctly"""
import unittest

import torch
import flowtorch.distributions as ftdist
import shapeflow as sf


class TestFlowTorchWrapper(unittest.TestCase):
    """"""

    def test_normflow(self):
        """Test flowtorch with resdiual transform from normflow"""
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
        """Test flowtorch with resdiual transform from normflow"""
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


if __name__ == "__main__":
    unittest.main()

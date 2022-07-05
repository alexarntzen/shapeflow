import unittest
import shapeflow.utils as utils

import torch


class TestUtils(unittest.TestCase):
    def test_conversion(self):
        """Test conversion between data and animations is correct"""
        size = (10, 20)
        data = torch.rand(size) * 2 * torch.pi - torch.pi

        # test inverse of forward is correct
        diff = utils.motion_array_to_data(utils.data_to_motion_array(data)) - data

        max_diff = torch.max(torch.abs(diff)).item()

        self.assertAlmostEqual(max_diff, 0, delta=1e-5)


if __name__ == "__main__":
    unittest.main()

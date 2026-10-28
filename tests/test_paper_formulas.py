import unittest

import torch

from algorithms import cmsaw, gcfmv
from algorithms.common import normalized_mv_weights


class PaperFormulaTests(unittest.TestCase):
    def test_inverse_normalization_preserves_denominator_sign(self):
        covariance = torch.diag(torch.tensor([-2.0, -1.0], dtype=torch.complex64))[None]
        steering = torch.ones((1, 2, 1), dtype=torch.complex64)
        weights = normalized_mv_weights(covariance, steering)
        torch.testing.assert_close(weights.real.squeeze(), torch.tensor([1 / 3, 2 / 3]))
        torch.testing.assert_close(steering.mH @ weights, torch.ones((1, 1, 1), dtype=torch.complex64))

    def test_singular_loaded_matrix_uses_pseudoinverse(self):
        floor = 20 * torch.finfo(torch.float32).eps
        covariance = torch.diag(torch.tensor([-floor, 1.0], dtype=torch.complex64))[None]
        steering = torch.ones((1, 2, 1), dtype=torch.complex64)
        weights = normalized_mv_weights(covariance, steering)
        torch.testing.assert_close(weights.squeeze(), torch.tensor([0.0, 1.0], dtype=torch.complex64))

    def test_joint_covariance_divides_by_actual_fractional_weight_sum(self):
        data = torch.tensor([[[[1.0], [2.0]], [[0.0], [0.0]]]], dtype=torch.complex64)
        valid = torch.tensor([[[[True], [True]], [[False], [False]]]])
        covariance = gcfmv._joint_spatial_covariance(
            data, valid, 2, 2, False, torch.tensor([0.25]),
        )
        expected = torch.tensor([[[0.25, 0.5], [0.5, 1.0]]], dtype=torch.complex64)
        torch.testing.assert_close(covariance, expected)
        empty = gcfmv._joint_spatial_covariance(
            data * 0, valid & False, 2, 2, False, torch.tensor([0.25]),
        )
        torch.testing.assert_close(empty, torch.zeros_like(expected))

    def test_cmsaw_default_keeps_raw_paper_msr(self):
        options = cmsaw.parser.parse_args([])
        self.assertEqual(options.gamma, 1.0)
        self.assertEqual(options.clip_percentile, 100.0)
        data = torch.tensor([[[[1.0, 2.0, 4.0, 8.0]]]], dtype=torch.complex64)
        weight = cmsaw.cmsaw_weight_from_delayed_data(
            data, [1.0], [0.0], 1.0, 1.5, False, "rect",
            0.5, 2, 0.5, options.gamma, options.clip_percentile, 1,
        )
        torch.testing.assert_close(weight, torch.tensor([[3.0]]))


if __name__ == "__main__":
    unittest.main()

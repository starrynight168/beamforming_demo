import unittest

import torch

from algorithms import ccmv, cmsaw, gcfmv
from algorithms.common import normalized_mv_weights


class PaperFormulaTests(unittest.TestCase):
    def test_cross_subarray_covariance_matches_explicit_full_pairs(self):
        generator = torch.Generator().manual_seed(13)
        data = torch.randn((2, 6, 3), dtype=torch.complex128, generator=generator)
        subarrays = data.unfold(1, 3, 1).permute(0, 3, 1, 2)
        for weights in (None, torch.tensor([0.0, 0.25, 1.0, 0.5], dtype=torch.float64),
                        torch.full((4,), 1e-12, dtype=torch.float64)):
            b = torch.ones(4, dtype=torch.float64) if weights is None else weights
            expected = torch.zeros((2, 3, 3), dtype=data.dtype)
            denominator = 0.0
            for i in range(4):
                for j in range(4):
                    expected += b[i] * b[j] * (subarrays[:, :, i] @ subarrays[:, :, j].mH)
                    denominator += b[i] * b[j] * 3
            expected /= denominator
            for fbss in (False, True):
                raw, actual = ccmv.cross_subarray_covariance(data, 3, 3, fbss, weights)
                reference = 0.5 * (expected + expected.conj().flip(-1, -2)) if fbss else expected
                torch.testing.assert_close(raw, subarrays)
                torch.testing.assert_close(actual, reference)
                torch.testing.assert_close(actual, actual.mH)
                self.assertGreaterEqual(torch.linalg.eigvalsh(actual).min().item(), -1e-12)

    def test_cross_subarray_covariance_rejects_undefined_estimates(self):
        data = torch.ones((1, 4, 3), dtype=torch.complex64)
        cases = ((5, 3, None), (2, 1, None), (2, 3, torch.ones(2)),
                 (2, 3, torch.zeros(3)),
                 (2, 3, torch.tensor([1.0, -1.0, 1.0])))
        for length, temporal_win, weights in cases:
            with self.subTest(length=length, temporal_win=temporal_win, weights=weights):
                with self.assertRaises(ValueError):
                    ccmv.cross_subarray_covariance(data, length, temporal_win, False, weights)

    def test_cross_covariance_accepts_a_single_effective_subarray(self):
        data = torch.tensor([[[1.0], [2.0], [3.0]]], dtype=torch.complex64)
        _, actual = ccmv.cross_subarray_covariance(data, 3, 1, False)
        torch.testing.assert_close(actual, data @ data.mH)
        _, actual = ccmv.cross_subarray_covariance(data, 2, 1, False, torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(actual, data[:, :2] @ data[:, :2].mH)

    def test_ccmv_explicit_loading_bounds_low_rank_weights(self):
        loading = 0.1
        data = torch.randn((64, 16, 1), dtype=torch.complex64,
                           generator=torch.Generator().manual_seed(71))
        _, covariance = ccmv.cross_subarray_covariance(data, 8, 1, True)
        trace = covariance.diagonal(dim1=-2, dim2=-1).real.sum(-1)
        loaded = covariance + (loading / 8) * trace[:, None, None] * torch.eye(8)
        steering = torch.ones((64, 8, 1), dtype=torch.complex64)
        weights = normalized_mv_weights(loaded, steering)
        torch.testing.assert_close(steering.mH @ weights, torch.ones((64, 1, 1), dtype=torch.complex64))
        self.assertTrue(torch.isfinite(weights).all())
        condition_bound = 1 + 8 / loading
        self.assertLessEqual(weights.abs().sum(-2).square().max().item(), condition_bound)

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

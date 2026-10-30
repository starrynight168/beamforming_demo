import unittest
from unittest.mock import patch

import numpy as np
import torch
from scipy.signal import filtfilt, firwin, hilbert, resample_poly

from algorithms import mv
from algorithms.shared_core.signals import rf_to_analytic, rf_to_iq


class SignalConversionTests(unittest.TestCase):
    def test_iq_matches_existing_packer_formula_including_decimation(self):
        rf = np.random.default_rng(17).normal(size=(2, 512, 4)).astype(np.float32)
        fs, fc, t0 = 20e6, 5e6, 0.8e-6
        coefficients = firwin(16, 0.4)
        oscillator = np.exp(-1j * 2 * np.pi * fc * (t0 + np.arange(512) / fs)).astype(np.complex64)
        mixed = rf * oscillator[None, :, None]
        reference = [filtfilt(coefficients, 1, component, axis=1).astype(np.float32)
                     for component in (mixed.real, mixed.imag)]
        for factor in (1, 2):
            with self.subTest(decimation=factor):
                actual = rf_to_iq(rf, fs, fc, t0, coefficients, factor)
                for output, component in zip(actual, reference):
                    expected = resample_poly(component, up=1, down=factor, axis=1, window=("kaiser", 12.0)).astype(np.float32)
                    np.testing.assert_array_equal(output, expected)

    def test_each_angle_uses_its_own_time_origin(self):
        rf = np.random.default_rng(19).normal(size=(2, 128, 2)).astype(np.float32)
        coefficients = firwin(8, 0.3)
        starts = [0.0, 0.17e-6]
        actual = rf_to_iq(rf, 20e6, 5e6, starts, coefficients)
        for angle, t0 in enumerate(starts):
            expected = rf_to_iq(rf[angle:angle + 1], 20e6, 5e6, t0, coefficients)
            for output, component in zip(actual, expected):
                np.testing.assert_array_equal(output[angle:angle + 1], component)

    def test_analytic_uses_full_time_axis_and_preserves_real_rf(self):
        rf = np.random.default_rng(23).normal(size=(2, 257, 4)).astype(np.float32)
        real, imag = rf_to_analytic(rf)
        np.testing.assert_array_equal(real, rf)
        np.testing.assert_array_equal(imag, hilbert(rf, axis=1).imag)
        self.assertEqual(imag.dtype, np.float32)

    def test_rejects_complex_source_and_invalid_sampling(self):
        source = np.ones((1, 128, 2), dtype=np.float32)
        coefficients = firwin(8, 0.3)
        with self.assertRaisesRegex(ValueError, "real"):
            rf_to_analytic(source.astype(np.complex64))
        with self.assertRaisesRegex(ValueError, "fs/2"):
            rf_to_iq(source, 20e6, 12e6, 0.0, coefficients)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            rf_to_iq(source, 20e6, 5e6, 0.0, coefficients, 1.5)


class MVSignalDomainTests(unittest.TestCase):
    def beamformer(self, domain=None, *, fc=0.25, n_elem=1, angles=(0.0,), **options):
        settings = mv.parser.parse_args([])
        settings.dynamic_aperture = False
        settings.interp = "linear"
        settings.window = "rect"
        vars(settings).update(options)
        arguments = {} if domain is None else {"signal_domain": domain}
        with patch.object(mv, "device", torch.device("cpu")):
            return mv.RowDynamicMVBeamformerIQ(
                np.array([0.75], dtype=np.float32),
                np.array([0.0], dtype=np.float32),
                n_elem, 0.1, 1.0, fc, 8.0, np.zeros(len(angles)),
                np.array(angles), mv_dl=0.1, subarray_ratio=0.5,
                temporal_win=1, options=settings, **arguments,
            )

    def test_default_iq_is_identical_to_explicit_iq(self):
        rng = np.random.default_rng(29)
        i_data = rng.normal(size=(2, 64, 4)).astype(np.float32)
        q_data = rng.normal(size=i_data.shape).astype(np.float32)
        angles = np.array([-0.05, 0.05])
        implicit = self.beamformer(n_elem=4, angles=angles)
        explicit = self.beamformer("iq", n_elem=4, angles=angles)
        for left, right in zip(implicit(i_data, q_data, angles, 0.0, 8.0), explicit(i_data, q_data, angles, 0.0, 8.0)):
            np.testing.assert_array_equal(left, right)

    def test_rf_and_analytic_rf_do_not_apply_iq_carrier_compensation(self):
        fc, fs, t0 = 0.25, 8.0, 0.125
        time = t0 + np.arange(64) / fs
        phase = 2 * np.pi * fc * time
        real = np.cos(phase).astype(np.float32)[None, :, None]
        imag = np.sin(phase).astype(np.float32)[None, :, None]
        expected = np.exp(1j * 2 * np.pi * fc * 1.5)
        rf_i, rf_q = self.beamformer("rf")(real, np.zeros_like(real), np.zeros(1), t0, fs)
        analytic_i, analytic_q = self.beamformer("analytic_rf")(real, imag, np.zeros(1), t0, fs)
        iq_i, iq_q = self.beamformer("iq")(np.ones_like(real), np.zeros_like(real), np.zeros(1), t0, fs)
        np.testing.assert_allclose(rf_i, expected.real, atol=2e-6)
        np.testing.assert_array_equal(rf_q, np.zeros_like(rf_q))
        np.testing.assert_allclose(analytic_i + 1j * analytic_q, expected, atol=2e-6)
        np.testing.assert_allclose(iq_i + 1j * iq_q, expected, atol=2e-6)

    def test_real_rf_uses_same_mv_solution_when_no_carrier_rotation_is_needed(self):
        source = np.random.default_rng(31).normal(size=(1, 64, 4)).astype(np.float32)
        zero = np.zeros_like(source)
        iq = self.beamformer("iq", fc=0.0, n_elem=4)(source, zero, np.zeros(1), 0.0, 8.0)
        rf = self.beamformer("rf", fc=0.0, n_elem=4)(source, zero, np.zeros(1), 0.0, 8.0)
        for left, right in zip(iq, rf):
            np.testing.assert_array_equal(left, right)

    def test_invalid_domain_and_nonzero_real_rf_imaginary_input_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "signal_domain"):
            self.beamformer("unknown")
        source = np.ones((1, 64, 1), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "all-zero"):
            self.beamformer("rf")(source, source, np.zeros(1), 0.0, 8.0)

    def test_all_axes_apply_carrier_compensation_only_to_iq(self):
        angles = np.array([-np.arccos(2 / 3), np.arccos(2 / 3)], dtype=np.float32)
        t0, fs, fc = .125, 8., .25
        phase = 2 * np.pi * fc * (t0 + np.arange(64) / fs)
        real = np.broadcast_to(np.cos(phase)[None, :, None], (2, 64, 1)).astype(np.float32).copy()
        imag = np.broadcast_to(np.sin(phase)[None, :, None], real.shape).astype(np.float32).copy()
        expected = np.exp(2j * np.pi * fc * 1.25)
        for axis in ("rx", "tx", "joint"):
            with self.subTest(axis=axis):
                rf = self.beamformer("rf", angles=angles, mv_axis=axis)(real, np.zeros_like(real), angles, t0, fs)
                analytic = self.beamformer("analytic_rf", angles=angles, mv_axis=axis)(real, imag, angles, t0, fs)
                iq = self.beamformer("iq", angles=angles, mv_axis=axis)(np.ones_like(real), np.zeros_like(real), angles, t0, fs)
                np.testing.assert_allclose(rf[0], expected.real, atol=2e-5)
                np.testing.assert_array_equal(rf[1], 0.)
                np.testing.assert_allclose(analytic[0] + 1j * analytic[1], expected, atol=2e-5)
                np.testing.assert_allclose(iq[0] + 1j * iq[1], expected, atol=2e-5)

    def test_tx_and_joint_still_require_multiple_acquisitions(self):
        real = np.ones((1, 64, 1), np.float32)
        for axis in ("tx", "joint"):
            for domain in ("rf", "analytic_rf", "iq"):
                with self.subTest(axis=axis, domain=domain), self.assertRaisesRegex(ValueError, "at least two"):
                    self.beamformer(domain, mv_axis=axis)(real, np.zeros_like(real), np.zeros(1), 0., 8.)

    def test_zero_imaginary_mv_matches_independent_real_solve(self):
        rng = np.random.default_rng(57)
        samples = rng.normal(size=(3, 40))
        covariance = samples @ samples.T / samples.shape[1]
        ones = np.ones((3, 1))
        expected = np.linalg.solve(covariance, ones)
        expected /= ones.T @ expected
        actual = mv.normalized_mv_weights(torch.tensor(covariance[None], dtype=torch.complex64), torch.ones((1, 3, 1), dtype=torch.complex64))
        np.testing.assert_allclose(actual.real.numpy()[0], expected, atol=2e-6)
        np.testing.assert_array_equal(actual.imag.numpy(), 0.)
        np.testing.assert_allclose((actual[0].mH @ torch.tensor(samples, dtype=torch.complex64)).real.numpy(), expected.T @ samples, atol=2e-6)


if __name__ == "__main__":
    unittest.main()

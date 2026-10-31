import argparse
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import h5py
import numpy as np
import torch

from algorithms import ccmv, cmsaw, das, esbmv, fdmas, gcfmv, mban, mv, template_algorithm
from algorithms.common import (
    WINDOW_CHOICES,
    aperture_start_indices,
    algorithm_source_paths,
    envelope_to_db,
    generalized_coherence_factor,
    merge_common_params,
    mv_row_output,
    normalized_mv_weights,
    spatial_temporal_covariance,
)
from algorithms.gcfmv import generalized_coherence_factor_2d


class BeamformerRegressionTests(unittest.TestCase):
    modules = (das, mv, ccmv, esbmv, gcfmv, fdmas, cmsaw)

    def options(self, module, **changes):
        options = module.parser.parse_args([])
        options.dynamic_aperture = False
        options.tgc = False
        options.interp = "linear"
        options.window = "rect"
        vars(options).update(changes)
        return options

    def beamformer(self, module, options, z=(1.0,), x=(0.0,), channels=8, **changes):
        cls = (module.DASBeamformerIQ if module is das else
               module.FDMASBeamformerIQ if module is fdmas else
               module.CMSAWBeamformerIQ if module is cmsaw else
               module.RowDynamicCCMVBeamformerIQ if module is ccmv else
               module.RowDynamicMVBeamformerIQ)
        parameters = dict(options=options)
        if module in (mv, ccmv, esbmv, gcfmv, cmsaw):
            parameters.update(temporal_win=1, subarray_ratio=0.5, mv_dl=0.0)
        parameters.update(changes)
        return cls(np.array(z), np.array(x), channels, 1.0, 1.0, 0.0, 1.0,
                   np.zeros(1), np.zeros(1), **parameters)

    def test_constructors_work_without_global_cli_args(self):
        for module in self.modules:
            with self.subTest(module=module.METHOD_NAME), patch.object(module, "args", None):
                self.beamformer(module, None)

    def test_ccmv_reuses_mv_parameters_and_dynamic_geometry(self):
        self.assertIs(ccmv.RowDynamicCCMVBeamformerIQ.__call__, mv.RowDynamicMVBeamformerIQ.__call__)
        expected = vars(mv.parser.parse_args([])).copy()
        actual = vars(ccmv.parser.parse_args([])).copy()
        self.assertEqual(expected.pop("mv_dl"), 0.0)
        self.assertEqual(actual.pop("mv_dl"), 0.1)
        self.assertEqual(actual, expected)
        geometry = (np.array([0.002, 0.012, 0.03]), np.array([-0.001, 0.0, 0.001]),
                    128, 0.0003, 1540.0, 5e6, 20e6, np.zeros(1), np.zeros(1))
        parameters = {"mv_dl": 0.1, **{name: actual[name] for name in ("fbss", "subarray_ratio", "temporal_win")}}
        baseline = mv.RowDynamicMVBeamformerIQ(*geometry, options=mv.parser.parse_args([]), **parameters)
        cross = ccmv.RowDynamicCCMVBeamformerIQ(*geometry, options=ccmv.parser.parse_args([]), **parameters)
        sizes = []
        for left, right in zip(baseline.row_cache, cross.row_cache):
            sizes.append(right["aperture_size"])
            for name in ("aperture_size", "subarray_size", "subarray_count"):
                self.assertEqual(left[name], right[name])
            torch.testing.assert_close(left["ch_idx_t"], right["ch_idx_t"])
        self.assertGreater(len(set(sizes)), 1)

    def test_config_rejects_unsupported_default_choices(self):
        for name in ("window", "interp"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                merge_common_params({"params": {name: "unsupported"}})

    def test_fixed_geometry_das_uses_full_physical_aperture(self):
        for window in WINDOW_CHOICES:
            geometry = self.beamformer(
                das, self.options(das, aperture_mode="geometry", window=window), x=(-2.0, 2.0),
            )
            discrete = self.beamformer(
                das, self.options(das, aperture_mode="discrete", window=window), x=(-2.0, 2.0),
            )
            with self.subTest(window=window):
                torch.testing.assert_close(geometry.window, discrete.window)
                self.assertTrue((geometry.window >= 0).all())

    def test_coherent_signal_gain_is_preserved_with_each_window(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        signal = np.ones((1, 16, 8), dtype=np.float32)
        for device in devices:
            with contextlib.ExitStack() as stack:
                for module in self.modules:
                    stack.enter_context(patch.object(module, "device", device))
                for module in self.modules:
                    for window in WINDOW_CHOICES:
                        with self.subTest(device=device, module=module.METHOD_NAME, window=window):
                            bf = self.beamformer(module, self.options(module, window=window))
                            i, q = bf(signal, np.zeros_like(signal), np.zeros(1), np.zeros(1), 1.0)
                            np.testing.assert_allclose(i, 1.0, atol=2e-4)
                            np.testing.assert_allclose(q, 0.0, atol=1e-6)

    def test_mv_methods_preserve_remaining_valid_channels_at_sample_boundary(self):
        signal = np.ones((1, 3, 3), dtype=np.float32)
        for module in (mv, esbmv, gcfmv):
            with self.subTest(module=module.METHOD_NAME):
                options = self.options(module)
                if module is gcfmv:
                    options.gcf_power = 0.0
                bf = self.beamformer(module, options, z=(1.0, 2.0), x=(1.0,),
                                     channels=3, subarray_ratio=1.0)
                i, q = bf(signal, np.zeros_like(signal), np.zeros(1), np.zeros(1), 1.0)
                np.testing.assert_allclose(i[:, 0], [1.0, 0.0], atol=1e-6)
                np.testing.assert_allclose(q, 0.0, atol=1e-6)

    def test_input_angle_channel_shape_and_sampling_frequency_are_checked(self):
        signal = np.ones((1, 16, 8), dtype=np.float32)
        for module in self.modules:
            bf = self.beamformer(module, self.options(module))
            for angles, samples, frequency in (([0.0, 1.0], signal, 1.0),
                                                ([0.0], signal[..., :4], 1.0),
                                                ([0.0], signal, 2.0)):
                with self.subTest(module=module.METHOD_NAME), self.assertRaises(ValueError):
                    bf(samples, np.zeros_like(samples), np.array(angles), np.zeros(1), frequency)

    def test_mv_fallback_keeps_valid_channels_at_zero_taper_weights(self):
        data = torch.tensor([[[3 + 2j], [0], [0]]], dtype=torch.complex64)
        valid = torch.tensor([[[1.0], [0.0], [0.0]]])
        row = dict(window=None, aperture_window=torch.tensor([0.0, 1.0, 0.0]),
                   ch_idx_t=torch.arange(3).view(1, 3, 1))
        subarrays = data.unfold(1, 3, 1).permute(0, 3, 1, 2)
        weights = torch.full((1, 3, 1), 1 / 3, dtype=torch.complex64)
        output = mv_row_output(weights, subarrays, data, valid, row)
        torch.testing.assert_close(output.squeeze(), torch.tensor(3 + 2j))

    def test_mv_solver_is_scale_invariant_and_regularizes_rank_deficiency(self):
        steering = torch.ones((1, 2, 1), dtype=torch.complex64)
        covariance = torch.diag(torch.tensor([2.0, 1.0], dtype=torch.complex64))[None]
        for scale in (1e-30, 1.0, 1e30):
            weights = normalized_mv_weights(covariance * scale, steering)
            torch.testing.assert_close(weights.real.squeeze(), torch.tensor([1 / 3, 2 / 3]))
        steering = torch.ones((2, 8, 1), dtype=torch.complex64)
        covariance = torch.ones((2, 8, 8), dtype=torch.complex64)
        covariance[1].zero_()
        weights = normalized_mv_weights(covariance, steering)
        self.assertTrue(torch.isfinite(weights).all())
        torch.testing.assert_close(weights[0].sum(), torch.tensor(1.0 + 0j))
        torch.testing.assert_close(weights[1], torch.zeros_like(weights[1]))

    def test_weighted_covariance_is_normalized_and_preserves_snapshot_data(self):
        data = torch.tensor([[[1, 2], [3, 4], [2, 1], [5, 3]]], dtype=torch.complex64)
        spatial_weights = torch.tensor([0.0, 1.0, 0.5])
        for scale in (1e-20, 1.0, 1e20):
            sub, actual = spatial_temporal_covariance(
                data * scale, 2, 2, False, spatial_weights=spatial_weights, normalize=True,
            )
            torch.testing.assert_close(sub / scale, data.unfold(1, 2, 1).permute(0, 3, 1, 2))
            snapshots = (data / 5).unfold(1, 2, 1).permute(0, 3, 1, 2).reshape(1, 2, -1)
            weights = spatial_weights.repeat_interleave(2)
            expected = (snapshots * weights) @ snapshots.mH / weights.sum()
            torch.testing.assert_close(actual, expected)

    def test_esbmv_projection_keeps_attenuation_and_relative_threshold(self):
        covariance = torch.diag(torch.tensor([10.0, 1.0], dtype=torch.complex64))[None]
        weights = torch.tensor([[[1 / 11], [10 / 11]]], dtype=torch.complex64)
        expected = torch.tensor([[[1 / 11], [0.0]]], dtype=torch.complex64)
        for scale in (1e-20, 1.0, 1e20):
            torch.testing.assert_close(esbmv.project_signal_subspace(covariance * scale, weights, 0, 0.5), expected)
        torch.testing.assert_close(esbmv.project_signal_subspace(covariance, weights, 1, 0.0), expected)
        torch.testing.assert_close(esbmv.project_signal_subspace(covariance, weights, 2, 0.0), weights)

    def test_gcf_uses_full_aperture_and_includes_even_length_nyquist_bin(self):
        self.assertEqual(gcfmv.parser.parse_args([]).gcf_power, 1.0)
        data = torch.tensor([[1, -1]], dtype=torch.complex64)
        torch.testing.assert_close(generalized_coherence_factor(data, 0), torch.zeros(1))
        torch.testing.assert_close(generalized_coherence_factor(data, 1), torch.ones(1))
        for scale in (1e-20, 1.0, 1e20):
            coherent = torch.full((1, 8), scale, dtype=torch.complex64)
            torch.testing.assert_close(generalized_coherence_factor(coherent, 0), torch.ones(1))
        channels = np.array([1, 2, 0, 1, 3, 1, 0, 2], dtype=np.float32)
        signal = np.broadcast_to(channels, (1, 16, 8)).copy()
        baseline = self.beamformer(mv, self.options(mv))
        weighted = self.beamformer(gcfmv, self.options(gcfmv, gcf_low_bins=0, gcf_power=1.0))
        i_mv, _ = baseline(signal, np.zeros_like(signal), np.zeros(1), np.zeros(1), 1.0)
        i_gcf, _ = weighted(signal, np.zeros_like(signal), np.zeros(1), np.zeros(1), 1.0)
        expected = channels.sum() ** 2 / (8 * np.sum(channels ** 2))
        np.testing.assert_allclose(i_gcf, i_mv * expected, atol=1e-6)

    def test_2d_gcf_uses_the_centered_tx_rx_frequency_rectangle(self):
        coherent = torch.ones((2, 4, 8), dtype=torch.complex64)
        torch.testing.assert_close(generalized_coherence_factor_2d(coherent, 0, 0), torch.ones(2))
        alternating = torch.tensor([[1, -1], [-1, 1]], dtype=torch.complex64)
        torch.testing.assert_close(
            generalized_coherence_factor_2d(alternating[None], 0, 0), torch.zeros(1),
        )
        factors = generalized_coherence_factor_2d(torch.randn(3, 5, 7, dtype=torch.complex64), 1, 2)
        self.assertTrue(torch.isfinite(factors).all())
        self.assertTrue(((factors >= 0) & (factors <= 1)).all())

    def test_joint_mv_applies_receive_and_transmit_weights(self):
        values = np.array([[1, 2, 4, 8], [3, 5, 7, 11], [13, 17, 19, 23]], dtype=np.float32)
        signal = np.broadcast_to(values[:, None], (3, 16, 4)).copy()
        angles = np.array([-0.1, 0.0, 0.1], dtype=np.float32)
        rx = torch.tensor([[[0.2], [0.8]]], dtype=torch.complex64)
        tx = torch.tensor([[[0.3], [0.7]]], dtype=torch.complex64)
        with patch.object(gcfmv, "device", torch.device("cpu")), patch.object(
            mv, "device", torch.device("cpu")
        ):
            bf = self.beamformer(gcfmv, self.options(gcfmv, mv_axis="joint", gcf_power=0.0), channels=4)
            with patch.object(mv, "normalized_mv_weights", side_effect=(rx, tx)):
                i, q = bf(signal, np.zeros_like(signal), angles, np.zeros(3), 1.0)
            baseline = self.beamformer(mv, self.options(mv, mv_axis="joint"), channels=4)
            with patch.object(mv, "normalized_mv_weights", side_effect=(rx, tx)):
                i_mv, q_mv = baseline(signal, np.zeros_like(signal), angles, np.zeros(3), 1.0)
        rx_values = 0.2 * values[:, :-1] + 0.8 * values[:, 1:]
        expected = (0.3 * rx_values[:-1] + 0.7 * rx_values[1:]).mean()
        np.testing.assert_allclose(i, expected, rtol=1e-6)
        np.testing.assert_allclose(q, 0.0, atol=1e-6)
        np.testing.assert_allclose(i, i_mv, rtol=1e-6)
        np.testing.assert_allclose(q, q_mv, atol=1e-6)

    def test_discrete_apertures_center_even_windows_on_pixels(self):
        x_grid = (-0.25, 0.0, 0.25)
        for module in self.modules:
            if module is cmsaw:
                continue
            with self.subTest(module=module.METHOD_NAME):
                options = self.options(module, dynamic_aperture=True, f_number=1.0)
                with patch.object(module, "device", torch.device("cpu")):
                    beamformer = self.beamformer(
                        module, options, z=(1.0,), x=x_grid, channels=4
                    )
                for pixel in range(len(x_grid)):
                    if module in (das, fdmas):
                        channels = torch.where(beamformer.window[0, pixel] > 0)[0]
                    else:
                        channels = beamformer.row_cache[0]["ch_idx_t"][pixel, :, 0]
                    self.assertEqual(channels.cpu().tolist(), [1, 2])

        data = torch.tensor(
            [[[[1, 2, 3, 4], [2, 4, 1, 3], [3, 1, 4, 2]]]],
            dtype=torch.complex64,
        )
        starts = []
        with patch.object(
            cmsaw,
            "aperture_start_indices",
            side_effect=lambda x, n, k, pitch: starts.append(
                aperture_start_indices(x, n, k, pitch)
            ) or starts[-1],
        ):
            weights = cmsaw.cmsaw_weight_from_delayed_data(
                data,
                np.array([1.0]),
                np.array(x_grid),
                1.0,
                1.0,
                True,
                "rect",
                0.5,
                2,
                1.0,
                0.4,
                90.0,
                1,
            )
        self.assertEqual(starts[0].cpu().tolist(), [1, 1, 1])
        self.assertTrue(torch.isfinite(weights).all())

    def test_rx_gcf_compounds_independent_angles_without_switching_method(self):
        rng = np.random.default_rng(71)
        signal = (rng.normal(size=(3, 16, 8)) + 1j * rng.normal(size=(3, 16, 8))).astype(np.complex64)
        angles = np.array([-0.17, 0.0, 0.11], dtype=np.float32)
        bf = self.beamformer(gcfmv, self.options(gcfmv))
        combined = bf(signal.real, signal.imag, angles, np.zeros(3), 1.0)
        separate = [bf(signal[i:i+1].real, signal[i:i+1].imag, angles[i:i+1], np.zeros(1), 1.0)
                    for i in range(3)]
        for actual, expected in zip(combined, np.mean(separate, axis=0)):
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_tx_mv_uses_receive_das_then_transmit_subarray_average(self):
        values = np.array([[1, 2, 4, 8], [3, 5, 7, 11], [13, 17, 19, 23]], dtype=np.float32)
        signal = np.broadcast_to(values[:, None], (3, 16, 4)).copy()
        angles = np.array([-0.1, 0.0, 0.1], dtype=np.float32)
        tx = torch.tensor([[[0.3], [0.7]]], dtype=torch.complex64)
        with patch.object(gcfmv, "device", torch.device("cpu")):
            for window in ("rect", "hann"):
                for power in (0.0, 1.0):
                    bf = self.beamformer(gcfmv, self.options(gcfmv, mv_axis="tx", window=window,
                        gcf_power=power, gcf_tx_low_bins=0), channels=4)
                    with patch.object(mv, "normalized_mv_weights", return_value=tx) as solve:
                        i, q = bf(signal, np.zeros_like(signal), angles, np.zeros(3), 1.0)
                    self.assertEqual(solve.call_count, 1)
                    rx_window = bf.row_cache[0]["aperture_window"]
                    aperture = np.ones(4) if rx_window is None else rx_window.numpy()
                    received = (values * aperture).sum(-1) / aperture.sum()
                    outputs = 0.3 * received[:-1] + 0.7 * received[1:]
                    gcf = outputs.sum() ** 2 / (2 * np.sum(outputs ** 2))
                    np.testing.assert_allclose(i, outputs.mean() * gcf ** power, rtol=1e-6)
                    np.testing.assert_allclose(q, 0.0, atol=1e-6)

    def test_tx_and_joint_reject_single_angle_instead_of_changing_method(self):
        signal = np.ones((1, 16, 8), dtype=np.float32)
        for axis in ("tx", "joint"):
            bf = self.beamformer(gcfmv, self.options(gcfmv, mv_axis=axis))
            with self.assertRaisesRegex(ValueError, "at least two"):
                bf(signal, np.zeros_like(signal), np.zeros(1), np.zeros(1), 1.0)

    def test_mv_variants_share_all_paths_and_preserve_coherent_signal(self):
        signal = np.ones((3, 64, 8), dtype=np.float32)
        angles = np.array([-0.1, 0.0, 0.1], dtype=np.float32)
        for module in (mv, ccmv, esbmv, gcfmv, cmsaw):
            for axis in ("rx", "tx", "joint"):
                with self.subTest(algorithm=module.METHOD_NAME, axis=axis):
                    bf = self.beamformer(module, self.options(module, mv_axis=axis), mv_dl=0.1)
                    engine = bf.mv_bf if module is cmsaw else bf
                    self.assertIs(type(engine).__call__, mv.RowDynamicMVBeamformerIQ.__call__)
                    i, q = bf(signal, np.zeros_like(signal), angles, np.zeros(3), 1.0)
                    np.testing.assert_allclose(i, 1, atol=3e-4)
                    np.testing.assert_allclose(q, 0.0, atol=1e-6)

    def test_esbmv_projects_weights_on_both_axes_in_joint_mode(self):
        signal = np.ones((3, 64, 8), dtype=np.float32)
        angles = np.array([-0.1, 0.0, 0.1], dtype=np.float32)
        bf = self.beamformer(esbmv, self.options(esbmv, mv_axis="joint"), mv_dl=0.1)
        with patch.object(esbmv, "project_signal_subspace", wraps=esbmv.project_signal_subspace) as project:
            bf(signal, np.zeros_like(signal), angles, np.zeros(3), 1.0)
        self.assertEqual([call.args[0].shape[-1] for call in project.call_args_list], [4, 2])

    def test_joint_mv_is_independent_of_column_block_size(self):
        rng = np.random.default_rng(37)
        signal = (rng.normal(size=(3, 16, 8)) + 1j * rng.normal(size=(3, 16, 8))).astype(np.complex64)
        angles = np.array([-0.1, 0.0, 0.1], dtype=np.float32)
        with patch.object(gcfmv, "device", torch.device("cpu")):
            bf = self.beamformer(gcfmv, self.options(gcfmv, mv_axis="joint"), x=np.linspace(-0.3, 0.3, 40),
                                 temporal_win=3)
            with patch.object(mv, "_JOINT_SNAPSHOT_TARGET_BYTES", 1):
                blocked = bf(signal.real, signal.imag, angles, np.zeros(3), 1.0)
            whole = bf(signal.real, signal.imag, angles, np.zeros(3), 1.0)
        for actual, expected in zip(blocked, whole):
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_joint_covariance_matches_valid_weighted_snapshots(self):
        generator = torch.Generator().manual_seed(47)
        data = torch.randn((2, 3, 4, 3), generator=generator, dtype=torch.complex64)
        valid = torch.ones_like(data, dtype=torch.bool)
        valid[0, 0, 0, 0] = False
        valid[1, 2, 3, 2] = False
        data = data * valid
        scaled = data / data.abs().amax(dim=(1, 2, 3), keepdim=True)
        for axis, weights in ((1, torch.tensor([0.25, 1.0])),
                              (2, torch.tensor([1.0, 0.2, 0.7]))):
            expected = torch.zeros((2, 2, 2), dtype=torch.complex64)
            totals = torch.zeros(2)
            for pixel in range(2):
                for start, weight in enumerate(weights):
                    for outer in range(data.shape[3 - axis]):
                        for time in range(3):
                            index = (pixel, slice(start, start + 2), outer, time) if axis == 1 else (pixel, outer, slice(start, start + 2), time)
                            if valid[index].all():
                                vector = scaled[index]
                                expected[pixel] += weight * vector[:, None] * vector.conj()[None, :]
                                totals[pixel] += weight
            expected /= totals[:, None, None]
            for fbss in (False, True):
                actual = mv.joint_spatial_covariance(data, valid, axis, 2, fbss, weights)
                target = 0.5 * (expected + expected.conj().flip(-1, -2)) if fbss else expected
                torch.testing.assert_close(actual, target)

    def test_cmsaw_sparse_signal_survives_percentile_clipping(self):
        data = torch.zeros((1, 4, 20, 8), dtype=torch.complex64)
        data[0, 1, 10] = 1.0
        for scale in (1e-20, 1.0, 1e20):
            weights = cmsaw.cmsaw_weight_from_delayed_data(
                data * scale, np.arange(1, 5), np.linspace(-2, 2, 20), 1.0, 1.5,
                False, "rect", 0.5, 2, 1.0, 0.5, 90.0, 1,
            )
            self.assertTrue(torch.isfinite(weights).all())
            self.assertGreater(weights[1, 10].item(), 0.0)
            self.assertEqual(torch.count_nonzero(weights).item(), 1)

    def test_cmsaw_retains_phase_cancellation_in_covariance_statistics(self):
        data = torch.ones((1, 1, 2, 8), dtype=torch.complex64)
        data[0, 0, 1] = torch.tensor([1, 1j, -1, -1j] * 2)
        weight = cmsaw.cmsaw_weight_from_delayed_data(
            data, [1.0], [-1.0, 1.0], 1.0, 1.5,
            False, "rect", 0.5, 4, 1.0, 1.0, 100.0, 1,
        )
        self.assertGreater(weight[0, 0].item(), 1.5 * weight[0, 1].item())

    def test_fdmas_pair_formula_preserves_amplitude_and_doubles_phase(self):
        phase = 0.7
        for scale in (1e-20, 1.0, 1e20):
            z = torch.full((1, 4), scale * np.exp(1j * phase), dtype=torch.complex64)
            weights = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
            output = fdmas.FDMASBeamformerIQ._complex_fdmas(z.real, z.imag, weights)
            torch.testing.assert_close(output / scale, torch.tensor([np.exp(2j * phase)], dtype=torch.complex64))
        zero = torch.zeros((1, 1))
        torch.testing.assert_close(fdmas.FDMASBeamformerIQ._complex_fdmas(zero, zero, torch.ones_like(zero)), zero.to(torch.complex64).squeeze(-1))

    def test_fdmas_rf_filter_selects_second_harmonic_and_checks_sampling_rate(self):
        fs, fc, count = 40e6, 5e6, 1024
        z = 0.005 + np.arange(count, dtype=np.float32) * 1540.0 / (2 * fs)
        options = self.options(fdmas, signal_domain="rf")
        bf = fdmas.FDMASBeamformerIQ(
            z, np.array([0.0]), 4, 0.0003, 1540.0, fc, fs,
            np.zeros(1), np.zeros(1), options=options,
        )
        t = np.arange(count) / fs
        signal = np.cos(2 * np.pi * 2 * fc * t) + np.cos(2 * np.pi * 0.5 * fc * t)
        filtered = bf._filter_rf_output(signal[:, None])[128:-128, 0]
        time = t[128:-128]
        passband_amplitude = abs(2 * np.mean(filtered * np.exp(-2j * np.pi * 2 * fc * time)))
        stopband_amplitude = abs(2 * np.mean(filtered * np.exp(-2j * np.pi * 0.5 * fc * time)))
        self.assertGreater(passband_amplitude, 0.8)
        self.assertLess(stopband_amplitude, passband_amplitude * 0.05)
        with self.assertRaisesRegex(ValueError, "Nyquist"):
            fdmas.design_rf_fdmas_filter(fc, 20e6)
        with self.assertRaisesRegex(ValueError, "5fc"):
            fdmas.FDMASBeamformerIQ(
                z, np.array([0.0]), 4, 0.0003, 1540.0, fc, 24e6,
                np.zeros(1), np.zeros(1), options=options,
            )

    def test_fdmas_preserves_pairs_with_imbalanced_weights(self):
        signal = torch.tensor([[1.0, 4.0]])
        for weights in ([[1.0, 1e-8]], [[1e-8, 1.0]]):
            actual = fdmas.FDMASBeamformerIQ._complex_fdmas(
                signal, torch.zeros_like(signal), torch.tensor(weights),
            )
            torch.testing.assert_close(actual, torch.tensor([2 + 0j]))
        actual = fdmas.FDMASBeamformerIQ._complex_fdmas(
            signal, torch.zeros_like(signal), torch.tensor([[1.0, 0.0]]),
        )
        torch.testing.assert_close(actual, torch.zeros(1, dtype=torch.complex64))

    def test_fdmas_compounds_two_angles_with_second_harmonic_tx_phase(self):
        z, x, fs, fc = 1.3, 0.4, 10.0, 1.0
        angles = np.array([-0.2, 0.2], dtype=np.float32)
        elements = np.arange(4) - 1.5
        rx = np.sqrt((x - elements) ** 2 + z ** 2)
        tx = z * np.cos(angles) + x * np.sin(angles)
        iq = np.exp(-2j * np.pi * fc * (tx[:, None] + rx[None]))
        iq = np.broadcast_to(iq[:, None, :], (2, 60, 4)).copy().astype(np.complex64)
        bf = fdmas.FDMASBeamformerIQ(np.array([z]), np.array([x]), 4, 1.0, 1.0, fc, fs,
                                      np.zeros(2), angles, options=self.options(fdmas))
        i, q = bf(iq.real, iq.imag, angles, np.zeros(2), fs)
        np.testing.assert_allclose(i, 1.0, atol=2e-5)
        np.testing.assert_allclose(q, 0.0, atol=2e-5)

    def test_envelope_db_handles_extreme_scale_and_large_tgc_without_overflow(self):
        values = np.array([[1.0, 0.1], [0.5, 0.05]], dtype=np.float32)
        reference = envelope_to_db(values, np.zeros_like(values), [0.0, 1.0], 5e6, False, 0.5)
        for scale in (1e-30, 1e30):
            actual = envelope_to_db(values * scale, np.zeros_like(values), [0.0, 1.0], 5e6, False, 0.5)
            np.testing.assert_allclose(actual, reference, atol=1e-5)
        actual = envelope_to_db(values, np.zeros_like(values), [0.0, 1.0], 5e6, True, 10.0)
        self.assertTrue(np.isfinite(actual).all())
        self.assertEqual(actual.max(), 0.0)

    def test_mban_reports_missing_training_stack_and_rejects_ignored_options(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(FileNotFoundError, "--train_root"):
                mban.load_evaluation_dependencies(Path(folder) / "train")
        with patch("sys.argv", ["mban", "--checkpoint", "model.pt", "--kind", "fp32", "--interp", "cubic"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                mban.parse_args()

    def test_ccmv_honors_its_device_independently_of_mv(self):
        with patch.object(ccmv, "device", torch.device("cpu")), patch.object(mv, "device", torch.device("meta")):
            bf = self.beamformer(ccmv, self.options(ccmv))
            signal = np.ones((1, 16, 8), dtype=np.float32)
            i, q = bf(signal, np.zeros_like(signal), np.zeros(1), np.zeros(1), 1.0)
        np.testing.assert_allclose(i, 1.0, atol=2e-4)
        np.testing.assert_allclose(q, 0.0, atol=1e-6)

    def test_mban_restores_checkpoint_snapshot_and_respects_explicit_config(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            train_root = root / "project" / "train"
            for name in ("eval_core/inference.py", "eval_core/models.py", "shared_core/runtime.py"):
                path = train_root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            checkpoint = train_root / "results" / "best.pth"
            checkpoint.parent.mkdir(parents=True)
            snapshot = checkpoint.with_name("effective_config.json")
            snapshot.write_text("{}", encoding="utf-8")
            argv = ["mban", "--checkpoint", str(checkpoint), "--kind", "fp32"]
            with patch("sys.argv", argv):
                options = mban.parse_args()
            self.assertEqual(options.train_root, train_root)
            self.assertEqual(options.config, snapshot)
            self.assertIsNone(options.eval_config)
            custom = root / "custom.yaml"
            with patch("sys.argv", argv + ["--config", str(custom)]):
                options = mban.parse_args()
            self.assertEqual(options.config, custom)
            self.assertEqual(options.eval_config, train_root / "hardware_eval.yaml")
            with patch("sys.argv", argv + ["--eval-config", str(custom)]):
                options = mban.parse_args()
            self.assertEqual(options.config, train_root / "config.yaml")
            self.assertEqual(options.eval_config, custom)
            with patch("sys.argv", argv + ["--config", str(snapshot), "--eval-config", str(custom)]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    mban.parse_args()

    def test_mban_saves_parameters_and_tracks_external_training_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            train_root = root / "train"
            for source in algorithm_source_paths("mban", train_root):
                if not source.is_relative_to(train_root):
                    continue
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text("source", encoding="utf-8")
            h5_path = root / "input.h5"
            with h5py.File(h5_path, "w") as handle:
                handle["angles"] = [0.0]
            grid = torch.tensor([0.0, 1.0])
            scene = SimpleNamespace(x_grid=grid, z_grid=grid, height=2, width=2)
            model = SimpleNamespace(runtime={}, signal_domain="iq")
            load_scene, load_model = Mock(return_value=scene), Mock(return_value=model)
            reconstruct = Mock(return_value=(np.array([[0.0, -5.0], [-10.0, -20.0]]), {}, [], 0.5))
            argv = ["mban", "--checkpoint", str(root / "model.pt"), "--kind", "fp32",
                    "--h5_path", str(h5_path), "--output_dir", str(root), "--train_root", str(train_root)]
            with patch("sys.argv", argv), patch.object(mban, "load_evaluation_dependencies", return_value=(load_scene, reconstruct, load_model)), patch("algorithms.common.load_from_h5", return_value=SimpleNamespace(angles=np.array([0.]), has_gt=False)), patch("algorithms.common.save_figure"), contextlib.redirect_stdout(io.StringIO()):
                mban.main()
            params = json.loads((root / "mban" / "params.json").read_text(encoding="utf-8"))
            self.assertEqual(params["runtime_sec"], 0.5)
            self.assertEqual(params["interp"], "linear")
            self.assertEqual(model.runtime["angle_reduction"], "mean")
            self.assertIn(str((train_root / "shared_core" / "model.py").resolve()), params["cache_signature"]["sources"])

    def test_template_requires_an_explicit_algorithm_implementation(self):
        with self.assertRaises(NotImplementedError):
            template_algorithm.beamform(None, argparse.Namespace())

    def test_each_physical_algorithm_reconstructs_and_saves_a_small_h5_scene(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "scene.h5"
            c, fc, fs, pitch = 1540.0, 3e6, 20e6, 0.0003
            angles = np.array([-0.1, 0.0, 0.1], dtype=np.float32)
            channels = (np.arange(8) - 3.5) * pitch
            tof = (0.003 * np.cos(angles[:, None]) + np.sqrt(channels[None] ** 2 + 0.003 ** 2)) / c
            pulse = np.exp(-0.5 * ((np.arange(160)[None, :, None] - tof[:, None] * fs) / 3.0) ** 2)
            iq = (pulse * np.exp(-2j * np.pi * fc * tof[:, None])).astype(np.complex64)
            with h5py.File(source, "w") as handle:
                for key, value in {
                    "all_multi_I": iq.real[None], "all_multi_Q": iq.imag[None],
                    "all_scale_ref": [1.0],
                    "angles": angles, "time_start_vector": np.zeros((1, 3)),
                    "valid_time_samples": [160], "c": c, "fc": fc, "fs": fs,
                    "pitch": pitch, "num_channels": 8,
                    "z_grid": np.linspace(0.002, 0.004, 5),
                    "x_grid": np.linspace(-0.0006, 0.0006, 5),
                }.items():
                    handle[key] = value
                handle["config_yaml"] = (
                    "generation:\n"
                    "  storage:\n"
                    "    inputs:\n"
                    "      iq:\n"
                    "        dtype: float32\n"
                    "        normalize: true\n"
                    "  ground_truth: {}\n"
                    "  training_targets: {}\n"
                )
            for module in self.modules:
                method = module.METHOD_NAME
                command = [sys.executable, "-m", f"algorithms.{method}",
                           "--h5_path", str(source), "--output_dir", str(root),
                           "--select_angles", "all", "--no_tgc", "--no_dynamic_aperture",
                           "--window", "hann", "--interp", "linear"]
                if module in (mv, ccmv, esbmv, gcfmv, cmsaw):
                    command += ["--subarray_ratio", "0.5", "--temporal_win", "3"]
                with self.subTest(method=method):
                    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                                            capture_output=True, text=True, timeout=120)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    image = np.load(root / method / f"{method}.npy")
                    self.assertEqual(image.shape, (5, 5))
                    self.assertTrue(np.isfinite(image).all())
                    self.assertAlmostEqual(float(image.max()), 0.0, places=3)
                    self.assertGreater(float(np.ptp(image)), 1e-3)
                    self.assertTrue((root / method / f"{method}.png").is_file())
                    params = json.loads((root / method / "params.json").read_text(encoding="utf-8"))
                    self.assertIn("cache_signature", params)
                    if module is gcfmv:
                        self.assertEqual(params["beamforming_mode"], "1D_RX")

            command = [sys.executable, "-m", "algorithms.gcfmv",
                       "--h5_path", str(source), "--output_dir", str(root),
                       "--select_angles", "1", "--no_tgc", "--no_dynamic_aperture"]
            result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            params = json.loads((root / "gcfmv" / "params.json").read_text(encoding="utf-8"))
            self.assertEqual(params["beamforming_mode"], "1D_RX")
            for axis, mode in (("tx", "1D_TX"), ("joint", "2D_JTR")):
                command[command.index("--select_angles") + 1] = "all"
                result = subprocess.run(command + ["--mv_axis", axis],
                    cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                params = json.loads((root / "gcfmv" / "params.json").read_text(encoding="utf-8"))
                self.assertEqual(params["mv_axis"], axis)
                self.assertEqual(params["beamforming_mode"], mode)


if __name__ == "__main__":
    unittest.main()

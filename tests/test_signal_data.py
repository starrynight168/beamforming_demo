import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import yaml
from scipy.signal import firwin, hilbert

from algorithms.shared_core.data import load_input, load_target
from algorithms.shared_core.signals import rf_to_analytic, rf_to_iq


class SignalDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "sample.h5"
        self.rf = np.random.default_rng(41).normal(size=(2, 260, 3)).astype(np.float32)
        self.starts = np.array([0.17e-6, 0.31e-6])
        self.config = {
            "acquisition": {"fc_hz": 5e6},
            "generation": {
                "input_iq": {
                    "decimation_factor": 2,
                    "source_sampling_frequency_hz": 20e6,
                    "packed_sampling_frequency_hz": 10e6,
                    "ddc": {"phase_sign_q": -1, "lowpass": "firwin_64_then_filtfilt",
                            "resampling": "scipy_resample_poly_kaiser_12"},
                    "normalization": {"mode": "complex_rms", "epsilon": 1e-12},
                },
                "training_targets": {
                    "complex_iq_targets": {"mv": {"i": "/mv_i", "q": "/mv_q"}},
                    "rf_targets": {"mv": "/mv_rf"},
                    "analytic_rf_targets": {"mv": {"real": "/mv_ar_i", "imag": "/mv_ar_q"}},
                    "output_angles": {"mv": {"selection": "input", "indices": [0, 1]}},
                },
            },
        }

    def create_file(self, *, scale=3.0, cached_iq=False):
        with h5py.File(self.path, "w") as hf:
            padded = np.pad(self.rf, ((0, 0), (0, 16), (0, 0)), constant_values=np.nan)
            hf["rf"] = padded[None]
            hf["rf_fs"] = 20e6
            hf["rf_t0"] = self.starts[None]
            hf["rf_angles"] = [-0.1, 0.1]
            hf["rf_length"] = [self.rf.shape[1]]
            hf["x_grid"] = [-0.001, 0.001]
            hf["z_grid"] = [0.01, 0.02, 0.03]
            if scale is not None:
                hf["all_scale_ref"] = [scale]
            if cached_iq:
                hf["all_multi_I"] = np.full((1, 2, 10, 3), 4.0, dtype=np.float16)
                hf["all_multi_Q"] = np.full((1, 2, 10, 3), -2.0, dtype=np.float16)
                hf["fs"] = 10e6
                hf["time_start_vector"] = self.starts[None]
                hf["valid_time_samples"] = [7]
                hf["angles"] = [-0.1, 0.1]
            for field, value in (("mv_i", 1.0), ("mv_q", 2.0), ("mv_rf", -3.0),
                                 ("mv_ar_i", 4.0), ("mv_ar_q", 5.0)):
                hf[field] = np.full((1, 1, 3, 2), value, dtype=np.float16)
            hf["config_yaml"] = yaml.safe_dump(self.config)

    def test_cached_iq_is_preferred_and_not_normalized_twice(self):
        self.create_file(cached_iq=True)
        with patch("algorithms.shared_core.data.rf_to_iq", side_effect=AssertionError("Cache should be used")):
            result = load_input(self.path, 0)
        self.assertEqual(result["real"].shape, (2, 7, 3))
        np.testing.assert_array_equal(result["real"], 4.0)
        np.testing.assert_array_equal(result["imag"], -2.0)
        self.assertEqual(result["fs"], 10e6)
        self.assertEqual(result["scale"], 3.0)

    def test_rf_and_analytic_use_valid_rf_length_and_common_scale(self):
        self.create_file()
        rf = load_input(self.path, 0, "rf")
        analytic = load_input(self.path, 0, "analytic_rf")
        np.testing.assert_array_equal(rf["real"], self.rf / 3.0)
        np.testing.assert_array_equal(rf["imag"], np.zeros_like(self.rf))
        np.testing.assert_array_equal(analytic["real"], self.rf / 3.0)
        np.testing.assert_array_equal(analytic["imag"], hilbert(self.rf, axis=1).imag / 3.0)
        np.testing.assert_array_equal(rf["t0"], self.starts)

    def test_raw_iq_cache_is_normalized_using_the_recorded_scale(self):
        for dtype in (np.float16, np.float32):
            with self.subTest(dtype=dtype):
                self.config["generation"]["storage"] = {"inputs": {"iq": {"dtype": np.dtype(dtype).name, "normalize": False}}}
                self.create_file(scale=4.0, cached_iq=True)
                with h5py.File(self.path, "r+") as hf:
                    for field in ("all_multi_I", "all_multi_Q"):
                        physical = hf[field][:].astype(np.float32) * 4.0
                        del hf[field]
                        hf[field] = physical.astype(dtype)
                result = load_input(self.path, 0)
                np.testing.assert_array_equal(result["real"], 4.0)
                np.testing.assert_array_equal(result["imag"], -2.0)

    def test_normalized_rf_is_restored_before_analytic_and_iq_conversion(self):
        self.config["generation"]["storage"] = {"inputs": {"rf": {"dtype": "float32", "normalize": True}}}
        self.create_file(scale=4.0)
        with h5py.File(self.path, "r+") as hf:
            hf["rf"][:] = hf["rf"][:] / 4.0
        np.testing.assert_array_equal(load_input(self.path, 0, "rf")["real"], self.rf / 4.0)
        np.testing.assert_array_equal(load_input(self.path, 0, "analytic_rf")["imag"], hilbert(self.rf, axis=1).imag / 4.0)
        expected = rf_to_iq(self.rf, 20e6, 5e6, self.starts, firwin(64, 0.4), 2)
        np.testing.assert_array_equal(load_input(self.path, 0, "iq")["real"], expected[0] / 4.0)

    def test_normalized_rf_requires_its_saved_scale(self):
        self.config["generation"]["storage"] = {"inputs": {"rf": {"dtype": "float16", "normalize": True}}}
        self.create_file(scale=None)
        with self.assertRaisesRegex(KeyError, "all_scale_ref"):
            load_input(self.path, 0, "rf")

    def test_new_input_normalization_declaration_must_be_boolean(self):
        self.config["generation"]["storage"] = {"inputs": {"iq": {"dtype": "float32", "normalize": "false"}}}
        self.create_file(cached_iq=True)
        with self.assertRaisesRegex(ValueError, "normalize must be explicit"):
            load_input(self.path, 0)

    def test_analytic_cache_can_be_used_without_rf_and_matches_derived_values(self):
        self.create_file()
        expected = load_input(self.path, 0, "analytic_rf")
        with h5py.File(self.path, "r+") as hf:
            config = yaml.safe_load(hf["config_yaml"][()])
            config["generation"]["storage"] = {"inputs": {"analytic_rf": {"dtype": "float32", "normalize": False}}}
            hf["config_yaml"][()] = yaml.safe_dump(config)
            hf["ar_i"] = expected["real"][None] * 3
            hf["ar_q"] = expected["imag"][None] * 3
            hf["ar_fs"], hf["ar_t0"] = 20e6, self.starts[None]
            hf["ar_angles"], hf["ar_len"] = [-.1, .1], [260]
            del hf["rf"]
        with patch("algorithms.shared_core.data.rf_to_analytic", side_effect=AssertionError("Cache must be used")):
            cached = load_input(self.path, 0, "analytic_rf")
        np.testing.assert_allclose(cached["real"], expected["real"], rtol=2e-7)
        np.testing.assert_allclose(cached["imag"], expected["imag"], rtol=2e-7)
        self.assertEqual(cached["fs"], 20e6)
        np.testing.assert_array_equal(load_target(self.path, 0, domain="analytic_rf")["real"], 4.)

    def test_partial_analytic_cache_never_falls_back_to_rf(self):
        self.create_file()
        with h5py.File(self.path, "r+") as hf:
            hf["ar_i"] = self.rf[None]
        with self.assertRaisesRegex(KeyError, "ANALYTIC_RF cache is incomplete"):
            load_input(self.path, 0, "analytic_rf")

    def test_physical_analytic_cache_without_teacher_scale_keeps_its_units(self):
        self.config["generation"]["storage"] = {"inputs": {"analytic_rf": {"dtype": "float32", "normalize": False}}}
        self.create_file(scale=None)
        expected_imag = hilbert(self.rf, axis=1).imag
        with h5py.File(self.path, "r+") as hf:
            hf["ar_i"], hf["ar_q"] = self.rf[None], expected_imag[None]
            hf["ar_fs"], hf["ar_t0"] = 20e6, self.starts[None]
            hf["ar_angles"], hf["ar_len"] = [-.1, .1], [260]
            del hf["rf"]
        result = load_input(self.path, 0, "analytic_rf")
        self.assertEqual(result["scale"], 1.)
        np.testing.assert_array_equal(result["real"], self.rf)
        np.testing.assert_array_equal(result["imag"], expected_imag)
        with h5py.File(self.path, "r+") as hf:
            self.config["generation"]["storage"]["inputs"]["analytic_rf"]["normalize"] = True
            hf["config_yaml"][()] = yaml.safe_dump(self.config)
        with self.assertRaisesRegex(KeyError, "all_scale_ref"):
            load_input(self.path, 0, "analytic_rf")

    def test_teacher_analytic_fields_are_not_channel_cache(self):
        self.create_file()
        with patch("algorithms.shared_core.data.rf_to_analytic", wraps=rf_to_analytic) as transform:
            value = load_input(self.path, 0, "analytic_rf")
        transform.assert_called_once()
        self.assertEqual(value["real"].shape, self.rf.shape)

    def test_iq_can_be_derived_from_rf_with_schema_one_ddc_definition(self):
        self.create_file()
        result = load_input(self.path, 0, "iq")
        expected = rf_to_iq(self.rf, 20e6, 5e6, self.starts, firwin(64, 0.4), 2)
        np.testing.assert_array_equal(result["real"], expected[0] / 3.0)
        np.testing.assert_array_equal(result["imag"], expected[1] / 3.0)
        self.assertEqual(result["fs"], 10e6)

    def test_explicit_filter_numbers_are_used(self):
        self.config["generation"]["input_iq"]["ddc"].update(
            mixing_gain=1.0, cutoff_hz=3e6, num_taps=32,
        )
        self.create_file()
        result = load_input(self.path, 0, "iq")
        expected = rf_to_iq(self.rf, 20e6, 5e6, self.starts, firwin(32, 0.3), 2)
        np.testing.assert_array_equal(result["real"], expected[0] / 3.0)
        np.testing.assert_array_equal(result["imag"], expected[1] / 3.0)

    def test_missing_scale_is_computed_from_recorded_normalization(self):
        iq = rf_to_iq(self.rf, 20e6, 5e6, self.starts, firwin(64, 0.4), 2)
        for mode in ("complex_rms", "env_std"):
            with self.subTest(mode=mode):
                self.config["generation"]["input_iq"]["normalization"]["mode"] = mode
                self.create_file(scale=None)
                result = load_input(self.path, 0, "iq")
                scale = float(np.sqrt(np.mean(iq[0]**2 + iq[1]**2) + 1e-12)) if mode == "complex_rms" else float(np.std(np.hypot(*iq)) + 1e-12)
                self.assertEqual(result["scale"], scale)
                np.testing.assert_array_equal(result["real"], iq[0] / scale)
        np.testing.assert_array_equal(load_input(self.path, 0, "rf")["real"], self.rf)

    def test_teachers_are_resolved_by_metadata_and_not_scaled_twice(self):
        self.create_file()
        for domain, expected in (("rf", (-3.0, 0.0)), ("analytic_rf", (4.0, 5.0)), ("iq", (1.0, 2.0))):
            with self.subTest(domain=domain):
                result = load_target(self.path, 0, domain=domain)
                np.testing.assert_array_equal(result["real"], expected[0])
                np.testing.assert_array_equal(result["imag"], expected[1])
                self.assertEqual(result["real"].shape, (3, 2))
                self.assertEqual(result["scale"], 3.0)
                np.testing.assert_array_equal(result["grid"]["x_grid"], np.array([-0.001, 0.001], dtype=np.float32))

    def test_wrong_angle_teacher_and_missing_fields_report_configuration(self):
        self.create_file()
        with self.assertRaisesRegex(ValueError, "uses angles='input'"):
            load_target(self.path, 0, angles="all")
        with self.assertRaisesRegex(KeyError, "outputs.rf.das"):
            load_target(self.path, 0, algorithm="das", domain="rf")
        with h5py.File(self.path, "r+") as hf:
            del hf["rf"]
        with self.assertRaisesRegex(KeyError, "storage.inputs"):
            load_input(self.path, 0, "rf")

    def test_unsupported_ddc_does_not_borrow_another_configuration(self):
        self.config["generation"]["input_iq"]["ddc"]["mixing_gain"] = 2.0
        self.create_file()
        with self.assertRaisesRegex(ValueError, "Unsupported.*ddc"):
            load_input(self.path, 0, "iq")

    def test_partial_iq_cache_is_not_silently_replaced_by_rf_conversion(self):
        self.create_file(cached_iq=True)
        with h5py.File(self.path, "r+") as hf:
            del hf["all_multi_Q"]
        with self.assertRaisesRegex(KeyError, "IQ cache is incomplete"):
            load_input(self.path, 0, "iq")

    def test_sample_idx_and_domain_are_explicit(self):
        self.create_file()
        with self.assertRaisesRegex(ValueError, "sample_idx"):
            load_input(self.path, "0")
        with self.assertRaisesRegex(ValueError, "domain"):
            load_target(self.path, 0, domain="unknown")
        with self.assertRaises(IndexError):
            load_input(self.path, 1, "rf")


if __name__ == "__main__":
    unittest.main()

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch
import yaml

from algorithms import das
from algorithms.cmsaw import validate_cmsaw_params
from algorithms.common import (
    aperture_window_1d,
    aperture_window_from_dx,
    interpolate_channel_samples,
    parse_selected_angles,
    result_cache_signature,
    save_algorithm_result,
)
from run_one import can_reuse_existing
from tools.run_ablation_cn import (
    can_reuse_flat_result,
    numeric_range_values,
    run_ablation_evaluation,
    validate_values,
)
from tools.run_wizard_cn import (
    inspect_h5,
    sample_can_be_evaluated,
    validate_algorithm_param,
)


class AlgorithmAuditTests(unittest.TestCase):
    def test_two_channel_aperture_windows_keep_active_channels(self):
        for window in ("tukey", "hann", "hamming", "blackman", "kaiser"):
            with self.subTest(window=window):
                np.testing.assert_array_equal(
                    aperture_window_1d(2, window, "cpu").numpy(), [1.0, 1.0]
                )
                geometry = aperture_window_from_dx(
                    torch.tensor([[[-1.0, 1.0]]]),
                    torch.tensor([[[1.0]]]),
                    window,
                )
                np.testing.assert_array_equal(geometry.numpy(), [[[1.0, 1.0]]])

    def test_cache_signature_tracks_input_and_dependency_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            h5_path = root / "input.h5"
            dependency = root / "weights.bin"
            h5_path.write_bytes(b"h5 input")
            dependency.write_bytes(b"weights")
            params = {"weights_path": str(dependency)}

            first = result_cache_signature(h5_path, "das", params)
            self.assertEqual(first, result_cache_signature(h5_path, "das", params))
            h5_path.write_bytes(b"changed h5 input")
            second = result_cache_signature(h5_path, "das", params)
            self.assertNotEqual(first["input"], second["input"])
            dependency.write_bytes(b"changed weights")
            third = result_cache_signature(h5_path, "das", params)
            self.assertNotEqual(second["dependencies"], third["dependencies"])

    def test_run_one_reuse_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            h5_path = root / "input.h5"
            h5_path.write_bytes(b"input")
            scene = {"id": "smoke", "h5_path": str(h5_path), "sample_idx": 0}
            config = {
                "params": {},
                "algorithm_params": {"das": {}},
            }
            scene_dir = root / "scene"
            scene_dir.mkdir()
            signature = result_cache_signature(h5_path, "das", {})
            (scene_dir / "run_params.json").write_text(
                json.dumps(
                    {
                        "scene": scene,
                        "params": {
                            "select_angles": "1",
                            "f_number": 1.5,
                            "dr": 60.0,
                            "dynamic_aperture": True,
                            "tgc": True,
                            "tgc_alpha": 0.5,
                            "window": "rect",
                            "interp": "cubic",
                        },
                        "algorithm_params": {"das": {}},
                        "extra_algorithm_args": [],
                        "cache_signatures": {"das": signature},
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(extra_algorithm_args=[])

            self.assertTrue(
                can_reuse_existing(scene_dir, config, scene, "das", args)[0]
            )
            h5_path.write_bytes(b"changed input")
            self.assertFalse(
                can_reuse_existing(scene_dir, config, scene, "das", args)[0]
            )

    def test_ablation_reuse_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            h5_path = root / "input.h5"
            h5_path.write_bytes(b"input")
            config = {
                "algorithms": ["das"],
                "params": {
                    "select_angles": "1",
                    "f_number": 1.5,
                    "dr": 60.0,
                    "dynamic_aperture": True,
                    "tgc": True,
                    "tgc_alpha": 0.5,
                    "window": "rect",
                    "interp": "cubic",
                },
                "algorithm_params": {"das": {}},
                "scenes": [
                    {"id": "smoke", "h5_path": str(h5_path), "sample_idx": 0}
                ],
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            scene_dir = root / "scene"
            scene_dir.mkdir()
            (scene_dir / "das.npy").write_bytes(b"output")
            (scene_dir / "params.json").write_text(
                json.dumps(
                    {
                        **config["params"],
                        "h5_path": str(h5_path.resolve()),
                        "h5_sample_idx": 0,
                        "method": "das",
                        "cache_signature": result_cache_signature(h5_path, "das", {}),
                    }
                ),
                encoding="utf-8",
            )

            self.assertTrue(can_reuse_flat_result(config_path, scene_dir, "das"))
            h5_path.write_bytes(b"changed input")
            self.assertFalse(can_reuse_flat_result(config_path, scene_dir, "das"))

    def test_cmsaw_depth_smoothing_requires_odd_window(self):
        base = (0.5, 2, 1.0, 0.5, 90.0, 9)
        validate_cmsaw_params(*base, 3)
        with self.assertRaisesRegex(ValueError, "positive odd"):
            validate_cmsaw_params(*base, 4)
        self.assertEqual(validate_algorithm_param("depth_smooth_rows", 3), 3)
        with self.assertRaisesRegex(ValueError, "正奇数"):
            validate_algorithm_param("depth_smooth_rows", 2)

    def test_ablation_values_reject_duplicates_after_normalization(self):
        with self.assertRaisesRegex(ValueError, "重复"):
            validate_values("fbss", True, [True, True])
        self.assertEqual(validate_values("fbss", True, [True, False]), [True, False])
        with self.assertRaisesRegex(ValueError, "重复"):
            validate_values("f_number", 1.5, [1, 1.0])
        with self.assertRaisesRegex(ValueError, "重复"):
            validate_values("window", "rect", ["hann", "HANN"])
        with self.assertRaisesRegex(ValueError, "正奇数"):
            validate_values("depth_smooth_rows", 1, [2, 3])
        self.assertEqual(validate_values("depth_smooth_rows", 1, [1, 3]), [1, 3])
        self.assertEqual(validate_values("f_number", 1.5, [1, 2]), [1, 2])

    def test_saved_cache_signature_cannot_be_overridden_by_extra_params(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            h5_path = root / "input.h5"
            h5_path.write_bytes(b"input")
            args = argparse.Namespace(
                h5_path=str(h5_path), output_dir=str(root), dr=60.0, save_gt=False
            )
            input_data = SimpleNamespace(
                extent_mm=(0, 1, 1, 0), sample=SimpleNamespace(has_gt=False)
            )
            with patch("algorithms.common.save_figure"):
                output = save_algorithm_result(
                    np.zeros((2, 2)), input_data, args, "das", "DAS", 0.0,
                    extra_params={"cache_signature": "stale"},
                )
            params = json.loads((output / "params.json").read_text(encoding="utf-8"))
            self.assertEqual(
                params["cache_signature"], result_cache_signature(h5_path, "das", vars(args))
            )

    def test_das_keeps_last_sample_and_zeros_out_of_range_output(self):
        angles = np.array([0.0], dtype=np.float32)
        i_data = np.zeros((1, 5, 1), dtype=np.float32)
        i_data[0, -1, 0] = 7.0
        q_data = np.zeros_like(i_data)
        for aperture in ("geometry", "centered"):
            for interpolation in ("nearest", "linear", "cubic", "quintic", "farrow", "sinc"):
                with self.subTest(aperture=aperture, interpolation=interpolation):
                    args = argparse.Namespace(
                        aperture_mode=aperture, dynamic_aperture=False, f_number=1.5,
                        window="rect", row_block=1, interp=interpolation,
                    )
                    with (
                        patch.object(das, "args", args, create=True),
                        patch.object(das, "device", torch.device("cpu")),
                    ):
                        beamformer = das.DASBeamformerIQ(
                            np.array([1.5, 2.0, 2.5]), np.array([0.0]),
                            1, 1.0, 1.0, 0.0, 1.0, np.zeros(1), angles,
                        )
                        i_output, q_output = beamformer(
                            i_data, q_data, angles, np.zeros(1), 1.0
                        )
                    np.testing.assert_allclose(i_output[:, 0], [0.0, 7.0, 0.0], atol=1e-6)
                    np.testing.assert_allclose(q_output, 0.0, atol=1e-6)

    def test_dr_ablation_skips_metric_subprocesses(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            np.save(root / "ablation_comparison.npy", np.zeros((2, 2, 2)))
            rows = [{"status": "ok", "parameter": "dr", "value": 40, "index": 1}]
            with patch("tools.run_ablation_cn.subprocess.run") as run:
                result = run_ablation_evaluation(rows, root, root / "input.h5", 0, True, 60)
            self.assertIsNone(result)
            run.assert_not_called()
            self.assertFalse((root / "metrics").exists())

    def test_numeric_range_rejects_nonfinite_and_nonadvancing_steps(self):
        with patch("tools.run_ablation_cn.ask_text", side_effect=["0", "1", "inf"]):
            with self.assertRaisesRegex(ValueError, "有限数"):
                numeric_range_values()
        with patch(
            "tools.run_ablation_cn.ask_text",
            side_effect=["1e308", "1e308", "1"],
        ):
            with self.assertRaisesRegex(ValueError, "无法推进"):
                numeric_range_values()

    def test_select_angles_count_one_selects_nearest_zero(self):
        angles = np.array([-0.2, -0.05, 0.1, 0.3])
        indices, selected = parse_selected_angles(angles, "1")
        np.testing.assert_array_equal(indices, [1])
        np.testing.assert_array_equal(selected, [-0.05])

    def test_interpolation_kernels_preserve_and_clamp_endpoint_samples(self):
        i_data = torch.arange(5, dtype=torch.float32).view(5, 1)
        q_data = -i_data
        samples = torch.tensor([-1.0, 4.0, 5.0])
        expected = torch.tensor([0.0, 4.0, 4.0])

        interpolations = ("nearest", "linear", "cubic", "quintic", "farrow", "sinc")
        for interpolation in interpolations:
            with self.subTest(interpolation=interpolation):
                i_values, q_values = interpolate_channel_samples(
                    i_data,
                    q_data,
                    samples,
                    torch.zeros_like(samples, dtype=torch.long),
                    interpolation,
                )
                torch.testing.assert_close(i_values, expected)
                torch.testing.assert_close(q_values, -expected)

    def test_wizard_accepts_single_channel_ground_truth(self):
        for gt_shape in ((1, 1, 2, 2), (1, 2, 2)):
            with (
                self.subTest(gt_shape=gt_shape),
                tempfile.TemporaryDirectory() as temp_dir,
            ):
                h5_path = Path(temp_dir) / "sample.h5"
                embedded_config = {
                    "dataset": {"id": "simulation"},
                    "source_samples": [
                        {
                            "acquisition_id": "simulation_contrast_speckle",
                            "phantom_mode": "simulation",
                            "phantom_source": "PICMUS",
                        }
                    ],
                }
                with h5py.File(h5_path, "w") as hf:
                    hf.create_dataset("all_multi_I", data=np.zeros((1, 3, 8, 4)))
                    hf.create_dataset("all_multi_Q", data=np.zeros((1, 3, 8, 4)))
                    hf.create_dataset("valid_time_samples", data=[8])
                    hf.create_dataset("time_start_vector", data=np.zeros((1, 3)))
                    hf.create_dataset("angles", data=[-0.1, 0.0, 0.1])
                    hf.create_dataset("z_grid", data=[0.001, 0.002])
                    hf.create_dataset("x_grid", data=[-0.001, 0.001])
                    hf.create_dataset("fs", data=20e6)
                    hf.create_dataset("c", data=1540.0)
                    hf.create_dataset("fc", data=5e6)
                    hf.create_dataset("pitch", data=0.0003)
                    hf.create_dataset("num_channels", data=4)
                    hf.create_dataset("all_envdb_norm", data=np.zeros(gt_shape))
                    hf.create_dataset(
                        "config_yaml",
                        data=yaml.safe_dump(embedded_config),
                    )

                info = inspect_h5(h5_path)
                self.assertTrue(sample_can_be_evaluated(info, 0))


if __name__ == "__main__":
    unittest.main()

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
from unittest.mock import patch

import h5py
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms import das
from algorithms.cmsaw import validate_cmsaw_params
from algorithms.common import (
    aperture_window_1d,
    aperture_window_from_dx,
    interpolate_channel_samples,
    normalized_mv_weights,
    parse_selected_angles,
    result_cache_signature,
    save_algorithm_result,
    spatial_temporal_covariance,
)
from run_one import can_reuse_existing, scene_has_gt
from run_one import main as run_scene
from tools.ablate_sound_speed import dense_grid_sweep, multi_section_search
from tools.run_ablation_cn import (
    can_reuse_flat_result,
    method_output_path,
    numeric_range_values,
    preserve_ground_truth,
    run_ablation_evaluation,
    validate_values,
)
from tools.run_wizard_cn import (
    ask_algorithm_param,
    configured_choice_index,
    inspect_h5,
    sample_can_be_evaluated,
    validate_algorithm_param,
)


class AlgorithmAuditTests(unittest.TestCase):
    def test_ablation_uses_only_flattened_method_outputs(self):
        scene_dir = Path("run") / "scene"
        self.assertEqual(
            method_output_path(scene_dir, "das", "npy"),
            scene_dir / "das.npy",
        )

    def test_wizard_rejects_unsupported_configured_choices(self):
        choices = [("rect", "rectangular"), ("hann", "Hann")]
        self.assertEqual(configured_choice_index(choices, "hann", "params.window"), 1)
        with self.assertRaisesRegex(ValueError, "params.window.*不受支持"):
            configured_choice_index(choices, "old_window", "params.window")
        with self.assertRaisesRegex(ValueError, "das.aperture_mode.*不受支持"):
            ask_algorithm_param("das", "aperture_mode", "legacy")

    def test_scene_rejects_truncated_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            h5_path = Path(directory) / "input.h5"
            with h5py.File(h5_path, "w") as handle:
                handle.create_dataset("all_multi_I", data=np.zeros((2, 1, 1, 1)))
                handle.create_dataset("all_envdb_norm", data=np.zeros((1, 1, 2, 2)))
            with self.assertRaisesRegex(ValueError, "样本数必须与 all_multi_I 一致"):
                scene_has_gt({"h5_path": str(h5_path), "sample_idx": 1})

    def test_cache_signature_tracks_cli_file_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h5_path = root / "input.h5"
            checkpoint = root / "model.pth"
            config = root / "model.yaml"
            h5_path.write_bytes(b"input")
            checkpoint.write_bytes(b"weights")
            config.write_bytes(b"config")
            extra = ["--checkpoint", str(checkpoint), f"--eval-config={config}", "--no_fbss"]
            first = result_cache_signature(h5_path, "das", extra_args=extra)
            self.assertEqual(len(first["dependencies"]), 2)
            checkpoint.write_bytes(b"new weights")
            second = result_cache_signature(h5_path, "das", extra_args=extra)
            self.assertNotEqual(first["dependencies"], second["dependencies"])

    def test_ablation_refreshes_gt_after_reconstruction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / "scene"
            scene.mkdir()
            gt = np.array([[0.0, -30.0], [-60.0, -45.0]], dtype=np.float32)
            np.save(root / "ground_truth.npy", np.zeros_like(gt))
            np.save(scene / "comparison.npy", np.stack([gt, gt - 1]))
            target = preserve_ground_truth(scene, root)
            np.testing.assert_array_equal(np.load(target), gt)

    def test_sound_speed_search_requires_a_shrinking_bracket(self):
        cache = {}

        def evaluate(speed):
            result = {
                "c": float(speed), "ssim": 1 - ((speed - 1300) / 300) ** 2,
                "psnr": 20, "mae_db": 1, "elapsed_sec": 0,
            }
            cache[float(speed)] = result
            return result

        evaluator = SimpleNamespace(cache=cache, evaluate_c=evaluate)
        with self.assertRaisesRegex(ValueError, "at least 3"):
            multi_section_search(evaluator, 1000, 1600, num_sections=2)
        with contextlib.redirect_stdout(io.StringIO()):
            for sections in (3, 4):
                cache.clear()
                best, history, interval = multi_section_search(
                    evaluator, 1000, 1600, num_sections=sections, tol=5, max_iter=20
                )
                self.assertLessEqual(interval[1] - interval[0], 5)
                self.assertAlmostEqual(best["c"], 1300, delta=5)
                self.assertTrue(history)
            grid = dense_grid_sweep(evaluator, 1000, 1600, step=250)
            self.assertEqual([row["c"] for row in grid], [1000, 1250, 1500, 1600])

    def test_runner_common_overrides_and_stored_gt_range(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            h5_path = root / "input.h5"
            gt = np.array([[1.0, 0.5], [0.0, 0.25]], dtype=np.float32)
            with h5py.File(h5_path, "w") as handle:
                handle.create_dataset("all_multi_I", data=np.zeros((1, 1, 2, 1)))
                handle.create_dataset("x_grid", data=[-0.001, 0.001])
                handle.create_dataset("z_grid", data=[0.001, 0.002])
                handle.create_dataset("all_envdb_norm", data=gt[None, None])
                handle.create_dataset("config_yaml", data=yaml.safe_dump({
                    "generation": {"ground_truth": {"dynamic_range_db": 60}}
                }))
            config = {
                "algorithms": ["das"], "params": {"dr": 60},
                "scenes": [{"id": "sample", "h5_path": str(h5_path), "sample_idx": 0}],
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            scene_dir = root / "output" / "sample"
            (scene_dir / "das").mkdir(parents=True)
            np.save(scene_dir / "das" / "das.npy", [[0.0, -10.0], [-20.0, -30.0]])
            args = argparse.Namespace(
                config=str(config_path), scene="sample", algorithms=None,
                output_root=str(root / "output"), python_exe=sys.executable,
                no_evaluate=True, no_individual_images=True, keep_existing=False,
                extra_algorithm_args=["--dr", "40", "--no_tgc", "--row_block", "2"],
            )
            with patch("run_one.parse_args", return_value=args), patch(
                "run_one.subprocess.run", return_value=SimpleNamespace(returncode=0)
            ) as process, patch("run_one.save_comparison"):
                run_scene()
            command = process.call_args.args[0]
            self.assertEqual(command.count("--dr"), 1)
            self.assertEqual(float(command[command.index("--dr") + 1]), 40)
            self.assertIn("--no_tgc", command)
            params = json.loads((scene_dir / "run_params.json").read_text(encoding="utf-8"))
            self.assertEqual(params["params"]["dr"], 40)
            self.assertFalse(params["params"]["tgc"])
            comparison = np.load(scene_dir / "comparison.npy")
            np.testing.assert_array_equal(comparison[0], gt * 60 - 60)

    def test_algorithm_entrypoints_support_direct_and_module_execution(self):
        project_root = Path(__file__).resolve().parents[1]
        for name in (
            "das", "mv", "esbmv", "gcfmv", "fdmas", "cmsaw", "template_algorithm"
        ):
            commands = (
                ([sys.executable, f"{name}.py", "--help"], project_root / "algorithms"),
                ([sys.executable, "-m", f"algorithms.{name}", "--help"], project_root),
            )
            for command, cwd in commands:
                with self.subTest(algorithm=name, command=command):
                    result = subprocess.run(
                        command, cwd=cwd, capture_output=True, text=True
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("--h5_path", result.stdout)

    def test_cache_signature_resolves_h5_relative_to_project_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            h5_path = root / "input.h5"
            h5_path.write_bytes(b"input")
            with patch("algorithms.common.PROJECT_ROOT", root), patch(
                "algorithms.common.algorithm_source_paths", return_value=()
            ):
                self.assertEqual(
                    result_cache_signature("input.h5", "das"),
                    result_cache_signature(h5_path, "das"),
                )

    def test_mv_weight_solver_matches_formula_and_handles_singular_covariance(self):
        covariance = torch.tensor(
            [[[2.0, 0.0], [0.0, 1.0]]], dtype=torch.complex64
        )
        steering = torch.ones((1, 2, 1), dtype=torch.complex64)
        expected_v = torch.linalg.solve(covariance, steering)
        expected = expected_v / (steering.mH @ expected_v + 1e-12)

        torch.testing.assert_close(
            normalized_mv_weights(covariance, steering), expected
        )
        singular = torch.zeros_like(covariance)
        result = normalized_mv_weights(singular, steering)
        self.assertTrue(torch.isfinite(result).all())
        torch.testing.assert_close(result, torch.zeros_like(result))

    def test_spatial_temporal_covariance_matches_forward_and_fbss_formulas(self):
        real = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        x_active = torch.complex(real, real.flip(-1) / 3)
        x_subarrays = x_active.unfold(1, 3, 1).permute(0, 3, 1, 2)
        x_snapshots = x_subarrays.reshape(2, 3, x_subarrays.shape[2] * 3)
        forward = x_snapshots @ x_snapshots.mH / x_snapshots.shape[-1]

        actual_subarrays, actual_forward = spatial_temporal_covariance(
            x_active, 3, 3, False
        )
        actual_fbss_subarrays, actual_fbss = spatial_temporal_covariance(
            x_active, 3, 3, True
        )

        self.assertEqual(actual_subarrays.shape, (2, 3, 2, 3))
        torch.testing.assert_close(actual_subarrays, x_subarrays)
        torch.testing.assert_close(actual_fbss_subarrays, x_subarrays)
        torch.testing.assert_close(actual_forward, forward)
        torch.testing.assert_close(
            actual_fbss, 0.5 * (forward + forward.conj().flip(-1, -2))
        )

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
            for interpolation in (
                "nearest",
                "linear",
                "cubic",
                "catmull_rom",
                "quintic",
                "farrow",
                "sinc",
            ):
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
                result = run_ablation_evaluation(
                    rows, root, root / "input.h5", 0, True, 60
                )
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

        interpolations = (
            "nearest",
            "linear",
            "cubic",
            "catmull_rom",
            "quintic",
            "farrow",
            "sinc",
        )
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
                            "phantom_mode": "in_vitro",
                            "phantom_source": "in_vitro",
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
                self.assertEqual(info["gt"], "有")
                self.assertEqual(info["names"], ["simulation_contrast_speckle"])
                self.assertFalse(sample_can_be_evaluated(h5_path, 0))

                with h5py.File(h5_path, "r+") as hf:
                    del hf["config_yaml"]
                    hf.create_dataset(
                        "config_yaml",
                        data=yaml.safe_dump({"source_samples": [{}]}),
                    )
                with self.assertRaisesRegex(ValueError, "必须包含 id 或 acquisition_id"):
                    inspect_h5(h5_path)


if __name__ == "__main__":
    unittest.main()

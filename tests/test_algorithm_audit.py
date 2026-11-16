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
from algorithms.shared_core.beamforming import (
    aperture_window_1d,
    aperture_window_from_dx,
    dynamic_aperture_channel_count,
    interpolate_channel_samples,
    normalized_mv_weights,
    spatial_temporal_covariance,
)
from algorithms.shared_core.common import (
    ground_truth_db,
    parse_selected_angles,
    result_cache_signature,
    save_algorithm_result,
)
from run_one import can_reuse_existing, scene_has_gt
from run_one import main as run_scene
from tools.code.ablate_sound_speed import method_c, parse_c_values
from tools.code.run_ablation import (
    can_reuse_flat_result,
    choose_values,
    collect_ablation_individual_images,
    find_matching_ablation_run,
    method_output_path,
    numeric_range_values,
    preserve_ground_truth,
    run_ablation_evaluation,
    save_ablation_overview,
    should_evaluate_ablation,
    validate_available_signal_domains,
    validate_values,
)
from tools.code.compare_rf_iq import main as compare_rf_iq_main, validate_time_axis
from tools.code.evaluate_point_targets import (
    main as point_target_main,
    parse_args as parse_point_target_args,
)
from tools.code import generate_point_target_images
from tools.shared.workflow_common import (
    ALGORITHMS,
    ask_multi_choice,
    configured_choice_index,
    inspect_h5,
    sample_can_be_evaluated,
)
from tools.shared.interactive import run as run_tool_process
from tools.shared.tool_wizard import run_registered_tool
from tools.code.run_wizard import (
    ask_algorithm_param,
    evaluation_enabled,
    reusable_scene_id,
    validate_algorithm_param,
    validate_select_angles,
)


class AlgorithmAuditTests(unittest.TestCase):
    def test_tool_wrapper_does_not_create_output_before_child_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pending-output"
            with patch("tools.shared.interactive.run_process", return_value=0) as run_process:
                run_tool_process("compare_rf_iq.py", ["--output_dir", str(output)])
            self.assertFalse(output.exists())
            self.assertIsNone(run_process.call_args.args[2])

    def test_menu_native_tools_dispatch_to_their_entry_modules(self):
        for script, module in (
            ("run_wizard.py", "tools.code.run_wizard"),
            ("run_ablation.py", "tools.code.run_ablation"),
        ):
            with self.subTest(script=script), patch(f"{module}.main") as main:
                run_registered_tool(script)
                main.assert_called_once_with()

    def test_wizard_enables_evaluation_only_with_display_and_picmus_gt(self):
        self.assertTrue(evaluation_enabled(True, True))
        self.assertFalse(evaluation_enabled(True, False))
        self.assertFalse(evaluation_enabled(False, True))
        self.assertFalse(evaluation_enabled(False, False))

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
            ask_algorithm_param("das", "aperture_mode", "unsupported")
        with self.assertRaisesRegex(ValueError, "config.algorithms"):
            ask_multi_choice("algorithms", list(ALGORITHMS.items()), ["das", "unknown"])
        with (
            patch("tools.shared.workflow_common.read_line", side_effect=["das,das", "das"]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(
                ask_multi_choice("algorithms", list(ALGORITHMS.items()), ["das"]),
                ["das"],
            )

    def test_wizard_numeric_parameter_types_follow_algorithm_cli(self):
        with patch("tools.code.run_wizard.ask_text", return_value="0.5"):
            self.assertEqual(ask_algorithm_param("mv", "mv_dl", 0), 0.5)
        with (
            patch("tools.code.run_wizard.ask_text", side_effect=["1.5", "3"]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(ask_algorithm_param("esbmv", "num_eig", 0.0), 3)

    def test_wizard_accepts_nullable_gcf_defaults_and_single_angle_index(self):
        with patch("tools.code.run_wizard.ask_text", return_value=""):
            self.assertIsNone(ask_algorithm_param("gcfmv", "tx_subarray_ratio", None))
            self.assertIsNone(ask_algorithm_param("gcfmv", "gcf_tx_low_bins", None))
        self.assertEqual(validate_select_angles("0,"), "0,")
        for malformed in ("0,,", ",", "0,1,"):
            with self.subTest(value=malformed), self.assertRaises(ValueError):
                validate_select_angles(malformed)

    def test_wizard_reusable_scene_id_tracks_effective_configuration(self):
        config = {
            "algorithms": ["das"],
            "scenes": [{"id": "old-id", "h5_path": "data/input.h5", "sample_idx": 0}],
            "params": {"select_angles": "1", "f_number": 1.5},
            "algorithm_params": {"das": {"aperture_mode": "discrete"}},
        }
        first = reusable_scene_id(config)
        config["scenes"][0]["id"] = "new-id"
        self.assertEqual(first, reusable_scene_id(config))
        config["params"]["f_number"] = 2.0
        self.assertNotEqual(first, reusable_scene_id(config))

    def test_scene_rejects_truncated_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            h5_path = Path(directory) / "input.h5"
            with h5py.File(h5_path, "w") as handle:
                handle["h5_schema_version"] = 3
                handle.create_dataset("iq_i", data=np.zeros((2, 1, 1, 1)))
                handle.create_dataset("mv_iq_db", data=np.zeros((1, 1, 2, 2)))
                handle.create_dataset("config_yaml", data=yaml.safe_dump({
                    "generation": {
                        "ground_truth": {"algorithm": "mv", "signal_domain": "iq"},
                        "training_targets": {
                            "domain_targets": {"iq": {"mv": {"db": "/mv_iq_db"}}}
                        },
                    }
                }))
            with self.assertRaisesRegex(ValueError, "GT 样本索引超出范围"):
                scene_has_gt({"h5_path": str(h5_path), "sample_idx": 1})

    def test_rf_input_uses_declared_iq_image_reference_without_rf_teacher_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.h5"
            image = np.array([[[[1.0, 0.5], [0.0, 0.25]]]], dtype=np.float32)
            with h5py.File(path, "w") as handle:
                handle["h5_schema_version"] = 3
                handle["reference_iq_db"] = image
                handle["config_yaml"] = yaml.safe_dump({"generation": {
                    "ground_truth": {"algorithm": "reference", "signal_domain": "iq", "dynamic_range_db": 60},
                    "training_targets": {"domain_targets": {"rf": {"reference": {
                        "db": "/reference_iq_db", "kind": "image_reference", "source_domain": "iq",
                    }}}},
                }})
                np.testing.assert_array_equal(
                    ground_truth_db(handle, 0, "rf"), image[0] * 60 - 60,
                )

    def test_scene_detects_complex_teacher_without_display_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            h5_path = Path(directory) / "input.h5"
            config = {
                "generation": {
                    "ground_truth": {"algorithm": "mv", "signal_domain": "iq"},
                    "training_targets": {
                        "domain_targets": {
                            "iq": {"mv": {"signal": {"real": "/mv_iq_i", "imag": "/mv_iq_q"}}}
                        }
                    },
                }
            }
            with h5py.File(h5_path, "w") as handle:
                handle["h5_schema_version"] = 3
                handle.create_dataset("config_yaml", data=yaml.safe_dump(config))
                handle.create_dataset("mv_iq_i", data=np.zeros((1, 1, 2, 2)))
                handle.create_dataset("mv_iq_q", data=np.ones((1, 1, 2, 2)))
            self.assertTrue(scene_has_gt({"h5_path": str(h5_path), "sample_idx": 0}))

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

    def test_signal_domain_ablation_overview_does_not_share_one_gt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            outputs = []
            for index, domain in enumerate(("iq", "rf"), 1):
                scene_dir = root / domain
                scene_dir.mkdir()
                output = np.full((2, 2), -index, dtype=np.float32)
                np.save(scene_dir / "das.npy", output)
                outputs.append(output)
                rows.append(
                    {
                        "status": "ok",
                        "scene_id": domain,
                        "result_dir": str(scene_dir),
                        "parameter": "signal_domain",
                        "value": domain,
                    }
                )
            np.save(root / "ground_truth.npy", np.zeros((2, 2), dtype=np.float32))
            save_ablation_overview(rows, root, "das", "signal_domain")
            comparison = np.load(root / "data" / "ablation_comparison.npy")
            self.assertEqual(comparison.shape, (2, 2, 2))
            np.testing.assert_array_equal(comparison, outputs)

    def test_signal_domain_ablation_keeps_each_ground_truth_with_its_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index, domain in enumerate(("iq", "rf"), 1):
                scene_dir = root / domain
                scene_dir.mkdir()
                (scene_dir / "das.png").write_bytes(b"result")
                (scene_dir / "ground_truth.png").write_bytes(domain.encode())
                rows.append(
                    {
                        "status": "ok",
                        "scene_id": domain,
                        "result_dir": str(scene_dir),
                        "parameter": "signal_domain",
                        "value": domain,
                        "index": index,
                    }
                )
            output_dir = collect_ablation_individual_images(
                rows, root, "das", "signal_domain"
            )
            self.assertEqual(
                (output_dir / "01_signal_domain_iq_ground_truth.png").read_bytes(),
                b"iq",
            )
            self.assertEqual(
                (output_dir / "02_signal_domain_rf_ground_truth.png").read_bytes(),
                b"rf",
            )

    def test_sound_speed_values_require_finite_unique_positive_values(self):
        self.assertEqual(parse_c_values("1430,1540,1620"), [1430.0, 1540.0, 1620.0])
        with self.assertRaisesRegex(ValueError, "至少需要两个取值"):
            parse_c_values("1540")
        with self.assertRaisesRegex(ValueError, "不能重复"):
            parse_c_values("1540.0001,1540.0002")
        with self.assertRaisesRegex(ValueError, "有限正数"):
            parse_c_values("-1,1540")
        with self.assertRaisesRegex(ValueError, "逗号分隔的数字"):
            parse_c_values("fast,1540")

    def test_sound_speed_metric_method_label(self):
        self.assertEqual(method_c("c=1540"), 1540.0)
        self.assertIsNone(method_c("c=invalid"))

    def test_runner_common_overrides_and_stored_gt_range(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            h5_path = root / "input.h5"
            gt = np.array([[1.0, 0.5], [0.0, 0.25]], dtype=np.float32)
            with h5py.File(h5_path, "w") as handle:
                handle["h5_schema_version"] = 3
                handle.create_dataset("iq_i", data=np.zeros((1, 1, 2, 1)))
                handle.create_dataset("x_grid", data=[-0.001, 0.001])
                handle.create_dataset("z_grid", data=[0.001, 0.002])
                handle.create_dataset("mv_iq_db", data=gt[None, None])
                handle.create_dataset("config_yaml", data=yaml.safe_dump({
                    "generation": {
                        "ground_truth": {
                            "algorithm": "mv",
                            "signal_domain": "iq",
                            "dynamic_range_db": 60,
                        },
                        "training_targets": {
                            "domain_targets": {
                                "iq": {"mv": {"db": "/mv_iq_db"}}
                            }
                        },
                    }
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

    def test_runner_rejects_unsupported_algorithm_before_reading_scene(self):
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "output"
            args = argparse.Namespace(
                config=str(project_root / "config.yaml"),
                scene="simulation_contrast_speckle",
                algorithms="das,missing_algorithm",
                output_root=str(output_root),
                python_exe=sys.executable,
                no_evaluate=True,
                no_individual_images=True,
                keep_existing=False,
                extra_algorithm_args=[],
            )
            with patch("run_one.parse_args", return_value=args):
                with self.assertRaisesRegex(FileNotFoundError, "Algorithm module not found"):
                    run_scene()
            self.assertFalse(output_root.exists())

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
            with patch("algorithms.shared_core.common.PROJECT_ROOT", root), patch(
                "algorithms.shared_core.common.algorithm_source_paths", return_value=()
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

    def test_dynamic_aperture_saturates_at_physical_channel_count(self):
        pitch = 0.00023
        self.assertEqual(dynamic_aperture_channel_count(pitch, 1.0, pitch, 192, True), 2)
        self.assertEqual(dynamic_aperture_channel_count(0.1, 1.0, pitch, 192, True), 192)
        self.assertEqual(dynamic_aperture_channel_count(0.1, 1.0, pitch, 192, False), 192)

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
                            "signal_domain": "iq",
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

    def test_ablation_keep_existing_finds_matching_prior_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            expected = {"angles1": {"params": {"select_angles": "1"}}}
            matching = root / "ablation_20261025_120000" / "configs"
            matching.mkdir(parents=True)
            (matching / "angles1.yaml").write_text(
                yaml.safe_dump(expected["angles1"]), encoding="utf-8"
            )
            newer = root / "ablation_20261026_120000" / "configs"
            newer.mkdir(parents=True)
            (newer / "angles1.yaml").write_text(
                yaml.safe_dump({"params": {"select_angles": "all"}}),
                encoding="utf-8",
            )

            self.assertEqual(find_matching_ablation_run(root, expected), matching.parent)

    def test_ablation_keep_existing_rejects_incomplete_prior_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            expected = {
                "angles1": {"params": {"select_angles": "1"}},
                "angles3": {"params": {"select_angles": "3"}},
            }
            config_dir = root / "ablation_20261026_120000" / "configs"
            config_dir.mkdir(parents=True)
            (config_dir / "angles1.yaml").write_text(
                yaml.safe_dump(expected["angles1"]), encoding="utf-8"
            )

            self.assertIsNone(find_matching_ablation_run(root, expected))

    def test_rf_iq_time_axis_accepts_a_valid_prefix_only(self):
        full = np.arange(8, dtype=np.float64) / 20.0
        np.testing.assert_allclose(validate_time_axis(full, 5, 0.0, 20.0), full[:5])
        with self.assertRaisesRegex(ValueError, "时间轴"):
            validate_time_axis(full + 0.01, 5, 0.0, 20.0)

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

    def test_nullable_numeric_ablation_defaults_reject_booleans(self):
        with self.assertRaisesRegex(ValueError, "tx_subarray_ratio"):
            validate_values("tx_subarray_ratio", None, [True])
        with self.assertRaisesRegex(ValueError, "gcf_tx_low_bins"):
            validate_values("gcf_tx_low_bins", None, [False])

    def test_point_target_duplicate_methods_and_loadings_are_rejected(self):
        for option, values in (("--methods", ("mv", "mv")), ("--loadings", ("0.1", "0.1"))):
            with self.subTest(option=option), patch.object(
                sys, "argv", ["evaluate_point_targets.py", option, *values]
            ), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    parse_point_target_args()
                self.assertEqual(error.exception.code, 2)

    def test_point_target_cli_reports_missing_optional_simulator(self):
        with (
            patch("tools.code.evaluate_point_targets.load_reference", None),
            patch("tools.code.evaluate_point_targets.simulate_rf", None),
            patch.object(sys, "argv", ["evaluate_point_targets.py"]),
            self.assertRaisesRegex(
                ModuleNotFoundError,
                "requires data.scripts.simulate_point_targets",
            ),
        ):
            point_target_main()

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
            with patch("algorithms.shared_core.common.save_figure"):
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
                        signal_domain="iq", aperture_mode=aperture, dynamic_aperture=False, f_number=1.5,
                        window="rect", row_block=1, interp=interpolation,
                    )
                    with (
                        patch.object(das, "args", args, create=True),
                        patch.object(das, "resolve_device", return_value=torch.device("cpu")),
                    ):
                        beamformer = das.DASBeamformerIQ(
                            np.array([1.5, 2.0, 2.5]), np.array([0.0]),
                            1, 1.0, 1.0, 0.0, 1.0, np.zeros(1), angles, options=args, device="cpu",
                        )
                        i_output, q_output = beamformer(
                            i_data, q_data, angles, np.zeros(1), 1.0
                        )
                    np.testing.assert_allclose(i_output[:, 0], [0.0, 7.0, 0.0], atol=1e-6)
                    np.testing.assert_allclose(q_output, 0.0, atol=1e-6)

    def test_non_comparable_ablation_skips_metric_subprocesses(self):
        for parameter in ("dr", "signal_domain"):
            with self.subTest(parameter=parameter), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                np.save(root / "ablation_comparison.npy", np.zeros((2, 2, 2)))
                rows = [{"status": "ok", "parameter": parameter, "value": 40, "index": 1}]
                with patch("tools.code.run_ablation.run_process") as run:
                    result = run_ablation_evaluation(
                        rows, root, root / "input.h5", 0, "das", True, 60
                    )
                self.assertIsNone(result)
                run.assert_not_called()
                self.assertFalse((root / "metrics").exists())

    def test_numeric_range_rejects_nonfinite_and_nonadvancing_steps(self):
        with patch("tools.code.run_ablation.ask_text", side_effect=["0", "1", "inf"]):
            with self.assertRaisesRegex(ValueError, "有限数"):
                numeric_range_values()
        with patch(
            "tools.code.run_ablation.ask_text",
            side_effect=["1e308", "1e308", "1"],
        ):
            with self.assertRaisesRegex(ValueError, "无法推进"):
                numeric_range_values()

    def test_integer_ablation_range_returns_integers_and_categorical_values_are_checked(self):
        with patch("tools.code.run_ablation.ask_text", side_effect=["1", "5", "2"]):
            self.assertEqual(numeric_range_values(integer=True), [1, 3, 5])
        with (
            patch("tools.code.run_ablation.ask_choice", return_value="range"),
            patch("tools.code.run_ablation.ask_text", side_effect=["1", "5", "2"]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            values = choose_values("temporal_win", 3)
        self.assertEqual(values, [1, 3, 5])
        self.assertTrue(all(isinstance(value, int) for value in values))
        self.assertEqual(validate_values("signal_domain", "iq", ["rf", "iq"]), ["rf", "iq"])
        self.assertFalse(should_evaluate_ablation("signal_domain", True))
        self.assertFalse(should_evaluate_ablation("dr", True))
        self.assertTrue(should_evaluate_ablation("mv_dl", True))
        self.assertFalse(should_evaluate_ablation("mv_dl", False))
        self.assertEqual(validate_available_signal_domains(["iq"], ["iq"]), ["iq"])
        with self.assertRaisesRegex(ValueError, "当前 H5 不支持 signal_domain"):
            validate_available_signal_domains(["rf", "iq"], ["iq"])
        with self.assertRaisesRegex(ValueError, "mv_axis"):
            validate_values("mv_axis", "rx", ["invalid"])

    def test_compare_rf_iq_rejects_nonfinite_continuous_arguments(self):
        for option, value in (("--mv_dl", "nan"), ("--subarray_ratio", "inf"),
                              ("--f_number", "nan"), ("--dr", "inf"),
                              ("--iq_gain", "nan")):
            with self.subTest(option=option), patch(
                "sys.argv", ["compare_rf_iq", "--h5_path", "sample.h5", option, value]
            ), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                compare_rf_iq_main()

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
                    "schema_version": 3,
                    "dataset": {"id": "simulation"},
                    "generation": {
                        "storage": {
                            "inputs": {"iq": {"dtype": "float32", "normalization": "none"}},
                            "outputs": {"dtype": "float32"},
                        },
                        "ground_truth": {"algorithm": "mv", "signal_domain": "iq"},
                        "training_targets": {
                            "domain_targets": {
                                "iq": {"mv": {"db": "/mv_iq_db"}}
                            }
                        },
                    },
                    "source_samples": [
                        {
                            "acquisition_id": "simulation_contrast_speckle",
                            "phantom_mode": "in_vitro",
                            "phantom_source": "in_vitro",
                        }
                    ],
                }
                with h5py.File(h5_path, "w") as hf:
                    hf.create_dataset("h5_schema_version", data=3)
                    hf.create_dataset("iq_i", data=np.zeros((1, 3, 8, 4)))
                    hf.create_dataset("iq_q", data=np.zeros((1, 3, 8, 4)))
                    hf.create_dataset("acquisition_id", data=["simulation_contrast_speckle"])
                    hf.create_dataset("iq_length", data=[8])
                    hf.create_dataset("iq_t0", data=np.zeros((1, 3)))
                    hf.create_dataset("iq_angles", data=[-0.1, 0.0, 0.1])
                    hf.create_dataset("z_grid", data=[0.001, 0.002])
                    hf.create_dataset("x_grid", data=[-0.001, 0.001])
                    hf.create_dataset("iq_fs", data=20e6)
                    hf.create_dataset("c", data=1540.0)
                    hf.create_dataset("fc", data=5e6)
                    hf.create_dataset("pitch", data=0.0003)
                    hf.create_dataset("num_channels", data=4)
                    hf.create_dataset("mv_iq_db", data=np.zeros(gt_shape))
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
                    data=yaml.safe_dump({**embedded_config, "source_samples": [{}]}),
                    )
                with self.assertRaisesRegex(ValueError, "必须包含 id 或 acquisition_id"):
                    inspect_h5(h5_path)

    def test_wizard_lists_derived_domains_for_rf_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            h5_path = Path(temp_dir) / "rf-only.h5"
            config = {
                "schema_version": 3,
                "dataset": {"id": "rf_sample"},
                "generation": {
                    "storage": {
                        "inputs": {"rf": {"dtype": "float32", "normalization": "none"}},
                        "outputs": {"dtype": "float32"},
                    },
                    "input_iq": {"decimation_factor": 1},
                    "ground_truth": {"algorithm": "mv", "signal_domain": "rf"},
                    "training_targets": {"domain_targets": {}},
                },
                "source_samples": [{"id": "rf_sample", "phantom_mode": "in_vivo"}],
            }
            with h5py.File(h5_path, "w") as hf:
                hf.create_dataset("h5_schema_version", data=3)
                hf.create_dataset("config_yaml", data=yaml.safe_dump(config))
                hf.create_dataset("acquisition_id", data=["rf_sample"])
                hf.create_dataset("body_region", data=["region"])
                hf.create_dataset("rf", data=np.zeros((1, 1, 8, 4), dtype=np.float32))
                hf.create_dataset("rf_fs", data=20e6)
                hf.create_dataset("rf_angles", data=[0.0])
                hf.create_dataset("rf_t0", data=np.zeros((1, 1)))
                hf.create_dataset("rf_length", data=[8])
                hf.create_dataset("z_grid", data=[0.001, 0.002])
                hf.create_dataset("x_grid", data=[-0.001, 0.001])
                hf.create_dataset("c", data=1540.0)
                hf.create_dataset("fc", data=5e6)
                hf.create_dataset("pitch", data=0.0003)
                hf.create_dataset("num_channels", data=4)

            info = inspect_h5(h5_path)
            self.assertEqual(info["domains"], ["rf", "analytic_rf", "iq"])
            self.assertEqual(info["default_domain"], "iq")
            for encoding in ("root_labels", "root_labels_with_path_template"):
                config["source_samples"] = {
                    "encoding": encoding, "count": 1,
                    "acquisition_id_dataset": "/acquisition_id",
                    "phantom_mode": "in_vivo", "phantom_source": "in_vivo",
                }
                with h5py.File(h5_path, "r+") as hf:
                    hf["config_yaml"][()] = yaml.safe_dump(config)
                info = inspect_h5(h5_path)
                self.assertEqual(info["in_vivo"], [True])
                self.assertEqual(info["names"], ["rf_sample"])
            config["source_samples"]["encoding"] = "unsupported"
            with h5py.File(h5_path, "r+") as hf:
                hf["config_yaml"][()] = yaml.safe_dump(config)
            with self.assertRaisesRegex(ValueError, "来源标识"):
                inspect_h5(h5_path)


class PointTargetSceneConfigTests(unittest.TestCase):
    def test_resolution_scenes_resolve_phantoms_under_picmus_root(self):
        root = Path(__file__).resolve().parents[1]
        picmus_root = root / "data" / "PICMUS"
        try:
            scenes = generate_point_target_images.load_scenes(root / "config.yaml")
        except FileNotFoundError as exc:
            self.skipTest(str(exc))
        if not picmus_root.is_dir():
            self.skipTest("PICMUS database unavailable")

        self.assertEqual(
            [scene["id"] for scene in scenes],
            [
                "simulation_resolution_distortion",
                "experiments_resolution_distortion",
            ],
        )
        expected_targets = {
            "simulation_resolution_distortion": 20,
            "experiments_resolution_distortion": 7,
        }
        expected_phantoms = {}
        for scene in scenes:
            meta = generate_point_target_images.read_sample_meta(
                scene["h5_path"], scene["sample_idx"]
            )
            expected = (picmus_root / meta["phantom_path"]).resolve()
            if not expected.is_file():
                self.skipTest(f"phantom file unavailable: {expected}")
            expected_phantoms[scene["id"]] = expected

        for scene in scenes:
            context = generate_point_target_images.load_picmus_context(
                scene["h5_path"], scene["sample_idx"]
            )
            self.assertEqual(context["mode"], "resolution_distortion")
            self.assertEqual(context["source"], scene["expected_source"])
            self.assertEqual(
                Path(context["phantom_path"]).resolve(),
                expected_phantoms[scene["id"]],
            )
            self.assertEqual(
                len(context["phantom"]["resolution_targets"]),
                expected_targets[scene["id"]],
            )


if __name__ == "__main__":
    unittest.main()

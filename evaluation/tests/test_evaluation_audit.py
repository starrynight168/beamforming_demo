import csv
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
import yaml

from evaluation.evaluate import (
    load_method_names,
    picmus_distortion_z_offset_mm,
    read_sample_meta,
    sample_evaluation_enabled,
)
from evaluation.evaluate import (
    main as evaluate_main,
)
from evaluation.plot_metrics import (
    infer_repeat_references,
    save_lateral_profile_plot,
)
from evaluation.plot_metrics import (
    main as plot_main,
)


def _write_scene_config(root, h5_path):
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump({"scenes": [{
        "id": "custom_scene", "h5_path": str(h5_path), "sample_idx": 0,
        "allow_evaluation": True,
    }]}), encoding="utf-8")
    return path


class EvaluationInputTests(unittest.TestCase):
    def test_disabled_scene_skips_metric_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_scene_config(root, root / "sample.h5")
            config = yaml.safe_load(path.read_text())
            config["scenes"][0]["allow_evaluation"] = False
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            argv = ["evaluate.py", "--config", str(path),
                    "--h5_path", str(root / "sample.h5"),
                    "--comparison_npy", str(root / "comparison.npy"),
                    "--out_dir", str(root / "metrics")]
            with patch.object(sys, "argv", argv):
                evaluate_main()
            self.assertFalse((root / "metrics").exists())

    def test_enabled_scene_rejects_unsupported_phantom_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h5_path = root / "sample.h5"
            config_path = _write_scene_config(root, h5_path)
            for mode, source in (("in_vivo", "in_vivo"), ("unknown", "simulation"),
                                 ("", "simulation"), ("contrast_speckle", "")):
                with self.subTest(mode=mode, source=source):
                    with h5py.File(h5_path, "w") as handle:
                        handle["all_multi_I"] = np.zeros((1, 1, 1, 1))
                        handle["config_yaml"] = yaml.safe_dump({
                            "source_samples": [{"id": "custom", "phantom_mode": mode,
                                                "phantom_source": source}]
                        })
                    argv = ["evaluate.py", "--config", str(config_path),
                            "--h5_path", str(h5_path)]
                    if mode == "in_vivo":
                        argv.extend(["--phantom_mode", "contrast_speckle",
                                     "--phantom_source", "simulation"])
                    with patch.object(sys, "argv", argv):
                        with self.assertRaisesRegex(ValueError, "不支持|缺少"):
                            evaluate_main()

    def test_sample_policy_uses_path_and_index_instead_of_scene_name(self):
        h5_path = Path("data/custom.h5").resolve()
        scene = {"id": "renamed", "h5_path": "data/custom.h5", "sample_idx": 2,
                 "allow_evaluation": True}
        config = {"scenes": [scene]}
        self.assertTrue(sample_evaluation_enabled(config, h5_path, 2))
        self.assertFalse(sample_evaluation_enabled(config, h5_path, 0))
        self.assertFalse(sample_evaluation_enabled(config, "data/other.h5", 2))
        scene["allow_evaluation"] = False
        self.assertFalse(sample_evaluation_enabled(config, h5_path, 2))
        del scene["allow_evaluation"]
        self.assertFalse(sample_evaluation_enabled(config, h5_path, 2))

    def test_sample_policy_rejects_invalid_and_ambiguous_configuration(self):
        scene = {"id": "custom", "h5_path": "data/custom.h5", "sample_idx": 0,
                 "allow_evaluation": True}
        invalid = (("allow_evaluation", "false"), ("sample_idx", True),
                   ("sample_idx", -1), ("h5_path", None))
        for name, value in invalid:
            with self.subTest(name=name, value=value):
                with self.assertRaisesRegex(ValueError, name):
                    sample_evaluation_enabled(
                        {"scenes": [{**scene, name: value}]}, "data/custom.h5", 0
                    )
        with self.assertRaisesRegex(ValueError, "不能重复"):
            sample_evaluation_enabled({"scenes": [scene, scene]}, "data/custom.h5", 0)

    def test_method_loader_rejects_non_object_run_params(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            comparison = Path(directory) / "comparison.npy"
            params = comparison.parent / "run_params.json"
            params.write_text("[]", encoding="utf-8")
            args = SimpleNamespace(methods=None, method_labels=None)

            with self.assertRaisesRegex(ValueError, "顶层必须是对象"):
                load_method_names(args, str(comparison), n_panels=1, has_gt=False)

    def test_method_loader_rejects_non_object_evaluation_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            comparison = Path(directory) / "comparison.npy"
            params = comparison.parent / "run_params.json"
            params.write_text(
                json.dumps({"models": {"model": {}}, "evaluation": []}),
                encoding="utf-8",
            )
            args = SimpleNamespace(methods=None, method_labels=None)

            with self.assertRaisesRegex(ValueError, "evaluation 必须是对象"):
                load_method_names(args, str(comparison), n_panels=1, has_gt=False)

    def test_method_loader_requires_boolean_skip_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            comparison = Path(directory) / "comparison.npy"
            params = comparison.parent / "run_params.json"
            params.write_text(
                json.dumps({"models": {"model": {}}, "evaluation": {"skip_baselines": "false"}}),
                encoding="utf-8",
            )
            args = SimpleNamespace(methods=None, method_labels=None)

            with self.assertRaisesRegex(ValueError, "skip_baselines 必须是布尔值"):
                load_method_names(args, str(comparison), n_panels=1, has_gt=False)

    def test_sample_metadata_rejects_non_object_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.h5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset("all_multi_I", data=np.zeros((1, 1, 1, 1)))
                handle.create_dataset(
                    "config_yaml",
                    data=b"dataset: []\nsource_samples:\n  - id: sample\n",
                )

            with self.assertRaisesRegex(ValueError, "dataset must be a mapping"):
                read_sample_meta(path, sample_idx=0)

    def test_picmus_config_rejects_non_object_evaluation_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.h5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset("config_yaml", data=b"evaluation: []\n")

            with self.assertRaisesRegex(ValueError, "config_yaml.evaluation must be a mapping"):
                picmus_distortion_z_offset_mm(path, "simulation")

    def test_evaluation_and_plotting_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h5_path = root / "sample.h5"
            comparison_path = root / "comparison.npy"
            metrics_dir = root / "metrics"
            image = np.linspace(-50.0, 0.0, 25).reshape(5, 5)
            np.save(comparison_path, np.stack((image, image - 1.0)))
            with h5py.File(h5_path, "w") as handle:
                handle.create_dataset("all_multi_I", data=np.zeros((1, 1, 1, 1)))
                handle.create_dataset("all_envdb_norm", data=np.ones((1, 1, 5, 5)))
                handle.create_dataset("x_grid", data=np.linspace(-0.002, 0.002, 5))
                handle.create_dataset("z_grid", data=np.linspace(0.02, 0.024, 5))
                handle.create_dataset(
                    "config_yaml",
                    data=(
                        b"dataset:\n  phantom_mode: contrast_speckle\n"
                        b"  phantom_source: simulation\nsource_samples:\n"
                        b"  - id: renamed_contrast_sample\n"
                        b"    phantom_mode: contrast_speckle\n"
                        b"    phantom_source: simulation\n"
                    ),
                )
            evaluate_args = SimpleNamespace(
                config=str(_write_scene_config(root, h5_path)),
                comparison_npy=str(comparison_path),
                methods="model",
                method_labels=None,
                h5_path=str(h5_path),
                h5_sample_idx=0,
                dr=60.0,
                out_dir=str(metrics_dir),
                phantom_path="none",
                phantom_mode="auto",
                phantom_source="auto",
                auto_roi_radius_mm=1.0,
                auto_target_count=4,
                auto_target_min_distance_mm=1.0,
                fwhm_window_mm=1.0,
            )
            command = [sys.executable, str(Path(__file__).resolve().parents[1] / "evaluate.py")]
            for name, value in vars(evaluate_args).items():
                if value is not None:
                    command.extend([f"--{name}", str(value)])
            subprocess.run(command, check=True, capture_output=True, text=True)

            plot_args = SimpleNamespace(
                metrics_dir=str(metrics_dir),
                dr=None,
                reference_mode="auto",
                profile_roi_index=None,
                profile_target_index=None,
            )
            with patch("evaluation.plot_metrics.parse_args", return_value=plot_args):
                plot_main()

            self.assertTrue((metrics_dir / "summary_metrics.csv").is_file())
            self.assertTrue((metrics_dir / "evaluation_meta.json").is_file())
            self.assertTrue((metrics_dir / "auxiliary_metrics.png").is_file())
            self.assertTrue((metrics_dir / "roi_targets.png").is_file())
            self.assertTrue((metrics_dir / "cyst_profile.png").is_file())
            with (metrics_dir / "summary_metrics.csv").open(encoding="utf-8-sig") as file:
                rows = list(csv.DictReader(file))
            model_row = next(row for row in rows if row["method"] != "GT")
            self.assertAlmostEqual(float(model_row["PSNR_dB_vs_GT"]), 20 * np.log10(60.0))
            metadata = json.loads((metrics_dir / "evaluation_meta.json").read_text())
            self.assertEqual(metadata["sample_name"], "renamed_contrast_sample")
            self.assertEqual(metadata["config_path"], evaluate_args.config)

    def test_resolution_scene_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h5_path = root / "sample.h5"
            comparison_path = root / "comparison.npy"
            metrics_dir = root / "metrics"
            ground_truth = np.full((5, 5), -40.0)
            ground_truth[2, 2] = -2.0
            ground_truth[1, 3] = -5.0
            np.save(comparison_path, np.stack((ground_truth, ground_truth - 1.0)))
            with h5py.File(h5_path, "w") as handle:
                handle.create_dataset("all_multi_I", data=np.zeros((1, 1, 1, 1)))
                handle.create_dataset("all_envdb_norm", data=np.ones((1, 1, 5, 5)))
                handle.create_dataset("x_grid", data=np.linspace(0.0, 0.004, 5))
                handle.create_dataset("z_grid", data=np.linspace(0.02, 0.024, 5))
                handle.create_dataset(
                    "config_yaml",
                    data=(
                        b"dataset:\n  phantom_mode: resolution_distorsion\n"
                        b"  phantom_source: simulation\nsource_samples:\n"
                        b"  - id: renamed_resolution_sample\n"
                        b"    phantom_mode: resolution_distorsion\n"
                        b"    phantom_source: simulation\n"
                    ),
                )
            evaluate_args = SimpleNamespace(
                config=str(_write_scene_config(root, h5_path)),
                comparison_npy=str(comparison_path),
                methods="model",
                method_labels=None,
                h5_path=str(h5_path),
                h5_sample_idx=0,
                dr=60.0,
                out_dir=str(metrics_dir),
                phantom_path="none",
                phantom_mode="auto",
                phantom_source="auto",
                auto_roi_radius_mm=1.0,
                auto_target_count=2,
                auto_target_min_distance_mm=1.0,
                fwhm_window_mm=1.0,
            )
            with patch("evaluation.evaluate.parse_args", return_value=evaluate_args):
                evaluate_main()

            plot_args = SimpleNamespace(
                metrics_dir=str(metrics_dir),
                dr=None,
                reference_mode="auto",
                profile_roi_index=None,
                profile_target_index=None,
            )
            with patch("evaluation.plot_metrics.parse_args", return_value=plot_args):
                plot_main()

            self.assertTrue((metrics_dir / "resolution_target_metrics.csv").is_file())
            self.assertTrue((metrics_dir / "resolution_group_metrics.csv").is_file())
            self.assertTrue((metrics_dir / "point_profile.png").is_file())

    def test_plot_reader_defaults_for_non_object_run_params(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = root / "metrics"
            metrics.mkdir()
            (root / "run_params.json").write_text("[]", encoding="utf-8")

            self.assertTrue(infer_repeat_references(metrics, "auto"))

    def test_plot_reader_rejects_non_object_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics = Path(directory)
            (metrics / "evaluation_meta.json").write_text("[]", encoding="utf-8")
            args = SimpleNamespace(
                metrics_dir=str(metrics),
                dr=None,
                reference_mode="auto",
                profile_roi_index=None,
                profile_target_index=None,
            )
            with patch("evaluation.plot_metrics.parse_args", return_value=args):
                with self.assertRaisesRegex(ValueError, "evaluation_meta.json 顶层"):
                    plot_main()


class PlotOutputTests(unittest.TestCase):
    def test_missing_comparison_cleans_stale_profile_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics = Path(directory)
            (metrics / "evaluation_meta.json").write_text(
                json.dumps({"comparison_npy": "missing.npy", "h5_path": "missing.h5"}),
                encoding="utf-8",
            )
            stale_paths = [
                metrics / "roi_targets.png",
                metrics / "cyst_profile.png",
                metrics / "cyst_profile_page2.png",
                metrics / "point_profile.png",
                metrics / "profile_selection.json",
            ]
            for path in stale_paths:
                path.write_bytes(b"stale")
            args = SimpleNamespace(
                metrics_dir=str(metrics),
                dr=None,
                reference_mode="auto",
                profile_roi_index=None,
                profile_target_index=None,
            )

            with patch("evaluation.plot_metrics.parse_args", return_value=args):
                plot_main()

            self.assertFalse(any(path.exists() for path in stale_paths))

    def test_invalid_profile_selection_preserves_existing_plots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            existing = [
                output / "cyst_profile.png",
                output / "cyst_profile_page2.png",
                output / "point_profile.png",
            ]
            for path in existing:
                path.write_bytes(b"keep")

            with self.assertRaisesRegex(ValueError, "Cyst ROI为空"):
                save_lateral_profile_plot(
                    output,
                    np.zeros((1, 2, 2)),
                    np.array([0.0, 1.0]),
                    np.array([0.0, 1.0]),
                    rois=[],
                    targets=[],
                    methods=["DAS"],
                    profile_roi_index=1,
                )

            self.assertEqual([path.read_bytes() for path in existing], [b"keep"] * 3)


if __name__ == "__main__":
    unittest.main()

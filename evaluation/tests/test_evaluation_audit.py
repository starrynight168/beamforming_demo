import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np

from evaluation.evaluate import (
    load_method_names,
    main as evaluate_main,
    picmus_distortion_z_offset_mm,
    read_sample_meta,
)
from evaluation.plot_metrics import (
    infer_repeat_references,
    main as plot_main,
    save_lateral_profile_plot,
)


class EvaluationInputTests(unittest.TestCase):
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
                        b"  - id: simulation_contrast_speckle\n"
                        b"    phantom_mode: contrast_speckle\n"
                        b"    phantom_source: simulation\n"
                    ),
                )
            evaluate_args = SimpleNamespace(
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

            self.assertTrue((metrics_dir / "summary_metrics.csv").is_file())
            self.assertTrue((metrics_dir / "evaluation_meta.json").is_file())
            self.assertTrue((metrics_dir / "auxiliary_metrics.png").is_file())
            self.assertTrue((metrics_dir / "roi_targets.png").is_file())
            self.assertTrue((metrics_dir / "cyst_profile.png").is_file())

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
                        b"  - id: simulation_resolution_distorsion\n"
                        b"    phantom_mode: resolution_distorsion\n"
                        b"    phantom_source: simulation\n"
                    ),
                )
            evaluate_args = SimpleNamespace(
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

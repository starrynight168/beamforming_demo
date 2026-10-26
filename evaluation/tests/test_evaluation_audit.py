import csv
import json
import shutil
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.evaluate import (
    load_method_names,
    load_picmus_context,
    picmus_distortion_z_offset_mm,
    picmus_reference_path,
    read_phantom,
    read_sample_meta,
    target_resolution,
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


def write_picmus_sample(
    root, mode="contrast_speckle", source="simulation", has_gt=True
):
    path = root / "sample.h5"
    phantom_path = (
        str(picmus_reference_path(mode, source))
        if source in {"simulation", "experiments"}
        else ""
    )
    with h5py.File(path, "w") as handle:
        handle["all_multi_I"] = np.zeros((1, 1, 1, 1))
        handle["x_grid"] = np.linspace(-0.025, 0.025, 201)
        handle["z_grid"] = np.linspace(0.005, 0.055, 201)
        if has_gt:
            handle["all_envdb_norm"] = np.zeros((1, 1, 201, 201))
        handle["config_yaml"] = yaml.safe_dump(
            {
                "source_samples": [
                    {
                        "id": "renamed_sample",
                        "phantom_mode": mode,
                        "phantom_source": source,
                        "phantom_path": phantom_path,
                    }
                ]
            }
        )
    return path


def update_sample_metadata(path, **changes):
    with h5py.File(path, "r+") as handle:
        config = yaml.safe_load(handle["config_yaml"][()])
        config["source_samples"][0].update(changes)
        del handle["config_yaml"]
        handle["config_yaml"] = yaml.safe_dump(config)


class EvaluationInputTests(unittest.TestCase):
    def test_method_loader_uses_root_algorithm_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {
                "algorithms": ["das", "mv"],
                "algorithm_labels": {"das": "DAS", "mv": "MV"},
            }
            (root / "run_params.json").write_text(json.dumps(metadata), encoding="utf-8")
            args = SimpleNamespace(methods=None, method_labels=None)
            self.assertEqual(
                load_method_names(args, str(root / "comparison.npy"), 3, True),
                ["GT", "DAS", "MV"],
            )

    def test_method_labels_must_match_explicit_method_count(self):
        args = SimpleNamespace(methods="das,mv", method_labels="DAS")
        with self.assertRaisesRegex(ValueError, "method_labels.*数量"):
            load_method_names(args, "comparison.npy", n_panels=2, has_gt=True)
        args = SimpleNamespace(methods=None, method_labels="DAS")
        with self.assertRaisesRegex(ValueError, "method_labels.*必须与 --methods"):
            load_method_names(args, "comparison.npy", n_panels=2, has_gt=True)

    def test_non_picmus_sample_skips_without_reading_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_picmus_sample(root, mode="in_vivo", source="in_vivo")
            argv = [
                "evaluate.py",
                "--h5_path",
                str(path),
                "--comparison_npy",
                str(root / "missing.npy"),
                "--out_dir",
                str(root / "metrics"),
            ]
            with patch.object(sys, "argv", argv):
                evaluate_main()
            self.assertFalse((root / "metrics").exists())

    def test_metadata_must_define_a_consistent_protocol(self):
        for mode, source in (
            ("unknown", "simulation"),
            ("contrast_speckle", "unknown"),
            ("", "simulation"),
            ("contrast_speckle", ""),
        ):
            with (
                self.subTest(mode=mode, source=source),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = write_picmus_sample(Path(directory))
                update_sample_metadata(path, phantom_mode=mode, phantom_source=source)
                with self.assertRaisesRegex(ValueError, "不支持|缺少"):
                    load_picmus_context(path, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = write_picmus_sample(
                Path(directory), mode="in_vivo", source="in_vivo"
            )
            with self.assertRaisesRegex(ValueError, "不一致"):
                load_picmus_context(
                    path,
                    0,
                    phantom_mode="contrast_speckle",
                    phantom_source="simulation",
                )

    def test_all_four_protocols_use_metadata_independent_of_sample_name(self):
        for mode in ("contrast_speckle", "resolution_distortion"):
            for source in ("simulation", "experiments"):
                with (
                    self.subTest(mode=mode, source=source),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    path = write_picmus_sample(
                        Path(directory), mode=mode, source=source
                    )
                    context = load_picmus_context(path, 0)
                    self.assertEqual(
                        (context["mode"], context["source"]), (mode, source)
                    )
                    self.assertEqual(context["meta"]["sample_name"], "renamed_sample")
                    self.assertTrue(context["meta"]["has_gt"])
        with tempfile.TemporaryDirectory() as directory:
            path = write_picmus_sample(
                Path(directory), mode="in_vitro", source="in_vitro"
            )
            self.assertIsNone(load_picmus_context(path, 0))

    def test_missing_phantom_never_uses_defaults_or_gt_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_picmus_sample(root)
            update_sample_metadata(path, phantom_path="")
            with self.assertRaisesRegex(ValueError, "缺少 phantom_path"):
                load_picmus_context(path, 0)
            update_sample_metadata(path, phantom_path=str(root / "missing.h5"))
            with self.assertRaises(FileNotFoundError):
                load_picmus_context(path, 0)
            with self.assertRaisesRegex(ValueError, "必须提供"):
                load_picmus_context(path, 0, phantom_path="none")

    def test_phantom_parameters_and_target_order_must_match_official_definition(self):
        for mode, dataset, operation in (
            ("contrast_speckle", "phantom_occlusionDiameter", "change"),
            ("contrast_speckle", "phantom_lateralResolution", "delete"),
            ("resolution_distortion", "phantom_xPts", "reverse"),
        ):
            with (
                self.subTest(dataset=dataset),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                path = write_picmus_sample(root, mode=mode)
                phantom = root / "modified_phantom.h5"
                shutil.copyfile(picmus_reference_path(mode, "simulation"), phantom)
                with h5py.File(phantom, "r+") as handle:
                    key = "US/US_DATASET0000/" + dataset
                    if operation == "delete":
                        del handle[key]
                    elif operation == "reverse":
                        handle[key][...] = handle[key][:][::-1]
                    else:
                        handle[key][...] = handle[key][:] * 2
                update_sample_metadata(path, phantom_path=str(phantom))
                with self.assertRaisesRegex(ValueError, "模体定义"):
                    load_picmus_context(path, 0)

    def test_grid_must_cover_complete_evaluation_regions(self):
        for mode in ("contrast_speckle", "resolution_distortion"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                path = write_picmus_sample(Path(directory), mode=mode)
                with h5py.File(path, "r+") as handle:
                    handle["x_grid"][...] = np.linspace(-0.002, 0.002, 201)
                with self.assertRaisesRegex(ValueError, "未完整覆盖"):
                    load_picmus_context(path, 0)

    def test_distortion_correction_must_keep_targets_inside_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_picmus_sample(Path(directory), mode="resolution_distortion")
            with h5py.File(path, "r+") as handle:
                config = yaml.safe_load(handle["config_yaml"][()])
                config["evaluation"] = {"picmus_distortion_z_offset_mm": 100.0}
                del handle["config_yaml"]
                handle["config_yaml"] = yaml.safe_dump(config)
            with self.assertRaisesRegex(ValueError, "未完整覆盖"):
                load_picmus_context(path, 0)

    def test_bad_picmus_input_fails_before_reconstruction_or_output_creation(self):
        from run_one import main as run_main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_picmus_sample(root)
            update_sample_metadata(path, phantom_path="")
            config = root / "config.yaml"
            config.write_text(
                yaml.safe_dump(
                    {
                        "scenes": [
                            {"id": "renamed", "h5_path": str(path), "sample_idx": 0}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                config=str(config), scene="renamed", no_evaluate=False,
                extra_algorithm_args=[],
            )
            with (
                patch("run_one.parse_args", return_value=args),
                patch("run_one.subprocess.run") as process,
            ):
                with self.assertRaisesRegex(ValueError, "phantom_path"):
                    run_main()
            process.assert_not_called()
            self.assertEqual(
                sorted(p.name for p in root.iterdir()), ["config.yaml", "sample.h5"]
            )

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
            args = SimpleNamespace(methods=None, method_labels=None)

            for metadata in (
                {"models": {"model": {}}},
                {"models": {"model": {}}, "evaluation": []},
            ):
                params.write_text(json.dumps(metadata), encoding="utf-8")
                with self.subTest(metadata=metadata), self.assertRaisesRegex(
                    ValueError, "evaluation 必须是对象"
                ):
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

    def test_method_loader_requires_skip_baselines_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            comparison = Path(directory) / "comparison.npy"
            params = comparison.parent / "run_params.json"
            params.write_text(
                json.dumps({"models": {"model": {}}, "evaluation": {}}),
                encoding="utf-8",
            )
            args = SimpleNamespace(methods=None, method_labels=None)

            with self.assertRaisesRegex(ValueError, "skip_baselines 必须是布尔值"):
                load_method_names(args, str(comparison), n_panels=1, has_gt=False)

    def test_method_loader_uses_current_model_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            comparison = Path(directory) / "comparison.npy"
            params = comparison.parent / "run_params.json"
            args = SimpleNamespace(methods=None, method_labels=None)
            for skip_baselines, expected in (
                (False, ["GT", "DAS", "MV", "MBAN"]),
                (True, ["GT", "MBAN"]),
            ):
                params.write_text(
                    json.dumps(
                        {
                            "models": {"MBAN": "model.pth"},
                            "evaluation": {"skip_baselines": skip_baselines},
                        }
                    ),
                    encoding="utf-8",
                )
                with self.subTest(skip_baselines=skip_baselines):
                    self.assertEqual(
                        load_method_names(
                            args,
                            str(comparison),
                            n_panels=len(expected),
                            has_gt=True,
                        ),
                        expected,
                    )

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

    def test_sample_metadata_rejects_truncated_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.h5"
            sample_meta = {
                "id": "sample",
                "phantom_mode": "contrast_speckle",
                "phantom_source": "simulation",
                "phantom_path": "phantom.h5",
            }
            with h5py.File(path, "w") as handle:
                handle.create_dataset("all_multi_I", data=np.zeros((2, 1, 1, 1)))
                handle.create_dataset("all_envdb_norm", data=np.zeros((1, 1, 2, 2)))
                handle.create_dataset(
                    "config_yaml",
                    data=yaml.safe_dump({"source_samples": [sample_meta, sample_meta]}),
                )
            with self.assertRaisesRegex(ValueError, "样本数必须与 all_multi_I 一致"):
                read_sample_meta(path, sample_idx=1)

    def test_picmus_config_rejects_non_object_evaluation_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.h5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset("config_yaml", data=b"evaluation: []\n")

            with self.assertRaisesRegex(ValueError, "config_yaml.evaluation must be a mapping"):
                picmus_distortion_z_offset_mm(path, "simulation")

    def test_evaluation_and_plotting_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h5_path = write_picmus_sample(root)
            comparison_path = root / "comparison.npy"
            metrics_dir = root / "metrics"
            (root / "run_params.json").write_text(
                json.dumps({"models": {"model": "model.pth"}}), encoding="utf-8"
            )
            image = np.linspace(-50.0, -5.0, 201 * 201).reshape(201, 201)
            np.save(comparison_path, np.stack((image, image - 1.0)))
            command = [
                sys.executable,
                str(Path(__file__).resolve().parents[1] / "evaluate.py"),
                "--comparison_npy",
                str(comparison_path),
                "--h5_path",
                str(h5_path),
                "--methods",
                "model",
                "--out_dir",
                str(metrics_dir),
            ]
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
            for name in (
                "summary_metrics.csv",
                "evaluation_meta.json",
                "auxiliary_metrics.png",
                "roi_targets.png",
                "cyst_profile.png",
            ):
                self.assertTrue((metrics_dir / name).is_file(), name)
            with (metrics_dir / "summary_metrics.csv").open(
                encoding="utf-8-sig"
            ) as file:
                rows = list(csv.DictReader(file))
            model_row = next(row for row in rows if row["method"] != "GT")
            self.assertAlmostEqual(
                float(model_row["PSNR_dB_vs_GT"]), 20 * np.log10(60.0)
            )
            metadata = json.loads((metrics_dir / "evaluation_meta.json").read_text())
            self.assertEqual(metadata["sample_name"], "renamed_sample")
            self.assertEqual(metadata["evaluation_protocol"], "picmus")
            self.assertEqual(len(metadata["contrast_rois"]), 9)

    def test_flat_frame_peak_stays_inside_resolution_target_roi(self):
        x_mm = np.linspace(-20, 20, 81)
        z_mm = np.linspace(0, 60, 121)
        target = {"x_mm": 0.0, "z_mm": 30.0}
        label_map = np.zeros((len(z_mm), len(x_mm)), dtype=np.int32)
        label_map[58:63, 38:43] = 1

        result = target_resolution(
            np.zeros_like(label_map, dtype=np.float64),
            x_mm,
            z_mm,
            target,
            window_mm=2.0,
            target_idx=1,
            label_map=label_map,
        )

        peak_x = np.flatnonzero(np.isclose(x_mm, result["peak_x_mm"]))[0]
        peak_z = np.flatnonzero(np.isclose(z_mm, result["peak_z_mm"]))[0]
        self.assertEqual(label_map[peak_z, peak_x], 1)

    def test_resolution_scene_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h5_path = write_picmus_sample(root, mode="resolution_distortion")
            comparison_path = root / "comparison.npy"
            metrics_dir = root / "metrics"
            (root / "run_params.json").write_text(
                json.dumps({"models": {"model": "model.pth"}}), encoding="utf-8"
            )
            with h5py.File(h5_path, "r") as handle:
                x = handle["x_grid"][:] * 1000
                z = handle["z_grid"][:] * 1000
            phantom = read_phantom(
                picmus_reference_path("resolution_distortion", "simulation")
            )
            image = np.full((len(z), len(x)), -50.0)
            for target in phantom["resolution_targets"]:
                dx = (x[None, :] - target["x_mm"]) / 0.4
                dz = (z[:, None] - target["z_mm"]) / 0.2
                image = np.maximum(image, -3.0 - 6.0 * (dx**2 + dz**2))
            np.save(comparison_path, np.stack((image, image - 1.0)))
            args = SimpleNamespace(
                comparison_npy=str(comparison_path),
                methods="model",
                method_labels=None,
                h5_path=str(h5_path),
                h5_sample_idx=0,
                dr=60.0,
                out_dir=str(metrics_dir),
                phantom_path="auto",
                phantom_mode="auto",
                phantom_source="auto",
                fwhm_window_mm=1.8,
            )
            with patch("evaluation.evaluate.parse_args", return_value=args):
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
            for name in (
                "resolution_target_metrics.csv",
                "resolution_group_metrics.csv",
                "point_profile.png",
            ):
                self.assertTrue((metrics_dir / name).is_file(), name)
            metadata = json.loads((metrics_dir / "evaluation_meta.json").read_text())
            self.assertEqual(len(metadata["resolution_targets"]), 20)
            with (metrics_dir / "resolution_target_metrics.csv").open(
                encoding="utf-8-sig"
            ) as file:
                targets = list(csv.DictReader(file))
            self.assertEqual(len(targets), 40)
            # 15 个采样点重采样至 150 点，-6 dB 区间分别跨 13 和 31 个间隔。
            interval = 3.5 / 149
            self.assertAlmostEqual(float(targets[0]["FWHM_axial_mm"]), 13 * interval)
            self.assertAlmostEqual(float(targets[0]["FWHM_lateral_mm"]), 31 * interval)

    def test_plot_reader_rejects_non_object_run_params(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = root / "metrics"
            metrics.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "auto reference mode requires"):
                infer_repeat_references(metrics, "auto")
            (root / "run_params.json").write_text("[]", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "顶层必须是对象"):
                infer_repeat_references(metrics, "auto")
            (root / "run_params.json").write_text(
                json.dumps({"models": {}, "algorithms": []}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "之一"):
                infer_repeat_references(metrics, "auto")

    def test_plot_reader_infers_algorithm_run_params(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = root / "metrics"
            metrics.mkdir()
            (root / "run_params.json").write_text(
                json.dumps({"algorithms": ["das"]}), encoding="utf-8"
            )

            self.assertFalse(infer_repeat_references(metrics, "auto"))

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
            root = Path(directory)
            metrics = root / "metrics"
            metrics.mkdir()
            (root / "run_params.json").write_text(
                json.dumps({"models": {"model": "model.pth"}}), encoding="utf-8"
            )
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

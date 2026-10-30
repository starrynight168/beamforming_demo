import json
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from algorithms.common import (
    aperture_window_1d,
    merge_common_params,
    result_cache_signature,
)
HAS_PACK_EPFL = (Path(__file__).resolve().parents[2] / "data" / "scripts" / "pack_EPFL.py").is_file()
if HAS_PACK_EPFL:
    from data.scripts.pack_EPFL import (
        load_pack_config,
        pack_code_fingerprint,
        pack_generation_signature,
    )
else:
    load_pack_config = None
    pack_code_fingerprint = None
    pack_generation_signature = None
from evaluation.tests.test_evaluation_audit import (
    update_sample_metadata,
    write_picmus_sample,
)
from evaluation.evaluate import load_method_names
from run_one import (
    can_reuse_existing,
    run_evaluation,
    scene_ids,
    select_algorithms,
    validate_algorithm_labels,
)
from tools.run_ablation_cn import can_reuse_flat_result, run_ablation_evaluation
from tools.run_wizard_cn import build_config, inspect_h5, sample_can_be_evaluated


class ResultCacheTests(unittest.TestCase):
    def test_mv_variants_cache_tracks_shared_mv_implementation(self):
        with tempfile.TemporaryDirectory() as directory:
            h5_path = Path(directory) / "input.h5"
            h5_path.write_bytes(b"input")
            for algorithm in ("ccmv", "esbmv", "gcfmv", "cmsaw"):
                with self.subTest(algorithm=algorithm):
                    sources = result_cache_signature(h5_path, algorithm)["sources"]
                    self.assertTrue(any(Path(path).name == "mv.py" for path in sources))

    def test_common_boolean_params_reject_string_values(self):
        with self.assertRaisesRegex(ValueError, "dynamic_aperture must be a boolean"):
            merge_common_params({"params": {"dynamic_aperture": "false"}})

    def test_common_params_reject_unknown_fields(self):
        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            merge_common_params({"params": {"f_nubmer": 1.5}})

    def test_run_one_cache_requires_matching_input_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h5_path = root / "input.h5"
            h5_path.write_bytes(b"first")
            checkpoint = root / "model.bin"
            checkpoint.write_bytes(b"weights-v1")
            scene = {"id": "custom", "h5_path": str(h5_path), "sample_idx": 0}
            config = {
                "params": {"select_angles": "1"},
                "algorithm_params": {
                    "das": {
                        "aperture_mode": "discrete",
                        "checkpoint": str(checkpoint),
                    }
                },
            }
            previous = {
                "scene": scene,
                "params": merge_common_params(config),
                "algorithm_params": config["algorithm_params"],
                "extra_algorithm_args": [],
                "cache_signatures": {
                    "das": result_cache_signature(
                        h5_path, "das", config["algorithm_params"]["das"]
                    )
                },
            }
            (root / "run_params.json").write_text(json.dumps(previous), encoding="utf-8")
            args = SimpleNamespace(extra_algorithm_args=[])

            self.assertTrue(
                can_reuse_existing(root, config, scene, "das", args)[0]
            )
            checkpoint.write_bytes(b"weights-v2")
            self.assertFalse(
                can_reuse_existing(root, config, scene, "das", args)[0]
            )
            previous["cache_signatures"]["das"] = result_cache_signature(
                h5_path, "das", config["algorithm_params"]["das"]
            )
            (root / "run_params.json").write_text(json.dumps(previous), encoding="utf-8")
            h5_path.write_bytes(b"replacement with same path")
            self.assertFalse(
                can_reuse_existing(root, config, scene, "das", args)[0]
            )

    def test_empty_algorithm_override_does_not_fall_back_to_config(self):
        config = {"algorithms": ["das", "mv"]}
        with self.assertRaisesRegex(ValueError, "至少需要选择一个算法"):
            select_algorithms(config, "")
        self.assertEqual(select_algorithms(config, None), ["das", "mv"])

    def test_algorithm_labels_validate_before_execution_and_allow_commas(self):
        self.assertEqual(
            validate_algorithm_labels(
                {"algorithm_labels": {"das": "DAS, reference"}},
                ["das"],
                True,
            ),
            ["DAS, reference"],
        )
        with self.assertRaisesRegex(ValueError, "必须唯一"):
            validate_algorithm_labels(
                {"algorithm_labels": {"das": "same", "mv": "same"}},
                ["das", "mv"],
                True,
            )
        with self.assertRaisesRegex(ValueError, "必须唯一"):
            validate_algorithm_labels(
                {"algorithm_labels": {"das": "GT"}}, ["das"], True
            )

    def test_ablation_flat_cache_requires_algorithm_and_input_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h5_path = root / "input.h5"
            h5_path.write_bytes(b"input")
            scene = {"id": "case", "h5_path": str(h5_path), "sample_idx": 0}
            config = {
                "algorithms": ["das"],
                "params": {"select_angles": "1"},
                "algorithm_params": {"das": {"aperture_mode": "discrete"}},
                "scenes": [scene],
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            (root / "das.npy").write_bytes(b"output")
            params = {
                **config["params"],
                **config["algorithm_params"]["das"],
                "h5_path": str(h5_path.resolve()),
                "h5_sample_idx": 0,
                "method": "das",
                "cache_signature": result_cache_signature(
                    h5_path, "das", config["algorithm_params"]["das"]
                ),
            }
            params_path = root / "params.json"
            params_path.write_text(json.dumps(params), encoding="utf-8")

            self.assertTrue(can_reuse_flat_result(config_path, root, "das"))
            params.pop("cache_signature")
            params_path.write_text(json.dumps(params), encoding="utf-8")
            self.assertFalse(can_reuse_flat_result(config_path, root, "das"))


class WizardFlowTests(unittest.TestCase):
    def test_removed_evaluation_switch_is_not_silently_ignored(self):
        for value in (True, False, "false"):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "已移除"),
            ):
                scene_ids({"scenes": [{"id": "custom", "allow_evaluation": value}]})

    def test_generated_wizard_config_needs_no_evaluation_switch(self):
        path = Path("data/custom.h5")
        config, _ = build_config({}, path, 0, ["das"], {})
        self.assertEqual(scene_ids(config), ["custom_sample0"])
        self.assertNotIn("allow_evaluation", config["scenes"][0])

    def test_evaluation_commands_use_selected_h5_without_root_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h5_path = write_picmus_sample(root)
            scene = {"id": "custom", "h5_path": str(h5_path), "sample_idx": 0}
            config = {"scenes": [scene]}
            (root / "run_params.json").write_text(
                json.dumps(
                    {
                        "algorithms": ["das"],
                        "algorithm_labels": {"das": "DAS, reference"},
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(python_exe="python")
            with patch(
                "run_one.subprocess.run", return_value=SimpleNamespace(returncode=0)
            ) as run:
                run_evaluation(
                    args, config, scene, root, root / "comparison.npy"
                )
            command = run.call_args.args[0]
            self.assertNotIn("--config", command)
            self.assertNotIn("--methods", command)
            self.assertNotIn("--method_labels", command)
            self.assertEqual(command[command.index("--h5_path") + 1], str(h5_path))
            self.assertEqual(
                load_method_names(
                    SimpleNamespace(methods=None, method_labels=None),
                    str(root / "comparison.npy"),
                    2,
                    True,
                ),
                ["GT", "DAS, reference"],
            )
            np.save(root / "ablation_comparison.npy", np.zeros((2, 201, 201)))
            rows = [{"status": "ok", "parameter": "f_number", "value": 1.5, "index": 1}]
            with patch(
                "tools.run_ablation_cn.subprocess.run",
                return_value=SimpleNamespace(returncode=0),
            ) as run:
                run_ablation_evaluation(rows, root, h5_path, 0, True, 60)
            command = run.call_args_list[0].args[0]
            self.assertNotIn("--config", command)
            self.assertEqual(command[command.index("--h5_path") + 1], str(h5_path))

    def test_wizard_requires_valid_picmus_metadata_and_gt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_picmus_sample(root)
            self.assertTrue(sample_can_be_evaluated(path, 0))
            update_sample_metadata(
                path,
                id="simulation_contrast_speckle",
                phantom_mode="in_vivo",
                phantom_source="in_vivo",
            )
            self.assertFalse(sample_can_be_evaluated(path, 0))
            path = write_picmus_sample(root, has_gt=False)
            self.assertFalse(sample_can_be_evaluated(path, 0))
            update_sample_metadata(path, phantom_path="")
            with self.assertRaisesRegex(ValueError, "phantom_path"):
                sample_can_be_evaluated(path, 0)

    def test_inspect_h5_accepts_single_image_channel_with_multiple_input_angles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packed.h5"
            with h5py.File(path, "w") as handle:
                handle["all_multi_I"] = np.zeros((1, 3, 8, 4), dtype=np.float32)
                handle["all_multi_Q"] = np.zeros((1, 3, 8, 4), dtype=np.float32)
                handle["valid_time_samples"] = np.array([8], dtype=np.int32)
                handle["time_start_vector"] = np.zeros((1, 3), dtype=np.float32)
                handle["angles"] = np.array([-0.2, 0.0, 0.2], dtype=np.float32)
                handle["z_grid"] = np.array([0.01, 0.02], dtype=np.float32)
                handle["x_grid"] = np.linspace(-0.01, 0.01, 5, dtype=np.float32)
                for name, value in {
                    "fs": 20e6,
                    "c": 1540.0,
                    "fc": 5e6,
                    "pitch": 0.0003,
                    "num_channels": 4,
                }.items():
                    handle[name] = value
                handle["all_envdb_norm"] = np.zeros((1, 1, 2, 5), dtype=np.float32)
                handle["config_yaml"] = yaml.safe_dump(
                    {
                        "source_samples": [
                            {"acquisition_id": "sample", "phantom_mode": "in_vitro"}
                        ]
                    }
                )

            info = inspect_h5(path)

        self.assertEqual(info["a"], 3)
        self.assertEqual(info["gt"], "有")


class BeamformingEdgeTests(unittest.TestCase):
    def test_two_channel_windows_keep_nonzero_aperture(self):
        for window in ("tukey", "hann", "hamming", "blackman", "kaiser"):
            values = aperture_window_1d(2, window, torch.device("cpu"))
            self.assertTrue(torch.equal(values, torch.ones_like(values)), window)


@unittest.skipUnless(HAS_PACK_EPFL, "Requires local data/scripts/pack_EPFL.py")
class PackConfigTests(unittest.TestCase):
    def test_pack_config_rejects_string_boolean(self):
        with tempfile.TemporaryDirectory() as directory:
            config = yaml.safe_load(
                Path("data/scripts/config.yaml").read_text(encoding="utf-8")
            )
            config["epfl_pack"]["reconstruction"]["tgc"] = "false"
            path = Path(directory) / "pack.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "reconstruction.tgc 必须是布尔值"):
                load_pack_config(path)

    def test_pack_config_rejects_unknown_reconstruction_field(self):
        with tempfile.TemporaryDirectory() as directory:
            config = yaml.safe_load(
                Path("data/scripts/config.yaml").read_text(encoding="utf-8")
            )
            config["epfl_pack"]["reconstruction"]["f_nubmer"] = 1.5
            path = Path(directory) / "pack.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "不支持的字段"):
                load_pack_config(path)

    def test_pack_config_rejects_ignored_teacher_parameter(self):
        with tempfile.TemporaryDirectory() as directory:
            config = yaml.safe_load(
                Path("data/scripts/config.yaml").read_text(encoding="utf-8")
            )
            config["epfl_pack"]["algorithm_params"]["mv"]["mv_dll"] = 0.01
            path = Path(directory) / "pack.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "不支持的字段"):
                load_pack_config(path)

    def test_pack_config_rejects_ignored_das_parameter(self):
        with tempfile.TemporaryDirectory() as directory:
            config = yaml.safe_load(
                Path("data/scripts/config.yaml").read_text(encoding="utf-8")
            )
            config["epfl_pack"]["algorithm_params"]["das"]["aperture_mdoe"] = "discrete"
            path = Path(directory) / "pack.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError, "algorithm_params.das.*不支持的字段"
            ):
                load_pack_config(path)

    def test_pack_config_rejects_string_algorithm_boolean(self):
        with tempfile.TemporaryDirectory() as directory:
            config = yaml.safe_load(
                Path("data/scripts/config.yaml").read_text(encoding="utf-8")
            )
            config["epfl_pack"]["algorithm_params"]["mv"]["fbss"] = "false"
            path = Path(directory) / "pack.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "algorithm_params.mv.fbss 必须是布尔值"):
                load_pack_config(path)

    def test_pack_code_fingerprint_tracks_cmsaw_mv_dependency(self):
        outputs = {
            "main": "cmsaw",
            "references": [],
            "save_complex": [],
            "save_envdb": ["cmsaw"],
            "save_amp": [],
            "save_norm_ref": [],
            "save_safe_max": [],
        }
        paths = pack_code_fingerprint(outputs)

        self.assertTrue(
            any(
                path.replace("\\", "/").endswith("algorithms/mv.py")
                for path in paths
            )
        )
        das_paths = pack_code_fingerprint(
            {
                "main": "das",
                "references": [],
                "save_complex": [],
                "save_envdb": ["das"],
                "save_amp": [],
                "save_norm_ref": [],
                "save_safe_max": [],
            }
        )
        self.assertTrue(
            any(
                path.replace("\\", "/").endswith("algorithms/common.py")
                for path in das_paths
            )
        )

    def test_pack_interpolation_is_configurable_per_algorithm(self):
        config_path = Path("data/scripts/config.yaml")
        pipeline = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        pack_config = load_pack_config(config_path)
        self.assertEqual(pack_config["reconstruction"]["interp"], "cubic")
        self.assertEqual(pack_config["algorithm_params"]["das"]["interp"], "linear")
        self.assertNotIn("interp", pack_config["algorithm_params"]["mv"])

        with tempfile.TemporaryDirectory() as directory:
            pipeline["epfl_pack"]["algorithm_params"]["das"]["interp"] = "cubic"
            variant = Path(directory) / "config.yaml"
            variant.write_text(yaml.safe_dump(pipeline), encoding="utf-8")
            pack_config = load_pack_config(variant)
            self.assertEqual(pack_config["algorithm_params"]["das"]["interp"], "cubic")

    def test_pack_generation_signature_detects_config_and_code_changes(self):
        original = {
            "schema_version": 1,
            "dataset": {"id": "epfl"},
            "h5_schema": {"compression": "gzip"},
            "generation": {
                "ground_truth": {"interpolation": "cubic"},
                "training_targets": {"algorithm_params": {"mv": {"temporal_win": 3}}},
                "failed_files": [],
            },
            "acquisition": {"c_m_per_s": 1540.0},
            "provenance": {"implementation_sha256": {"pack.py": "old"}},
            "source_samples": [],
        }
        appended = json.loads(json.dumps(original))
        appended["source_samples"] = [{"acquisition_id": "new"}]
        appended["generation"]["failed_files"] = ["older failure"]
        self.assertEqual(pack_generation_signature(original), pack_generation_signature(appended))

        changed = json.loads(json.dumps(appended))
        changed["provenance"]["implementation_sha256"]["pack.py"] = "new"
        self.assertNotEqual(pack_generation_signature(original), pack_generation_signature(changed))

    def test_pack_config_rejects_even_temporal_window(self):
        with tempfile.TemporaryDirectory() as directory:
            config = yaml.safe_load(
                Path("data/scripts/config.yaml").read_text(encoding="utf-8")
            )
            config["epfl_pack"]["algorithm_params"]["mv"]["temporal_win"] = 4
            path = Path(directory) / "pack.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "temporal_win 必须为正奇数"):
                load_pack_config(path)


if __name__ == "__main__":
    unittest.main()

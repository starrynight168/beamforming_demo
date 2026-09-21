from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from train import evaluate_mban as suite_module
from train.evaluate_mban import DEFAULT_EVAL, build_eval_argv, load_eval, parse_route


class EvaluationConfigTests(unittest.TestCase):
    def test_software_cli_dispatches_every_mode(self) -> None:
        with (
            mock.patch.object(suite_module.diagnostics, "_run_ssim") as test_mode,
            mock.patch.object(suite_module.diagnostics, "run_ptq") as ptq_mode,
            mock.patch.object(suite_module.diagnostics, "run_mc") as mc_mode,
            mock.patch.object(suite_module.diagnostics, "run_weight_diagnostics") as weights_mode,
            mock.patch.object(suite_module.diagnostics, "run_paired_stats_evaluation") as stats_mode,
            mock.patch.object(suite_module.software, "run_figures") as figures_mode,
            mock.patch.object(suite_module.hardware_trace, "run") as trace_mode,
            mock.patch.object(suite_module.hardware, "main") as tia_mode,
            mock.patch.object(suite_module, "checkpoint_main") as checkpoint_mode,
            mock.patch.object(suite_module.software, "_run_validation") as scenes_mode,
        ):
            argv = ["--smoke"]
            for mode in ("test", "ptq", "mc", "weights", "stats", "figures", "trace", "tia", "checkpoint", "scenes"):
                suite_module.run_software_cli(mode, argv)

        test_mode.assert_called_once_with(argv=argv)
        ptq_mode.assert_called_once_with(argv)
        mc_mode.assert_called_once_with(argv)
        weights_mode.assert_called_once_with(argv)
        stats_mode.assert_called_once_with(argv)
        figures_mode.assert_called_once_with(argv)
        trace_mode.assert_called_once_with(argv)
        tia_mode.assert_called_once_with(argv)
        checkpoint_mode.assert_called_once_with(argv)
        scenes_mode.assert_called_once_with(["--mode", "scenes", *argv])

    def test_dispatch_rejects_unknown_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported software mode"):
            suite_module.run_software_cli("unknown", [])
        with self.assertRaisesRegex(ValueError, "unsupported evaluation mode"):
            suite_module._run_task("unknown", [])

    def test_hardware_cli_dispatches_every_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            with (
                mock.patch.object(suite_module, "run_hardware_stress") as hardware_mode,
                mock.patch.object(suite_module, "run_calibration") as calibrate_mode,
                mock.patch.object(suite_module, "run_deployment") as deployment_mode,
                mock.patch.object(suite_module, "run_mapping") as mapping_mode,
                mock.patch.object(suite_module, "run_crosssim") as crosssim_mode,
                mock.patch.object(suite_module, "run_ppa") as ppa_mode,
                mock.patch.object(suite_module, "load_hardware_eval_config", return_value={}),
                mock.patch.object(
                    suite_module,
                    "expand_hardware_eval_ppa_cases",
                    return_value=[{"name": "smoke"}],
                ),
            ):
                for mode in suite_module.HARDWARE_MODES:
                    suite_module.run_hardware_cli(
                        [
                            "--mode",
                            mode,
                            "--model-type",
                            "ptq",
                            "--model",
                            "model=checkpoint.pth",
                            "--output",
                            str(output_root / mode),
                        ]
                    )

        hardware_mode.assert_called_once()
        calibrate_mode.assert_called_once()
        deployment_mode.assert_called_once()
        mapping_mode.assert_called_once()
        crosssim_mode.assert_called_once()
        ppa_mode.assert_called_once()

    def test_eval_route_uses_default_or_explicit_config(self) -> None:
        self.assertEqual(parse_route([]), ("eval", [str(DEFAULT_EVAL)]))
        self.assertEqual(parse_route(["--eval", "custom.yaml"]), ("eval", ["custom.yaml"]))

    def test_eval_resolves_models_and_builds_mode_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            eval_path = root / "eval.yaml"
            eval_path.write_text(
                "paths:\n"
                "  split_file: split.csv\n"
                "enabled:\n"
                "  evaluations: [invivo]\n"
                "models:\n"
                "  model_type: fp32\n"
                "  items:\n"
                "    - checkpoint: fp32.pth\n"
                "evaluations:\n"
                "  invivo:\n"
                "    mode: test\n"
                "    test_frames: 3\n",
                encoding="utf-8",
            )
            evaluation = load_eval(eval_path)
            args = build_eval_argv(evaluation, evaluation["evaluations"][0], root / "out" / "invivo")
            self.assertIn("--models-config", args)
            self.assertIn("--split-file", args)
            self.assertIn("--test-frames", args)
            self.assertEqual(evaluation["models"][0]["checkpoint"], str((root / "fp32.pth").resolve()))
            self.assertEqual(evaluation["evaluations"][0]["models"], ["model1"])

    def test_eval_uses_ordered_display_names_and_rejects_invalid_model_type(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            eval_path = root / "eval.yaml"
            eval_path.write_text(
                "enabled:\n"
                "  evaluations: [trace]\n"
                "models:\n"
                "  model_type: fp32\n"
                "  items:\n"
                "    - checkpoint: first.pth\n"
                "    - display_name: second\n"
                "      checkpoint: second.pth\n"
                "evaluations:\n"
                "  trace:\n"
                "    mode: trace\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "trace.*fp32.*qat, ptq"):
                load_eval(eval_path)

            eval_path.write_text(
                "enabled:\n"
                "  evaluations: [test]\n"
                "models:\n"
                "  model_type: fp32\n"
                "  items:\n"
                "    - checkpoint: first.pth\n"
                "    - display_name: second\n"
                "      checkpoint: second.pth\n"
                "evaluations:\n"
                "  test:\n"
                "    mode: test\n",
                encoding="utf-8",
            )
            evaluation = load_eval(eval_path)
            self.assertEqual([model["display_name"] for model in evaluation["models"]], ["model1", "second"])

    def test_eval_manifest_records_failed_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            eval_path = root / "eval.yaml"
            eval_path.write_text(
                "paths:\n"
                "  split_file: split.csv\n"
                "enabled:\n"
                "  evaluations: [broken]\n"
                "models:\n"
                "  model_type: fp32\n"
                "  items:\n"
                "    - checkpoint: fp32.pth\n"
                "evaluations:\n"
                "  broken:\n"
                "    mode: test\n",
                encoding="utf-8",
            )
            with mock.patch.object(suite_module, "_run_task", side_effect=RuntimeError("smoke failure")):
                with self.assertRaisesRegex(RuntimeError, "smoke failure"):
                    suite_module.run_eval(eval_path)
            manifest = json.loads((root / "results" / "eval" / "eval_manifest.json").read_text())
            self.assertEqual(manifest["evaluations"][0]["status"], "failed")
            self.assertIn("smoke failure", manifest["evaluations"][0]["error"])


if __name__ == "__main__":
    unittest.main()

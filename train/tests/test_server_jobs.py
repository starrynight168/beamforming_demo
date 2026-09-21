import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from train.server_jobs import evaluate_all_si
from train.server_jobs import run_mainline_adc_study
from train.server_jobs.ablations import ablation_common


class AblationEvaluationTests(unittest.TestCase):
    def test_checkpoint_group_uses_exact_directory_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fp32 = root / "qattext" / "FP32" / "baseline" / "best_val.pth"
            qat = root / "QAT4" / "adc4" / "best_val.pth"
            fp32.parent.mkdir(parents=True)
            qat.parent.mkdir(parents=True)
            fp32.touch()
            qat.touch()

            models = evaluate_all_si.discover_models(root)

        self.assertEqual(models["FP32"], [("baseline", fp32)])
        self.assertEqual(models["QAT4"], [("adc4", qat)])
        self.assertEqual(models["QAT"], [])

    def test_evaluation_runner_fails_on_child_process_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(evaluate_all_si.subprocess, "run") as run:
                evaluate_all_si.run_evaluation(["python"], output, {})

        run.assert_called_once()
        self.assertTrue(run.call_args.kwargs["check"])

    def test_mainline_summary_reports_failed_parallel_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            good_log = open(Path(directory) / "good.log", "w", encoding="utf-8")
            bad_log = open(Path(directory) / "bad.log", "w", encoding="utf-8")
            failed = run_mainline_adc_study.summarize_parallel_runs(
                [
                    {"name": "good", "out": good_log, "proc": SimpleNamespace(returncode=0)},
                    {"name": "bad", "out": bad_log, "proc": SimpleNamespace(returncode=2)},
                ]
            )

        self.assertEqual(failed, ["bad"])

    def test_model_arguments_requires_every_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete = root / "complete.pth"
            complete.write_bytes(b"checkpoint")

            arguments = ablation_common.model_arguments(
                (("complete", complete),),
            )

        self.assertEqual(arguments, ["--model", f"complete={complete.resolve()}"])

    def test_model_arguments_reports_all_missing_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, r"missing-a[\s\S]*missing-b"):
                ablation_common.model_arguments(
                    (("a", root / "missing-a.pth"), ("b", root / "missing-b.pth")),
                )

    def test_fp32_evaluation_uses_test_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(ablation_common.subprocess, "run") as run:
                ablation_common.evaluate_models(
                    ["--model", "model=checkpoint.pth"],
                    Path(directory),
                    "2",
                )

        self.assertEqual(run.call_count, 2)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertIn("test", commands[1])
        self.assertNotIn("mc", commands[1])
        self.assertNotIn("--runs", commands[1])
        self.assertNotIn("--monte-carlo-runs", commands[0])

    def test_qat_evaluation_uses_mc_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(ablation_common.subprocess, "run") as run:
                ablation_common.evaluate_models(
                    ["--model", "model=checkpoint.pth"],
                    Path(directory),
                    "3",
                    model_type="qat",
                    test_frames=7,
                    mc_runs=5,
                )

        commands = [call.args[0] for call in run.call_args_list]
        self.assertIn("--monte-carlo-runs", commands[0])
        self.assertIn("5", commands[0])
        self.assertIn("mc", commands[1])
        self.assertIn("--runs", commands[1])
        self.assertIn("5", commands[1])
        self.assertIn("--test-frames", commands[1])
        self.assertIn("7", commands[1])


if __name__ == "__main__":
    unittest.main()

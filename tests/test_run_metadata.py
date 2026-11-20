import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from train.shared_core.run_metadata import (
    file_record,
    finalize_training_metadata,
    relative_path,
    write_evaluation_manifest,
)


class RunMetadataTests(unittest.TestCase):
    def test_completion_log_hash_matches_final_log(self):
        from train.mban_core import training

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            output = Path(directory)
            (output / "effective_config.json").write_text("{}", encoding="utf-8")
            args = SimpleNamespace(output_directory=str(output))
            with (
                patch.object(training._runtime, "args", args),
                patch.object(training._runtime, "LOG_PATH", str(output / "training_log.txt")),
                patch.object(training, "_train"),
            ):
                training.train()
            config = json.loads((output / "effective_config.json").read_text(encoding="utf-8"))
            self.assertEqual(
                config["run_manifest"]["log"]["sha256"],
                file_record(output / "training_log.txt")["sha256"],
            )

    def test_finalize_adds_manifest_and_code_revision(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            output = Path(directory)
            (output / "effective_config.json").write_text(
                json.dumps(
                    {
                        "signal_domain": "analytic_rf",
                        "model_semantics_version": 50,
                        "training_data_fingerprint": "abc",
                    }
                ),
                encoding="utf-8",
            )
            (output / "training_log.txt").write_text("log\n", encoding="utf-8")
            (output / "best_val.pth").write_bytes(b"checkpoint")

            manifest = finalize_training_metadata(output)

            self.assertIsNotNone(manifest)
            config = json.loads(
                (output / "effective_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(config["run_manifest"]["signal_domain"], "analytic_rf")
            self.assertEqual(
                config["run_manifest"]["checkpoints"]["best_val.pth"]["sha256"],
                file_record(output / "best_val.pth")["sha256"],
            )
            self.assertTrue((output / "code_revision.json").is_file())
            self.assertEqual(
                config["run_manifest"]["config"]["path"],
                relative_path(output / "effective_config.json"),
            )

    def test_file_record_rejects_missing_file(self):
        self.assertIsNone(file_record(Path.cwd() / "definitely-missing.pth"))

    def test_evaluation_manifest_records_relative_checkpoint_and_semantics(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            output = Path(directory)
            checkpoint = output / "best_val.pth"
            checkpoint.write_bytes(b"checkpoint")
            spec = SimpleNamespace(
                display_name="H32_OC160",
                path=checkpoint,
                model_type="fp32",
                semantics_version=50,
                runtime={"signal_domain": "iq"},
            )

            write_evaluation_manifest(output, [spec], mode="scenes")

            manifest = json.loads(
                (output / "eval_manifest.json").read_text(encoding="utf-8")
            )
            model = manifest["models"][0]
            self.assertEqual(manifest["status"], "running")
            self.assertEqual(model["signal_domain"], "iq")
            self.assertEqual(model["model_semantics_version"], 50)
            self.assertEqual(model["checkpoint"], relative_path(checkpoint))
            self.assertEqual(model["checkpoint_sha256"], file_record(checkpoint)["sha256"])


if __name__ == "__main__":
    unittest.main()

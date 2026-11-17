import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.code import ablate_sound_speed as tool
from evaluation.evaluate import resolve_signal_domain


class SoundSpeedEvaluationTest(unittest.TestCase):
    def test_signal_domain_resolution(self):
        self.assertEqual(resolve_signal_domain(None, ["rf"]), "rf")
        self.assertEqual(resolve_signal_domain("iq", ["rf"]), "iq")
        with self.assertRaises(ValueError):
            resolve_signal_domain(None, ["rf", "iq"])
        with self.assertRaises(ValueError):
            resolve_signal_domain(None, [])

    def test_metrics_passes_configured_signal_domain_to_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scene = {"id": "test", "h5_path": root / "input.h5", "sample_idx": 0,
                     "role": "picmus", "nominal_c": 1540.0}
            args = argparse.Namespace(config="unused", c_values="1520,1540", output_dir=root,
                                      run_dir=None, metrics=True)
            config = {"params": {"signal_domain": "rf", "dr": 55}}

            def overview(rows, scene_root, *_):
                data = scene_root / "data"
                data.mkdir(parents=True, exist_ok=True)
                np.save(data / "ablation_comparison.npy", np.zeros((2, 2, 2)))

            with (patch.object(tool, "parse_args", return_value=args),
                  patch.object(tool, "load_config", return_value=config),
                  patch.object(tool, "select_scenes", return_value=[scene]),
                  patch.object(tool.workflow, "run_one", return_value=(0, None)),
                  patch.object(tool.workflow, "preserve_ground_truth"),
                  patch.object(tool.workflow, "flatten_ablation_result"),
                  patch.object(tool.workflow, "save_ablation_overview", side_effect=overview),
                  patch.object(tool.workflow, "collect_ablation_individual_images"),
                  patch.object(tool.workflow, "method_output_path", return_value=Path(__file__)),
                  patch.object(tool.workflow, "load_picmus_context", return_value={}),
                  patch.object(tool.workflow, "run_process", return_value=0) as run_process,
                  patch.object(tool.workflow, "append_run_log"),
                  patch.object(tool, "collect_metrics", return_value=None)):
                tool.main()

            self.assertEqual(run_process.call_count, 2)
            command = run_process.call_args_list[0].args[0]
            self.assertEqual(command[command.index("--signal_domain") + 1], "rf")
            self.assertEqual(command[command.index("--dr") + 1], "55.0")


if __name__ == "__main__":
    unittest.main()

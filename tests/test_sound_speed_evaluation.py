import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.code import ablate_sound_speed as tool
from tools.code import compare_interpolation as interp_tool
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


class InterpolationMatrixEvaluationTest(unittest.TestCase):
    def test_parse_domains_and_interps_validation(self):
        self.assertEqual(
            interp_tool.parse_domains("rf,analytic_rf,iq"),
            ["rf", "analytic_rf", "iq"],
        )
        self.assertEqual(
            interp_tool.parse_interps("linear,sinc"),
            ["linear", "sinc"],
        )
        with self.assertRaises(ValueError):
            interp_tool.parse_domains("")
        with self.assertRaises(ValueError):
            interp_tool.parse_domains("unknown")
        with self.assertRaises(ValueError):
            interp_tool.parse_domains("iq,iq")
        with self.assertRaises(ValueError):
            interp_tool.parse_interps("linear")
        with self.assertRaises(ValueError):
            interp_tool.parse_interps("linear,unknown")
        with self.assertRaises(ValueError):
            interp_tool.parse_interps("linear,linear")

    def test_matrix_dispatches_signal_domains_to_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scene = {
                "id": "simulation_contrast_speckle",
                "h5_path": root / "input.h5",
                "sample_idx": 0,
                "role": "picmus",
            }
            args = argparse.Namespace(
                config="unused",
                algorithm="mv",
                domains="rf,iq",
                interps="linear,sinc",
                output_dir=root,
                run_dir=None,
                metrics=True,
            )
            config = {"params": {"signal_domain": "iq", "dr": 60}}

            def overview(rows, scene_root, *_):
                data = scene_root / "data"
                data.mkdir(parents=True, exist_ok=True)
                np.save(data / "ablation_comparison.npy", np.zeros((2, 2, 2)))

            with (
                patch.object(interp_tool, "parse_args", return_value=args),
                patch.object(interp_tool, "load_config", return_value=config),
                patch.object(interp_tool, "select_scenes", return_value=[scene]),
                patch.object(
                    interp_tool.workflow, "run_one", return_value=(0, None)
                ) as run_one,
                patch.object(interp_tool.workflow, "preserve_ground_truth"),
                patch.object(interp_tool.workflow, "flatten_ablation_result"),
                patch.object(
                    interp_tool.workflow, "save_ablation_overview", side_effect=overview
                ),
                patch.object(
                    interp_tool.workflow, "collect_ablation_individual_images"
                ),
                patch.object(
                    interp_tool.workflow, "method_output_path", return_value=Path(__file__)
                ),
                patch.object(
                    interp_tool.workflow, "load_picmus_context", return_value={}
                ),
                patch.object(
                    interp_tool.workflow, "run_process", return_value=0
                ) as run_process,
                patch.object(interp_tool.workflow, "append_run_log"),
                patch.object(interp_tool, "collect_metrics", return_value=None),
            ):
                interp_tool.main()

            self.assertEqual(run_one.call_count, 4)
            # 2 domains x 2 commands (evaluate + plot_metrics)
            self.assertEqual(run_process.call_count, 4)
            eval_domains = [
                call.args[0][call.args[0].index("--signal_domain") + 1]
                for call in run_process.call_args_list
                if "evaluation.evaluate" in call.args[0]
            ]
            self.assertEqual(eval_domains, ["rf", "iq"])

    def test_save_interp_sweep_curve_renders_dot_plot(self):
        with tempfile.TemporaryDirectory() as temp:
            out_dir = Path(temp)
            scenes = [
                {"id": "simulation_contrast_speckle", "role": "picmus"},
                {"id": "simulation_resolution_distortion", "role": "picmus"},
            ]
            domains = ["rf", "iq"]
            interps = ["linear", "cubic", "sinc"]
            records = []
            for domain in domains:
                for idx, interp in enumerate(interps):
                    records.append(
                        {
                            "signal_domain": domain,
                            "scenario": "simulation_contrast_speckle",
                            "method": f"interp={interp}",
                            "CR_dB": str(15.0 + idx * 2.0),
                            "ROI2_CR_dB": str(18.0 + idx * 2.0),
                            "runtime_sec": str(1.0 + idx),
                        }
                    )
                    records.append(
                        {
                            "signal_domain": domain,
                            "scenario": "simulation_resolution_distortion",
                            "method": f"interp={interp}",
                            "FWHM_lateral_mm": str(0.5 - idx * 0.05),
                            "runtime_sec": str(1.0 + idx),
                        }
                    )
            path = interp_tool.save_interp_sweep_curve(
                records, scenes, domains, interps, out_dir
            )
            self.assertIsNotNone(path)
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)

    def test_select_scenes_keeps_only_four_standard_picmus_scenes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sim_h5 = root / "sim.h5"
            exp_h5 = root / "exp.h5"
            invivo_h5 = root / "invivo.h5"
            config = {
                "scenes": [
                    {"id": "simulation_contrast_speckle", "h5_path": str(sim_h5), "sample_idx": 0},
                    {"id": "simulation_resolution_distortion", "h5_path": str(sim_h5), "sample_idx": 1},
                    {"id": "experiments_contrast_speckle", "h5_path": str(exp_h5), "sample_idx": 0},
                    {"id": "experiments_resolution_distortion", "h5_path": str(exp_h5), "sample_idx": 1},
                    {"id": "invivo_15002", "h5_path": str(invivo_h5), "sample_idx": 0},
                ]
            }
            inspected = []

            def fake_inspect(path):
                inspected.append(Path(path))
                if Path(path) == invivo_h5:
                    raise AssertionError("in_vivo H5 should not be inspected")
                return {"n": 2, "domains": ["iq", "rf", "analytic_rf"]}

            with (
                patch.object(interp_tool.workflow, "inspect_h5", side_effect=fake_inspect),
                patch.object(interp_tool.workflow, "load_picmus_context", return_value={}),
            ):
                selected = interp_tool.select_scenes(
                    config, domains=["rf", "analytic_rf", "iq"]
                )

            self.assertEqual(
                [scene["id"] for scene in selected],
                list(interp_tool.STANDARD_SCENE_IDS),
            )
            self.assertNotIn(invivo_h5, inspected)

            with (
                patch.object(
                    interp_tool.workflow,
                    "inspect_h5",
                    return_value={"n": 2, "domains": ["iq"]},
                ),
                patch.object(interp_tool.workflow, "load_picmus_context", return_value={}),
            ):
                with self.assertRaises(ValueError):
                    interp_tool.select_scenes(config, domains=["rf", "iq"])


if __name__ == "__main__":
    unittest.main()

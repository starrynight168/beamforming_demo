import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import yaml

from data.check_data import decode_compact_sequence, display_width, run_checks, shorten_text
from data.pack_data import (
    compact_sequence_config,
    das_reference_from_iq,
    pack_dataset,
    process_scene,
    validate_grid,
)


class DataUtilityTests(unittest.TestCase):
    def test_compact_sequence_round_trip(self) -> None:
        values = np.arange(12, dtype=np.int32)
        encoded = compact_sequence_config(values)

        np.testing.assert_array_equal(decode_compact_sequence(encoded), values)
        np.testing.assert_array_equal(decode_compact_sequence([1, 3]), [1, 3])

    def test_short_text_respects_display_width(self) -> None:
        result = shorten_text("数据" * 80, limit=24)

        self.assertLessEqual(display_width(result), 24)
        self.assertTrue(result.endswith(" ..."))

    def test_grid_requires_finite_increasing_coordinates(self) -> None:
        np.testing.assert_array_equal(validate_grid([0, 1], "x"), [0.0, 1.0])
        for values in ([0, 0], [0, np.nan], [[0, 1]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_grid(values, "x")

    def test_das_interpolation_branches(self) -> None:
        rng = np.random.default_rng(7)
        i_data = rng.normal(size=(2, 128, 4)).astype(np.float32)
        q_data = rng.normal(size=(2, 128, 4)).astype(np.float32)
        grids = ([-0.001, 0.0, 0.001], [0.001, 0.002, 0.003])

        for interpolation in ("nearest", "linear", "cubic"):
            with self.subTest(interpolation=interpolation):
                image, peak = das_reference_from_iq(
                    i_data,
                    q_data,
                    20e6,
                    1540.0,
                    5e6,
                    0.0003,
                    [0.0, 0.0],
                    [-0.1, 0.1],
                    *grids,
                    interp=interpolation,
                    row_block=0,
                )
                self.assertEqual(image.shape, (1, 1, 3, 3))
                self.assertTrue(np.isfinite(image).all())
                self.assertGreater(peak, 0.0)

        with self.assertRaisesRegex(ValueError, "不支持的 DAS 插值方式"):
            das_reference_from_iq(
                i_data,
                q_data,
                20e6,
                1540.0,
                5e6,
                0.0003,
                [0.0, 0.0],
                [-0.1, 0.1],
                *grids,
                interp="quadratic",
            )

    def test_das_interpolation_keeps_last_input_sample(self) -> None:
        i_data = np.zeros((1, 20, 1), dtype=np.float32)
        q_data = np.zeros_like(i_data)
        i_data[0, -1, 0] = 1.0
        z_grid = np.array([17.5, 18.75]) * 1540.0 / (2.0 * 20e6)

        for interpolation in ("nearest", "linear", "cubic"):
            with self.subTest(interpolation=interpolation):
                image, _ = das_reference_from_iq(
                    i_data,
                    q_data,
                    20e6,
                    1540.0,
                    5e6,
                    0.0003,
                    [0.0],
                    [0.0],
                    [-1e-6, 1e-6],
                    z_grid,
                    interp=interpolation,
                )
                self.assertGreater(float(np.max(image)), 0.0)


class DataPipelineTests(unittest.TestCase):
    def _write_sources(self, root: Path) -> list[dict]:
        rng = np.random.default_rng(11)
        iq_names = ("iq_direct.h5", "iq_geometry.h5")
        for index, iq_name in enumerate(iq_names):
            with h5py.File(root / iq_name, "w") as handle:
                acquisition = handle.create_group("US/US_DATASET0000")
                data = acquisition.create_group("data")
                data.create_dataset(
                    "real",
                    data=rng.normal(size=(2, 4, 128)).astype(np.float32),
                )
                data.create_dataset(
                    "imag",
                    data=rng.normal(size=(2, 4, 128)).astype(np.float32),
                )
                acquisition.create_dataset("sampling_frequency", data=np.float32(20e6))
                acquisition.create_dataset("sound_speed", data=np.float32(1540.0))
                acquisition.create_dataset("initial_time", data=np.zeros(2, np.float32))
                acquisition.create_dataset("angles", data=np.array([-0.1, 0.1], np.float32))
                if index == 0:
                    acquisition.create_dataset("fc", data=np.float32(5e6))
                    acquisition.create_dataset("pitch", data=np.float32(0.0003))
                else:
                    acquisition.create_dataset("modulation_frequency", data=np.float32(5e6))
                    acquisition.create_dataset(
                        "probe_geometry",
                        data=np.array([[-0.00045, -0.00015, 0.00015, 0.00045]]),
                    )

        with h5py.File(root / "scan.h5", "w") as handle:
            scan = handle.create_group("US/US_DATASET0000")
            scan.create_dataset("x_axis", data=np.array([-0.001, 0.0, 0.001]))
            scan.create_dataset("z_axis", data=np.array([0.001, 0.002, 0.003]))

        with h5py.File(root / "gt.h5", "w") as handle:
            data = handle.create_group("US/US_DATASET0000/data")
            data.create_dataset("real", data=rng.normal(size=(1, 3, 3)))
            data.create_dataset("imag", data=rng.normal(size=(1, 3, 3)))

        common = {"scan": "scan.h5", "phantom": ""}
        return [
            {
                **common,
                "iq": iq_names[0],
                "name": "simulation_contrast_speckle",
                "mode": "contrast_speckle",
                "source": "simulation",
                "gt": "gt.h5",
            },
            {
                **common,
                "iq": iq_names[1],
                "name": "carotid_cross",
                "mode": "in_vivo",
                "source": "in_vivo",
                "gt": "generated:multi_angle_das",
            },
        ]

    def test_pack_and_check_both_ground_truth_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenes = self._write_sources(root)
            output = root / "packed.h5"

            pack_dataset(scenes, output, root, row_block=2)
            self.assertTrue((root / "gt_preview" / "simulation_contrast_speckle.png").is_file())
            self.assertTrue((root / "gt_preview" / "carotid_cross.png").is_file())

            with h5py.File(output, "r") as handle:
                status, problems = run_checks(handle)
                self.assertEqual(status, "通过", problems)
                self.assertEqual(handle["all_multi_I"].shape, (2, 2, 128, 4))
                config = yaml.safe_load(handle["config_yaml"][()])
                self.assertIn(
                    "generated_multi_angle_das",
                    config["generation"]["ground_truth"],
                )

            config["dataset"] = []
            config["h5_schema"]["units"]["c"] = "cm/s"
            with h5py.File(output, "r+") as handle:
                handle["config_yaml"][()] = yaml.safe_dump(config, sort_keys=False)
                status, problems = run_checks(handle)
                self.assertEqual(status, "失败")
                self.assertIn("config_yaml.dataset 必须是字典", problems)
                self.assertIn("config_yaml.h5_schema.units.c 应为 m/s", problems)

                del handle["valid_time_samples"]
                handle.create_dataset("valid_time_samples", data=np.array([64.5, 64.5]))
                status, problems = run_checks(handle)
                self.assertEqual(status, "失败")
                self.assertIn(
                    "valid_time_samples 必须是长度 2 的一维整数数组",
                    problems,
                )

                handle["all_envdb_norm"][0, 0, 0, 0] = 1.5
                status, problems = run_checks(handle)
                self.assertTrue(
                    any("all_envdb_norm 必须位于 [0,1]" in problem for problem in problems),
                    problems,
                )

                del handle["config_yaml"]
                handle.create_dataset("config_yaml", data=np.float32(1.0))
                status, problems = run_checks(handle)
                self.assertEqual(status, "失败")
                self.assertTrue(
                    any("config_yaml 不是合法 YAML" in problem for problem in problems),
                    problems,
                )

            with h5py.File(root / "iq_geometry.h5", "r+") as handle:
                handle["US/US_DATASET0000/probe_geometry"][0] = [
                    -0.00045,
                    -0.00015,
                    -0.00045,
                    -0.00015,
                ]
            with self.assertRaisesRegex(ValueError, "不是均匀线阵"):
                process_scene(scenes[1], root)


if __name__ == "__main__":
    unittest.main()

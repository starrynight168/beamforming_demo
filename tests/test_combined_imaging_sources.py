import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DRIVE_PATH = re.compile(r"(?:^|[=])([A-Za-z]:[\\/])")
POSIX_PATH = re.compile(r"^/(?:home|mnt|Users)/")


def load_combined_imaging():
    path = ROOT / "train" / "figure" / "code" / "combined_imaging.py"
    spec = importlib.util.spec_from_file_location("combined_imaging_sources", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CombinedImagingSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_combined_imaging()

    def test_tracked_sources_build_all_three_rows(self):
        rows, extents = self.module.build_panels()

        self.assertEqual([len(row) for row in rows], [4, 4, 4])
        self.assertEqual([image.shape for image in rows[0]], [(609, 387)] * 4)
        self.assertEqual([image.shape for image in rows[1]], [(609, 387)] * 4)
        self.assertEqual([image.shape for image in rows[2]], [(576, 320)] * 4)
        self.assertEqual(len(extents), 3)

    def test_figure_inputs_live_under_tracked_figure_data(self):
        data_root = ROOT / "train" / "figure" / "data"
        for path in (
            self.module.FP32_SCENES,
            self.module.QAT_IDEAL_SCENES,
            self.module.INVIVO,
        ):
            self.assertTrue(path.is_relative_to(data_root), path)

    def test_plotter_does_not_open_hdf5_or_evaluation_results(self):
        source = (ROOT / "train" / "figure" / "code" / "combined_imaging.py").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("h5py", source)
        self.assertNotIn("train/results", source)
        self.assertNotIn("results", source)

    def test_tracked_metadata_has_no_machine_absolute_paths(self):
        offenders = []

        def walk(value, path):
            if isinstance(value, str):
                if DRIVE_PATH.search(value) or POSIX_PATH.match(value):
                    offenders.append(f"{path}: {value}")
            elif isinstance(value, list):
                for item in value:
                    walk(item, path)
            elif isinstance(value, dict):
                for item in value.values():
                    walk(item, path)

        for path in sorted((ROOT / "train" / "figure" / "data").rglob("*.json")):
            walk(json.loads(path.read_text(encoding="utf-8")), path.name)

        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()

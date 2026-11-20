import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import h5py
import json

from evaluation import replot_existing


class ReplotExistingTests(unittest.TestCase):
    def test_replot_comparison_rejects_mismatched_grid_before_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h5_path = root / "source.h5"
            with h5py.File(h5_path, "w") as handle:
                handle["x_grid"] = [0.0, 1.0, 2.0]
                handle["z_grid"] = [0.0, 1.0]
            (root / "evaluation_meta.json").write_text(
                json.dumps({"h5_path": str(h5_path), "methods": ["DAS"]}),
                encoding="utf-8",
            )
            np.save(root / "comparison.npy", np.zeros((1, 2, 2)))
            output = root / "comparison.png"
            output.write_bytes(b"previous image")

            with self.assertRaisesRegex(ValueError, "坐标网格"):
                replot_existing.replot_comparison(root)
            self.assertEqual(output.read_bytes(), b"previous image")

    def test_failed_training_replot_keeps_existing_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(
                {
                    "method": ["roi05", "roi05"],
                    "epoch": [1, 2],
                    "train_loss": [0.5, 0.4],
                    "validation_loss": [0.6, 0.5],
                }
            ).to_csv(root / "training_curves.csv", index=False)
            old_page = root / "training_curves_page2.png"
            old_page.write_bytes(b"old-page")

            with patch.object(
                replot_existing, "save_figure", side_effect=RuntimeError("render failed")
            ):
                with self.assertRaisesRegex(RuntimeError, "render failed"):
                    replot_existing.replot_training_curves(root)

            self.assertEqual(old_page.read_bytes(), b"old-page")
            self.assertEqual(list(root.glob(".*.png")), [])

    def test_replace_pages_restores_every_target_after_install_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            targets = [root / "page1.png", root / "page2.png"]
            targets[0].write_bytes(b"old-page-1")
            targets[1].write_bytes(b"old-page-2")
            rendered = [root / ".page1.tmp.png", root / ".page2.tmp.png"]
            rendered[0].write_bytes(b"new-page-1")
            rendered[1].write_bytes(b"new-page-2")
            real_replace = Path.replace

            def flaky_replace(source, target):
                if Path(target) == targets[1] and Path(source) == rendered[1]:
                    raise OSError("disk full")
                return real_replace(source, target)

            with patch.object(Path, "replace", flaky_replace):
                with self.assertRaisesRegex(OSError, "disk full"):
                    replot_existing.replace_pages(
                        [(rendered[0], targets[0]), (rendered[1], targets[1])]
                    )

            self.assertEqual(targets[0].read_bytes(), b"old-page-1")
            self.assertEqual(targets[1].read_bytes(), b"old-page-2")
            self.assertEqual(list(root.glob(".*.bak")), [])

    def test_failed_page_install_keeps_the_previous_training_page_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            methods = [f"method{index}" for index in range(7)]
            pd.DataFrame(
                {
                    "method": methods * 2,
                    "epoch": [1] * len(methods) + [2] * len(methods),
                    "train_loss": [0.5] * (2 * len(methods)),
                    "validation_loss": [0.6] * (2 * len(methods)),
                }
            ).to_csv(root / "training_curves.csv", index=False)
            page1 = root / "training_curves.png"
            page2 = root / "training_curves_page2.png"
            page1.write_bytes(b"old-page-1")
            page2.write_bytes(b"old-page-2")
            real_replace = Path.replace

            def flaky_replace(source, target):
                source_path, target_path = Path(source), Path(target)
                if (
                    target_path.name == "training_curves_page2.png"
                    and source_path.suffix == ".png"
                    and source_path.name.startswith(".")
                ):
                    raise OSError("disk full")
                return real_replace(source_path, target_path)

            with patch.object(Path, "replace", flaky_replace):
                with self.assertRaisesRegex(OSError, "disk full"):
                    replot_existing.replot_training_curves(root)

            self.assertEqual(page1.read_bytes(), b"old-page-1")
            self.assertEqual(page2.read_bytes(), b"old-page-2")
            self.assertEqual(list(root.glob(".*.png")), [])
            self.assertEqual(list(root.glob(".*.bak")), [])

    def test_portable_path_resolves_relative_and_absolute_paths(self):
        expected = replot_existing.PROJECT_ROOT / "train" / "results" / "x.json"
        self.assertEqual(replot_existing.portable_path(expected), expected)
        self.assertEqual(
            replot_existing.portable_path("train/results/x.json"), expected
        )


if __name__ == "__main__":
    unittest.main()

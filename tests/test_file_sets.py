import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shared_core.file_sets import replace_file_set


class ReplaceFileSetTests(unittest.TestCase):
    def test_replaces_generated_files_and_removes_stale_managed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged = root / "staged"
            target = root / "target"
            staged.mkdir()
            target.mkdir()
            (staged / "current.txt").write_text("new", encoding="utf-8")
            (target / "current.txt").write_text("old", encoding="utf-8")
            (target / "stale.txt").write_text("stale", encoding="utf-8")

            replace_file_set(
                staged,
                target,
                [Path("current.txt")],
                [Path("current.txt"), Path("stale.txt")],
            )

            self.assertEqual((target / "current.txt").read_text(encoding="utf-8"), "new")
            self.assertFalse((target / "stale.txt").exists())

    def test_empty_managed_set_installs_generated_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged = root / "staged"
            target = root / "target"
            staged.mkdir()
            (staged / "new.txt").write_text("new", encoding="utf-8")

            replace_file_set(staged, target, [Path("new.txt")], [])

            self.assertEqual((target / "new.txt").read_text(encoding="utf-8"), "new")

    def test_install_failure_restores_all_managed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged = root / "staged"
            target = root / "target"
            staged.mkdir()
            target.mkdir()
            for name in ("first.txt", "second.txt"):
                (staged / name).write_text(f"new {name}", encoding="utf-8")
                (target / name).write_text(f"old {name}", encoding="utf-8")
            stale = target / "stale.txt"
            stale.write_text("old stale", encoding="utf-8")
            original_replace = Path.replace

            def fail_second_install(source, destination):
                if source == staged / "second.txt":
                    raise OSError("disk full")
                return original_replace(source, destination)

            with patch.object(Path, "replace", fail_second_install):
                with self.assertRaisesRegex(OSError, "disk full"):
                    replace_file_set(
                        staged,
                        target,
                        [Path("first.txt"), Path("second.txt")],
                        [Path("first.txt"), Path("second.txt"), Path("stale.txt")],
                    )

            self.assertEqual((target / "first.txt").read_text(encoding="utf-8"), "old first.txt")
            self.assertEqual((target / "second.txt").read_text(encoding="utf-8"), "old second.txt")
            self.assertEqual(stale.read_text(encoding="utf-8"), "old stale")


if __name__ == "__main__":
    unittest.main()

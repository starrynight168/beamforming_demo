from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path


def replace_file_set(
    staged_root: Path,
    target_root: Path,
    generated: Iterable[Path],
    managed: Iterable[Path],
) -> None:
    """Replace a related set of files and restore the old set on failure."""

    generated_paths = tuple(Path(path) for path in generated)
    managed_paths = tuple(dict.fromkeys(Path(path) for path in managed))
    if any(
        path.is_absolute() or ".." in path.parts
        for path in (*generated_paths, *managed_paths)
    ):
        raise ValueError("file-set paths must stay inside their staging and target roots")
    target_root.mkdir(parents=True, exist_ok=True)
    backup_root = Path(
        tempfile.mkdtemp(prefix=".file-set-backup-", dir=target_root)
    )
    backups: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    preserve_backup = False
    try:
        for relative in managed_paths:
            target = target_root / relative
            if target.is_file():
                backup = backup_root / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                target.replace(backup)
                backups.append((backup, target))
        for relative in generated_paths:
            source = staged_root / relative
            target = target_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
            installed.append(target)
    except BaseException as error:
        rollback_errors: list[OSError] = []
        for target in reversed(installed):
            try:
                target.unlink(missing_ok=True)
            except OSError as rollback_error:
                rollback_errors.append(rollback_error)
        for backup, target in reversed(backups):
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                backup.replace(target)
            except OSError as rollback_error:
                rollback_errors.append(rollback_error)
        if rollback_errors:
            preserve_backup = True
            raise RuntimeError(
                f"file-set replacement failed and rollback data remains in {backup_root}"
            ) from error
        raise
    finally:
        if not preserve_backup:
            shutil.rmtree(backup_root, ignore_errors=True)

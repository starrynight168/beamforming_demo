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
    replace_files(
        (
            (staged_root / relative, target_root / relative)
            for relative in generated_paths
        ),
        (target_root / relative for relative in managed_paths),
        target_root,
    )


def replace_files(
    pending: Iterable[tuple[Path, Path]],
    managed: Iterable[Path],
    backup_root: Path,
) -> None:
    pending_paths = tuple((Path(source), Path(target)) for source, target in pending)
    managed_targets = [Path(target) for target in managed]
    managed_targets.extend(target for _, target in pending_paths)
    managed_paths = tuple(dict.fromkeys(managed_targets))
    backup_root = Path(backup_root).absolute()
    for target in managed_paths:
        try:
            target.absolute().relative_to(backup_root)
        except ValueError as error:
            raise ValueError(
                f"managed target is outside backup root: {target}"
            ) from error
    backup_root.mkdir(parents=True, exist_ok=True)
    backup_directory = Path(
        tempfile.mkdtemp(prefix=".file-set-backup-", dir=backup_root)
    )
    backups: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    preserve_backup = False
    try:
        for target in managed_paths:
            if target.is_file():
                backup = backup_directory / target.absolute().relative_to(backup_root)
                backup.parent.mkdir(parents=True, exist_ok=True)
                target.replace(backup)
                backups.append((backup, target))
        for source, target in pending_paths:
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
                "file-set replacement failed and rollback data remains in "
                f"{backup_directory}"
            ) from error
        raise
    finally:
        if not preserve_backup:
            shutil.rmtree(backup_directory, ignore_errors=True)

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def portable_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative_path(path: str | Path, root: Path = PROJECT_ROOT) -> str:
    candidate = Path(path).expanduser()
    try:
        return candidate.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return candidate.as_posix()

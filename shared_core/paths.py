from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def portable_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path

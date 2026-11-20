import csv
from pathlib import Path


def read_dict_csv(
    path,
    required_fields=(),
    *,
    encoding="utf-8-sig",
    missing_ok=False,
    field_error_label="CSV",
):
    path = Path(path)
    if missing_ok and not path.exists():
        return []
    with path.open(encoding=encoding, newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(required_fields) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{field_error_label} 缺少列 {sorted(missing)}: {path}")
        return list(reader)

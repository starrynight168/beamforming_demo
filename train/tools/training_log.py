from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

EPOCH = re.compile(r"\[EPOCH\] Ep (\d+)/\d+ \| Tr=([\deE+\-.]+) \| Val=([\deE+\-.]+)")
DATA = re.compile(r"监督损失 \| Train Data=([\deE+\-.]+).*?\| Val Data=([\deE+\-.]+)")
RANGE = re.compile(r"附加损失 \| Train Lrange=([\deE+\-.]+)")
STAGE = re.compile(r"QAT \| stage=(\S+) \| stage_profile=(\S+)")
HEALTH = re.compile(r"数值健康 \| 非有限梯度步=(\d+)/(\d+).*?跳过更新步=(\d+)")
SCALE = re.compile(r"偏置/映射 \| (fc\d+): .*?wscale=([\deE+\-.]+)")


def parse_log(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    stage = profile = ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = STAGE.search(line)
        if match:
            stage, profile = match.groups()
        match = EPOCH.search(line)
        if match:
            if current is not None:
                rows.append(current)
            current = {
                "epoch": int(match.group(1)),
                "stage": stage,
                "profile": profile,
                "train_total": float(match.group(2)),
                "val_total": float(match.group(3)),
            }
            continue
        if current is None:
            continue
        match = DATA.search(line)
        if match:
            current["train_data"], current["val_data"] = map(float, match.groups())
        match = RANGE.search(line)
        if match:
            current["train_lrange"] = float(match.group(1))
        match = HEALTH.search(line)
        if match:
            current["nonfinite_grad_steps"] = int(match.group(1))
            current["gradient_steps"] = int(match.group(2))
            current["skipped_steps"] = int(match.group(3))
        match = SCALE.search(line)
        if match:
            current[f"{match.group(1)}_weight_scale"] = float(match.group(2))
    if current is not None:
        rows.append(current)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="提取 QAT 日志的逐轮损失、数值健康和映射尺度")
    parser.add_argument("--log", action="append", required=True, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    for path in args.log:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = parse_log(path)
        output = args.output / f"{path.stem}_series.csv"
        fields = sorted({key for row in rows for key in row})
        with output.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        bad = [row["epoch"] for row in rows if row.get("nonfinite_grad_steps", 0)]
        print(f"[{path.name}] epochs={len(rows)} | nonfinite_epochs={bad} | output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

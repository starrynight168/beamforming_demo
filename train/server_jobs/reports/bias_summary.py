"""Print the existing bias-ablation result tables without modifying results."""

import argparse
import csv
import math
from pathlib import Path

STAGES = ("FP32", "QAT", "QAT4", "PTQ")
CONFIGS = ("nobias", "fc1", "fc2", "fc3", "fc12", "fc23", "fc123")
SCENES = (
    "experiments_contrast_speckle",
    "experiments_resolution_distorsion",
    "simulation_contrast_speckle",
    "simulation_resolution_distorsion",
)
BIAS_LAYER_COUNT = {"nobias": 0, "fc1": 1, "fc2": 1, "fc3": 1, "fc12": 2, "fc23": 2, "fc123": 3}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def value_or_nan(value: str | None) -> float:
    try:
        return float(value) if value is not None else math.nan
    except (TypeError, ValueError):
        return math.nan


def fmt(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:8.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "train/results/SI/evaluation/bias",
    )
    return parser.parse_args()


def main() -> None:
    base = parse_args().results.resolve()
    scene_data: dict[tuple[str, str, str], float] = {}
    invivo_data: dict[tuple[str, str], float] = {}
    for stage in STAGES:
        for row in read_csv(base / stage / "scenes/all_scene_metrics.csv"):
            if row.get("method") in CONFIGS:
                scene_data[(stage, row["method"], row.get("scene", ""))] = value_or_nan(row.get("SSIM_dB_vs_GT"))
        summary = read_csv(base / stage / "invivo/test_mc_summary.csv")
        summary = summary or read_csv(base / stage / "invivo/test_ssim_summary.csv")
        for row in summary:
            if row.get("model") in CONFIGS:
                invivo_data[(stage, row["model"])] = value_or_nan(row.get("ssim_db_mean"))

    print("Bias 消融综合结果 (invivo SSIM_dB)")
    print(f"{'配置':<10} {'bias层':<8} " + " ".join(f"{stage:>8}" for stage in STAGES))
    for config in CONFIGS:
        values = [fmt(invivo_data.get((stage, config), math.nan)) for stage in STAGES]
        print(f"{config:<10} {BIAS_LAYER_COUNT[config]:<8} " + " ".join(values))

    print("\n四场景 SSIM_dB_vs_GT")
    for scene in SCENES:
        print(f"\n--- {scene} ---")
        print(f"{'配置':<10} " + " ".join(f"{stage:>8}" for stage in STAGES))
        for config in CONFIGS:
            values = [fmt(scene_data.get((stage, config, scene), math.nan)) for stage in STAGES]
            print(f"{config:<10} " + " ".join(values))


if __name__ == "__main__":
    main()

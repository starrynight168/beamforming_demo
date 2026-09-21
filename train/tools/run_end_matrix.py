from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


SCENES = (
    "simulation_contrast_speckle",
    "simulation_resolution_distorsion",
    "experiments_contrast_speckle",
    "experiments_resolution_distorsion",
)


def model_config(checkpoint: Path, name: str, profile: str, overrides: dict[str, object]) -> dict[str, object]:
    return {
        "models": [
            {
                "display_name": name,
                "checkpoint": str(checkpoint.resolve()),
                "model_type": "ptq",
                "profile": profile,
                "overrides": overrides,
            }
        ]
    }


def run_command(command: list[str], cwd: Path) -> None:
    print("[RUN] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def run_variant(
    train_dir: Path,
    output_root: Path,
    checkpoint: Path,
    split_file: Path,
    name: str,
    profile: str,
    overrides: dict[str, object],
    runs: int,
    frames: int,
    device: str,
) -> None:
    variant_dir = output_root / name
    invivo_dir = variant_dir / "invivo"
    scenes_dir = variant_dir / "scenes"
    invivo_done = (invivo_dir / "test_mc_summary.csv").is_file()
    scenes_done = (scenes_dir / "monte_carlo_ssim_summary.csv").is_file()
    config = model_config(checkpoint, name, profile, overrides)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", prefix="end_eval_", delete=False
    ) as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        config_path = Path(handle.name)
    try:
        common = [
            sys.executable,
            "-X",
            "utf8",
            str(train_dir / "evaluate_mban.py"),
            "--models-config",
            str(config_path),
            "--config",
            str(train_dir / "config.yaml"),
            "--eval-config",
            str(train_dir / "hardware_eval.yaml"),
            "--split-file",
            str(split_file.resolve()),
            "--device",
            device,
            "--micro-batch",
            "8192",
            "--seed",
            "42",
        ]
        if not invivo_done:
            run_command(
                [
                    *common,
                    "--mode",
                    "mc",
                    "--output",
                    str(invivo_dir),
                    "--test-frames",
                    str(frames),
                    "--runs",
                    str(runs),
                ],
                train_dir,
            )
        else:
            print(f"[SKIP] {name}/invivo", flush=True)
        if not scenes_done:
            run_command(
                [
                    *common,
                    "--mode",
                    "scenes",
                    "--output",
                    str(scenes_dir),
                    "--scenes",
                    ",".join(SCENES),
                    "--monte-carlo-runs",
                    str(runs),
                ],
                train_dir,
            )
        else:
            print(f"[SKIP] {name}/scenes", flush=True)
    finally:
        config_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--only",
        choices=("all", "ladder", "none_factors", "none_tail", "linf_factors"),
        default="all",
    )
    args = parser.parse_args()
    train_dir = Path(__file__).resolve().parent
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    ladder_root = output_root / "PTQ_ladder"
    factor_root = output_root / "single_factors"
    variants: list[tuple[Path, str, str, dict[str, object]]] = []
    for bits in (3, 4, 6, 8):
        for normalization in ("none", "linf"):
            name = f"PTQ{bits}_{normalization}"
            variants.append(
                (
                    ladder_root,
                    name,
                    "common",
                    {
                        "weight_bits": bits,
                        "bias_bits": bits,
                        "control_bits": bits,
                        "control_normalization": normalization,
                    },
                )
            )
    factor_overrides = {
        "adc_noise": {
            "adc_noise_std": 0.0333333333,
            "adc_nonlinearity": 0.02,
            "adc_nonlinearity_beta": 3.0,
            "noise_enabled": True,
        },
        "drift": {"drift_std": 0.07, "noise_enabled": True},
        "ir_drop": {
            "ir_drop_enabled": True,
            "ir_drop_g_max": 3.13e-5,
            "ir_drop_tile_cols": 64,
            "ir_drop_tile_rows": 64,
            "ir_drop_v_read": 0.3,
            "ir_drop_wire_resistance": 0.35,
            "noise_enabled": True,
        },
        "read_noise": {
            "noise_enabled": True,
            "read_noise_mode": "common_mode_relative",
            "read_noise_std": 0.02,
        },
        "stuck_at": {
            "noise_enabled": True,
            "stuck_at_high_fraction": 0.5,
            "stuck_at_prob": 0.01,
        },
        "tia_mismatch": {
            "noise_enabled": True,
            "tia_channel_gain_mismatch_std": 0.02,
            "tia_noise_mode": "full_scale",
            "tia_noise_reference": 1.0,
            "tia_noise_std": 0.005,
        },
        "write_error": {
            "noise_enabled": True,
            "write_error_std": 0.05,
            "write_noise_mode": "full_scale",
        },
    }
    for normalization in ("none", "linf"):
        for factor, factor_values in factor_overrides.items():
            name = f"PTQ4_{normalization}_{factor}"
            variants.append(
                (
                    factor_root,
                    name,
                    "ideal",
                    {
                        "weight_bits": 4,
                        "bias_bits": 4,
                        "control_bits": 4,
                        "control_normalization": normalization,
                        **factor_values,
                    },
                )
            )
    if args.only == "ladder":
        variants = [item for item in variants if item[0].name == "PTQ_ladder"]
    elif args.only == "none_factors":
        variants = [item for item in variants if item[0].name == "single_factors" and "_none_" in item[1]]
    elif args.only == "none_tail":
        variants = [
            item
            for item in variants
            if item[0].name == "single_factors"
            and "_none_" in item[1]
            and not item[1].endswith(("_adc_noise", "_drift", "_ir_drop"))
        ]
    elif args.only == "linf_factors":
        variants = [item for item in variants if item[0].name == "single_factors" and "_linf_" in item[1]]
    for root, name, profile, overrides in variants:
        run_variant(
            train_dir,
            root,
            args.checkpoint,
            args.split_file,
            name,
            profile,
            overrides,
            args.runs,
            args.frames,
            args.device,
        )
    print(f"[DONE] variants={len(variants)}", flush=True)


if __name__ == "__main__":
    main()

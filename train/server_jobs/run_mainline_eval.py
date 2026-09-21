#!/usr/bin/env python3
"""
Orchestrator for Mainline Study Evaluation.
Evaluates FP32 and 3 QAT models (ADC 4/6/8-bit) across In Vivo test split and 4 PICMUS physical scenes.
Generates comprehensive benchmark comparison tables and JSON summaries.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
TRAIN_ROOT = ROOT_DIR / "train"
PYTHON_BIN = os.environ.get("PYTHON", sys.executable)

CONFIG_PATH = TRAIN_ROOT / "config.yaml"
ROOT_CONFIG_PATH = ROOT_DIR / "config.yaml"
HARDWARE_CONFIG_PATH = TRAIN_ROOT / "hardware_eval.yaml"
SPLIT_FILE = Path(
    os.environ.get(
        "SPLIT_FILE",
        TRAIN_ROOT / "results/SI/shared_split_seed42_0p7-0p2-0p1.csv",
    )
).expanduser().resolve()

STUDY_ROOT = TRAIN_ROOT / "results/SI/model/adc_bit"
FP32_CKPT = STUDY_ROOT / "FP32/best_val.pth"
ADC4_CKPT = STUDY_ROOT / "QAT4/ADC4/best_val.pth"
ADC6_CKPT = STUDY_ROOT / "QAT4/ADC6/best_val.pth"
ADC8_CKPT = STUDY_ROOT / "QAT4/ADC8/best_val.pth"

EVAL_DIR = TRAIN_ROOT / "results/SI/evaluation/adc_bit"

BASE_ENV = os.environ.copy()
if conda_prefix := os.environ.get("CONDA_PREFIX"):
    conda_lib = Path(conda_prefix) / "lib"
    if conda_lib.is_dir():
        BASE_ENV["LD_LIBRARY_PATH"] = f"{conda_lib}:{BASE_ENV.get('LD_LIBRARY_PATH', '')}"
BASE_ENV["OMP_NUM_THREADS"] = "16"
BASE_ENV["MKL_NUM_THREADS"] = "16"
BASE_ENV["OPENBLAS_NUM_THREADS"] = "16"
BASE_ENV["NUMEXPR_NUM_THREADS"] = "16"
BASE_ENV["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
BASE_ENV["PYTHONUNBUFFERED"] = "1"

SCENES_LIST = (
    "experiments_contrast_speckle,"
    "experiments_resolution_distorsion,"
    "simulation_contrast_speckle,"
    "simulation_resolution_distorsion"
)


def run_cmd(cmd: list[str], log_file: Path, gpu_id: int = 3) -> None:
    env = BASE_ENV.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] [RUN] {' '.join(cmd)}")
    with open(log_file, "w", encoding="utf-8") as out:
        subprocess.run(cmd, env=env, cwd=str(TRAIN_ROOT), stdout=out, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    gpu_id = 3

    print("==================================================================")
    print("=== Mainline Study Comprehensive Benchmark Evaluation ===")
    print(f"=== GPU: {gpu_id} | Output: {EVAL_DIR} ===")
    print("==================================================================")

    # 1. FP32 Evaluation
    print("\n--- [Step 1/2] Evaluating FP32 Baseline ---")
    fp32_invivo_out = EVAL_DIR / "FP32/invivo"
    fp32_scenes_out = EVAL_DIR / "FP32/scenes"

    if not (fp32_invivo_out / "test_ssim_summary.csv").is_file():
        cmd_fp32_invivo = [
            PYTHON_BIN, "-u", str(TRAIN_ROOT / "evaluate_mban.py"),
            "--mode", "test", "--model-type", "fp32",
            f"--model=FP32={FP32_CKPT}",
            "--split-file", str(SPLIT_FILE),
            "--config", str(CONFIG_PATH),
            "--output", str(fp32_invivo_out),
            "--micro-batch", "8192",
            "--device", "cuda:0",
        ]
        run_cmd(cmd_fp32_invivo, fp32_invivo_out / "eval.log", gpu_id=gpu_id)
        print(f"[{time.strftime('%H:%M:%S')}] [DONE] FP32 In Vivo evaluated.")
    else:
        print("[SKIP] FP32 In Vivo already done.")

    if not (fp32_scenes_out / "all_scene_metrics.csv").is_file():
        cmd_fp32_scenes = [
            PYTHON_BIN, "-u", str(TRAIN_ROOT / "evaluate_mban.py"),
            "--mode", "scenes", "--model-type", "fp32",
            f"--model=FP32={FP32_CKPT}",
            "--config", str(CONFIG_PATH),
            "--output", str(fp32_scenes_out),
            "--scene-config", str(ROOT_CONFIG_PATH),
            "--scenes", SCENES_LIST,
            "--seed", "42",
            "--micro-batch", "8192",
            "--device", "cuda:0",
        ]
        run_cmd(cmd_fp32_scenes, fp32_scenes_out / "eval.log", gpu_id=gpu_id)
        print(f"[{time.strftime('%H:%M:%S')}] [DONE] FP32 Scenes evaluated.")
    else:
        print("[SKIP] FP32 Scenes already done.")

    # 2. QAT Evaluation (3 models in parallel or batch)
    print("\n--- [Step 2/2] Evaluating 3 QAT Models (ADC 4, 6, 8-bit) ---")
    qat_models = [
        ("ADC4", ADC4_CKPT),
        ("ADC6", ADC6_CKPT),
        ("ADC8", ADC8_CKPT),
    ]
    qat_model_args = [f"--model={name}={ckpt}" for name, ckpt in qat_models]

    qat_invivo_out = EVAL_DIR / "QAT4/invivo"
    qat_scenes_out = EVAL_DIR / "QAT4/scenes"

    if not (qat_invivo_out / "test_mc_summary.csv").is_file():
        cmd_qat_invivo = [
            PYTHON_BIN, "-u", str(TRAIN_ROOT / "evaluate_mban.py"),
            "--mode", "mc", "--model-type", "qat",
            *qat_model_args,
            "--split-file", str(SPLIT_FILE),
            "--config", str(CONFIG_PATH),
            "--eval-config", str(HARDWARE_CONFIG_PATH),
            "--profile", "common",
            "--output", str(qat_invivo_out),
            "--test-frames", "10",
            "--seed", "42",
            "--runs", "10",
            "--micro-batch", "8192",
            "--device", "cuda:0",
        ]
        run_cmd(cmd_qat_invivo, qat_invivo_out / "eval.log", gpu_id=gpu_id)
        print(f"[{time.strftime('%H:%M:%S')}] [DONE] QAT In Vivo MC-10 evaluated.")
    else:
        print("[SKIP] QAT In Vivo already done.")

    if not (qat_scenes_out / "monte_carlo_ssim_summary.csv").is_file():
        cmd_qat_scenes = [
            PYTHON_BIN, "-u", str(TRAIN_ROOT / "evaluate_mban.py"),
            "--mode", "scenes", "--model-type", "qat",
            *qat_model_args,
            "--config", str(CONFIG_PATH),
            "--eval-config", str(HARDWARE_CONFIG_PATH),
            "--profile", "common",
            "--output", str(qat_scenes_out),
            "--scene-config", str(ROOT_CONFIG_PATH),
            "--scenes", SCENES_LIST,
            "--seed", "42",
            "--monte-carlo-runs", "10",
            "--micro-batch", "8192",
            "--device", "cuda:0",
        ]
        run_cmd(cmd_qat_scenes, qat_scenes_out / "eval.log", gpu_id=gpu_id)
        print(f"[{time.strftime('%H:%M:%S')}] [DONE] QAT Scenes MC-10 evaluated.")
    else:
        print("[SKIP] QAT Scenes already done.")

    # 3. Aggregate & Report
    print("\n==================================================================")
    print("=== Aggregating Evaluation Results ===")
    print("==================================================================")
    summary_rows = []

    # Read FP32 invivo
    fp32_invivo_ssim_db = None
    fp32_invivo_ssim_env = None
    fp32_invivo_csv = fp32_invivo_out / "test_ssim_summary.csv"
    if fp32_invivo_csv.is_file():
        with open(fp32_invivo_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("model") == "FP32":
                    fp32_invivo_ssim_db = float(row["ssim_db_mean"])
                    fp32_invivo_ssim_env = float(row["ssim_env_mean"])
                    break

    # Read FP32 scenes
    fp32_scenes_dict = {}
    fp32_scenes_csv = fp32_scenes_out / "all_scene_metrics.csv"
    if fp32_scenes_csv.is_file():
        with open(fp32_scenes_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("method") == "FP32":
                    scene = row.get("scene", "")
                    fp32_scenes_dict[scene] = {
                        "ssim_db": float(row["SSIM_dB_vs_GT"]),
                        "ssim_env": float(row["SSIM_envelope_vs_GT"]),
                        "cnr": float(row["CNR"]) if row.get("CNR") and row["CNR"] != "nan" else None,
                        "gcnr": float(row["gCNR"]) if row.get("gCNR") and row["gCNR"] != "nan" else None,
                    }

    fp32_scenes_mean_db = (
        sum(v["ssim_db"] for v in fp32_scenes_dict.values()) / len(fp32_scenes_dict)
        if fp32_scenes_dict else None
    )

    summary_rows.append({
        "model": "FP32",
        "adc_bits": "32 (FP32)",
        "invivo_ssim_db": fp32_invivo_ssim_db,
        "invivo_ssim_env": fp32_invivo_ssim_env,
        "scenes_mean_ssim_db": fp32_scenes_mean_db,
        "exp_contrast_ssim_db": fp32_scenes_dict.get("experiments_contrast_speckle", {}).get("ssim_db"),
        "exp_res_ssim_db": fp32_scenes_dict.get("experiments_resolution_distorsion", {}).get("ssim_db"),
        "sim_contrast_ssim_db": fp32_scenes_dict.get("simulation_contrast_speckle", {}).get("ssim_db"),
        "sim_res_ssim_db": fp32_scenes_dict.get("simulation_resolution_distorsion", {}).get("ssim_db"),
    })

    # Read QAT invivo
    qat_invivo_dict = {}
    qat_invivo_csv = qat_invivo_out / "test_mc_summary.csv"
    if qat_invivo_csv.is_file():
        with open(qat_invivo_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model = row.get("model", "")
                qat_invivo_dict[model] = {
                    "ssim_db_mean": float(row["ssim_db_mean"]),
                    "ssim_db_std": float(row["ssim_db_std"]),
                    "ssim_env_mean": float(row["ssim_env_mean"]),
                }

    # Read QAT scenes
    qat_scenes_dict = {}
    qat_scenes_csv = qat_scenes_out / "monte_carlo_ssim_summary.csv"
    if qat_scenes_csv.is_file():
        with open(qat_scenes_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model = row.get("method", "")
                scene = row.get("scene", "")
                if model not in qat_scenes_dict:
                    qat_scenes_dict[model] = {}
                qat_scenes_dict[model][scene] = {
                    "ssim_db_mean": float(row["ssim_db_mean"]),
                    "ssim_db_std": float(row["ssim_db_std"]),
                    "ssim_env_mean": float(row["ssim_env_mean"]),
                }

    for name, bits in [("ADC4", 4), ("ADC6", 6), ("ADC8", 8)]:
        inv_data = qat_invivo_dict.get(name, {})
        sc_data = qat_scenes_dict.get(name, {})
        mean_sc_db = (
            sum(v["ssim_db_mean"] for v in sc_data.values()) / len(sc_data)
            if sc_data else None
        )
        summary_rows.append({
            "model": name,
            "adc_bits": f"{bits}-bit",
            "invivo_ssim_db": inv_data.get("ssim_db_mean"),
            "invivo_ssim_env": inv_data.get("ssim_env_mean"),
            "scenes_mean_ssim_db": mean_sc_db,
            "exp_contrast_ssim_db": sc_data.get("experiments_contrast_speckle", {}).get("ssim_db_mean"),
            "exp_res_ssim_db": sc_data.get("experiments_resolution_distorsion", {}).get("ssim_db_mean"),
            "sim_contrast_ssim_db": sc_data.get("simulation_contrast_speckle", {}).get("ssim_db_mean"),
            "sim_res_ssim_db": sc_data.get("simulation_resolution_distorsion", {}).get("ssim_db_mean"),
        })

    # Save summary
    summary_csv = EVAL_DIR / "mainline_benchmark_summary.csv"
    summary_json = EVAL_DIR / "mainline_benchmark_summary.json"
    if summary_rows:
        with open(summary_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary_rows, f, indent=2, ensure_ascii=False)

    print("\n🎉 [EVALUATION COMPLETE] Summary Results:")
    print("-" * 100)
    fmt = "{:<12} | {:<10} | {:<15} | {:<15} | {:<15}"
    print(fmt.format("Model", "ADC Bits", "InVivo SSIM(dB)", "Scenes Mean(dB)", "Exp Contrast(dB)"))
    print("-" * 100)
    for r in summary_rows:
        inv_str = f"{r['invivo_ssim_db']:.4f}" if r['invivo_ssim_db'] is not None else "N/A"
        sc_str = f"{r['scenes_mean_ssim_db']:.4f}" if r['scenes_mean_ssim_db'] is not None else "N/A"
        exp_str = f"{r['exp_contrast_ssim_db']:.4f}" if r['exp_contrast_ssim_db'] is not None else "N/A"
        print(fmt.format(r['model'], r['adc_bits'], inv_str, sc_str, exp_str))
    print("-" * 100)


if __name__ == "__main__":
    main()

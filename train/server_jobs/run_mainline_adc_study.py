#!/usr/bin/env python3
"""
Mainline Experiment Orchestrator:
1. Trains 1 FP32 baseline with ordinary analog inter-layer bias (100 ep).
2. Automatically launches 3 QAT4 models warm-started from the FP32 checkpoint,
   sweeping output control ADC bit-widths: ADC 4-bit, ADC 6-bit, ADC 8-bit.
"""

from __future__ import annotations

import os
import sys
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
TRAIN_ROOT = PROJECT_ROOT / "train"
PYTHON_BIN = os.environ.get("PYTHON", sys.executable)
RESULTS_ROOT = TRAIN_ROOT / "results/SI/model/adc_bit"
LOGS_ROOT = TRAIN_ROOT / "results/SI/logs/adc_bit"

CONFIG_PATH = TRAIN_ROOT / "config.yaml"
HARDWARE_CONFIG_PATH = TRAIN_ROOT / "hardware_eval.yaml"
SPLIT_FILE = Path(
    os.environ.get(
        "SPLIT_FILE",
        TRAIN_ROOT / "results/SI/shared_split_seed42_0p7-0p2-0p1.csv",
    )
).expanduser().resolve()

LOSS_STRING = "{error_function: mse, charbonnier_epsilon: 0.001, envelope_epsilon: 0.000001, terms: {iq: {weight: 1.0}, envelope: {weight: 0.1}, unity: {weight: 0.0}, d1: {domain: shape, weight: 0.0}, d2: {domain: shape, weight: 0.0}, lrange: {limit: 2.0, weight: 0.1}}}"

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


def run_process_and_monitor(name: str, cmd: list[str], log_file: Path, gpu_id: int) -> int:
    env = BASE_ENV.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[{time.strftime('%H:%M:%S')}] [LAUNCH] {name} on GPU {gpu_id}")
    print(f"       Command: {' '.join(cmd)}")
    print(f"       Log: {log_file}")
    with open(log_file, "w", encoding="utf-8") as out:
        p = subprocess.Popen(cmd, env=env, cwd=str(TRAIN_ROOT), stdout=out, stderr=subprocess.STDOUT)
    
    while p.poll() is None:
        time.sleep(15)
        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                for line in reversed(lines):
                    if "[EPOCH]" in line or "[SUCCESS]" in line or "[ERROR]" in line:
                        print(f"[{time.strftime('%H:%M:%S')}] [{name}] {line}")
                        break
        except OSError:
            pass

    code = p.returncode
    if code == 0:
        print(f"[{time.strftime('%H:%M:%S')}] [SUCCESS] {name} completed with code 0!\n")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] [ERROR] {name} failed with code {code}!\n")
    return code


def summarize_parallel_runs(procs: list[dict[str, object]]) -> list[str]:
    failed = []
    for item in procs:
        item["out"].close()
        code = item["proc"].returncode
        print(f"[{time.strftime('%H:%M:%S')}] [{item['name']}] Exited with code {code}")
        if code != 0:
            failed.append(str(item["name"]))
    return failed


def main():
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    fp32_dir = RESULTS_ROOT / "FP32"
    fp32_ckpt = fp32_dir / "best_val.pth"

    # Step 1: Run FP32 if not already done
    if not fp32_ckpt.is_file():
        print("==================================================================")
        print("=== Step 1: Training FP32 Baseline (Ordinary + Analog Inter-layer)")
        print("==================================================================")
        fp32_cmd = [
            PYTHON_BIN, "-u", "mban.py",
            "--config", str(CONFIG_PATH),
            "--mode", "software",
            "--inter-layer", "analog",
            "--set", f"output_directory={fp32_dir}",
            "--set", f"split_file={SPLIT_FILE}",
            "--set", "bias_implementation=ordinary",
            "--set", "bias_layers=[fc1, fc2, fc3]",
            "--set", "interpolation_bits=32",
            "--set", f"loss={LOSS_STRING}",
            "--set", "resume=none",
        ]
        code = run_process_and_monitor("mainline_FP32", fp32_cmd, LOGS_ROOT / "fp32.log", gpu_id=3)
        if code != 0 or not fp32_ckpt.is_file():
            print("[FATAL] FP32 training failed. Aborting QAT runs.")
            return 1
    else:
        print(f"[FOUND] Existing FP32 checkpoint at {fp32_ckpt}, proceeding to QAT directly!")

    # Step 2: Run 3 QAT models in parallel
    print("==================================================================")
    print("=== Step 2: Training 3 QAT Models in Parallel (ADC 4, 6, 8 bits) ==")
    print("==================================================================")
    
    adc_gpus = [int(value) for value in os.environ.get("ADC_GPUS", "3,5,6").split(",") if value.strip()]
    if len(adc_gpus) != 3:
        raise ValueError("ADC_GPUS must contain exactly three GPU indices")
    adc_configs = [(f"qat4_adc{bits}", bits, gpu) for bits, gpu in zip((4, 6, 8), adc_gpus, strict=True)]

    procs = []
    for name, adc_bits, gpu_id in adc_configs:
        out_dir = RESULTS_ROOT / "QAT4" / f"ADC{adc_bits}"
        log_f = LOGS_ROOT / f"{name}.log"
        cmd = [
            PYTHON_BIN, "-u", "mban.py",
            "--config", str(CONFIG_PATH),
            "--eval-config", str(HARDWARE_CONFIG_PATH),
            "--mode", "qat",
            "--profile", "common",
            "--inter-layer", "analog",
            "--set", f"output_directory={out_dir}",
            "--set", f"initial_checkpoint={fp32_ckpt}",
            "--set", f"split_file={SPLIT_FILE}",
            "--set", "bias_implementation=ordinary",
            "--set", "bias_layers=[fc1, fc2, fc3]",
            "--set", f"control_bits={adc_bits}",
            "--set", "interpolation_bits=32",
            "--set", f"loss={LOSS_STRING}",
            "--set", "resume=none",
        ]
        env = BASE_ENV.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[{time.strftime('%H:%M:%S')}] [SPAWN] {name} (ADC={adc_bits}-bit) on GPU {gpu_id}")
        out = open(log_f, "w", encoding="utf-8")
        p = subprocess.Popen(cmd, env=env, cwd=str(TRAIN_ROOT), stdout=out, stderr=subprocess.STDOUT)
        procs.append({
            "name": name,
            "adc_bits": adc_bits,
            "gpu": gpu_id,
            "proc": p,
            "out": out,
            "log": log_f,
        })

    # Monitor parallel QAT runs
    while any(item["proc"].poll() is None for item in procs):
        time.sleep(15)
        for item in procs:
            if item["proc"].poll() is None:
                try:
                    with open(item["log"], "r", encoding="utf-8", errors="ignore") as f:
                        lines = [line.strip() for line in f.readlines() if line.strip()]
                        for line in reversed(lines):
                            if "[EPOCH]" in line:
                                print(f"[{time.strftime('%H:%M:%S')}] [{item['name']} (GPU {item['gpu']})] {line}")
                                break
                except OSError:
                    pass

    failed = summarize_parallel_runs(procs)
    if failed:
        print(f"\n[ERROR] Mainline study failed: {', '.join(failed)}")
        return 1
    print("\n🎉 [ALL COMPLETE] Mainline study FP32 and 3 QAT models finished successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

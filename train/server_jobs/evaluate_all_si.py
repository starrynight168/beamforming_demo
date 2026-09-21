#!/usr/bin/env python3
"""Parallel Evaluation for SI ablation checkpoints across GPUs with multi-core CPU."""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
TRAIN_ROOT = PROJECT_ROOT / "train"
PYTHON_BIN = os.environ.get("PYTHON", sys.executable)
SI_ROOT = TRAIN_ROOT / "results/SI"
MODEL_ROOT = SI_ROOT / "model"
EVAL_ROOT = SI_ROOT / "evaluation"
CONFIG_PATH = TRAIN_ROOT / "config.yaml"
ROOT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
HARDWARE_CONFIG_PATH = TRAIN_ROOT / "hardware_eval.yaml"
SPLIT_FILE = Path(
    os.environ.get(
        "SPLIT_FILE",
        next(
            (
                str(path)
                for path in (
                    SI_ROOT / "shared_split_seed42_0p7-0p2-0p1.csv",
                    TRAIN_ROOT / "results/model_ablation/shared_split_seed42_0p7-0p2-0p1.csv",
                )
                if path.is_file()
            ),
            str(SI_ROOT / "shared_split_seed42_0p7-0p2-0p1.csv"),
        ),
    )
).expanduser().resolve()
LOGS_DIR = SI_ROOT / "logs"

BASE_ENV = os.environ.copy()
if conda_prefix := os.environ.get("CONDA_PREFIX"):
    conda_lib = Path(conda_prefix) / "lib"
    if conda_lib.is_dir():
        BASE_ENV["LD_LIBRARY_PATH"] = f"{conda_lib}:{BASE_ENV.get('LD_LIBRARY_PATH', '')}"
BASE_ENV["OMP_NUM_THREADS"] = "2"
BASE_ENV["MKL_NUM_THREADS"] = "2"
BASE_ENV["OPENBLAS_NUM_THREADS"] = "2"
BASE_ENV["NUMEXPR_NUM_THREADS"] = "2"
BASE_ENV["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
BASE_ENV["PYTHONUNBUFFERED"] = "1"
BASE_ENV["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

SCENES_LIST = (
    "experiments_contrast_speckle,experiments_resolution_distorsion,"
    "simulation_contrast_speckle,simulation_resolution_distorsion"
)
DEFAULT_MODULES = (
    "input_norm",
    "interpolation",
    "activation",
    "outlimit",
    "bias",
    "bias_interlayer",
    "envelope",
    "loss",
    "lrange",
    "qattext",
    "normalization",
    "control_adc",
    "windows",
)
MODEL_GROUP_KEYS = frozenset(("FP32", "QAT4", "QAT", "QATLINF"))
MODEL_GROUPS = ("FP32", "QAT4", "QAT", "QATlinf")


def normalize_group(raw: str) -> str:
    u = raw.upper().replace("_", "")
    if u == "QATLINF":
        return "QATlinf"
    return u if u in {"FP32", "QAT4", "QAT"} else raw


def discover_models(mod_dir: Path) -> dict[str, list[tuple[str, Path]]]:
    models = {group: [] for group in MODEL_GROUPS}
    for checkpoint in sorted(mod_dir.rglob("best_val.pth")):
        rel_parts = checkpoint.relative_to(mod_dir).parts[:-1]
        group_index = next(
            (index for index, part in enumerate(rel_parts) if part.upper().replace("_", "") in MODEL_GROUP_KEYS),
            None,
        )
        if group_index is None:
            group = "FP32"
            label = rel_parts[-1] if rel_parts else mod_dir.name
        else:
            group = normalize_group(rel_parts[group_index])
            label = rel_parts[group_index + 1] if group_index + 1 < len(rel_parts) else "baseline"
        models.setdefault(group, []).append((label, checkpoint))
    return models


def run_evaluation(cmd: list[str], output_dir: Path, env: dict[str, str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "eval.log").open("w", encoding="utf-8") as log_file:
        subprocess.run(
            cmd,
            env=env,
            cwd=str(TRAIN_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )


def evaluate_module(mod_name: str, gpu_id: int = 0) -> str:
    mod_dir = MODEL_ROOT / mod_name
    eval_dir = EVAL_ROOT / mod_name
    log_file = LOGS_DIR / f"eval_{mod_name}.log"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    def log(msg: str):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        full_msg = f"[{timestamp}][{mod_name}|GPU:{gpu_id}] {msg}"
        print(full_msg, flush=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(full_msg + "\n")
            f.flush()

    if not mod_dir.exists():
        log(f"[SKIP] Module {mod_name} does not exist.")
        return f"{mod_name}: skipped"

    log(f"=== Starting Evaluation for Module: {mod_name} on GPU {gpu_id} ===")
    models = discover_models(mod_dir)

    proc_env = BASE_ENV.copy()
    proc_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 1. Evaluate FP32 models if present
    fp32_models = models.get("FP32", [])
    if fp32_models:
        log(f"Running FP32 Evaluation ({len(fp32_models)} models)...")
        fp32_args = [f"--model={name}={p}" for name, p in fp32_models]

        # In Vivo 50 frames
        invivo_out = eval_dir / "FP32" / "invivo"
        if (invivo_out / "test_ssim_summary.csv").exists():
            log(f"[SKIP] FP32 In Vivo already evaluated -> {invivo_out}")
        else:
            log(f"Running FP32 In Vivo ({len(fp32_models)} models)...")
            cmd_invivo = [
                PYTHON_BIN, "-u", str(TRAIN_ROOT / "evaluate_mban.py"),
                "--mode", "test", "--model-type", "fp32",
                *fp32_args,
                "--split-file", str(SPLIT_FILE),
                "--config", str(CONFIG_PATH),
                "--output", str(invivo_out),
                "--micro-batch", "8192",
                "--device", "cuda:0",
            ]
            run_evaluation(cmd_invivo, invivo_out, proc_env)
            log(f"[DONE] FP32 In Vivo evaluated -> {invivo_out}")

        # Physical Scenes
        scenes_out = eval_dir / "FP32" / "scenes"
        if (scenes_out / "all_scene_metrics.csv").exists():
            log(f"[SKIP] FP32 Scenes already evaluated -> {scenes_out}")
        else:
            log(f"Running FP32 Scenes ({len(fp32_models)} models)...")
            cmd_scenes = [
                PYTHON_BIN, "-u", str(TRAIN_ROOT / "evaluate_mban.py"),
                "--mode", "scenes", "--model-type", "fp32",
                *fp32_args,
                "--config", str(CONFIG_PATH),
                "--output", str(scenes_out),
                "--scene-config", str(ROOT_CONFIG_PATH),
                "--scenes", SCENES_LIST,
                "--seed", "42",
                "--micro-batch", "8192",
                "--device", "cuda:0",
            ]
            run_evaluation(cmd_scenes, scenes_out, proc_env)
            log(f"[DONE] FP32 Scenes evaluated -> {scenes_out}")

    # 2. Evaluate all non-FP32 groups (QAT, QAT4, QATlinf, etc.)
    for qat_group, qat_models in models.items():
        if qat_group == "FP32" or not qat_models:
            continue
        log(f"Running {qat_group} Evaluation ({len(qat_models)} models)...")
        qat_args = [f"--model={name}={p}" for name, p in qat_models]

        # In Vivo 10-run MC 10 frames
        invivo_out = eval_dir / qat_group / "invivo"
        if (invivo_out / "test_run_ssim_summary.csv").exists():
            log(f"[SKIP] {qat_group} In Vivo already evaluated -> {invivo_out}")
        else:
            log(f"Running {qat_group} In Vivo ({len(qat_models)} models)...")
            cmd_invivo = [
                PYTHON_BIN, "-u", str(TRAIN_ROOT / "evaluate_mban.py"),
                "--mode", "mc", "--model-type", "qat",
                *qat_args,
                "--split-file", str(SPLIT_FILE),
                "--config", str(CONFIG_PATH),
                "--eval-config", str(HARDWARE_CONFIG_PATH),
                "--profile", "common",
                "--output", str(invivo_out),
                "--test-frames", "10",
                "--seed", "42",
                "--runs", "10",
                "--micro-batch", "8192",
                "--device", "cuda:0",
            ]
            run_evaluation(cmd_invivo, invivo_out, proc_env)
            log(f"[DONE] {qat_group} In Vivo MC-10 evaluated -> {invivo_out}")

        # Physical Scenes 10-run MC
        scenes_out = eval_dir / qat_group / "scenes"
        if (scenes_out / "monte_carlo_ssim_summary.csv").exists():
            log(f"[SKIP] {qat_group} Scenes already evaluated -> {scenes_out}")
        else:
            log(f"Running {qat_group} Scenes ({len(qat_models)} models)...")
            cmd_scenes = [
                PYTHON_BIN, "-u", str(TRAIN_ROOT / "evaluate_mban.py"),
                "--mode", "scenes", "--model-type", "qat",
                *qat_args,
                "--config", str(CONFIG_PATH),
                "--eval-config", str(HARDWARE_CONFIG_PATH),
                "--profile", "common",
                "--output", str(scenes_out),
                "--scene-config", str(ROOT_CONFIG_PATH),
                "--scenes", SCENES_LIST,
                "--seed", "42",
                "--monte-carlo-runs", "10",
                "--micro-batch", "8192",
                "--device", "cuda:0",
            ]
            run_evaluation(cmd_scenes, scenes_out, proc_env)
            log(f"[DONE] {qat_group} Scenes MC-10 evaluated -> {scenes_out}")

    log(f"=== Completed Evaluation for Module: {mod_name} ===")
    return f"{mod_name}: done"


def evaluate_gpu_group(gpu_id: int, modules: list[str]) -> list[str]:
    return [evaluate_module(module, gpu_id) for module in modules]


def evaluate_parallel(tasks: list[tuple[str, int]]) -> None:
    if not tasks:
        raise ValueError("no evaluation tasks requested")
    grouped: dict[int, list[str]] = {}
    for module, gpu in tasks:
        grouped.setdefault(gpu, []).append(module)
    print(f"Launching {len(tasks)} evaluation tasks across {len(grouped)} GPUs...", flush=True)
    with ProcessPoolExecutor(max_workers=len(grouped)) as executor:
        futures = {
            executor.submit(evaluate_gpu_group, gpu, modules): gpu
            for gpu, modules in grouped.items()
        }
        failures = []
        for future in as_completed(futures):
            gpu = futures[future]
            try:
                for result in future.result():
                    print(f"[SUCCESS] {result}", flush=True)
            except Exception as e:
                print(f"[ERROR] GPU {gpu} evaluation failed: {e}", flush=True)
                failures.append(f"GPU{gpu}: {e}")
    if failures:
        raise RuntimeError("evaluation failed: " + "; ".join(failures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all SI ablation checkpoints")
    parser.add_argument("--modules", default=",".join(DEFAULT_MODULES))
    parser.add_argument("--gpus", default=os.environ.get("EVAL_GPUS", "0,5"))
    args = parser.parse_args()
    modules = [item.strip() for item in args.modules.split(",") if item.strip()]
    gpus = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise SystemExit("--gpus 不能为空")
    if any(gpu < 0 for gpu in gpus):
        raise SystemExit("--gpus 只能包含非负整数")
    evaluate_parallel([(module, gpus[index % len(gpus)]) for index, module in enumerate(modules)])

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TextIO


PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[3])).expanduser().resolve()
TRAIN_ROOT = PROJECT_ROOT / "train"
DATA_H5 = Path(os.environ.get("DATA_H5", PROJECT_ROOT / "data/volunteer_005.h5")).expanduser().resolve()
PYTHON_BIN = os.environ.get("PYTHON", sys.executable)
SPLIT_CANDIDATES = (
    TRAIN_ROOT / "results/SI/shared_split_seed42_0p7-0p2-0p1.csv",
    TRAIN_ROOT / "results/model_ablation/shared_split_seed42_0p7-0p2-0p1.csv",
)
SPLIT_FILE = Path(
    os.environ.get(
        "SPLIT_FILE",
        next((str(path) for path in SPLIT_CANDIDATES if path.is_file()), str(SPLIT_CANDIDATES[-1])),
    )
).expanduser().resolve()
BASE_CONFIG_PATH = TRAIN_ROOT / "config.yaml"
HARDWARE_CONFIG_PATH = TRAIN_ROOT / "hardware_eval.yaml"


def task_environment(gpu: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment.setdefault("OMP_NUM_THREADS", "8")
    environment.setdefault("MKL_NUM_THREADS", "8")
    return environment


def wait_for_processes(processes: list[tuple[str, subprocess.Popen, TextIO]]) -> None:
    failures = []
    for name, process, log in processes:
        status = process.wait()
        log.close()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {name} finished with code {status}")
        if status:
            failures.append(f"{name}={status}")
    if failures:
        raise RuntimeError("子任务失败: " + ", ".join(failures))


def load_base_config() -> dict:
    import yaml

    with BASE_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{BASE_CONFIG_PATH} 必须是 YAML 映射")
    return config


def require_inputs() -> None:
    for path in (BASE_CONFIG_PATH, HARDWARE_CONFIG_PATH, DATA_H5, SPLIT_FILE):
        if not path.is_file():
            raise FileNotFoundError(path)


def model_arguments(models: Iterable[tuple[str, Path]]) -> list[str]:
    arguments: list[str] = []
    missing: list[Path] = []
    for label, raw_path in models:
        path = Path(raw_path).resolve()
        if not path.is_file():
            missing.append(path)
            continue
        arguments.extend(("--model", f"{label}={path}"))
    if missing:
        details = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"missing checkpoints:\n{details}")
    return arguments


def evaluate_models(
    model_args: list[str],
    output_root: Path,
    eval_gpu: str,
    *,
    model_type: str = "fp32",
    test_frames: int = 50,
    mc_runs: int = 1,
) -> None:
    if model_type not in {"fp32", "qat"}:
        raise ValueError(f"unsupported model type: {model_type}")
    if test_frames <= 0 or mc_runs <= 0:
        raise ValueError("test_frames and mc_runs must be positive")
    environment = task_environment(int(eval_gpu))
    scenes_output = output_root / "scenes"
    invivo_output = output_root / "invivo"
    scenes_output.mkdir(parents=True, exist_ok=True)
    invivo_output.mkdir(parents=True, exist_ok=True)
    common = [
        "--model-type",
        model_type,
        "--config",
        str(BASE_CONFIG_PATH),
        "--eval-config",
        str(HARDWARE_CONFIG_PATH),
        "--profile",
        "common",
        "--seed",
        "42",
        "--micro-batch",
        "8192",
        "--device",
        "cuda:0",
    ]
    scenes_command = [
        PYTHON_BIN,
        str(TRAIN_ROOT / "evaluate_mban.py"),
        "--mode",
        "scenes",
        *model_args,
        *common,
        "--output",
        str(scenes_output),
    ]
    if model_type == "qat":
        scenes_command.extend(("--monte-carlo-runs", str(mc_runs)))
    subprocess.run(scenes_command, env=environment, check=True)
    test_command = [
        PYTHON_BIN,
        str(TRAIN_ROOT / "evaluate_mban.py"),
        "--mode",
        "mc" if model_type == "qat" else "test",
        *model_args,
        *common,
        "--output",
        str(invivo_output),
        "--split-file",
        str(SPLIT_FILE),
        "--test-frames",
        str(test_frames),
    ]
    if model_type == "qat":
        test_command.extend(("--runs", str(mc_runs)))
    subprocess.run(test_command, env=environment, check=True)

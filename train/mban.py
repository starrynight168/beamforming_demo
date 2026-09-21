"""
MBAN training script for adaptive receive beamforming from aligned complex IQ input.

Design:
  1. Keep the existing EPFL H5 loading, delay alignment, visualization, and checkpoint flow.
  2. One pixel gets its time-aligned aperture vector and predicts apodization weights.
  3. Dynamic aperture masks inactive channels; ReLU-hard applies one common output gain after accumulation.
  4. The first layer can run in exact input-feature tiles.
  5. MV losses combine complex-IQ error with a linear-envelope error term.

Usage::

    python mban.py --config config.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from mban_core.config import (
    CONFIG_FIELDS,
    MODEL_SEMANTICS_VERSION,
    load_config,
    load_nonideal_profile,
    runtime,
)
from mban_core.hardware import QATConfig
from mban_core.model import MBAN, set_runtime_args
from mban_core.beamforming import (
    beamform_iq_with_tx_phase,
    effective_aperture_weights,
    normalize_input_iq,
    pack_dynamic_input,
    predict_aperture_weights,
    predict_weights,
    split_complex_weights,
)
from mban_core.data import (
    compute_aperture_mask,
    derive_network_channels,
    extract_windows_on_gpu,
    load_or_create_mixed_split,
)
from mban_core.training import train

TRAIN_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = TRAIN_DIR / "config.yaml"

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


cli = argparse.ArgumentParser(description="MBAN training")
cli.add_argument(
    "--config",
    default=os.environ.get("MBAN_CONFIG", str(DEFAULT_CONFIG)),
    help="统一 YAML 配置路径",
)
cli.add_argument(
    "--eval-config",
    default=os.environ.get("MBAN_EVAL_CONFIG"),
    help="硬件评估配置路径",
)
cli.add_argument(
    "--mode",
    choices=("software", "qat"),
    default=os.environ.get("MBAN_MODE"),
    help="训练模式；覆盖 config.run.mode。PTQ 使用 evaluate_mban.py --mode ptq",
)
cli.add_argument(
    "--profile",
    default=os.environ.get("MBAN_NONIDEAL_PROFILE"),
    help="硬件压力 profile；只表示 ideal/common，覆盖 active_profile",
)
cli.add_argument(
    "--inter-layer",
    choices=("none", "analog", "digital"),
    default=os.environ.get("MBAN_INTER_LAYER"),
    help="层间数据流；独立于硬件压力 profile",
)
cli.add_argument(
    "--set",
    action="append",
    default=[],
    metavar="KEY=VALUE",
    help="命令行覆盖配置字段，可多次使用，例如 --set normalization=l1",
)
args: SimpleNamespace | None = None
CONFIG_PATH = Path(os.environ.get("MBAN_CONFIG", DEFAULT_CONFIG)).resolve()
HARDWARE_CONFIG_PATH: Path | None = None
NONIDEAL_PROFILE_NAME = "ideal"

# =====================================================================
# 路径 / 设备常量
# =====================================================================
LOG_PATH = ""
LATEST_CHECKPOINT_PATH = ""
BEST_VAL_REG_WEIGHT_PATH = ""
BEST_VAL_WEIGHT_PATH = ""
BEST_TRAIN_WEIGHT_PATH = ""
NUM_WORKERS = 0
_NON_SEMANTIC_FIELDS = {
    "epochs",
    "learning_rate",
    "scheduler",
    "scheduler_step_epochs",
    "scheduler_gamma",
    "scheduler_milestones",
    "scheduler_patience",
    "cosine_restart_epochs",
    "onecycle_pct_start",
    "gradient_clip_norm",
    "seed",
    "optimization_batch_pixels",
    "train_pixels_per_image",
    "images_per_batch",
    "loader_workers",
    "resume",
    "initial_checkpoint",
    "h5_files",
    "train_ratio",
    "validation_ratio",
    "test_ratio",
    "split_file",
    "make_new_split",
    "use_train_as_validation",
    "output_directory",
    "save_images_every_epochs",
    "visualization_acquisition",
}
SEMANTIC_CONFIG: dict[str, object] = {}
SEMANTIC_HASH = ""


def _apply_runtime(
    runtime_args: SimpleNamespace,
    config_path: Path,
    profile_name: str | None = None,
    hardware_config_path: Path | None = None,
) -> None:
    global args, CONFIG_PATH, NONIDEAL_PROFILE_NAME
    global HARDWARE_CONFIG_PATH
    global LOG_PATH, LATEST_CHECKPOINT_PATH, BEST_VAL_REG_WEIGHT_PATH
    global BEST_VAL_WEIGHT_PATH, BEST_TRAIN_WEIGHT_PATH, NUM_WORKERS
    global SEMANTIC_CONFIG, SEMANTIC_HASH
    args = runtime_args
    config_path = Path(config_path).resolve()
    hardware_config_path = Path(hardware_config_path).resolve() if hardware_config_path is not None else None
    CONFIG_PATH = config_path
    if hardware_config_path is not None:
        HARDWARE_CONFIG_PATH = hardware_config_path
    else:
        sibling = CONFIG_PATH.with_name("hardware_eval.yaml")
        HARDWARE_CONFIG_PATH = sibling if sibling.is_file() else None
    NONIDEAL_PROFILE_NAME = profile_name or getattr(runtime_args, "_active_profile_name", "ideal")
    os.makedirs(args.output_directory, exist_ok=True)
    LOG_PATH = os.path.join(args.output_directory, "training_log.txt")
    LATEST_CHECKPOINT_PATH = os.path.join(args.output_directory, "latest.pth")
    BEST_VAL_REG_WEIGHT_PATH = os.path.join(args.output_directory, "best_val_reg.pth")
    BEST_VAL_WEIGHT_PATH = os.path.join(args.output_directory, "best_val.pth")
    BEST_TRAIN_WEIGHT_PATH = os.path.join(args.output_directory, "best_train.pth")
    NUM_WORKERS = min(multiprocessing.cpu_count(), 4) if args.loader_workers == -1 else args.loader_workers
    SEMANTIC_CONFIG = {
        "model_semantics_version": MODEL_SEMANTICS_VERSION,
        "config": {key: getattr(args, key) for key in sorted(CONFIG_FIELDS - _NON_SEMANTIC_FIELDS)},
    }
    SEMANTIC_HASH = hashlib.sha256(
        json.dumps(SEMANTIC_CONFIG, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    set_runtime_args(args)
    _sync_runtime_state()


def initialize_runtime(cli_args: argparse.Namespace) -> SimpleNamespace:
    """仅在真正启动训练时解析配置并创建输出目录。"""
    config_path = Path(cli_args.config).resolve()
    overrides = [*(cli_args.set or [])]
    if getattr(cli_args, "mode", None):
        overrides.append(f"mode={cli_args.mode}")
    if getattr(cli_args, "inter_layer", None):
        overrides.append(f"inter_layer={cli_args.inter_layer}")
    eval_config = getattr(cli_args, "eval_config", None)
    hardware_config_path = Path(eval_config).resolve() if eval_config else None
    runtime_args = load_config(
        config_path,
        overrides=overrides,
        profile_name=cli_args.profile,
        hardware_config_path=hardware_config_path,
    )
    _apply_runtime(runtime_args, config_path, cli_args.profile, hardware_config_path)
    return runtime_args


def configure_runtime(
    runtime_args: SimpleNamespace,
    config_path: Path | None = None,
    profile_name: str | None = None,
    hardware_config_path: Path | None = None,
) -> None:
    """Replace the active namespace when an evaluator switches runtime."""
    _apply_runtime(runtime_args, Path(config_path or CONFIG_PATH), profile_name, hardware_config_path)


def _sync_runtime_state() -> None:
    if args is None:
        return
    runtime.set_state(
        args,
        CONFIG_PATH,
        HARDWARE_CONFIG_PATH,
        NONIDEAL_PROFILE_NAME,
        args.output_directory,
        NUM_WORKERS,
        SEMANTIC_CONFIG,
        SEMANTIC_HASH,
    )


# =====================================================================
# 工具
# =====================================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def write_log(message: str, level: str = "INFO") -> None:
    runtime.write_log(message, level=level)


__all__ = (
    "MBAN",
    "QATConfig",
    "MODEL_SEMANTICS_VERSION",
    "args",
    "runtime",
    "configure_runtime",
    "extract_windows_on_gpu",
    "load_config",
    "load_nonideal_profile",
    "load_or_create_mixed_split",
    "train",
    "beamform_iq_with_tx_phase",
    "effective_aperture_weights",
    "compute_aperture_mask",
    "derive_network_channels",
    "normalize_input_iq",
    "pack_dynamic_input",
    "predict_aperture_weights",
    "predict_weights",
    "split_complex_weights",
)


def main() -> None:
    initialize_runtime(cli.parse_args())
    if (
        args.mode == "qat"
        and NONIDEAL_PROFILE_NAME != "ideal"
        and (args.resume != "none" or not args.initial_checkpoint)
    ):
        raise ValueError("HWA-QAT必须使用 resume=none，并显式提供同一语义版本的 initial_checkpoint")
    set_seed(args.seed)
    train()


if __name__ == "__main__":
    main()

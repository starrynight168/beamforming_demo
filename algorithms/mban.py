"""Run MBAN inside the root beamforming comparison pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT))

from algorithms.common import (
    add_common_arguments,
    add_io_arguments,
    parse_selected_angles,
    positive_int,
    resolve_project_path,
    save_algorithm_result,
    validate_db_output,
)

TRAIN_DIR = ROOT / "train"

METHOD_NAME = "mban"


def load_evaluation_dependencies(train_root=TRAIN_DIR):
    """Load the evaluation stack after command-line parsing."""
    train_root = Path(train_root).resolve()
    required = ("eval_core/inference.py", "eval_core/models.py", "shared_core/runtime.py")
    if train_root.name != "train" or any(not (train_root / name).is_file() for name in required):
        raise FileNotFoundError(
            f"MBAN requires the training evaluation package at {train_root}; "
            "use --train_root to select an existing train directory"
        )
    sys.path.insert(0, str(train_root.parent))
    from train.eval_core.inference import load_scene, reconstruct
    from train.eval_core.models import load_evaluation_model

    return load_scene, reconstruct, load_evaluation_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MBAN in the root beamforming comparison"
    )
    add_common_arguments(parser)
    add_io_arguments(parser)
    parser.set_defaults(signal_domain=None)
    parser.set_defaults(interp="linear")
    parser.add_argument("--train_root", type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--kind", required=True, choices=("fp32", "ptq", "qat"))
    parser.add_argument("--config", type=Path, help="Runtime config; defaults to the checkpoint effective_config.json when available")
    parser.add_argument("--eval-config", type=Path)
    parser.add_argument("--profile", default="ideal")
    parser.add_argument("--micro_batch_size", type=positive_int, default=8192)
    parser.add_argument(
        "--split-file",
        type=Path,
        help="PTQ 校准使用的训练划分文件（--kind ptq 时必需）",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.window != "rect" or args.interp != "linear":
        parser.error("MBAN uses learned receive weights and linear IQ sampling; require --window rect --interp linear")
    args.checkpoint = Path(resolve_project_path(args.checkpoint, ROOT)).resolve()
    if args.train_root is not None:
        args.train_root = Path(resolve_project_path(args.train_root, ROOT)).resolve()
    else:
        required = ("eval_core/inference.py", "eval_core/models.py", "shared_core/runtime.py")
        candidates = [*args.checkpoint.parents, TRAIN_DIR]
        args.train_root = next(
            (folder for folder in candidates if folder.name == "train"
             and all((folder / item).is_file() for item in required)),
            TRAIN_DIR,
        )
    snapshot = args.checkpoint.parent / "effective_config.json"
    args.config = (
        Path(resolve_project_path(args.config, ROOT)).resolve() if args.config else
        snapshot if snapshot.is_file() and args.eval_config is None else args.train_root / "config.yaml"
    )
    if args.eval_config is not None:
        args.eval_config = Path(resolve_project_path(args.eval_config, ROOT)).resolve()
        if args.config.suffix.lower() == ".json":
            parser.error("effective_config.json restores saved hardware profiles; use a YAML --config with --eval-config")
    elif args.config.suffix.lower() != ".json":
        args.eval_config = args.train_root / "hardware_eval.yaml"
    if args.kind == "ptq" and args.split_file is None:
        parser.error("--kind ptq requires --split-file")
    if args.split_file is not None:
        args.split_file = Path(resolve_project_path(args.split_file, ROOT)).resolve()
    return args


def main() -> None:
    args = parse_args()
    print(f"MBAN runtime config: {args.config}")
    load_scene, reconstruct, load_evaluation_model = load_evaluation_dependencies(args.train_root)
    h5_path = Path(resolve_project_path(args.h5_path, str(ROOT))).resolve()
    output_dir = Path(args.output_dir).resolve() / METHOD_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    spec = load_evaluation_model(
        METHOD_NAME,
        Path(resolve_project_path(args.checkpoint, ROOT)).resolve(),
        args.kind,
        device,
        config_path=args.config.resolve(),
        hardware_config_path=args.eval_config.resolve() if args.eval_config is not None else None,
        profile=args.profile,
    )
    if args.signal_domain is not None and args.signal_domain != spec.signal_domain:
        raise ValueError(f"checkpoint 使用 {spec.signal_domain}，不能按 {args.signal_domain} 推理")
    args.signal_domain = spec.signal_domain
    from algorithms.common import load_from_h5

    sample = load_from_h5(h5_path, args.h5_sample_idx, signal_domain=args.signal_domain)
    angle_indices, _ = parse_selected_angles(sample.angles, args.select_angles)
    has_gt = sample.has_gt
    data = load_scene(h5_path, args.h5_sample_idx, gt_key=None,
                      angle_indices=angle_indices, signal_domain=args.signal_domain)
    spec.runtime.update(
        dynamic_aperture=bool(args.dynamic_aperture),
        f_number=float(args.f_number),
        use_tgc=bool(args.tgc),
        tgc_alpha=float(args.tgc_alpha),
        angle_selection="all",
        angle_reduction="mean",
    )
    if args.kind == "ptq":
        from train.eval_core.calibration import calibrate_ptq_observers

        spec.runtime_args.split_file = args.split_file.resolve()
        calibration_args = argparse.Namespace(
            micro_batch=args.micro_batch_size,
            output=output_dir,
        )
        calibrate_ptq_observers(calibration_args, [spec], device)
    image, _, _, elapsed = reconstruct(
        spec, data, device, args.micro_batch_size, collect_weight_stats=False
    )
    extent = [
        float(data.x_grid[0] * 1000.0),
        float(data.x_grid[-1] * 1000.0),
        float(data.z_grid[-1] * 1000.0),
        float(data.z_grid[0] * 1000.0),
    ]
    image = validate_db_output(image, (data.height, data.width), METHOD_NAME)
    input_data = SimpleNamespace(extent_mm=extent, sample=SimpleNamespace(has_gt=has_gt))
    save_algorithm_result(image, input_data, args, METHOD_NAME, "MBAN", elapsed)
    print(f"Saved {output_dir / f'{METHOD_NAME}.npy'}")


if __name__ == "__main__":
    main()

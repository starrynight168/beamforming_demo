"""Run MBAN inside the root beamforming comparison pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch

from algorithms.common import (
    add_common_arguments,
    parse_selected_angles,
    positive_int,
    resolve_project_path,
    validate_db_output,
)

matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
TRAIN_DIR = ROOT / "train"

METHOD_NAME = "mban"
DEFAULT_CONFIG = TRAIN_DIR / "config.yaml"
DEFAULT_HARDWARE_CONFIG = TRAIN_DIR / "hardware_eval.yaml"


def load_evaluation_dependencies():
    """Load the evaluation stack after command-line parsing."""
    if str(TRAIN_DIR) not in sys.path:
        sys.path.insert(0, str(TRAIN_DIR))
    from eval_core.config import load_evaluation_model
    from eval_core.inference import load_scene, reconstruct

    return load_scene, reconstruct, load_evaluation_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MBAN in the root beamforming comparison")
    add_common_arguments(parser)
    parser.add_argument("--h5_path", default="data/simulation.h5")
    parser.add_argument("--h5_sample_idx", type=int, default=0)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--kind", required=True, choices=("fp32", "ptq", "qat"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--eval-config", type=Path, default=DEFAULT_HARDWARE_CONFIG)
    parser.add_argument("--profile", default="ideal")
    parser.add_argument("--micro_batch_size", type=positive_int, default=8192)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def save_preview(image: np.ndarray, output: Path, extent: list[float], dr: float) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(5, 5), dpi=180)
    handle = axis.imshow(image, cmap="gray", vmin=-dr, vmax=0.0, extent=extent, aspect="equal")
    axis.set_title("MBAN")
    axis.set_xlabel("Lateral (mm)")
    axis.set_ylabel("Depth (mm)")
    fig.colorbar(handle, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    load_scene, reconstruct, load_evaluation_model = load_evaluation_dependencies()
    h5_path = Path(resolve_project_path(args.h5_path, str(ROOT))).resolve()
    output_dir = Path(args.output_dir).resolve() / METHOD_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "r") as handle:
        angle_values = np.asarray(handle["angles"][:], dtype=np.float32)
    angle_indices, _ = parse_selected_angles(angle_values, args.select_angles)
    device = torch.device(args.device)
    data = load_scene(h5_path, args.h5_sample_idx, gt_key=None, angle_indices=angle_indices)
    spec = load_evaluation_model(
        METHOD_NAME,
        Path(args.checkpoint).resolve(),
        args.kind,
        device,
        config_path=args.config.resolve(),
        hardware_config_path=args.eval_config.resolve(),
        profile=args.profile,
    )
    spec.runtime.update(
        dynamic_aperture=bool(args.dynamic_aperture),
        f_number=float(args.f_number),
        use_tgc=bool(args.tgc),
        tgc_alpha=float(args.tgc_alpha),
    )
    image, _, _, _ = reconstruct(spec, data, device, args.micro_batch_size, collect_weight_stats=False)
    extent = [
        float(data.x_grid[0] * 1000.0),
        float(data.x_grid[-1] * 1000.0),
        float(data.z_grid[-1] * 1000.0),
        float(data.z_grid[0] * 1000.0),
    ]
    image = validate_db_output(image, (data.height, data.width), METHOD_NAME)
    np.save(output_dir / f"{METHOD_NAME}.npy", image)
    save_preview(image, output_dir / f"{METHOD_NAME}.png", extent, float(args.dr))
    print(f"Saved {output_dir / f'{METHOD_NAME}.npy'}")


if __name__ == "__main__":
    main()

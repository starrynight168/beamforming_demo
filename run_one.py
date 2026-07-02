import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

try:
    import yaml
except Exception:
    yaml = None


ROOT = Path(__file__).resolve().parent
ALGORITHM_LABELS = {
    "das": "DAS",
    "mv": "MV",
    "esbmv": "ESBMV",
    "cmsaw": "CMSAW",
    "fdmas": "F-DMAS",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run one beamforming comparison scene.")
    parser.add_argument(
        "--scene",
        default="simulation_contrast_speckle",
        help="Scene id in config.yaml (default: simulation_contrast_speckle)",
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--algorithms", default=None, help="Override algorithms, comma-separated")
    parser.add_argument("--output_root", default="results")
    parser.add_argument("--python_exe", default=sys.executable)
    parser.add_argument("--no_evaluate", action="store_true", default=False)
    parser.add_argument("--keep_existing", action="store_true", default=False, help="Reuse existing algorithm npy files")
    return parser.parse_args()


def load_config(path):
    if yaml is None:
        raise RuntimeError("PyYAML is required to read config.yaml. Install with: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_scene(config, scene_id):
    for scene in config.get("scenes", []):
        if scene.get("id") == scene_id:
            return scene
    raise ValueError(f"Scene not found: {scene_id}")


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def flag(cmd, enabled, name, no_name):
    cmd.append(name if enabled else no_name)


def build_algorithm_cmd(args, config, scene, algorithm, scene_dir):
    params = config.get("params", {})
    mv_params = config.get("mv", {})
    esbmv_params = config.get("esbmv", {})
    script = ROOT / "algorithms" / f"{algorithm}.py"
    if not script.exists():
        raise FileNotFoundError(f"Algorithm script not found: {script}")

    cmd = [
        args.python_exe, str(script),
        "--h5_path", str(resolve_path(scene["h5_path"])),
        "--h5_sample_idx", str(scene["sample_idx"]),
        "--output_dir", str(scene_dir),
        "--select_angles", str(params.get("select_angles", "center")),
        "--f_number", str(params.get("f_number", 1.5)),
        "--dr", str(params.get("dr", 60)),
        "--window", str(params.get("window", "rect")),
        "--interp", str(params.get("interp", "cubic")),
    ]
    flag(cmd, bool(params.get("dynamic_aperture", True)), "--dynamic_aperture", "--no_dynamic_aperture")
    flag(cmd, bool(params.get("tgc", True)), "--tgc", "--no_tgc")
    cmd.extend(["--tgc_alpha", str(params.get("tgc_alpha", 0.5))])

    if algorithm in {"mv", "esbmv", "cmsaw"}:
        cmd.extend([
            "--mv_dl", str(mv_params.get("mv_dl", 0.1)),
            "--subarray_ratio", str(mv_params.get("subarray_ratio", 0.25)),
            "--temporal_win", str(mv_params.get("temporal_win", 9)),
        ])
        flag(cmd, bool(mv_params.get("fbss", True)), "--fbss", "--no_fbss")
    if algorithm == "esbmv":
        cmd.extend([
            "--num_eig", str(esbmv_params.get("num_eig", 0)),
            "--eig_threshold", str(esbmv_params.get("eig_threshold", 0.05)),
        ])
    return cmd


def load_grid_and_gt(h5_path, sample_idx, has_gt):
    with h5py.File(h5_path, "r") as hf:
        x_grid = hf["x_grid"][:].astype(np.float32)
        z_grid = hf["z_grid"][:].astype(np.float32)
        gt = None
        if has_gt and "all_envdb_norm" in hf:
            gt = hf["all_envdb_norm"][sample_idx].astype(np.float32)
    if gt is not None and gt.ndim == 3:
        gt = gt[0] if gt.shape[0] == 1 else gt[:, :, 0]
    extent_mm = [x_grid[0] * 1000, x_grid[-1] * 1000, z_grid[-1] * 1000, z_grid[0] * 1000]
    return extent_mm, gt


def gt_norm_to_db(gt, dr):
    return np.clip(gt, 0.0, 1.0) * dr - dr


def add_scale_bar(ax, extent_mm):
    bar_length = 5.0
    bar_x = extent_mm[1] - bar_length - 2.0
    bar_y = extent_mm[2] - 2.0
    ax.add_patch(patches.Rectangle((bar_x, bar_y), bar_length, 0.5, color="white", zorder=5))
    ax.text(bar_x + bar_length / 2, bar_y - 1.0, "5 mm",
            color="white", fontsize=9, ha="center", va="bottom", fontweight="bold")


def save_comparison(images, titles, extent_mm, output_path, dr):
    fig = plt.figure(figsize=(4.0 * len(images), 4.6), dpi=300)
    gs = fig.add_gridspec(1, len(images) + 1, width_ratios=[1] * len(images) + [0.05])
    im = None
    for idx, (image, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(gs[0, idx])
        im = ax.imshow(image, cmap="gray", vmin=-dr, vmax=0, extent=extent_mm, aspect="equal")
        ax.set_title(title, fontsize=12, pad=8, fontweight="bold")
        ax.set_xlabel("Lateral (mm)")
        if idx == 0:
            ax.set_ylabel("Depth (mm)")
        else:
            ax.set_yticklabels([])
        add_scale_bar(ax, extent_mm)
    cax = fig.add_subplot(gs[0, len(images)])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Amplitude (dB)")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_evaluation(args, config, scene, scene_dir, comparison_path, methods):
    params = config.get("params", {})
    cmd = [
        args.python_exe, str(ROOT / "evaluation" / "evaluate.py"),
        "--comparison_npy", str(comparison_path),
        "--h5_path", str(resolve_path(scene["h5_path"])),
        "--h5_sample_idx", str(scene["sample_idx"]),
        "--methods", ",".join(methods),
        "--dr", str(params.get("dr", 60)),
        "--out_dir", str(scene_dir / "metrics"),
        "--phantom_mode", scene.get("phantom_mode", "auto"),
        "--phantom_source", scene.get("phantom_source", "auto"),
        "--has_gt", "true" if scene.get("has_gt", True) else "false",
    ]
    result = subprocess.run(cmd, cwd=ROOT, text=True)
    if result.returncode != 0:
        raise RuntimeError("Evaluation failed")


def main():
    args = parse_args()
    config = load_config(resolve_path(args.config))
    scene = find_scene(config, args.scene)
    methods = [m.strip().lower() for m in (args.algorithms or ",".join(config["algorithms"])).split(",") if m.strip()]
    scene_dir = resolve_path(args.output_root) / scene["id"]
    scene_dir.mkdir(parents=True, exist_ok=True)

    start_all = time.time()
    for method in methods:
        expected = scene_dir / method / f"{method}.npy"
        if args.keep_existing and expected.exists():
            print(f"Reuse {expected}")
            continue
        cmd = build_algorithm_cmd(args, config, scene, method, scene_dir)
        print("\n" + "=" * 70)
        print(f"Running {method}: {' '.join(cmd)}")
        print("=" * 70)
        result = subprocess.run(cmd, cwd=ROOT, text=True)
        if result.returncode != 0 or not expected.exists():
            raise RuntimeError(f"Algorithm failed or output missing: {method}")

    params = config.get("params", {})
    dr = float(params.get("dr", 60))
    has_gt = bool(scene.get("has_gt", True))
    extent_mm, gt = load_grid_and_gt(resolve_path(scene["h5_path"]), int(scene["sample_idx"]), has_gt)

    images = []
    titles = []
    if has_gt and gt is not None:
        images.append(gt_norm_to_db(gt, dr))
        titles.append("Ground Truth")
    for method in methods:
        path = scene_dir / method / f"{method}.npy"
        images.append(np.load(path).astype(np.float32))
        titles.append(ALGORITHM_LABELS.get(method, method.upper()))

    comparison = np.stack(images, axis=0)
    comparison_path = scene_dir / "comparison.npy"
    np.save(comparison_path, comparison)
    save_comparison(images, titles, extent_mm, scene_dir / "comparison.png", dr)

    run_params = {
        "scene": scene,
        "algorithms": methods,
        "params": params,
        "mv": config.get("mv", {}),
        "esbmv": config.get("esbmv", {}),
        "runtime_sec": time.time() - start_all,
        "output": {"methods": ",".join(methods)},
    }
    with open(scene_dir / "run_params.json", "w", encoding="utf-8") as f:
        json.dump(run_params, f, ensure_ascii=False, indent=2)

    if not args.no_evaluate:
        run_evaluation(args, config, scene, scene_dir, comparison_path, methods)

    print(f"\nDone: {scene_dir}")


if __name__ == "__main__":
    main()

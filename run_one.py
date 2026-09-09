"""Provide Python utilities for run_one."""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

ROOT = Path(__file__).resolve().parent
ALGORITHMS_DIR = ROOT / "algorithms"
if str(ALGORITHMS_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHMS_DIR))
from common_params import COMMON_PARAMS  # noqa: E402

COMPARISON_VALUE_3 = 3
MAX_COMPARISON_IMAGES_PER_PAGE = 8
EVALUATION_SCENE_IDS = frozenset(
    {
        "simulation_contrast_speckle",
        "simulation_resolution_distorsion",
        "experiments_contrast_speckle",
        "experiments_resolution_distorsion",
    }
)

try:
    import yaml
except ImportError:
    yaml = None


def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser(
        description="Run one beamforming comparison scene.",
    )
    parser.add_argument(
        "--scene",
        default="simulation_contrast_speckle",
        help="Scene id in config.yaml (default: simulation_contrast_speckle)",
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--algorithms",
        default=None,
        help="Override algorithms, comma-separated",
    )
    parser.add_argument("--output_root", default="results")
    parser.add_argument("--python_exe", default=sys.executable)
    parser.add_argument("--no_evaluate", action="store_true", default=False)
    parser.add_argument(
        "--no_individual_images",
        action="store_true",
        default=False,
        help="不生成场景目录下的 individual_images 单图目录",
    )
    parser.add_argument(
        "--keep_existing",
        action="store_true",
        default=False,
        help="Reuse existing algorithm npy files",
    )
    args, extra_algorithm_args = parser.parse_known_args()
    args.extra_algorithm_args = extra_algorithm_args
    return args


def load_config(path):
    """Load config."""
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to read config.yaml. Install with: pip install pyyaml",
        )
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def find_scene(config, scene_id):
    """Execute find scene."""
    scene_ids(config)
    for scene in config["scenes"]:
        if scene.get("id") == scene_id:
            return scene
    raise ValueError(f"Scene not found: {scene_id}")


def scene_ids(config):
    """Return validated scene ids."""
    scenes = config.get("scenes")
    if not isinstance(scenes, list) or not all(isinstance(scene, dict) for scene in scenes):
        raise ValueError("config.scenes 必须是场景字典列表")
    ids = [scene.get("id") for scene in scenes]
    if (
        not ids
        or any(not isinstance(value, str) or not value.strip() for value in ids)
        or len(ids) != len(set(ids))
    ):
        raise ValueError("config.scenes 必须包含非空且不重复的 id")
    return ids


def resolve_path(path):
    """Execute resolve path."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def flag(cmd, enabled, name, no_name):
    """Execute flag."""
    cmd.append(name if enabled else no_name)


def value_to_cli_args(name, value):
    """Execute value to cli args."""
    if value is None:
        return []
    option = f"--{name}"
    if isinstance(value, bool):
        return [option] if value else [f"--no_{name}"]
    if isinstance(value, (list, tuple)):
        return [option, ",".join(str(item) for item in value)]
    return [option, str(value)]


def algorithm_params(config, algorithm):
    """Execute algorithm params."""
    all_params = config.get("algorithm_params", {}) or {}
    return all_params.get(algorithm, {}) or {}


def common_params(config):
    """Execute common params."""
    params = COMMON_PARAMS.copy()
    params.update(config.get("params", {}) or {})
    params["select_angles"] = str(
        params.get("select_angles", COMMON_PARAMS["select_angles"]),
    )
    return params


def algorithm_label(config, algorithm):
    """Execute algorithm label."""
    labels = config.get("algorithm_labels", {}) or {}
    return labels.get(algorithm, algorithm.upper())


def build_algorithm_cmd(args, config, scene, algorithm, scene_dir):
    """Build algorithm cmd."""
    params = common_params(config)
    method_params = algorithm_params(config, algorithm)
    script = ROOT / "algorithms" / f"{algorithm}.py"
    if not script.exists():
        raise FileNotFoundError(f"Algorithm script not found: {script}")

    cmd = [
        args.python_exe,
        str(script),
        "--h5_path",
        str(resolve_path(scene["h5_path"])),
        "--h5_sample_idx",
        str(scene["sample_idx"]),
        "--output_dir",
        str(scene_dir),
        "--select_angles",
        str(params["select_angles"]),
        "--f_number",
        str(params["f_number"]),
        "--dr",
        str(params["dr"]),
        "--window",
        str(params["window"]),
        "--interp",
        str(params["interp"]),
    ]
    flag(
        cmd,
        bool(params["dynamic_aperture"]),
        "--dynamic_aperture",
        "--no_dynamic_aperture",
    )
    flag(cmd, bool(params["tgc"]), "--tgc", "--no_tgc")
    cmd.extend(["--tgc_alpha", str(params["tgc_alpha"])])
    for name, value in method_params.items():
        cmd.extend(value_to_cli_args(name, value))
    cmd.extend(args.extra_algorithm_args)
    return cmd


def load_grid_and_gt(h5_path, sample_idx, has_gt):
    """Load grid and gt."""
    with h5py.File(h5_path, "r") as hf:
        x_grid = hf["x_grid"][:].astype(np.float32)
        z_grid = hf["z_grid"][:].astype(np.float32)
        gt = None
        if has_gt and "all_envdb_norm" in hf:
            gt = hf["all_envdb_norm"][sample_idx].astype(np.float32)
    for name, grid in (("x_grid", x_grid), ("z_grid", z_grid)):
        if grid.ndim != 1 or grid.size < 2 or not np.isfinite(grid).all() or not np.all(np.diff(grid) > 0):
            raise ValueError(f"{name} 必须是至少含两个点的有限严格递增一维网格")
    if gt is not None and gt.ndim == COMPARISON_VALUE_3:
        if gt.shape[0] != 1:
            raise ValueError(f"GT 三维形状必须为 [1,H,W]，实际 {gt.shape}")
        gt = gt[0]
    if gt is not None and (gt.shape != (z_grid.size, x_grid.size) or not np.isfinite(gt).all()):
        raise ValueError(f"GT 形状或数值非法: {gt.shape}")
    extent_mm = [
        x_grid[0] * 1000,
        x_grid[-1] * 1000,
        z_grid[-1] * 1000,
        z_grid[0] * 1000,
    ]
    return extent_mm, gt


def scene_has_gt(scene):
    """Execute scene has gt."""
    h5_path = resolve_path(scene["h5_path"])
    sample_idx = int(scene["sample_idx"])
    with h5py.File(h5_path, "r") as hf:
        available = "all_envdb_norm" in hf and 0 <= sample_idx < hf["all_envdb_norm"].shape[0]
    return available


def scene_is_non_controlled(scene):
    """Return whether a scene is outside the four controlled metric scenes."""
    return str(scene.get("id", "")) not in EVALUATION_SCENE_IDS


def gt_norm_to_db(gt, dr):
    """Execute gt norm to db."""
    return np.clip(gt, 0.0, 1.0) * dr - dr


def add_scale_bar(ax, extent_mm):
    """Execute add scale bar."""
    bar_length = 5.0
    bar_x = extent_mm[1] - bar_length - 2.0
    bar_y = extent_mm[2] - 2.0
    ax.add_patch(
        patches.Rectangle((bar_x, bar_y), bar_length, 0.5, color="white", zorder=5),
    )
    ax.text(
        bar_x + bar_length / 2,
        bar_y - 1.0,
        "5 mm",
        color="white",
        fontsize=9,
        ha="center",
        va="bottom",
        fontweight="bold",
    )


def save_comparison(images, titles, extent_mm, output_path, dr, _cleanup_pages=True):
    """Save comparison."""
    n_images = len(images)
    max_images_per_page = MAX_COMPARISON_IMAGES_PER_PAGE
    if _cleanup_pages:
        for stale in output_path.parent.glob(f"{output_path.stem}_page*{output_path.suffix}"):
            stale.unlink(missing_ok=True)
    if n_images > max_images_per_page:
        for page_idx, start in enumerate(range(0, n_images, max_images_per_page), start=1):
            page_path = output_path if page_idx == 1 else output_path.with_name(
                f"{output_path.stem}_page{page_idx}{output_path.suffix}"
            )
            save_comparison(
                images[start : start + max_images_per_page],
                titles[start : start + max_images_per_page],
                extent_mm,
                page_path,
                dr,
                _cleanup_pages=False,
            )
        return
    cols = 4 if n_images > 4 else min(2, n_images)
    rows = math.ceil(n_images / cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.8 * cols, 4.2 * rows),
        dpi=300,
        squeeze=False,
        constrained_layout=True,
    )
    im = None
    for idx, (image, title) in enumerate(zip(images, titles, strict=True)):
        ax = axes[idx // cols][idx % cols]
        im = ax.imshow(
            image,
            cmap="gray",
            vmin=-dr,
            vmax=0,
            extent=extent_mm,
            aspect="equal",
        )
        ax.set_title(title, fontsize=12, pad=8, fontweight="bold")
        ax.set_xlabel("Lateral (mm)")
        if idx % cols == 0:
            ax.set_ylabel("Depth (mm)")
        else:
            ax.set_yticklabels([])
        add_scale_bar(ax, extent_mm)

    for idx in range(n_images, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Amplitude (dB)")
    plt.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_single_image(image, title, extent_mm, output_path, dr):
    """Save single image."""
    fig, ax = plt.subplots(figsize=(4.2, 4.6), dpi=300, constrained_layout=True)
    im = ax.imshow(
        image,
        cmap="gray",
        vmin=-dr,
        vmax=0,
        extent=extent_mm,
        aspect="equal",
    )
    ax.set_title(title, fontsize=12, pad=8, fontweight="bold")
    ax.set_xlabel("Lateral (mm)")
    ax.set_ylabel("Depth (mm)")
    add_scale_bar(ax, extent_mm)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Amplitude (dB)")
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_individual_images(images, titles, image_names, extent_mm, output_dir, dr):
    """Save individual images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.png"):
        stale.unlink(missing_ok=True)
    for image, title, name in zip(images, titles, image_names, strict=True):
        save_single_image(image, title, extent_mm, output_dir / f"{name}.png", dr)


def can_reuse_existing(scene_dir, config, scene, method, args):
    """Execute can reuse existing."""
    params_path = scene_dir / "run_params.json"
    if not params_path.exists():
        return False, "缺少 run_params.json,无法确认旧结果参数"
    try:
        with open(params_path, encoding="utf-8") as f:
            previous = json.load(f)
    except Exception as exc:
        return False, f"无法读取旧 run_params.json: {exc}"

    expected = {
        "scene": scene,
        "params": common_params(config),
        "method_params": algorithm_params(config, method),
        "extra_algorithm_args": args.extra_algorithm_args,
    }
    actual = {
        "scene": previous.get("scene"),
        "params": previous.get("params", {}) or {},
        "method_params": (previous.get("algorithm_params", {}) or {}).get(method, {}) or {},
        "extra_algorithm_args": previous.get("extra_algorithm_args", []),
    }
    if actual != expected:
        return False, "旧结果参数与当前配置不一致"
    return True, "参数一致"


def run_evaluation(args, config, scene, scene_dir, comparison_path, methods):
    """Execute run evaluation."""
    params = common_params(config)
    labels = [algorithm_label(config, method) for method in methods]
    cmd = [
        args.python_exe,
        str(ROOT / "evaluation" / "evaluate.py"),
        "--comparison_npy",
        str(comparison_path),
        "--h5_path",
        str(resolve_path(scene["h5_path"])),
        "--h5_sample_idx",
        str(scene["sample_idx"]),
        "--methods",
        ",".join(methods),
        "--method_labels",
        ",".join(labels),
        "--dr",
        str(params["dr"]),
        "--out_dir",
        str(scene_dir / "metrics"),
        "--phantom_mode",
        "auto",
        "--phantom_source",
        "auto",
    ]
    result = subprocess.run(cmd, cwd=ROOT, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError("Evaluation failed")
    metrics_dir = scene_dir / "metrics"
    if (metrics_dir / "evaluation_meta.json").exists():
        plot_cmd = [
            args.python_exe,
            str(ROOT / "evaluation" / "plot_metrics.py"),
            "--metrics_dir",
            str(metrics_dir),
            "--dr",
            str(params["dr"]),
            "--reference-mode",
            "algorithms",
        ]
        result = subprocess.run(plot_cmd, cwd=ROOT, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError("Metric plotting failed")


def main():
    """Run the command-line workflow."""
    args = parse_args()
    config = load_config(resolve_path(args.config))
    scene = find_scene(config, args.scene)
    methods = [m.strip().lower() for m in (args.algorithms or ",".join(config["algorithms"])).split(",") if m.strip()]
    if not methods:
        raise ValueError("至少需要选择一个算法")
    if len(methods) != len(set(methods)):
        raise ValueError(f"算法列表包含重复项: {methods}")
    scene_dir = resolve_path(args.output_root) / scene["id"]
    scene_dir.mkdir(parents=True, exist_ok=True)
    start_all = time.time()
    for method in methods:
        expected = scene_dir / method / f"{method}.npy"
        if args.keep_existing and expected.exists():
            reusable, reason = can_reuse_existing(
                scene_dir,
                config,
                scene,
                method,
                args,
            )
            if reusable:
                print(f"Reuse {expected}")
                continue
            print(f"Ignore existing {expected}: {reason}")
        cmd = build_algorithm_cmd(args, config, scene, method, scene_dir)
        print("\n" + "=" * 70)
        print(f"Running {method}: {' '.join(cmd)}")
        print("=" * 70)
        result = subprocess.run(cmd, cwd=ROOT, text=True, check=False)
        if result.returncode != 0 or not expected.exists():
            raise RuntimeError(f"Algorithm failed or output missing: {method}")

    params = common_params(config)
    dr = float(params["dr"])
    has_gt = scene_has_gt(scene)
    extent_mm, gt = load_grid_and_gt(
        resolve_path(scene["h5_path"]),
        int(scene["sample_idx"]),
        has_gt,
    )
    with h5py.File(resolve_path(scene["h5_path"]), "r") as hf:
        expected_shape = (int(hf["z_grid"].size), int(hf["x_grid"].size))

    images = []
    titles = []
    image_names = []
    if has_gt and gt is not None:
        images.append(gt_norm_to_db(gt, dr))
        titles.append("Ground Truth")
        image_names.append("ground_truth")
    for method in methods:
        path = scene_dir / method / f"{method}.npy"
        image = np.load(path).astype(np.float32)
        if image.shape != expected_shape or not np.all(np.isfinite(image)):
            raise ValueError(f"算法 {method} 输出必须是有限二维数组: {image.shape}")
        if not np.isclose(float(np.max(image)), 0.0, rtol=0.0, atol=1.0e-3):
            raise ValueError(f"算法 {method} 输出未归一化到 0 dB")
        if not np.any(image < -1.0e-3):
            raise ValueError(f"算法 {method} 输出没有可测动态范围")
        images.append(image)
        titles.append(algorithm_label(config, method))
        image_names.append(method)

    comparison = np.stack(images, axis=0)
    comparison_path = scene_dir / "comparison.npy"
    np.save(comparison_path, comparison)
    save_comparison(images, titles, extent_mm, scene_dir / "comparison.png", dr)
    individual_dir = None
    if not args.no_individual_images:
        individual_dir = scene_dir / "individual_images"
        save_individual_images(
            images,
            titles,
            image_names,
            extent_mm,
            individual_dir,
            dr,
        )

    run_params = {
        "scene": scene,
        "algorithms": methods,
        "algorithm_labels": {method: algorithm_label(config, method) for method in methods},
        "params": params,
        "algorithm_params": {method: algorithm_params(config, method) for method in methods},
        "extra_algorithm_args": args.extra_algorithm_args,
        "runtime_sec": time.time() - start_all,
        "output": {"methods": ",".join(methods)},
    }
    with open(scene_dir / "run_params.json", "w", encoding="utf-8") as f:
        json.dump(run_params, f, ensure_ascii=False, indent=2)

    non_controlled = scene_is_non_controlled(scene)
    if not args.no_evaluate and has_gt and not non_controlled:
        run_evaluation(args, config, scene, scene_dir, comparison_path, methods)
    elif not args.no_evaluate and non_controlled:
        print("Skip evaluation: non-controlled scenes only generate images.")
    elif not args.no_evaluate:
        print("Skip evaluation: H5 没有 all_envdb_norm ground truth.")

    print(f"\nDone: {scene_dir}")
    if individual_dir is not None:
        print(f"Individual images: {individual_dir}")


if __name__ == "__main__":
    main()

import argparse
import itertools
import math
import sys
import tempfile
import time
from datetime import datetime
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "train"
sys.path.insert(0, str(TRAIN))
sys.path.insert(0, str(ROOT))

from train.eval_core.inference import load_scene, reconstruct
from train.eval_core.models import load_evaluation_model
from tools.shared.workflow_common import TOOLS_OUTPUT_ROOT, inspect_h5
from tools.shared.run_output import (
    emit,
    prepare_output_tree,
    resolve_output_dir,
    save_figure,
    set_plot_style,
    write_csv,
    write_json,
)


DEFAULT_DATA = ROOT / "data" / "volunteer_005.h5"
DEFAULT_CONFIGS = {
    "fp32": TRAIN
    / "results"
    / "MAIN"
    / "configs"
    / "figure_experiments"
    / "main_fp32_roi05.yaml",
    "qat": TRAIN
    / "results"
    / "MAIN"
    / "configs"
    / "figure_experiments"
    / "main_qat_roi05.yaml",
}
DEFAULT_CHECKPOINTS = {
    "fp32": TRAIN / "results" / "MAIN" / "model" / "FP32" / "roi05" / "best_val.pth",
    "qat": TRAIN / "results" / "MAIN" / "model" / "QAT" / "roi05" / "best_val.pth",
}
DEFAULT_OUTPUT = TOOLS_OUTPUT_ROOT / "rebeamform_invivo_grid"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在多个横向/深度网格上重新重建 H5 样本的 MBAN B-mode 图像。"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--frame", type=int, default=300, help="H5 帧号，从 1 开始。")
    parser.add_argument("--model-name", default="roi05")
    parser.add_argument("--model-type", choices=("fp32", "qat"), default="fp32")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--x-range-mm", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--z-range-mm", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument(
        "--widths-mm",
        type=parse_float_list,
        help="逗号分隔的横向视野宽度；与深度长度列表做组合扫描。",
    )
    parser.add_argument(
        "--depth-lengths-mm",
        type=parse_float_list,
        help="逗号分隔的深度长度；与横向宽度列表做组合扫描。",
    )
    parser.add_argument(
        "--x-center-mm", type=float, help="横向宽度的中心；默认使用原网格中心。"
    )
    parser.add_argument(
        "--depth-start-mm", type=float, help="深度长度的起点；默认使用原网格起点。"
    )
    parser.add_argument(
        "--width-pixels",
        type=parse_int_list,
        help="逗号分隔的横向像素数；省略时保持原横向采样间距。",
    )
    parser.add_argument(
        "--depth-pixels",
        type=parse_int_list,
        help="逗号分隔的深度像素数；省略时保持原深度采样间距。",
    )
    parser.add_argument("--micro-batch", type=int, default=8192)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    try:
        values = [float(value.strip()) for value in text.split(",") if value.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("请输入逗号分隔的数字") from error
    if not values or not np.isfinite(values).all():
        raise argparse.ArgumentTypeError("列表不能为空，且所有值必须为有限数")
    return values


def parse_int_list(text: str) -> list[int]:
    try:
        values = [int(value.strip()) for value in text.split(",") if value.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("请输入逗号分隔的整数") from error
    if not values or any(value < 2 for value in values):
        raise argparse.ArgumentTypeError("像素数列表不能为空，且每个值至少为 2")
    return values


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def compatible_config(config_path: Path) -> tuple[Path, Path | None]:
    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        return config_path, None
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"配置文件不是 YAML 字典：{config_path}")
    data_config = config.get("data")
    if (
        not isinstance(data_config, dict)
        or data_config.pop("roi_subset_file", None) is None
    ):
        return config_path, None
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".yaml", dir=config_path.parent, delete=False
    ) as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
        temporary_path = Path(handle.name)
    return temporary_path, temporary_path


def make_grid(
    native_grid: np.ndarray, requested_range: list[float] | None, count: int | None
) -> np.ndarray:
    low_mm, high_mm = (
        (float(native_grid[0] * 1000.0), float(native_grid[-1] * 1000.0))
        if requested_range is None
        else (float(requested_range[0]), float(requested_range[1]))
    )
    if not np.isfinite((low_mm, high_mm)).all() or high_mm <= low_mm:
        raise ValueError("网格范围必须是有限值且 MAX 大于 MIN")
    if count is None:
        native_spacing_mm = float(np.median(np.diff(native_grid)) * 1000.0)
        count = int(round((high_mm - low_mm) / native_spacing_mm)) + 1
    if count < 2:
        raise ValueError("网格像素数至少为 2")
    return np.linspace(low_mm * 1e-3, high_mm * 1e-3, count, dtype=np.float32)


def make_ranges(
    scene, args: argparse.Namespace
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    native_x = scene.x_grid.cpu().numpy() * 1000.0
    native_z = scene.z_grid.cpu().numpy() * 1000.0
    if args.x_range_mm is not None and args.widths_mm is not None:
        raise ValueError("--x-range-mm 与 --widths-mm 只能选一个")
    if args.z_range_mm is not None and args.depth_lengths_mm is not None:
        raise ValueError("--z-range-mm 与 --depth-lengths-mm 只能选一个")
    if args.x_range_mm is not None and args.x_center_mm is not None:
        raise ValueError("--x-center-mm 不能与 --x-range-mm 同时使用")
    if args.z_range_mm is not None and args.depth_start_mm is not None:
        raise ValueError("--depth-start-mm 不能与 --z-range-mm 同时使用")

    native_center = float((native_x[0] + native_x[-1]) / 2.0)
    native_width = float(native_x[-1] - native_x[0])
    x_center = native_center if args.x_center_mm is None else args.x_center_mm
    if args.x_range_mm is not None:
        x_ranges = [tuple(args.x_range_mm)]
    elif args.widths_mm is not None:
        if any(width <= 0 for width in args.widths_mm) or len(
            set(args.widths_mm)
        ) != len(args.widths_mm):
            raise ValueError("--widths-mm 必须是互不重复的正数")
        x_ranges = [
            (x_center - width / 2.0, x_center + width / 2.0) for width in args.widths_mm
        ]
    elif args.x_center_mm is not None:
        x_ranges = [(x_center - native_width / 2.0, x_center + native_width / 2.0)]
    else:
        x_ranges = [(float(native_x[0]), float(native_x[-1]))]

    native_depth = float(native_z[-1] - native_z[0])
    depth_start = (
        float(native_z[0]) if args.depth_start_mm is None else args.depth_start_mm
    )
    if args.z_range_mm is not None:
        z_ranges = [tuple(args.z_range_mm)]
    elif args.depth_lengths_mm is not None:
        if any(length <= 0 for length in args.depth_lengths_mm) or len(
            set(args.depth_lengths_mm)
        ) != len(args.depth_lengths_mm):
            raise ValueError("--depth-lengths-mm 必须是互不重复的正数")
        z_ranges = [
            (depth_start, depth_start + length) for length in args.depth_lengths_mm
        ]
    elif args.depth_start_mm is not None:
        z_ranges = [(depth_start, depth_start + native_depth)]
    else:
        z_ranges = [(float(native_z[0]), float(native_z[-1]))]
    return x_ranges, z_ranges


def ticks_with_limits(low: float, high: float, max_labels: int = 7) -> np.ndarray:
    span = high - low
    if not np.isfinite(span) or span <= 0:
        raise ValueError("刻度范围必须是有限正数")
    target_step = span / max(max_labels - 1, 2)
    exponent = math.floor(math.log10(target_step))
    scale = 10.0**exponent
    step = next(
        multiplier * scale
        for multiplier in (1, 2, 2.5, 5, 10)
        if multiplier * scale >= target_step
    )
    interior = np.arange(np.ceil(low / step) * step, high, step)
    edge_gap = step * 0.65
    interior = interior[(interior - low >= edge_gap) & (high - interior >= edge_gap)]
    return np.unique(np.round(np.r_[low, interior, high], 10))


def align_boundary_ticklabels(
    axis, x_range: tuple[float, float], z_range: tuple[float, float]
) -> None:
    for label, tick in zip(axis.get_xticklabels(), axis.get_xticks()):
        if np.isclose(tick, x_range[0]):
            label.set_ha("right")
        elif np.isclose(tick, x_range[1]):
            label.set_ha("left")
    for label, tick in zip(axis.get_yticklabels(), axis.get_yticks()):
        if np.isclose(tick, z_range[0]):
            label.set_va("bottom")
        elif np.isclose(tick, z_range[1]):
            label.set_va("top")


def number_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def plot_title(
    source_label: str,
    frame: int,
    x_range: tuple[float, float],
    z_range: tuple[float, float],
) -> str:
    return (
        f"{source_label} | Frame {frame}\n"
        f"Width: {x_range[0]:g} to {x_range[1]:g} mm; "
        f"Depth: {z_range[0]:g} to {z_range[1]:g} mm"
    )


def save_bmode(
    image: np.ndarray,
    x_range: tuple[float, float],
    z_range: tuple[float, float],
    source_label: str,
    frame: int,
    output_dir: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(5.0, 4.0))
    extent = [x_range[0], x_range[1], z_range[1], z_range[0]]
    handle = axis.imshow(
        image,
        extent=extent,
        cmap="gray",
        vmin=-60,
        vmax=0,
        aspect="equal",
        interpolation="nearest",
    )
    axis.set_xlim(x_range)
    axis.set_ylim(z_range[1], z_range[0])
    axis.set_xticks(ticks_with_limits(*x_range))
    axis.set_yticks(ticks_with_limits(*z_range))
    align_boundary_ticklabels(axis, x_range, z_range)
    axis.set_xlabel("Lateral (mm)")
    axis.set_ylabel("Depth (mm)")
    axis.set_title(plot_title(source_label, frame, x_range, z_range), fontsize=8)
    axis.tick_params(labelsize=7, length=2)
    colorbar = fig.colorbar(handle, ax=axis, ticks=(-60, -40, -20, 0), pad=0.04)
    colorbar.set_label("dB")
    fig.tight_layout()
    save_figure(fig, output_dir / "bmode.png", dpi=600, formats=("png", "svg", "pdf"))


def save_grid_comparison(
    results: list[dict], output_dir: Path, source_label: str, frame: int
) -> None:
    cols = min(3, len(results))
    rows = math.ceil(len(results) / cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(3.3 * cols, 3.5 * rows), squeeze=False
    )
    handle = None
    for axis, result in zip(axes.flat, results):
        x_range = result["x_range_mm"]
        z_range = result["z_range_mm"]
        handle = axis.imshow(
            result["image"],
            extent=[x_range[0], x_range[1], z_range[1], z_range[0]],
            cmap="gray",
            vmin=-60,
            vmax=0,
            aspect="equal",
            interpolation="nearest",
        )
        axis.set_xlim(x_range)
        axis.set_ylim(z_range[1], z_range[0])
        axis.set_xticks(ticks_with_limits(*x_range))
        axis.set_yticks(ticks_with_limits(*z_range))
        align_boundary_ticklabels(axis, x_range, z_range)
        axis.set_title(plot_title(source_label, frame, x_range, z_range), fontsize=8)
        axis.set_xlabel("Lateral (mm)")
        if axis is axes.flat[0]:
            axis.set_ylabel("Depth (mm)")
        axis.tick_params(labelsize=7, length=2)
    for axis in list(axes.flat)[len(results) :]:
        axis.set_visible(False)
    fig.colorbar(
        handle, ax=list(axes.flat), ticks=(-60, -40, -20, 0), fraction=0.025, pad=0.025
    ).set_label("dB")
    save_figure(fig, output_dir / "grid_comparison.png", dpi=400)


def main() -> None:
    args = parse_args()
    if args.frame < 1 or args.micro_batch < 1:
        raise ValueError("frame 和 micro-batch 必须为正整数")
    data_path = resolve_path(args.data)
    checkpoint_path = resolve_path(
        args.checkpoint or DEFAULT_CHECKPOINTS[args.model_type]
    )
    config_path = resolve_path(args.config or DEFAULT_CONFIGS[args.model_type])
    if (
        not data_path.is_file()
        or not checkpoint_path.is_file()
        or not config_path.is_file()
    ):
        raise FileNotFoundError("data、checkpoint 或 config 文件不存在")
    h5_info = inspect_h5(data_path)
    if args.frame > h5_info["n"]:
        raise IndexError(f"frame {args.frame} 超出 H5 样本数 {h5_info['n']}")
    source_label = "In-vivo" if h5_info["in_vivo"][args.frame - 1] else "Simulation"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前环境没有可用 CUDA 设备")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )

    runtime_config, temporary_config = compatible_config(config_path)
    try:
        spec = load_evaluation_model(
            args.model_name,
            checkpoint_path,
            args.model_type,
            device,
            config_path=runtime_config,
        )
    finally:
        if temporary_config is not None:
            temporary_config.unlink(missing_ok=True)

    scene = load_scene(
        data_path,
        args.frame - 1,
        gt_key=None,
        signal_domain=spec.signal_domain,
    )
    x_ranges, z_ranges = make_ranges(scene, args)
    width_counts = args.width_pixels or [None]
    depth_counts = args.depth_pixels or [None]
    if len(set(width_counts)) != len(width_counts) or len(set(depth_counts)) != len(
        depth_counts
    ):
        raise ValueError("像素数列表不能包含重复值")
    grids = list(itertools.product(x_ranges, z_ranges, width_counts, depth_counts))
    if len(grids) > 36:
        raise ValueError("一次最多重建 36 个网格组合")

    sample_tag = data_path.stem.replace("volunteer_", "v", 1)
    if args.output_dir is None:
        output_group = (
            "in_vivo_grid" if source_label == "In-vivo" else "simulation_grid"
        )
        run_parent = DEFAULT_OUTPUT / output_group / f"{sample_tag}_f{args.frame}"
        output_root = resolve_output_dir(
            "rebeamform_invivo_grid",
            run_parent / datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        )
    else:
        output_root = resolve_output_dir(
            "rebeamform_invivo_grid", resolve_path(args.output_dir)
        )
    prepare_output_tree(output_root)

    set_plot_style()
    emit(
        output_root,
        "prepare",
        f"重建 {len(grids)} 个网格；样本 {sample_tag}，帧 {args.frame}。",
    )
    write_json(
        output_root / "run_config.json",
        {
            "data": str(data_path),
            "source_type": source_label,
            "frame": args.frame,
            "model_name": args.model_name,
            "model_type": args.model_type,
            "checkpoint": str(checkpoint_path),
            "config": str(config_path),
            "device": str(device),
            "micro_batch": args.micro_batch,
            "x_ranges_mm": x_ranges,
            "z_ranges_mm": z_ranges,
            "width_pixels": width_counts,
            "depth_pixels": depth_counts,
        },
    )
    results = []
    for index, (
        x_range_requested,
        z_range_requested,
        width_count,
        depth_count,
    ) in enumerate(grids, 1):
        x_grid = make_grid(
            scene.x_grid.cpu().numpy(), list(x_range_requested), width_count
        )
        z_grid = make_grid(
            scene.z_grid.cpu().numpy(), list(z_range_requested), depth_count
        )
        output_scene = replace(
            scene, x_grid=torch.from_numpy(x_grid), z_grid=torch.from_numpy(z_grid)
        )
        started = time.perf_counter()
        image_db, _, _, _ = reconstruct(
            spec, output_scene, device, args.micro_batch, collect_weight_stats=False
        )
        elapsed = time.perf_counter() - started
        x_mm = x_grid.astype(np.float64) * 1000.0
        z_mm = z_grid.astype(np.float64) * 1000.0
        x_range = (round(float(x_mm[0]), 4), round(float(x_mm[-1]), 4))
        z_range = (round(float(z_mm[0]), 4), round(float(z_mm[-1]), 4))
        width_mm = round(x_range[1] - x_range[0], 3)
        depth_mm = round(z_range[1] - z_range[0], 3)
        center_mm = round((x_range[0] + x_range[1]) / 2.0, 3)
        folder = (
            f"x{number_token(center_mm)}_w{number_token(width_mm)}"
            f"_z{number_token(z_range[0])}h{number_token(depth_mm)}_{len(x_grid)}x{len(z_grid)}"
        )
        output_dir = output_root / folder
        figure_dir = output_dir / "figures"
        data_dir = output_dir / "data"
        figure_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            data_dir / "reconstruction.npz",
            bmode_db=image_db,
            x_grid_mm=x_mm.astype(np.float32),
            z_grid_mm=z_mm.astype(np.float32),
        )
        run_config = {
            "data": str(data_path),
            "source_type": source_label,
            "frame": args.frame,
            "model_name": args.model_name,
            "model_type": args.model_type,
            "checkpoint": str(checkpoint_path),
            "config": str(config_path),
            "device": str(device),
            "x_range_mm": list(x_range),
            "z_range_mm": list(z_range),
            "width_pixels": int(len(x_grid)),
            "depth_pixels": int(len(z_grid)),
            "dx_mm": round(float(np.diff(x_mm).mean()), 6),
            "dz_mm": round(float(np.diff(z_mm).mean()), 6),
            "micro_batch": args.micro_batch,
            "beamforming_seconds": elapsed,
        }
        write_json(output_dir / "run_config.json", run_config)
        save_bmode(
            image_db,
            x_range,
            z_range,
            source_label,
            args.frame,
            figure_dir,
        )
        results.append(
            {
                "index": index,
                "image": image_db,
                "x_range_mm": x_range,
                "z_range_mm": z_range,
                "width_mm": width_mm,
                "depth_mm": depth_mm,
                "width_pixels": len(x_grid),
                "depth_pixels": len(z_grid),
                "dx_mm": run_config["dx_mm"],
                "dz_mm": run_config["dz_mm"],
                "beamforming_seconds": elapsed,
                "output_dir": str(output_dir),
            }
        )
        emit(
            output_root,
            "progress",
            f"{index}/{len(grids)}：{folder}（{elapsed:.2f} 秒）。",
        )

    summary_path = output_root / "metrics" / "grid_sweep_summary.csv"
    fields = (
        "index",
        "width_mm",
        "depth_mm",
        "x_range_mm",
        "z_range_mm",
        "width_pixels",
        "depth_pixels",
        "dx_mm",
        "dz_mm",
        "beamforming_seconds",
        "output_dir",
    )
    write_csv(
        summary_path,
        ({key: result[key] for key in fields} for result in results),
        fields,
    )
    if len(results) > 1:
        save_grid_comparison(results, output_root / "figures", source_label, args.frame)
    emit(output_root, "done", f"结果目录：{output_root}；汇总表：{summary_path}。")


if __name__ == "__main__":
    main()

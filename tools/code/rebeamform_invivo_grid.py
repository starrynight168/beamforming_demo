import argparse
import itertools
import math
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from algorithms.das import DASBeamformerIQ
from algorithms.shared_core.common import (
    add_common_arguments,
    beamformed_image_db,
    beamforming_depth_grid,
    prepare_beamforming_input,
    run_beamformer,
    validate_db_output,
)
from evaluation.context import read_sample_meta
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


DEFAULT_OUTPUT = TOOLS_OUTPUT_ROOT / "rebeamform_invivo_grid"
DEFAULT_SCENE = "simulation_contrast_speckle"
SCENE_IDS = (
    "simulation_contrast_speckle",
    "simulation_resolution_distortion",
    "experiments_contrast_speckle",
    "experiments_resolution_distortion",
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 DAS 扫描横向视野、深度范围和像素数。"
    )
    add_common_arguments(parser)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--scene", choices=SCENE_IDS, default=DEFAULT_SCENE)
    parser.add_argument("--data", type=Path, help="显式指定 H5 时覆盖 config 场景。")
    parser.add_argument("--frame", type=int, help="H5 帧号，从 1 开始；默认使用 config 的 sample_idx。")
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
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--row_block", type=int, default=24)
    parser.add_argument("--aperture_mode", choices=("discrete", "geometry"), default="discrete")
    return parser.parse_args(argv)


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


def configured_scene(config_path: Path, scene_id: str) -> tuple[Path, int]:
    with resolve_path(config_path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    scenes = config.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError("config.yaml 必须包含 scenes 列表")
    matches = [scene for scene in scenes if isinstance(scene, dict) and scene.get("id") == scene_id]
    if len(matches) != 1:
        raise ValueError(f"config.yaml 中必须恰好包含场景 {scene_id!r}")
    scene = matches[0]
    return resolve_path(Path(scene["h5_path"])), int(scene.get("sample_idx", 0)) + 1


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
    x_grid: np.ndarray, z_grid: np.ndarray, args: argparse.Namespace
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    native_x = numpy_grid(x_grid).astype(np.float64) * 1000.0
    native_z = numpy_grid(z_grid).astype(np.float64) * 1000.0
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


def numpy_grid(grid) -> np.ndarray:
    if isinstance(grid, torch.Tensor):
        grid = grid.detach().cpu().numpy()
    return np.asarray(grid, dtype=np.float32)


def run_das(input_data, x_grid, z_grid, args, device):
    sample = input_data.sample
    c = sample.c
    compute_z = beamforming_depth_grid(z_grid, c, sample.fs, args.signal_domain)
    beamformer = DASBeamformerIQ(
        compute_z, x_grid, sample.n_elem, sample.pitch, c, sample.fc, sample.fs,
        input_data.t0, sample.angles, options=args, device=device,
    )
    i_out, q_out, elapsed = run_beamformer(
        beamformer,
        input_data.i_data,
        input_data.q_data,
        input_data.selected_angles,
        input_data.t0,
        sample.fs,
        device,
    )
    image_db = beamformed_image_db(
        i_out,
        q_out,
        compute_z,
        z_grid,
        sample.fc,
        args.tgc,
        args.tgc_alpha,
        args.signal_domain,
    )
    return validate_db_output(image_db, (len(z_grid), len(x_grid)), "das"), elapsed


def das_config(args):
    return {
        "algorithm": "das",
        "signal_domain": args.signal_domain,
        "select_angles": args.select_angles,
        "f_number": args.f_number,
        "dynamic_aperture": args.dynamic_aperture,
        "window": args.window,
        "interp": args.interp,
        "tgc": args.tgc,
        "tgc_alpha": args.tgc_alpha,
        "row_block": args.row_block,
        "aperture_mode": args.aperture_mode,
    }


def main() -> None:
    args = parse_args()
    if args.data is None:
        data_path, default_frame = configured_scene(args.config, args.scene)
        scene_id = args.scene
    else:
        data_path, default_frame = resolve_path(args.data), 1
        scene_id = None
    args.frame = args.frame if args.frame is not None else default_frame
    if args.frame < 1:
        raise ValueError("frame 必须为正整数")
    if not data_path.is_file():
        raise FileNotFoundError(f"H5 文件不存在：{data_path}")
    h5_info = inspect_h5(data_path)
    if args.frame > h5_info["n"]:
        raise IndexError(f"frame {args.frame} 超出 H5 样本数 {h5_info['n']}")
    sample_meta = read_sample_meta(data_path, args.frame - 1)
    phantom_source = sample_meta["phantom_source"]
    if not phantom_source:
        raise ValueError("所选 H5 样本缺少 phantom_source 元数据")
    source_label = {
        "in_vivo": "In-vivo",
        "simulation": "Simulation",
        "experiments": "Experiments",
    }.get(phantom_source, phantom_source.replace("_", " ").title())
    source_group = phantom_source if phantom_source in {"in_vivo", "simulation", "experiments"} else "other"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前环境没有可用 CUDA 设备")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )

    input_data = prepare_beamforming_input(
        data_path, args.frame - 1, args.select_angles, signal_domain=args.signal_domain
    )
    native_x, native_z = input_data.sample.x_grid, input_data.sample.z_grid
    x_ranges, z_ranges = make_ranges(native_x, native_z, args)
    width_counts = args.width_pixels or [None]
    depth_counts = args.depth_pixels or [None]
    if len(set(width_counts)) != len(width_counts) or len(set(depth_counts)) != len(depth_counts):
        raise ValueError("像素数列表不能包含重复值")
    grids = list(itertools.product(x_ranges, z_ranges, width_counts, depth_counts))
    if len(grids) > 36:
        raise ValueError("一次最多重建 36 个网格组合")

    native_x = numpy_grid(native_x)
    native_z = numpy_grid(native_z)
    sample_tag = data_path.stem.replace("volunteer_", "v", 1)
    if args.output_dir is None:
        output_group = f"{source_group}_grid"
        run_parent = DEFAULT_OUTPUT / output_group / f"{sample_tag}_f{args.frame}"
        output_root = resolve_output_dir(
            "rebeamform_invivo_grid", run_parent / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
    else:
        output_root = resolve_output_dir("rebeamform_invivo_grid", resolve_path(args.output_dir))
    prepare_output_tree(output_root)
    set_plot_style()
    emit(output_root, "prepare", f"DAS 扫描 {len(grids)} 个网格；样本 {sample_tag}，帧 {args.frame}。")
    write_json(
        output_root / "run_config.json",
        {
            "algorithm": "das",
            "data": str(data_path),
            "source_type": source_label,
            "phantom_mode": sample_meta["phantom_mode"],
            "phantom_source": phantom_source,
            "config_scene": scene_id,
            "frame": args.frame,
            "signal_domain": args.signal_domain,
            "speed_of_sound_m_s": input_data.sample.c,
            "device": str(device),
            "x_ranges_mm": x_ranges,
            "z_ranges_mm": z_ranges,
            "width_pixels": width_counts,
            "depth_pixels": depth_counts,
            "algorithm_parameters": das_config(args),
        },
    )

    results = []
    for index, (x_range_requested, z_range_requested, width_count, depth_count) in enumerate(grids, 1):
        x_grid = make_grid(native_x, list(x_range_requested), width_count)
        z_grid = make_grid(native_z, list(z_range_requested), depth_count)
        image_db, elapsed = run_das(input_data, x_grid, z_grid, args, device)
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
        figure_dir, data_dir = output_dir / "figures", output_dir / "data"
        figure_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            data_dir / "reconstruction.npz",
            bmode_db=image_db,
            x_grid_mm=x_mm.astype(np.float32),
            z_grid_mm=z_mm.astype(np.float32),
        )
        run_config = das_config(args) | {
            "data": str(data_path),
            "source_type": source_label,
            "frame": args.frame,
            "speed_of_sound_m_s": input_data.sample.c,
            "device": str(device),
            "x_range_mm": list(x_range),
            "z_range_mm": list(z_range),
            "width_pixels": int(len(x_grid)),
            "depth_pixels": int(len(z_grid)),
            "dx_mm": round(float(np.diff(x_mm).mean()), 6),
            "dz_mm": round(float(np.diff(z_mm).mean()), 6),
            "beamforming_seconds": elapsed,
        }
        write_json(output_dir / "run_config.json", run_config)
        save_bmode(image_db, x_range, z_range, source_label, args.frame, figure_dir)
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
        emit(output_root, "progress", f"DAS {index}/{len(grids)}：{folder}（{elapsed:.2f} 秒）。")

    summary_path = output_root / "metrics" / "grid_sweep_summary.csv"
    fields = (
        "index", "width_mm", "depth_mm", "x_range_mm", "z_range_mm",
        "width_pixels", "depth_pixels", "dx_mm", "dz_mm", "beamforming_seconds", "output_dir",
    )
    write_csv(summary_path, ({key: result[key] for key in fields} for result in results), fields)
    if len(results) > 1:
        save_grid_comparison(results, output_root / "figures", source_label, args.frame)
    emit(output_root, "done", f"结果目录：{output_root}；汇总表：{summary_path}。")


if __name__ == "__main__":
    main()

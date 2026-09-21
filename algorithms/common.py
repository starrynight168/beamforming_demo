"""Shared runtime, numerical, data, and output helpers for algorithms."""

import argparse
import json
import math
from pathlib import Path
from typing import NamedTuple

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib import patches
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WINDOW_CHOICES = ["rect", "tukey", "hann", "hamming", "blackman", "kaiser"]
INTERP_CHOICES = ["nearest", "linear", "cubic", "quintic", "farrow", "sinc"]

COMMON_PARAMS = {
    "select_angles": "1",
    "f_number": 1.5,
    "dr": 60.0,
    "dynamic_aperture": True,
    "tgc": True,
    "tgc_alpha": 0.5,
    "window": "rect",
    "interp": "cubic",
}


class PackedSample(NamedTuple):
    c: float
    fc: float
    fs: float
    pitch: float
    n_elem: int
    angles: np.ndarray
    t0_vec: np.ndarray
    z_grid: np.ndarray
    x_grid: np.ndarray
    i_data: np.ndarray
    q_data: np.ndarray
    gt_data: np.ndarray | None
    has_gt: bool


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than 0")
    return value


def unit_interval_float(value):
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= 1:
        raise argparse.ArgumentTypeError("must be a finite number in (0, 1]")
    return value


def nonnegative_float(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(
            "must be a finite number greater than or equal to 0",
        )
    return value


def closed_unit_interval_float(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("must be a finite number in [0, 1]")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return value


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return value


def positive_odd_int(value):
    value = int(value)
    if value < 1 or value % 2 == 0:
        raise argparse.ArgumentTypeError("must be a positive odd integer")
    return value


def load_common_params(config_path=None):
    config_path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    return merge_common_params(config)


def merge_common_params(config):
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    configured = config.get("params", {}) or {}
    if not isinstance(configured, dict):
        raise ValueError("config.params must be a mapping")
    params = COMMON_PARAMS | configured
    params["select_angles"] = str(params["select_angles"])
    return params


def add_common_arguments(
    parser,
    params=None,
    select_help="角度选择: all, center, N(数量)",
    window_help="窗函数",
):
    params = load_common_params() if params is None else params
    parser.add_argument(
        "--select_angles",
        type=str,
        default=params["select_angles"],
        help=select_help,
    )
    parser.add_argument(
        "--f_number",
        type=positive_float,
        default=positive_float(params["f_number"]),
        help="F-Number",
    )
    parser.add_argument(
        "--dr",
        type=positive_float,
        default=positive_float(params["dr"]),
        help="动态范围 dB",
    )
    parser.add_argument(
        "--dynamic_aperture",
        action="store_true",
        default=bool(params["dynamic_aperture"]),
        help="启用动态孔径",
    )
    parser.add_argument(
        "--no_dynamic_aperture",
        dest="dynamic_aperture",
        action="store_false",
        help="禁用动态孔径",
    )
    parser.add_argument(
        "--tgc",
        action="store_true",
        default=bool(params["tgc"]),
        help="启用 TGC",
    )
    parser.add_argument("--no_tgc", dest="tgc", action="store_false", help="禁用 TGC")
    parser.add_argument(
        "--tgc_alpha",
        type=nonnegative_float,
        default=nonnegative_float(params["tgc_alpha"]),
        help="TGC 衰减系数",
    )
    parser.add_argument(
        "--window",
        type=str,
        default=params["window"],
        choices=WINDOW_CHOICES,
        help=window_help,
    )
    parser.add_argument(
        "--interp",
        type=str,
        default=params["interp"],
        choices=INTERP_CHOICES,
        help="插值方式",
    )


def add_io_arguments(parser, save_gt_help="保存GT图像"):
    parser.add_argument(
        "--h5_path",
        type=str,
        default="data/simulation.h5",
        help="H5 数据文件路径",
    )
    parser.add_argument("--h5_sample_idx", type=int, default=0, help="H5 样本索引")
    parser.add_argument("--output_dir", type=str, default="results", help="输出目录")
    parser.add_argument(
        "--save_gt",
        action="store_true",
        default=False,
        help=save_gt_help,
    )


def resolve_project_path(path, project_root):
    path = Path(path)
    return str(path if path.is_absolute() else Path(project_root) / path)


def load_from_h5(h5_path, sample_idx=0):
    with h5py.File(h5_path, "r") as hf:
        i_all = hf["all_multi_I"]
        q_all = hf["all_multi_Q"]
        if i_all.shape != q_all.shape or i_all.ndim != 4:
            raise ValueError("all_multi_I/all_multi_Q 必须是形状一致的 [N,A,T,C] 数组")
        if min(i_all.shape[0], i_all.shape[1], i_all.shape[3]) < 1 or i_all.shape[2] < 2:
            raise ValueError("IQ 的样本、角度、通道维必须非空，时间维至少为 2")
        if not 0 <= sample_idx < i_all.shape[0]:
            raise IndexError(
                f"h5_sample_idx={sample_idx} 超出 [0,{i_all.shape[0] - 1}]",
            )

        i_data = i_all[sample_idx].astype(np.float32)
        q_data = q_all[sample_idx].astype(np.float32)
        angles = hf["angles"][:].astype(np.float32)
        t0_vec = hf["time_start_vector"][sample_idx].astype(np.float32)
        if angles.ndim != 1 or len(angles) != i_data.shape[0]:
            raise ValueError("angles 必须与 IQ 的角度维度一致")
        if t0_vec.ndim != 1 or len(t0_vec) != i_data.shape[0]:
            raise ValueError("time_start_vector 必须与 IQ 的角度维度一致")

        valid_raw = hf["valid_time_samples"][sample_idx]
        valid_time = int(valid_raw)
        if float(valid_raw) != valid_time or not 2 <= valid_time <= i_data.shape[1]:
            raise ValueError(
                f"valid_time_samples[{sample_idx}]={valid_raw} 不是 [2,{i_data.shape[1]}] 内的整数",
            )
        i_data = i_data[:, :valid_time, :]
        q_data = q_data[:, :valid_time, :]
        if not np.all(np.isfinite(i_data)) or not np.all(np.isfinite(q_data)):
            raise ValueError("所选样本的 IQ 数据包含 NaN/Inf")

        c = float(hf["c"][()])
        fc = float(hf["fc"][()])
        fs = float(hf["fs"][()])
        pitch = float(hf["pitch"][()])
        if not all(np.isfinite(value) and value > 0 for value in (c, fc, fs, pitch)):
            raise ValueError("c/fc/fs/pitch 必须是有限正数")

        n_elem = int(hf["num_channels"][()])
        if n_elem != i_data.shape[2]:
            raise ValueError(
                f"num_channels={n_elem} 与 IQ 通道数={i_data.shape[2]} 不一致",
            )

        z_grid = hf["z_grid"][:].astype(np.float32)
        x_grid = hf["x_grid"][:].astype(np.float32)
        for name, grid in (("z_grid", z_grid), ("x_grid", x_grid)):
            if grid.ndim != 1 or grid.size < 2:
                raise ValueError(f"{name} 必须是至少含 2 点的一维数组")
            if not np.all(np.isfinite(grid)) or not np.all(np.diff(grid) > 0):
                raise ValueError(f"{name} 必须全部有限且严格递增")
        if not np.all(np.isfinite(angles)) or not np.all(np.isfinite(t0_vec)):
            raise ValueError("angles/time_start_vector 包含 NaN/Inf")

        gt_data = None
        has_gt = "all_envdb_norm" in hf
        if has_gt:
            if hf["all_envdb_norm"].shape[0] <= sample_idx:
                raise ValueError("GT 样本数少于 IQ 样本数")
            gt_data = hf["all_envdb_norm"][sample_idx].astype(np.float32)
            expected_gt_shape = (1, z_grid.size, x_grid.size)
            if gt_data.shape not in (expected_gt_shape, expected_gt_shape[1:]):
                raise ValueError(
                    f"GT 形状 {gt_data.shape} 与网格不匹配,期望 {expected_gt_shape}",
                )
            if not np.all(np.isfinite(gt_data)):
                raise ValueError("GT 包含 NaN/Inf")

        return (
            c,
            fc,
            fs,
            pitch,
            n_elem,
            angles,
            t0_vec,
            z_grid,
            x_grid,
            i_data,
            q_data,
            gt_data,
            has_gt,
        )


def format_method_name(method_name):
    return {
        "das": "DAS",
        "mv": "MV",
        "esbmv": "ESBMV",
        "gcfmv": "GCF-MV",
        "cmsaw": "CMSAW",
        "fdmas": "F-DMAS",
        "mban": "MBAN",
    }.get(method_name, method_name.upper())


def print_physical_summary(
    *,
    method_name,
    h5_path,
    sample_idx,
    output_dir,
    device,
    c,
    fc,
    fs,
    pitch,
    n_elem,
    angles,
    t0,
    height,
    width,
    dz,
    dx,
    depth_min,
    depth_max,
    selected_angles,
    has_gt,
    parameters,
):
    wavelength = c / fc
    fov_lateral = (n_elem - 1) * pitch * 1000
    fov_depth = (depth_max - depth_min) * 1000
    angle_display = f"{len(selected_angles)}"
    if len(selected_angles) == 1:
        angle_display += f" ({np.degrees(selected_angles[0]):.1f}°)"
    t0_value = t0[0] if isinstance(t0, np.ndarray) else t0

    print(f"\n{'=' * 70}")
    print(f"  {format_method_name(method_name)} Beamforming")
    print(f"{'=' * 70}")
    print(f"  H5 file      : {h5_path} (sample {sample_idx})")
    print(f"  GT           : {'Available' if has_gt else 'Not available'}")
    print(f"  Angles       : {angle_display}")
    print(f"{'=' * 70}")
    print(f"  Hardware     : {device}", end="")
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f" ({gpu_mem:.1f} GB)")
    else:
        print()
    print(
        f"  Transducer   : {n_elem}ch, fc={fc / 1e6:.1f}MHz, λ={wavelength * 1e3:.3f}mm, pitch={pitch * 1e3:.3f}mm",
    )
    print(f"  Sampling     : fs={fs / 1e6:.1f}MHz, {len(angles)} angles, t0={t0_value * 1e6:.3f}μs")
    print(f"  Grid         : {height}x{width}, dz={dz * 1e3:.4f}mm, dx={dx * 1e3:.4f}mm")
    print(
        f"  FOV          : {fov_depth:.1f}mm x {fov_lateral:.1f}mm "
        f"(depth {depth_min * 1e3:.1f}~{depth_max * 1e3:.1f}mm)",
    )
    print(f"{'=' * 70}")
    for label, value in parameters:
        print(f"  {label:<12}: {value}")
    print(f"  Output       : {output_dir}")
    print(f"{'=' * 70}\n")


def validate_db_output(image, expected_shape, method_name="algorithm"):
    image = np.asarray(image)
    if image.ndim != 2 or image.shape != tuple(expected_shape):
        raise ValueError(
            f"{method_name} output shape {image.shape} does not match {tuple(expected_shape)}",
        )
    if not np.all(np.isfinite(image)):
        bad_count = int(image.size - np.count_nonzero(np.isfinite(image)))
        raise FloatingPointError(
            f"{method_name} output contains {bad_count} NaN/Inf values",
        )
    peak = float(np.max(image))
    if not np.isclose(peak, 0.0, rtol=0.0, atol=1e-3):
        raise ValueError(
            f"{method_name} output is not normalized to 0 dB (peak={peak:g} dB)",
        )
    if not np.any(image < -1e-3):
        raise ValueError(f"{method_name} output has no measurable dynamic range")
    return image.astype(np.float32, copy=False)


def parse_selected_angles(angles, select_str):
    angles = np.asarray(angles)
    if angles.ndim != 1 or len(angles) == 0:
        raise ValueError("angles must be a non-empty 1D array")
    select_str = str(select_str).strip().lower()
    if select_str == "all":
        return np.arange(len(angles)), angles
    if select_str == "center":
        center_idx = int(np.argmin(np.abs(angles)))
        return np.array([center_idx]), np.array([angles[center_idx]])
    if select_str.isdigit():
        count = int(select_str)
        if not 1 <= count <= len(angles):
            raise ValueError(
                f"Angle count {count} is outside the valid range [1,{len(angles)}]",
            )
        if count == 1:
            center_idx = int(np.argmin(np.abs(angles)))
            return np.array([center_idx]), np.array([angles[center_idx]])
        sorted_indices = np.argsort(angles)
        selected = np.linspace(0, len(angles) - 1, count, dtype=int)
        indices = sorted_indices[selected]
        return indices, angles[indices]
    try:
        indices = [int(item.strip()) for item in select_str.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid angle selection: {select_str!r}") from exc
    if not indices:
        raise ValueError(
            f"Angle selection {select_str!r} contains no valid indices in [0,{len(angles) - 1}]",
        )
    invalid = [idx for idx in indices if not 0 <= idx < len(angles)]
    if invalid:
        raise ValueError(
            f"Angle selection {select_str!r} contains out-of-range indices {invalid}; valid range is [0,{len(angles) - 1}]",
        )
    if len(set(indices)) != len(indices):
        raise ValueError(f"Angle selection {select_str!r} contains duplicate indices")
    indices = np.asarray(indices, dtype=np.int64)
    return indices, angles[indices]


def aperture_half_width(depth, f_number, n_channels, pitch, dynamic_aperture):
    if dynamic_aperture:
        return depth / (2.0 * f_number)
    return torch.full_like(depth, (n_channels - 1) * pitch / 2.0)


def dynamic_aperture_channel_count(
    depth,
    f_number,
    pitch,
    n_channels,
    dynamic_aperture,
):
    if not dynamic_aperture:
        return n_channels
    count = int(depth / (f_number * pitch)) + 1
    return min(max(count, 1), n_channels)


def tgc_gain(z_grid, fc, tgc_alpha):
    return 10 ** (tgc_alpha * (fc / 1e6) * (z_grid * 100) * 2.0 / 20.0)


def aperture_window_from_dx(dx, half_a, window_type, tukey_alpha=0.25, kaiser_beta=8.6):
    x_norm = dx.abs() / (half_a + 1e-9)
    in_aperture = (x_norm <= 1.0).float()
    if window_type == "rect":
        return in_aperture

    if window_type == "tukey":
        win = torch.ones_like(x_norm)
        transition = (x_norm > (1.0 - tukey_alpha)) & (x_norm <= 1.0)
        val = 0.5 * (1.0 + torch.cos(torch.pi * (x_norm - (1.0 - tukey_alpha)) / tukey_alpha))
        win[transition] = val[transition]
        win[x_norm > 1.0] = 0.0
        return win

    if window_type == "hann":
        win = 0.5 * (1.0 + torch.cos(torch.pi * x_norm))
    elif window_type == "hamming":
        win = 0.54 + 0.46 * torch.cos(torch.pi * x_norm)
    elif window_type == "blackman":
        win = 0.42 + 0.5 * torch.cos(torch.pi * x_norm) + 0.08 * torch.cos(2.0 * torch.pi * x_norm)
    elif window_type == "kaiser":
        beta = torch.as_tensor(kaiser_beta, dtype=dx.dtype, device=dx.device)
        arg = beta * torch.sqrt(torch.clamp(1.0 - x_norm.square(), min=0.0))
        win = torch.i0(arg) / torch.i0(beta)
    else:
        raise ValueError(f"Unsupported window type: {window_type}")
    return win * in_aperture


def aperture_window_1d(
    k,
    window_type,
    device,
    dtype=torch.float32,
    tukey_alpha=0.25,
    kaiser_beta=8.6,
):
    if k < 1:
        raise ValueError("aperture size must be positive")
    if window_type == "rect" or k == 1:
        return torch.ones(k, dtype=dtype, device=device)
    x = torch.linspace(-1.0, 1.0, k, dtype=dtype, device=device)
    ax = x.abs()
    if window_type == "tukey":
        win = torch.ones_like(x)
        edge = ax > 1.0 - tukey_alpha
        win[edge] = 0.5 * (1.0 + torch.cos(torch.pi * (ax[edge] - (1.0 - tukey_alpha)) / tukey_alpha))
        return win
    if window_type == "hann":
        return 0.5 * (1.0 + torch.cos(torch.pi * x))
    if window_type == "hamming":
        return 0.54 + 0.46 * torch.cos(torch.pi * x)
    if window_type == "blackman":
        return 0.42 + 0.5 * torch.cos(torch.pi * x) + 0.08 * torch.cos(2.0 * torch.pi * x)
    if window_type == "kaiser":
        beta = torch.as_tensor(kaiser_beta, dtype=dtype, device=device)
        arg = beta * torch.sqrt(torch.clamp(1.0 - ax.square(), min=0.0))
        return torch.i0(arg) / torch.i0(beta)
    raise ValueError(f"Unsupported window type: {window_type}")


def optional_aperture_window_1d(k, window_type, device, dtype=torch.float32):
    if window_type == "rect":
        return None
    return aperture_window_1d(k, window_type, device, dtype=dtype)


def _lagrange_weight(frac, offset, offsets):
    weight = torch.ones_like(frac)
    for other in offsets:
        if other != offset:
            weight = weight * (frac - other) / (offset - other)
    return weight


def _sinc_weight(frac, offset, radius):
    x = frac - offset
    sinc = torch.sinc(x)
    window_arg = x / (radius + 1.0)
    window = 0.5 * (1.0 + torch.cos(torch.pi * window_arg))
    return sinc * window


def _interpolate_samples(sample, n_samples, interp, gather):
    sample = sample.clamp(0.0, float(n_samples - 1))

    if interp == "nearest":
        idx = sample.round().long().clamp(0, n_samples - 1)
        return gather(idx)

    if interp == "linear":
        idx0 = sample.floor().long()
        frac = sample - idx0.float()
        idx1 = torch.clamp(idx0 + 1, 0, n_samples - 1)
        idx0 = torch.clamp(idx0, 0, n_samples - 1)
        i0, q0 = gather(idx0)
        i1, q1 = gather(idx1)
        i = i0 * (1.0 - frac) + i1 * frac
        q = q0 * (1.0 - frac) + q1 * frac
        return i, q

    idx0 = sample.floor().long()
    frac = sample - idx0.float()
    if interp == "cubic":
        offsets = [-1, 0, 1, 2]
    elif interp == "quintic":
        offsets = [-2, -1, 0, 1, 2, 3]
    elif interp in {"farrow", "sinc"}:
        offsets = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    else:
        raise ValueError(f"Unsupported interpolation type: {interp}")

    i_out = torch.zeros_like(sample)
    q_out = torch.zeros_like(sample)
    norm = torch.zeros_like(sample)
    for offset in offsets:
        idx = torch.clamp(idx0 + offset, 0, n_samples - 1)
        if interp == "sinc":
            weight = _sinc_weight(frac, offset, radius=4)
            norm = norm + weight
        else:
            weight = _lagrange_weight(frac, offset, offsets)
        i_part, q_part = gather(idx)
        i_out = i_out + i_part * weight
        q_out = q_out + q_part * weight

    if interp == "sinc":
        i_out = i_out / (norm + 1e-9)
        q_out = q_out / (norm + 1e-9)
    return i_out, q_out


def interpolate_channel_samples(i_data, q_data, sample, channel_index, interp):
    return _interpolate_samples(
        sample,
        i_data.shape[0],
        interp,
        lambda index: (i_data[index, channel_index], q_data[index, channel_index]),
    )


def interpolate_multi_angle_channel_samples(
    i_data,
    q_data,
    sample,
    angle_index,
    channel_index,
    interp,
):
    n_samples = i_data.shape[1]

    def gather(idx):
        idx = torch.clamp(idx, 0, n_samples - 1)
        return i_data[angle_index, idx, channel_index], q_data[
            angle_index,
            idx,
            channel_index,
        ]

    return _interpolate_samples(sample, n_samples, interp, gather)


def db_display_range(dynamic_range):
    return -float(dynamic_range), 0.0


def _to_2d(array):
    array = np.asarray(array)
    if array.ndim == 3:
        return array[0] if array.shape[0] == 1 else array[:, :, 0]
    if array.ndim != 2:
        raise ValueError(f"image must be 2D or compatible 3D, got {array.shape}")
    return array


def _add_scale_bar(axis, extent_mm, fontsize=10):
    bar_length = 5.0
    bar_x = extent_mm[1] - bar_length - 2.0
    bar_y = extent_mm[2] - 2.0
    axis.add_patch(
        patches.Rectangle((bar_x, bar_y), bar_length, 0.5, color="white", zorder=5),
    )
    axis.text(
        bar_x + bar_length / 2,
        bar_y - 1.0,
        "5 mm",
        color="white",
        fontsize=fontsize,
        ha="center",
        va="bottom",
        fontweight="bold",
    )


def save_comparison_figure(
    image_db,
    gt_norm,
    extent_mm,
    out_path,
    title_str,
    dr=60.0,
    method_name="Beamforming",
):
    vmin, vmax = db_display_range(dr)
    image_db = _to_2d(image_db)
    gt_norm = _to_2d(gt_norm)

    fig = plt.figure(figsize=(12, 8), dpi=300)
    grid = GridSpec(1, 3, width_ratios=[1, 1, 0.05], figure=fig)

    gt_axis = fig.add_subplot(grid[0, 0])
    gt_axis.imshow(gt_norm, cmap="gray", vmin=0, vmax=1, extent=extent_mm, aspect="equal")
    gt_axis.set_title("Ground Truth", fontsize=12, pad=10)
    gt_axis.set_xlabel("Lateral (mm)")
    gt_axis.set_ylabel("Depth (mm)")

    image_axis = fig.add_subplot(grid[0, 1])
    image_handle = image_axis.imshow(
        image_db,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        extent=extent_mm,
        aspect="equal",
    )
    image_axis.set_title(f"{format_method_name(method_name)}\n{title_str}", fontsize=9, pad=10)
    image_axis.set_xlabel("Lateral (mm)")
    image_axis.set_ylabel("Depth (mm)")
    _add_scale_bar(image_axis, extent_mm)

    color_axis = fig.add_subplot(grid[0, 2])
    colorbar = fig.colorbar(image_handle, cax=color_axis, fraction=0.8)
    colorbar.set_label("Amplitude (dB)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_figure(image_db, extent_mm, out_path, title_str, dr=60.0, method_name="Beamforming"):
    vmin, vmax = db_display_range(dr)
    fig, axis = plt.subplots(figsize=(6, 8), dpi=300)
    image_handle = axis.imshow(
        image_db,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        extent=extent_mm,
        aspect="equal",
    )
    axis.set_xlabel("Lateral (mm)")
    axis.set_ylabel("Depth (mm)")
    axis.set_title(f"{format_method_name(method_name)}\n{title_str}", fontsize=9, pad=15)
    colorbar = fig.colorbar(image_handle, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Amplitude (dB)")
    _add_scale_bar(axis, extent_mm)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_params(path, params):
    Path(path).write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")

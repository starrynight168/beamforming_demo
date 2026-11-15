"""Shared runtime, numerical, data, and output helpers for algorithms."""

import argparse
import hashlib
import json
import math
import time
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
from shared_core.images import (
    ground_truth_db as _ground_truth_db,
    tgc_gain_db_from_depth,
    to_db,
    to_envelope,
)
from shared_core.signals import rf_to_analytic as rf_to_analytic, rf_to_iq as rf_to_iq
from shared_core.data import load_input as load_input, load_input_from_h5, load_target as load_target
from shared_core.h5_schema import require_h5_schema as require_h5_schema

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WINDOW_CHOICES = ["rect", "tukey", "hann", "hamming", "blackman", "kaiser"]
INTERP_CHOICES = ["nearest", "linear", "cubic", "catmull_rom", "quintic", "farrow", "sinc"]

COMMON_PARAMS = {
    "signal_domain": "iq",
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
    signal_domain: str = "iq"


class BeamformingInput(NamedTuple):
    sample: PackedSample
    selected_indices: np.ndarray
    selected_angles: np.ndarray
    i_data: np.ndarray
    q_data: np.ndarray
    t0: np.ndarray
    extent_mm: tuple[float, float, float, float]


def resolve_device(device=None):
    return torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))


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
    unsupported = set(configured) - set(COMMON_PARAMS)
    if unsupported:
        raise ValueError(
            f"config.params contains unsupported fields: {sorted(unsupported)}"
        )
    params = COMMON_PARAMS | configured
    for name in ("dynamic_aperture", "tgc"):
        if not isinstance(params[name], bool):
            raise ValueError(f"config.params.{name} must be a boolean")
    params["select_angles"] = str(params["select_angles"])
    for name, choices in (("window", WINDOW_CHOICES), ("interp", INTERP_CHOICES), ("signal_domain", ("rf", "analytic_rf", "iq"))):
        if params[name] not in choices:
            raise ValueError(f"config.params.{name} must be one of {choices}")
    return params


def add_common_arguments(
    parser,
    params=None,
    select_help="角度选择: all, center, N(数量)",
    window_help="窗函数",
):
    params = load_common_params() if params is None else params
    parser.add_argument("--signal_domain", choices=("rf", "analytic_rf", "iq"),
                        default=params.get("signal_domain", "iq"), help="输入和成像信号域；优先读取同域缓存，缺失时从 RF 派生")
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


def ground_truth_db(hf, sample_idx, signal_domain=None):
    return _ground_truth_db(hf, sample_idx, signal_domain=signal_domain)[None]


def _algorithm_channels(hf, sample_idx, signal_domain):
    values = load_input_from_h5(hf, sample_idx, domain=signal_domain)
    return (values["real"], values["imag"], values["angles"].astype(np.float32),
            values["t0"].astype(np.float32), values["fs"])


def load_from_h5(h5_path, sample_idx=0, signal_domain="iq"):
    with h5py.File(h5_path, "r") as hf:
        i_data, q_data, angles, t0_vec, fs = _algorithm_channels(hf, sample_idx, signal_domain)
        c = float(hf["c"][()])
        fc = float(hf["fc"][()])
        pitch = float(hf["pitch"][()])
        if not all(np.isfinite(value) and value > 0 for value in (c, fc, fs, pitch)):
            raise ValueError("c/fc/fs/pitch 必须是有限正数")

        n_elem = int(hf["num_channels"][()])
        if n_elem != i_data.shape[2]:
            raise ValueError(
                f"num_channels={n_elem} 与 {signal_domain} 通道数={i_data.shape[2]} 不一致",
            )

        z_grid = hf["z_grid"][:].astype(np.float32)
        x_grid = hf["x_grid"][:].astype(np.float32)
        for name, grid in (("z_grid", z_grid), ("x_grid", x_grid)):
            if grid.ndim != 1 or grid.size < 2:
                raise ValueError(f"{name} 必须是至少含 2 点的一维数组")
            if not np.all(np.isfinite(grid)) or not np.all(np.diff(grid) > 0):
                raise ValueError(f"{name} 必须全部有限且严格递增")
        if not np.all(np.isfinite(angles)) or not np.all(np.isfinite(t0_vec)):
            raise ValueError("采集角度或时间起点包含 NaN/Inf")

        gt_data = None
        has_gt = False
        generation = yaml.safe_load(hf["config_yaml"][()])["generation"]
        definition = generation["ground_truth"]
        targets = generation["training_targets"]
        algorithm = definition.get("algorithm")
        entry = targets.get("domain_targets", {}).get(signal_domain, {}).get(algorithm, {}) if algorithm else {}
        image_path = entry.get("db")
        if image_path is not None and (image_path not in hf or len(hf[image_path]) <= sample_idx):
            raise ValueError("GT 字段缺失或样本数少于输入样本数")
        has_gt = bool(image_path or entry.get("envelope") or
                      (signal_domain != "rf" and entry.get("signal")))
        if has_gt:
            dynamic_range = float(definition["dynamic_range_db"])
            gt_data = (ground_truth_db(hf, sample_idx, signal_domain=signal_domain) + dynamic_range) / dynamic_range
        if has_gt:
            expected_gt_shape = (1, z_grid.size, x_grid.size)
            if gt_data.shape not in (expected_gt_shape, expected_gt_shape[1:]):
                raise ValueError(f"GT 形状 {gt_data.shape} 与网格不匹配,期望 {expected_gt_shape}")
            if not np.all(np.isfinite(gt_data)):
                raise ValueError("GT 包含 NaN/Inf")

        return PackedSample(
            c=c,
            fc=fc,
            fs=fs,
            pitch=pitch,
            n_elem=n_elem,
            angles=angles,
            t0_vec=t0_vec,
            z_grid=z_grid,
            x_grid=x_grid,
            i_data=i_data,
            q_data=q_data,
            gt_data=gt_data,
            has_gt=has_gt,
            signal_domain=signal_domain,
        )


def prepare_beamforming_input(h5_path, sample_idx, select_str, signal_domain="iq"):
    sample = load_from_h5(h5_path, sample_idx, signal_domain=signal_domain)
    selected_indices, selected_angles = parse_selected_angles(
        sample.angles,
        select_str,
    )
    return BeamformingInput(
        sample=sample,
        selected_indices=selected_indices,
        selected_angles=selected_angles,
        i_data=sample.i_data[selected_indices],
        q_data=sample.q_data[selected_indices],
        t0=sample.t0_vec[selected_indices],
        extent_mm=(
            float(sample.x_grid[0] * 1000),
            float(sample.x_grid[-1] * 1000),
            float(sample.z_grid[-1] * 1000),
            float(sample.z_grid[0] * 1000),
        ),
    )


def time_start_tensor(t_starts, n_angles, fs, device):
    values = np.asarray(t_starts, dtype=np.float32).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("time starts must be finite")
    if values.size == 1:
        values = np.repeat(values, n_angles)
    if values.size != n_angles:
        raise ValueError(
            f"time starts must contain one value or exactly {n_angles} values, got {values.size}",
        )
    return torch.from_numpy(values).to(device) * fs


def validate_beamformer_input(i_data, q_data, angles, fs, expected_fs, n_channels):
    if i_data.shape != q_data.shape or i_data.ndim != 3 or min(i_data.shape) < 1:
        raise ValueError("I/Q must have matching nonempty [angles,time,channels] shapes")
    angles = np.asarray(angles)
    if angles.ndim != 1 or angles.size != i_data.shape[0] or not np.isfinite(angles).all():
        raise ValueError("selected_angles must match the IQ angle dimension and be finite")
    if i_data.shape[2] != n_channels:
        raise ValueError("IQ channel count does not match beamformer geometry")
    if not np.isfinite(fs) or fs <= 0 or not np.isclose(fs, expected_fs, rtol=1e-6, atol=0):
        raise ValueError("sampling frequency does not match beamformer geometry")
    if not np.isfinite(i_data).all() or not np.isfinite(q_data).all():
        raise ValueError("I/Q contains NaN/Inf")


def envelope_to_db(i_output, q_output, z_grid, fc, tgc_enabled, tgc_alpha):
    i_output = np.asarray(i_output, dtype=np.float32)
    q_output = np.asarray(q_output, dtype=np.float32)
    if i_output.ndim != 2 or i_output.shape != q_output.shape:
        raise ValueError(
            f"beamformer I/Q output must have matching 2D shapes, got {i_output.shape} and {q_output.shape}",
        )
    if not np.isfinite(i_output).all() or not np.isfinite(q_output).all():
        raise FloatingPointError("beamformer I/Q output contains NaN/Inf")
    z_grid = np.asarray(z_grid, dtype=np.float64)
    if z_grid.ndim != 1 or z_grid.size != i_output.shape[0] or not np.isfinite(z_grid).all():
        raise ValueError("z_grid must match the output depth dimension and be finite")
    envelope = to_envelope(i_output, q_output)
    gain_db = None
    if tgc_enabled:
        gain_db = tgc_gain_db_from_depth(z_grid, fc, tgc_alpha)
    return to_db(envelope, gain_db=gain_db)


def beamforming_depth_grid(z_grid, c, fs, signal_domain):
    if signal_domain != "rf":
        return z_grid
    count = int(np.ceil((z_grid[-1] - z_grid[0]) * 2 * fs / c)) + 1
    return np.linspace(z_grid[0], z_grid[-1], max(2, count), dtype=np.float32)


def beamformed_image_db(real, imag, computed_z, image_z, fc, tgc_enabled, tgc_alpha, signal_domain):
    if signal_domain == "rf":
        from scipy.signal import hilbert

        if np.any(imag != 0):
            raise ValueError("RF 成像输出必须为实数，不能丢弃非零虚部")
        pad_before = len(real) // 2
        pad_after = len(real) - pad_before
        padded = np.pad(real, ((pad_before, pad_after), (0, 0)))
        envelope = np.abs(hilbert(padded, axis=0)[pad_before:pad_before + len(real)])
        envelope = np.column_stack([np.interp(image_z, computed_z, column) for column in envelope.T])
        return envelope_to_db(envelope, np.zeros_like(envelope), image_z, fc, tgc_enabled, tgc_alpha)
    return envelope_to_db(real, imag, image_z, fc, tgc_enabled, tgc_alpha)


def run_beamformer(beamformer, i_data, q_data, selected_angles, t0, fs, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    i_output, q_output = beamformer(i_data, q_data, selected_angles, t0, fs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return i_output, q_output, time.perf_counter() - start


def save_algorithm_result(
    image_db,
    input_data,
    args,
    method_name,
    title,
    runtime_sec,
    extra_arrays=None,
    extra_params=None,
):
    """Save the standard algorithm artifacts and return the method directory."""
    method_dir = Path(args.output_dir) / method_name
    method_dir.mkdir(parents=True, exist_ok=True)
    image_path = method_dir / f"{method_name}.npy"
    np.save(image_path, np.asarray(image_db, dtype=np.float32))
    save_figure(
        image_db,
        input_data.extent_mm,
        method_dir / f"{method_name}.png",
        title,
        dr=args.dr,
        method_name=method_name,
    )
    sample = input_data.sample
    if sample.has_gt and args.save_gt:
        with h5py.File(resolve_project_path(args.h5_path, PROJECT_ROOT), "r") as hf:
            gt_db = ground_truth_db(hf, args.h5_sample_idx, signal_domain=args.signal_domain)
        save_comparison_figure(
            image_db,
            gt_db,
            input_data.extent_mm,
            method_dir / f"{method_name}_comparison.png",
            title,
            dr=args.dr,
            method_name=method_name,
        )
    for suffix, values in (extra_arrays or {}).items():
        np.save(method_dir / f"{method_name}_{suffix}.npy", np.asarray(values))
    params = {**vars(args), "method": method_name, "runtime_sec": float(runtime_sec)}
    if isinstance(sample, PackedSample):
        params["image_depth_rows"] = len(sample.z_grid)
        params["beamforming_depth_rows"] = len(beamforming_depth_grid(sample.z_grid, sample.c, sample.fs, sample.signal_domain))
    params.update(extra_params or {})
    params["cache_signature"] = result_cache_signature(
        args.h5_path, method_name, vars(args)
    )
    write_params(method_dir / "params.json", params)
    return method_dir


def format_method_name(method_name):
    return {
        "das": "DAS",
        "mv": "MV",
        "esbmv": "ESBMV",
        "gcfmv": "GCF-MV",
        "cmsaw": "CMSAW",
        "ccmv": "CC-MV",
        "fdmas": "F-DMAS",
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
    print(
        f"  Sampling     : fs={fs / 1e6:.1f}MHz, {len(angles)} angles, t0={t0_value * 1e6:.3f}μs"
    )
    print(
        f"  Grid         : {height}x{width}, dz={dz * 1e3:.4f}mm, dx={dx * 1e3:.4f}mm"
    )
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


def db_display_range(dynamic_range):
    return -float(dynamic_range), 0.0


def _to_2d(array):
    array = np.asarray(array)
    if array.ndim == 3:
        return array[0] if array.shape[0] == 1 else array[:, :, 0]
    if array.ndim != 2:
        raise ValueError(f"image must be 2D or compatible 3D, got {array.shape}")
    return array


def add_scale_bar(axis, extent_mm, fontsize=10):
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
    gt_db,
    extent_mm,
    out_path,
    title_str,
    dr=60.0,
    method_name="Beamforming",
):
    vmin, vmax = db_display_range(dr)
    image_db = _to_2d(image_db)
    gt_db = _to_2d(gt_db)

    fig = plt.figure(figsize=(12, 8), dpi=300)
    grid = GridSpec(1, 3, width_ratios=[1, 1, 0.05], figure=fig)

    gt_axis = fig.add_subplot(grid[0, 0])
    gt_axis.imshow(
        gt_db, cmap="gray", vmin=vmin, vmax=vmax, extent=extent_mm, aspect="equal"
    )
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
    image_axis.set_title(
        f"{format_method_name(method_name)}\n{title_str}", fontsize=9, pad=10
    )
    image_axis.set_xlabel("Lateral (mm)")
    image_axis.set_ylabel("Depth (mm)")
    add_scale_bar(image_axis, extent_mm)

    color_axis = fig.add_subplot(grid[0, 2])
    colorbar = fig.colorbar(image_handle, cax=color_axis, fraction=0.8)
    colorbar.set_label("Amplitude (dB)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_figure(
    image_db, extent_mm, out_path, title_str, dr=60.0, method_name="Beamforming"
):
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
    axis.set_title(
        f"{format_method_name(method_name)}\n{title_str}", fontsize=9, pad=15
    )
    colorbar = fig.colorbar(image_handle, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Amplitude (dB)")
    add_scale_bar(axis, extent_mm)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_params(path, params):
    Path(path).write_text(
        json.dumps(params, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def result_cache_signature(h5_path, method_name, parameters=None, *, extra_args=()):
    input_path = Path(resolve_project_path(h5_path, PROJECT_ROOT)).resolve(strict=True)
    input_stat = input_path.stat()
    effective_parameters = dict(parameters or {})
    for index, argument in enumerate(extra_args):
        if not argument.startswith("--"):
            continue
        name, separator, value = argument[2:].partition("=")
        if separator:
            effective_parameters[name] = value
        elif index + 1 < len(extra_args) and not extra_args[index + 1].startswith("--"):
            effective_parameters[name] = extra_args[index + 1]
    sources = {}
    for source_path in algorithm_source_paths(method_name, effective_parameters.get("train_root")):
        sources[str(source_path.resolve())] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    dependencies = {}
    for name, value in effective_parameters.items():
        normalized_name = str(name).lower().replace("-", "_")
        if normalized_name == "h5_path" or not any(
            token in normalized_name
            for token in ("_path", "_file", "config", "checkpoint", "weights")
        ):
            continue
        if not isinstance(value, (str, Path)):
            continue
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path = path.resolve()
        if path.is_file():
            stat = path.stat()
            dependencies[str(path)] = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "file_id": getattr(stat, "st_ino", None),
            }

    return {
        "input": {
            "path": str(input_path),
            "size": input_stat.st_size,
            "mtime_ns": input_stat.st_mtime_ns,
            "file_id": getattr(input_stat, "st_ino", None),
        },
        "sources": sources,
        "dependencies": dependencies,
    }


def algorithm_source_paths(method_name, train_root=None):
    script_path = Path(__file__).resolve()
    algorithm_dir = script_path.parent.parent
    mban_sources = (
        "eval_core/inference.py",
        "eval_core/models.py",
        "eval_core/calibration.py",
        "eval_core/hardware.py",
        "shared_core/beamforming.py",
        "shared_core/checkpoints.py",
        "shared_core/config.py",
        "shared_core/config_schema.py",
        "shared_core/controls.py",
        "shared_core/crossbar.py",
        "shared_core/data.py",
        "shared_core/geometry.py",
        "shared_core/hardware.py",
        "shared_core/hardware_config.py",
        "shared_core/hardware_layers.py",
        "shared_core/hashing.py",
        "shared_core/interpolation.py",
        "shared_core/model.py",
        "shared_core/operators.py",
        "shared_core/runtime.py",
        "shared_core/signal_chain.py",
        "shared_core/signals.py",
        "shared_core/images.py",
        "shared_core/splits.py",
    )
    train_root = Path(resolve_project_path(train_root, PROJECT_ROOT)) if train_root else PROJECT_ROOT / "train"
    dependencies = {
        "ccmv": (algorithm_dir / "mv.py",),
        "esbmv": (algorithm_dir / "mv.py",),
        "gcfmv": (algorithm_dir / "mv.py",),
        "cmsaw": (algorithm_dir / "mv.py",),
        "mban": tuple(train_root / source for source in mban_sources),
    }
    beamforming_sources = (
        ()
        if method_name == "mban"
        else (algorithm_dir / "shared_core/beamforming.py",)
    )
    return (
        algorithm_dir / f"{method_name}.py",
        script_path,
        PROJECT_ROOT / "shared_core/images.py",
        PROJECT_ROOT / "shared_core/signals.py",
        PROJECT_ROOT / "shared_core/normalization.py",
        PROJECT_ROOT / "shared_core/data.py",
        PROJECT_ROOT / "shared_core/h5_schema.py",
        *beamforming_sources,
        *dependencies.get(method_name, ()),
    )

"""Provide Python utilities for cmsaw."""

import argparse
import importlib
import json
import os
import subprocess
import sys
import time

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from beamforming_utils import (
    aperture_window_1d,
    db_display_range,
    dynamic_aperture_channel_count,
    interpolate_multi_angle_channel_samples,
    parse_selected_angles,
    resolve_project_path,
    validate_db_output,
)
from common_params import (
    add_common_arguments,
    add_io_arguments,
    nonnegative_float,
    positive_odd_int,
    unit_interval_float,
)
from h5_loader import load_from_h5
from matplotlib import patches
from matplotlib.gridspec import GridSpec

COMPARISON_VALUE_0_5 = 0.5
COMPARISON_VALUE_100 = 100
COMPARISON_VALUE_2 = 2
COMPARISON_VALUE_3 = 3
COMPARISON_VALUE_4 = 4

# ================= 命令行参数配置 =================
parser = argparse.ArgumentParser(
    description="CMSAW - Coherence-based Minimum Variance Adaptive Weighting",
)
add_common_arguments(parser, select_help="角度选择: center, 0,1,2(逗号分隔)")

# ---- MV参数(用于自动生成基线) ----
parser.add_argument(
    "--mv_dl",
    type=nonnegative_float,
    default=0.0,
    help="MV 对角加载系数(自动生成基线用)",
)
parser.add_argument(
    "--subarray_ratio",
    type=unit_interval_float,
    default=0.25,
    help="MV 子阵列比例(自动生成基线用)",
)
parser.add_argument(
    "--temporal_win",
    type=positive_odd_int,
    default=9,
    help="MV 时间平均窗口(自动生成基线用)",
)
parser.add_argument(
    "--fbss",
    action="store_true",
    default=True,
    help="MV 启用FBSS(自动生成基线用)",
)
parser.add_argument("--no_fbss", dest="fbss", action="store_false", help="MV 禁用FBSS")
# ---- CMSAW 核心参数 ----
parser.add_argument(
    "--baseline_mv",
    type=str,
    default="mv.npy",
    help="MV基线结果文件路径",
)
parser.add_argument(
    "--lmax_ratio",
    type=float,
    default=0.5,
    help="最大子阵列长度比例 Lmax = ratio x aperture_size",
)
parser.add_argument("--min_subarray_len", type=int, default=2, help="最小子阵列长度")
parser.add_argument("--delta_max", type=float, default=1.0, help="对角线缩放因子最大值")
parser.add_argument("--gamma", type=float, default=0.5, help="权重幂次")
parser.add_argument(
    "--clip_percentile",
    type=float,
    default=90.0,
    help="权重裁剪百分位数",
)
parser.add_argument(
    "--depth_smooth_rows",
    type=int,
    default=1,
    help="深度平滑行数 (1=禁用)",
)

add_io_arguments(parser, save_gt_help="保存GT对比图")
args = parser.parse_args()

METHOD_NAME = "cmsaw"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


H5_PATH = resolve_project_path(args.h5_path, PROJECT_ROOT)
BASE_OUTPUT_DIR = args.output_dir
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, METHOD_NAME)


def print_physical_summary(
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
    n_files,
    selected_angles,
    has_gt,
    baseline_path,
):
    """Execute print physical summary."""
    wavelength = c / fc
    pw = (n_elem - 1) * pitch
    fov_lateral = pw * 1000
    fov_depth = (depth_max - depth_min) * 1000

    angle_display = f"{len(selected_angles)}"
    if len(selected_angles) == 1:
        angle_display += f" ({np.degrees(selected_angles[0]):.1f}°)"

    print(f"\n{'=' * 70}")
    print("  CMSAW Beamforming")
    print(f"{'=' * 70}")
    print(f"  H5 file      : {H5_PATH} (sample {args.h5_sample_idx})")
    print(f"  GT           : {'Available' if has_gt else 'Not available'}")
    print(f"  Angles       : {angle_display}")
    print(f"  MV baseline  : {baseline_path}")
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
        f"  Sampling     : fs={fs / 1e6:.1f}MHz, {len(angles)} angles, t0={t0[0] * 1e6 if isinstance(t0, np.ndarray) else t0 * 1e6:.3f}μs",
    )
    print(f"  Grid         : {height}x{width}, dz={dz * 1e3:.4f}mm, dx={dx * 1e3:.4f}mm")
    print(
        f"  FOV          : {fov_depth:.1f}mm x {fov_lateral:.1f}mm (depth {depth_min * 1e3:.1f}~{depth_max * 1e3:.1f}mm)",
    )
    print(f"{'=' * 70}")
    print(f"  F-Number     : {args.f_number}")
    print(f"  Aperture     : {'Dynamic' if args.dynamic_aperture else 'Fixed Full'}")
    print(f"  Window       : {args.window.upper()}")
    print(f"  Interp       : {args.interp}")
    print(f"  CMSAW Lmax   : {args.lmax_ratio} x aperture_size")
    print(f"  CMSAW minL   : {args.min_subarray_len}")
    print(f"  CMSAW delta  : {args.delta_max}")
    print(f"  CMSAW gamma  : {args.gamma}")
    print(f"  CMSAW clip   : {args.clip_percentile}%")
    print(f"  CMSAW smooth : {args.depth_smooth_rows} rows")
    print(f"  DR           : {args.dr} dB")
    print(f"  Output       : {OUTPUT_DIR}")
    print(f"{'=' * 70}\n")


def ensure_mv_baseline(baseline_path):
    """验证 MV 基线参数;默认基线缺失或过期时自动重新生成。."""
    managed_baseline = not os.path.isabs(baseline_path) and baseline_path == "mv.npy"
    baseline_output_dir = os.path.join(OUTPUT_DIR, "_mv_baseline") if managed_baseline else BASE_OUTPUT_DIR
    if not os.path.isabs(baseline_path):
        full_path = os.path.join(baseline_output_dir, "mv", "mv.npy") if managed_baseline else os.path.join(BASE_OUTPUT_DIR, baseline_path)
    else:
        full_path = baseline_path

    expected = {
        "h5_path": os.path.normcase(os.path.abspath(H5_PATH)),
        "h5_sample_idx": int(args.h5_sample_idx),
        "select_angles": str(args.select_angles),
        "f_number": float(args.f_number),
        "dr": float(args.dr),
        "dynamic_aperture": bool(args.dynamic_aperture),
        "tgc": bool(args.tgc),
        "tgc_alpha": float(args.tgc_alpha),
        "window": args.window,
        "interp": args.interp,
        "mv_dl": float(args.mv_dl),
        "fbss": bool(args.fbss),
        "subarray_ratio": float(args.subarray_ratio),
        "temporal_win": int(args.temporal_win),
    }

    def baseline_matches():
        """Execute baseline matches."""
        params_path = os.path.join(os.path.dirname(full_path), "params.json")
        if not os.path.exists(full_path) or not os.path.exists(params_path):
            return False, "缺少基线文件或 params.json"
        try:
            with open(params_path, encoding="utf-8") as file:
                actual = json.load(file)
        except (OSError, ValueError) as exc:
            return False, f"无法读取 params.json: {exc}"
        actual = {key: actual.get(key) for key in expected}
        if actual.get("h5_path") is not None:
            actual["h5_path"] = os.path.normcase(os.path.abspath(actual["h5_path"]))
        if managed_baseline and actual == expected:
            baseline_mtime = os.path.getmtime(full_path)
            dependencies = [H5_PATH, os.path.join(SCRIPT_DIR, "mv.py")]
            dependencies.extend(os.path.join(SCRIPT_DIR, name) for name in ("beamforming_utils.py", "common_params.py", "h5_loader.py"))
            if any(os.path.getmtime(path) > baseline_mtime for path in dependencies):
                return False, "输入数据或 MV 源码比基线更新"
        return (actual == expected, "参数一致" if actual == expected else "参数不一致")

    matches, reason = baseline_matches()
    if matches:
        print(f"MV基线已存在且参数一致: {full_path}")
        return full_path
    if not managed_baseline:
        raise ValueError(f"自定义 MV 基线不可用({reason}): {full_path}")

    print(f"MV基线需要生成({reason}): {full_path}")

    mv_script = os.path.join(SCRIPT_DIR, "mv.py")
    if not os.path.exists(mv_script):
        raise FileNotFoundError(f"找不到mv.py: {mv_script},请确保mv.py在相同目录下")

    cmd = [
        sys.executable,
        mv_script,
        "--h5_path",
        H5_PATH,
        "--h5_sample_idx",
        str(args.h5_sample_idx),
        "--select_angles",
        args.select_angles,
        "--f_number",
        str(args.f_number),
        "--dr",
        str(args.dr),
        "--output_dir",
        baseline_output_dir,
    ]

    if args.dynamic_aperture:
        cmd.append("--dynamic_aperture")
    else:
        cmd.append("--no_dynamic_aperture")

    cmd.extend(["--window", args.window])
    cmd.extend(["--interp", args.interp])

    # MV参数
    cmd.extend(["--mv_dl", str(args.mv_dl)])
    cmd.extend(["--subarray_ratio", str(args.subarray_ratio)])
    cmd.extend(["--temporal_win", str(args.temporal_win)])
    if args.fbss:
        cmd.append("--fbss")
    else:
        cmd.append("--no_fbss")

    if args.tgc:
        cmd.append("--tgc")
        cmd.extend(["--tgc_alpha", str(args.tgc_alpha)])
    else:
        cmd.append("--no_tgc")

    print(f"执行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        print(f"MV生成失败: {result.stderr}")
        raise RuntimeError("自动生成MV基线失败")

    print(result.stdout)
    matches, reason = baseline_matches()
    if not matches:
        raise RuntimeError(f"MV基线生成后校验失败: {reason}")
    print("MV基线生成并校验完成")
    return full_path


def delayed_iq(sample, angle_idx):
    """Execute delayed iq."""
    i_data = torch.from_numpy(sample["I"][angle_idx]).to(device)
    q_data = torch.from_numpy(sample["Q"][angle_idx]).to(device)
    angles = torch.from_numpy(sample["angles"][angle_idx]).to(device)
    t0 = torch.from_numpy(np.atleast_1d(sample["t0"][angle_idx])).to(device)
    z = torch.from_numpy(sample["z"]).to(device)
    x = torch.from_numpy(sample["x"]).to(device)
    n_angles, n_samples, n_channels = i_data.shape

    z_mesh, x_mesh = torch.meshgrid(z, x, indexing="ij")
    elements = (torch.arange(n_channels, device=device) - (n_channels - 1) / 2.0) * sample["pitch"]
    receive = torch.sqrt((x_mesh[..., None] - elements) ** 2 + z_mesh[..., None] ** 2)
    transmit = z_mesh[None, ..., None] * torch.cos(angles)[:, None, None, None]
    transmit += x_mesh[None, ..., None] * torch.sin(angles)[:, None, None, None]
    tof = (transmit + receive[None]) / sample["c"]
    exact = (tof - t0[:, None, None, None]) * sample["fs"]
    valid = (exact >= 0) & (exact <= n_samples - 1)

    angle_index = torch.arange(n_angles, device=device)[:, None, None, None]
    channel_offset = torch.arange(n_channels, device=device)[None, None, None, :]

    i, q = interpolate_multi_angle_channel_samples(
        i_data,
        q_data,
        exact,
        angle_index,
        channel_offset,
        args.interp,
    )

    phase_rx = 2.0 * torch.pi * sample["fc"] * (receive / sample["c"])
    cos_rx = torch.cos(phase_rx)[None]
    sin_rx = torch.sin(phase_rx)[None]
    aligned_i = i * cos_rx - q * sin_rx
    aligned_q = i * sin_rx + q * cos_rx
    return torch.complex(aligned_i, aligned_q) * valid


def cmsaw_weight_from_delayed_data(
    data,
    z_grid,
    x_grid,
    pitch,
    f_number,
    dynamic_aperture,
    window,
    lmax_ratio,
    min_subarray_len,
    delta_max,
    gamma,
    clip_percentile,
    depth_smooth_rows,
    return_lengths=False,
):
    """Compute one CMSAW map per angle and average the normalized maps."""
    if data.ndim == COMPARISON_VALUE_3:
        data = data.unsqueeze(0)
    if data.ndim != COMPARISON_VALUE_4 or data.shape[0] < 1:
        raise ValueError("delayed IQ 必须是 [A,height,width,C] 或 [height,width,C]")

    _, height, width, n_channels = data.shape
    z_grid = np.asarray(z_grid, dtype=np.float32)
    x_grid = np.asarray(x_grid, dtype=np.float32)
    x_t = torch.from_numpy(x_grid).to(device)
    element_x = (torch.arange(n_channels, device=device) - (n_channels - 1) / 2.0) * pitch
    centers = torch.argmin(torch.abs(x_t[:, None] - element_x[None]), dim=1)

    row_cache = []
    for depth in z_grid:
        k = dynamic_aperture_channel_count(
            depth,
            f_number,
            pitch,
            n_channels,
            dynamic_aperture,
        )
        starts = torch.clamp(centers - k // 2, 0, n_channels - k)
        channels = starts[:, None] + torch.arange(k, device=device)[None]
        row_cache.append(
            (k, channels, aperture_window_1d(k, window, device)[None]),
        )

    angle_weights = []
    angle_lengths = []
    for angle_data in data:
        sigma = torch.empty((height, width), dtype=torch.float32, device=device)
        active_rows = []
        k_rows = []
        for iz, (k, channels, aperture_window) in enumerate(row_cache):
            active = torch.gather(angle_data[iz], 1, channels) * aperture_window
            sigma[iz] = torch.std(torch.abs(active), dim=1, unbiased=False)
            active_rows.append(active)
            k_rows.append(k)

        sigma_prime = torch.pow(sigma + 1e-12, -1.0 / 3.0)
        sigma_prime = (sigma_prime - sigma_prime.min()) / (sigma_prime.max() - sigma_prime.min() + 1e-12)
        weight = torch.zeros_like(sigma)
        length_map = torch.zeros(
            (height, width),
            dtype=torch.int16,
            device=device,
        )

        for iz, active in enumerate(active_rows):
            k = k_rows[iz]
            lmax = min(
                max(min_subarray_len, int(np.floor(k * lmax_ratio))),
                k,
            )
            min_l = min(min_subarray_len, lmax)
            lengths = (
                torch.floor(sigma_prime[iz] * lmax)
                .long()
                .clamp(
                    min_l,
                    lmax,
                )
            )
            length_map[iz] = lengths.to(torch.int16)
            coherent = torch.abs(active.sum(dim=1)).square()
            total = k * torch.abs(active).square().sum(dim=1)
            coherence = (coherent / (total + 1e-12)).clamp(0, 1)
            delta = torch.pow(coherence + 1e-12, sigma_prime[iz]) * delta_max

            for length_tensor in torch.unique(lengths):
                length = int(length_tensor.item())
                columns = torch.where(lengths == length)[0]
                subset = active[columns]
                sub = subset.unfold(1, length, 1).transpose(1, 2)
                covariance = sub @ sub.mH / float(k - length + 1)
                eye = torch.eye(length, dtype=torch.complex64, device=device)
                exchange = torch.flip(eye, dims=(0,))
                transpose = covariance.transpose(-2, -1)
                rotary = 0.25 * (covariance + exchange @ transpose + exchange @ covariance @ exchange + transpose @ exchange)
                diagonal = torch.diag_embed(
                    torch.diagonal(rotary, dim1=-2, dim2=-1),
                )
                matrix = torch.abs(
                    rotary - delta[columns, None, None] * diagonal,
                )
                weight[iz, columns] = matrix.mean(dim=(-2, -1)) / (matrix.std(dim=(-2, -1), unbiased=False) + 1e-12)

        if depth_smooth_rows > 1:
            rows = depth_smooth_rows + 1 if depth_smooth_rows % 2 == 0 else depth_smooth_rows
            if rows > height:
                raise ValueError(
                    f"depth_smooth_rows={depth_smooth_rows} 超过图像深度 {height}",
                )
            coord = torch.arange(rows, device=device).float() - rows // 2
            kernel = torch.exp(
                -0.5 * (coord / max(rows / 4.0, 1.0)) ** 2,
            )
            kernel = (kernel / kernel.sum()).view(1, 1, rows, 1)
            log_weight = torch.log(weight.clamp_min(1e-12))[None, None]
            log_weight = torch.nn.functional.pad(
                log_weight,
                (0, 0, rows // 2, rows // 2),
                mode="reflect",
            )
            weight = torch.exp(
                torch.nn.functional.conv2d(log_weight, kernel)[0, 0],
            )

        weight /= torch.median(weight.clamp_min(1e-12))
        weight.clamp_(max=torch.quantile(weight, clip_percentile / 100.0))
        weight.pow_(gamma)
        angle_weights.append(weight)
        angle_lengths.append(length_map)

    combined = torch.stack(angle_weights, dim=0).mean(dim=0)
    if return_lengths:
        return combined, torch.stack(angle_lengths, dim=0)
    return combined


class CMSAWBeamformerIQ:
    """Pack-time CMSAW teacher interface.

    CMSAW 本身是基于 MV baseline 的显示域/幅值加权方法;这里为了统一 H5 GT 打包接口,
    采用“MV 复数 IQ x CMSAW 实值权重”的定义:
      - 相位沿用 MV baseline;
      - 幅值按 CMSAW coherence 权重调整;
      - 返回 i_output, q_output,供 pack_EPFL.py 后续统一生成 envdb。
    """

    def __init__(
        self,
        z_grid,
        x_grid,
        n_elem,
        pitch,
        c,
        fc,
        fs,
        t0_all,
        angles_rad,
        mv_dl=0.0,
        fbss=True,
        subarray_ratio=0.25,
        temporal_win=9,
        lmax_ratio=0.5,
        min_subarray_len=2,
        delta_max=1.0,
        gamma=0.5,
        clip_percentile=90.0,
        depth_smooth_rows=1,
        baseline_mv=None,
    ):
        """Initialize the instance."""
        if not 0 < lmax_ratio <= COMPARISON_VALUE_0_5 or min_subarray_len < COMPARISON_VALUE_2:
            raise ValueError("Require lmax_ratio in (0, 0.5] and min_subarray_len >= 2")
        if not 0 <= delta_max <= 1 or not 0 < gamma <= 1 or not 0 < clip_percentile <= COMPARISON_VALUE_100:
            raise ValueError(
                "Require delta_max in [0,1], gamma in (0,1], clip_percentile in (0,100]",
            )
        if temporal_win < 1 or temporal_win % 2 == 0:
            raise ValueError("temporal_win must be a positive odd integer")
        if depth_smooth_rows < 1:
            raise ValueError("depth_smooth_rows must be at least 1")

        self.z_grid = np.asarray(z_grid, dtype=np.float32)
        self.x_grid = np.asarray(x_grid, dtype=np.float32)
        self.n_elem = int(n_elem)
        self.pitch = float(pitch)
        self.c = float(c)
        self.fs = float(fs)
        self.fc = float(fc)
        self.lmax_ratio = float(lmax_ratio)
        self.min_subarray_len = int(min_subarray_len)
        self.delta_max = float(delta_max)
        self.gamma = float(gamma)
        self.clip_percentile = float(clip_percentile)
        self.depth_smooth_rows = int(depth_smooth_rows)

        old_argv = sys.argv[:]
        try:
            sys.argv = ["mv.py"]
            self.mv_module = importlib.import_module("mv")
        finally:
            sys.argv = old_argv
        self.mv_module.args = args
        self.mv_bf = self.mv_module.RowDynamicMVBeamformerIQ(
            self.z_grid,
            self.x_grid,
            self.n_elem,
            self.pitch,
            self.c,
            self.fc,
            self.fs,
            t0_all,
            angles_rad,
            mv_dl=mv_dl,
            fbss=fbss,
            subarray_ratio=subarray_ratio,
            temporal_win=temporal_win,
        )

    def _weight_from_delayed_data(self, data):
        """Execute  weight from delayed data."""
        return cmsaw_weight_from_delayed_data(
            data,
            self.z_grid,
            self.x_grid,
            self.pitch,
            args.f_number,
            args.dynamic_aperture,
            args.window,
            self.lmax_ratio,
            self.min_subarray_len,
            self.delta_max,
            self.gamma,
            self.clip_percentile,
            self.depth_smooth_rows,
        )

    def __call__(self, i_data, q_data, selected_angles, t_starts, fs):
        """Run the callable operation."""
        i_mv, q_mv = self.mv_bf(i_data, q_data, selected_angles, t_starts, fs)
        t0 = np.asarray(t_starts, dtype=np.float32).reshape(-1)
        if t0.size == 1:
            t0 = np.repeat(t0, len(selected_angles))
        sample = {
            "I": i_data.astype(np.float32),
            "Q": q_data.astype(np.float32),
            "t0": t0.astype(np.float32),
            "z": self.z_grid,
            "x": self.x_grid,
            "angles": np.asarray(selected_angles, dtype=np.float32),
            "fs": float(fs),
            "c": self.c,
            "fc": self.fc,
            "pitch": self.pitch,
        }
        angle_weights = []
        for angle_idx in range(len(selected_angles)):
            data = delayed_iq(sample, np.array([angle_idx]))
            angle_weights.append(self._weight_from_delayed_data(data))
        weight = torch.stack(angle_weights, dim=0).mean(dim=0)
        weight = weight.cpu().numpy().astype(np.float32)
        return i_mv * weight, q_mv * weight


# ================= 保存函数 =================
def save_comparison_figure(cmsaw_db, gt_norm, extent_mm, out_path, title_str, dr=60.0):
    """Save comparison figure."""
    vmin, vmax = db_display_range(dr)

    def to_2d(arr):
        """Execute to 2d."""
        if arr.ndim == COMPARISON_VALUE_3:
            return arr[0] if arr.shape[0] == 1 else arr[:, :, 0]
        return arr

    cmsaw_db, gt_norm = to_2d(cmsaw_db), to_2d(gt_norm)

    fig = plt.figure(figsize=(12, 8), dpi=300)
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(gt_norm, cmap="gray", vmin=0, vmax=1, extent=extent_mm, aspect="equal")
    ax1.set_title("Ground Truth", fontsize=12, pad=10)
    ax1.set_xlabel("Lateral (mm)")
    ax1.set_ylabel("Depth (mm)")

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(
        cmsaw_db,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        extent=extent_mm,
        aspect="equal",
    )
    ax2.set_title(f"CMSAW\n{title_str}", fontsize=9, pad=10)
    ax2.set_xlabel("Lateral (mm)")
    ax2.set_ylabel("Depth (mm)")

    cax = fig.add_subplot(gs[0, 2])
    cbar = fig.colorbar(im2, cax=cax, fraction=0.8)
    cbar.set_label("Amplitude (dB)")

    bar_length = 5.0
    bar_x = extent_mm[1] - bar_length - 2.0
    bar_y = extent_mm[2] - 2.0
    scale_bar = patches.Rectangle(
        (bar_x, bar_y),
        bar_length,
        0.5,
        color="white",
        zorder=5,
    )
    ax2.add_patch(scale_bar)
    ax2.text(
        bar_x + bar_length / 2,
        bar_y - 1.0,
        "5 mm",
        color="white",
        fontsize=10,
        ha="center",
        va="bottom",
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_figure(db_img, extent_mm, out_path, title_str, dr=60.0):
    """Save figure."""
    vmin, vmax = db_display_range(dr)
    fig, ax = plt.subplots(figsize=(6, 8), dpi=300)
    im = ax.imshow(
        db_img,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        extent=extent_mm,
        aspect="equal",
    )
    ax.set_xlabel("Lateral (mm)")
    ax.set_ylabel("Depth (mm)")
    ax.set_title(f"CMSAW\n{title_str}", fontsize=9, pad=15)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Amplitude (dB)")

    bar_length = 5.0
    bar_x = extent_mm[1] - bar_length - 2.0
    bar_y = extent_mm[2] - 2.0
    scale_bar = patches.Rectangle(
        (bar_x, bar_y),
        bar_length,
        0.5,
        color="white",
        zorder=5,
    )
    ax.add_patch(scale_bar)
    ax.text(
        bar_x + bar_length / 2,
        bar_y - 1.0,
        "5 mm",
        color="white",
        fontsize=10,
        ha="center",
        va="bottom",
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ================= 主程序 =================
def main():
    # 参数验证
    """Run the command-line workflow."""
    if not 0 < args.lmax_ratio <= COMPARISON_VALUE_0_5 or args.min_subarray_len < COMPARISON_VALUE_2:
        raise ValueError("Require lmax_ratio in (0, 0.5] and min_subarray_len >= 2")
    if not 0 <= args.delta_max <= 1 or not 0 < args.gamma <= 1 or not 0 < args.clip_percentile <= COMPARISON_VALUE_100:
        raise ValueError(
            "Require delta_max in [0,1], gamma in (0,1], clip_percentile in (0,100]",
        )
    if args.depth_smooth_rows < 1:
        raise ValueError("depth_smooth_rows must be at least 1")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ========== 自动生成MV基线 ==========
    baseline_path = ensure_mv_baseline(args.baseline_mv)

    # 加载H5数据
    (
        c,
        fc,
        fs,
        pitch,
        n_elem,
        angles_all,
        t0_all,
        z_grid,
        x_grid,
        i_data,
        q_data,
        gt_data,
        has_gt,
    ) = load_from_h5(
        H5_PATH,
        args.h5_sample_idx,
    )

    selected_indices, selected_angles = parse_selected_angles(
        angles_all,
        args.select_angles,
    )
    t0_sub = t0_all[selected_indices] if isinstance(t0_all, (np.ndarray, list)) else t0_all

    height, width = len(z_grid), len(x_grid)
    dz, dx = z_grid[1] - z_grid[0], x_grid[1] - x_grid[0]
    depth_min, depth_max = z_grid[0], z_grid[-1]
    extent_mm = [
        x_grid[0] * 1000,
        x_grid[-1] * 1000,
        z_grid[-1] * 1000,
        z_grid[0] * 1000,
    ]

    # 加载MV基线
    baseline = np.load(baseline_path).astype(np.float32)
    baseline = validate_db_output(baseline, (height, width), "CMSAW MV baseline")

    print_physical_summary(
        c,
        fc,
        fs,
        pitch,
        n_elem,
        angles_all,
        t0_sub,
        height,
        width,
        dz,
        dx,
        depth_min,
        depth_max,
        1,
        selected_angles,
        has_gt,
        baseline_path,
    )

    # 准备数据
    sample = {
        "I": i_data,
        "Q": q_data,
        "t0": t0_all,
        "z": z_grid,
        "x": x_grid,
        "angles": angles_all,
        "fs": fs,
        "c": c,
        "fc": fc,
        "pitch": pitch,
    }

    print(f"Processing CMSAW (Lmax={args.lmax_ratio}aperture_size, gamma={args.gamma})...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()

    with torch.no_grad():
        angle_weights = []
        angle_lengths = []
        for angle_number, selected_index in enumerate(selected_indices, 1):
            print(
                f"  CMSAW angle {angle_number}/{len(selected_indices)} ({np.degrees(angles_all[selected_index]):.1f}°)",
            )
            data = delayed_iq(sample, np.array([selected_index]))
            angle_weight, angle_length = cmsaw_weight_from_delayed_data(
                data,
                z_grid,
                x_grid,
                pitch,
                args.f_number,
                args.dynamic_aperture,
                args.window,
                args.lmax_ratio,
                args.min_subarray_len,
                args.delta_max,
                args.gamma,
                args.clip_percentile,
                args.depth_smooth_rows,
                return_lengths=True,
            )
            angle_weights.append(angle_weight)
            angle_lengths.append(angle_length[0])
        weight = torch.stack(angle_weights, dim=0).mean(dim=0)
        dynamic_l = torch.stack(angle_lengths, dim=0)
        baseline_env_sq = torch.from_numpy(np.power(10.0, baseline / 10.0)).to(device)
        output_env_sq = baseline_env_sq * (weight**2)
        output_env_sq /= output_env_sq.max() + 1e-24
        output = (10.0 * torch.log10(output_env_sq + 1e-24)).cpu().numpy().astype(np.float32)
        output = validate_db_output(output, (height, width), METHOD_NAME)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.time() - t1

    weight_np = weight.cpu().numpy().astype(np.float32)

    # ========== 生成图标题(所有参数简写) ==========
    title_parts = [
        f"F{args.f_number}",
        f"{args.window.upper()}",
        f"{args.interp[:3]}",
        f"subarray_size{args.lmax_ratio}",
        f"min{args.min_subarray_len}",
        f"g{args.gamma}",
        f"P{args.clip_percentile:.0f}",
    ]
    if len(selected_angles) == 1:
        title_parts.append(f"{np.degrees(selected_angles[0]):.1f}°")
    else:
        title_parts.append(f"{len(selected_angles)}A")
    if args.dynamic_aperture:
        title_parts.append("Dyn")
    else:
        title_parts.append("Full")
    if args.depth_smooth_rows > 1:
        title_parts.append(f"Sm{args.depth_smooth_rows}")
    title_params = " | ".join(title_parts)

    out_name = METHOD_NAME

    np.save(os.path.join(OUTPUT_DIR, f"{out_name}.npy"), output)
    np.save(os.path.join(OUTPUT_DIR, f"{out_name}_weight.npy"), weight_np)
    dynamic_l_np = dynamic_l.cpu().numpy()
    np.save(
        os.path.join(OUTPUT_DIR, f"{out_name}_subarray_length.npy"),
        dynamic_l_np,
    )

    save_figure(
        output,
        extent_mm,
        os.path.join(OUTPUT_DIR, f"{out_name}.png"),
        title_params,
        dr=args.dr,
    )

    if has_gt and args.save_gt:
        save_comparison_figure(
            output,
            gt_data,
            extent_mm,
            os.path.join(OUTPUT_DIR, f"{out_name}_comparison.png"),
            title_params,
            dr=args.dr,
        )

    params = vars(args).copy()
    params["method"] = METHOD_NAME
    params["output_dir"] = args.output_dir
    params["method_dir"] = OUTPUT_DIR
    params["mv_baseline"] = baseline_path
    params["runtime_sec"] = dt
    with open(os.path.join(OUTPUT_DIR, "params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    print(f"  Weight p1/p50/p99 = {np.percentile(weight_np, [1, 50, 99])}")
    print(f"  GPU Time: {dt:.2f}s | Saved -> {out_name}")
    print(f"\nDone | Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

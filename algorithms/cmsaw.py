import os
import sys
import subprocess
import json
import importlib
import numpy as np
import torch
import time
import argparse
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from beamforming_utils import (
    aperture_window_1d,
    db_display_range,
    dynamic_aperture_channel_count,
    interpolate_multi_angle_channel_samples,
    parse_selected_angles,
    resolve_project_path,
)
from common_params import add_common_arguments, add_io_arguments

# ================= 命令行参数配置 =================
parser = argparse.ArgumentParser(description='CMSAW - Coherence-based Minimum Variance Adaptive Weighting')
add_common_arguments(parser, select_help='角度选择: center, 0,1,2(逗号分隔)')

# ---- MV参数（用于自动生成基线） ----
parser.add_argument('--mv_dl', type=float, default=0.0, help='MV 对角加载系数（自动生成基线用）')
parser.add_argument('--subarray_ratio', type=float, default=0.25, help='MV 子阵列比例（自动生成基线用）')
parser.add_argument('--temporal_win', type=int, default=9, help='MV 时间平均窗口（自动生成基线用）')
parser.add_argument('--fbss', action='store_true', default=True, help='MV 启用FBSS（自动生成基线用）')
parser.add_argument('--no_fbss', dest='fbss', action='store_false', help='MV 禁用FBSS')
# ---- CMSAW 核心参数 ----
parser.add_argument('--baseline_mv', type=str, default='mv.npy', help='MV基线结果文件路径')
parser.add_argument('--lmax_ratio', type=float, default=0.5, help='最大子阵列长度比例 Lmax = ratio × K')
parser.add_argument('--min_subarray_len', type=int, default=2, help='最小子阵列长度')
parser.add_argument('--delta_max', type=float, default=1.0, help='对角线缩放因子最大值')
parser.add_argument('--gamma', type=float, default=0.5, help='权重幂次')
parser.add_argument('--clip_percentile', type=float, default=90.0, help='权重裁剪百分位数')
parser.add_argument('--depth_smooth_rows', type=int, default=1, help='深度平滑行数 (1=禁用)')

add_io_arguments(parser, save_gt_help='保存GT对比图')
args = parser.parse_args()

METHOD_NAME = "cmsaw"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


H5_PATH = resolve_project_path(args.h5_path, PROJECT_ROOT)
BASE_OUTPUT_DIR = args.output_dir
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, METHOD_NAME)

c_global, fc_global, NUM_CHANNELS, IMG_H, IMG_W = None, None, None, None, None


# ================= 从 H5 加载数据 =================
def load_from_h5(h5_path, sample_idx=0):
    global c_global, fc_global, NUM_CHANNELS, IMG_H, IMG_W
    with h5py.File(h5_path, 'r') as hf:
        c_global = float(hf['c'][()])
        fc_global = float(hf['fc'][()])
        fs = float(hf['fs'][()])
        pitch = float(hf['pitch'][()])
        n_elem = int(hf['num_channels'][()])
        angles = hf['angles'][:].astype(np.float32)

        gt_data, has_gt = None, False
        if 'all_envdb_norm' in hf:
            gt_data = hf['all_envdb_norm'][sample_idx].astype(np.float32)
            has_gt = True

        t0_vec = hf['time_start_vector'][sample_idx].astype(np.float32)
        z_grid = hf['z_grid'][:].astype(np.float32)
        x_grid = hf['x_grid'][:].astype(np.float32)
        I_data = hf['all_multi_I'][sample_idx].astype(np.float32)
        Q_data = hf['all_multi_Q'][sample_idx].astype(np.float32)

        NUM_CHANNELS, IMG_H, IMG_W = n_elem, len(z_grid), len(x_grid)
    return c_global, fc_global, fs, pitch, n_elem, angles, t0_vec, z_grid, x_grid, I_data, Q_data, gt_data, has_gt


def print_physical_summary(c, fc, fs, pitch, n_elem, angles, t0, H, W, dz, dx,
                           depth_min, depth_max, n_files, selected_angles, has_gt, baseline_path):
    wavelength = c / fc
    pw = (n_elem - 1) * pitch
    fov_lateral = pw * 1000
    fov_depth = (depth_max - depth_min) * 1000
    
    angle_display = f"{len(selected_angles)}"
    if len(selected_angles) == 1:
        angle_display += f" ({selected_angles[0]:.1f}°)"
    
    print(f"\n{'=' * 70}")
    print(f"  CMSAW Beamforming")
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
    print(f"  Transducer   : {n_elem}ch, fc={fc/1e6:.1f}MHz, λ={wavelength*1e3:.3f}mm, pitch={pitch*1e3:.3f}mm")
    print(f"  Sampling     : fs={fs/1e6:.1f}MHz, {len(angles)} angles, t0={t0[0]*1e6 if isinstance(t0, np.ndarray) else t0*1e6:.3f}μs")
    print(f"  Grid         : {H}×{W}, dz={dz*1e3:.4f}mm, dx={dx*1e3:.4f}mm")
    print(f"  FOV          : {fov_depth:.1f}mm × {fov_lateral:.1f}mm (depth {depth_min*1e3:.1f}~{depth_max*1e3:.1f}mm)")
    print(f"{'=' * 70}")
    print(f"  F-Number     : {args.f_number}")
    print(f"  Aperture     : {'Dynamic' if args.dynamic_aperture else 'Fixed Full'}")
    print(f"  Window       : {args.window.upper()}")
    print(f"  Interp       : {args.interp}")
    print(f"  CMSAW Lmax   : {args.lmax_ratio} × K")
    print(f"  CMSAW minL   : {args.min_subarray_len}")
    print(f"  CMSAW delta  : {args.delta_max}")
    print(f"  CMSAW gamma  : {args.gamma}")
    print(f"  CMSAW clip   : {args.clip_percentile}%")
    print(f"  CMSAW smooth : {args.depth_smooth_rows} rows")
    print(f"  DR           : {args.dr} dB")
    print(f"  Output       : {OUTPUT_DIR}")
    print(f"{'=' * 70}\n")


def ensure_mv_baseline(baseline_path):
    """如果MV基线不存在，自动调用mv.py生成"""
    if not os.path.isabs(baseline_path):
        if baseline_path == "mv.npy":
            full_path = os.path.join(BASE_OUTPUT_DIR, "mv", "mv.npy")
        else:
            full_path = os.path.join(BASE_OUTPUT_DIR, baseline_path)
    else:
        full_path = baseline_path
    
    if os.path.exists(full_path):
        print(f"MV基线已存在: {full_path}")
        return full_path
    
    print(f"MV基线不存在，正在自动生成: {full_path}")
    
    mv_script = os.path.join(SCRIPT_DIR, 'mv.py')
    if not os.path.exists(mv_script):
        raise FileNotFoundError(f"找不到mv.py: {mv_script}，请确保mv.py在相同目录下")
    
    cmd = [
        sys.executable, mv_script,
        '--h5_path', H5_PATH,
        '--h5_sample_idx', str(args.h5_sample_idx),
        '--select_angles', args.select_angles,
        '--f_number', str(args.f_number),
        '--dr', str(args.dr),
        '--output_dir', BASE_OUTPUT_DIR,
    ]
    
    if args.dynamic_aperture:
        cmd.append('--dynamic_aperture')
    else:
        cmd.append('--no_dynamic_aperture')
    
    cmd.extend(['--window', args.window])
    cmd.extend(['--interp', args.interp])
    
    # MV参数
    cmd.extend(['--mv_dl', str(args.mv_dl)])
    cmd.extend(['--subarray_ratio', str(args.subarray_ratio)])
    cmd.extend(['--temporal_win', str(args.temporal_win)])
    if args.fbss:
        cmd.append('--fbss')
    else:
        cmd.append('--no_fbss')
    
    if args.tgc:
        cmd.append('--tgc')
        cmd.extend(['--tgc_alpha', str(args.tgc_alpha)])
    else:
        cmd.append('--no_tgc')
    
    print(f"执行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"MV生成失败: {result.stderr}")
        raise RuntimeError("自动生成MV基线失败")
    
    print(result.stdout)
    print("MV基线生成完成")
    return full_path


def delayed_iq(sample, angle_idx):
    I = torch.from_numpy(sample["I"][angle_idx]).to(device)
    Q = torch.from_numpy(sample["Q"][angle_idx]).to(device)
    angles = torch.from_numpy(sample["angles"][angle_idx]).to(device)
    t0 = torch.from_numpy(np.atleast_1d(sample["t0"][angle_idx])).to(device)
    z = torch.from_numpy(sample["z"]).to(device)
    x = torch.from_numpy(sample["x"]).to(device)
    n_angles, n_samples, n_channels = I.shape

    Z, X = torch.meshgrid(z, x, indexing="ij")
    elements = (torch.arange(n_channels, device=device) - (n_channels - 1) / 2.0) * sample["pitch"]
    receive = torch.sqrt((X[..., None] - elements) ** 2 + Z[..., None] ** 2)
    transmit = Z[None, ..., None] * torch.cos(angles)[:, None, None, None]
    transmit += X[None, ..., None] * torch.sin(angles)[:, None, None, None]
    tof = (transmit + receive[None]) / sample["c"]
    exact = (tof - t0[:, None, None, None]) * sample["fs"]
    valid = (exact >= 0) & (exact <= n_samples - 1)

    angle_index = torch.arange(n_angles, device=device)[:, None, None, None]
    channel_offset = torch.arange(n_channels, device=device)[None, None, None, :]

    i, q = interpolate_multi_angle_channel_samples(I, Q, exact, angle_index, channel_offset, args.interp)

    phase_rx = 2.0 * torch.pi * sample["fc"] * (receive / sample["c"])
    cos_rx = torch.cos(phase_rx)[None]
    sin_rx = torch.sin(phase_rx)[None]
    aligned_i = i * cos_rx - q * sin_rx
    aligned_q = i * sin_rx + q * cos_rx
    return torch.complex(aligned_i, aligned_q) * valid


class CMSAWBeamformerIQ:
    """
    Pack-time CMSAW teacher interface.

    CMSAW 本身是基于 MV baseline 的显示域/幅值加权方法；这里为了统一 H5 GT 打包接口，
    采用“MV 复数 IQ × CMSAW 实值权重”的定义：
      - 相位沿用 MV baseline；
      - 幅值按 CMSAW coherence 权重调整；
      - 返回 I_out, Q_out，供 pack_EPFL.py 后续统一生成 envdb。
    """

    def __init__(self, z_grid, x_grid, n_elem, pitch, c, fs, t0_all, angles_rad,
                 mv_dl=0.0, fbss=True, subarray_ratio=0.25, temporal_win=9,
                 lmax_ratio=0.5, min_subarray_len=2, delta_max=1.0,
                 gamma=0.5, clip_percentile=90.0, depth_smooth_rows=1,
                 baseline_mv=None):
        if not 0 < lmax_ratio <= 0.5 or min_subarray_len < 2:
            raise ValueError("Require lmax_ratio in (0, 0.5] and min_subarray_len >= 2")
        if not 0 <= delta_max <= 1 or not 0 < gamma <= 1 or not 0 < clip_percentile <= 100:
            raise ValueError("Require delta_max in [0,1], gamma in (0,1], clip_percentile in (0,100]")
        if temporal_win < 1 or temporal_win % 2 == 0:
            raise ValueError("temporal_win must be a positive odd integer")

        self.z_grid = np.asarray(z_grid, dtype=np.float32)
        self.x_grid = np.asarray(x_grid, dtype=np.float32)
        self.n_elem = int(n_elem)
        self.pitch = float(pitch)
        self.c = float(c)
        self.fs = float(fs)
        self.fc = float(fc_global)
        self.lmax_ratio = float(lmax_ratio)
        self.min_subarray_len = int(min_subarray_len)
        self.delta_max = float(delta_max)
        self.gamma = float(gamma)
        self.clip_percentile = float(clip_percentile)
        self.depth_smooth_rows = int(depth_smooth_rows)

        old_argv = sys.argv[:]
        try:
            sys.argv = ['mv.py']
            self.mv_module = importlib.import_module('mv')
        finally:
            sys.argv = old_argv
        self.mv_module.args = args
        self.mv_module.fc_global = self.fc
        self.mv_module.c_global = self.c
        self.mv_bf = self.mv_module.RowDynamicMVBeamformerIQ(
            self.z_grid, self.x_grid, self.n_elem, self.pitch, self.c, self.fs,
            t0_all, angles_rad,
            mv_dl=mv_dl, fbss=fbss,
            subarray_ratio=subarray_ratio, temporal_win=temporal_win,
        )

    def _weight_from_delayed_data(self, data):
        H, W, n_channels = data.shape
        x_t = torch.from_numpy(self.x_grid).to(device)
        element_x = (torch.arange(n_channels, device=device) - (n_channels - 1) / 2.0) * self.pitch
        centers = torch.argmin(torch.abs(x_t[:, None] - element_x[None]), dim=1)

        row_cache = []
        for depth in self.z_grid:
            k = dynamic_aperture_channel_count(depth, args.f_number, self.pitch, n_channels, args.dynamic_aperture)
            starts = torch.clamp(centers - k // 2, 0, n_channels - k)
            channels = starts[:, None] + torch.arange(k, device=device)[None]
            row_cache.append((k, channels, aperture_window_1d(k, args.window, device)[None]))

        sigma = torch.empty((H, W), dtype=torch.float32, device=device)
        active_rows, k_rows = [], []
        for iz, (k, channels, win) in enumerate(row_cache):
            active = torch.gather(data[iz], 1, channels) * win
            sigma[iz] = torch.std(torch.abs(active), dim=1, unbiased=False)
            active_rows.append(active)
            k_rows.append(k)

        sigma_prime = torch.pow(sigma + 1e-12, -1.0 / 3.0)
        sigma_prime = (sigma_prime - sigma_prime.min()) / (sigma_prime.max() - sigma_prime.min() + 1e-12)
        weight = torch.zeros_like(sigma)

        for iz, active in enumerate(active_rows):
            k = k_rows[iz]
            lmax = max(self.min_subarray_len, int(np.floor(k * self.lmax_ratio)))
            min_l = min(self.min_subarray_len, lmax)
            lengths = torch.floor(sigma_prime[iz] * lmax).long().clamp(min_l, lmax)
            coherent = torch.abs(active.sum(dim=1)).square()
            total = k * torch.abs(active).square().sum(dim=1)
            coherence = (coherent / (total + 1e-12)).clamp(0, 1)
            delta = torch.pow(coherence + 1e-12, sigma_prime[iz]) * self.delta_max

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
                diagonal = torch.diag_embed(torch.diagonal(rotary, dim1=-2, dim2=-1))
                matrix = torch.abs(rotary - delta[columns, None, None] * diagonal)
                weight[iz, columns] = matrix.mean(dim=(-2, -1)) / (matrix.std(dim=(-2, -1), unbiased=False) + 1e-12)

        if self.depth_smooth_rows > 1:
            rows = self.depth_smooth_rows + 1 if self.depth_smooth_rows % 2 == 0 else self.depth_smooth_rows
            coord = torch.arange(rows, device=device).float() - rows // 2
            kernel = torch.exp(-0.5 * (coord / max(rows / 4.0, 1.0)) ** 2)
            kernel = (kernel / kernel.sum()).view(1, 1, rows, 1)
            log_weight = torch.log(weight.clamp_min(1e-12))[None, None]
            log_weight = torch.nn.functional.pad(log_weight, (0, 0, rows // 2, rows // 2), mode="reflect")
            weight = torch.exp(torch.nn.functional.conv2d(log_weight, kernel)[0, 0])

        weight /= torch.median(weight.clamp_min(1e-12))
        weight.clamp_(max=torch.quantile(weight, self.clip_percentile / 100.0))
        weight.pow_(self.gamma)
        return weight

    def __call__(self, I_data, Q_data, selected_angles, t_starts, fs):
        if len(selected_angles) != 1:
            raise ValueError("CMSAWBeamformerIQ requires exactly one selected angle")

        I_mv, Q_mv = self.mv_bf(I_data, Q_data, selected_angles, t_starts, fs)
        t0 = np.asarray(t_starts, dtype=np.float32).reshape(-1)
        if t0.size == 1:
            t0 = np.repeat(t0, len(selected_angles))
        sample = {
            "I": I_data.astype(np.float32),
            "Q": Q_data.astype(np.float32),
            "t0": t0.astype(np.float32),
            "z": self.z_grid,
            "x": self.x_grid,
            "angles": np.asarray(selected_angles, dtype=np.float32),
            "fs": float(fs),
            "c": self.c,
            "fc": self.fc,
            "pitch": self.pitch,
        }
        data = delayed_iq(sample, np.arange(len(selected_angles)))[0]
        weight = self._weight_from_delayed_data(data).cpu().numpy().astype(np.float32)
        return I_mv * weight, Q_mv * weight


# ================= 保存函数 =================
def save_comparison_figure(cmsaw_db, gt_norm, extent_mm, out_path, title_str, dr=60.0):
    vmin, vmax = db_display_range(dr)
    def to_2d(arr):
        if arr.ndim == 3:
            return arr[0] if arr.shape[0] == 1 else arr[:, :, 0]
        return arr
    
    cmsaw_db, gt_norm = to_2d(cmsaw_db), to_2d(gt_norm)
    
    fig = plt.figure(figsize=(12, 8), dpi=300)
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(gt_norm, cmap='gray', vmin=0, vmax=1, extent=extent_mm, aspect='equal')
    ax1.set_title("Ground Truth", fontsize=12, pad=10)
    ax1.set_xlabel("Lateral (mm)")
    ax1.set_ylabel("Depth (mm)")

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(cmsaw_db, cmap='gray', vmin=vmin, vmax=vmax, extent=extent_mm, aspect='equal')
    ax2.set_title(f"CMSAW\n{title_str}", fontsize=9, pad=10)
    ax2.set_xlabel("Lateral (mm)")
    ax2.set_ylabel("Depth (mm)")

    cax = fig.add_subplot(gs[0, 2])
    cbar = fig.colorbar(im2, cax=cax, fraction=0.8)
    cbar.set_label("Amplitude (dB)")

    bar_length = 5.0
    bar_x = extent_mm[1] - bar_length - 2.0
    bar_y = extent_mm[2] - 2.0
    scale_bar = patches.Rectangle((bar_x, bar_y), bar_length, 0.5, color='white', zorder=5)
    ax2.add_patch(scale_bar)
    ax2.text(bar_x + bar_length / 2, bar_y - 1.0, '5 mm', color='white',
             fontsize=10, ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def save_figure(db_img, extent_mm, out_path, title_str, dr=60.0):
    vmin, vmax = db_display_range(dr)
    fig, ax = plt.subplots(figsize=(6, 8), dpi=300)
    im = ax.imshow(db_img, cmap='gray', vmin=vmin, vmax=vmax, extent=extent_mm, aspect='equal')
    ax.set_xlabel("Lateral (mm)")
    ax.set_ylabel("Depth (mm)")
    ax.set_title(f"CMSAW\n{title_str}", fontsize=9, pad=15)
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Amplitude (dB)")

    bar_length = 5.0
    bar_x = extent_mm[1] - bar_length - 2.0
    bar_y = extent_mm[2] - 2.0
    scale_bar = patches.Rectangle((bar_x, bar_y), bar_length, 0.5, color='white', zorder=5)
    ax.add_patch(scale_bar)
    ax.text(bar_x + bar_length / 2, bar_y - 1.0, '5 mm', color='white',
            fontsize=10, ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ================= 主程序 =================
def main():
    # 参数验证
    if not 0 < args.lmax_ratio <= 0.5 or args.min_subarray_len < 2:
        raise ValueError("Require lmax_ratio in (0, 0.5] and min_subarray_len >= 2")
    if not 0 <= args.delta_max <= 1 or not 0 < args.gamma <= 1 or not 0 < args.clip_percentile <= 100:
        raise ValueError("Require delta_max in [0,1], gamma in (0,1], clip_percentile in (0,100]")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ========== 自动生成MV基线 ==========
    baseline_path = ensure_mv_baseline(args.baseline_mv)

    # 加载H5数据
    c, fc, fs, pitch, n_elem, angles_all, t0_all, z_grid, x_grid, I_data, Q_data, gt_data, has_gt = load_from_h5(
        H5_PATH, args.h5_sample_idx
    )

    selected_indices, selected_angles = parse_selected_angles(angles_all, args.select_angles)
    if len(selected_indices) != 1:
        raise ValueError("CMSAW requires exactly one selected angle")
    
    I_sub, Q_sub = I_data[selected_indices], Q_data[selected_indices]
    t0_sub = t0_all[selected_indices] if isinstance(t0_all, (np.ndarray, list)) else t0_all

    H, W = len(z_grid), len(x_grid)
    dz, dx = z_grid[1] - z_grid[0], x_grid[1] - x_grid[0]
    depth_min, depth_max = z_grid[0], z_grid[-1]
    extent_mm = [x_grid[0]*1000, x_grid[-1]*1000, z_grid[-1]*1000, z_grid[0]*1000]

    # 加载MV基线
    baseline = np.load(baseline_path).astype(np.float32)
    if baseline.shape != (H, W):
        raise ValueError(f"Baseline shape {baseline.shape}, expected ({H}, {W})")

    print_physical_summary(c, fc, fs, pitch, n_elem, angles_all, t0_sub, H, W, dz, dx,
                           depth_min, depth_max, 1, selected_angles, has_gt, baseline_path)

    # 准备数据
    sample = {
        "I": I_data, "Q": Q_data, "t0": t0_all,
        "z": z_grid, "x": x_grid, "angles": angles_all,
        "fs": fs, "c": c, "fc": fc, "pitch": pitch
    }

    print(f"Processing CMSAW (Lmax={args.lmax_ratio}K, gamma={args.gamma})...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()

    with torch.no_grad():
        data = delayed_iq(sample, selected_indices)[0]  # [H, W, N]
        n_channels = data.shape[-1]
        element_x = (torch.arange(n_channels, device=device) - (n_channels - 1) / 2.0) * pitch
        centers = torch.argmin(
            torch.abs(torch.from_numpy(x_grid).to(device)[:, None] - element_x[None]), dim=1
        )
        row_cache = []
        for depth in z_grid:
            k = dynamic_aperture_channel_count(depth, args.f_number, pitch, n_channels, args.dynamic_aperture)
            starts = torch.clamp(centers - k // 2, 0, n_channels - k)
            channels = starts[:, None] + torch.arange(k, device=device)[None]
            row_cache.append((k, channels, aperture_window_1d(k, args.window, device)[None]))

        sigma = torch.empty((H, W), dtype=torch.float32, device=device)
        active_rows, k_rows = [], []

        for iz, (k, channels, win) in enumerate(row_cache):
            active = torch.gather(data[iz], 1, channels) * win
            sigma[iz] = torch.std(torch.abs(active), dim=1, unbiased=False)
            active_rows.append(active)
            k_rows.append(k)

        sigma_prime = torch.pow(sigma + 1e-12, -1.0 / 3.0)
        sigma_prime = (sigma_prime - sigma_prime.min()) / (sigma_prime.max() - sigma_prime.min() + 1e-12)
        weight = torch.zeros_like(sigma)
        dynamic_l = torch.zeros((H, W), dtype=torch.int16, device=device)

        for iz, active in enumerate(active_rows):
            k = k_rows[iz]
            lmax = max(args.min_subarray_len, int(np.floor(k * args.lmax_ratio)))
            min_l = min(args.min_subarray_len, lmax)
            lengths = torch.floor(sigma_prime[iz] * lmax).long().clamp(min_l, lmax)
            dynamic_l[iz] = lengths.to(torch.int16)
            coherent = torch.abs(active.sum(dim=1)).square()
            total = k * torch.abs(active).square().sum(dim=1)
            coherence = (coherent / (total + 1e-12)).clamp(0, 1)
            delta = torch.pow(coherence + 1e-12, sigma_prime[iz]) * args.delta_max

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
                diagonal = torch.diag_embed(torch.diagonal(rotary, dim1=-2, dim2=-1))
                matrix = torch.abs(rotary - delta[columns, None, None] * diagonal)
                weight[iz, columns] = matrix.mean(dim=(-2, -1)) / (matrix.std(dim=(-2, -1), unbiased=False) + 1e-12)

        if args.depth_smooth_rows > 1:
            rows = args.depth_smooth_rows + 1 if args.depth_smooth_rows % 2 == 0 else args.depth_smooth_rows
            coord = torch.arange(rows, device=device).float() - rows // 2
            kernel = torch.exp(-0.5 * (coord / max(rows / 4.0, 1.0)) ** 2)
            kernel = (kernel / kernel.sum()).view(1, 1, rows, 1)
            log_weight = torch.log(weight.clamp_min(1e-12))[None, None]
            log_weight = torch.nn.functional.pad(log_weight, (0, 0, rows // 2, rows // 2), mode="reflect")
            weight = torch.exp(torch.nn.functional.conv2d(log_weight, kernel)[0, 0])

        weight /= torch.median(weight.clamp_min(1e-12))
        weight.clamp_(max=torch.quantile(weight, args.clip_percentile / 100.0))
        weight.pow_(args.gamma)
        baseline_env_sq = torch.from_numpy(np.power(10.0, baseline / 10.0)).to(device)
        output_env_sq = baseline_env_sq * (weight ** 2)
        output_env_sq /= (output_env_sq.max() + 1e-24)
        output = (10.0 * torch.log10(output_env_sq + 1e-24)).cpu().numpy().astype(np.float32)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.time() - t1

    weight_np = weight.cpu().numpy().astype(np.float32)

    # ========== 生成图标题（所有参数简写） ==========
    title_parts = [
        f"F{args.f_number}",
        f"{args.window.upper()}",
        f"{args.interp[:3]}",
        f"L{args.lmax_ratio}",
        f"min{args.min_subarray_len}",
        f"g{args.gamma}",
        f"P{args.clip_percentile:.0f}",
    ]
    if len(selected_angles) == 1:
        title_parts.append(f"{selected_angles[0]:.1f}°")
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

    np.save(os.path.join(OUTPUT_DIR, f'{out_name}.npy'), output)
    np.save(os.path.join(OUTPUT_DIR, f'{out_name}_weight.npy'), weight_np)
    np.save(os.path.join(OUTPUT_DIR, f'{out_name}_subarray_length.npy'), dynamic_l.cpu().numpy())

    save_figure(output, extent_mm,
                os.path.join(OUTPUT_DIR, f'{out_name}.png'),
                title_params,
                dr=args.dr)

    if has_gt and args.save_gt:
        save_comparison_figure(output, gt_data, extent_mm,
                               os.path.join(OUTPUT_DIR, f'{out_name}_comparison.png'),
                               title_params,
                               dr=args.dr)

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


if __name__ == '__main__':
    main()

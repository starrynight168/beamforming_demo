import os
import sys
import subprocess
import json
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

# ================= 命令行参数配置 =================
parser = argparse.ArgumentParser(description='CMSAW - Coherence-based Minimum Variance Adaptive Weighting')
parser.add_argument('--select_angles', type=str, default='center', help='角度选择: center, 0,1,2(逗号分隔)')
parser.add_argument('--f_number', type=float, default=1.5, help='F-Number')
parser.add_argument('--dr', type=float, default=60.0, help='动态范围 dB')
parser.add_argument('--dynamic_aperture', action='store_true', default=True, help='启用动态孔径')
parser.add_argument('--no_dynamic_aperture', dest='dynamic_aperture', action='store_false', help='禁用动态孔径')
parser.add_argument('--window', type=str, default='rect', choices=['hann', 'rect', 'tukey'], help='窗函数')
parser.add_argument('--interp', type=str, default='cubic', choices=['linear', 'nearest', 'cubic'], help='插值方式')

# ---- MV参数（用于自动生成基线） ----
parser.add_argument('--mv_dl', type=float, default=0.0, help='MV 对角加载系数（自动生成基线用）')
parser.add_argument('--subarray_ratio', type=float, default=0.25, help='MV 子阵列比例（自动生成基线用）')
parser.add_argument('--temporal_win', type=int, default=9, help='MV 时间平均窗口（自动生成基线用）')
parser.add_argument('--fbss', action='store_true', default=True, help='MV 启用FBSS（自动生成基线用）')
parser.add_argument('--no_fbss', dest='fbss', action='store_false', help='MV 禁用FBSS')
parser.add_argument('--tgc', action='store_true', default=True, help='启用 TGC（自动生成基线用）')
parser.add_argument('--no_tgc', dest='tgc', action='store_false', help='禁用 TGC')
parser.add_argument('--tgc_alpha', type=float, default=0.5, help='TGC 衰减系数')

# ---- CMSAW 核心参数 ----
parser.add_argument('--baseline_mv', type=str, default='mv.npy', help='MV基线结果文件路径')
parser.add_argument('--lmax_ratio', type=float, default=0.5, help='最大子阵列长度比例 Lmax = ratio × K')
parser.add_argument('--min_subarray_len', type=int, default=2, help='最小子阵列长度')
parser.add_argument('--delta_max', type=float, default=1.0, help='对角线缩放因子最大值')
parser.add_argument('--gamma', type=float, default=0.5, help='权重幂次')
parser.add_argument('--clip_percentile', type=float, default=90.0, help='权重裁剪百分位数')
parser.add_argument('--depth_smooth_rows', type=int, default=1, help='深度平滑行数 (1=禁用)')

parser.add_argument('--h5_path', type=str, default='data/simulation.h5', help='H5数据文件路径')
parser.add_argument('--h5_sample_idx', type=int, default=0, help='H5样本索引')
parser.add_argument('--output_dir', type=str, default='results', help='输出目录')
parser.add_argument('--save_gt', action='store_true', default=False, help='保存GT对比图')
args = parser.parse_args()

METHOD_NAME = "cmsaw"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def resolve_input_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


H5_PATH = resolve_input_path(args.h5_path)
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


def parse_selected_angles(angles, select_str):
    select_str = select_str.strip().lower()
    if select_str == 'center':
        ci = int(np.argmin(np.abs(angles)))
        return np.array([ci]), np.array([angles[ci]])
    else:
        indices = [int(x) for x in select_str.split(',')]
        indices = [i for i in indices if 0 <= i < len(angles)]
        if len(indices) == 0:
            ci = int(np.argmin(np.abs(angles)))
            return np.array([ci]), np.array([angles[ci]])
        return np.array(indices), angles[indices]


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


def aperture_window(k):
    if args.window == "rect":
        return torch.ones(k, device=device)
    x = torch.linspace(-1.0, 1.0, k, device=device)
    if args.window == "hann":
        return 0.5 * (1.0 + torch.cos(torch.pi * x))
    beta = 0.25
    weight = torch.ones_like(x)
    edge = x.abs() > 1.0 - beta
    weight[edge] = 0.5 * (1.0 + torch.cos(torch.pi * (x[edge].abs() - (1.0 - beta)) / beta))
    return weight


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

    flat_i, flat_q = I.reshape(-1), Q.reshape(-1)
    angle_offset = torch.arange(n_angles, device=device)[:, None, None, None] * n_samples * n_channels
    channel_offset = torch.arange(n_channels, device=device)[None, None, None, :]

    if args.interp == "nearest":
        index0 = torch.round(exact).long().clamp(0, n_samples - 1)
        linear0 = angle_offset + index0 * n_channels + channel_offset
        i = flat_i[linear0]
        q = flat_q[linear0]
    elif args.interp == "linear":
        index0 = torch.floor(exact).long().clamp(0, n_samples - 1)
        index1 = (index0 + 1).clamp(0, n_samples - 1)
        frac = exact - torch.floor(exact)
        linear0 = angle_offset + index0 * n_channels + channel_offset
        linear1 = angle_offset + index1 * n_channels + channel_offset
        i = flat_i[linear0] * (1.0 - frac) + flat_i[linear1] * frac
        q = flat_q[linear0] * (1.0 - frac) + flat_q[linear1] * frac
    else: # cubic
        index0 = torch.floor(exact).long().clamp(0, n_samples - 1)
        index1 = (index0 + 1).clamp(0, n_samples - 1)
        index_m1 = (index0 - 1).clamp(0, n_samples - 1)
        index2 = (index0 + 2).clamp(0, n_samples - 1)
        frac = exact - torch.floor(exact)
        
        linear0 = angle_offset + index0 * n_channels + channel_offset
        linear1 = angle_offset + index1 * n_channels + channel_offset
        linear_m1 = angle_offset + index_m1 * n_channels + channel_offset
        linear2 = angle_offset + index2 * n_channels + channel_offset
        
        w2, w3 = frac.square(), frac.pow(3)
        cm1 = -0.5 * w3 + w2 - 0.5 * frac
        c0 = 1.5 * w3 - 2.5 * w2 + 1.0
        c1 = -1.5 * w3 + 2.0 * w2 + 0.5 * frac
        c2 = 0.5 * w3 - 0.5 * w2
        
        i = flat_i[linear_m1] * cm1 + flat_i[linear0] * c0 + flat_i[linear1] * c1 + flat_i[linear2] * c2
        q = flat_q[linear_m1] * cm1 + flat_q[linear0] * c0 + flat_q[linear1] * c1 + flat_q[linear2] * c2

    phase = 2.0 * torch.pi * sample["fc"] * tof
    aligned_i = i * torch.cos(phase) - q * torch.sin(phase)
    aligned_q = i * torch.sin(phase) + q * torch.cos(phase)
    return torch.complex(aligned_i, aligned_q) * valid


# ================= 保存函数 =================
def save_comparison_figure(cmsaw_db, gt_norm, extent_mm, out_path, title_str, dr=60.0):
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
    im2 = ax2.imshow(cmsaw_db, cmap='gray', vmin=-dr, vmax=0, extent=extent_mm, aspect='equal')
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
    fig, ax = plt.subplots(figsize=(6, 8), dpi=300)
    im = ax.imshow(db_img, cmap='gray', vmin=-dr, vmax=0, extent=extent_mm, aspect='equal')
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
            if args.dynamic_aperture:
                k = min(max(int(depth / (args.f_number * pitch)) + 1, 4), n_channels)
            else:
                k = n_channels
            starts = torch.clamp(centers - k // 2, 0, n_channels - k)
            channels = starts[:, None] + torch.arange(k, device=device)[None]
            row_cache.append((k, channels, aperture_window(k)[None]))

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

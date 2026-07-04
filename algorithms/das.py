import os
import numpy as np
import torch
import time
import argparse
import json
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# ================= 命令行参数配置 =================
parser = argparse.ArgumentParser(description='DAS Beamforming - Baseband IQ Data (H5)')
parser.add_argument('--select_angles', type=str, default='center', help='角度选择: all, center, N(数量)')
parser.add_argument('--f_number', type=float, default=1.5, help='F-Number')
parser.add_argument('--dr', type=float, default=60.0, help='动态范围 dB')
parser.add_argument('--dynamic_aperture', action='store_true', default=True, help='启用动态孔径')
parser.add_argument('--no_dynamic_aperture', dest='dynamic_aperture', action='store_false', help='禁用动态孔径')
parser.add_argument('--tgc', action='store_true', default=True, help='启用 TGC')
parser.add_argument('--no_tgc', dest='tgc', action='store_false', help='禁用 TGC')
parser.add_argument('--tgc_alpha', type=float, default=0.5, help='TGC 衰减系数')
parser.add_argument('--window', type=str, default='rect', choices=['hann', 'rect', 'tukey'], help='窗函数')
parser.add_argument('--interp', type=str, default='cubic', choices=['linear', 'nearest', 'cubic'], help='插值方式')
parser.add_argument('--h5_path', type=str, default='data/simulation.h5', help='H5 数据文件路径')
parser.add_argument('--h5_sample_idx', type=int, default=0, help='H5 样本索引')
parser.add_argument('--output_dir', type=str, default='results', help='输出目录')
parser.add_argument('--save_gt', action='store_true', default=False, help='保存GT图像')
args = parser.parse_args()

METHOD_NAME = 'das'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def resolve_input_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


H5_PATH = resolve_input_path(args.h5_path)
OUTPUT_DIR = os.path.join(args.output_dir, METHOD_NAME)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    if select_str == 'all':
        return np.arange(len(angles)), angles
    elif select_str == 'center':
        ci = int(np.argmin(np.abs(angles)))
        return np.array([ci]), np.array([angles[ci]])
    elif select_str.isdigit():
        K = max(1, min(int(select_str), len(angles)))
        if K == 1:
            ci = int(np.argmin(np.abs(angles)))
            return np.array([ci]), np.array([angles[ci]])
        sorted_indices = np.argsort(angles)
        sel = np.linspace(0, len(angles) - 1, K, dtype=int)
        indices = sorted_indices[sel]
        return indices, angles[indices]
    else:
        indices = [int(x) for x in select_str.split(',')]
        indices = [i for i in indices if 0 <= i < len(angles)]
        return np.array(indices), angles[indices]


def print_physical_summary(c, fc, fs, pitch, n_elem, angles, t0, H, W, dz, dx,
                           depth_min, depth_max, n_files, selected_angles, has_gt):
    wavelength = c / fc
    pw = (n_elem - 1) * pitch
    fov_lateral = pw * 1000
    fov_depth = (depth_max - depth_min) * 1000
    
    angle_display = f"{len(selected_angles)}"
    if len(selected_angles) == 1:
        angle_display += f" ({selected_angles[0]:.1f}°)"
    
    print(f"\n{'=' * 70}")
    print(f"  DAS Beamforming")
    print(f"{'=' * 70}")
    print(f"  H5 file      : {H5_PATH} (sample {args.h5_sample_idx})")
    print(f"  GT           : {'Available' if has_gt else 'Not available'}")
    print(f"  Angles       : {angle_display}")
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
    print(f"  TGC          : {'Enabled' if args.tgc else 'Disabled'}")
    if args.tgc:
        print(f"  TGC Alpha    : {args.tgc_alpha} dB/MHz/cm")
    print(f"  DR           : {args.dr} dB")
    print(f"  Output       : {OUTPUT_DIR}")
    print(f"{'=' * 70}\n")


def build_window(dx, half_a, window_type):
    x_norm = dx.abs() / (half_a + 1e-9)
    if window_type == 'hann':
        win = 0.5 * (1.0 + torch.cos(torch.pi * dx / (half_a + 1e-9)))
        win = win * (dx.abs() <= half_a).float()
    elif window_type == 'rect':
        win = (dx.abs() <= half_a).float()
    elif window_type == 'tukey':
        beta = 0.25
        win = torch.ones_like(x_norm)
        transition_mask = (x_norm > (1.0 - beta)) & (x_norm <= 1.0)
        val = 0.5 * (1.0 + torch.cos(torch.pi * (x_norm - (1.0 - beta)) / beta))
        win[transition_mask] = val[transition_mask]
        win[x_norm > 1.0] = 0.0
    return win





class DASBeamformerIQ:
    def __init__(self, z_grid, x_grid, n_elem, pitch, c, fs, t0_all, angles_rad):
        self.H, self.W, self.N = len(z_grid), len(x_grid), n_elem
        self.x_grid = torch.from_numpy(x_grid).float().to(device)
        self.z_grid = torch.from_numpy(z_grid).float().to(device)
        self.angles_rad = torch.from_numpy(angles_rad).float().to(device)
        self.pitch, self.fs = pitch, fs
        self.sc = fs / c

        self.ep = torch.linspace(-(n_elem-1)/2*pitch, (n_elem-1)/2*pitch, n_elem, device=device)
        X, Z = torch.meshgrid(self.x_grid, self.z_grid, indexing='xy')
        self.X, self.Z = X, Z
        self.drs = torch.sqrt((X[..., None] - self.ep)**2 + Z[..., None]**2) * self.sc

        dx = X[..., None] - self.ep
        if args.dynamic_aperture:
            half_a = Z[..., None] / (2 * args.f_number)
        else:
            half_a = torch.full_like(Z[..., None], (n_elem - 1) * pitch / 2)
        win = build_window(dx, half_a, args.window)
        self.window = win / (win.sum(-1, keepdim=True) + 1e-9)
        self.ch = torch.arange(n_elem, device=device).view(1, 1, -1)

    def __call__(self, I_data, Q_data, selected_angles, t_starts, fs):
        n_a = I_data.shape[0]
        n_s = I_data.shape[1]
        max_sample = float(n_s - 2)

        I_t = torch.from_numpy(I_data.astype(np.float32)).to(device)
        Q_t = torch.from_numpy(Q_data.astype(np.float32)).to(device)
        cos_a = torch.from_numpy(np.cos(selected_angles).astype(np.float32)).to(device)
        sin_a = torch.from_numpy(np.sin(selected_angles).astype(np.float32)).to(device)

        t_starts_arr = np.asarray(t_starts, dtype=np.float32).reshape(-1)
        if t_starts_arr.size == 1:
            t_starts_arr = np.repeat(t_starts_arr, n_a)
        t_starts_t = torch.from_numpy(t_starts_arr[:n_a]).to(device) * fs

        I_beam_sum = torch.zeros((self.H, self.W), dtype=torch.float32, device=device)
        Q_beam_sum = torch.zeros((self.H, self.W), dtype=torch.float32, device=device)
        tx_z = self.Z * self.sc
        tx_x = self.X * self.sc
        weights = self.window
        ch = self.ch

        # 1. 预计算接收端相位旋转因子 (在所有角度循环中完全相同)
        phi_rx = 2.0 * np.pi * fc_global * (self.drs / fs)
        cos_rx = torch.cos(phi_rx)
        sin_rx = torch.sin(phi_rx)

        for i in range(n_a):
            sample_no_t0 = (tx_z * cos_a[i] + tx_x * sin_a[i]).unsqueeze(-1) + self.drs
            sample = sample_no_t0 - t_starts_t[i]
            valid = (sample >= 0) & (sample < n_s - 1)
            sample.clamp_(0.0, max_sample)

            if args.interp == 'nearest':
                idx = sample.round().long().clamp(0, n_s - 1)
                I_center = I_t[i][idx, ch]
                Q_center = Q_t[i][idx, ch]
            elif args.interp == 'cubic':
                idx0 = sample.floor().long()
                frac = sample - idx0.float()
                frac2 = frac * frac
                frac3 = frac2 * frac
                c_m1 = -0.5 * frac3 + frac2 - 0.5 * frac
                c_0 = 1.5 * frac3 - 2.5 * frac2 + 1.0
                c_1 = -1.5 * frac3 + 2.0 * frac2 + 0.5 * frac
                c_2 = 0.5 * frac3 - 0.5 * frac2
                idx_m1 = torch.clamp(idx0 - 1, 0, n_s - 1)
                idx_0 = torch.clamp(idx0, 0, n_s - 1)
                idx_1 = torch.clamp(idx0 + 1, 0, n_s - 1)
                idx_2 = torch.clamp(idx0 + 2, 0, n_s - 1)
                I_angle = I_t[i]
                Q_angle = Q_t[i]
                I_center = (
                    I_angle[idx_m1, ch] * c_m1
                    + I_angle[idx_0, ch] * c_0
                    + I_angle[idx_1, ch] * c_1
                    + I_angle[idx_2, ch] * c_2
                )
                Q_center = (
                    Q_angle[idx_m1, ch] * c_m1
                    + Q_angle[idx_0, ch] * c_0
                    + Q_angle[idx_1, ch] * c_1
                    + Q_angle[idx_2, ch] * c_2
                )
            else:
                idx0 = sample.floor().long()
                frac = sample - idx0.float()
                I_angle = I_t[i]
                Q_angle = Q_t[i]
                I_center = I_angle[idx0, ch] * (1.0 - frac) + I_angle[idx0 + 1, ch] * frac
                Q_center = Q_angle[idx0, ch] * (1.0 - frac) + Q_angle[idx0 + 1, ch] * frac

            valid_w = valid.float() * weights

            # 2. 接收端相位旋转并求和 (3D -> 2D)
            I_rx = I_center * cos_rx - Q_center * sin_rx
            Q_rx = I_center * sin_rx + Q_center * cos_rx
            I_sum = (I_rx * valid_w).sum(dim=-1)
            Q_sum = (Q_rx * valid_w).sum(dim=-1)

            # 3. 发射端相位旋转 (2D)
            phi_tx = 2.0 * np.pi * fc_global * ((tx_z * cos_a[i] + tx_x * sin_a[i]) / fs)
            cos_tx = torch.cos(phi_tx)
            sin_tx = torch.sin(phi_tx)
            I_aligned = I_sum * cos_tx - Q_sum * sin_tx
            Q_aligned = I_sum * sin_tx + Q_sum * cos_tx

            I_beam_sum.add_(I_aligned)
            Q_beam_sum.add_(Q_aligned)

        I_out = I_beam_sum / n_a
        Q_out = Q_beam_sum / n_a
        return I_out.cpu().numpy(), Q_out.cpu().numpy()


# ================= 保存函数 =================
def save_comparison_figure(das_db, gt_norm, extent_mm, out_path, title_str, dr=60.0):
    def to_2d(arr):
        if arr.ndim == 3:
            return arr[0] if arr.shape[0] == 1 else arr[:, :, 0]
        return arr
    
    das_db, gt_norm = to_2d(das_db), to_2d(gt_norm)
    
    fig = plt.figure(figsize=(12, 8), dpi=300)
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(gt_norm, cmap='gray', vmin=0, vmax=1, extent=extent_mm, aspect='equal')
    ax1.set_title("Ground Truth", fontsize=12, pad=10)
    ax1.set_xlabel("Lateral (mm)")
    ax1.set_ylabel("Depth (mm)")

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(das_db, cmap='gray', vmin=-dr, vmax=0, extent=extent_mm, aspect='equal')
    ax2.set_title(f"DAS\n{title_str}", fontsize=9, pad=10)
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
    ax.set_title(f"DAS\n{title_str}", fontsize=9, pad=15)
    
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    c, fc, fs, pitch, n_elem, angles_all, t0_all, z_grid, x_grid, I_data, Q_data, gt_data, has_gt = load_from_h5(
        H5_PATH, args.h5_sample_idx
    )

    selected_indices, selected_angles = parse_selected_angles(angles_all, args.select_angles)
    I_sub, Q_sub = I_data[selected_indices], Q_data[selected_indices]
    t0_sub = t0_all[selected_indices] if isinstance(t0_all, (np.ndarray, list)) else t0_all

    H, W = len(z_grid), len(x_grid)
    dz, dx = z_grid[1] - z_grid[0], x_grid[1] - x_grid[0]
    depth_min, depth_max = z_grid[0], z_grid[-1]
    extent_mm = [x_grid[0]*1000, x_grid[-1]*1000, z_grid[-1]*1000, z_grid[0]*1000]

    print_physical_summary(c, fc, fs, pitch, n_elem, angles_all, t0_sub, H, W, dz, dx,
                           depth_min, depth_max, 1, selected_angles, has_gt)

    bf = DASBeamformerIQ(z_grid, x_grid, n_elem, pitch, c, fs, t0_sub, angles_all)

    if args.tgc:
        tgc = 10 ** (args.tgc_alpha * (fc / 1e6) * (z_grid * 100) * 2.0 / 20.0)
    else:
        tgc = np.ones_like(z_grid)

    print(f"Processing DAS...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    I, Q = bf(I_sub, Q_sub, selected_angles, t0_sub, fs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.time() - t1

    env = (I**2 + Q**2) * (tgc[:, None] ** 2)
    env /= env.max() + 1e-24
    das_db = 10 * np.log10(env + 1e-24)

    # ========== 生成图标题（所有参数简写） ==========
    title_parts = [
        f"F{args.f_number}",
        f"{args.window.upper()}",
        f"{args.interp[:3]}",
    ]
    if len(selected_angles) == 1:
        title_parts.append(f"{selected_angles[0]:.1f}°")
    else:
        title_parts.append(f"{len(selected_angles)}A")
    if args.tgc:
        title_parts.append(f"TGC{args.tgc_alpha}")
    else:
        title_parts.append("noTGC")
    if args.dynamic_aperture:
        title_parts.append("Dyn")
    else:
        title_parts.append("Full")
    title_params = " | ".join(title_parts)

    out_name = METHOD_NAME

    np.save(os.path.join(OUTPUT_DIR, f'{out_name}.npy'), das_db)
    
    save_figure(das_db, extent_mm,
                os.path.join(OUTPUT_DIR, f'{out_name}.png'),
                title_params,
                dr=args.dr)
    
    if has_gt and args.save_gt:
        save_comparison_figure(das_db, gt_data, extent_mm,
                               os.path.join(OUTPUT_DIR, f'{out_name}_comparison.png'),
                               title_params,
                               dr=args.dr)

    params = vars(args).copy()
    params["method"] = METHOD_NAME
    params["runtime_sec"] = float(dt)
    with open(os.path.join(OUTPUT_DIR, "params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    print(f"  GPU Time: {dt:.2f}s | Saved -> {out_name}")
    print(f"\nDone | Output: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()

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
from beamforming_utils import (
    dynamic_aperture_channel_count,
    db_display_range,
    interpolate_channel_samples,
    optional_aperture_window_1d,
    parse_selected_angles,
    resolve_project_path,
    tgc_gain,
)
from common_params import add_common_arguments, add_io_arguments

# ================= 命令行参数配置 =================
parser = argparse.ArgumentParser(description='MV+FBSS Beamforming')
add_common_arguments(parser, window_help='子孔径预加权窗')

# ---- MV 核心参数 ----
parser.add_argument('--mv_dl', type=float, default=0.0, help='对角加载系数')
parser.add_argument('--fbss', action='store_true', default=True, help='启用前向-后向空间平滑')
parser.add_argument('--no_fbss', dest='fbss', action='store_false', help='禁用 FBSS')
parser.add_argument('--subarray_ratio', type=float, default=0.25, help='子阵列比例 L = K * ratio')
parser.add_argument('--temporal_win', type=int, default=9, help='协方差时间平均窗口，必须为正奇数；1 表示关闭')
add_io_arguments(parser)
args = parser.parse_args()
if args.temporal_win < 1 or args.temporal_win % 2 == 0:
    parser.error('--temporal_win must be a positive odd integer')

METHOD_NAME = 'mv'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


H5_PATH = resolve_project_path(args.h5_path, PROJECT_ROOT)
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
    print(f"  MV Beamforming")
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
    print(f"  FBSS         : {'Enabled' if args.fbss else 'Disabled'}")
    print(f"  DL factor    : {args.mv_dl}")
    print(f"  Subarray     : {args.subarray_ratio}")
    print(f"  Temporal win : {args.temporal_win}")
    print(f"  Window       : {args.window.upper()}")
    print(f"  Interp       : {args.interp}")
    print(f"  TGC          : {'Enabled' if args.tgc else 'Disabled'}")
    if args.tgc:
        print(f"  TGC Alpha    : {args.tgc_alpha} dB/MHz/cm")
    print(f"  DR           : {args.dr} dB")
    print(f"  Output       : {OUTPUT_DIR}")
    print(f"{'=' * 70}\n")


class RowDynamicMVBeamformerIQ:
    """
    标准 MV + FBSS 自适应波束形成器
    固定使用 positive IQ 相位补偿
    """
    def __init__(self, z_grid, x_grid, n_elem, pitch, c, fs, t0_all, angles_rad,
                 dl_factor=0.01, use_fbss=True, subarray_ratio=0.5,
                 temporal_win=1):
        self.H, self.W, self.N = len(z_grid), len(x_grid), n_elem
        self.x_grid = torch.from_numpy(x_grid).float().to(device)
        self.z_grid = torch.from_numpy(z_grid).float().to(device)
        self.angles_rad = torch.from_numpy(angles_rad).float().to(device)
        self.pitch, self.fs, self.sc = pitch, fs, fs / c
        self.dl_factor = dl_factor
        self.use_fbss = use_fbss
        self.subarray_ratio = subarray_ratio
        self.temporal_win = temporal_win

        # 预计算每个横向像素对应的最近阵元索引
        lateral_channel = (torch.arange(self.N, device=device).float() - (self.N - 1) / 2.0) * self.pitch
        self.c_idx = torch.argmin(torch.abs(self.x_grid.unsqueeze(1) - lateral_channel.unsqueeze(0)), dim=1)
        X, Z = torch.meshgrid(self.x_grid, self.z_grid, indexing='xy')
        self.X, self.Z = X, Z
        self.drs = torch.sqrt((X[..., None] - lateral_channel) ** 2 + Z[..., None] ** 2) * self.sc
        self.ch = torch.arange(self.N, device=device).view(1, self.N, 1)
        self.row_cache = self._build_row_cache()

    def _aperture_window(self, k):
        return optional_aperture_window_1d(k, args.window, device)

    def _build_row_cache(self):
        cache = []
        for hz in range(self.H):
            depth = float(self.z_grid[hz].item())
            k = dynamic_aperture_channel_count(depth, args.f_number, self.pitch, self.N, args.dynamic_aperture)
            l = max(int(k * self.subarray_ratio), 2)
            m = k - l + 1
            idx_start = torch.clamp(self.c_idx - k // 2, 0, self.N - k)
            ch_idx = idx_start.unsqueeze(1) + torch.arange(k, device=device).unsqueeze(0)
            cache.append({
                'K': k,
                'L': l,
                'M': m,
                'ch_idx_t': ch_idx.unsqueeze(-1).expand(-1, -1, self.temporal_win),
                'window': self._aperture_window(k),
                'eye': torch.eye(l, dtype=torch.complex64, device=device).unsqueeze(0),
                'ones': torch.ones((self.W, l, 1), dtype=torch.complex64, device=device),
            })
        return cache

    def __call__(self, I_data, Q_data, selected_angles, t_starts, fs):
        n_a = I_data.shape[0]
        I_t = torch.from_numpy(I_data.astype(np.float32)).to(device)
        Q_t = torch.from_numpy(Q_data.astype(np.float32)).to(device)
        n_s = I_t.shape[1]

        half_win = self.temporal_win // 2
        offsets = torch.arange(-half_win, half_win + 1, device=device)
        offsets_f = offsets.float().view(1, 1, -1)
        cos_a = torch.from_numpy(np.cos(selected_angles).astype(np.float32)).to(device)
        sin_a = torch.from_numpy(np.sin(selected_angles).astype(np.float32)).to(device)
        t_starts_arr = np.asarray(t_starts, dtype=np.float32).reshape(-1)
        if t_starts_arr.size == 1:
            t_starts_arr = np.repeat(t_starts_arr, n_a)
        t_starts_t = torch.from_numpy(t_starts_arr[:n_a]).to(device) * fs
        max_sample = float(n_s - 2)

        I_beam_sum = torch.zeros((self.H, self.W), dtype=torch.float32, device=device)
        Q_beam_sum = torch.zeros((self.H, self.W), dtype=torch.float32, device=device)

        # 1. 预计算接收端相位旋转因子 (全网格 3D)
        phi_rx_global = 2.0 * np.pi * fc_global * (self.drs / fs)
        cos_rx_global = torch.cos(phi_rx_global)
        sin_rx_global = torch.sin(phi_rx_global)

        for i in range(n_a):
            I_angle = I_t[i]
            Q_angle = Q_t[i]

            for hz in range(self.H):
                tx_row = self.Z[hz] * self.sc * cos_a[i] + self.X[hz] * self.sc * sin_a[i]
                sample_no_t0 = tx_row.unsqueeze(-1) + self.drs[hz]
                sample_center = sample_no_t0 - t_starts_t[i]
                valid_mask_center = ((sample_center >= 0) & (sample_center < n_s - 1)).float().unsqueeze(-1)
                sample = sample_center.unsqueeze(-1) + offsets_f
                sample.clamp_(0.0, max_sample)

                I_samples, Q_samples = interpolate_channel_samples(I_angle, Q_angle, sample, self.ch, args.interp)

                # 2. 接收端相位旋转 (3D)
                cos_rx = cos_rx_global[hz].unsqueeze(-1)
                sin_rx = sin_rx_global[hz].unsqueeze(-1)
                I_rx = I_samples * cos_rx - Q_samples * sin_rx
                Q_rx = I_samples * sin_rx + Q_samples * cos_rx

                X_flat = torch.complex(I_rx, Q_rx) * valid_mask_center
                row = self.row_cache[hz]
                K, L, M = row['K'], row['L'], row['M']

                X_active = torch.gather(X_flat, 1, row['ch_idx_t'])  # [W, K, T]
                if row['window'] is not None:
                    X_active = X_active * row['window'].view(1, K, 1)

                # ========== 空间平滑 (前向) ==========
                X_sub = X_active.unfold(1, L, 1).permute(0, 3, 1, 2)  # [W, L, M, T]
                snapshots = M * self.temporal_win
                X_snap = X_sub.reshape(self.W, L, snapshots)
                R_forward = torch.matmul(X_snap, X_snap.mH) / snapshots  # [W, L, L]

                # ========== 前向-后向空间平滑 (FBSS) ==========
                if self.use_fbss:
                    R = 0.5 * (R_forward + R_forward.conj().flip(-1, -2))
                else:
                    R = R_forward

                # ========== 对角加载 ==========
                trace = R.diagonal(dim1=-2, dim2=-1).real.sum(-1)  # [W]
                R_dl = R + (self.dl_factor / L) * trace.view(self.W, 1, 1) * row['eye']

                # ========== 标准 MVDR 求解 ==========
                ones_L = row['ones']

                try:
                    v = torch.linalg.solve(R_dl, ones_L)  # [W, L, 1]
                except RuntimeError:
                    v = torch.linalg.pinv(R_dl) @ ones_L

                denom = torch.matmul(ones_L.mH, v)  # [W, 1, 1]
                w = v / (denom + 1e-12)              # [W, L, 1]  归一化 MV 权重

                # ========== 合成输出 ==========
                t0 = self.temporal_win // 2
                X_sub_center = X_sub[:, :, :, t0]              # [W, L, M]
                X_mean = X_sub_center.mean(dim=2).unsqueeze(-1)  # [W, L, 1]
                y_row_rx = torch.matmul(w.mH, X_mean)             # [W, 1, 1]

                # 3. 发射端相位旋转 (2D)
                phi_tx = 2.0 * np.pi * fc_global * (tx_row / fs)
                cos_tx = torch.cos(phi_tx).view(self.W, 1, 1)
                sin_tx = torch.sin(phi_tx).view(self.W, 1, 1)

                y_row = torch.complex(
                    y_row_rx.real * cos_tx - y_row_rx.imag * sin_tx,
                    y_row_rx.real * sin_tx + y_row_rx.imag * cos_tx
                )

                y_row_2d = y_row.squeeze(-1).squeeze(-1)
                I_beam_sum[hz].add_(y_row_2d.real)
                Q_beam_sum[hz].add_(y_row_2d.imag)

        I_out = (I_beam_sum / n_a).cpu().numpy()
        Q_out = (Q_beam_sum / n_a).cpu().numpy()
        return I_out, Q_out


# ================= 保存函数 =================
def save_comparison_figure(mv_db, gt_norm, extent_mm, out_path, title_str, dr=60.0):
    vmin, vmax = db_display_range(dr)
    def to_2d(arr):
        if arr.ndim == 3:
            return arr[0] if arr.shape[0] == 1 else arr[:, :, 0]
        return arr
    
    mv_db, gt_norm = to_2d(mv_db), to_2d(gt_norm)
    
    fig = plt.figure(figsize=(12, 8), dpi=300)
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], figure=fig)

    # Ground Truth
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(gt_norm, cmap='gray', vmin=0, vmax=1, extent=extent_mm, aspect='equal')
    ax1.set_title("Ground Truth", fontsize=12, pad=10)
    ax1.set_xlabel("Lateral (mm)")
    ax1.set_ylabel("Depth (mm)")

    # MV Result
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(mv_db, cmap='gray', vmin=vmin, vmax=vmax, extent=extent_mm, aspect='equal')
    ax2.set_title(f"MV\n{title_str}", fontsize=9, pad=10)
    ax2.set_xlabel("Lateral (mm)")
    ax2.set_ylabel("Depth (mm)")

    # Colorbar
    cax = fig.add_subplot(gs[0, 2])
    cbar = fig.colorbar(im2, cax=cax, fraction=0.8)
    cbar.set_label("Amplitude (dB)")

    # Scale bar (5mm)
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
    ax.set_title(f"MV\n{title_str}", fontsize=9, pad=15)
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Amplitude (dB)")

    # Scale bar (5mm)
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
    dz = z_grid[1] - z_grid[0]
    dx = x_grid[1] - x_grid[0]
    depth_min = z_grid[0]
    depth_max = z_grid[-1]
    extent_mm = [x_grid[0]*1000, x_grid[-1]*1000, z_grid[-1]*1000, z_grid[0]*1000]

    print_physical_summary(c, fc, fs, pitch, n_elem, angles_all, t0_sub, H, W, dz, dx,
                           depth_min, depth_max, 1, selected_angles, has_gt)

    # 初始化标准 MV + FBSS 波束合成器
    bf = RowDynamicMVBeamformerIQ(
        z_grid, x_grid, n_elem, pitch, c, fs, t0_sub, angles_all,
        dl_factor=args.mv_dl, use_fbss=args.fbss, subarray_ratio=args.subarray_ratio,
        temporal_win=args.temporal_win
    )

    # TGC
    if args.tgc:
        tgc = tgc_gain(z_grid, fc, args.tgc_alpha)
    else:
        tgc = np.ones_like(z_grid)

    print(f"Processing MV (DL={args.mv_dl})...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    I, Q = bf(I_sub, Q_sub, selected_angles, t0_sub, fs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.time() - t1

    env = (I**2 + Q**2) * (tgc[:, None] ** 2)
    env /= env.max() + 1e-24
    mv_db = 10 * np.log10(env + 1e-24)

    # ========== 生成图标题（所有参数简写） ==========
    title_parts = [
        f"F{args.f_number}",
        f"DL{args.mv_dl}",
        f"sub{args.subarray_ratio}",
        f"Tw{args.temporal_win}",
        f"{args.window.upper()}",
        f"{args.interp[:3]}",
    ]
    # 角度信息
    if len(selected_angles) == 1:
        title_parts.append(f"{selected_angles[0]:.1f}°")
    else:
        title_parts.append(f"{len(selected_angles)}A")
    # TGC
    if args.tgc:
        title_parts.append(f"TGC{args.tgc_alpha}")
    else:
        title_parts.append("noTGC")
    # 孔径
    if args.dynamic_aperture:
        title_parts.append("Dyn")
    else:
        title_parts.append("Full")
    # FBSS
    if args.fbss:
        title_parts.append("FBSS")
    else:
        title_parts.append("noFBSS")
    title_params = " | ".join(title_parts)

    out_name = METHOD_NAME

    np.save(os.path.join(OUTPUT_DIR, f'{out_name}.npy'), mv_db)

    save_figure(mv_db, extent_mm,
                os.path.join(OUTPUT_DIR, f'{out_name}.png'),
                title_params,
                dr=args.dr)

    if has_gt and args.save_gt:
        save_comparison_figure(mv_db, gt_data, extent_mm,
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



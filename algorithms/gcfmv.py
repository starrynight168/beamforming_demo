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
parser = argparse.ArgumentParser(description='GCF-MV Beamforming')
parser.add_argument('--select_angles', type=str, default='center', help='角度选择: all, center, N(数量)')
parser.add_argument('--f_number', type=float, default=1.5, help='F-Number')
parser.add_argument('--dr', type=float, default=60.0, help='动态范围 dB')
parser.add_argument('--dynamic_aperture', action='store_true', default=True, help='启用动态孔径')
parser.add_argument('--no_dynamic_aperture', dest='dynamic_aperture', action='store_false', help='禁用动态孔径')
parser.add_argument('--tgc', action='store_true', default=True, help='启用 TGC')
parser.add_argument('--no_tgc', dest='tgc', action='store_false', help='禁用 TGC')
parser.add_argument('--tgc_alpha', type=float, default=0.5, help='TGC 衰减系数')

# ---- MV 核心参数 ----
parser.add_argument('--mv_dl', type=float, default=0.0, help='对角加载系数')
parser.add_argument('--fbss', action='store_true', default=True, help='启用前向-后向空间平滑')
parser.add_argument('--no_fbss', dest='fbss', action='store_false', help='禁用 FBSS')
parser.add_argument('--subarray_ratio', type=float, default=0.25, help='子阵列比例 L = K * ratio')
parser.add_argument('--temporal_win', type=int, default=9, help='协方差时间平均窗口，必须为正奇数；1 表示关闭')
parser.add_argument('--window', type=str, default='rect', choices=['hann', 'rect', 'tukey'], help='子孔径预加权窗')
parser.add_argument('--interp', type=str, default='cubic', choices=['linear', 'nearest', 'cubic'], help='插值方式')
parser.add_argument('--gcf_low_bins', type=int, default=1, help='GCF低空间频率半宽；1表示保留中心频点及左右各1个频点')
parser.add_argument('--gcf_power', type=float, default=2.0, help='GCF后加权指数；0表示退化为MV')

parser.add_argument('--h5_path', type=str, default='data/simulation.h5')
parser.add_argument('--h5_sample_idx', type=int, default=0)
parser.add_argument('--output_dir', type=str, default='results')
parser.add_argument('--save_gt', action='store_true', default=False)
args = parser.parse_args()
if args.temporal_win < 1 or args.temporal_win % 2 == 0:
    parser.error('--temporal_win must be a positive odd integer')

METHOD_NAME = 'gcfmv'
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
    print(f"  GCF-MV Beamforming")
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
    print(f"  GCF low bins : {args.gcf_low_bins}")
    print(f"  GCF power    : {args.gcf_power}")
    print(f"  TGC          : {'Enabled' if args.tgc else 'Disabled'}")
    if args.tgc:
        print(f"  TGC Alpha    : {args.tgc_alpha} dB/MHz/cm")
    print(f"  DR           : {args.dr} dB")
    print(f"  Output       : {OUTPUT_DIR}")
    print(f"{'=' * 70}\n")


# ================= 提取函数 =================
def extract_windows_on_gpu(rf_I_pad, rf_Q_pad, t_starts, img_idx, pixel_idx,
                           fs, offsets, angle_idx, max_t_len,
                           pitch, x_grid, z_grid, selected_angles_rad, interp='cubic'):
    P, T, N = len(pixel_idx), len(offsets), NUM_CHANNELS
    device_local = rf_I_pad.device
    z_idx, x_idx = (pixel_idx // IMG_W).long(), (pixel_idx % IMG_W).long()
    depth, lateral_pixel = z_grid[z_idx], x_grid[x_idx]
    lateral_channel = (torch.arange(N, device=device_local).float() - (N - 1) / 2.0) * pitch

    dx = lateral_pixel.unsqueeze(1) - lateral_channel.unsqueeze(0)
    receive_dist = torch.sqrt(depth.unsqueeze(1)**2 + dx**2)

    A = len(angle_idx)
    theta = selected_angles_rad
    tx_dist = depth.unsqueeze(1) * torch.cos(theta).unsqueeze(0) + lateral_pixel.unsqueeze(1) * torch.sin(theta).unsqueeze(0)
    total_dist = tx_dist.unsqueeze(2) + receive_dist.unsqueeze(1)
    total_tof = total_dist / c_global

    if isinstance(t_starts, (int, float)):
        ts = torch.full((P, A), float(t_starts), device=device_local)
    elif t_starts.dim() == 0:
        ts = torch.full((P, A), t_starts.item(), device=device_local)
    elif t_starts.dim() == 1:
        ts = t_starts.unsqueeze(0).expand(P, A)
    else:
        ts = t_starts.expand(P, A)

    exact_idxs = ((total_tof - ts.unsqueeze(2)) * fs)
    valid_mask = (exact_idxs >= 0) & (exact_idxs < max_t_len - 1)

    c_ind = torch.arange(N, device=device_local).view(1, 1, N, 1).expand(P, A, N, T)
    img_ind = img_idx.view(-1, 1, 1, 1).expand(P, A, N, T)
    a_ind = angle_idx[:A].view(1, -1, 1, 1).expand(P, A, N, T)

    if interp == 'cubic':
        idx_floor = torch.floor(exact_idxs).long()
        w = exact_idxs - idx_floor.float()
        w2, w3 = w * w, w * w * w
        c_m1 = -0.5 * w3 + w2 - 0.5 * w
        c_0  =  1.5 * w3 - 2.5 * w2 + 1.0
        c_1  = -1.5 * w3 + 2.0 * w2 + 0.5 * w
        c_2  =  0.5 * w3 - 0.5 * w2
        idx_m1 = torch.clamp(idx_floor - 1, 0, max_t_len - 1)
        idx_0  = torch.clamp(idx_floor,     0, max_t_len - 1)
        idx_1  = torch.clamp(idx_floor + 1, 0, max_t_len - 1)
        idx_2  = torch.clamp(idx_floor + 2, 0, max_t_len - 1)
        t_ind_m1 = torch.clamp(idx_m1.unsqueeze(-1) + offsets.view(1, 1, 1, -1), 0, max_t_len - 1).long()
        t_ind_0  = torch.clamp(idx_0.unsqueeze(-1)  + offsets.view(1, 1, 1, -1), 0, max_t_len - 1).long()
        t_ind_1  = torch.clamp(idx_1.unsqueeze(-1)  + offsets.view(1, 1, 1, -1), 0, max_t_len - 1).long()
        t_ind_2  = torch.clamp(idx_2.unsqueeze(-1)  + offsets.view(1, 1, 1, -1), 0, max_t_len - 1).long()
        I_m1 = rf_I_pad[img_ind, a_ind, t_ind_m1, c_ind]
        I_0  = rf_I_pad[img_ind, a_ind, t_ind_0,  c_ind]
        I_1  = rf_I_pad[img_ind, a_ind, t_ind_1,  c_ind]
        I_2  = rf_I_pad[img_ind, a_ind, t_ind_2,  c_ind]
        Q_m1 = rf_Q_pad[img_ind, a_ind, t_ind_m1, c_ind]
        Q_0  = rf_Q_pad[img_ind, a_ind, t_ind_0,  c_ind]
        Q_1  = rf_Q_pad[img_ind, a_ind, t_ind_1,  c_ind]
        Q_2  = rf_Q_pad[img_ind, a_ind, t_ind_2,  c_ind]
        extracted_I = I_m1 * c_m1.unsqueeze(-1) + I_0 * c_0.unsqueeze(-1) + I_1 * c_1.unsqueeze(-1) + I_2 * c_2.unsqueeze(-1)
        extracted_Q = Q_m1 * c_m1.unsqueeze(-1) + Q_0 * c_0.unsqueeze(-1) + Q_1 * c_1.unsqueeze(-1) + Q_2 * c_2.unsqueeze(-1)
    else:
        center_idxs = torch.clamp(exact_idxs.long(), 0, max_t_len - 1)
        t_ind = torch.clamp(center_idxs.unsqueeze(-1) + offsets.view(1, 1, 1, -1), 0, max_t_len - 1).long()
        extracted_I = rf_I_pad[img_ind, a_ind, t_ind, c_ind]
        extracted_Q = rf_Q_pad[img_ind, a_ind, t_ind, c_ind]

    valid_mask_expanded = valid_mask.unsqueeze(-1).expand_as(extracted_I)
    extracted_I = extracted_I * valid_mask_expanded.float()
    extracted_Q = extracted_Q * valid_mask_expanded.float()
    return extracted_I, extracted_Q, total_tof, valid_mask


class RowDynamicMVBeamformerIQ:
    """
    GCF-MV + FBSS 自适应波束形成器
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

    def __call__(self, I_data, Q_data, selected_angles, t_starts, fs):
        n_a = I_data.shape[0]
        I_t = torch.from_numpy(I_data.astype(np.float32)).to(device).unsqueeze(0)
        Q_t = torch.from_numpy(Q_data.astype(np.float32)).to(device).unsqueeze(0)
        max_t_len = I_t.shape[2]

        half_win = self.temporal_win // 2
        offsets = torch.arange(-half_win, half_win + 1, device=device)

        I_beam_sum = torch.zeros((self.H, self.W), dtype=torch.float32, device=device)
        Q_beam_sum = torch.zeros((self.H, self.W), dtype=torch.float32, device=device)

        for i in range(n_a):
            angle_idx_single = torch.tensor([i], device=device)
            selected_angles_t = torch.as_tensor([selected_angles[i]], dtype=torch.float32, device=device)
            t_starts_t = (float(t_starts) if isinstance(t_starts, (int, float))
                          else torch.as_tensor([t_starts[i]], dtype=torch.float32, device=device))

            Y_out = torch.zeros((self.H, self.W), dtype=torch.complex64, device=device)

            for hz in range(self.H):
                # 提取当前行像素对应的 window
                row_pixel_idx = torch.arange(hz * self.W, (hz + 1) * self.W, device=device)
                row_img_idx_t = torch.zeros(self.W, dtype=torch.long, device=device)

                ext_I, ext_Q, tof, valid_mask = extract_windows_on_gpu(
                    I_t, Q_t, t_starts_t, row_img_idx_t, row_pixel_idx,
                    fs, offsets, angle_idx_single, max_t_len,
                    self.pitch, self.x_grid, self.z_grid, selected_angles_t, interp=args.interp
                )

                I_samples = ext_I[:, 0, :, :]   # [W, N, T]
                Q_samples = ext_Q[:, 0, :, :]   # [W, N, T]
                tof_center = tof[:, 0, :]       # [W, N]
                valid_mask_center = valid_mask[:, 0, :].float().unsqueeze(-1)

                # ========== 固定使用 positive IQ 相位补偿 ==========
                tof_samples = tof_center.unsqueeze(-1) + offsets.float().view(1, 1, -1) / fs
                phase = 2.0 * np.pi * fc_global * tof_samples
                cos_phi = torch.cos(phase)
                sin_phi = torch.sin(phase)
                I_aligned = I_samples * cos_phi - Q_samples * sin_phi
                Q_aligned = I_samples * sin_phi + Q_samples * cos_phi
                X_flat = torch.complex(I_aligned, Q_aligned)

                X_flat = X_flat * valid_mask_center  # [W, N, T]

                depth = self.z_grid[hz]

                # 动态计算活动孔径大小 K
                if args.dynamic_aperture:
                    half_a = depth / (2 * args.f_number)
                    K = int(2 * half_a / self.pitch) + 1
                    K = min(max(K, 4), self.N)
                else:
                    K = self.N

                L = max(int(K * self.subarray_ratio), 2)
                M = K - L + 1  # 子阵列数量

                # 提取当前行所有像素的活动通道信号
                idx_start = torch.clamp(self.c_idx - K // 2, 0, self.N - K)
                ch_idx = idx_start.unsqueeze(1) + torch.arange(K, device=device).unsqueeze(0)  # [W, K]

                ch_idx_t = ch_idx.unsqueeze(-1).expand(-1, -1, self.temporal_win)
                X_active = torch.gather(X_flat, 1, ch_idx_t)  # [W, K, T]

                # 可选预加权窗
                if args.window != 'rect':
                    x_norm = torch.linspace(-1, 1, K, device=device)
                    if args.window == 'hann':
                        win_w = 0.5 * (1.0 + torch.cos(np.pi * x_norm))
                    elif args.window == 'tukey':
                        beta = 0.25
                        win_w = torch.ones_like(x_norm)
                        transition = (x_norm.abs() > (1.0 - beta))
                        win_w[transition] = 0.5 * (1.0 + torch.cos(
                            np.pi * (x_norm[transition].abs() - (1.0 - beta)) / beta))
                    X_active = X_active * win_w.view(1, K, 1)

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
                dl_matrix = torch.eye(L, dtype=torch.complex64, device=device).unsqueeze(0)
                R_dl = R + (self.dl_factor / L) * trace.view(self.W, 1, 1) * dl_matrix

                # ========== 标准 MVDR 求解 ==========
                ones_L = torch.ones((self.W, L, 1), dtype=torch.complex64, device=device)

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
                y_row = torch.matmul(w.mH, X_mean)             # [W, 1, 1]
                if args.gcf_power > 0:
                    spatial_spectrum = torch.fft.fftshift(torch.fft.fft(X_mean.squeeze(-1), dim=1), dim=1)
                    power_spectrum = torch.abs(spatial_spectrum).square()
                    center = L // 2
                    low_bins = max(0, min(args.gcf_low_bins, center, L - center - 1))
                    low_energy = power_spectrum[:, center - low_bins:center + low_bins + 1].sum(dim=1)
                    total_energy = power_spectrum.sum(dim=1) + 1e-12
                    gcf = (low_energy / total_energy).clamp(0, 1)
                    y_row = y_row * torch.pow(gcf.view(self.W, 1, 1), args.gcf_power)

                Y_out[hz] = y_row.squeeze(-1).squeeze(-1)

            I_beam_sum += Y_out.real
            Q_beam_sum += Y_out.imag

        I_out = (I_beam_sum / n_a).cpu().numpy()
        Q_out = (Q_beam_sum / n_a).cpu().numpy()
        return I_out, Q_out


# ================= 保存函数 =================
def save_comparison_figure(mv_db, gt_norm, extent_mm, out_path, title_str, dr=60.0):
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
    im2 = ax2.imshow(mv_db, cmap='gray', vmin=-dr, vmax=0, extent=extent_mm, aspect='equal')
    ax2.set_title(f"GCF-MV\n{title_str}", fontsize=9, pad=10)
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
    fig, ax = plt.subplots(figsize=(6, 8), dpi=300)
    im = ax.imshow(db_img, cmap='gray', vmin=-dr, vmax=0, extent=extent_mm, aspect='equal')
    ax.set_xlabel("Lateral (mm)")
    ax.set_ylabel("Depth (mm)")
    ax.set_title(f"GCF-MV\n{title_str}", fontsize=9, pad=15)
    
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

    # 初始化 GCF-MV + FBSS 波束合成器
    bf = RowDynamicMVBeamformerIQ(
        z_grid, x_grid, n_elem, pitch, c, fs, t0_sub, angles_all,
        dl_factor=args.mv_dl, use_fbss=args.fbss, subarray_ratio=args.subarray_ratio,
        temporal_win=args.temporal_win
    )

    # TGC
    if args.tgc:
        tgc = 10 ** (args.tgc_alpha * (fc / 1e6) * (z_grid * 100) * 2.0 / 20.0)
    else:
        tgc = np.ones_like(z_grid)

    print(f"Processing GCF-MV (DL={args.mv_dl}, GCF bins={args.gcf_low_bins}, power={args.gcf_power})...")
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
        f"GCFb{args.gcf_low_bins}",
        f"GCFp{args.gcf_power:g}",
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


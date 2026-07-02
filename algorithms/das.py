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
    print(f"  H5 file      : {args.h5_path} (sample {args.h5_sample_idx})")
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

    if interp == 'linear':
        idx_floor = torch.floor(exact_idxs).long()
        idx_ceil = idx_floor + 1
        w_ceil = exact_idxs - idx_floor.float()
        w_floor = 1.0 - w_ceil
        idx_floor_c = torch.clamp(idx_floor, 0, max_t_len - 1)
        idx_ceil_c = torch.clamp(idx_ceil, 0, max_t_len - 1)
        t_ind_floor = torch.clamp(idx_floor_c.unsqueeze(-1) + offsets.view(1, 1, 1, -1), 0, max_t_len - 1).long()
        t_ind_ceil = torch.clamp(idx_ceil_c.unsqueeze(-1) + offsets.view(1, 1, 1, -1), 0, max_t_len - 1).long()
        I_floor = rf_I_pad[img_ind, a_ind, t_ind_floor, c_ind]
        I_ceil = rf_I_pad[img_ind, a_ind, t_ind_ceil, c_ind]
        extracted_I = I_floor * w_floor.unsqueeze(-1) + I_ceil * w_ceil.unsqueeze(-1)
        Q_floor = rf_Q_pad[img_ind, a_ind, t_ind_floor, c_ind]
        Q_ceil = rf_Q_pad[img_ind, a_ind, t_ind_ceil, c_ind]
        extracted_Q = Q_floor * w_floor.unsqueeze(-1) + Q_ceil * w_ceil.unsqueeze(-1)
    elif interp == 'cubic':
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

    def __call__(self, I_data, Q_data, selected_angles, t_starts, fs):
        n_a = I_data.shape[0]
        I_t = torch.from_numpy(I_data.astype(np.float32)).to(device).unsqueeze(0)
        Q_t = torch.from_numpy(Q_data.astype(np.float32)).to(device).unsqueeze(0)
        max_t_len = I_t.shape[2]

        offsets = torch.tensor([0], device=device)
        pixel_idx = torch.arange(self.H * self.W, device=device)
        img_idx_t = torch.zeros(self.H * self.W, dtype=torch.long, device=device)

        I_beam_sum = torch.zeros(self.H * self.W, device=device)
        Q_beam_sum = torch.zeros(self.H * self.W, device=device)
        weights = self.window.view(-1, self.N)

        for i in range(n_a):
            angle_idx_single = torch.tensor([i], device=device)
            selected_angles_t = torch.as_tensor([selected_angles[i]], dtype=torch.float32, device=device)
            t_starts_t = (float(t_starts) if isinstance(t_starts, (int, float))
                          else torch.as_tensor([t_starts[i]], dtype=torch.float32, device=device))

            ext_I, ext_Q, tof, valid_mask = extract_windows_on_gpu(
                I_t, Q_t, t_starts_t, img_idx_t, pixel_idx,
                fs, offsets, angle_idx_single, max_t_len,
                self.pitch, self.x_grid, self.z_grid, selected_angles_t,
                interp=args.interp
            )

            center_t = ext_I.shape[-1] // 2
            I_center = ext_I[:, 0, :, center_t]
            Q_center = ext_Q[:, 0, :, center_t]
            tof_center = tof[:, 0, :]
            valid_mask_center = valid_mask[:, 0, :].float()

            phase = 2.0 * np.pi * fc_global * tof_center
            cos_phi, sin_phi = torch.cos(phase), torch.sin(phase)
            
            I_center = I_center * valid_mask_center
            Q_center = Q_center * valid_mask_center
            
            I_aligned = I_center * cos_phi - Q_center * sin_phi
            Q_aligned = I_center * sin_phi + Q_center * cos_phi

            I_beam_sum += (I_aligned * weights).sum(dim=1)
            Q_beam_sum += (Q_aligned * weights).sum(dim=1)

        I_out = (I_beam_sum / n_a).view(self.H, self.W)
        Q_out = (Q_beam_sum / n_a).view(self.H, self.W)
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
        args.h5_path, args.h5_sample_idx
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

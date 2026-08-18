"""Provide Python utilities for fdmas."""

import argparse
import json
import os
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
    interpolate_channel_samples,
    parse_selected_angles,
    resolve_project_path,
    tgc_gain,
    validate_db_output,
)
from common_params import add_common_arguments, add_io_arguments
from h5_loader import load_from_h5
from matplotlib import patches
from matplotlib.gridspec import GridSpec

COMPARISON_VALUE_3 = 3

# ================= 命令行参数配置 =================
parser = argparse.ArgumentParser(
    description="F-DMAS Beamforming - Baseband IQ Data (H5)",
)
add_common_arguments(parser)
add_io_arguments(parser, save_gt_help="保存GT对比图")
parser.add_argument(
    "--row_block",
    type=int,
    default=24,
    help="GPU按深度方向分块行数;<=0 表示整幅一次计算",
)
args = None

METHOD_NAME = "fdmas"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


H5_PATH = None
OUTPUT_DIR = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    print("  F-DMAS Beamforming")
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
    print(f"  TGC          : {'Enabled' if args.tgc else 'Disabled'}")
    if args.tgc:
        print(f"  TGC Alpha    : {args.tgc_alpha} dB/MHz/cm")
    print(f"  DR           : {args.dr} dB")
    print(f"  Output       : {OUTPUT_DIR}")
    print(f"{'=' * 70}\n")


class FDMASBeamformerIQ:
    """Represent FDMASBeamformerIQ."""

    def __init__(self, z_grid, x_grid, n_elem, pitch, c, fc, fs, t0_all, angles_rad):
        """Initialize the instance."""
        self.height, self.width, self.N = len(z_grid), len(x_grid), n_elem
        self.x_grid = torch.from_numpy(x_grid).float().to(device)
        self.z_grid = torch.from_numpy(z_grid).float().to(device)
        self.angles_rad = torch.from_numpy(angles_rad).float().to(device)
        self.pitch, self.fc, self.fs = pitch, float(fc), fs
        self.sc = fs / c

        self.ep = torch.linspace(
            -(n_elem - 1) / 2 * pitch,
            (n_elem - 1) / 2 * pitch,
            n_elem,
            device=device,
        )
        x_mesh, z_mesh = torch.meshgrid(self.x_grid, self.z_grid, indexing="xy")
        self.x_mesh, self.z_mesh = x_mesh, z_mesh
        self.drs = torch.sqrt((x_mesh[..., None] - self.ep) ** 2 + z_mesh[..., None] ** 2) * self.sc

        win = torch.zeros((self.height, self.width, self.N), dtype=torch.float32, device=device)
        centers = torch.argmin(
            torch.abs(self.x_grid[:, None] - self.ep[None, :]),
            dim=1,
        )
        for iz, depth in enumerate(self.z_grid):
            k = dynamic_aperture_channel_count(
                float(depth.item()),
                args.f_number,
                pitch,
                n_elem,
                args.dynamic_aperture,
            )
            starts = torch.clamp(centers - k // 2, 0, n_elem - k)
            channels = starts[:, None] + torch.arange(k, device=device)[None, :]
            row_window = aperture_window_1d(k, args.window, device).expand(self.width, -1)
            win[iz].scatter_(1, channels, row_window)
        self.window = win / (win.sum(-1, keepdim=True) + 1e-9)
        self.ch = torch.arange(n_elem, device=device).view(1, 1, -1)

    @staticmethod
    def _complex_fdmas(i_aligned, q_aligned, weights, eps=1e-12):
        """Execute  complex fdmas."""
        z = torch.complex(i_aligned, q_aligned)
        q = z / torch.sqrt(torch.abs(z) + eps)
        qw = q * weights
        pair_sum = 0.5 * (qw.sum(dim=-1).square() - qw.square().sum(dim=-1))
        pair_norm = 0.5 * (weights.sum(dim=-1).square() - weights.square().sum(dim=-1))
        pair_sum = pair_sum / torch.clamp(pair_norm, min=eps)
        return pair_sum

    def __call__(self, i_data, q_data, selected_angles, t_starts, fs):
        """Run the callable operation."""
        n_a = i_data.shape[0]
        n_s = i_data.shape[1]
        max_sample = float(n_s - 2)

        i_tensor = torch.from_numpy(i_data.astype(np.float32)).to(device)
        q_tensor = torch.from_numpy(q_data.astype(np.float32)).to(device)
        cos_a = torch.from_numpy(np.cos(selected_angles).astype(np.float32)).to(device)
        sin_a = torch.from_numpy(np.sin(selected_angles).astype(np.float32)).to(device)

        t_starts_arr = np.asarray(t_starts, dtype=np.float32).reshape(-1)
        if t_starts_arr.size == 1:
            t_starts_arr = np.repeat(t_starts_arr, n_a)
        t_starts_t = torch.from_numpy(t_starts_arr[:n_a]).to(device) * fs

        beam_out = torch.zeros((self.height, self.width), dtype=torch.complex64, device=device)
        ch = self.ch
        row_block = self.height if args.row_block <= 0 else max(1, args.row_block)

        for z0 in range(0, self.height, row_block):
            z1 = min(z0 + row_block, self.height)
            x_block = self.x_mesh[z0:z1]
            z_block = self.z_mesh[z0:z1]
            drs_b = self.drs[z0:z1]
            weights_b = self.window[z0:z1]
            tx_z = z_block * self.sc
            tx_x = x_block * self.sc

            block_sum = torch.zeros(
                (z1 - z0, self.width),
                dtype=torch.complex64,
                device=device,
            )

            phi_rx = 2.0 * np.pi * self.fc * (drs_b / fs)
            cos_rx = torch.cos(phi_rx)
            sin_rx = torch.sin(phi_rx)

            for i in range(n_a):
                tx_samples = tx_z * cos_a[i] + tx_x * sin_a[i]
                sample = tx_samples.unsqueeze(-1) + drs_b - t_starts_t[i]
                valid = (sample >= 0) & (sample < n_s - 1)
                sample.clamp_(0.0, max_sample)
                i_angle = i_tensor[i]
                q_angle = q_tensor[i]

                i_center, q_center = interpolate_channel_samples(
                    i_angle,
                    q_angle,
                    sample,
                    ch,
                    args.interp,
                )

                valid_w = valid.float() * weights_b
                valid_w = valid_w / (valid_w.sum(dim=-1, keepdim=True) + 1e-9)
                i_rx = i_center * cos_rx - q_center * sin_rx
                q_rx = i_center * sin_rx + q_center * cos_rx
                pair_sum = self._complex_fdmas(i_rx, q_rx, valid_w)

                phi_tx = 2.0 * np.pi * self.fc * (tx_samples / fs)
                cos_tx = torch.cos(2.0 * phi_tx)
                sin_tx = torch.sin(2.0 * phi_tx)
                block_sum.add_(
                    torch.complex(
                        pair_sum.real * cos_tx - pair_sum.imag * sin_tx,
                        pair_sum.real * sin_tx + pair_sum.imag * cos_tx,
                    ),
                )

            beam_out[z0:z1] = block_sum / n_a

        return beam_out.real.cpu().numpy(), beam_out.imag.cpu().numpy()


# ================= 保存函数 =================
def save_comparison_figure(fdmas_db, gt_norm, extent_mm, out_path, title_str, dr=60.0):
    """Save comparison figure."""
    vmin, vmax = db_display_range(dr)

    def to_2d(arr):
        """Execute to 2d."""
        if arr.ndim == COMPARISON_VALUE_3:
            return arr[0] if arr.shape[0] == 1 else arr[:, :, 0]
        return arr

    fdmas_db, gt_norm = to_2d(fdmas_db), to_2d(gt_norm)

    fig = plt.figure(figsize=(12, 8), dpi=300)
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(gt_norm, cmap="gray", vmin=0, vmax=1, extent=extent_mm, aspect="equal")
    ax1.set_title("Ground Truth", fontsize=12, pad=10)
    ax1.set_xlabel("Lateral (mm)")
    ax1.set_ylabel("Depth (mm)")

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(
        fdmas_db,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        extent=extent_mm,
        aspect="equal",
    )
    ax2.set_title(f"F-DMAS\n{title_str}", fontsize=9, pad=10)
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
    ax.set_title(f"F-DMAS\n{title_str}", fontsize=9, pad=15)

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
    """Run the command-line workflow."""
    global args, H5_PATH, OUTPUT_DIR
    args = parser.parse_args()
    H5_PATH = resolve_project_path(args.h5_path, PROJECT_ROOT)
    OUTPUT_DIR = os.path.join(args.output_dir, METHOD_NAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    i_sub, q_sub = i_data[selected_indices], q_data[selected_indices]
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
    )

    beamformer = FDMASBeamformerIQ(
        z_grid,
        x_grid,
        n_elem,
        pitch,
        c,
        fc,
        fs,
        t0_sub,
        angles_all,
    )

    tgc = tgc_gain(z_grid, fc, args.tgc_alpha) if args.tgc else np.ones_like(z_grid)

    print("Processing F-DMAS...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    i_out, q_out = beamformer(i_sub, q_sub, selected_angles, t0_sub, fs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.time() - t1

    env = (i_out**2 + q_out**2) * (tgc[:, None] ** 2)
    env /= env.max() + 1e-24
    fdmas_db = 10 * np.log10(env + 1e-24)
    fdmas_db = validate_db_output(fdmas_db, (height, width), METHOD_NAME)

    # ========== 生成图标题(所有参数简写) ==========
    title_parts = [
        f"F{args.f_number}",
        f"{args.window.upper()}",
        f"{args.interp[:3]}",
    ]
    if len(selected_angles) == 1:
        title_parts.append(f"{np.degrees(selected_angles[0]):.1f}°")
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

    np.save(os.path.join(OUTPUT_DIR, f"{out_name}.npy"), fdmas_db)

    save_figure(
        fdmas_db,
        extent_mm,
        os.path.join(OUTPUT_DIR, f"{out_name}.png"),
        title_params,
        dr=args.dr,
    )

    if has_gt and args.save_gt:
        save_comparison_figure(
            fdmas_db,
            gt_data,
            extent_mm,
            os.path.join(OUTPUT_DIR, f"{out_name}_comparison.png"),
            title_params,
            dr=args.dr,
        )

    params = vars(args).copy()
    params["method"] = METHOD_NAME
    params["runtime_sec"] = float(dt)
    with open(os.path.join(OUTPUT_DIR, "params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    print(f"  GPU Time: {dt:.2f}s | Saved -> {out_name}")
    print(f"\nDone | Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

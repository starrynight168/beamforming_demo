"""Provide Python utilities for das."""

import argparse
import os

import numpy as np
import torch

from algorithms.common import (
    add_common_arguments,
    add_io_arguments,
    aperture_half_width,
    aperture_window_1d,
    aperture_window_from_dx,
    dynamic_aperture_channel_count,
    envelope_to_db,
    interpolate_channel_samples,
    prepare_beamforming_input,
    print_physical_summary,
    resolve_project_path,
    run_beamformer,
    save_comparison_figure,
    save_figure,
    time_start_tensor,
    validate_db_output,
    write_params,
)

# ================= 命令行参数配置 =================
parser = argparse.ArgumentParser(description="DAS Beamforming - Baseband IQ Data (H5)")
add_common_arguments(parser)
add_io_arguments(parser, save_gt_help="保存GT图像")
parser.add_argument(
    "--row_block",
    type=int,
    default=24,
    help="GPU按深度方向分块行数;<=0 表示整幅一次计算",
)
parser.add_argument(
    "--aperture_mode",
    choices=["discrete", "geometry"],
    default="discrete",
    help="DAS 接收孔径:discrete=与 MV 相同的离散 aperture_size;geometry=连续几何截断",
)
args = None

METHOD_NAME = "das"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


H5_PATH = None
OUTPUT_DIR = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DASBeamformerIQ:
    """Represent DASBeamformerIQ."""

    def __init__(self, z_grid, x_grid, n_elem, pitch, c, fc, fs, _t0_all, angles_rad):
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

        if args.aperture_mode == "geometry":
            dx = x_mesh[..., None] - self.ep
            half_a = aperture_half_width(
                z_mesh[..., None],
                args.f_number,
                n_elem,
                pitch,
                args.dynamic_aperture,
            )
            win = aperture_window_from_dx(dx, half_a, args.window)
        else:
            win = torch.zeros(
                (self.height, self.width, self.N),
                dtype=torch.float32,
                device=device,
            )
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
                row_window = aperture_window_1d(k, args.window, device).expand(
                    self.width,
                    -1,
                )
                win[iz].scatter_(1, channels, row_window)
        self.window = win / (win.sum(-1, keepdim=True) + 1e-9)
        self.ch = torch.arange(n_elem, device=device).view(1, 1, -1)

    def __call__(self, i_data, q_data, selected_angles, t_starts, fs):
        """Run the callable operation."""
        n_a = i_data.shape[0]
        n_s = i_data.shape[1]
        max_sample = float(n_s - 2)

        i_tensor = torch.from_numpy(i_data.astype(np.float32)).to(device)
        q_tensor = torch.from_numpy(q_data.astype(np.float32)).to(device)
        cos_a = torch.from_numpy(np.cos(selected_angles).astype(np.float32)).to(device)
        sin_a = torch.from_numpy(np.sin(selected_angles).astype(np.float32)).to(device)

        t_starts_t = time_start_tensor(t_starts, n_a, fs, device)

        i_output = torch.zeros((self.height, self.width), dtype=torch.float32, device=device)
        q_output = torch.zeros((self.height, self.width), dtype=torch.float32, device=device)
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

            i_block = torch.zeros((z1 - z0, self.width), dtype=torch.float32, device=device)
            q_block = torch.zeros((z1 - z0, self.width), dtype=torch.float32, device=device)

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
                i_sum = (i_rx * valid_w).sum(dim=-1)
                q_sum = (q_rx * valid_w).sum(dim=-1)

                phi_tx = 2.0 * np.pi * self.fc * (tx_samples / fs)
                cos_tx = torch.cos(phi_tx)
                sin_tx = torch.sin(phi_tx)
                i_block.add_(i_sum * cos_tx - q_sum * sin_tx)
                q_block.add_(i_sum * sin_tx + q_sum * cos_tx)

            i_output[z0:z1] = i_block / n_a
            q_output[z0:z1] = q_block / n_a

        return i_output.cpu().numpy(), q_output.cpu().numpy()


# ================= 主程序 =================
def main():
    """Run the command-line workflow."""
    global args, H5_PATH, OUTPUT_DIR
    args = parser.parse_args()
    H5_PATH = resolve_project_path(args.h5_path, PROJECT_ROOT)
    OUTPUT_DIR = os.path.join(args.output_dir, METHOD_NAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_data = prepare_beamforming_input(
        H5_PATH,
        args.h5_sample_idx,
        args.select_angles,
    )
    sample = input_data.sample
    selected_angles = input_data.selected_angles
    c, fc, fs, pitch, n_elem = sample.c, sample.fc, sample.fs, sample.pitch, sample.n_elem
    z_grid, x_grid = sample.z_grid, sample.x_grid
    i_sub, q_sub, t0_sub = input_data.i_data, input_data.q_data, input_data.t0
    angles_all = sample.angles
    gt_data, has_gt = sample.gt_data, sample.has_gt
    height, width = len(z_grid), len(x_grid)
    dz, dx = z_grid[1] - z_grid[0], x_grid[1] - x_grid[0]
    depth_min, depth_max = z_grid[0], z_grid[-1]
    extent_mm = input_data.extent_mm

    print_physical_summary(
        method_name=METHOD_NAME,
        h5_path=H5_PATH,
        sample_idx=args.h5_sample_idx,
        output_dir=OUTPUT_DIR,
        device=device,
        c=c,
        fc=fc,
        fs=fs,
        pitch=pitch,
        n_elem=n_elem,
        angles=angles_all,
        t0=t0_sub,
        height=height,
        width=width,
        dz=dz,
        dx=dx,
        depth_min=depth_min,
        depth_max=depth_max,
        selected_angles=selected_angles,
        has_gt=has_gt,
        parameters=[
            ("F-Number", args.f_number),
            ("Aperture", "Dynamic" if args.dynamic_aperture else "Fixed Full"),
            ("Aperture mode", args.aperture_mode),
            ("Window", args.window.upper()),
            ("Interp", args.interp),
            ("TGC", "Enabled" if args.tgc else "Disabled"),
            *([("TGC Alpha", f"{args.tgc_alpha} dB/MHz/cm")] if args.tgc else []),
            ("DR", f"{args.dr} dB"),
        ],
    )

    bf = DASBeamformerIQ(z_grid, x_grid, n_elem, pitch, c, fc, fs, t0_sub, angles_all)

    print("Processing DAS...")
    i_out, q_out, dt = run_beamformer(
        bf,
        i_sub,
        q_sub,
        selected_angles,
        t0_sub,
        fs,
        device,
    )
    das_db = envelope_to_db(i_out, q_out, z_grid, fc, args.tgc, args.tgc_alpha)
    das_db = validate_db_output(das_db, (height, width), METHOD_NAME)

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

    np.save(os.path.join(OUTPUT_DIR, f"{out_name}.npy"), das_db)

    save_figure(
        das_db,
        extent_mm,
        os.path.join(OUTPUT_DIR, f"{out_name}.png"),
        f"{title_params} | {args.aperture_mode}",
        dr=args.dr,
        method_name=METHOD_NAME,
    )

    if has_gt and args.save_gt:
        save_comparison_figure(
            das_db,
            gt_data,
            extent_mm,
            os.path.join(OUTPUT_DIR, f"{out_name}_comparison.png"),
            f"{title_params} | {args.aperture_mode}",
            dr=args.dr,
        )

    params = vars(args).copy()
    params["method"] = METHOD_NAME
    params["runtime_sec"] = float(dt)
    write_params(os.path.join(OUTPUT_DIR, "params.json"), params)

    print(f"  GPU Time: {dt:.2f}s | Saved -> {out_name}")
    print(f"\nDone | Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

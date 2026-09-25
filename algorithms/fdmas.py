"""Provide Python utilities for fdmas."""

import argparse
from pathlib import Path

import numpy as np
import torch

from algorithms.common import (
    add_common_arguments,
    add_io_arguments,
    aperture_window_1d,
    dynamic_aperture_channel_count,
    envelope_to_db,
    interpolate_channel_samples,
    prepare_beamforming_input,
    print_physical_summary,
    resolve_project_path,
    run_beamformer,
    save_algorithm_result,
    time_start_tensor,
    validate_db_output,
)

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
PROJECT_ROOT = Path(__file__).resolve().parents[1]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FDMASBeamformerIQ:
    """Represent FDMASBeamformerIQ."""

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
        self.drs = (
            torch.sqrt((x_mesh[..., None] - self.ep) ** 2 + z_mesh[..., None] ** 2)
            * self.sc
        )

        win = torch.zeros(
            (self.height, self.width, self.N), dtype=torch.float32, device=device
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
                self.width, -1
            )
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
        return pair_sum / torch.clamp(pair_norm, min=eps)

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

        beam_out = torch.zeros(
            (self.height, self.width), dtype=torch.complex64, device=device
        )
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


# ================= 主程序 =================
def main():
    """Run the command-line workflow."""
    global args
    args = parser.parse_args()
    h5_path = resolve_project_path(args.h5_path, PROJECT_ROOT)
    output_dir = f"{args.output_dir}/{METHOD_NAME}"

    input_data = prepare_beamforming_input(
        h5_path,
        args.h5_sample_idx,
        args.select_angles,
    )
    sample = input_data.sample
    c, fc, fs, pitch, n_elem = (
        sample.c,
        sample.fc,
        sample.fs,
        sample.pitch,
        sample.n_elem,
    )
    angles_all, z_grid, x_grid = sample.angles, sample.z_grid, sample.x_grid
    selected_angles = input_data.selected_angles
    i_sub, q_sub, t0_sub = input_data.i_data, input_data.q_data, input_data.t0
    has_gt = sample.has_gt
    height, width = len(z_grid), len(x_grid)
    dz, dx = z_grid[1] - z_grid[0], x_grid[1] - x_grid[0]
    depth_min, depth_max = z_grid[0], z_grid[-1]

    print_physical_summary(
        method_name=METHOD_NAME,
        h5_path=h5_path,
        sample_idx=args.h5_sample_idx,
        output_dir=output_dir,
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
            ("Window", args.window.upper()),
            ("Interp", args.interp),
            ("TGC", "Enabled" if args.tgc else "Disabled"),
            *([("TGC Alpha", f"{args.tgc_alpha} dB/MHz/cm")] if args.tgc else []),
            ("DR", f"{args.dr} dB"),
        ],
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

    print("Processing F-DMAS...")
    i_out, q_out, dt = run_beamformer(
        beamformer,
        i_sub,
        q_sub,
        selected_angles,
        t0_sub,
        fs,
        device,
    )
    fdmas_db = envelope_to_db(i_out, q_out, z_grid, fc, args.tgc, args.tgc_alpha)
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

    method_dir = save_algorithm_result(
        fdmas_db,
        input_data,
        args,
        METHOD_NAME,
        title_params,
        dt,
    )

    print(f"  GPU Time: {dt:.2f}s | Saved -> {out_name}")
    print(f"\nDone | Output: {method_dir}")


if __name__ == "__main__":
    main()

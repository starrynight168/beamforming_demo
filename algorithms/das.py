"""Provide Python utilities for das."""

# ruff: noqa: E402

import argparse

import numpy as np
import torch


if __package__:
    from algorithms.bootstrap import initialize_project_path
else:
    from bootstrap import initialize_project_path

PROJECT_ROOT = initialize_project_path(__file__)

from shared_core.geometry import plane_wave_distance, receive_distance
from shared_core.phase import carrier_phase, phase_factors, rotate_components

from algorithms.shared_core.beamforming import (
    aperture_start_indices,
    aperture_half_width,
    aperture_window_1d,
    aperture_window_from_dx,
    dynamic_aperture_channel_count,
    interpolate_channel_samples,
)
from algorithms.shared_core.common import (
    build_beamformer,
    resolve_device,
    add_common_arguments,
    add_io_arguments,
    beamforming_depth_grid,
    beamformed_image_db,
    prepare_beamforming_input,
    print_physical_summary,
    resolve_project_path,
    run_beamformer,
    save_algorithm_result,
    time_start_tensor,
    validate_db_output,
    validate_beamformer_input,
)

# ================= 命令行参数配置 =================
def build_parser(params=None):
    parser = argparse.ArgumentParser(description="DAS Beamforming - Baseband IQ Data (H5)")
    add_common_arguments(parser, params=params)
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
    return parser



METHOD_NAME = "das"




class DASBeamformerIQ:
    """Represent DASBeamformerIQ."""

    def __init__(self, z_grid, x_grid, n_elem, pitch, c, fc, fs, _t0_all, angles_rad, options=None, device=None):
        """Initialize the instance."""
        self.device = resolve_device(device)
        self.options = options if options is not None else build_parser().parse_args([])
        self.signal_domain = self.options.signal_domain
        self.height, self.width, self.N = len(z_grid), len(x_grid), n_elem
        self.x_grid = torch.from_numpy(x_grid).float().to(self.device)
        self.z_grid = torch.from_numpy(z_grid).float().to(self.device)
        self.pitch, self.fc, self.fs = pitch, float(fc), fs
        self.sc = fs / c

        self.ep = torch.linspace(
            -(n_elem - 1) / 2 * pitch,
            (n_elem - 1) / 2 * pitch,
            n_elem,
            device=self.device,
        )
        x_mesh, z_mesh = torch.meshgrid(self.x_grid, self.z_grid, indexing="xy")
        self.x_mesh, self.z_mesh = x_mesh, z_mesh
        self.drs = (
            receive_distance(x_mesh[..., None], z_mesh[..., None], self.ep)
            * self.sc
        )

        if self.options.aperture_mode == "geometry":
            if self.options.dynamic_aperture:
                dx = x_mesh[..., None] - self.ep
                half_a = aperture_half_width(
                    z_mesh[..., None], self.options.f_number, n_elem, pitch, True,
                )
                win = aperture_window_from_dx(dx, half_a, self.options.window)
            else:
                win = aperture_window_1d(n_elem, self.options.window, self.device).expand(
                    self.height, self.width, -1,
                )
        else:
            win = torch.zeros(
                (self.height, self.width, self.N),
                dtype=torch.float32,
                device=self.device,
            )
            for iz, depth in enumerate(self.z_grid):
                k = dynamic_aperture_channel_count(
                    float(depth.item()),
                    self.options.f_number,
                    pitch,
                    n_elem,
                    self.options.dynamic_aperture,
                )
                starts = aperture_start_indices(self.x_grid, n_elem, k, pitch)
                channels = starts[:, None] + torch.arange(k, device=self.device)[None, :]
                row_window = aperture_window_1d(k, self.options.window, self.device).expand(
                    self.width,
                    -1,
                )
                win[iz].scatter_(1, channels, row_window)
        self.window = win / (win.sum(-1, keepdim=True) + 1e-9)
        self.ch = torch.arange(n_elem, device=self.device).view(1, 1, -1)

    def __call__(self, i_data, q_data, selected_angles, t_starts, fs):
        """Run the callable operation."""
        validate_beamformer_input(i_data, q_data, selected_angles, fs, self.fs, self.N)
        if self.signal_domain == "rf" and np.any(q_data != 0):
            raise ValueError("RF 输入的虚部必须为零")
        n_a = i_data.shape[0]
        n_s = i_data.shape[1]
        max_sample = float(n_s - 1)

        i_tensor = torch.from_numpy(i_data.astype(np.float32)).to(self.device)
        q_tensor = torch.from_numpy(q_data.astype(np.float32)).to(self.device)
        cos_a = torch.from_numpy(np.cos(selected_angles).astype(np.float32)).to(self.device)
        sin_a = torch.from_numpy(np.sin(selected_angles).astype(np.float32)).to(self.device)

        t_starts_t = time_start_tensor(t_starts, n_a, fs, self.device)

        i_output = torch.zeros(
            (self.height, self.width), dtype=torch.float32, device=self.device
        )
        q_output = torch.zeros(
            (self.height, self.width), dtype=torch.float32, device=self.device
        )
        ch = self.ch
        row_block = self.height if self.options.row_block <= 0 else max(1, self.options.row_block)

        for z0 in range(0, self.height, row_block):
            z1 = min(z0 + row_block, self.height)
            x_block = self.x_mesh[z0:z1]
            z_block = self.z_mesh[z0:z1]
            drs_b = self.drs[z0:z1]
            weights_b = self.window[z0:z1]
            tx_z = z_block * self.sc
            tx_x = x_block * self.sc

            i_block = torch.zeros(
                (z1 - z0, self.width), dtype=torch.float32, device=self.device
            )
            q_block = torch.zeros(
                (z1 - z0, self.width), dtype=torch.float32, device=self.device
            )

            phi_rx = carrier_phase(self.fc, drs_b / fs) if self.signal_domain == "iq" else torch.zeros_like(drs_b)
            cos_rx, sin_rx = phase_factors(phi_rx)

            for i in range(n_a):
                tx_samples = plane_wave_distance(tx_x, tx_z, cos_a[i], sin_a[i])
                sample = tx_samples.unsqueeze(-1) + drs_b - t_starts_t[i]
                valid = (sample >= 0) & (sample <= max_sample)
                sample.clamp_(0.0, max_sample)
                i_angle = i_tensor[i]
                q_angle = q_tensor[i]

                i_center, q_center = interpolate_channel_samples(
                    i_angle,
                    q_angle,
                    sample,
                    ch,
                    self.options.interp,
                )

                valid_w = valid.float() * weights_b
                valid_w = valid_w / (valid_w.sum(dim=-1, keepdim=True) + 1e-9)
                i_rx, q_rx = rotate_components(i_center, q_center, cos_rx, sin_rx)
                i_sum = (i_rx * valid_w).sum(dim=-1)
                q_sum = (q_rx * valid_w).sum(dim=-1)

                phi_tx = carrier_phase(self.fc, tx_samples / fs) if self.signal_domain == "iq" else torch.zeros_like(tx_samples)
                cos_tx, sin_tx = phase_factors(phi_tx)
                i_rot, q_rot = rotate_components(i_sum, q_sum, cos_tx, sin_tx)
                i_block.add_(i_rot)
                q_block.add_(q_rot)

            i_output[z0:z1] = i_block / n_a
            q_output[z0:z1] = q_block / n_a

        return i_output.cpu().numpy(), q_output.cpu().numpy()


# ================= 主程序 =================
def main(argv=None):
    """Run the command-line workflow."""
    args = build_parser().parse_args(argv)
    device = resolve_device()
    h5_path = resolve_project_path(args.h5_path, PROJECT_ROOT)
    output_dir = f"{args.output_dir}/{METHOD_NAME}"

    input_data = prepare_beamforming_input(
        h5_path,
        args.h5_sample_idx,
        args.select_angles,
        signal_domain=args.signal_domain,
    )
    sample = input_data.sample
    selected_angles = input_data.selected_angles
    c, fc, fs, pitch, n_elem = (
        sample.c,
        sample.fc,
        sample.fs,
        sample.pitch,
        sample.n_elem,
    )
    z_grid, x_grid = sample.z_grid, sample.x_grid
    compute_z = beamforming_depth_grid(z_grid, c, fs, args.signal_domain)
    i_sub, q_sub, t0_sub = input_data.i_data, input_data.q_data, input_data.t0
    angles_all = sample.angles
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
            ("Signal domain", args.signal_domain),
            ("Computed depth rows", len(compute_z)),
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

    bf = build_beamformer(
        DASBeamformerIQ, compute_z, x_grid, n_elem, pitch, c, fc, fs,
        t0_sub, angles_all, args, device,
    )

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
    das_db = beamformed_image_db(i_out, q_out, compute_z, z_grid, fc, args.tgc, args.tgc_alpha, args.signal_domain)
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

    title = f"{title_params} | {args.aperture_mode}"
    method_dir = save_algorithm_result(
        das_db,
        input_data,
        args,
        METHOD_NAME,
        title,
        dt,
    )

    print(f"  GPU Time: {dt:.2f}s | Saved -> {METHOD_NAME}")
    print(f"\nDone | Output: {method_dir}")


if __name__ == "__main__":
    main()

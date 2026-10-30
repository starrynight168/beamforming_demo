"""Provide Python utilities for fdmas."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import filtfilt, firwin

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.common import (
    add_common_arguments,
    add_io_arguments,
    aperture_window_1d,
    dynamic_aperture_channel_count,
    beamforming_depth_grid,
    beamformed_image_db,
    interpolate_channel_samples,
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
parser = argparse.ArgumentParser(
    description="F-DMAS beamforming for RF, analytic RF, and IQ data (H5)",
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
RF_FILTER_TAPS = 33


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FDMASBeamformerIQ:
    """Filtered delay-multiply-and-sum beamformer for channel signals."""

    def __init__(self, z_grid, x_grid, n_elem, pitch, c, fc, fs, _t0_all, angles_rad, options=None):
        """Initialize the instance."""
        self.options = options if options is not None else (args if args is not None else parser.parse_args([]))
        self.signal_domain = self.options.signal_domain
        self.height, self.width, self.N = len(z_grid), len(x_grid), n_elem
        self.x_grid = torch.from_numpy(x_grid).float().to(device)
        self.z_grid = torch.from_numpy(z_grid).float().to(device)
        self.pitch, self.fc, self.fs = pitch, float(fc), fs
        self.sc = fs / c
        self.rf_filter = None
        self.rf_filter_padlen = None
        if self.signal_domain == "rf":
            if fs <= 5.0 * self.fc:
                raise ValueError(
                    f"RF 输入采样率必须高于 2.5fc 谐波带的 Nyquist 下限 5fc={5.0 * self.fc:.6g} Hz，"
                    f"当前 fs={fs:.6g} Hz"
                )
            depth_fs = c / (2.0 * float(np.diff(z_grid).mean()))
            self.rf_filter = design_rf_fdmas_filter(self.fc, depth_fs)
            self.rf_filter_padlen = 3 * (RF_FILTER_TAPS - 1)
            if self.height <= self.rf_filter_padlen:
                raise ValueError(
                    f"RF F-DMAS 带通滤波至少需要 {self.rf_filter_padlen + 1} 个密深度行，"
                    f"当前只有 {self.height} 行"
                )

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
                self.options.f_number,
                pitch,
                n_elem,
                self.options.dynamic_aperture,
            )
            starts = torch.clamp(centers - k // 2, 0, n_elem - k)
            channels = starts[:, None] + torch.arange(k, device=device)[None, :]
            row_window = aperture_window_1d(k, self.options.window, device).expand(
                self.width, -1
            )
            win[iz].scatter_(1, channels, row_window)
        self.window = win / (win.sum(-1, keepdim=True) + 1e-9)
        self.ch = torch.arange(n_elem, device=device).view(1, 1, -1)

    @staticmethod
    def _complex_fdmas(i_aligned, q_aligned, weights):
        """Average complex channel pairs after square-root amplitude compression."""
        z = torch.complex(i_aligned, q_aligned)
        tiny = torch.finfo(z.real.dtype).tiny
        q = z / torch.sqrt(torch.abs(z).clamp_min(tiny))
        qw = q * weights
        pair_sum = (qw[..., 1:] * qw[..., :-1].cumsum(dim=-1)).sum(dim=-1)
        pair_norm = (
            weights[..., 1:] * weights[..., :-1].cumsum(dim=-1)
        ).sum(dim=-1)
        return torch.where(
            pair_norm > 0, pair_sum / pair_norm.clamp_min(tiny), torch.zeros_like(pair_sum),
        )

    def _filter_rf_output(self, values):
        pad = self.rf_filter_padlen
        padded = np.pad(values, ((pad, pad), (0, 0)), mode="reflect")
        filtered = filtfilt(
            self.rf_filter,
            [1.0],
            padded,
            axis=0,
            padlen=pad,
        )
        return filtered[pad:-pad].astype(np.float32)

    def __call__(self, i_data, q_data, selected_angles, t_starts, fs):
        """Run the callable operation."""
        validate_beamformer_input(i_data, q_data, selected_angles, fs, self.fs, self.N)
        if self.signal_domain == "rf" and np.any(q_data != 0):
            raise ValueError("RF F-DMAS requires real RF input with a zero imaginary component")
        n_a = i_data.shape[0]
        n_s = i_data.shape[1]
        max_sample = float(n_s - 1)

        i_tensor = torch.from_numpy(i_data.astype(np.float32)).to(device)
        q_tensor = torch.from_numpy(q_data.astype(np.float32)).to(device)
        cos_a = torch.from_numpy(np.cos(selected_angles).astype(np.float32)).to(device)
        sin_a = torch.from_numpy(np.sin(selected_angles).astype(np.float32)).to(device)

        t_starts_t = time_start_tensor(t_starts, n_a, fs, device)

        beam_out = torch.zeros(
            (self.height, self.width), dtype=torch.complex64, device=device
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

            block_sum = torch.zeros(
                (z1 - z0, self.width),
                dtype=torch.complex64,
                device=device,
            )

            phi_rx = 2.0 * np.pi * self.fc * (drs_b / fs) if self.signal_domain == "iq" else torch.zeros_like(drs_b)
            cos_rx = torch.cos(phi_rx)
            sin_rx = torch.sin(phi_rx)

            for i in range(n_a):
                tx_samples = tx_z * cos_a[i] + tx_x * sin_a[i]
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
                i_rx = i_center * cos_rx - q_center * sin_rx
                q_rx = i_center * sin_rx + q_center * cos_rx
                pair_sum = self._complex_fdmas(i_rx, q_rx, valid_w)

                phi_tx = 2.0 * np.pi * self.fc * (tx_samples / fs) if self.signal_domain == "iq" else torch.zeros_like(tx_samples)
                cos_tx = torch.cos(2.0 * phi_tx)
                sin_tx = torch.sin(2.0 * phi_tx)
                block_sum.add_(
                    torch.complex(
                        pair_sum.real * cos_tx - pair_sum.imag * sin_tx,
                        pair_sum.real * sin_tx + pair_sum.imag * cos_tx,
                    ),
                )

            beam_out[z0:z1] = block_sum / n_a

        real = beam_out.real.cpu().numpy()
        if self.signal_domain == "rf":
            return self._filter_rf_output(real), np.zeros_like(real)
        return real, beam_out.imag.cpu().numpy()


def design_rf_fdmas_filter(fc, sample_rate):
    passband = (1.5 * fc, 2.5 * fc)
    if passband[1] >= sample_rate / 2:
        raise ValueError(
            f"RF F-DMAS 需要足够采样率通过 2f0 谐波带："
            f"2.5fc={passband[1]:.6g} Hz 必须低于 Nyquist={sample_rate / 2:.6g} Hz"
        )
    return firwin(RF_FILTER_TAPS, passband, pass_zero=False, fs=sample_rate, window=("kaiser", 8.6))


def fdmas_depth_grid(z_grid, c, fc, fs, signal_domain):
    if signal_domain != "rf":
        return z_grid
    # Leave transition room above the 2.5fc filter edge.
    filter_sample_rate = 6.0 * fc
    dense_sample_rate = max(fs, np.nextafter(filter_sample_rate, np.inf))
    # Keep the deep image boundary away from filter edges; the shallow edge is reflected before filtering.
    pad_rows = 2 * 3 * (RF_FILTER_TAPS - 1)
    end_margin = pad_rows * c / (2.0 * dense_sample_rate)
    bounds = np.array([z_grid[0], z_grid[-1] + end_margin], dtype=np.float64)
    return beamforming_depth_grid(bounds, c, dense_sample_rate, signal_domain)


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
        signal_domain=args.signal_domain,
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
    compute_z = fdmas_depth_grid(z_grid, c, fc, fs, args.signal_domain)
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
            ("Signal domain", args.signal_domain),
            ("Computed depth rows", len(compute_z)),
            *([("RF output depth sampling", f"{c / (2.0 * np.diff(compute_z).mean()) / 1e6:.3f} MHz")] if args.signal_domain == "rf" else []),
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
        compute_z,
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
    fdmas_db = beamformed_image_db(i_out, q_out, compute_z, z_grid, fc, args.tgc, args.tgc_alpha, args.signal_domain)
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

    method_dir = save_algorithm_result(
        fdmas_db,
        input_data,
        args,
        METHOD_NAME,
        title_params,
        dt,
        extra_params={"beamforming_depth_rows": len(compute_z)},
    )

    print(f"  GPU Time: {dt:.2f}s | Saved -> {METHOD_NAME}")
    print(f"\nDone | Output: {method_dir}")


if __name__ == "__main__":
    main()

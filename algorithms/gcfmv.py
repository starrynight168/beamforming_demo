"""Provide Python utilities for gcfmv."""

import argparse
from pathlib import Path

import numpy as np
import torch

from algorithms.common import (
    add_common_arguments,
    add_io_arguments,
    build_row_dynamic_geometry,
    envelope_to_db,
    interpolate_channel_samples,
    nonnegative_float,
    nonnegative_int,
    prepare_beamforming_input,
    print_physical_summary,
    positive_odd_int,
    resolve_project_path,
    run_beamformer,
    save_algorithm_result,
    time_start_tensor,
    unit_interval_float,
    validate_db_output,
)

# ================= 命令行参数配置 =================
parser = argparse.ArgumentParser(description="GCF-MV Beamforming")
add_common_arguments(parser, window_help="子孔径预加权窗")

# ---- MV 核心参数 ----
parser.add_argument("--mv_dl", type=nonnegative_float, default=0.0, help="对角加载系数")
parser.add_argument(
    "--fbss",
    action="store_true",
    default=True,
    help="启用前向-后向空间平滑",
)
parser.add_argument("--no_fbss", dest="fbss", action="store_false", help="禁用 FBSS")
parser.add_argument(
    "--subarray_ratio",
    type=unit_interval_float,
    default=0.25,
    help="子阵列比例 subarray_size = aperture_size * ratio",
)
parser.add_argument(
    "--temporal_win",
    type=positive_odd_int,
    default=9,
    help="协方差时间平均窗口,必须为正奇数;1 表示关闭",
)
parser.add_argument(
    "--gcf_low_bins",
    type=nonnegative_int,
    default=1,
    help="GCF低空间频率半宽;1表示保留中心频点及左右各1个频点",
)
parser.add_argument(
    "--gcf_power",
    type=nonnegative_float,
    default=2.0,
    help="GCF后加权指数;0表示退化为MV",
)

add_io_arguments(parser)
args = None

METHOD_NAME = "gcfmv"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RowDynamicMVBeamformerIQ:
    """GCF-MV + FBSS 自适应波束形成器.

    固定使用 positive IQ 相位补偿.
    """

    def __init__(
        self,
        z_grid,
        x_grid,
        n_elem,
        pitch,
        c,
        fc,
        fs,
        _t0_all,
        angles_rad,
        mv_dl=0.01,
        fbss=True,
        subarray_ratio=0.5,
        temporal_win=1,
    ):
        """Initialize the instance."""
        geometry = build_row_dynamic_geometry(
            z_grid,
            x_grid,
            n_elem,
            pitch,
            c,
            fc,
            fs,
            args.f_number,
            args.dynamic_aperture,
            args.window,
            subarray_ratio,
            temporal_win,
            device,
        )
        self.__dict__.update(geometry)
        self.angles_rad = torch.from_numpy(angles_rad).float().to(device)
        self.dl_factor = mv_dl
        self.use_fbss = fbss
        self.subarray_ratio = subarray_ratio
        self.temporal_win = temporal_win

    def __call__(self, i_data, q_data, selected_angles, t_starts, fs):
        """Run the callable operation."""
        n_a = i_data.shape[0]
        i_tensor = torch.from_numpy(i_data.astype(np.float32)).to(device)
        q_tensor = torch.from_numpy(q_data.astype(np.float32)).to(device)
        n_s = i_tensor.shape[1]

        half_win = self.temporal_win // 2
        offsets = torch.arange(-half_win, half_win + 1, device=device)
        offsets_f = offsets.float().view(1, 1, -1)
        cos_a = torch.from_numpy(np.cos(selected_angles).astype(np.float32)).to(device)
        sin_a = torch.from_numpy(np.sin(selected_angles).astype(np.float32)).to(device)
        t_starts_t = time_start_tensor(t_starts, n_a, fs, device)
        max_sample = float(n_s - 2)

        i_beam_sum = torch.zeros(
            (self.height, self.width), dtype=torch.float32, device=device
        )
        q_beam_sum = torch.zeros(
            (self.height, self.width), dtype=torch.float32, device=device
        )

        # 1. 预计算接收端相位旋转因子 (全网格 3D)
        phi_rx_global = 2.0 * np.pi * self.fc * (self.drs / fs)
        cos_rx_global = torch.cos(phi_rx_global)
        sin_rx_global = torch.sin(phi_rx_global)

        for i in range(n_a):
            i_angle = i_tensor[i]
            q_angle = q_tensor[i]

            for hz in range(self.height):
                tx_row = (
                    self.z_mesh[hz] * self.sc * cos_a[i]
                    + self.x_mesh[hz] * self.sc * sin_a[i]
                )
                sample_no_t0 = tx_row.unsqueeze(-1) + self.drs[hz]
                sample_center = sample_no_t0 - t_starts_t[i]
                sample = sample_center.unsqueeze(-1) + offsets_f
                valid_mask = ((sample >= 0) & (sample < n_s - 1)).float()
                sample.clamp_(0.0, max_sample)

                i_samples, q_samples = interpolate_channel_samples(
                    i_angle,
                    q_angle,
                    sample,
                    self.ch,
                    args.interp,
                )

                # 2. 接收端相位旋转 (3D)
                cos_rx = cos_rx_global[hz].unsqueeze(-1)
                sin_rx = sin_rx_global[hz].unsqueeze(-1)
                i_rx = i_samples * cos_rx - q_samples * sin_rx
                q_rx = i_samples * sin_rx + q_samples * cos_rx

                x_flat = torch.complex(i_rx, q_rx) * valid_mask
                row = self.row_cache[hz]
                aperture_size, subarray_size, subarray_count = (
                    row["aperture_size"],
                    row["subarray_size"],
                    row["subarray_count"],
                )

                x_active = torch.gather(
                    x_flat, 1, row["ch_idx_t"]
                )  # [width, aperture_size, T]
                if row["window"] is not None:
                    x_active = x_active * row["window"].view(1, aperture_size, 1)

                # ========== 空间平滑 (前向) ==========
                x_subarrays = x_active.unfold(1, subarray_size, 1).permute(
                    0,
                    3,
                    1,
                    2,
                )  # [width, subarray_size, subarray_count, T]
                snapshots = subarray_count * self.temporal_win
                x_snapshots = x_subarrays.reshape(self.width, subarray_size, snapshots)
                covariance_forward = (
                    torch.matmul(x_snapshots, x_snapshots.mH) / snapshots
                )  # [width, subarray_size, subarray_size]

                # ========== 前向-后向空间平滑 (FBSS) ==========
                covariance = (
                    0.5 * (covariance_forward + covariance_forward.conj().flip(-1, -2))
                    if self.use_fbss
                    else covariance_forward
                )

                # ========== 对角加载 ==========
                trace = covariance.diagonal(dim1=-2, dim2=-1).real.sum(-1)  # [width]
                covariance_loaded = (
                    covariance
                    + (self.dl_factor / subarray_size)
                    * trace.view(self.width, 1, 1)
                    * row["eye"]
                )

                # ========== 标准 MVDR 求解 ==========
                ones_subarray = row["ones"]

                try:
                    v = torch.linalg.solve(
                        covariance_loaded,
                        ones_subarray,
                    )  # [width, subarray_size, 1]
                except RuntimeError:
                    v = torch.linalg.pinv(covariance_loaded) @ ones_subarray

                denom = torch.matmul(ones_subarray.mH, v)  # [width, 1, 1]
                w = v / (denom + 1e-12)  # [width, subarray_size, 1]  归一化 MV 权重

                # ========== 合成输出 ==========
                t0 = self.temporal_win // 2
                x_subarray_center = x_subarrays[
                    :,
                    :,
                    :,
                    t0,
                ]  # [width, subarray_size, subarray_count]
                x_mean = x_subarray_center.mean(dim=2).unsqueeze(
                    -1
                )  # [width, subarray_size, 1]
                y_row_rx = torch.matmul(w.mH, x_mean)  # [width, 1, 1]
                if args.gcf_power > 0:
                    spatial_spectrum = torch.fft.fftshift(
                        torch.fft.fft(x_mean.squeeze(-1), dim=1),
                        dim=1,
                    )
                    power_spectrum = torch.abs(spatial_spectrum).square()
                    center = subarray_size // 2
                    low_bins = max(
                        0, min(args.gcf_low_bins, center, subarray_size - center - 1)
                    )
                    low_energy = power_spectrum[
                        :,
                        center - low_bins : center + low_bins + 1,
                    ].sum(dim=1)
                    total_energy = power_spectrum.sum(dim=1) + 1e-12
                    gcf = (low_energy / total_energy).clamp(0, 1)
                    y_row_rx = y_row_rx * torch.pow(
                        gcf.view(self.width, 1, 1),
                        args.gcf_power,
                    )

                # 3. 发射端相位旋转 (2D)
                phi_tx = 2.0 * np.pi * self.fc * (tx_row / fs)
                cos_tx = torch.cos(phi_tx).view(self.width, 1, 1)
                sin_tx = torch.sin(phi_tx).view(self.width, 1, 1)

                y_row = torch.complex(
                    y_row_rx.real * cos_tx - y_row_rx.imag * sin_tx,
                    y_row_rx.real * sin_tx + y_row_rx.imag * cos_tx,
                )

                y_row_2d = y_row.squeeze(-1).squeeze(-1)
                i_beam_sum[hz].add_(y_row_2d.real)
                q_beam_sum[hz].add_(y_row_2d.imag)

        i_output = (i_beam_sum / n_a).cpu().numpy()
        q_output = (q_beam_sum / n_a).cpu().numpy()
        return i_output, q_output


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
    dz = z_grid[1] - z_grid[0]
    dx = x_grid[1] - x_grid[0]
    depth_min = z_grid[0]
    depth_max = z_grid[-1]

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
            ("FBSS", "Enabled" if args.fbss else "Disabled"),
            ("DL factor", args.mv_dl),
            ("Subarray", args.subarray_ratio),
            ("Temporal win", args.temporal_win),
            ("GCF bins", args.gcf_low_bins),
            ("GCF power", args.gcf_power),
            ("Window", args.window.upper()),
            ("Interp", args.interp),
            ("TGC", "Enabled" if args.tgc else "Disabled"),
            *([("TGC Alpha", f"{args.tgc_alpha} dB/MHz/cm")] if args.tgc else []),
            ("DR", f"{args.dr} dB"),
        ],
    )

    # 初始化 GCF-MV + FBSS 波束合成器
    bf = RowDynamicMVBeamformerIQ(
        z_grid,
        x_grid,
        n_elem,
        pitch,
        c,
        fc,
        fs,
        t0_sub,
        angles_all,
        mv_dl=args.mv_dl,
        fbss=args.fbss,
        subarray_ratio=args.subarray_ratio,
        temporal_win=args.temporal_win,
    )

    print(
        f"Processing GCF-MV (DL={args.mv_dl}, GCF bins={args.gcf_low_bins}, power={args.gcf_power})...",
    )
    i_out, q_out, dt = run_beamformer(
        bf,
        i_sub,
        q_sub,
        selected_angles,
        t0_sub,
        fs,
        device,
    )
    mv_db = envelope_to_db(i_out, q_out, z_grid, fc, args.tgc, args.tgc_alpha)
    mv_db = validate_db_output(mv_db, (height, width), METHOD_NAME)

    # ========== 生成图标题(所有参数简写) ==========
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
        title_parts.append(f"{np.degrees(selected_angles[0]):.1f}°")
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

    method_dir = save_algorithm_result(
        mv_db,
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

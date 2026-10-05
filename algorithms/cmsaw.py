"""Provide Python utilities for cmsaw."""

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
    interpolate_multi_angle_channel_samples,
    nonnegative_float,
    prepare_beamforming_input,
    print_physical_summary,
    positive_odd_int,
    resolve_project_path,
    run_beamformer,
    save_algorithm_result,
    unit_interval_float,
    validate_db_output,
)

# ================= 命令行参数配置 =================
parser = argparse.ArgumentParser(
    description="CMSAW - Coherence-based Minimum Variance Adaptive Weighting",
)
add_common_arguments(parser, select_help="角度选择: center, 0,1,2(逗号分隔)")

# ---- MV参数(用于相位基线) ----
parser.add_argument(
    "--mv_dl",
    type=nonnegative_float,
    default=0.0,
    help="MV 对角加载系数",
)
parser.add_argument(
    "--subarray_ratio",
    type=unit_interval_float,
    default=0.25,
    help="MV 子阵列比例",
)
parser.add_argument(
    "--temporal_win",
    type=positive_odd_int,
    default=9,
    help="MV 时间平均窗口",
)
parser.add_argument(
    "--fbss",
    action="store_true",
    default=True,
    help="MV 启用FBSS",
)
parser.add_argument("--no_fbss", dest="fbss", action="store_false", help="MV 禁用FBSS")
# ---- CMSAW 核心参数 ----
parser.add_argument(
    "--lmax_ratio",
    type=float,
    default=0.5,
    help="最大子阵列长度比例 Lmax = ratio x aperture_size",
)
parser.add_argument("--min_subarray_len", type=int, default=2, help="最小子阵列长度")
parser.add_argument("--delta_max", type=float, default=1.0, help="对角线缩放因子最大值")
parser.add_argument("--gamma", type=float, default=0.5, help="权重幂次")
parser.add_argument(
    "--clip_percentile",
    type=float,
    default=90.0,
    help="权重裁剪百分位数",
)
parser.add_argument(
    "--depth_smooth_rows",
    type=int,
    default=1,
    help="深度平滑行数 (1=禁用)",
)

add_io_arguments(parser, save_gt_help="保存GT对比图")
args = None

METHOD_NAME = "cmsaw"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def validate_cmsaw_params(
    lmax_ratio,
    min_subarray_len,
    delta_max,
    gamma,
    clip_percentile,
    temporal_win,
    depth_smooth_rows,
):
    if not 0 < lmax_ratio <= 0.5 or min_subarray_len < 2:
        raise ValueError("Require lmax_ratio in (0, 0.5] and min_subarray_len >= 2")
    if not 0 <= delta_max <= 1 or not 0 < gamma <= 1 or not 0 < clip_percentile <= 100:
        raise ValueError(
            "Require delta_max in [0,1], gamma in (0,1], clip_percentile in (0,100]",
        )
    if temporal_win < 1 or temporal_win % 2 == 0:
        raise ValueError("temporal_win must be a positive odd integer")
    if depth_smooth_rows < 1:
        raise ValueError("depth_smooth_rows must be at least 1")


def delayed_iq(sample, angle_idx):
    """Execute delayed iq."""
    i_data = torch.from_numpy(sample["I"][angle_idx]).to(device)
    q_data = torch.from_numpy(sample["Q"][angle_idx]).to(device)
    angles = torch.from_numpy(sample["angles"][angle_idx]).to(device)
    t0 = torch.from_numpy(np.atleast_1d(sample["t0"][angle_idx])).to(device)
    z = torch.from_numpy(sample["z"]).to(device)
    x = torch.from_numpy(sample["x"]).to(device)
    n_angles, n_samples, n_channels = i_data.shape

    z_mesh, x_mesh = torch.meshgrid(z, x, indexing="ij")
    elements = (
        torch.arange(n_channels, device=device) - (n_channels - 1) / 2.0
    ) * sample["pitch"]
    receive = torch.sqrt((x_mesh[..., None] - elements) ** 2 + z_mesh[..., None] ** 2)
    transmit = z_mesh[None, ..., None] * torch.cos(angles)[:, None, None, None]
    transmit += x_mesh[None, ..., None] * torch.sin(angles)[:, None, None, None]
    tof = (transmit + receive[None]) / sample["c"]
    exact = (tof - t0[:, None, None, None]) * sample["fs"]
    valid = (exact >= 0) & (exact <= n_samples - 1)

    angle_index = torch.arange(n_angles, device=device)[:, None, None, None]
    channel_offset = torch.arange(n_channels, device=device)[None, None, None, :]

    i, q = interpolate_multi_angle_channel_samples(
        i_data,
        q_data,
        exact,
        angle_index,
        channel_offset,
        args.interp,
    )

    phase_rx = 2.0 * torch.pi * sample["fc"] * (receive / sample["c"])
    cos_rx = torch.cos(phase_rx)[None]
    sin_rx = torch.sin(phase_rx)[None]
    aligned_i = i * cos_rx - q * sin_rx
    aligned_q = i * sin_rx + q * cos_rx
    return torch.complex(aligned_i, aligned_q) * valid


def cmsaw_weight_from_delayed_data(
    data,
    z_grid,
    x_grid,
    pitch,
    f_number,
    dynamic_aperture,
    window,
    lmax_ratio,
    min_subarray_len,
    delta_max,
    gamma,
    clip_percentile,
    depth_smooth_rows,
    return_lengths=False,
):
    """Compute one CMSAW map per angle and average the normalized maps."""
    if data.ndim == 3:
        data = data.unsqueeze(0)
    if data.ndim != 4 or data.shape[0] < 1:
        raise ValueError("delayed IQ 必须是 [A,height,width,C] 或 [height,width,C]")

    _, height, width, n_channels = data.shape
    data_device = data.device
    z_grid = np.asarray(z_grid, dtype=np.float32)
    x_grid = np.asarray(x_grid, dtype=np.float32)
    x_t = torch.from_numpy(x_grid).to(data_device)
    element_x = (
        torch.arange(n_channels, device=data_device) - (n_channels - 1) / 2.0
    ) * pitch
    centers = torch.argmin(torch.abs(x_t[:, None] - element_x[None]), dim=1)

    row_cache = []
    for depth in z_grid:
        k = dynamic_aperture_channel_count(
            depth,
            f_number,
            pitch,
            n_channels,
            dynamic_aperture,
        )
        starts = torch.clamp(centers - k // 2, 0, n_channels - k)
        channels = starts[:, None] + torch.arange(k, device=data_device)[None]
        row_cache.append(
            (k, channels, aperture_window_1d(k, window, data_device)[None]),
        )

    angle_weights = []
    angle_lengths = []
    for angle_data in data:
        sigma = torch.empty((height, width), dtype=torch.float32, device=data_device)
        active_rows = []
        k_rows = []
        for iz, (k, channels, aperture_window) in enumerate(row_cache):
            active = torch.gather(angle_data[iz], 1, channels) * aperture_window
            sigma[iz] = torch.std(torch.abs(active), dim=1, unbiased=False)
            active_rows.append(active)
            k_rows.append(k)

        sigma_prime = torch.pow(sigma + 1e-12, -1.0 / 3.0)
        sigma_prime = (sigma_prime - sigma_prime.min()) / (
            sigma_prime.max() - sigma_prime.min() + 1e-12
        )
        weight = torch.zeros_like(sigma)
        length_map = torch.zeros(
            (height, width),
            dtype=torch.int16,
            device=data_device,
        )

        for iz, active in enumerate(active_rows):
            k = k_rows[iz]
            lmax = min(
                max(min_subarray_len, int(np.floor(k * lmax_ratio))),
                k,
            )
            min_l = min(min_subarray_len, lmax)
            lengths = (
                torch.floor(sigma_prime[iz] * lmax)
                .long()
                .clamp(
                    min_l,
                    lmax,
                )
            )
            length_map[iz] = lengths.to(torch.int16)
            coherent = torch.abs(active.sum(dim=1)).square()
            total = k * torch.abs(active).square().sum(dim=1)
            coherence = (coherent / (total + 1e-12)).clamp(0, 1)
            delta = torch.pow(coherence + 1e-12, sigma_prime[iz]) * delta_max

            for length_tensor in torch.unique(lengths):
                length = int(length_tensor.item())
                columns = torch.where(lengths == length)[0]
                subset = active[columns]
                sub = subset.unfold(1, length, 1).transpose(1, 2)
                covariance = sub @ sub.mH / float(k - length + 1)
                eye = torch.eye(length, dtype=torch.complex64, device=data_device)
                exchange = torch.flip(eye, dims=(0,))
                transpose = covariance.transpose(-2, -1)
                rotary = 0.25 * (
                    covariance
                    + exchange @ transpose
                    + exchange @ covariance @ exchange
                    + transpose @ exchange
                )
                diagonal = torch.diag_embed(
                    torch.diagonal(rotary, dim1=-2, dim2=-1),
                )
                matrix = torch.abs(
                    rotary - delta[columns, None, None] * diagonal,
                )
                weight[iz, columns] = matrix.mean(dim=(-2, -1)) / (
                    matrix.std(dim=(-2, -1), unbiased=False) + 1e-12
                )

        if depth_smooth_rows > 1:
            rows = (
                depth_smooth_rows + 1
                if depth_smooth_rows % 2 == 0
                else depth_smooth_rows
            )
            if rows > height:
                raise ValueError(
                    f"depth_smooth_rows={depth_smooth_rows} 超过图像深度 {height}",
                )
            coord = torch.arange(rows, device=data_device).float() - rows // 2
            kernel = torch.exp(
                -0.5 * (coord / max(rows / 4.0, 1.0)) ** 2,
            )
            kernel = (kernel / kernel.sum()).view(1, 1, rows, 1)
            log_weight = torch.log(weight.clamp_min(1e-12))[None, None]
            log_weight = torch.nn.functional.pad(
                log_weight,
                (0, 0, rows // 2, rows // 2),
                mode="reflect",
            )
            weight = torch.exp(
                torch.nn.functional.conv2d(log_weight, kernel)[0, 0],
            )

        weight /= torch.median(weight.clamp_min(1e-12))
        weight.clamp_(max=torch.quantile(weight, clip_percentile / 100.0))
        weight.pow_(gamma)
        angle_weights.append(weight)
        angle_lengths.append(length_map)

    combined = torch.stack(angle_weights, dim=0).mean(dim=0)
    if return_lengths:
        return combined, torch.stack(angle_lengths, dim=0)
    return combined


class CMSAWBeamformerIQ:
    """Pack-time CMSAW teacher interface.

    CMSAW 本身是基于 MV baseline 的显示域/幅值加权方法;这里为了统一 H5 GT 打包接口,
    采用“MV 复数 IQ x CMSAW 实值权重”的定义:
      - 相位沿用 MV baseline;
      - 幅值按 CMSAW coherence 权重调整;
      - 返回 i_output, q_output,供 pack_EPFL.py 后续统一生成 envdb。
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
        t0_all,
        angles_rad,
        mv_dl=0.0,
        fbss=True,
        subarray_ratio=0.25,
        temporal_win=9,
        lmax_ratio=0.5,
        min_subarray_len=2,
        delta_max=1.0,
        gamma=0.5,
        clip_percentile=90.0,
        depth_smooth_rows=1,
    ):
        """Initialize the instance."""
        validate_cmsaw_params(
            lmax_ratio,
            min_subarray_len,
            delta_max,
            gamma,
            clip_percentile,
            temporal_win,
            depth_smooth_rows,
        )

        self.z_grid = np.asarray(z_grid, dtype=np.float32)
        self.x_grid = np.asarray(x_grid, dtype=np.float32)
        self.n_elem = int(n_elem)
        self.pitch = float(pitch)
        self.c = float(c)
        self.fs = float(fs)
        self.fc = float(fc)
        self.lmax_ratio = float(lmax_ratio)
        self.min_subarray_len = int(min_subarray_len)
        self.delta_max = float(delta_max)
        self.gamma = float(gamma)
        self.clip_percentile = float(clip_percentile)
        self.depth_smooth_rows = int(depth_smooth_rows)

        from algorithms import mv as mv_module

        self.mv_module = mv_module
        self.mv_module.args = args
        self.mv_bf = self.mv_module.RowDynamicMVBeamformerIQ(
            self.z_grid,
            self.x_grid,
            self.n_elem,
            self.pitch,
            self.c,
            self.fc,
            self.fs,
            t0_all,
            angles_rad,
            mv_dl=mv_dl,
            fbss=fbss,
            subarray_ratio=subarray_ratio,
            temporal_win=temporal_win,
        )
        self.last_weight = None
        self.last_lengths = None

    def __call__(self, i_data, q_data, selected_angles, t_starts, fs):
        """Run the callable operation."""
        i_mv, q_mv = self.mv_bf(i_data, q_data, selected_angles, t_starts, fs)
        t0 = np.asarray(t_starts, dtype=np.float32).reshape(-1)
        if t0.size == 1:
            t0 = np.repeat(t0, len(selected_angles))
        sample = {
            "I": i_data.astype(np.float32),
            "Q": q_data.astype(np.float32),
            "t0": t0.astype(np.float32),
            "z": self.z_grid,
            "x": self.x_grid,
            "angles": np.asarray(selected_angles, dtype=np.float32),
            "fs": float(fs),
            "c": self.c,
            "fc": self.fc,
            "pitch": self.pitch,
        }
        angle_weights = []
        angle_lengths = []
        for angle_idx in range(len(selected_angles)):
            data = delayed_iq(sample, np.array([angle_idx]))
            angle_weight, angle_length = cmsaw_weight_from_delayed_data(
                data,
                self.z_grid,
                self.x_grid,
                self.pitch,
                args.f_number,
                args.dynamic_aperture,
                args.window,
                self.lmax_ratio,
                self.min_subarray_len,
                self.delta_max,
                self.gamma,
                self.clip_percentile,
                self.depth_smooth_rows,
                return_lengths=True,
            )
            angle_weights.append(angle_weight)
            angle_lengths.append(angle_length[0])
        weight = torch.stack(angle_weights, dim=0).mean(dim=0)
        self.last_weight = weight.detach()
        self.last_lengths = torch.stack(angle_lengths, dim=0).detach()
        weight_np = weight.cpu().numpy().astype(np.float32)
        return i_mv * weight_np, q_mv * weight_np


# ================= 主程序 =================
def main():
    """Run the command-line workflow."""
    global args
    args = parser.parse_args()
    h5_path = resolve_project_path(args.h5_path, PROJECT_ROOT)
    output_dir = Path(args.output_dir) / METHOD_NAME

    validate_cmsaw_params(
        args.lmax_ratio,
        args.min_subarray_len,
        args.delta_max,
        args.gamma,
        args.clip_percentile,
        args.temporal_win,
        args.depth_smooth_rows,
    )

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
            ("MV phase", "in-memory MV"),
            ("F-Number", args.f_number),
            ("Aperture", "Dynamic" if args.dynamic_aperture else "Fixed Full"),
            ("FBSS", "Enabled" if args.fbss else "Disabled"),
            ("DL factor", args.mv_dl),
            ("Subarray", args.subarray_ratio),
            ("Temporal win", args.temporal_win),
            ("Window", args.window.upper()),
            ("Interp", args.interp),
            ("Lmax", args.lmax_ratio),
            ("Min subarray", args.min_subarray_len),
            ("Delta max", args.delta_max),
            ("Gamma", args.gamma),
            ("Clip", f"{args.clip_percentile}%"),
            ("Depth smooth", args.depth_smooth_rows),
            ("TGC", "Enabled" if args.tgc else "Disabled"),
            *([("TGC Alpha", f"{args.tgc_alpha} dB/MHz/cm")] if args.tgc else []),
            ("DR", f"{args.dr} dB"),
        ],
    )

    print(
        f"Processing CMSAW (Lmax={args.lmax_ratio}aperture_size, gamma={args.gamma})..."
    )
    mv_bf = CMSAWBeamformerIQ(
        z_grid,
        x_grid,
        n_elem,
        pitch,
        c,
        fc,
        fs,
        t0_sub,
        selected_angles,
        mv_dl=args.mv_dl,
        fbss=args.fbss,
        subarray_ratio=args.subarray_ratio,
        temporal_win=args.temporal_win,
        lmax_ratio=args.lmax_ratio,
        min_subarray_len=args.min_subarray_len,
        delta_max=args.delta_max,
        gamma=args.gamma,
        clip_percentile=args.clip_percentile,
        depth_smooth_rows=args.depth_smooth_rows,
    )
    with torch.no_grad():
        i_out, q_out, dt = run_beamformer(
            mv_bf,
            i_sub,
            q_sub,
            selected_angles,
            t0_sub,
            fs,
            device,
        )
    output = envelope_to_db(i_out, q_out, z_grid, fc, args.tgc, args.tgc_alpha)
    output = validate_db_output(output, (height, width), METHOD_NAME)
    weight_np = mv_bf.last_weight.cpu().numpy().astype(np.float32)
    dynamic_l_np = mv_bf.last_lengths.cpu().numpy()

    # ========== 生成图标题(所有参数简写) ==========
    title_parts = [
        f"F{args.f_number}",
        f"{args.window.upper()}",
        f"{args.interp[:3]}",
        f"subarray_size{args.lmax_ratio}",
        f"min{args.min_subarray_len}",
        f"g{args.gamma}",
        f"P{args.clip_percentile:.0f}",
    ]
    if len(selected_angles) == 1:
        title_parts.append(f"{np.degrees(selected_angles[0]):.1f}°")
    else:
        title_parts.append(f"{len(selected_angles)}A")
    if args.dynamic_aperture:
        title_parts.append("Dyn")
    else:
        title_parts.append("Full")
    if args.depth_smooth_rows > 1:
        title_parts.append(f"Sm{args.depth_smooth_rows}")
    title_params = " | ".join(title_parts)

    out_name = METHOD_NAME

    method_dir = save_algorithm_result(
        output,
        input_data,
        args,
        METHOD_NAME,
        title_params,
        dt,
        extra_arrays={"weight": weight_np, "subarray_length": dynamic_l_np},
        extra_params={
            "method_dir": str(Path(args.output_dir) / METHOD_NAME),
            "mv_phase_source": "in_memory",
        },
    )

    print(f"  Weight p1/p50/p99 = {np.percentile(weight_np, [1, 50, 99])}")
    print(f"  GPU Time: {dt:.2f}s | Saved -> {out_name}")
    print(f"\nDone | Output: {method_dir}")


if __name__ == "__main__":
    main()

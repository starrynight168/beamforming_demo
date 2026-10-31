"""Provide Python utilities for mv."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.common import (
    add_common_arguments,
    add_io_arguments,
    build_row_dynamic_geometry,
    beamforming_depth_grid,
    beamformed_image_db,
    interpolate_channel_samples,
    interpolate_multi_angle_channel_samples,
    optional_aperture_window_1d,
    nonnegative_float,
    normalized_mv_weights,
    mv_row_output,
    positive_odd_int,
    prepare_beamforming_input,
    print_physical_summary,
    resolve_project_path,
    run_beamformer,
    save_algorithm_result,
    spatial_temporal_covariance,
    time_start_tensor,
    unit_interval_float,
    validate_db_output,
    validate_beamformer_input,
)

# ================= 命令行参数配置 =================
parser = argparse.ArgumentParser(description="MV+FBSS Beamforming")
add_common_arguments(parser, window_help="子阵空间平滑加权窗")
parser.add_argument(
    "--mv_axis", choices=("rx", "tx", "joint"), default="rx",
    help="MV方向：rx接收端，tx接收DAS后发射端，joint收发联合",
)
parser.add_argument(
    "--tx_subarray_ratio", type=unit_interval_float, default=None,
    help="发射子阵比例；默认沿用subarray_ratio",
)

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
add_io_arguments(parser)
args = None

METHOD_NAME = "mv"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_JOINT_SNAPSHOT_TARGET_BYTES = 64 * 1024**2


class RowDynamicMVBeamformerIQ:
    """标准 MV + FBSS 自适应波束形成器.

    IQ 使用 positive 相位补偿；实数 RF 和解析 RF 直接按传播时间取样。
    """

    normalize_covariance = False

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
        options=None,
        signal_domain=None,
    ):
        """Initialize the instance."""
        self.options = options if options is not None else self.get_options()
        signal_domain = self.options.signal_domain if signal_domain is None else signal_domain
        self.mv_axis = getattr(self.options, "mv_axis", "rx")
        if signal_domain not in ("iq", "rf", "analytic_rf"):
            raise ValueError("signal_domain must be iq, rf or analytic_rf")
        self.signal_domain = signal_domain
        self.device = self.get_device()
        mv_dl = nonnegative_float(mv_dl)
        geometry = build_row_dynamic_geometry(
            z_grid,
            x_grid,
            n_elem,
            pitch,
            c,
            fc,
            fs,
            self.options.f_number,
            self.options.dynamic_aperture,
            self.options.window,
            subarray_ratio,
            temporal_win,
            self.device,
        )
        self.__dict__.update(geometry)
        self.dl_factor = mv_dl
        self.use_fbss = fbss
        self.subarray_ratio = subarray_ratio
        self.temporal_win = temporal_win
        self._covariance_windows = {
            row["aperture_size"]: row["window"] for row in self.row_cache
        }

    def __call__(self, i_data, q_data, selected_angles, t_starts, fs):
        """Run the callable operation."""
        validate_beamformer_input(i_data, q_data, selected_angles, fs, self.fs, self.N)
        if self.signal_domain == "rf" and np.any(q_data != 0):
            raise ValueError("Real RF requires an all-zero imaginary component")
        n_a = i_data.shape[0]
        i_tensor = torch.from_numpy(i_data.astype(np.float32)).to(self.device)
        q_tensor = torch.from_numpy(q_data.astype(np.float32)).to(self.device)
        n_s = i_tensor.shape[1]
        if self.mv_axis != "rx":
            if n_a < 2:
                raise ValueError("tx/joint MV requires at least two transmit angles")
            return self._beamform_transmit(
                i_tensor, q_tensor, np.asarray(selected_angles), t_starts, fs,
            )

        half_win = self.temporal_win // 2
        offsets = torch.arange(-half_win, half_win + 1, device=self.device)
        offsets_f = offsets.float().view(1, 1, -1)
        cos_a = torch.from_numpy(np.cos(selected_angles).astype(np.float32)).to(self.device)
        sin_a = torch.from_numpy(np.sin(selected_angles).astype(np.float32)).to(self.device)
        t_starts_t = time_start_tensor(t_starts, n_a, fs, self.device)
        max_sample = float(n_s - 1)

        i_beam_sum = torch.zeros(
            (self.height, self.width), dtype=torch.float32, device=self.device
        )
        q_beam_sum = torch.zeros(
            (self.height, self.width), dtype=torch.float32, device=self.device
        )

        # 1. 预计算接收端相位旋转因子 (全网格 3D)
        if self.signal_domain == "iq":
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
                valid_mask = ((sample >= 0) & (sample <= max_sample)).float()
                sample.clamp_(0.0, max_sample)

                i_samples, q_samples = interpolate_channel_samples(
                    i_angle,
                    q_angle,
                    sample,
                    self.ch,
                    self.options.interp,
                )

                # 2. 接收端相位旋转 (3D)
                if self.signal_domain == "iq":
                    cos_rx = cos_rx_global[hz].unsqueeze(-1)
                    sin_rx = sin_rx_global[hz].unsqueeze(-1)
                    i_rx = i_samples * cos_rx - q_samples * sin_rx
                    q_rx = i_samples * sin_rx + q_samples * cos_rx
                else:
                    i_rx, q_rx = i_samples, q_samples

                x_flat = torch.complex(i_rx, q_rx) * valid_mask
                row = self.row_cache[hz]
                aperture_size, subarray_size, _subarray_count = (
                    row["aperture_size"],
                    row["subarray_size"],
                    row["subarray_count"],
                )

                x_active = torch.gather(
                    x_flat, 1, row["ch_idx_t"]
                )  # [width, aperture_size, T]
                x_subarrays, covariance = self.estimate_covariance(
                    x_active,
                    subarray_size,
                    self.temporal_win,
                    self.use_fbss,
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

                w = self.solve_weights(covariance_loaded, ones_subarray)

                # ========== 合成输出 ==========
                y_row_rx = mv_row_output(w, x_subarrays, x_active, valid_mask, row)
                y_row_rx = self.weight_rx_output(y_row_rx, x_active)

                # 3. 发射端相位旋转 (2D)
                if self.signal_domain == "iq":
                    phi_tx = 2.0 * np.pi * self.fc * (tx_row / fs)
                    cos_tx = torch.cos(phi_tx).view(self.width, 1, 1)
                    sin_tx = torch.sin(phi_tx).view(self.width, 1, 1)
                    y_row = torch.complex(
                        y_row_rx.real * cos_tx - y_row_rx.imag * sin_tx,
                        y_row_rx.real * sin_tx + y_row_rx.imag * cos_tx,
                    )
                else:
                    y_row = y_row_rx

                y_row_2d = y_row.squeeze(-1).squeeze(-1)
                i_beam_sum[hz].add_(y_row_2d.real)
                q_beam_sum[hz].add_(y_row_2d.imag)

        i_output = (i_beam_sum / n_a).cpu().numpy()
        q_output = (q_beam_sum / n_a).cpu().numpy()
        return i_output, q_output

    def _beamform_transmit(self, i_tensor, q_tensor, selected_angles, t_starts, fs):
        n_angles = i_tensor.shape[0]
        order = np.argsort(selected_angles)
        angles = np.asarray(selected_angles, dtype=np.float64)[order]
        spacing = np.diff(angles)
        if np.any(spacing <= 0):
            raise ValueError("Transmit MV requires unique transmit angles")
        median_spacing = float(np.median(spacing))
        if np.max(np.abs(spacing - median_spacing)) > 0.2 * median_spacing:
            raise ValueError(
                "Transmit MV requires approximately uniform transmit-angle spacing; "
                "use all angles or a uniformly spaced selection",
            )
        order_t = torch.as_tensor(order, device=self.device)
        i_tensor = i_tensor[order_t]
        q_tensor = q_tensor[order_t]
        starts = time_start_tensor(t_starts, n_angles, fs, self.device)[order_t]
        cos_a = torch.as_tensor(np.cos(angles), dtype=torch.float32, device=self.device)
        sin_a = torch.as_tensor(np.sin(angles), dtype=torch.float32, device=self.device)
        half_win = self.temporal_win // 2
        offsets = torch.arange(-half_win, half_win + 1, device=self.device).float()
        max_sample = float(i_tensor.shape[1] - 1)
        i_output = torch.zeros((self.height, self.width), device=self.device)
        q_output = torch.zeros_like(i_output)
        angle_index = torch.arange(n_angles, device=self.device).view(n_angles, 1, 1, 1)
        channel_index = self.ch.expand(1, 1, self.N, 1)
        tx_ratio = self.options.tx_subarray_ratio
        if tx_ratio is None:
            tx_ratio = self.subarray_ratio
        tx_size = min(max(int(n_angles * tx_ratio), 2), n_angles)
        tx_count = n_angles - tx_size + 1
        tx_weights = optional_aperture_window_1d(
            tx_count, self.options.window, self.device,
        )
        tx_eye = torch.eye(tx_size, dtype=torch.complex64, device=self.device).unsqueeze(0)

        for hz in range(self.height):
            row = self.row_cache[hz]
            rx_size = 1 if self.mv_axis == "tx" else row["subarray_size"]
            snapshot_elements = max(
                rx_size * min(row["subarray_count"], 16) * n_angles * self.temporal_win,
                tx_size * min(tx_count, 16) * row["aperture_size"] * self.temporal_win,
                n_angles * self.N * self.temporal_win,
            )
            column_block = min(
                self.width, max(32, _JOINT_SNAPSHOT_TARGET_BYTES // (8 * snapshot_elements)),
            )
            for x_start in range(0, self.width, column_block):
                x_stop = min(x_start + column_block, self.width)
                x_slice = slice(x_start, x_stop)
                block_width = x_stop - x_start
                tx_row = (
                    self.z_mesh[hz, x_slice].unsqueeze(0) * self.sc * cos_a.unsqueeze(1)
                    + self.x_mesh[hz, x_slice].unsqueeze(0) * self.sc * sin_a.unsqueeze(1)
                )
                drs = self.drs[hz, x_slice]
                sample = (
                    tx_row[:, :, None]
                    + drs.unsqueeze(0)
                    - starts.view(n_angles, 1, 1)
                ).unsqueeze(-1) + offsets.view(1, 1, 1, -1)
                valid = (sample >= 0) & (sample <= max_sample)
                sample = sample.clamp(0.0, max_sample)
                i_samples, q_samples = interpolate_multi_angle_channel_samples(
                    i_tensor,
                    q_tensor,
                    sample,
                    angle_index,
                    channel_index,
                    self.options.interp,
                )
                if self.signal_domain == "iq":
                    phi_tx = 2.0 * np.pi * self.fc * tx_row / fs
                    phi_rx = 2.0 * np.pi * self.fc * drs / fs
                    phi = phi_tx[:, :, None, None] + phi_rx.unsqueeze(0).unsqueeze(-1)
                    cos_phi, sin_phi = torch.cos(phi), torch.sin(phi)
                    i_aligned = i_samples * cos_phi - q_samples * sin_phi
                    q_aligned = i_samples * sin_phi + q_samples * cos_phi
                else:
                    i_aligned, q_aligned = i_samples, q_samples
                data = torch.complex(i_aligned, q_aligned) * valid
                row_indices = row["ch_idx_t"][x_slice]
                data = torch.gather(
                    data,
                    2,
                    row_indices.unsqueeze(0).expand(n_angles, -1, -1, -1),
                ).permute(1, 0, 2, 3)
                valid = torch.gather(
                    valid,
                    2,
                    row_indices.unsqueeze(0).expand(n_angles, -1, -1, -1),
                ).permute(1, 0, 2, 3)

                if self.mv_axis == "tx":
                    receive_weights = valid.to(data.real.dtype)
                    if row["aperture_window"] is not None:
                        receive_weights = receive_weights * row["aperture_window"].view(1, 1, -1, 1)
                    weight_sum = receive_weights.sum(dim=2, keepdim=True)
                    data = (data * receive_weights).sum(dim=2, keepdim=True) / weight_sum.clamp_min(1e-12)
                    valid = weight_sum > 0
                    rx_mv = torch.ones((block_width, 1, 1), dtype=data.dtype, device=self.device)
                else:
                    rx_cov = self.estimate_joint_covariance(
                        data, valid, 2, rx_size, row["window"],
                    )
                    rx_cov = _load_joint_covariance(rx_cov, self.dl_factor, row["eye"])
                    rx_mv = self.solve_weights(rx_cov, row["ones"][x_slice])
                tx_cov = self.estimate_joint_covariance(
                    data, valid, 1, tx_size, tx_weights,
                )
                tx_cov = _load_joint_covariance(tx_cov, self.dl_factor, tx_eye)
                tx_steering = torch.ones(
                    (block_width, tx_size, 1), dtype=torch.complex64, device=self.device,
                )
                tx_mv = self.solve_weights(tx_cov, tx_steering)

                center = self.temporal_win // 2
                rx_values = data[..., center].unfold(2, rx_size, 1)
                rx_filtered = torch.einsum(
                    "wr,wagr->wag", rx_mv.conj().squeeze(-1), rx_values,
                )
                tx_values = rx_filtered.unfold(1, tx_size, 1).permute(0, 1, 3, 2)
                z_matrix = torch.einsum(
                    "wl,wglr->wgr", tx_mv.conj().squeeze(-1), tx_values,
                )
                valid_rx = valid[..., center].unfold(2, rx_size, 1).all(dim=-1)
                valid_both = valid_rx.unfold(1, tx_size, 1).all(dim=-1)
                z_matrix = torch.where(valid_both, z_matrix, torch.zeros_like(z_matrix))
                output = self.combine_subarrays(z_matrix)
                i_output[hz, x_slice] = output.real
                q_output[hz, x_slice] = output.imag

        return i_output.cpu().numpy(), q_output.cpu().numpy()

    def get_options(self):
        return args if args is not None else parser.parse_args([])

    def solve_weights(self, covariance, steering):
        return normalized_mv_weights(covariance, steering)

    def weight_rx_output(self, output, data):
        return output

    def combine_subarrays(self, values):
        return values.mean(dim=(-2, -1))

    def estimate_joint_covariance(self, data, valid, axis, subarray_size, weights):
        return joint_spatial_covariance(data, valid, axis, subarray_size, self.use_fbss, weights)

    def get_device(self):
        return device

    def estimate_covariance(self, x_active, subarray_size, temporal_win, use_fbss):
        return spatial_temporal_covariance(
            x_active, subarray_size, temporal_win, use_fbss,
            spatial_weights=self._covariance_windows[x_active.shape[1]],
            normalize=self.normalize_covariance,
        )


def joint_spatial_covariance(data, valid, axis, subarray_size, use_fbss, weights, cross=False):
    width = data.shape[0]
    scale = data.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(
        torch.finfo(data.real.dtype).tiny,
    )
    unfolded = (data / scale).unfold(axis, subarray_size, 1)
    valid_windows = valid.unfold(axis, subarray_size, 1).all(dim=-1)
    if axis == 2:
        window_count = unfolded.shape[2]
        n_outer, n_inner = data.shape[1], data.shape[3]
    else:
        window_count = unfolded.shape[1]
        n_outer, n_inner = data.shape[2], data.shape[3]
    covariance = torch.zeros(
        (width, subarray_size, subarray_size), dtype=data.dtype, device=data.device,
    )
    denominator = torch.zeros(width, dtype=data.real.dtype, device=data.device)
    if cross:
        mean = data.new_zeros((width, subarray_size, n_outer, n_inner))
        weight_sum = data.real.new_zeros((width, n_outer, n_inner))
    for start in range(0, window_count, 16):
        stop = min(start + 16, window_count)
        count = stop - start
        if axis == 2:
            block = unfolded[:, :, start:stop]
            snapshots = block.permute(0, 4, 1, 2, 3).reshape(width, subarray_size, -1)
            mask = valid_windows[:, :, start:stop]
            weights_shape = (1, 1, count, 1)
            repeats = (width, n_outer, count, n_inner)
        else:
            block = unfolded[:, start:stop]
            snapshots = block.permute(0, 4, 1, 2, 3).reshape(width, subarray_size, -1)
            mask = valid_windows[:, start:stop]
            weights_shape = (1, count, 1, 1)
            repeats = (width, count, n_outer, n_inner)
        if weights is None:
            sample_weights = torch.ones(
                weights_shape, dtype=data.real.dtype, device=data.device,
            )
        else:
            sample_weights = weights[start:stop].view(weights_shape)
        sample_weights = sample_weights.expand(repeats).reshape(width, -1)
        sample_weights = sample_weights * mask.reshape(width, -1).to(data.real.dtype)
        if cross:
            block_weights = sample_weights.view(repeats)
            mean += (block * block_weights.unsqueeze(-1)).sum(dim=axis).permute(0, 3, 1, 2)
            weight_sum += block_weights.sum(dim=axis)
        else:
            snapshots = snapshots * sample_weights.sqrt().unsqueeze(1)
            covariance += snapshots @ snapshots.mH
            denominator += sample_weights.sum(dim=-1)
    if cross:
        mean = mean / weight_sum.clamp_min(torch.finfo(data.real.dtype).tiny).unsqueeze(1)
        mean = mean.flatten(2)
        covariance = mean @ mean.mH
        denominator = (weight_sum > 0).sum(dim=(1, 2)).to(data.real.dtype)
    covariance = covariance / denominator.clamp_min(
        torch.finfo(data.real.dtype).tiny,
    ).view(-1, 1, 1)
    if use_fbss:
        covariance = 0.5 * (covariance + covariance.conj().flip(-1, -2))
    return covariance


def _load_joint_covariance(covariance, dl_factor, eye):
    trace = covariance.diagonal(dim1=-2, dim2=-1).real.sum(-1)
    return covariance + (dl_factor / covariance.shape[-1]) * trace.view(-1, 1, 1) * eye


# ================= 主程序 =================
def main(
    beamformer_class=RowDynamicMVBeamformerIQ,
    method_name=METHOD_NAME,
    method_label="MV",
    covariance_label="Subarray auto-covariance",
    argument_parser=None,
):
    """Run the command-line workflow."""
    global args
    args = (parser if argument_parser is None else argument_parser).parse_args()
    h5_path = resolve_project_path(args.h5_path, PROJECT_ROOT)
    output_dir = f"{args.output_dir}/{method_name}"

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
    compute_z = beamforming_depth_grid(z_grid, c, fs, args.signal_domain)
    selected_angles = input_data.selected_angles
    i_sub, q_sub, t0_sub = input_data.i_data, input_data.q_data, input_data.t0
    has_gt = sample.has_gt
    height, width = len(z_grid), len(x_grid)
    dz = z_grid[1] - z_grid[0]
    dx = x_grid[1] - x_grid[0]
    depth_min = z_grid[0]
    depth_max = z_grid[-1]

    print_physical_summary(
        method_name=method_name,
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
            ("MV axis", args.mv_axis),
            ("Covariance estimator", covariance_label),
            ("F-Number", args.f_number),
            ("Aperture", "Dynamic" if args.dynamic_aperture else "Fixed Full"),
            ("FBSS", "Enabled" if args.fbss else "Disabled"),
            ("DL factor", args.mv_dl),
            ("Subarray", args.subarray_ratio),
            ("Temporal win", args.temporal_win),
            ("Window", args.window.upper()),
            ("Interp", args.interp),
            ("TGC", "Enabled" if args.tgc else "Disabled"),
            *([("TGC Alpha", f"{args.tgc_alpha} dB/MHz/cm")] if args.tgc else []),
            ("DR", f"{args.dr} dB"),
        ],
    )

    # 初始化标准 MV + FBSS 波束合成器
    bf = beamformer_class(
        compute_z,
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
        options=args,
    )

    print(f"Processing {method_label} (DL={args.mv_dl})...")
    i_out, q_out, dt = run_beamformer(
        bf,
        i_sub,
        q_sub,
        selected_angles,
        t0_sub,
        fs,
        device,
    )
    mv_db = beamformed_image_db(i_out, q_out, compute_z, z_grid, fc, args.tgc, args.tgc_alpha, args.signal_domain)
    mv_db = validate_db_output(mv_db, (height, width), method_name)

    # ========== 生成图标题(所有参数简写) ==========
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

    method_dir = save_algorithm_result(
        mv_db,
        input_data,
        args,
        method_name,
        title_params,
        dt,
    )

    print(f"  GPU Time: {dt:.2f}s | Saved -> {method_name}")
    print(f"\nDone | Output: {method_dir}")


if __name__ == "__main__":
    main()

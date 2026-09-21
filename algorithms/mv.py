"""Provide Python utilities for mv."""

import argparse
import os
import time

import numpy as np
import torch

from algorithms.common import (
    add_common_arguments,
    add_io_arguments,
    dynamic_aperture_channel_count,
    interpolate_channel_samples,
    load_from_h5,
    nonnegative_float,
    optional_aperture_window_1d,
    parse_selected_angles,
    print_physical_summary,
    positive_odd_int,
    resolve_project_path,
    save_comparison_figure,
    save_figure,
    tgc_gain,
    unit_interval_float,
    validate_db_output,
    write_params,
)

COMPARISON_VALUE_3 = 3

# ================= 命令行参数配置 =================
parser = argparse.ArgumentParser(description="MV+FBSS Beamforming")
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
add_io_arguments(parser)
args = None

METHOD_NAME = "mv"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


H5_PATH = None
OUTPUT_DIR = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RowDynamicMVBeamformerIQ:
    """标准 MV + FBSS 自适应波束形成器.

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
        self.height, self.width, self.N = len(z_grid), len(x_grid), n_elem
        self.x_grid = torch.from_numpy(x_grid).float().to(device)
        self.z_grid = torch.from_numpy(z_grid).float().to(device)
        self.angles_rad = torch.from_numpy(angles_rad).float().to(device)
        self.pitch, self.fc, self.fs, self.sc = pitch, float(fc), fs, fs / c
        self.dl_factor = mv_dl
        self.use_fbss = fbss
        self.subarray_ratio = subarray_ratio
        self.temporal_win = temporal_win

        # 预计算每个横向像素对应的最近阵元索引
        lateral_channel = (torch.arange(self.N, device=device).float() - (self.N - 1) / 2.0) * self.pitch
        self.c_idx = torch.argmin(
            torch.abs(self.x_grid.unsqueeze(1) - lateral_channel.unsqueeze(0)),
            dim=1,
        )
        x_mesh, z_mesh = torch.meshgrid(self.x_grid, self.z_grid, indexing="xy")
        self.x_mesh, self.z_mesh = x_mesh, z_mesh
        self.drs = torch.sqrt((x_mesh[..., None] - lateral_channel) ** 2 + z_mesh[..., None] ** 2) * self.sc
        self.ch = torch.arange(self.N, device=device).view(1, self.N, 1)
        self.row_cache = self._build_row_cache()

    def _aperture_window(self, k):
        """Execute  aperture window."""
        return optional_aperture_window_1d(k, args.window, device)

    def _build_row_cache(self):
        """Execute  build row cache."""
        cache = []
        for hz in range(self.height):
            depth = float(self.z_grid[hz].item())
            k = dynamic_aperture_channel_count(
                depth,
                args.f_number,
                self.pitch,
                self.N,
                args.dynamic_aperture,
            )
            subarray_len = min(max(int(k * self.subarray_ratio), 2), k)
            m = k - subarray_len + 1
            idx_start = torch.clamp(self.c_idx - k // 2, 0, self.N - k)
            ch_idx = idx_start.unsqueeze(1) + torch.arange(k, device=device).unsqueeze(
                0,
            )
            cache.append(
                {
                    "aperture_size": k,
                    "subarray_size": subarray_len,
                    "subarray_count": m,
                    "ch_idx_t": ch_idx.unsqueeze(-1).expand(-1, -1, self.temporal_win),
                    "window": self._aperture_window(k),
                    "eye": torch.eye(
                        subarray_len,
                        dtype=torch.complex64,
                        device=device,
                    ).unsqueeze(0),
                    "ones": torch.ones(
                        (self.width, subarray_len, 1),
                        dtype=torch.complex64,
                        device=device,
                    ),
                },
            )
        return cache

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
        t_starts_arr = np.asarray(t_starts, dtype=np.float32).reshape(-1)
        if t_starts_arr.size == 1:
            t_starts_arr = np.repeat(t_starts_arr, n_a)
        t_starts_t = torch.from_numpy(t_starts_arr[:n_a]).to(device) * fs
        max_sample = float(n_s - 2)

        i_beam_sum = torch.zeros((self.height, self.width), dtype=torch.float32, device=device)
        q_beam_sum = torch.zeros((self.height, self.width), dtype=torch.float32, device=device)

        # 1. 预计算接收端相位旋转因子 (全网格 3D)
        phi_rx_global = 2.0 * np.pi * self.fc * (self.drs / fs)
        cos_rx_global = torch.cos(phi_rx_global)
        sin_rx_global = torch.sin(phi_rx_global)

        for i in range(n_a):
            i_angle = i_tensor[i]
            q_angle = q_tensor[i]

            for hz in range(self.height):
                tx_row = self.z_mesh[hz] * self.sc * cos_a[i] + self.x_mesh[hz] * self.sc * sin_a[i]
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

                x_active = torch.gather(x_flat, 1, row["ch_idx_t"])  # [width, aperture_size, T]
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
                    covariance + (self.dl_factor / subarray_size) * trace.view(self.width, 1, 1) * row["eye"]
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
                x_mean = x_subarray_center.mean(dim=2).unsqueeze(-1)  # [width, subarray_size, 1]
                y_row_rx = torch.matmul(w.mH, x_mean)  # [width, 1, 1]

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
    dz = z_grid[1] - z_grid[0]
    dx = x_grid[1] - x_grid[0]
    depth_min = z_grid[0]
    depth_max = z_grid[-1]
    extent_mm = [
        x_grid[0] * 1000,
        x_grid[-1] * 1000,
        z_grid[-1] * 1000,
        z_grid[0] * 1000,
    ]

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

    # TGC
    tgc = tgc_gain(z_grid, fc, args.tgc_alpha) if args.tgc else np.ones_like(z_grid)

    print(f"Processing MV (DL={args.mv_dl})...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    i_out, q_out = bf(i_sub, q_sub, selected_angles, t0_sub, fs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.time() - t1

    env = (i_out**2 + q_out**2) * (tgc[:, None] ** 2)
    env /= env.max() + 1e-24
    mv_db = 10 * np.log10(env + 1e-24)
    mv_db = validate_db_output(mv_db, (height, width), METHOD_NAME)

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

    out_name = METHOD_NAME

    np.save(os.path.join(OUTPUT_DIR, f"{out_name}.npy"), mv_db)

    save_figure(
        mv_db,
        extent_mm,
        os.path.join(OUTPUT_DIR, f"{out_name}.png"),
        title_params,
        dr=args.dr,
        method_name=METHOD_NAME,
    )

    if has_gt and args.save_gt:
        save_comparison_figure(
            mv_db,
            gt_data,
            extent_mm,
            os.path.join(OUTPUT_DIR, f"{out_name}_comparison.png"),
            title_params,
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

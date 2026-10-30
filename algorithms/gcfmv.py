"""Provide Python utilities for gcfmv."""

from copy import deepcopy
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from algorithms import mv
from algorithms.common import (
    beamforming_depth_grid,
    beamformed_image_db,
    nonnegative_float,
    nonnegative_int,
    generalized_coherence_factor,
    prepare_beamforming_input,
    print_physical_summary,
    resolve_project_path,
    run_beamformer,
    save_algorithm_result,
    validate_db_output,
)

# ================= 命令行参数配置 =================
parser = deepcopy(mv.parser)
parser.description = "GCFMV Beamforming"
parser.add_argument(
    "--gcf_low_bins",
    type=nonnegative_int,
    default=1,
    help="GCF低空间频率半宽;1表示保留中心频点及左右各1个频点",
)
parser.add_argument(
    "--gcf_power",
    type=nonnegative_float,
    default=1.0,
    help="GCF后加权指数;1为标准GCF,0表示退化为MV",
)
parser.add_argument(
    "--gcf_tx_low_bins",
    type=nonnegative_int,
    default=None,
    help="多角度 Tx 方向 GCF 低频半宽;默认沿用 gcf_low_bins",
)

args = None

METHOD_NAME = "gcfmv"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RowDynamicMVBeamformerIQ(mv.RowDynamicMVBeamformerIQ):
    normalize_covariance = True

    def get_options(self):
        return args if args is not None else parser.parse_args([])

    def get_device(self):
        return device

    def weight_rx_output(self, output, data):
        if self.options.gcf_power == 0:
            return output
        factor = generalized_coherence_factor(data[..., self.temporal_win // 2], self.options.gcf_low_bins)
        return output * factor.pow(self.options.gcf_power).view(-1, 1, 1)

    def combine_subarrays(self, values):
        output = super().combine_subarrays(values) if self.mv_axis == "tx" else values.sum(dim=(-2, -1))
        if self.options.gcf_power == 0:
            return output
        tx_bins = self.options.gcf_tx_low_bins
        if tx_bins is None:
            tx_bins = self.options.gcf_low_bins
        return output * generalized_coherence_factor_2d(
            values, tx_bins, self.options.gcf_low_bins,
        ).pow(self.options.gcf_power)


def generalized_coherence_factor_2d(data, tx_low_bins, rx_low_bins):
    scale = data.abs().amax(dim=(-2, -1), keepdim=True)
    data = data / scale.clamp_min(torch.finfo(data.real.dtype).tiny)
    spectrum = torch.fft.fftshift(
        torch.fft.fft2(data, dim=(-2, -1)), dim=(-2, -1),
    ).abs().square()
    tx_center, rx_center = spectrum.shape[-2] // 2, spectrum.shape[-1] // 2
    tx_slice = slice(max(0, tx_center - tx_low_bins), min(spectrum.shape[-2], tx_center + tx_low_bins + 1))
    rx_slice = slice(max(0, rx_center - rx_low_bins), min(spectrum.shape[-1], rx_center + rx_low_bins + 1))
    low = spectrum[..., tx_slice, rx_slice].sum(dim=(-2, -1))
    total = spectrum.sum(dim=(-2, -1))
    return (low / total.clamp_min(torch.finfo(total.dtype).tiny)).clamp(0, 1)


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
    compute_z = beamforming_depth_grid(z_grid, c, fs, args.signal_domain)
    selected_angles = input_data.selected_angles
    beamforming_mode = {"rx": "1D_RX", "tx": "1D_TX", "joint": "2D_JTR"}[args.mv_axis]
    tx_subarray_ratio = (
        args.tx_subarray_ratio
        if args.tx_subarray_ratio is not None
        else args.subarray_ratio
    )
    gcf_tx_low_bins = (
        args.gcf_tx_low_bins
        if args.gcf_tx_low_bins is not None
        else args.gcf_low_bins
    )
    args.beamforming_mode = beamforming_mode
    args.tx_subarray_ratio_effective = tx_subarray_ratio
    args.gcf_tx_low_bins_effective = gcf_tx_low_bins
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
            ("Signal domain", args.signal_domain),
            ("Computed depth rows", len(compute_z)),
            ("F-Number", args.f_number),
            ("Aperture", "Dynamic" if args.dynamic_aperture else "Fixed Full"),
            ("FBSS", "Enabled" if args.fbss else "Disabled"),
            ("DL factor", args.mv_dl),
            ("Subarray", args.subarray_ratio),
            ("Temporal win", args.temporal_win),
            ("GCF bins", args.gcf_low_bins),
            ("GCF power", args.gcf_power),
            ("Beamforming mode", beamforming_mode),
            ("Tx subarray ratio", tx_subarray_ratio),
            ("Tx GCF bins", gcf_tx_low_bins),
            ("Window", args.window.upper()),
            ("Interp", args.interp),
            ("TGC", "Enabled" if args.tgc else "Disabled"),
            *([("TGC Alpha", f"{args.tgc_alpha} dB/MHz/cm")] if args.tgc else []),
            ("DR", f"{args.dr} dB"),
        ],
    )

    # 初始化 GCF-MV + FBSS 波束合成器
    bf = RowDynamicMVBeamformerIQ(
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
    )

    print(
        f"Processing GCF-MV {beamforming_mode} (DL={args.mv_dl}, GCF bins={args.gcf_low_bins}, power={args.gcf_power})...",
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
    mv_db = beamformed_image_db(i_out, q_out, compute_z, z_grid, fc, args.tgc, args.tgc_alpha, args.signal_domain)
    mv_db = validate_db_output(mv_db, (height, width), METHOD_NAME)

    # ========== 生成图标题(所有参数简写) ==========
    title_parts = [
        beamforming_mode,
        f"F{args.f_number}",
        f"DL{args.mv_dl}",
        f"sub{args.subarray_ratio}",
        f"Tw{args.temporal_win}",
        f"{args.window.upper()}",
        f"{args.interp[:3]}",
        f"GCFb{args.gcf_low_bins}",
        f"GCFp{args.gcf_power:g}",
    ]
    if len(selected_angles) > 1:
        title_parts.extend(
            [f"txsub{tx_subarray_ratio:g}", f"GCFtx{gcf_tx_low_bins}"],
        )
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
        METHOD_NAME,
        title_params,
        dt,
    )

    print(f"  GPU Time: {dt:.2f}s | Saved -> {METHOD_NAME}")
    print(f"\nDone | Output: {method_dir}")


if __name__ == "__main__":
    main()

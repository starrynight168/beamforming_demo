"""Provide Python utilities for esbmv."""

from copy import deepcopy
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from algorithms import mv
from algorithms.common import (
    closed_unit_interval_float,
    envelope_to_db,
    nonnegative_int,
    prepare_beamforming_input,
    print_physical_summary,
    resolve_project_path,
    run_beamformer,
    save_algorithm_result,
    validate_db_output,
)

MAX_GPU_EIGH_SUBARRAY_SIZE = 32

# ================= 命令行参数配置 =================
parser = deepcopy(mv.parser)
parser.description = "ESBMV Beamforming"
# ---- ESBMV 子空间参数 ----
parser.add_argument(
    "--num_eig",
    type=nonnegative_int,
    default=0,
    help="信号子空间特征向量数量;0 时使用 eig_threshold 自动选择",
)
parser.add_argument(
    "--eig_threshold",
    type=closed_unit_interval_float,
    default=0.05,
    help="自动选择时保留加载协方差中 lambda >= threshold * lambda_max 的特征向量",
)
parser.add_argument(
    "--force_gpu_eigh",
    action="store_true",
    default=False,
    help="强制在 GPU 上进行特征值分解",
)

args = None

METHOD_NAME = "esbmv"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def project_signal_subspace(covariance, weights, num_eig, threshold, force_gpu=False):
    target_device = covariance.device
    if target_device.type == "cuda" and covariance.shape[-1] > MAX_GPU_EIGH_SUBARRAY_SIZE and not force_gpu:
        covariance = covariance.cpu()
    eigvals, eigvecs = torch.linalg.eigh(covariance)
    eigvals = eigvals.flip(-1).to(target_device)
    eigvecs = eigvecs.flip(-1).to(target_device)
    if num_eig > 0:
        space = eigvecs[..., :min(num_eig, weights.shape[-2])]
        return space @ (space.mH @ weights)
    keep = eigvals >= threshold * eigvals[..., :1].clamp_min(0)
    keep[..., 0] = True
    return eigvecs @ ((eigvecs.mH @ weights) * keep.unsqueeze(-1))


class RowDynamicMVBeamformerIQ(mv.RowDynamicMVBeamformerIQ):
    normalize_covariance = True

    def get_options(self):
        return args if args is not None else parser.parse_args([])

    def get_device(self):
        return device

    def solve_weights(self, covariance, steering):
        weights = super().solve_weights(covariance, steering)
        return project_signal_subspace(
            covariance, weights, self.options.num_eig,
            self.options.eig_threshold, self.options.force_gpu_eigh,
        )


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
            ("MV axis", args.mv_axis),
            ("F-Number", args.f_number),
            ("Aperture", "Dynamic" if args.dynamic_aperture else "Fixed Full"),
            ("FBSS", "Enabled" if args.fbss else "Disabled"),
            ("DL factor", args.mv_dl),
            ("Subarray", args.subarray_ratio),
            ("Temporal win", args.temporal_win),
            ("Num eig", args.num_eig),
            ("Eig threshold", args.eig_threshold),
            ("Window", args.window.upper()),
            ("Interp", args.interp),
            ("TGC", "Enabled" if args.tgc else "Disabled"),
            *([("TGC Alpha", f"{args.tgc_alpha} dB/MHz/cm")] if args.tgc else []),
            ("DR", f"{args.dr} dB"),
        ],
    )

    # 初始化 ESBMV + FBSS 波束合成器
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

    print(f"Processing ESBMV (DL={args.mv_dl})...")
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
        f"ne{args.num_eig}" if args.num_eig > 0 else f"thr{args.eig_threshold}",
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
        METHOD_NAME,
        title_params,
        dt,
    )

    print(f"  GPU Time: {dt:.2f}s | Saved -> {METHOD_NAME}")
    print(f"\nDone | Output: {method_dir}")


if __name__ == "__main__":
    main()

"""Shared reader for the project's current packed beamforming H5 schema."""

import h5py
import numpy as np

COMPARISON_VALUE_2 = 2
COMPARISON_VALUE_4 = 4


def load_from_h5(h5_path, sample_idx=0):
    """Load one sample and remove pack-time temporal padding.

    This intentionally requires the current schema, including
    ``valid_time_samples``; legacy H5 files are not supported.
    """
    with h5py.File(h5_path, "r") as hf:
        i_all = hf["all_multi_I"]
        q_all = hf["all_multi_Q"]
        if i_all.shape != q_all.shape or i_all.ndim != COMPARISON_VALUE_4:
            raise ValueError("all_multi_I/all_multi_Q 必须是形状一致的 [N,A,T,C] 数组")
        if min(i_all.shape[0], i_all.shape[1], i_all.shape[3]) < 1 or i_all.shape[2] < 2:
            raise ValueError("IQ 的样本、角度、通道维必须非空，时间维至少为 2")
        if not 0 <= sample_idx < i_all.shape[0]:
            raise IndexError(
                f"h5_sample_idx={sample_idx} 超出 [0,{i_all.shape[0] - 1}]",
            )

        i_data = i_all[sample_idx].astype(np.float32)
        q_data = q_all[sample_idx].astype(np.float32)
        angles = hf["angles"][:].astype(np.float32)
        t0_vec = hf["time_start_vector"][sample_idx].astype(np.float32)
        if angles.ndim != 1 or len(angles) != i_data.shape[0]:
            raise ValueError("angles 必须与 IQ 的角度维度一致")
        if t0_vec.ndim != 1 or len(t0_vec) != i_data.shape[0]:
            raise ValueError("time_start_vector 必须与 IQ 的角度维度一致")

        valid_raw = hf["valid_time_samples"][sample_idx]
        valid_time = int(valid_raw)
        if float(valid_raw) != valid_time or not 2 <= valid_time <= i_data.shape[1]:
            raise ValueError(
                f"valid_time_samples[{sample_idx}]={valid_raw} 不是 [2,{i_data.shape[1]}] 内的整数",
            )
        i_data = i_data[:, :valid_time, :]
        q_data = q_data[:, :valid_time, :]
        if not np.all(np.isfinite(i_data)) or not np.all(np.isfinite(q_data)):
            raise ValueError("所选样本的 IQ 数据包含 NaN/Inf")

        c = float(hf["c"][()])
        fc = float(hf["fc"][()])
        fs = float(hf["fs"][()])
        pitch = float(hf["pitch"][()])
        if not all(np.isfinite(value) and value > 0 for value in (c, fc, fs, pitch)):
            raise ValueError("c/fc/fs/pitch 必须是有限正数")

        n_elem = int(hf["num_channels"][()])
        if n_elem != i_data.shape[2]:
            raise ValueError(
                f"num_channels={n_elem} 与 IQ 通道数={i_data.shape[2]} 不一致",
            )

        z_grid = hf["z_grid"][:].astype(np.float32)
        x_grid = hf["x_grid"][:].astype(np.float32)
        for name, grid in (("z_grid", z_grid), ("x_grid", x_grid)):
            if grid.ndim != 1 or grid.size < COMPARISON_VALUE_2:
                raise ValueError(f"{name} 必须是至少含 2 点的一维数组")
            if not np.all(np.isfinite(grid)) or not np.all(np.diff(grid) > 0):
                raise ValueError(f"{name} 必须全部有限且严格递增")
        if not np.all(np.isfinite(angles)) or not np.all(np.isfinite(t0_vec)):
            raise ValueError("angles/time_start_vector 包含 NaN/Inf")

        gt_data = None
        has_gt = "all_envdb_norm" in hf
        if has_gt:
            if hf["all_envdb_norm"].shape[0] <= sample_idx:
                raise ValueError("GT 样本数少于 IQ 样本数")
            gt_data = hf["all_envdb_norm"][sample_idx].astype(np.float32)
            expected_gt_shape = (1, z_grid.size, x_grid.size)
            if gt_data.shape not in (expected_gt_shape, expected_gt_shape[1:]):
                raise ValueError(
                    f"GT 形状 {gt_data.shape} 与网格不匹配,期望 {expected_gt_shape}",
                )
            if not np.all(np.isfinite(gt_data)):
                raise ValueError("GT 包含 NaN/Inf")

        return (
            c,
            fc,
            fs,
            pitch,
            n_elem,
            angles,
            t0_vec,
            z_grid,
            x_grid,
            i_data,
            q_data,
            gt_data,
            has_gt,
        )

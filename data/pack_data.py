"""Provide Python utilities for pack_data."""

import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches

COMPARISON_VALUE_2 = 2

BASE = "US/US_DATASET0000"
DYNAMIC_RANGE = 60.0
TGC_ALPHA = 0.5
F_NUMBER = 1.5
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "data" / "PICMUS"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SIMULATION_SCENES = [
    {
        "name": "simulation_contrast_speckle",
        "mode": "contrast_speckle",
        "source": "simulation",
        "iq": "database/simulation/contrast_speckle/contrast_speckle_simu_dataset_iq.hdf5",
        "scan": "database/simulation/contrast_speckle/contrast_speckle_simu_scan.hdf5",
        "phantom": "database/simulation/contrast_speckle/contrast_speckle_simu_phantom.hdf5",
        "gt": "reconstructed_image/simulation/contrast_speckle/contrast_speckle_simu_img_from_iq.hdf5",
    },
    {
        "name": "simulation_resolution_distorsion",
        "mode": "resolution_distorsion",
        "source": "simulation",
        "iq": "database/simulation/resolution_distorsion/resolution_distorsion_simu_dataset_iq.hdf5",
        "scan": "database/simulation/resolution_distorsion/resolution_distorsion_simu_scan.hdf5",
        "phantom": "database/simulation/resolution_distorsion/resolution_distorsion_simu_phantom.hdf5",
        "gt": "reconstructed_image/simulation/resolution_distorsion/resolution_distorsion_simu_img_from_iq.hdf5",
    },
]

EXPERIMENT_SCENES = [
    {
        "name": "experiments_contrast_speckle",
        "mode": "contrast_speckle",
        "source": "experiments",
        "iq": "database/experiments/contrast_speckle/contrast_speckle_expe_dataset_iq.hdf5",
        "scan": "database/experiments/contrast_speckle/contrast_speckle_expe_scan.hdf5",
        "phantom": "database/experiments/contrast_speckle/contrast_speckle_expe_phantom.hdf5",
        "gt": "reconstructed_image/experiments/contrast_speckle/contrast_speckle_expe_img_from_iq.hdf5",
    },
    {
        "name": "experiments_resolution_distorsion",
        "mode": "resolution_distorsion",
        "source": "experiments",
        "iq": "database/experiments/resolution_distorsion/resolution_distorsion_expe_dataset_iq.hdf5",
        "scan": "database/experiments/resolution_distorsion/resolution_distorsion_expe_scan.hdf5",
        "phantom": "database/experiments/resolution_distorsion/resolution_distorsion_expe_phantom.hdf5",
        "gt": "reconstructed_image/experiments/resolution_distorsion/resolution_distorsion_expe_img_from_iq.hdf5",
    },
]

IN_VIVO_SCENES = [
    {
        "name": "carotid_cross",
        "mode": "in_vivo",
        "source": "in_vivo",
        "iq": "database/in_vivo/carotid_cross/carotid_cross_expe_dataset_iq.hdf5",
        "scan": "database/in_vivo/carotid_cross/carotid_cross_expe_scan.hdf5",
        "phantom": "",
        "gt": "generated:multi_angle_das",
    },
    {
        "name": "carotid_long",
        "mode": "in_vivo",
        "source": "in_vivo",
        "iq": "database/in_vivo/carotid_long/carotid_long_expe_dataset_iq.hdf5",
        "scan": "database/in_vivo/carotid_long/carotid_long_expe_scan.hdf5",
        "phantom": "",
        "gt": "generated:multi_angle_das",
    },
]


def complex_rms_normalization(i_data, q_data):
    """Execute complex rms normalization."""
    i_data = np.asarray(i_data, dtype=np.float32)
    q_data = np.asarray(q_data, dtype=np.float32)
    if i_data.shape != q_data.shape or i_data.ndim != 3 or min(i_data.shape) < 1:
        raise ValueError(
            f"输入 IQ 必须是形状一致的非空 [angle,channel,time]，实际 {i_data.shape}/{q_data.shape}"
        )
    if not np.isfinite(i_data).all() or not np.isfinite(q_data).all():
        raise ValueError("输入 IQ 包含 NaN/Inf")
    signal_power = float(np.mean(i_data**2 + q_data**2))
    if not np.isfinite(signal_power) or signal_power <= 0:
        raise ValueError("输入 IQ 没有有效信号")
    rms = np.sqrt(signal_power + 1e-12)
    return i_data / rms, q_data / rms, rms


def read_gt(gt_path):
    """Read gt."""
    with h5py.File(gt_path, "r") as f:
        real = f[f"{BASE}/data/real"][:][-1].T
        imag = f[f"{BASE}/data/imag"][:][-1].T
    if real.shape != imag.shape or real.ndim != 2 or min(real.shape) < 1:
        raise ValueError(
            f"GT real/imag 必须是形状一致的非空二维数组: {real.shape}/{imag.shape}"
        )
    if not np.isfinite(real).all() or not np.isfinite(imag).all():
        raise ValueError("GT real/imag 包含 NaN/Inf")
    env_sq = real**2 + imag**2
    safe_max = float(np.sqrt(np.max(env_sq)))
    if not np.isfinite(safe_max) or safe_max <= 0:
        raise ValueError("GT 包络没有有效正峰值")
    env_sq /= safe_max**2 + 1e-24
    db = 10.0 * np.log10(env_sq + 1e-24)
    norm = (np.clip(db, -DYNAMIC_RANGE, 0.0) + DYNAMIC_RANGE) / DYNAMIC_RANGE
    return norm[np.newaxis, np.newaxis, ...].astype(np.float32), safe_max


def das_reference_from_iq(
    i_data,
    q_data,
    fs,
    c,
    fc,
    pitch,
    t0,
    angles,
    x_grid,
    z_grid,
    interp="cubic",
    row_block=24,
):
    """Build an in-vivo multi-angle DAS reference with GPU tensor operations using row blocks."""
    i_data = np.asarray(i_data, dtype=np.float32)
    q_data = np.asarray(q_data, dtype=np.float32)
    angles = np.asarray(angles, dtype=np.float32).reshape(-1)
    x_grid = np.asarray(x_grid, dtype=np.float32).reshape(-1)
    z_grid = np.asarray(z_grid, dtype=np.float32).reshape(-1)
    if i_data.shape != q_data.shape or i_data.ndim != 3:
        raise ValueError("DAS 输入 IQ 必须是形状一致的 [angle,time,channel]")
    n_angles, n_times, n_channels = i_data.shape
    if n_angles < 1 or n_times < 2 or n_channels < 1:
        raise ValueError("DAS 输入的角度/通道不能为空，时间维至少为 2")
    if not np.isfinite(i_data).all() or not np.isfinite(q_data).all():
        raise ValueError("DAS 输入 IQ 包含 NaN/Inf")
    if not all(
        np.isfinite(value) and value > 0 for value in (fs, c, fc, pitch, F_NUMBER)
    ):
        raise ValueError("DAS 的 fs/c/fc/pitch/F_NUMBER 必须是有限正数")
    if angles.size != n_angles or not np.isfinite(angles).all():
        raise ValueError("DAS angles 必须匹配 IQ 角度维且全部有限")
    for name, grid in (("x_grid", x_grid), ("z_grid", z_grid)):
        if (
            grid.size < 2
            or not np.isfinite(grid).all()
            or not np.all(np.diff(grid) > 0)
        ):
            raise ValueError(f"DAS {name} 必须是至少含两个点的有限严格递增数组")
    if interp not in {"nearest", "linear", "cubic"}:
        raise ValueError(f"不支持的 DAS 插值方式: {interp}")
    t0 = np.asarray(t0, dtype=np.float32).reshape(-1)
    if t0.size == 1:
        t0 = np.repeat(t0, n_angles)
    if t0.size != n_angles or not np.isfinite(t0).all():
        raise ValueError("DAS initial_time 必须是标量或与角度数一致的有限数组")

    with torch.no_grad():
        z_t = torch.from_numpy(z_grid.astype(np.float32)).to(DEVICE)
        x_t = torch.from_numpy(x_grid.astype(np.float32)).to(DEVICE)
        x_mesh, z_mesh = torch.meshgrid(x_t, z_t, indexing="xy")

        sc = fs / c
        elements = torch.linspace(
            -(n_channels - 1) / 2 * pitch,
            (n_channels - 1) / 2 * pitch,
            n_channels,
            device=DEVICE,
        )
        transmit_z = z_mesh * sc
        transmit_x = x_mesh * sc
        ch = torch.arange(n_channels, device=DEVICE, dtype=torch.long).view(1, 1, -1)

        i_tensor = torch.from_numpy(i_data.astype(np.float32)).to(DEVICE)
        q_tensor = torch.from_numpy(q_data.astype(np.float32)).to(DEVICE)
        cos_a = torch.from_numpy(np.cos(angles).astype(np.float32)).to(DEVICE)
        sin_a = torch.from_numpy(np.sin(angles).astype(np.float32)).to(DEVICE)
        t_starts_t = torch.from_numpy(t0.astype(np.float32)).to(DEVICE) * fs

        out_i = torch.zeros(
            (len(z_grid), len(x_grid)),
            dtype=torch.float32,
            device=DEVICE,
        )
        out_q = torch.zeros_like(out_i)
        max_sample = float(n_times - 2)
        row_block = len(z_grid) if row_block <= 0 else max(1, row_block)

        for z0 in range(0, len(z_grid), row_block):
            z1 = min(z0 + row_block, len(z_grid))
            x_b = x_mesh[z0:z1]
            z_b = z_mesh[z0:z1]
            tx_z = transmit_z[z0:z1]
            tx_x = transmit_x[z0:z1]

            receive_samples = (
                torch.sqrt((x_b[..., None] - elements) ** 2 + z_b[..., None] ** 2) * sc
            )
            dx = x_b[..., None] - elements
            half_aperture = z_b[..., None] / (2.0 * F_NUMBER)
            aperture = (dx.abs() <= half_aperture).float()
            weights = aperture / (aperture.sum(-1, keepdim=True) + 1e-9)

            phi_rx = 2.0 * np.pi * fc * (receive_samples / fs)
            cos_rx = torch.cos(phi_rx)
            sin_rx = torch.sin(phi_rx)
            block_i = torch.zeros(
                (z1 - z0, len(x_grid)),
                dtype=torch.float32,
                device=DEVICE,
            )
            block_q = torch.zeros_like(block_i)

            for i in range(n_angles):
                tx_samples = tx_z * cos_a[i] + tx_x * sin_a[i]
                sample = tx_samples[..., None] + receive_samples - t_starts_t[i]
                valid = (sample >= 0) & (sample < n_times - 1)
                sample.clamp_(0.0, max_sample)
                i_angle = i_tensor[i]
                q_angle = q_tensor[i]

                if interp == "nearest":
                    idx = sample.round().long().clamp(0, n_times - 1)
                    i_center = i_angle[idx, ch]
                    q_center = q_angle[idx, ch]
                elif interp == "cubic":
                    idx0 = sample.floor().long()
                    frac = sample - idx0.float()
                    frac2 = frac * frac
                    frac3 = frac2 * frac
                    c_m1 = -0.5 * frac3 + frac2 - 0.5 * frac
                    c_0 = 1.5 * frac3 - 2.5 * frac2 + 1.0
                    c_1 = -1.5 * frac3 + 2.0 * frac2 + 0.5 * frac
                    c_2 = 0.5 * frac3 - 0.5 * frac2
                    idx_m1 = torch.clamp(idx0 - 1, 0, n_times - 1)
                    idx_0 = torch.clamp(idx0, 0, n_times - 1)
                    idx_1 = torch.clamp(idx0 + 1, 0, n_times - 1)
                    idx_2 = torch.clamp(idx0 + 2, 0, n_times - 1)
                    i_center = (
                        i_angle[idx_m1, ch] * c_m1
                        + i_angle[idx_0, ch] * c_0
                        + i_angle[idx_1, ch] * c_1
                        + i_angle[idx_2, ch] * c_2
                    )
                    q_center = (
                        q_angle[idx_m1, ch] * c_m1
                        + q_angle[idx_0, ch] * c_0
                        + q_angle[idx_1, ch] * c_1
                        + q_angle[idx_2, ch] * c_2
                    )
                else:
                    idx0 = sample.floor().long()
                    frac = sample - idx0.float()
                    i_center = (
                        i_angle[idx0, ch] * (1.0 - frac) + i_angle[idx0 + 1, ch] * frac
                    )
                    q_center = (
                        q_angle[idx0, ch] * (1.0 - frac) + q_angle[idx0 + 1, ch] * frac
                    )

                valid_w = valid.float() * weights
                valid_w = valid_w / (valid_w.sum(dim=-1, keepdim=True) + 1e-9)
                i_rx = i_center * cos_rx - q_center * sin_rx
                q_rx = i_center * sin_rx + q_center * cos_rx
                i_sum = (i_rx * valid_w).sum(dim=-1)
                q_sum = (q_rx * valid_w).sum(dim=-1)

                phi_tx = 2.0 * np.pi * fc * (tx_samples / fs)
                cos_tx = torch.cos(phi_tx)
                sin_tx = torch.sin(phi_tx)
                block_i.add_(i_sum * cos_tx - q_sum * sin_tx)
                block_q.add_(i_sum * sin_tx + q_sum * cos_tx)

            out_i[z0:z1] = block_i / n_angles
            out_q[z0:z1] = block_q / n_angles

        tgc = 10.0 ** (TGC_ALPHA * (fc / 1e6) * (z_t * 100.0) * 2.0 / 20.0)
        env = torch.sqrt(out_i * out_i + out_q * out_q) * tgc[:, None]
        safe_max = env.max()
        if not torch.isfinite(safe_max) or safe_max <= 0:
            raise ValueError("DAS 参考包络没有有效正峰值")
        env = env / (safe_max + 1e-12)
        db = 20.0 * torch.log10(torch.clamp(env, min=1e-12))
        norm = (torch.clamp(db, -DYNAMIC_RANGE, 0.0) + DYNAMIC_RANGE) / DYNAMIC_RANGE
        return norm.cpu().numpy()[np.newaxis, np.newaxis, ...].astype(
            np.float32,
        ), float(safe_max.cpu())


def process_scene(scene, source_root, row_block=24):
    """Execute process scene."""
    required_scene_keys = {"name", "mode", "source", "iq", "scan", "phantom", "gt"}
    present_scene_keys = set(scene) if isinstance(scene, dict) else set()
    if not required_scene_keys <= present_scene_keys:
        raise ValueError(
            f"场景配置缺少字段: {sorted(required_scene_keys - present_scene_keys)}"
        )
    for key in ("name", "mode", "source", "iq", "scan", "gt"):
        if not isinstance(scene[key], str) or not scene[key].strip():
            raise ValueError(f"场景字段 {key} 必须是非空字符串")
    print(f"Processing {scene['name']}")
    source_root = Path(source_root)
    iq_path = source_root / scene["iq"]
    scan_path = source_root / scene["scan"]
    gt_path = (
        source_root / scene["gt"]
        if scene["gt"] and not scene["gt"].startswith("generated:")
        else None
    )

    with h5py.File(iq_path, "r") as f:
        i_data = f[f"{BASE}/data/real"][:]
        q_data = f[f"{BASE}/data/imag"][:]
        i_norm, q_norm, scale_ref = complex_rms_normalization(i_data, q_data)
        i_raw_trans = np.transpose(i_data.astype(np.float32), (0, 2, 1))
        q_raw_trans = np.transpose(q_data.astype(np.float32), (0, 2, 1))
        i_trans = np.transpose(i_norm, (0, 2, 1)).astype(np.float32)
        q_trans = np.transpose(q_norm, (0, 2, 1)).astype(np.float32)

        fs = float(np.array(f[f"{BASE}/sampling_frequency"]).flatten()[0])
        c = float(np.array(f[f"{BASE}/sound_speed"]).flatten()[0])
        if f"{BASE}/fc" in f:
            fc = float(np.array(f[f"{BASE}/fc"]).flatten()[0])
        elif f"{BASE}/modulation_frequency" in f:
            fc = float(np.array(f[f"{BASE}/modulation_frequency"]).flatten()[0])
        else:
            raise KeyError("IQ 源文件缺少 fc/modulation_frequency，不能推断载波频率")
        if f"{BASE}/pitch" in f:
            pitch = float(np.array(f[f"{BASE}/pitch"]).flatten()[0])
        elif f"{BASE}/probe_geometry" in f:
            geom = np.asarray(f[f"{BASE}/probe_geometry"], dtype=np.float64)
            if geom.ndim != 2 or geom.shape[0] < 1 or geom.shape[1] != i_trans.shape[2]:
                raise ValueError(f"probe_geometry 形状与通道数不一致: {geom.shape}")
            spacing = np.diff(geom[0])
            if (
                spacing.size == 0
                or not np.isfinite(spacing).all()
                or np.any(spacing == 0)
                or not np.allclose(
                    np.abs(spacing), np.median(np.abs(spacing)), rtol=1e-4, atol=1e-9
                )
            ):
                raise ValueError("probe_geometry 不是均匀线阵，无法推导 pitch")
            pitch = float(np.median(np.abs(spacing)))
        else:
            raise KeyError("IQ 源文件缺少 pitch/probe_geometry，不能推导阵元间距")
        t0_raw = np.asarray(f[f"{BASE}/initial_time"], dtype=np.float32).reshape(-1)
        if t0_raw.size == 1:
            t0 = np.repeat(t0_raw, i_trans.shape[0])
        elif t0_raw.size == i_trans.shape[0]:
            t0 = t0_raw
        else:
            raise ValueError(
                f"initial_time 长度 {t0_raw.size} 必须为 1 或角度数 {i_trans.shape[0]}",
            )
        angles = np.array(f[f"{BASE}/angles"]).flatten().astype(np.float32)

    if not all(np.isfinite(value) and value > 0 for value in (fs, c, fc, pitch)):
        raise ValueError("fs/c/fc/pitch 必须是有限正数")
    if angles.size != i_trans.shape[0] or not np.isfinite(angles).all():
        raise ValueError("angles 必须匹配 IQ 角度维且全部有限")
    if not np.isfinite(t0).all():
        raise ValueError("initial_time 包含 NaN/Inf")

    with h5py.File(scan_path, "r") as f:
        x_grid = np.array(f[f"{BASE}/x_axis"]).flatten().astype(np.float32)
        z_grid = np.array(f[f"{BASE}/z_axis"]).flatten().astype(np.float32)
    for name, grid in (("x_grid", x_grid), ("z_grid", z_grid)):
        if (
            grid.size < 2
            or not np.isfinite(grid).all()
            or not np.all(np.diff(grid) > 0)
        ):
            raise ValueError(f"{name} 必须是至少含两个点的有限严格递增数组")

    if scene["gt"] == "generated:multi_angle_das":
        gt, safe_max = das_reference_from_iq(
            i_raw_trans,
            q_raw_trans,
            fs,
            c,
            fc,
            pitch,
            t0,
            angles,
            x_grid,
            z_grid,
            row_block=row_block,
        )
    else:
        gt, safe_max = read_gt(gt_path) if gt_path else (None, np.nan)

    if gt is None or gt.shape != (1, 1, z_grid.size, x_grid.size):
        raise ValueError(
            f"GT 形状必须为 [1,1,{z_grid.size},{x_grid.size}]，实际 {None if gt is None else gt.shape}",
        )
    if (
        not np.isfinite(gt).all()
        or float(np.min(gt)) < -1e-6
        or float(np.max(gt)) > 1.0 + 1e-6
    ):
        raise ValueError("GT 必须是位于 [0,1] 的有限数组")

    norm_ref = safe_max / (float(scale_ref) + 1e-12)
    if not np.isfinite(norm_ref) or norm_ref <= 0:
        raise ValueError("norm_ref 必须是有限正数")

    return {
        "I": i_trans[np.newaxis, ...],
        "Q": q_trans[np.newaxis, ...],
        "gt": gt,
        "t0": t0[np.newaxis, ...],
        "fs": fs,
        "c": c,
        "fc": fc,
        "pitch": pitch,
        "num_channels": i_trans.shape[2],
        "z_grid": z_grid,
        "x_grid": x_grid,
        "angles": angles,
        "scale_ref": scale_ref,
        "norm_ref": norm_ref,
        "meta": scene,
    }


def pad_and_concat(items, key):
    """Execute pad and concat."""
    arrays = [item[key] for item in items]
    max_a = max(arr.shape[1] for arr in arrays)
    max_t = max(arr.shape[2] for arr in arrays)
    max_c = max(arr.shape[3] for arr in arrays)
    padded = []
    for arr in arrays:
        pad_cfg = (
            (0, 0),
            (0, max_a - arr.shape[1]),
            (0, max_t - arr.shape[2]),
            (0, max_c - arr.shape[3]),
        )
        padded.append(np.pad(arr, pad_cfg, mode="constant"))
    return np.concatenate(padded, axis=0)


def compact_sequence_config(values):
    """Encode a long uniform numeric sequence without repeating every value."""
    array = np.asarray(values).reshape(-1)
    if array.size <= 8:
        return array.tolist()
    differences = np.diff(array.astype(np.float64))
    if differences.size and np.allclose(
        differences,
        differences[0],
        rtol=1e-5,
        atol=1e-9,
    ):
        integer_sequence = np.issubdtype(array.dtype, np.integer)
        return {
            "encoding": "arithmetic_sequence",
            "start": int(array[0]) if integer_sequence else float(array[0]),
            "step": int(round(differences[0]))
            if integer_sequence
            else float(differences[0]),
            "count": int(array.size),
            "dtype": str(array.dtype),
        }
    return array.tolist()


def build_embedded_config(items, output_path, row_block):
    """Build embedded config."""
    has_official_gt = any(
        item["meta"]["gt"] and not item["meta"]["gt"].startswith("generated:")
        for item in items
    )
    has_generated_gt = any(
        item["meta"]["gt"] == "generated:multi_angle_das" for item in items
    )
    ground_truth = {
        "dataset": "/all_envdb_norm",
        "reference_sampling_frequency_hz": float(items[0]["fs"]),
        "output_domain": "envelope_db_mapped_to_unit_interval",
        "dynamic_range_db": DYNAMIC_RANGE,
        "normalization": {
            "reference": "per_sample_envelope_peak",
            "db_formula": "20*log10(envelope/reference)",
            "clipped_db_range": [-DYNAMIC_RANGE, 0.0],
            "mapped_range": [0.0, 1.0],
            "norm_reference_dataset": "/all_norm_ref",
            "norm_reference_formula": "ground_truth_peak / input_scale_ref",
        },
    }
    if has_official_gt:
        ground_truth["official_picmus_reference"] = {
            "source_domain": "complex_baseband_iq",
            "frame_selection": "last",
            "spatial_transpose": True,
            "envelope": "sqrt(real^2 + imag^2)",
            "peak_epsilon": 1.0e-12,
            "power_log_epsilon": 1.0e-24,
        }
    if has_generated_gt:
        ground_truth["generated_multi_angle_das"] = {
            "source_domain": "unnormalized_PICMUS_baseband_iq",
            "angle_selection": "all_available_input_angles",
            "element_positions": "centered_uniform_linear_array_from_pitch",
            "transmit_model": "plane_wave",
            "receive_model": "spherical_distance",
            "baseband_phase_rotation": "receive_then_transmit",
            "aperture_mode": "geometry",
            "dynamic_aperture": True,
            "f_number": F_NUMBER,
            "window": "rect",
            "aperture_weight_normalization": True,
            "interpolation": "cubic_convolution_a_minus_0.5",
            "interpolation_boundary": "clamped_indices",
            "out_of_range_samples": "zero",
            "valid_weight_renormalization": True,
            "angle_compounding": "complex_mean",
            "peak_division_epsilon": 1.0e-12,
            "envelope_log_floor": 1.0e-12,
            "tgc": {
                "enabled": True,
                "alpha_db_cm_mhz": TGC_ALPHA,
                "two_way": True,
                "formula": "10**(alpha*(fc_MHz)*(depth_cm)*2/20)",
            },
            "implementation": {
                "depth_row_block": int(row_block),
                "nonpositive_row_block_means_full_image": True,
            },
        }

    return {
        "schema_version": 1,
        "dataset": {
            "id": output_path.stem,
            "type": "PICMUS",
            "purpose": "algorithm_evaluation",
        },
        "provenance": {
            "generator": "data/pack_data.py",
            "source_kind": "official_PICMUS_data",
        },
        "h5_schema": {
            "iq_layout": ["sample", "angle", "time", "channel"],
            "ground_truth_layout": ["sample", "image_channel", "z", "x"],
            "padding": {
                "value": 0.0,
                "time_valid_length_dataset": "/valid_time_samples",
            },
            "sample_metadata": "config_yaml.source_samples",
            "units": {
                "fs": "Hz",
                "c": "m/s",
                "fc": "Hz",
                "pitch": "m",
                "x_grid": "m",
                "z_grid": "m",
                "angles": "rad",
                "time_start_vector": "s",
            },
            "coordinate_system": {
                "x": "lateral_positive_right",
                "z": "depth_positive_away_from_probe",
                "array_origin": "center_of_linear_array",
            },
        },
        "generation": {
            "input_iq": {
                "source": "PICMUS_baseband_iq",
                "source_layout": ["angle", "channel", "time"],
                "input_angle_selection": "all_available",
                "all_steering_angles_rad": compact_sequence_config(items[0]["angles"]),
                "selected_angle_indices": compact_sequence_config(
                    np.arange(len(items[0]["angles"]), dtype=np.int64),
                ),
                "normalization": {
                    "mode": "complex_rms",
                    "formula": "sqrt(mean(I^2 + Q^2) + epsilon)",
                    "epsilon": 1.0e-12,
                    "scale_reference_dataset": "/all_scale_ref",
                },
                # PICMUS 的输入本身就是基带 IQ；没有在打包阶段执行抽取。
                # 采样率相关元数据统一放在 config_yaml，避免遗留 gt_fs 等
                # 根字段造成“GT 采样率/输入采样率”语义混淆。
                "decimation_factor": 1,
                "source_sampling_frequency_hz": float(items[0]["fs"]),
                "packed_sampling_frequency_hz": float(items[0]["fs"]),
            },
            "ground_truth": ground_truth,
        },
        "source_samples": [
            {
                "sample_index": index,
                "id": item["meta"]["name"],
                "phantom_mode": item["meta"]["mode"],
                "phantom_source": item["meta"]["source"],
                "iq_path": item["meta"]["iq"],
                "scan_path": item["meta"]["scan"],
                "phantom_path": item["meta"]["phantom"],
                "gt_path": item["meta"]["gt"],
                "path_roles": {
                    "iq_path": "read_input",
                    "scan_path": "read_input",
                    "phantom_path": "reference_only_not_read",
                    "gt_path": (
                        "generated_not_read"
                        if str(item["meta"]["gt"] or "").startswith("generated:")
                        else "read_input"
                    ),
                },
            }
            for index, item in enumerate(items)
        ],
        "path_convention": {
            "type": "relative",
            "base": "PICMUS_ROOT",
        },
    }


def save_gt_images(items, image_dir, dr=DYNAMIC_RANGE):
    """Save gt images."""
    if not np.isfinite(dr) or dr <= 0:
        raise ValueError("GT 预览动态范围必须是有限正数")
    image_dir.mkdir(parents=True, exist_ok=True)

    def add_scale_bar(ax, extent_mm):
        """Execute add scale bar."""
        bar_length = 5.0
        bar_x = extent_mm[1] - bar_length - 2.0
        bar_y = extent_mm[2] - 2.0
        ax.add_patch(
            patches.Rectangle((bar_x, bar_y), bar_length, 0.5, color="white", zorder=5),
        )
        ax.text(
            bar_x + bar_length / 2,
            bar_y - 1.0,
            "5 mm",
            color="white",
            fontsize=9,
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    for item in items:
        gt = item["gt"]
        if gt is None:
            continue
        gt_sample = np.squeeze(gt).astype(np.float32)
        sample_name = item["meta"]["name"]
        out_path = image_dir / f"{sample_name}.png"

        x_grid = np.asarray(item.get("x_grid", []), dtype=np.float32)
        z_grid = np.asarray(item.get("z_grid", []), dtype=np.float32)
        if x_grid.size < COMPARISON_VALUE_2 or z_grid.size < COMPARISON_VALUE_2:
            plt.imsave(out_path, gt_sample, cmap="gray", vmin=0.0, vmax=1.0)
            continue

        extent_mm = [
            x_grid[0] * 1000.0,
            x_grid[-1] * 1000.0,
            z_grid[-1] * 1000.0,
            z_grid[0] * 1000.0,
        ]
        gt_db = np.clip(gt_sample, 0.0, 1.0) * dr - dr

        fig, ax = plt.subplots(figsize=(6, 7), dpi=150)
        im = ax.imshow(
            gt_db,
            cmap="gray",
            vmin=-dr,
            vmax=0.0,
            extent=extent_mm,
            aspect="equal",
        )
        ax.set_title(f"GT: {sample_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Lateral (mm)")
        ax.set_ylabel("Depth (mm)")
        add_scale_bar(ax, extent_mm)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Normalized Envelope (dB)", rotation=270, labelpad=15)

        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def validate_shared_metadata(items):
    """Validate the shared physical and grid definition."""
    if not items:
        raise ValueError("至少需要一个待打包样本")
    base = items[0]
    required_keys = {
        "I",
        "Q",
        "gt",
        "t0",
        "fs",
        "c",
        "fc",
        "pitch",
        "num_channels",
        "z_grid",
        "x_grid",
        "angles",
        "scale_ref",
        "norm_ref",
        "meta",
    }
    for index, item in enumerate(items):
        missing = required_keys - set(item)
        if missing:
            raise ValueError(f"样本 {index} 缺少字段: {sorted(missing)}")
        i_data = np.asarray(item["I"])
        q_data = np.asarray(item["Q"])
        if (
            i_data.shape != q_data.shape
            or i_data.ndim != 4
            or i_data.shape[0] != 1
            or min(i_data.shape[1:]) < 1
            or i_data.shape[2] < 2
            or not np.isfinite(i_data).all()
            or not np.isfinite(q_data).all()
        ):
            raise ValueError(f"样本 {index} 的 IQ 必须是有限非空 [1,A,T,C] 且 T>=2")
        if (
            np.asarray(item["t0"]).shape != i_data.shape[:2]
            or not np.isfinite(item["t0"]).all()
        ):
            raise ValueError(f"样本 {index} 的 t0 必须是匹配 IQ 的有限 [1,A]")
        angles = np.asarray(item["angles"])
        if angles.shape != (i_data.shape[1],) or not np.isfinite(angles).all():
            raise ValueError(f"样本 {index} 的 angles 数量与 IQ 不一致")
        if (
            int(item["num_channels"]) != item["num_channels"]
            or int(item["num_channels"]) != i_data.shape[3]
        ):
            raise ValueError(f"样本 {index} 的 num_channels 与 IQ 不一致")
        for grid_name in ("z_grid", "x_grid"):
            grid = np.asarray(item[grid_name])
            if (
                grid.ndim != 1
                or grid.size < 2
                or not np.isfinite(grid).all()
                or not np.all(np.diff(grid) > 0)
            ):
                raise ValueError(
                    f"样本 {index} 的 {grid_name} 必须是有限严格递增一维数组"
                )
        gt = np.asarray(item["gt"])
        expected_gt_shape = (1, 1, len(item["z_grid"]), len(item["x_grid"]))
        if gt.shape != expected_gt_shape or not np.isfinite(gt).all():
            raise ValueError(f"样本 {index} 的 GT 形状或数值非法: {gt.shape}")
        if float(np.min(gt)) < -1e-6 or float(np.max(gt)) > 1.0 + 1e-6:
            raise ValueError(f"样本 {index} 的 GT 必须位于 [0,1]")
        if not isinstance(item["meta"], dict):
            raise ValueError(f"样本 {index} 的 meta 必须是字典")
        if not all(
            np.isfinite(value) and value > 0
            for value in (
                item["fs"],
                item["c"],
                item["fc"],
                item["pitch"],
                item["scale_ref"],
                item["norm_ref"],
            )
        ):
            raise ValueError(f"样本 {index} 的物理参数或归一化参考必须是有限正数")
    scalar_keys = ("fs", "c", "fc", "pitch")
    exact_keys = ("num_channels",)
    array_keys = ("z_grid", "x_grid", "angles")
    for index, item in enumerate(items[1:], 1):
        for key in scalar_keys:
            if not np.isclose(item[key], base[key], rtol=1e-6, atol=0.0):
                raise ValueError(
                    f"样本 {index} 的 {key}={item[key]} 与首样本 {base[key]} 不一致",
                )
        for key in exact_keys:
            if item[key] != base[key]:
                raise ValueError(
                    f"样本 {index} 的 {key}={item[key]} 与首样本 {base[key]} 不一致",
                )
        for key in array_keys:
            current = np.asarray(item[key])
            reference = np.asarray(base[key])
            if current.shape != reference.shape or not np.allclose(
                current,
                reference,
                rtol=1e-6,
                atol=1e-9,
            ):
                raise ValueError(
                    f"样本 {index} 的 {key} 与首样本不一致,当前 H5 schema 无法共同打包",
                )


def pack_dataset(
    scenes,
    output_path,
    source_root,
    row_block=24,
    save_gt_images_enabled=True,
):
    """Execute pack dataset."""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"输出文件已存在，请先移除或更换路径: {output_path}")
    items = [process_scene(scene, source_root, row_block=row_block) for scene in scenes]
    validate_shared_metadata(items)
    base = items[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    packed_i = pad_and_concat(items, "I").astype(np.float16, copy=False)
    packed_q = pad_and_concat(items, "Q").astype(np.float16, copy=False)
    packed_gt = np.concatenate([item["gt"] for item in items], axis=0).astype(
        np.float16,
        copy=False,
    )
    if not np.isfinite(packed_i).all() or not np.isfinite(packed_q).all():
        raise OverflowError("IQ 转换为 float16 后产生 NaN/Inf")
    if not np.isfinite(packed_gt).all():
        raise OverflowError("GT 转换为 float16 后产生 NaN/Inf")

    try:
        output_file = h5py.File(output_path, "x")
    except FileExistsError as exc:
        raise FileExistsError(
            f"输出文件已存在，请先移除或更换路径: {output_path}"
        ) from exc

    with output_file as hf:
        hf.create_dataset(
            "all_multi_I",
            data=packed_i,
            compression="gzip",
            compression_opts=4,
        )
        hf.create_dataset(
            "all_multi_Q",
            data=packed_q,
            compression="gzip",
            compression_opts=4,
        )

        hf.create_dataset(
            "all_envdb_norm",
            data=packed_gt,
            compression="gzip",
            compression_opts=4,
        )

        hf.create_dataset(
            "all_scale_ref",
            data=np.array([item["scale_ref"] for item in items], dtype=np.float32),
        )
        hf.create_dataset(
            "all_norm_ref",
            data=np.array([item["norm_ref"] for item in items], dtype=np.float32),
        )
        hf.create_dataset(
            "valid_time_samples",
            data=np.array([item["I"].shape[2] for item in items], dtype=np.int32),
        )

        max_a = max(item["t0"].shape[1] for item in items)
        t0_padded = [
            np.pad(
                item["t0"],
                ((0, 0), (0, max_a - item["t0"].shape[1])),
                mode="constant",
            )
            for item in items
        ]
        hf.create_dataset(
            "time_start_vector",
            data=np.concatenate(t0_padded, axis=0).astype(np.float32, copy=False),
            compression="gzip",
            compression_opts=4,
        )

        scalar_dtypes = {
            "fs": np.float32,
            "c": np.float32,
            "fc": np.float32,
            "pitch": np.float32,
            "num_channels": np.int32,
        }
        for meta_key, dtype in scalar_dtypes.items():
            hf.create_dataset(meta_key, data=dtype(base[meta_key]))
        for meta_key in ("z_grid", "x_grid", "angles"):
            hf.create_dataset(
                meta_key, data=np.asarray(base[meta_key], dtype=np.float32)
            )

        string_dtype = h5py.string_dtype(encoding="utf-8")
        config_yaml = yaml.safe_dump(
            build_embedded_config(items, output_path, row_block),
            sort_keys=False,
            allow_unicode=True,
        )
        hf.create_dataset("config_yaml", data=config_yaml, dtype=string_dtype)

    if save_gt_images_enabled:
        save_gt_images(items, output_path.parent / "gt_preview")

    print(f"Saved {output_path}")


def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser(description="Pack project H5 datasets.")
    parser.add_argument("--source_root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--out_dir", default=str(PROJECT_ROOT / "data"))
    parser.add_argument(
        "--only",
        default="all",
        choices=("all", "simulation", "experiments", "in_vivo"),
        help="all, simulation, experiments, in_vivo",
    )
    parser.add_argument(
        "--row_block",
        type=int,
        default=24,
        help="GPU按深度方向分块行数;<=0 表示整幅一次计算",
    )
    parser.add_argument(
        "--no_save_gt_images",
        action="store_true",
        help="不额外输出GT预览图",
    )
    return parser.parse_args()


def main():
    """Run the command-line workflow."""
    args = parse_args()
    source_root = Path(args.source_root)
    out_dir = Path(args.out_dir)
    choices = (
        ["simulation", "experiments", "in_vivo"] if args.only == "all" else [args.only]
    )
    scene_map = {
        "simulation": (SIMULATION_SCENES, out_dir / "simulation.h5"),
        "experiments": (EXPERIMENT_SCENES, out_dir / "experiments.h5"),
        "in_vivo": (IN_VIVO_SCENES, out_dir / "in_vivo.h5"),
    }
    for choice in choices:
        scenes, output_path = scene_map[choice]
        pack_dataset(
            scenes,
            output_path,
            source_root,
            row_block=args.row_block,
            save_gt_images_enabled=not args.no_save_gt_images,
        )


if __name__ == "__main__":
    main()

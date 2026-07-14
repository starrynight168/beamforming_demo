import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches


BASE = "US/US_DATASET0000"
DYNAMIC_RANGE = 60.0
TGC_ALPHA = 0.5
F_NUMBER = 1.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def project_root():
    return Path(__file__).resolve().parents[1]


def default_source_root():
    return project_root() / "data" / "PICMUS"


def complex_rms_normalization(i_data, q_data):
    i_data = i_data.astype(np.float32)
    q_data = q_data.astype(np.float32)
    rms = np.sqrt(np.mean(i_data ** 2 + q_data ** 2) + 1e-12)
    return i_data / rms, q_data / rms, rms


def read_gt(gt_path):
    with h5py.File(gt_path, "r") as f:
        real = f[f"{BASE}/data/real"][:][-1].T
        imag = f[f"{BASE}/data/imag"][:][-1].T
    env_sq = real ** 2 + imag ** 2
    safe_max = float(np.sqrt(np.max(env_sq)) + 1e-12)
    env_sq /= safe_max ** 2 + 1e-24
    db = 10.0 * np.log10(env_sq + 1e-24)
    norm = (np.clip(db, -DYNAMIC_RANGE, 0.0) + DYNAMIC_RANGE) / DYNAMIC_RANGE
    return norm[np.newaxis, np.newaxis, ...].astype(np.float32), safe_max

def das_reference_from_iq(i_data, q_data, fs, c, fc, pitch, t0, angles, x_grid, z_grid, interp='cubic', row_block=24):
    """Build an in-vivo multi-angle DAS reference with GPU tensor operations using row blocks."""
    n_angles, n_times, n_channels = i_data.shape
    t0 = np.asarray(t0, dtype=np.float32).reshape(-1)
    if t0.size == 1:
        t0 = np.repeat(t0, n_angles)

    with torch.no_grad():
        z_t = torch.from_numpy(z_grid.astype(np.float32)).to(device)
        x_t = torch.from_numpy(x_grid.astype(np.float32)).to(device)
        x_mesh, z_mesh = torch.meshgrid(x_t, z_t, indexing='xy')

        sc = fs / c
        elements = torch.linspace(
            -(n_channels - 1) / 2 * pitch,
            (n_channels - 1) / 2 * pitch,
            n_channels,
            device=device,
        )
        transmit_z = z_mesh * sc
        transmit_x = x_mesh * sc
        ch = torch.arange(n_channels, device=device, dtype=torch.long).view(1, 1, -1)

        I_t = torch.from_numpy(i_data.astype(np.float32)).to(device)
        Q_t = torch.from_numpy(q_data.astype(np.float32)).to(device)
        cos_a = torch.from_numpy(np.cos(angles).astype(np.float32)).to(device)
        sin_a = torch.from_numpy(np.sin(angles).astype(np.float32)).to(device)
        t_starts_t = torch.from_numpy(t0.astype(np.float32)).to(device) * fs

        out_i = torch.zeros((len(z_grid), len(x_grid)), dtype=torch.float32, device=device)
        out_q = torch.zeros_like(out_i)
        max_sample = float(n_times - 2)
        row_block = len(z_grid) if row_block <= 0 else max(1, row_block)

        for z0 in range(0, len(z_grid), row_block):
            z1 = min(z0 + row_block, len(z_grid))
            x_b = x_mesh[z0:z1]
            z_b = z_mesh[z0:z1]
            tx_z = transmit_z[z0:z1]
            tx_x = transmit_x[z0:z1]

            receive_samples = torch.sqrt((x_b[..., None] - elements) ** 2 + z_b[..., None] ** 2) * sc
            dx = x_b[..., None] - elements
            half_aperture = z_b[..., None] / (2.0 * F_NUMBER)
            aperture = (dx.abs() <= half_aperture).float()
            weights = aperture / (aperture.sum(-1, keepdim=True) + 1e-9)

            phi_rx = 2.0 * np.pi * fc * (receive_samples / fs)
            cos_rx = torch.cos(phi_rx)
            sin_rx = torch.sin(phi_rx)
            block_i = torch.zeros((z1 - z0, len(x_grid)), dtype=torch.float32, device=device)
            block_q = torch.zeros_like(block_i)

            for i in range(n_angles):
                tx_samples = tx_z * cos_a[i] + tx_x * sin_a[i]
                sample = tx_samples[..., None] + receive_samples - t_starts_t[i]
                valid = (sample >= 0) & (sample < n_times - 1)
                sample.clamp_(0.0, max_sample)
                I_angle = I_t[i]
                Q_angle = Q_t[i]

                if interp == 'nearest':
                    idx = sample.round().long().clamp(0, n_times - 1)
                    I_center = I_angle[idx, ch]
                    Q_center = Q_angle[idx, ch]
                elif interp == 'cubic':
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
                    I_center = (
                        I_angle[idx_m1, ch] * c_m1
                        + I_angle[idx_0, ch] * c_0
                        + I_angle[idx_1, ch] * c_1
                        + I_angle[idx_2, ch] * c_2
                    )
                    Q_center = (
                        Q_angle[idx_m1, ch] * c_m1
                        + Q_angle[idx_0, ch] * c_0
                        + Q_angle[idx_1, ch] * c_1
                        + Q_angle[idx_2, ch] * c_2
                    )
                else: # linear
                    idx0 = sample.floor().long()
                    frac = sample - idx0.float()
                    I_center = I_angle[idx0, ch] * (1.0 - frac) + I_angle[idx0 + 1, ch] * frac
                    Q_center = Q_angle[idx0, ch] * (1.0 - frac) + Q_angle[idx0 + 1, ch] * frac

                valid_w = valid.float() * weights
                I_rx = I_center * cos_rx - Q_center * sin_rx
                Q_rx = I_center * sin_rx + Q_center * cos_rx
                I_sum = (I_rx * valid_w).sum(dim=-1)
                Q_sum = (Q_rx * valid_w).sum(dim=-1)

                phi_tx = 2.0 * np.pi * fc * (tx_samples / fs)
                cos_tx = torch.cos(phi_tx)
                sin_tx = torch.sin(phi_tx)
                block_i.add_(I_sum * cos_tx - Q_sum * sin_tx)
                block_q.add_(I_sum * sin_tx + Q_sum * cos_tx)

            out_i[z0:z1] = block_i / n_angles
            out_q[z0:z1] = block_q / n_angles

        tgc = 10.0 ** (TGC_ALPHA * (fc / 1e6) * (z_t * 100.0) * 2.0 / 20.0)
        env = torch.sqrt(out_i * out_i + out_q * out_q) * tgc[:, None]
        safe_max = env.max()
        env = env / (safe_max + 1e-12)
        db = 20.0 * torch.log10(torch.clamp(env, min=1e-12))
        norm = (torch.clamp(db, -DYNAMIC_RANGE, 0.0) + DYNAMIC_RANGE) / DYNAMIC_RANGE
        return norm.cpu().numpy()[np.newaxis, np.newaxis, ...].astype(np.float32), float(safe_max.cpu())
def process_scene(scene, source_root, row_block=24):
    print(f"Processing {scene['name']}")
    iq_path = source_root / scene["iq"]
    scan_path = source_root / scene["scan"]
    gt_path = source_root / scene["gt"] if scene["gt"] and not scene["gt"].startswith("generated:") else None

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
            fc = fs
        if f"{BASE}/pitch" in f:
            pitch = float(np.array(f[f"{BASE}/pitch"]).flatten()[0])
        elif f"{BASE}/probe_geometry" in f:
            geom = np.array(f[f"{BASE}/probe_geometry"])
            pitch = float(abs(np.median(np.diff(geom[0]))))
        else:
            pitch = 0.300e-3
        t0 = np.repeat(np.array(f[f"{BASE}/initial_time"]).flatten(), i_trans.shape[0]).astype(np.float32)
        angles = np.array(f[f"{BASE}/angles"]).flatten().astype(np.float32)

    with h5py.File(scan_path, "r") as f:
        x_grid = np.array(f[f"{BASE}/x_axis"]).flatten().astype(np.float32)
        z_grid = np.array(f[f"{BASE}/z_axis"]).flatten().astype(np.float32)

    if scene["gt"] == "generated:multi_angle_das":
        gt, safe_max = das_reference_from_iq(i_raw_trans, q_raw_trans, fs, c, fc, pitch, t0, angles, x_grid, z_grid, row_block=row_block)
    else:
        gt, safe_max = read_gt(gt_path) if gt_path else (None, np.nan)

    norm_ref = safe_max / (float(scale_ref) + 1e-12) if gt is not None else np.nan

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
    arrays = [item[key] for item in items]
    max_a = max(arr.shape[1] for arr in arrays)
    max_t = max(arr.shape[2] for arr in arrays)
    max_c = max(arr.shape[3] for arr in arrays)
    padded = []
    for arr in arrays:
        pad_cfg = ((0, 0), (0, max_a - arr.shape[1]), (0, max_t - arr.shape[2]), (0, max_c - arr.shape[3]))
        padded.append(np.pad(arr, pad_cfg, mode="constant"))
    return np.concatenate(padded, axis=0)


def save_gt_images(items, image_dir, dr=DYNAMIC_RANGE):
    image_dir.mkdir(parents=True, exist_ok=True)

    def add_scale_bar(ax, extent_mm):
        bar_length = 5.0
        bar_x = extent_mm[1] - bar_length - 2.0
        bar_y = extent_mm[2] - 2.0
        ax.add_patch(patches.Rectangle((bar_x, bar_y), bar_length, 0.5, color="white", zorder=5))
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
        if x_grid.size < 2 or z_grid.size < 2:
            plt.imsave(out_path, gt_sample, cmap="gray", vmin=0.0, vmax=1.0)
            continue

        extent_mm = [x_grid[0] * 1000.0, x_grid[-1] * 1000.0, z_grid[-1] * 1000.0, z_grid[0] * 1000.0]
        gt_db = np.clip(gt_sample, 0.0, 1.0) * dr - dr

        fig, ax = plt.subplots(figsize=(6, 7), dpi=150)
        im = ax.imshow(gt_db, cmap="gray", vmin=-dr, vmax=0.0, extent=extent_mm, aspect="equal")
        ax.set_title(f"GT: {sample_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Lateral (mm)")
        ax.set_ylabel("Depth (mm)")
        add_scale_bar(ax, extent_mm)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Normalized Envelope (dB)", rotation=270, labelpad=15)

        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
def pack_dataset(scenes, output_path, source_root, row_block=24, save_gt_images_enabled=True):
    items = [process_scene(scene, source_root, row_block=row_block) for scene in scenes]
    base = items[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as hf:
        hf.create_dataset("all_multi_I", data=pad_and_concat(items, "I"), compression="gzip", compression_opts=4)
        hf.create_dataset("all_multi_Q", data=pad_and_concat(items, "Q"), compression="gzip", compression_opts=4)

        if all(item["gt"] is not None for item in items):
            hf.create_dataset("all_envdb_norm", data=np.concatenate([item["gt"] for item in items], axis=0),
                              compression="gzip", compression_opts=4)

        hf.create_dataset("all_scale_ref", data=np.array([item["scale_ref"] for item in items], dtype=np.float32))
        hf.create_dataset("all_norm_ref", data=np.array([item["norm_ref"] for item in items], dtype=np.float32))

        max_a = max(item["t0"].shape[1] for item in items)
        t0_padded = []
        for item in items:
            t0_padded.append(np.pad(item["t0"], ((0, 0), (0, max_a - item["t0"].shape[1])), mode="constant"))
        hf.create_dataset("time_start_vector", data=np.concatenate(t0_padded, axis=0), compression="gzip", compression_opts=4)

        for meta_key in ["fs", "c", "fc", "pitch", "num_channels", "z_grid", "x_grid", "angles"]:
            hf.create_dataset(meta_key, data=base[meta_key])

        string_dtype = h5py.string_dtype(encoding="utf-8")
        for key in ["name", "mode", "source", "iq", "scan", "phantom", "gt"]:
            values = [item["meta"][key] for item in items]
            dataset_name = {
                "name": "sample_names",
                "mode": "phantom_mode",
                "source": "phantom_source",
                "iq": "iq_path",
                "scan": "scan_path",
                "phantom": "phantom_path",
                "gt": "gt_path",
            }[key]
            hf.create_dataset(dataset_name, data=np.array(values, dtype=object), dtype=string_dtype)
        hf.create_dataset("has_gt", data=np.array([item["gt"] is not None for item in items], dtype=np.bool_))

    if save_gt_images_enabled:
        save_gt_images(items, output_path.parent / "gt_preview")

    print(f"Saved {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Pack project H5 datasets.")
    parser.add_argument("--source_root", default=str(default_source_root()))
    parser.add_argument("--out_dir", default=str(project_root() / "data"))
    parser.add_argument("--only", default="all", help="all, simulation, experiments, in_vivo")
    parser.add_argument("--row_block", type=int, default=24, help="GPU按深度方向分块行数；<=0 表示整幅一次计算")
    parser.add_argument("--no_save_gt_images", action="store_true", help="不额外输出GT预览图")
    return parser.parse_args()


def main():
    args = parse_args()
    source_root = Path(args.source_root)
    out_dir = Path(args.out_dir)
    choices = ["simulation", "experiments", "in_vivo"] if args.only == "all" else [args.only]
    scene_map = {
        "simulation": (SIMULATION_SCENES, out_dir / "simulation.h5"),
        "experiments": (EXPERIMENT_SCENES, out_dir / "experiments.h5"),
        "in_vivo": (IN_VIVO_SCENES, out_dir / "in_vivo.h5"),
    }
    for choice in choices:
        scenes, output_path = scene_map[choice]
        pack_dataset(scenes, output_path, source_root, row_block=args.row_block, save_gt_images_enabled=not args.no_save_gt_images)


if __name__ == "__main__":
    main()







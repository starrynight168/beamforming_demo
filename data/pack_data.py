import argparse
import os
from pathlib import Path

import h5py
import numpy as np


BASE = "US/US_DATASET0000"
DYNAMIC_RANGE = 60.0


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
        "gt": "",
    },
    {
        "name": "carotid_long",
        "mode": "in_vivo",
        "source": "in_vivo",
        "iq": "database/in_vivo/carotid_long/carotid_long_expe_dataset_iq.hdf5",
        "scan": "database/in_vivo/carotid_long/carotid_long_expe_scan.hdf5",
        "phantom": "",
        "gt": "",
    },
]


def project_root():
    return Path(__file__).resolve().parents[1]


def default_source_root():
    return project_root() / "PICMUS"


def complex_standardization(i_data, q_data):
    i_data = i_data.astype(np.float32)
    q_data = q_data.astype(np.float32)
    envelope = np.sqrt(i_data ** 2 + q_data ** 2)
    scale = np.std(envelope) + 1e-12
    return i_data / scale, q_data / scale


def read_gt(gt_path):
    with h5py.File(gt_path, "r") as f:
        real = f[f"{BASE}/data/real"][:][-1].T
        imag = f[f"{BASE}/data/imag"][:][-1].T
    env_sq = real ** 2 + imag ** 2
    env_sq /= np.max(env_sq) + 1e-24
    db = 10.0 * np.log10(env_sq + 1e-24)
    norm = (np.clip(db, -DYNAMIC_RANGE, 0.0) + DYNAMIC_RANGE) / DYNAMIC_RANGE
    return norm[np.newaxis, np.newaxis, ...].astype(np.float32)


def process_scene(scene, source_root):
    print(f"Processing {scene['name']}")
    iq_path = source_root / scene["iq"]
    scan_path = source_root / scene["scan"]
    gt_path = source_root / scene["gt"] if scene["gt"] else None

    gt = read_gt(gt_path) if gt_path else None

    with h5py.File(iq_path, "r") as f:
        i_data = f[f"{BASE}/data/real"][:]
        q_data = f[f"{BASE}/data/imag"][:]
        i_norm, q_norm = complex_standardization(i_data, q_data)
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


def pack_dataset(scenes, output_path, source_root):
    items = [process_scene(scene, source_root) for scene in scenes]
    base = items[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as hf:
        hf.create_dataset("all_multi_I", data=pad_and_concat(items, "I"), compression="gzip", compression_opts=4)
        hf.create_dataset("all_multi_Q", data=pad_and_concat(items, "Q"), compression="gzip", compression_opts=4)

        if all(item["gt"] is not None for item in items):
            hf.create_dataset("all_envdb_norm", data=np.concatenate([item["gt"] for item in items], axis=0),
                              compression="gzip", compression_opts=4)

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

    print(f"Saved {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Pack project H5 datasets.")
    parser.add_argument("--source_root", default=str(default_source_root()))
    parser.add_argument("--out_dir", default=str(project_root() / "data"))
    parser.add_argument("--only", default="all", help="all, simulation, experiments, in_vivo")
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
        pack_dataset(scenes, output_path, source_root)


if __name__ == "__main__":
    main()

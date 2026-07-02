import os
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent
PICMUS = ROOT / "PICMUS"

PACKED_FILES = [
    ("simulation", ROOT / "data" / "simulation.h5", True, 2),
    ("experiments", ROOT / "data" / "experiments.h5", True, 2),
    ("in_vivo", ROOT / "data" / "in_vivo.h5", False, 2),
]

REQUIRED_H5_KEYS = [
    "all_multi_I",
    "all_multi_Q",
    "time_start_vector",
    "fs",
    "c",
    "fc",
    "pitch",
    "num_channels",
    "z_grid",
    "x_grid",
    "angles",
]

PICMUS_PHANTOMS = [
    ("simulation contrast", PICMUS / "database/simulation/contrast_speckle/contrast_speckle_simu_phantom.hdf5", "contrast"),
    ("simulation resolution", PICMUS / "database/simulation/resolution_distorsion/resolution_distorsion_simu_phantom.hdf5", "resolution"),
    ("experiments contrast", PICMUS / "database/experiments/contrast_speckle/contrast_speckle_expe_phantom.hdf5", "contrast"),
    ("experiments resolution", PICMUS / "database/experiments/resolution_distorsion/resolution_distorsion_expe_phantom.hdf5", "resolution"),
]

PICMUS_IN_VIVO = [
    PICMUS / "database/in_vivo/carotid_cross/carotid_cross_expe_dataset_iq.hdf5",
    PICMUS / "database/in_vivo/carotid_cross/carotid_cross_expe_scan.hdf5",
    PICMUS / "database/in_vivo/carotid_long/carotid_long_expe_dataset_iq.hdf5",
    PICMUS / "database/in_vivo/carotid_long/carotid_long_expe_scan.hdf5",
]


def ok(message):
    print(f"[OK]   {message}")


def warn(message):
    print(f"[WARN] {message}")


def fail(message):
    print(f"[FAIL] {message}")


def exists(path):
    if path.exists():
        ok(str(path.relative_to(ROOT)))
        return True
    fail(f"missing: {path.relative_to(ROOT)}")
    return False


def read_strings(dataset):
    values = dataset[:]
    out = []
    for value in values:
        out.append(value.decode("utf-8") if isinstance(value, bytes) else str(value))
    return out


def check_packed_h5(name, path, expect_gt, expected_samples):
    if not exists(path):
        return False
    passed = True
    with h5py.File(path, "r") as hf:
        for key in REQUIRED_H5_KEYS:
            if key not in hf:
                fail(f"{path.name}: missing key {key}")
                passed = False
        has_gt = "all_envdb_norm" in hf
        if has_gt != expect_gt:
            fail(f"{path.name}: GT presence is {has_gt}, expected {expect_gt}")
            passed = False
        if "all_multi_I" in hf and "all_multi_Q" in hf:
            i_shape = hf["all_multi_I"].shape
            q_shape = hf["all_multi_Q"].shape
            if i_shape != q_shape:
                fail(f"{path.name}: I/Q shape mismatch {i_shape} vs {q_shape}")
                passed = False
            elif i_shape[0] != expected_samples:
                fail(f"{path.name}: sample count {i_shape[0]}, expected {expected_samples}")
                passed = False
            else:
                ok(f"{name}: I/Q shape {i_shape}")
        if "sample_names" in hf:
            ok(f"{name}: samples = {', '.join(read_strings(hf['sample_names']))}")
        if "has_gt" in hf:
            ok(f"{name}: has_gt = {hf['has_gt'][:].astype(bool).tolist()}")
    return passed


def check_phantom(label, path, kind):
    if not exists(path):
        return False
    base = "US/US_DATASET0000"
    passed = True
    with h5py.File(path, "r") as hf:
        if kind == "contrast":
            required = [
                "phantom_occlusionCenterX",
                "phantom_occlusionCenterZ",
                "phantom_occlusionDiameter",
                "phantom_RoiCenterX",
                "phantom_RoiCenterZ",
                "phantom_RoiPsfTimeX",
                "phantom_RoiPsfTimeZ",
            ]
        else:
            required = ["phantom_xPts", "phantom_zPts"]
        for key in required:
            full_key = f"{base}/{key}"
            if full_key not in hf:
                fail(f"{label}: missing {key}")
                passed = False
        if kind == "contrast" and f"{base}/phantom_occlusionCenterX" in hf:
            ok(f"{label}: contrast ROIs = {len(hf[f'{base}/phantom_occlusionCenterX'])}")
        if kind == "resolution" and f"{base}/phantom_xPts" in hf:
            xs = np.asarray(hf[f"{base}/phantom_xPts"][:])
            ok(f"{label}: point targets = {int(np.isfinite(xs).sum())}")
    return passed


def main():
    print(f"Project: {ROOT}")
    print()

    passed = True
    for name, path, expect_gt, expected_samples in PACKED_FILES:
        passed = check_packed_h5(name, path, expect_gt, expected_samples) and passed

    print()
    for label, path, kind in PICMUS_PHANTOMS:
        passed = check_phantom(label, path, kind) and passed

    print()
    for path in PICMUS_IN_VIVO:
        passed = exists(path) and passed

    print()
    for path in [ROOT / "config.yaml", ROOT / "run_one.py", ROOT / "run_all.py", ROOT / "evaluation" / "evaluate.py"]:
        passed = exists(path) and passed

    print()
    if passed:
        ok("data check passed")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

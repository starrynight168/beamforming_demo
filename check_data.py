import os
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent
PICMUS = ROOT / "data" / "PICMUS"

PACKED_FILES = [
    ("simulation", ROOT / "data" / "simulation.h5", True, 2),
    ("experiments", ROOT / "data" / "experiments.h5", True, 2),
    ("in_vivo", ROOT / "data" / "in_vivo.h5", True, 2),
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

REQUIRED_STRING_KEYS = [
    "sample_names",
    "phantom_mode",
    "phantom_source",
    "iq_path",
    "scan_path",
    "phantom_path",
    "gt_path",
]

NUMERIC_FIELD_INFO = {
    "fs": ("fs", "sampling frequency", "Hz", 1e5, 1e9),
    "c": ("c", "sound speed", "m/s", 1000.0, 2000.0),
    "fc": ("fc", "center frequency", "Hz", 1e5, 1e9),
    "pitch": ("pitch", "element pitch", "m", 1e-6, 5e-3),
    "num_channels": ("num_channels", "channel count", "", 1, 4096),
}

GRID_FIELD_INFO = {
    "z_grid": ("z_grid", "depth grid", "m"),
    "x_grid": ("x_grid", "lateral grid", "m"),
    "angles": ("angles", "plane-wave angles", "rad"),
}

SEPARATOR = "-" * 90

META_FIELD_MAP = {
    "sample_names": "meta.name",
    "phantom_mode": "meta.mode",
    "phantom_source": "meta.source",
    "iq_path": "meta.iq",
    "scan_path": "meta.scan",
    "phantom_path": "meta.phantom",
    "gt_path": "meta.gt",
}

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


def section(title):
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def print_item(label, value):
    print(f"  {label:<28}: {value}")


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


def check_string_dataset(name, path_name, hf, key, expected_samples, allow_empty=False):
    if key not in hf:
        fail(f"{path_name}: missing string key {key}")
        return False

    passed = True
    values = read_strings(hf[key])
    if len(values) != expected_samples:
        fail(f"{path_name}: {key} count {len(values)}, expected {expected_samples}")
        passed = False

    empty_indices = [idx for idx, value in enumerate(values) if value == ""]
    if empty_indices and not allow_empty:
        fail(f"{path_name}: {key} has empty values at {empty_indices}")
        passed = False

    try:
        [value.encode("utf-8").decode("utf-8") for value in values]
    except UnicodeError:
        fail(f"{path_name}: {key} contains non-UTF8 string values")
        passed = False

    if os.name == "nt":
        absolute_values = [value for value in values if len(value) > 2 and value[1:3] in (":\\", ":/")]
    else:
        absolute_values = [value for value in values if value.startswith("/")]
    if absolute_values:
        fail(f"{path_name}: {key} contains absolute paths: {absolute_values}")
        passed = False

    if passed:
        display_key = META_FIELD_MAP.get(key, key)
        print_item(f"{display_key} -> {key}", values)
    return passed


def scalar_value(dataset):
    value = dataset[()]
    return value.item() if hasattr(value, "item") else value


def check_numeric_fields(name, path_name, hf):
    passed = True
    for key, (field_name, label, unit, low, high) in NUMERIC_FIELD_INFO.items():
        if key not in hf:
            fail(f"{path_name}: missing numeric key {key}")
            passed = False
            continue
        value = scalar_value(hf[key])
        if not np.isscalar(value) or not np.isfinite(value):
            fail(f"{path_name}: {key} is not a finite scalar: {value}")
            passed = False
            continue
        if not (low <= float(value) <= high):
            fail(f"{path_name}: {key}={value} outside expected range [{low}, {high}]")
            passed = False
        else:
            suffix = f" {unit}" if unit else ""
            print_item(f"{field_name} -> {key}", f"{value}{suffix} ({label})")

    for key, (field_name, label, unit) in GRID_FIELD_INFO.items():
        if key not in hf:
            fail(f"{path_name}: missing grid key {key}")
            passed = False
            continue
        values = np.asarray(hf[key][:], dtype=float)
        if values.ndim != 1 or len(values) == 0:
            fail(f"{path_name}: {key} must be a non-empty 1D array, got shape {values.shape}")
            passed = False
            continue
        if not np.all(np.isfinite(values)):
            fail(f"{path_name}: {key} contains non-finite values")
            passed = False
            continue
        diffs = np.diff(values)
        monotonic = len(values) == 1 or np.all(diffs > 0) or np.all(diffs < 0)
        if not monotonic:
            fail(f"{path_name}: {key} is not monotonic")
            passed = False
        unit_suffix = f" {unit}" if unit else ""
        if len(values) > 1:
            step = float(np.median(np.abs(diffs)))
            print_item(
                f"{field_name} -> {key}",
                f"len={len(values)}, range=[{values.min():.6g}, {values.max():.6g}]{unit_suffix}, step~{step:.6g}{unit_suffix} ({label})",
            )
        else:
            print_item(f"{field_name} -> {key}", f"len=1, value={values[0]:.6g}{unit_suffix} ({label})")

    if "time_start_vector" in hf:
        t0 = np.asarray(hf["time_start_vector"][:], dtype=float)
        if t0.shape[0] == 0 or not np.all(np.isfinite(t0)):
            fail(f"{path_name}: time_start_vector is empty or non-finite")
            passed = False
        else:
            print_item("t0 -> time_start_vector", f"shape={t0.shape}, range=[{t0.min():.6g}, {t0.max():.6g}] s")

    if "all_multi_I" in hf and "num_channels" in hf:
        num_channels = int(scalar_value(hf["num_channels"]))
        actual_channels = int(hf["all_multi_I"].shape[-1])
        if num_channels != actual_channels:
            fail(f"{path_name}: num_channels={num_channels}, I channel dim={actual_channels}")
            passed = False
    return passed


def check_packed_h5(name, path, expect_gt, expected_samples):
    if not exists(path):
        return False
    passed = True
    section(f"{name} | {path.relative_to(ROOT)}")
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
                print_item("I -> all_multi_I", f"shape={i_shape}, dtype={hf['all_multi_I'].dtype}")
                print_item("Q -> all_multi_Q", f"shape={q_shape}, dtype={hf['all_multi_Q'].dtype}")
        if "all_envdb_norm" in hf:
            gt_shape = hf["all_envdb_norm"].shape
            print_item("gt -> all_envdb_norm", f"shape={gt_shape}, dtype={hf['all_envdb_norm'].dtype}")
        elif expect_gt:
            fail(f"{path.name}: missing gt -> all_envdb_norm")
            passed = False
        else:
            print_item("gt -> all_envdb_norm", "absent (expected)")
        print(SEPARATOR)
        passed = check_numeric_fields(name, path.name, hf) and passed
        print(SEPARATOR)
        for key in REQUIRED_STRING_KEYS:
            allow_empty = key == "phantom_path" and name == "in_vivo" or (key in {"phantom_path", "gt_path"} and not expect_gt)
            passed = check_string_dataset(name, path.name, hf, key, expected_samples, allow_empty) and passed
        if "has_gt" in hf:
            has_gt_values = hf["has_gt"][:].astype(bool).tolist()
            if len(has_gt_values) != expected_samples:
                fail(f"{path.name}: has_gt count {len(has_gt_values)}, expected {expected_samples}")
                passed = False
            elif any(value != expect_gt for value in has_gt_values):
                fail(f"{path.name}: has_gt = {has_gt_values}, expected all {expect_gt}")
                passed = False
            else:
                print_item("meta.has_gt -> has_gt", has_gt_values)
        else:
            fail(f"{path.name}: missing key has_gt")
            passed = False
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

"""
Evaluate compare.py output.

Default input:
    results/<scene>/comparison.npy
"""

import argparse
import csv
import json
import math
import os

import h5py
import numpy as np
from scipy.ndimage import convolve, uniform_filter
from scipy.stats import kstest

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
except Exception:
    peak_signal_noise_ratio = None
    structural_similarity = None


DEFAULT_METHODS = ["DAS", "MV", "ESBMV", "F-DMAS"]
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
SOURCE_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, os.pardir))
PICMUS_ROOT = os.path.join(PROJECT_ROOT, "PICMUS")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate compare.py output."
    )
    parser.add_argument("--comparison_npy", default=os.path.join("results", "comparison.npy"))
    parser.add_argument("--methods", default=None, help="逗号分隔算法名，不含GT；默认从run_params.json读取")
    parser.add_argument("--method_labels", default=None, help="逗号分隔显示名，不含GT；默认使用run_params.json或算法名大写")
    parser.add_argument("--h5_path", default=os.path.join("data", "simulation.h5"))
    parser.add_argument("--h5_sample_idx", type=int, default=0)
    parser.add_argument("--dr", type=float, default=60.0)
    parser.add_argument("--out_dir", default=os.path.join("results", "metrics"))
    parser.add_argument(
        "--phantom_path",
        default="auto",
        help="Phantom hdf5 for predefined ROI/targets; use 'none' to disable.",
    )
    parser.add_argument(
        "--phantom_mode",
        choices=["auto", "contrast_speckle", "resolution_distorsion", "in_vivo"],
        default="auto",
    )
    parser.add_argument(
        "--phantom_source",
        choices=["auto", "simulation", "experiments", "in_vivo"],
        default="auto",
        help="Phantom source; auto uses experiments for validation H5 files and simulation otherwise.",
    )
    parser.add_argument("--has_gt", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--roi_ring_inner_mm", type=float, default=3.5)
    parser.add_argument("--roi_ring_outer_mm", type=float, default=5.5)
    parser.add_argument("--auto_roi_radius_mm", type=float, default=2.0)
    parser.add_argument("--auto_target_count", type=int, default=20)
    parser.add_argument("--auto_target_min_distance_mm", type=float, default=3.0)
    parser.add_argument("--fwhm_window_mm", type=float, default=1.8)
    return parser.parse_args()


def load_method_names(args, comparison_path, n_panels, has_gt):
    if args.methods:
        methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
        if args.method_labels:
            names = [m.strip() for m in args.method_labels.split(",") if m.strip()]
        else:
            names = [m.upper() for m in methods]
    else:
        params_path = os.path.join(os.path.dirname(comparison_path), "run_params.json")
        names = None
        if os.path.exists(params_path):
            with open(params_path, "r", encoding="utf-8") as f:
                params = json.load(f)
            label_map = params.get("algorithm_labels", {}) or {}
            method_text = params.get("output", {}).get("methods")
            if method_text:
                methods = [m.strip().lower() for m in method_text.split(",") if m.strip()]
                names = [label_map.get(m, m.upper()) for m in methods]
        if names is None:
            n_methods = n_panels - 1 if has_gt else n_panels
            if n_methods <= len(DEFAULT_METHODS):
                names = DEFAULT_METHODS[:n_methods]
            else:
                names = DEFAULT_METHODS + [f"Method{i + 1}" for i in range(len(DEFAULT_METHODS), n_methods)]
    if has_gt:
        names = ["GT"] + names
    if len(names) != n_panels:
        raise ValueError(f"Method count {len(names)} does not match stacked image count {n_panels}")
    return names


def resolve(path):
    if path in ("auto", "none", None):
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def resolve_existing(relative_path):
    if not relative_path:
        return "none"
    if os.path.isabs(relative_path):
        return relative_path if os.path.exists(relative_path) else "none"
    candidates = [
        os.path.join(PICMUS_ROOT, relative_path),
        os.path.join(PROJECT_ROOT, "data", relative_path),
        os.path.join(PROJECT_ROOT, relative_path),
        os.path.join(SOURCE_ROOT, relative_path),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return "none"


def h5_string_at(hf, key, idx, default=""):
    if key not in hf:
        return default
    value = hf[key][idx]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def h5_bool_at(hf, key, idx, default=False):
    if key not in hf:
        return default
    return bool(hf[key][idx])


def read_sample_meta(h5_path, sample_idx):
    out = {}
    with h5py.File(h5_path, "r") as hf:
        out["has_gt"] = h5_bool_at(hf, "has_gt", sample_idx, "all_envdb_norm" in hf)
        out["phantom_mode"] = h5_string_at(hf, "phantom_mode", sample_idx, "")
        out["phantom_source"] = h5_string_at(hf, "phantom_source", sample_idx, "")
        out["phantom_path"] = h5_string_at(hf, "phantom_path", sample_idx, "")
        out["sample_name"] = h5_string_at(hf, "sample_names", sample_idx, "")
    return out


def load_grids(h5_path):
    with h5py.File(h5_path, "r") as hf:
        x_mm = hf["x_grid"][:].astype(float) * 1000.0
        z_mm = hf["z_grid"][:].astype(float) * 1000.0
    return x_mm, z_mm


def db_to_display(db, dr):
    return np.clip((db + dr) / dr, 0.0, 1.0)


def db_to_envelope(db):
    return 10.0 ** (db / 20.0)


def psnr(image, reference, data_range=1.0):
    mse = float(np.mean((image - reference) ** 2))
    if mse == 0.0:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def local_ssim(image, reference, data_range=1.0, win_size=11):
    image = image.astype(np.float64)
    reference = reference.astype(np.float64)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ux = uniform_filter(image, win_size)
    uy = uniform_filter(reference, win_size)
    uxx = uniform_filter(image * image, win_size)
    uyy = uniform_filter(reference * reference, win_size)
    uxy = uniform_filter(image * reference, win_size)
    vx = uxx - ux * ux
    vy = uyy - uy * uy
    vxy = uxy - ux * uy
    score = ((2.0 * ux * uy + c1) * (2.0 * vxy + c2)) / (
        (ux * ux + uy * uy + c1) * (vx + vy + c2)
    )
    return float(np.mean(score))


def disk_kernel(radius_mm, dx_mm, dz_mm):
    rx = max(int(np.ceil(radius_mm / abs(dx_mm))), 1)
    rz = max(int(np.ceil(radius_mm / abs(dz_mm))), 1)
    yy, xx = np.mgrid[-rz:rz + 1, -rx:rx + 1]
    kernel = ((xx * dx_mm) ** 2 + (yy * dz_mm) ** 2 <= radius_mm ** 2).astype(float)
    return kernel / max(kernel.sum(), 1.0)


def auto_dark_rois(gt_db, x_mm, z_mm, radius_mm):
    gt_env = db_to_envelope(gt_db)
    kernel = disk_kernel(radius_mm, x_mm[1] - x_mm[0], z_mm[1] - z_mm[0])
    local_mean = convolve(gt_env, kernel, mode="constant", cval=float(np.nanmax(gt_env)))
    x_margin = 0.15 * (x_mm[-1] - x_mm[0])
    z_margin = 0.15 * (z_mm[-1] - z_mm[0])
    valid = (
        (x_mm[None, :] >= x_mm[0] + x_margin)
        & (x_mm[None, :] <= x_mm[-1] - x_margin)
        & (z_mm[:, None] >= z_mm[0] + z_margin)
        & (z_mm[:, None] <= z_mm[-1] - z_margin)
    )
    score = np.where(valid, local_mean, np.inf)
    iz, ix = np.unravel_index(np.argmin(score), score.shape)
    return [{"x_mm": float(x_mm[ix]), "z_mm": float(z_mm[iz]), "diameter_mm": 2.0 * radius_mm, "source": "auto_gt_dark"}]


def auto_bright_targets(gt_db, x_mm, z_mm, max_targets, min_distance_mm):
    valid = np.isfinite(gt_db)
    x_margin = 0.08 * (x_mm[-1] - x_mm[0])
    z_margin = 0.08 * (z_mm[-1] - z_mm[0])
    valid &= (
        (x_mm[None, :] >= x_mm[0] + x_margin)
        & (x_mm[None, :] <= x_mm[-1] - x_margin)
        & (z_mm[:, None] >= z_mm[0] + z_margin)
        & (z_mm[:, None] <= z_mm[-1] - z_margin)
    )
    score = np.where(valid, gt_db, -np.inf)
    flat_order = np.argsort(score.ravel())[::-1]
    targets = []
    for flat_idx in flat_order:
        if len(targets) >= max_targets:
            break
        iz, ix = np.unravel_index(flat_idx, score.shape)
        if not np.isfinite(score[iz, ix]):
            break
        x = float(x_mm[ix])
        z = float(z_mm[iz])
        too_close = any((x - t["x_mm"]) ** 2 + (z - t["z_mm"]) ** 2 < min_distance_mm ** 2 for t in targets)
        if not too_close:
            targets.append({"x_mm": x, "z_mm": z, "source": "auto_gt_bright"})
    targets.sort(key=lambda t: (t["z_mm"], t["x_mm"]))
    return targets


def read_phantom(phantom_path):
    out = {
        "contrast_rois": [],
        "speckle_rois": [],
        "resolution_targets": [],
        "lateral_resolution_mm": None,
        "axial_resolution_mm": None,
    }
    if not phantom_path or phantom_path == "none":
        return out
    with h5py.File(phantom_path, "r") as hf:
        base = "US/US_DATASET0000"
        if f"{base}/phantom_occlusionCenterX" in hf:
            xs = hf[f"{base}/phantom_occlusionCenterX"][:].astype(float) * 1000.0
            zs = hf[f"{base}/phantom_occlusionCenterZ"][:].astype(float) * 1000.0
            ds = hf[f"{base}/phantom_occlusionDiameter"][:].astype(float) * 1000.0
            if f"{base}/phantom_lateralResolution" in hf:
                out["lateral_resolution_mm"] = float(np.squeeze(hf[f"{base}/phantom_lateralResolution"][()])) * 1000.0
            if f"{base}/phantom_axialResolution" in hf:
                out["axial_resolution_mm"] = float(np.squeeze(hf[f"{base}/phantom_axialResolution"][()])) * 1000.0
            for x, z, d in zip(xs, zs, ds):
                if d > 0:
                    out["contrast_rois"].append(
                        {"x_mm": float(x), "z_mm": float(z), "diameter_mm": float(d), "source": "phantom"}
                    )
        if f"{base}/phantom_RoiCenterX" in hf:
            xs = hf[f"{base}/phantom_RoiCenterX"][:].astype(float) * 1000.0
            zs = hf[f"{base}/phantom_RoiCenterZ"][:].astype(float) * 1000.0
            txs = hf[f"{base}/phantom_RoiPsfTimeX"][:].astype(float)
            tzs = hf[f"{base}/phantom_RoiPsfTimeZ"][:].astype(float)
            if f"{base}/phantom_lateralResolution" in hf:
                out["lateral_resolution_mm"] = float(np.squeeze(hf[f"{base}/phantom_lateralResolution"][()])) * 1000.0
            if f"{base}/phantom_axialResolution" in hf:
                out["axial_resolution_mm"] = float(np.squeeze(hf[f"{base}/phantom_axialResolution"][()])) * 1000.0
            for x, z, tx, tz in zip(xs, zs, txs, tzs):
                out["speckle_rois"].append(
                    {"x_mm": float(x), "z_mm": float(z), "psf_time_x": float(tx), "psf_time_z": float(tz), "source": "phantom"}
                )
        if f"{base}/phantom_xPts" in hf:
            xs = hf[f"{base}/phantom_xPts"][:].astype(float) * 1000.0
            zs = hf[f"{base}/phantom_zPts"][:].astype(float) * 1000.0
            for x, z in zip(xs, zs):
                if np.isfinite(x) and np.isfinite(z) and (abs(x) > 1e-9 or abs(z) > 1e-9):
                    out["resolution_targets"].append({"x_mm": float(x), "z_mm": float(z), "source": "phantom"})
    return out


def infer_source(args):
    if args.phantom_source != "auto":
        return args.phantom_source
    meta = read_sample_meta(resolve(args.h5_path), args.h5_sample_idx)
    if meta.get("phantom_source"):
        return meta["phantom_source"]
    h5_name = os.path.basename(resolve(args.h5_path)).lower()
    if "val" in h5_name or "valid" in h5_name:
        return "experiments"
    return "simulation"


def default_phantom(mode, source):
    if mode == "in_vivo" or source == "in_vivo":
        return "none"
    if mode == "resolution_distorsion":
        prefix = "expe" if source == "experiments" else "simu"
        rel = os.path.join("database", source, "resolution_distorsion", f"resolution_distorsion_{prefix}_phantom.hdf5")
    else:
        prefix = "expe" if source == "experiments" else "simu"
        rel = os.path.join("database", source, "contrast_speckle", f"contrast_speckle_{prefix}_phantom.hdf5")
    return resolve_existing(rel)


def infer_mode(args, comparison):
    if args.phantom_mode != "auto":
        return args.phantom_mode
    meta = read_sample_meta(resolve(args.h5_path), args.h5_sample_idx)
    if meta.get("phantom_mode"):
        return meta["phantom_mode"]
    if "resolution" in os.path.basename(args.comparison_npy).lower():
        return "resolution_distorsion"
    if args.h5_sample_idx == 1:
        return "resolution_distorsion"
    return "contrast_speckle"


def infer_has_gt(args, h5_path, sample_idx):
    if args.has_gt == "true":
        return True
    if args.has_gt == "false":
        return False
    return read_sample_meta(h5_path, sample_idx).get("has_gt", False)


def standard_contrast_score(db_img, x_mm, z_mm, roi, lateral_resolution_mm, padding=1.0):
    if not lateral_resolution_mm or not np.isfinite(lateral_resolution_mm):
        return np.nan
    x = x_mm[None, :]
    z = z_mm[:, None]
    radius = roi["diameter_mm"] / 2.0
    rin = radius - padding * lateral_resolution_mm
    rout1 = radius + padding * lateral_resolution_mm
    rout2 = 1.2 * math.sqrt(rin ** 2 + rout1 ** 2)
    dist2 = (x - roi["x_mm"]) ** 2 + (z - roi["z_mm"]) ** 2
    inside = db_img[dist2 <= rin ** 2]
    outside = db_img[(dist2 >= rout1 ** 2) & (dist2 <= rout2 ** 2)]
    if inside.size < 8 or outside.size < 8:
        return np.nan
    denom = math.sqrt((float(np.var(inside)) + float(np.var(outside))) / 2.0)
    if denom <= 0:
        return np.nan
    value = 20.0 * math.log10(abs(float(np.mean(inside)) - float(np.mean(outside))) / denom)
    return float(round(value * 10.0) / 10.0)


def contrast_roi_metrics(db_img, x_mm, z_mm, roi, lateral_resolution_mm, padding=1.0):
    if not lateral_resolution_mm or not np.isfinite(lateral_resolution_mm):
        return None
    x = x_mm[None, :]
    z = z_mm[:, None]
    radius = roi["diameter_mm"] / 2.0
    rin = radius - padding * lateral_resolution_mm
    rout1 = radius + padding * lateral_resolution_mm
    rout2 = 1.2 * math.sqrt(max(rin ** 2 + rout1 ** 2, 0.0))
    if rin <= 0 or rout2 <= rout1:
        return None
    dist2 = (x - roi["x_mm"]) ** 2 + (z - roi["z_mm"]) ** 2
    inside_db = db_img[dist2 <= rin ** 2]
    outside_db = db_img[(dist2 >= rout1 ** 2) & (dist2 <= rout2 ** 2)]
    inside_db = inside_db[np.isfinite(inside_db)]
    outside_db = outside_db[np.isfinite(outside_db)]
    if inside_db.size < 8 or outside_db.size < 8:
        return None

    inside_env = db_to_envelope(inside_db)
    outside_env = db_to_envelope(outside_db)
    mu_in = float(np.mean(inside_env))
    mu_out = float(np.mean(outside_env))
    var_in = float(np.var(inside_env))
    var_out = float(np.var(outside_env))
    cnr = abs(mu_out - mu_in) / math.sqrt(var_in + var_out + 1e-24)
    cr_db = 20.0 * math.log10((mu_out + 1e-12) / (mu_in + 1e-12))
    residual_db = 20.0 * math.log10((mu_in + 1e-12) / (mu_out + 1e-12))

    combined = np.concatenate([inside_env, outside_env])
    lo, hi = float(np.min(combined)), float(np.max(combined))
    if hi <= lo:
        gcnr = np.nan
    else:
        hist_in, bins = np.histogram(inside_env, bins=128, range=(lo, hi), density=False)
        hist_out, _ = np.histogram(outside_env, bins=bins, density=False)
        p_in = hist_in.astype(np.float64) / max(float(hist_in.sum()), 1.0)
        p_out = hist_out.astype(np.float64) / max(float(hist_out.sum()), 1.0)
        gcnr = 1.0 - float(np.sum(np.minimum(p_in, p_out)))

    return {
        "CR_dB": float(cr_db),
        "CNR": float(cnr),
        "gCNR": float(gcnr),
        "cyst_residual_dB": float(residual_db),
    }


def standard_speckle_quality(env, x_mm, z_mm, roi, lateral_resolution_mm, axial_resolution_mm):
    if not lateral_resolution_mm or not axial_resolution_mm:
        return None
    pad_x = roi["psf_time_x"] * lateral_resolution_mm
    pad_z = roi["psf_time_z"] * axial_resolution_mm
    x_mask = (x_mm > roi["x_mm"] - pad_x) & (x_mm < roi["x_mm"] + pad_x)
    z_mask = (z_mm > roi["z_mm"] - pad_z) & (z_mm < roi["z_mm"] + pad_z)
    env_roi = env[np.ix_(z_mask, x_mask)]
    if env_roi.size == 0:
        return None
    sample = env_roi[::5, ::5].ravel()
    sample = sample[np.isfinite(sample)]
    if sample.size < 8:
        return None
    rayleigh_var = float(np.sum(sample ** 2) / (2.0 * sample.size))
    scale = math.sqrt(max(rayleigh_var, 1e-24))
    ks = kstest(sample, "rayleigh", args=(0.0, scale))
    return {
        "speckle_pass": 1.0 if ks.pvalue >= 0.05 else 0.0,
        "speckle_KS_D": float(ks.statistic),
        "speckle_KS_p": float(ks.pvalue),
        "speckle_SNR": float(np.mean(sample) / (np.std(sample) + 1e-12)),
        "ENL": float((np.mean(sample) ** 2) / (np.var(sample) + 1e-24)),
    }


def compute_6db_resolution(coord, profile_db):
    profile = np.asarray(profile_db, dtype=np.float64)
    coord = np.asarray(coord, dtype=np.float64)
    if profile.size < 2 or coord.size != profile.size:
        return np.nan
    nb_interp = profile.size * 10
    coord_interp = np.linspace(coord[0], coord[-1], nb_interp)
    profile_interp = np.interp(coord_interp, coord, profile)
    valid = np.where(profile_interp >= (np.nanmax(profile_interp) - 6.0))[0]
    if valid.size == 0:
        return np.nan
    return float(coord_interp[valid[-1]] - coord_interp[valid[0]])


def profile_sidelobe_metrics(profile_db):
    profile = np.asarray(profile_db, dtype=np.float64)
    profile = profile[np.isfinite(profile)]
    if profile.size < 5:
        return np.nan, np.nan

    peak_idx = int(np.argmax(profile))
    peak_db = float(profile[peak_idx])
    main_threshold = peak_db - 6.0

    left = peak_idx
    while left > 0 and profile[left - 1] >= main_threshold:
        left -= 1
    right = peak_idx
    while right < profile.size - 1 and profile[right + 1] >= main_threshold:
        right += 1

    main = profile[left:right + 1]
    side = np.concatenate([profile[:left], profile[right + 1:]])
    if main.size == 0 or side.size == 0:
        return np.nan, np.nan

    peak_env = float(np.max(db_to_envelope(main)))
    side_env = db_to_envelope(side)
    main_env = db_to_envelope(main)
    if peak_env <= 0:
        return np.nan, np.nan

    pslr = 20.0 * math.log10((float(np.max(side_env)) + 1e-12) / (peak_env + 1e-12))
    islr = 10.0 * math.log10((float(np.sum(side_env ** 2)) + 1e-24) / (float(np.sum(main_env ** 2)) + 1e-24))
    return float(pslr), float(islr)


def target_resolution(db_img, x_mm, z_mm, target, window_mm, target_idx=None, source="simulation"):
    x_mask = np.abs(x_mm - target["x_mm"]) <= window_mm
    z_mask = np.abs(z_mm - target["z_mm"]) <= window_mm
    patch = db_img[np.ix_(z_mask, x_mask)]
    if patch.size == 0:
        return None
    pz, px = np.unravel_index(np.argmax(patch), patch.shape)
    z_ids = np.where(z_mask)[0]
    x_ids = np.where(x_mask)[0]
    iz = int(z_ids[pz])
    ix = int(x_ids[px])
    axial = compute_6db_resolution(z_mm[z_mask], patch[:, px])
    lateral = compute_6db_resolution(x_mm[x_mask], patch[pz, :])
    PSLR_axial_dB, ISLR_axial_dB = profile_sidelobe_metrics(patch[:, px])
    PSLR_lateral_dB, ISLR_lateral_dB = profile_sidelobe_metrics(patch[pz, :])
    PSLR_dB = mean_or_nan([PSLR_axial_dB, PSLR_lateral_dB])
    ISLR_dB = mean_or_nan([ISLR_axial_dB, ISLR_lateral_dB])
    peak_offset = math.sqrt((x_mm[ix] - target["x_mm"]) ** 2 + (z_mm[iz] - target["z_mm"]) ** 2)
    distortion_pass = np.nan
    if source == "simulation" and target_idx is not None and target_idx in {1, 5, 8, 9, 14, 15, 20}:
        corrected_z = target["z_mm"] + 0.2
        inside = (
            abs(x_mm[ix] - target["x_mm"]) < 0.2957
            and abs(z_mm[iz] - corrected_z) < 0.2957
        )
        distortion_pass = 1.0 if inside else 0.0
    return {
        "target_x_mm": target["x_mm"],
        "target_z_mm": target["z_mm"],
        "peak_x_mm": float(x_mm[ix]),
        "peak_z_mm": float(z_mm[iz]),
        "FWHM_axial_mm": axial,
        "FWHM_lateral_mm": lateral,
        "PSLR_axial_dB": float(PSLR_axial_dB),
        "PSLR_lateral_dB": float(PSLR_lateral_dB),
        "PSLR_dB": float(PSLR_dB),
        "ISLR_axial_dB": float(ISLR_axial_dB),
        "ISLR_lateral_dB": float(ISLR_lateral_dB),
        "ISLR_dB": float(ISLR_dB),
        "distortion_mm": float(peak_offset),
        "distortion_pass": distortion_pass,
    }


def mean_or_nan(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else np.nan


def picmus_resolution_groups(source, target_count):
    if source == "simulation" and target_count >= 20:
        return [
            ("vertical_targets", list(range(1, 9))),
            ("horizontal_targets_2cm", [9, 10, 11, 3, 12, 13, 14]),
            ("horizontal_targets_4cm", [15, 16, 17, 7, 18, 19, 20]),
        ]
    if source == "experiments" and target_count >= 7:
        return [
            ("vertical_targets", list(range(1, 6))),
            ("horizontal_targets_near_4cm", [6, 4, 7]),
        ]
    return [("all_targets", list(range(1, target_count + 1)))] if target_count else []


def picmus_contrast_groups(source, roi_count):
    if source == "simulation" and roi_count >= 9:
        return [
            ("left_column", [4, 5, 6]),
            ("middle_column", [1, 2, 3]),
            ("right_column", [7, 8, 9]),
        ]
    if source == "experiments" and roi_count >= 2:
        return [("middle_column", [1, 2])]
    return [("all_cysts", list(range(1, roi_count + 1)))] if roi_count else []


def rows_for_method_index(rows, method, index_key):
    out = {}
    for row in rows:
        if row.get("method") != method or index_key not in row:
            continue
        try:
            out[int(row[index_key])] = row
        except (TypeError, ValueError):
            continue
    return out


def speckle_penalty_for_method(roi_rows, method):
    passes = [
        row["speckle_pass"]
        for row in roi_rows
        if row.get("method") == method and "speckle_pass" in row and np.isfinite(row["speckle_pass"])
    ]
    if not passes:
        return np.nan
    return -40.0 if any(value < 0.5 for value in passes) else 0.0


def build_resolution_group_rows(target_rows, source, target_count, methods):
    groups = picmus_resolution_groups(source, target_count)
    rows = []
    for method in methods:
        indexed = rows_for_method_index(target_rows, method, "target_idx")
        for group_name, indices in groups:
            selected = [indexed[i] for i in indices if i in indexed]
            if not selected:
                continue
            pass_values = [row["distortion_pass"] for row in selected if np.isfinite(row.get("distortion_pass", np.nan))]
            if source == "simulation" and pass_values:
                penalty = -40.0 if any(value < 0.5 for value in pass_values) else 0.0
                pass_rate = mean_or_nan(pass_values)
            else:
                penalty = np.nan
                pass_rate = np.nan
            rows.append({
                "method": method,
                "group": group_name,
                "target_indices": " ".join(str(i) for i in indices),
                "n_targets": len(selected),
                "FWHM_axial_mm": mean_or_nan([row.get("FWHM_axial_mm") for row in selected]),
                "FWHM_lateral_mm": mean_or_nan([row.get("FWHM_lateral_mm") for row in selected]),
                "PSLR_dB": mean_or_nan([row.get("PSLR_dB") for row in selected]),
                "ISLR_dB": mean_or_nan([row.get("ISLR_dB") for row in selected]),
                "distortion_mm": mean_or_nan([row.get("distortion_mm") for row in selected]),
                "distortion_pass_rate": pass_rate,
                "PICMUS_distortion_penalty": penalty,
            })
    return rows


def build_contrast_group_rows(roi_rows, source, roi_count, methods):
    groups = picmus_contrast_groups(source, roi_count)
    rows = []
    for method in methods:
        indexed = rows_for_method_index([row for row in roi_rows if "contrast_dB" in row], method, "roi_idx")
        for group_name, indices0 in groups:
            indices = [i - 1 for i in indices0]
            selected = [indexed[i] for i in indices if i in indexed]
            if not selected:
                continue
            rows.append({
                "method": method,
                "group": group_name,
                "roi_indices": " ".join(str(i) for i in indices0),
                "n_rois": len(selected),
                "contrast_dB": mean_or_nan([row.get("contrast_dB") for row in selected]),
                "CR_dB": mean_or_nan([row.get("CR_dB") for row in selected]),
                "CNR": mean_or_nan([row.get("CNR") for row in selected]),
                "gCNR": mean_or_nan([row.get("gCNR") for row in selected]),
                "cyst_residual_dB": mean_or_nan([row.get("cyst_residual_dB") for row in selected]),
                "PICMUS_speckle_penalty": speckle_penalty_for_method(roi_rows, method),
            })
    return rows


def fmt_metric(value, precision=3):
    try:
        if value is None or not np.isfinite(value):
            return "NA"
    except TypeError:
        return "NA"
    return f"{float(value):.{precision}f}"


def write_picmus_report(path, mode, source, methods, summary_rows, contrast_group_rows, resolution_group_rows, roi_rows):
    summary_by_method = {row["method"]: row for row in summary_rows}
    with open(path, "w", encoding="utf-8") as file:
        file.write("PICMUS-style evaluation summary\n")
        file.write(f"mode: {mode}\nsource: {source}\n\n")
        for method in methods:
            row = summary_by_method.get(method, {})
            file.write(f"[{method}]\n")
            if mode == "contrast_speckle":
                file.write(f"mean_contrast_dB: {fmt_metric(row.get('contrast_dB'), 1)}\n")
                file.write(f"mean_CR_dB: {fmt_metric(row.get('CR_dB'), 3)}\n")
                file.write(f"mean_CNR: {fmt_metric(row.get('CNR'), 3)}\n")
                file.write(f"mean_gCNR: {fmt_metric(row.get('gCNR'), 3)}\n")
                file.write(f"speckle_pass_rate: {fmt_metric(row.get('speckle_pass_rate'), 3)}\n")
                file.write(f"PICMUS_speckle_penalty: {fmt_metric(speckle_penalty_for_method(roi_rows, method), 1)}\n")
                for group in [r for r in contrast_group_rows if r["method"] == method]:
                    file.write(
                        f"  {group['group']} ({group['roi_indices']}): "
                        f"contrast={fmt_metric(group.get('contrast_dB'), 1)}, "
                        f"CNR={fmt_metric(group.get('CNR'), 3)}, "
                        f"gCNR={fmt_metric(group.get('gCNR'), 3)}\n"
                    )
            elif mode == "resolution_distorsion":
                file.write(f"mean_FWHM_axial_mm: {fmt_metric(row.get('FWHM_axial_mm'), 4)}\n")
                file.write(f"mean_FWHM_lateral_mm: {fmt_metric(row.get('FWHM_lateral_mm'), 4)}\n")
                file.write(f"mean_PSLR_dB: {fmt_metric(row.get('PSLR_dB'), 3)}\n")
                file.write(f"mean_ISLR_dB: {fmt_metric(row.get('ISLR_dB'), 3)}\n")
                file.write(f"distortion_pass_rate: {fmt_metric(row.get('distortion_pass_rate'), 3)}\n")
                for group in [r for r in resolution_group_rows if r["method"] == method]:
                    file.write(
                        f"  {group['group']} ({group['target_indices']}): "
                        f"axial={fmt_metric(group.get('FWHM_axial_mm'), 4)}, "
                        f"lateral={fmt_metric(group.get('FWHM_lateral_mm'), 4)}, "
                        f"penalty={fmt_metric(group.get('PICMUS_distortion_penalty'), 1)}\n"
                    )
            else:
                file.write("No PICMUS phantom score for this mode.\n")
            file.write("\n")


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    comparison_path = resolve(args.comparison_npy)
    h5_path = resolve(args.h5_path)
    out_dir = resolve(args.out_dir)
    meta_from_h5 = read_sample_meta(h5_path, args.h5_sample_idx)
    if (
        args.phantom_mode == "in_vivo"
        or args.phantom_source == "in_vivo"
        or meta_from_h5.get("phantom_mode") == "in_vivo"
        or meta_from_h5.get("phantom_source") == "in_vivo"
    ):
        print("In-vivo scene: skipped metric export.")
        return

    comparison = np.load(comparison_path).astype(np.float64)
    x_mm, z_mm = load_grids(h5_path)
    if comparison.shape[1:] != (len(z_mm), len(x_mm)):
        raise ValueError(f"Image shape {comparison.shape[1:]} does not match H5 grids {(len(z_mm), len(x_mm))}")
    has_gt = infer_has_gt(args, h5_path, args.h5_sample_idx)
    methods = load_method_names(args, comparison_path, comparison.shape[0], has_gt)

    mode = infer_mode(args, comparison)
    source = infer_source(args)
    os.makedirs(out_dir, exist_ok=True)
    if args.phantom_path == "auto" and meta_from_h5.get("phantom_path"):
        phantom_path = resolve_existing(meta_from_h5["phantom_path"])
    else:
        phantom_path = default_phantom(mode, source) if args.phantom_path == "auto" else resolve(args.phantom_path)
    phantom = read_phantom(phantom_path)
    rois = phantom["contrast_rois"]
    speckle_rois = phantom["speckle_rois"]
    targets = phantom["resolution_targets"] if mode == "resolution_distorsion" else []
    lateral_resolution_mm = phantom.get("lateral_resolution_mm")
    axial_resolution_mm = phantom.get("axial_resolution_mm")
    gt_db = comparison[0] if has_gt else None
    if not rois and mode == "contrast_speckle" and has_gt:
        rois = auto_dark_rois(comparison[0], x_mm, z_mm, args.auto_roi_radius_mm)
    if not targets and mode == "resolution_distorsion" and has_gt:
        targets = auto_bright_targets(
            comparison[0],
            x_mm,
            z_mm,
            args.auto_target_count,
            args.auto_target_min_distance_mm,
        )

    gt_display = db_to_display(gt_db, args.dr) if has_gt else None
    rows = []
    roi_rows = []
    target_rows = []
    for method, db_img in zip(methods, comparison):
        display = db_to_display(db_img, args.dr)
        env = db_to_envelope(db_img)
        row = {"method": method}
        if has_gt and method == "GT":
            row["SSIM_vs_GT"] = 1.0
            row["PSNR_dB_vs_GT"] = float("inf")
            row["MAE_dB_vs_GT"] = 0.0
        elif has_gt:
            if structural_similarity is not None:
                row["SSIM_vs_GT"] = float(structural_similarity(gt_display, display, data_range=1.0))
            else:
                row["SSIM_vs_GT"] = local_ssim(display, gt_display, data_range=1.0)
            if peak_signal_noise_ratio is not None:
                row["PSNR_dB_vs_GT"] = float(peak_signal_noise_ratio(gt_display, display, data_range=1.0))
            else:
                row["PSNR_dB_vs_GT"] = psnr(display, gt_display, data_range=1.0)
            row["MAE_dB_vs_GT"] = float(np.mean(np.abs(np.clip(db_img, -args.dr, 0.0) - np.clip(gt_db, -args.dr, 0.0))))
        else:
            row["SSIM_vs_GT"] = np.nan
            row["PSNR_dB_vs_GT"] = np.nan
            row["MAE_dB_vs_GT"] = np.nan

        contrast_scores = []
        cr_scores = []
        cnr_scores = []
        gcnr_scores = []
        residual_scores = []
        for idx, roi in enumerate(rois):
            score = standard_contrast_score(db_img, x_mm, z_mm, roi, lateral_resolution_mm)
            metrics = contrast_roi_metrics(db_img, x_mm, z_mm, roi, lateral_resolution_mm)
            if not np.isfinite(score) and metrics is None:
                continue
            roi_row = {"method": method, "roi_idx": idx, **roi, "contrast_dB": score}
            if metrics is not None:
                roi_row.update(metrics)
                cr_scores.append(metrics["CR_dB"])
                cnr_scores.append(metrics["CNR"])
                gcnr_scores.append(metrics["gCNR"])
                residual_scores.append(metrics["cyst_residual_dB"])
            roi_rows.append(roi_row)
            contrast_scores.append(score)
        row["contrast_dB"] = mean_or_nan(contrast_scores)
        row["CR_dB"] = mean_or_nan(cr_scores)
        row["CNR"] = mean_or_nan(cnr_scores)
        row["gCNR"] = mean_or_nan(gcnr_scores)
        row["cyst_residual_dB"] = mean_or_nan(residual_scores)


        speckle_stats = []
        for idx, roi in enumerate(speckle_rois):
            stats = standard_speckle_quality(env, x_mm, z_mm, roi, lateral_resolution_mm, axial_resolution_mm)
            if stats is None:
                continue
            roi_rows.append({"method": method, "roi_idx": idx, **roi, **stats})
            speckle_stats.append(stats)
        row["speckle_pass_rate"] = mean_or_nan([s["speckle_pass"] for s in speckle_stats])
        row["speckle_KS_D"] = mean_or_nan([s["speckle_KS_D"] for s in speckle_stats])
        row["speckle_KS_p"] = mean_or_nan([s["speckle_KS_p"] for s in speckle_stats])
        row["speckle_SNR"] = mean_or_nan([s["speckle_SNR"] for s in speckle_stats])
        row["ENL"] = mean_or_nan([s["ENL"] for s in speckle_stats])


        target_stats = []
        for idx, target in enumerate(targets):
            stats = target_resolution(db_img, x_mm, z_mm, target, args.fwhm_window_mm, target_idx=idx + 1, source=source)
            if stats is None:
                continue
            target_rows.append({"method": method, "target_idx": idx + 1, **stats})
            target_stats.append(stats)
        row["FWHM_axial_mm"] = mean_or_nan([s["FWHM_axial_mm"] for s in target_stats])
        row["FWHM_lateral_mm"] = mean_or_nan([s["FWHM_lateral_mm"] for s in target_stats])
        row["PSLR_dB"] = mean_or_nan([s["PSLR_dB"] for s in target_stats])
        row["ISLR_dB"] = mean_or_nan([s["ISLR_dB"] for s in target_stats])
        row["distortion_mm"] = mean_or_nan([s["distortion_mm"] for s in target_stats])
        row["distortion_pass_rate"] = mean_or_nan([s["distortion_pass"] for s in target_stats])
        rows.append(row)

    summary_fields = [
        "method", "SSIM_vs_GT", "PSNR_dB_vs_GT", "MAE_dB_vs_GT",
        "contrast_dB", "CR_dB", "CNR", "gCNR", "cyst_residual_dB",
        "speckle_pass_rate", "speckle_KS_D", "speckle_KS_p", "speckle_SNR", "ENL",
        "FWHM_axial_mm", "FWHM_lateral_mm", "PSLR_dB", "ISLR_dB",
        "distortion_mm", "distortion_pass_rate",
    ]
    write_csv(os.path.join(out_dir, "summary_metrics.csv"), rows, summary_fields)
    contrast_group_rows = build_contrast_group_rows(roi_rows, source, len(rois), methods)
    resolution_group_rows = build_resolution_group_rows(target_rows, source, len(targets), methods)
    optional_outputs = [
        ("contrast_roi_metrics.csv", roi_rows),
        ("resolution_target_metrics.csv", target_rows),
        ("contrast_group_metrics.csv", contrast_group_rows),
        ("resolution_group_metrics.csv", resolution_group_rows),
    ]
    for name, data_rows in optional_outputs:
        path = os.path.join(out_dir, name)
        if data_rows:
            fieldnames = []
            for data_row in data_rows:
                for key in data_row.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)
            write_csv(path, data_rows, fieldnames)
        else:
            write_csv(path, [{"status": "not_applicable"}], ["status"])
    write_picmus_report(
        os.path.join(out_dir, "picmus_challenge_summary.txt"),
        mode,
        source,
        methods,
        rows,
        contrast_group_rows,
        resolution_group_rows,
        roi_rows,
    )

    meta = {
        "comparison_npy": comparison_path,
        "h5_path": h5_path,
        "h5_sample_idx": args.h5_sample_idx,
        "mode": mode,
        "phantom_source": source,
        "phantom_path": phantom_path,
        "sample_name": meta_from_h5.get("sample_name", ""),
        "has_gt": has_gt,
        "methods": methods,
        "contrast_roi_count": len(rois),
        "speckle_roi_count": len(speckle_rois),
        "resolution_target_count": len(targets),
        "contrast_groups": picmus_contrast_groups(source, len(rois)),
        "resolution_groups": picmus_resolution_groups(source, len(targets)),
        "note": "Standard metrics use the predefined phantom ROI and target definitions when available.",
    }
    with open(os.path.join(out_dir, "evaluation_meta.json"), "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2, ensure_ascii=False)

    print(f"Saved summary: {os.path.join(out_dir, 'summary_metrics.csv')}")
    if contrast_group_rows:
        print(f"Saved contrast group metrics: {os.path.join(out_dir, 'contrast_group_metrics.csv')}")
    if resolution_group_rows:
        print(f"Saved resolution group metrics: {os.path.join(out_dir, 'resolution_group_metrics.csv')}")
    print(f"Saved PICMUS-style report: {os.path.join(out_dir, 'picmus_challenge_summary.txt')}")
    print(f"Mode={mode}, source={source}, contrast_rois={len(rois)}, resolution_targets={len(targets)}")


if __name__ == "__main__":
    main()

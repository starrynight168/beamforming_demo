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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
            names = DEFAULT_METHODS[:n_panels]
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


def target_resolution(db_img, x_mm, z_mm, target, window_mm, target_idx=None):
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
    peak_offset = math.sqrt((x_mm[ix] - target["x_mm"]) ** 2 + (z_mm[iz] - target["z_mm"]) ** 2)
    distortion_pass = np.nan
    if target_idx is not None and target_idx in {1, 5, 8, 9, 14, 15, 20}:
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
        "distortion_mm": float(peak_offset),
        "distortion_pass": distortion_pass,
    }


def mean_or_nan(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else np.nan


def save_roi_plot(path, gt_display, x_mm, z_mm, rois, targets):
    fig, ax = plt.subplots(figsize=(6, 8), dpi=220)
    ax.imshow(
        gt_display,
        cmap="gray",
        vmin=0,
        vmax=1,
        extent=[x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]],
        aspect="equal",
    )
    for roi in rois:
        ax.add_patch(plt.Circle((roi["x_mm"], roi["z_mm"]), roi["diameter_mm"] / 2.0, fill=False, color="red"))
    if targets:
        ax.scatter([t["x_mm"] for t in targets], [t["z_mm"] for t in targets], marker="+", c="cyan", s=20)
    ax.set_xlabel("Lateral (mm)")
    ax.set_ylabel("Depth (mm)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_bar_plot(path, metrics_to_plot):
    import matplotlib.pyplot as plt
    import numpy as np

    if not metrics_to_plot:
        return

    n_plots = len(metrics_to_plot)
    cols = min(n_plots, 3)
    rows_grid = (n_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows_grid, cols, figsize=(5 * cols, 4 * rows_grid), dpi=150)
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    color_map = {
        "GT": "#333333",       # Dark Grey
        "DAS": "#4C72B0",      # Classic Blue
        "MV": "#55A868",       # Green
        "ESBMV": "#8172B3",    # Purple
        "GCF-MV": "#64B5CD",   # Cyan
        "CMSAW": "#FF7F0E",    # Orange
        "F-DMAS": "#C44E52",   # Red
    }

    for i, (title, key, vals, m_list) in enumerate(metrics_to_plot):
        ax = axes[i]
        colors = [color_map.get(m, "#4C72B0") for m in m_list]
        use_horizontal = len(m_list) > 6
        positions = np.arange(len(m_list))
        if use_horizontal:
            bars = ax.barh(positions, vals, color=colors, height=0.55, edgecolor='black', linewidth=0.8)
            ax.set_yticks(positions)
            ax.set_yticklabels(m_list, fontsize=8)
            ax.invert_yaxis()
        else:
            bars = ax.bar(m_list, vals, color=colors, width=0.5, edgecolor='black', linewidth=0.8)
            ax.tick_params(axis='x', labelrotation=25)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        ax.grid(axis='x' if use_horizontal else 'y', linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)

        for bar in bars:
            value = bar.get_width() if use_horizontal else bar.get_height()
            if np.isfinite(value):
                label = f'{value:.3f}' if 'SSIM' in title or 'Rate' in title or 'mm' in title else f'{value:.1f}'
                if use_horizontal:
                    ax.annotate(label,
                                xy=(value, bar.get_y() + bar.get_height() / 2),
                                xytext=(3, 0),
                                textcoords="offset points",
                                ha='left', va='center', fontsize=8, fontweight='bold')
                else:
                    ax.annotate(label,
                                xy=(bar.get_x() + bar.get_width() / 2, value),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=8, fontweight='bold')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def build_metric_plots(rows):
    methods_all = [r["method"] for r in rows]
    non_gt = [r for r in rows if r["method"] != "GT"]

    standard = []
    if any(np.isfinite(r["contrast_dB"]) for r in rows):
        standard.append(("Contrast (dB)", "contrast_dB", [r["contrast_dB"] for r in rows], methods_all))
    if any(np.isfinite(r["speckle_pass_rate"]) for r in rows):
        standard.append(("Speckle Pass Rate", "speckle_pass_rate", [r["speckle_pass_rate"] for r in rows], methods_all))
    if any(np.isfinite(r["FWHM_axial_mm"]) for r in rows):
        standard.append(("Axial FWHM (mm)", "FWHM_axial_mm", [r["FWHM_axial_mm"] for r in rows], methods_all))
        standard.append(("Lateral FWHM (mm)", "FWHM_lateral_mm", [r["FWHM_lateral_mm"] for r in rows], methods_all))
    if any(np.isfinite(r["distortion_pass_rate"]) for r in rows):
        standard.append(("Distortion Pass Rate", "distortion_pass_rate", [r["distortion_pass_rate"] for r in rows], methods_all))

    auxiliary = []
    if any(np.isfinite(r["SSIM_vs_GT"]) for r in non_gt):
        auxiliary.append(("SSIM vs GT", "SSIM_vs_GT", [r["SSIM_vs_GT"] for r in non_gt], [r["method"] for r in non_gt]))
    if any(np.isfinite(r["PSNR_dB_vs_GT"]) for r in non_gt):
        auxiliary.append(("PSNR vs GT (dB)", "PSNR_dB_vs_GT", [r["PSNR_dB_vs_GT"] for r in non_gt], [r["method"] for r in non_gt]))
    if any(np.isfinite(r["MAE_dB_vs_GT"]) for r in non_gt):
        auxiliary.append(("MAE vs GT (dB)", "MAE_dB_vs_GT", [r["MAE_dB_vs_GT"] for r in non_gt], [r["method"] for r in non_gt]))
    if any(np.isfinite(r["black_pixel_ratio"]) for r in non_gt):
        auxiliary.append(("Black Pixel Ratio (%)", "black_pixel_ratio", [100.0 * r["black_pixel_ratio"] for r in non_gt], [r["method"] for r in non_gt]))
    if any(np.isfinite(r["mean_raw_dB"]) for r in rows):
        auxiliary.append(("Mean Raw (dB)", "mean_raw_dB", [r["mean_raw_dB"] for r in rows], methods_all))
    if any(np.isfinite(r["std_display_dB"]) for r in rows):
        auxiliary.append(("Display Std (dB)", "std_display_dB", [r["std_display_dB"] for r in rows], methods_all))

    return standard, auxiliary


def save_lateral_profile_plot(out_dir, comparison, x_mm, z_mm, rois, targets, methods):
    import matplotlib.pyplot as plt
    import numpy as np
    import math

    color_map = {
        "GT": "#333333",       # Dark Grey
        "DAS": "#4C72B0",      # Classic Blue
        "MV": "#55A868",       # Green
        "ESBMV": "#8172B3",    # Purple
        "F-DMAS": "#C44E52",   # Red
    }
    linestyle_map = {
        "GT": "--",
        "DAS": "-",
        "MV": "-",
        "ESBMV": "-",
        "F-DMAS": "-",
    }

    # 1. Plot Cyst Profile if contrast ROIs exist
    if rois:
        target_cyst = min(rois, key=lambda c: (c["x_mm"]-0.0)**2 + (c["z_mm"]-25.0)**2)
        cyst_x = target_cyst["x_mm"]
        cyst_z = target_cyst["z_mm"]
        cyst_d = target_cyst["diameter_mm"]

        iz = np.argmin(np.abs(z_mm - cyst_z))
        actual_z = z_mm[iz]

        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        for i, method in enumerate(methods):
            profile = comparison[i, iz, :]
            ax.plot(x_mm, profile, label=method, color=color_map.get(method, "#4C72B0"),
                    linestyle=linestyle_map.get(method, "-"), linewidth=1.5 if method != "GT" else 1.2)

        ax.axvline(cyst_x - cyst_d/2, color='grey', linestyle=':', alpha=0.7, label='Cyst Boundary')
        ax.axvline(cyst_x + cyst_d/2, color='grey', linestyle=':')

        ax.set_title(f"1D Lateral Cyst Profile (Depth z = {actual_z:.1f} mm)", fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel("Lateral coordinate (mm)", fontsize=10)
        ax.set_ylabel("Amplitude (dB)", fontsize=10)
        ax.set_ylim(-65, 5)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right", fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "cyst_profile.png"), bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved cyst profile plot: {os.path.join(out_dir, 'cyst_profile.png')}")

    # 2. Plot Point Profile if resolution targets exist
    if targets:
        target_point = min(targets, key=lambda t: (t["x_mm"]-0.0)**2 + (t["z_mm"]-25.0)**2)
        pt_x = target_point["x_mm"]
        pt_z = target_point["z_mm"]

        iz = np.argmin(np.abs(z_mm - pt_z))
        actual_z = z_mm[iz]

        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        for i, method in enumerate(methods):
            profile = comparison[i, iz, :]
            ax.plot(x_mm, profile, label=method, color=color_map.get(method, "#4C72B0"),
                    linestyle=linestyle_map.get(method, "-"), linewidth=1.5 if method != "GT" else 1.2)

        ax.axvline(pt_x, color='grey', linestyle=':', alpha=0.7, label='Point Target Center')

        ax.set_title(f"1D Lateral Point Profile (Depth z = {actual_z:.1f} mm)", fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel("Lateral coordinate (mm)", fontsize=10)
        ax.set_ylabel("Amplitude (dB)", fontsize=10)
        ax.set_ylim(-65, 5)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right", fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "point_profile.png"), bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved point target profile plot: {os.path.join(out_dir, 'point_profile.png')}")


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
    os.makedirs(out_dir, exist_ok=True)

    comparison = np.load(comparison_path).astype(np.float64)
    x_mm, z_mm = load_grids(h5_path)
    if comparison.shape[1:] != (len(z_mm), len(x_mm)):
        raise ValueError(f"Image shape {comparison.shape[1:]} does not match H5 grids {(len(z_mm), len(x_mm))}")
    has_gt = infer_has_gt(args, h5_path, args.h5_sample_idx)
    methods = load_method_names(args, comparison_path, comparison.shape[0], has_gt)

    mode = infer_mode(args, comparison)
    source = infer_source(args)
    meta_from_h5 = read_sample_meta(h5_path, args.h5_sample_idx)
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
        clipped_db = np.clip(db_img, -args.dr, 0.0)
        display = db_to_display(db_img, args.dr)
        env = db_to_envelope(db_img)
        row = {"method": method}
        row["mean_raw_dB"] = float(np.mean(db_img))
        row["median_raw_dB"] = float(np.median(db_img))
        row["std_raw_dB"] = float(np.std(db_img))
        row["dynamic_range_raw_dB"] = float(np.percentile(db_img, 99.9) - np.percentile(db_img, 0.1))
        row["mean_display_dB"] = float(np.mean(clipped_db))
        row["median_display_dB"] = float(np.median(clipped_db))
        row["std_display_dB"] = float(np.std(clipped_db))
        row["black_pixel_ratio"] = float(np.mean(db_img <= -args.dr))
        row["white_pixel_ratio"] = float(np.mean(db_img >= 0.0))
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
        for idx, roi in enumerate(rois):
            score = standard_contrast_score(db_img, x_mm, z_mm, roi, lateral_resolution_mm)
            if not np.isfinite(score):
                continue
            roi_row = {"method": method, "roi_idx": idx, **roi, "contrast_dB": score}
            roi_rows.append(roi_row)
            contrast_scores.append(score)
        row["contrast_dB"] = mean_or_nan(contrast_scores)

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

        target_stats = []
        for idx, target in enumerate(targets):
            stats = target_resolution(db_img, x_mm, z_mm, target, args.fwhm_window_mm, target_idx=idx + 1)
            if stats is None:
                continue
            target_rows.append({"method": method, "target_idx": idx + 1, **stats})
            target_stats.append(stats)
        row["FWHM_axial_mm"] = mean_or_nan([s["FWHM_axial_mm"] for s in target_stats])
        row["FWHM_lateral_mm"] = mean_or_nan([s["FWHM_lateral_mm"] for s in target_stats])
        row["distortion_mm"] = mean_or_nan([s["distortion_mm"] for s in target_stats])
        row["distortion_pass_rate"] = mean_or_nan([s["distortion_pass"] for s in target_stats])
        rows.append(row)

    summary_fields = [
        "method", "SSIM_vs_GT", "PSNR_dB_vs_GT", "MAE_dB_vs_GT",
        "contrast_dB", "speckle_pass_rate", "speckle_KS_D", "speckle_KS_p",
        "FWHM_axial_mm", "FWHM_lateral_mm", "distortion_mm", "distortion_pass_rate",
        "mean_raw_dB", "median_raw_dB", "std_raw_dB", "dynamic_range_raw_dB",
        "mean_display_dB", "median_display_dB", "std_display_dB",
        "black_pixel_ratio", "white_pixel_ratio",
    ]
    write_csv(os.path.join(out_dir, "summary_metrics.csv"), rows, summary_fields)
    optional_outputs = [
        ("contrast_roi_metrics.csv", roi_rows),
        ("resolution_target_metrics.csv", target_rows),
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
    if has_gt:
        save_roi_plot(os.path.join(out_dir, "roi_targets.png"), gt_display, x_mm, z_mm, rois, targets)
    old_plot = os.path.join(out_dir, "metrics_" + "comparison.png")
    if os.path.exists(old_plot):
        os.remove(old_plot)
    standard_metrics, auxiliary_metrics = build_metric_plots(rows)
    save_bar_plot(os.path.join(out_dir, "standard_metrics.png"), standard_metrics)
    save_bar_plot(os.path.join(out_dir, "auxiliary_metrics.png"), auxiliary_metrics)
    save_lateral_profile_plot(out_dir, comparison, x_mm, z_mm, rois, targets, methods)

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
        "note": "Standard metrics use the predefined phantom ROI and target definitions when available.",
    }
    with open(os.path.join(out_dir, "evaluation_meta.json"), "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2, ensure_ascii=False)

    print(f"Saved summary: {os.path.join(out_dir, 'summary_metrics.csv')}")
    if standard_metrics:
        print(f"Saved standard metrics plot: {os.path.join(out_dir, 'standard_metrics.png')}")
    if auxiliary_metrics:
        print(f"Saved auxiliary metrics plot: {os.path.join(out_dir, 'auxiliary_metrics.png')}")
    if has_gt:
        print(f"Saved ROI/target plot: {os.path.join(out_dir, 'roi_targets.png')}")
    print(f"Mode={mode}, source={source}, contrast_rois={len(rois)}, resolution_targets={len(targets)}")


if __name__ == "__main__":
    main()




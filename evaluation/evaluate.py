"""Evaluate run_one.py comparison output.

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
import yaml
from scipy.ndimage import convolve, gaussian_filter
from scipy.stats import kstest

EPSILON = 1e-9
PASS_THRESHOLD = 0.5
MIN_PROFILE_POINTS = 2
MIN_SIDE_LOBE_POINTS = 5
MIN_ROI_SAMPLES = 8
SIMULATION_RESOLUTION_TARGETS = 20
EXPERIMENT_RESOLUTION_TARGETS = 7
SIMULATION_CONTRAST_ROIS = 9
EXPERIMENT_CONTRAST_ROIS = 2
IMAGE_STACK_DIMENSIONS = 3
SPECKLE_PASS_ALPHA = 0.05
PICMUS_OFFICIAL_Z_CORRECTION_MM = 0.2

try:
    from skimage.metrics import peak_signal_noise_ratio
except ImportError:
    peak_signal_noise_ratio = None


CONTROLLED_SCENES = frozenset(
    {
        "simulation_contrast_speckle",
        "simulation_resolution_distorsion",
        "experiments_contrast_speckle",
        "experiments_resolution_distorsion",
    }
)
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
PICMUS_ROOT = os.path.join(PROJECT_ROOT, "data", "PICMUS")


def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser(
        description="Evaluate run_one.py comparison output.",
    )
    parser.add_argument(
        "--comparison_npy",
        default=os.path.join("results", "comparison.npy"),
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="逗号分隔算法名,不含GT;默认从run_params.json读取",
    )
    parser.add_argument(
        "--method_labels",
        default=None,
        help="逗号分隔显示名,不含GT;默认使用run_params.json或算法名大写",
    )
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
        choices=["auto", "contrast_speckle", "resolution_distorsion"],
        default="auto",
    )
    parser.add_argument(
        "--phantom_source",
        choices=["auto", "simulation", "experiments"],
        default="auto",
        help="Phantom source; auto uses experiments for validation H5 files and simulation otherwise.",
    )
    parser.add_argument("--auto_roi_radius_mm", type=float, default=2.0)
    parser.add_argument("--auto_target_count", type=int, default=20)
    parser.add_argument("--auto_target_min_distance_mm", type=float, default=3.0)
    parser.add_argument("--fwhm_window_mm", type=float, default=1.8)
    return parser.parse_args()


def load_method_names(args, comparison_path, n_panels, has_gt):
    """Load method names."""
    if args.methods:
        methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
        names = (
            [m.strip() for m in args.method_labels.split(",") if m.strip()]
            if args.method_labels
            else [m.upper() for m in methods]
        )
    else:
        params_path = os.path.join(os.path.dirname(comparison_path), "run_params.json")
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"缺少当前格式的 run_params.json: {params_path}")
        with open(params_path, encoding="utf-8") as f:
            params = json.load(f)
        if not isinstance(params, dict):
            raise ValueError(f"run_params.json 顶层必须是对象: {params_path}")
        model_map = params.get("models")
        if not isinstance(model_map, dict) or not model_map:
            raise ValueError(f"run_params.json 缺少当前格式的 models 字典: {params_path}")
        model_names = list(model_map)
        evaluation = params.get("evaluation")
        if evaluation is None:
            evaluation = {}
        if not isinstance(evaluation, dict):
            raise ValueError(f"run_params.json evaluation 必须是对象: {params_path}")
        skip_baselines = evaluation.get("skip_baselines", False)
        if not isinstance(skip_baselines, bool):
            raise ValueError(f"run_params.json evaluation.skip_baselines 必须是布尔值: {params_path}")
        baseline_names = [] if skip_baselines else ["DAS", "MV"]
        names = [*baseline_names, *model_names]
        n_methods = n_panels - 1 if has_gt else n_panels
        if len(names) != n_methods:
            raise ValueError(
                f"run_params.json 中的模型/基线数量与 comparison 不一致: "
                f"expected={n_methods}, actual={len(names)}"
            )
    if has_gt:
        names = ["GT", *names]
    if len(names) != n_panels:
        raise ValueError(
            f"Method count {len(names)} does not match stacked image count {n_panels}",
        )
    if len(names) != len(set(names)):
        raise ValueError(f"Method labels must be unique: {names}")
    return names


def resolve(path):
    """Execute resolve."""
    if path in ("auto", "none", None):
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def resolve_existing(relative_path):
    """Execute resolve existing."""
    if not relative_path:
        return "none"
    if os.path.isabs(relative_path):
        return relative_path if os.path.exists(relative_path) else "none"
    candidates = [
        os.path.join(PICMUS_ROOT, relative_path),
        os.path.join(PROJECT_ROOT, "data", relative_path),
        os.path.join(PROJECT_ROOT, relative_path),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return "none"


def h5_embedded_config(hf):
    """Execute h5 embedded config."""
    value = hf["config_yaml"][()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        config = yaml.safe_load(value) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid config_yaml: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("config_yaml top level must be a mapping")
    return config


def read_sample_meta(h5_path, sample_idx):
    """Read sample meta."""
    with h5py.File(h5_path, "r") as hf:
        n_samples = int(hf["all_multi_I"].shape[0])
        if not 0 <= sample_idx < n_samples:
            raise IndexError(f"sample index {sample_idx} is outside [0, {n_samples - 1}]")
        has_gt = "all_envdb_norm" in hf and 0 <= sample_idx < hf["all_envdb_norm"].shape[0]
        config = h5_embedded_config(hf)
        dataset_meta = config.get("dataset", {})
        if dataset_meta is None:
            dataset_meta = {}
        if not isinstance(dataset_meta, dict):
            raise ValueError("config_yaml.dataset must be a mapping")
        source_samples = config.get("source_samples", [])
        if isinstance(source_samples, list):
            if not 0 <= sample_idx < len(source_samples):
                raise IndexError(f"source_samples has no sample index {sample_idx}")
            sample_config = source_samples[sample_idx]
            if not isinstance(sample_config, dict):
                raise ValueError(f"source_samples[{sample_idx}] must be a mapping")
            return {
                "has_gt": has_gt,
                "sample_name": str(sample_config.get("id") or sample_config.get("acquisition_id") or ""),
                "phantom_mode": str(sample_config.get("phantom_mode") or dataset_meta.get("phantom_mode") or ""),
                "phantom_source": str(sample_config.get("phantom_source") or dataset_meta.get("phantom_source") or ""),
                "phantom_path": str(sample_config.get("phantom_path") or ""),
            }
        if (
            isinstance(source_samples, dict)
            and source_samples.get("encoding") == "root_labels_with_path_template"
            and source_samples.get("count") == hf["all_multi_I"].shape[0]
        ):
            id_dataset = str(source_samples.get("acquisition_id_dataset", "/acquisition_id"))
            if id_dataset not in hf or not 0 <= sample_idx < hf[id_dataset].shape[0]:
                raise IndexError(f"{id_dataset} has no sample index {sample_idx}")
            sample_name = hf[id_dataset][sample_idx]
            if isinstance(sample_name, bytes):
                sample_name = sample_name.decode("utf-8")
            phantom_mode = str(source_samples.get("phantom_mode") or dataset_meta.get("phantom_mode", ""))
            phantom_source = str(source_samples.get("phantom_source") or dataset_meta.get("phantom_source", ""))
            return {
                "has_gt": has_gt,
                "sample_name": str(sample_name),
                "phantom_mode": phantom_mode,
                "phantom_source": phantom_source,
                "phantom_path": "",
            }
        raise IndexError(f"source_samples has no sample index {sample_idx}")


def picmus_distortion_z_offset_mm(h5_path, source):
    """Return the distortion-mask depth correction for the packed dataset."""
    if source != "simulation":
        return 0.0
    with h5py.File(h5_path, "r") as hf:
        if "config_yaml" not in hf:
            return PICMUS_OFFICIAL_Z_CORRECTION_MM
        config = h5_embedded_config(hf)
    evaluation = config.get("evaluation")
    if evaluation is None:
        evaluation = {}
    if not isinstance(evaluation, dict):
        raise ValueError("config_yaml.evaluation must be a mapping")
    configured = evaluation.get("picmus_distortion_z_offset_mm")
    if configured is not None:
        value = float(configured)
        if not np.isfinite(value):
            raise ValueError("evaluation.picmus_distortion_z_offset_mm 必须是有限数")
        return value
    provenance = config.get("provenance")
    if provenance is None:
        provenance = {}
    if not isinstance(provenance, dict):
        raise ValueError("config_yaml.provenance must be a mapping")
    source_kind = str(provenance.get("source_kind") or "").strip().lower()
    if source_kind == "synthetic_rf_from_picmus_phantom":
        return 0.0
    return PICMUS_OFFICIAL_Z_CORRECTION_MM


def load_grids(h5_path):
    """Load grids."""
    with h5py.File(h5_path, "r") as hf:
        x_mm = hf["x_grid"][:].astype(float) * 1000.0
        z_mm = hf["z_grid"][:].astype(float) * 1000.0
    for name, grid in (("x_grid", x_mm), ("z_grid", z_mm)):
        if grid.ndim != 1 or grid.size < 2 or not np.all(np.isfinite(grid)) or not np.all(np.diff(grid) > 0):
            raise ValueError(f"{name} must be a finite, strictly increasing 1D grid with at least two points")
    return x_mm, z_mm


def db_to_display(db, dr):
    """Execute db to display."""
    return np.clip((db + dr) / dr, 0.0, 1.0)


def db_to_envelope(db):
    """Execute db to envelope."""
    return 10.0 ** (db / 20.0)


def psnr(image, reference, data_range=1.0):
    """Execute psnr."""
    mse = float(np.mean((image - reference) ** 2))
    if mse == 0.0:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def local_ssim(image, reference, data_range=1.0, sigma=1.5):
    """Gaussian-window SSIM with the standard sigma=1.5 population covariance."""
    image = image.astype(np.float64)
    reference = reference.astype(np.float64)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    truncate = 3.5
    ux = gaussian_filter(image, sigma=sigma, truncate=truncate, mode="reflect")
    uy = gaussian_filter(reference, sigma=sigma, truncate=truncate, mode="reflect")
    uxx = gaussian_filter(image * image, sigma=sigma, truncate=truncate, mode="reflect")
    uyy = gaussian_filter(reference * reference, sigma=sigma, truncate=truncate, mode="reflect")
    uxy = gaussian_filter(image * reference, sigma=sigma, truncate=truncate, mode="reflect")
    vx = uxx - ux * ux
    vy = uyy - uy * uy
    vxy = uxy - ux * uy
    score = ((2.0 * ux * uy + c1) * (2.0 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
    pad = int(truncate * sigma + 0.5)
    if min(score.shape) > 2 * pad:
        score = score[pad:-pad, pad:-pad]
    return float(np.mean(score))


def disk_kernel(radius_mm, dx_mm, dz_mm):
    """Execute disk kernel."""
    rx = max(int(np.ceil(radius_mm / abs(dx_mm))), 1)
    rz = max(int(np.ceil(radius_mm / abs(dz_mm))), 1)
    yy, xx = np.mgrid[-rz : rz + 1, -rx : rx + 1]
    kernel = ((xx * dx_mm) ** 2 + (yy * dz_mm) ** 2 <= radius_mm**2).astype(float)
    return kernel / max(kernel.sum(), 1.0)


def auto_dark_rois(gt_db, x_mm, z_mm, radius_mm):
    """Execute auto dark rois."""
    gt_env = db_to_envelope(gt_db)
    kernel = disk_kernel(radius_mm, x_mm[1] - x_mm[0], z_mm[1] - z_mm[0])
    local_mean = convolve(
        gt_env,
        kernel,
        mode="constant",
        cval=float(np.nanmax(gt_env)),
    )
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
    return [
        {
            "x_mm": float(x_mm[ix]),
            "z_mm": float(z_mm[iz]),
            "diameter_mm": 2.0 * radius_mm,
            "source": "auto_gt_dark",
        },
    ]


def auto_bright_targets(gt_db, x_mm, z_mm, max_targets, min_distance_mm):
    """Execute auto bright targets."""
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
        too_close = any((x - t["x_mm"]) ** 2 + (z - t["z_mm"]) ** 2 < min_distance_mm**2 for t in targets)
        if not too_close:
            targets.append({"x_mm": x, "z_mm": z, "source": "auto_gt_bright"})
    targets.sort(key=lambda t: (t["z_mm"], t["x_mm"]))
    return targets


def read_phantom(phantom_path):
    """Read phantom."""
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
            for x, z, d in zip(xs, zs, ds, strict=True):
                if d > 0:
                    out["contrast_rois"].append(
                        {
                            "x_mm": float(x),
                            "z_mm": float(z),
                            "diameter_mm": float(d),
                            "source": "phantom",
                        },
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
            for x, z, tx, tz in zip(xs, zs, txs, tzs, strict=True):
                out["speckle_rois"].append(
                    {
                        "x_mm": float(x),
                        "z_mm": float(z),
                        "psf_time_x": float(tx),
                        "psf_time_z": float(tz),
                        "source": "phantom",
                    },
                )
        if f"{base}/phantom_xPts" in hf:
            xs = hf[f"{base}/phantom_xPts"][:].astype(float) * 1000.0
            zs = hf[f"{base}/phantom_zPts"][:].astype(float) * 1000.0
            for x, z in zip(xs, zs, strict=True):
                if (
                    np.isfinite(x)
                    and np.isfinite(z)
                    and (abs(x) > EPSILON or abs(z) > EPSILON)
                ):
                    out["resolution_targets"].append(
                        {"x_mm": float(x), "z_mm": float(z), "source": "phantom"},
                    )
    return out


def infer_source(args, meta=None):
    """Execute infer source."""
    if args.phantom_source != "auto":
        return args.phantom_source
    if meta is None:
        meta = read_sample_meta(resolve(args.h5_path), args.h5_sample_idx)
    if meta.get("phantom_source"):
        return meta["phantom_source"]
    h5_name = os.path.basename(resolve(args.h5_path)).lower()
    if "val" in h5_name or "valid" in h5_name:
        return "experiments"
    return "simulation"


def default_phantom(mode, source):
    """Execute default phantom."""
    if mode == "in_vivo" or source == "in_vivo":
        return "none"
    if mode == "resolution_distorsion":
        prefix = "expe" if source == "experiments" else "simu"
        rel = os.path.join(
            "database",
            source,
            "resolution_distorsion",
            f"resolution_distorsion_{prefix}_phantom.hdf5",
        )
    else:
        prefix = "expe" if source == "experiments" else "simu"
        rel = os.path.join(
            "database",
            source,
            "contrast_speckle",
            f"contrast_speckle_{prefix}_phantom.hdf5",
        )
    return resolve_existing(rel)


def infer_mode(args, meta=None):
    """Execute infer mode."""
    if args.phantom_mode != "auto":
        return args.phantom_mode
    if meta is None:
        meta = read_sample_meta(resolve(args.h5_path), args.h5_sample_idx)
    if meta.get("phantom_mode"):
        return meta["phantom_mode"]
    if "resolution" in os.path.basename(args.comparison_npy).lower():
        return "resolution_distorsion"
    if args.h5_sample_idx == 1:
        return "resolution_distorsion"
    return "contrast_speckle"


def standard_contrast_score(
    db_img,
    x_mm,
    z_mm,
    roi,
    lateral_resolution_mm,
    padding=1.0,
):
    """Execute standard contrast score."""
    masks = _contrast_roi_masks(x_mm, z_mm, roi, lateral_resolution_mm, padding)
    if masks is None:
        return np.nan
    inside_mask, outside_mask = masks
    inside = db_img[inside_mask]
    outside = db_img[outside_mask]
    if inside.size < MIN_ROI_SAMPLES or outside.size < MIN_ROI_SAMPLES:
        return np.nan
    # Match MATLAB var() in the official PICMUS evaluator (sample variance, N-1).
    denom = math.sqrt(
        (float(np.var(inside, ddof=1)) + float(np.var(outside, ddof=1))) / 2.0,
    )
    if denom <= 0:
        return np.nan
    ratio = abs(float(np.mean(inside)) - float(np.mean(outside))) / denom
    if ratio <= 0:
        return np.nan
    value = 20.0 * math.log10(ratio)
    return float(round(value * 10.0) / 10.0)


def contrast_roi_metrics(db_img, x_mm, z_mm, roi, lateral_resolution_mm, padding=1.0):
    """Execute contrast roi metrics."""
    masks = _contrast_roi_masks(x_mm, z_mm, roi, lateral_resolution_mm, padding)
    if masks is None:
        return None
    inside_mask, outside_mask = masks
    inside_db = db_img[inside_mask]
    outside_db = db_img[outside_mask]
    inside_db = inside_db[np.isfinite(inside_db)]
    outside_db = outside_db[np.isfinite(outside_db)]
    if inside_db.size < MIN_ROI_SAMPLES or outside_db.size < MIN_ROI_SAMPLES:
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

    if not np.isfinite(inside_env).all() or not np.isfinite(outside_env).all():
        gcnr = np.nan
    else:
        hist_in, bins = np.histogram(
            inside_env,
            bins=256,
            range=(0.0, 1.0),
            density=False,
        )
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


def _contrast_roi_masks(x_mm, z_mm, roi, lateral_resolution_mm, padding):
    if lateral_resolution_mm is None or not np.isfinite(lateral_resolution_mm) or lateral_resolution_mm <= 0:
        return None
    radius = roi["diameter_mm"] / 2.0
    inner_radius = radius - padding * lateral_resolution_mm
    outer_inner_radius = radius + padding * lateral_resolution_mm
    outer_radius = 1.2 * math.sqrt(inner_radius**2 + outer_inner_radius**2)
    if inner_radius <= 0 or outer_radius <= outer_inner_radius:
        return None
    distance_squared = (
        (x_mm[None, :] - roi["x_mm"]) ** 2
        + (z_mm[:, None] - roi["z_mm"]) ** 2
    )
    return (
        distance_squared <= inner_radius**2,
        (distance_squared >= outer_inner_radius**2)
        & (distance_squared <= outer_radius**2),
    )


def standard_speckle_quality(
    env,
    x_mm,
    z_mm,
    roi,
    lateral_resolution_mm,
    axial_resolution_mm,
):
    """Execute standard speckle quality."""
    if (
        lateral_resolution_mm is None
        or axial_resolution_mm is None
        or not np.isfinite(lateral_resolution_mm)
        or not np.isfinite(axial_resolution_mm)
        or lateral_resolution_mm <= 0
        or axial_resolution_mm <= 0
    ):
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
    if sample.size < MIN_ROI_SAMPLES:
        return None
    rayleigh_var = float(np.sum(sample**2) / (2.0 * sample.size))
    scale = math.sqrt(max(rayleigh_var, 1e-24))
    ks = kstest(sample, "rayleigh", args=(0.0, scale))
    return {
        "speckle_pass": 1.0 if ks.pvalue >= SPECKLE_PASS_ALPHA else 0.0,
        "speckle_KS_D": float(ks.statistic),
        "speckle_KS_p": float(ks.pvalue),
        "speckle_SNR": float(np.mean(sample) / (np.std(sample) + 1e-12)),
        "ENL": float((np.mean(sample) ** 2) / (np.var(sample) + 1e-24)),
    }


def compute_6db_resolution(coord, profile_db):
    """Reproduce PICMUS ``Compute_6dB_Resolution.m`` exactly.

    The reference code linearly resamples the whole profile to ``10 * N``
    points, then measures the span from the first to the last sample whose
    value is at least 6 dB below the interpolated maximum.
    """
    profile = np.asarray(profile_db, dtype=np.float64)
    coord = np.asarray(coord, dtype=np.float64)
    if (
        profile.size < MIN_PROFILE_POINTS
        or coord.size != profile.size
        or not np.all(np.isfinite(coord))
        or not np.any(np.isfinite(profile))
    ):
        return np.nan
    finite = np.isfinite(profile)
    if not np.all(finite):
        coord = coord[finite]
        profile = profile[finite]
    if profile.size < MIN_PROFILE_POINTS:
        return np.nan
    interp_coord = np.linspace(coord[0], coord[-1], profile.size * 10)
    interp_profile = np.interp(interp_coord, coord, profile)
    above = np.flatnonzero(interp_profile >= np.max(interp_profile) - 6.0)
    if above.size == 0:
        return np.nan
    return float(interp_coord[above[-1]] - interp_coord[above[0]])


def profile_sidelobe_metrics(profile_db):
    """Execute profile sidelobe metrics."""
    profile = np.asarray(profile_db, dtype=np.float64)
    profile = profile[np.isfinite(profile)]
    if profile.size < MIN_SIDE_LOBE_POINTS:
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

    main = profile[left : right + 1]
    side = np.concatenate([profile[:left], profile[right + 1 :]])
    if main.size == 0 or side.size == 0:
        return np.nan, np.nan

    peak_env = float(np.max(db_to_envelope(main)))
    side_env = db_to_envelope(side)
    main_env = db_to_envelope(main)
    if peak_env <= 0:
        return np.nan, np.nan

    pslr = 20.0 * math.log10((float(np.max(side_env)) + 1e-12) / (peak_env + 1e-12))
    islr = 10.0 * math.log10(
        (float(np.sum(side_env**2)) + 1e-24) / (float(np.sum(main_env**2)) + 1e-24),
    )
    return float(pslr), float(islr)


def build_square_label_map(x_mm, z_mm, targets, half_width_mm, z_offset_mm=0.0):
    """Build the additive, one-based target mask used by PICMUS MATLAB code."""
    labels = np.zeros((len(z_mm), len(x_mm)), dtype=np.int32)
    for target_idx, target in enumerate(targets, 1):
        x_mask = np.abs(x_mm - target["x_mm"]) < half_width_mm
        z_mask = np.abs(z_mm - (target["z_mm"] + z_offset_mm)) < half_width_mm
        labels[np.ix_(z_mask, x_mask)] += target_idx
    return labels


def target_resolution(
    db_img,
    x_mm,
    z_mm,
    target,
    window_mm,
    target_idx=None,
    label_map=None,
):
    # The official implementation first adds all numbered ROIs into one label
    # image. Thus overlaps are intentionally excluded from each individual ROI.
    """Execute target resolution."""
    if target_idx is None:
        raise ValueError("target_idx is required for PICMUS resolution evaluation")
    if label_map is None:
        label_map = build_square_label_map(x_mm, z_mm, [target], window_mm)
        target_label = 1
    else:
        target_label = target_idx
    target_mask = label_map == target_label
    z_ids, x_ids = np.where(target_mask)
    if z_ids.size == 0:
        return None
    masked = np.full_like(db_img, np.nanmin(db_img), dtype=np.float64)
    masked[target_mask] = db_img[target_mask]
    if not np.any(np.isfinite(masked[target_mask])):
        return None
    iz, ix = np.unravel_index(np.nanargmax(masked), masked.shape)
    z_slice = slice(int(z_ids.min()), int(z_ids.max()) + 1)
    x_slice = slice(int(x_ids.min()), int(x_ids.max()) + 1)
    axial_profile = masked[z_slice, ix]
    lateral_profile = masked[iz, x_slice]
    axial = compute_6db_resolution(z_mm[z_slice], axial_profile)
    lateral = compute_6db_resolution(x_mm[x_slice], lateral_profile)
    pslr_axial_db, islr_axial_db = profile_sidelobe_metrics(axial_profile)
    pslr_lateral_db, islr_lateral_db = profile_sidelobe_metrics(lateral_profile)
    pslr_db = mean_or_nan([pslr_axial_db, pslr_lateral_db])
    islr_db = mean_or_nan([islr_axial_db, islr_lateral_db])
    peak_offset = math.sqrt(
        (x_mm[ix] - target["x_mm"]) ** 2 + (z_mm[iz] - target["z_mm"]) ** 2,
    )
    return {
        "target_x_mm": target["x_mm"],
        "target_z_mm": target["z_mm"],
        "peak_x_mm": float(x_mm[ix]),
        "peak_z_mm": float(z_mm[iz]),
        "FWHM_axial_mm": axial,
        "FWHM_lateral_mm": lateral,
        "pslr_axial_db": float(pslr_axial_db),
        "pslr_lateral_db": float(pslr_lateral_db),
        "pslr_db": float(pslr_db),
        "islr_axial_db": float(islr_axial_db),
        "islr_lateral_db": float(islr_lateral_db),
        "islr_db": float(islr_db),
        "distortion_mm": float(peak_offset),
    }


def picmus_distortion_pass(
    db_img,
    target_idx,
    source,
    roi_label_map=None,
    inside_label_map=None,
):
    """Reproduce the official simulated PICMUS distortion test."""
    if source != "simulation" or target_idx not in {1, 5, 8, 9, 14, 15, 20}:
        return np.nan

    if roi_label_map is None or inside_label_map is None:
        return np.nan
    target_mask = roi_label_map == target_idx
    if not np.any(target_mask):
        return np.nan
    masked = np.full_like(db_img, np.nanmin(db_img), dtype=np.float64)
    masked[target_mask] = db_img[target_mask]
    peak = np.unravel_index(np.nanargmax(masked), masked.shape)
    return 1.0 if inside_label_map[peak] == target_idx else 0.0


def mean_or_nan(values):
    """Execute mean or nan."""
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else np.nan


def picmus_resolution_groups(source, target_count):
    """Return the official PICMUS resolution target groupings."""
    if source == "simulation" and target_count >= SIMULATION_RESOLUTION_TARGETS:
        return [
            ("vertical_targets", list(range(1, 9))),
            ("horizontal_targets_2cm", [9, 10, 11, 3, 12, 13, 14]),
            ("horizontal_targets_4cm", [15, 16, 17, 7, 18, 19, 20]),
        ]
    if source == "experiments" and target_count >= EXPERIMENT_RESOLUTION_TARGETS:
        return [
            ("vertical_targets", list(range(1, 6))),
            ("horizontal_targets_near_4cm", [6, 4, 7]),
        ]
    return [("all_targets", list(range(1, target_count + 1)))] if target_count else []


def picmus_contrast_groups(source, roi_count):
    """Execute picmus contrast groups."""
    if source == "simulation" and roi_count >= SIMULATION_CONTRAST_ROIS:
        return [
            ("left_column", [4, 5, 6]),
            ("middle_column", [1, 2, 3]),
            ("right_column", [7, 8, 9]),
        ]
    if source == "experiments" and roi_count >= EXPERIMENT_CONTRAST_ROIS:
        return [("middle_column", [1, 2])]
    return [("all_cysts", list(range(1, roi_count + 1)))] if roi_count else []


def rows_for_method_index(rows, method, index_key):
    """Execute rows for method index."""
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
    """Execute speckle penalty for method."""
    passes = [
        row["speckle_pass"]
        for row in roi_rows
        if row.get("method") == method and "speckle_pass" in row and np.isfinite(row["speckle_pass"])
    ]
    if not passes:
        return np.nan
    return -40.0 if any(value < PASS_THRESHOLD for value in passes) else 0.0


def distortion_penalty_for_method(target_rows, method, source):
    """Return the official PICMUS simulated distortion penalty."""
    if source != "simulation":
        return np.nan
    passes = [
        row["distortion_pass"]
        for row in target_rows
        if row.get("method") == method
        and "distortion_pass" in row
        and np.isfinite(row["distortion_pass"])
    ]
    if not passes:
        return np.nan
    return -40.0 if any(value < PASS_THRESHOLD for value in passes) else 0.0


def build_resolution_group_rows(target_rows, source, target_count, methods):
    """Build rows using the official PICMUS target group order."""
    groups = picmus_resolution_groups(source, target_count)
    rows = []
    for method in methods:
        indexed = rows_for_method_index(target_rows, method, "target_idx")
        for group_name, indices in groups:
            selected = [indexed[i] for i in indices if i in indexed]
            if not selected:
                continue
            pass_values = [
                row["distortion_pass"]
                for row in selected
                if np.isfinite(row.get("distortion_pass", np.nan))
            ]
            if source == "simulation" and pass_values:
                penalty = -40.0 if any(value < PASS_THRESHOLD for value in pass_values) else 0.0
                pass_rate = mean_or_nan(pass_values)
            else:
                penalty = np.nan
                pass_rate = np.nan
            rows.append(
                {
                    "method": method,
                    "group": group_name,
                    "target_indices": " ".join(str(i) for i in indices),
                    "n_targets": len(selected),
                    "FWHM_axial_mm": mean_or_nan(
                        [row.get("FWHM_axial_mm") for row in selected],
                    ),
                    "FWHM_lateral_mm": mean_or_nan(
                        [row.get("FWHM_lateral_mm") for row in selected],
                    ),
                    "pslr_db": mean_or_nan([row.get("pslr_db") for row in selected]),
                    "islr_db": mean_or_nan([row.get("islr_db") for row in selected]),
                    "distortion_mm": mean_or_nan(
                        [row.get("distortion_mm") for row in selected],
                    ),
                    "distortion_pass_rate": pass_rate,
                    "distortion_penalty": penalty,
                },
            )
    return rows


def build_contrast_group_rows(roi_rows, source, roi_count, methods):
    """Build contrast group rows."""
    groups = picmus_contrast_groups(source, roi_count)
    rows = []
    for method in methods:
        indexed = rows_for_method_index(
            [row for row in roi_rows if "contrast_score_dB" in row],
            method,
            "roi_idx",
        )
        for group_name, indices0 in groups:
            indices = [i - 1 for i in indices0]
            selected = [indexed[i] for i in indices if i in indexed]
            if not selected:
                continue
            rows.append(
                {
                    "method": method,
                    "group": group_name,
                    "roi_indices": " ".join(str(i) for i in indices0),
                    "n_rois": len(selected),
                    "contrast_score_dB": mean_or_nan(
                        [row.get("contrast_score_dB") for row in selected],
                    ),
                    "CR_dB": mean_or_nan([row.get("CR_dB") for row in selected]),
                    "CNR": mean_or_nan([row.get("CNR") for row in selected]),
                    "gCNR": mean_or_nan([row.get("gCNR") for row in selected]),
                    "cyst_residual_dB": mean_or_nan(
                        [row.get("cyst_residual_dB") for row in selected],
                    ),
                    "speckle_penalty": speckle_penalty_for_method(
                        roi_rows,
                        method,
                    ),
                },
            )
    return rows


def fmt_metric(value, precision=3):
    """Execute fmt metric."""
    try:
        if value is None or not np.isfinite(value):
            return "NA"
    except TypeError:
        return "NA"
    return f"{float(value):.{precision}f}"


def write_picmus_report(
    path,
    mode,
    source,
    methods,
    summary_rows,
    contrast_group_rows,
    resolution_group_rows,
    roi_rows,
):
    """Execute write picmus report."""
    summary_by_method = {row["method"]: row for row in summary_rows}
    with open(path, "w", encoding="utf-8") as file:
        file.write("PICMUS-inspired evaluation summary (ours protocol)\n")
        file.write(f"mode: {mode}\nsource: {source}\n\n")
        for method in methods:
            row = summary_by_method.get(method, {})
            file.write(f"[{method}]\n")
            if mode == "contrast_speckle":
                file.write(
                    f"mean_contrast_score_dB: {fmt_metric(row.get('contrast_score_dB'), 1)}\n",
                )
                file.write(f"mean_CR_dB: {fmt_metric(row.get('CR_dB'), 3)}\n")
                file.write(f"mean_CNR: {fmt_metric(row.get('CNR'), 3)}\n")
                file.write(f"mean_gCNR: {fmt_metric(row.get('gCNR'), 3)}\n")
                file.write(
                    f"speckle_pass_rate: {fmt_metric(row.get('speckle_pass_rate'), 3)}\n",
                )
                file.write(
                    f"speckle_penalty: {fmt_metric(speckle_penalty_for_method(roi_rows, method), 1)}\n",
                )
                for group in [r for r in contrast_group_rows if r["method"] == method]:
                    file.write(
                        f"  {group['group']} ({group['roi_indices']}): "
                        f"contrast={fmt_metric(group.get('contrast_score_dB'), 1)}, "
                        f"CNR={fmt_metric(group.get('CNR'), 3)}, "
                        f"gCNR={fmt_metric(group.get('gCNR'), 3)}\n",
                    )
            elif mode == "resolution_distorsion":
                file.write(
                    f"mean_FWHM_axial_mm: {fmt_metric(row.get('FWHM_axial_mm'), 4)}\n",
                )
                file.write(
                    f"mean_FWHM_lateral_mm: {fmt_metric(row.get('FWHM_lateral_mm'), 4)}\n",
                )
                file.write(f"mean_PSLR_dB: {fmt_metric(row.get('pslr_db'), 3)}\n")
                file.write(f"mean_ISLR_dB: {fmt_metric(row.get('islr_db'), 3)}\n")
                file.write(
                    f"distortion_pass_rate: {fmt_metric(row.get('distortion_pass_rate'), 3)}\n",
                )
                file.write(
                    f"distortion_penalty: {fmt_metric(row.get('distortion_penalty'), 1)}\n",
                )
                for group in [r for r in resolution_group_rows if r["method"] == method]:
                    file.write(
                        f"  {group['group']} ({group['target_indices']}): "
                        f"axial={fmt_metric(group.get('FWHM_axial_mm'), 4)}, "
                        f"lateral={fmt_metric(group.get('FWHM_lateral_mm'), 4)}, "
                        f"penalty={fmt_metric(group.get('distortion_penalty'), 1)}\n",
                    )
            else:
                file.write("No controlled-scene score for this mode.\n")
            file.write("\n")


def write_csv(path, rows, fieldnames):
    """Execute write csv."""
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Run the command-line workflow."""
    args = parse_args()
    positive_args = {
        "dr": args.dr,
        "auto_roi_radius_mm": args.auto_roi_radius_mm,
        "auto_target_min_distance_mm": args.auto_target_min_distance_mm,
        "fwhm_window_mm": args.fwhm_window_mm,
    }
    invalid = [name for name, value in positive_args.items() if not math.isfinite(value) or value <= 0]
    if invalid:
        raise ValueError(f"参数必须是有限正数: {', '.join(invalid)}")
    if args.auto_target_count < 1:
        raise ValueError("auto_target_count 必须大于等于 1")
    comparison_path = resolve(args.comparison_npy)
    h5_path = resolve(args.h5_path)
    out_dir = resolve(args.out_dir)
    meta_from_h5 = read_sample_meta(h5_path, args.h5_sample_idx)
    has_gt = bool(meta_from_h5.get("has_gt", False))
    sample_name = str(meta_from_h5.get("sample_name", ""))
    if sample_name not in CONTROLLED_SCENES:
        print(f"Non-controlled scene {sample_name or '<unknown>'}: skipped full phantom metric export.")
        return

    comparison = np.load(comparison_path).astype(np.float64)
    x_mm, z_mm = load_grids(h5_path)
    if comparison.ndim != IMAGE_STACK_DIMENSIONS or comparison.shape[0] < 1:
        raise ValueError(
            f"comparison 必须是非空 [method,z,x] 三维数组,实际为 {comparison.shape}",
        )
    if comparison.shape[1:] != (len(z_mm), len(x_mm)):
        raise ValueError(
            f"Image shape {comparison.shape[1:]} does not match H5 grids {(len(z_mm), len(x_mm))}",
        )
    if not np.all(np.isfinite(comparison)):
        bad_count = int(comparison.size - np.count_nonzero(np.isfinite(comparison)))
        raise ValueError(f"comparison 包含 {bad_count} 个 NaN/Inf,拒绝生成无效指标")
    methods = load_method_names(args, comparison_path, comparison.shape[0], has_gt)

    mode = infer_mode(args, meta_from_h5)
    source = infer_source(args, meta_from_h5)
    distortion_z_offset_mm = picmus_distortion_z_offset_mm(h5_path, source)
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

    resolution_label_map = None
    distortion_roi_label_map = None
    distortion_inside_label_map = None
    if targets:
        resolution_label_map = build_square_label_map(
            x_mm,
            z_mm,
            targets,
            args.fwhm_window_mm,
        )
        if source == "simulation":
            distortion_roi_label_map = build_square_label_map(
                x_mm,
                z_mm,
                targets,
                args.fwhm_window_mm,
                z_offset_mm=distortion_z_offset_mm,
            )
            distortion_inside_label_map = build_square_label_map(
                x_mm,
                z_mm,
                targets,
                0.29570,
                z_offset_mm=distortion_z_offset_mm,
            )

    gt_display = db_to_display(gt_db, args.dr) if has_gt else None
    gt_linear_envelope = db_to_envelope(gt_db) if has_gt else None
    rows = []
    roi_rows = []
    target_rows = []
    for method, db_img in zip(methods, comparison, strict=True):
        display = db_to_display(db_img, args.dr)
        env = db_to_envelope(db_img)
        row = {"method": method}
        if has_gt and method == "GT":
            row["SSIM_dB_vs_GT"] = 1.0
            row["SSIM_envelope_vs_GT"] = 1.0
            row["PSNR_dB_vs_GT"] = float("inf")
            row["MAE_dB_vs_GT"] = 0.0
        elif has_gt:
            row["SSIM_dB_vs_GT"] = local_ssim(display, gt_display, data_range=1.0)
            row["SSIM_envelope_vs_GT"] = local_ssim(
                env,
                gt_linear_envelope,
                data_range=1.0,
            )
            if peak_signal_noise_ratio is not None:
                row["PSNR_dB_vs_GT"] = float(
                    peak_signal_noise_ratio(gt_display, display, data_range=1.0),
                )
            else:
                row["PSNR_dB_vs_GT"] = psnr(display, gt_display, data_range=1.0)
            clipped_db = np.clip(db_img, -args.dr, 0.0)
            clipped_gt_db = np.clip(gt_db, -args.dr, 0.0)
            row["MAE_dB_vs_GT"] = float(np.mean(np.abs(clipped_db - clipped_gt_db)))
        else:
            row["SSIM_dB_vs_GT"] = np.nan
            row["SSIM_envelope_vs_GT"] = np.nan
            row["PSNR_dB_vs_GT"] = np.nan
            row["MAE_dB_vs_GT"] = np.nan

        contrast_scores = []
        cr_scores = []
        cnr_scores = []
        gcnr_scores = []
        residual_scores = []
        for idx, roi in enumerate(rois):
            score = standard_contrast_score(
                db_img,
                x_mm,
                z_mm,
                roi,
                lateral_resolution_mm,
            )
            metrics = contrast_roi_metrics(
                db_img,
                x_mm,
                z_mm,
                roi,
                lateral_resolution_mm,
            )
            if not np.isfinite(score) and metrics is None:
                continue
            roi_row = {"method": method, "roi_idx": idx, **roi, "contrast_score_dB": score}
            if metrics is not None:
                roi_row.update(metrics)
                cr_scores.append(metrics["CR_dB"])
                cnr_scores.append(metrics["CNR"])
                gcnr_scores.append(metrics["gCNR"])
                residual_scores.append(metrics["cyst_residual_dB"])
            roi_rows.append(roi_row)
            contrast_scores.append(score)
        row["contrast_score_dB"] = mean_or_nan(contrast_scores)
        row["CR_dB"] = mean_or_nan(cr_scores)
        row["CNR"] = mean_or_nan(cnr_scores)
        row["gCNR"] = mean_or_nan(gcnr_scores)
        row["cyst_residual_dB"] = mean_or_nan(residual_scores)

        speckle_stats = []
        for idx, roi in enumerate(speckle_rois):
            stats = standard_speckle_quality(
                env,
                x_mm,
                z_mm,
                roi,
                lateral_resolution_mm,
                axial_resolution_mm,
            )
            if stats is None:
                continue
            roi_rows.append({"method": method, "roi_idx": idx, **roi, **stats})
            speckle_stats.append(stats)
        row["speckle_pass_rate"] = mean_or_nan(
            [s["speckle_pass"] for s in speckle_stats],
        )
        row["speckle_penalty"] = speckle_penalty_for_method(
            roi_rows,
            method,
        )
        row["speckle_KS_D"] = mean_or_nan([s["speckle_KS_D"] for s in speckle_stats])
        row["speckle_KS_p"] = mean_or_nan([s["speckle_KS_p"] for s in speckle_stats])
        row["speckle_SNR"] = mean_or_nan([s["speckle_SNR"] for s in speckle_stats])
        row["ENL"] = mean_or_nan([s["ENL"] for s in speckle_stats])

        target_stats = []
        for idx, target in enumerate(targets):
            stats = target_resolution(
                db_img,
                x_mm,
                z_mm,
                target,
                args.fwhm_window_mm,
                target_idx=idx + 1,
                label_map=resolution_label_map,
            )
            if stats is None:
                continue
            stats["distortion_pass"] = picmus_distortion_pass(
                db_img,
                idx + 1,
                source,
                roi_label_map=distortion_roi_label_map,
                inside_label_map=distortion_inside_label_map,
            )
            target_rows.append({"method": method, "target_idx": idx + 1, **stats})
            target_stats.append(stats)
        row["FWHM_axial_mm"] = mean_or_nan([s["FWHM_axial_mm"] for s in target_stats])
        row["FWHM_lateral_mm"] = mean_or_nan(
            [s["FWHM_lateral_mm"] for s in target_stats],
        )
        row["pslr_db"] = mean_or_nan([s["pslr_db"] for s in target_stats])
        row["islr_db"] = mean_or_nan([s["islr_db"] for s in target_stats])
        row["distortion_mm"] = mean_or_nan([s["distortion_mm"] for s in target_stats])
        row["distortion_pass_rate"] = mean_or_nan(
            [s["distortion_pass"] for s in target_stats],
        )
        row["distortion_penalty"] = distortion_penalty_for_method(
            target_rows,
            method,
            source,
        )
        rows.append(row)

    summary_fields = [
        "method",
        "SSIM_dB_vs_GT",
        "SSIM_envelope_vs_GT",
        "PSNR_dB_vs_GT",
        "MAE_dB_vs_GT",
        "contrast_score_dB",
        "CR_dB",
        "CNR",
        "gCNR",
        "cyst_residual_dB",
        "speckle_pass_rate",
        "speckle_penalty",
        "speckle_KS_D",
        "speckle_KS_p",
        "speckle_SNR",
        "ENL",
        "FWHM_axial_mm",
        "FWHM_lateral_mm",
        "pslr_db",
        "islr_db",
        "distortion_mm",
        "distortion_pass_rate",
        "distortion_penalty",
    ]
    write_csv(os.path.join(out_dir, "summary_metrics.csv"), rows, summary_fields)
    contrast_group_rows = build_contrast_group_rows(
        roi_rows,
        source,
        len(rois),
        methods,
    )
    resolution_group_rows = build_resolution_group_rows(
        target_rows,
        source,
        len(targets),
        methods,
    )
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
                for key in data_row:
                    if key not in fieldnames:
                        fieldnames.append(key)
            write_csv(path, data_rows, fieldnames)
        else:
            write_csv(path, [{"status": "not_applicable"}], ["status"])
    write_picmus_report(
        os.path.join(out_dir, "evaluation_protocol_summary.txt"),
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
        "fwhm_window_mm": args.fwhm_window_mm,
        "picmus_distortion_z_offset_mm": distortion_z_offset_mm,
        "contrast_groups": picmus_contrast_groups(source, len(rois)),
        "resolution_groups": picmus_resolution_groups(source, len(targets)),
        "contrast_rois": rois,
        "resolution_targets": targets,
        "dr": args.dr,
        "protocol": "ours_picmus_inspired",
        "metric_domains": {
            "contrast_score_dB": "relative dB domain from normalized envelope",
            "CR_dB": "linear envelope mean ratio, reported in dB",
            "CNR": "linear envelope domain",
            "gCNR": "linear envelope domain, fixed 256-bin histogram on [0, 1]",
            "speckle_KS_D": "linear envelope domain with 5-pixel block sampling",
            "speckle_SNR": "linear envelope domain with 5-pixel block sampling",
            "ENL": "linear envelope domain with 5-pixel block sampling",
            "FWHM_axial_mm": "relative dB domain, -6 dB width",
            "FWHM_lateral_mm": "relative dB domain, -6 dB width",
            "pslr_db": "relative dB domain",
            "islr_db": "relative dB domain",
            "SSIM_dB_vs_GT": "[-dr, 0] dB display domain",
            "SSIM_envelope_vs_GT": "normalized linear envelope domain",
            "PSNR_dB_vs_GT": "[-dr, 0] dB display domain",
            "MAE_dB_vs_GT": "clipped [-dr, 0] dB domain",
        },
        "normalization": "Each output and GT frame is independently peak-normalized to 0 dB before relative metrics.",
        "note": "Standard metrics use the predefined phantom ROI and target definitions when available.",
    }
    with open(
        os.path.join(out_dir, "evaluation_meta.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(meta, file, indent=2, ensure_ascii=False)

    print(f"Saved summary: {os.path.join(out_dir, 'summary_metrics.csv')}")
    if contrast_group_rows:
        print(
            f"Saved contrast group metrics: {os.path.join(out_dir, 'contrast_group_metrics.csv')}",
        )
    if resolution_group_rows:
        print(
            f"Saved resolution group metrics: {os.path.join(out_dir, 'resolution_group_metrics.csv')}",
        )
    print(
        f"Saved evaluation protocol report: {os.path.join(out_dir, 'evaluation_protocol_summary.txt')}",
    )
    print(
        f"Mode={mode}, source={source}, contrast_rois={len(rois)}, resolution_targets={len(targets)}",
    )


if __name__ == "__main__":
    main()

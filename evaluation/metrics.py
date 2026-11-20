"""Numerical image and PICMUS evaluation metrics."""

import math

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import kstest

from shared_core.images import (
    db_to_display as db_to_display,
    envelope_from_db as db_to_envelope,
)

EPSILON = 1e-9
MIN_PROFILE_POINTS = 2
MIN_SIDE_LOBE_POINTS = 5
MIN_ROI_SAMPLES = 8
TARGET_MIN_PROMINENCE_DB = 6.0
SPECKLE_PASS_ALPHA = 0.05


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
        "CNR_env": float(cnr),
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
        "ENL_env": float((np.mean(sample) ** 2) / (np.var(sample) + 1e-24)),
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


def target_response_peak(db_img, target_mask):
    z_ids, x_ids = np.where(target_mask)
    if z_ids.size == 0:
        return None
    values = db_img[target_mask]
    if not np.any(np.isfinite(values)):
        return None
    peak_flat_index = int(np.nanargmax(values))
    peak_value = float(values[peak_flat_index])
    prominence = peak_value - float(np.nanmedian(values))
    return (
        int(z_ids[peak_flat_index]),
        int(x_ids[peak_flat_index]),
        prominence >= TARGET_MIN_PROMINENCE_DB,
    )


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
    peak = target_response_peak(db_img, target_mask)
    if peak is None:
        return None
    iz, ix, response_detected = peak
    result = {
        "target_x_mm": target["x_mm"],
        "target_z_mm": target["z_mm"],
        "peak_x_mm": np.nan,
        "peak_z_mm": np.nan,
        "FWHM_axial_mm": np.nan,
        "FWHM_lateral_mm": np.nan,
        "pslr_axial_db": np.nan,
        "pslr_lateral_db": np.nan,
        "pslr_db": np.nan,
        "islr_axial_db": np.nan,
        "islr_lateral_db": np.nan,
        "islr_db": np.nan,
        "localization_error_mm": np.nan,
        "target_response_detected": response_detected,
    }
    if not response_detected:
        return result
    masked = np.full_like(db_img, np.nanmin(db_img), dtype=np.float64)
    masked[target_mask] = db_img[target_mask]
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
    result.update({
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
        "localization_error_mm": float(peak_offset),
    })
    return result


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
    peak = target_response_peak(db_img, target_mask)
    if peak is None:
        return np.nan
    z_index, x_index, response_detected = peak
    if not response_detected:
        return 0.0
    peak = z_index, x_index
    return 1.0 if inside_label_map[peak] == target_idx else 0.0


def mean_or_nan(values):
    """Execute mean or nan."""
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else np.nan


def cyst_lateral_profile(db_image, x_mm, z_mm, cyst):
    """Peak-normalized lateral profile through the central band of a cyst ROI.

    Returns ``(offsets_mm, profile_db)``. The profile is NaN when the band has
    no positive envelope peak; callers can skip such a method.
    """
    image = np.asarray(db_image)
    x_mm = np.asarray(x_mm, dtype=np.float64)
    z_mm = np.asarray(z_mm, dtype=np.float64)
    if image.shape != (z_mm.size, x_mm.size):
        raise ValueError("cyst profile image shape must match the coordinate grids")
    diameter = float(cyst["diameter_mm"])
    radius = diameter / 2.0
    half_width = max(5.0, 1.75 * radius)
    x_mask = np.abs(x_mm - float(cyst["x_mm"])) < half_width
    offsets = x_mm[x_mask] - float(cyst["x_mm"])
    center = int(np.argmin(np.abs(z_mm - float(cyst["z_mm"]))))
    spacing = float(np.median(np.abs(np.diff(z_mm)))) if z_mm.size > 1 else 1.0
    half_rows = max(1, min(3, int(round(0.35 * diameter / max(spacing, 1.0e-9)))))
    rows = np.arange(max(0, center - half_rows), min(z_mm.size, center + half_rows + 1))
    envelope = np.power(10.0, image[np.ix_(rows, x_mask)] / 20.0)
    profile_env = np.nanmean(envelope, axis=0)
    peak = float(np.nanmax(profile_env)) if profile_env.size else np.nan
    if not np.isfinite(peak) or peak <= 0:
        return offsets, np.full(offsets.shape, np.nan)
    profile = 20.0 * np.log10(np.maximum(profile_env, 1.0e-12))
    return offsets, np.minimum(profile, 0.0)

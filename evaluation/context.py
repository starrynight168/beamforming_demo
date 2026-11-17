"""HDF5 and PICMUS input context for evaluation workflows."""

import math
import os
from pathlib import Path

import h5py
import numpy as np
import yaml

from shared_core.h5_schema import require_h5_schema
from evaluation.metrics import EPSILON

PROJECT_ROOT = os.fspath(Path(__file__).resolve().parents[1])
PICMUS_ROOT = os.path.join(PROJECT_ROOT, "data", "PICMUS")

SIMULATION_RESOLUTION_TARGETS = 20
EXPERIMENT_RESOLUTION_TARGETS = 7
SIMULATION_CONTRAST_ROIS = 9
EXPERIMENT_CONTRAST_ROIS = 2
NON_PICMUS_PROTOCOLS = {
    ("in_vivo", "in_vivo"),
    ("in_vitro", "in_vitro"),
    ("in_silico", "random_point_scattering"),
    ("unknown", "unknown"),
}
PICMUS_OFFICIAL_Z_CORRECTION_MM = 0.2


def resolve(path):
    """Execute resolve."""
    if path in ("auto", "none", None):
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


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
        require_h5_schema(hf)
        config = h5_embedded_config(hf)
        generation = config.get("generation", {})
        storage = generation.get("storage", {}) if isinstance(generation, dict) else {}
        stored_inputs = storage.get("inputs", {}) if isinstance(storage, dict) else {}
        input_domains = [
            domain for domain in ("iq", "rf", "analytic_rf") if domain in stored_inputs
        ]
        input_domain = input_domains[0] if input_domains else None
        if input_domain is None:
            raise ValueError("H5 config_yaml.generation.storage.inputs 未声明输入域")
        if "acquisition_id" in hf:
            sample_axis = "acquisition_id"
        elif input_domain == "iq":
            sample_axis = "iq_i"
        elif input_domain == "rf":
            sample_axis = "rf"
        else:
            sample_axis = "ar_i"
        n_samples = int(hf[sample_axis].shape[0])
        if not 0 <= sample_idx < n_samples:
            raise IndexError(f"sample index {sample_idx} is outside [0, {n_samples - 1}]")
        ground_truth = generation.get("ground_truth", {})
        targets = generation.get("training_targets", {})
        domain = ground_truth["signal_domain"]
        algorithm = ground_truth["algorithm"]
        entry = targets.get("domain_targets", {}).get(domain, {}).get(algorithm, {})
        target_paths = [path for path in (entry.get("db"), entry.get("envelope")) if path]
        if domain != "rf":
            pair = entry.get("signal", {})
            if not isinstance(pair, dict):
                raise ValueError(f"domain_targets.{domain}.{algorithm}.signal must be a mapping")
            target_paths.extend(path for path in pair.values() if path)
        has_gt = any(path and str(path).lstrip("/") in hf for path in target_paths)
        for path in target_paths:
            key = str(path).lstrip("/") if path else None
            if key in hf and (not hf[key].shape or hf[key].shape[0] != n_samples):
                raise ValueError(f"{key} 样本数必须与 {sample_axis} 一致")
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
                "signal_domain": input_domain,
                "signal_domains": input_domains,
            }
        if (
            isinstance(source_samples, dict)
            and source_samples.get("encoding") in ("root_labels", "root_labels_with_path_template")
            and source_samples.get("count") == n_samples
        ):
            id_dataset = str(source_samples["acquisition_id_dataset"])
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
                "signal_domain": input_domain,
                "signal_domains": input_domains,
            }
        raise IndexError(f"source_samples has no sample index {sample_idx}")


def picmus_distortion_z_offset_mm(h5_path, source):
    """Return the distortion-mask depth correction for the packed dataset."""
    if source != "simulation":
        return 0.0
    with h5py.File(h5_path, "r") as hf:
        require_h5_schema(hf)
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
        require_h5_schema(hf)
        x_mm = hf["x_grid"][:].astype(float) * 1000.0
        z_mm = hf["z_grid"][:].astype(float) * 1000.0
    for name, grid in (("x_grid", x_mm), ("z_grid", z_mm)):
        if grid.ndim != 1 or grid.size < 2 or not np.all(np.isfinite(grid)) or not np.all(np.diff(grid) > 0):
            raise ValueError(f"{name} must be a finite, strictly increasing 1D grid with at least two points")
    return x_mm, z_mm


def read_phantom(phantom_path):
    """Read phantom."""
    out = {
        "contrast_rois": [],
        "speckle_rois": [],
        "resolution_targets": [],
        "lateral_resolution_mm": None,
        "axial_resolution_mm": None,
    }
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


def picmus_reference_path(mode, source):
    if mode == "resolution_distortion":
        prefix = "expe" if source == "experiments" else "simu"
        rel = os.path.join(
            "database",
            source,
            "resolution_distortion",
            f"resolution_distortion_{prefix}_phantom.hdf5",
        )
    else:
        prefix = "expe" if source == "experiments" else "simu"
        rel = os.path.join(
            "database",
            source,
            "contrast_speckle",
            f"contrast_speckle_{prefix}_phantom.hdf5",
        )
    return Path(PICMUS_ROOT) / rel


def load_picmus_context(
    h5_path,
    sample_idx,
    *,
    phantom_path="auto",
    phantom_mode="auto",
    phantom_source="auto",
    fwhm_window_mm=1.8,
):
    h5_path = resolve(h5_path)
    meta = read_sample_meta(h5_path, sample_idx)
    values = []
    for name, selected in (
        ("phantom_mode", phantom_mode),
        ("phantom_source", phantom_source),
    ):
        stored = meta[name]
        if selected != "auto" and stored and selected != stored:
            raise ValueError(f"{name} 与 H5 元数据不一致: {selected} != {stored}")
        value = stored if selected == "auto" else selected
        if not value:
            raise ValueError(f"H5 缺少 {name}，请显式指定 --{name}")
        values.append(value)
    mode, source = values
    if (mode, source) in NON_PICMUS_PROTOCOLS:
        return None
    supported_mode = mode in {"contrast_speckle", "resolution_distortion"}
    supported_source = source in {"simulation", "experiments"}
    if not supported_mode or not supported_source:
        raise ValueError(f"不支持的 PICMUS 模体组合: mode={mode}, source={source}")
    if phantom_path == "auto":
        if not meta["phantom_path"]:
            raise ValueError("PICMUS 样本缺少 phantom_path，请显式指定 --phantom_path")
        path = Path(meta["phantom_path"])
        if not path.is_absolute():
            path = Path(PICMUS_ROOT) / path
    else:
        if not phantom_path or phantom_path == "none":
            raise ValueError("PICMUS 评估必须提供模体文件")
        path = Path(resolve(phantom_path))
    phantom = read_phantom(path)
    reference = read_phantom(picmus_reference_path(mode, source))
    if phantom != reference:
        raise ValueError(
            "模体定义与对应 PICMUS 官方模体不一致，需保留 ROI 和点目标顺序"
        )
    x_mm, z_mm = load_grids(h5_path)
    z_offset = picmus_distortion_z_offset_mm(h5_path, source)
    validate_picmus_geometry(
        phantom, mode, source, x_mm, z_mm, fwhm_window_mm, z_offset
    )
    return {
        "meta": meta,
        "mode": mode,
        "source": source,
        "phantom_path": str(path.resolve()),
        "phantom": phantom,
        "x_mm": x_mm,
        "z_mm": z_mm,
        "distortion_z_offset_mm": z_offset,
        "signal_domain": meta["signal_domain"],
        "signal_domains": meta["signal_domains"],
    }


def validate_picmus_geometry(phantom, mode, source, x_mm, z_mm, window_mm, z_offset_mm):
    lateral = phantom["lateral_resolution_mm"]
    axial = phantom["axial_resolution_mm"]
    if any(
        value is None or not math.isfinite(value) or value <= 0
        for value in (lateral, axial)
    ):
        raise ValueError("PICMUS 模体缺少有限正数的横向或轴向分辨率")
    if mode == "contrast_speckle":
        counts = (
            (SIMULATION_CONTRAST_ROIS, 4)
            if source == "simulation"
            else (EXPERIMENT_CONTRAST_ROIS, 3)
        )
        if (len(phantom["contrast_rois"]), len(phantom["speckle_rois"])) != counts:
            raise ValueError("PICMUS 囊肿或散斑 ROI 数量不符合对应协议")
        regions = []
        for roi in phantom["contrast_rois"]:
            radius = roi["diameter_mm"] / 2
            outer = 1.2 * math.sqrt((radius - lateral) ** 2 + (radius + lateral) ** 2)
            regions.append((roi, outer, outer))
        regions.extend(
            (roi, roi["psf_time_x"] * lateral, roi["psf_time_z"] * axial)
            for roi in phantom["speckle_rois"]
        )
    else:
        count = (
            SIMULATION_RESOLUTION_TARGETS
            if source == "simulation"
            else EXPERIMENT_RESOLUTION_TARGETS
        )
        if len(phantom["resolution_targets"]) != count:
            raise ValueError("PICMUS 点目标数量不符合对应协议")
        regions = [
            (target, window_mm, window_mm) for target in phantom["resolution_targets"]
        ]
    for region, half_x, half_z in regions:
        x, z = region["x_mm"], region["z_mm"]
        shifted_z = (
            z + z_offset_mm
            if mode == "resolution_distortion" and source == "simulation"
            else z
        )
        if not (
            x_mm[0] <= x - half_x
            and x + half_x <= x_mm[-1]
            and z_mm[0] <= min(z, shifted_z) - half_z
            and max(z, shifted_z) + half_z <= z_mm[-1]
        ):
            raise ValueError("H5 坐标网格未完整覆盖 PICMUS 评估区域")

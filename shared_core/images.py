"""Linear envelope and B-mode conversion shared by algorithm and data tools."""

import numpy as np

from shared_core.h5_schema import require_h5_schema
from shared_core.numerics import component_magnitude


RF_ENVELOPE_OPERATION = "beamformed RF envelope with symmetric depth-Hilbert zero-padding and linear resampling to z_grid"


def require_current_rf_envelope_method(training_targets):
    method = training_targets.get("rf_envelope_method") if isinstance(training_targets, dict) else None
    if not isinstance(method, dict) or method.get("operation") != RF_ENVELOPE_OPERATION:
        raise ValueError("RF 包络缓存方法缺失或过期；请从 NPZ 重建 RF 包络")


def to_envelope(real, imag=None, *, domain="iq"):
    if isinstance(real, dict):
        domain = real["domain"]
        imag = real["imag"]
        real = real["real"]
    if domain == "rf":
        raise ValueError("有符号 RF 不是包络；请在 outputs.rf 中保存 envelope，读取密网格生成的包络缓存")
    if domain not in ("iq", "analytic_rf") or imag is None:
        raise ValueError("取包络需要明确的 IQ/解析 RF 实部和虚部")
    real = np.asarray(real, dtype=np.float32)
    imag = np.asarray(imag, dtype=np.float32)
    if real.shape != imag.shape or real.ndim != 2:
        raise ValueError("复数成像输出必须为形状一致的二维实部和虚部")
    if not np.isfinite(real).all() or not np.isfinite(imag).all():
        raise ValueError("复数成像输出包含 NaN/Inf")
    return component_magnitude(real, imag)


def to_db(envelope, *, reference=None, dr=None, gain_db=None):
    envelope = np.asarray(envelope, dtype=np.float64)
    if envelope.ndim != 2 or not np.isfinite(envelope).all() or np.any(envelope < 0):
        raise ValueError("包络必须为有限非负二维数组")
    if not np.any(envelope > 0):
        raise ValueError("包络峰值必须为正数")
    db = np.full_like(envelope, -np.inf)
    np.log10(envelope, out=db, where=envelope > 0)
    db *= 20.0
    if gain_db is not None:
        gain_db = np.asarray(gain_db, dtype=np.float64)
        if gain_db.shape != (envelope.shape[0],) or not np.isfinite(gain_db).all():
            raise ValueError("深度增益必须为每行一个有限 dB 值")
        db += gain_db[:, None]
    if reference is None:
        db -= db.max()
    else:
        if not np.isfinite(reference) or reference <= 0:
            raise ValueError("参考幅度必须为有限正数")
        db -= 20.0 * np.log10(reference)
    db = np.maximum(db, -240.0)
    if dr is not None:
        dynamic_range = _dynamic_range_value(dr)
        db = np.clip(db, -dynamic_range, 0.0)
    return db.astype(np.float32)


def _dynamic_range_value(dynamic_range):
    value = float(dynamic_range)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("动态范围必须为有限正数")
    return value


def _unit_interval(values):
    clamp = getattr(values, "clamp", None)
    if callable(clamp):
        return clamp(0.0, 1.0)
    return np.clip(np.asarray(values), 0.0, 1.0)


def db_to_display(db, dynamic_range):
    """Map dB values into the normalized [0, 1] display interval.

    NumPy arrays and torch tensors are both accepted so that one definition of
    the display mapping serves generation, algorithms, training, and evaluation.
    """
    dynamic_range = _dynamic_range_value(dynamic_range)
    values = db if hasattr(db, "clamp") else np.asarray(db)
    return _unit_interval((values + dynamic_range) / dynamic_range)


def envelope_to_display(envelope, dynamic_range):
    try:
        import torch

        if torch.is_tensor(envelope):
            db = 20.0 * torch.log10(envelope + 1e-12)
            return db_to_display(db, dynamic_range)
    except ImportError:
        pass
    db = 20.0 * np.log10(np.asarray(envelope) + 1e-12)
    return db_to_display(db, dynamic_range)


def display_to_db(display, dynamic_range):
    """Map normalized [0, 1] display values back to dB (inverse of ``db_to_display``)."""
    dynamic_range = _dynamic_range_value(dynamic_range)
    return np.clip(np.asarray(display), 0.0, 1.0) * dynamic_range - dynamic_range


def envelope_from_db(db):
    """Convert dB amplitudes back to linear envelope values."""
    return np.power(10.0, np.asarray(db) / 20.0)


def _image(hf, path, sample_idx):
    if path not in hf:
        raise KeyError(f"H5 缺少显示目标字段 {path}")
    value = np.asarray(hf[path][sample_idx], dtype=np.float32)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(f"显示目标 {path} 必须是有限二维图像")
    return value


def _dynamic_range(ground_truth):
    dr = ground_truth.get("dynamic_range_db")
    if dr is None:
        raise ValueError("H5 ground_truth.dynamic_range_db 必须为有限正数")
    return _dynamic_range_value(dr)


def tgc_attenuation_db(depth, fc, alpha):
    """Depth-dependent TGC attenuation in dB.

    This is the single definition of the TGC law; NumPy and torch callers apply
    it through :func:`tgc_gain_db_from_depth` / :func:`tgc_gain_from_depth_tensor`.
    """
    return alpha * (fc / 1e6) * (depth * 100.0) * 2.0


def _validate_tgc_parameters(fc, alpha):
    if not np.isfinite(fc) or fc <= 0:
        raise ValueError("TGC 中心频率必须为有限正数")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("TGC 衰减系数必须为有限非负数")


def tgc_gain_db_from_depth(depth, fc, alpha):
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 1 or not np.isfinite(depth).all():
        raise ValueError("TGC 深度必须为有限一维数组")
    fc, alpha = float(fc), float(alpha)
    _validate_tgc_parameters(fc, alpha)
    return tgc_attenuation_db(depth, fc, alpha)


def tgc_gain_from_depth(depth, fc, alpha, enabled=True):
    if not enabled:
        return np.ones_like(np.asarray(depth), dtype=np.float32)
    return np.power(10.0, tgc_gain_db_from_depth(depth, fc, alpha) / 20.0)


def tgc_gain_from_depth_tensor(depth, fc, alpha, enabled=True):
    """Torch counterpart of :func:`tgc_gain_from_depth` for tensor depth grids."""
    import torch

    fc, alpha = float(fc), float(alpha)
    _validate_tgc_parameters(fc, alpha)
    if not enabled:
        return torch.ones_like(depth)
    return torch.pow(10.0, tgc_attenuation_db(depth, fc, alpha) / 20.0)


def apply_tgc(envelope, depth, fc, alpha, enabled=True):
    try:
        import torch

        if torch.is_tensor(envelope):
            gain = tgc_gain_from_depth_tensor(depth, fc, alpha, enabled)
        else:
            gain = tgc_gain_from_depth(depth, fc, alpha, enabled)
    except ImportError:
        gain = tgc_gain_from_depth(depth, fc, alpha, enabled)
    if getattr(gain, "ndim", 0) == 1 and getattr(envelope, "ndim", 0) > 1:
        if envelope.shape[0] == gain.shape[0]:
            gain = gain.reshape((gain.shape[0],) + (1,) * (envelope.ndim - 1))
    return envelope * gain


def tgc_gain_db(hf, ground_truth, height):
    tgc = ground_truth.get("tgc")
    if type(tgc) is not bool:
        raise ValueError("H5 ground_truth.tgc 必须明确为布尔值")
    if not tgc:
        return np.zeros(height, dtype=np.float64)
    alpha = ground_truth.get("tgc_alpha_db_cm_mhz")
    if alpha is None or not np.isfinite(alpha) or alpha < 0:
        raise ValueError("H5 ground_truth.tgc_alpha_db_cm_mhz 必须为有限非负数")
    if "fc" not in hf or "z_grid" not in hf:
        raise KeyError("H5 必须包含 fc/z_grid 才能生成 TGC 显示目标")
    fc = float(hf["fc"][()])
    depth = np.asarray(hf["z_grid"][:], dtype=np.float32)
    if depth.shape != (height,):
        raise ValueError("H5 fc/z_grid 与 TGC 显示目标不匹配")
    return tgc_gain_db_from_depth(depth, fc, alpha)


def ground_truth_db(hf, sample_idx, algorithm=None, normalized=False, *, signal_domain=None):
    from shared_core.data import embedded_config

    require_h5_schema(hf)
    config = embedded_config(hf)
    if not isinstance(config.get("generation"), dict):
        raise ValueError("H5 config_yaml.generation 必须是映射")
    generation = config["generation"]
    ground_truth = generation.get("ground_truth", {})
    targets = generation.get("training_targets", {})
    if not isinstance(ground_truth, dict) or not isinstance(targets, dict):
        raise ValueError("H5 ground_truth/training_targets 必须是映射")
    main_algorithm = ground_truth.get("algorithm")
    main_domain = ground_truth["signal_domain"]
    domain = main_domain if signal_domain is None else signal_domain
    algorithm = main_algorithm if algorithm is None else algorithm
    if not isinstance(algorithm, str) or not algorithm:
        raise ValueError("H5 ground_truth.algorithm 必须明确主目标算法")
    if domain not in ("rf", "analytic_rf", "iq"):
        raise ValueError("H5 ground_truth.signal_domain 必须为 rf/analytic_rf/iq")
    entry = targets.get("domain_targets", {}).get(domain, {}).get(algorithm, {})
    image_path = entry.get("db")
    envelope_path = entry.get("envelope")
    if (domain == "rf" and (image_path is not None or envelope_path is not None)
            and (image_path is None or entry.get("kind") != "image_reference")):
        require_current_rf_envelope_method(targets)
    if image_path is not None:
        image = _image(hf, image_path, sample_idx)
        if np.any((image < 0) | (image > 1)):
            raise ValueError(f"缓存显示目标 {image_path} 必须位于 [0,1]")
        if normalized:
            return image
        dr = _dynamic_range(ground_truth)
        return display_to_db(image, dr)

    if envelope_path is not None:
        envelope = _image(hf, envelope_path, sample_idx).astype(np.float64)
        if np.any(envelope < 0):
            raise ValueError("显示目标包络必须非负")
    elif domain == "rf":
        raise KeyError(f"H5 缺少 RF 包络；请配置 outputs.rf.{algorithm} 保存 envelope，不能对有符号 RF 直接取绝对值")
    else:
        pair = entry.get("signal")
        if not isinstance(pair, dict) or not pair.get("real") or not pair.get("imag"):
            raise KeyError(f"H5 缺少 {domain}/{algorithm} 显示缓存或复数标签；请配置 outputs.{domain}.{algorithm}")
        real = _image(hf, pair["real"], sample_idx)
        imag = _image(hf, pair["imag"], sample_idx)
        if real.shape != imag.shape:
            raise ValueError("显示目标实部/虚部必须具有相同形状")
        envelope = to_envelope(real, imag, domain=domain).astype(np.float64)
    dr = _dynamic_range(ground_truth)
    if not np.any(envelope > 0):
        raise ValueError("显示目标包络峰值必须为正数")
    db = to_db(envelope, dr=dr, gain_db=tgc_gain_db(hf, ground_truth, envelope.shape[0]))
    return db_to_display(db, dr).astype(np.float32) if normalized else db

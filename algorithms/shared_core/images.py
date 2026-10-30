"""Linear envelope and B-mode conversion shared by algorithm and data tools."""

import numpy as np


RF_ENVELOPE_OPERATION = "real RF-MV on dense depth grid, append N deep-end zeros before depth Hilbert, magnitude, linear resampling to z_grid"


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
    return np.hypot(real, imag)


def to_db(envelope, *, reference=None, dr=None, gain_db=None):
    envelope = np.asarray(envelope, dtype=np.float64)
    if envelope.ndim != 2 or not np.isfinite(envelope).all() or np.any(envelope < 0):
        raise ValueError("包络必须为有限非负二维数组")
    if not np.any(envelope > 0):
        raise ValueError("包络峰值必须为正数")
    db = 20.0 * np.log10(np.maximum(envelope, np.finfo(np.float64).tiny))
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
        if not np.isfinite(dr) or dr <= 0:
            raise ValueError("动态范围必须为有限正数")
        db = np.clip(db, -dr, 0.0)
    return db.astype(np.float32)

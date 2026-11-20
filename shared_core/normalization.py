import numpy as np

from shared_core.numerics import component_magnitude


def storage_normalization_mode(inputs):
    if not isinstance(inputs, dict) or set(inputs) - {"rf", "analytic_rf", "iq"}:
        raise ValueError("storage.inputs must select rf, analytic_rf or iq")
    modes = set()
    for domain, options in inputs.items():
        if (not isinstance(options, dict) or set(options) != {"dtype", "normalization"}
                or options["dtype"] not in ("float16", "float32")
                or options["normalization"] not in ("rms", "env_std", "none")):
            raise ValueError(f"storage.inputs.{domain} requires dtype=float16/float32 and normalization=rms/env_std/none")
        if options["normalization"] != "none":
            modes.add(options["normalization"])
    if len(modes) > 1:
        raise ValueError("Stored domains must share one normalization method; rms and env_std cannot be mixed")
    return next(iter(modes), "none")


def normalization_statistic(real, imag, mode, epsilon=1e-12):
    if mode == "rms":
        return float(np.sqrt(np.mean(real * real + imag * imag) + epsilon))
    if mode == "env_std":
        return float(np.std(component_magnitude(real, imag)) + epsilon)
    raise ValueError(f"Unsupported normalization: {mode}")


def normalization_scale(real, imag, mode, epsilon=1e-12):
    real, imag = np.asarray(real, dtype=np.float32), np.asarray(imag, dtype=np.float32)
    if real.shape != imag.shape or real.size == 0 or not np.isfinite(real).all() or not np.isfinite(imag).all():
        raise ValueError("Normalization requires matching finite nonempty real/imaginary arrays")
    if mode == "none":
        return 1.0
    scale = normalization_statistic(real, imag, mode, epsilon)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Normalization scale must be finite and positive")
    return scale

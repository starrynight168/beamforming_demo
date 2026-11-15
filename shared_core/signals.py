"""Channel RF conversions using the acquisition time axis."""

import numpy as np
from scipy.signal import filtfilt, hilbert, resample_poly


def _rf_array(rf):
    values = np.asarray(rf)
    if values.ndim != 3 or min(values.shape) == 0:
        raise ValueError("RF must have nonempty [angle,time,channel] dimensions")
    if np.iscomplexobj(values) or not np.isfinite(values).all():
        raise ValueError("RF must contain finite real samples")
    return values.astype(np.float32, copy=False)


def rf_to_analytic(rf):
    values = _rf_array(rf)
    return values, hilbert(values, axis=1).imag.astype(np.float32)


def rf_to_iq(rf, fs, fc, t0, lpf_coeffs, decimation_factor=1):
    values = _rf_array(rf)
    if not np.isfinite(fs) or fs <= 0 or not np.isfinite(fc) or not 0 < fc < fs / 2:
        raise ValueError("IQ conversion requires fs > 0 and 0 < fc < fs/2")
    if type(decimation_factor) is not int or decimation_factor < 1:
        raise ValueError("decimation_factor must be a positive integer")
    starts = np.asarray(t0, dtype=np.float64)
    if starts.ndim == 0:
        starts = np.full(values.shape[0], starts.item())
    if starts.shape != (values.shape[0],) or not np.isfinite(starts).all():
        raise ValueError("t0 must be finite and scalar or one value per angle")
    coefficients = np.asarray(lpf_coeffs)
    if coefficients.ndim != 1 or coefficients.size < 2 or not np.isfinite(coefficients).all():
        raise ValueError("lpf_coeffs must be a finite one-dimensional FIR filter")
    if values.shape[1] <= 3 * coefficients.size:
        raise ValueError("RF time samples must exceed the filtfilt padding length")
    time = starts[:, None] + np.arange(values.shape[1])[None, :] / fs
    oscillator = np.exp(-1j * 2 * np.pi * fc * time).astype(np.complex64)
    mixed = values * oscillator[:, :, None]
    i_data = filtfilt(coefficients, 1, mixed.real, axis=1).astype(np.float32)
    q_data = filtfilt(coefficients, 1, mixed.imag, axis=1).astype(np.float32)
    return tuple(
        resample_poly(component, up=1, down=decimation_factor, axis=1, window=("kaiser", 12.0)).astype(np.float32)
        for component in (i_data, q_data)
    )

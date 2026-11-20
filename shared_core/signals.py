"""Channel RF conversions using the acquisition time axis."""

import numpy as np
from scipy.signal import filtfilt, firwin, hilbert, resample_poly

DDC_LOWPASS_TAPS = 64
DDC_LOWPASS_CUTOFF_OVER_FC = 0.8


def ddc_lowpass_coefficients(
    fs,
    fc,
    taps=DDC_LOWPASS_TAPS,
    cutoff_over_fc=DDC_LOWPASS_CUTOFF_OVER_FC,
    *,
    cutoff_hz=None,
    window="hamming",
):
    """FIR anti-alias low-pass used by the RF→IQ demodulation path.

    Defaults to the project DDC settings. The cutoff and window may be
    specified explicitly by an acquisition's metadata or simulator.
    """
    fs, fc = float(fs), float(fc)
    if not np.isfinite(fs) or fs <= 0 or not np.isfinite(fc) or fc <= 0:
        raise ValueError("DDC 低通需要有限正采样率和中心频率")
    if type(taps) is not int or taps < 2:
        raise ValueError("DDC 低通阶数必须是至少为 2 的整数")
    cutoff = cutoff_over_fc * fc if cutoff_hz is None else cutoff_hz
    if not 0.0 < cutoff < fs / 2.0:
        raise ValueError("DDC 低通截止频率必须位于 (0, fs/2)")
    return firwin(taps, cutoff / (fs / 2.0), window=window)


def ddc_lowpass_name(taps):
    if type(taps) is not int or taps < 2:
        raise ValueError("DDC 低通阶数必须是至少为 2 的整数")
    return f"firwin_{taps}_then_filtfilt"


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

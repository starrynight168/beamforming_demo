"""RF-domain MV computations shared by evaluation tools."""

import numpy as np
import torch
from scipy.signal import hilbert, welch

from .run_output import emit

from algorithms.shared_core.beamforming import (
    dynamic_aperture_channel_count,
    interpolate_channel_samples,
    spatial_temporal_covariance,
)


def rf_mv(raw, x, z, angle, fs, c, pitch, t0, loading, ratio, temporal_win, device, f_number=1.5, dynamic_aperture=True, interp="cubic", fbss=True, complex_weights=False):
    real = torch.as_tensor(raw, device=device)
    imag = torch.as_tensor(hilbert(raw, axis=0).imag.copy(), device=device) if complex_weights else torch.zeros_like(real)
    xs = torch.as_tensor(x, device=device)
    positions = (torch.arange(raw.shape[1], device=device) - (raw.shape[1] - 1) / 2) * pitch
    angle_t = torch.tensor(angle, device=device)
    channel = torch.arange(raw.shape[1], device=device)[None, :, None]
    offsets = torch.arange(-(temporal_win // 2), temporal_win // 2 + 1, device=device)
    center = torch.argmin(torch.abs(xs[:, None] - positions[None, :]), dim=1)
    result = np.empty((len(z), len(x)), dtype=np.complex64 if complex_weights else np.float32)
    with torch.no_grad():
        for row, depth in enumerate(z):
            aperture = dynamic_aperture_channel_count(float(depth), f_number, pitch, raw.shape[1], dynamic_aperture)
            length = min(max(int(aperture * ratio), 2), aperture)
            start = (center - aperture // 2).clamp(0, raw.shape[1] - aperture)
            active = start[:, None, None] + torch.arange(aperture, device=device)[None, :, None]
            active = active.expand(-1, -1, temporal_win)
            matrix_dtype = torch.complex64 if complex_weights else torch.float32
            eye = torch.eye(length, dtype=matrix_dtype, device=device)
            ones = torch.ones((len(x), length, 1), dtype=matrix_dtype, device=device)
            tx = (float(depth) * torch.cos(angle_t) + xs * torch.sin(angle_t)) / c
            rx = torch.sqrt((xs[:, None] - positions[None, :]).square() + float(depth) ** 2) / c
            index = ((tx[:, None] + rx - t0) * fs)[:, :, None] + offsets
            valid = (index >= 0) & (index <= len(raw) - 1)
            ir, iq = interpolate_channel_samples(real, imag, index, channel, interp)
            ir, iq = ir * valid, iq * valid
            ir, iq = ir.gather(1, active), iq.gather(1, active)
            if complex_weights:
                snapshots, cov = spatial_temporal_covariance(torch.complex(ir, iq), length, temporal_win, fbss)
                trace = cov.diagonal(dim1=-2, dim2=-1).real.sum(-1)
            else:
                snapshots, cov = spatial_temporal_covariance(ir, length, temporal_win, fbss)
                trace = cov.diagonal(dim1=-2, dim2=-1).sum(-1)
            eps = torch.finfo(cov.real.dtype if complex_weights else cov.dtype).eps
            cov = cov + (loading * (trace / length).clamp_min(eps))[:, None, None] * eye
            solution = torch.linalg.solve(cov, ones)
            denominator = (ones.mH @ solution).real.clamp_min(eps)
            weights = solution / denominator
            if complex_weights:
                center_snapshots = snapshots[:, :, :, temporal_win // 2].mean(dim=2)
                result[row] = (weights.squeeze(-1).conj() * center_snapshots).sum(dim=1).cpu().numpy()
            else:
                center_real = snapshots[:, :, :, temporal_win // 2].mean(dim=2)
                y_real = (weights.squeeze(-1) * center_real).sum(dim=1)
                result[row] = y_real.cpu().numpy()
            if (row + 1) % 100 == 0:
                emit(None, "progress", f"RF MV {row + 1}/{len(z)}")
    return result


def rf_mv_signed_and_envelope(raw, x, z, angle, fs, c, pitch, t0, loading, ratio, temporal_win, device, f_number=1.5, dynamic_aperture=True, interp="cubic", fbss=True):
    dz = c / (2.0 * fs)
    z_dense = np.arange(float(z[0]), float(z[-1]) + 0.5 * dz, dz, dtype=np.float32)
    signed_dense = rf_mv(raw, x, z_dense, angle, fs, c, pitch, t0, loading, ratio, temporal_win, device, f_number, dynamic_aperture, interp, fbss)
    envelope_dense = np.abs(hilbert(signed_dense, N=2 * len(signed_dense), axis=0)[:len(signed_dense)])
    right = np.searchsorted(z_dense, z, side="right").clip(1, len(z_dense) - 1)
    left = right - 1
    fraction = ((z - z_dense[left]) / (z_dense[right] - z_dense[left])).astype(np.float32)
    signed = signed_dense[left] * (1.0 - fraction[:, None]) + signed_dense[right] * fraction[:, None]
    envelope = envelope_dense[left] * (1.0 - fraction[:, None]) + envelope_dense[right] * fraction[:, None]
    return signed.astype(np.float32), envelope.astype(np.float32)


def estimate_stft_frequency_bands(raw, fs, fc, window_len):
    nperseg = min(1024, len(raw))
    freq, power = welch(raw, fs=fs, axis=0, nperseg=nperseg)
    mean_power = power.mean(axis=1)
    search = (freq >= 0.5 * fc) & (freq <= 1.5 * fc)
    peak_idx = np.flatnonzero(search)[np.argmax(mean_power[search])]
    peak_db = 10 * np.log10(np.maximum(mean_power, 1e-30))
    stft_freq = np.fft.fftfreq(window_len, d=1 / fs)
    positive_bins = np.arange(1, (window_len + 1) // 2)
    bands = {}
    for threshold in (6, 20):
        mask = peak_db >= peak_db[peak_idx] - threshold
        left = right = int(peak_idx)
        while left > 0 and mask[left - 1]:
            left -= 1
        while right + 1 < len(mask) and mask[right + 1]:
            right += 1
        low, high = float(freq[left]), float(freq[right])
        bins = positive_bins[(stft_freq[positive_bins] >= low) & (stft_freq[positive_bins] <= high)]
        if not len(bins):
            bins = np.array([positive_bins[np.argmin(np.abs(stft_freq[positive_bins] - freq[peak_idx]))]])
        bands[threshold] = {
            "peak_hz": float(freq[peak_idx]),
            "band_hz": [low, high],
            "bins": bins.astype(int).tolist(),
            "frequencies_hz": stft_freq[bins].astype(float).tolist(),
        }
    return bands


def stft_frequency_mv(raw, x, z, angle, fs, fc, c, pitch, t0, loading, ratio, temporal_win, device, window_len=16, selected_bins=None, f_number=1.5, dynamic_aperture=True, interp="cubic", fbss=True):
    analytic = hilbert(raw, axis=0)
    real = torch.as_tensor(raw, device=device)
    imag = torch.as_tensor(analytic.imag.copy(), device=device)
    xs = torch.as_tensor(x, device=device)
    positions = (torch.arange(raw.shape[1], device=device) - (raw.shape[1] - 1) / 2) * pitch
    angle_t = torch.tensor(angle, device=device)
    channel = torch.arange(raw.shape[1], device=device)[None, :, None, None]
    center_channel = torch.argmin(torch.abs(xs[:, None] - positions[None, :]), dim=1)
    frame_hop = max(1, round(window_len * 0.1))
    frame_offsets = (torch.arange(temporal_win, device=device) - temporal_win // 2) * frame_hop
    sample_offsets = torch.arange(window_len, device=device) - window_len // 2
    window = torch.hann_window(window_len, periodic=False, device=device)
    window_sum = window.sum()
    frequencies = torch.fft.fftfreq(window_len, d=1 / fs).to(device)
    positive_bins = torch.arange(1, (window_len + 1) // 2, device=device)
    if selected_bins is None:
        selected_bins = positive_bins[torch.argsort(torch.abs(frequencies[positive_bins] - fc))[:3]]
    selected_bins = torch.as_tensor(selected_bins, dtype=torch.long, device=device).sort().values
    selected_bin_values = selected_bins.cpu().tolist()
    selected_frequencies = frequencies[selected_bins].cpu().numpy()
    phase = 2 * torch.pi * selected_bins.float() * (window_len // 2) / window_len
    reconstruction_phase = torch.complex(torch.cos(phase), torch.sin(phase))
    result = np.empty((len(z), len(x)), dtype=np.complex64)
    with torch.no_grad():
        for row, depth in enumerate(z):
            aperture = dynamic_aperture_channel_count(float(depth), f_number, pitch, raw.shape[1], dynamic_aperture)
            length = min(max(int(aperture * ratio), 2), aperture)
            start = (center_channel - aperture // 2).clamp(0, raw.shape[1] - aperture)
            active = start[:, None] + torch.arange(aperture, device=device)[None, :]
            active = active[:, :, None, None].expand(-1, -1, temporal_win, window_len)
            eye = torch.eye(length, dtype=torch.complex64, device=device)
            ones = torch.ones((len(x), length, 1), dtype=torch.complex64, device=device)
            tx = (float(depth) * torch.cos(angle_t) + xs * torch.sin(angle_t)) / c
            rx = torch.sqrt((xs[:, None] - positions[None, :]).square() + float(depth) ** 2) / c
            center_sample = ((tx[:, None] + rx - t0) * fs)[:, :, None, None]
            offsets = frame_offsets[:, None] + sample_offsets[None, :]
            sample_index = center_sample + offsets[None, None, :, :]
            valid = (sample_index >= 0) & (sample_index <= len(raw) - 1)
            ir, iq = interpolate_channel_samples(real, imag, sample_index, channel, interp)
            signal = torch.complex(ir, iq) * valid * window
            signal = signal.gather(1, active)
            spectrum = torch.fft.fft(signal, dim=-1)
            row_output = torch.zeros(len(x), dtype=torch.complex64, device=device)
            eps = torch.finfo(torch.float32).eps
            for freq_idx, phase_factor in zip(selected_bin_values, reconstruction_phase):
                snapshots, cov = spatial_temporal_covariance(spectrum[..., freq_idx], length, temporal_win, fbss)
                trace = cov.diagonal(dim1=-2, dim2=-1).real.sum(-1)
                cov = cov + (loading * (trace / length).clamp_min(eps))[:, None, None] * eye
                solution = torch.linalg.solve(cov, ones)
                denominator = (ones.mH @ solution).real.clamp_min(eps)
                weights = solution / denominator
                center = snapshots[:, :, :, temporal_win // 2].mean(dim=2)
                row_output += (weights.squeeze(-1).conj() * center).sum(dim=1) * phase_factor
            result[row] = (row_output / window_sum).cpu().numpy()
            if (row + 1) % 100 == 0:
                emit(None, "progress", f"STFT-FD MV {row + 1}/{len(z)}")
    return result, selected_frequencies

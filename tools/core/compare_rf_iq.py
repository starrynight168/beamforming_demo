import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import hilbert, welch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from algorithms.common import dynamic_aperture_channel_count, interpolate_channel_samples, spatial_temporal_covariance
from algorithms.mv import RowDynamicMVBeamformerIQ


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
                print(f"RF MV {row + 1}/{len(z)}", flush=True)
    return result


def rf_mv_signed_and_envelope(raw, x, z, angle, fs, c, pitch, t0, loading, ratio, temporal_win, device, f_number=1.5, dynamic_aperture=True, interp="cubic", fbss=True):
    dz = c / (2.0 * fs)
    z_dense = np.arange(float(z[0]), float(z[-1]) + 0.5 * dz, dz, dtype=np.float32)
    signed_dense = rf_mv(raw, x, z_dense, angle, fs, c, pitch, t0, loading, ratio, temporal_win, device, f_number, dynamic_aperture, interp, fbss)
    envelope_dense = np.abs(hilbert(signed_dense, axis=0))
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
                print(f"STFT-FD MV {row + 1}/{len(z)}", flush=True)
    return result, selected_frequencies


def main():
    parser = argparse.ArgumentParser(description="Compare RF, analytic-RF, IQ and STFT frequency-domain MV from matching EPFL acquisitions", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--h5_path", type=Path, required=True)
    parser.add_argument("--rf_root", type=Path, help="Directory containing region/acquisition.npz")
    parser.add_argument("--rf_path", type=Path, help="Explicit raw RF NPZ path")
    parser.add_argument("--settings_dir", type=Path, default=ROOT / "data/EPFL/settings")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "results/rf_iq_comparison")
    parser.add_argument("--mv_dl", type=float, default=0.01)
    parser.add_argument("--subarray_ratio", type=float, default=0.25)
    parser.add_argument("--temporal_win", type=int, default=3)
    parser.add_argument("--f_number", type=float, default=1.5)
    parser.add_argument("--no_dynamic_aperture", action="store_true")
    parser.add_argument("--no_fbss", action="store_true")
    parser.add_argument("--interp", choices=("linear", "cubic"), default="cubic")
    parser.add_argument("--dr", type=float, default=60.0)
    parser.add_argument("--display_scale", choices=("independent", "shared"), default="independent")
    parser.add_argument("--iq_gain", type=float, default=2.0, help="IQ amplitude correction for shared display")
    parser.add_argument("--stft_window", type=int, default=16, help="Axial STFT window length in RF samples")
    args = parser.parse_args()
    if args.sample_idx < 0 or args.mv_dl <= 0 or not 0 < args.subarray_ratio <= 1 or args.temporal_win < 1 or args.temporal_win % 2 != 1 or args.f_number <= 0 or args.dr <= 0 or args.iq_gain <= 0 or args.stft_window < 8 or args.stft_window % 2 != 0:
        parser.error("Invalid sample, MV, aperture or display parameters")
    if args.rf_root is None and args.rf_path is None:
        parser.error("Specify --rf_root or --rf_path")
    def resolve(path):
        return path if path.is_absolute() else ROOT / path
    h5_path = resolve(args.h5_path)
    settings = resolve(args.settings_dir)
    args.output = resolve(args.output_dir) / f"{h5_path.stem}_sample_{args.sample_idx}"
    with h5py.File(h5_path, "r") as f:
        if args.sample_idx >= len(f["acquisition_id"]):
            parser.error("sample_idx exceeds H5 sample count")
        acquisition = f["acquisition_id"][args.sample_idx].decode()
        region = f["body_region"][args.sample_idx].decode()
        x = f["x_grid"][:].astype(np.float32)
        z = f["z_grid"][:].astype(np.float32)
        angles_iq = f["angles"][:]
        selected = int(np.argmin(np.abs(angles_iq)))
        angle = float(angles_iq[selected])
        fs, c, pitch, fc = (float(f[k][()]) for k in ("fs", "c", "pitch", "fc"))
        t0 = float(f["time_start_vector"][args.sample_idx, selected])
        scale = float(f["all_scale_ref"][args.sample_idx])
        i = f["all_multi_I"][args.sample_idx, selected:selected+1].astype(np.float32) * scale
        q = f["all_multi_Q"][args.sample_idx, selected:selected+1].astype(np.float32) * scale
    path = resolve(args.rf_path) if args.rf_path else resolve(args.rf_root) / region / (acquisition + ".npz")
    angles = np.load(settings / "steering_angles.npy")
    angle_idx = int(np.argmin(np.abs(angles - angle)))
    if abs(float(angles[angle_idx]) - angle) > 1e-6:
        raise ValueError("RF 与 IQ 角度不一致")
    print(f"读取 {path}", flush=True)
    with np.load(path) as f:
        rf = f["data"][0, angle_idx].T.astype(np.float32).copy()
    time_axis = np.load(settings / "time_axis.npy")
    if len(time_axis) != len(rf) or abs(float(time_axis[0]) - t0) > 1e-9 or not np.isclose(float(time_axis[1] - time_axis[0]), 1 / fs, rtol=1e-5):
        raise ValueError("RF 与 IQ 时间轴不一致")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loading, ratio, temporal_win = args.mv_dl, args.subarray_ratio, args.temporal_win
    dynamic, fbss = not args.no_dynamic_aperture, not args.no_fbss
    spectral_bands = estimate_stft_frequency_bands(rf, fs, fc, args.stft_window)
    for threshold, band in spectral_bands.items():
        print(
            f"RF 频谱峰值 {band['peak_hz'] / 1e6:.3f} MHz; -{threshold} dB 带宽 "
            f"{band['band_hz'][0] / 1e6:.3f}-{band['band_hz'][1] / 1e6:.3f} MHz; "
            f"STFT 频点 {np.array(band['frequencies_hz']) / 1e6}",
            flush=True,
        )
    start = time.time()
    rf_result, rf_envelope = rf_mv_signed_and_envelope(rf, x, z, angle, fs, c, pitch, t0, loading, ratio, temporal_win, device, args.f_number, dynamic, args.interp, fbss)
    print(f"实 RF MV 计算完成 {time.time()-start:.1f}s", flush=True)
    start = time.time()
    analytic_rf_result = rf_mv(rf, x, z, angle, fs, c, pitch, t0, loading, ratio, temporal_win, device, args.f_number, dynamic, args.interp, fbss, complex_weights=True)
    print(f"解析 RF MV 计算完成 {time.time()-start:.1f}s", flush=True)
    options = SimpleNamespace(f_number=args.f_number, dynamic_aperture=dynamic, window="rect", interp=args.interp)
    beamformer = RowDynamicMVBeamformerIQ(z, x, rf.shape[1], pitch, c, fc, fs, t0,
                                         np.array([angle], dtype=np.float32), mv_dl=loading,
                                         fbss=fbss, subarray_ratio=ratio, temporal_win=temporal_win, options=options)
    with torch.no_grad():
        i_out, q_out = beamformer(i, q, np.array([angle], dtype=np.float32), t0, fs)
    iq_result = i_out + 1j * q_out
    stft_results = {}
    for threshold, band in spectral_bands.items():
        start = time.time()
        stft_results[threshold], _ = stft_frequency_mv(
            rf, x, z, angle, fs, fc, c, pitch, t0, loading, ratio, temporal_win, device,
            window_len=args.stft_window, selected_bins=band["bins"], f_number=args.f_number,
            dynamic_aperture=dynamic, interp=args.interp, fbss=fbss,
        )
        print(f"STFT 频域 MV (-{threshold} dB 带) 完成 {time.time()-start:.1f}s", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    db = []
    displayed_results = (
        rf_envelope,
        analytic_rf_result,
        args.iq_gain * iq_result,
        stft_results[6],
        stft_results[20],
    )
    reference = float(rf_envelope.max())
    for result in displayed_results:
        env = np.abs(result)
        norm = float(env.max()) if args.display_scale == "independent" else reference
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Reconstruction contains invalid or zero amplitudes")
        db.append(np.maximum(20 * np.log10(np.maximum(env / norm, 1e-8)), -args.dr))
    corr_rf_iq = float(np.corrcoef(db[0].ravel(), db[2].ravel())[0, 1])
    mae_rf_iq = float(np.mean(np.abs(db[0] - db[2])))
    corr_rf_analytic = float(np.corrcoef(db[0].ravel(), db[1].ravel())[0, 1])
    mae_rf_analytic = float(np.mean(np.abs(db[0] - db[1])))
    corr_rf_stft_6db = float(np.corrcoef(db[0].ravel(), db[3].ravel())[0, 1])
    mae_rf_stft_6db = float(np.mean(np.abs(db[0] - db[3])))
    corr_rf_stft_20db = float(np.corrcoef(db[0].ravel(), db[4].ravel())[0, 1])
    mae_rf_stft_20db = float(np.mean(np.abs(db[0] - db[4])))
    fig, axes = plt.subplots(1, 5, figsize=(25, 6), constrained_layout=True)
    extent = [1000*x[0], 1000*x[-1], 1000*z[-1], 1000*z[0]]
    titles = (
        "RF MV (real weights)",
        "Analytic RF MV (complex weights)",
        "IQ MV (complex weights)",
        f"STFT-FD MV (-6 dB, {len(spectral_bands[6]['bins'])} bins)",
        f"STFT-FD MV (-20 dB, {len(spectral_bands[20]['bins'])} bins)",
    )
    for ax, image, title in zip(axes, db, titles):
        panel = ax.imshow(image, cmap="gray", vmin=-args.dr, vmax=0, extent=extent, aspect="equal")
        ax.set(title=title, xlabel="Lateral (mm)", ylabel="Depth (mm)")
    aperture_label = f"dynamic aperture F/{args.f_number:g}" if dynamic else "full aperture"
    fig.suptitle(f"{acquisition} | {aperture_label}, {np.degrees(angle):.2f} deg | {args.dr:g} dB")
    reference_label = "RF MV peak reference" if args.display_scale == "shared" else "per-image peak reference"
    fig.colorbar(panel, ax=axes, label=f"dB ({reference_label})")
    fig.savefig(args.output / "rf_iq_mv_comparison.png", dpi=180)
    plt.close(fig)
    np.savez_compressed(
        args.output / "rf_iq_mv_data.npz",
        rf=rf_result,
        rf_signed=rf_result,
        rf_envelope=rf_envelope,
        analytic_rf=analytic_rf_result,
        iq=iq_result,
        stft_fd=stft_results[20],
        stft_fd_6db=stft_results[6],
        stft_fd_20db=stft_results[20],
        x=x,
        z=z,
    )
    metadata = dict(acquisition=acquisition, region=region, source_rf=str(path), source_h5=str(h5_path),
                    angle_rad=angle, channels=rf.shape[1], aperture="dynamic" if dynamic else "full", f_number=args.f_number, interpolation=args.interp, fbss=fbss,
                    subarray_ratio=ratio, temporal_win=temporal_win, diagonal_loading=loading,
                    loading_formula="loading*trace(R)/L", tgc=False, dynamic_range_db=args.dr,
                    display_normalization=args.display_scale, iq_display_gain=args.iq_gain,
                    display_reference=("RF MV peak" if args.display_scale == "shared" else "peak of each image"),
                    rf_mv="ordinary MV on signed RF; saved signed RF output, envelope from beamformed RF sampled at dz=c/(2*fs)",
                    analytic_rf_mv="Hilbert analytic RF covariance and complex weights",
                    iq_mv="complex baseband IQ covariance and complex weights",
                    stft_fd_mv="local STFT of analytic RF; per-bin complex MV; inverse spectral sum at window center",
                    stft_window_samples=args.stft_window, rf_spectral_bands=spectral_bands,
                    db_pearson_rf_iq=corr_rf_iq, db_mae_rf_iq=mae_rf_iq,
                    db_pearson_rf_analytic=corr_rf_analytic, db_mae_rf_analytic=mae_rf_analytic,
                    db_pearson_rf_stft_6db=corr_rf_stft_6db, db_mae_rf_stft_6db=mae_rf_stft_6db,
                    db_pearson_rf_stft_20db=corr_rf_stft_20db, db_mae_rf_stft_20db=mae_rf_stft_20db)
    (args.output / "comparison.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

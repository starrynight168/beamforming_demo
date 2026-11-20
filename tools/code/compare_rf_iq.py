import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms import mv
from algorithms.shared_core.common import build_beamformer, run_beamformer
from shared_core.h5_schema import require_h5_schema
from shared_core.geometry import image_extent_mm
from tools.shared.rf_beamforming import (
    estimate_stft_frequency_bands,
    rf_mv,
    rf_mv_signed_and_envelope,
    stft_frequency_mv,
)
from shared_core.plot_style import DIAGNOSTIC, apply_style, draw_bmode
from tools.shared.run_output import ToolOutput


def decode_h5_text(value):
    return (
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
    )


def validate_time_axis(time_axis, sample_count, t0, fs):
    time_axis = np.asarray(time_axis, dtype=np.float64)
    if time_axis.ndim != 1 or len(time_axis) < sample_count or sample_count < 2:
        raise ValueError("RF samples exceed or do not fit the configured time axis")
    time_axis = time_axis[:sample_count]
    if (
        not np.isfinite(time_axis).all()
        or abs(float(time_axis[0]) - t0) > 1e-9
        or not np.allclose(np.diff(time_axis), 1 / fs, rtol=1e-5, atol=1e-12)
    ):
        raise ValueError("RF 与 IQ 时间轴不一致")
    return time_axis


def main():
    parser = argparse.ArgumentParser(
        description="Compare RF, analytic-RF, IQ and STFT frequency-domain MV from matching EPFL acquisitions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--h5_path", type=Path, required=True)
    rf_source = parser.add_mutually_exclusive_group(required=True)
    rf_source.add_argument(
        "--rf_root", type=Path, help="Directory containing region/acquisition.npz"
    )
    rf_source.add_argument("--rf_path", type=Path, help="Explicit raw RF NPZ path")
    parser.add_argument(
        "--settings_dir", type=Path, default=ROOT / "data/EPFL/settings"
    )
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument(
        "--output_dir", type=Path
    )
    parser.add_argument("--mv_dl", type=float, default=0.01)
    parser.add_argument("--subarray_ratio", type=float, default=0.25)
    parser.add_argument("--temporal_win", type=int, default=3)
    parser.add_argument("--f_number", type=float, default=1.5)
    parser.add_argument("--no_dynamic_aperture", action="store_true")
    parser.add_argument("--no_fbss", action="store_true")
    parser.add_argument("--interp", choices=("linear", "cubic"), default="cubic")
    parser.add_argument("--dr", type=float, default=60.0)
    parser.add_argument(
        "--display_scale", choices=("independent", "shared"), default="independent"
    )
    parser.add_argument(
        "--iq_gain",
        type=float,
        default=2.0,
        help="IQ amplitude correction for shared display",
    )
    parser.add_argument(
        "--stft_window",
        type=int,
        default=16,
        help="Axial STFT window length in RF samples",
    )
    args = parser.parse_args()
    continuous_values = (
        args.mv_dl,
        args.subarray_ratio,
        args.f_number,
        args.dr,
        args.iq_gain,
    )
    valid_parameters = (
        args.sample_idx >= 0
        and args.mv_dl >= 0
        and 0 < args.subarray_ratio <= 1
        and args.temporal_win >= 1
        and args.temporal_win % 2 == 1
        and args.f_number > 0
        and args.dr > 0
        and args.iq_gain > 0
        and args.stft_window >= 8
        and args.stft_window % 2 == 0
    )
    if not np.isfinite(continuous_values).all() or not valid_parameters:
        parser.error("Invalid sample, MV, aperture or display parameters")

    def resolve(path):
        path = path.expanduser()
        return path.resolve() if path.is_absolute() else (ROOT / path).resolve()

    h5_path = resolve(args.h5_path)
    settings = resolve(args.settings_dir)
    output_root = resolve(args.output_dir) if args.output_dir else None
    output_root = (
        output_root / f"{h5_path.stem}_sample_{args.sample_idx}"
        if output_root is not None
        else None
    )
    with h5py.File(h5_path, "r") as f:
        require_h5_schema(f)
        iq_fields = ("iq_angles", "iq_fs", "iq_t0", "iq_i", "iq_q", "linear_scale_ref")
        missing = [name for name in iq_fields if name not in f]
        if missing:
            raise ValueError(
                f"H5 lacks IQ fields required for comparison: {', '.join(missing)}"
            )
        if args.sample_idx >= len(f["acquisition_id"]):
            parser.error("sample_idx exceeds H5 sample count")
        acquisition = decode_h5_text(f["acquisition_id"][args.sample_idx])
        region = decode_h5_text(f["body_region"][args.sample_idx])
        x = f["x_grid"][:].astype(np.float32)
        z = f["z_grid"][:].astype(np.float32)
        angles_iq = f["iq_angles"][:]
        selected = int(np.argmin(np.abs(angles_iq)))
        angle = float(angles_iq[selected])
        fs, c, pitch, fc = (float(f[k][()]) for k in ("iq_fs", "c", "pitch", "fc"))
        t0 = float(f["iq_t0"][args.sample_idx, selected])
        scale = float(f["linear_scale_ref"][args.sample_idx])
        channel_count = int(f["num_channels"][()])
        i = (
            f["iq_i"][args.sample_idx, selected : selected + 1].astype(np.float32)
            * scale
        )
        q = (
            f["iq_q"][args.sample_idx, selected : selected + 1].astype(np.float32)
            * scale
        )
    path = (
        resolve(args.rf_path)
        if args.rf_path
        else resolve(args.rf_root) / region / (acquisition + ".npz")
    )
    angles = np.load(settings / "steering_angles.npy")
    angle_idx = int(np.argmin(np.abs(angles - angle)))
    if abs(float(angles[angle_idx]) - angle) > 1e-6:
        raise ValueError("RF 与 IQ 角度不一致")
    with np.load(path) as f:
        if "data" not in f:
            raise ValueError(f"RF NPZ 缺少 data 字段: {path}")
        raw_data = np.asarray(f["data"])
    if (
        raw_data.ndim != 4
        or raw_data.shape[0] != 1
        or raw_data.shape[1] != len(angles)
        or raw_data.shape[2] != channel_count
        or angle_idx >= raw_data.shape[1]
        or raw_data.shape[3] < 2
    ):
        raise ValueError(
            f"RF data 必须为 [1,{len(angles)},{channel_count},T]，实际 {raw_data.shape}"
        )
    if np.iscomplexobj(raw_data) or not np.isfinite(raw_data).all():
        raise ValueError("RF data 必须是有限实数数组")
    rf = raw_data[0, angle_idx].T.astype(np.float32, copy=True)
    time_axis = np.load(settings / "time_axis.npy")
    validate_time_axis(time_axis, len(rf), t0, fs)
    output = ToolOutput("compare_rf_iq", output_root)
    output.log("prepare", f"H5：{h5_path}；样本：{args.sample_idx}。")
    output.log("info", f"RF 输入已验证：{path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loading, ratio, temporal_win = args.mv_dl, args.subarray_ratio, args.temporal_win
    dynamic, fbss = not args.no_dynamic_aperture, not args.no_fbss
    spectral_bands = estimate_stft_frequency_bands(rf, fs, fc, args.stft_window)
    for threshold, band in spectral_bands.items():
        output.log(
            "info",
            f"频谱峰值 {band['peak_hz'] / 1e6:.3f} MHz；-{threshold} dB 带宽 "
            f"{band['band_hz'][0] / 1e6:.3f}–{band['band_hz'][1] / 1e6:.3f} MHz；"
            f"STFT 频点数 {len(band['frequencies_hz'])}。",
        )
    start = time.time()
    rf_result, rf_envelope = rf_mv_signed_and_envelope(
        rf,
        x,
        z,
        angle,
        fs,
        c,
        pitch,
        t0,
        loading,
        ratio,
        temporal_win,
        device,
        args.f_number,
        dynamic,
        args.interp,
        fbss,
    )
    output.log("progress", f"实 RF MV 完成，用时 {time.time() - start:.1f} s。")
    start = time.time()
    analytic_rf_result = rf_mv(
        rf,
        x,
        z,
        angle,
        fs,
        c,
        pitch,
        t0,
        loading,
        ratio,
        temporal_win,
        device,
        args.f_number,
        dynamic,
        args.interp,
        fbss,
        complex_weights=True,
    )
    output.log("progress", f"解析 RF MV 完成，用时 {time.time() - start:.1f} s。")
    options = SimpleNamespace(
        f_number=args.f_number,
        dynamic_aperture=dynamic,
        window="rect",
        interp=args.interp,
        signal_domain="iq",
        mv_dl=loading,
        fbss=fbss,
        subarray_ratio=ratio,
        temporal_win=temporal_win,
    )
    angles_rad = np.array([angle], dtype=np.float32)
    beamformer = build_beamformer(
        mv.RowDynamicMVBeamformerIQ, z, x, rf.shape[1], pitch, c, fc, fs,
        t0, angles_rad, options, device,
    )
    with torch.no_grad():
        i_out, q_out, _ = run_beamformer(
            beamformer, i, q, angles_rad, t0, fs, device
        )
    iq_result = i_out + 1j * q_out
    stft_results = {}
    for threshold, band in spectral_bands.items():
        start = time.time()
        stft_results[threshold], _ = stft_frequency_mv(
            rf,
            x,
            z,
            angle,
            fs,
            fc,
            c,
            pitch,
            t0,
            loading,
            ratio,
            temporal_win,
            device,
            window_len=args.stft_window,
            selected_bins=band["bins"],
            f_number=args.f_number,
            dynamic_aperture=dynamic,
            interp=args.interp,
            fbss=fbss,
        )
        output.log(
            "progress",
            f"STFT 频域 MV（-{threshold} dB 带）完成，用时 {time.time() - start:.1f} s。",
        )
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
    apply_style(DIAGNOSTIC)
    fig, axes = plt.subplots(1, 5, figsize=(25, 6), constrained_layout=True)
    extent = image_extent_mm(x, z)
    titles = (
        "RF MV (real weights)",
        "Analytic RF MV (complex weights)",
        "IQ MV (complex weights)",
        f"STFT-FD MV (-6 dB, {len(spectral_bands[6]['bins'])} bins)",
        f"STFT-FD MV (-20 dB, {len(spectral_bands[20]['bins'])} bins)",
    )
    for ax, image, title in zip(axes, db, titles):
        panel = draw_bmode(ax, image, extent, args.dr)
        ax.set(title=title, xlabel="Lateral (mm)", ylabel="Depth (mm)")
    aperture_label = (
        f"dynamic aperture F/{args.f_number:g}" if dynamic else "full aperture"
    )
    fig.suptitle(
        f"{acquisition} | {aperture_label}, {np.degrees(angle):.2f} deg | {args.dr:g} dB"
    )
    reference_label = (
        "RF MV peak reference"
        if args.display_scale == "shared"
        else "per-image peak reference"
    )
    fig.colorbar(panel, ax=axes, label=f"dB ({reference_label})")
    output.figure("rf_iq_mv_comparison.png", fig, dpi=300)
    np.savez_compressed(
        output.path("data", "rf_iq_mv_data.npz"),
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
    metadata = dict(
        acquisition=acquisition,
        region=region,
        source_rf=str(path),
        source_h5=str(h5_path),
        angle_rad=angle,
        channels=rf.shape[1],
        aperture="dynamic" if dynamic else "full",
        f_number=args.f_number,
        interpolation=args.interp,
        fbss=fbss,
        subarray_ratio=ratio,
        temporal_win=temporal_win,
        diagonal_loading=loading,
        loading_formula="loading*trace(R)/L",
        tgc=False,
        dynamic_range_db=args.dr,
        display_normalization=args.display_scale,
        iq_display_gain=args.iq_gain,
        display_reference=(
            "RF MV peak" if args.display_scale == "shared" else "peak of each image"
        ),
        rf_mv="ordinary MV on signed RF; saved signed RF output, envelope from beamformed RF sampled at dz=c/(2*fs)",
        analytic_rf_mv="Hilbert analytic RF covariance and complex weights",
        iq_mv="complex baseband IQ covariance and complex weights",
        stft_fd_mv="local STFT of analytic RF; per-bin complex MV; inverse spectral sum at window center",
        stft_window_samples=args.stft_window,
        rf_spectral_bands=spectral_bands,
        db_pearson_rf_iq=corr_rf_iq,
        db_mae_rf_iq=mae_rf_iq,
        db_pearson_rf_analytic=corr_rf_analytic,
        db_mae_rf_analytic=mae_rf_analytic,
        db_pearson_rf_stft_6db=corr_rf_stft_6db,
        db_mae_rf_stft_6db=mae_rf_stft_6db,
        db_pearson_rf_stft_20db=corr_rf_stft_20db,
        db_mae_rf_stft_20db=mae_rf_stft_20db,
    )
    output.config(
        {
            "tool": "compare_rf_iq",
            "parameters": {
                "sample_idx": args.sample_idx,
                "angle_rad": angle,
                "f_number": args.f_number,
                "dynamic_aperture": dynamic,
                "interpolation": args.interp,
                "fbss": fbss,
                "subarray_ratio": ratio,
                "temporal_win": temporal_win,
                "diagonal_loading": loading,
                "dynamic_range_db": args.dr,
                "display_normalization": args.display_scale,
                "iq_display_gain": args.iq_gain,
                "stft_window_samples": args.stft_window,
            },
            "inputs": {
                "h5_path": str(h5_path),
                "rf_path": str(path),
                "settings_dir": str(settings),
            },
            "details": metadata,
        },
    )
    output.csv(
        "metrics",
        "rf_iq_metrics.csv",
        [
            {"comparison": pair, "metric": metric, "value": value, "unit": unit}
            for pair, metric, value, unit in (
                ("RF vs IQ", "pearson_db", corr_rf_iq, "1"),
                ("RF vs IQ", "mae_db", mae_rf_iq, "dB"),
                ("RF vs analytic RF", "pearson_db", corr_rf_analytic, "1"),
                ("RF vs analytic RF", "mae_db", mae_rf_analytic, "dB"),
                ("RF vs STFT-FD -6 dB", "pearson_db", corr_rf_stft_6db, "1"),
                ("RF vs STFT-FD -6 dB", "mae_db", mae_rf_stft_6db, "dB"),
                ("RF vs STFT-FD -20 dB", "pearson_db", corr_rf_stft_20db, "1"),
                ("RF vs STFT-FD -20 dB", "mae_db", mae_rf_stft_20db, "dB"),
            )
        ],
    )
    output.log("done", f"图像、数据和指标已保存到 {output.root}。")


if __name__ == "__main__":
    main()

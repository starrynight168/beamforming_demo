import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import algorithms.mv as mv
try:
    from data.scripts.simulate_point_targets import load_reference, simulate_rf
except ModuleNotFoundError:
    load_reference = None
    simulate_rf = None
from tools.core.compare_rf_iq import rf_mv, rf_mv_signed_and_envelope, stft_frequency_mv

METHODS = ("RF MV", "Analytic RF MV", "IQ MV", "STFT-FD MV")
METRICS = ("lateral_fwhm_mm", "lateral_pslr_db", "lateral_islr_db", "lateral_peak_error_mm",
           "axial_fwhm_mm", "axial_pslr_db", "axial_islr_db", "axial_peak_error_mm")


def scaled_grid(grid, scale):
    if scale == 1:
        return grid.copy()
    result = np.linspace(float(grid[0]), float(grid[-1]), (len(grid) - 1) * scale + 1, dtype=np.float32)
    result[::scale] = grid
    return result


def measure_profile(profile, coordinates, expected, roi_half, peak_half, smooth_sigma=1.0):
    profile = np.asarray(profile, dtype=np.float64)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    ids = np.flatnonzero(np.abs(coordinates - expected) <= roi_half)
    if len(ids) < 5 or not np.isfinite(profile[ids]).all():
        return (float("nan"),) * 4
    values, positions = profile[ids], coordinates[ids]
    near = np.abs(positions - expected) <= peak_half
    if not near.any():
        return (float("nan"),) * 4
    peak_idx = int(np.flatnonzero(near)[np.argmax(values[near])])
    peak = float(values[peak_idx])
    if not np.isfinite(peak) or peak <= 0:
        return (float("nan"),) * 4
    half = peak * 0.5
    left = peak_idx
    while left > 0 and values[left] > half:
        left -= 1
    right = peak_idx
    while right < len(values) - 1 and values[right] > half:
        right += 1

    def crossing(a, b):
        if values[b] == values[a]:
            return float((positions[a] + positions[b]) * 0.5)
        return float(positions[a] + (half - values[a]) * (positions[b] - positions[a]) / (values[b] - values[a]))

    fwhm = float("nan")
    if left < peak_idx and right > peak_idx and values[left] <= half and values[right] <= half:
        fwhm = crossing(right, right - 1) - crossing(left, left + 1)
    minima, _ = find_peaks(-gaussian_filter1d(values, smooth_sigma))
    left_min = int(minima[minima < peak_idx][-1]) if np.any(minima < peak_idx) else 0
    right_min = int(minima[minima > peak_idx][0]) if np.any(minima > peak_idx) else len(values) - 1
    main_lobe = np.zeros(len(values), dtype=bool)
    main_lobe[left_min:right_min + 1] = True
    side_peak = float(values[~main_lobe].max()) if np.any(~main_lobe) else 0.0
    pslr = 20 * np.log10(max(side_peak, 1e-15) / peak)
    main_energy = float(np.sum(values[main_lobe] ** 2))
    side_energy = float(np.sum(values[~main_lobe] ** 2))
    islr = 10 * np.log10(max(side_energy, 1e-30) / max(main_energy, 1e-30))
    return fwhm * 1e3, pslr, islr, float(positions[peak_idx] - expected) * 1e3


def point_profiles(method, raw, iq_i, iq_q, xs, target_x, target_z, geom, args, loading, device):
    c, fs, fc, pitch, t0 = (geom[key] for key in ("c", "fs", "fc", "pitch", "t0"))
    angle = float(geom["angles"][0])
    z_grid = geom["z_grid"].astype(np.float32)
    z_lateral = np.array([z_grid[np.argmin(np.abs(z_grid - target_z))]], dtype=np.float32)
    z_axial = z_grid[np.abs(z_grid - target_z) <= args.axial_roi_mm / 1000].astype(np.float32)
    x_axial = np.array([target_x], dtype=np.float32)
    dynamic = not args.no_dynamic_aperture
    fbss = not args.no_fbss

    if method == "RF MV":
        dz = c / (2.0 * fs)
        z_fine = np.arange(max(float(z_grid[0]), target_z - args.axial_roi_mm / 1000),
                           min(float(z_grid[-1]), target_z + args.axial_roi_mm / 1000) + 0.5 * dz, dz,
                           dtype=np.float32)
        _, envelope = rf_mv_signed_and_envelope(raw, xs, z_fine, angle, fs, c, pitch, t0, loading,
                                                args.subarray_ratio, args.temporal_win, device,
                                                args.f_number, dynamic, args.interp, fbss)
        ix = int(np.argmin(np.abs(xs - target_x)))
        iz = int(np.argmin(np.abs(z_fine - target_z)))
        axial = np.abs(z_fine - target_z) <= args.axial_roi_mm / 1000
        return envelope[iz], xs, envelope[axial, ix], z_fine[axial]

    if method == "Analytic RF MV":
        lateral = rf_mv(raw, xs, z_lateral, angle, fs, c, pitch, t0, loading, args.subarray_ratio,
                        args.temporal_win, device, args.f_number, dynamic, args.interp, fbss,
                        complex_weights=True)
        axial = rf_mv(raw, x_axial, z_axial, angle, fs, c, pitch, t0, loading, args.subarray_ratio,
                      args.temporal_win, device, args.f_number, dynamic, args.interp, fbss,
                      complex_weights=True)
    elif method == "IQ MV":
        options = SimpleNamespace(f_number=args.f_number, dynamic_aperture=dynamic,
                                  window="rect", interp=args.interp)

        def beam(i_data, q_data, x_grid, z_values):
            beamformer = mv.RowDynamicMVBeamformerIQ(
                z_values, x_grid, geom["channels"], pitch, c, fc, fs, t0,
                np.array([angle], dtype=np.float32), mv_dl=loading, fbss=fbss,
                subarray_ratio=args.subarray_ratio, temporal_win=args.temporal_win, options=options,
            )
            with torch.no_grad():
                out_i, out_q = beamformer(i_data, q_data, np.array([angle], dtype=np.float32), t0, fs)
            return out_i + 1j * out_q

        lateral = beam(iq_i, iq_q, xs, z_lateral)
        axial = beam(iq_i, iq_q, x_axial, z_axial)
    else:
        lateral, _ = stft_frequency_mv(
            raw, xs, z_lateral, angle, fs, fc, c, pitch, t0, loading, args.subarray_ratio,
            args.temporal_win, device, window_len=args.stft_window, selected_bins=args.stft_bins,
            f_number=args.f_number, dynamic_aperture=dynamic, interp=args.interp, fbss=fbss,
        )
        axial, _ = stft_frequency_mv(
            raw, x_axial, z_axial, angle, fs, fc, c, pitch, t0, loading, args.subarray_ratio,
            args.temporal_win, device, window_len=args.stft_window, selected_bins=args.stft_bins,
            f_number=args.f_number, dynamic_aperture=dynamic, interp=args.interp, fbss=fbss,
        )
    return np.abs(np.asarray(lateral).reshape(-1)), xs, np.abs(np.asarray(axial).reshape(-1)), z_axial


def summarize(rows):
    summary = {}
    keys = sorted({(row["method"], row["grid_scale"], row["mv_dl"]) for row in rows})
    for method, scale, loading in keys:
        subset = [row for row in rows if row["method"] == method and row["grid_scale"] == scale and row["mv_dl"] == loading]
        summary_key = f"{method}|scale={scale}|mv_dl={loading:g}"
        summary[summary_key] = {"n": len(subset)}
        for metric in METRICS:
            values = np.asarray([row[metric] for row in subset], dtype=float)
            values = values[np.isfinite(values)]
            summary[summary_key][metric] = {
                "n": int(len(values)),
                "median": float(np.median(values)) if len(values) else None,
                "q25": float(np.percentile(values, 25)) if len(values) else None,
                "q75": float(np.percentile(values, 75)) if len(values) else None,
            }
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="测量单点目标的分辨率、旁瓣和定位误差，并比较信号域、网格密度与 MV 对角加载。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--h5_path", type=Path, default=ROOT / "data/point_targets.h5")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "results/point_target_metrics")
    parser.add_argument("--sample_indices", type=int, nargs="+", help="只评估指定的 H5 样本；默认使用所有 single-point 场景")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--grid_scales", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    parser.add_argument("--loadings", type=float, nargs="+", default=[0.0])
    parser.add_argument("--f_number", type=float, default=1.5)
    parser.add_argument("--subarray_ratio", type=float, default=0.25)
    parser.add_argument("--temporal_win", type=int, default=9)
    parser.add_argument("--interp", choices=("linear", "cubic"), default="cubic")
    parser.add_argument("--no_dynamic_aperture", action="store_true")
    parser.add_argument("--no_fbss", action="store_true")
    parser.add_argument("--stft_window", type=int, default=16)
    parser.add_argument("--stft_bins", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--lateral_roi_mm", type=float, default=5.0)
    parser.add_argument("--axial_roi_mm", type=float, default=3.0)
    parser.add_argument("--peak_search_mm", type=float, default=1.5)
    args = parser.parse_args()
    if (not args.grid_scales or any(scale < 1 for scale in args.grid_scales)
            or len(set(args.grid_scales)) != len(args.grid_scales)):
        parser.error("grid_scales must be unique positive integers")
    if len(set(args.methods)) != len(args.methods):
        parser.error("methods must be unique")
    if not args.loadings or any(not np.isfinite(value) or value < 0 for value in args.loadings):
        parser.error("loadings must be finite nonnegative values")
    if len(set(args.loadings)) != len(args.loadings):
        parser.error("loadings must be unique")
    numeric = (args.f_number, args.subarray_ratio, args.lateral_roi_mm,
               args.axial_roi_mm, args.peak_search_mm)
    if (not args.methods or not np.isfinite(numeric).all()
            or args.f_number <= 0 or not 0 < args.subarray_ratio <= 1):
        parser.error("methods, f_number and subarray_ratio are invalid")
    if args.temporal_win < 1 or args.temporal_win % 2 != 1:
        parser.error("temporal_win must be a positive odd integer")
    if args.stft_window < 8 or args.stft_window % 2 != 0 or any(
        value < 1 or value >= (args.stft_window + 1) // 2 for value in args.stft_bins
    ):
        parser.error("STFT window must be even and at least 8; bins must be positive and below Nyquist")
    if min(args.lateral_roi_mm, args.axial_roi_mm, args.peak_search_mm) <= 0:
        parser.error("ROI and peak search sizes must be positive")
    return args


def main():
    args = parse_args()
    h5_path = args.h5_path if args.h5_path.is_absolute() else ROOT / args.h5_path
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    geom, config = load_reference(h5_path)
    scenes = config["generation"]["simulation"]["scenes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    with h5py.File(h5_path, "r") as handle:
        ids = handle["acquisition_id"].asstr()[:]
        if len(scenes) != len(ids):
            raise ValueError("Embedded scene metadata does not match H5 sample count")
        selected = [(idx, scene) for idx, scene in enumerate(scenes)
                    if scene.get("kind") == "points" and scene.get("point_layout") == "single"]
        if args.sample_indices is not None:
            requested = set(args.sample_indices)
            available = {idx for idx, _ in selected}
            missing = sorted(requested - available)
            if missing:
                raise ValueError(f"Samples are not single-point scenes: {missing}")
            selected = [(idx, scene) for idx, scene in selected if idx in requested]
        if not selected:
            raise ValueError("No single-point scenes selected")
        for n, (idx, scene) in enumerate(selected, 1):
            target_data = np.asarray(scene["point_targets_x_z_amplitude"], dtype=np.float32)
            if target_data.shape[0] != 1:
                raise ValueError(f"Scene {idx} does not contain exactly one target")
            target_x, target_z = map(float, target_data[0, :2])
            raw, measured_snr = simulate_rf(target_data, geom, int(scene["seed"]),
                                            float(scene["snr_db_target"]), device)
            raw = raw.T.copy()
            scale_ref = float(handle["all_scale_ref"][idx])
            iq_i = handle["all_multi_I"][idx, 0].astype(np.float32)[None] * scale_ref
            iq_q = handle["all_multi_Q"][idx, 0].astype(np.float32)[None] * scale_ref
            for scale in args.grid_scales:
                full_x = scaled_grid(geom["x_grid"].astype(np.float32), scale)
                xs = full_x[np.abs(full_x - target_x) <= args.lateral_roi_mm / 1000]
                for loading in args.loadings:
                    for method in args.methods:
                        lateral, lateral_x, axial, axial_z = point_profiles(
                            method, raw, iq_i, iq_q, xs, target_x, target_z, geom, args, loading, device,
                        )
                        lateral_metrics = measure_profile(
                            lateral, lateral_x, target_x, args.lateral_roi_mm / 1000,
                            args.peak_search_mm / 1000, smooth_sigma=float(scale),
                        )
                        axial_metrics = measure_profile(
                            axial, axial_z, target_z, args.axial_roi_mm / 1000, args.peak_search_mm / 1000,
                        )
                        rows.append({
                            "sample_index": idx, "scene_id": str(ids[idx]), "method": method,
                            "grid_scale": scale, "mv_dl": loading,
                            "dx_mm": float(np.diff(full_x).mean() * 1e3),
                            "target_x_mm": target_x * 1e3, "target_z_mm": target_z * 1e3,
                            "snr_db": measured_snr,
                            **dict(zip(METRICS[:4], lateral_metrics)),
                            **dict(zip(METRICS[4:], axial_metrics)),
                        })
            print(f"point targets {n}/{len(selected)}: {ids[idx]}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "point_target_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    (output_dir / "point_target_summary.json").write_text(
        json.dumps({"parameters": vars(args) | {"h5_path": str(h5_path), "output_dir": str(output_dir)},
                    "summary": summary}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    for method in args.methods:
        for loading in args.loadings:
            values = []
            for scale in args.grid_scales:
                key = f"{method}|scale={scale}|mv_dl={loading:g}"
                value = summary[key]["lateral_fwhm_mm"]["median"]
                values.append(np.nan if value is None else value)
            label = f"{method}, DL={loading:g}"
            ax.plot(args.grid_scales, values, marker="o", label=label)
    ax.set_xticks(args.grid_scales)
    ax.set_xlabel("Lateral grid scale")
    ax.set_ylabel("Median lateral FWHM (mm)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "lateral_fwhm_vs_grid.png", facecolor="white")
    plt.close(fig)
    print(f"Saved metrics to {output_dir}", flush=True)


if __name__ == "__main__":
    main()

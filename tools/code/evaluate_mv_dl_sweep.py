import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms import mv
from evaluation.evaluate import target_resolution
from tools.shared.run_output import (
    ToolOutput,
    save_figure,
    set_plot_style,
    write_csv as write_output_csv,
    write_json,
)
import matplotlib.pyplot as plt


def resolve_path(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def write_csv(path, rows):
    if not rows:
        return
    write_output_csv(path, rows)


def summarize(rows):
    summaries = []
    for dl in sorted({row["mv_dl"] for row in rows}):
        selected = [row for row in rows if row["mv_dl"] == dl]
        detected = sum(bool(row["target_response_detected"]) for row in selected)
        summary = {
            "mv_dl": dl,
            "n_targets": len(selected),
            "n_detected": detected,
            "target_response_detection_rate": detected / len(selected),
        }
        for name, unit in (
            ("FWHM_lateral", "mm"),
            ("PSLR_lateral", "db"),
            ("ISLR_lateral", "db"),
        ):
            values = np.asarray(
                [row[f"{name}_{unit}"] for row in selected], dtype=float
            )
            values = values[np.isfinite(values)]
            summary[f"{name}_n_valid"] = len(values)
            summary[f"{name}_mean_{unit}"] = (
                float(values.mean()) if len(values) else float("nan")
            )
            summary[f"{name}_sd_{unit}"] = (
                float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            )
        summaries.append(summary)
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Sweep MV diagonal loading on PICMUS point targets."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--meta",
        type=Path,
        required=True,
        help="Point-target metadata; h5_sample_idx must match sample-index.",
    )
    parser.add_argument(
        "--output-dir", type=Path
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=1,
        help="Zero-based H5 sample index; 1 matches the existing fine-grid report.",
    )
    parser.add_argument("--grid-factor", type=int, default=8)
    parser.add_argument("--window-mm", type=float, default=1.8)
    parser.add_argument("--subarray-ratio", type=float, default=0.25)
    parser.add_argument("--temporal-win", type=int, default=9)
    parser.add_argument("--dls", default="0,0.0001,0.0003,0.001,0.003,0.01,0.03")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)

    if args.sample_index < 0:
        parser.error("sample-index must be nonnegative (zero-based)")
    if args.grid_factor < 1 or not np.isfinite(args.window_mm) or args.window_mm <= 0:
        parser.error(
            "grid-factor must be positive and window-mm must be finite and positive"
        )
    if args.temporal_win < 1 or args.temporal_win % 2 == 0:
        parser.error("temporal-win must be a positive odd number")
    if not 0 < args.subarray_ratio <= 1:
        parser.error("subarray-ratio must be in (0, 1]")
    try:
        dls = [float(value.strip()) for value in args.dls.split(",")]
    except ValueError:
        parser.error("dls must be comma-separated finite nonnegative values")
    if not dls or any(not np.isfinite(value) or value < 0 for value in dls):
        parser.error("dls must be comma-separated finite nonnegative values")
    if len(dls) != len(set(dls)):
        parser.error("dls must not contain duplicate values")

    args.data = resolve_path(args.data)
    args.meta = resolve_path(args.meta)

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    options = SimpleNamespace(
        f_number=1.5,
        dynamic_aperture=True,
        window="rect",
        interp="cubic",
        signal_domain="iq",
        mv_axis="rx",
        tx_subarray_ratio=None,
    )

    with args.meta.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("h5_sample_idx") != args.sample_index:
        parser.error("meta.h5_sample_idx must match the zero-based sample-index")
    targets = metadata["resolution_targets"]
    if not targets:
        parser.error("meta.resolution_targets must not be empty")
    sample = mv.prepare_beamforming_input(args.data, args.sample_index, "all")
    x_grid = np.linspace(
        float(sample.sample.x_grid[0]),
        float(sample.sample.x_grid[-1]),
        (len(sample.sample.x_grid) - 1) * args.grid_factor + 1,
        dtype=np.float32,
    )
    x_mm_all = x_grid * 1000.0
    z_mm_all = sample.sample.z_grid * 1000.0
    target_windows = []
    for target_idx, target in enumerate(targets, start=1):
        x_ids = np.flatnonzero(np.abs(x_mm_all - target["x_mm"]) < args.window_mm)
        z_ids = np.flatnonzero(np.abs(z_mm_all - target["z_mm"]) < args.window_mm)
        if len(x_ids) < 2 or len(z_ids) < 2:
            parser.error(
                f"target {target_idx} window must contain at least two lateral and depth samples"
            )
        target_windows.append((x_ids, z_ids))
    target_rows = []

    output = ToolOutput("evaluate_mv_dl_sweep", args.output_dir)
    output_dir = output.root

    output.config(
        {
            "tool": "evaluate_mv_dl_sweep",
            "data": str(args.data),
            "meta": str(args.meta),
            "sample_idx": args.sample_index,
            "grid_factor": args.grid_factor,
            "window_mm": args.window_mm,
            "subarray_ratio": args.subarray_ratio,
            "fbss": True,
            "temporal_win": args.temporal_win,
            "dls": dls,
            "device": str(device),
        },
    )
    output.log(
        "prepare",
        f"样本 {args.sample_index}；角度 {len(sample.selected_angles)}；点靶 {len(targets)}；"
        f"网格 {args.grid_factor}×；DL {dls}；设备 {device}。",
    )
    set_plot_style()
    for dl in dls:
        for target_idx, (target, (x_ids, z_ids)) in enumerate(
            zip(targets, target_windows), start=1
        ):
            local_x = x_grid[x_ids]
            local_z = sample.sample.z_grid[z_ids]
            beamformer = mv.RowDynamicMVBeamformerIQ(
                local_z,
                local_x,
                sample.sample.n_elem,
                sample.sample.pitch,
                sample.sample.c,
                sample.sample.fc,
                sample.sample.fs,
                sample.t0,
                sample.selected_angles,
                mv_dl=dl,
                fbss=True,
                subarray_ratio=args.subarray_ratio,
                temporal_win=args.temporal_win,
                options=options,
                device=device,
            )
            out_i, out_q, elapsed = mv.run_beamformer(
                beamformer,
                sample.i_data,
                sample.q_data,
                sample.selected_angles,
                sample.t0,
                sample.sample.fs,
                device,
            )
            image_db = mv.beamformed_image_db(
                out_i,
                out_q,
                local_z,
                local_z,
                sample.sample.fc,
                True,
                0.5,
                "iq",
            )
            metric = target_resolution(
                image_db,
                x_mm_all[x_ids],
                z_mm_all[z_ids],
                target,
                args.window_mm,
                target_idx=1,
            )
            if metric is None:
                raise ValueError(
                    f"target {target_idx} has no measurable ROI on the selected grid"
                )
            target_rows.append(
                {
                    "sample_idx": args.sample_index,
                    "h5_path": str(args.data),
                    "mv_dl": dl,
                    "target_idx": target_idx,
                    "target_x_mm": target["x_mm"],
                    "target_z_mm": target["z_mm"],
                    "grid_factor_x": args.grid_factor,
                    "angle_count": len(sample.selected_angles),
                    "dx_mm": float(np.diff(x_mm_all[x_ids]).mean()),
                    "target_response_detected": metric["target_response_detected"],
                    "FWHM_lateral_mm": metric["FWHM_lateral_mm"],
                    "PSLR_lateral_db": metric["pslr_lateral_db"],
                    "ISLR_lateral_db": metric["islr_lateral_db"],
                    "localization_error_mm": metric["localization_error_mm"],
                    "beamform_seconds": elapsed,
                }
            )
        write_csv(output_dir / "metrics" / "mv_dl_target_metrics.csv", target_rows)
        summaries = summarize(target_rows)
        write_csv(output_dir / "metrics" / "mv_dl_summary.csv", summaries)
        current = next(row for row in summaries if row["mv_dl"] == dl)
        output.log(
            "progress",
            f"DL={dl:g}：FWHM {current['FWHM_lateral_mean_mm']:.5f}"
            f"±{current['FWHM_lateral_sd_mm']:.5f} mm; "
            f"检出 {current['n_detected']}/{current['n_targets']}；"
            f"PSLR {current['PSLR_lateral_mean_db']:.3f} dB；"
            f"ISLR {current['ISLR_lateral_mean_db']:.3f} dB。",
        )

    summaries = summarize(target_rows)
    x = np.arange(len(summaries))
    labels = [f"{row['mv_dl']:g}" for row in summaries]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=160)
    axes[0].errorbar(
        x,
        [r["FWHM_lateral_mean_mm"] for r in summaries],
        yerr=[r["FWHM_lateral_sd_mm"] for r in summaries],
        marker="o",
        capsize=3,
    )
    axes[0].set_ylabel("FWHM (mm)")
    for ax, metric, title in (
        (axes[1], "PSLR_lateral_mean_db", "PSLR (dB)"),
        (axes[2], "ISLR_lateral_mean_db", "ISLR (dB)"),
    ):
        ax.plot(x, [r[metric] for r in summaries], marker="o")
        ax.set_ylabel(title)
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.set_xlabel("MV diagonal loading (dl)")
        ax.grid(alpha=0.25)
    fig.suptitle(
        f"PICMUS sample {args.sample_index}, {len(targets)} point targets, "
        f"{args.grid_factor}x grid; sub={args.subarray_ratio}, temporal win={args.temporal_win}"
    )
    fig.tight_layout()
    save_figure(fig, output_dir / "figures" / "mv_dl_sweep.png")
    write_json(
        output_dir / "run_config.json",
        {
            "tool": "evaluate_mv_dl_sweep",
            "data": str(args.data),
            "meta": str(args.meta),
            "sample_index_h5": args.sample_index,
            "selected_angles_rad": sample.selected_angles.tolist(),
            "grid_factor_x": args.grid_factor,
            "window_mm": args.window_mm,
            "subarray_ratio": args.subarray_ratio,
            "fbss": True,
            "temporal_win": args.temporal_win,
            "dls": dls,
            "f_number": 1.5,
            "dynamic_aperture": True,
            "window": "rect",
            "interp": "cubic",
            "tgc": True,
            "tgc_alpha": 0.5,
            "device": str(device),
        },
    )
    output.log("done", f"结果已保存到 {output_dir}。")


if __name__ == "__main__":
    main()

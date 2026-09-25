"""Speed-of-Sound (SoS) Ablation & Optimization for Ultrasound Beamforming.

This script ablates the speed of sound (c) for single-angle MV beamforming
and evaluates SSIM against multi-angle DAS on in-vivo carotid data.
Supports multi-section / bracketed bisection search and grid sweep.
"""

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root and evaluation to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from algorithms.common import (  # noqa: E402
    INTERP_CHOICES,
    WINDOW_CHOICES,
    load_from_h5,
    nonnegative_float,
    nonnegative_int,
    positive_float,
    positive_int,
    positive_odd_int,
    unit_interval_float,
)
from algorithms.common import tgc_gain  # noqa: E402
from algorithms import mv  # noqa: E402
from evaluation.evaluate import db_to_display, local_ssim, psnr  # noqa: E402


def resolve_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else (ROOT_DIR / path).resolve()


def validate_search_config(c_min, c_max, num_sections, tol, max_iter):
    values = {
        "c_min": c_min,
        "c_max": c_max,
        "tol": tol,
    }
    if any(not np.isfinite(value) for value in values.values()):
        raise ValueError("c_min, c_max and tol must be finite")
    if c_min <= 0 or c_max <= 0 or c_min >= c_max:
        raise ValueError("sound-speed range must satisfy 0 < c_min < c_max")
    if num_sections < 2:
        raise ValueError("num_sections must be at least 2")
    if tol <= 0:
        raise ValueError("tol must be greater than 0")
    if max_iter < 1:
        raise ValueError("max_iter must be greater than 0")


def best_cached_result(evaluator):
    if not evaluator.cache:
        raise RuntimeError("No sound-speed evaluations were completed")
    return max(evaluator.cache.values(), key=lambda result: result["ssim"])


class SpeedOfSoundEvaluator:
    """Evaluates single-angle MV beamforming across different sound speeds against multi-angle DAS."""

    def __init__(
        self,
        h5_path="data/in_vivo.h5",
        sample_idx=0,
        mv_dl=0.0,
        fbss=True,
        subarray_ratio=0.25,
        temporal_win=9,
        f_number=1.5,
        dynamic_aperture=True,
        tgc=True,
        tgc_alpha=0.5,
        window="rect",
        interp="linear",
        dr=60.0,
    ):
        self.h5_path = resolve_path(h5_path)
        if not self.h5_path.is_file():
            raise FileNotFoundError(f"H5 dataset not found: {self.h5_path}")
        if sample_idx < 0:
            raise ValueError("sample_idx must be nonnegative")
        self.sample_idx = int(sample_idx)
        self.mv_dl = mv_dl
        self.fbss = fbss
        self.subarray_ratio = subarray_ratio
        self.temporal_win = temporal_win
        self.f_number = f_number
        self.dynamic_aperture = dynamic_aperture
        self.tgc = tgc
        self.tgc_alpha = tgc_alpha
        self.window = window
        self.interp = interp
        self.dr = dr

        mv.args = argparse.Namespace(
            f_number=self.f_number,
            dynamic_aperture=self.dynamic_aperture,
            window=self.window,
            interp=self.interp,
            tgc=self.tgc,
            tgc_alpha=self.tgc_alpha,
            dr=self.dr,
        )

        # Load H5 dataset
        print(
            f"[Dataset] Loading H5 file: {self.h5_path} (Sample {self.sample_idx})..."
        )
        (
            self.c_nominal,
            self.fc,
            self.fs,
            self.pitch,
            self.n_elem,
            self.angles,
            self.t0,
            self.z_grid,
            self.x_grid,
            self.i_data,
            self.q_data,
            self.gt_data,
            self.has_gt,
        ) = load_from_h5(self.h5_path, self.sample_idx)

        if not self.has_gt or self.gt_data is None:
            raise ValueError(
                f"H5 dataset {self.h5_path} does not contain multi-angle DAS ground truth (all_envdb_norm)!"
            )

        # Multi-angle DAS reference (ground truth)
        self.gt_display = np.asarray(self.gt_data.squeeze(), dtype=np.float32)
        if self.gt_display.ndim != 2:
            raise ValueError(
                f"Expected 2D ground truth display array, got shape {self.gt_display.shape}"
            )
        expected_shape = (len(self.z_grid), len(self.x_grid))
        if self.gt_display.shape != expected_shape:
            raise ValueError(
                f"Ground truth shape {self.gt_display.shape} does not match imaging grid {expected_shape}",
            )

        # Pick single angle closest to 0.0 radians
        self.c_idx = int(np.argmin(np.abs(self.angles)))
        self.single_angle = float(self.angles[self.c_idx])
        self.single_angle_deg = np.degrees(self.single_angle)
        print(
            f"[Imaging] Single-angle selected: index={self.c_idx}, angle={self.single_angle_deg:.2f} deg ({self.single_angle:.4f} rad)"
        )
        print(
            f"[Transducer] {self.n_elem} channels, fc={self.fc / 1e6:.2f} MHz, pitch={self.pitch * 1e3:.3f} mm, nominal c={self.c_nominal:.1f} m/s"
        )
        print(
            f"[Grid] Depth: {self.z_grid[0] * 1e3:.1f} ~ {self.z_grid[-1] * 1e3:.1f} mm ({len(self.z_grid)} pts), Lateral: {self.x_grid[0] * 1e3:.1f} ~ {self.x_grid[-1] * 1e3:.1f} mm ({len(self.x_grid)} pts)"
        )

        # Prepare single-angle slice
        self.sel_angles = self.angles[self.c_idx : self.c_idx + 1]
        self.sel_t0 = self.t0[self.c_idx : self.c_idx + 1]
        self.sel_i = self.i_data[self.c_idx : self.c_idx + 1]
        self.sel_q = self.q_data[self.c_idx : self.c_idx + 1]

        # Cache for evaluated speeds of sound to avoid redundant computation
        self.cache = {}

    @property
    def config(self):
        return {
            "mv_dl": self.mv_dl,
            "fbss": self.fbss,
            "subarray_ratio": self.subarray_ratio,
            "temporal_win": self.temporal_win,
            "f_number": self.f_number,
            "dynamic_aperture": self.dynamic_aperture,
            "tgc": self.tgc,
            "tgc_alpha": self.tgc_alpha,
            "window": self.window,
            "interp": self.interp,
            "dr": self.dr,
        }

    def evaluate_c(self, c_val):
        """Run single-angle MV beamforming with sound speed c_val and compute SSIM vs multi-angle DAS."""
        c_key = round(float(c_val), 3)
        if not np.isfinite(c_key) or c_key <= 0:
            raise ValueError(
                f"sound speed must be a finite positive value, got {c_val!r}"
            )
        if c_key in self.cache:
            return self.cache[c_key]

        start_time = time.perf_counter()
        # Instantiate MV beamformer with candidate sound speed c_val
        beamformer = mv.RowDynamicMVBeamformerIQ(
            self.z_grid,
            self.x_grid,
            self.n_elem,
            self.pitch,
            c_key,
            self.fc,
            self.fs,
            self.sel_t0,
            self.sel_angles,
            mv_dl=self.mv_dl,
            fbss=self.fbss,
            subarray_ratio=self.subarray_ratio,
            temporal_win=self.temporal_win,
        )

        with torch.inference_mode():
            i_out, q_out = beamformer(
                self.sel_i, self.sel_q, self.sel_angles, self.sel_t0, self.fs
            )

        # Envelope detection
        envelope = np.hypot(i_out, q_out)
        if self.tgc:
            tgc_vector = tgc_gain(self.z_grid, self.fc, self.tgc_alpha)
            envelope *= tgc_vector[:, None]

        peak = float(np.max(envelope))
        if not np.isfinite(peak) or peak <= 0:
            raise ValueError(
                f"Beamformer output is invalid for c={c_key:.3f} m/s: peak={peak!r}"
            )
        envelope_norm = envelope / (peak + 1e-12)
        db_img = 20.0 * np.log10(np.clip(envelope_norm, 1e-12, None))
        display = db_to_display(db_img, self.dr)

        # Evaluation metrics vs multi-angle DAS reference
        ssim_val = float(local_ssim(display, self.gt_display, data_range=1.0))
        psnr_val = float(psnr(display, self.gt_display, data_range=1.0))
        gt_db = np.clip(self.gt_display * self.dr - self.dr, -self.dr, 0.0)
        clipped_db = np.clip(db_img, -self.dr, 0.0)
        mae_db = float(np.mean(np.abs(clipped_db - gt_db)))
        elapsed = time.perf_counter() - start_time

        result = {
            "c": c_key,
            "ssim": ssim_val,
            "psnr": psnr_val,
            "mae_db": mae_db,
            "elapsed_sec": elapsed,
            "display": display,
            "db_img": db_img,
        }
        self.cache[c_key] = result
        return result


def multi_section_search(
    evaluator, c_min=1430.0, c_max=1620.0, num_sections=4, tol=2.0, max_iter=8
):
    """Perform multi-section / bracketed bisection search to locate the optimal sound speed."""
    validate_search_config(c_min, c_max, num_sections, tol, max_iter)
    print(f"\n{'=' * 70}")
    print(
        f"  Starting Multi-Section Bracketed Search (Sections: {num_sections}, Tol: {tol:.1f} m/s)"
    )
    print(f"  Initial Search Range: [{c_min:.1f}, {c_max:.1f}] m/s")
    print(f"{'=' * 70}")

    a, b = float(c_min), float(c_max)
    history = []
    iteration = 0

    while (b - a) > tol and iteration < max_iter:
        iteration += 1
        print(
            f"\n--- Iteration {iteration}: Current Interval = [{a:.2f}, {b:.2f}] m/s (width = {b - a:.2f} m/s) ---"
        )
        probe_c = np.linspace(a, b, num_sections + 1)
        probe_results = []
        for idx, c_val in enumerate(probe_c):
            res = evaluator.evaluate_c(c_val)
            probe_results.append(res)
            print(
                f"  Point {idx + 1}/{len(probe_c)}: c = {c_val:7.2f} m/s | SSIM = {res['ssim']:.6f} | PSNR = {res['psnr']:.2f} dB | MAE = {res['mae_db']:.2f} dB ({res['elapsed_sec']:.2f}s)"
            )

        best_idx = int(np.argmax([r["ssim"] for r in probe_results]))
        best_res = probe_results[best_idx]
        print(
            f"  -> Best in iteration {iteration}: c = {best_res['c']:.2f} m/s with SSIM = {best_res['ssim']:.6f}"
        )

        history.append(
            {
                "iteration": iteration,
                "interval_before": [a, b],
                "probes": [
                    {
                        "c": r["c"],
                        "ssim": r["ssim"],
                        "psnr": r["psnr"],
                        "mae_db": r["mae_db"],
                    }
                    for r in probe_results
                ],
                "best_c": best_res["c"],
                "best_ssim": best_res["ssim"],
            }
        )

        # Update bracket around the best probe point
        left_idx = max(0, best_idx - 1)
        right_idx = min(len(probe_c) - 1, best_idx + 1)
        new_a = float(probe_c[left_idx])
        new_b = float(probe_c[right_idx])

        if new_b == new_a:
            break
        a, b = new_a, new_b

    if not evaluator.cache:
        evaluator.evaluate_c(a)
        evaluator.evaluate_c(b)
    overall_best = best_cached_result(evaluator)

    print(f"\n{'=' * 70}")
    print("  Search Completed!")
    print(f"  Optimal Sound Speed : c* = {overall_best['c']:.2f} m/s")
    print(f"  Optimal SSIM        : {overall_best['ssim']:.6f}")
    print(f"  PSNR vs Multi-DAS   : {overall_best['psnr']:.2f} dB")
    print(f"  MAE vs Multi-DAS    : {overall_best['mae_db']:.2f} dB")
    print(f"  Final Bracket Range : [{a:.2f}, {b:.2f}] m/s")
    print(f"{'=' * 70}\n")

    return overall_best, history, (a, b)


def dense_grid_sweep(evaluator, c_min=1430.0, c_max=1620.0, step=10.0):
    """Evaluate sound speeds across a uniform grid for curve visualization."""
    validate_search_config(c_min, c_max, 2, step, 1)
    grid_c = np.arange(c_min, c_max, step, dtype=float)
    if grid_c.size == 0 or grid_c[-1] < c_max:
        grid_c = np.append(grid_c, c_max)
    print(
        f"[Grid Sweep] Scanning {len(grid_c)} points from {c_min:.1f} to {c_max:.1f} m/s (step={step:.1f} m/s)..."
    )
    results = []
    for idx, c_val in enumerate(grid_c):
        res = evaluator.evaluate_c(c_val)
        results.append(res)
        print(
            f"  [{idx + 1:2d}/{len(grid_c):2d}] c = {c_val:7.2f} m/s | SSIM = {res['ssim']:.6f} | PSNR = {res['psnr']:.2f} dB"
        )
    return results


def plot_results(evaluator, search_history, overall_best, output_dir):
    """Generate high-resolution ablation curve and B-mode comparison images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nom_res = evaluator.evaluate_c(evaluator.c_nominal)

    all_res = sorted(evaluator.cache.values(), key=lambda x: x["c"])
    if not all_res:
        raise RuntimeError(
            "Cannot plot results before evaluating at least one sound speed"
        )
    c_list = [r["c"] for r in all_res]
    ssim_list = [r["ssim"] for r in all_res]
    psnr_list = [r["psnr"] for r in all_res]
    mae_list = [r["mae_db"] for r in all_res]

    # 1. Curve Plot: SSIM / PSNR / MAE vs Speed of Sound
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Upper panel: SSIM
    ax1.plot(
        c_list,
        ssim_list,
        "o-",
        color="#1f77b4",
        linewidth=2.0,
        markersize=5,
        label="Single-angle MV vs Multi-angle DAS",
    )
    ax1.axvline(
        evaluator.c_nominal,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label=f"Nominal c = {evaluator.c_nominal:.0f} m/s",
    )
    ax1.axvline(
        overall_best["c"],
        color="#d62728",
        linestyle=":",
        linewidth=1.8,
        label=f"Optimal c* = {overall_best['c']:.1f} m/s",
    )
    ax1.plot(
        overall_best["c"],
        overall_best["ssim"],
        marker="*",
        color="#d62728",
        markersize=14,
        label=f"Max SSIM = {overall_best['ssim']:.4f}",
    )

    # Mark search iteration points
    colors = ["#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]
    for i, step in enumerate(search_history):
        col = colors[i % len(colors)]
        probe_cs = [p["c"] for p in step["probes"]]
        probe_ssims = [p["ssim"] for p in step["probes"]]
        ax1.scatter(
            probe_cs,
            probe_ssims,
            color=col,
            s=35,
            zorder=5,
            label=f"Search Iter {step['iteration']}" if i < 3 else None,
        )

    ax1.set_ylabel("SSIM", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Speed of Sound (SoS) Ablation & Optimization\n(Single-angle MV vs Multi-angle DAS Ground Truth)",
        fontsize=13,
        fontweight="bold",
    )
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=9)

    # Lower panel: PSNR & MAE
    ax2.plot(
        c_list,
        psnr_list,
        "s-",
        color="#2ca02c",
        linewidth=1.8,
        markersize=4,
        label="PSNR (dB)",
    )
    ax2.set_ylabel("PSNR (dB)", color="#2ca02c", fontsize=11, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.grid(True, linestyle="--", alpha=0.6)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(
        c_list,
        mae_list,
        "^-",
        color="#ff7f0e",
        linewidth=1.8,
        markersize=4,
        label="MAE (dB)",
    )
    ax2_twin.set_ylabel("MAE (dB)", color="#ff7f0e", fontsize=11, fontweight="bold")
    ax2_twin.tick_params(axis="y", labelcolor="#ff7f0e")

    ax2.set_xlabel("Speed of Sound c (m/s)", fontsize=12, fontweight="bold")
    ax2.axvline(overall_best["c"], color="#d62728", linestyle=":", linewidth=1.5)

    plt.tight_layout()
    curve_path = output_dir / "sound_speed_vs_ssim_curve.png"
    fig.savefig(curve_path, dpi=200)
    plt.close(fig)
    print(f"[Plot] Saved ablation curve to: {curve_path}")

    # 2. B-Mode Images Comparison
    min_res = min(all_res, key=lambda x: x["c"])
    max_res = max(all_res, key=lambda x: x["c"])

    images_to_show = [
        (
            f"Multi-angle DAS (GT, {len(evaluator.angles)} angles)",
            evaluator.gt_display * evaluator.dr - evaluator.dr,
            "Reference (SSIM=1.0)",
        ),
        (
            f"Single-angle MV (c = {min_res['c']:.0f} m/s)",
            min_res["db_img"],
            f"SSIM = {min_res['ssim']:.4f}",
        ),
        (
            f"Single-angle MV (c = {nom_res['c']:.0f} m/s, Nominal)",
            nom_res["db_img"],
            f"SSIM = {nom_res['ssim']:.4f}",
        ),
        (
            f"Single-angle MV (c* = {overall_best['c']:.0f} m/s, Optimal)",
            overall_best["db_img"],
            f"SSIM = {overall_best['ssim']:.4f}",
        ),
        (
            f"Single-angle MV (c = {max_res['c']:.0f} m/s)",
            max_res["db_img"],
            f"SSIM = {max_res['ssim']:.4f}",
        ),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5.5))
    x_mm = evaluator.x_grid * 1e3
    z_mm = evaluator.z_grid * 1e3
    extent = [x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]]

    for ax, (title, db_map, metric_text) in zip(axes, images_to_show):
        im = ax.imshow(
            db_map,
            cmap="gray",
            vmin=-evaluator.dr,
            vmax=0.0,
            extent=extent,
            aspect="auto",
        )
        ax.set_title(f"{title}\n[{metric_text}]", fontsize=10, fontweight="bold")
        ax.set_xlabel("Lateral (mm)", fontsize=9)
        ax.set_ylabel("Depth (mm)", fontsize=9)

    cbar = fig.colorbar(
        im, ax=axes.ravel().tolist(), orientation="vertical", shrink=0.8, pad=0.015
    )
    cbar.set_label("Dynamic Range (dB)", fontsize=10)

    bmode_path = output_dir / "bmode_comparison.png"
    fig.savefig(bmode_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved B-mode comparison to: {bmode_path}")


def save_results(evaluator, search_history, overall_best, final_bracket, output_dir):
    """Save CSV, JSON and TXT summary reports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nom_res = evaluator.evaluate_c(evaluator.c_nominal)

    # 1. Save all evaluated points to CSV
    csv_path = output_dir / "sound_speed_ablation_results.csv"
    all_res = sorted(evaluator.cache.values(), key=lambda x: x["c"])
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["c_mps", "ssim_vs_gt", "psnr_db_vs_gt", "mae_db_vs_gt", "elapsed_sec"]
        )
        for r in all_res:
            writer.writerow(
                [
                    f"{r['c']:.3f}",
                    f"{r['ssim']:.6f}",
                    f"{r['psnr']:.4f}",
                    f"{r['mae_db']:.4f}",
                    f"{r['elapsed_sec']:.3f}",
                ]
            )
    print(f"[Export] Saved results CSV to: {csv_path}")

    # 2. Save JSON search log
    json_path = output_dir / "sound_speed_search_log.json"
    log_data = {
        "dataset": str(evaluator.h5_path),
        "sample_idx": evaluator.sample_idx,
        "nominal_c": float(evaluator.c_nominal),
        "optimal_c": float(overall_best["c"]),
        "optimal_ssim": float(overall_best["ssim"]),
        "optimal_psnr": float(overall_best["psnr"]),
        "optimal_mae_db": float(overall_best["mae_db"]),
        "final_bracket": [float(final_bracket[0]), float(final_bracket[1])],
        "evaluated_count": len(all_res),
        "config": evaluator.config,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_device": str(mv.device),
        },
        "search_history": search_history,
        "all_evaluated_points": [
            {"c": r["c"], "ssim": r["ssim"], "psnr": r["psnr"], "mae_db": r["mae_db"]}
            for r in all_res
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    print(f"[Export] Saved search log JSON to: {json_path}")

    # 3. Save Summary TXT
    summary_path = output_dir / "summary.txt"
    ssim_delta = overall_best["ssim"] - nom_res["ssim"]
    ssim_pct = ssim_delta / nom_res["ssim"] * 100 if nom_res["ssim"] else float("nan")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  Speed-of-Sound (SoS) Ablation & Optimization Report\n")
        f.write("=" * 60 + "\n")
        f.write(
            f"Dataset             : {evaluator.h5_path} (Sample {evaluator.sample_idx})\n"
        )
        f.write(
            f"Imaging Mode        : Single-angle MV ({evaluator.single_angle_deg:.2f} deg, angle_idx={evaluator.c_idx})\n"
        )
        f.write(
            f"Reference Baseline  : Multi-angle DAS ({len(evaluator.angles)} angles CPWC ground truth)\n"
        )
        f.write(f"Nominal Sound Speed : {evaluator.c_nominal:.1f} m/s\n")
        f.write(f"  - Nominal SSIM    : {nom_res['ssim']:.6f}\n")
        f.write(f"  - Nominal PSNR    : {nom_res['psnr']:.2f} dB\n")
        f.write(f"  - Nominal MAE     : {nom_res['mae_db']:.2f} dB\n")
        f.write("-" * 60 + "\n")
        f.write(f"Optimal Sound Speed : {overall_best['c']:.2f} m/s\n")
        f.write(f"  - Optimal SSIM    : {overall_best['ssim']:.6f}\n")
        f.write(f"  - Optimal PSNR    : {overall_best['psnr']:.2f} dB\n")
        f.write(f"  - Optimal MAE     : {overall_best['mae_db']:.2f} dB\n")
        f.write(
            f"Final Search Bracket: [{final_bracket[0]:.2f}, {final_bracket[1]:.2f}] m/s\n"
        )
        f.write(f"SSIM Improvement    : {ssim_delta:+.6f} ({ssim_pct:+.2f}%)\n")
        f.write("=" * 60 + "\n")
    print(f"[Export] Saved summary report to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Speed of Sound Ablation & Optimization for Single-angle MV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--h5_path", default="data/invivo_15002.h5", help="Path to in-vivo H5 dataset"
    )
    parser.add_argument(
        "--sample_idx", type=nonnegative_int, default=0, help="H5 sample index"
    )
    parser.add_argument(
        "--c_min",
        type=positive_float,
        default=1430.0,
        help="Minimum sound speed in search (m/s)",
    )
    parser.add_argument(
        "--c_max",
        type=positive_float,
        default=1620.0,
        help="Maximum sound speed in search (m/s)",
    )
    parser.add_argument(
        "--sections",
        type=positive_int,
        default=4,
        help="Number of sections per bracket iteration",
    )
    parser.add_argument(
        "--tol",
        type=positive_float,
        default=2.0,
        help="Interval tolerance for search termination (m/s)",
    )
    parser.add_argument(
        "--max_iter", type=positive_int, default=6, help="Maximum search iterations"
    )
    parser.add_argument(
        "--grid_step",
        type=nonnegative_float,
        default=15.0,
        help="Step for dense curve sweep (0 to disable)",
    )
    parser.add_argument(
        "--output_dir", default="results/ablation_sound_speed", help="Output directory"
    )
    parser.add_argument(
        "--mv_dl",
        type=nonnegative_float,
        default=0.0,
        help="MV diagonal loading factor",
    )
    parser.add_argument(
        "--fbss",
        action="store_true",
        default=True,
        help="Enable forward-backward spatial smoothing",
    )
    parser.add_argument(
        "--no_fbss",
        dest="fbss",
        action="store_false",
        help="Disable forward-backward spatial smoothing",
    )
    parser.add_argument(
        "--subarray_ratio",
        type=unit_interval_float,
        default=0.25,
        help="MV subarray ratio",
    )
    parser.add_argument(
        "--temporal_win",
        type=positive_odd_int,
        default=9,
        help="Temporal covariance window; must be odd",
    )
    parser.add_argument(
        "--f_number", type=positive_float, default=1.5, help="Receive aperture F-number"
    )
    parser.add_argument(
        "--dynamic_aperture",
        action="store_true",
        default=True,
        help="Enable dynamic receive aperture",
    )
    parser.add_argument(
        "--no_dynamic_aperture",
        dest="dynamic_aperture",
        action="store_false",
        help="Disable dynamic receive aperture",
    )
    parser.add_argument(
        "--tgc", action="store_true", default=True, help="Enable time-gain compensation"
    )
    parser.add_argument(
        "--no_tgc",
        dest="tgc",
        action="store_false",
        help="Disable time-gain compensation",
    )
    parser.add_argument(
        "--tgc_alpha",
        type=nonnegative_float,
        default=0.5,
        help="TGC attenuation coefficient",
    )
    parser.add_argument(
        "--window",
        choices=WINDOW_CHOICES,
        default="rect",
        help="Receive aperture window",
    )
    parser.add_argument(
        "--interp",
        choices=INTERP_CHOICES,
        default="linear",
        help="Channel interpolation method",
    )
    parser.add_argument(
        "--dr", type=positive_float, default=60.0, help="Display dynamic range (dB)"
    )
    args = parser.parse_args()

    try:
        validate_search_config(
            args.c_min, args.c_max, args.sections, args.tol, args.max_iter
        )
    except ValueError as exc:
        parser.error(str(exc))

    evaluator = SpeedOfSoundEvaluator(
        h5_path=args.h5_path,
        sample_idx=args.sample_idx,
        mv_dl=args.mv_dl,
        fbss=args.fbss,
        subarray_ratio=args.subarray_ratio,
        temporal_win=args.temporal_win,
        f_number=args.f_number,
        dynamic_aperture=args.dynamic_aperture,
        tgc=args.tgc,
        tgc_alpha=args.tgc_alpha,
        window=args.window,
        interp=args.interp,
        dr=args.dr,
    )

    # 1. Run Bracketed Multi-section / Bisection Search
    overall_best, history, final_bracket = multi_section_search(
        evaluator,
        c_min=args.c_min,
        c_max=args.c_max,
        num_sections=args.sections,
        tol=args.tol,
        max_iter=args.max_iter,
    )

    # 2. Dense Grid Sweep for Complete Curve Visualization
    if args.grid_step > 0:
        dense_grid_sweep(
            evaluator, c_min=args.c_min, c_max=args.c_max, step=args.grid_step
        )
        overall_best = best_cached_result(evaluator)

    # 3. Output Figures and Data
    sample_name = (
        "carotid_cross" if args.sample_idx == 0 else f"carotid_sample_{args.sample_idx}"
    )
    out_dir = resolve_path(args.output_dir) / sample_name
    plot_results(evaluator, history, overall_best, out_dir)
    save_results(evaluator, history, overall_best, final_bracket, out_dir)

    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()

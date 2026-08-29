"""Plot evaluation outputs written by evaluate.py.

Default input:
    results/<scene>/metrics
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb, to_hex

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.evaluate import db_to_display, load_grids, read_phantom

COMPARISON_VALUE_12 = 12
COMPARISON_VALUE_4 = 4
COMPARISON_VALUE_5 = 5
COMPARISON_VALUE_6 = 6
COMPARISON_VALUE_8 = 8


METHOD_PALETTE = [
    "#4C72B0",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#CCB974",
    "#64B5CD",
    "#E17C05",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
    "#2A9D8F",
    "#E76F51",
]


def get_color(method_name):
    name = method_name.upper().replace("-", "").replace(" ", "")
    mapping = {
        "GT": "#333333",  # 深灰
        "DAS": "#4C72B0",  # 蓝色
        "MV": "#55A868",  # 绿色
        "ESBMV": "#8172B3",  # 紫色
        "GCFMV": "#64B5CD",  # 青色
        "CMSAW": "#FF7F0E",  # 橙色
        "FDMAS": "#C44E52",  # 红色
    }
    if name in mapping:
        return mapping[name]
    index = int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16) % len(METHOD_PALETTE)
    return METHOD_PALETTE[index]


def build_method_colors(methods):
    methods = list(dict.fromkeys(methods))
    result = {}
    used = set()
    for method in methods:
        color = get_color(method)
        if color not in used:
            result[method] = color
            used.add(color)
    available = []
    for colormap_name in ("tab20", "tab20b", "tab20c"):
        for color in plt.get_cmap(colormap_name).colors:
            color = to_hex(color)
            if color not in used and color not in available:
                available.append(color)
    index = 0
    while len(result) + len(available) < len(methods):
        color = to_hex(hsv_to_rgb((index * 0.61803398875) % 1.0, 0.68, 0.86))
        index += 1
        if color not in used and color not in available:
            available.append(color)
    for method in methods:
        if method not in result:
            if not available:
                raise RuntimeError("无法为所有评估方法生成唯一颜色")
            result[method] = available.pop(0)
            used.add(result[method])
    return result


def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser(description="Plot evaluation CSV outputs.")
    parser.add_argument("--metrics_dir", default=os.path.join("results", "metrics"))
    parser.add_argument("--dr", type=float)
    return parser.parse_args()


def resolve(path):
    """Execute resolve."""
    if os.path.isabs(path):
        return path
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir),
    )
    return os.path.abspath(os.path.join(project_root, path))


def read_rows(path):
    """Read rows."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if rows and rows[0].get("status") == "not_applicable":
        return []
    out = []
    for row in rows:
        converted = {}
        for key, value in row.items():
            if value is None or value.strip() == "":
                converted[key] = np.nan
                continue
            try:
                converted[key] = float(value)
            except ValueError:
                converted[key] = value
        out.append(converted)
    return out


def finite_value(value):
    """Return a finite metric value or NaN."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def metric_precision(title, key):
    """Execute metric precision."""
    key_l = key.lower()
    title_l = title.lower()
    if "gcnr" in key_l or "ssim" in key_l or "ratio" in key_l or "rate" in title_l:
        return 4
    if "fwhm" in key_l or "distortion" in key_l or "contrast" in key_l:
        return 4
    if "cnr" in key_l or "snr" in key_l or "enl" in key_l:
        return 3
    if "psnr" in key_l or key_l.endswith("_db") or "(db)" in title_l:
        return 2
    return 3


def padded_axis_limits(values, include_zero=True, pad_fraction=0.20):
    """Execute padded axis limits."""
    finite = np.asarray([finite_value(v) for v in values], dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if include_zero:
        vmin = min(vmin, 0.0)
        vmax = max(vmax, 0.0)
    span = vmax - vmin
    if span <= 0:
        span = max(abs(vmax), 1.0)
    return vmin - pad_fraction * span, vmax + pad_fraction * span


def apply_value_padding(ax, values, horizontal=False):
    """Execute apply value padding."""
    limits = padded_axis_limits(values, include_zero=True, pad_fraction=0.18)
    if limits is None:
        return
    if horizontal:
        ax.set_xlim(*limits)
    else:
        ax.set_ylim(*limits)


def save_bar_plot(path, metrics_to_plot):
    """Save bar plot."""
    if not metrics_to_plot:
        return

    max_plots_per_page = 6
    if len(metrics_to_plot) > max_plots_per_page:
        root, ext = os.path.splitext(path)
        for page_idx, start in enumerate(
            range(0, len(metrics_to_plot), max_plots_per_page),
            start=1,
        ):
            page_path = path if page_idx == 1 else f"{root}_page{page_idx}{ext}"
            save_bar_plot(
                page_path,
                metrics_to_plot[start : start + max_plots_per_page],
            )
        return

    n_plots = len(metrics_to_plot)
    max_methods = max(len(item[3]) for item in metrics_to_plot)
    cols = min(n_plots, 2) if max_methods > COMPARISON_VALUE_8 else min(n_plots, 3)
    rows_grid = (n_plots + cols - 1) // cols
    panel_width = max(7.8, 0.32 * max_methods + 3.6) if max_methods > COMPARISON_VALUE_6 else 5.0
    panel_height = max(4.6, 0.34 * max_methods + 1.8) if max_methods > COMPARISON_VALUE_6 else 4.0
    fig, axes = plt.subplots(
        rows_grid,
        cols,
        figsize=(panel_width * cols, panel_height * rows_grid),
        dpi=150,
    )
    axes = np.asarray([axes]).reshape(-1) if n_plots == 1 else axes.flatten()

    for i, (title, key, vals, methods) in enumerate(metrics_to_plot):
        ax = axes[i]
        color_map = build_method_colors(methods)
        colors = [color_map[m] for m in methods]
        use_horizontal = len(methods) > COMPARISON_VALUE_6
        positions = np.arange(len(methods))
        if use_horizontal:
            bars = ax.barh(
                positions,
                vals,
                color=colors,
                height=0.62,
                edgecolor="black",
                linewidth=0.8,
            )
            ax.set_yticks(positions)
            ax.set_yticklabels(methods, fontsize=8 if len(methods) <= COMPARISON_VALUE_12 else 7)
            ax.invert_yaxis()
        else:
            bars = ax.bar(
                methods,
                vals,
                color=colors,
                width=0.5,
                edgecolor="black",
                linewidth=0.8,
            )
            ax.tick_params(axis="x", labelrotation=25 if len(methods) <= COMPARISON_VALUE_5 else 35)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.grid(axis="x" if use_horizontal else "y", linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        apply_value_padding(ax, vals, horizontal=use_horizontal)

        for bar in bars:
            value = finite_value(bar.get_width() if use_horizontal else bar.get_height())
            if np.isfinite(value):
                label = f"{value:.{metric_precision(title, key)}f}"
                if use_horizontal:
                    ax.annotate(
                        label,
                        xy=(value, bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0),
                        textcoords="offset points",
                        ha="left",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                    )
                else:
                    ax.annotate(
                        label,
                        xy=(bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        fontweight="bold",
                    )

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.tight_layout(pad=1.4, h_pad=2.0, w_pad=1.6)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_group_metric_plot(path, group_rows, metric_keys, title_prefix):
    """Save group metric plot."""
    if not group_rows:
        return
    groups = []
    methods = []
    for row in group_rows:
        if row["group"] not in groups:
            groups.append(row["group"])
        if row["method"] not in methods:
            methods.append(row["method"])
    max_methods_per_page = 12
    if len(methods) > max_methods_per_page:
        baseline_names = {"GT", "DAS", "MV", "ESBMV", "GCFMV", "CMSAW", "FDMAS"}
        baselines = [method for method in methods if str(method).upper() in baseline_names]
        model_methods = [method for method in methods if method not in baselines]
        chunk_size = max(max_methods_per_page - len(baselines), 1)
        model_groups: list[list[str]] = []
        group_keys: list[str] = []
        for method in model_methods:
            method_text = str(method)
            group_key = method_text.split("_", 1)[0] if "_" in method_text else method_text
            if group_key not in group_keys:
                group_keys.append(group_key)
                model_groups.append([])
            model_groups[group_keys.index(group_key)].append(method)
        chunks: list[list[str]] = []
        current: list[str] = []
        for model_group in model_groups:
            if len(model_group) > chunk_size:
                if current:
                    chunks.append(current)
                    current = []
                chunks.extend(
                    model_group[start : start + chunk_size]
                    for start in range(0, len(model_group), chunk_size)
                )
            elif current and len(current) + len(model_group) > chunk_size:
                chunks.append(current)
                current = list(model_group)
            else:
                current.extend(model_group)
        if current:
            chunks.append(current)
        root, ext = os.path.splitext(path)
        for page_idx, chunk in enumerate(chunks, start=1):
            page_methods = set(baselines + chunk)
            page_rows = [row for row in group_rows if row["method"] in page_methods]
            page_path = path if page_idx == 1 else f"{root}_page{page_idx}{ext}"
            page_title = title_prefix if page_idx == 1 else f"{title_prefix} (page {page_idx}/{len(chunks)})"
            save_group_metric_plot(page_path, page_rows, metric_keys, page_title)
        return
    available_metrics = [
        (key, title)
        for key, title in metric_keys
        if any(np.isfinite(finite_value(row.get(key, np.nan))) for row in group_rows)
    ]
    if not available_metrics:
        return

    color_map = build_method_colors(methods)
    n_plots = len(available_metrics)
    cols = min(n_plots, 2)
    rows_grid = (n_plots + cols - 1) // cols
    panel_width = max(6.2, 1.15 * len(groups) + 0.35 * len(methods) + 3.0)
    panel_height = 4.8 if len(methods) <= COMPARISON_VALUE_8 else 5.4
    fig, axes = plt.subplots(
        rows_grid,
        cols,
        figsize=(panel_width * cols, panel_height * rows_grid),
        dpi=160,
    )
    axes = np.asarray(axes).reshape(-1)

    x = np.arange(len(groups))
    bar_width = min(0.78 / max(len(methods), 1), 0.13)
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * bar_width
    show_value_labels = len(methods) <= COMPARISON_VALUE_8 and len(groups) <= COMPARISON_VALUE_4

    for ax_idx, (key, title) in enumerate(available_metrics):
        ax = axes[ax_idx]
        all_values = []
        for method_idx, method in enumerate(methods):
            values = []
            for group in groups:
                row = next(
                    (r for r in group_rows if r["method"] == method and r["group"] == group),
                    None,
                )
                values.append(finite_value(row.get(key, np.nan)) if row else np.nan)
            all_values.extend(values)
            bars = ax.bar(
                x + offsets[method_idx],
                values,
                width=bar_width,
                label=method,
                color=color_map[method],
                edgecolor="black",
                linewidth=0.5,
            )
            for bar, value in zip(bars, values, strict=True):
                if show_value_labels and np.isfinite(finite_value(value)):
                    ax.annotate(
                        f"{value:.{metric_precision(title, key)}f}",
                        xy=(bar.get_x() + bar.get_width() / 2.0, value),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                    )
        ax.set_title(f"{title_prefix}: {title}", fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([g.replace("_", "\n") for g in groups], fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.45)
        ax.set_axisbelow(True)
        limits = padded_axis_limits(
            all_values,
            include_zero=True,
            pad_fraction=0.24 if show_value_labels else 0.14,
        )
        if limits is not None:
            ax.set_ylim(*limits)

    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(methods), 6),
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),
        )
    fig.tight_layout(
        rect=(0.0, 0.08 if handles else 0.0, 1.0, 1.0),
        pad=1.4,
        h_pad=2.2,
        w_pad=1.8,
    )
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_metric_plots(rows):
    """Build metric plots."""
    methods_all = [r["method"] for r in rows]
    non_gt = [r for r in rows if r["method"] != "GT"]
    standard = []
    for title, key in [
        ("Contrast (dB)", "contrast_dB"),
        ("CNR", "CNR"),
        ("gCNR", "gCNR"),
        ("Speckle Pass Rate", "speckle_pass_rate"),
        ("Speckle SNR", "speckle_SNR"),
        ("ENL", "ENL"),
        ("Axial FWHM (mm)", "FWHM_axial_mm"),
        ("Lateral FWHM (mm)", "FWHM_lateral_mm"),
        ("PSLR (dB)", "pslr_db"),
        ("ISLR (dB)", "islr_db"),
        ("Distortion Pass Rate", "distortion_pass_rate"),
    ]:
        if any(np.isfinite(finite_value(r.get(key, np.nan))) for r in rows):
            standard.append(
                (title, key, [finite_value(r.get(key, np.nan)) for r in rows], methods_all),
            )

    auxiliary = []
    aux_methods = [r["method"] for r in non_gt]
    for title, key in [
        ("Gaussian SSIM vs GT (dB)", "SSIM_dB_vs_GT"),
        ("Gaussian SSIM vs GT (linear envelope)", "SSIM_envelope_vs_GT"),
        ("PSNR vs GT (dB)", "PSNR_dB_vs_GT"),
        ("MAE vs GT (dB)", "MAE_dB_vs_GT"),
    ]:
        if any(np.isfinite(finite_value(r.get(key, np.nan))) for r in non_gt):
            auxiliary.append(
                (title, key, [finite_value(r.get(key, np.nan)) for r in non_gt], aux_methods),
            )
    return standard, auxiliary


def save_roi_plot(
    path,
    display,
    x_mm,
    z_mm,
    rois,
    targets,
    target_rows=None,
    peak_method=None,
):
    """Save roi plot."""
    fig, ax = plt.subplots(figsize=(6, 8), dpi=220)
    ax.imshow(
        display,
        cmap="gray",
        vmin=0,
        vmax=1,
        extent=[x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]],
        aspect="equal",
    )
    for idx, roi in enumerate(rois, start=1):
        ax.add_patch(
            plt.Circle(
                (roi["x_mm"], roi["z_mm"]),
                roi["diameter_mm"] / 2.0,
                fill=False,
                color="red",
                linewidth=0.9,
            ),
        )
        ax.text(
            roi["x_mm"],
            roi["z_mm"],
            str(idx),
            color="yellow",
            fontsize=6,
            ha="center",
            va="center",
        )
    if targets:
        ax.scatter(
            [t["x_mm"] for t in targets],
            [t["z_mm"] for t in targets],
            marker="+",
            c="cyan",
            s=20,
            label="target",
        )
        for idx, target in enumerate(targets, start=1):
            ax.text(
                target["x_mm"] + 0.25,
                target["z_mm"] - 0.25,
                str(idx),
                color="cyan",
                fontsize=5,
            )
    if target_rows:
        if peak_method is None:
            non_gt = [row["method"] for row in target_rows if row.get("method") != "GT"]
            peak_method = non_gt[0] if non_gt else target_rows[0].get("method")
        peaks = [row for row in target_rows if row.get("method") == peak_method]
        if peaks:
            ax.scatter(
                [row["peak_x_mm"] for row in peaks],
                [row["peak_z_mm"] for row in peaks],
                marker="x",
                c="yellow",
                s=14,
                linewidths=0.8,
                label=f"{peak_method} peak",
            )
            for row in peaks:
                ax.plot(
                    [row["target_x_mm"], row["peak_x_mm"]],
                    [row["target_z_mm"], row["peak_z_mm"]],
                    color="yellow",
                    linewidth=0.35,
                    alpha=0.65,
                )
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower right", fontsize=6, framealpha=0.7)
    ax.set_xlabel("Lateral (mm)")
    ax.set_ylabel("Depth (mm)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_lateral_profile_plot(
    out_dir,
    comparison,
    x_mm,
    z_mm,
    rois,
    targets,
    methods,
    dr=60.0,
    fwhm_window_mm=1.8,
    display_window_mm=5.0,
):
    """Save lateral profile plot."""
    color_map = build_method_colors(methods)
    if rois:
        target_cyst = min(
            rois,
            key=lambda c: (c["x_mm"] - 0.0) ** 2 + (c["z_mm"] - 25.0) ** 2,
        )
        iz = int(np.argmin(np.abs(z_mm - target_cyst["z_mm"])))
        cyst_radius = target_cyst["diameter_mm"] / 2.0
        display_half_width = max(5.0, 1.75 * cyst_radius)
        display_x_mask = np.abs(x_mm - target_cyst["x_mm"]) < display_half_width
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        for i, method in enumerate(methods):
            ax.plot(
                x_mm[display_x_mask],
                comparison[i, iz, display_x_mask],
                label=method,
                color=color_map[method],
                linestyle="--" if method.upper() == "GT" else "-",
                linewidth=1.5 if method.upper() != "GT" else 1.2,
            )
        ax.axvline(
            target_cyst["x_mm"] - target_cyst["diameter_mm"] / 2,
            color="grey",
            linestyle=":",
            alpha=0.7,
            label="Cyst Boundary",
        )
        ax.axvline(
            target_cyst["x_mm"] + target_cyst["diameter_mm"] / 2,
            color="grey",
            linestyle=":",
            alpha=0.7,
        )
        ax.set_title("Lateral Cyst Profile", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Lateral coordinate (mm)")
        ax.set_ylabel("Amplitude (dB)")
        ax.set_ylim(-dr, 0)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        fig.savefig(
            os.path.join(out_dir, "cyst_profile.png"),
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)
    if targets:
        target_point = min(
            targets,
            key=lambda t: (t["x_mm"] - 0.0) ** 2 + (t["z_mm"] - 25.0) ** 2,
        )
        peak_x_mask = np.abs(x_mm - target_point["x_mm"]) < fwhm_window_mm
        z_mask = np.abs(z_mm - target_point["z_mm"]) < fwhm_window_mm
        display_x_mask = np.abs(x_mm - target_point["x_mm"]) < display_window_mm
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        for i, method in enumerate(methods):
            patch = comparison[i][np.ix_(z_mask, peak_x_mask)]
            if patch.size == 0 or not np.any(np.isfinite(patch)):
                continue
            peak_z_local, peak_x_local = np.unravel_index(
                np.nanargmax(patch),
                patch.shape,
            )
            peak_z_index = np.where(z_mask)[0][peak_z_local]
            peak_db = float(patch[peak_z_local, peak_x_local])
            profile = comparison[i, peak_z_index, display_x_mask].astype(np.float64) - peak_db
            ax.plot(
                x_mm[display_x_mask],
                profile,
                label=method,
                color=color_map[method],
                linestyle="--" if method.upper() == "GT" else "-",
                linewidth=1.5 if method.upper() != "GT" else 1.2,
            )
        ax.set_title(
            "Normalized Lateral Beam Profile",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("Lateral coordinate (mm)")
        ax.set_ylabel("Amplitude relative to local target peak (dB)")
        ax.set_ylim(-dr, 0)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        fig.savefig(
            os.path.join(out_dir, "point_profile.png"),
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)


def main():
    """Run the command-line workflow."""
    args = parse_args()
    metrics_dir = resolve(args.metrics_dir)
    meta_path = os.path.join(metrics_dir, "evaluation_meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as file:
            meta = json.load(file)
    dr = float(meta.get("dr", 60.0) if args.dr is None else args.dr)
    if not np.isfinite(dr) or dr <= 0:
        raise ValueError("dr 必须是有限正数")

    summary_rows = read_rows(os.path.join(metrics_dir, "summary_metrics.csv"))
    contrast_group_rows = read_rows(
        os.path.join(metrics_dir, "contrast_group_metrics.csv"),
    )
    resolution_group_rows = read_rows(
        os.path.join(metrics_dir, "resolution_group_metrics.csv"),
    )
    target_rows = read_rows(os.path.join(metrics_dir, "resolution_target_metrics.csv"))

    standard_metrics, auxiliary_metrics = build_metric_plots(summary_rows)
    save_bar_plot(os.path.join(metrics_dir, "standard_metrics.png"), standard_metrics)
    save_bar_plot(os.path.join(metrics_dir, "auxiliary_metrics.png"), auxiliary_metrics)
    save_group_metric_plot(
        os.path.join(metrics_dir, "contrast_group_metrics.png"),
        contrast_group_rows,
        [
            ("contrast_dB", "Contrast (dB)"),
            ("CNR", "CNR"),
            ("gCNR", "gCNR"),
            ("CR_dB", "CR (dB)"),
        ],
        "Contrast groups",
    )
    save_group_metric_plot(
        os.path.join(metrics_dir, "resolution_group_metrics.png"),
        resolution_group_rows,
        [
            ("FWHM_axial_mm", "Axial FWHM (mm)"),
            ("FWHM_lateral_mm", "Lateral FWHM (mm)"),
            ("pslr_db", "PSLR (dB)"),
            ("islr_db", "ISLR (dB)"),
        ],
        "Resolution groups",
    )

    comparison_path = meta.get("comparison_npy")
    h5_path = meta.get("h5_path")
    phantom_path = meta.get("phantom_path")
    if comparison_path and h5_path and os.path.exists(comparison_path) and os.path.exists(h5_path):
        comparison = np.load(comparison_path).astype(np.float64)
        x_mm, z_mm = load_grids(h5_path)
        methods = meta.get("methods") or [row["method"] for row in summary_rows]
        if (
            comparison.ndim != 3
            or not np.isfinite(comparison).all()
            or comparison.shape[1:] != (len(z_mm), len(x_mm))
            or len(methods) != comparison.shape[0]
            or any(not isinstance(method, str) or not method for method in methods)
            or len(methods) != len(set(methods))
        ):
            raise ValueError(f"comparison 数值/形状或 methods 非法: shape={comparison.shape}, methods={methods}")
        comparison = np.clip(comparison, -dr, 0.0)
        phantom = (
            read_phantom(phantom_path)
            if phantom_path and phantom_path != "none"
            else {
                "contrast_rois": [],
                "resolution_targets": [],
            }
        )
        rois = phantom.get("contrast_rois", []) or meta.get("contrast_rois", [])
        targets = (
            (phantom.get("resolution_targets", []) or meta.get("resolution_targets", []))
            if meta.get("mode") == "resolution_distorsion"
            else []
        )
        if meta.get("has_gt") and comparison.size:
            display = db_to_display(comparison[0], dr)
            peak_method = next(
                (method for method in methods if method != "GT"),
                methods[0] if methods else None,
            )
            save_roi_plot(
                os.path.join(metrics_dir, "roi_targets.png"),
                display,
                x_mm,
                z_mm,
                rois,
                targets,
                target_rows,
                peak_method,
            )
        save_lateral_profile_plot(
            metrics_dir,
            comparison,
            x_mm,
            z_mm,
            rois,
            targets,
            methods,
            dr,
            float(meta.get("fwhm_window_mm", 1.8)),
        )

    print(f"Saved plots in: {metrics_dir}")


if __name__ == "__main__":
    main()

"""Plot evaluation outputs written by evaluate.py.

Default input:
    results/<scene>/metrics
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.evaluate import db_to_display, load_grids, read_phantom

COMPARISON_VALUE_12 = 12
PAGE_MODEL_PALETTE = [
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#17BECF",
    "#8C564B",
    "#BCBD22",
    "#E377C2",
    "#1F77B4",
    "#E69F00",
    "#6A3D9A",
]
MAX_METHODS_PER_PAGE = 8
MAX_PROFILE_MODELS_PER_PAGE = 3
MAX_PROFILE_ALGORITHMS_PER_PAGE = 4
MAX_CURVE_METHODS_PER_PAGE = 6
MAX_METRICS_PER_PAGE = 4
METRIC_PANEL_WIDTH = 10.5
METRIC_PANEL_HEIGHT = 7.5
METRIC_COLUMNS = 2
REFERENCE_METHODS = {"GT", "DAS", "MV", "ESBMV", "GCFMV", "CMSAW", "FDMAS"}
GROUP_HATCHES = ["", "//", "..", "\\\\", "xx"]


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


def method_sort_key(method):
    text = str(method)
    normalized = text.upper().replace("-", "").replace(" ", "")
    baseline_order = {"GT": 0, "DAS": 1, "MV": 2, "ESBMV": 3, "GCFMV": 4, "CMSAW": 5, "FDMAS": 6}
    if normalized in baseline_order:
        return (0, baseline_order[normalized], ())
    natural = tuple(
        (0, int(token)) if token.isdigit() else (1, token.lower())
        for token in re.split(r"(\d+)", text)
    )
    return (1, 0, natural)


def is_reference_method(method):
    normalized = str(method).upper().replace("-", "").replace(" ", "")
    return normalized in REFERENCE_METHODS


def balanced_pages(items, max_per_page):
    items = list(items)
    if not items:
        return [[]]
    page_count = math.ceil(len(items) / max_per_page)
    base_size, extra = divmod(len(items), page_count)
    pages = []
    start = 0
    for page_index in range(page_count):
        size = base_size + (1 if page_index < extra else 0)
        pages.append(items[start : start + size])
        start += size
    return pages


def paginate_methods(methods, repeat_references=True):
    methods = sorted(set(methods), key=method_sort_key)
    if not repeat_references:
        return balanced_pages(methods, MAX_METHODS_PER_PAGE)
    references = [method for method in methods if is_reference_method(method)]
    models = [method for method in methods if not is_reference_method(method)]
    if not models:
        return [references] if references else [[]]
    model_capacity = max(MAX_METHODS_PER_PAGE - len(references), 1)
    pages = []
    for page_models in balanced_pages(models, model_capacity):
        pages.append(sorted(references + page_models, key=method_sort_key))
    return pages


def paginate_profile_methods(methods, repeat_references=True):
    methods = sorted(set(methods), key=method_sort_key)
    if not repeat_references:
        return balanced_pages(methods, MAX_PROFILE_ALGORITHMS_PER_PAGE)
    references = [method for method in methods if is_reference_method(method)]
    models = [method for method in methods if not is_reference_method(method)]
    if not models:
        return [references] if references else [[]]
    if not references:
        return balanced_pages(models, MAX_PROFILE_ALGORITHMS_PER_PAGE)
    return [
        sorted(references + page_models, key=method_sort_key)
        for page_models in balanced_pages(models, MAX_PROFILE_MODELS_PER_PAGE)
    ]


def get_color(method_name):
    name = method_name.upper().replace("-", "").replace(" ", "")
    mapping = {
        "GT": "#333333",  # 深灰
        "DAS": "#4C72B0",  # 蓝色
        "MV": "#D55E00",
        "ESBMV": "#8172B3",  # 紫色
        "GCFMV": "#64B5CD",  # 青色
        "CMSAW": "#FF7F0E",  # 橙色
        "FDMAS": "#C44E52",  # 红色
    }
    if name in mapping:
        return mapping[name]
    index = int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16) % len(METHOD_PALETTE)
    return METHOD_PALETTE[index]


def build_page_method_colors(methods):
    methods = sorted(set(methods), key=method_sort_key)
    result = {}
    model_methods = []
    for method in methods:
        if is_reference_method(method):
            result[method] = get_color(method)
        else:
            model_methods.append(method)
    if len(model_methods) > len(PAGE_MODEL_PALETTE):
        raise ValueError(
            f"单页模型数 {len(model_methods)} 超过高对比颜色数 {len(PAGE_MODEL_PALETTE)}，请先分页"
        )
    result.update(
        {
            method: PAGE_MODEL_PALETTE[index]
            for index, method in enumerate(model_methods)
        }
    )
    return result


PROFILE_LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]
PROFILE_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def build_profile_styles(methods, page_methods=None):
    methods = list(dict.fromkeys(methods))
    styles = {}
    fixed = {
        "GT": {"color": "#222222", "linestyle": "--", "marker": None, "linewidth": 2.1},
        "DAS": {"color": "#0072B2", "linestyle": ":", "marker": "s", "linewidth": 1.8},
        "MV": {"color": "#D55E00", "linestyle": "-.", "marker": "D", "linewidth": 1.9},
    }
    model_methods = []
    for method in methods:
        normalized = str(method).upper().replace("-", "").replace(" ", "")
        if normalized in fixed:
            styles[method] = fixed[normalized].copy()
            continue
        model_methods.append(method)
    page_methods = methods if page_methods is None else page_methods
    page_colors = build_page_method_colors(page_methods)
    ordered_model_methods = sorted(model_methods, key=method_sort_key)
    for method in sorted(page_methods, key=method_sort_key):
        if method not in model_methods:
            continue
        member_index = ordered_model_methods.index(method)
        styles[method] = {
            "color": page_colors[method],
            "linestyle": PROFILE_LINESTYLES[member_index % len(PROFILE_LINESTYLES)],
            "marker": PROFILE_MARKERS[(member_index // len(PROFILE_LINESTYLES)) % len(PROFILE_MARKERS)],
            "linewidth": 1.55,
        }
    return styles


def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser(description="Plot evaluation CSV outputs.")
    parser.add_argument("--metrics_dir", default=os.path.join("results", "metrics"))
    parser.add_argument("--dr", type=float)
    parser.add_argument(
        "--reference-mode",
        choices=("auto", "model", "algorithms"),
        default="auto",
    )
    parser.add_argument("--profile-roi-index", type=int, help="Cyst ROI编号，从1开始")
    parser.add_argument("--profile-target-index", type=int, help="Point target编号，从1开始")
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


def select_profile_index(items, requested_index, label, center_x_mm=0.0, center_z_mm=25.0):
    if not items:
        if requested_index is not None:
            raise ValueError(f"{label}为空，不能选择编号 {requested_index}")
        return None
    if requested_index is not None:
        if requested_index < 1 or requested_index > len(items):
            raise ValueError(f"{label}编号必须在1到{len(items)}之间")
        return requested_index - 1
    return min(
        range(len(items)),
        key=lambda index: (items[index]["x_mm"] - center_x_mm) ** 2
        + (items[index]["z_mm"] - center_z_mm) ** 2,
    )


def infer_repeat_references(metrics_dir, reference_mode):
    if reference_mode == "model":
        return True
    if reference_mode == "algorithms":
        return False
    params_path = Path(metrics_dir).parent / "run_params.json"
    if params_path.exists():
        try:
            with params_path.open(encoding="utf-8") as file:
                params = json.load(file)
        except (OSError, json.JSONDecodeError):
            params = {}
        if isinstance(params.get("models"), dict):
            return True
        if isinstance(params.get("algorithms"), list):
            return False
    return True


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


def remove_paged_plot_files(path):
    """Remove stale pages when the current layout is a single figure."""
    output = Path(path)
    for stale_path in output.parent.glob(f"{output.stem}_page*{output.suffix}"):
        stale_path.unlink(missing_ok=True)


def save_bar_plot(
    path,
    metrics_to_plot,
    _cleanup_pages=True,
    _page_title=None,
    _repeat_references=True,
):
    """Save bar plot."""
    path = Path(path)
    if _cleanup_pages:
        remove_paged_plot_files(path)
    if not metrics_to_plot:
        path.unlink(missing_ok=True)
        return

    methods = sorted(
        {method for _, _, _, item_methods in metrics_to_plot for method in item_methods},
        key=method_sort_key,
    )
    method_pages = paginate_methods(methods, repeat_references=_repeat_references)
    metric_pages = balanced_pages(metrics_to_plot, MAX_METRICS_PER_PAGE)
    page_count = len(method_pages) * len(metric_pages)
    if page_count > 1:
        page_index = 0
        for metric_page in metric_pages:
            for method_page in method_pages:
                page_index += 1
                page_items = []
                for title, key, values, item_methods in metric_page:
                    value_map = dict(zip(item_methods, values, strict=True))
                    page_items.append(
                        (
                            title,
                            key,
                            [value_map.get(method, np.nan) for method in method_page],
                            method_page,
                        )
                    )
                page_path = path if page_index == 1 else path.with_name(
                    f"{path.stem}_page{page_index}{path.suffix}"
                )
                save_bar_plot(
                    page_path,
                    page_items,
                    _cleanup_pages=False,
                    _page_title=f"page {page_index}/{page_count}",
                    _repeat_references=_repeat_references,
                )
        return

    cols = METRIC_COLUMNS
    rows_grid = 2
    panel_width = METRIC_PANEL_WIDTH
    panel_height = METRIC_PANEL_HEIGHT
    fig, axes = plt.subplots(
        rows_grid,
        cols,
        figsize=(panel_width * cols, panel_height * rows_grid),
        dpi=240,
    )
    axes = np.asarray(axes).reshape(-1)
    color_map = build_page_method_colors(methods)

    for i, (title, key, vals, item_methods) in enumerate(metrics_to_plot):
        ax = axes[i]
        value_map = dict(zip(item_methods, vals, strict=True))
        vals = [value_map.get(method, np.nan) for method in methods]
        colors = [color_map[m] for m in methods]
        positions = np.arange(len(methods))
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
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        apply_value_padding(ax, vals, horizontal=True)

        for bar in bars:
            value = finite_value(bar.get_width())
            if np.isfinite(value):
                label = f"{value:.{metric_precision(title, key)}f}"
                positive = value >= 0
                ax.annotate(
                    label,
                    xy=(value, bar.get_y() + bar.get_height() / 2),
                    xytext=(3 if positive else -3, 0),
                    textcoords="offset points",
                    ha="left" if positive else "right",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )

    for j in range(len(metrics_to_plot), len(axes)):
        fig.delaxes(axes[j])
    if _page_title:
        fig.suptitle(_page_title, fontsize=10, y=0.995)
    fig.tight_layout(pad=1.4, h_pad=2.0, w_pad=1.6)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def save_group_metric_plot(
    path,
    group_rows,
    metric_keys,
    title_prefix,
    _cleanup_pages=True,
    _page_title=None,
    _repeat_references=True,
):
    """Save group metric plot."""
    path = Path(path)
    if _cleanup_pages:
        remove_paged_plot_files(path)
    if not group_rows:
        path.unlink(missing_ok=True)
        return
    groups = []
    methods = []
    for row in group_rows:
        if row["group"] not in groups:
            groups.append(row["group"])
        if row["method"] not in methods:
            methods.append(row["method"])
    methods.sort(key=method_sort_key)
    method_pages = paginate_methods(methods, repeat_references=_repeat_references)
    if len(method_pages) > 1:
        for page_idx, page_methods in enumerate(method_pages, start=1):
            page_rows = [row for row in group_rows if row["method"] in page_methods]
            page_path = path if page_idx == 1 else path.with_name(
                f"{path.stem}_page{page_idx}{path.suffix}"
            )
            save_group_metric_plot(
                page_path,
                page_rows,
                metric_keys,
                title_prefix,
                _cleanup_pages=False,
                _page_title=f"page {page_idx}/{len(method_pages)}",
                _repeat_references=_repeat_references,
            )
        return
    available_metrics = [
        (key, title)
        for key, title in metric_keys
        if any(np.isfinite(finite_value(row.get(key, np.nan))) for row in group_rows)
    ]
    if not available_metrics:
        path.unlink(missing_ok=True)
        return

    group_labels = {
        "vertical_targets": "Vertical targets",
        "horizontal_targets_2cm": "Horizontal targets (2 cm)",
        "horizontal_targets_4cm": "Horizontal targets (4 cm)",
        "horizontal_targets_near_4cm": "Horizontal targets (near 4 cm)",
        "left_column": "Left column",
        "middle_column": "Middle column",
        "right_column": "Right column",
    }
    method_colors = build_page_method_colors(methods)
    n_plots = len(available_metrics)
    cols = METRIC_COLUMNS
    rows_grid = 2
    panel_width = METRIC_PANEL_WIDTH
    panel_height = METRIC_PANEL_HEIGHT
    fig, axes = plt.subplots(
        rows_grid,
        cols,
        figsize=(panel_width * cols, panel_height * rows_grid),
        dpi=240,
    )
    axes = np.asarray(axes).reshape(-1)
    positions = np.arange(len(methods), dtype=float)
    group_step = 0.78 / max(len(groups), 1)
    offsets = (np.arange(len(groups)) - (len(groups) - 1) / 2.0) * group_step
    bar_height = group_step * 0.86
    row_lookup = {
        (str(row["method"]), str(row["group"])): row
        for row in group_rows
    }

    for ax_idx, (key, title) in enumerate(available_metrics):
        ax = axes[ax_idx]
        all_values = []
        method_values = []
        for method in methods:
            values = [
                finite_value(row_lookup.get((str(method), str(group)), {}).get(key, np.nan))
                for group in groups
            ]
            method_values.append(values)
            all_values.extend(value for value in values if np.isfinite(value))
        for group_idx, group in enumerate(groups):
            values = [method_values[method_idx][group_idx] for method_idx in range(len(methods))]
            bars = ax.barh(
                positions + offsets[group_idx],
                values,
                height=bar_height,
                color=[method_colors[method] for method in methods],
                hatch=GROUP_HATCHES[group_idx % len(GROUP_HATCHES)],
                edgecolor="black",
                linewidth=0.7,
                label=group_labels.get(group, str(group).replace("_", " ").title()),
            )
            for bar, value in zip(bars, values, strict=True):
                if np.isfinite(value):
                    ax.annotate(
                        f"{value:.{metric_precision(title, key)}f}",
                        xy=(value, bar.get_y() + bar.get_height() / 2.0),
                        xytext=(3 if value >= 0 else -3, 0),
                        textcoords="offset points",
                        ha="left" if value >= 0 else "right",
                        va="center",
                        fontsize=5.7,
                        color="#333333",
                        zorder=4,
                    )
        limits = padded_axis_limits(all_values, include_zero=True, pad_fraction=0.20)
        if limits is not None:
            ax.set_xlim(*limits)
        ax.set_title(f"{title_prefix}: {title}", fontsize=10, fontweight="bold")
        ax.set_yticks(positions)
        ax.set_yticklabels(methods, fontsize=8)
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", alpha=0.45)
        ax.set_axisbelow(True)
        ax.set_xlabel(title)
        if ax_idx % cols == 0:
            ax.set_ylabel("Method")

    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])
    if _page_title:
        fig.suptitle(_page_title, fontsize=10, y=0.995)
    handles = [
        Patch(
            facecolor="#BDBDBD",
            edgecolor="black",
            hatch=GROUP_HATCHES[index % len(GROUP_HATCHES)],
            label=group_labels.get(group, str(group).replace("_", " ").title()),
        )
        for index, group in enumerate(groups)
    ]
    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="lower center",
        ncol=min(len(groups), 4),
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 0.012),
    )
    fig.tight_layout(
        rect=(0.0, 0.08, 1.0, 1.0),
        pad=1.4,
        h_pad=2.2,
        w_pad=1.8,
    )
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def build_metric_plots(rows):
    """Build metric plots."""
    ordered_rows = sorted(rows, key=lambda row: method_sort_key(row["method"]))
    methods_all = [r["method"] for r in ordered_rows]
    non_gt = [r for r in ordered_rows if r["method"] != "GT"]
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
        ("PICMUS Speckle Penalty", "PICMUS_speckle_penalty"),
        ("PICMUS Distortion Penalty", "PICMUS_distortion_penalty"),
    ]:
        if any(np.isfinite(finite_value(r.get(key, np.nan))) for r in ordered_rows):
            standard.append(
                (title, key, [finite_value(r.get(key, np.nan)) for r in ordered_rows], methods_all),
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
    repeat_references=True,
    target_rows=None,
    profile_roi_index=None,
    profile_target_index=None,
):
    """Save lateral profile plot."""
    for stem in ("cyst_profile", "point_profile"):
        base_path = Path(out_dir) / f"{stem}.png"
        base_path.unlink(missing_ok=True)
        remove_paged_plot_files(base_path)
    method_pages = paginate_profile_methods(methods, repeat_references=repeat_references)
    page_count = len(method_pages)
    center_x_mm = float((x_mm[0] + x_mm[-1]) / 2.0)
    center_z_mm = float((z_mm[0] + z_mm[-1]) / 2.0)
    selected_roi_position = select_profile_index(
        rois,
        profile_roi_index,
        "Cyst ROI",
        center_x_mm,
        center_z_mm,
    )
    selected_target_position = select_profile_index(
        targets,
        profile_target_index,
        "Point target",
        center_x_mm,
        center_z_mm,
    )
    for page_idx, page_methods in enumerate(method_pages, start=1):
        profile_styles = build_profile_styles(methods, page_methods)
        page_suffix = "" if page_idx == 1 else f"_page{page_idx}"
        page_title_suffix = "" if page_count == 1 else f" (page {page_idx}/{page_count})"
        method_indices = [methods.index(method) for method in page_methods]
        if rois:
            target_cyst = rois[selected_roi_position]
            cyst_radius = target_cyst["diameter_mm"] / 2.0
            display_half_width = max(5.0, 1.75 * cyst_radius)
            display_x_mask = np.abs(x_mm - target_cyst["x_mm"]) < display_half_width
            cyst_center_z_index = int(np.argmin(np.abs(z_mm - target_cyst["z_mm"])))
            z_spacing = float(np.median(np.abs(np.diff(z_mm)))) if len(z_mm) > 1 else 1.0
            z_half_rows = max(
                1,
                min(3, int(round(0.35 * target_cyst["diameter_mm"] / max(z_spacing, 1.0e-9)))),
            )
            cyst_z_indices = np.arange(
                max(0, cyst_center_z_index - z_half_rows),
                min(len(z_mm), cyst_center_z_index + z_half_rows + 1),
            )
            fig, ax = plt.subplots(figsize=(12.5, 5.8), dpi=150)
            for i, method in zip(method_indices, page_methods, strict=True):
                style = profile_styles[method]
                env = 10.0 ** (
                    comparison[i][np.ix_(cyst_z_indices, display_x_mask)] / 20.0
                )
                profile_env = np.nanmean(env, axis=0)
                peak = float(np.nanmax(profile_env))
                if not np.isfinite(peak) or peak <= 0:
                    continue
                profile = 20.0 * np.log10(np.maximum(profile_env, 1.0e-12))
                profile = np.minimum(profile, 0.0)
                ax.plot(
                    x_mm[display_x_mask] - target_cyst["x_mm"],
                    profile,
                    label=method,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    marker=style["marker"],
                    markersize=3.8 if style["marker"] else 0.0,
                    markevery=max(1, int(display_x_mask.sum() / 12)),
                    markerfacecolor="white" if style["marker"] else None,
                    markeredgewidth=0.7 if style["marker"] else 0.0,
                )
            ax.axvspan(
                -cyst_radius,
                cyst_radius,
                color="#808080",
                alpha=0.12,
                label="Cyst ROI",
            )
            ax.axvline(-cyst_radius, color="grey", linestyle=":", alpha=0.7)
            ax.axvline(cyst_radius, color="grey", linestyle=":", alpha=0.7)
            ax.axvline(0.0, color="grey", linestyle="--", alpha=0.45)
            ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.55)
            ax.set_title(
                f"Lateral Cyst Profile (ROI {selected_roi_position + 1}; "
                f"central axial-band envelope mean){page_title_suffix}",
                fontsize=12,
                fontweight="bold",
                pad=12,
            )
            ax.set_xlabel("Lateral offset from cyst center (mm)")
            ax.set_ylabel("Amplitude relative to image peak (dB)")
            ax.set_ylim(-dr, 0.0)
            ax.grid(True, linestyle="--", alpha=0.5)
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(0.76, 0.5),
                ncol=2 if len(handles) > 4 else 1,
                fontsize=8,
            )
            fig.tight_layout(rect=(0.0, 0.0, 0.74, 1.0))
            fig.savefig(
                os.path.join(out_dir, f"cyst_profile{page_suffix}.png"),
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close(fig)
        if targets:
            target_point = targets[selected_target_position]
            target_x_indices = np.flatnonzero(np.abs(x_mm - target_point["x_mm"]) < fwhm_window_mm)
            target_z_indices = np.flatnonzero(np.abs(z_mm - target_point["z_mm"]) < fwhm_window_mm)
            display_x_mask = np.abs(x_mm - target_point["x_mm"]) < display_window_mm
            fig, ax = plt.subplots(figsize=(12.5, 5.8), dpi=150)
            for i, method in zip(method_indices, page_methods, strict=True):
                style = profile_styles[method]
                if target_x_indices.size == 0 or target_z_indices.size == 0:
                    continue
                target_patch = comparison[i][np.ix_(target_z_indices, target_x_indices)]
                if not np.any(np.isfinite(target_patch)):
                    continue
                peak_flat_index = int(np.nanargmax(target_patch))
                peak_z_index = int(target_z_indices[peak_flat_index // target_x_indices.size])
                profile = comparison[i, peak_z_index, display_x_mask].astype(np.float64)
                if profile.size == 0 or not np.any(np.isfinite(profile)):
                    continue
                profile -= np.nanmax(profile)
                profile = np.minimum(profile, 0.0)
                plot_label = method
                for row in target_rows or []:
                    try:
                        row_target_index = int(float(row.get("target_idx", -1)))
                    except (TypeError, ValueError):
                        continue
                    if row.get("method") != method or row_target_index != selected_target_position + 1:
                        continue
                    fwhm = finite_value(row.get("FWHM_lateral_mm", np.nan))
                    if np.isfinite(fwhm):
                        plot_label = f"{method} (FWHM={fwhm:.3f} mm)"
                    break
                ax.plot(
                    x_mm[display_x_mask] - target_point["x_mm"],
                    profile,
                    label=plot_label,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    marker=style["marker"],
                    markersize=3.8 if style["marker"] else 0.0,
                    markevery=max(1, int(display_x_mask.sum() / 12)),
                    markerfacecolor="white" if style["marker"] else None,
                    markeredgewidth=0.7 if style["marker"] else 0.0,
                )
            ax.axvline(0.0, color="grey", linestyle="--", alpha=0.55)
            ax.axvline(-fwhm_window_mm, color="grey", linestyle=":", alpha=0.65)
            ax.axvline(fwhm_window_mm, color="grey", linestyle=":", alpha=0.65, label="Target window")
            ax.axhline(-6.0, color="grey", linestyle=":", linewidth=0.8, alpha=0.7)
            ax.set_title(
                f"Lateral Point-Target Profile (peak row within target window){page_title_suffix}",
                fontsize=12,
                fontweight="bold",
                pad=12,
            )
            ax.set_xlabel("Lateral offset from target (mm)")
            ax.set_ylabel("Amplitude relative to local peak (dB)")
            ax.set_ylim(-dr, 0)
            ax.grid(True, linestyle="--", alpha=0.5)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                fig.legend(
                    handles,
                    labels,
                    loc="center left",
                    bbox_to_anchor=(0.76, 0.5),
                    ncol=2 if len(handles) > 4 else 1,
                    fontsize=8,
                )
                fig.tight_layout(rect=(0.0, 0.0, 0.74, 1.0))
            else:
                fig.tight_layout()
            fig.savefig(
                os.path.join(out_dir, f"point_profile{page_suffix}.png"),
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
    repeat_references = infer_repeat_references(metrics_dir, args.reference_mode)

    summary_rows = read_rows(os.path.join(metrics_dir, "summary_metrics.csv"))
    contrast_group_rows = read_rows(
        os.path.join(metrics_dir, "contrast_group_metrics.csv"),
    )
    resolution_group_rows = read_rows(
        os.path.join(metrics_dir, "resolution_group_metrics.csv"),
    )
    target_rows = read_rows(os.path.join(metrics_dir, "resolution_target_metrics.csv"))

    standard_metrics, auxiliary_metrics = build_metric_plots(summary_rows)
    save_bar_plot(
        os.path.join(metrics_dir, "standard_metrics.png"),
        standard_metrics,
        _repeat_references=repeat_references,
    )
    save_bar_plot(
        os.path.join(metrics_dir, "auxiliary_metrics.png"),
        auxiliary_metrics,
        _repeat_references=repeat_references,
    )
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
        _repeat_references=repeat_references,
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
        _repeat_references=repeat_references,
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
        center_x_mm = float((x_mm[0] + x_mm[-1]) / 2.0)
        center_z_mm = float((z_mm[0] + z_mm[-1]) / 2.0)
        selected_roi_position = select_profile_index(
            rois,
            args.profile_roi_index,
            "Cyst ROI",
            center_x_mm,
            center_z_mm,
        )
        selected_target_position = select_profile_index(
            targets,
            args.profile_target_index,
            "Point target",
            center_x_mm,
            center_z_mm,
        )
        with open(os.path.join(metrics_dir, "profile_selection.json"), "w", encoding="utf-8") as file:
            json.dump(
                {
                    "roi_index": selected_roi_position + 1 if selected_roi_position is not None else None,
                    "target_index": selected_target_position + 1 if selected_target_position is not None else None,
                },
                file,
                indent=2,
                ensure_ascii=False,
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
        profile_order = sorted(range(len(methods)), key=lambda index: method_sort_key(methods[index]))
        profile_methods = [methods[index] for index in profile_order]
        profile_comparison = comparison[profile_order]
        save_lateral_profile_plot(
            metrics_dir,
            profile_comparison,
            x_mm,
            z_mm,
            rois,
            targets,
            profile_methods,
            dr,
            float(meta.get("fwhm_window_mm", 1.8)),
            repeat_references=repeat_references,
            target_rows=target_rows,
            profile_roi_index=selected_roi_position + 1 if selected_roi_position is not None else None,
            profile_target_index=selected_target_position + 1 if selected_target_position is not None else None,
        )

    print(f"Saved plots in: {metrics_dir}")


if __name__ == "__main__":
    main()

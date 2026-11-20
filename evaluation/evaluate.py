"""Evaluate run_one.py comparison output.

Default input:
    results/<scene>/comparison.npy
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from evaluation.context import (
    EXPERIMENT_CONTRAST_ROIS,
    EXPERIMENT_RESOLUTION_TARGETS,
    SIMULATION_CONTRAST_ROIS,
    SIMULATION_RESOLUTION_TARGETS,
    load_picmus_context,
    resolve,
)
from evaluation.metrics import (
    TARGET_MIN_PROMINENCE_DB,
    build_square_label_map,
    contrast_roi_metrics,
    db_to_display,
    db_to_envelope,
    local_ssim,
    mean_or_nan,
    picmus_distortion_pass,
    psnr,
    standard_contrast_score,
    standard_speckle_quality,
    target_resolution,
)
from shared_core.paths import relative_path


PASS_THRESHOLD = 0.5
IMAGE_STACK_DIMENSIONS = 3
SIGNAL_DOMAINS = ("iq", "rf", "analytic_rf")


def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser(
        description="Evaluate run_one.py comparison output.",
    )
    parser.add_argument(
        "--comparison_npy",
        default=os.path.join("results", "comparison.npy"),
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="逗号分隔算法名,不含GT;省略或设为auto时从run_params.json读取",
    )
    parser.add_argument(
        "--method_labels",
        default=None,
        help="逗号分隔显示名,不含GT;默认使用run_params.json或算法名大写",
    )
    parser.add_argument("--h5_path", default=os.path.join("data", "simulation.h5"))
    parser.add_argument("--h5_sample_idx", type=int, default=0)
    parser.add_argument("--dr", type=float, default=60.0)
    parser.add_argument("--out_dir", default=os.path.join("results", "metrics"))
    parser.add_argument(
        "--phantom_path",
        default="auto",
        help="显式指定对应的 PICMUS 模体文件；默认读取 H5 元数据。",
    )
    parser.add_argument(
        "--phantom_mode",
        choices=["auto", "contrast_speckle", "resolution_distortion"],
        default="auto",
    )
    parser.add_argument(
        "--phantom_source",
        choices=["auto", "simulation", "experiments"],
        default="auto",
        help="默认读取 H5 的 phantom_source；显式值须与元数据一致。",
    )
    parser.add_argument("--fwhm_window_mm", type=float, default=1.8)
    parser.add_argument(
        "--signal_domain",
        choices=SIGNAL_DOMAINS,
        default=None,
        help="输入信号域；默认读取 H5 声明，显式值须与 H5 输入域一致。",
    )
    parser.add_argument(
        "--model_semantics_version",
        type=int,
        default=None,
        help="评估所用 checkpoint 的 model_semantics_version。",
    )
    return parser.parse_args()


def resolve_signal_domain(explicit, stored_domains):
    """Resolve the evaluated signal domain without silently assuming IQ.

    ``stored_domains`` comes from the H5 ``generation.storage.inputs`` block.
    An explicit value wins because an H5 may store RF while the evaluation
    processes a derived IQ/analytic domain.  Without an explicit value a
    single stored domain is used, while a missing or multi-domain declaration
    is an error.
    """

    domains = tuple(dict.fromkeys(str(item) for item in stored_domains if item))
    if explicit is not None:
        if explicit not in SIGNAL_DOMAINS:
            raise ValueError(f"不支持的 signal_domain: {explicit!r}")
        return explicit
    if len(domains) == 1:
        return domains[0]
    if not domains:
        raise ValueError("H5 未声明输入信号域，请显式指定 --signal_domain")
    raise ValueError(
        "H5 声明了多个输入信号域 "
        f"({', '.join(domains)})，请显式指定 --signal_domain"
    )


def load_method_names(args, comparison_path, n_panels, has_gt):
    """Load method names."""
    methods_arg = args.methods
    if methods_arg is not None and methods_arg.strip().lower() == "auto":
        methods_arg = None
    if args.method_labels and not methods_arg:
        raise ValueError("--method_labels 必须与 --methods 一起使用")
    if methods_arg:
        methods = [m.strip().lower() for m in methods_arg.split(",") if m.strip()]
        names = [m.upper() for m in methods]
        if args.method_labels:
            labels = [m.strip() for m in args.method_labels.split(",") if m.strip()]
            if len(labels) != len(methods):
                raise ValueError("--method_labels 数量必须与 --methods 数量一致")
            names = labels
    else:
        params_path = os.path.join(os.path.dirname(comparison_path), "run_params.json")
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"缺少当前格式的 run_params.json: {params_path}")
        with open(params_path, encoding="utf-8") as f:
            params = json.load(f)
        if not isinstance(params, dict):
            raise ValueError(f"run_params.json 顶层必须是对象: {params_path}")
        algorithms = params.get("algorithms")
        if algorithms is not None:
            if not isinstance(algorithms, list) or not all(
                isinstance(name, str) and name for name in algorithms
            ):
                raise ValueError("run_params.json algorithms 必须是算法名列表")
            labels = params.get("algorithm_labels")
            if not isinstance(labels, dict):
                raise ValueError("run_params.json algorithm_labels 必须是标签字典")
            names = [labels[name] for name in algorithms]
            if not all(isinstance(name, str) and name for name in names):
                raise ValueError("run_params.json 算法标签必须是非空字符串")
        else:
            names = load_model_names(params, n_panels, has_gt, params_path)
    if has_gt:
        names = ["GT", *names]
    if len(names) != n_panels:
        raise ValueError(
            f"Method count {len(names)} does not match stacked image count {n_panels}",
        )
    if len(names) != len(set(names)):
        raise ValueError(f"Method labels must be unique: {names}")
    return names


def load_model_names(params, n_panels, has_gt, params_path):
    model_map = params.get("models")
    if not isinstance(model_map, dict) or not model_map:
        raise ValueError(f"run_params.json 缺少当前格式的 models 字典: {params_path}")
    model_names = list(model_map)
    evaluation = params.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError(f"run_params.json evaluation 必须是对象: {params_path}")
    skip_baselines = evaluation.get("skip_baselines")
    if not isinstance(skip_baselines, bool):
        raise ValueError(f"run_params.json evaluation.skip_baselines 必须是布尔值: {params_path}")
    baseline_names = [] if skip_baselines else ["DAS", "MV"]
    names = [*baseline_names, *model_names]
    n_methods = n_panels - 1 if has_gt else n_panels
    if len(names) != n_methods:
        raise ValueError(
            f"run_params.json 中的模型/基线数量与 comparison 不一致: "
            f"expected={n_methods}, actual={len(names)}"
        )
    return names


def picmus_resolution_groups(source, target_count):
    """Return the official PICMUS resolution target groupings."""
    if source == "simulation" and target_count >= SIMULATION_RESOLUTION_TARGETS:
        return [
            ("vertical_targets", list(range(1, 9))),
            ("horizontal_targets_2cm", [9, 10, 11, 3, 12, 13, 14]),
            ("horizontal_targets_4cm", [15, 16, 17, 7, 18, 19, 20]),
        ]
    if source == "experiments" and target_count >= EXPERIMENT_RESOLUTION_TARGETS:
        return [
            ("vertical_targets", list(range(1, 6))),
            ("horizontal_targets_near_4cm", [6, 4, 7]),
        ]
    return [("all_targets", list(range(1, target_count + 1)))] if target_count else []


def picmus_contrast_groups(source, roi_count):
    """Execute picmus contrast groups."""
    if source == "simulation" and roi_count >= SIMULATION_CONTRAST_ROIS:
        return [
            ("left_column", [4, 5, 6]),
            ("middle_column", [1, 2, 3]),
            ("right_column", [7, 8, 9]),
        ]
    if source == "experiments" and roi_count >= EXPERIMENT_CONTRAST_ROIS:
        return [("middle_column", [1, 2])]
    return [("all_cysts", list(range(1, roi_count + 1)))] if roi_count else []


def rows_for_method_index(rows, method, index_key):
    """Execute rows for method index."""
    out = {}
    for row in rows:
        if row.get("method") != method or index_key not in row:
            continue
        try:
            out[int(row[index_key])] = row
        except (TypeError, ValueError):
            continue
    return out


def speckle_penalty_for_method(roi_rows, method):
    """Execute speckle penalty for method."""
    passes = [
        row["speckle_pass"]
        for row in roi_rows
        if row.get("method") == method and "speckle_pass" in row and np.isfinite(row["speckle_pass"])
    ]
    if not passes:
        return np.nan
    return -40.0 if any(value < PASS_THRESHOLD for value in passes) else 0.0


def distortion_penalty_for_method(target_rows, method, source):
    """Return the official PICMUS simulated distortion penalty."""
    if source != "simulation":
        return np.nan
    passes = [
        row["distortion_pass"]
        for row in target_rows
        if row.get("method") == method
        and "distortion_pass" in row
        and np.isfinite(row["distortion_pass"])
    ]
    if not passes:
        return np.nan
    return -40.0 if any(value < PASS_THRESHOLD for value in passes) else 0.0


def build_resolution_group_rows(target_rows, source, target_count, methods):
    """Build rows using the official PICMUS target group order."""
    groups = picmus_resolution_groups(source, target_count)
    rows = []
    for method in methods:
        indexed = rows_for_method_index(target_rows, method, "target_idx")
        for group_name, indices in groups:
            selected = [indexed[i] for i in indices if i in indexed]
            if not selected:
                continue
            pass_values = [
                row["distortion_pass"]
                for row in selected
                if np.isfinite(row.get("distortion_pass", np.nan))
            ]
            if source == "simulation" and pass_values:
                penalty = -40.0 if any(value < PASS_THRESHOLD for value in pass_values) else 0.0
                pass_rate = mean_or_nan(pass_values)
            else:
                penalty = np.nan
                pass_rate = np.nan
            rows.append(
                {
                    "method": method,
                    "group": group_name,
                    "target_indices": " ".join(str(i) for i in indices),
                    "n_targets": len(selected),
                    "n_detected": sum(
                        bool(row["target_response_detected"]) for row in selected
                    ),
                    "target_response_detection_rate": mean_or_nan(
                        [row["target_response_detected"] for row in selected]
                    ),
                    "FWHM_axial_mm": mean_or_nan(
                        [row.get("FWHM_axial_mm") for row in selected],
                    ),
                    "FWHM_lateral_mm": mean_or_nan(
                        [row.get("FWHM_lateral_mm") for row in selected],
                    ),
                    "pslr_db": mean_or_nan([row.get("pslr_db") for row in selected]),
                    "islr_db": mean_or_nan([row.get("islr_db") for row in selected]),
                    "localization_error_mm": mean_or_nan(
                        [row.get("localization_error_mm") for row in selected],
                    ),
                    "distortion_pass_rate": pass_rate,
                    "distortion_penalty": penalty,
                },
            )
    return rows


def build_contrast_group_rows(roi_rows, source, roi_count, methods):
    """Build contrast group rows."""
    groups = picmus_contrast_groups(source, roi_count)
    rows = []
    for method in methods:
        indexed = rows_for_method_index(
            [row for row in roi_rows if "contrast_score_dB" in row],
            method,
            "roi_idx",
        )
        for group_name, indices0 in groups:
            indices = [i - 1 for i in indices0]
            selected = [indexed[i] for i in indices if i in indexed]
            if not selected:
                continue
            rows.append(
                {
                    "method": method,
                    "group": group_name,
                    "roi_indices": " ".join(str(i) for i in indices0),
                    "n_rois": len(selected),
                    "contrast_score_dB": mean_or_nan(
                        [row.get("contrast_score_dB") for row in selected],
                    ),
                    "CR_dB": mean_or_nan([row.get("CR_dB") for row in selected]),
                    "CNR_env": mean_or_nan([row.get("CNR_env") for row in selected]),
                    "gCNR": mean_or_nan([row.get("gCNR") for row in selected]),
                    "cyst_residual_dB": mean_or_nan(
                        [row.get("cyst_residual_dB") for row in selected],
                    ),
                    "speckle_penalty": speckle_penalty_for_method(
                        roi_rows,
                        method,
                    ),
                },
            )
    return rows


def fmt_metric(value, precision=3):
    """Execute fmt metric."""
    try:
        if value is None or not np.isfinite(value):
            return "NA"
    except TypeError:
        return "NA"
    return f"{float(value):.{precision}f}"


def write_picmus_report(
    path,
    mode,
    source,
    methods,
    summary_rows,
    contrast_group_rows,
    resolution_group_rows,
    roi_rows,
):
    """Execute write picmus report."""
    summary_by_method = {row["method"]: row for row in summary_rows}
    with open(path, "w", encoding="utf-8") as file:
        file.write("PICMUS-inspired evaluation summary (ours protocol)\n")
        file.write(f"mode: {mode}\nsource: {source}\n\n")
        for method in methods:
            row = summary_by_method.get(method, {})
            file.write(f"[{method}]\n")
            if mode == "contrast_speckle":
                file.write(
                    f"mean_contrast_score_dB: {fmt_metric(row.get('contrast_score_dB'), 1)}\n",
                )
                file.write(f"mean_CR_dB: {fmt_metric(row.get('CR_dB'), 3)}\n")
                file.write(f"mean_CNR_env: {fmt_metric(row.get('CNR_env'), 3)}\n")
                file.write(f"mean_gCNR: {fmt_metric(row.get('gCNR'), 3)}\n")
                file.write(
                    f"speckle_pass_rate: {fmt_metric(row.get('speckle_pass_rate'), 3)}\n",
                )
                file.write(
                    f"speckle_penalty: {fmt_metric(speckle_penalty_for_method(roi_rows, method), 1)}\n",
                )
                for group in [r for r in contrast_group_rows if r["method"] == method]:
                    file.write(
                        f"  {group['group']} ({group['roi_indices']}): "
                        f"contrast={fmt_metric(group.get('contrast_score_dB'), 1)}, "
                        f"CNR_env={fmt_metric(group.get('CNR_env'), 3)}, "
                        f"gCNR={fmt_metric(group.get('gCNR'), 3)}\n",
                    )
            elif mode == "resolution_distortion":
                file.write(
                    f"mean_FWHM_axial_mm: {fmt_metric(row.get('FWHM_axial_mm'), 4)}\n",
                )
                file.write(
                    f"mean_FWHM_lateral_mm: {fmt_metric(row.get('FWHM_lateral_mm'), 4)}\n",
                )
                file.write(f"mean_PSLR_dB: {fmt_metric(row.get('pslr_db'), 3)}\n")
                file.write(f"mean_ISLR_dB: {fmt_metric(row.get('islr_db'), 3)}\n")
                file.write(
                    "target_response_detection_rate: "
                    f"{fmt_metric(row.get('target_response_detection_rate'), 3)}\n",
                )
                file.write(
                    f"distortion_pass_rate: {fmt_metric(row.get('distortion_pass_rate'), 3)}\n",
                )
                file.write(
                    f"distortion_penalty: {fmt_metric(row.get('distortion_penalty'), 1)}\n",
                )
                for group in [r for r in resolution_group_rows if r["method"] == method]:
                    file.write(
                        f"  {group['group']} ({group['target_indices']}): "
                        f"detected={group['n_detected']}/{group['n_targets']}, "
                        f"axial={fmt_metric(group.get('FWHM_axial_mm'), 4)}, "
                        f"lateral={fmt_metric(group.get('FWHM_lateral_mm'), 4)}, "
                        f"penalty={fmt_metric(group.get('distortion_penalty'), 1)}\n",
                    )
            else:
                file.write("No controlled-scene score for this mode.\n")
            file.write("\n")


def write_csv(path, rows, fieldnames):
    """Execute write csv."""
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Run the command-line workflow."""
    args = parse_args()
    positive_args = {
        "dr": args.dr,
        "fwhm_window_mm": args.fwhm_window_mm,
    }
    invalid = [name for name, value in positive_args.items() if not math.isfinite(value) or value <= 0]
    if invalid:
        raise ValueError(f"参数必须是有限正数: {', '.join(invalid)}")
    comparison_path = resolve(args.comparison_npy)
    h5_path = resolve(args.h5_path)
    out_dir = resolve(args.out_dir)
    context = load_picmus_context(
        h5_path, args.h5_sample_idx, phantom_path=args.phantom_path,
        phantom_mode=args.phantom_mode, phantom_source=args.phantom_source,
        fwhm_window_mm=args.fwhm_window_mm,
    )
    if context is None:
        print("Non-PICMUS sample: skipped phantom metric export.")
        return
    meta_from_h5 = context["meta"]
    has_gt = bool(meta_from_h5.get("has_gt", False))
    mode, source = context["mode"], context["source"]
    signal_domain = resolve_signal_domain(
        getattr(args, "signal_domain", None),
        context.get("signal_domains") or [meta_from_h5.get("signal_domain")],
    )

    comparison = np.load(comparison_path).astype(np.float64)
    x_mm, z_mm = context["x_mm"], context["z_mm"]
    if comparison.ndim != IMAGE_STACK_DIMENSIONS or comparison.shape[0] < 1:
        raise ValueError(
            f"comparison 必须是非空 [method,z,x] 三维数组,实际为 {comparison.shape}",
        )
    if comparison.shape[1:] != (len(z_mm), len(x_mm)):
        raise ValueError(
            f"Image shape {comparison.shape[1:]} does not match H5 grids {(len(z_mm), len(x_mm))}",
        )
    if not np.all(np.isfinite(comparison)):
        bad_count = int(comparison.size - np.count_nonzero(np.isfinite(comparison)))
        raise ValueError(f"comparison 包含 {bad_count} 个 NaN/Inf,拒绝生成无效指标")
    methods = load_method_names(args, comparison_path, comparison.shape[0], has_gt)

    distortion_z_offset_mm = context["distortion_z_offset_mm"]
    os.makedirs(out_dir, exist_ok=True)
    phantom_path = context["phantom_path"]
    phantom = context["phantom"]
    rois = phantom["contrast_rois"]
    speckle_rois = phantom["speckle_rois"]
    targets = phantom["resolution_targets"] if mode == "resolution_distortion" else []
    lateral_resolution_mm = phantom.get("lateral_resolution_mm")
    axial_resolution_mm = phantom.get("axial_resolution_mm")
    gt_db = comparison[0] if has_gt else None

    resolution_label_map = None
    distortion_roi_label_map = None
    distortion_inside_label_map = None
    if targets:
        resolution_label_map = build_square_label_map(
            x_mm,
            z_mm,
            targets,
            args.fwhm_window_mm,
        )
        if source == "simulation":
            distortion_roi_label_map = build_square_label_map(
                x_mm,
                z_mm,
                targets,
                args.fwhm_window_mm,
                z_offset_mm=distortion_z_offset_mm,
            )
            distortion_inside_label_map = build_square_label_map(
                x_mm,
                z_mm,
                targets,
                0.29570,
                z_offset_mm=distortion_z_offset_mm,
            )

    gt_display = db_to_display(gt_db, args.dr) if has_gt else None
    gt_linear_envelope = db_to_envelope(gt_db) if has_gt else None
    rows = []
    roi_rows = []
    target_rows = []
    for method, db_img in zip(methods, comparison, strict=True):
        display = db_to_display(db_img, args.dr)
        env = db_to_envelope(db_img)
        row = {"method": method}
        if has_gt and method == "GT":
            row["SSIM_dB_vs_GT"] = 1.0
            row["SSIM_env_vs_GT"] = 1.0
            row["PSNR_dB_vs_GT"] = float("inf")
            row["MAE_dB_vs_GT"] = 0.0
        elif has_gt:
            row["SSIM_dB_vs_GT"] = local_ssim(display, gt_display, data_range=1.0)
            row["SSIM_env_vs_GT"] = local_ssim(
                env,
                gt_linear_envelope,
                data_range=1.0,
            )
            row["PSNR_dB_vs_GT"] = psnr(display, gt_display, data_range=1.0)
            clipped_db = np.clip(db_img, -args.dr, 0.0)
            clipped_gt_db = np.clip(gt_db, -args.dr, 0.0)
            row["MAE_dB_vs_GT"] = float(np.mean(np.abs(clipped_db - clipped_gt_db)))
        else:
            row["SSIM_dB_vs_GT"] = np.nan
            row["SSIM_env_vs_GT"] = np.nan
            row["PSNR_dB_vs_GT"] = np.nan
            row["MAE_dB_vs_GT"] = np.nan

        contrast_scores = []
        cr_scores = []
        cnr_scores = []
        gcnr_scores = []
        residual_scores = []
        for idx, roi in enumerate(rois):
            score = standard_contrast_score(
                db_img,
                x_mm,
                z_mm,
                roi,
                lateral_resolution_mm,
            )
            metrics = contrast_roi_metrics(
                db_img,
                x_mm,
                z_mm,
                roi,
                lateral_resolution_mm,
            )
            if not np.isfinite(score) and metrics is None:
                continue
            roi_row = {"method": method, "roi_idx": idx, **roi, "contrast_score_dB": score}
            if metrics is not None:
                roi_row.update(metrics)
                cr_scores.append(metrics["CR_dB"])
                cnr_scores.append(metrics["CNR_env"])
                gcnr_scores.append(metrics["gCNR"])
                residual_scores.append(metrics["cyst_residual_dB"])
            roi_rows.append(roi_row)
            contrast_scores.append(score)
        row["contrast_score_dB"] = mean_or_nan(contrast_scores)
        row["CR_dB"] = mean_or_nan(cr_scores)
        row["CNR_env"] = mean_or_nan(cnr_scores)
        row["gCNR"] = mean_or_nan(gcnr_scores)
        row["cyst_residual_dB"] = mean_or_nan(residual_scores)

        speckle_stats = []
        for idx, roi in enumerate(speckle_rois):
            stats = standard_speckle_quality(
                env,
                x_mm,
                z_mm,
                roi,
                lateral_resolution_mm,
                axial_resolution_mm,
            )
            if stats is None:
                continue
            roi_rows.append({"method": method, "roi_idx": idx, **roi, **stats})
            speckle_stats.append(stats)
        row["speckle_pass_rate"] = mean_or_nan(
            [s["speckle_pass"] for s in speckle_stats],
        )
        row["speckle_penalty"] = speckle_penalty_for_method(
            roi_rows,
            method,
        )
        row["speckle_KS_D"] = mean_or_nan([s["speckle_KS_D"] for s in speckle_stats])
        row["speckle_KS_p"] = mean_or_nan([s["speckle_KS_p"] for s in speckle_stats])
        row["speckle_SNR"] = mean_or_nan([s["speckle_SNR"] for s in speckle_stats])
        row["ENL_env"] = mean_or_nan([s["ENL_env"] for s in speckle_stats])

        target_stats = []
        for idx, target in enumerate(targets):
            stats = target_resolution(
                db_img,
                x_mm,
                z_mm,
                target,
                args.fwhm_window_mm,
                target_idx=idx + 1,
                label_map=resolution_label_map,
            )
            if stats is None:
                continue
            stats["distortion_pass"] = picmus_distortion_pass(
                db_img,
                idx + 1,
                source,
                roi_label_map=distortion_roi_label_map,
                inside_label_map=distortion_inside_label_map,
            )
            target_rows.append({"method": method, "target_idx": idx + 1, **stats})
            target_stats.append(stats)
        row["FWHM_axial_mm"] = mean_or_nan([s["FWHM_axial_mm"] for s in target_stats])
        row["FWHM_lateral_mm"] = mean_or_nan(
            [s["FWHM_lateral_mm"] for s in target_stats],
        )
        row["pslr_db"] = mean_or_nan([s["pslr_db"] for s in target_stats])
        row["islr_db"] = mean_or_nan([s["islr_db"] for s in target_stats])
        row["localization_error_mm"] = mean_or_nan(
            [s["localization_error_mm"] for s in target_stats]
        )
        row["target_response_detection_rate"] = mean_or_nan(
            [s["target_response_detected"] for s in target_stats]
        )
        row["distortion_pass_rate"] = mean_or_nan(
            [s["distortion_pass"] for s in target_stats],
        )
        row["distortion_penalty"] = distortion_penalty_for_method(
            target_rows,
            method,
            source,
        )
        rows.append(row)

    summary_fields = [
        "method",
        "SSIM_dB_vs_GT",
        "SSIM_env_vs_GT",
        "PSNR_dB_vs_GT",
        "MAE_dB_vs_GT",
        "contrast_score_dB",
        "CR_dB",
        "CNR_env",
        "gCNR",
        "cyst_residual_dB",
        "speckle_pass_rate",
        "speckle_penalty",
        "speckle_KS_D",
        "speckle_KS_p",
        "speckle_SNR",
        "ENL_env",
        "FWHM_axial_mm",
        "FWHM_lateral_mm",
        "pslr_db",
        "islr_db",
        "localization_error_mm",
        "target_response_detection_rate",
        "distortion_pass_rate",
        "distortion_penalty",
    ]
    write_csv(os.path.join(out_dir, "summary_metrics.csv"), rows, summary_fields)
    contrast_group_rows = build_contrast_group_rows(
        roi_rows,
        source,
        len(rois),
        methods,
    )
    resolution_group_rows = build_resolution_group_rows(
        target_rows,
        source,
        len(targets),
        methods,
    )
    optional_outputs = [
        ("contrast_roi_metrics.csv", roi_rows),
        ("resolution_target_metrics.csv", target_rows),
        ("contrast_group_metrics.csv", contrast_group_rows),
        ("resolution_group_metrics.csv", resolution_group_rows),
    ]
    for name, data_rows in optional_outputs:
        path = os.path.join(out_dir, name)
        if data_rows:
            fieldnames = []
            for data_row in data_rows:
                for key in data_row:
                    if key not in fieldnames:
                        fieldnames.append(key)
            write_csv(path, data_rows, fieldnames)
        else:
            write_csv(path, [{"status": "not_applicable"}], ["status"])
    write_picmus_report(
        os.path.join(out_dir, "evaluation_protocol_summary.txt"),
        mode,
        source,
        methods,
        rows,
        contrast_group_rows,
        resolution_group_rows,
        roi_rows,
    )

    meta = {
        "evaluation_protocol": "picmus",
        "comparison_npy": relative_path(comparison_path),
        "h5_path": relative_path(h5_path),
        "h5_sample_idx": args.h5_sample_idx,
        "mode": mode,
        "phantom_source": source,
        "phantom_path": relative_path(phantom_path),
        "sample_name": meta_from_h5.get("sample_name", ""),
        "has_gt": has_gt,
        "methods": methods,
        "signal_domain": signal_domain,
        "model_semantics_version": getattr(args, "model_semantics_version", None),
        "contrast_roi_count": len(rois),
        "speckle_roi_count": len(speckle_rois),
        "resolution_target_count": len(targets),
        "fwhm_window_mm": args.fwhm_window_mm,
        "target_min_prominence_db": TARGET_MIN_PROMINENCE_DB,
        "picmus_distortion_z_offset_mm": distortion_z_offset_mm,
        "contrast_groups": picmus_contrast_groups(source, len(rois)),
        "resolution_groups": picmus_resolution_groups(source, len(targets)),
        "contrast_rois": rois,
        "resolution_targets": targets,
        "dr": args.dr,
        "protocol": "ours_picmus_inspired",
        "metric_domains": {
            "contrast_score_dB": "relative dB domain from normalized envelope",
            "CR_dB": "linear envelope mean ratio, reported in dB",
            "CNR_env": "linear envelope domain",
            "gCNR": "linear envelope domain, fixed 256-bin histogram on [0, 1]",
            "speckle_KS_D": "linear envelope domain with 5-pixel block sampling",
            "speckle_SNR": "linear envelope domain with 5-pixel block sampling",
            "ENL_env": "linear envelope domain with 5-pixel block sampling; envelope mean squared divided by envelope variance",
            "FWHM_axial_mm": "relative dB domain, -6 dB width",
            "FWHM_lateral_mm": "relative dB domain, -6 dB width",
            "pslr_db": "relative dB domain",
            "islr_db": "relative dB domain",
            "localization_error_mm": "mean Euclidean distance from each point target to the measured peak position",
            "target_response_detection_rate": (
                "fraction of point targets whose peak is at least "
                "6 dB above the target-window median"
            ),
            "SSIM_dB_vs_GT": "[-dr, 0] dB display domain",
            "SSIM_env_vs_GT": "normalized linear envelope domain",
            "PSNR_dB_vs_GT": "[-dr, 0] dB display domain",
            "MAE_dB_vs_GT": "clipped [-dr, 0] dB domain",
        },
        "normalization": "Each output and GT frame is independently peak-normalized to 0 dB before relative metrics.",
        "note": "Standard metrics use the predefined phantom ROI and target definitions when available.",
    }
    with open(
        os.path.join(out_dir, "evaluation_meta.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(meta, file, indent=2, ensure_ascii=False)

    print(f"Saved summary: {os.path.join(out_dir, 'summary_metrics.csv')}")
    if contrast_group_rows:
        print(
            f"Saved contrast group metrics: {os.path.join(out_dir, 'contrast_group_metrics.csv')}",
        )
    if resolution_group_rows:
        print(
            f"Saved resolution group metrics: {os.path.join(out_dir, 'resolution_group_metrics.csv')}",
        )
    print(
        f"Saved evaluation protocol report: {os.path.join(out_dir, 'evaluation_protocol_summary.txt')}",
    )
    print(
        f"Mode={mode}, source={source}, contrast_rois={len(rois)}, resolution_targets={len(targets)}",
    )


if __name__ == "__main__":
    main()

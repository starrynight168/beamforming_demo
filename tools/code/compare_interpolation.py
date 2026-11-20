"""Run a signal_domain x interpolation combination matrix ablation across scenes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = Path(__file__).resolve().parent
for path in (ROOT, CODE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ablate_sound_speed as sound_speed_tool  # noqa: E402
import run_ablation as workflow  # noqa: E402
from evaluation.plot_metrics import (  # noqa: E402
    METHOD_PALETTE,
    PROFILE_LINESTYLES,
    PROFILE_MARKERS,
)
from shared_core.plot_style import (  # noqa: E402
    DIAGNOSTIC,
    apply_style,
    draw_bmode,
    save_figure,
    style_axes,
)
from tools.shared.run_output import (  # noqa: E402
    emit,
    prepare_output_tree,
    write_csv,
    write_json,
)
from tools.shared.scenes import (  # noqa: E402
    SCENE_LABELS,
    STANDARD_SCENE_IDS,
    resolve_project_path as resolve,
)

DEFAULT_DOMAINS = ("rf", "analytic_rf", "iq")
DEFAULT_INTERPS = (
    "nearest",
    "linear",
    "catmull_rom",
    "cubic",
    "quintic",
    "farrow",
    "sinc",
)
DOMAIN_LABELS = {
    "rf": "Raw RF",
    "analytic_rf": "Analytic RF",
    "iq": "IQ",
}
CURVE_METRICS = (
    ("SSIM_dB_vs_GT", "SSIM to reference"),
    ("ROI2_CR_dB", "Center cyst ROI2 CR (dB)"),
    ("CR_dB", "Mean cyst contrast ratio (dB)"),
    ("gCNR", "gCNR"),
    ("FWHM_lateral_mm", "Lateral FWHM (mm)"),
    ("FWHM_axial_mm", "Axial FWHM (mm)"),
    ("runtime_sec", "Beamforming time (s/frame)"),
)


def parse_domains(raw: str) -> list[str]:
    """Parse and validate comma-separated signal domains."""
    values = [item.strip().lower() for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("信号域消融至少需要指定一个取值。")
    unsupported = [value for value in values if value not in DEFAULT_DOMAINS]
    if unsupported:
        raise ValueError(
            f"不支持的 signal_domain: {', '.join(unsupported)}；"
            f"只能从 {', '.join(DEFAULT_DOMAINS)} 中选择。",
        )
    if len(set(values)) != len(values):
        raise ValueError("信号域取值不能重复。")
    return values


def parse_interps(raw: str) -> list[str]:
    """Parse and validate comma-separated interpolation methods."""
    allowed = workflow.CATEGORICAL_PARAM_CHOICES["interp"]
    values = [item.strip().lower() for item in str(raw).split(",") if item.strip()]
    if len(values) < 2:
        raise ValueError("插值消融至少需要两个取值。")
    unsupported = [value for value in values if value not in allowed]
    if unsupported:
        raise ValueError(
            f"不支持的 interp: {', '.join(unsupported)}；"
            f"只能从 {', '.join(allowed)} 中选择。",
        )
    if len(set(values)) != len(values):
        raise ValueError("插值取值不能重复。")
    return values


def load_config(path):
    """Load and validate the base YAML config."""
    return sound_speed_tool.load_config(path)


def select_scenes(config, domains=None):
    """Select the four standard PICMUS scenes and validate requested signal domains."""
    raw_scenes = config.get("scenes")
    if not isinstance(raw_scenes, list):
        raise ValueError("场景配置必须包含 scenes 列表。")
    by_id = {
        str(scene.get("id")): scene
        for scene in raw_scenes
        if isinstance(scene, dict)
    }
    missing = [scene_id for scene_id in STANDARD_SCENE_IDS if scene_id not in by_id]
    if missing:
        raise ValueError(f"配置缺少标准四场景: {missing}")

    selected = []
    info_by_path = {}
    for scene_id in STANDARD_SCENE_IDS:
        scene = by_id[scene_id]
        h5_path = resolve(scene["h5_path"])
        sample_idx = int(scene.get("sample_idx", 0))
        h5_key = str(h5_path)
        info = info_by_path.get(h5_key)
        if info is None:
            info = workflow.inspect_h5(h5_path)
            info_by_path[h5_key] = info
        if not 0 <= sample_idx < info["n"]:
            raise IndexError(f"{scene_id} 的 sample_idx 超出 H5 范围。")
        if workflow.load_picmus_context(h5_path, sample_idx) is None:
            raise ValueError(f"{scene_id} 不是有效的 PICMUS 标准场景。")
        if domains:
            workflow.validate_available_signal_domains(domains, info["domains"])
        selected.append(
            {
                "id": scene_id,
                "h5_path": h5_path,
                "sample_idx": sample_idx,
                "role": "picmus",
            }
        )
    return selected


def method_interp(method):
    """Extract the interpolation name from a method label like 'interp=sinc'."""
    text = str(method)
    if not text.startswith("interp="):
        return None
    value = text.split("=", 1)[1].strip()
    return value or None


def write_all_rows(rows, path):
    """Write the complete (domain, scenario, interp) run summary CSV."""
    fields = [
        "signal_domain",
        "scenario",
        "index",
        "parameter",
        "value",
        "scene_id",
        "status",
        "result_dir",
    ]
    return write_csv(path, rows, fields)


def collect_metrics(scenes, domains, run_root, algorithm):
    """Collect evaluated PICMUS metrics and beamforming runtimes across (domain x scene)."""
    del algorithm
    records = []
    metric_fields = None
    for domain in domains:
        for scene in scenes:
            scene_root = run_root / domain / scene["id"]
            metrics_path = scene_root / "metrics" / "summary_metrics.csv"
            if not metrics_path.is_file():
                continue
            roi2_by_method = {}
            contrast_path = scene_root / "metrics" / "contrast_roi_metrics.csv"
            if contrast_path.is_file():
                with contrast_path.open(encoding="utf-8-sig", newline="") as stream:
                    for roi_row in csv.DictReader(stream):
                        if str(roi_row.get("roi_idx", "")).strip() == "1" and "CR_dB" in roi_row:
                            roi2_by_method[roi_row.get("method", "")] = roi_row["CR_dB"]
            with metrics_path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                metric_fields = list(reader.fieldnames or ())
                for row in reader:
                    method_label = row.get("method", "")
                    interp = method_interp(method_label)
                    runtime_sec = ""
                    if interp is not None:
                        params_path = (
                            scene_root
                            / workflow.value_scene_id("interp", interp)
                            / "params.json"
                        )
                        if params_path.is_file():
                            params_data = json.loads(
                                params_path.read_text(encoding="utf-8")
                            )
                            if "runtime_sec" in params_data:
                                runtime_sec = params_data["runtime_sec"]
                    records.append(
                        {
                            "signal_domain": domain,
                            "scenario": scene["id"],
                            "h5_path": str(scene["h5_path"]),
                            "sample_idx": scene["sample_idx"],
                            **row,
                            "ROI2_CR_dB": roi2_by_method.get(method_label, ""),
                            "runtime_sec": runtime_sec,
                        }
                    )
    if not records or not metric_fields:
        return None
    output_path = run_root / "metrics" / "four_scene_metrics.csv"
    fields = ["signal_domain", "scenario", "h5_path", "sample_idx"]
    for field in (*metric_fields, "ROI2_CR_dB", "runtime_sec"):
        if field not in fields:
            fields.append(field)
    write_csv(output_path, records, fields)
    return output_path, records


def save_interp_sweep_curve(records, scenes, domains, interps, output_dir):
    """Plot dot-and-line metric sweeps across (signal_domain x interp) using shared style."""
    apply_style(DIAGNOSTIC)
    available = []
    for metric, label in CURVE_METRICS:
        seen_interps = set()
        for row in records:
            interp = method_interp(row.get("method", ""))
            if interp not in interps:
                continue
            try:
                value = float(row.get(metric, "nan"))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                seen_interps.add(interp)
        if len(seen_interps) >= 2:
            available.append((metric, label))
    if not available:
        return None

    ncols = 3
    nrows = math.ceil(len(available) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.6 * ncols, 3.2 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    x_ticks = list(range(len(interps)))
    interp_to_x = {name: idx for idx, name in enumerate(interps)}

    series_specs = []
    for d_idx, domain in enumerate(domains):
        for s_idx, scene in enumerate(scenes):
            series_specs.append((d_idx, domain, s_idx, scene))

    for ax, (metric, label) in zip(axes.flat, available, strict=False):
        plotted = False
        for d_idx, domain, s_idx, scene in series_specs:
            points = []
            for row in records:
                if (
                    row.get("signal_domain") != domain
                    or row.get("scenario") != scene["id"]
                ):
                    continue
                interp = method_interp(row.get("method", ""))
                if interp not in interp_to_x:
                    continue
                try:
                    value = float(row.get(metric, "nan"))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    points.append((interp_to_x[interp], value))
            if points:
                points.sort()
                x_vals, y_vals = zip(*points, strict=True)
                color = METHOD_PALETTE[(d_idx * 4 + s_idx) % len(METHOD_PALETTE)]
                linestyle = PROFILE_LINESTYLES[d_idx % len(PROFILE_LINESTYLES)]
                marker = PROFILE_MARKERS[s_idx % len(PROFILE_MARKERS)]
                series_label = (
                    f"{DOMAIN_LABELS.get(domain, domain)} - "
                    f"{SCENE_LABELS.get(scene['id'], scene['id'])}"
                )
                ax.plot(
                    x_vals,
                    y_vals,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=1.2,
                    markersize=4.5,
                    label=series_label,
                )
                plotted = True
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(interps, rotation=20, ha="right")
        ax.set_title(label)
        ax.set_xlabel("Interpolation method")
        ax.set_ylabel(label)
        style_axes(ax, grid=True)
        if plotted:
            ax.legend(fontsize=6.5, ncol=2 if len(series_specs) > 6 else 1)

    for ax in list(axes.flat)[len(available):]:
        ax.set_visible(False)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return save_figure(
        fig,
        output_dir / "interp_sweep_metrics.png",
        profile=DIAGNOSTIC,
    )[0]


def save_scene_matrix_overviews(scenes, domains, interps, run_root, algorithm, dr):
    """Render (signal_domain x interp) B-mode image matrix per scene using shared style."""
    apply_style(DIAGNOSTIC)
    saved = []
    figures_dir = Path(run_root) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        grid = []
        has_any = False
        for domain in domains:
            row_images = []
            for interp in interps:
                path = (
                    Path(run_root)
                    / domain
                    / scene["id"]
                    / workflow.value_scene_id("interp", interp)
                    / f"{algorithm}.npy"
                )
                if path.is_file():
                    row_images.append(np.load(path).astype(np.float32))
                    has_any = True
                else:
                    row_images.append(None)
            grid.append(row_images)
        if not has_any:
            continue

        nrows, ncols = len(domains), len(interps)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(2.4 * ncols, 2.8 * nrows),
            squeeze=False,
            constrained_layout=True,
        )
        for r_idx, domain in enumerate(domains):
            for c_idx, interp in enumerate(interps):
                ax = axes[r_idx][c_idx]
                image = grid[r_idx][c_idx]
                if image is not None:
                    draw_bmode(ax, image, None, dynamic_range=dr)
                else:
                    ax.text(
                        0.5, 0.5, "N/A",
                        ha="center", va="center", transform=ax.transAxes,
                    )
                if r_idx == 0:
                    ax.set_title(interp)
                if c_idx == 0:
                    ax.set_ylabel(DOMAIN_LABELS.get(domain, domain))
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

        out_path = figures_dir / f"{scene['id']}_domain_interp_matrix.png"
        saved.append(save_figure(fig, out_path, profile=DIAGNOSTIC)[0])
    return saved


def parse_args(argv=None):
    """Parse command-line options for the signal_domain x interp matrix tool."""
    parser = argparse.ArgumentParser(
        description="信号域 (rf/analytic_rf/iq) × 插值方法组合矩阵消融与评测"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--algorithm",
        default="mv",
        choices=tuple(workflow.ALGORITHMS),
        help="固定使用的波束合成算法（默认 mv）",
    )
    parser.add_argument(
        "--domains",
        default=None,
        help="逗号分隔信号域，例如 rf,analytic_rf,iq",
    )
    parser.add_argument(
        "--interps",
        default=None,
        help="逗号分隔插值方式，例如 nearest,linear,catmull_rom,cubic,quintic,farrow,sinc",
    )
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default=None)
    parser.add_argument(
        "--run-dir", "--run_dir", dest="run_dir", default=None,
        help="指定已有运行目录以续跑",
    )
    parser.add_argument(
        "--metrics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否计算四个 PICMUS 场景的标准指标",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Run the signal_domain x interpolation matrix workflow."""
    args = parse_args(argv)
    config = load_config(args.config)

    default_domains = ",".join(DEFAULT_DOMAINS)
    raw_domains = args.domains
    if raw_domains is None and sys.stdin.isatty():
        raw_domains = workflow.ask_text(
            "请输入信号域列表（逗号分隔）", default=default_domains
        )
    domains = parse_domains(raw_domains or default_domains)

    default_interps = ",".join(DEFAULT_INTERPS)
    raw_interps = args.interps
    if raw_interps is None and sys.stdin.isatty():
        raw_interps = workflow.ask_text(
            "请输入插值方法列表（逗号分隔）", default=default_interps
        )
    interps = parse_interps(raw_interps or default_interps)

    scenes = select_scenes(config, domains=domains)

    metrics_enabled = args.metrics
    if metrics_enabled is None:
        metrics_enabled = (
            workflow.ask_yes_no("是否计算四个 PICMUS 场景的标准指标？", default=True)
            if sys.stdin.isatty()
            else True
        )

    default_output = workflow.TOOLS_OUTPUT_ROOT / "compare_interpolation"
    output_root = args.output_dir
    if output_root is None and sys.stdin.isatty() and not args.run_dir:
        output_root = workflow.ask_text("请输入结果根目录", default=str(default_output))
    output_root = resolve(output_root or default_output)
    run_root = (
        resolve(args.run_dir)
        if args.run_dir
        else output_root / f"matrix_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    )

    protocol_path = run_root / "protocol.json"
    protocol = {
        "parameter_matrix": ["signal_domain", "interp"],
        "algorithm": args.algorithm,
        "signal_domains": domains,
        "interps": interps,
        "metrics_enabled": metrics_enabled,
        "scenes": [
            {
                "id": scene["id"],
                "role": scene["role"],
                "h5_path": str(scene["h5_path"]),
                "sample_idx": scene["sample_idx"],
            }
            for scene in scenes
        ],
    }
    if args.run_dir:
        if not protocol_path.is_file():
            raise FileNotFoundError("续跑目录中没有 protocol.json。")
        saved_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if saved_protocol != protocol:
            raise ValueError("续跑配置与原运行的信号域、插值方式、算法或场景不一致。")
    else:
        run_root.mkdir(parents=True, exist_ok=False)
        write_json(protocol_path, protocol)

    prepare_output_tree(run_root)
    write_json(run_root / "run_config.json", protocol)

    emit(run_root, "prepare", f"算法：{args.algorithm}；信号域：{domains}；插值：{interps}。")
    emit(run_root, "info", f"标准四场景：{[s['id'] for s in scenes]}。")
    emit(run_root, "info", f"四场景指标：{'开启' if metrics_enabled else '关闭'}。")

    dr = float((config.get("params", {}) or {}).get("dr", 60.0))
    all_rows = []
    for domain in domains:
        domain_config = deepcopy(config)
        domain_config.setdefault("params", {})["signal_domain"] = domain
        for scene in scenes:
            scene_root = run_root / domain / scene["id"]
            config_dir = scene_root / "configs"
            rows = []
            for index, interp in enumerate(interps, 1):
                scene_value_id = workflow.value_scene_id("interp", interp)
                run_config = workflow.build_config(
                    domain_config,
                    scene["h5_path"],
                    scene["sample_idx"],
                    args.algorithm,
                    "global",
                    "interp",
                    interp,
                    scene_value_id,
                )
                config_path = workflow.save_config(
                    run_config,
                    config_dir,
                    "interp",
                    interp,
                )
                emit(
                    run_root,
                    "progress",
                    f"{domain}/{scene['id']}：interp={interp}（{index}/{len(interps)}）。",
                )
                code, _ = workflow.run_one(
                    config_path,
                    scene_value_id,
                    scene_root,
                    bool(args.run_dir),
                )
                result_dir = scene_root / scene_value_id
                status = "ok" if code == 0 else f"failed({code})"
                if code == 0:
                    if not (
                        (scene_root / "data" / "ground_truth.npy").exists()
                        or (scene_root / "ground_truth.npy").exists()
                    ):
                        workflow.preserve_ground_truth(result_dir, scene_root)
                    workflow.flatten_ablation_result(result_dir, args.algorithm)
                row = {
                    "index": index,
                    "parameter": "interp",
                    "value": interp,
                    "scene_id": scene_value_id,
                    "status": status,
                    "result_dir": str(result_dir),
                }
                rows.append(row)
                all_rows.append(
                    {"signal_domain": domain, "scenario": scene["id"], **row}
                )
            workflow.write_summary(rows, scene_root / "metrics" / "ablation_summary.csv")
            workflow.save_ablation_overview(rows, scene_root, args.algorithm, "interp")
            workflow.collect_ablation_individual_images(
                rows, scene_root, args.algorithm, "interp"
            )
            if metrics_enabled:
                workflow.run_ablation_evaluation(
                    rows,
                    scene_root,
                    scene["h5_path"],
                    scene["sample_idx"],
                    args.algorithm,
                    True,
                    dr,
                    domain,
                )

    all_rows_path = run_root / "metrics" / "all_scene_runs.csv"
    write_all_rows(all_rows, all_rows_path)

    matrix_images = save_scene_matrix_overviews(
        scenes, domains, interps, run_root, args.algorithm, dr
    )
    curve_path = None
    metrics_path = None
    if metrics_enabled:
        collected = collect_metrics(scenes, domains, run_root, args.algorithm)
        if collected:
            metrics_path, records = collected
            curve_path = save_interp_sweep_curve(
                records, scenes, domains, interps, run_root / "figures"
            )

    workflow.append_run_log(
        {
            "type": "interpolation_matrix_ablation",
            "dir": str(workflow.relative_to_root(run_root)).replace("\\", "/"),
            "algorithm": args.algorithm,
            "signal_domains": domains,
            "interps": interps,
            "metrics": metrics_enabled,
            "scenes": [scene["id"] for scene in scenes],
        },
    )
    emit(run_root, "done", f"汇总表：{all_rows_path}。")
    if metrics_path:
        emit(run_root, "info", f"四场景统一指标：{metrics_path}。")
    if curve_path:
        emit(run_root, "info", f"插值指标点线图：{curve_path}。")
    if matrix_images:
        emit(run_root, "info", f"组合矩阵成像图：{len(matrix_images)} 张，位于 {run_root / 'figures'}。")


if __name__ == "__main__":
    main()

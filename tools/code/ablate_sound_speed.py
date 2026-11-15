"""Run a sound-speed sweep across the standard scenes and an in-vivo sample."""

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = Path(__file__).resolve().parent
for path in (ROOT, CODE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_ablation as workflow
from tools.shared.run_output import (
    emit,
    prepare_output_tree,
    save_figure,
    set_plot_style,
    write_csv,
    write_json,
)

DEFAULT_C_VALUES = (1430, 1460, 1490, 1520, 1540, 1570, 1600, 1620)
CURVE_METRICS = (
    ("SSIM_dB_vs_GT", "SSIM to reference"),
    ("CR_dB", "Cyst contrast ratio (dB)"),
    ("gCNR", "gCNR"),
    ("FWHM_lateral_mm", "Lateral FWHM (mm)"),
    ("FWHM_axial_mm", "Axial FWHM (mm)"),
)
SCENE_LABELS = {
    "simulation_contrast_speckle": "Sim. contrast",
    "simulation_resolution_distortion": "Sim. resolution",
    "experiments_contrast_speckle": "Exp. contrast",
    "experiments_resolution_distortion": "Exp. resolution",
}


def resolve(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def parse_c_values(raw):
    try:
        values = [float(item.strip()) for item in str(raw).split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("声速取值必须是逗号分隔的数字。") from exc
    if len(values) < 2:
        raise ValueError("声速消融至少需要两个取值。")
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("声速取值必须是有限正数。")
    rounded = [round(value, 3) for value in values]
    if len(set(rounded)) != len(rounded):
        raise ValueError("声速取值不能重复。")
    return sorted(rounded)


def load_config(path):
    with resolve(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict) or not config.get("scenes"):
        raise ValueError("场景配置必须包含 scenes 列表。")
    return config


def select_scenes(config):
    selected = []
    info_by_path = {}
    for scene in config["scenes"]:
        h5_path = resolve(scene["h5_path"])
        sample_idx = int(scene.get("sample_idx", 0))
        info = info_by_path.setdefault(str(h5_path), None)
        if info is None:
            info = workflow.inspect_h5(h5_path)
            info_by_path[str(h5_path)] = info
        if not 0 <= sample_idx < info["n"]:
            raise IndexError(f"{scene['id']} 的 sample_idx 超出 H5 范围。")
        is_picmus = workflow.load_picmus_context(h5_path, sample_idx) is not None
        is_in_vivo = bool(info["in_vivo"][sample_idx])
        if is_picmus:
            role = "picmus"
        elif is_in_vivo:
            role = "in_vivo"
        else:
            continue
        with h5py.File(h5_path, "r") as handle:
            nominal_c = float(handle["c"][()])
        selected.append(
            {
                "id": str(scene["id"]),
                "h5_path": h5_path,
                "sample_idx": sample_idx,
                "role": role,
                "nominal_c": nominal_c,
            },
        )
    picmus = [scene for scene in selected if scene["role"] == "picmus"]
    in_vivo = [scene for scene in selected if scene["role"] == "in_vivo"]
    if len(picmus) != 4 or not in_vivo:
        raise ValueError(
            "配置需要包含四个 PICMUS 场景和至少一个活体场景；"
            f"当前识别到 {len(picmus)} 个 PICMUS、{len(in_vivo)} 个活体。",
        )
    return selected


def write_all_rows(rows, path):
    fields = [
        "scenario",
        "index",
        "parameter",
        "value",
        "scene_id",
        "status",
        "result_dir",
    ]
    return write_csv(path, rows, fields)


def collect_metrics(scenes, run_root):
    records = []
    metric_fields = None
    for scene in scenes:
        if scene["role"] != "picmus":
            continue
        metrics_path = run_root / scene["id"] / "metrics" / "summary_metrics.csv"
        if not metrics_path.is_file():
            continue
        with metrics_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            metric_fields = reader.fieldnames
            for row in reader:
                records.append(
                    {
                        "scenario": scene["id"],
                        "h5_path": str(scene["h5_path"]),
                        "sample_idx": scene["sample_idx"],
                        **row,
                    },
                )
    if not records or not metric_fields:
        return None
    output_path = run_root / "metrics" / "four_scene_metrics.csv"
    fields = ["scenario", "h5_path", "sample_idx", *metric_fields]
    write_csv(output_path, records, fields)
    return output_path, records


def method_c(method):
    try:
        return float(str(method).split("=", 1)[1])
    except (IndexError, ValueError):
        return None


def save_c_sweep_curve(records, scenes, output_dir):
    set_plot_style()
    available = []
    for metric, label in CURVE_METRICS:
        speeds = set()
        for row in records:
            c_value = method_c(row.get("method", ""))
            try:
                metric_value = float(row.get(metric, "nan"))
            except (TypeError, ValueError):
                continue
            if c_value is not None and np.isfinite(metric_value):
                speeds.add(c_value)
        if len(speeds) >= 2:
            available.append((metric, label))
    if not available:
        return None
    ncols = 3
    nrows = math.ceil(len(available) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.4 * ncols, 3.2 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    phantom_scenes = [scene for scene in scenes if scene["role"] == "picmus"]
    nominal_speeds = sorted({scene["nominal_c"] for scene in phantom_scenes})
    for ax, (metric, label) in zip(axes.flat, available, strict=False):
        plotted = False
        for scene in phantom_scenes:
            points = []
            for row in records:
                if row["scenario"] != scene["id"]:
                    continue
                c_value = method_c(row.get("method", ""))
                try:
                    metric_value = float(row.get(metric, "nan"))
                except (TypeError, ValueError):
                    continue
                if c_value is not None and np.isfinite(metric_value):
                    points.append((c_value, metric_value))
            if len(points) >= 2:
                points.sort()
                x_values, y_values = zip(*points, strict=True)
                ax.plot(
                    x_values,
                    y_values,
                    marker="o",
                    linewidth=1.4,
                    markersize=4,
                    label=SCENE_LABELS.get(scene["id"], scene["id"]),
                )
                plotted = True
        for index, nominal_c in enumerate(nominal_speeds):
            ax.axvline(
                nominal_c,
                color="#666666",
                linestyle="--",
                linewidth=0.9,
                label="H5 nominal c" if index == 0 else None,
            )
        ax.set_title(label)
        ax.set_xlabel("Sound speed used by MV (m/s)")
        ax.set_ylabel(label)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        if plotted or nominal_speeds:
            ax.legend(frameon=False, fontsize=7)
    for ax in list(axes.flat)[len(available):]:
        ax.set_visible(False)
    return save_figure(fig, output_dir / "c_sweep_metrics.png")


def parse_args():
    parser = argparse.ArgumentParser(description="四场景和活体的 MV 声速消融")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--c-values", default=None, help="逗号分隔，例如 1460,1540,1620")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-dir", default=None, help="指定已有运行目录以续跑")
    parser.add_argument(
        "--metrics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否计算四个 PICMUS 场景的标准指标",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    scenes = select_scenes(config)
    default_values = ",".join(str(value) for value in DEFAULT_C_VALUES)
    raw_values = args.c_values
    if raw_values is None and sys.stdin.isatty():
        raw_values = workflow.ask_text("请输入声速取值", default=default_values)
    values = parse_c_values(raw_values or default_values)
    for scene in scenes:
        if not any(np.isclose(value, scene["nominal_c"], atol=1e-3, rtol=0) for value in values):
            values.append(scene["nominal_c"])
    values = sorted(set(round(value, 3) for value in values))

    metrics_enabled = args.metrics
    if metrics_enabled is None:
        metrics_enabled = (
            workflow.ask_yes_no("是否计算四个 PICMUS 场景的标准指标？", default=True)
            if sys.stdin.isatty()
            else True
        )
    default_output = workflow.TOOLS_OUTPUT_ROOT / "ablate_sound_speed"
    output_root = args.output_dir
    if output_root is None and sys.stdin.isatty() and not args.run_dir:
        output_root = workflow.ask_text("请输入结果根目录", default=str(default_output))
    output_root = resolve(output_root or default_output)
    run_root = (
        resolve(args.run_dir)
        if args.run_dir
        else output_root / f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    )
    protocol_path = run_root / "protocol.json"
    protocol = {
        "parameter": "MV sound speed c (m/s)",
        "c_values_mps": values,
        "metrics_enabled": metrics_enabled,
        "scenes": [
            {
                "id": scene["id"],
                "role": scene["role"],
                "h5_path": str(scene["h5_path"]),
                "sample_idx": scene["sample_idx"],
                "nominal_c_mps": scene["nominal_c"],
            }
            for scene in scenes
        ],
        "in_vivo_metrics": "image comparison only; no independent ground truth",
    }
    if args.run_dir:
        if not protocol_path.is_file():
            raise FileNotFoundError("续跑目录中没有 protocol.json。")
        saved_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if saved_protocol != protocol:
            raise ValueError("续跑配置与原运行的声速取值、场景或指标选项不一致。")
    else:
        run_root.mkdir(parents=True, exist_ok=False)
        write_json(protocol_path, protocol)
    prepare_output_tree(run_root)
    write_json(run_root / "run_config.json", protocol)

    emit(run_root, "prepare", f"声速取值：{values}。")
    emit(run_root, "info", f"标准场景：{[s['id'] for s in scenes if s['role'] == 'picmus']}。")
    emit(run_root, "info", f"活体场景：{[s['id'] for s in scenes if s['role'] == 'in_vivo']}。")
    emit(run_root, "info", f"四场景指标：{'开启' if metrics_enabled else '关闭'}。")

    rows_by_scene = {}
    all_rows = []
    for scene in scenes:
        scene_root = run_root / scene["id"]
        config_dir = scene_root / "configs"
        rows = []
        for index, value in enumerate(values, 1):
            scene_value_id = workflow.value_scene_id("c", value)
            run_config = workflow.build_config(
                config,
                scene["h5_path"],
                scene["sample_idx"],
                "mv",
                "algorithm",
                "c",
                value,
                scene_value_id,
            )
            config_path = workflow.save_config(
                run_config,
                config_dir,
                "c",
                value,
            )
            emit(run_root, "progress", f"{scene['id']}：c={value:g} m/s（{index}/{len(values)}）。")
            code, _ = workflow.run_one(
                config_path,
                scene_value_id,
                scene_root,
                bool(args.run_dir),
            )
            result_dir = scene_root / scene_value_id
            status = "ok" if code == 0 else f"failed({code})"
            if code == 0:
                if (
                    scene["role"] == "picmus"
                    and not (
                        (scene_root / "data" / "ground_truth.npy").exists()
                        or (scene_root / "ground_truth.npy").exists()
                    )
                ):
                    workflow.preserve_ground_truth(result_dir, scene_root)
                workflow.flatten_ablation_result(result_dir, "mv")
            row = {
                "index": index,
                "parameter": "c",
                "value": value,
                "scene_id": scene_value_id,
                "status": status,
                "result_dir": str(result_dir),
            }
            rows.append(row)
            all_rows.append({"scenario": scene["id"], **row})
        rows_by_scene[scene["id"]] = rows
        workflow.write_summary(rows, scene_root / "metrics" / "ablation_summary.csv")

    for scene in scenes:
        scene_root = run_root / scene["id"]
        rows = rows_by_scene[scene["id"]]
        workflow.save_ablation_overview(rows, scene_root, "mv", "c")
        workflow.collect_ablation_individual_images(rows, scene_root, "mv", "c")
        if metrics_enabled and scene["role"] == "picmus":
            workflow.run_ablation_evaluation(
                rows,
                scene_root,
                scene["h5_path"],
                scene["sample_idx"],
                "mv",
                True,
                float((config.get("params", {}) or {}).get("dr", 60.0)),
            )

    all_rows_path = run_root / "metrics" / "all_scene_runs.csv"
    write_all_rows(all_rows, all_rows_path)
    curve_path = None
    metrics_path = None
    if metrics_enabled:
        collected = collect_metrics(scenes, run_root)
        if collected:
            metrics_path, records = collected
            curve_path = save_c_sweep_curve(records, scenes, run_root / "figures")

    workflow.append_run_log(
        {
            "type": "sound_speed_ablation",
            "dir": str(workflow.relative_to_root(run_root)).replace("\\", "/"),
            "c_values": values,
            "metrics": metrics_enabled,
            "scenes": [scene["id"] for scene in scenes],
        },
    )
    emit(run_root, "done", f"汇总表：{all_rows_path}。")
    if metrics_path:
        emit(run_root, "info", f"四场景统一指标：{metrics_path}。")
    if curve_path:
        emit(run_root, "info", f"声速指标曲线：{curve_path}。")
    emit(run_root, "info", f"各场景图像对照位于：{run_root}。")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import copy
import dataclasses
import json
import math
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TRAIN_DIR = Path(__file__).resolve().parent.parent
ROOT = TRAIN_DIR.parent
for path in (ROOT, TRAIN_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import mban as training
from eval_core.config import (
    DEFAULT_CONFIG,
    DEFAULT_HARDWARE_CONFIG,
    DEFAULT_SCENE_CONFIG,
    DEFAULT_SCENES,
    evaluation_stage,
    parse_checkpoint_model,
    resolve_deployment_runtime_transforms,
    validate_deployment_checkpoint,
)
import run_one
from eval_core.inference import load_scene, metric_pair, read_split_indices, read_test_indices, reconstruct, write_rows
from eval_core.config import ModelSpec, load_evaluation_model
from mban_core.config import (
    MODEL_SEMANTICS_VERSION,
    configured_h5_files,
    format_hardware_eval_overrides,
    hardware_eval_default_overrides,
    load_hardware_eval_config,
)
from mban_core.hardware import (
    BIAS_IMPLEMENTATIONS,
    HARDWARE_STRESS_PROFILES,
    INTER_LAYER_MODES,
    resolve_layer_bias_implementation,
)
from mban_core.naming import layer_name, output_controls_name

@contextmanager
def tia_limit_probe(hardware: object, output_tia_name: str, limit: float, enabled: bool) -> Iterator[dict[str, object]]:
    original_tia = hardware.tia
    stats: dict[str, object] = {
        "total": 0,
        "clipped": 0,
        "tia_names": {},
        "tia_maxima": {},
    }

    def wrapped_tia(x, name="tia", common_mode=None):
        tia_names = stats["tia_names"]
        tia_maxima = stats["tia_maxima"]
        tia_names[name] = tia_names.get(name, 0) + int(x.numel())
        y = original_tia(x, name, common_mode=common_mode)
        if y.numel():
            tia_maxima[name] = max(tia_maxima.get(name, 0.0), float(y.detach().abs().max().item()))
        if enabled and name == output_tia_name:
            values = y.detach().abs()
            stats["total"] += values.numel()
            stats["clipped"] += int((values > limit).sum().item())
            y = y.clamp(-limit, limit)
        return y

    hardware.tia = wrapped_tia
    try:
        yield stats
    finally:
        hardware.tia = original_tia
def nominal_hardware_fields() -> dict[str, object]:
    return {
        "dac_gain_error": 0.0,
        "dac_offset": 0.0,
        "dac_noise_std": 0.0,
        "dac_noise_mode": "full_scale",
        "dac_nonlinearity": 0.0,
        "dac_nonlinearity_beta": 3.0,
        "bias_programming_gain_error": 0.0,
        "bias_programming_offset": 0.0,
        "bias_noise_std": 0.0,
        "bias_channel_gain_mismatch_std": 0.0,
        "tia_gain_error": 0.0,
        "tia_channel_gain_mismatch_std": 0.0,
        "tia_offset": 0.0,
        "tia_channel_offset_mismatch_std": 0.0,
        "tia_noise_std": 0.0,
        "tia_saturation": 0.0,
        "activation_threshold": 0.0,
        "activation_threshold_mismatch": 0.0,
        "activation_gain_mismatch": 0.0,
        "buffer_gain_error": 0.0,
        "buffer_noise_std": 0.0,
        "write_error_std": 0.0,
        "read_noise_std": 0.0,
        "control_reference_gain_error": 0.0,
        "control_reference_offset": 0.0,
        "control_reference_noise_std": 0.0,
        "control_reference_hold_error": 0.0,
        "ir_drop_enabled": False,
        "adc_gain_error": 0.0,
        "adc_channel_gain_mismatch_std": 0.0,
        "adc_offset": 0.0,
        "adc_channel_offset_mismatch_std": 0.0,
        "adc_noise_std": 0.0,
        "adc_nonlinearity": 0.0,
        "drift_std": 0.0,
        "stuck_at_prob": 0.0,
        "noise_enabled": False,
        "static_mismatch_std": {},
    }

def calibrate_ptq_observers(
    args: argparse.Namespace,
    specs: list[ModelSpec],
    device: torch.device,
    *,
    split_file: Path | None = None,
) -> None:
    quantized = [
        spec
        for spec in specs
        if spec.model_type == "ptq" and hasattr(spec.model, "hardware_qat") and spec.model.hardware_qat.enabled
    ]
    if not quantized:
        raise ValueError("PTQ校准至少需要一个启用硬件量化的模型")
    runtime_args = quantized[0].runtime_args or training.args
    if runtime_args is None:
        raise RuntimeError("PTQ校准缺少运行时配置")
    data_fields = ("h5_files", "train_ratio", "validation_ratio", "test_ratio", "seed", "make_new_split")
    for spec in quantized[1:]:
        other_args = spec.runtime_args or training.args
        if other_args is None or any(
            getattr(other_args, field, None) != getattr(runtime_args, field, None)
            for field in data_fields
        ):
            raise ValueError("同一次 PTQ 校准的模型必须共享 h5_files、split 比例、seed 和 make_new_split")
    requested_split = split_file or getattr(args, "split_file", None)
    if requested_split is None:
        raise ValueError("PTQ校准必须显式提供 split_file")
    h5_paths = [str(Path(path).resolve()) for path in runtime_args.h5_files]
    split = training.load_or_create_mixed_split(
        h5_paths,
        runtime_args.train_ratio,
        runtime_args.validation_ratio,
        runtime_args.test_ratio,
        split_file=Path(requested_split).resolve(),
        seed=int(runtime_args.seed),
        make_new_split=bool(runtime_args.make_new_split),
    )
    active_configs = {}
    nominal_fields = nominal_hardware_fields()
    for spec in quantized:
        hardware = spec.model.hardware_qat
        active_configs[spec.display_name] = hardware.config
        hardware.config = dataclasses.replace(hardware.config, **nominal_fields)
        hardware.disable_noise()
        hardware.clear_observer()
        hardware.enable_observer()
        spec.model.eval()
    try:
        for h5_path in h5_paths:
            train_indices = split[h5_path]["train"]
            count = min(int(runtime_args.ptq_calibration_samples), len(train_indices))
            positions = np.linspace(0, len(train_indices) - 1, count).round().astype(int)
            for sample_index in train_indices[positions]:
                data = load_scene(Path(h5_path), int(sample_index))
                for spec in quantized:
                    reconstruct(spec, data, device, args.micro_batch, collect_weight_stats=False)
    finally:
        for spec in quantized:
            hardware = spec.model.hardware_qat
            hardware.disable_observer()
            active_config = active_configs[spec.display_name]
            hardware.config = active_config
            if active_config.noise_enabled:
                hardware.enable_noise()
    for spec in quantized:
        hardware_config = spec.model.hardware_qat.config
        quantized_fields = ["input_bits", "control_bits"]
        if hardware_config.inter_layer == "digital":
            quantized_fields.append("inter_layer_bits")
        needs_observer_ranges = any(int(getattr(hardware_config, field)) < 32 for field in quantized_fields)
        if needs_observer_ranges and not spec.model.hardware_qat.activation_ranges:
            raise RuntimeError(f"模型{spec.display_name}未获得任何PTQ校准量程")
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", spec.display_name)
        (args.output / f"{safe_label}_ptq_calibration.json").write_text(
            json.dumps(spec.model.hardware_qat.state_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def load_tia_model(source: Path, label: str, args: argparse.Namespace, device: torch.device):
    spec = load_evaluation_model(
        label,
        source.resolve(),
        "qat",
        device,
        config_path=args.config.resolve(),
        profile=args.profile,
        hardware_config_path=args.eval_config.resolve(),
    )
    hardware = spec.model.hardware_qat
    if hardware is None:
        raise RuntimeError("TIA 饱和测试需要 QAT checkpoint")
    hardware.config = dataclasses.replace(hardware.config, noise_seed=args.seed)
    hardware.reset_noise_counter()
    hardware.begin_noise_realization()
    return spec, hardware


def evaluate_test_frames(
    source: Path,
    label: str,
    args: argparse.Namespace,
    test_frames: list[int],
    saturated: bool,
) -> dict[str, object]:
    device = torch.device(args.device)
    spec, hardware = load_tia_model(source, label, args, device)
    output_tia_name = f"fc{int(spec.model.hidden_layers) + 1}.tia"
    values_db: list[float] = []
    values_env: list[float] = []
    h5_path = args.test_h5.resolve()
    with tia_limit_probe(hardware, output_tia_name, args.tia_limit, saturated) as probe:
        for frame in test_frames:
            data = load_scene(h5_path, frame)
            image, _, _, _ = reconstruct(spec, data, device, args.micro_batch, collect_weight_stats=False)
            db, env = metric_pair(image, data.gt_db)
            values_db.append(db)
            values_env.append(env)
    return {
        "model": label,
        "saturated": saturated,
        "tia_limit": float(args.tia_limit),
        "tia_saturation_fraction": int(probe["clipped"]) / max(int(probe["total"]), 1),
        "tia_names": probe["tia_names"],
        "tia_maxima": probe["tia_maxima"],
        "frames": len(values_db),
        "ssim_db_mean": float(np.mean(values_db)),
        "ssim_db_std": float(np.std(values_db, ddof=1)) if len(values_db) > 1 else 0.0,
        "ssim_env_mean": float(np.mean(values_env)),
        "ssim_env_std": float(np.std(values_env, ddof=1)) if len(values_env) > 1 else 0.0,
    }


def evaluate_scene(
    source: Path,
    label: str,
    scene_id: str,
    data,
    args: argparse.Namespace,
    saturated: bool,
) -> tuple[dict[str, object], np.ndarray]:
    device = torch.device(args.device)
    spec, hardware = load_tia_model(source, label, args, device)
    output_tia_name = f"fc{int(spec.model.hidden_layers) + 1}.tia"
    with tia_limit_probe(hardware, output_tia_name, args.tia_limit, saturated) as probe:
        image, _, _, _ = reconstruct(spec, data, device, args.micro_batch, collect_weight_stats=False)
    ssim_db, ssim_env = metric_pair(image, data.gt_db)
    return (
        {
            "scene": scene_id,
            "saturated": saturated,
            "tia_limit": float(args.tia_limit),
            "tia_saturation_fraction": int(probe["clipped"]) / max(int(probe["total"]), 1),
            "tia_names": json.dumps(probe["tia_names"], ensure_ascii=False, sort_keys=True),
            "tia_maxima": json.dumps(probe["tia_maxima"], ensure_ascii=False, sort_keys=True),
            "ssim_db": ssim_db,
            "ssim_env": ssim_env,
        },
        image,
    )


def run_saturation(args: argparse.Namespace) -> None:
    test_frames = read_test_indices(args.split_file.resolve(), args.test_h5.resolve())
    results = [
        evaluate_test_frames(args.source, args.label, args, test_frames, False),
        evaluate_test_frames(args.source, args.label, args, test_frames, True),
    ]
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.output.resolve().with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    for row in results:
        print(
            f"[RESULT] saturated={row['saturated']} | SSIM={row['ssim_db_mean']:.6f}±{row['ssim_db_std']:.6f} | "
            f"TIA_sat={row['tia_saturation_fraction']:.6e}"
        )


def run_scenes(args: argparse.Namespace) -> None:
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    scene_config = run_one.load_config(args.scene_config.resolve())
    scenes = [run_one.find_scene(scene_config, item.strip()) for item in args.scenes.split(",") if item.strip()]
    all_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for scene in scenes:
        scene_id = str(scene["id"])
        h5_path = run_one.resolve_path(scene["h5_path"])
        data = load_scene(h5_path, int(scene["sample_idx"]))
        scene_dir = args.output / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        no_row, no_image = evaluate_scene(args.source, args.label, scene_id, data, args, False)
        sat_row, sat_image = evaluate_scene(args.source, args.label, scene_id, data, args, True)
        all_rows.extend((no_row, sat_row))
        comparison_path = scene_dir / "comparison.npy"
        np.save(comparison_path, np.stack((data.gt_db, no_image, sat_image)))
        metrics_dir = scene_dir / "metrics"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "evaluation" / "evaluate.py"),
                "--comparison_npy",
                str(comparison_path),
                "--methods",
                f"{args.label},{args.label}_tia1",
                "--method_labels",
                f"{args.label} (no clamp),{args.label} (TIA=1.0)",
                "--h5_path",
                str(h5_path),
                "--h5_sample_idx",
                str(scene["sample_idx"]),
                "--dr",
                "60",
                "--out_dir",
                str(metrics_dir),
                "--phantom_mode",
                "auto",
                "--phantom_source",
                "auto",
            ],
            cwd=ROOT,
            check=True,
        )
        summary_path = metrics_dir / "summary_metrics.csv"
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    row["scene"] = scene_id
                    row["tia_limit"] = args.tia_limit
                    method_name = str(row.get("method", "")).lower()
                    row["tia_saturation_fraction"] = (
                        sat_row["tia_saturation_fraction"]
                        if "tia=1.0" in method_name or method_name == f"{args.label}_tia1".lower()
                        else no_row["tia_saturation_fraction"]
                    )
                    metric_rows.append(row)
        (scene_dir / "tia_summary.json").write_text(
            json.dumps({"no_clamp": no_row, "tia_limit": sat_row}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    write_rows(args.output / "tia_scene_summary.csv", all_rows)
    write_rows(args.output / "four_scene_metrics.csv", metric_rows)
    print(f"[OUTPUT] {args.output}")
    for row in all_rows:
        print(
            f"[RESULT] scene={row['scene']} | saturated={row['saturated']} | "
            f"SSIM={float(row['ssim_db']):.6f} | TIA_sat={float(row['tia_saturation_fraction']):.6e}"
        )


def add_tia_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--label", default="source")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--eval-config", type=Path, default=DEFAULT_HARDWARE_CONFIG)
    parser.add_argument("--profile", default="common")
    parser.add_argument("--tia-limit", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--micro-batch", type=int, default=8192)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MBAN TIA behavior evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    saturation_parser = subparsers.add_parser("saturation", help="测试集 TIA 饱和")
    add_tia_arguments(saturation_parser)
    saturation_parser.add_argument("--split-file", type=Path, required=True)

    scenes_parser = subparsers.add_parser("scenes", help="四场景 TIA 饱和")
    add_tia_arguments(scenes_parser)
    scenes_parser.add_argument("--scene-config", type=Path, default=DEFAULT_SCENE_CONFIG)
    scenes_parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))

    args = parser.parse_args(argv)
    if args.command == "saturation":
        args.test_h5 = configured_h5_files(args.config)[0]
    if args.tia_limit <= 0.0 or args.micro_batch <= 0:
        raise ValueError("tia-limit and micro-batch must be positive")
    if args.command == "saturation":
        run_saturation(args)
    else:
        run_scenes(args)

def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_sweep_models(values: list[str]) -> list[tuple[str, Path]]:
    models: list[tuple[str, Path]] = []
    labels: set[str] = set()
    for item in values:
        label, separator, path = item.partition("=")
        if not separator or not label.strip() or not path.strip():
            raise ValueError(f"model must be LABEL=CHECKPOINT: {item}")
        label = label.strip()
        if label in labels:
            raise ValueError(f"model labels must be unique: {label}")
        labels.add(label)
        models.append((label, Path(path.strip()).resolve()))
    return models


def deduplicate_hardware_overrides(overrides: list[str]) -> list[str]:
    result: list[str] = []
    positions: dict[str, int] = {}
    for item in overrides:
        key, separator, _ = item.partition("=")
        if not separator:
            result.append(item)
            continue
        if key in positions:
            result[positions[key]] = item
        else:
            positions[key] = len(result)
            result.append(item)
    return result


def write_hardware_eval_snapshot(
    args: argparse.Namespace, config: dict[str, object], inter_layer: str | None = None
) -> None:
    runtime = {
        "profile": args.profile or "common",
        "tile_rows": args.tile_rows,
        "tile_cols": args.tile_cols,
        "interpolation_bits": (config.get("hardware", {}) or {}).get("interpolation_bits", 32),
        "inter_layer": inter_layer or args.inter_layer or config.get("hardware", {}).get("inter_layer"),
        "inter_layer_bits": (config.get("hardware", {}) or {}).get("inter_layer_bits", 4),
        "mc_runs": args.mc_runs,
    }
    (args.output / "hardware_eval_config_used.json").write_text(
        json.dumps(
            {"source": str(args.eval_config.resolve()), "config": config, "runtime": runtime},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def checkpoint_layer_index(key: str) -> int | None:
    match = re.fullmatch(r"fc(\d+)\.weight", key)
    return int(match.group(1)) if match else None


def checkpoint_hidden_layers(model_config: dict[str, object], state: dict[str, object]) -> int:
    configured = model_config.get("hidden_layers")
    if configured is not None:
        hidden_layers = int(configured)
    else:
        indices = [index for key in state if (index := checkpoint_layer_index(key)) is not None]
        if not indices:
            raise ValueError("checkpoint 中无法推导 hidden_layers")
        hidden_layers = max(indices) - 1
    if hidden_layers < 1:
        raise ValueError("checkpoint 必须包含至少一个隐藏层")
    return hidden_layers


def checkpoint_bias_layers(model_config: dict[str, object], hidden_layers: int) -> set[str]:
    value = model_config.get("bias_layers")
    if value is None or (isinstance(value, str) and value.strip().lower() == "all"):
        return {f"fc{i}" for i in range(1, hidden_layers + 2)}
    return {str(layer) for layer in value}


def checkpoint_bias_implementation(
    model_config: dict[str, object], hardware_config: dict[str, object]
) -> str:
    value = model_config.get("bias_implementation", hardware_config.get("bias_implementation", "ordinary"))
    if value not in BIAS_IMPLEMENTATIONS:
        raise ValueError(f"不支持的 bias_implementation: {value!r}")
    return str(value)


def checkpoint_layer_bias_implementation(bias_implementation: str, inter_layer: str) -> str:
    return resolve_layer_bias_implementation(bias_implementation, inter_layer)


def checkpoint_hidden_width(model_config: dict[str, object], state: dict[str, object]) -> int:
    configured = int(model_config.get("hidden_width", 0) or 0)
    if configured > 0:
        return configured
    first_hidden = state.get("fc1.weight")
    if first_hidden is None or len(first_hidden.shape) != 2:
        raise ValueError("checkpoint 中无法推导 hidden_width")
    return int(first_hidden.shape[0])


def read_hidden_signal_references(payload: dict[str, object]) -> dict[str, float]:
    shared = payload.get("shared_reference", {})
    references = shared.get("hidden_signal_references") if isinstance(shared, dict) else None
    if references is None:
        definition = payload.get("definition", {})
        inter_layer = definition.get("inter_layer") if isinstance(definition, dict) else None
        architectures = {inter_layer} if isinstance(inter_layer, str) else set(inter_layer or [])
        if architectures and architectures <= {"digital", "none"}:
            return {}
        raise ValueError("校准文件缺少 Analog hidden_signal_references")
    if not isinstance(references, dict):
        raise ValueError("校准文件的 hidden_signal_references 必须是映射")
    result = {str(layer): float(value) for layer, value in references.items()}
    if any(not math.isfinite(value) or value <= 0.0 for value in result.values()):
        raise ValueError("校准文件的 hidden_signal_references 必须是有限正数")
    return result


def run_hardware_stress(args: argparse.Namespace, root: Path) -> None:
    if args.split_file is None or args.references_json is None:
        raise ValueError("hardware 模式需要 --split-file 和 --references-json")
    if args.profile is not None and args.profile not in HARDWARE_STRESS_PROFILES:
        raise ValueError(f"hardware 模式只支持压力 profile: {', '.join(sorted(HARDWARE_STRESS_PROFILES))}")
    selected_profile = str(args.profile or "common")
    payload = json.loads(args.references_json.resolve().read_text(encoding="utf-8"))
    references = read_hidden_signal_references(payload)
    requests = getattr(args, "model_requests", None)
    models = (
        [(request.display_name, request.path, request.overrides) for request in requests]
        if requests is not None
        else [(label, path, ()) for label, path in parse_sweep_models(args.model)]
    )
    for _, checkpoint_path, _ in models:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"{checkpoint_path} 不是有效训练检查点")
        validate_deployment_checkpoint(checkpoint, checkpoint_path)
    eval_config = load_hardware_eval_config(args.eval_config)
    configured_inter_layer = (eval_config.get("hardware", {}) or {}).get("inter_layer")
    effective_inter_layer = str(args.inter_layer or configured_inter_layer or "").strip()
    if effective_inter_layer not in INTER_LAYER_MODES:
        raise ValueError("hardware 模式必须明确 inter_layer=none、analog 或 digital")
    write_hardware_eval_snapshot(args, eval_config, effective_inter_layer)
    ref_text = "{" + ", ".join(f"{key}: {float(value):.12g}" for key, value in references.items()) + "}"
    context = {
        "references": ref_text,
        "tile_rows": args.tile_rows,
        "tile_cols": args.tile_cols,
        "seed": args.seed,
    }
    named_cases = [
        case
        for case in eval_config["stress"]["cases"]
        if str(case["name"]) == selected_profile
    ]
    cases = named_cases or [
        case for case in eval_config["stress"]["cases"] if str(case["profile"]) == selected_profile
    ]
    if not cases and selected_profile == "ideal":
        cases = [
            {"name": "ideal", "profile": "ideal", "protocol": "mc", "noise_enabled": False, "overrides": {}}
        ]
    if len(cases) != 1:
        raise ValueError(f"hardware_eval.stress 必须为场景 {selected_profile} 提供唯一 case")
    if cases[0]["protocol"] != "mc":
        raise ValueError(f"场景 {selected_profile} 必须使用 mc 协议")
    evaluator = root / "evaluate_mban.py"
    rows: list[dict[str, object]] = []
    for case in cases:
        case_name = f"{selected_profile}_{effective_inter_layer}"
        default_mc_runs = eval_config["stress"]["mc_runs"]
        mc_runs = int(args.mc_runs if args.mc_runs is not None else case.get("mc_runs", default_mc_runs))
        if mc_runs <= 1:
            raise ValueError("mc-runs must be greater than one")
        profile_overrides = training.load_nonideal_profile(
            args.config.resolve(),
            selected_profile,
            args.eval_config,
            inter_layer=effective_inter_layer,
        )
        all_overrides = format_hardware_eval_overrides(hardware_eval_default_overrides(eval_config), context)
        all_overrides.extend(profile_overrides)
        all_overrides.extend(format_hardware_eval_overrides(eval_config["stress"]["base_overrides"], context))
        all_overrides.extend(format_hardware_eval_overrides(case["overrides"], context))
        all_overrides.extend(format_hardware_eval_overrides({"noise_enabled": case["noise_enabled"]}, context))
        all_overrides.append(f"inter_layer={effective_inter_layer}")
        all_overrides = deduplicate_hardware_overrides(all_overrides)
        factor_dir = args.output / case_name
        factor_dir.mkdir(parents=True, exist_ok=True)
        for label, checkpoint, request_overrides in models:
            run_dir = factor_dir / label
            model_label = f"{label}_{case_name}"
            overrides = [*request_overrides, *all_overrides]
            override_config = {}
            for item in overrides:
                key, separator, raw_value = item.partition("=")
                if not separator:
                    raise ValueError(f"硬件评估 override 必须是 KEY=VALUE: {item!r}")
                try:
                    override_config[key] = json.loads(raw_value)
                except json.JSONDecodeError:
                    override_config[key] = raw_value
            with tempfile.TemporaryDirectory(prefix="hardware_stress_") as temp_dir:
                model_config = Path(temp_dir) / "models.json"
                model_config.write_text(
                    json.dumps(
                        {
                            "models": [
                                {
                                    "display_name": model_label,
                                    "checkpoint": str(checkpoint),
                                    "model_type": "ptq",
                                    "overrides": override_config,
                                }
                            ]
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                command = [
                    sys.executable,
                    "-X",
                    "utf8",
                    str(evaluator),
                    "--mode",
                    "mc",
                    "--models-config",
                    str(model_config),
                    "--split-file",
                    str(args.split_file.resolve()),
                    "--output",
                    str(run_dir),
                    "--device",
                    args.device,
                    "--micro-batch",
                    str(args.micro_batch),
                    "--runs",
                    str(mc_runs),
                    "--seed",
                    str(args.seed),
                    "--eval-config",
                    str(args.eval_config.resolve()),
                    "--config",
                    str(args.config.resolve()),
                    "--profile",
                    selected_profile,
                ]
                subprocess.run(command, cwd=root, check=True)
            summary = read_csv_rows(run_dir / "test_mc_summary.csv")[0]
            rows.append(
                {
                    "factor": case_name,
                    "protocol": "mc",
                    "model": label,
                    "runs": int(summary["runs"]),
                    "frames_per_run": int(summary["frames_per_run"]),
                    "test_ssim_db_mean": float(summary["ssim_db_mean"]),
                    "test_ssim_db_std": float(summary["ssim_db_std"]),
                    "test_ssim_db_p05": float(summary["ssim_db_p05"]),
                    "test_ssim_db_p95": float(summary["ssim_db_p95"]),
                    "test_ssim_env_mean": float(summary["ssim_env_mean"]),
                    "test_ssim_env_std": float(summary["ssim_env_std"]),
                    "test_ssim_env_p05": float(summary["ssim_env_p05"]),
                    "test_ssim_env_p95": float(summary["ssim_env_p95"]),
                }
            )
    fields = sorted({key for row in rows for key in row})
    with (args.output / f"{selected_profile}_{effective_inter_layer}_mc.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def quantiles(values: list[np.ndarray]) -> dict[str, float]:
    if not values:
        raise ValueError("quantiles requires at least one sample array")
    merged = np.concatenate(values)
    if merged.size == 0 or not np.isfinite(merged).all():
        raise ValueError("quantiles 输入必须包含有限样本")
    q = np.quantile(merged, [0.5, 0.95, 0.99, 0.999, 1.0])
    return {
        "p50": float(q[0]),
        "p95": float(q[1]),
        "p99": float(q[2]),
        "p999": float(q[3]),
        "max": float(q[4]),
        "samples": int(merged.size),
    }


def run_calibration(args: argparse.Namespace, device: torch.device) -> None:
    if args.split_file is None:
        raise ValueError("calibrate 模式需要 --split-file")
    eval_config = load_hardware_eval_config(args.eval_config)
    write_hardware_eval_snapshot(args, eval_config)
    base_overrides = {
        **hardware_eval_default_overrides(eval_config),
        **eval_config["stress"]["base_overrides"],
    }
    calibration_fields = {
        "weight_bits",
        "bias_bits",
        "input_bits",
        "control_bits",
        "interpolation_bits",
        "inter_layer",
        "inter_layer_bits",
    }
    context = {}
    overrides = ["mode=ptq"] + format_hardware_eval_overrides(
        {key: value for key, value in base_overrides.items() if key in calibration_fields}, context
    )
    overrides.append("noise_enabled=false")
    if args.inter_layer is not None:
        overrides.append(f"inter_layer={args.inter_layer}")
    overrides = deduplicate_hardware_overrides(overrides)
    specs = [
        load_evaluation_model(
            label,
            path,
            "ptq",
            device,
            config_path=args.config,
            profile=args.profile,
            hardware_config_path=args.eval_config,
            overrides=overrides,
        )
        for label, path in args.model
    ]
    calibration_args = argparse.Namespace(
        micro_batch=args.micro_batch,
        output=args.output,
        split_file=args.split_file,
    )
    calibrate_ptq_observers(calibration_args, specs, device)
    split = training.load_or_create_mixed_split(
        [str(Path(path).resolve()) for path in training.args.h5_files],
        training.args.train_ratio,
        training.args.validation_ratio,
        training.args.test_ratio,
        split_file=args.split_file.resolve(),
        seed=int(training.args.seed),
        make_new_split=bool(training.args.make_new_split),
    )
    samples: list[tuple[Path, int]] = []
    for h5_path in training.args.h5_files:
        train_indices = split[str(Path(h5_path).resolve())]["train"]
        count = min(int(training.args.ptq_calibration_samples), len(train_indices))
        positions = np.linspace(0, len(train_indices) - 1, count).round().astype(int)
        samples.extend((Path(h5_path).resolve(), int(train_indices[pos])) for pos in positions)
    if not samples:
        raise RuntimeError("PTQ 校准未找到任何训练样本")
    collected: dict[str, dict[str, object]] = {spec.display_name: {"tia": [], "split_signal": {}} for spec in specs}
    originals: dict[str, tuple[object, object, object]] = {}
    for spec in specs:
        hardware = spec.model.hardware_qat
        original_tia, original_dual = hardware.tia, hardware.dual_mismatch
        original_activation = hardware.activation_mismatch
        originals[spec.display_name] = (original_tia, original_dual, original_activation)

        def wrapped_tia(
            x,
            name="tia",
            common_mode=None,
            *,
            _original=original_tia,
            _values=collected[spec.display_name]["tia"],
        ):
            _values.append(x.detach().abs().float().cpu().numpy().reshape(-1))
            return _original(x, name, common_mode=common_mode)

        def wrapped_dual(
            x_positive,
            x_negative,
            layer=None,
            *,
            _original=original_dual,
            _values=collected[spec.display_name]["split_signal"],
        ):
            positive, negative = _original(x_positive, x_negative, layer)
            _values.setdefault(layer or "unknown", []).append(
                torch.cat((positive.detach().abs().flatten(), negative.detach().abs().flatten())).float().cpu().numpy()
            )
            return positive, negative

        def wrapped_activation(
            x,
            layer,
            *,
            _original=original_activation,
            _values=collected[spec.display_name]["split_signal"],
        ):
            result = _original(x, layer)
            _values.setdefault(layer or "unknown", []).append(
                result.detach().abs().float().cpu().numpy().reshape(-1)
            )
            return result

        hardware.tia = wrapped_tia
        hardware.dual_mismatch = wrapped_dual
        hardware.activation_mismatch = wrapped_activation
    try:
        for h5_path, frame in samples:
            data = load_scene(h5_path, frame)
            for spec in specs:
                reconstruct(spec, data, device, args.micro_batch, collect_weight_stats=False)
    finally:
        for spec in specs:
            hardware = spec.model.hardware_qat
            original_tia, original_dual, original_activation = originals[spec.display_name]
            hardware.tia = original_tia
            hardware.dual_mismatch = original_dual
            hardware.activation_mismatch = original_activation
    model_stats = {}
    inter_layers = sorted({str(spec.model.hardware_qat.config.inter_layer) for spec in specs})
    if "analog" in inter_layers and any(
        not collected[spec.display_name]["split_signal"] for spec in specs
    ):
        raise RuntimeError("Analog 校准未采集到隐层信号；请确认模型包含可观测的隐层激活")
    for spec in specs:
        split_values = collected[spec.display_name]["split_signal"]
        model_stats[spec.display_name] = {
            "tia": quantiles(collected[spec.display_name]["tia"]),
            "hidden_signal": {layer: quantiles(values) for layer, values in split_values.items()},
        }
    tia_reference = max(values["tia"]["p99"] for values in model_stats.values())
    layers = sorted({layer for values in model_stats.values() for layer in values["hidden_signal"]})
    hidden_signal_reference = {
        layer: max(
            values["hidden_signal"][layer]["p99"] for values in model_stats.values() if layer in values["hidden_signal"]
        )
        for layer in layers
    }
    result = {
        "definition": {
            "tia_reference": "max model q99 of |TIA input| over fixed training calibration frames",
            "hidden_signal_reference": "max model q99 per hidden layer of post-split, pre-normalization signal over fixed training calibration frames",
            "calibration_samples_per_h5": int(training.args.ptq_calibration_samples),
            "samples": [{"h5": str(path), "frame": frame} for path, frame in samples],
            "models": {spec.display_name: str(spec.path) for spec in specs},
            "inter_layer": inter_layers,
        },
        "shared_reference": {"tia_reference": tia_reference, "hidden_signal_references": hidden_signal_reference},
        "models": model_stats,
    }
    (args.output / "nonideal_references.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output / "nonideal_references.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        signal_fields = [f"hidden_signal_{layer}_p99" for layer in layers]
        writer = csv.DictWriter(handle, fieldnames=("model", "tia_p99", *signal_fields))
        writer.writeheader()
        for label, values in model_stats.items():
            writer.writerow(
                {
                    "model": label,
                    "tia_p99": values["tia"]["p99"],
                    **{
                        f"hidden_signal_{layer}_p99": values["hidden_signal"].get(layer, {}).get("p99", "")
                        for layer in layers
                    },
                }
            )
        writer.writerow(
            {
                "model": "SHARED_MAX_Q99",
                "tia_p99": tia_reference,
                **{f"hidden_signal_{layer}_p99": hidden_signal_reference.get(layer, "") for layer in layers},
            }
        )


def percentile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.float().reshape(-1), values.new_tensor(q)).item())


def quantize_mapped_weight(
    weight: torch.Tensor,
    bits: int,
    g_min: float,
    g_max: float,
    per_output_channel: bool,
) -> dict[str, object]:
    weight = weight.detach().float().cpu()
    absolute = weight.abs()
    scale = absolute.amax(dim=1, keepdim=True) if per_output_channel else absolute.amax().reshape(1, 1)
    safe_scale = scale.clamp_min(1.0e-12)
    span = float(g_max - g_min)
    plus = float(g_min) + weight.relu() * span / safe_scale
    minus = float(g_min) + (-weight).relu() * span / safe_scale
    q_plus = q_minus = None
    states = 0
    if bits < 32:
        states = 1 << bits
        step = span / max(states - 1, 1)
        q_plus = torch.round((plus - g_min) / step).clamp(0, states - 1).to(torch.int64)
        q_minus = torch.round((minus - g_min) / step).clamp(0, states - 1).to(torch.int64)
        mapped = safe_scale * ((q_plus - q_minus) * step) / span
        endpoint = ((q_plus >= states - 1) | (q_minus >= states - 1)).float().mean()
    else:
        mapped = weight
        endpoint = ((plus >= g_max - 1.0e-12) | (minus >= g_max - 1.0e-12)).float().mean()
    return {
        "weight": weight,
        "scale": scale,
        "mapped": mapped,
        "q_plus": q_plus,
        "q_minus": q_minus,
        "states": states,
        "endpoint": endpoint,
    }


def mapping_metrics(
    weight: torch.Tensor,
    bits: int,
    g_min: float,
    g_max: float,
    per_output_channel: bool,
) -> dict[str, float]:
    mapped = quantize_mapped_weight(weight, bits, g_min, g_max, per_output_channel)
    source = mapped["weight"]
    target = mapped["mapped"]
    scale = mapped["scale"]
    quantization_rmse = torch.sqrt(torch.mean((target - source).square()))
    return {
        "mapping_scale_max": float(scale.max().item()),
        "mapping_scale_mean": float(scale.mean().item()),
        "conductance_endpoint_fraction": float(mapped["endpoint"].item()),
        "quantization_rmse": float(quantization_rmse.item()),
    }


def mapping_row_metrics(
    weight: torch.Tensor,
    bits: int,
    g_min: float,
    g_max: float,
    per_output_channel: bool,
) -> list[dict[str, object]]:
    mapped = quantize_mapped_weight(weight, bits, g_min, g_max, per_output_channel)
    source = mapped["weight"]
    target = mapped["mapped"]
    q_plus = mapped["q_plus"]
    q_minus = mapped["q_minus"]
    states = int(mapped["states"])
    scale = mapped["scale"].reshape(-1)
    row_max = source.abs().amax(dim=1)
    layer_max = source.abs().amax().clamp_min(1.0e-12)
    if scale.numel() == 1:
        scale = scale.expand_as(row_max)
    rms = torch.sqrt(torch.mean(source.square(), dim=1)).clamp_min(1.0e-12)
    error = torch.sqrt(torch.mean((target - source).square(), dim=1)) / rms
    rows: list[dict[str, object]] = []
    for index in range(source.shape[0]):
        if q_plus is None or q_minus is None:
            plus_states = minus_states = signed_states = 0
            zero_fraction = 0.0
            endpoint_fraction = 0.0
        else:
            plus_row = q_plus[index]
            minus_row = q_minus[index]
            plus_states = int(torch.unique(plus_row).numel())
            minus_states = int(torch.unique(minus_row).numel())
            signed_states = int(torch.unique(plus_row - minus_row).numel())
            zero_fraction = float(((plus_row == 0) & (minus_row == 0)).float().mean().item())
            endpoint_fraction = float(
                ((plus_row == (states - 1)) | (minus_row == (states - 1))).float().mean().item()
            )
        rows.append(
            {
                "row": index,
                "r_j": float((row_max[index] / layer_max).item()),
                "E_j": float(error[index].item()),
                "mapping_scale": float(scale[index].item()),
                "plus_state_count": plus_states,
                "minus_state_count": minus_states,
                "signed_state_count": signed_states,
                "zero_code_fraction": zero_fraction,
                "endpoint_fraction": endpoint_fraction,
            }
        )
    return rows


def run_deployment(args: argparse.Namespace, device: torch.device) -> None:
    requests = getattr(args, "model_requests", None)
    if requests is None:
        models = [(label, path, "ptq", None, ()) for label, path in (parse_checkpoint_model(value) for value in args.model)]
    else:
        models = [
            (request.display_name, request.path, request.model_type, request.profile, request.overrides)
            for request in requests
        ]
    specs = [
        load_evaluation_model(
            label,
            path,
            model_type,
            device,
            config_path=args.config,
            profile=profile or args.profile,
            hardware_config_path=args.eval_config,
            overrides=overrides,
            require_deployment=True,
        )
        for label, path, model_type, profile, overrides in models
    ]
    rows: list[dict[str, object]] = []
    row_metrics: list[dict[str, object]] = []
    for spec in specs:
        state = {
            name: tensor.detach().cpu()
            for name, tensor in spec.model.state_dict().items()
            if name.startswith("fc") and (name.endswith(".weight") or name.endswith(".bias"))
        }
        hardware = getattr(spec.model, "hardware_qat", None)
        cfg = getattr(hardware, "config", None)
        bits = int(getattr(cfg, "weight_bits", 32))
        g_min = float(args.g_min if args.g_min is not None else getattr(cfg, "g_min", 0.0))
        g_max = float(args.g_max if args.g_max is not None else getattr(cfg, "g_max", 1.0))
        per_channel = bool(args.per_output_channel or getattr(cfg, "per_output_channel", False))
        if not 2 <= bits <= 32:
            raise ValueError("weight-bits 必须在 [2, 32]")
        if g_max <= g_min:
            raise ValueError("g-max 必须大于 g-min")
        for name, tensor in sorted(state.items()):
            if not name.endswith(".weight"):
                continue
            weight = tensor.detach().float().cpu()
            absolute = weight.abs()
            row: dict[str, object] = {
                "method": spec.display_name,
                "stage": evaluation_stage(spec.model),
                "layer": name.removesuffix(".weight"),
                "rows": int(weight.shape[0]),
                "cols": int(weight.shape[1]),
                "parameters": int(weight.numel()),
                "abs_min": float(absolute.min().item()),
                "abs_p1": percentile(absolute, 0.01),
                "abs_p99": percentile(absolute, 0.99),
                "abs_p999": percentile(absolute, 0.999),
                "abs_max": float(absolute.max().item()),
                "rms": float(torch.sqrt(torch.mean(weight.square())).item()),
                "positive_fraction": float((weight > 0).float().mean().item()),
                "negative_fraction": float((weight < 0).float().mean().item()),
                "weight_bits": bits,
                "g_min": g_min,
                "g_max": g_max,
                "per_output_channel": per_channel,
            }
            row.update(mapping_metrics(weight, bits, g_min, g_max, per_channel))
            rows.append(row)
            row_metrics.extend(
                {
                "method": spec.display_name,
                    "stage": evaluation_stage(spec.model),
                    "layer": name.removesuffix(".weight"),
                    "rows": int(weight.shape[0]),
                    "cols": int(weight.shape[1]),
                    "weight_bits": bits,
                    "per_output_channel": per_channel,
                    **item,
                }
                for item in mapping_row_metrics(weight, bits, g_min, g_max, per_channel)
            )
    if not rows:
        raise RuntimeError("checkpoint 中没有可部署线性权重")
    write_rows(args.output / "deployment_weight_metrics.csv", rows)
    write_rows(args.output / "deployment_weight_row_metrics.csv", row_metrics)
    summary: list[dict[str, object]] = []
    for method in dict.fromkeys(str(row["method"]) for row in rows):
        method_rows = [row for row in rows if row["method"] == method]
        summary.append(
            {
                "method": method,
                "stage": method_rows[0]["stage"],
                "layers": len(method_rows),
                "parameters": sum(int(row["parameters"]) for row in method_rows),
                "max_abs_weight": max(float(row["abs_max"]) for row in method_rows),
                "max_abs_p99": max(float(row["abs_p99"]) for row in method_rows),
                "max_abs_p999": max(float(row["abs_p999"]) for row in method_rows),
                "max_mapping_scale": max(float(row["mapping_scale_max"]) for row in method_rows),
                "mean_quantization_rmse": float(np.mean([float(row["quantization_rmse"]) for row in method_rows])),
                "mean_endpoint_fraction": float(
                    np.mean([float(row["conductance_endpoint_fraction"]) for row in method_rows])
                ),
            }
        )
    write_rows(args.output / "deployment_weight_metrics_summary.csv", summary)


def mark_derived_checkpoint(checkpoint: dict[str, object], source: Path, operation: str) -> None:
    checkpoint["derived_checkpoint"] = True
    checkpoint["derived_from"] = str(source.resolve())
    checkpoint["derived_operation"] = operation
    checkpoint["training_resume_allowed"] = False


def _linear_state_layout(
    state: dict[str, object], deployment: dict[str, object], model_config: dict[str, object], source: Path
) -> list[tuple[str, bool]]:
    try:
        total_layers = int(model_config["total_fc_layers"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source} 缺少有效的 model_config.total_fc_layers") from exc
    if total_layers < 1:
        raise ValueError(f"{source} 的 total_fc_layers 必须为正数")
    expected: set[str] = set()
    layers: list[tuple[str, bool]] = []
    for index in range(1, total_layers + 1):
        name = f"fc{index}"
        weight_key = f"{name}.weight"
        bias_key = f"{name}.bias"
        if weight_key not in state:
            raise ValueError(f"{source} 缺少 {weight_key}")
        if not isinstance(state[weight_key], torch.Tensor) or state[weight_key].ndim != 2:
            raise ValueError(f"{source} 的 {weight_key} 必须是二维张量")
        expected.add(weight_key)
        has_bias = bias_key in state
        if has_bias:
            if not isinstance(state[bias_key], torch.Tensor) or state[bias_key].ndim != 1:
                raise ValueError(f"{source} 的 {bias_key} 必须是一维张量")
            expected.add(bias_key)
        layers.append((name, has_bias))
    actual = set(deployment)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"缺少 {missing}")
        if unexpected:
            details.append(f"多余 {unexpected}")
        raise ValueError(f"{source} 的 deployment_linear_state 不完整：{'；'.join(details)}")
    for key in sorted(expected):
        value = deployment[key]
        source_value = state[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{source} 的 deployment_linear_state[{key}] 不是张量")
        if value.shape != source_value.shape:
            raise ValueError(
                f"{source} 的 deployment_linear_state[{key}] 形状 {tuple(value.shape)} "
                f"与 model_state_dict 形状 {tuple(source_value.shape)} 不一致"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"{source} 的 deployment_linear_state[{key}] 含非有限值")
    return layers


def _effective_checkpoint_weight(
    state: dict[str, object], layer: str, model_config: dict[str, object]
) -> torch.Tensor:
    weight = state[f"{layer}.weight"]
    assert isinstance(weight, torch.Tensor)
    weight = weight.detach().float()
    transform_layers = {str(value) for value in model_config.get("weight_transform_layers", [])}
    if model_config.get("weight_transform", "none") == "ws" and layer in transform_layers:
        mean = weight.mean(dim=1, keepdim=True)
        variance = (weight - mean).square().mean(dim=1, keepdim=True)
        eps = float(model_config.get("weight_transform_epsilon", 1.0e-5))
        weight = (weight - mean) * torch.rsqrt(variance + eps)
    return weight


def _checkpoint_bias_layers(model_config: dict[str, object], total_layers: int) -> set[str]:
    value = model_config.get("bias_layers", [])
    if isinstance(value, str) and value.strip().lower() == "all":
        return {f"fc{index}" for index in range(1, total_layers + 1)}
    return {str(layer) for layer in value}


def _apply_static_normalization(
    value: torch.Tensor, layer: str, state: dict[str, object], model_config: dict[str, object]
) -> torch.Tensor:
    prefix = f"running_standardizers.{layer}."
    mean = state.get(prefix + "running_mean")
    variance = state.get(prefix + "running_var")
    if not isinstance(mean, torch.Tensor) or not isinstance(variance, torch.Tensor):
        raise ValueError(f"缺少 {prefix}running_mean/running_var")
    eps = float(model_config.get("running_stat_epsilon", 1.0e-5))
    normalized = (value - mean.detach().float()) * torch.rsqrt(variance.detach().float() + eps)
    if model_config.get("normalization") == "batch_renorm":
        affine_weight = state.get(prefix + "weight")
        affine_bias = state.get(prefix + "bias")
        if not isinstance(affine_weight, torch.Tensor) or not isinstance(affine_bias, torch.Tensor):
            raise ValueError(f"缺少 {prefix}weight/bias")
        normalized = normalized * affine_weight.detach().float() + affine_bias.detach().float()
    return normalized


def _fold_parity(
    state: dict[str, object], deployment: dict[str, object], model_config: dict[str, object], layers: list[tuple[str, bool]]
) -> dict[str, object]:
    normalization = str(model_config.get("normalization", "none"))
    normalization_layers = {str(layer) for layer in model_config.get("normalization_layers", [])}
    bias_layers = _checkpoint_bias_layers(model_config, len(layers))
    generator = torch.Generator().manual_seed(94040)
    reports: list[dict[str, object]] = []
    for layer, has_bias in layers:
        source_weight = _effective_checkpoint_weight(state, layer, model_config)
        folded_weight = deployment[f"{layer}.weight"].detach().float()
        source_bias = state.get(f"{layer}.bias")
        folded_bias = deployment.get(f"{layer}.bias")
        array_bias = model_config.get("bias_implementation", "ordinary") == "array" and layer in bias_layers
        inputs = torch.randn(32, source_weight.shape[1], generator=generator)
        if array_bias:
            inputs[:, -1] = 1.0
        source_output = F.linear(inputs, source_weight, None if array_bias or not has_bias else source_bias.float())
        if layer in normalization_layers:
            source_output = _apply_static_normalization(source_output, layer, state, model_config)
        folded_output = F.linear(inputs, folded_weight, None if folded_bias is None else folded_bias.detach().float())
        error = (source_output - folded_output).abs()
        max_abs = float(error.max().item()) if error.numel() else 0.0
        mean_abs = float(error.mean().item()) if error.numel() else 0.0
        reports.append({"layer": layer, "max_abs_error": max_abs, "mean_abs_error": mean_abs})
    max_error = max((float(item["max_abs_error"]) for item in reports), default=0.0)
    mean_error = float(np.mean([float(item["mean_abs_error"]) for item in reports])) if reports else 0.0
    tolerance = 2.0e-4
    if max_error > tolerance:
        raise ValueError(f"折叠前后线性等价性校验失败：max_abs_error={max_error:.6e} > {tolerance:.6e}")
    return {
        "passed": True,
        "seed": 94040,
        "samples_per_layer": 32,
        "max_abs_error": max_error,
        "mean_abs_error": mean_error,
        "tolerance": tolerance,
        "layers": reports,
        "normalization": normalization,
    }


def fold_checkpoint(source: Path, destination: Path) -> None:
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError(f"无效 checkpoint: {source}")
    if checkpoint.get("model_semantics_version") != MODEL_SEMANTICS_VERSION:
        raise ValueError(
            f"{source} 的 model_semantics_version={checkpoint.get('model_semantics_version')!r}，"
            f"当前要求 {MODEL_SEMANTICS_VERSION}；请先用当前代码重新训练"
        )
    original_model_config = copy.deepcopy(checkpoint.get("model_config", {}))
    if not isinstance(original_model_config, dict):
        raise ValueError(f"{source} 缺少有效 model_config")
    model_config = copy.deepcopy(original_model_config)
    normalization = model_config.get("normalization", "none")
    if normalization not in {"running_zscore", "batch_renorm", "none"}:
        raise ValueError(f"{source} 含动态归一化 {normalization}，不能融合")
    if model_config.get("centering", "none") != "none" or model_config.get("centering_layers", []):
        raise ValueError(f"{source} 含动态 sample_mean 中心化，不能形成部署态 checkpoint")
    normalization_layers = model_config.get("normalization_layers", [])
    if not isinstance(normalization_layers, (list, tuple)):
        raise ValueError(f"{source} 的 normalization_layers 必须是层名列表")
    if (normalization == "none") != (not normalization_layers):
        raise ValueError(f"{source} 的 normalization 与 normalization_layers 不一致")
    weight_transform = model_config.get("weight_transform", "none")
    weight_transform_layers = model_config.get("weight_transform_layers", [])
    if not isinstance(weight_transform_layers, (list, tuple)):
        raise ValueError(f"{source} 的 weight_transform_layers 必须是层名列表")
    if weight_transform not in {"none", "ws"}:
        raise ValueError(f"{source} 的 weight_transform={weight_transform!r} 不支持部署折叠")
    if (weight_transform == "none") != (not weight_transform_layers):
        raise ValueError(f"{source} 的 weight_transform 与 weight_transform_layers 不一致")
    if not isinstance(checkpoint.get("evaluation_config"), dict):
        raise ValueError(f"{source} 缺少有效 evaluation_config")
    runtime_transforms = resolve_deployment_runtime_transforms(checkpoint, source)
    deployment = checkpoint.get("deployment_linear_state")
    if not isinstance(deployment, dict):
        raise ValueError(f"{source} 缺少 deployment_linear_state")
    layers = _linear_state_layout(checkpoint["model_state_dict"], deployment, original_model_config, source)
    parity = _fold_parity(checkpoint["model_state_dict"], deployment, original_model_config, layers)

    state = dict(checkpoint["model_state_dict"])
    for key, value in deployment.items():
        state[key] = value.detach().cpu().to(dtype=state[key].dtype)
    if normalization in {"running_zscore", "batch_renorm"}:
        model_config["normalization"] = "none"
        model_config["normalization_layers"] = []
    model_config["weight_transform"] = "none"
    model_config["weight_transform_layers"] = []
    for key in list(state):
        if key.startswith("running_standardizers."):
            del state[key]
    checkpoint["model_state_dict"] = state
    hardware_state = checkpoint.get("hardware_qat_state")
    if isinstance(hardware_state, dict):
        checkpoint["hardware_qat_state"] = dict(hardware_state)
    checkpoint.pop("deployment_linear_state", None)
    checkpoint["model_config"] = model_config
    checkpoint["deployment_folded"] = True
    checkpoint["deployment_folded_from_model_config"] = original_model_config
    checkpoint["deployment_runtime_transforms"] = runtime_transforms
    checkpoint["deployment_fold_parity"] = parity
    mark_derived_checkpoint(checkpoint, source, "fold")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)
    destination.with_name(destination.name + ".fold_parity.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "destination": str(destination),
                "model_semantics_version": MODEL_SEMANTICS_VERSION,
                "deployment_runtime_transforms": runtime_transforms,
                "parity": parity,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def read_train_frames(split_file: Path, h5_path: Path, count: int) -> list[int]:
    if count <= 0:
        raise ValueError("calibration-frames must be positive")
    rows = read_split_indices(split_file, h5_path, "train")
    count = min(count, len(rows))
    positions = np.linspace(0, len(rows) - 1, count).round().astype(int)
    return [rows[int(position)] for position in positions]


def scale_output_checkpoint(args: argparse.Namespace) -> None:
    if args.calibration_frames <= 0 or args.target_utilization <= 0.0 or args.target_utilization > 1.0:
        raise ValueError("calibration-frames must be positive and target-utilization must be in (0, 1]")
    checkpoint = torch.load(args.source.resolve(), map_location="cpu", weights_only=False)
    if checkpoint.get("deployment_folded") is not True:
        raise ValueError(f"{args.source} 不是部署态 checkpoint；请先执行 checkpoint fold")
    device = torch.device(args.device)
    spec = load_evaluation_model(
        "source",
        args.source.resolve(),
        "qat",
        device,
        config_path=args.config.resolve(),
        profile=args.profile,
        hardware_config_path=args.eval_config.resolve(),
    )
    h5_path = Path(training.args.h5_files[0]).resolve()
    frames = read_train_frames(args.split_file, h5_path, args.calibration_frames)
    spec.model.begin_signal_diagnostics()
    for frame in frames:
        data = load_scene(h5_path, frame)
        reconstruct(spec, data, device, args.micro_batch, collect_weight_stats=False)
    diagnostics = spec.model.end_signal_diagnostics()
    vector_stats = diagnostics.get("raw_control_vector_max")
    if not vector_stats:
        raise RuntimeError("未收集到 raw control vector max")
    p999 = float(vector_stats["p999"])
    if not math.isfinite(p999) or p999 <= 0.0:
        raise RuntimeError(f"raw_control_vector_max.p999 必须是有限正数，实际为 {p999!r}")
    scale = p999 / args.target_utilization
    state = dict(checkpoint["model_state_dict"])
    output_layer = int(checkpoint["model_config"]["hidden_layers"]) + 1
    state[f"fc{output_layer}.weight"] = state[f"fc{output_layer}.weight"] / scale
    output_bias_key = f"fc{output_layer}.bias"
    if output_bias_key in state:
        state[output_bias_key] = state[output_bias_key] / scale
    checkpoint["model_state_dict"] = state
    hardware_state = checkpoint.get("hardware_qat_state")
    if isinstance(hardware_state, dict):
        hardware_state = dict(hardware_state)
        activation_ranges = dict(hardware_state.get("activation_ranges", {}) or {})
        control_name = output_controls_name(layer_name(output_layer))
        if control_name not in activation_ranges:
            raise RuntimeError(f"checkpoint 缺少输出控制量程 {control_name}")
        activation_ranges[control_name] = float(activation_ranges[control_name]) / scale
        hardware_state["activation_ranges"] = activation_ranges
        hardware_state["weight_scales"] = {}
        checkpoint["hardware_qat_state"] = hardware_state
    checkpoint["static_output_scale"] = scale
    checkpoint["static_output_scale_definition"] = {
        "output_layer": f"fc{output_layer}",
        "calibration_frames": frames,
        "raw_control_vector_max_p999": p999,
        "target_tia_utilization": args.target_utilization,
        "profile": args.profile,
        "formula": "scale=p999(raw_control_vector_max)/target_tia_utilization",
    }
    mark_derived_checkpoint(checkpoint, args.source, "scale-output")
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output.resolve())
    args.output.resolve().with_suffix(".json").write_text(
        json.dumps(
            {
                "source": str(args.source.resolve()),
                "output": str(args.output.resolve()),
                "scale": scale,
                "diagnostics": diagnostics,
                "calibration_frames": frames,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"scale": scale, "p999": p999, "frames": frames}, ensure_ascii=False))


def checkpoint_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MBAN checkpoint deployment tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fold_parser = subparsers.add_parser("fold", help="融合静态归一化和权重变换")
    fold_parser.add_argument("source", type=Path)
    fold_parser.add_argument("destination", type=Path)

    scale_parser = subparsers.add_parser("scale-output", help="按校准集缩放输出层")
    scale_parser.add_argument("--source", type=Path, required=True)
    scale_parser.add_argument("--output", type=Path, required=True)
    scale_parser.add_argument("--split-file", type=Path, required=True)
    scale_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    scale_parser.add_argument("--eval-config", type=Path, default=DEFAULT_HARDWARE_CONFIG)
    scale_parser.add_argument("--profile", default="common")
    scale_parser.add_argument("--calibration-frames", type=int, default=32)
    scale_parser.add_argument("--target-utilization", type=float, default=0.8)
    scale_parser.add_argument("--micro-batch", type=int, default=8192)
    scale_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args(argv)
    if args.command == "fold":
        fold_checkpoint(args.source, args.destination)
        print(args.destination)
    else:
        scale_output_checkpoint(args)

if __name__ == "__main__":
    main()

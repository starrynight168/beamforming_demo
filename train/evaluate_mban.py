"""统一评估入口；具体实现位于 eval_core。"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml
import torch

_TRAIN_DIR = Path(__file__).resolve().parent
_ROOT = _TRAIN_DIR.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_TRAIN_DIR))

from eval_core import diagnostics, hardware, hardware_trace, software
from eval_core.config import (
    DEFAULT_CONFIG,
    DEFAULT_HARDWARE_CONFIG,
    DEFAULT_SCENE_CONFIG,
    DEFAULT_SCENES,
    TRAIN_DIR,
    add_model_arguments,
    parse_checkpoint_model,
    resolve_model_requests,
    resolve_path,
)
from eval_core.hardware_ppa import run_crosssim, run_mapping, run_ppa
from eval_core.hardware import checkpoint_main, run_calibration, run_deployment, run_hardware_stress
from mban_core.config import expand_hardware_eval_ppa_cases, load_hardware_eval_config

SOFTWARE_MODES = ("scenes", "test", "ptq", "mc", "weights", "stats", "figures", "trace")
HARDWARE_MODES = ("deployment", "hardware", "calibrate", "mapping", "crosssim", "ppa")
UTILITY_MODES = ("tia", "checkpoint")
ALL_MODES = (*SOFTWARE_MODES, *HARDWARE_MODES, *UTILITY_MODES)
EVAL_MODES = (*SOFTWARE_MODES, *HARDWARE_MODES)
DEFAULT_EVAL = TRAIN_DIR / "eval.yaml"
MODE_MODEL_TYPES = {
    "ptq": {"ptq"},
    "mc": {"qat", "ptq"},
    "trace": {"qat", "ptq"},
    "hardware": {"ptq"},
    "calibrate": {"ptq"},
    "deployment": {"qat", "ptq"},
    "mapping": {"qat", "ptq"},
    "crosssim": {"qat", "ptq"},
    "ppa": {"qat", "ptq"},
}


def route_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_mban.py",
        description="MBAN 软件成像、量化、硬件非理想与部署评估",
        epilog="指定 --mode 后使用 -h/--help 可查看该模式的完整参数。",
        add_help=add_help,
    )
    parser.add_argument("--mode", choices=ALL_MODES, help="单一评估协议")
    parser.add_argument("--eval", type=Path, help="评估配置 YAML；默认使用 train/eval.yaml")
    return parser


def parse_route(argv: list[str]) -> tuple[str, list[str]]:
    if not argv:
        return "eval", [str(DEFAULT_EVAL)]
    if (("-h" in argv or "--help" in argv) and "--mode" not in argv and "--eval" not in argv):
        route_parser().parse_args(argv)
    args, remaining = route_parser(add_help=False).parse_known_args(argv)
    if args.eval is not None:
        if args.mode is not None or remaining:
            raise ValueError("--eval 不能与 --mode 或其他参数同时使用")
        return "eval", [str(args.eval)]
    if args.mode is None:
        raise ValueError("必须提供 --mode 或 --eval")
    return args.mode, remaining


def parse_hardware_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MBAN 器件非理想、阵列映射、CrossSim 与 PPA")
    parser.add_argument("--mode", choices=HARDWARE_MODES, required=True)
    add_model_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--eval-config", type=Path, default=DEFAULT_HARDWARE_CONFIG)
    parser.add_argument("--profile", help="硬件压力 profile；只表示 ideal/common")
    parser.add_argument("--micro-batch", type=int, default=8192)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--weight-bits", type=int)
    parser.add_argument("--g-min", type=float)
    parser.add_argument("--g-max", type=float)
    parser.add_argument("--per-output-channel", action="store_true")
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--references-json", type=Path)
    parser.add_argument("--mc-runs", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mapping-mode", choices=("custom_compact", "fixed_tile"), default="custom_compact")
    parser.add_argument("--tile-rows", type=int, default=64)
    parser.add_argument("--tile-cols", type=int, default=64)
    parser.add_argument("--converter-schedule", choices=("time_multiplexed", "parallel"), default="parallel")
    parser.add_argument("--input-timing", choices=("analog_single_pulse", "bit_serial"), default="analog_single_pulse")
    parser.add_argument(
        "--differential-readout", choices=("post_tia_subtractor", "separate_adc"), default="post_tia_subtractor"
    )
    parser.add_argument("--crosssim-runs", type=int, default=8)
    parser.add_argument("--input-bits", type=int)
    parser.add_argument("--adc-bits", type=int)
    parser.add_argument("--inter-layer", choices=("none", "analog", "digital"))
    parser.add_argument("--inter-layer-bits", type=int, choices=range(2, 33))
    parser.add_argument("--interpolation-bits", type=int, choices=range(2, 33))
    parser.add_argument("--technology-node", type=int, default=32)
    parser.add_argument("--neurosim-distro", default="Ubuntu-24.04")
    parser.add_argument("--skip-neurosim", action="store_true")
    return parser.parse_args(argv)


def run_hardware_cli(argv: list[str]) -> None:
    args = parse_hardware_args(argv)
    if args.micro_batch <= 0:
        raise ValueError("micro_batch 必须大于 0")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    requests = resolve_model_requests(args.model, args.models_config, args.model_type)
    args.model_requests = requests
    args.model = [f"{request.display_name}={request.path}" for request in requests]
    display_names = [value.split("=", 1)[0] for value in args.model]
    if len(display_names) != len(set(display_names)):
        raise ValueError(f"display_name 必须唯一: {display_names}")
    details = [
        f"models={len(display_names)} ({', '.join(display_names)})",
        f"device={device}",
        f"micro_batch={args.micro_batch}",
    ]
    if args.mode in {"mapping", "crosssim", "ppa"}:
        details.append(f"mapping={args.mapping_mode} | tile={args.tile_rows}x{args.tile_cols}")
    if args.mode == "hardware":
        details.append(f"mc_runs={args.mc_runs if args.mc_runs is not None else 'eval-config'}")
    print("=" * 72)
    print(f"[EVAL:{args.mode}] " + " | ".join(details))
    if args.mode == "hardware":
        run_hardware_stress(args, TRAIN_DIR)
    elif args.mode == "calibrate":
        args.model = [parse_checkpoint_model(value) for value in args.model]
        run_calibration(args, device)
    elif args.mode == "mapping":
        _run_ppa_cases(args, "mapping", run_mapping)
    elif args.mode == "crosssim":
        _run_ppa_cases(args, "crosssim", run_crosssim)
    elif args.mode == "ppa":
        _run_ppa_cases(args, "ppa", run_ppa)
    elif args.mode == "deployment":
        run_deployment(args, device)
    else:
        raise ValueError(f"unsupported hardware mode: {args.mode}")
    print(f"[OUTPUT] {args.output}")
    print("=" * 72)


def _run_ppa_cases(args: argparse.Namespace, mode: str, runner) -> None:
    eval_config = load_hardware_eval_config(args.eval_config)
    cases = expand_hardware_eval_ppa_cases(eval_config, mode)
    (args.output / "hardware_eval_config_used.json").write_text(
        json.dumps(
            {"source": str(args.eval_config.resolve()), "mode": mode, "config": eval_config, "cases": cases},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    for case in cases:
        case_args = argparse.Namespace(**vars(args))
        for key, value in case.items():
            if key != "name":
                setattr(case_args, key, value)
        case_args.output = args.output / str(case["name"])
        case_args.output.mkdir(parents=True, exist_ok=True)
        runner(case_args)


def run_software_cli(mode: str, argv: list[str]) -> None:
    if mode == "test":
        diagnostics._run_ssim(argv=argv)
    elif mode == "ptq":
        diagnostics.run_ptq(argv)
    elif mode == "mc":
        diagnostics.run_mc(argv)
    elif mode == "weights":
        diagnostics.run_weight_diagnostics(argv)
    elif mode == "stats":
        diagnostics.run_paired_stats_evaluation(argv)
    elif mode == "figures":
        software.run_figures(argv)
    elif mode == "trace":
        hardware_trace.run(argv)
    elif mode == "tia":
        hardware.main(argv)
    elif mode == "checkpoint":
        checkpoint_main(argv)
    elif mode == "scenes":
        software._run_validation(["--mode", mode, *argv])
    else:
        raise ValueError(f"unsupported software mode: {mode}")


def load_eval(path: Path) -> dict[str, object]:
    path = path.resolve()
    with path.open("r", encoding="utf-8") as handle:
        evaluation = yaml.safe_load(handle) or {}
    if not isinstance(evaluation, dict):
        raise ValueError(f"{path} 必须是 YAML 映射")
    enabled = evaluation.get("enabled")
    if not isinstance(enabled, dict):
        raise ValueError(f"{path} 必须包含 enabled 映射")
    active_evaluations = enabled.get("evaluations")
    if not isinstance(active_evaluations, list) or not active_evaluations:
        raise ValueError(f"{path}.enabled.evaluations 必须是非空列表")
    details = evaluation.get("evaluations")
    model_config = evaluation.get("models")
    if not isinstance(model_config, dict) or not isinstance(details, dict):
        raise ValueError(f"{path} 必须包含 models 和 evaluations 映射")
    unknown_model_keys = set(model_config) - {"model_type", "profile", "overrides", "items"}
    if unknown_model_keys:
        raise ValueError(f"{path}.models 包含不支持的字段: {', '.join(sorted(unknown_model_keys))}")
    model_type = str(model_config.get("model_type", "")).strip().lower()
    if model_type not in {"fp32", "qat", "ptq"}:
        raise ValueError(f"{path}.models.model_type 必须是 fp32、qat 或 ptq")
    items = model_config.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path}.models.items 必须是非空列表")
    profile = model_config.get("profile")
    overrides = model_config.get("overrides", {}) or {}
    if not isinstance(overrides, dict):
        raise ValueError(f"{path}.models.overrides 必须是映射")
    models = []
    labels = set()
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{path}.models.items[{index}] 必须是映射")
        unknown_item_keys = set(item) - {"display_name", "checkpoint"}
        if unknown_item_keys:
            raise ValueError(f"{path}.models.items[{index}] 包含不支持的字段: {', '.join(sorted(unknown_item_keys))}")
        display_name = str(item.get("display_name") or f"model{index}").strip()
        checkpoint = str(item.get("checkpoint", "")).strip()
        if not checkpoint or display_name in labels:
            raise ValueError(f"{path}.models.items[{index}] 必须有唯一 display_name 和 checkpoint")
        models.append(
            {
                "display_name": display_name,
                "checkpoint": str(resolve_path(checkpoint, path.parent)),
                "model_type": model_type,
                "profile": None if profile is None else str(profile),
                "overrides": overrides,
            }
        )
        labels.add(display_name)
    active_evaluations = [str(name).strip() for name in active_evaluations]
    if len(active_evaluations) != len(set(active_evaluations)) or any(
        name not in details for name in active_evaluations
    ):
        raise ValueError(f"{path}.enabled.evaluations 包含未知或重复评估")
    validated_details = {}
    for raw_name, entry in details.items():
        name = str(raw_name).strip()
        if not name or not isinstance(entry, dict):
            raise ValueError(f"{path}.evaluations.{name} 必须是映射")
        if "models" in entry or "enabled" in entry:
            raise ValueError(f"{path}.evaluations.{name} 不再配置 models/enabled，请放到顶层")
        mode = str(entry.get("mode", "")).strip().lower()
        if mode not in EVAL_MODES:
            raise ValueError(f"{path}.evaluations.{name}.mode 不是有效评估模式")
        allowed = MODE_MODEL_TYPES.get(mode)
        if allowed is not None and model_type not in allowed and name in active_evaluations:
            options = ", ".join(kind for kind in ("fp32", "qat", "ptq") if kind in allowed)
            raise ValueError(
                f"评估 {name} 使用 model_type={model_type}，该模式只支持 {options}；请拆分评估计划"
            )
        validated_details[name] = {**entry, "name": name, "mode": mode}
    evaluations = [
        validated_details[name] | {"models": [model["display_name"] for model in models]}
        for name in active_evaluations
    ]
    return {
        **evaluation,
        "path": path,
        "models": models,
        "enabled": {"evaluations": active_evaluations},
        "evaluations": evaluations,
    }


def _paths(evaluation: dict[str, object]) -> dict[str, Path]:
    path = evaluation["path"]
    configured = evaluation.get("paths", {}) or {}
    if not isinstance(configured, dict):
        raise ValueError("eval.paths 必须是映射")
    defaults = {
        "config": DEFAULT_CONFIG,
        "hardware_config": DEFAULT_HARDWARE_CONFIG,
        "scene_config": DEFAULT_SCENE_CONFIG,
        "output": Path(path).parent / "results" / "eval",
    }
    paths = {
        key: resolve_path(str(configured[key]), Path(path).parent) if key in configured else Path(value).resolve()
        for key, value in defaults.items()
    }
    if "split_file" in configured:
        paths["split_file"] = resolve_path(str(configured["split_file"]), Path(path).parent)
    return paths


def _model_config(task_dir: Path, models: list[dict[str, object]]) -> Path:
    task_dir.mkdir(parents=True, exist_ok=True)
    path = task_dir / "models.yaml"
    payload = {
        "models": [
            {
                "display_name": model["display_name"],
                "checkpoint": model["checkpoint"],
                "model_type": model["model_type"],
                **({"profile": model["profile"]} if model.get("profile") else {}),
                **({"overrides": model["overrides"]} if model.get("overrides") else {}),
            }
            for model in models
        ]
    }
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def _common_args(
    evaluation: dict[str, object], paths: dict[str, Path], task_dir: Path, task: dict[str, object], model_config: Path
) -> list[str]:
    args = [
        "--models-config",
        str(model_config),
        "--output",
        str(task_dir),
        "--config",
        str(paths["config"]),
        "--eval-config",
        str(paths["hardware_config"]),
    ]
    runtime = dict(evaluation.get("runtime", {}) or {})
    runtime.update(dict(task.get("runtime", {}) or {}))
    runtime.update({key: value for key, value in task.items() if key in {"device", "micro_batch", "seed"}})
    for key, flag in (("device", "--device"), ("micro_batch", "--micro-batch"), ("seed", "--seed")):
        if key in runtime:
            args.extend((flag, str(runtime[key])))
    profile = task.get("profile")
    if profile:
        args.extend(("--profile", str(profile)))
    return args


def build_eval_argv(evaluation: dict[str, object], task: dict[str, object], task_dir: Path) -> list[str]:
    paths = _paths(evaluation)
    mode = task["mode"]
    model_config = _model_config(task_dir, evaluation["models"]) if mode != "stats" else None
    args = (
        _common_args(evaluation, paths, task_dir, task, model_config)
        if model_config is not None
        else ["--output", str(task_dir)]
    )
    if mode == "scenes":
        args.extend(("--scene-config", str(paths["scene_config"])))
        args.extend(("--scenes", ",".join(task.get("scenes", DEFAULT_SCENES))))
        args.extend(("--monte-carlo-runs", str(task.get("monte_carlo_runs", 1))))
        if paths.get("split_file"):
            args.extend(("--split-file", str(paths["split_file"])))
        if task.get("skip_baselines"):
            args.append("--skip-baselines")
        if task.get("collect_weight_stats"):
            args.append("--collect-weight-stats")
        if task.get("no_weight_curves", True):
            args.append("--no-weight-curves")
        if task.get("reference_method"):
            args.extend(("--reference-method", str(task["reference_method"])))
    elif mode in {"test", "ptq", "mc", "figures", "trace"}:
        if "split_file" not in paths:
            raise ValueError(f"评估 {task['name']} 需要 paths.split_file")
        args.extend(("--split-file", str(paths["split_file"])))
        if "test_frames" in task:
            args.extend(("--test-frames", str(task["test_frames"])))
        if mode == "mc":
            args.extend(("--runs", str(task.get("runs", 30))))
        elif mode == "trace":
            if task.get("frames") is not None:
                args.extend(("--frames", str(task["frames"])))
            args.extend(("--runs", str(task.get("runs", 1))))
        elif mode == "figures":
            args.extend(("--f-number", str(task.get("f_number", 1.5))))
    elif mode == "weights":
        args.extend(("--scene-config", str(paths["scene_config"])))
        args.extend(("--scenes", ",".join(task.get("scenes", DEFAULT_SCENES))))
        if paths.get("split_file"):
            args.extend(("--split-file", str(paths["split_file"])))
    elif mode == "stats":
        args.extend(("--input", str(resolve_path(str(task["input"]), Path(evaluation["path"]).parent))))
        args.extend(("--reference", str(task["reference"])))
        args.extend(("--models", ",".join(task["models"])))
        args.extend(("--bootstrap-samples", str(task.get("bootstrap_samples", 10000))))
    elif mode in HARDWARE_MODES:
        if mode in {"hardware", "calibrate"} and paths.get("split_file"):
            args.extend(("--split-file", str(paths["split_file"])))
        if mode == "hardware":
            for key, flag in (("references_json", "--references-json"), ("mc_runs", "--mc-runs")):
                if task.get(key) is not None:
                    value = (
                        resolve_path(str(task[key]), Path(evaluation["path"]).parent)
                        if key == "references_json"
                        else task[key]
                    )
                    args.extend((flag, str(value)))
        for key, flag in (
            ("weight_bits", "--weight-bits"),
            ("g_min", "--g-min"),
            ("g_max", "--g-max"),
            ("tile_rows", "--tile-rows"),
            ("tile_cols", "--tile-cols"),
            ("mapping_mode", "--mapping-mode"),
            ("converter_schedule", "--converter-schedule"),
            ("input_timing", "--input-timing"),
            ("differential_readout", "--differential-readout"),
            ("input_bits", "--input-bits"),
            ("adc_bits", "--adc-bits"),
            ("inter_layer", "--inter-layer"),
            ("inter_layer_bits", "--inter-layer-bits"),
            ("interpolation_bits", "--interpolation-bits"),
            ("technology_node", "--technology-node"),
            ("crosssim_runs", "--crosssim-runs"),
        ):
            if task.get(key) is not None:
                args.extend((flag, str(task[key])))
    return args


def _run_task(mode: str, argv: list[str]) -> None:
    if mode == "scenes":
        software._run_validation(["--mode", mode, *argv])
    elif mode == "test":
        diagnostics._run_ssim("test", argv)
    elif mode == "ptq":
        diagnostics.run_ptq(argv)
    elif mode == "mc":
        diagnostics.run_mc(argv)
    elif mode == "weights":
        diagnostics.run_weight_diagnostics(argv)
    elif mode == "stats":
        diagnostics.run_paired_stats_evaluation(argv)
    elif mode == "figures":
        software.run_figures(argv)
    elif mode == "trace":
        hardware_trace.run(argv)
    elif mode in HARDWARE_MODES:
        run_hardware_cli(["--mode", mode, *argv])
    else:
        raise ValueError(f"unsupported evaluation mode: {mode}")


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def run_eval(path: Path) -> None:
    evaluation = load_eval(path)
    paths = _paths(evaluation)
    output = paths["output"]
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(evaluation["path"], output / "eval_config_used.yaml")
    manifest = {
        "schema_version": 2,
        "eval": str(evaluation["path"]),
        "output": str(output),
        "models": evaluation["models"],
        "evaluations": [],
    }
    manifest_path = output / "eval_manifest.json"
    _write_manifest(manifest_path, manifest)
    for task in evaluation["evaluations"]:
        task_dir = output / str(task.get("output", task["name"]))
        record = {
            "name": task["name"],
            "mode": task["mode"],
            "models": task["models"],
            "output": str(task_dir),
        }
        manifest["evaluations"].append(record)
        print(f"[EVAL] {task['name']} | mode={task['mode']} | models={','.join(task['models'])}", flush=True)
        record["status"] = "running"
        _write_manifest(manifest_path, manifest)
        try:
            argv = build_eval_argv(evaluation, task, task_dir)
            _run_task(str(task["mode"]), argv)
        except Exception as error:
            record["status"] = "failed"
            record["error"] = f"{type(error).__name__}: {error}"
            _write_manifest(manifest_path, manifest)
            raise
        record["status"] = "completed"
        _write_manifest(manifest_path, manifest)
    print(f"[OUTPUT] {output}")



def main() -> None:
    argv = sys.argv[1:]
    mode, remaining = parse_route(argv)
    if mode == "eval":
        run_eval(Path(remaining[0]))
    elif mode in HARDWARE_MODES:
        run_hardware_cli(argv)
    else:
        run_software_cli(mode, remaining)


if __name__ == "__main__":
    main()

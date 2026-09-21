from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import yaml
import torch

import mban as training
from mban_core.config import TEST_REALIZATION_SEED_OFFSET, normalize_angle_selection
from mban_core.hardware import QATConfig

TRAIN_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = TRAIN_DIR / "config.yaml"
DEFAULT_HARDWARE_CONFIG = TRAIN_DIR / "hardware_eval.yaml"

ROOT = TRAIN_DIR.parent
RESULTS_DIR = TRAIN_DIR / "results"
DEFAULT_SCENE_CONFIG = ROOT / "config.yaml"

DEFAULT_SCENES = (
    "simulation_contrast_speckle",
    "simulation_resolution_distorsion",
    "experiments_contrast_speckle",
    "experiments_resolution_distorsion",
)

_INPUT_NORMALIZATIONS = {"none", "rms", "std"}
_CONTROL_NORMALIZATIONS = {"none", "linf"}


def resolve_path(value: str | Path, base: Path = ROOT) -> Path:
    path = Path(str(value).strip().strip('"').strip("'")).expanduser()
    return path.resolve() if path.is_absolute() else (Path(base) / path).resolve()


def resolve_hardware_config(config_path: Path, hardware_config_path: Path | None = None) -> Path | None:
    if hardware_config_path is not None:
        return Path(hardware_config_path).resolve()
    sibling = Path(config_path).resolve().with_name("hardware_eval.yaml")
    return sibling if sibling.is_file() else None


@dataclass(frozen=True)
class ModelRequest:
    display_name: str
    path: Path
    model_type: str
    profile: str | None
    overrides: tuple[str, ...]


def _parse_model_parts(value: str) -> tuple[str, Path]:
    display_name, separator, path = value.partition("=")
    if not separator or not display_name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--model 格式必须为 DISPLAY_NAME=CHECKPOINT")
    return display_name.strip(), Path(path.strip()).resolve()


def parse_checkpoint_model(value: str) -> tuple[str, Path]:
    return _parse_model_parts(value)


def parse_model_value(value: str, model_type: str) -> ModelRequest:
    display_name, path = _parse_model_parts(value)
    return ModelRequest(display_name, path, model_type, None, ())


def _yaml_override_value(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def load_model_requests(path: Path) -> list[ModelRequest]:
    path = path.resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict) or not isinstance(config.get("models"), list):
        raise ValueError(f"{path} 必须包含 models 列表")
    requests = []
    for index, item in enumerate(config["models"], start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{path} 的 models[{index}] 必须是映射")
        unknown_keys = set(item) - {"display_name", "checkpoint", "model_type", "profile", "overrides"}
        if unknown_keys:
            raise ValueError(f"{path} 的 models[{index}] 包含不支持的字段: {', '.join(sorted(unknown_keys))}")
        display_name = str(item.get("display_name", "")).strip()
        checkpoint = str(item.get("checkpoint", "")).strip()
        model_type = str(item.get("model_type", "")).strip().lower()
        if not display_name or not checkpoint or model_type not in {"fp32", "qat", "ptq"}:
            raise ValueError(f"{path} 的 models[{index}] 必须包含 display_name、checkpoint 和有效 model_type")
        profile = item.get("profile")
        if profile is not None:
            profile = str(profile)
        raw_overrides = item.get("overrides", {}) or {}
        if not isinstance(raw_overrides, dict):
            raise ValueError(f"{path} 的 models[{index}].overrides 必须是映射")
        overrides = tuple(f"{key}={_yaml_override_value(value)}" for key, value in raw_overrides.items())
        requests.append(
            ModelRequest(
                display_name,
                (path.parent / checkpoint).resolve(),
                model_type,
                profile,
                overrides,
            )
        )
    if not requests:
        raise ValueError(f"{path} 的 models 不能为空")
    display_names = [request.display_name for request in requests]
    if len(display_names) != len(set(display_names)):
        raise ValueError(f"{path} 的 display_name 必须唯一")
    return requests


def resolve_model_requests(
    values: list[str],
    models_config: Path | None = None,
    model_type: str | None = None,
) -> list[ModelRequest]:
    if values and models_config is not None:
        raise ValueError("--model 与 --models-config 只能二选一")
    if models_config is None and model_type is None:
        raise ValueError("使用 --model 时必须提供 --model-type")
    requests = (
        load_model_requests(models_config)
        if models_config is not None
        else [parse_model_value(value, model_type) for value in values]
    )
    if not requests:
        raise ValueError("必须提供 --model 或 --models-config")
    return requests


def explicit_override(overrides: list[str] | tuple[str, ...], key: str) -> str | None:
    value = None
    for item in overrides:
        name, separator, raw_value = item.partition("=")
        if separator and name.strip() == key:
            value = raw_value.strip()
    return value


def split_execution_overrides(overrides: list[str] | tuple[str, ...]) -> tuple[list[str], str | None]:
    config_overrides = []
    implementation = None
    for item in overrides:
        key, separator, value = item.partition("=")
        if separator and key.strip() == "beamforming_implementation":
            implementation = value.strip()
        else:
            config_overrides.append(item)
    if implementation is not None and implementation not in {"explicit", "factorized"}:
        raise ValueError(f"不支持的 beamforming_implementation: {implementation}")
    return config_overrides, implementation


def add_model_arguments(
    parser: argparse.ArgumentParser,
    model_types: tuple[str, ...] = ("fp32", "qat", "ptq"),
) -> None:
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--models-config", type=Path)
    parser.add_argument("--model-type", choices=model_types)


def resolve_model_arguments(args: argparse.Namespace) -> argparse.Namespace:
    args.model = resolve_model_requests(args.model, args.models_config, args.model_type)
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统一 MBAN 验证")
    add_model_arguments(parser)
    parser.add_argument("--mode", choices=("scenes",), default="scenes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--eval-config", type=Path, default=DEFAULT_HARDWARE_CONFIG)
    parser.add_argument("--profile", help="硬件压力 profile；层间流程通过 inter_layer 选择")
    parser.add_argument("--split-file", type=Path, default=None, help="PTQ 校准使用的划分文件（PTQ 场景评估必需）")
    parser.add_argument("--scene-config", type=Path, default=DEFAULT_SCENE_CONFIG)
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--micro-batch", type=int, default=8192)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--monte-carlo-runs", type=int, default=1)
    parser.add_argument("--weight-z-index", type=int)
    parser.add_argument("--weight-x-index", type=int)
    parser.add_argument("--collect-weight-stats", action="store_true")
    parser.add_argument("--no-weight-curves", action="store_true")
    parser.add_argument("--reference-method", help="为 all_scene_metrics.csv 增加相对该模型的 delta_* 列")
    args = parser.parse_args(argv)
    return resolve_model_arguments(args)

@dataclasses.dataclass
class ModelSpec:
    display_name: str
    path: Path
    model: torch.nn.Module
    runtime: dict[str, object]
    training_module: object
    model_type: str
    runtime_args: object | None = None
    config_path: Path | None = None
    profile: str | None = None
    hardware_config_path: Path | None = None


def load_specs(
    args: argparse.Namespace,
    device: torch.device,
    *,
    allowed_model_types: set[str] | None = None,
    require_deployment: bool = False,
) -> list[ModelSpec]:
    requests = args.model
    display_names = [request.display_name for request in requests]
    if len(display_names) != len(set(display_names)):
        raise ValueError("display_name 必须唯一")
    specs = [
        load_evaluation_model(
            request.display_name,
            request.path,
            request.model_type,
            device,
            config_path=args.config,
            hardware_config_path=getattr(args, "eval_config", None),
            profile=request.profile or getattr(args, "profile", None),
            overrides=request.overrides,
            require_deployment=require_deployment,
        )
        for request in requests
    ]
    model_types = {spec.model_type for spec in specs}
    if len(model_types) > 1:
        values = ", ".join(sorted(model_types))
        raise ValueError(f"同一次评估只能使用同一种 model_type，当前包含: {values}；请拆分评估计划")
    if allowed_model_types is not None:
        invalid = [spec.display_name for spec in specs if spec.model_type not in allowed_model_types]
        if invalid:
            names = ", ".join(sorted(allowed_model_types))
            raise ValueError(f"该协议只支持 {names}: {', '.join(invalid)}")
    angle_policies = {
        json.dumps(
            (spec.runtime.get("angle_selection", "center"), spec.runtime.get("angle_reduction", "sum")),
            sort_keys=True,
        )
        for spec in specs
    }
    if len(angle_policies) > 1:
        raise ValueError("同一次评估的模型必须共享 angle_selection 和 angle_reduction")
    return specs


def calibrate_models(args: argparse.Namespace, specs: list[ModelSpec], device: torch.device) -> None:
    if not any(spec.model_type == "ptq" for spec in specs):
        return
    from eval_core.hardware import calibrate_ptq_observers

    calibration_args = argparse.Namespace(
        micro_batch=args.micro_batch,
        output=args.output,
        split_file=args.split_file,
    )
    calibrate_ptq_observers(calibration_args, specs, device, split_file=args.split_file)


def prepare_test_specs(args: argparse.Namespace, device: torch.device) -> list[ModelSpec]:
    specs = load_specs(args, device)
    calibrate_models(args, specs, device)
    for spec in specs:
        hardware = getattr(spec.model, "hardware_qat", None)
        if hardware is not None and hardware.config.noise_enabled:
            test_seed = int(args.seed) + TEST_REALIZATION_SEED_OFFSET
            hardware.config = dataclasses.replace(hardware.config, noise_seed=test_seed)
            hardware.reset_noise_counter()
            hardware.begin_noise_realization()
    return specs

def resolve_execution_implementation(evaluation_config: dict[str, object], runtime_args: object | None) -> str:
    configured = getattr(runtime_args, "beamforming_implementation", None)
    implementation = configured or evaluation_config.get("beamforming_implementation", "explicit")
    if implementation not in {"explicit", "factorized"}:
        raise ValueError(f"不支持的 beamforming_implementation: {implementation}")
    return str(implementation)
def evaluation_stage(model: torch.nn.Module) -> str:
    """标记结果所属阶段；不把软件时间误写成硬件延迟。"""
    hardware = getattr(model, "hardware_qat", None)
    if hardware is None or not hardware.enabled:
        return "FP32"
    mode = str(hardware.config.mode).upper()
    return f"{mode}_nonideal" if bool(hardware.config.noise_enabled) else mode

def load_evaluation_runtime(
    model_type: str,
    config_path: Path = DEFAULT_CONFIG,
    profile: str | None = None,
    hardware_config_path: Path | None = None,
    overrides: list[str] | tuple[str, ...] = (),
) -> object:
    config_path = config_path.resolve()
    hardware_config_path = resolve_hardware_config(config_path, hardware_config_path)
    config_overrides, implementation = split_execution_overrides(overrides)
    runtime_args = training.load_config(
        config_path,
        overrides=[*config_overrides, f"mode={'software' if model_type == 'fp32' else model_type}"],
        profile_name=profile,
        hardware_config_path=hardware_config_path,
    )
    if implementation is not None:
        runtime_args.beamforming_implementation = implementation
    training.configure_runtime(runtime_args, config_path, profile, hardware_config_path)
    return runtime_args


def load_evaluation_model(
    display_name: str,
    path: Path,
    model_type: str,
    device: torch.device,
    *,
    config_path: Path = DEFAULT_CONFIG,
    profile: str | None = None,
    hardware_config_path: Path | None = None,
    overrides: list[str] | tuple[str, ...] = (),
    require_deployment: bool = False,
) -> ModelSpec:
    checkpoint = torch.load(path.resolve(), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path} 不是训练检查点")
    identity = checkpoint.get("checkpoint_identity")
    if not isinstance(identity, dict) or identity.get("training_mode") not in {"fp32", "qat"}:
        raise ValueError(f"{path} 缺少 checkpoint_identity.training_mode，请用当前代码重新训练")
    expected_training_mode = "qat" if model_type == "qat" else "fp32"
    if identity["training_mode"] != expected_training_mode:
        raise ValueError(
            f"{path} 的 checkpoint_identity.training_mode={identity['training_mode']!r}，"
            f"与 model_type={model_type} 不匹配"
        )
    hardware_config_path = resolve_hardware_config(config_path, hardware_config_path)
    runtime_args = load_evaluation_runtime(
        model_type,
        config_path,
        profile,
        hardware_config_path,
        overrides,
    )
    interpolation = explicit_override(overrides, "interpolation_bits")
    output_interpolation = explicit_override(overrides, "output_interpolation")
    return load_model(
        display_name,
        path.resolve(),
        device,
        runtime_args=runtime_args,
        config_path=config_path.resolve(),
        profile=profile,
        hardware_config_path=hardware_config_path,
        model_type=model_type,
        interpolation_bits=None if interpolation is None else int(interpolation),
        output_interpolation=output_interpolation,
        require_deployment=require_deployment,
        checkpoint=checkpoint,
    )

def activate_runtime(runtime: dict[str, object], module: object = training) -> None:
    implementation = getattr(module, "_runtime", module)
    for key, value in runtime.items():
        setattr(module.args, key, value)
        if implementation is not module:
            setattr(implementation.args, key, value)

def resolve_deployment_runtime_transforms(
    checkpoint: dict[str, object], path: Path, *, require_metadata: bool = False
) -> dict[str, object]:
    model_config = checkpoint.get("model_config", {})
    evaluation_config = checkpoint.get("evaluation_config", {})
    metadata = checkpoint.get("deployment_runtime_transforms")
    if not isinstance(model_config, dict) or not isinstance(evaluation_config, dict):
        raise ValueError(f"{path} 缺少有效 model_config/evaluation_config")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError(f"{path} 的 deployment_runtime_transforms 必须是映射")
    if require_metadata and metadata is None:
        raise ValueError(f"{path} 缺少 deployment_runtime_transforms")
    metadata = metadata or {}
    input_normalization = str(
        metadata.get(
            "input_normalization",
            evaluation_config.get("input_normalization", model_config.get("input_normalization", "rms")),
        )
    )
    control_normalization = str(
        metadata.get("control_normalization", evaluation_config.get("control_normalization", "none"))
    )
    if input_normalization not in _INPUT_NORMALIZATIONS:
        raise ValueError(f"{path} 的 input_normalization={input_normalization!r} 无法用于部署")
    if control_normalization not in _CONTROL_NORMALIZATIONS:
        raise ValueError(f"{path} 的 control_normalization={control_normalization!r} 无法用于部署")
    for field in ("input_normalization_folded", "control_normalization_folded"):
        if metadata.get(field, False) is not False:
            raise ValueError(f"{path} 的 {field} 必须为 false；该变换只能运行时执行")
    for field, value in (
        ("input_normalization", input_normalization),
        ("control_normalization", control_normalization),
    ):
        configured = evaluation_config.get(field)
        if configured is not None and str(configured) != value:
            raise ValueError(f"{path} 的部署运行时合同与 evaluation_config.{field} 不一致")
    return {
        "input_normalization": input_normalization,
        "control_normalization": control_normalization,
        "input_normalization_folded": False,
        "control_normalization_folded": False,
    }


def validate_deployment_checkpoint(checkpoint: dict[str, object], path: Path) -> None:
    if checkpoint.get("deployment_folded") is not True:
        raise ValueError(f"{path} 不是部署态 checkpoint；请先执行 checkpoint fold")
    model_config = checkpoint.get("model_config", {})
    if not isinstance(model_config, dict) or (
        model_config.get("normalization", "none") != "none"
        or model_config.get("normalization_layers", [])
        or model_config.get("centering", "none") != "none"
        or model_config.get("centering_layers", [])
        or model_config.get("weight_transform", "none") != "none"
        or model_config.get("weight_transform_layers", [])
    ):
        raise ValueError(f"{path} 的部署态必须移除动态中心化、归一化和权重变换")
    if "deployment_linear_state" in checkpoint and checkpoint.get("derived_operation") == "fold":
        raise ValueError(f"{path} 的部署态仍包含 deployment_linear_state；请重新执行 checkpoint fold")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or any(str(key).startswith("running_standardizers.") for key in state):
        raise ValueError(f"{path} 的部署态仍包含 running_standardizers 参数")
    resolve_deployment_runtime_transforms(
        checkpoint, path, require_metadata=checkpoint.get("derived_operation") == "fold"
    )
    if checkpoint.get("derived_operation") == "fold" and not isinstance(
        checkpoint.get("deployment_fold_parity"), dict
    ):
        raise ValueError(f"{path} 缺少 deployment_fold_parity 审计记录")

def load_model(
    display_name: str,
    path: Path,
    device: torch.device,
    *,
    runtime_args: object | None = None,
    config_path: Path | None = None,
    profile: str | None = None,
    hardware_config_path: Path | None = None,
    model_type: str,
    interpolation_bits: int | None = None,
    output_interpolation: str | None = None,
    require_deployment: bool = False,
    checkpoint: dict[str, object] | None = None,
) -> ModelSpec:
    checkpoint = checkpoint or torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"{path} 不是训练检查点（缺少 model_state_dict）")
    checkpoint_version = checkpoint.get("model_semantics_version")
    expected_version = training.MODEL_SEMANTICS_VERSION
    if checkpoint_version != expected_version:
        raise ValueError(
            f"{path} 的 model_semantics_version={checkpoint_version!r}，"
            f"当前要求 {expected_version!r}；请用当前代码重新训练"
        )
    if require_deployment:
        validate_deployment_checkpoint(checkpoint, path)
    state = checkpoint["model_state_dict"]
    if not isinstance(state, dict) or "fc1.weight" not in state:
        raise ValueError(f"{path} 不是可识别的训练检查点")
    model_config = dict(checkpoint["model_config"])
    evaluation_config = dict(checkpoint["evaluation_config"])
    hardware_state = checkpoint.get("hardware_qat_state")
    qat_checkpoint_config = None
    if model_type == "qat":
        if not isinstance(hardware_state, dict) or not isinstance(hardware_state.get("config"), dict):
            raise ValueError(f"{path} 不是可复现的 QAT checkpoint（缺少 hardware_qat_state.config）")
        qat_checkpoint_config = QATConfig(**hardware_state["config"])
        if not qat_checkpoint_config.enabled or qat_checkpoint_config.mode != "qat":
            raise ValueError(f"{path} 的 hardware_qat_state.config 不是 QAT 硬件合同")
    stored_bias_implementation = str(model_config.get("bias_implementation", "ordinary"))
    stored_bias_layers = model_config.get("bias_layers")
    has_fc1_bias = stored_bias_layers is None or (
        isinstance(stored_bias_layers, str) and stored_bias_layers.strip().lower() == "all"
    ) or "fc1" in stored_bias_layers
    array_fc1_bias = stored_bias_implementation == "array" and has_fc1_bias
    hidden_width = int(model_config["hidden_width"])
    hidden_layers = int(model_config["hidden_layers"])
    fc1_weight = state["fc1.weight"]
    if not isinstance(fc1_weight, torch.Tensor) or fc1_weight.ndim != 2:
        raise ValueError(f"{path} 的 fc1.weight 必须是偶数输入维度的二维张量")
    fc1_input_width = fc1_weight.shape[1] - int(array_fc1_bias)
    if fc1_input_width % 2:
        raise ValueError(f"{path} 的 fc1.weight 必须是偶数输入维度的二维张量")
    input_channels = fc1_input_width // 2
    if "network_channels" not in model_config:
        raise ValueError(f"{path} 缺少 model_config.network_channels")
    network_channels = int(model_config["network_channels"])
    if network_channels != input_channels:
        raise ValueError(
            f"{path} 的 network_channels={network_channels} 与 fc1 输入维度推导值 {input_channels} 不一致"
        )
    expected_network_channels = int(model_config.get("expected_network_channels", network_channels))
    if expected_network_channels != network_channels:
        raise ValueError(
            f"{path} 的 expected_network_channels={expected_network_channels} 与 "
            f"network_channels={network_channels} 不一致"
        )
    physical_channels = int(model_config.get("physical_channels", network_channels))
    if physical_channels < network_channels:
        raise ValueError(
            f"{path} 的 physical_channels={physical_channels} 小于 network_channels={network_channels}"
        )
    output_weights = str(evaluation_config["output_weights"])
    output_layer = f"fc{hidden_layers + 1}"
    output_width = state[f"{output_layer}.weight"].shape[0]
    output_controls = int(
        model_config.get("output_controls", output_width // (2 if output_weights == "complex" else 1))
    )
    expected_output_width = output_controls * (2 if output_weights == "complex" else 1)
    if output_width != expected_output_width:
        raise ValueError(
            f"{path} 的 {output_layer} 输出维度={output_width} 与 output_controls={output_controls}"
            f"、output_weights={output_weights} 不一致"
        )
    activation = model_config.get("activation", {})
    branch_mode = str(model_config["branch_mode"])
    hardware_enabled = model_type in {"qat", "ptq"}
    module = training
    runtime_config = runtime_args or training.args
    if hardware_enabled and not bool(getattr(runtime_config, "qat_enabled", False)):
        raise ValueError("硬件 checkpoint 必须使用 config.run.mode=ptq 或 qat；当前配置是 software")
    hardware_config = (
        qat_checkpoint_config
        if qat_checkpoint_config is not None
        else module.QATConfig.from_namespace(runtime_config)
        if hardware_enabled
        else None
    )
    runtime = {
        "output_weights": output_weights,
        "output_domain": str(evaluation_config["output_domain"]),
        "unity_constraint": str(evaluation_config["unity_constraint"]),
        "unity_scope": str(evaluation_config.get("unity_scope", "global")),
        "angle_selection": normalize_angle_selection(evaluation_config.get("angle_selection", "center")),
        "angle_reduction": str(evaluation_config.get("angle_reduction", "sum")),
        "control_normalization": str(
            evaluation_config.get(
                "control_normalization",
                getattr(runtime_config, "control_normalization", "none"),
            )
        ),
        "control_adc_range": str(evaluation_config["control_adc_range"]),
        "control_adc_full_scale": float(evaluation_config["control_adc_full_scale"]),
        "beamforming_implementation": resolve_execution_implementation(evaluation_config, runtime_config),
        "dynamic_aperture": bool(evaluation_config["dynamic_aperture"]),
        "f_number": float(evaluation_config["f_number"]),
        "input_normalization": evaluation_config.get(
            "input_normalization",
            model_config.get("input_normalization", getattr(runtime_config, "input_normalization", "rms")),
        ),
        "projection_controls": int(evaluation_config.get("projection_controls", 0)),
        "output_interpolation": str(
            output_interpolation
            if output_interpolation is not None
            else evaluation_config.get("output_interpolation", getattr(runtime_config, "output_interpolation", "linear"))
        ),
        "interpolation_bits": int(
            interpolation_bits
            if interpolation_bits is not None
            else evaluation_config.get("interpolation_bits", getattr(runtime_config, "interpolation_bits", 32))
        ),
        "use_tgc": bool(evaluation_config.get("use_tgc", getattr(runtime_config, "use_tgc", False))),
        "tgc_alpha": float(getattr(runtime_config, "tgc_alpha", 0.5)),
    }
    if runtime["output_domain"] not in {"nonnegative", "signed"}:
        raise ValueError(f"{path} 使用不支持的 output_domain: {runtime['output_domain']}")
    if runtime["unity_constraint"] not in {"hard", "free"}:
        raise ValueError(f"{path} 使用不支持的 unity_constraint: {runtime['unity_constraint']}")
    if runtime["unity_scope"] not in {"global", "per_angle"}:
        raise ValueError(f"{path} 使用不支持的 unity_scope: {runtime['unity_scope']}")
    if not (
        runtime["angle_selection"] in {"center", "all"}
        if isinstance(runtime["angle_selection"], str)
        else isinstance(runtime["angle_selection"], (int, list))
    ):
        raise ValueError(f"{path} 使用不支持的 angle_selection: {runtime['angle_selection']}")
    if runtime["angle_reduction"] not in {"sum", "mean"}:
        raise ValueError(f"{path} 使用不支持的 angle_reduction: {runtime['angle_reduction']}")
    activate_runtime(runtime, module)
    model = module.MBAN(
        num_channels=network_channels,
        dropout=float(model_config.get("dropout", 0.0)),
        hidden_width=hidden_width,
        activation=activation,
        branch_mode=branch_mode,
        hidden_layers=hidden_layers,
        output_controls=output_controls,
        input_tile_features=int(model_config.get("input_tile_features", 0)),
        centering=str(model_config.get("centering", "none")),
        centering_layers=model_config.get("centering_layers", []),
        normalization=str(model_config.get("normalization", "none")),
        normalization_layers=model_config.get("normalization_layers", []),
        bias_layers=model_config.get("bias_layers"),
        bias_implementation=stored_bias_implementation,
        weight_transform=str(model_config.get("weight_transform", "none")),
        weight_transform_layers=model_config.get("weight_transform_layers", []),
        weight_transform_epsilon=float(model_config.get("weight_transform_epsilon", 1.0e-5)),
        running_stat_momentum=float(model_config.get("running_stat_momentum", 0.1)),
        running_stat_epsilon=float(model_config.get("running_stat_epsilon", 1.0e-5)),
        batch_renorm_rmax=float(model_config.get("batch_renorm_rmax", 3.0)),
        batch_renorm_dmax=float(model_config.get("batch_renorm_dmax", 5.0)),
        hardware_config=hardware_config,
        hardware_enabled=hardware_enabled,
    ).to(device)
    model.load_state_dict(state)
    if hardware_enabled:
        if hardware_state:
            model.hardware_qat.load_state_dict(hardware_state)
        model.hardware_qat.disable_observer()
    model.eval()
    return ModelSpec(
        display_name,
        path,
        model,
        runtime,
        module,
        model_type,
        runtime_args or training.args,
        config_path or getattr(training, "CONFIG_PATH", None),
        profile or getattr(training, "NONIDEAL_PROFILE_NAME", None),
        hardware_config_path or getattr(training, "HARDWARE_CONFIG_PATH", None),
    )

def activate_model_runtime(spec: ModelSpec, module: object = training) -> None:
    if spec.runtime_args is not None and hasattr(module, "configure_runtime"):
        module.configure_runtime(
            spec.runtime_args,
            spec.config_path,
            spec.profile,
            spec.hardware_config_path,
        )
    activate_runtime(spec.runtime, module)

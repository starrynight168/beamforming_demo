from __future__ import annotations

import itertools
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from .hardware import (
    BIAS_IMPLEMENTATIONS,
    HARDWARE_STRESS_PROFILES,
    INTER_LAYER_MODES,
    NONIDEAL_PROFILE_FIELDS,
    QATConfig,
    resolve_nonideal_profile_raw,
)

MODEL_SEMANTICS_VERSION = 40
TEST_REALIZATION_SEED_OFFSET = 2_000_000
VISUALIZATION_REALIZATION_SEED_OFFSET = 3_000_000

_LEVEL_COLORS = {
    "INFO": "\033[1;37m",  # bold white
    "DEBUG": "\033[1;90m",  # bold bright black/gray
    "SUCCESS": "\033[1;92m",  # bold bright green
    "WARNING": "\033[1;93;43m",  # bold bright yellow on yellow-black bg
    "ERROR": "\033[1;91;41m",  # bold bright red on red bg
    "CRITICAL": "\033[1;97;41m",
}
_RESET = "\033[0m"
_TIME_COLOR = "\033[37m"
_LEVEL_COLORED_TAG = {level: f"{color}{level:<7}{_RESET}" for level, color in _LEVEL_COLORS.items()}


def _use_color() -> bool:
    if os.environ.get("MBAN_NO_COLOR", "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    if os.environ.get("NO_COLOR", "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    if os.environ.get("MBAN_FORCE_COLOR", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    return sys.stdout.isatty()


CONFIG_FIELDS = {
    "epochs",
    "learning_rate",
    "scheduler",
    "scheduler_step_epochs",
    "scheduler_gamma",
    "scheduler_milestones",
    "scheduler_patience",
    "cosine_restart_epochs",
    "onecycle_pct_start",
    "gradient_clip_norm",
    "qat_clip_outside_grad",
    "seed",
    "optimization_batch_pixels",
    "train_pixels_per_image",
    "validation_pixels_per_image",
    "images_per_batch",
    "loader_workers",
    "resume",
    "initial_checkpoint",
    "target_algorithm",
    "loss",
    "control_normalization",
    "control_adc_range",
    "control_adc_full_scale",
    "projection_controls",
    "output_domain",
    "unity_constraint",
    "unity_scope",
    "angle_selection",
    "angle_reduction",
    "distillation_weight",
    "distillation_checkpoint",
    "output_weights",
    "hidden_width",
    "hidden_layers",
    "dropout",
    "activation",
    "branch_mode",
    "centering",
    "centering_layers",
    "normalization",
    "normalization_layers",
    "bias_layers",
    "weight_transform",
    "weight_transform_layers",
    "weight_transform_epsilon",
    "running_stat_momentum",
    "running_stat_epsilon",
    "batch_renorm_rmax",
    "batch_renorm_dmax",
    "input_normalization",
    "input_tile_features",
    "output_controls",
    "expected_network_channels",
    "output_interpolation",
    "interpolation_bits",
    "dynamic_aperture",
    "f_number",
    "mode",
    "qat_schedule",
    "h5_files",
    "train_ratio",
    "validation_ratio",
    "test_ratio",
    "split_file",
    "make_new_split",
    "use_train_as_validation",
    "output_directory",
    "use_tgc",
    "save_images_every_epochs",
    "visualization_acquisition",
    "qat_enabled",
    "qat_mode",
    "inter_layer",
    "inter_layer_bits",
    "weight_bits",
    "input_bits",
    "control_bits",
    "bias_bits",
    "bias_implementation",
    "bias_programming_gain_error",
    "bias_programming_offset",
    "bias_noise_std",
    "bias_channel_gain_mismatch_std",
    "bias_noise_reference",
    "g_min",
    "g_max",
    "static_mismatch_std",
    "per_output_channel",
    "observer_enabled",
    "fixed_observer_calibration",
    "observer_calibration_epochs",
    "ptq_calibration_samples",
    "public_gain_max",
    "dac_gain_error",
    "dac_offset",
    "dac_noise_std",
    "dac_noise_mode",
    "dac_nonlinearity",
    "dac_nonlinearity_beta",
    "tia_gain_error",
    "tia_channel_gain_mismatch_std",
    "tia_offset",
    "tia_channel_offset_mismatch_std",
    "tia_noise_std",
    "tia_noise_mode",
    "tia_noise_reference",
    "tia_saturation",
    "tia_saturation_mode",
    "tia_saturation_reference",
    "activation_threshold",
    "activation_threshold_mode",
    "activation_threshold_mismatch",
    "activation_threshold_mismatch_mode",
    "activation_threshold_reference",
    "activation_threshold_references",
    "activation_gain_mismatch",
    "buffer_gain_error",
    "buffer_noise_std",
    "write_error_std",
    "write_noise_mode",
    "read_noise_std",
    "read_noise_mode",
    "control_reference_gain_error",
    "control_reference_offset",
    "control_reference_noise_std",
    "control_reference_hold_error",
    "ir_drop_enabled",
    "ir_drop_tile_rows",
    "ir_drop_tile_cols",
    "ir_drop_wire_resistance",
    "ir_drop_g_max",
    "ir_drop_v_read",
    "adc_gain_error",
    "adc_channel_gain_mismatch_std",
    "adc_offset",
    "adc_channel_offset_mismatch_std",
    "adc_noise_std",
    "adc_noise_mode",
    "adc_nonlinearity",
    "adc_nonlinearity_beta",
    "drift_std",
    "stuck_at_prob",
    "stuck_at_high_fraction",
    "noise_enabled",
    "noise_seed",
}

QAT_STAGE_OVERRIDE_FIELDS = {
    "noise_seed",
}


class RuntimeState(SimpleNamespace):
    """共享训练运行状态；配置解析与运行期状态保持在同一模块。"""

    def __init__(self) -> None:
        super().__init__(
            args=None,
            CONFIG_PATH=None,
            HARDWARE_CONFIG_PATH=None,
            NONIDEAL_PROFILE_NAME="ideal",
            MODEL_SEMANTICS_VERSION=MODEL_SEMANTICS_VERSION,
            SEMANTIC_CONFIG={},
            SEMANTIC_HASH="",
            LOG_PATH="",
            LATEST_CHECKPOINT_PATH="",
            BEST_VAL_REG_WEIGHT_PATH="",
            BEST_VAL_WEIGHT_PATH="",
            BEST_TRAIN_WEIGHT_PATH="",
            NUM_WORKERS=0,
            NUM_CHANNELS=-1,
            IMG_H=None,
            IMG_W=None,
            NUM_PIXELS=None,
            c_global=None,
            fc_global=None,
            fs_global=None,
            DR=60.0,
        )

    def set_state(
        self,
        runtime_args,
        config_path: Path,
        hardware_config_path: Path | None,
        profile_name: str,
        output_directory: str | os.PathLike[str],
        num_workers: int,
        semantic_config: dict[str, object],
        semantic_hash: str,
    ) -> None:
        self.args = runtime_args
        self.CONFIG_PATH = Path(config_path).resolve()
        self.HARDWARE_CONFIG_PATH = Path(hardware_config_path).resolve() if hardware_config_path else None
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        self.NONIDEAL_PROFILE_NAME = profile_name
        self.LOG_PATH = str(output / "training_log.txt")
        self.LATEST_CHECKPOINT_PATH = str(output / "latest.pth")
        self.BEST_VAL_REG_WEIGHT_PATH = str(output / "best_val_reg.pth")
        self.BEST_VAL_WEIGHT_PATH = str(output / "best_val.pth")
        self.BEST_TRAIN_WEIGHT_PATH = str(output / "best_train.pth")
        self.NUM_WORKERS = int(num_workers)
        self.SEMANTIC_CONFIG = semantic_config
        self.SEMANTIC_HASH = semantic_hash

    def write_log(self, message: str, level: str = "INFO") -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        color = _use_color()
        if color:
            tag = _LEVEL_COLORED_TAG.get(level, _LEVEL_COLORED_TAG["INFO"])
            colored = f"{_TIME_COLOR}[{ts}]{_RESET} {tag} {_LEVEL_COLORS.get(level, '')}{message}{_RESET}"
        else:
            colored = f"[{ts}] [{level}] {message}"
        try:
            print(colored)
        except UnicodeEncodeError:
            encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
            print(colored.encode(encoding, errors="replace").decode(encoding))
        if self.LOG_PATH:
            with open(self.LOG_PATH, "a", encoding="utf-8") as handle:
                handle.write(f"[{ts}] [{level}] {message}\n")


runtime = RuntimeState()


def _coerce_override(key: str, raw: str, current: object) -> object:
    """Convert a ``--set key=value`` string to the config field's type."""
    if key == "angle_selection":
        parsed = yaml.safe_load(raw)
        return parsed if isinstance(parsed, (list, int)) and not isinstance(parsed, bool) else raw
    if isinstance(current, bool):
        lowered = raw.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"--set {key}: 无法解析为布尔值: {raw!r}")
    if isinstance(current, int) and not isinstance(current, bool):
        if "." in raw or "e" in raw.lower():
            return float(raw)
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        if key == "bias_layers" and raw.strip().lower() == "all":
            return "all"
        parsed = yaml.safe_load(raw)
        if not isinstance(parsed, list):
            raise ValueError(f"--set {key}: 期望 YAML 列表: {raw!r}")
        return parsed
    if isinstance(current, dict):
        parsed = yaml.safe_load(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"--set {key}: 期望 YAML 映射: {raw!r}")
        return parsed
    if current is None:
        if raw.strip().lower() in {"none", "null", ""}:
            return None
        return raw
    return raw


def normalize_angle_selection(value: object) -> str | int | list[int]:
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"center", "single"}:
            return "center"
        if token == "all":
            return "all"
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError("angle_selection 必须是 center、all、角度数或索引列表") from exc
    if isinstance(value, bool):
        raise ValueError("angle_selection 不能是布尔值")
    if isinstance(value, (int, np.integer)):
        if int(value) < 1:
            raise ValueError("angle_selection 的角度数必须大于 0")
        return int(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        result = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
                raise ValueError("angle_selection 索引列表必须只包含整数")
            result.append(int(item))
        if not result:
            raise ValueError("angle_selection 索引列表不能为空")
        if len(set(result)) != len(result) or any(item < 0 for item in result):
            raise ValueError("angle_selection 索引列表必须是非负且不重复的整数")
        return result
    raise ValueError("angle_selection 必须是 center、all、角度数或索引列表")


def _override_value(overrides: list[str] | None, key: str) -> str | None:
    value = None
    for item in overrides or []:
        name, separator, raw = item.partition("=")
        if separator and name.strip() == key:
            value = raw.strip()
    return value


def _validate_loss_config(loss: object, output_domain: str, unity_constraint: str) -> None:
    if not isinstance(loss, dict):
        raise ValueError("loss 必须是映射")
    if set(loss) != {"error_function", "charbonnier_epsilon", "envelope_epsilon", "terms"}:
        raise ValueError("loss 必须包含 error_function、charbonnier_epsilon、envelope_epsilon、terms")
    if loss["error_function"] not in {"mse", "l1", "charbonnier"}:
        raise ValueError("loss.error_function 必须是 mse、l1 或 charbonnier")
    for name in ("charbonnier_epsilon", "envelope_epsilon"):
        value = float(loss[name])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"loss.{name} 必须是有限正数")
        loss[name] = value
    terms = loss["terms"]
    if not isinstance(terms, dict):
        raise ValueError("loss.terms 必须是映射")
    term_names = {"iq", "envelope", "unity", "d1", "d2", "lrange"}
    if set(terms) != term_names:
        raise ValueError("loss.terms 必须包含 iq、envelope、unity、d1、d2、lrange")
    for name in ("iq", "envelope", "unity"):
        term = terms[name]
        if not isinstance(term, dict) or set(term) != {"weight"}:
            raise ValueError(f"loss.terms.{name} 必须是 {{weight}} 映射")
        term["weight"] = float(term["weight"])
    for name in ("d1", "d2"):
        term = terms[name]
        if not isinstance(term, dict) or set(term) != {"domain", "weight"}:
            raise ValueError(f"loss.terms.{name} 必须是 {{domain, weight}} 映射")
        if term["domain"] not in {"shape", "absolute"}:
            raise ValueError(f"loss.terms.{name}.domain 必须是 shape 或 absolute")
        term["weight"] = float(term["weight"])
    lrange = terms["lrange"]
    if not isinstance(lrange, dict) or set(lrange) != {"limit", "weight"}:
        raise ValueError("loss.terms.lrange 必须是 {limit, weight} 映射")
    lrange["limit"] = float(lrange["limit"])
    lrange["weight"] = float(lrange["weight"])
    weights = {name: float(terms[name]["weight"]) for name in term_names}
    if any(not np.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("loss terms 的 weight 必须是有限非负数")
    if weights["iq"] + weights["envelope"] <= 0:
        raise ValueError("loss.terms.iq 与 loss.terms.envelope 不能同时为0")
    if unity_constraint == "hard" and weights["unity"] != 0.0:
        raise ValueError("unity_constraint=hard 时 loss.terms.unity.weight 必须为0")
    if float(lrange["limit"]) < 0 or not np.isfinite(float(lrange["limit"])):
        raise ValueError("loss.terms.lrange.limit 必须是有限非负数")
    if (
        output_domain == "nonnegative"
        and unity_constraint == "hard"
        and any(weights[name] > 0 and terms[name]["domain"] == "absolute" for name in ("d1", "d2"))
    ):
        raise ValueError("nonnegative+hard 下 d1/d2 的 absolute 域无意义，请使用 shape 域")


def _validate_layer_list(value: object, field_name: str, valid_layers: set[str]) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} 必须是列表")
    if any(not isinstance(layer, str) or not layer.strip() for layer in value):
        raise ValueError(f"{field_name} 必须只包含非空字符串层名")
    if len(set(value)) != len(value):
        raise ValueError(f"{field_name} 不能包含重复层名")
    unknown = set(value) - valid_layers
    if unknown:
        raise ValueError(f"{field_name} 只能从 {sorted(valid_layers)} 选择")
    return sorted(value, key=lambda layer: int(layer[2:]))


def _read_hardware_sections(
    config_path: Path,
    hardware_config_path: Path | None = None,
) -> tuple[dict[str, object], dict[str, object], dict[str, object] | None]:
    if hardware_config_path is not None:
        candidates = [Path(hardware_config_path).resolve()]
        if not candidates[0].is_file():
            raise FileNotFoundError(f"硬件配置不存在: {candidates[0]}")
    else:
        candidates = [
            config_path.with_name("hardware_eval.yaml"),
        ]
    source = next((candidate for candidate in candidates if candidate.is_file()), None)
    if source is None:
        return {}, {}, None
    validated = load_hardware_eval_config(source)
    hardware = validated.get("hardware", {})
    nonidealities = validated.get("nonidealities", {})
    qat_schedule = validated.get("qat_schedule")
    return dict(hardware), dict(nonidealities), dict(qat_schedule) if qat_schedule is not None else None


def _bind_qat_schedule_profile(qat_schedule: dict | None, profile_name: str) -> dict:
    schedule = deepcopy(qat_schedule or {"enabled": False, "stages": []})
    stages = schedule.get("stages", []) if isinstance(schedule, dict) else []
    if not schedule.get("enabled", False) or not isinstance(stages, list):
        return schedule
    for stage in stages:
        if isinstance(stage, dict) and str(stage.get("profile", "")) != "ideal":
            stage["profile"] = profile_name
    return schedule


def load_config(
    config_path: Path,
    overrides: list[str] | None = None,
    raw_config: dict | None = None,
    profile_name: str | None = None,
    hardware_config_path: Path | None = None,
) -> SimpleNamespace:
    config_path = Path(config_path).resolve()
    hardware_config_path = Path(hardware_config_path).resolve() if hardware_config_path is not None else None
    if raw_config is None:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    else:
        raw = raw_config
    if not isinstance(raw, dict):
        raise ValueError("训练配置必须是 YAML 映射")
    raw = dict(raw)
    embedded_hardware = sorted(set(raw) & {"hardware", "hardware_execution", "nonidealities", "qat_schedule"})
    if embedded_hardware:
        raise ValueError(f"训练配置不得包含硬件配置段 {embedded_hardware}；请将硬件配置放入 hardware_eval.yaml")
    companion_hardware, companion_nonidealities, companion_qat_schedule = _read_hardware_sections(
        config_path, hardware_config_path
    )
    run_raw = raw.pop("run", {}) or {}
    hardware_raw = companion_hardware
    nonideal_raw = companion_nonidealities
    qat_schedule = companion_qat_schedule
    if not isinstance(run_raw, dict) or not isinstance(hardware_raw, dict) or not isinstance(nonideal_raw, dict):
        raise ValueError("run、hardware、nonidealities 必须是 YAML 映射")
    if set(run_raw) - {"mode"}:
        raise ValueError(f"run 包含未知字段: {sorted(set(run_raw) - {'mode'})}")
    if set(nonideal_raw) - {"active_profile", "parameters", "profiles"}:
        raise ValueError(
            f"nonidealities 包含未知字段: {sorted(set(nonideal_raw) - {'active_profile', 'parameters', 'profiles'})}"
        )
    if not isinstance(nonideal_raw.get("parameters", {}) or {}, dict):
        raise ValueError("nonidealities.parameters 必须是映射")
    if qat_schedule is not None and not isinstance(qat_schedule, dict):
        raise ValueError("qat_schedule 必须是映射")
    values: dict[str, object] = {}
    if "mode" in run_raw:
        values["mode"] = run_raw["mode"]
    values.update(hardware_raw)
    values.update(nonideal_raw.get("parameters", {}) or {})
    values["qat_schedule"] = qat_schedule or {"enabled": False, "stages": []}
    profile_config = {
        "version": 1,
        "active_profile": nonideal_raw.get("active_profile", "ideal"),
        "hardware": nonideal_raw.get("parameters", {}) or {},
        "profiles": nonideal_raw.get("profiles", {}) or {},
    }
    if not isinstance(profile_config["profiles"], dict):
        raise ValueError("nonidealities.profiles 必须是映射")
    selected_profile = str(profile_name or profile_config["active_profile"] or "ideal")
    profile_inter_layer = str(
        _override_value(overrides, "inter_layer")
        or hardware_raw.get("inter_layer", "none")
    )
    profile_bias_implementation = str(
        _override_value(overrides, "bias_implementation")
        or hardware_raw.get("bias_implementation", "ordinary")
    )
    values["qat_schedule"] = (
        _bind_qat_schedule_profile(qat_schedule, selected_profile)
        if profile_name is not None
        else deepcopy(qat_schedule or {"enabled": False, "stages": []})
    )
    profile_overrides = resolve_nonideal_profile_raw(
        profile_config,
        selected_profile,
        profile_inter_layer,
        profile_bias_implementation,
    )
    # 这些段的值本身就是嵌套结构，作为整体保留，不扁平化。
    NESTED_SECTIONS = {"activation_functions", "loss"}
    for section, section_values in raw.items():
        if not isinstance(section_values, dict):
            raise ValueError(f"配置段 {section!r} 必须是映射")
        if section in NESTED_SECTIONS:
            if section in values:
                raise ValueError(f"配置键重复: {section}")
            values[section] = section_values
            continue
        duplicate = values.keys() & section_values.keys()
        if duplicate:
            raise ValueError(f"配置键重复: {sorted(duplicate)}")
        values.update(section_values)
    has_functions = "activation_functions" in values
    has_layers = "layer_activations" in values
    if has_functions != has_layers:
        raise ValueError("activation_functions 与 backbone.layer_activations 必须同时配置")
    if has_functions:
        values["activation"] = {
            "functions": values.pop("activation_functions"),
            "layers": values.pop("layer_activations"),
        }
    defaults = {
        "output_domain": "signed",
        "unity_constraint": "hard",
        "unity_scope": "global",
        "angle_selection": "center",
        "angle_reduction": "sum",
        "train_pixels_per_image": 0,
        "validation_pixels_per_image": 8192,
        "onecycle_pct_start": 0.1,
        "hidden_layers": 3,
        "branch_mode": "single",
        "centering": "none",
        "centering_layers": [],
        "activation": {
            "functions": {"relu": {}, "pwl": {"knee": 1.0, "tail_slope": 0.2}, "tanh": {"beta": 1.0}},
            "layers": {},
        },
        "normalization": "none",
        "normalization_layers": [],
        "bias_layers": "all",
        "weight_transform": "none",
        "weight_transform_layers": [],
        "weight_transform_epsilon": 1.0e-5,
        "qat_clip_outside_grad": 0.1,
        "running_stat_momentum": 0.1,
        "running_stat_epsilon": 1.0e-5,
        "batch_renorm_rmax": 3.0,
        "batch_renorm_dmax": 5.0,
        "loss": {
            "error_function": "mse",
            "charbonnier_epsilon": 1.0e-3,
            "envelope_epsilon": 1.0e-6,
            "terms": {
                "iq": {"weight": 1.0},
                "envelope": {"weight": 0.0},
                "unity": {"weight": 0.0},
                "d1": {"domain": "shape", "weight": 0.0},
                "d2": {"domain": "shape", "weight": 0.0},
                "lrange": {"limit": 8.0, "weight": 0.0},
            },
        },
        "control_normalization": "none",
        "control_adc_range": "full_scale",
        "control_adc_full_scale": 1.0,
        "projection_controls": 0,
        "input_tile_features": 0,
        "input_normalization": "rms",
        "output_controls": 0,
        "expected_network_channels": None,
        "output_interpolation": "linear",
        "interpolation_bits": 32,
        "distillation_weight": 0.0,
        "distillation_checkpoint": None,
        "initial_checkpoint": None,
        "mode": "software",
        "qat_enabled": False,
        "qat_mode": "qat",
        "inter_layer": "none",
        "inter_layer_bits": 4,
        "weight_bits": 4,
        "input_bits": 4,
        "control_bits": 4,
        "bias_bits": 4,
        "bias_implementation": "ordinary",
        "bias_programming_gain_error": 0.0,
        "bias_programming_offset": 0.0,
        "bias_noise_std": 0.0,
        "bias_channel_gain_mismatch_std": 0.0,
        "bias_noise_reference": 1.0,
        "g_min": 0.2428115,
        "g_max": 1.0,
        "per_output_channel": False,
        "observer_enabled": True,
        "fixed_observer_calibration": True,
        "observer_calibration_epochs": 3,
        "ptq_calibration_samples": 4,
        "public_gain_max": 16.0,
        "dac_gain_error": 0.0,
        "dac_offset": 0.0,
        "dac_noise_std": 0.0,
        "dac_noise_mode": "full_scale",
        "dac_nonlinearity": 0.0,
        "dac_nonlinearity_beta": 3.0,
        "tia_gain_error": 0.0,
        "tia_channel_gain_mismatch_std": 0.0,
        "tia_offset": 0.0,
        "tia_channel_offset_mismatch_std": 0.0,
        "tia_noise_std": 0.0,
        "tia_noise_mode": "full_scale",
        "tia_noise_reference": 1.0,
        "tia_saturation": 0.0,
        "tia_saturation_mode": "full_scale",
        "tia_saturation_reference": 1.0,
        "activation_threshold": 0.0,
        "activation_threshold_mode": "full_scale",
        "activation_threshold_mismatch": 0.0,
        "activation_threshold_mismatch_mode": "full_scale",
        "activation_threshold_reference": 1.0,
        "activation_threshold_references": {},
        "activation_gain_mismatch": 0.0,
        "buffer_gain_error": 0.0,
        "buffer_noise_std": 0.0,
        "write_error_std": 0.0,
        "write_noise_mode": "full_scale",
        "read_noise_std": 0.0,
        "read_noise_mode": "common_mode_relative",
        "control_reference_gain_error": 0.0,
        "control_reference_offset": 0.0,
        "control_reference_noise_std": 0.0,
        "control_reference_hold_error": 0.0,
        "ir_drop_enabled": False,
        "ir_drop_tile_rows": 64,
        "ir_drop_tile_cols": 64,
        "ir_drop_wire_resistance": 0.35,
        "ir_drop_g_max": 3.13e-5,
        "ir_drop_v_read": 0.3,
        "adc_gain_error": 0.0,
        "adc_channel_gain_mismatch_std": 0.0,
        "adc_offset": 0.0,
        "adc_channel_offset_mismatch_std": 0.0,
        "adc_noise_std": 0.0,
        "adc_noise_mode": "full_scale",
        "adc_nonlinearity": 0.0,
        "adc_nonlinearity_beta": 3.0,
        "drift_std": 0.0,
        "stuck_at_prob": 0.0,
        "stuck_at_high_fraction": 0.5,
        "noise_enabled": False,
        "noise_seed": 0,
        "static_mismatch_std": {},
        "qat_schedule": {"enabled": False, "stages": []},
    }
    for key, value in defaults.items():
        values.setdefault(key, value)
    nonideal_base_values = {
        key: (dict(values[key]) if isinstance(values[key], dict) else values[key]) for key in NONIDEAL_PROFILE_FIELDS
    }
    for item in profile_overrides:
        key, separator, raw_value = item.partition("=")
        if not separator or key not in CONFIG_FIELDS:
            raise ValueError(f"非理想 profile 产生了未知字段: {key!r}")
        values[key] = _coerce_override(key, raw_value.strip(), values[key])
    for item in overrides or []:
        key, separator, raw = item.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"--set 必须为 KEY=VALUE 格式: {item!r}")
        key = key.strip()
        if key not in CONFIG_FIELDS:
            raise ValueError(f"--set 未知配置字段: {key}")
        values[key] = _coerce_override(key, raw.strip(), values[key])
    missing, unknown = CONFIG_FIELDS - values.keys(), values.keys() - CONFIG_FIELDS
    if missing or unknown:
        problems = ([f"缺少 {sorted(missing)}"] if missing else []) + ([f"未知 {sorted(unknown)}"] if unknown else [])
        raise ValueError("配置字段错误: " + "; ".join(problems))
    if values["mode"] not in {"software", "ptq", "qat"}:
        raise ValueError("run.mode 必须是 software、ptq 或 qat")
    values["qat_enabled"] = values["mode"] != "software"
    values["qat_mode"] = values["mode"] if values["mode"] in {"ptq", "qat"} else "qat"
    schedule = values["qat_schedule"]
    if not isinstance(schedule, dict) or not isinstance(schedule.get("enabled", False), bool):
        raise ValueError("qat_schedule 必须包含布尔 enabled")
    stages = schedule.get("stages", [])
    if not isinstance(stages, list):
        raise ValueError("qat_schedule.stages 必须是列表")
    for stage in stages:
        if not isinstance(stage, dict) or not stage.get("name") or not stage.get("profile"):
            raise ValueError("每个 QAT stage 必须包含 name、epochs、profile")
        if str(stage["profile"]) in INTER_LAYER_MODES:
            raise ValueError("QAT stage.profile 只能表示硬件压力 profile；层间流程请使用 inter_layer")
        if int(stage.get("epochs", 0)) <= 0:
            raise ValueError("QAT stage 的 epochs 必须大于 0")
        stage_overrides = stage.get("overrides", {}) or {}
        if not isinstance(stage_overrides, dict):
            raise ValueError("QAT stage.overrides 必须是映射")
        unknown_stage_overrides = set(stage_overrides) - QAT_STAGE_OVERRIDE_FIELDS
        if unknown_stage_overrides:
            raise ValueError(f"QAT stage.overrides 包含不允许的字段: {sorted(unknown_stage_overrides)}")
        resolve_nonideal_profile_raw(
            profile_config,
            str(stage["profile"]),
            str(values["inter_layer"]),
            str(values["bias_implementation"]),
        )
    if not isinstance(values["dynamic_aperture"], bool):
        raise ValueError("dynamic_aperture 必须是 true 或 false")
    if values["resume"] not in {"all", "weights", "none"}:
        raise ValueError("resume 必须是 all、weights 或 none")
    if values["resume"] == "all" and values["initial_checkpoint"]:
        raise ValueError("resume=all 不能同时设置 initial_checkpoint；请使用 resume=none 或 weights")
    values["start_fresh"] = values["resume"] == "none"
    if values["scheduler"] not in {"none", "cosine", "step", "multistep", "onecycle", "plateau"}:
        raise ValueError("scheduler 配置无效")
    if values["epochs"] < 0 or not np.isfinite(values["learning_rate"]) or values["learning_rate"] <= 0:
        raise ValueError("epochs 必须大于等于 0，learning_rate 必须为有限正数")
    if values["images_per_batch"] <= 0 or values["loader_workers"] < -1:
        raise ValueError("images_per_batch 必须大于 0，loader_workers 必须为 -1 或非负整数")
    if not 0.0 <= values["dropout"] < 1.0 or values["f_number"] <= 0:
        raise ValueError("dropout 必须在 [0, 1) 内，f_number 必须大于 0")
    if values["scheduler_step_epochs"] <= 0 or not 0.0 < values["scheduler_gamma"] <= 1.0:
        raise ValueError("scheduler_step_epochs 必须大于 0，scheduler_gamma 必须在 (0, 1] 内")
    milestones = values["scheduler_milestones"]
    if not isinstance(milestones, list) or any(int(value) <= 0 for value in milestones):
        raise ValueError("scheduler_milestones 必须是正整数列表")
    if values["scheduler_patience"] < 0:
        raise ValueError("scheduler_patience 必须大于等于 0")
    if values["cosine_restart_epochs"] <= 0 or values["gradient_clip_norm"] < 0:
        raise ValueError("调度周期必须大于 0，gradient_clip_norm 必须大于等于 0")
    if not 0.0 < values["onecycle_pct_start"] < 1.0:
        raise ValueError("onecycle_pct_start 必须在 (0, 1) 内")
    if (
        values["optimization_batch_pixels"] <= 0
        or values["train_pixels_per_image"] < 0
        or values["validation_pixels_per_image"] < 0
    ):
        raise ValueError(
            "optimization_batch_pixels 必须大于 0，train_pixels_per_image 和 validation_pixels_per_image "
            "必须大于等于 0"
        )
    if values["output_domain"] not in {"nonnegative", "signed"}:
        raise ValueError("output_domain 必须是 nonnegative 或 signed")
    if values["unity_constraint"] not in {"hard", "free"}:
        raise ValueError("unity_constraint 必须是 hard 或 free")
    if values["unity_scope"] not in {"global", "per_angle"}:
        raise ValueError("unity_scope 必须是 global 或 per_angle")
    values["angle_selection"] = normalize_angle_selection(values["angle_selection"])
    if not (
        values["angle_selection"] in {"center", "all"}
        if isinstance(values["angle_selection"], str)
        else isinstance(values["angle_selection"], (int, list))
    ):
        raise ValueError("angle_selection 必须是 center、all、角度数或索引列表")
    if values["angle_reduction"] not in {"sum", "mean"}:
        raise ValueError("angle_reduction 必须是 sum 或 mean")
    output_domain = values["output_domain"]
    unity_constraint = values["unity_constraint"]
    if output_domain == "nonnegative" and values["output_weights"] != "real":
        raise ValueError("relu 输出约束只支持 real 权重")
    _validate_loss_config(values["loss"], output_domain, unity_constraint)
    if not isinstance(values["hidden_layers"], int) or values["hidden_layers"] < 1:
        raise ValueError("hidden_layers 必须是正整数")
    if values["input_normalization"] not in {"none", "rms", "std"}:
        raise ValueError("输入归一化配置无效")
    if values["input_tile_features"] < 0:
        raise ValueError("input_tile_features 必须大于等于 0")
    if values["expected_network_channels"] is not None and (
        not isinstance(values["expected_network_channels"], int) or values["expected_network_channels"] < 1
    ):
        raise ValueError("expected_network_channels 若指定则必须是正整数")
    if values["output_interpolation"] not in {"linear", "nearest", "cubic"}:
        raise ValueError("output_interpolation 必须是 linear、nearest 或 cubic")
    if not 2 <= values["interpolation_bits"] <= 32:
        raise ValueError("interpolation_bits 必须是 2 到 32")
    if values["branch_mode"] not in {"single", "dual"}:
        raise ValueError("branch_mode 必须是 single 或 dual")

    # ---- activation：集中定义函数参数，再逐层选择 ----
    valid_layers = {f"fc{i}" for i in range(1, int(values["hidden_layers"]) + 1)}
    activation_config = values["activation"]
    if not isinstance(activation_config, dict) or set(activation_config) != {"functions", "layers"}:
        raise ValueError("activation 必须包含 functions 和 layers 两个映射")
    act_types = {"relu", "pwl", "tanh"}
    functions = activation_config["functions"]
    layers = activation_config["layers"]
    if not isinstance(functions, dict) or not isinstance(layers, dict):
        raise ValueError("activation.functions/layers 必须是映射")
    if set(functions) - act_types:
        raise ValueError(f"activation.functions 包含未知类型: {sorted(set(functions) - act_types)}")
    if set(layers) - valid_layers:
        raise ValueError(f"activation.layers 的层只能从 {sorted(valid_layers)} 选择")
    allowed_params = {"relu": set(), "pwl": {"knee", "tail_slope"}, "tanh": {"beta"}}
    for function_name, function_params in functions.items():
        if not isinstance(function_params, dict):
            raise ValueError(f"activation.functions[{function_name}] 必须是参数映射")
        unknown_params = set(function_params) - allowed_params[function_name]
        if unknown_params:
            raise ValueError(f"activation.functions[{function_name}] 包含未知参数: {sorted(unknown_params)}")
        if function_name == "pwl" and (
            float(function_params.get("knee", 1.0)) <= 0
            or not 0.0 <= float(function_params.get("tail_slope", 0.2)) <= 1.0
        ):
            raise ValueError("activation.functions[pwl] 需要 knee>0 且 0<=tail_slope<=1")
        if function_name == "tanh" and float(function_params.get("beta", 1.0)) <= 0:
            raise ValueError("activation.functions[tanh].beta 必须大于 0")
    activation_spec: dict[str, dict[str, object]] = {}
    for layer, assignment in layers.items():
        if isinstance(assignment, str):
            spec_type, overrides = assignment, {}
        elif isinstance(assignment, dict) and "type" in assignment:
            spec_type = str(assignment["type"])
            overrides = {key: value for key, value in assignment.items() if key != "type"}
        else:
            raise ValueError(f"activation.layers[{layer}] 必须是函数名或包含 type 的映射")
        if spec_type not in functions:
            raise ValueError(f"activation.layers[{layer}] 引用了未定义函数: {spec_type}")
        function_params = functions[spec_type]
        unknown_overrides = set(overrides) - allowed_params[spec_type]
        if unknown_overrides:
            raise ValueError(f"activation.layers[{layer}] 包含未知参数: {sorted(unknown_overrides)}")
        spec = {"type": spec_type, **function_params, **overrides}
        if spec_type == "pwl":
            knee = float(spec.get("knee", 1.0))
            tail_slope = float(spec.get("tail_slope", 0.2))
            if knee <= 0 or not 0.0 <= tail_slope <= 1.0:
                raise ValueError(f"activation[{layer}] pwl 需要 knee>0 且 0<=tail_slope<=1")
        elif spec_type == "tanh":
            if float(spec.get("beta", 1.0)) <= 0:
                raise ValueError(f"activation[{layer}] tanh beta 必须大于 0")
        activation_spec[layer] = spec
    values["activation"] = activation_spec

    # ---- normalization：互斥单选，L1 需 dual ----
    if values["centering"] not in {"none", "sample_mean"}:
        raise ValueError("centering 必须是 none 或 sample_mean")
    centering_layers = _validate_layer_list(values["centering_layers"], "centering_layers", valid_layers)
    values["centering_layers"] = centering_layers
    if (values["centering"] == "none") != (not centering_layers):
        raise ValueError("centering=none 时 centering_layers 必须为空；启用中心化时必须指定层")
    if not set(centering_layers).issubset(activation_spec):
        raise ValueError("centering_layers 必须是已配置激活函数的隐藏层")
    if values["centering"] == "sample_mean" and values["normalization"] in {"running_zscore", "batch_renorm"}:
        raise ValueError("sample_mean 中心化不能与运行统计归一化同时启用")
    if values["normalization"] not in {"none", "l1", "l2", "running_zscore", "batch_renorm"}:
        raise ValueError("normalization 必须是 none、l1、l2、running_zscore 或 batch_renorm")
    norm_layers = _validate_layer_list(values["normalization_layers"], "normalization_layers", valid_layers)
    values["normalization_layers"] = norm_layers
    if (values["normalization"] == "none") != (not norm_layers):
        raise ValueError("normalization=none 时 normalization_layers 必须为空；启用归一化时必须指定层")
    if values["normalization"] == "l1" and values["branch_mode"] != "dual":
        raise ValueError("L1归一化只允许在 branch_mode=dual 时启用")
    if (
        values["normalization"] in {"l1", "l2"}
        and values["branch_mode"] == "dual"
        and not set(norm_layers).issubset(activation_spec)
    ):
        raise ValueError("dual分支的 L1/L2 normalization_layers 必须是 activation 已配置层")
    valid_bias_layers = {f"fc{i}" for i in range(1, int(values["hidden_layers"]) + 2)}
    bias_layers = values["bias_layers"]
    if isinstance(bias_layers, str) and bias_layers.strip().lower() == "all":
        bias_layers = sorted(valid_bias_layers, key=lambda layer: int(layer[2:]))
    else:
        bias_layers = _validate_layer_list(bias_layers, "bias_layers", valid_bias_layers)
    values["bias_layers"] = bias_layers
    if values["bias_implementation"] not in BIAS_IMPLEMENTATIONS:
        raise ValueError(f"bias_implementation 必须是 {sorted(BIAS_IMPLEMENTATIONS)} 之一")
    if values["weight_transform"] not in {"none", "ws"}:
        raise ValueError("weight_transform 必须是 none 或 ws")
    if values["weight_transform_epsilon"] <= 0:
        raise ValueError("weight_transform_epsilon 必须大于 0")
    valid_weight_layers = {f"fc{i}" for i in range(1, int(values["hidden_layers"]) + 2)}
    weight_layers = _validate_layer_list(
        values["weight_transform_layers"], "weight_transform_layers", valid_weight_layers
    )
    values["weight_transform_layers"] = weight_layers
    if values["weight_transform"] == "none" and weight_layers:
        raise ValueError("weight_transform=none 时 weight_transform_layers 必须为空")
    if values["weight_transform"] == "ws" and not weight_layers:
        values["weight_transform_layers"] = sorted(valid_weight_layers)
    if not 0 < values["running_stat_momentum"] <= 1 or values["running_stat_epsilon"] <= 0:
        raise ValueError("运行统计动量和epsilon必须为正")
    if values["batch_renorm_rmax"] < 1 or values["batch_renorm_dmax"] < 0:
        raise ValueError("batch_renorm_rmax 必须大于等于1，batch_renorm_dmax 必须大于等于0")
    if values["control_normalization"] not in {"none", "linf"}:
        raise ValueError("control_normalization 必须是 none 或 linf")
    if values["control_adc_range"] not in {"observer", "full_scale"}:
        raise ValueError("control_adc_range 必须是 observer 或 full_scale")
    if values["control_adc_full_scale"] <= 0:
        raise ValueError("control_adc_full_scale 必须大于0")
    if values["output_controls"] < 0:
        raise ValueError("控制点数量必须大于等于 0")
    if values["projection_controls"] < 0:
        raise ValueError("投影控制点数量必须大于等于 0")
    if values["output_controls"] > 0 and values["projection_controls"] > 0:
        raise ValueError("output_controls 与 projection_controls 不能同时启用；Direct+Proj 请将 output_controls 设为 0")
    if values["distillation_weight"] < 0:
        raise ValueError("distillation_weight 必须大于等于 0")
    if not isinstance(values["target_algorithm"], str) or not values["target_algorithm"].strip():
        raise ValueError("target_algorithm 必须是非空算法名")
    values["target_algorithm"] = values["target_algorithm"].strip().lower()
    if values["distillation_weight"] > 0 and values["distillation_checkpoint"] is None:
        raise ValueError("启用蒸馏时必须提供 distillation_checkpoint")
    if values["save_images_every_epochs"] < 0:
        raise ValueError("save_images_every_epochs 必须大于等于 0")
    if values["qat_mode"] not in {"ptq", "qat"}:
        raise ValueError("qat_mode 必须是 ptq 或 qat")
    if values["inter_layer"] not in INTER_LAYER_MODES:
        raise ValueError("inter_layer 必须是 none、analog 或 digital")
    if not 2 <= values["inter_layer_bits"] <= 32:
        raise ValueError("inter_layer_bits 必须是 2 到 32")
    if (
        values["observer_calibration_epochs"] < 0
        or values["ptq_calibration_samples"] <= 0
        or values["public_gain_max"] <= 0
    ):
        raise ValueError("observer_calibration_epochs必须>=0，ptq_calibration_samples和public_gain_max必须>0")
    if not values["fixed_observer_calibration"] and values["observer_calibration_epochs"] <= 0:
        raise ValueError("关闭固定 observer 校准时，observer_calibration_epochs 必须大于 0")
    if not isinstance(values["activation_threshold_references"], dict):
        raise ValueError("activation_threshold_references必须是映射")
    QATConfig.from_namespace(SimpleNamespace(**values))
    values["_nonideal_profile_config"] = profile_config
    values["_nonideal_base_values"] = nonideal_base_values
    values["_active_profile_name"] = selected_profile
    base_dir = config_path.parent
    for key in ("output_directory", "split_file", "distillation_checkpoint", "initial_checkpoint"):
        value = values[key]
        if value is not None and not os.path.isabs(str(value)):
            values[key] = str((base_dir / str(value)).resolve())
    if not isinstance(values["h5_files"], list) or not values["h5_files"]:
        raise ValueError("h5_files 必须是非空列表")
    ratios = np.asarray([values["train_ratio"], values["validation_ratio"], values["test_ratio"]], dtype=np.float64)
    if ratios[0] <= 0 or ratios[1] <= 0 or ratios[2] < 0 or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("训练和验证比例必须>0，测试比例必须>=0，三者之和必须为1")
    values["h5_files"] = [
        str((base_dir / str(path)).resolve()) if not os.path.isabs(str(path)) else str(path)
        for path in values["h5_files"]
    ]
    return SimpleNamespace(**values)


def configured_h5_files(config_path: Path) -> tuple[Path, ...]:
    config = load_config(Path(config_path).resolve())
    return tuple(Path(path) for path in config.h5_files)


def load_nonideal_profile(
    config_path: Path,
    profile_name: str | None = None,
    hardware_config_path: Path | None = None,
    inter_layer: str | None = None,
    bias_implementation: str | None = None,
) -> list[str]:
    config_path = Path(config_path).resolve()
    hardware_config_path = Path(hardware_config_path).resolve() if hardware_config_path is not None else None
    with config_path.resolve().open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("训练配置必须是 YAML 映射")
    embedded_hardware = sorted(set(raw) & {"hardware", "hardware_execution", "nonidealities", "qat_schedule"})
    if embedded_hardware:
        raise ValueError(f"训练配置不得包含硬件配置段 {embedded_hardware}；请将硬件配置放入 hardware_eval.yaml")
    companion_hardware, companion_nonidealities, _ = _read_hardware_sections(config_path.resolve(), hardware_config_path)
    nonideal = companion_nonidealities or {}
    if not isinstance(nonideal, dict):
        raise ValueError("nonidealities 必须是映射")
    profile_config = {
        "version": 1,
        "active_profile": nonideal.get("active_profile", "ideal"),
        "hardware": nonideal.get("parameters", {}) or {},
        "profiles": nonideal.get("profiles", {}) or {},
    }
    return resolve_nonideal_profile_raw(
        profile_config,
        profile_name,
        str(inter_layer or companion_hardware.get("inter_layer", "none")),
        str(bias_implementation or companion_hardware.get("bias_implementation", "ordinary")),
    )


HARDWARE_EVAL_OVERRIDE_FIELDS = set(QATConfig.__dataclass_fields__) | {
    "input_tile_features",
    "interpolation_bits",
    "fixed_observer_calibration",
    "observer_calibration_epochs",
    "ptq_calibration_samples",
    "public_gain_max",
}


HARDWARE_BIT_FIELDS = {
    "weight_bits",
    "bias_bits",
    "input_bits",
    "inter_layer_bits",
    "control_bits",
    "interpolation_bits",
}
HARDWARE_STRESS_OVERRIDE_FIELDS = HARDWARE_EVAL_OVERRIDE_FIELDS - HARDWARE_BIT_FIELDS - {
    "enabled",
    "mode",
    "inter_layer",
    "bias_implementation",
    "g_min",
    "g_max",
    "per_output_channel",
    "observer_enabled",
}


def _validate_hardware_eval_overrides(
    value: object,
    label: str,
    allowed_fields: set[str] | frozenset[str] | None = None,
) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} 必须是映射")
    normalized = {str(key): item for key, item in value.items()}
    valid_fields = HARDWARE_EVAL_OVERRIDE_FIELDS if allowed_fields is None else set(allowed_fields)
    unknown = set(normalized) - valid_fields
    if unknown:
        raise ValueError(f"{label} 包含未知硬件字段: {sorted(unknown)}")
    return normalized


def _validate_hardware_eval_grid(value: object, label: str) -> dict[str, list[object]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} 必须是映射")
    grid: dict[str, list[object]] = {}
    for key, values in value.items():
        key = str(key)
        if key not in {
            "tile_rows",
            "tile_cols",
            "mapping_mode",
            "converter_schedule",
            "input_timing",
            "differential_readout",
            "weight_bits",
            "input_bits",
            "adc_bits",
            "inter_layer",
            "inter_layer_bits",
            "technology_node",
            "crosssim_runs",
            "seed",
        }:
            raise ValueError(f"{label} 包含未知字段: {key}")
        if not isinstance(values, list) or not values:
            raise ValueError(f"{label}.{key} 必须是非空列表")
        grid[key] = list(values)
    return grid


def load_hardware_eval_config(config_path: Path) -> dict[str, object]:
    config_path = Path(config_path).resolve()
    with config_path.resolve().open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("硬件评估配置必须是 YAML 映射")
    if raw.get("version", 1) != 1:
        raise ValueError("硬件评估配置 version 必须为 1")
    unknown = set(raw) - {"version", "hardware", "qat_schedule", "nonidealities", "stress", "ppa"}
    if unknown:
        raise ValueError(f"硬件评估配置包含未知顶层字段: {sorted(unknown)}")

    hardware_raw = raw.get("hardware", {}) or {}
    if not isinstance(hardware_raw, dict):
        raise ValueError("hardware_eval.hardware 必须是映射")
    hardware_defaults = _validate_hardware_eval_overrides(hardware_raw, "hardware_eval.hardware")
    qat_schedule = raw.get("qat_schedule")
    if qat_schedule is not None and not isinstance(qat_schedule, dict):
        raise ValueError("hardware_eval.qat_schedule 必须是映射")
    nonidealities = raw.get("nonidealities", {}) or {}
    if not isinstance(nonidealities, dict):
        raise ValueError("hardware_eval.nonidealities 必须是映射")
    if set(nonidealities) - {"active_profile", "parameters", "profiles"}:
        raise ValueError(
            "hardware_eval.nonidealities 包含未知字段: "
            f"{sorted(set(nonidealities) - {'active_profile', 'parameters', 'profiles'})}"
        )
    nonideal_parameters = _validate_hardware_eval_overrides(
        nonidealities.get("parameters", {}),
        "hardware_eval.nonidealities.parameters",
        NONIDEAL_PROFILE_FIELDS,
    )
    nonideal_profiles = nonidealities.get("profiles", {}) or {}
    if not isinstance(nonideal_profiles, dict):
        raise ValueError("hardware_eval.nonidealities.profiles 必须是映射")
    for profile_name, profile in nonideal_profiles.items():
        if not isinstance(profile, dict):
            raise ValueError(f"hardware_eval.nonidealities.profiles.{profile_name} 必须是映射")
        if set(profile) - {"extends", "requires_parameters", "parameters"}:
            raise ValueError(f"hardware_eval.nonidealities.profiles.{profile_name} 包含未知字段")
        _validate_hardware_eval_overrides(
            profile.get("parameters", {}),
            f"hardware_eval.nonidealities.profiles.{profile_name}.parameters",
            NONIDEAL_PROFILE_FIELDS,
        )

    stress_raw = raw.get("stress", {}) or {}
    if not isinstance(stress_raw, dict):
        raise ValueError("hardware_eval.stress 必须是映射")
    unknown_stress = set(stress_raw) - {"mc_runs", "base_overrides", "cases"}
    if unknown_stress:
        raise ValueError(f"hardware_eval.stress 包含未知字段: {sorted(unknown_stress)}")
    mc_runs = int(stress_raw.get("mc_runs", 30))
    if mc_runs <= 1:
        raise ValueError("hardware_eval.stress.mc_runs 必须大于 1")
    cases_raw = stress_raw.get("cases", []) or []
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ValueError("hardware_eval.stress.cases 必须是非空列表")
    cases: list[dict[str, object]] = []
    names: set[str] = set()
    for index, case_raw in enumerate(cases_raw):
        if not isinstance(case_raw, dict):
            raise ValueError(f"hardware_eval.stress.cases[{index}] 必须是映射")
        unknown_case = set(case_raw) - {
            "name",
            "profile",
            "protocol",
            "noise_enabled",
            "mc_runs",
            "overrides",
        }
        if unknown_case:
            raise ValueError(f"hardware_eval.stress.cases[{index}] 包含未知字段: {sorted(unknown_case)}")
        name = str(case_raw.get("name", "")).strip()
        if not name or name in names:
            raise ValueError(f"hardware_eval.stress.cases 名称必须非空且不重复: {name!r}")
        protocol = str(case_raw.get("protocol", "test")).strip().lower()
        if protocol not in {"test", "mc"}:
            raise ValueError(f"hardware_eval.stress.cases[{name}].protocol 必须为 test 或 mc")
        profile_value = case_raw.get("profile")
        profile = str(profile_value).strip() if profile_value is not None else None
        if profile == "":
            raise ValueError(f"hardware_eval.stress.cases[{name}].profile 不能为空")
        if profile is not None and profile in INTER_LAYER_MODES:
            raise ValueError(
                f"hardware_eval.stress.cases[{name}].profile 只能表示硬件压力 profile；"
                "analog/digital 请通过 hardware.inter_layer 选择"
            )
        if profile is not None and profile not in nonideal_profiles and profile not in HARDWARE_STRESS_PROFILES:
            raise ValueError(f"hardware_eval.stress.cases[{name}].profile 不存在: {profile}")
        if profile is None:
            raise ValueError(f"hardware_eval.stress.cases[{name}] 必须指定 profile")
        case_mc_runs = case_raw.get("mc_runs")
        if case_mc_runs is not None and int(case_mc_runs) <= 1:
            raise ValueError(f"hardware_eval.stress.cases[{name}].mc_runs 必须大于 1")
        names.add(name)
        cases.append(
            {
                "name": name,
                "profile": profile,
                "protocol": protocol,
                "noise_enabled": bool(case_raw.get("noise_enabled", protocol == "mc")),
                "mc_runs": int(case_mc_runs) if case_mc_runs is not None else mc_runs,
                "overrides": _validate_hardware_eval_overrides(
                    case_raw.get("overrides"),
                    f"case {name}.overrides",
                    HARDWARE_STRESS_OVERRIDE_FIELDS,
                ),
            }
        )

    ppa_raw = raw.get("ppa", {}) or {}
    if not isinstance(ppa_raw, dict):
        raise ValueError("hardware_eval.ppa 必须是映射")
    if set(ppa_raw) - {"mapping", "crosssim", "ppa"}:
        raise ValueError(f"hardware_eval.ppa 包含未知字段: {sorted(set(ppa_raw) - {'mapping', 'crosssim', 'ppa'})}")
    ppa_modes: dict[str, dict[str, dict[str, list[object]]]] = {}
    for mode in ("mapping", "crosssim", "ppa"):
        mode_raw = ppa_raw.get(mode, {}) or {}
        if not isinstance(mode_raw, dict) or set(mode_raw) - {"grid"}:
            raise ValueError(f"hardware_eval.ppa.{mode} 必须只包含 grid")
        ppa_modes[mode] = {"grid": _validate_hardware_eval_grid(mode_raw.get("grid"), f"ppa.{mode}.grid")}
    return {
        "version": 1,
        "hardware": hardware_defaults,
        "qat_schedule": qat_schedule,
        "nonidealities": {
            **nonidealities,
            "parameters": nonideal_parameters,
            "profiles": nonideal_profiles,
        },
        "stress": {
            "mc_runs": mc_runs,
            "base_overrides": _validate_hardware_eval_overrides(
                stress_raw.get("base_overrides"),
                "stress.base_overrides",
                HARDWARE_STRESS_OVERRIDE_FIELDS,
            ),
            "cases": cases,
        },
        "ppa": ppa_modes,
    }


def format_hardware_eval_overrides(overrides: dict[str, object], context: dict[str, object] | None = None) -> list[str]:
    context = context or {}
    result = []
    for key, raw_value in overrides.items():
        if key not in HARDWARE_EVAL_OVERRIDE_FIELDS:
            raise ValueError(f"硬件评估 override 包含未知字段: {key}")
        value = raw_value
        if isinstance(value, str):
            for name, replacement in context.items():
                value = value.replace(f"${{{name}}}", str(replacement))
        if isinstance(value, bool):
            text = "true" if value else "false"
        elif value is None:
            text = "null"
        elif isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        else:
            text = str(value)
        result.append(f"{key}={text}")
    return result


def hardware_eval_default_overrides(config: dict[str, object]) -> dict[str, object]:
    nonidealities = config.get("nonidealities", {}) or {}
    parameters = nonidealities.get("parameters", {}) if isinstance(nonidealities, dict) else {}
    if not isinstance(parameters, dict):
        raise ValueError("hardware_eval.nonidealities.parameters 必须是映射")
    hardware = config.get("hardware", {}) or {}
    if not isinstance(hardware, dict):
        raise ValueError("hardware_eval.hardware 必须是映射")
    return {**hardware, **parameters}


def expand_hardware_eval_ppa_cases(config: dict[str, object], mode: str) -> list[dict[str, object]]:
    if mode not in {"mapping", "crosssim", "ppa"}:
        raise ValueError(f"不支持的 PPA 评估模式: {mode}")
    grid = config["ppa"][mode]["grid"]
    if not grid:
        return [{"name": "default"}]
    keys = list(grid)
    cases = []
    for values in itertools.product(*(grid[key] for key in keys)):
        fields = dict(zip(keys, values, strict=True))
        name = "_".join(f"{key}-{str(value).replace('/', '-')}" for key, value in fields.items())
        cases.append({"name": name, **fields})
    return cases

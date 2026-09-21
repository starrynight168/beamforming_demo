from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import torch
import yaml


def stable_hash(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) & ((1 << 63) - 1)


NONIDEAL_PROFILE_FIELDS = {
    "static_mismatch_std",
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
    "bias_programming_gain_error",
    "bias_programming_offset",
    "bias_noise_std",
    "bias_channel_gain_mismatch_std",
    "bias_noise_reference",
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

HARDWARE_STRESS_PROFILES = frozenset({"ideal", "common"})
INTER_LAYER_MODES = frozenset({"none", "analog", "digital"})
BIAS_IMPLEMENTATIONS = frozenset({"ordinary", "array"})

ANALOG_INTER_LAYER_FIELDS = frozenset(
    {
        "activation_threshold",
        "activation_threshold_mismatch",
        "activation_gain_mismatch",
        "buffer_gain_error",
        "buffer_noise_std",
    }
)
BIAS_NONIDEAL_FIELDS = frozenset(
    {
        "bias_programming_gain_error",
        "bias_programming_offset",
        "bias_noise_std",
        "bias_channel_gain_mismatch_std",
    }
)


def resolve_layer_bias_implementation(requested: str, inter_layer: str) -> str:
    if requested not in BIAS_IMPLEMENTATIONS:
        raise ValueError(f"bias_implementation must be one of {sorted(BIAS_IMPLEMENTATIONS)}")
    if inter_layer not in INTER_LAYER_MODES:
        raise ValueError(f"inter_layer must be one of {sorted(INTER_LAYER_MODES)}")
    if requested == "array":
        return "array"
    return inter_layer


def _profile_override_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return yaml.safe_dump(value, allow_unicode=True, default_flow_style=True).strip()
    return str(value)


def resolve_nonideal_profile_raw(
    raw: dict,
    profile_name: str | None = None,
    inter_layer: str = "none",
    bias_implementation: str = "ordinary",
) -> list[str]:
    if not isinstance(raw, dict) or not isinstance(raw.get("profiles"), dict):
        raise ValueError("硬件配置必须包含 profiles 映射")
    unknown_root = set(raw) - {"version", "active_profile", "hardware", "profiles"}
    if unknown_root:
        raise ValueError(f"硬件配置包含未知顶层字段: {sorted(unknown_root)}")
    profiles = raw["profiles"]
    requested_name = str(profile_name or raw.get("active_profile", "ideal"))
    name = requested_name
    resolving: set[str] = set()

    def resolve(current: str) -> dict[str, object]:
        if current not in profiles or not isinstance(profiles[current], dict):
            raise ValueError(f"非理想 profile 不存在或格式错误: {current}")
        if current in resolving:
            raise ValueError(f"非理想 profile extends 循环: {current}")
        resolving.add(current)
        item = profiles[current]
        unknown = set(item) - {"extends", "enabled", "parameters", "metadata", "requires_parameters"}
        if unknown:
            raise ValueError(f"非理想 profile {current} 包含未知字段: {sorted(unknown)}")
        parent = item.get("extends")
        result: dict[str, object] = {"parameters": {}, "provided": set(), "required": []}
        if parent is not None:
            result = resolve(str(parent))
        item_parameters = item.get("parameters", {}) or {}
        if not isinstance(item_parameters, dict):
            raise ValueError(f"非理想 profile {current}.parameters 必须是映射")
        parameters = dict(result["parameters"])
        parameters.update(item_parameters)
        result["parameters"] = parameters
        result["provided"] = set(result["provided"]) | (set(item_parameters) if current != "ideal" else set())
        result["required"] = list(result["required"]) + list(item.get("requires_parameters", []) or [])
        resolving.remove(current)
        return result

    resolved = resolve(name)
    parameters = resolved["parameters"]
    unknown = set(parameters) - NONIDEAL_PROFILE_FIELDS
    if unknown:
        raise ValueError(f"非理想 profile {name} 包含未知字段: {sorted(unknown)}")
    missing = [key for key in resolved["required"] if key not in resolved["provided"] or parameters.get(key) is None]
    if missing:
        raise ValueError(f"非理想 profile {name} 尚未填写有来源参数: {sorted(missing)}")
    if any(value is None for value in parameters.values()):
        raise ValueError(f"非理想 profile {name} 不能包含 null 参数")
    resolved_bias_implementation = resolve_layer_bias_implementation(bias_implementation, inter_layer)
    inactive_fields = set()
    if inter_layer != "analog":
        inactive_fields.update(ANALOG_INTER_LAYER_FIELDS)
    if inter_layer == "none" or resolved_bias_implementation != "analog":
        inactive_fields.update(BIAS_NONIDEAL_FIELDS)
    for key in inactive_fields:
        parameters[key] = 0.0
    static_mismatch = parameters.get("static_mismatch_std")
    if isinstance(static_mismatch, dict):
        inactive_mismatch = set(inactive_fields) & set(static_mismatch)
        parameters["static_mismatch_std"] = {
            key: value for key, value in static_mismatch.items() if key not in inactive_mismatch
        }
    return [f"{key}={_profile_override_value(value)}" for key, value in parameters.items()]


class FakeQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        scale: torch.Tensor,
        qmin: int,
        qmax: int,
    ) -> torch.Tensor:
        q = torch.clamp(torch.round(x / scale), qmin, qmax)
        return q * scale

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        return grad_output, None, None, None


@dataclass(frozen=True)
class QATConfig:
    enabled: bool = False
    mode: str = "qat"
    inter_layer: str = "none"
    inter_layer_bits: int = 4
    weight_bits: int = 4
    input_bits: int = 4
    control_bits: int = 4
    bias_bits: int = 4
    bias_implementation: str = "ordinary"
    qat_clip_outside_grad: float = 0.1
    bias_programming_gain_error: float = 0.0
    bias_programming_offset: float = 0.0
    bias_noise_std: float = 0.0
    bias_channel_gain_mismatch_std: float = 0.0
    bias_noise_reference: float = 1.0
    g_min: float = 0.2428115
    g_max: float = 1.0
    per_output_channel: bool = False
    observer_enabled: bool = True
    epsilon: float = 1.0e-12
    dac_gain_error: float = 0.0
    dac_offset: float = 0.0
    dac_noise_std: float = 0.0
    dac_noise_mode: str = "full_scale"
    dac_nonlinearity: float = 0.0
    dac_nonlinearity_beta: float = 3.0
    tia_gain_error: float = 0.0
    tia_channel_gain_mismatch_std: float = 0.0
    tia_offset: float = 0.0
    tia_channel_offset_mismatch_std: float = 0.0
    tia_noise_std: float = 0.0
    tia_noise_mode: str = "local_relative"
    tia_noise_reference: float = 1.0
    tia_saturation: float = 0.0
    tia_saturation_mode: str = "absolute"
    tia_saturation_reference: float = 1.0
    activation_threshold: float = 0.0
    activation_threshold_mode: str = "absolute"
    activation_threshold_mismatch: float = 0.0
    activation_threshold_mismatch_mode: str = "absolute"
    activation_threshold_reference: float = 1.0
    activation_threshold_references: dict[str, float] | None = None
    activation_gain_mismatch: float = 0.0
    buffer_gain_error: float = 0.0
    buffer_noise_std: float = 0.0
    write_error_std: float = 0.0
    write_noise_mode: str = "local_relative"
    read_noise_std: float = 0.0
    read_noise_mode: str = "common_mode_relative"
    control_reference_gain_error: float = 0.0
    control_reference_offset: float = 0.0
    control_reference_noise_std: float = 0.0
    control_reference_hold_error: float = 0.0
    ir_drop_enabled: bool = False
    ir_drop_tile_rows: int = 64
    ir_drop_tile_cols: int = 64
    ir_drop_wire_resistance: float = 0.35
    ir_drop_g_max: float = 3.13e-5
    ir_drop_v_read: float = 0.3
    adc_gain_error: float = 0.0
    adc_channel_gain_mismatch_std: float = 0.0
    adc_offset: float = 0.0
    adc_channel_offset_mismatch_std: float = 0.0
    adc_noise_std: float = 0.0
    adc_noise_mode: str = "full_scale"
    adc_nonlinearity: float = 0.0
    adc_nonlinearity_beta: float = 3.0
    drift_std: float = 0.0
    stuck_at_prob: float = 0.0
    stuck_at_high_fraction: float = 0.5
    noise_enabled: bool = False
    noise_seed: int = 0
    static_mismatch_std: dict[str, float] | None = None

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.activation_threshold_references is not None and any(
            not math.isfinite(float(value)) for value in self.activation_threshold_references.values()
        ):
            raise ValueError("activation_threshold_references values must be finite")
        if self.static_mismatch_std is not None:
            allowed_mismatch_fields = {
                "tia_gain_error",
                "tia_offset",
                "bias_programming_gain_error",
                "bias_programming_offset",
                "control_reference_gain_error",
                "control_reference_offset",
                "activation_gain_mismatch",
                "activation_threshold",
                "activation_threshold_mismatch",
                "buffer_gain_error",
                "adc_gain_error",
                "adc_offset",
                "dac_gain_error",
                "dac_offset",
            }
            unknown_mismatch_fields = set(self.static_mismatch_std) - allowed_mismatch_fields
            if unknown_mismatch_fields:
                raise ValueError(f"static_mismatch_std contains unsupported fields: {sorted(unknown_mismatch_fields)}")
            if any(
                float(value) < 0.0 or not math.isfinite(float(value)) for value in self.static_mismatch_std.values()
            ):
                raise ValueError("static_mismatch_std values must be finite and non-negative")
        if self.mode not in {"ptq", "qat"}:
            raise ValueError("mode must be 'ptq' or 'qat'")
        if self.inter_layer not in INTER_LAYER_MODES:
            raise ValueError("inter_layer must be 'none', 'analog', or 'digital'")
        if self.bias_implementation not in BIAS_IMPLEMENTATIONS:
            raise ValueError(f"bias_implementation must be one of {sorted(BIAS_IMPLEMENTATIONS)}")
        if self.tia_noise_mode not in {"local_relative", "full_scale"}:
            raise ValueError("tia_noise_mode must be local_relative or full_scale")
        if self.tia_saturation_mode not in {"absolute", "full_scale"}:
            raise ValueError("tia_saturation_mode must be absolute or full_scale")
        if self.activation_threshold_mismatch_mode not in {"absolute", "full_scale"}:
            raise ValueError("activation_threshold_mismatch_mode must be absolute or full_scale")
        if self.activation_threshold_mode not in {"absolute", "full_scale"}:
            raise ValueError("activation_threshold_mode must be absolute or full_scale")
        if self.write_noise_mode not in {"local_relative", "full_scale"}:
            raise ValueError("write_noise_mode must be local_relative or full_scale")
        if self.read_noise_mode not in {"common_mode_relative", "device_current_propagated"}:
            raise ValueError("read_noise_mode must be common_mode_relative or device_current_propagated")
        if self.adc_noise_mode not in {"local_relative", "full_scale"}:
            raise ValueError("adc_noise_mode must be local_relative or full_scale")
        if self.dac_noise_mode not in {"local_relative", "full_scale"}:
            raise ValueError("dac_noise_mode must be local_relative or full_scale")
        if self.ir_drop_tile_rows <= 0 or self.ir_drop_tile_cols <= 0:
            raise ValueError("ir_drop tile sizes must be positive")
        if self.ir_drop_wire_resistance < 0 or self.ir_drop_g_max <= 0 or self.ir_drop_v_read <= 0:
            raise ValueError("ir_drop physical parameters are invalid")
        if self.adc_nonlinearity_beta <= 0:
            raise ValueError("adc_nonlinearity_beta must be positive")
        if self.dac_nonlinearity_beta <= 0:
            raise ValueError("dac_nonlinearity_beta must be positive")
        if any(
            not 2 <= bits <= 32
            for bits in (
                self.weight_bits,
                self.input_bits,
                self.inter_layer_bits,
                self.control_bits,
                self.bias_bits,
            )
        ):
            raise ValueError("quantization bits must be in [2, 32]")
        if self.g_min < 0.0 or self.g_max <= self.g_min:
            raise ValueError("g_min must be non-negative and g_max must be greater than g_min")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if not 0.0 <= self.qat_clip_outside_grad <= 1.0:
            raise ValueError("qat_clip_outside_grad must be in [0, 1]")
        if self.bias_noise_reference <= 0.0:
            raise ValueError("bias_noise_reference must be positive")
        for name in (
            "tia_noise_reference",
            "tia_saturation_reference",
            "activation_threshold_reference",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.activation_threshold_references is not None and any(
            float(value) <= 0.0 for value in self.activation_threshold_references.values()
        ):
            raise ValueError("activation_threshold_references values must be positive")
        for name in (
            "dac_gain_error",
            "tia_gain_error",
            "buffer_gain_error",
        ):
            if not -1.0 < getattr(self, name) < 1.0:
                raise ValueError(f"{name} must satisfy -1 < value < 1")
        for name in ("tia_channel_gain_mismatch_std", "adc_channel_gain_mismatch_std"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must satisfy 0 <= value < 1")
        for name in (
            "dac_noise_std",
            "tia_noise_std",
            "tia_saturation",
            "tia_channel_offset_mismatch_std",
            "activation_threshold",
            "activation_threshold_mismatch",
            "buffer_noise_std",
            "write_error_std",
            "read_noise_std",
            "adc_noise_std",
            "adc_channel_offset_mismatch_std",
            "drift_std",
            "bias_noise_std",
            "control_reference_noise_std",
            "control_reference_hold_error",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= self.activation_gain_mismatch < 1.0:
            raise ValueError("activation_gain_mismatch must satisfy 0 <= value < 1")
        for name in ("bias_programming_gain_error", "control_reference_gain_error"):
            if not -1.0 < getattr(self, name) < 1.0:
                raise ValueError(f"{name} must satisfy -1 < value < 1")
        if not 0.0 <= self.bias_channel_gain_mismatch_std < 1.0:
            raise ValueError("bias_channel_gain_mismatch_std must satisfy 0 <= value < 1")
        if not -1.0 < self.adc_gain_error < 1.0:
            raise ValueError("adc_gain_error must satisfy -1 < value < 1")
        for name in ("adc_nonlinearity", "dac_nonlinearity"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0.0 <= self.stuck_at_prob <= 1.0 or not 0.0 <= self.stuck_at_high_fraction <= 1.0:
            raise ValueError("stuck-at probabilities must be in [0, 1]")

    @classmethod
    def from_namespace(cls, values: Any) -> QATConfig:
        return cls(
            enabled=bool(getattr(values, "qat_enabled", False)),
            mode=str(getattr(values, "qat_mode", "qat")),
            inter_layer=str(getattr(values, "inter_layer", "none")),
            inter_layer_bits=int(getattr(values, "inter_layer_bits", 4)),
            weight_bits=int(getattr(values, "weight_bits", 4)),
            input_bits=int(getattr(values, "input_bits", 4)),
            control_bits=int(getattr(values, "control_bits", 4)),
            bias_bits=int(getattr(values, "bias_bits", 4)),
            bias_implementation=str(getattr(values, "bias_implementation", "ordinary")),
            qat_clip_outside_grad=float(getattr(values, "qat_clip_outside_grad", 0.1)),
            bias_programming_gain_error=float(getattr(values, "bias_programming_gain_error", 0.0)),
            bias_programming_offset=float(getattr(values, "bias_programming_offset", 0.0)),
            bias_noise_std=float(getattr(values, "bias_noise_std", 0.0)),
            bias_channel_gain_mismatch_std=float(getattr(values, "bias_channel_gain_mismatch_std", 0.0)),
            bias_noise_reference=float(getattr(values, "bias_noise_reference", 1.0)),
            g_min=float(getattr(values, "g_min", 0.2428115)),
            g_max=float(getattr(values, "g_max", 1.0)),
            per_output_channel=bool(getattr(values, "per_output_channel", False)),
            observer_enabled=bool(getattr(values, "observer_enabled", True)),
            dac_gain_error=float(getattr(values, "dac_gain_error", 0.0)),
            dac_offset=float(getattr(values, "dac_offset", 0.0)),
            dac_noise_std=float(getattr(values, "dac_noise_std", 0.0)),
            dac_noise_mode=str(getattr(values, "dac_noise_mode", "full_scale")),
            dac_nonlinearity=float(getattr(values, "dac_nonlinearity", 0.0)),
            dac_nonlinearity_beta=float(getattr(values, "dac_nonlinearity_beta", 3.0)),
            tia_gain_error=float(getattr(values, "tia_gain_error", 0.0)),
            tia_channel_gain_mismatch_std=float(getattr(values, "tia_channel_gain_mismatch_std", 0.0)),
            tia_offset=float(getattr(values, "tia_offset", 0.0)),
            tia_channel_offset_mismatch_std=float(getattr(values, "tia_channel_offset_mismatch_std", 0.0)),
            tia_noise_std=float(getattr(values, "tia_noise_std", 0.0)),
            tia_noise_mode=str(getattr(values, "tia_noise_mode", "local_relative")),
            tia_noise_reference=float(getattr(values, "tia_noise_reference", 1.0)),
            tia_saturation=float(getattr(values, "tia_saturation", 0.0)),
            tia_saturation_mode=str(getattr(values, "tia_saturation_mode", "absolute")),
            tia_saturation_reference=float(getattr(values, "tia_saturation_reference", 1.0)),
            activation_threshold=float(getattr(values, "activation_threshold", 0.0)),
            activation_threshold_mode=str(getattr(values, "activation_threshold_mode", "absolute")),
            activation_threshold_mismatch=float(getattr(values, "activation_threshold_mismatch", 0.0)),
            activation_threshold_mismatch_mode=str(getattr(values, "activation_threshold_mismatch_mode", "absolute")),
            activation_threshold_reference=float(getattr(values, "activation_threshold_reference", 1.0)),
            activation_threshold_references=getattr(values, "activation_threshold_references", None),
            activation_gain_mismatch=float(getattr(values, "activation_gain_mismatch", 0.0)),
            buffer_gain_error=float(getattr(values, "buffer_gain_error", 0.0)),
            buffer_noise_std=float(getattr(values, "buffer_noise_std", 0.0)),
            write_error_std=float(getattr(values, "write_error_std", 0.0)),
            write_noise_mode=str(getattr(values, "write_noise_mode", "local_relative")),
            read_noise_std=float(getattr(values, "read_noise_std", 0.0)),
            read_noise_mode=str(getattr(values, "read_noise_mode", "common_mode_relative")),
            control_reference_gain_error=float(getattr(values, "control_reference_gain_error", 0.0)),
            control_reference_offset=float(getattr(values, "control_reference_offset", 0.0)),
            control_reference_noise_std=float(getattr(values, "control_reference_noise_std", 0.0)),
            control_reference_hold_error=float(getattr(values, "control_reference_hold_error", 0.0)),
            ir_drop_enabled=bool(getattr(values, "ir_drop_enabled", False)),
            ir_drop_tile_rows=int(getattr(values, "ir_drop_tile_rows", 64)),
            ir_drop_tile_cols=int(getattr(values, "ir_drop_tile_cols", 64)),
            ir_drop_wire_resistance=float(getattr(values, "ir_drop_wire_resistance", 0.35)),
            ir_drop_g_max=float(getattr(values, "ir_drop_g_max", 3.13e-5)),
            ir_drop_v_read=float(getattr(values, "ir_drop_v_read", 0.3)),
            adc_gain_error=float(getattr(values, "adc_gain_error", 0.0)),
            adc_channel_gain_mismatch_std=float(getattr(values, "adc_channel_gain_mismatch_std", 0.0)),
            adc_offset=float(getattr(values, "adc_offset", 0.0)),
            adc_channel_offset_mismatch_std=float(getattr(values, "adc_channel_offset_mismatch_std", 0.0)),
            adc_noise_std=float(getattr(values, "adc_noise_std", 0.0)),
            adc_noise_mode=str(getattr(values, "adc_noise_mode", "full_scale")),
            adc_nonlinearity=float(getattr(values, "adc_nonlinearity", 0.0)),
            adc_nonlinearity_beta=float(getattr(values, "adc_nonlinearity_beta", 3.0)),
            drift_std=float(getattr(values, "drift_std", 0.0)),
            stuck_at_prob=float(getattr(values, "stuck_at_prob", 0.0)),
            stuck_at_high_fraction=float(getattr(values, "stuck_at_high_fraction", 0.5)),
            noise_enabled=bool(getattr(values, "noise_enabled", False)),
            noise_seed=int(getattr(values, "noise_seed", 0)),
            static_mismatch_std=getattr(values, "static_mismatch_std", None),
        )


def _signed_fake_quant(
    x: torch.Tensor,
    bits: int,
    abs_max: torch.Tensor,
    epsilon: float,
    outside_grad: float = 1.0,
) -> torch.Tensor:
    qmax = (1 << (bits - 1)) - 1
    safe_abs_max = abs_max.detach().clamp_min(epsilon)
    normalized = x / safe_abs_max
    clipped = normalized.clamp(-1.0, 1.0)
    gradient = torch.where(
        normalized.abs() <= 1.0,
        torch.ones_like(normalized),
        normalized.new_tensor(float(outside_grad)),
    )
    normalized = clipped.detach() + (normalized - normalized.detach()) * gradient.detach()
    scale = safe_abs_max / qmax
    return FakeQuantSTE.apply(normalized * safe_abs_max, scale, -qmax, qmax)


@dataclass(frozen=True)
class ConductanceState:
    effective_weight: torch.Tensor
    g_plus: torch.Tensor
    g_minus: torch.Tensor
    weight_scale: torch.Tensor

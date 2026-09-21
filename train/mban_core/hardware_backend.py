from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hardware import ConductanceState, FakeQuantSTE, QATConfig, _signed_fake_quant, stable_hash
from .naming import bias_dac_name, bias_range_name, input_dac_name


def _ste_clamp(
    x: torch.Tensor,
    minimum: float | torch.Tensor,
    maximum: float | torch.Tensor,
    outside_grad: float,
) -> torch.Tensor:
    clipped = x.clamp(minimum, maximum)
    if outside_grad == 0.0:
        return clipped
    distance = torch.where(x < minimum, minimum - x, torch.where(x > maximum, x - maximum, torch.zeros_like(x)))
    gradient = torch.where(distance == 0.0, torch.ones_like(x), outside_grad / (1.0 + distance))
    return clipped.detach() + (x - x.detach()) * gradient.detach()


class HardwareQAT:
    def __init__(self, config: QATConfig) -> None:
        self.config = config
        self.profile_name = "ideal"
        self.stage_name = None
        self.observer_enabled = config.observer_enabled
        self.activation_ranges: dict[str, torch.Tensor] = {}
        self.weight_scales: dict[str, torch.Tensor] = {}
        self._diagnostics_enabled = False
        self._diagnostic_weight_sums: list[torch.Tensor] = []
        self._diagnostic_depth_fractions: list[torch.Tensor] = []
        self._diagnostic_control_groups = 0
        self._diagnostic_control_pixels = 0
        self._diagnostic_control_codes = 0
        self._diagnostic_all_zero_groups = 0
        self._diagnostic_all_zero_pixels = 0
        self._diagnostic_zero_codes = 0
        self._diagnostic_saturated_control_codes = 0
        self._diagnostic_endpoint_control_codes = 0
        self._diagnostic_used_control_codes: set[int] = set()
        self._diagnostic_quantizer_total: dict[str, int] = {}
        self._diagnostic_quantizer_saturated: dict[str, int] = {}
        self._diagnostic_control_values: list[torch.Tensor] = []
        self._diagnostic_control_codes_by_depth: list[torch.Tensor] = []
        self._diagnostic_control_depths: list[torch.Tensor] = []
        self._noise_enabled = config.noise_enabled
        self._dynamic_noise_counters: dict[tuple[str, str], int] = {}
        self._write_noise_cache: dict[tuple[str, str], torch.Tensor] = {}
        self._device_state_cache: dict[tuple[str, str], torch.Tensor] = {}
        self._static_mismatch_cache: dict[tuple[str, str], float] = {}
        self._channel_mismatch_cache: dict[tuple[str, str], torch.Tensor] = {}
        self._realization_index = 0
        self._realization_seed_base: int | None = None
        self.adaptive_valid: torch.Tensor | None = None
        self.output_control_range_name: str | None = None
        self.output_preactivation_range_name: str | None = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def enable_observer(self) -> None:
        self.observer_enabled = True

    def disable_observer(self) -> None:
        self.observer_enabled = False

    def clear_observer(self) -> None:
        self.activation_ranges.clear()

    def enable_noise(self) -> None:
        self._noise_enabled = True

    def disable_noise(self) -> None:
        self._noise_enabled = False

    def set_config(self, config: QATConfig, profile_name: str | None = None, stage_name: str | None = None) -> None:
        self.config = config
        if profile_name is not None:
            self.profile_name = str(profile_name)
        if stage_name is not None:
            self.stage_name = str(stage_name)
        self.observer_enabled = config.observer_enabled
        self._noise_enabled = config.noise_enabled
        self._realization_index = 0
        self._realization_seed_base = None
        self.adaptive_valid = None
        self.reset_noise_counter()

    def reset_noise_counter(self) -> None:
        self._dynamic_noise_counters.clear()
        self._write_noise_cache.clear()
        self._device_state_cache.clear()
        self._static_mismatch_cache.clear()
        self._channel_mismatch_cache.clear()

    def begin_noise_realization(self) -> None:
        if self._realization_seed_base != self.config.noise_seed:
            self._realization_seed_base = self.config.noise_seed
            self._realization_index = 0
        self._realization_index += 1
        self.reset_noise_counter()

    def realization_state(self) -> dict[str, Any]:
        return {
            "realization_index": self._realization_index,
            "realization_seed_base": self._realization_seed_base,
            "dynamic_noise_counters": dict(self._dynamic_noise_counters),
            "write_noise_cache": {key: value.detach().clone() for key, value in self._write_noise_cache.items()},
            "device_state_cache": {key: value.detach().clone() for key, value in self._device_state_cache.items()},
            "static_mismatch_cache": dict(self._static_mismatch_cache),
            "channel_mismatch_cache": {
                key: value.detach().clone() for key, value in self._channel_mismatch_cache.items()
            },
        }

    def restore_realization_state(self, state: dict[str, Any] | None) -> None:
        if state is None:
            return
        self._realization_index = int(state["realization_index"])
        self._realization_seed_base = state["realization_seed_base"]
        self._dynamic_noise_counters = dict(state["dynamic_noise_counters"])
        self._write_noise_cache = {
            key: value.detach().clone() for key, value in state["write_noise_cache"].items()
        }
        self._device_state_cache = {
            key: value.detach().clone() for key, value in state["device_state_cache"].items()
        }
        self._static_mismatch_cache = dict(state["static_mismatch_cache"])
        self._channel_mismatch_cache = {
            key: value.detach().clone() for key, value in state["channel_mismatch_cache"].items()
        }

    def _stable_seed(self, *parts: object) -> int:
        return stable_hash(self.config.noise_seed, self._realization_index, *parts)

    def _generator(self, device: torch.device, *parts: object) -> torch.Generator:
        return torch.Generator(device=device).manual_seed(self._stable_seed(*parts))

    def _has_static_mismatch(self, *names: str) -> bool:
        configured = self.config.static_mismatch_std or {}
        return self._noise_enabled and any(float(configured.get(name, 0.0)) > 0.0 for name in names)

    def _static_parameter(self, name: str | None, value: float, stream: str = "global") -> float:
        if name is None or not self._noise_enabled:
            return value
        std = float((self.config.static_mismatch_std or {}).get(name, 0.0))
        if std <= 0.0:
            return value
        key = (name, stream)
        mismatch = self._static_mismatch_cache.get(key)
        if mismatch is None:
            generator = self._generator(torch.device("cpu"), "static_mismatch", name, stream)
            mismatch = float(torch.randn((), generator=generator)) * std
        self._static_mismatch_cache[key] = mismatch
        return value + mismatch

    def _static_channel_parameter(
        self,
        like: torch.Tensor,
        name: str,
        std: float,
        stream: str,
    ) -> torch.Tensor:
        if not self._noise_enabled or std <= 0.0:
            return like.new_zeros((like.shape[-1],))
        channels = int(like.shape[-1])
        key = (name, stream)
        mismatch = self._channel_mismatch_cache.get(key)
        if mismatch is None or mismatch.numel() != channels:
            generator = self._generator(torch.device("cpu"), "static_channel_mismatch", name, stream)
            mismatch = torch.randn((channels,), generator=generator, dtype=torch.float32) * std
            self._channel_mismatch_cache[key] = mismatch
        return mismatch.to(device=like.device, dtype=like.dtype)

    def _noise(
        self,
        like: torch.Tensor,
        std: float,
        mode: str = "local_relative",
        reference: float | None = None,
        *,
        factor: str = "dynamic_noise",
        stream: str = "global",
    ) -> torch.Tensor:
        """Draw dynamic noise from a stable factor/stream sequence."""
        if not self._noise_enabled or std <= 0.0 or like.numel() == 0:
            return like.new_zeros(like.shape)
        key = (factor, stream)
        occurrence = self._dynamic_noise_counters.get(key, 0)
        self._dynamic_noise_counters[key] = occurrence + 1
        generator = self._generator(like.device, factor, stream, occurrence)
        if mode == "full_scale":
            if reference is None or reference <= 0.0:
                raise ValueError("full_scale noise requires a positive reference")
            magnitude = like.new_tensor(reference)
        else:
            magnitude = like.abs().detach().clamp_min(self.config.epsilon)
        return torch.randn(like.shape, generator=generator, device=like.device, dtype=like.dtype) * std * magnitude

    def _noise_from_magnitude(
        self,
        like: torch.Tensor,
        magnitude: torch.Tensor,
        std: float,
        factor: str,
        stream: str,
    ) -> torch.Tensor:
        if not self._noise_enabled or std <= 0.0 or like.numel() == 0:
            return like.new_zeros(like.shape)
        key = (factor, stream)
        occurrence = self._dynamic_noise_counters.get(key, 0)
        self._dynamic_noise_counters[key] = occurrence + 1
        generator = self._generator(like.device, factor, stream, occurrence)
        magnitude = magnitude.to(device=like.device, dtype=like.dtype).detach()
        return torch.randn(like.shape, generator=generator, device=like.device, dtype=like.dtype) * std * magnitude

    def _write_noise(
        self,
        like: torch.Tensor,
        std: float,
        name: str,
        mode: str = "local_relative",
        reference: float | None = None,
    ) -> torch.Tensor:
        key = ("write_error", name)
        cached = self._write_noise_cache.get(key)
        if cached is None or cached.shape != like.shape or cached.device != like.device:
            generator = self._generator(like.device, "write_error", name)
            cached = torch.randn(like.shape, generator=generator, device=like.device, dtype=torch.float32)
            self._write_noise_cache[key] = cached
        cached = cached.to(dtype=like.dtype)
        if mode == "full_scale":
            if reference is None or reference <= 0.0:
                raise ValueError("full_scale write noise requires a positive reference")
            magnitude = like.new_tensor(reference)
        else:
            magnitude = like.abs().detach().clamp_min(self.config.epsilon)
        return cached * std * magnitude

    def _realization_random(self, like: torch.Tensor, factor: str, stream: str, distribution: str) -> torch.Tensor:
        key = (factor, stream)
        cached = self._device_state_cache.get(key)
        if cached is None or cached.shape != like.shape or cached.device != like.device:
            generator = self._generator(like.device, factor, stream)
            if distribution == "normal":
                cached = torch.randn(like.shape, generator=generator, device=like.device, dtype=torch.float32)
            elif distribution == "uniform":
                cached = torch.rand(like.shape, generator=generator, device=like.device, dtype=torch.float32)
            else:
                raise ValueError(f"unknown realization distribution: {distribution}")
            self._device_state_cache[key] = cached
        return cached.to(dtype=like.dtype)

    def _apply_static_device_state(self, conductance: torch.Tensor, name: str) -> torch.Tensor:
        """Apply one realization's static conductance stress.

        ``drift_std`` is a normalized stress amplitude here; it is not a
        time- or temperature-dependent retention law.
        """
        if not self._noise_enabled:
            return conductance
        cfg = self.config
        state = conductance
        if cfg.drift_std > 0.0:
            drift = self._realization_random(state, "drift", name, "normal")
            state = state + drift * cfg.drift_std * (cfg.g_max - cfg.g_min)
        if cfg.stuck_at_prob > 0.0:
            fault = self._realization_random(state, "stuck_at", name, "uniform")
            low_probability = cfg.stuck_at_prob * (1.0 - cfg.stuck_at_high_fraction)
            low = fault < low_probability
            high = (fault >= low_probability) & (fault < cfg.stuck_at_prob)
            state = torch.where(low, state.new_tensor(cfg.g_min), state)
            state = torch.where(high, state.new_tensor(cfg.g_max), state)
        return state.clamp(cfg.g_min, cfg.g_max)

    def _analog_signal(
        self,
        x: torch.Tensor,
        gain_error: float,
        offset: float,
        noise_std: float,
        saturation: float,
        name: str,
        noise_mode: str = "local_relative",
        noise_reference: float | None = None,
        saturation_mode: str = "absolute",
        saturation_reference: float = 1.0,
        gain_key: str | None = None,
        offset_key: str | None = None,
        channel_gain_mismatch_std: float = 0.0,
        channel_offset_mismatch_std: float = 0.0,
        channel_mismatch_name: str | None = None,
        noise_factor: str = "analog_noise",
        common_mode: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Deterministic analog non-idealities: gain error, offset, noise, saturation."""
        if not self.enabled:
            return x
        gain_error = self._static_parameter(gain_key, gain_error, name)
        offset = self._static_parameter(offset_key, offset, name)
        if channel_mismatch_name is not None:
            gain_error = gain_error + self._static_channel_parameter(
                x,
                f"{channel_mismatch_name}.gain",
                channel_gain_mismatch_std,
                name,
            ).view(*([1] * (x.ndim - 1)), -1)
            offset = offset + self._static_channel_parameter(
                x,
                f"{channel_mismatch_name}.offset",
                channel_offset_mismatch_std,
                name,
            ).view(*([1] * (x.ndim - 1)), -1)
        y = x * (1.0 + gain_error) + offset
        y = y + self._noise(y, noise_std, noise_mode, noise_reference, factor=noise_factor, stream=name)
        if saturation > 0.0:
            limit = saturation if saturation_mode == "absolute" else saturation * saturation_reference
            if limit <= 0.0:
                raise ValueError("TIA saturation limit must be positive")
            available = y.new_full((), limit)
            if common_mode is not None:
                pre_saturation = y
                available = (available - common_mode).clamp_min(0.0)
                y = y.sign() * torch.minimum(y.abs(), available)
                self._record_saturation(name, pre_saturation.detach(), available, False)
            else:
                pre_saturation = y
                y = y.clamp(-limit, limit)
                self._record_saturation(name, pre_saturation.detach(), y.new_tensor(limit), False)
        return y

    def dac(
        self,
        x: torch.Tensor,
        name: str,
        bits: int,
    ) -> torch.Tensor:
        return self._dac(
            x,
            name,
            bits,
            unsigned=False,
            value_max=None,
            gain_error=self.config.dac_gain_error,
            offset=self.config.dac_offset,
            noise_std=self.config.dac_noise_std,
            noise_mode=self.config.dac_noise_mode,
            nonlinearity=self.config.dac_nonlinearity,
            nonlinearity_beta=self.config.dac_nonlinearity_beta,
            gain_key="dac_gain_error",
            offset_key="dac_offset",
            noise_factor="dac_noise",
        )

    def tia(
        self,
        x: torch.Tensor,
        name: str = "tia",
        common_mode: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """TIA output stage: continuous analog current/voltage to voltage."""
        cfg = self.config
        normalized_common_mode = None
        if common_mode is not None and cfg.tia_saturation > 0.0:
            normalized_common_mode = (
                common_mode / (cfg.ir_drop_g_max * cfg.ir_drop_v_read)
            ) * cfg.tia_saturation_reference
        return self._analog_signal(
            x,
            cfg.tia_gain_error,
            cfg.tia_offset,
            cfg.tia_noise_std,
            cfg.tia_saturation,
            name,
            cfg.tia_noise_mode,
            cfg.tia_noise_reference,
            cfg.tia_saturation_mode,
            cfg.tia_saturation_reference,
            gain_key="tia_gain_error",
            offset_key="tia_offset",
            channel_gain_mismatch_std=cfg.tia_channel_gain_mismatch_std,
            channel_offset_mismatch_std=cfg.tia_channel_offset_mismatch_std,
            channel_mismatch_name="tia",
            noise_factor="tia_noise",
            common_mode=normalized_common_mode,
        )

    def buffer(
        self,
        x: torch.Tensor,
        name: str = "buffer",
    ) -> torch.Tensor:
        """Analog buffer / sample-and-hold: continuous, keeps polarity."""
        cfg = self.config
        layer = name.split(".", 1)[0]
        return self._analog_signal(
            x,
            cfg.buffer_gain_error,
            0.0,
            cfg.buffer_noise_std,
            0.0,
            name,
            noise_mode="full_scale",
            noise_reference=self._activation_reference(layer),
            gain_key="buffer_gain_error",
            noise_factor="buffer_noise",
        )

    def _activation_reference(self, layer: str) -> float:
        references = self.config.activation_threshold_references
        if references and layer in references:
            return float(references[layer])
        return self.config.activation_threshold_reference

    def _activation_parameters(self, layer: str, reference: float | None = None) -> tuple[float, float, float]:
        cfg = self.config
        gain = self._static_parameter("activation_gain_mismatch", cfg.activation_gain_mismatch, layer)
        reference = self._activation_reference(layer) if reference is None else float(reference)
        threshold = self._static_parameter("activation_threshold", cfg.activation_threshold, layer)
        if cfg.activation_threshold_mode == "full_scale":
            threshold *= reference
        threshold_mismatch = self._static_parameter(
            "activation_threshold_mismatch", cfg.activation_threshold_mismatch, layer
        )
        if cfg.activation_threshold_mismatch_mode == "full_scale":
            threshold_mismatch *= reference
        return gain, threshold, threshold_mismatch

    def activation_mismatch(self, x: torch.Tensor, layer: str) -> torch.Tensor:
        cfg = self.config
        if not self.enabled or (
            cfg.activation_gain_mismatch == 0.0
            and cfg.activation_threshold == 0.0
            and cfg.activation_threshold_mismatch == 0.0
            and not self._has_static_mismatch(
                "activation_gain_mismatch", "activation_threshold", "activation_threshold_mismatch"
            )
        ):
            return x
        gain, threshold, threshold_mismatch = self._activation_parameters(layer)
        return (x - threshold - threshold_mismatch) * (1.0 + gain)

    def output_activation_mismatch(
        self,
        x: torch.Tensor,
        layer: str,
        reference: float | None = None,
    ) -> torch.Tensor:
        """Apply the analog nonnegative output ReLU with fixed channel mismatch."""
        cfg = self.config
        if not self.enabled:
            return torch.relu(x)
        if (
            cfg.activation_gain_mismatch == 0.0
            and cfg.activation_threshold == 0.0
            and cfg.activation_threshold_mismatch == 0.0
            and not self._has_static_mismatch(
                "activation_gain_mismatch", "activation_threshold", "activation_threshold_mismatch"
            )
        ):
            return torch.relu(x)
        gain, threshold, threshold_mismatch = self._activation_parameters(layer, reference)
        threshold_reference = x.new_tensor(
            self._activation_reference(layer) if reference is None else float(reference)
        )
        static_std = cfg.static_mismatch_std or {}
        stream = f"{layer}.output_relu"
        channel_gain = self._static_channel_parameter(
            x,
            "activation_gain_mismatch",
            float(static_std.get("activation_gain_mismatch", 0.0)),
            stream,
        )
        channel_threshold = self._static_channel_parameter(
            x,
            "activation_threshold_mismatch",
            float(static_std.get("activation_threshold_mismatch", 0.0)),
            stream,
        )
        shape = (*([1] * (x.ndim - 1)), x.shape[-1])
        effective_gain = x.new_tensor(1.0 + gain) + channel_gain
        effective_threshold = x.new_tensor(threshold + threshold_mismatch) + channel_threshold * (
            threshold_reference if cfg.activation_threshold_mismatch_mode == "full_scale" else 1.0
        )
        return torch.relu((x - effective_threshold.view(shape)) * effective_gain.view(shape))

    def _ir_drop_block(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        g_plus: torch.Tensor | None = None,
        g_minus: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        with torch.autocast(device_type=weight.device.type, enabled=False):
            x = x.float()
            weight = weight.float()
            safe_w_max = weight_scale.to(device=weight.device, dtype=torch.float32).clamp_min(cfg.epsilon)
            if g_plus is None or g_minus is None:
                span = cfg.g_max - cfg.g_min
                normalized_positive = torch.relu(weight.detach()) / safe_w_max
                normalized_negative = torch.relu(-weight.detach()) / safe_w_max
                g_plus = cfg.g_min + normalized_positive * span
                g_minus = cfg.g_min + normalized_negative * span
            else:
                g_plus = g_plus.float()
                g_minus = g_minus.float()
            conductance = cfg.ir_drop_g_max * (g_plus + g_minus)
            voltage = x * cfg.ir_drop_v_read
            input_line_current = voltage.abs() * conductance.sum(dim=0)
            input_drop = cfg.ir_drop_wire_resistance * (input_line_current.cumsum(dim=1) - input_line_current)
            input_factor = (1.0 - input_drop / cfg.ir_drop_v_read).clamp_min(0.0)
            output_line_current = self._conductance_current(voltage.abs() * input_factor, conductance)
            output_drop = cfg.ir_drop_wire_resistance * (output_line_current.cumsum(dim=1) - output_line_current)
            output_factor = (1.0 - output_drop / cfg.ir_drop_v_read).clamp_min(0.0)
            common_mode_current = output_line_current * output_factor
            differential_output = F.linear(x * input_factor, weight, None) * output_factor
            return differential_output, common_mode_current

    def _weight_scale(self, weight: torch.Tensor) -> torch.Tensor:
        if self.config.per_output_channel:
            return weight.detach().abs().amax(dim=1, keepdim=True)
        return weight.detach().abs().amax().reshape(1, 1)

    @staticmethod
    def _conductance_current(magnitude: torch.Tensor, conductance: torch.Tensor) -> torch.Tensor:
        dtype = torch.promote_types(magnitude.dtype, conductance.dtype)
        with torch.autocast(device_type=magnitude.device.type, enabled=False):
            return F.linear(magnitude.to(dtype), conductance.to(dtype))

    def _requires_common_mode(self) -> bool:
        cfg = self.config
        return cfg.tia_saturation > 0.0 or (
            self._noise_enabled and cfg.read_noise_std > 0.0 and cfg.read_noise_mode == "common_mode_relative"
        )

    def _common_mode_current(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor | None = None,
        g_plus: torch.Tensor | None = None,
        g_minus: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cfg = self.config
        if g_plus is not None and g_minus is not None:
            conductance = cfg.ir_drop_g_max * (g_plus + g_minus)
            return self._conductance_current(x.abs() * cfg.ir_drop_v_read, conductance)
        safe_w_max = (
            (weight_scale if weight_scale is not None else self._weight_scale(weight))
            .to(
                device=weight.device,
                dtype=weight.dtype,
            )
            .clamp_min(cfg.epsilon)
        )
        conductance = cfg.ir_drop_g_max * (
            2.0 * cfg.g_min + (torch.relu(weight) + torch.relu(-weight)) * (cfg.g_max - cfg.g_min) / safe_w_max
        )
        return self._conductance_current(x.abs() * cfg.ir_drop_v_read, conductance)

    def crossbar_mvm(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        return_common_mode: bool = False,
        weight_scale_override: torch.Tensor | None = None,
        conductance_state: ConductanceState | None = None,
        name: str = "mvm",
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        cfg = self.config
        if not self.enabled:
            result = F.linear(x, weight, bias)
            return (result, None) if return_common_mode else result
        effective_weight = conductance_state.effective_weight if conductance_state is not None else weight
        weight_scale = (
            conductance_state.weight_scale
            if conductance_state is not None
            else (weight_scale_override if weight_scale_override is not None else self._weight_scale(weight))
        )
        if not cfg.ir_drop_enabled:
            result = F.linear(x, effective_weight, bias)
            common_mode = (
                self._common_mode_current(
                    x,
                    effective_weight,
                    weight_scale,
                    conductance_state.g_plus if conductance_state is not None else None,
                    conductance_state.g_minus if conductance_state is not None else None,
                )
                if self._requires_common_mode()
                else None
            )
            result = self._apply_read_noise(
                result,
                name,
                common_mode=common_mode,
                x=x,
                weight_scale=weight_scale,
                g_plus=conductance_state.g_plus if conductance_state is not None else None,
                g_minus=conductance_state.g_minus if conductance_state is not None else None,
            )
            return (result, common_mode) if return_common_mode else result
        result = torch.zeros(
            (x.shape[0], effective_weight.shape[0]),
            device=x.device,
            dtype=torch.float32,
        )
        common_mode = torch.zeros_like(result)
        row_tile = cfg.ir_drop_tile_rows
        col_tile = cfg.ir_drop_tile_cols
        for row_start in range(0, effective_weight.shape[0], row_tile):
            row_end = min(row_start + row_tile, effective_weight.shape[0])
            for col_start in range(0, effective_weight.shape[1], col_tile):
                col_end = min(col_start + col_tile, effective_weight.shape[1])
                block_scale = weight_scale if not cfg.per_output_channel else weight_scale[row_start:row_end]
                differential_block, common_block = self._ir_drop_block(
                    x[:, col_start:col_end],
                    effective_weight[row_start:row_end, col_start:col_end],
                    block_scale,
                    conductance_state.g_plus[row_start:row_end, col_start:col_end]
                    if conductance_state is not None
                    else None,
                    conductance_state.g_minus[row_start:row_end, col_start:col_end]
                    if conductance_state is not None
                    else None,
                )
                result[:, row_start:row_end] += differential_block
                common_mode[:, row_start:row_end] += common_block
        if bias is not None:
            result = result + bias.unsqueeze(0)
        result = self._apply_read_noise(
            result,
            name,
            common_mode=common_mode,
            x=x,
            weight_scale=weight_scale,
            g_plus=conductance_state.g_plus if conductance_state is not None else None,
            g_minus=conductance_state.g_minus if conductance_state is not None else None,
        )
        return (result, common_mode) if return_common_mode else result

    def _apply_read_noise(
        self,
        result: torch.Tensor,
        name: str,
        common_mode: torch.Tensor | None = None,
        x: torch.Tensor | None = None,
        weight_scale: torch.Tensor | None = None,
        g_plus: torch.Tensor | None = None,
        g_minus: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cfg = self.config
        if not self._noise_enabled or cfg.read_noise_std <= 0.0:
            return result
        if cfg.read_noise_mode == "common_mode_relative":
            if common_mode is None:
                magnitude = result.abs().detach().clamp_min(cfg.epsilon)
            else:
                span = cfg.g_max - cfg.g_min
                scale = (weight_scale if weight_scale is not None else result.new_ones((1, 1))).to(
                    device=result.device, dtype=result.dtype
                )
                if scale.numel() == result.shape[-1]:
                    scale = scale.reshape(1, -1)
                magnitude = common_mode.detach() * scale / (cfg.ir_drop_g_max * cfg.ir_drop_v_read * span)
                magnitude = magnitude.clamp_min(cfg.epsilon)
        elif cfg.read_noise_mode == "device_current_propagated":
            if x is None or weight_scale is None or g_plus is None or g_minus is None:
                raise ValueError("device_current_propagated read noise requires x, conductance state and weight scale")
            scale = weight_scale.to(device=result.device, dtype=result.dtype)
            if scale.numel() == result.shape[-1]:
                scale = scale.reshape(1, -1)
            variance = (x.square().unsqueeze(1) * (g_plus.square() + g_minus.square()).unsqueeze(0)).sum(dim=2)
            magnitude = (scale / (cfg.g_max - cfg.g_min)) * variance.clamp_min(0.0).sqrt()
            magnitude = magnitude.clamp_min(cfg.epsilon)
        else:
            raise ValueError(f"unsupported read_noise_mode: {cfg.read_noise_mode}")
        return result + self._noise_from_magnitude(result, magnitude, cfg.read_noise_std, "read_noise", name)

    def control_reference(self, maximum: torch.Tensor, name: str) -> torch.Tensor:
        cfg = self.config
        if not self.enabled:
            return maximum
        magnitude = maximum.detach().abs().clamp_min(cfg.epsilon)
        gain = self._static_parameter("control_reference_gain_error", cfg.control_reference_gain_error, name)
        offset = self._static_parameter("control_reference_offset", cfg.control_reference_offset, name)
        target = maximum * (1.0 + gain) + magnitude * offset
        target = target + self._noise_from_magnitude(
            target,
            magnitude,
            cfg.control_reference_noise_std,
            "control_reference_noise",
            name,
        )
        return target + self._noise_from_magnitude(
            target,
            magnitude,
            cfg.control_reference_hold_error,
            "control_reference_hold",
            name,
        )

    def adc_unsigned(
        self,
        x: torch.Tensor,
        name: str,
        bits: int,
        value_max: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        if not self.enabled:
            return x
        if x.numel() == 0:
            return x
        cfg = self.config
        transfer_error = (
            cfg.adc_gain_error != 0.0
            or cfg.adc_channel_gain_mismatch_std != 0.0
            or cfg.adc_offset != 0.0
            or cfg.adc_channel_offset_mismatch_std != 0.0
            or self._has_static_mismatch("adc_gain_error", "adc_offset")
            or (self._noise_enabled and cfg.adc_noise_std > 0.0)
            or cfg.adc_nonlinearity > 0.0
        )
        if bits >= 32 and not transfer_error:
            return x
        observed = x.detach().amax() if x.numel() else x.new_zeros(())
        if value_max is not None:
            value_max = torch.as_tensor(value_max, device=x.device, dtype=x.dtype).clamp_min(cfg.epsilon)
        elif bits >= 32 and not self.observer_enabled and name not in self.activation_ranges:
            value_max = observed.clamp_min(cfg.epsilon)
        else:
            value_max = self._range(name, observed)
        self._record_saturation(name, x.detach(), value_max, True)
        normalized = x / value_max.clamp_min(cfg.epsilon)
        adc_gain_error = self._static_parameter("adc_gain_error", cfg.adc_gain_error, name)
        adc_offset = self._static_parameter("adc_offset", cfg.adc_offset, name)
        channel_shape = (*([1] * (normalized.ndim - 1)), normalized.shape[-1])
        adc_gain_error = adc_gain_error + self._static_channel_parameter(
            normalized,
            "adc_channel_gain_mismatch",
            cfg.adc_channel_gain_mismatch_std,
            name,
        ).view(channel_shape)
        adc_offset = adc_offset + self._static_channel_parameter(
            normalized,
            "adc_channel_offset_mismatch",
            cfg.adc_channel_offset_mismatch_std,
            name,
        ).view(channel_shape)
        normalized = normalized * (1.0 + adc_gain_error) + adc_offset
        normalized = normalized + self._noise(
            normalized,
            cfg.adc_noise_std,
            cfg.adc_noise_mode,
            1.0 if cfg.adc_noise_mode == "full_scale" else None,
            factor="adc_noise",
            stream=name,
        )
        if cfg.adc_nonlinearity > 0.0:
            beta = cfg.adc_nonlinearity_beta
            ideal = 0.5 + 0.5 * torch.tanh(beta * (normalized - 0.5)) / torch.tanh(normalized.new_tensor(beta / 2.0))
            normalized = (1.0 - cfg.adc_nonlinearity) * normalized + cfg.adc_nonlinearity * ideal
        normalized = _ste_clamp(normalized, 0.0, 1.0, cfg.qat_clip_outside_grad)
        if bits >= 32:
            return normalized * value_max
        qmax = (1 << bits) - 1
        scale = value_max.detach().clamp_min(cfg.epsilon) / qmax
        return FakeQuantSTE.apply(normalized * value_max, scale, 0, qmax)

    def adc_signed(
        self,
        x: torch.Tensor,
        name: str,
        bits: int,
        value_max: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Signed final-control ADC with the same transfer errors as the unsigned path."""
        if not self.enabled:
            return x
        if x.numel() == 0:
            return x
        cfg = self.config
        transfer_error = (
            cfg.adc_gain_error != 0.0
            or cfg.adc_channel_gain_mismatch_std != 0.0
            or cfg.adc_offset != 0.0
            or cfg.adc_channel_offset_mismatch_std != 0.0
            or self._has_static_mismatch("adc_gain_error", "adc_offset")
            or (self._noise_enabled and cfg.adc_noise_std > 0.0)
            or cfg.adc_nonlinearity > 0.0
        )
        if bits >= 32 and not transfer_error:
            return x
        return self._adc(
            x,
            name,
            bits,
            cfg.adc_gain_error,
            cfg.adc_offset,
            cfg.adc_noise_std,
            cfg.adc_noise_mode,
            cfg.adc_nonlinearity,
            cfg.adc_nonlinearity_beta,
            "adc_gain_error",
            "adc_offset",
            noise_factor="adc_noise",
            channel_gain_mismatch_std=cfg.adc_channel_gain_mismatch_std,
            channel_offset_mismatch_std=cfg.adc_channel_offset_mismatch_std,
            value_max=value_max,
        )

    def quantize_output_controls(
        self,
        controls: torch.Tensor,
        *,
        name: str,
        bits: int,
        unsigned: bool,
        value_max: torch.Tensor | None = None,
        depth_index: torch.Tensor | None = None,
        depth_count: int | None = None,
    ) -> torch.Tensor:
        if value_max is not None:
            self.activation_ranges[name] = value_max.detach()
        adc = self.adc_unsigned if unsigned else self.adc_signed
        quantized = adc(controls, name, bits, value_max=value_max)
        self.record_controls(
            controls,
            quantized,
            name,
            bits,
            depth_index=depth_index,
            depth_count=depth_count,
            unsigned=unsigned,
        )
        return quantized

    def dual_mismatch(
        self,
        x_positive: torch.Tensor,
        x_negative: torch.Tensor,
        layer: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rail mismatch applied after the dual split, before joint normalization and activation.

        ``x_+ = g_+ * ReLU(x^+ - tau_+)`` and ``x_- = g_- * ReLU(x^- - tau_-)``
        with ``tau_+ = tau0 + Δtau`` and ``tau_- = tau0 - Δtau``, modelling the
        physical threshold/gain difference between the two unipolar rails.
        """
        cfg = self.config
        if not self.enabled or (
            cfg.activation_gain_mismatch == 0.0
            and cfg.activation_threshold == 0.0
            and cfg.activation_threshold_mismatch == 0.0
            and not self._has_static_mismatch(
                "activation_gain_mismatch", "activation_threshold", "activation_threshold_mismatch"
            )
        ):
            return x_positive, x_negative
        mismatch_stream = layer or "global"
        activation_gain_mismatch, base_threshold, threshold_delta = self._activation_parameters(mismatch_stream)
        gain_plus = 1.0 + activation_gain_mismatch
        gain_minus = 1.0 - activation_gain_mismatch
        threshold_plus = base_threshold + threshold_delta
        threshold_minus = base_threshold - threshold_delta
        x_positive = gain_plus * torch.relu(x_positive - threshold_plus)
        x_negative = gain_minus * torch.relu(x_negative - threshold_minus)
        return x_positive, x_negative

    def _range(self, name: str, observed: torch.Tensor) -> torch.Tensor:
        if self.observer_enabled:
            previous = self.activation_ranges.get(name)
            self.activation_ranges[name] = (
                observed if previous is None else torch.maximum(previous.to(observed.device), observed)
            )
        if name not in self.activation_ranges:
            raise RuntimeError(f"observer已冻结，但量化范围{name!r}尚未校准")
        return self.activation_ranges[name].to(device=observed.device, dtype=observed.dtype)

    def _record_saturation(self, name: str, x: torch.Tensor, limit: torch.Tensor, unsigned: bool) -> None:
        if not self._diagnostics_enabled:
            return
        saturated = x > limit if unsigned else x.abs() > limit
        self._diagnostic_quantizer_total[name] = self._diagnostic_quantizer_total.get(name, 0) + x.numel()
        self._diagnostic_quantizer_saturated[name] = self._diagnostic_quantizer_saturated.get(name, 0) + int(
            saturated.sum().item()
        )

    def tensor(
        self,
        x: torch.Tensor,
        name: str,
        bits: int,
        *,
        quantization_range: str | None = None,
    ) -> torch.Tensor:
        if not self.enabled or bits >= 32:
            return x
        observed = x.detach().abs().amax() if x.numel() else x.new_zeros(())
        if quantization_range is None:
            abs_max = self._range(name, observed)
        else:
            abs_max = self.activation_ranges[quantization_range].to(
                device=observed.device,
                dtype=observed.dtype,
            )
        self._record_saturation(name, x.detach(), abs_max, False)
        return _signed_fake_quant(
            x,
            bits,
            abs_max,
            self.config.epsilon,
            self.config.qat_clip_outside_grad,
        )

    def _adc(
        self,
        x: torch.Tensor,
        name: str,
        bits: int,
        gain_error: float,
        offset: float,
        noise_std: float,
        noise_mode: str,
        nonlinearity: float,
        nonlinearity_beta: float,
        gain_key: str | None = None,
        offset_key: str | None = None,
        noise_factor: str = "adc_noise",
        channel_gain_mismatch_std: float = 0.0,
        channel_offset_mismatch_std: float = 0.0,
        value_max: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        if not self.enabled:
            return x
        if x.numel() == 0:
            return x
        cfg = self.config
        observed = x.detach().abs().amax()
        if value_max is not None:
            value_max = torch.as_tensor(value_max, device=x.device, dtype=x.dtype).clamp_min(cfg.epsilon)
        elif bits >= 32 and not self.observer_enabled and name not in self.activation_ranges:
            value_max = observed.clamp_min(cfg.epsilon)
        else:
            value_max = self._range(name, observed).clamp_min(cfg.epsilon)
        self._record_saturation(name, x.detach(), value_max, False)
        normalized = x / value_max
        gain_error = self._static_parameter(gain_key, gain_error, name)
        offset = self._static_parameter(offset_key, offset, name)
        channel_shape = (*([1] * (normalized.ndim - 1)), normalized.shape[-1])
        gain_error = gain_error + self._static_channel_parameter(
            normalized,
            "adc_channel_gain_mismatch",
            channel_gain_mismatch_std,
            name,
        ).view(channel_shape)
        offset = offset + self._static_channel_parameter(
            normalized,
            "adc_channel_offset_mismatch",
            channel_offset_mismatch_std,
            name,
        ).view(channel_shape)
        normalized = normalized * (1.0 + gain_error) + offset
        if self._noise_enabled and noise_std > 0.0:
            reference = 1.0 if noise_mode == "full_scale" else None
            normalized = normalized + self._noise(
                normalized,
                noise_std,
                noise_mode,
                reference,
                factor=noise_factor,
                stream=name,
            )
        if nonlinearity > 0.0:
            ideal = torch.tanh(nonlinearity_beta * normalized) / torch.tanh(normalized.new_tensor(nonlinearity_beta))
            normalized = (1.0 - nonlinearity) * normalized + nonlinearity * ideal
        qmax = (1 << (bits - 1)) - 1
        normalized = _ste_clamp(normalized, -1.0, 1.0, cfg.qat_clip_outside_grad)
        scale = value_max.detach() / qmax
        return FakeQuantSTE.apply(normalized * value_max, scale, -qmax, qmax) if bits < 32 else normalized * value_max

    def _dac(
        self,
        x: torch.Tensor,
        name: str,
        bits: int,
        unsigned: bool,
        value_max: torch.Tensor | None,
        gain_error: float,
        offset: float,
        noise_std: float,
        noise_mode: str,
        nonlinearity: float,
        nonlinearity_beta: float,
        gain_key: str | None = None,
        offset_key: str | None = None,
        noise_factor: str = "dac_noise",
    ) -> torch.Tensor:
        if not self.enabled:
            return x
        if x.numel() == 0:
            return x
        cfg = self.config
        observed = x.detach().amax() if unsigned else x.detach().abs().amax()
        if value_max is None:
            value_max = self._range(name, observed)
        value_max = value_max.to(device=x.device, dtype=x.dtype).clamp_min(cfg.epsilon)
        self._record_saturation(name, x.detach(), value_max, unsigned)
        normalized = x / value_max
        if unsigned:
            qmax = (1 << bits) - 1
            normalized = _ste_clamp(normalized, 0.0, 1.0, cfg.qat_clip_outside_grad)
            scale = value_max.detach() / qmax
            quantized = (
                FakeQuantSTE.apply(normalized * value_max, scale, 0, qmax) if bits < 32 else normalized * value_max
            )
        else:
            qmax = (1 << (bits - 1)) - 1
            normalized = _ste_clamp(normalized, -1.0, 1.0, cfg.qat_clip_outside_grad)
            scale = value_max.detach() / qmax
            quantized = (
                FakeQuantSTE.apply(normalized * value_max, scale, -qmax, qmax) if bits < 32 else normalized * value_max
            )
        normalized = quantized / value_max
        gain_error = self._static_parameter(gain_key, gain_error, name)
        offset = self._static_parameter(offset_key, offset, name)
        normalized = normalized * (1.0 + gain_error) + offset
        if self._noise_enabled and noise_std > 0.0:
            reference = 1.0 if noise_mode == "full_scale" else None
            normalized = normalized + self._noise(
                normalized,
                noise_std,
                noise_mode,
                reference,
                factor=noise_factor,
                stream=name,
            )
        if nonlinearity > 0.0:
            if unsigned:
                ideal = 0.5 + 0.5 * torch.tanh(nonlinearity_beta * (normalized - 0.5)) / torch.tanh(
                    normalized.new_tensor(nonlinearity_beta / 2.0)
                )
            else:
                ideal = torch.tanh(nonlinearity_beta * normalized) / torch.tanh(
                    normalized.new_tensor(nonlinearity_beta)
                )
            normalized = (1.0 - nonlinearity) * normalized + nonlinearity * ideal
        if unsigned:
            normalized = normalized.clamp_min(0.0)
        return normalized * value_max

    def digital_adc(self, x: torch.Tensor, name: str, bits: int) -> torch.Tensor:
        """Signed hidden ADC using the shared ADC device model."""
        return self.adc_signed(x, f"{name}.adc", bits)

    def digital_dac(
        self,
        x: torch.Tensor,
        name: str,
        bits: int,
        unsigned: bool,
        value_max: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """DAC after digital activation processing; dual mode passes magnitudes only."""
        cfg = self.config
        return self._dac(
            x,
            f"{name}.dac",
            bits,
            unsigned,
            value_max,
            cfg.dac_gain_error,
            cfg.dac_offset,
            cfg.dac_noise_std,
            cfg.dac_noise_mode,
            cfg.dac_nonlinearity,
            cfg.dac_nonlinearity_beta,
            "dac_gain_error",
            "dac_offset",
            "dac_noise",
        )

    @staticmethod
    def route_dual_rails(
        magnitude: torch.Tensor,
        positive_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Route one post-DAC magnitude to exactly one of the two physical rails."""
        if magnitude.ndim != positive_mask.ndim:
            raise ValueError("magnitude and positive_mask must have the same rank")
        positive = magnitude * positive_mask.to(dtype=magnitude.dtype)
        negative = magnitude * (~positive_mask).to(dtype=magnitude.dtype)
        return torch.cat((positive, negative), dim=1)

    def map_weight(self, weight: torch.Tensor, name: str) -> ConductanceState:
        if not self.enabled:
            zeros = torch.zeros_like(weight)
            return ConductanceState(weight, zeros, zeros, weight.new_ones((1, 1)))
        cfg = self.config
        reduce_dim = 1 if cfg.per_output_channel else None
        if reduce_dim is None:
            w_max = weight.detach().abs().amax().reshape(1, 1)
        else:
            w_max = weight.detach().abs().amax(dim=reduce_dim, keepdim=True)
        safe_w_max = w_max.clamp_min(cfg.epsilon)
        span = cfg.g_max - cfg.g_min
        g_plus = cfg.g_min + torch.relu(weight) * span / safe_w_max
        g_minus = cfg.g_min + torch.relu(-weight) * span / safe_w_max
        if cfg.weight_bits >= 32:
            q_plus, q_minus = g_plus, g_minus
        else:
            states = 1 << cfg.weight_bits
            g_scale = weight.new_tensor(span / (states - 1))
            q_plus = FakeQuantSTE.apply(g_plus - cfg.g_min, g_scale, 0, states - 1) + cfg.g_min
            q_minus = FakeQuantSTE.apply(g_minus - cfg.g_min, g_scale, 0, states - 1) + cfg.g_min
        # Programming error: applied to the conductance states (bounded stay in-range).
        write_std = cfg.write_error_std
        if self._noise_enabled and write_std > 0.0:
            write_reference = (cfg.g_max - cfg.g_min) if cfg.write_noise_mode == "full_scale" else None
            q_plus = q_plus + self._write_noise(
                q_plus,
                write_std,
                f"{name}.plus",
                cfg.write_noise_mode,
                write_reference,
            )
            q_minus = q_minus + self._write_noise(
                q_minus,
                write_std,
                f"{name}.minus",
                cfg.write_noise_mode,
                write_reference,
            )
            q_plus = q_plus.clamp(cfg.g_min, cfg.g_max)
            q_minus = q_minus.clamp(cfg.g_min, cfg.g_max)
        q_plus = self._apply_static_device_state(q_plus, f"{name}.plus")
        q_minus = self._apply_static_device_state(q_minus, f"{name}.minus")
        restored = safe_w_max * (q_plus - q_minus) / span
        restored = torch.where(w_max > cfg.epsilon, restored, torch.zeros_like(restored))
        self.weight_scales[name] = safe_w_max.detach().flatten()
        return ConductanceState(restored, q_plus, q_minus, safe_w_max)

    def parameter_diagnostics(
        self,
        weight: torch.Tensor,
        name: str,
        implementation: str,
        bias: torch.Tensor | None = None,
        quantization_range: str | None = None,
    ) -> dict[str, float | None]:
        weight = weight.detach()
        previous_noise = self._noise_enabled
        self._noise_enabled = False
        try:
            with torch.no_grad():
                if implementation == "array":
                    raw_bias = weight[:, -1]
                    effective_bias = self.bias(raw_bias, name, "array")
                    mapped_weight = torch.cat((weight[:, :-1], effective_bias.unsqueeze(1)), dim=1)
                else:
                    raw_bias = bias.detach() if bias is not None else weight.new_empty(0)
                    effective_bias = (
                        self.bias(raw_bias, name, implementation, quantization_range=quantization_range)
                        if raw_bias.numel()
                        else raw_bias
                    )
                    mapped_weight = weight
                mapped = self.map_weight(mapped_weight, name)
        finally:
            self._noise_enabled = previous_noise
        if implementation == "array":
            core = weight[:, :-1]
            mapped_core = mapped.effective_weight[:, :-1]
        else:
            core = weight
            mapped_core = mapped.effective_weight
        bias_range = self.activation_ranges.get(quantization_range or bias_range_name(name))
        endpoint = (
            bias_range
            if bias_range is not None
            else effective_bias.abs().amax().clamp_min(self.config.epsilon)
        ).to(device=weight.device, dtype=weight.dtype)
        error = (mapped_core - core).abs()
        endpoint_tolerance = endpoint * 1.0e-6
        positive_endpoint = (
            ((effective_bias >= endpoint - endpoint_tolerance) & (effective_bias > 0)).float().mean()
            if effective_bias.numel()
            else weight.new_zeros(())
        )
        negative_endpoint = (
            ((effective_bias <= -endpoint + endpoint_tolerance) & (effective_bias < 0)).float().mean()
            if effective_bias.numel()
            else weight.new_zeros(())
        )
        return {
            "raw_bias_max": float(raw_bias.abs().amax().detach().cpu()) if raw_bias.numel() else 0.0,
            "effective_bias_max": float(effective_bias.abs().amax().detach().cpu()) if effective_bias.numel() else 0.0,
            "bias_range": float(bias_range.detach().cpu()) if bias_range is not None else None,
            "bias_endpoint_positive": float(positive_endpoint.detach().cpu()),
            "bias_endpoint_negative": float(negative_endpoint.detach().cpu()),
            "weight_scale": float(mapped.weight_scale.max().detach().cpu()),
            "core_error_rms": float(error.square().mean().sqrt().detach().cpu()) if error.numel() else 0.0,
            "core_error_max": float(error.max().detach().cpu()) if error.numel() else 0.0,
        }

    def bias(
        self,
        bias: torch.Tensor | None,
        name: str,
        implementation: str,
        *,
        quantization_range: str | None = None,
    ) -> torch.Tensor | None:
        if bias is None or not self.enabled:
            return bias
        cfg = self.config
        bits = cfg.weight_bits if implementation == "array" else cfg.bias_bits
        value = self.tensor(
            bias,
            f"{name}.bias",
            bits,
            quantization_range=quantization_range,
        )
        if implementation != "analog":
            return value
        has_error = any(
            float(getattr(cfg, field)) != 0.0
            for field in (
                "bias_programming_gain_error",
                "bias_programming_offset",
                "bias_noise_std",
                "bias_channel_gain_mismatch_std",
            )
        ) or self._has_static_mismatch("bias_programming_gain_error", "bias_programming_offset")
        if not has_error:
            return value
        gain = self._static_parameter("bias_programming_gain_error", cfg.bias_programming_gain_error, name)
        offset = self._static_parameter("bias_programming_offset", cfg.bias_programming_offset, name)
        channel_gain = self._static_channel_parameter(
            value.unsqueeze(0), "bias.channel_gain", cfg.bias_channel_gain_mismatch_std, name
        ).view_as(value)
        value = value * (1.0 + gain + channel_gain) + offset
        return value + self._noise_from_magnitude(
            value,
            value.new_full(value.shape, cfg.bias_noise_reference),
            cfg.bias_noise_std,
            "bias_noise",
            name,
        )

    def begin_diagnostics(self) -> None:
        self._diagnostics_enabled = True
        self._diagnostic_weight_sums.clear()
        self._diagnostic_depth_fractions.clear()
        self._diagnostic_control_groups = 0
        self._diagnostic_control_pixels = 0
        self._diagnostic_control_codes = 0
        self._diagnostic_all_zero_groups = 0
        self._diagnostic_all_zero_pixels = 0
        self._diagnostic_zero_codes = 0
        self._diagnostic_saturated_control_codes = 0
        self._diagnostic_endpoint_control_codes = 0
        self._diagnostic_used_control_codes.clear()
        self._diagnostic_quantizer_total.clear()
        self._diagnostic_quantizer_saturated.clear()
        self._diagnostic_control_values.clear()
        self._diagnostic_control_codes_by_depth.clear()
        self._diagnostic_control_depths.clear()

    def record_controls(
        self,
        inputs: torch.Tensor,
        controls: torch.Tensor,
        name: str,
        bits: int,
        depth_index: torch.Tensor | None = None,
        depth_count: int | None = None,
        unsigned: bool = True,
    ) -> None:
        if not self._diagnostics_enabled:
            return
        detached = controls.detach()
        groups = detached.reshape(-1, detached.shape[-1])
        pixels = detached.reshape(detached.shape[0], -1)
        self._diagnostic_control_groups += groups.shape[0]
        self._diagnostic_control_pixels += pixels.shape[0]
        self._diagnostic_control_codes += detached.numel()
        zero_epsilon = max(float(self.config.epsilon), 1.0e-12)
        group_max = groups.amax(dim=-1) if unsigned else groups.abs().amax(dim=-1)
        pixel_max = pixels.amax(dim=-1) if unsigned else pixels.abs().amax(dim=-1)
        self._diagnostic_all_zero_groups += int((group_max <= zero_epsilon).sum().item())
        self._diagnostic_all_zero_pixels += int((pixel_max <= zero_epsilon).sum().item())
        if bits < 32 and name in self.activation_ranges:
            value_max = self.activation_ranges[name].to(device=detached.device, dtype=detached.dtype)
            qmax = (1 << bits) - 1 if unsigned else (1 << (bits - 1)) - 1
            scale = value_max.clamp_min(self.config.epsilon) / qmax
            if unsigned:
                codes = torch.clamp(torch.round(detached / scale), 0, qmax).long()
                zero_codes = codes <= 0
                endpoints = codes == qmax
                saturated = inputs.detach() > value_max
            else:
                codes = torch.clamp(torch.round(detached / scale), -qmax, qmax).long()
                zero_codes = codes == 0
                endpoints = codes.abs() == qmax
                saturated = inputs.detach().abs() > value_max
            self._diagnostic_zero_codes += int(zero_codes.sum().item())
            self._diagnostic_used_control_codes.update(int(value) for value in torch.unique(codes).cpu().tolist())
            self._diagnostic_saturated_control_codes += int(saturated.sum().item())
            self._diagnostic_endpoint_control_codes += int(endpoints.sum().item())
            raw_values = inputs.detach().float().reshape(-1)
            sampled_codes = codes.reshape(-1)
            if depth_index is not None and depth_count is not None:
                depth_values = (
                    depth_index.detach()
                    .float()
                    .view(-1, *([1] * (detached.ndim - 1)))
                    .expand_as(detached)
                    .reshape(-1)
                    / max(int(depth_count) - 1, 1)
                )
            else:
                depth_values = torch.full_like(raw_values, 0.5)
            sample_count = min(32, raw_values.numel())
            if raw_values.numel() > sample_count:
                select = torch.linspace(0, raw_values.numel() - 1, sample_count, device=raw_values.device).long()
                raw_values = raw_values[select]
                sampled_codes = sampled_codes[select]
                depth_values = depth_values[select]
            self._diagnostic_control_values.append(raw_values.cpu())
            self._diagnostic_control_codes_by_depth.append(sampled_codes.cpu())
            self._diagnostic_control_depths.append(depth_values.cpu())

    def record_weight_sums(
        self,
        weight_sums: torch.Tensor,
        depth_index: torch.Tensor | None = None,
        depth_count: int | None = None,
    ) -> None:
        if not self._diagnostics_enabled:
            return
        remaining = 100000 - sum(value.numel() for value in self._diagnostic_weight_sums)
        if remaining > 0:
            values = weight_sums.detach().float().flatten().cpu()
            depths = None
            if depth_index is not None and depth_count is not None:
                depths = depth_index.detach().float().flatten().cpu() / max(depth_count - 1, 1)
            if values.numel() > remaining:
                indices = torch.linspace(0, values.numel() - 1, remaining).long()
                values = values[indices]
                if depths is not None:
                    depths = depths[indices]
            self._diagnostic_weight_sums.append(values)
            if depths is not None:
                self._diagnostic_depth_fractions.append(depths)

    @staticmethod
    def _distribution(values: torch.Tensor) -> dict[str, float]:
        q = torch.quantile(values.float(), torch.tensor([0.001, 0.01, 0.5, 0.99, 0.999]))
        return {
            "min": float(values.min()),
            "p0_1": float(q[0]),
            "p1": float(q[1]),
            "median": float(q[2]),
            "p99": float(q[3]),
            "p99_9": float(q[4]),
            "max": float(values.max()),
            "mean": float(values.float().mean()),
            "std": float(values.float().std(unbiased=False)),
        }

    @staticmethod
    def _control_percentiles(values: torch.Tensor) -> dict[str, float]:
        q = torch.quantile(values.float(), torch.tensor([0.5, 0.9, 0.99, 0.999]))
        return {
            "p50": float(q[0]),
            "p90": float(q[1]),
            "p99": float(q[2]),
            "p99_9": float(q[3]),
            "max": float(values.max()),
        }

    def end_diagnostics(self, public_gain_max: float) -> dict[str, Any]:
        self._diagnostics_enabled = False
        groups = max(self._diagnostic_control_groups, 1)
        pixels = max(self._diagnostic_control_pixels, 1)
        codes = max(self._diagnostic_control_codes, 1)
        result: dict[str, Any] = {
            "all_zero_control_group_fraction": self._diagnostic_all_zero_groups / groups,
            "all_zero_control_pixel_fraction": self._diagnostic_all_zero_pixels / pixels,
            "zero_control_code_fraction": self._diagnostic_zero_codes / codes,
            "saturated_control_code_fraction": self._diagnostic_saturated_control_codes / codes,
            "endpoint_control_code_fraction": self._diagnostic_endpoint_control_codes / codes,
            "used_control_code_count": len(self._diagnostic_used_control_codes),
            "used_control_codes": sorted(self._diagnostic_used_control_codes),
            "quantizer_saturation_fraction": {
                name: self._diagnostic_quantizer_saturated.get(name, 0) / max(total, 1)
                for name, total in sorted(self._diagnostic_quantizer_total.items())
            },
        }
        output_range_name = (
            self.output_preactivation_range_name
            if self.config.inter_layer == "digital"
            else self.output_control_range_name
        )
        observer_range = self.activation_ranges.get(output_range_name) if output_range_name is not None else None
        observer_limit = float(observer_range.detach().cpu()) if observer_range is not None else None
        if observer_range is not None and self.config.control_bits < 32:
            control_qmax = (
                (1 << (self.config.control_bits - 1)) - 1
                if self.config.inter_layer == "digital"
                else (1 << self.config.control_bits) - 1
            )
            result["control_observer_range"] = observer_limit
            result["control_observer_lsb"] = float(
                observer_range.detach().cpu() / control_qmax
            )
        if self._diagnostic_control_values:
            raw_values = torch.cat(self._diagnostic_control_values)
            result["control_distribution"] = self._control_percentiles(raw_values)
            depth_codes = torch.cat(self._diagnostic_control_codes_by_depth).long()
            depth_values = torch.cat(self._diagnostic_control_depths)
            depth_bands: dict[str, Any] = {}
            for band_name, low, high in (
                ("shallow", 0.0, 1.0 / 3.0),
                ("middle", 1.0 / 3.0, 2.0 / 3.0),
                ("deep", 2.0 / 3.0, 1.000001),
            ):
                mask = (depth_values >= low) & (depth_values < high)
                if not mask.any():
                    continue
                band_raw = raw_values[mask]
                code_values = depth_codes[mask]
                depth_bands[band_name] = {
                    "control_distribution": self._control_percentiles(band_raw),
                    "used_code_count": int(torch.unique(code_values).numel()),
                    "used_codes": sorted(int(value) for value in torch.unique(code_values).tolist()),
                    "zero_code_fraction": float((code_values == 0).float().mean()),
                    "saturated_code_fraction": float(
                        ((band_raw.abs() if self.config.inter_layer == "digital" else band_raw) > observer_limit)
                        .float()
                        .mean()
                    ) if observer_limit is not None else 0.0,
                    "endpoint_code_fraction": float(
                        (
                            code_values.abs() == (
                                (1 << (self.config.control_bits - 1)) - 1
                                if self.config.inter_layer == "digital"
                                else (1 << self.config.control_bits) - 1
                            )
                        )
                        .float()
                        .mean()
                    ),
                }
            result["control_depth_bands"] = depth_bands
        self._diagnostic_control_values.clear()
        self._diagnostic_control_codes_by_depth.clear()
        self._diagnostic_control_depths.clear()
        if self._diagnostic_weight_sums:
            sums = torch.cat(self._diagnostic_weight_sums)
            safe_sums = sums.clamp_min(self.config.epsilon)
            gains = safe_sums.reciprocal()
            clipped_gains = gains.clamp_max(public_gain_max)
            ideal_sums = sums * gains
            clipped_sums = sums * clipped_gains
            result.update(
                {
                    "s": self._distribution(sums),
                    "inverse_s": self._distribution(gains),
                    "ideal_sum_w": self._distribution(ideal_sums),
                    "clipped_sum_w": self._distribution(clipped_sums),
                    "public_gain_saturation_fraction": float((gains > public_gain_max).float().mean()),
                }
            )
            if self._diagnostic_depth_fractions:
                depths = torch.cat(self._diagnostic_depth_fractions)
                depth_stats = {}
                for label, selected in (
                    ("shallow", depths < 1.0 / 3.0),
                    ("middle", (depths >= 1.0 / 3.0) & (depths < 2.0 / 3.0)),
                    ("deep", depths >= 2.0 / 3.0),
                ):
                    if selected.any():
                        depth_stats[label] = self._distribution(gains[selected])
                result["inverse_s_by_depth"] = depth_stats
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "runtime": {
                "profile_name": self.profile_name,
                "stage_name": self.stage_name,
                "observer_enabled": self.observer_enabled,
                "noise_enabled": self._noise_enabled,
            },
            "activation_ranges": {key: float(value.cpu()) for key, value in self.activation_ranges.items()},
            "weight_scales": {key: value.cpu().tolist() for key, value in self.weight_scales.items()},
        }

    def checkpoint_state_dict(self) -> dict[str, Any]:
        state = self.state_dict()
        state["realization_state"] = self.realization_state()
        return state

    def load_state_dict(self, state: dict[str, Any] | None, restore_config: bool = False) -> None:
        if not state:
            return
        runtime = state.get("runtime", {}) or {}
        if runtime.get("profile_name") is not None:
            self.profile_name = str(runtime["profile_name"])
        self.stage_name = runtime.get("stage_name")
        if restore_config and state.get("config"):
            config = dict(state["config"])
            self.config = QATConfig(**config)
            self.reset_noise_counter()
        if "observer_enabled" in runtime:
            self.observer_enabled = bool(runtime["observer_enabled"])
        if "noise_enabled" in runtime:
            self._noise_enabled = bool(runtime["noise_enabled"])
        self.activation_ranges = {
            str(k): torch.tensor(float(v)) for k, v in state.get("activation_ranges", {}).items()
        }
        self.weight_scales = {
            str(k): torch.tensor([float(value) for value in values])
            for k, values in state.get("weight_scales", {}).items()
        }
        self.restore_realization_state(state.get("realization_state"))


class MemristorLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        hardware: HardwareQAT,
        name: str,
        bias_implementation: str,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        object.__setattr__(self, "hardware", hardware)
        self.hardware_name = name
        self.bias_implementation = bias_implementation

    def _weight_and_bias(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        hw = self.hardware
        if self.bias_implementation == "array":
            bias = hw.bias(weight[:, -1], self.hardware_name, "array")
            return torch.cat((weight[:, :-1], bias.unsqueeze(1)), dim=1), None
        bias = (
            hw.bias(self.bias, self.hardware_name, self.bias_implementation)
            if self.bias_implementation in {"none", "analog"}
            else None
        )
        return weight, bias

    def _input_signal(self, x: torch.Tensor, input_bits: int) -> torch.Tensor:
        hw = self.hardware
        if self.bias_implementation != "array":
            return hw.dac(x, input_dac_name(self.hardware_name), input_bits)
        signal = hw.dac(x[..., :-1], input_dac_name(self.hardware_name), input_bits)
        return torch.cat((signal, self._array_bias_input(x[..., -1:], input_bits)), dim=-1)

    def _array_bias_input(self, x: torch.Tensor, input_bits: int) -> torch.Tensor:
        return self.hardware.digital_dac(
            x,
            bias_dac_name(self.hardware_name),
            input_bits,
            unsigned=True,
            value_max=x.new_tensor(1.0),
        )

    def _forward_input(self, x: torch.Tensor, input_bits: int, quantize_input: bool) -> torch.Tensor:
        if quantize_input or self.bias_implementation != "array":
            return self._input_signal(x, input_bits) if quantize_input else x
        return torch.cat((x[..., :-1], self._array_bias_input(x[..., -1:], input_bits)), dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        effective_weight: torch.Tensor | None = None,
        input_bits: int | None = None,
        quantize_input: bool = True,
    ) -> torch.Tensor:
        hw = self.hardware
        weight = self.weight if effective_weight is None else effective_weight
        x_hat = self._forward_input(x, input_bits or hw.config.input_bits, quantize_input)
        weight, bias_hat = self._weight_and_bias(weight)
        weight_state = hw.map_weight(weight, self.hardware_name)
        output, common_mode = hw.crossbar_mvm(
            x_hat,
            weight_state.effective_weight,
            bias_hat,
            return_common_mode=True,
            conductance_state=weight_state,
            name=self.hardware_name,
        )
        return hw.tia(output, f"{self.hardware_name}.tia", common_mode=common_mode)

    def forward_tiled(
        self,
        x: torch.Tensor,
        tile: int,
        effective_weight: torch.Tensor,
        input_bits: int | None = None,
        quantize_input: bool = True,
    ) -> torch.Tensor:
        hw = self.hardware
        x_hat = self._forward_input(x, input_bits or hw.config.input_bits, quantize_input)
        effective_weight, bias_hat = self._weight_and_bias(effective_weight)
        weight_state = hw.map_weight(effective_weight, self.hardware_name)
        if hw.config.ir_drop_enabled:
            result, common_mode = hw.crossbar_mvm(
                x_hat,
                weight_state.effective_weight,
                bias_hat,
                return_common_mode=True,
                conductance_state=weight_state,
                name=self.hardware_name,
            )
            return hw.tia(result, f"{self.hardware_name}.tia", common_mode=common_mode)
        weight_hat = weight_state.effective_weight
        weight_scale = weight_state.weight_scale
        common_mode = (
            hw._common_mode_current(
                x_hat,
                weight_hat,
                weight_scale,
                weight_state.g_plus,
                weight_state.g_minus,
            )
            if hw._requires_common_mode()
            else None
        )
        result = (
            bias_hat.unsqueeze(0).expand(x.shape[0], -1)
            if bias_hat is not None
            else x.new_zeros((x.shape[0], self.out_features))
        )
        for start in range(0, x.shape[1], tile):
            end = min(start + tile, x.shape[1])
            result = result + F.linear(x_hat[:, start:end], weight_hat[:, start:end], None)
        result = hw._apply_read_noise(
            result,
            self.hardware_name,
            common_mode=common_mode,
            x=x_hat,
            weight_scale=weight_scale,
            g_plus=weight_state.g_plus,
            g_minus=weight_state.g_minus,
        )
        return hw.tia(result, f"{self.hardware_name}.tia", common_mode=common_mode)

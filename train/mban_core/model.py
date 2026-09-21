from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hardware import BIAS_IMPLEMENTATIONS, QATConfig
from .hardware_backend import HardwareQAT, MemristorLinear
from .naming import layer_name, output_controls_name, output_preactivation_name

_runtime_args = None


def set_runtime_args(runtime_args) -> None:
    global _runtime_args
    _runtime_args = runtime_args


class SymmetricPWL(nn.Module):
    def __init__(self, knee: float, tail_slope: float) -> None:
        super().__init__()
        self.knee = float(knee)
        self.tail_slope = float(tail_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        excess = torch.relu(x.abs() - self.knee)
        return x - (1.0 - self.tail_slope) * torch.sign(x) * excess


class TanhBeta(nn.Module):
    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = float(beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.beta * x)


class BatchRenorm1d(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1.0e-5,
        momentum: float = 0.1,
        rmax: float = 3.0,
        dmax: float = 5.0,
    ) -> None:
        super().__init__()
        if not 0.0 < momentum <= 1.0 or eps <= 0.0 or rmax < 1.0 or dmax < 0.0:
            raise ValueError("Batch Renormalization 参数无效")
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.rmax = float(rmax)
        self.dmax = float(dmax)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        running_mean = self.running_mean.detach()
        running_var = self.running_var.detach()
        if self.training and x_fp32.shape[0] > 1:
            batch_mean = x_fp32.mean(dim=0)
            batch_var = x_fp32.var(dim=0, unbiased=False)
            batch_std = torch.sqrt(batch_var + self.eps)
            running_std = torch.sqrt(running_var + self.eps)
            correction_r = (batch_std / running_std).clamp(1.0 / self.rmax, self.rmax).detach()
            correction_d = (
                (batch_mean - running_mean) / running_std
            ).clamp(-self.dmax, self.dmax).detach()
            normalized = (x_fp32 - batch_mean) / batch_std * correction_r + correction_d
            with torch.no_grad():
                self.running_mean.lerp_(batch_mean, self.momentum)
                self.running_var.lerp_(x_fp32.var(dim=0, unbiased=True), self.momentum)
        else:
            normalized = (x_fp32 - running_mean) / torch.sqrt(running_var + self.eps)
        return (normalized * self.weight.float() + self.bias.float()).to(x.dtype)


def build_activation(spec: dict) -> nn.Module:
    """按逐层 spec 构建激活。"""
    spec_type = spec["type"]
    if spec_type == "relu":
        return nn.ReLU()
    if spec_type == "pwl":
        return SymmetricPWL(float(spec.get("knee", 1.0)), float(spec.get("tail_slope", 0.2)))
    if spec_type == "tanh":
        return TanhBeta(float(spec.get("beta", 1.0)))
    raise ValueError(f"未知激活函数: {spec_type}")


def split_dual_branches(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dual-branch split: x -> (x^+, x^-) with x^+ = ReLU(x), x^- = ReLU(-x)."""
    return torch.relu(x), torch.relu(-x)


def apply_hidden_normalization(x: torch.Tensor, normalization: str) -> torch.Tensor:
    if normalization not in {"l1", "l2"}:
        return x
    x_dtype = x.dtype
    x_fp32 = x.float()
    if normalization == "l1":
        denominator = x_fp32.abs().sum(dim=1, keepdim=True)
    else:
        denominator = torch.linalg.vector_norm(x_fp32, ord=2, dim=1, keepdim=True)
    return (x_fp32 / denominator.clamp_min(1.0e-8)).to(x_dtype)


def standardize_weight(weight: torch.Tensor, eps: float) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError("Weight Standardization 只支持二维线性层权重")
    weight_fp32 = weight.float()
    mean = weight_fp32.mean(dim=1, keepdim=True)
    variance = (weight_fp32 - mean).square().mean(dim=1, keepdim=True)
    return ((weight_fp32 - mean) * torch.rsqrt(variance + float(eps))).to(weight.dtype)


def fold_standardizer(
    weight: torch.Tensor,
    bias: torch.Tensor,
    normalization: str,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float,
    affine_weight: torch.Tensor | None = None,
    affine_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = torch.rsqrt(running_var.detach() + float(eps))
    if normalization == "batch_renorm":
        if affine_weight is None or affine_bias is None:
            raise ValueError("Batch Renorm 折叠需要 affine weight/bias")
        scale = scale * affine_weight.detach()
        bias = (bias - running_mean.detach()) * scale + affine_bias.detach()
    elif normalization == "running_zscore":
        bias = (bias - running_mean.detach()) * scale
    else:
        raise ValueError(f"不支持折叠的归一化: {normalization}")
    return weight * scale[:, None], bias


def _validate_layer_names(
    value: tuple[str, ...] | list[str], field_name: str, valid_layers: set[str]
) -> frozenset[str]:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{field_name} 必须是层名序列")
    if any(not isinstance(layer, str) or not layer.strip() for layer in value):
        raise ValueError(f"{field_name} 必须只包含非空字符串层名")
    layers = frozenset(value)
    if len(layers) != len(value):
        raise ValueError(f"{field_name} 不能包含重复层名")
    unknown = layers - valid_layers
    if unknown:
        raise ValueError(f"{field_name} 只能从 {sorted(valid_layers)} 选择")
    return layers


class MBAN(nn.Module):
    def __init__(
        self,
        num_channels: int,
        dropout: float = 0.2,
        hidden_width: int = 0,
        activation: dict | None = None,
        branch_mode: str = "single",
        hidden_layers: int = 3,
        output_controls: int = 0,
        input_tile_features: int = 0,
        centering: str = "none",
        centering_layers: tuple[str, ...] | list[str] = (),
        normalization: str = "none",
        normalization_layers: tuple[str, ...] | list[str] = (),
        bias_layers: tuple[str, ...] | list[str] | str | None = None,
        bias_implementation: str = "ordinary",
        weight_transform: str = "none",
        weight_transform_layers: tuple[str, ...] | list[str] = (),
        weight_transform_epsilon: float = 1.0e-5,
        running_stat_momentum: float = 0.1,
        running_stat_epsilon: float = 1.0e-5,
        batch_renorm_rmax: float = 3.0,
        batch_renorm_dmax: float = 5.0,
        hardware_config: QATConfig | None = None,
        hardware_enabled: bool | None = None,
    ) -> None:
        super().__init__()
        if _runtime_args is None:
            raise RuntimeError("MBAN 运行参数尚未初始化，请先调用 set_runtime_args()")
        if centering not in {"none", "sample_mean"}:
            raise ValueError("centering 必须是 none 或 sample_mean")
        if centering == "sample_mean" and normalization in {"running_zscore", "batch_renorm"}:
            raise ValueError("sample_mean 中心化不能与运行统计归一化同时启用")
        channel_count = int(num_channels)
        if channel_count < 1:
            raise ValueError("num_channels 必须为正整数")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout 必须在 [0, 1) 内")
        if hidden_width < 0:
            raise ValueError("hidden_width 必须大于等于 0")
        hidden_dim = max(1, channel_count // 4) if hidden_width == 0 else int(hidden_width)
        if hidden_dim < 1:
            raise ValueError("hidden_width 必须为正整数或 0")
        if hidden_layers < 1:
            raise ValueError("hidden_layers 必须为正整数")
        if branch_mode not in {"single", "dual"}:
            raise ValueError("branch_mode 必须是 single 或 dual")
        if weight_transform not in {"none", "ws"}:
            raise ValueError("weight_transform 必须是 none 或 ws")
        if weight_transform_epsilon <= 0:
            raise ValueError("weight_transform_epsilon 必须大于 0")
        self.network_channels = channel_count
        self.input_channels = channel_count
        self.output_controls = channel_count if output_controls == 0 else output_controls
        if self.output_controls < 1:
            raise ValueError("output_controls 必须为正整数或 0")
        self.hidden_layers = hidden_layers
        self.total_fc_layers = hidden_layers + 1
        self.output_layer_name = layer_name(self.total_fc_layers)
        self.output_preactivation_range_name = output_preactivation_name(self.output_layer_name)
        self.output_control_range_name = output_controls_name(self.output_layer_name)
        if input_tile_features < 0:
            raise ValueError("input_tile_features 必须大于等于 0")
        self.input_tile_features = int(input_tile_features)
        if bias_implementation not in BIAS_IMPLEMENTATIONS:
            raise ValueError(f"bias_implementation 必须是 {sorted(BIAS_IMPLEMENTATIONS)} 之一")
        valid_bias_layers = {f"fc{i}" for i in range(1, self.total_fc_layers + 1)}
        if bias_layers is None or (isinstance(bias_layers, str) and bias_layers.strip().lower() == "all"):
            bias_layer_set = valid_bias_layers
        else:
            bias_layer_set = _validate_layer_names(bias_layers, "bias_layers", valid_bias_layers)
        self.bias_layers = frozenset(bias_layer_set)
        valid_layers = {f"fc{i}" for i in range(1, hidden_layers + 1)}
        self.activation_specs = dict(activation or {})
        if set(self.activation_specs) - valid_layers:
            raise ValueError(f"activation 层只能从 {sorted(valid_layers)} 选择")
        centering_layer_set = _validate_layer_names(centering_layers, "centering_layers", valid_layers)
        if (centering == "none") != (not centering_layer_set):
            raise ValueError("centering=none 时 centering_layers 必须为空；启用中心化时必须指定层")
        if centering_layer_set - valid_layers or not centering_layer_set.issubset(self.activation_specs):
            raise ValueError("centering_layers 必须是已配置激活函数的隐藏层")
        self.activations = nn.ModuleDict(
            {layer: build_activation(spec) for layer, spec in self.activation_specs.items()}
        )
        self.normalization = normalization
        if normalization not in {"none", "l1", "l2", "running_zscore", "batch_renorm"}:
            raise ValueError("normalization 配置无效")
        self.normalization_layers = _validate_layer_names(normalization_layers, "normalization_layers", valid_layers)
        if (normalization == "none") != (not self.normalization_layers):
            raise ValueError("normalization=none 时 normalization_layers 必须为空；启用归一化时必须指定层")
        if normalization == "l1" and branch_mode != "dual":
            raise ValueError("L1归一化只允许在 branch_mode=dual 时启用")
        if normalization in {"l1", "l2"} and branch_mode == "dual" and not self.normalization_layers.issubset(
            self.activation_specs
        ):
            raise ValueError("dual分支的 L1/L2 normalization_layers 必须是 activation 已配置层")
        valid_weight_layers = {f"fc{i}" for i in range(1, self.total_fc_layers + 1)}
        weight_layer_set = set(weight_transform_layers)
        if weight_layer_set - valid_weight_layers:
            raise ValueError(f"weight_transform_layers 只能从 {sorted(valid_weight_layers)} 选择")
        if weight_transform == "ws" and not weight_layer_set:
            weight_layer_set = valid_weight_layers
        if weight_transform == "none" and weight_layer_set:
            raise ValueError("weight_transform=none 时 weight_transform_layers 必须为空")
        self.weight_transform = weight_transform
        self.weight_transform_layers = frozenset(weight_layer_set)
        self.weight_transform_epsilon = float(weight_transform_epsilon)
        self.centering = centering
        self.centering_layers = centering_layer_set
        self.hidden_width = hidden_dim
        self.branch_mode = branch_mode
        use_hardware = _runtime_args.qat_enabled if hardware_enabled is None else bool(hardware_enabled)
        self.hardware_qat = (
            HardwareQAT(hardware_config or QATConfig.from_namespace(_runtime_args)) if use_hardware else None
        )
        if self.hardware_qat is not None:
            self.hardware_qat.output_preactivation_range_name = self.output_preactivation_range_name
            self.hardware_qat.output_control_range_name = self.output_control_range_name
        self.bias_implementation = bias_implementation
        self.array_bias = self.bias_implementation == "array"
        branch_multiplier = 2 if branch_mode == "dual" else 1
        output_width = self.output_controls * (2 if _runtime_args.output_weights == "complex" else 1)

        def layer_width(name):
            return hidden_dim * (branch_multiplier if name in self.activation_specs else 1)

        input_width = 2 * self.input_channels
        for index in range(1, self.total_fc_layers + 1):
            name = f"fc{index}"
            out_width = output_width if index == self.total_fc_layers else hidden_dim
            has_bias = name in self.bias_layers
            layer_input_width = input_width + int(self.array_bias and has_bias)
            layer_bias_implementation = self._layer_bias_implementation(name)
            init_rng_state = torch.random.get_rng_state() if self.array_bias and has_bias else None
            if self.hardware_qat is None:
                setattr(
                    self,
                    name,
                    nn.Linear(
                        layer_input_width,
                        out_width,
                        bias=has_bias and layer_bias_implementation in {"none", "analog"},
                    ),
                )
            else:
                setattr(
                    self,
                    name,
                    MemristorLinear(
                        layer_input_width,
                        out_width,
                        has_bias and layer_bias_implementation in {"none", "analog", "digital"},
                        self.hardware_qat,
                        name,
                        layer_bias_implementation,
                    ),
                )
            if init_rng_state is not None:
                torch.random.set_rng_state(init_rng_state)
                nn.Linear(input_width, out_width, bias=has_bias)
            if index <= self.hidden_layers:
                input_width = layer_width(name)
        if self.normalization == "running_zscore":
            self.running_standardizers = nn.ModuleDict(
                {
                    layer: nn.BatchNorm1d(
                        hidden_dim, eps=running_stat_epsilon, momentum=running_stat_momentum, affine=False
                    )
                    for layer in self.normalization_layers
                }
            )
        elif self.normalization == "batch_renorm":
            self.running_standardizers = nn.ModuleDict(
                {
                    layer: BatchRenorm1d(
                        hidden_dim,
                        eps=running_stat_epsilon,
                        momentum=running_stat_momentum,
                        rmax=batch_renorm_rmax,
                        dmax=batch_renorm_dmax,
                    )
                    for layer in self.normalization_layers
                }
            )
        else:
            self.running_standardizers = nn.ModuleDict()
        self.dropout = nn.Dropout(p=dropout)
        self._activation_diagnostic_enabled = False
        self._preactivation_samples: dict[str, list[torch.Tensor]] = {}
        self._signal_diagnostic_enabled = False
        self._signal_diagnostic_samples: dict[str, list[torch.Tensor]] = {}
        self._control_max_diagnostic_samples: list[torch.Tensor] = []
        self._control_value_diagnostic_samples: list[torch.Tensor] = []
        self._init()

    def _init(self) -> None:
        def initialize_weight(weight: torch.Tensor, array_bias: bool, initializer) -> None:
            if not array_bias:
                initializer(weight)
                return
            initialized = torch.empty_like(weight[:, :-1])
            initializer(initialized)
            with torch.no_grad():
                weight[:, :-1].copy_(initialized)

        # 直接预测实权重，或复权重的 wr/wi 两个分量。
        for index in range(1, self.hidden_layers + 1):
            name = f"fc{index}"
            layer = getattr(self, name)
            array_bias = self.array_bias and name in self.bias_layers
            spec_type = self.activation_specs.get(name, {}).get("type")
            if spec_type == "relu":
                initialize_weight(
                    layer.weight,
                    array_bias,
                    lambda weight: nn.init.kaiming_normal_(weight, nonlinearity="relu"),
                )
            elif spec_type == "tanh":
                initialize_weight(
                    layer.weight,
                    array_bias,
                    lambda weight: nn.init.xavier_normal_(weight, gain=nn.init.calculate_gain("tanh")),
                )
            else:
                initialize_weight(
                    layer.weight,
                    array_bias,
                    lambda weight: nn.init.xavier_normal_(weight, gain=1.0),
                )
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            elif array_bias:
                nn.init.zeros_(layer.weight[:, -1])
        output_layer = getattr(self, f"fc{self.total_fc_layers}")
        output_array_bias = self.array_bias and f"fc{self.total_fc_layers}" in self.bias_layers
        initialize_weight(
            output_layer.weight,
            output_array_bias,
            lambda weight: nn.init.xavier_normal_(weight, gain=1.0),
        )
        if output_layer.bias is not None:
            nn.init.zeros_(output_layer.bias)
        elif output_array_bias:
            nn.init.zeros_(output_layer.weight[:, -1])
        if _runtime_args.unity_constraint == "hard":
            if output_layer.bias is not None:
                nn.init.zeros_(output_layer.weight)
                nn.init.constant_(output_layer.bias, 1.0 / self.output_controls)
            elif self.array_bias and f"fc{self.total_fc_layers}" in self.bias_layers:
                nn.init.zeros_(output_layer.weight)
                nn.init.constant_(output_layer.weight[:, -1], 1.0 / self.output_controls)
        elif _runtime_args.output_domain == "nonnegative":
            if output_layer.bias is not None:
                nn.init.zeros_(output_layer.weight)
                nn.init.constant_(output_layer.bias, 1.0 / self.network_channels)
            elif self.array_bias and f"fc{self.total_fc_layers}" in self.bias_layers:
                nn.init.zeros_(output_layer.weight)
                nn.init.constant_(output_layer.weight[:, -1], 1.0 / self.network_channels)

    def forward(self, rf_I: torch.Tensor, rf_Q: torch.Tensor) -> torch.Tensor:
        x = torch.cat([rf_I, rf_Q], dim=1)
        self._record_signal_diagnostic(f"{layer_name(1)}.input", x)
        for index in range(1, self.hidden_layers + 1):
            name = layer_name(index)
            layer = getattr(self, name)
            x = self._first_linear(x) if index == 1 else self._linear(x, layer, name)
            self._record_signal_diagnostic(f"{name}.tia_output", x)
            x = self.dropout(self._activate(x, name))
            if index < self.hidden_layers:
                self._record_signal_diagnostic(f"{layer_name(index + 1)}.input", x)
        output_name = self.output_layer_name
        self._record_signal_diagnostic(f"{output_name}.input", x)
        x = self._linear(x, getattr(self, output_name), output_name)
        self._record_signal_diagnostic(f"{output_name}.tia_output", x)
        if (
            self._signal_diagnostic_enabled
            and _runtime_args.output_domain == "nonnegative"
            and _runtime_args.unity_constraint == "hard"
        ):
            controls = torch.relu(x)
            remaining_values = 100000 - sum(item.numel() for item in self._control_value_diagnostic_samples)
            if remaining_values > 0:
                self._control_value_diagnostic_samples.append(controls.detach().float().reshape(-1)[:remaining_values].cpu())
            maxima = controls.reshape(-1, controls.shape[-1]).amax(dim=-1)
            remaining = 100000 - sum(item.numel() for item in self._control_max_diagnostic_samples)
            if remaining > 0:
                self._control_max_diagnostic_samples.append(maxima.detach().float().flatten()[:remaining].cpu())
        return x

    def _record_signal_diagnostic(self, name: str, value: torch.Tensor) -> None:
        if not self._signal_diagnostic_enabled:
            return
        flat = value.detach().float().reshape(-1)
        if flat.numel() > 4096:
            indices = torch.linspace(0, flat.numel() - 1, 4096, device=flat.device).round().long()
            flat = flat[indices]
        current = self._signal_diagnostic_samples.setdefault(name, [])
        remaining = 100000 - sum(item.numel() for item in current)
        if remaining > 0:
            current.append(flat[:remaining].cpu())

    def _activate(self, x: torch.Tensor, layer: str) -> torch.Tensor:
        if self.normalization in {"running_zscore", "batch_renorm"} and layer in self.normalization_layers:
            norm = self.running_standardizers[layer]
            if self.normalization == "running_zscore" and self.training and x.shape[0] == 1:
                x = F.batch_norm(x, norm.running_mean, norm.running_var, None, None, False, 0.0, norm.eps)
            else:
                x = norm(x)
        if self.centering == "sample_mean" and layer in self.centering_layers:
            x = x - x.mean(dim=1, keepdim=True)
        inter_layer = self.hardware_qat.config.inter_layer if self.hardware_qat is not None else "none"
        digital_inter_layer = inter_layer == "digital"
        analog_inter_layer = inter_layer == "analog"
        if digital_inter_layer:
            x = self.hardware_qat.digital_adc(x, layer, self.hardware_qat.config.inter_layer_bits)
            x = self._add_digital_bias(x, layer)
        spec = self.activation_specs.get(layer)
        if self.branch_mode == "dual" and spec is not None:
            x_positive, x_negative = split_dual_branches(x)
            if self.hardware_qat is not None and analog_inter_layer:
                x_positive, x_negative = self.hardware_qat.dual_mismatch(x_positive, x_negative, layer)
            x = torch.cat((x_positive, x_negative), dim=1)
            if layer in self.normalization_layers:
                x = apply_hidden_normalization(x, self.normalization)
            x_positive, x_negative = torch.chunk(x, 2, dim=1)
            if self._activation_diagnostic_enabled:
                sample = torch.cat((x_positive, x_negative), dim=1).detach().flatten()[:4096].float().cpu()
                self._preactivation_samples.setdefault(layer, []).append(sample)
            if spec["type"] != "relu":
                act = self.activations[layer]
                x_positive = act(x_positive)
                x_negative = act(x_negative)
            if digital_inter_layer:
                positive_mask = x_positive >= x_negative
                magnitude = self.hardware_qat.digital_dac(
                    x_positive + x_negative,
                    layer,
                    self.hardware_qat.config.inter_layer_bits,
                    unsigned=True,
                )
                return self.hardware_qat.route_dual_rails(magnitude, positive_mask)
            x = torch.cat((x_positive, x_negative), dim=1)
        else:
            if layer in self.normalization_layers:
                x = apply_hidden_normalization(x, self.normalization)
            if analog_inter_layer and spec is not None:
                x = self.hardware_qat.activation_mismatch(x, layer)
            if spec is not None:
                if self._activation_diagnostic_enabled:
                    sample = x.detach().flatten()[:4096].float().cpu()
                    self._preactivation_samples.setdefault(layer, []).append(sample)
                x = self.activations[layer](x)
        if self.hardware_qat is None:
            return x
        if analog_inter_layer:
            return self.hardware_qat.buffer(x, f"{layer}.activation")
        if inter_layer == "none":
            return x
        is_unsigned = spec is not None and spec["type"] == "relu"
        return self.hardware_qat.digital_dac(
            x,
            layer,
            self.hardware_qat.config.inter_layer_bits,
            unsigned=is_unsigned,
        )

    def begin_activation_diagnostics(self) -> None:
        self._preactivation_samples = {}
        self._activation_diagnostic_enabled = True

    def end_activation_diagnostics(self) -> dict[str, dict[str, float]]:
        self._activation_diagnostic_enabled = False
        result: dict[str, dict[str, float]] = {}
        for layer, chunks in self._preactivation_samples.items():
            values = torch.cat(chunks)
            absolute = values.abs()
            quantiles = torch.quantile(absolute, torch.tensor([0.5, 0.9, 0.95, 0.99, 0.999]))
            stats = {
                name: float(value) for name, value in zip(("q50", "q90", "q95", "q99", "q999"), quantiles, strict=True)
            }
            spec = self.activation_specs[layer]
            if spec["type"] == "pwl":
                stats["tail_ratio"] = float((absolute > float(spec.get("knee", 1.0))).float().mean())
            elif spec["type"] == "tanh":
                activated = torch.tanh(float(spec.get("beta", 1.0)) * values)
                stats["mean_local_gain"] = float((1.0 - activated.square()).mean())
            result[layer] = stats
        self._preactivation_samples = {}
        return result

    def begin_signal_diagnostics(self) -> None:
        self._signal_diagnostic_samples = {}
        self._control_max_diagnostic_samples = []
        self._control_value_diagnostic_samples = []
        self._signal_diagnostic_enabled = True

    def end_signal_diagnostics(self) -> dict[str, dict[str, float]]:
        self._signal_diagnostic_enabled = False
        result: dict[str, dict[str, float]] = {}
        for name, chunks in self._signal_diagnostic_samples.items():
            values = torch.cat(chunks)
            absolute = values.abs()
            quantiles = torch.quantile(absolute, torch.tensor([0.5, 0.95, 0.99, 0.999]))
            result[name] = {
                "min": float(values.min()),
                "max": float(values.max()),
                "abs_p50": float(quantiles[0]),
                "abs_p95": float(quantiles[1]),
                "abs_p99": float(quantiles[2]),
                "abs_p999": float(quantiles[3]),
                "abs_max": float(absolute.max()),
                "mean": float(values.mean()),
                "std": float(values.std(unbiased=False)),
                "samples": int(values.numel()),
            }
        if self._control_max_diagnostic_samples:
            values = torch.cat(self._control_max_diagnostic_samples)
            quantiles = torch.quantile(values, torch.tensor([0.5, 0.95, 0.99, 0.999]))
            result["raw_control_vector_max"] = {
                "p50": float(quantiles[0]),
                "p95": float(quantiles[1]),
                "p99": float(quantiles[2]),
                "p999": float(quantiles[3]),
                "max": float(values.max()),
                "samples": int(values.numel()),
            }
        if self._control_value_diagnostic_samples:
            values = torch.cat(self._control_value_diagnostic_samples)
            quantiles = torch.quantile(values, torch.tensor([0.5, 0.95, 0.99, 0.999]))
            result["raw_control_after_relu"] = {
                "p50": float(quantiles[0]),
                "p95": float(quantiles[1]),
                "p99": float(quantiles[2]),
                "p999": float(quantiles[3]),
                "max": float(values.max()),
                "samples": int(values.numel()),
            }
        self._signal_diagnostic_samples = {}
        self._control_max_diagnostic_samples = []
        self._control_value_diagnostic_samples = []
        return result

    def _effective_weight(self, name: str, weight: torch.Tensor) -> torch.Tensor:
        if self.weight_transform != "ws" or name not in self.weight_transform_layers:
            return weight
        return standardize_weight(weight, self.weight_transform_epsilon)

    def _prepare_linear_input(self, x: torch.Tensor, name: str) -> torch.Tensor:
        if self.array_bias and name in self.bias_layers:
            return torch.cat((x, x.new_ones((*x.shape[:-1], 1))), dim=-1)
        return x

    def _layer_bias_implementation(self, name: str) -> str:
        if name not in self.bias_layers:
            return "none"
        if self.array_bias:
            return "array"
        if self.hardware_qat is not None:
            return self.hardware_qat.config.inter_layer
        return "none"

    def _digital_bias(self, name: str, quantization_range: str | None = None) -> torch.Tensor | None:
        if self.hardware_qat is None or self._layer_bias_implementation(name) != "digital":
            return None
        layer = getattr(self, name)
        if layer.bias is None:
            return None
        return self.hardware_qat.bias(
            layer.bias,
            name,
            "digital",
            quantization_range=quantization_range,
        )

    def _add_digital_bias(
        self,
        x: torch.Tensor,
        name: str,
        quantization_range: str | None = None,
    ) -> torch.Tensor:
        """Add quantized Bias in ADC-dequantized units using float accumulation without overflow modeling."""
        bias = self._digital_bias(name, quantization_range=quantization_range)
        if bias is None:
            return x
        return x + bias.reshape((1,) * (x.ndim - 1) + (-1,))

    def prepare_output_controls(
        self,
        controls: torch.Tensor,
        depth_index: torch.Tensor | None = None,
        depth_count: int | None = None,
        active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.hardware_qat is None:
            return controls
        if self.hardware_qat.config.inter_layer == "analog":
            if _runtime_args.output_domain != "nonnegative":
                return controls
            reference = (
                float(_runtime_args.control_adc_full_scale)
                if _runtime_args.control_adc_range == "full_scale"
                else None
            )
            return self.hardware_qat.output_activation_mismatch(controls, self.output_layer_name, reference)
        if self.hardware_qat.config.inter_layer != "digital":
            return controls
        value_max = None
        if _runtime_args.control_adc_range == "full_scale":
            value_max = controls.new_tensor(_runtime_args.control_adc_full_scale)
        adc_controls = controls
        if active_mask is not None:
            adc_controls = controls * active_mask.to(controls.dtype).unsqueeze(1)
        adc_controls = self.hardware_qat.quantize_output_controls(
            adc_controls,
            name=self.output_preactivation_range_name,
            bits=self.hardware_qat.config.control_bits,
            unsigned=False,
            value_max=value_max,
            depth_index=depth_index,
            depth_count=depth_count,
        )
        if self._layer_bias_implementation(self.output_layer_name) == "digital":
            return self._add_digital_bias(
                adc_controls,
                self.output_layer_name,
                quantization_range=self.output_preactivation_range_name,
            )
        if _runtime_args.output_domain == "nonnegative":
            return F.relu(adc_controls)
        return adc_controls

    def _linear(self, x: torch.Tensor, layer: nn.Module, name: str) -> torch.Tensor:
        x = self._prepare_linear_input(x, name)
        weight = self._effective_weight(name, layer.weight)
        if self.hardware_qat is None:
            if self.array_bias and name in self.bias_layers:
                return F.linear(x[..., :-1], weight[..., :-1], weight[..., -1])
            return F.linear(x, weight, layer.bias)
        return layer(x, weight, quantize_input=False)

    def quantize_controls(
        self,
        controls: torch.Tensor,
        depth_index: torch.Tensor | None = None,
        depth_count: int | None = None,
    ) -> torch.Tensor:
        if self.hardware_qat is None:
            return controls
        if self.hardware_qat.config.inter_layer == "digital":
            return controls
        unsigned = _runtime_args.output_domain == "nonnegative"
        control_range_mode = _runtime_args.control_adc_range
        value_max = None
        if control_range_mode == "full_scale":
            value_max = controls.new_tensor(_runtime_args.control_adc_full_scale)
        return self.hardware_qat.quantize_output_controls(
            controls,
            name=self.output_control_range_name,
            bits=self.hardware_qat.config.control_bits,
            unsigned=unsigned,
            value_max=value_max,
            depth_index=depth_index,
            depth_count=depth_count,
        )

    def deployment_linear_state(self) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for index in range(1, self.total_fc_layers + 1):
            linear = getattr(self, f"fc{index}")
            name = f"fc{index}"
            weight = self._effective_weight(f"fc{index}", linear.weight).detach()
            bias = linear.bias.detach() if linear.bias is not None else None
            if self.normalization in {"running_zscore", "batch_renorm"} and name in self.normalization_layers:
                norm = self.running_standardizers[name]
                if bias is None:
                    if self.array_bias and name in self.bias_layers:
                        core_weight, array_bias = weight[:, :-1], weight[:, -1]
                        core_weight, array_bias = fold_standardizer(
                            core_weight,
                            array_bias,
                            self.normalization,
                            norm.running_mean,
                            norm.running_var,
                            norm.eps,
                            getattr(norm, "weight", None),
                            getattr(norm, "bias", None),
                        )
                        weight = torch.cat((core_weight, array_bias.unsqueeze(1)), dim=1)
                    else:
                        zero_bias = weight.new_zeros(weight.shape[0])
                        weight, folded_bias = fold_standardizer(
                            weight,
                            zero_bias,
                            self.normalization,
                            norm.running_mean,
                            norm.running_var,
                            norm.eps,
                            getattr(norm, "weight", None),
                            getattr(norm, "bias", None),
                        )
                        if torch.count_nonzero(folded_bias).item():
                            raise ValueError(f"{name} 的归一化折叠产生非零 bias，但该层未配置 bias")
                else:
                    weight, bias = fold_standardizer(
                        weight,
                        bias,
                        self.normalization,
                        norm.running_mean,
                        norm.running_var,
                        norm.eps,
                        getattr(norm, "weight", None),
                        getattr(norm, "bias", None),
                    )
            result[f"fc{index}.weight"] = weight.cpu()
            if bias is not None:
                result[f"fc{index}.bias"] = bias.cpu()
        return result

    def _first_linear(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_linear_input(x, "fc1")
        tile = self.input_tile_features
        first_layer = self.fc1
        weight = self._effective_weight("fc1", first_layer.weight)
        if tile <= 0 or tile >= x.shape[1]:
            if self.hardware_qat is None:
                if self.array_bias and "fc1" in self.bias_layers:
                    return F.linear(x[..., :-1], weight[..., :-1], weight[..., -1])
                return F.linear(x, weight, first_layer.bias)
            return first_layer(x, weight, self.hardware_qat.config.input_bits)
        if self.hardware_qat is None:
            if self.array_bias and "fc1" in self.bias_layers:
                x = x[..., :-1]
                weight = weight[..., :-1]
                bias = first_layer.weight[:, -1]
            else:
                bias = first_layer.bias
            result = (
                bias.unsqueeze(0).expand(x.shape[0], -1)
                if bias is not None
                else x.new_zeros((x.shape[0], first_layer.out_features))
            )
            for start in range(0, x.shape[1], tile):
                end = min(start + tile, x.shape[1])
                result = result + F.linear(x[:, start:end], weight[:, start:end], None)
            return result
        return first_layer.forward_tiled(x, tile, weight, self.hardware_qat.config.input_bits)

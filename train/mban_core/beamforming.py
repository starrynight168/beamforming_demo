from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import runtime as _runtime
# =====================================================================
# 损失与推理辅助
# =====================================================================


def _loss_terms() -> dict:
    return _runtime.args.loss["terms"]


def selected_error(delta: torch.Tensor) -> torch.Tensor:
    loss = _runtime.args.loss
    if loss["error_function"] == "mse":
        return delta.square()
    if loss["error_function"] == "l1":
        return delta.abs()
    epsilon = loss["charbonnier_epsilon"]
    return torch.sqrt(delta.square() + epsilon**2) - epsilon


def iq_data_loss_components(
    pred_i: torch.Tensor,
    pred_q: torch.Tensor,
    target_i: torch.Tensor,
    target_q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    iq_loss = (selected_error(pred_i - target_i) + selected_error(pred_q - target_q)).mean()
    envelope_epsilon = _runtime.args.loss["envelope_epsilon"]
    pred_amp = torch.sqrt(pred_i.square() + pred_q.square() + envelope_epsilon)
    target_amp = torch.sqrt(target_i.square() + target_q.square() + envelope_epsilon)
    envelope_loss = selected_error(pred_amp - target_amp).mean()
    terms = _loss_terms()
    total = float(terms["iq"]["weight"]) * iq_loss + float(terms["envelope"]["weight"]) * envelope_loss
    return total, iq_loss, envelope_loss


def unity_loss(weights: torch.Tensor) -> torch.Tensor:
    """Soft 无失真约束；复权重同时要求 Σwr=1、Σwi=0。"""
    wr, wi = split_complex_weights(weights)
    if getattr(_runtime.args, "unity_scope", "global") == "per_angle":
        real_sum = wr.sum(dim=2)
        imag_sum = wi.sum(dim=2)
    else:
        real_sum = wr.flatten(start_dim=1).sum(dim=1)
        imag_sum = wi.flatten(start_dim=1).sum(dim=1)
    return torch.mean((real_sum - 1.0).square() + imag_sum.square())


def _aperture_smoothness(
    weights: torch.Tensor,
    mask: torch.Tensor | None,
    domain: str,
    order: int,
) -> torch.Tensor:
    wr, wi = split_complex_weights(weights)
    wr, wi = wr.float(), wi.float()
    if mask is None:
        active = torch.ones_like(wr)
    else:
        active = mask.to(device=wr.device, dtype=wr.dtype).unsqueeze(1).expand_as(wr)
    if domain == "shape":
        magnitude = torch.sqrt(wr.square() + wi.square() + 1.0e-12)
        shape_scale = (magnitude * active).sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        wr_norm = wr * active / shape_scale
        wi_norm = wi * active / shape_scale
    else:
        wr_norm = wr * active
        wi_norm = wi * active

    if order == 1:
        edge_active = active[..., 1:] * active[..., :-1]
        d1 = (wr_norm[..., 1:] - wr_norm[..., :-1]).square()
        d1 = d1 + (wi_norm[..., 1:] - wi_norm[..., :-1]).square()
        return (d1 * edge_active).sum() / edge_active.sum().clamp_min(1.0)
    second_active = active[..., 2:] * active[..., 1:-1] * active[..., :-2]
    d2_real = wr_norm[..., 2:] - 2.0 * wr_norm[..., 1:-1] + wr_norm[..., :-2]
    d2_imag = wi_norm[..., 2:] - 2.0 * wi_norm[..., 1:-1] + wi_norm[..., :-2]
    second = (d2_real.square() + d2_imag.square()) * second_active
    return second.sum() / second_active.sum().clamp_min(1.0)


def aperture_weight_regularization(
    weights: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    terms = _loss_terms()
    total = weights.new_zeros(())
    for name, order in (("d1", 1), ("d2", 2)):
        term = terms[name]
        weight = float(term["weight"])
        if weight > 0.0:
            total = total + weight * _aperture_smoothness(weights, mask, term["domain"], order)
    return total


def raw_control_range_penalty(raw_controls: torch.Tensor) -> torch.Tensor:
    """作用于 K 维 raw_controls：E[max(|c_raw|-R, 0)^2]。"""
    term = _loss_terms()["lrange"]
    limit = float(term["limit"])
    weight = float(term["weight"])
    if weight <= 0.0:
        return raw_controls.new_zeros(())
    excess = torch.relu(raw_controls.abs() - limit)
    return (excess.square()).mean()


def training_loss(
    loss_iq: torch.Tensor,
    loss_unity: torch.Tensor,
    loss_aperture: torch.Tensor,
    loss_range: torch.Tensor,
    loss_distillation: torch.Tensor,
) -> torch.Tensor:
    """各损失使用独立系数，避免隐式混合。"""
    terms = _loss_terms()
    unity_term = 0.0 if _runtime.args.unity_constraint == "hard" else float(terms["unity"]["weight"]) * loss_unity
    return (
        loss_iq
        + unity_term
        + loss_aperture
        + float(terms["lrange"]["weight"]) * loss_range
        + _runtime.args.distillation_weight * loss_distillation
    )


def init_weight_sum_stats(device: torch.device) -> dict:
    return {
        "count": 0,
        "sum": torch.zeros((), dtype=torch.float64, device=device),
        "sum_sq": torch.zeros((), dtype=torch.float64, device=device),
        "min": torch.full((), float("inf"), dtype=torch.float32, device=device),
        "max": torch.full((), float("-inf"), dtype=torch.float32, device=device),
    }


@torch.no_grad()
def update_weight_sum_stats(stats: dict, weights: torch.Tensor) -> None:
    wr, _ = split_complex_weights(weights)
    weight_sums = wr.detach().float().sum(dim=-1)
    update_scalar_stats(stats, weight_sums)


@torch.no_grad()
def update_scalar_stats(stats: dict, values: torch.Tensor) -> None:
    if values.numel() == 0:
        return
    stats["count"] += values.numel()
    stats["sum"] += values.sum(dtype=torch.float64)
    stats["sum_sq"] += values.square().sum(dtype=torch.float64)
    stats["min"] = torch.minimum(stats["min"], values.min())
    stats["max"] = torch.maximum(stats["max"], values.max())


def init_depth_weight_sum_stats(device: torch.device) -> list[dict]:
    return [init_weight_sum_stats(device) for _ in range(3)]


@torch.no_grad()
def update_depth_weight_sum_stats(stats: list[dict], weights: torch.Tensor, pixel_idx: torch.Tensor) -> None:
    wr, _ = split_complex_weights(weights)
    weight_sums = wr.detach().float().sum(dim=-1)
    z_idx = pixel_idx // _runtime.IMG_W
    depth_band = torch.clamp(3 * z_idx // _runtime.IMG_H, max=2)
    for band in range(3):
        update_scalar_stats(stats[band], weight_sums[depth_band == band])


def finalize_weight_sum_stats(stats: dict) -> tuple[float, float, float, float]:
    count = max(int(stats["count"]), 1)
    mean = stats["sum"] / count
    variance = (stats["sum_sq"] / count - mean.square()).clamp_min(0.0)
    return (
        mean.item(),
        variance.sqrt().item(),
        stats["min"].item(),
        stats["max"].item(),
    )


def normalize_db(env_p: torch.Tensor) -> torch.Tensor:
    pred_dB = (20.0 * torch.log10(env_p + 1e-12) + _runtime.DR) / _runtime.DR
    return torch.clamp(pred_dB, 0.0, 1.0)


def predict_weights(
    model: nn.Module,
    I_a: torch.Tensor,
    Q_a: torch.Tensor,
) -> torch.Tensor:
    _, A, _ = I_a.shape
    weights_list = []

    for a in range(A):
        weights = model(I_a[:, a, :], Q_a[:, a, :])
        weights_list.append(weights)

    return torch.stack(weights_list, dim=1)


def normalize_input_iq(
    i_data: torch.Tensor,
    q_data: torch.Tensor,
    active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (*i_data.shape[:-1], 1)
    if _runtime.args.input_normalization == "none":
        return i_data, q_data, i_data.new_ones(shape), i_data.new_zeros(shape)
    mask = active.to(i_data.dtype).unsqueeze(1).expand_as(i_data)
    count = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    if _runtime.args.input_normalization == "rms":
        scale = (
            (((i_data.square() + q_data.square()) * mask).sum(dim=-1, keepdim=True) / count).sqrt().clamp_min(1.0e-8)
        )
        return i_data / scale, q_data / scale, scale, torch.zeros_like(scale)
    joint_count = 2.0 * count
    mean = ((i_data + q_data) * mask).sum(dim=-1, keepdim=True) / joint_count
    variance = (((i_data - mean).square() + (q_data - mean).square()) * mask).sum(dim=-1, keepdim=True) / joint_count
    scale = variance.sqrt().clamp_min(1.0e-8)
    return (i_data - mean) * mask / scale, (q_data - mean) * mask / scale, scale, mean


def _quantize_interpolation_fraction(fraction: torch.Tensor) -> torch.Tensor:
    bits = int(_runtime.args.interpolation_bits)
    if bits >= 32:
        return fraction
    levels = float((1 << bits) - 1)
    return torch.round(fraction.clamp(0.0, 1.0) * levels) / levels


def _interpolation_taps(
    coordinates: torch.Tensor,
    control_count: int | torch.Tensor,
    interpolation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.is_tensor(control_count):
        limit = control_count.to(device=coordinates.device, dtype=torch.long).reshape(
            -1, *([1] * (coordinates.ndim - 1))
        ) - 1
    else:
        limit = coordinates.new_tensor(control_count - 1, dtype=torch.long)

    def clamp_indices(indices: torch.Tensor) -> torch.Tensor:
        broadcast_limit = limit
        while broadcast_limit.ndim < indices.ndim:
            broadcast_limit = broadcast_limit.unsqueeze(-1)
        return indices.clamp_min(0).minimum(broadcast_limit)

    if interpolation == "nearest":
        indices = clamp_indices(coordinates.round().long()).unsqueeze(-1)
        return indices, torch.ones_like(indices, dtype=coordinates.dtype)
    left = clamp_indices(coordinates.floor().long())
    if interpolation == "linear":
        right = clamp_indices(left + 1)
        fraction = _quantize_interpolation_fraction(coordinates - left.to(coordinates.dtype))
        return torch.stack((left, right), dim=-1), torch.stack((1.0 - fraction, fraction), dim=-1)
    if interpolation == "cubic":
        base = coordinates.floor().long()
        fraction = _quantize_interpolation_fraction(coordinates - base.to(coordinates.dtype))
        offsets = torch.tensor((-1, 0, 1, 2), device=coordinates.device, dtype=torch.long)
        indices = clamp_indices(base.unsqueeze(-1) + offsets)
        t2 = fraction.square()
        t3 = t2 * fraction
        coefficients = torch.stack(
            (
                -0.5 * fraction + t2 - 0.5 * t3,
                1.0 - 2.5 * t2 + 1.5 * t3,
                0.5 * fraction + 2.0 * t2 - 1.5 * t3,
                -0.5 * t2 + 0.5 * t3,
            ),
            dim=-1,
        )
        return indices, coefficients
    raise ValueError(f"不支持的 output_interpolation: {interpolation}")


def _dynamic_interpolation_coordinates(
    relative: torch.Tensor,
    sizes: torch.Tensor,
    control_count: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    coordinates = (
        relative.clamp_min(0).to(dtype)
        * (control_count - 1)
        / sizes[:, None].clamp_min(2).sub(1).to(dtype)
    )
    return torch.where(
        sizes[:, None] == 1,
        torch.full_like(coordinates, (control_count - 1) / 2.0),
        coordinates,
    )


def _dynamic_interpolation_map(
    aperture_size: torch.Tensor,
    channels: int,
    control_count: int,
    interpolation: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sizes = aperture_size.to(device=device, dtype=torch.long).reshape(-1).clamp_min(1)
    positions = torch.arange(channels, device=device, dtype=torch.long).view(1, -1)
    relative = positions - (channels - sizes)[:, None] // 2
    valid = (relative >= 0) & (relative < sizes[:, None])
    coordinates = _dynamic_interpolation_coordinates(
        relative, sizes, control_count, torch.float32
    )
    indices, coefficients = _interpolation_taps(coordinates, control_count, interpolation)
    return indices, coefficients, valid


def dynamic_interpolation_cache_bytes(
    network_channels: int,
    control_count: int,
    interpolation: str,
) -> int:
    if network_channels <= 0 or control_count <= 0:
        raise ValueError("network_channels 和 control_count 必须为正数")
    if control_count > network_channels:
        raise ValueError("control_count 不能超过 network_channels")
    if interpolation not in {"nearest", "linear", "cubic"}:
        raise ValueError(f"不支持的 output_interpolation: {interpolation}")
    return (network_channels + 1) * (control_count + network_channels * control_count * 4)


def _expand_dynamic_with_taps(
    controls: torch.Tensor,
    indices: torch.Tensor,
    coefficients: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    pixels, angles, channels = controls.shape[0], controls.shape[1], indices.shape[1]
    expanded = torch.zeros(
        (pixels, angles, channels), device=controls.device, dtype=controls.dtype
    )
    for tap in range(indices.shape[-1]):
        tap_indices = indices[..., tap].unsqueeze(1).expand(-1, angles, -1)
        tap_values = controls.gather(-1, tap_indices)
        expanded = expanded + tap_values * coefficients[..., tap].unsqueeze(1)
    return expanded * valid[:, None, :].to(expanded.dtype)


class _CachedDynamicInterpolation(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        controls: torch.Tensor,
        basis: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(basis)
        with torch.autocast(device_type=controls.device.type, enabled=False):
            expanded = torch.bmm(controls.to(torch.float32), basis)
        return expanded.to(dtype=controls.dtype)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        (basis,) = ctx.saved_tensors
        with torch.autocast(device_type=grad_output.device.type, enabled=False):
            grad_controls = torch.matmul(grad_output.to(torch.float32), basis.transpose(1, 2))
        return grad_controls.to(grad_output.dtype), None


def expand_dynamic_output_controls(
    controls: torch.Tensor,
    aperture_size: torch.Tensor,
    channels: int,
    interpolation_cache: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    components = torch.chunk(controls, 2, dim=-1) if _runtime.args.output_weights == "complex" else (controls,)
    expanded_components = []
    for component in components:
        if component.shape[-1] == channels:
            expanded_components.append(component)
            continue
        pixels, angles, control_count = component.shape
        flat = component.reshape(pixels * angles, control_count)
        cached_basis = None
        if interpolation_cache is not None:
            cached_basis = interpolation_cache.get("dynamic_output_basis_by_size")
        use_cache = (
            cached_basis is not None
            and cached_basis.ndim == 3
            and cached_basis.shape[0] > channels
            and cached_basis.shape[1] == control_count
            and cached_basis.shape[2] == channels
            and flat.dtype in (torch.float16, torch.bfloat16, torch.float32)
        )
        if use_cache:
            size_indices = aperture_size.reshape(-1).long()
            expanded_components.append(
                _CachedDynamicInterpolation.apply(
                    component,
                    cached_basis[size_indices],
                )
            )
            continue
        indices, coefficients, valid = _dynamic_interpolation_map(
            aperture_size,
            channels,
            control_count,
            _runtime.args.output_interpolation,
            controls.device,
        )
        coefficients = coefficients.to(dtype=flat.dtype)
        expanded_components.append(
            _expand_dynamic_with_taps(component, indices, coefficients, valid)
        )
    return torch.cat(expanded_components, dim=-1)


def dynamic_control_active_mask(
    aperture_size: torch.Tensor,
    control_count: int,
    interpolation: str,
) -> torch.Tensor:
    if control_count <= 0:
        return torch.zeros((*aperture_size.shape, 0), device=aperture_size.device, dtype=torch.bool)
    sizes = aperture_size.to(dtype=torch.long).clamp_min(1)
    mask = torch.ones((*sizes.shape, control_count), device=sizes.device, dtype=torch.bool)
    need = sizes < control_count
    if not bool(need.any().item()):
        return mask
    sub_sizes = sizes[need]
    max_size = int(sub_sizes.max().item())
    indices, coefficients, valid = _dynamic_interpolation_map(
        sub_sizes,
        max_size,
        control_count,
        interpolation,
        sizes.device,
    )
    mask[need] = _active_control_mask_from_taps(indices, coefficients, valid, control_count)
    return mask


def _active_control_mask_from_taps(
    indices: torch.Tensor,
    coefficients: torch.Tensor,
    valid: torch.Tensor,
    control_count: int,
) -> torch.Tensor:
    active = torch.zeros(
        (indices.shape[0], control_count), device=indices.device, dtype=torch.bool
    )
    row_index = torch.arange(indices.shape[0], device=indices.device).view(-1, 1).expand_as(valid)
    for tap in range(indices.shape[-1]):
        tap_valid = valid & (coefficients[..., tap] != 0)
        active[row_index[tap_valid], indices[..., tap][tap_valid]] = True
    return active


def build_dynamic_interpolation_cache(
    network_channels: int,
    control_count: int,
    interpolation: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if network_channels <= 0 or control_count <= 0:
        raise ValueError("network_channels 和 control_count 必须为正数")
    if control_count > network_channels:
        raise ValueError("control_count 不能超过 network_channels")
    sizes = torch.arange(1, network_channels + 1, device=device, dtype=torch.long)
    indices, coefficients, valid = _dynamic_interpolation_map(
        sizes,
        network_channels,
        control_count,
        interpolation,
        device,
    )
    active = torch.ones((network_channels, control_count), device=device, dtype=torch.bool)
    need = sizes < control_count
    if bool(need.any().item()):
        active[need] = _active_control_mask_from_taps(
            indices[need], coefficients[need], valid[need], control_count
        )
    zero_active = torch.zeros((1, control_count), device=device, dtype=torch.bool)
    cache = {
        "dynamic_control_active_mask_by_size": torch.cat((zero_active, active), dim=0),
    }
    basis = torch.zeros(
        (network_channels + 1, control_count, network_channels),
        device=device,
        dtype=torch.float32,
    )
    valid_coefficients = coefficients * valid.unsqueeze(-1).to(torch.float32)
    for tap in range(indices.shape[-1]):
        basis[1:].scatter_add_(
            1,
            indices[..., tap].unsqueeze(1),
            valid_coefficients[..., tap].unsqueeze(1),
        )
    cache["dynamic_output_basis_by_size"] = basis
    return cache


def _linear_resize_last(values: torch.Tensor, size: int) -> torch.Tensor:
    old_size = values.shape[-1]
    if old_size == size:
        return values
    if old_size == 1 or size == 1:
        return values[..., :1].expand(*values.shape[:-1], size)
    coordinates = torch.linspace(0.0, float(old_size - 1), int(size), device=values.device, dtype=values.dtype)
    left = coordinates.floor().long().clamp_max(old_size - 1)
    right = (left + 1).clamp_max(old_size - 1)
    fraction = _quantize_interpolation_fraction(coordinates - left.to(values.dtype))
    return values[..., left] * (1.0 - fraction) + values[..., right] * fraction


def _cubic_resize_last(values: torch.Tensor, size: int) -> torch.Tensor:
    old_size = values.shape[-1]
    if old_size == size:
        return values
    if old_size <= 0 or size <= 0:
        raise ValueError("resize 的输入和输出长度必须为正数")
    if old_size == 1:
        return values[..., :1].expand(*values.shape[:-1], size)
    coordinates = torch.linspace(0.0, float(old_size - 1), int(size), device=values.device, dtype=values.dtype)
    indices, coefficients = _interpolation_taps(coordinates, old_size, "cubic")
    result = torch.zeros((*values.shape[:-1], size), device=values.device, dtype=values.dtype)
    for tap in range(4):
        result = result + values[..., indices[..., tap]] * coefficients[..., tap]
    return result


def _linear_interpolation_basis(control_count: int, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if control_count <= 0 or channels <= 0:
        raise ValueError("control_count 和 channels 必须为正数")
    if control_count == channels:
        return torch.eye(channels, device=device, dtype=dtype)
    if control_count == 1:
        return torch.ones((1, channels), device=device, dtype=dtype)
    if channels == 1:
        basis = torch.zeros((control_count, 1), device=device, dtype=dtype)
        basis[0, 0] = 1.0
        return basis
    coordinates = torch.linspace(0.0, float(control_count - 1), channels, device=device, dtype=dtype)
    left = coordinates.floor().long().clamp_max(control_count - 1)
    right = (left + 1).clamp_max(control_count - 1)
    fraction = _quantize_interpolation_fraction(coordinates - left.to(dtype))
    basis = torch.zeros((control_count, channels), device=device, dtype=dtype)
    basis.scatter_add_(0, left.unsqueeze(0), (1.0 - fraction).unsqueeze(0))
    basis.scatter_add_(0, right.unsqueeze(0), fraction.unsqueeze(0))
    return basis


def _cubic_interpolation_basis(control_count: int, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if control_count <= 0 or channels <= 0:
        raise ValueError("control_count 和 channels 必须为正数")
    if control_count == channels:
        return torch.eye(channels, device=device, dtype=dtype)
    coordinates = torch.linspace(0.0, float(control_count - 1), channels, device=device, dtype=dtype)
    indices, coefficients = _interpolation_taps(coordinates, control_count, "cubic")
    basis = torch.zeros((control_count, channels), device=device, dtype=dtype)
    for tap in range(4):
        basis.scatter_add_(0, indices[..., tap].unsqueeze(0), coefficients[..., tap].unsqueeze(0))
    return basis


def _nearest_interpolation_basis(control_count: int, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if control_count <= 0 or channels <= 0:
        raise ValueError("control_count 和 channels 必须为正数")
    positions = torch.arange(channels, device=device, dtype=torch.long)
    indices = (positions * control_count // channels).clamp_max(control_count - 1)
    basis = torch.zeros((control_count, channels), device=device, dtype=dtype)
    basis[indices, positions] = 1.0
    return basis


def expand_fixed_output_controls(controls: torch.Tensor, channels: int) -> torch.Tensor:
    components = torch.chunk(controls, 2, dim=-1) if _runtime.args.output_weights == "complex" else (controls,)
    expanded = []
    for component in components:
        if component.shape[-1] == channels:
            expanded.append(component)
            continue
        flat = component.reshape(-1, 1, component.shape[-1])
        if _runtime.args.output_interpolation == "nearest":
            value = F.interpolate(flat, size=channels, mode="nearest")
        elif _runtime.args.output_interpolation == "linear":
            value = _linear_resize_last(component, channels).reshape(-1, 1, channels)
        else:
            value = _cubic_resize_last(component, channels).reshape(-1, 1, channels)
        expanded.append(value.reshape(*component.shape[:-1], channels))
    return torch.cat(expanded, dim=-1)


def normalize_controls_for_adc(
    controls: torch.Tensor,
    active_mask: torch.Tensor | None = None,
    hardware: object | None = None,
    control_range_name: str | None = None,
) -> torch.Tensor:
    mode = getattr(_runtime.args, "control_normalization", "none")
    if mode not in {"none", "linf"}:
        raise ValueError(f"不支持的 control_normalization: {mode}")
    values = controls
    mask = None
    if active_mask is not None:
        mask = active_mask.to(device=controls.device).unsqueeze(1)
        values = controls * mask.to(controls.dtype)
    if _runtime.args.output_domain == "signed":
        denominator = values.abs().amax(dim=-1, keepdim=True)
    else:
        denominator = values.amax(dim=-1, keepdim=True)
    epsilon = max(float(getattr(getattr(hardware, "config", None), "epsilon", 1.0e-12)), 1.0e-12)
    reference = denominator
    if mode == "linf" and hardware is not None:
        if control_range_name is None:
            raise ValueError("linf 控制归一化必须提供动态输出量程名")
        reference = hardware.control_reference(reference, control_range_name)
    reference_check = reference.float()
    reference_valid = torch.isfinite(reference_check) & (reference_check > epsilon)
    safe_reference = torch.where(reference_valid, reference, torch.ones_like(reference))
    if mode == "none":
        normalized = values
    else:
        normalized = (values.float() / safe_reference.float()).to(values.dtype)
    zero = denominator.float() <= epsilon
    invalid_reference = ~reference_valid if mode == "linf" else torch.zeros_like(zero)
    fallback_mask = zero | invalid_reference if mode == "linf" else zero
    active = torch.ones_like(values, dtype=torch.bool) if mask is None else mask
    bits = int(getattr(_runtime.args, "control_bits", 32))
    if bits < 32:
        fallback_value = float(getattr(_runtime.args, "control_adc_full_scale", 1.0))
        fallback = active.to(values.dtype) * fallback_value
    else:
        active_count = active.sum(dim=-1, keepdim=True).clamp_min(1)
        fallback = active.to(values.dtype) / active_count
    normalized = torch.where(fallback_mask, fallback, normalized)
    if hardware is not None:
        hardware.adaptive_valid = (~fallback_mask).detach()
    return normalized


def apply_output_domain(controls: torch.Tensor) -> torch.Tensor:
    if _runtime.args.output_domain == "nonnegative":
        return F.relu(controls)
    if _runtime.args.output_domain == "signed":
        return controls
    raise ValueError(f"不支持的 output_domain: {_runtime.args.output_domain}")


def prepare_output_controls(
    model: nn.Module,
    controls: torch.Tensor,
    depth_index: torch.Tensor | None = None,
    depth_count: int | None = None,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    hardware = getattr(model, "hardware_qat", None)
    if hardware is not None and hardware.config.inter_layer in {"analog", "digital"}:
        return model.prepare_output_controls(controls, depth_index, depth_count, active_mask)
    return apply_output_domain(controls)


def project_output_weights(weights: torch.Tensor, controls: int) -> torch.Tensor:
    if controls <= 0:
        return weights
    components = torch.chunk(weights, 2, dim=-1) if _runtime.args.output_weights == "complex" else (weights,)
    projected = []
    for component in components:
        if controls == component.shape[-1]:
            projected.append(component)
            continue
        flat = component.reshape(-1, 1, component.shape[-1])
        if _runtime.args.output_interpolation == "nearest":
            compact = F.interpolate(flat, size=controls, mode="nearest")
            restored = F.interpolate(compact, size=component.shape[-1], mode="nearest")
        elif _runtime.args.output_interpolation == "linear":
            compact = _linear_resize_last(component, controls).reshape(-1, 1, controls)
            restored = _linear_resize_last(compact.squeeze(1), component.shape[-1]).reshape(-1, 1, component.shape[-1])
        else:
            compact = _cubic_resize_last(component, controls).reshape(-1, 1, controls)
            restored = _cubic_resize_last(compact.squeeze(1), component.shape[-1]).reshape(-1, 1, component.shape[-1])
        projected.append(restored.reshape_as(component))
    return torch.cat(projected, dim=-1)


def _project_active_aperture_vectorized(
    component: torch.Tensor,
    aperture_size: torch.Tensor,
    controls: int,
) -> torch.Tensor:
    """全向量化 M→Kp→M 投影：两阶段批量线性重采样，避免按孔径尺寸的 Python 循环。"""
    channels = component.shape[-1]
    device = component.device
    dtype = component.dtype
    need = aperture_size > controls
    if not bool(need.any().item()):
        return component
    projected = component.clone()
    sub = component[need]
    sub_s = aperture_size[need]
    n = sub.shape[0]
    compact = controls
    if n == 0:
        return projected
    start = (channels - sub_s) // 2
    angle_count = sub.shape[1]
    max_s = int(sub_s.max().item())
    t = torch.arange(max_s, device=device)
    if _runtime.args.output_interpolation == "nearest":
        down_index = start[:, None] + torch.arange(compact, device=device)[None, :] * sub_s[:, None] // compact
        compact_vals = component[need].gather(2, down_index[:, None, :].expand(-1, angle_count, -1))
        up_index = (t[None, :] * compact // sub_s[:, None]).clamp_max(compact - 1)
        restored = compact_vals.gather(2, up_index[:, None, :].expand(-1, angle_count, -1))
    elif _runtime.args.output_interpolation == "linear":
        # 阶段1：s→Kp（对齐角点线性重采样），所有像素并行 [n, Kp]
        x_norm = torch.linspace(0.0, 1.0, compact, device=device)
        x = x_norm.unsqueeze(0) * (sub_s - 1).unsqueeze(1).to(dtype)
        abs_x = start.unsqueeze(1).to(dtype) + x
        left = abs_x.floor().long().clamp_max(channels - 1)
        right = (left + 1).clamp_max(channels - 1)
        frac = _quantize_interpolation_fraction(abs_x - left.to(dtype))
        left_all = left[:, None, :].expand(-1, angle_count, -1)
        right_all = right[:, None, :].expand(-1, angle_count, -1)
        frac_all = frac[:, None, :]
        compact_vals = sub.gather(2, left_all) * (1.0 - frac_all) + sub.gather(2, right_all) * frac_all
        # 阶段2：Kp→s（每像素变长，按最大孔径补齐后掩码） [n, max_s]
        y = t.unsqueeze(0).to(dtype) / (sub_s - 1).unsqueeze(1).to(dtype).clamp_min(1.0) * (compact - 1)
        c_left = y.floor().long().clamp_max(compact - 1)
        c_right = (c_left + 1).clamp_max(compact - 1)
        frac_y = _quantize_interpolation_fraction(y - c_left.to(dtype))
        c_left_all = c_left[:, None, :].expand(-1, angle_count, -1)
        c_right_all = c_right[:, None, :].expand(-1, angle_count, -1)
        frac_y_all = frac_y[:, None, :]
        restored = (
            compact_vals.gather(2, c_left_all) * (1.0 - frac_y_all)
            + compact_vals.gather(2, c_right_all) * frac_y_all
        )
    else:
        x_norm = torch.linspace(0.0, 1.0, compact, device=device, dtype=dtype)
        x = x_norm.unsqueeze(0) * (sub_s - 1).unsqueeze(1).to(dtype)
        down_indices, down_coefficients = _interpolation_taps(x, sub_s, "cubic")
        compact_vals = torch.zeros((n, angle_count, compact), device=device, dtype=dtype)
        for tap in range(4):
            local_index = down_indices[..., tap]
            absolute_index = start[:, None] + local_index
            gathered = sub.gather(2, absolute_index[:, None, :].expand(-1, angle_count, -1))
            compact_vals = compact_vals + gathered * down_coefficients[:, None, :, tap]
        y = (
            t.unsqueeze(0).to(dtype)
            / (sub_s - 1).unsqueeze(1).to(dtype).clamp_min(1.0)
            * (compact - 1)
        ).clamp_max(compact - 1)
        up_indices, up_coefficients = _interpolation_taps(y, compact, "cubic")
        restored = torch.zeros((n, angle_count, max_s), device=device, dtype=dtype)
        for tap in range(4):
            gathered = compact_vals.gather(
                2,
                up_indices[..., tap][:, None, :].expand(-1, angle_count, -1),
            )
            restored = restored + gathered * up_coefficients[:, None, :, tap]
    valid = t.unsqueeze(0) < sub_s.unsqueeze(1)
    restored = restored * valid[:, None, :].to(dtype)
    out_pos = start.unsqueeze(1).long() + t.unsqueeze(0).long()
    global_row = torch.nonzero(need, as_tuple=False).flatten()
    row_idx = global_row[:, None, None].expand(n, angle_count, max_s)
    angle_idx = torch.arange(angle_count, device=device).view(1, -1, 1).expand(n, -1, max_s)
    out_pos = out_pos[:, None, :].expand(n, angle_count, max_s)
    valid_flat = valid[:, None, :].expand(n, angle_count, max_s)
    projected.index_put_(
        (row_idx[valid_flat], angle_idx[valid_flat], out_pos[valid_flat]),
        restored[valid_flat].to(projected.dtype),
    )
    return projected


def project_active_aperture_weights(
    packed_weights: torch.Tensor,
    aperture_size: torch.Tensor,
    controls: int,
) -> torch.Tensor:
    """在每个像素的有效局部孔径内执行 M→Kp→M 投影。"""
    if controls <= 0:
        return packed_weights
    components = (
        torch.chunk(packed_weights, 2, dim=-1) if _runtime.args.output_weights == "complex" else (packed_weights,)
    )
    projected_components = []
    for component in components:
        projected = _project_active_aperture_vectorized(component, aperture_size, controls)
        projected_components.append(projected)
    return torch.cat(projected_components, dim=-1)


def split_complex_weights(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return real/imaginary weights; real-weight configurations use zero imaginary weights."""
    if _runtime.args.output_weights == "real":
        return weights, torch.zeros_like(weights)
    if weights.shape[-1] % 2:
        raise ValueError(f"复权重最后一维必须为 2N，实际为 {weights.shape[-1]}")
    return torch.chunk(weights, 2, dim=-1)


def effective_aperture_weights(weights: torch.Tensor) -> torch.Tensor:
    if _runtime.args.output_domain != "nonnegative" or _runtime.args.unity_constraint != "hard":
        return weights
    wr, _ = split_complex_weights(weights)
    if getattr(_runtime.args, "unity_scope", "global") == "per_angle":
        weight_sum = wr.float().sum(dim=2, keepdim=True).clamp_min(1.0e-12)
    else:
        weight_sum = wr.float().sum(dim=(1, 2), keepdim=True).clamp_min(1.0e-12)
    return weights / weight_sum


def _factorized_interpolation_map(
    control_count: int,
    batch_size: int,
    channels: int,
    aperture_start: torch.Tensor | None,
    aperture_size: torch.Tensor | None,
    network_channels: int | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if control_count <= 0 or channels <= 0:
        raise ValueError("control_count 和 channels 必须为正数")
    if aperture_size is None:
        positions = torch.arange(channels, device=device, dtype=torch.long)
        valid = torch.ones((batch_size, channels), device=device, dtype=torch.bool)
        if _runtime.args.output_interpolation == "nearest":
            indices = (positions * control_count // channels).clamp_max(control_count - 1)
            indices = indices.view(1, -1, 1).expand(batch_size, -1, -1)
            coefficients = torch.ones_like(indices, dtype=dtype)
        else:
            coordinates = torch.linspace(0.0, float(control_count - 1), channels, device=device, dtype=dtype)
            coordinates = coordinates.view(1, -1).expand(batch_size, -1)
            indices, coefficients = _interpolation_taps(
                coordinates, control_count, _runtime.args.output_interpolation
            )
        return indices, coefficients, valid
    if aperture_start is None or network_channels is None:
        raise ValueError("factorized 动态孔径必须提供 aperture_start")
    starts = aperture_start.to(device=device, dtype=torch.long).reshape(-1)
    sizes = aperture_size.to(device=device, dtype=torch.long).reshape(-1).clamp_min(1)
    if starts.numel() != batch_size or sizes.numel() != batch_size:
        raise ValueError("动态孔径参数长度必须匹配 controls 的 batch 维")
    if bool((sizes > network_channels).any().item()):
        raise ValueError("动态孔径不能超过 network_channels")
    physical = torch.arange(channels, device=device).view(1, -1)
    relative = physical - starts[:, None]
    valid = (relative >= 0) & (relative < sizes[:, None])
    if control_count == network_channels:
        indices = ((network_channels - sizes[:, None]) // 2 + relative).clamp(0, control_count - 1)
        return indices.unsqueeze(-1), torch.ones((*indices.shape, 1), device=device, dtype=dtype), valid
    coordinates = _dynamic_interpolation_coordinates(
        relative, sizes, control_count, dtype
    )
    if _runtime.args.output_interpolation == "nearest":
        indices = coordinates.round().long().clamp(0, control_count - 1)
        indices = indices.unsqueeze(-1)
        coefficients = torch.ones_like(indices, dtype=dtype)
    else:
        indices, coefficients = _interpolation_taps(
            coordinates, control_count, _runtime.args.output_interpolation
        )
    return indices, coefficients, valid


def _validate_factorized_direct_mode() -> None:
    if (
        getattr(_runtime.args, "beamforming_implementation", "explicit") == "factorized"
        and int(getattr(_runtime.args, "projection_controls", 0)) > 0
    ):
        raise ValueError("F/factorized 仅支持直接 output_controls；projection_controls 请使用 E/explicit")


def _factorized_dynamic_weighted_terms(
    components: tuple[torch.Tensor, ...],
    iq_i: torch.Tensor,
    iq_q: torch.Tensor,
    indices: torch.Tensor,
    coefficients: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, angle_count, _ = components[0].shape
    tap_count = indices.shape[-1]
    gather_index = indices.reshape(batch_size, -1).unsqueeze(1).expand(-1, angle_count, -1)
    active = valid[:, None, :, None].to(iq_i.dtype)
    weights = active * coefficients[:, None, :, :]

    control_r = components[0].gather(-1, gather_index).reshape(batch_size, angle_count, -1, tap_count)
    weighted_r = weights * control_r
    i_angle = torch.sum(iq_i.unsqueeze(-1) * weighted_r, dim=(-2, -1))
    q_angle = torch.sum(iq_q.unsqueeze(-1) * weighted_r, dim=(-2, -1))
    wr_sum = torch.sum(weighted_r, dim=(-2, -1))

    if len(components) == 2:
        control_i = components[1].gather(-1, gather_index).reshape(batch_size, angle_count, -1, tap_count)
        weighted_i = weights * control_i
        i_angle = i_angle + torch.sum(iq_q.unsqueeze(-1) * weighted_i, dim=(-2, -1))
        q_angle = q_angle - torch.sum(iq_i.unsqueeze(-1) * weighted_i, dim=(-2, -1))
        wi_sum = torch.sum(weighted_i, dim=(-2, -1))
    else:
        wi_sum = torch.zeros_like(wr_sum)
    return i_angle, q_angle, wr_sum, wi_sum


def _factorized_weighted_iq(
    controls: torch.Tensor,
    iq_i: torch.Tensor,
    iq_q: torch.Tensor,
    aperture_start: torch.Tensor | None,
    aperture_size: torch.Tensor | None,
    network_channels: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    components = torch.chunk(controls, 2, dim=-1) if _runtime.args.output_weights == "complex" else (controls,)
    if aperture_size is None:
        if _runtime.args.output_interpolation == "nearest":
            basis = _nearest_interpolation_basis(
                components[0].shape[-1], iq_i.shape[-1], iq_i.device, iq_i.dtype
            )
        elif _runtime.args.output_interpolation == "linear":
            basis = _linear_interpolation_basis(
                components[0].shape[-1], iq_i.shape[-1], iq_i.device, iq_i.dtype
            )
        else:
            basis = _cubic_interpolation_basis(
                components[0].shape[-1], iq_i.shape[-1], iq_i.device, iq_i.dtype
            )
        batch_size, angle_count, channels = iq_i.shape
        basis_t = basis.transpose(0, 1).contiguous()
        basis_i = torch.mm(iq_i.reshape(-1, channels), basis_t).reshape(batch_size, angle_count, -1)
        basis_q = torch.mm(iq_q.reshape(-1, channels), basis_t).reshape(batch_size, angle_count, -1)
        basis_sum = basis.sum(dim=1).view(1, 1, -1)
        control_r = components[0]
        wr_sum = torch.sum(control_r * basis_sum, dim=-1)
        active_count = None
        if _runtime.args.output_domain == "signed" and _runtime.args.unity_constraint == "hard":
            if getattr(_runtime.args, "unity_scope", "global") == "per_angle":
                active_count = controls.new_full((batch_size, angle_count), float(channels))
                correction_r = (1.0 - wr_sum) / active_count
            else:
                active_count = controls.new_full((batch_size,), float(channels * angle_count))
                correction_r = (1.0 - wr_sum.sum(dim=1)) / active_count
            i_angle = torch.sum(control_r * basis_i, dim=-1)
            q_angle = torch.sum(control_r * basis_q, dim=-1)
            active_i = iq_i.sum(dim=-1)
            active_q = iq_q.sum(dim=-1)
            correction_angle = correction_r if correction_r.ndim == 2 else correction_r[:, None]
            i_angle = i_angle + correction_angle * active_i
            q_angle = q_angle + correction_angle * active_q
            wr_sum = wr_sum + correction_angle * channels
        else:
            i_angle = torch.sum(control_r * basis_i, dim=-1)
            q_angle = torch.sum(control_r * basis_q, dim=-1)
        if len(components) == 2:
            control_i = components[1]
            wi_sum = torch.sum(control_i * basis_sum, dim=-1)
            if active_count is not None:
                if getattr(_runtime.args, "unity_scope", "global") == "per_angle":
                    correction_i = -wi_sum / active_count
                else:
                    correction_i = -wi_sum.sum(dim=1) / active_count
                correction_angle = correction_i if correction_i.ndim == 2 else correction_i[:, None]
                i_angle = i_angle + correction_angle * active_q
                q_angle = q_angle - correction_angle * active_i
                wi_sum = wi_sum + correction_angle * channels
            i_angle = i_angle + torch.sum(control_i * basis_q, dim=-1)
            q_angle = q_angle - torch.sum(control_i * basis_i, dim=-1)
        else:
            wi_sum = torch.zeros_like(wr_sum)
        return i_angle, q_angle, wr_sum, wi_sum
    control_count = components[0].shape[-1]
    indices, coefficients, valid = _factorized_interpolation_map(
        control_count,
        controls.shape[0],
        iq_i.shape[-1],
        aperture_start,
        aperture_size,
        network_channels,
        iq_i.device,
        iq_i.dtype,
    )
    active_angle = valid.unsqueeze(1).to(iq_i.dtype)
    i_angle, q_angle, wr_sum, wi_sum = _factorized_dynamic_weighted_terms(
        components, iq_i, iq_q, indices, coefficients, valid
    )
    active_count = None
    if _runtime.args.output_domain == "signed" and _runtime.args.unity_constraint == "hard":
        active_channels = valid.sum(dim=-1).to(controls.dtype).clamp_min(1.0)
        if getattr(_runtime.args, "unity_scope", "global") == "per_angle":
            active_count = active_channels[:, None]
            correction_r = (1.0 - wr_sum) / active_count
        else:
            active_count = (active_channels * controls.shape[1]).clamp_min(1.0)
            correction_r = (1.0 - wr_sum.sum(dim=1)) / active_count
        active_i = torch.sum(iq_i * active_angle, dim=-1)
        active_q = torch.sum(iq_q * active_angle, dim=-1)
        correction_angle = correction_r if correction_r.ndim == 2 else correction_r[:, None]
        i_angle = i_angle + correction_angle * active_i
        q_angle = q_angle + correction_angle * active_q
        wr_sum = wr_sum + correction_angle * active_channels[:, None]
    if len(components) == 2:
        if active_count is not None:
            if getattr(_runtime.args, "unity_scope", "global") == "per_angle":
                correction_i = -wi_sum / active_count
            else:
                correction_i = -wi_sum.sum(dim=1) / active_count
            correction_angle = correction_i if correction_i.ndim == 2 else correction_i[:, None]
            i_angle = i_angle + correction_angle * active_q
            q_angle = q_angle - correction_angle * active_i
            wi_sum = wi_sum + correction_angle * active_channels[:, None]
    else:
        wi_sum = torch.zeros_like(wr_sum)
    return i_angle, q_angle, wr_sum, wi_sum


def beamform_iq_with_tx_phase(
    weights: torch.Tensor | None,
    iq_i: torch.Tensor,
    iq_q: torch.Tensor,
    tx_tof: torch.Tensor,
    input_scale: torch.Tensor,
    input_mean: torch.Tensor,
    controls: torch.Tensor | None = None,
    aperture_start: torch.Tensor | None = None,
    aperture_size: torch.Tensor | None = None,
    network_channels: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对 IQ 完成权重加权和角度相干合成。"""
    implementation = getattr(_runtime.args, "beamforming_implementation", "explicit")
    if implementation == "factorized":
        _validate_factorized_direct_mode()
        if controls is None:
            raise ValueError("factorized 波束合成必须提供量化后的 controls")
        i_angle, q_angle, wr_sum, wi_sum = _factorized_weighted_iq(
            controls,
            iq_i,
            iq_q,
            aperture_start,
            aperture_size,
            network_channels,
        )
    elif implementation == "explicit":
        if weights is None:
            raise ValueError("explicit 波束合成必须提供 materialized weights")
        wr, wi = split_complex_weights(weights)
        i_angle = torch.sum(wr * iq_i + wi * iq_q, dim=2)
        q_angle = torch.sum(wr * iq_q - wi * iq_i, dim=2)
        wr_sum = wr.sum(dim=2)
        wi_sum = wi.sum(dim=2)
    else:
        raise ValueError(f"不支持的 beamforming_implementation: {implementation}")
    scale = input_scale.squeeze(-1)
    mean = input_mean.squeeze(-1)
    i_angle = scale * i_angle + mean * (wr_sum + wi_sum)
    q_angle = scale * q_angle + mean * (wr_sum - wi_sum)
    per_angle_nonnegative = (
        _runtime.args.output_domain == "nonnegative"
        and _runtime.args.unity_constraint == "hard"
        and getattr(_runtime.args, "unity_scope", "global") == "per_angle"
    )
    if per_angle_nonnegative:
        weight_sum = wr_sum.float().clamp_min(1.0e-12)
        i_angle = i_angle.float() / weight_sum
        q_angle = q_angle.float() / weight_sum
    tx_phase = 2.0 * np.pi * _runtime.fc_global * tx_tof
    cos_tx, sin_tx = torch.cos(tx_phase), torch.sin(tx_phase)
    output_i = torch.sum(i_angle * cos_tx - q_angle * sin_tx, dim=1)
    output_q = torch.sum(i_angle * sin_tx + q_angle * cos_tx, dim=1)
    if getattr(_runtime.args, "angle_reduction", "sum") == "mean":
        output_i = output_i / max(i_angle.shape[1], 1)
        output_q = output_q / max(i_angle.shape[1], 1)
    if (
        _runtime.args.output_domain == "nonnegative"
        and _runtime.args.unity_constraint == "hard"
        and not per_angle_nonnegative
    ):
        weight_sum = wr_sum.float().sum(dim=1).clamp_min(1.0e-12)
        output_i = output_i.float() / weight_sum
        output_q = output_q.float() / weight_sum
    return output_i, output_q


def mask_iq_to_aperture(
    I_a: torch.Tensor,
    Q_a: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mask is None:
        return I_a, Q_a
    mask_f = mask.to(device=I_a.device, dtype=I_a.dtype).unsqueeze(1)
    return I_a * mask_f, Q_a * mask_f


def mask_weights_to_aperture(weights: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mask inactive weights; signed-hard keeps its affine unity projection."""
    if _runtime.args.output_weights == "real":
        active = None if mask is None else mask.to(device=weights.device, dtype=weights.dtype).unsqueeze(1)
        wr = weights if active is None else weights * active
        if _runtime.args.output_domain != "signed" or _runtime.args.unity_constraint != "hard":
            return wr
        if active is None:
            active = torch.ones_like(wr)
        if getattr(_runtime.args, "unity_scope", "global") == "per_angle":
            active_count = active.sum(dim=2, keepdim=True).clamp_min(1.0)
            real_sum = wr.sum(dim=2, keepdim=True)
        else:
            active_count = (active.sum(dim=2, keepdim=True) * wr.shape[1]).clamp_min(1.0)
            real_sum = wr.sum(dim=(1, 2), keepdim=True)
        return active * (1.0 / active_count + wr - real_sum / active_count)

    n_channels = weights.shape[-1] // 2
    if mask is None:
        active = torch.ones((*weights.shape[:-1], n_channels), device=weights.device, dtype=weights.dtype)
    else:
        active = mask.to(device=weights.device, dtype=weights.dtype).unsqueeze(1)
    wr, wi = split_complex_weights(weights)
    wr, wi = wr * active, wi * active
    if _runtime.args.output_domain == "signed" and _runtime.args.unity_constraint == "hard":
        if getattr(_runtime.args, "unity_scope", "global") == "per_angle":
            active_count = active.sum(dim=2, keepdim=True).clamp_min(1.0)
            real_sum = wr.sum(dim=2, keepdim=True)
            imag_sum = wi.sum(dim=2, keepdim=True)
        else:
            active_count = (active.sum(dim=2, keepdim=True) * weights.shape[1]).clamp_min(1.0)
            real_sum = wr.sum(dim=(1, 2), keepdim=True)
            imag_sum = wi.sum(dim=(1, 2), keepdim=True)
        wr = active * (1.0 / active_count + wr - real_sum / active_count)
        wi = active * (wi - imag_sum / active_count)
    return wr if _runtime.args.output_weights == "real" else torch.cat((wr, wi), dim=-1)


def _dynamic_pack_indices(
    aperture_start: torch.Tensor,
    aperture_size: torch.Tensor,
    network_channels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    local = torch.arange(network_channels, device=aperture_size.device).view(1, -1)
    packed_start = (network_channels - aperture_size).view(-1, 1) // 2
    source = local - packed_start + aperture_start.view(-1, 1)
    active = (local >= packed_start) & (local < packed_start + aperture_size.view(-1, 1))
    return source, active


def pack_dynamic_input(
    values: torch.Tensor,
    aperture_start: torch.Tensor,
    aperture_size: torch.Tensor,
    network_channels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    source, active = _dynamic_pack_indices(aperture_start, aperture_size, network_channels)
    return _pack_dynamic_input(values, source, active, values.shape[-1])


def _pack_dynamic_input(
    values: torch.Tensor,
    source: torch.Tensor,
    active: torch.Tensor,
    physical_channels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    source = source.clamp(0, physical_channels - 1).unsqueeze(1).expand(-1, values.shape[1], -1)
    active = active.unsqueeze(1)
    return values.gather(2, source) * active.to(values.dtype), active[:, 0]


def unpack_dynamic_weights(
    weights: torch.Tensor,
    aperture_start: torch.Tensor,
    aperture_size: torch.Tensor,
    physical_channels: int,
) -> torch.Tensor:
    components = torch.chunk(weights, 2, dim=-1) if _runtime.args.output_weights == "complex" else (weights,)
    network_channels = components[0].shape[-1]
    physical = torch.arange(physical_channels, device=weights.device).view(1, 1, -1)
    packed_start = (network_channels - aperture_size) // 2
    source = physical - aperture_start.view(-1, 1, 1) + packed_start.view(-1, 1, 1)
    source = source.clamp(0, network_channels - 1).expand(-1, weights.shape[1], -1)
    active = (physical >= aperture_start.view(-1, 1, 1)) & (
        physical < (aperture_start + aperture_size).view(-1, 1, 1)
    )
    unpacked = []
    for component in components:
        unpacked.append(component.gather(2, source) * active.to(component.dtype))
    return torch.cat(unpacked, dim=-1)


def predict_aperture_weights(
    model: nn.Module,
    i_aligned: torch.Tensor,
    q_aligned: torch.Tensor,
    mask: torch.Tensor | None,
    aperture_start: torch.Tensor | None = None,
    aperture_size: torch.Tensor | None = None,
    depth_index: torch.Tensor | None = None,
    depth_count: int | None = None,
    dynamic_cache: dict[str, torch.Tensor] | None = None,
    packed_beamforming: bool = False,
    materialize_weights: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    _validate_factorized_direct_mode()
    implementation = getattr(_runtime.args, "beamforming_implementation", "explicit")
    projection_controls = int(getattr(_runtime.args, "projection_controls", 0))
    hardware = getattr(model, "hardware_qat", None)
    output_control_range = model.output_control_range_name if hardware is not None else None
    use_explicit_weights = implementation == "explicit"
    weights: torch.Tensor | None = None
    beam_weights: torch.Tensor | None = None
    packed_fast_path = (
        packed_beamforming
        and not materialize_weights
        and implementation == "explicit"
        and mask is not None
        and _runtime.args.output_domain == "nonnegative"
        and _runtime.args.unity_constraint == "hard"
    )
    if mask is not None:
        if aperture_start is None or aperture_size is None:
            raise ValueError("动态孔径必须提供 aperture_start 和 aperture_size")
        input_source, local_active = _dynamic_pack_indices(
            aperture_start, aperture_size, model.network_channels
        )
        i_model, local_active = _pack_dynamic_input(
            i_aligned, input_source, local_active, i_aligned.shape[-1]
        )
        q_model, _ = _pack_dynamic_input(
            q_aligned, input_source, local_active, q_aligned.shape[-1]
        )
        i_model, q_model, input_scale, input_mean = normalize_input_iq(i_model, q_model, local_active)
        raw_controls = predict_weights(model, i_model, q_model)
        component_count = 2 if _runtime.args.output_weights == "complex" else 1
        control_count = raw_controls.shape[-1] // component_count
        if control_count == model.network_channels:
            active_controls = local_active
        else:
            cached_active = None if dynamic_cache is None else dynamic_cache.get("dynamic_control_active_mask_by_size")
            if (
                cached_active is not None
                and cached_active.ndim == 2
                and cached_active.shape[0] > model.network_channels
                and cached_active.shape[1] == control_count
            ):
                active_controls = cached_active[aperture_size.long()]
            else:
                active_controls = dynamic_control_active_mask(
                    aperture_size,
                    control_count,
                    _runtime.args.output_interpolation,
                )
        if component_count == 2:
            active_controls = torch.cat((active_controls, active_controls), dim=-1)
        controls = normalize_controls_for_adc(
            raw_controls,
            active_controls,
            hardware,
            output_control_range,
        )
        controls = prepare_output_controls(model, controls, depth_index, depth_count, active_controls)
        controls = model.quantize_controls(controls, depth_index=depth_index, depth_count=depth_count)
        if use_explicit_weights:
            packed_weights = expand_dynamic_output_controls(
                controls,
                aperture_size,
                model.network_channels,
                interpolation_cache=dynamic_cache,
            )
            packed_weights = project_active_aperture_weights(packed_weights, aperture_size, projection_controls)
            if packed_fast_path:
                beam_weights = packed_weights
            if materialize_weights or not packed_fast_path:
                candidate_weights = unpack_dynamic_weights(
                    packed_weights, aperture_start, aperture_size, i_aligned.shape[-1]
                )
    else:
        i_model, q_model = i_aligned, q_aligned
        active = torch.ones((i_model.shape[0], i_model.shape[-1]), device=i_model.device, dtype=torch.bool)
        i_model, q_model, input_scale, input_mean = normalize_input_iq(i_model, q_model, active)
        raw_controls = predict_weights(model, i_model, q_model)
        controls = normalize_controls_for_adc(
            raw_controls,
            hardware=hardware,
            control_range_name=output_control_range,
        )
        controls = prepare_output_controls(model, controls, depth_index, depth_count)
        controls = model.quantize_controls(controls, depth_index=depth_index, depth_count=depth_count)
        if use_explicit_weights:
            candidate_weights = expand_fixed_output_controls(controls, model.network_channels)
    if use_explicit_weights:
        if mask is None:
            candidate_weights = project_output_weights(candidate_weights, projection_controls)
        if _runtime.args.output_domain == "signed" and _runtime.args.unity_constraint == "hard":
            candidate_weights = mask_weights_to_aperture(candidate_weights, mask)
        if materialize_weights or not packed_fast_path:
            weights = candidate_weights
    if (
        weights is not None
        and _runtime.args.output_domain == "nonnegative"
        and _runtime.args.unity_constraint == "hard"
        and getattr(model, "hardware_qat", None) is not None
    ):
        wr, _ = split_complex_weights(weights)
        if getattr(_runtime.args, "unity_scope", "global") == "per_angle":
            weight_sums = wr.sum(dim=2)
            diagnostic_depth = depth_index
            if diagnostic_depth is not None and depth_count is not None:
                diagnostic_depth = diagnostic_depth.reshape(-1, 1).expand_as(weight_sums)
        else:
            weight_sums = wr.sum(dim=(1, 2))
            diagnostic_depth = depth_index
        model.hardware_qat.record_weight_sums(
            weight_sums,
            depth_index=diagnostic_depth,
            depth_count=depth_count,
        )
    if packed_fast_path:
        i_use, q_use = i_model, q_model
    else:
        i_use, q_use = mask_iq_to_aperture(
            (i_aligned - input_mean) / input_scale,
            (q_aligned - input_mean) / input_scale,
            mask,
        )
        beam_weights = weights
    if packed_beamforming:
        return i_use, q_use, weights, controls, input_scale, input_mean, raw_controls, beam_weights
    return i_use, q_use, weights, controls, input_scale, input_mean, raw_controls

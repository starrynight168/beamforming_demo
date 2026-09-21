from __future__ import annotations

import dataclasses
import math
import random
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .beamforming import (
    aperture_weight_regularization,
    beamform_iq_with_tx_phase,
    iq_data_loss_components,
    predict_aperture_weights,
    raw_control_range_penalty,
    training_loss,
    unity_loss,
)
from .config import (
    CONFIG_FIELDS,
    QAT_STAGE_OVERRIDE_FIELDS,
    TEST_REALIZATION_SEED_OFFSET,
    _coerce_override,
    resolve_nonideal_profile_raw,
)
from .config import runtime as _runtime
from .data import (
    _amp_enabled,
    compute_aperture_mask,
    extract_aligned_iq_cached,
    extract_windows_on_gpu,
)
from .hardware import QATConfig
from .model import MBAN
from .naming import layer_name, output_preactivation_name

def _validate_checkpoint_semantics(
    checkpoint: dict,
    path: str | Path,
    require_hash: bool = False,
    for_resume: bool = False,
) -> None:
    expected = int(_runtime.MODEL_SEMANTICS_VERSION)
    actual = checkpoint.get("model_semantics_version")
    if actual != expected:
        raise ValueError(
            f"{path} 的 model_semantics_version={actual!r}，当前要求 {expected!r}；"
            "请使用当前代码重新训练或设置 resume=none"
        )
    if require_hash and checkpoint.get("semantic_hash") != _runtime.SEMANTIC_HASH:
        raise ValueError(
            f"{path} 的 semantic_hash={checkpoint.get('semantic_hash')!r}，"
            f"当前配置要求 {_runtime.SEMANTIC_HASH!r}；请使用 resume=weights 或新输出目录"
        )
    if for_resume and checkpoint.get("training_resume_allowed") is False:
        raise ValueError(f"{path} 是派生 checkpoint，不能用于训练恢复；请使用原始训练 checkpoint")


def _validate_training_checkpoint(
    checkpoint: object,
    path: str | Path,
    required_fields: tuple[str, ...] = (),
    require_semantic_hash: bool = False,
    for_resume: bool = False,
    expected_network_channels: int | None = None,
    expected_output_controls: int | None = None,
    expected_geometry: dict[str, float] | None = None,
) -> None:
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path} 不是有效训练检查点")
    _validate_checkpoint_semantics(checkpoint, path, require_hash=require_semantic_hash, for_resume=for_resume)
    required = {"model_state_dict", "model_config", *required_fields}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"{path} 缺少当前检查点字段: {', '.join(missing)}")
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError(f"{path} 的 model_config 必须是映射")
    checks = (
        ("network_channels", expected_network_channels),
        ("output_controls", expected_output_controls),
    )
    for key, expected_value in checks:
        if expected_value is None:
            continue
        actual_value = model_config.get(key)
        if actual_value is None or int(actual_value) != int(expected_value):
            raise ValueError(
                f"{path} 的 model_config.{key}={actual_value!r} 与当前配置的 {int(expected_value)} 不一致"
            )
    if expected_geometry is not None:
        geometry = checkpoint.get("geometry_config")
        if not isinstance(geometry, dict):
            raise ValueError(f"{path} 缺少 geometry_config")
        for key, expected_value in expected_geometry.items():
            actual_value = geometry.get(key)
            if actual_value is None or not math.isclose(
                float(actual_value), float(expected_value), rel_tol=1.0e-6, abs_tol=1.0e-9
            ):
                raise ValueError(
                    f"{path} 的 geometry_config.{key}={actual_value!r} 与当前配置的 {expected_value!r} 不一致"
                )


# =====================================================================
# MBAN 模型
# =====================================================================


def _set_geometry(meta) -> tuple[int, int, int, float, float, float]:
    values = {
        "IMG_H": meta.gt_height,
        "IMG_W": meta.gt_width,
        "NUM_PIXELS": meta.gt_height * meta.gt_width,
        "NUM_CHANNELS": meta.num_channels,
        "c_global": meta.c,
        "fc_global": meta.fc,
        "fs_global": meta.fs,
    }
    for name, value in values.items():
        setattr(_runtime, name, value)
    return (
        values["IMG_H"],
        values["IMG_W"],
        values["NUM_PIXELS"],
        values["c_global"],
        values["fc_global"],
        values["fs_global"],
    )


def effective_input_complex_features(network_channels: int) -> int:
    return network_channels


def checkpoint_metadata(
    network_channels: int,
    physical_channels: int,
    output_controls: int,
    max_depth: float,
    pitch: float,
) -> dict:
    input_features = effective_input_complex_features(network_channels)
    input_real_features = 2 * input_features + int(
        _runtime.args.bias_implementation == "array" and "fc1" in _runtime.args.bias_layers
    )
    tile_passes = (
        1
        if _runtime.args.input_tile_features <= 0
        else math.ceil(input_real_features / _runtime.args.input_tile_features)
    )
    return {
        "checkpoint_identity": {
            "family": "MBAN",
            "training_mode": "qat" if _runtime.args.mode == "qat" else "fp32",
        },
        "activation": _runtime.args.activation,
        "model_semantics_version": _runtime.MODEL_SEMANTICS_VERSION,
        "semantic_config": _runtime.SEMANTIC_CONFIG,
        "semantic_hash": _runtime.SEMANTIC_HASH,
        "deployment_folded": (
            _runtime.args.normalization == "none"
            and not _runtime.args.normalization_layers
            and _runtime.args.centering == "none"
            and not _runtime.args.centering_layers
            and _runtime.args.weight_transform == "none"
            and not _runtime.args.weight_transform_layers
        ),
        "deployment_runtime_transforms": {
            "input_normalization": _runtime.args.input_normalization,
            "control_normalization": _runtime.args.control_normalization,
            "input_normalization_folded": False,
            "control_normalization_folded": False,
        },
        "hardware_config": {
            "enabled": bool(_runtime.args.qat_enabled),
            "path": str(_runtime.HARDWARE_CONFIG_PATH) if _runtime.HARDWARE_CONFIG_PATH else None,
            "profile": _runtime.NONIDEAL_PROFILE_NAME,
        },
        "model_config": {
            "network_channels": network_channels,
            "expected_network_channels": network_channels,
            "physical_channels": physical_channels,
            "output_controls": output_controls,
            "hidden_width": _runtime.args.hidden_width,
            "hidden_layers": _runtime.args.hidden_layers,
            "total_fc_layers": _runtime.args.hidden_layers + 1,
            "dropout": _runtime.args.dropout,
            "activation": dict(_runtime.args.activation),
            "branch_mode": _runtime.args.branch_mode,
            "centering": _runtime.args.centering,
            "centering_layers": list(_runtime.args.centering_layers),
            "normalization": _runtime.args.normalization,
            "normalization_layers": list(_runtime.args.normalization_layers),
            "bias_layers": list(_runtime.args.bias_layers),
            "bias_implementation": _runtime.args.bias_implementation,
            "weight_transform": _runtime.args.weight_transform,
            "weight_transform_layers": list(_runtime.args.weight_transform_layers),
            "weight_transform_epsilon": _runtime.args.weight_transform_epsilon,
            "running_stat_momentum": _runtime.args.running_stat_momentum,
            "running_stat_epsilon": _runtime.args.running_stat_epsilon,
            "batch_renorm_rmax": _runtime.args.batch_renorm_rmax,
            "batch_renorm_dmax": _runtime.args.batch_renorm_dmax,
            "input_normalization": _runtime.args.input_normalization,
            "input_tile_features": _runtime.args.input_tile_features,
            "input_complex_features": input_features,
            "input_real_features": input_real_features,
            "input_tile_passes": tile_passes,
        },
        "geometry_config": {
            "f_number": _runtime.args.f_number,
            "max_depth": float(max_depth),
            "pitch": float(pitch),
        },
        "evaluation_config": {
            "target_algorithm": _runtime.args.target_algorithm,
            "output_weights": _runtime.args.output_weights,
            "output_domain": _runtime.args.output_domain,
            "unity_constraint": _runtime.args.unity_constraint,
            "unity_scope": getattr(_runtime.args, "unity_scope", "global"),
            "angle_selection": getattr(_runtime.args, "angle_selection", "center"),
            "angle_reduction": getattr(_runtime.args, "angle_reduction", "sum"),
            "control_normalization": _runtime.args.control_normalization,
            "control_adc_range": _runtime.args.control_adc_range,
            "control_adc_full_scale": _runtime.args.control_adc_full_scale,
            "beamforming_implementation": "explicit",
            "dynamic_aperture": _runtime.args.dynamic_aperture,
            "f_number": _runtime.args.f_number,
            "input_normalization": _runtime.args.input_normalization,
            "projection_controls": _runtime.args.projection_controls,
            "output_interpolation": _runtime.args.output_interpolation,
            "interpolation_bits": _runtime.args.interpolation_bits,
            "use_tgc": _runtime.args.use_tgc,
        },
    }


def hardware_checkpoint_state(model: nn.Module) -> dict:
    hardware = getattr(model, "hardware_qat", None)
    return {"hardware_qat_state": hardware.checkpoint_state_dict()} if hardware is not None else {}


def training_rng_state(loader_generator: torch.Generator) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "train_loader": loader_generator.get_state(),
    }


def restore_training_rng_state(state: dict | None, loader_generator: torch.Generator) -> None:
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch_state = state["torch"]
        if isinstance(torch_state, torch.Tensor):
            torch_state = torch_state.detach().cpu()
            if torch_state.dtype != torch.uint8:
                torch_state = torch_state.to(torch.uint8)
        torch.set_rng_state(torch_state)
    if state.get("cuda") is not None and torch.cuda.is_available():
        cuda_states = []
        for device_state in state["cuda"]:
            if isinstance(device_state, torch.Tensor):
                device_state = device_state.detach().cpu()
                if device_state.dtype != torch.uint8:
                    device_state = device_state.to(torch.uint8)
            cuda_states.append(device_state)
        torch.cuda.set_rng_state_all(cuda_states)
    if state.get("train_loader") is not None:
        loader_state = state["train_loader"]
        if isinstance(loader_state, torch.Tensor):
            loader_state = loader_state.detach().cpu()
            if loader_state.dtype != torch.uint8:
                loader_state = loader_state.to(torch.uint8)
        loader_generator.set_state(loader_state)


def _qat_stage_for_epoch(epoch: int) -> tuple[int, dict[str, object]]:
    schedule = getattr(_runtime.args, "qat_schedule", {}) or {}
    stages = schedule.get("stages", []) if isinstance(schedule, dict) else []
    if not schedule.get("enabled", False) or not stages:
        return -1, {"name": "config", "profile": _runtime.NONIDEAL_PROFILE_NAME}
    if not isinstance(stages, list):
        raise ValueError("qat_schedule.stages 必须是列表")
    remaining = int(epoch)
    last = None
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict) or not stage.get("name") or not stage.get("profile"):
            raise ValueError("每个 QAT stage 必须包含 name、epochs、profile")
        duration = int(stage.get("epochs", 0))
        if duration <= 0:
            raise ValueError("QAT stage 的 epochs 必须大于 0")
        stage_overrides = stage.get("overrides", {}) or {}
        if not isinstance(stage_overrides, dict):
            raise ValueError("QAT stage.overrides 必须是映射")
        unknown_stage_overrides = set(stage_overrides) - QAT_STAGE_OVERRIDE_FIELDS
        if unknown_stage_overrides:
            raise ValueError(f"QAT stage.overrides 包含不允许的字段: {sorted(unknown_stage_overrides)}")
        last = (index, stage)
        if remaining < duration:
            return index, stage
        remaining -= duration
    return last


def apply_qat_stage(model: nn.Module, epoch: int) -> None:
    hardware = getattr(model, "hardware_qat", None)
    if hardware is None:
        return
    stage_index, stage = _qat_stage_for_epoch(epoch)
    profile_name = str(stage.get("profile", _runtime.NONIDEAL_PROFILE_NAME))
    stage_name = str(stage.get("name", "config"))
    profile_config = getattr(_runtime.args, "_nonideal_profile_config", None)
    if not isinstance(profile_config, dict):
        raise ValueError("QAT阶段需要 hardware_eval.yaml 的 nonidealities 配置")
    values = vars(_runtime.args).copy()
    profile_overrides = resolve_nonideal_profile_raw(
        profile_config,
        profile_name,
        str(values["inter_layer"]),
        str(values["bias_implementation"]),
    )
    base_nonideal_values = getattr(_runtime.args, "_nonideal_base_values", None)
    if not isinstance(base_nonideal_values, dict):
        raise ValueError("QAT阶段缺少非理想基线配置")
    values.update(base_nonideal_values)
    for item in profile_overrides:
        key, separator, raw_value = item.partition("=")
        if not separator or key not in CONFIG_FIELDS:
            raise ValueError(f"QAT profile 产生了未知字段: {key!r}")
        values[key] = _coerce_override(key, raw_value.strip(), values[key])
    for key, raw_value in (stage.get("overrides", {}) or {}).items():
        if key not in QAT_STAGE_OVERRIDE_FIELDS:
            raise ValueError(f"QAT stage override 产生了未知字段: {key!r}")
        values[key] = _coerce_override(key, str(raw_value), values[key])
    values["mode"] = "qat"
    values["qat_enabled"] = True
    values["qat_mode"] = "qat"
    stage_config = QATConfig.from_namespace(SimpleNamespace(**values))
    if hardware.profile_name == profile_name and hardware.stage_name == stage_name and hardware.config == stage_config:
        return
    hardware.set_config(stage_config, profile_name, stage_name)
    _runtime.write_log(f"QAT stage {stage_index}: {stage_name} | stage_profile={profile_name}", level="INFO")


def build_learning_rate_scheduler(optimizer: optim.Optimizer, steps_per_epoch: int):
    """Create the selected scheduler; ``none`` deliberately returns no scheduler."""
    if _runtime.args.scheduler == "none":
        return None
    if _runtime.args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=_runtime.args.cosine_restart_epochs, T_mult=1, eta_min=1e-8
        )
    if _runtime.args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=_runtime.args.scheduler_step_epochs, gamma=_runtime.args.scheduler_gamma
        )
    if _runtime.args.scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=_runtime.args.learning_rate,
            epochs=_runtime.args.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=_runtime.args.onecycle_pct_start,
            final_div_factor=1e4,
        )
    if _runtime.args.scheduler == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=_runtime.args.scheduler_milestones, gamma=_runtime.args.scheduler_gamma
        )
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=_runtime.args.scheduler_gamma,
        patience=_runtime.args.scheduler_patience,
        min_lr=1e-8,
    )


def restore_learning_rate_scheduler(scheduler, state: dict | None) -> None:
    if scheduler is None or not state:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
        saved_total_steps = int(state.get("total_steps", scheduler.total_steps))
        if saved_total_steps != scheduler.total_steps:
            saved_step = int(state.get("last_epoch", -1))
            if scheduler.total_steps <= saved_step:
                raise ValueError(
                    f"OneCycleLR 新总步数 {scheduler.total_steps} 不足以恢复到第 {saved_step} 步；"
                    "请增加 epochs 或使用 resume=weights"
                )
            schedule_phases = scheduler._schedule_phases
            scheduler.load_state_dict(
                {
                    key: value
                    for key, value in state.items()
                    if key not in {"total_steps", "_schedule_phases", "_last_lr"}
                }
            )
            scheduler._schedule_phases = schedule_phases
            get_lr_called = scheduler._get_lr_called_within_step
            scheduler._get_lr_called_within_step = True
            try:
                current_lrs = scheduler.get_lr()
            finally:
                scheduler._get_lr_called_within_step = get_lr_called
            for group, lr in zip(scheduler.optimizer.param_groups, current_lrs, strict=True):
                group["lr"] = lr
            scheduler._last_lr = list(current_lrs)
            return
    scheduler.load_state_dict(state)


def _count_nonfinite_tensors(
    tensors: Iterable[torch.Tensor],
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, int]:
    total = 0
    nonfinite: torch.Tensor | None = None
    for tensor in tensors:
        total += tensor.numel()
        count = (~torch.isfinite(tensor)).sum(dtype=torch.int64)
        nonfinite = count if nonfinite is None else nonfinite + count
    if nonfinite is None:
        nonfinite = torch.zeros((), dtype=torch.int64, device=device)
    return nonfinite, total


def count_nonfinite_gradients(
    parameters,
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, int]:
    """返回设备端的 (非有限梯度元素数, 梯度元素总数)。"""
    return _count_nonfinite_tensors(
        (parameter.grad for parameter in parameters if parameter.grad is not None),
        device=device,
    )


def count_nonfinite_parameters(
    parameters,
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, int]:
    """返回设备端的 (非有限参数元素数, 参数元素总数)。"""
    return _count_nonfinite_tensors(
        (parameter.detach() for parameter in parameters),
        device=device,
    )


def qat_parameter_diagnostics(model: nn.Module) -> dict[str, dict[str, float | None]]:
    hardware = getattr(model, "hardware_qat", None)
    if hardware is None:
        return {}
    result = {}
    with torch.no_grad():
        for name in sorted(getattr(model, "bias_layers", ())):
            layer = getattr(model, name)
            weight = model._effective_weight(name, layer.weight)
            implementation = getattr(layer, "bias_implementation", "ordinary")
            quantization_range = (
                output_preactivation_name(layer_name(model.total_fc_layers))
                if implementation == "digital" and name == layer_name(model.total_fc_layers)
                else None
            )
            result[name] = hardware.parameter_diagnostics(
                weight,
                name,
                implementation,
                layer.bias,
                quantization_range=quantization_range,
            )
    return result


# =====================================================================
# 当前 EPFL-pack H5 schema
# =====================================================================


def report_activation_diagnostics(model: MBAN) -> None:
    """训练开始前对逐层激活做一次 pre-activation 采样诊断，检测非线性是否退化。

    判定规则（不自动改参，仅输出警告与修复建议）：
    - PWL：tail_ratio = P(|x| > knee)，接近 0 表示几乎未进入尾区（形同 relu）；
      tail_slope 接近 1 时 PWL 即使进入尾区也近似恒等映射，同样视为线性。
    - tanh：mean_local_gain = E[1 - tanh^2(beta*x)]，接近 1 表示 tanh 基本线性。

    注意 dual 模式：采样的是拆分后的 [x+, x-]，每对正负支路各占一半，结构性存在
    约 50% 的零值，因此 q50≈0 属正常；tail_ratio 最大值约为 0.5，请按此口径解读。
    """
    diag = model.end_activation_diagnostics()
    if not diag:
        return
    for layer, stats in diag.items():
        spec = model.activation_specs.get(layer, {})
        spec_type = spec.get("type")
        detail = " | ".join(f"{k}={v:.4f}" for k, v in stats.items())
        if spec_type == "pwl":
            tail = stats.get("tail_ratio", 0.0)
            tail_slope = float(spec.get("tail_slope", 0.2))
            if tail_slope >= 0.99:
                _runtime.write_log(
                    f"激活退化 {layer}: PWL tail_slope={tail_slope:.2f}≈1，近似恒等映射（{detail}）。"
                    f"建议：减小 tail_slope 以保留尾部非线性。",
                    level="WARNING",
                )
            elif tail < 0.01:
                _runtime.write_log(
                    f"激活退化 {layer}: PWL tail_ratio={tail:.4f}≈0，几乎未进入尾区（{detail}）。"
                    f"建议：减小 knee、改用 L2 归一化，或确认 L1 下分量确实超过 knee。",
                    level="WARNING",
                )
            else:
                _runtime.write_log(
                    f"[ACT] {layer} PWL: tail_ratio={tail:.4f}, tail_slope={tail_slope:.2f}（{detail}）",
                    level="INFO",
                )
        elif spec_type == "tanh":
            gain = stats.get("mean_local_gain", 1.0)
            if gain > 0.99:
                _runtime.write_log(
                    f"激活退化 {layer}: tanh mean_local_gain={gain:.4f}≈1，基本线性（{detail}）。"
                    f"建议：增大 beta、改用 L2 归一化，或确认 L1 下分量确实触发饱和。",
                    level="WARNING",
                )
            else:
                _runtime.write_log(f"[ACT] {layer} tanh: mean_local_gain={gain:.4f}（{detail}）", level="INFO")
        elif spec_type == "relu":
            _runtime.write_log(f"[ACT] {layer} relu: {detail}", level="INFO")


def forward_pixel_batch(
    model: nn.Module,
    rf_i: torch.Tensor,
    rf_q: torch.Tensor,
    t_starts: torch.Tensor,
    img_idx: torch.Tensor,
    pixel_idx: torch.Tensor,
    target_i: torch.Tensor,
    target_q: torch.Tensor,
    max_t_len: int,
    valid_time_samples: torch.Tensor,
    geometry_cache_ids: torch.Tensor,
    angle_idx: torch.Tensor,
    offsets: torch.Tensor,
    pitch: float,
    x_grid: torch.Tensor,
    depth_grid: torch.Tensor,
    angles_rad: torch.Tensor,
    distillation_model: nn.Module | None = None,
    geometry_caches: dict[int, dict[str, torch.Tensor] | None] | None = None,
) -> dict[str, torch.Tensor]:
    """Shared train/validation/test forward path for one pixel micro-batch."""
    if geometry_cache_ids.numel() != rf_i.shape[0]:
        raise ValueError("geometry_cache_ids 长度必须与图像批大小一致")

    def align_uncached(selected: torch.Tensor):
        selected_img_idx = img_idx[selected]
        selected_pixel_idx = pixel_idx[selected]
        ext_i, ext_q, tof, tx_tof = extract_windows_on_gpu(
            rf_i,
            rf_q,
            t_starts,
            selected_img_idx,
            selected_pixel_idx,
            _runtime.fs_global,
            offsets,
            angle_idx,
            max_t_len,
            valid_time_samples,
            pitch,
            x_grid,
            depth_grid,
            angles_rad,
        )
        center_t = ext_i.shape[-1] // 2
        i_sample = ext_i[:, :, :, center_t]
        q_sample = ext_q[:, :, :, center_t]
        phase = 2.0 * np.pi * _runtime.fc_global * (tof - tx_tof.unsqueeze(2))
        cos_p, sin_p = torch.cos(phase), torch.sin(phase)
        return i_sample * cos_p - q_sample * sin_p, i_sample * sin_p + q_sample * cos_p, tx_tof

    cache_ids_by_image = [int(value) for value in geometry_cache_ids.reshape(-1).tolist()]
    if not geometry_caches:
        selected_all = torch.arange(pixel_idx.numel(), device=pixel_idx.device)
        i_aligned, q_aligned, tx_tof = align_uncached(selected_all)
    elif len(cache_ids_by_image) == 1:
        cache = geometry_caches.get(cache_ids_by_image[0])
        if cache is None:
            selected_all = torch.arange(pixel_idx.numel(), device=pixel_idx.device)
            i_aligned, q_aligned, tx_tof = align_uncached(selected_all)
        else:
            i_aligned, q_aligned, tx_tof = extract_aligned_iq_cached(
                rf_i,
                rf_q,
                img_idx,
                pixel_idx,
                cache,
            )
    else:
        i_aligned = rf_i.new_empty((pixel_idx.numel(), angle_idx.numel(), rf_i.shape[-1]))
        q_aligned = torch.empty_like(i_aligned)
        tx_tof = rf_i.new_empty((pixel_idx.numel(), angle_idx.numel()))
        uncached_groups = []
        for cache_id in dict.fromkeys(cache_ids_by_image):
            image_indices = [index for index, value in enumerate(cache_ids_by_image) if value == cache_id]
            image_mask = torch.zeros(rf_i.shape[0], dtype=torch.bool, device=img_idx.device)
            image_mask[image_indices] = True
            selected = torch.nonzero(image_mask[img_idx], as_tuple=False).flatten()
            if selected.numel() == 0:
                continue
            cache = geometry_caches.get(cache_id)
            if cache is None:
                uncached_groups.append(selected)
                continue
            aligned_i, aligned_q, selected_tx_tof = extract_aligned_iq_cached(
                rf_i,
                rf_q,
                img_idx[selected],
                pixel_idx[selected],
                cache,
            )
            i_aligned[selected] = aligned_i
            q_aligned[selected] = aligned_q
            tx_tof[selected] = selected_tx_tof
        if uncached_groups:
            selected = torch.cat(uncached_groups)
            aligned_i, aligned_q, selected_tx_tof = align_uncached(selected)
            i_aligned[selected] = aligned_i
            q_aligned[selected] = aligned_q
            tx_tof[selected] = selected_tx_tof
    with torch.amp.autocast("cuda", enabled=_amp_enabled(i_aligned.device)):
        aperture_cache = None
        if _runtime.args.dynamic_aperture:
            aperture_cache = next(
                (cache for cache in (geometry_caches or {}).values() if cache is not None and "mask" in cache),
                None,
            )
            if aperture_cache is None:
                mask, aperture_start, aperture_size = compute_aperture_mask(
                    pixel_idx,
                    depth_grid,
                    x_grid,
                    pitch,
                    return_geometry=True,
                )
            else:
                mask = aperture_cache["mask"][pixel_idx].to(i_aligned.dtype)
                aperture_start = aperture_cache["aperture_start"][pixel_idx]
                aperture_size = aperture_cache["aperture_size"][pixel_idx]
        else:
            mask = aperture_start = aperture_size = None
        loss_terms = _runtime.args.loss["terms"]
        aperture_regularization_enabled = any(
            float(loss_terms[name]["weight"]) > 0.0 for name in ("d1", "d2")
        )
        materialize_weights = (
            _runtime.args.unity_constraint != "hard"
            or aperture_regularization_enabled
            or bool(getattr(getattr(model, "hardware_qat", None), "_diagnostics_enabled", False))
        )
        (
            i_use,
            q_use,
            weights,
            controls,
            input_scale,
            input_mean,
            raw_controls,
            beam_weights,
        ) = predict_aperture_weights(
            model,
            i_aligned,
            q_aligned,
            mask,
            aperture_start,
            aperture_size,
            depth_index=pixel_idx // _runtime.IMG_W,
            depth_count=_runtime.IMG_H,
            dynamic_cache=aperture_cache,
            packed_beamforming=True,
            materialize_weights=materialize_weights,
        )
        pred_i, pred_q = beamform_iq_with_tx_phase(
            beam_weights,
            i_use,
            q_use,
            tx_tof,
            input_scale,
            input_mean,
            controls=controls,
            aperture_start=aperture_start,
            aperture_size=aperture_size,
            network_channels=model.network_channels,
        )

        data_loss, iq_loss, envelope_loss = iq_data_loss_components(pred_i, pred_q, target_i, target_q)
        zero_loss = controls.new_zeros(())
        loss_unity = zero_loss if _runtime.args.unity_constraint == "hard" else unity_loss(weights)
        loss_aperture = (
            aperture_weight_regularization(weights, mask)
            if aperture_regularization_enabled
            else zero_loss
        )
        loss_range = raw_control_range_penalty(raw_controls)
        loss_distillation = data_loss.new_zeros(())
        if distillation_model is not None:
            with torch.no_grad():
                (
                    distill_i_use,
                    distill_q_use,
                    _,
                    distill_controls,
                    distill_scale,
                    distill_mean,
                    _,
                    distill_beam_weights,
                ) = predict_aperture_weights(
                    distillation_model,
                    i_aligned,
                    q_aligned,
                    mask,
                    aperture_start,
                    aperture_size,
                    dynamic_cache=aperture_cache,
                    packed_beamforming=True,
                    materialize_weights=materialize_weights,
                )
            distill_i, distill_q = beamform_iq_with_tx_phase(
                distill_beam_weights,
                distill_i_use,
                distill_q_use,
                tx_tof,
                distill_scale,
                distill_mean,
                controls=distill_controls,
                aperture_start=aperture_start,
                aperture_size=aperture_size,
                network_channels=distillation_model.network_channels,
            )
            loss_distillation = ((pred_i - distill_i).square() + (pred_q - distill_q).square()).mean()
        loss = training_loss(
            data_loss,
            loss_unity,
            loss_aperture,
            loss_range,
            loss_distillation,
        )

    return {
        "loss": loss,
        "data": data_loss,
        "iq": iq_loss,
        "envelope": envelope_loss,
        "unity": loss_unity,
        "aperture": loss_aperture,
        "range": loss_range,
        "distillation": loss_distillation,
        "weights": weights,
    }


TEST_METRIC_NAMES = ("data", "iq", "envelope", "unity", "aperture", "range", "distillation")


def evaluate_loss_components(
    model,
    loader,
    device,
    angle_idx,
    offsets,
    pitch,
    x_grid,
    depth_grid,
    angles_rad,
    distillation_model=None,
    geometry_caches=None,
) -> dict[str, float]:
    """Evaluate the held-out loss and its components on the current hardware realization.

    ``loss`` 与训练期 ``training_loss`` 同口径，已含各损失项系数；其余键是未加系数的分量均值，
    用于区分监督误差（data/iq/envelope）与正则项（range 等）。``pixels`` 为实际累计的像素数。
    """
    model.eval()
    loss_sum, pixel_count = 0.0, 0
    component_sums = dict.fromkeys(TEST_METRIC_NAMES, 0.0)
    with torch.no_grad():
        for rf_imgs_I, rf_imgs_Q, _target_db, target_i, target_q, t_starts, _norm_ref, valid_times, cache_ids in loader:
            b_img = rf_imgs_I.shape[0]
            rf_imgs_I = rf_imgs_I.to(device, non_blocking=True)
            rf_imgs_Q = rf_imgs_Q.to(device, non_blocking=True)
            max_t_len = rf_imgs_I.shape[2]
            target_i_flat = target_i.view(b_img, _runtime.NUM_PIXELS).to(device, non_blocking=True)
            target_q_flat = target_q.view(b_img, _runtime.NUM_PIXELS).to(device, non_blocking=True)
            t_starts = t_starts.to(device, non_blocking=True)
            valid_times = valid_times.to(device, non_blocking=True)
            # 均匀覆盖该批的所有帧，避免测试只落在第 1 帧。
            eval_indices = torch.arange(b_img * _runtime.NUM_PIXELS, device=device)
            for start in range(0, len(eval_indices), _runtime.args.optimization_batch_pixels):
                idx_chunk = eval_indices[start : start + _runtime.args.optimization_batch_pixels]
                img_idx, pixel_idx = idx_chunk // _runtime.NUM_PIXELS, idx_chunk % _runtime.NUM_PIXELS
                result = forward_pixel_batch(
                    model,
                    rf_imgs_I,
                    rf_imgs_Q,
                    t_starts,
                    img_idx,
                    pixel_idx,
                    target_i_flat[img_idx, pixel_idx],
                    target_q_flat[img_idx, pixel_idx],
                    max_t_len,
                    valid_times,
                    cache_ids,
                    angle_idx,
                    offsets,
                    pitch,
                    x_grid,
                    depth_grid,
                    angles_rad,
                    distillation_model,
                    geometry_caches,
                )
                chunk_pixels = len(idx_chunk)
                loss_sum += result["loss"].item() * chunk_pixels
                for name in TEST_METRIC_NAMES:
                    component_sums[name] += result[name].item() * chunk_pixels
                pixel_count += chunk_pixels
    count = max(pixel_count, 1)
    return {
        "loss": loss_sum / count,
        "pixels": pixel_count,
        **{name: value / count for name, value in component_sums.items()},
    }


def prepare_test_realization(model: nn.Module) -> tuple[QATConfig, int] | None:
    hardware = getattr(model, "hardware_qat", None)
    if hardware is None or not hardware.config.noise_enabled:
        return None
    original_config = hardware.config
    test_seed = int(_runtime.args.seed) + TEST_REALIZATION_SEED_OFFSET
    hardware.config = dataclasses.replace(original_config, noise_seed=test_seed)
    hardware.reset_noise_counter()
    hardware.begin_noise_realization()
    return original_config, test_seed


def _training_validation_indices(batch_size: int, device: torch.device) -> torch.Tensor:
    pixels_per_image = int(_runtime.args.validation_pixels_per_image)
    if pixels_per_image == 0:
        pixels_per_image = _runtime.NUM_PIXELS
    pixels_per_image = min(pixels_per_image, _runtime.NUM_PIXELS)
    if pixels_per_image == _runtime.NUM_PIXELS:
        pixel_indices = torch.arange(_runtime.NUM_PIXELS, device=device)
    else:
        pixel_indices = torch.linspace(
            0,
            _runtime.NUM_PIXELS - 1,
            steps=pixels_per_image,
            device=device,
        ).long()
    image_offsets = torch.arange(batch_size, device=device).view(-1, 1) * _runtime.NUM_PIXELS
    return (image_offsets + pixel_indices.view(1, -1)).reshape(-1)


def _collate_single_image(batch):
    if len(batch) != 1:
        raise ValueError("单图零拷贝 collate 只接受 batch_size=1")
    return tuple(value.unsqueeze(0) for value in batch[0])


def _load_distillation_model(network_channels: int, device: torch.device):
    if _runtime.args.distillation_weight <= 0:
        return None
    checkpoint = torch.load(_runtime.args.distillation_checkpoint, map_location=device, weights_only=False)
    _validate_training_checkpoint(checkpoint, _runtime.args.distillation_checkpoint)
    config = checkpoint["model_config"]
    model = MBAN(
        num_channels=network_channels,
        dropout=float(config["dropout"]),
        hidden_width=int(config["hidden_width"]),
        activation=dict(config["activation"]),
        branch_mode=str(config["branch_mode"]),
        hidden_layers=int(config["hidden_layers"]),
        output_controls=int(config["output_controls"]),
        input_tile_features=int(config["input_tile_features"]),
        centering=str(config.get("centering", "none")),
        centering_layers=config.get("centering_layers", []),
        normalization=str(config["normalization"]),
        normalization_layers=config["normalization_layers"],
        bias_layers=config.get("bias_layers"),
        bias_implementation=str(config.get("bias_implementation", "ordinary")),
        weight_transform=str(config.get("weight_transform", "none")),
        weight_transform_layers=config.get("weight_transform_layers", []),
        weight_transform_epsilon=float(config.get("weight_transform_epsilon", 1.0e-5)),
        running_stat_momentum=float(config["running_stat_momentum"]),
        running_stat_epsilon=float(config["running_stat_epsilon"]),
        batch_renorm_rmax=float(config.get("batch_renorm_rmax", 3.0)),
        batch_renorm_dmax=float(config.get("batch_renorm_dmax", 5.0)),
        hardware_enabled=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model

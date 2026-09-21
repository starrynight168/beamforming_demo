from __future__ import annotations

import dataclasses
import json
import math
import os
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from .beamforming import (
    finalize_weight_sum_stats,
    init_depth_weight_sum_stats,
    init_weight_sum_stats,
    update_depth_weight_sum_stats,
    update_weight_sum_stats,
)
from .config import (
    VISUALIZATION_REALIZATION_SEED_OFFSET,
)
from .config import runtime as _runtime
from .data import (
    _amp_enabled,
    UltrasoundImageDataset,
    build_fixed_geometry_cache,
    derive_network_channels,
    load_or_create_mixed_split,
    pick_visualization_dataset,
    resolve_angle_indices,
    save_reconstruction_image,
)
from .hardware import QATConfig
from .model import MBAN

from .training_support import (
    _collate_single_image,
    _load_distillation_model,
    _qat_stage_for_epoch,
    _set_geometry,
    _training_validation_indices,
    _validate_training_checkpoint,
    apply_qat_stage,
    build_learning_rate_scheduler,
    checkpoint_metadata,
    count_nonfinite_gradients,
    count_nonfinite_parameters,
    evaluate_loss_components,
    forward_pixel_batch,
    hardware_checkpoint_state,
    prepare_test_realization,
    qat_parameter_diagnostics,
    report_activation_diagnostics,
    restore_learning_rate_scheduler,
    restore_training_rng_state,
    TEST_METRIC_NAMES,
    training_rng_state,
)
def train():
    if _runtime.args is None:
        raise RuntimeError("训练运行时尚未初始化，请先调用 train.initialize_runtime() 或 train.configure_runtime()")
    if getattr(_runtime.args, "beamforming_implementation", "explicit") != "explicit":
        raise ValueError("训练必须使用 explicit 波束合成；factorized 仅用于评估和部署")
    if _runtime.args.qat_enabled and _runtime.args.qat_mode != "qat":
        raise RuntimeError("训练入口只执行 FP32 或 QAT；PTQ 请使用 evaluate_mban.py --mode ptq")
    output = Path(_runtime.args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    config_copy = output / "config_used.yaml"
    if config_copy.resolve() != _runtime.CONFIG_PATH:
        shutil.copy2(_runtime.CONFIG_PATH, config_copy)
    effective_config = {key: value for key, value in vars(_runtime.args).items() if not key.startswith("_")}
    effective_config.update(
        {
            "config_path": str(_runtime.CONFIG_PATH),
            "hardware_config_path": str(_runtime.HARDWARE_CONFIG_PATH) if _runtime.HARDWARE_CONFIG_PATH else None,
            "active_profile": _runtime.NONIDEAL_PROFILE_NAME,
            "semantic_hash": _runtime.SEMANTIC_HASH,
        }
    )
    (output / "effective_config.json").write_text(
        json.dumps(effective_config, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    if _runtime.args.epochs < 0:
        raise ValueError("epochs 必须大于等于 0")

    _runtime.write_log("=" * 80, level="INFO")
    _runtime.write_log(
        f"[START] MBAN | target={_runtime.args.target_algorithm} | mode={_runtime.args.mode} | seed={_runtime.args.seed}",
        level="INFO",
    )
    _runtime.write_log(
        f"[RUN] config={_runtime.CONFIG_PATH} | output={Path(output).resolve()}",
        level="INFO",
    )
    _runtime.write_log(
        f"[OPT] epochs={_runtime.args.epochs} | lr={_runtime.args.learning_rate} | "
        f"scheduler={_runtime.args.scheduler} | grad_clip={_runtime.args.gradient_clip_norm:g} | "
        f"images/batch={_runtime.args.images_per_batch} | pixels/update={_runtime.args.optimization_batch_pixels} | "
        f"train_pixels/image={_runtime.args.train_pixels_per_image}",
        level="INFO",
    )
    norm_desc = _runtime.args.normalization
    if norm_desc != "none":
        norm_desc += "@" + ",".join(sorted(_runtime.args.normalization_layers))
    weight_desc = _runtime.args.weight_transform
    if weight_desc != "none":
        weight_desc += "@" + ",".join(sorted(_runtime.args.weight_transform_layers))
    center_desc = _runtime.args.centering
    if center_desc != "none":
        center_desc += "@" + ",".join(sorted(_runtime.args.centering_layers))
    bias_desc = ",".join(_runtime.args.bias_layers) if _runtime.args.bias_layers else "none"
    activation_detail = []
    for layer, spec in _runtime.args.activation.items():
        params = ", ".join(f"{k}={v:g}" for k, v in spec.items() if k != "type")
        activation_detail.append(f"{layer}={spec.get('type')}({params})" if params else f"{layer}={spec.get('type')}")
    model_detail = (
        f"dropout={_runtime.args.dropout} | branch={_runtime.args.branch_mode} | "
        f"activation={'; '.join(activation_detail) or 'none'} | "
        f"centering={center_desc} | normalization={norm_desc} | weight_transform={weight_desc} | "
        f"bias={bias_desc} ({_runtime.args.bias_implementation}) | "
        f"projection={_runtime.args.projection_controls} | interpolation={_runtime.args.output_interpolation} | "
        f"input_norm={_runtime.args.input_normalization}"
    )
    loss_terms = _runtime.args.loss["terms"]
    loss_description = (
        f"{_runtime.args.loss['error_function']}(iq={loss_terms['iq']['weight']:g}, "
        f"envelope={loss_terms['envelope']['weight']:g})"
    )
    enabled_losses = []
    for name in ("d1", "d2"):
        term = loss_terms[name]
        if term["weight"]:
            enabled_losses.append(f"{name.upper()}({term['domain']})×{term['weight']:g}")
    if loss_terms["lrange"]["weight"]:
        enabled_losses.append(
            f"Lrange(T={loss_terms['lrange']['limit']:g})×{loss_terms['lrange']['weight']:g}"
        )
    if _runtime.args.distillation_weight:
        enabled_losses.append(f"Distill×{_runtime.args.distillation_weight:g}")
    if _runtime.args.unity_constraint != "hard" and loss_terms["unity"]["weight"]:
        enabled_losses.append(f"Unity×{loss_terms['unity']['weight']:g}")
    unity_mode = _runtime.args.unity_constraint
    _runtime.write_log(
        f"[LOSS] {loss_description} | weights={_runtime.args.output_weights} | "
        f"domain={_runtime.args.output_domain} | unity={unity_mode} | "
        f"extras={', '.join(enabled_losses) if enabled_losses else 'none'}",
        level="INFO",
    )

    h5_paths = [os.path.abspath(path) for path in _runtime.args.h5_files]
    with h5py.File(h5_paths[0], "r") as handle:
        angles = handle["angles"][:]
    selected_angles = resolve_angle_indices(angles, getattr(_runtime.args, "angle_selection", "center"))
    angle_indices = None if getattr(_runtime.args, "angle_selection", "center") == "all" else selected_angles.tolist()
    split_by_h5 = load_or_create_mixed_split(
        h5_paths,
        _runtime.args.train_ratio,
        _runtime.args.validation_ratio,
        _runtime.args.test_ratio,
    )
    train_parts, val_parts, test_parts = [], [], []
    for h5_path in h5_paths:
        name = os.path.splitext(os.path.basename(h5_path))[0]
        split = split_by_h5[h5_path]
        train_parts.append(
            UltrasoundImageDataset(
                h5_path,
                name=f"训练集/{name}",
                angle_indices=angle_indices,
                target_algorithm=_runtime.args.target_algorithm,
                sample_indices=split["train"],
            )
        )
        val_parts.append(
            UltrasoundImageDataset(
                h5_path,
                name=f"验证集/{name}",
                angle_indices=angle_indices,
                target_algorithm=_runtime.args.target_algorithm,
                sample_indices=split["val"],
            )
        )
        if len(split["test"]) > 0:
            test_parts.append(
                UltrasoundImageDataset(
                    h5_path,
                    name=f"测试集/{name}",
                    angle_indices=angle_indices,
                    target_algorithm=_runtime.args.target_algorithm,
                    sample_indices=split["test"],
                )
            )
    train_dataset = ConcatDataset(train_parts)
    val_dataset = train_dataset if _runtime.args.use_train_as_validation else ConcatDataset(val_parts)
    test_dataset = ConcatDataset(test_parts) if test_parts else None
    visualization_dataset, visualization_sample_idx = pick_visualization_dataset(val_parts)
    if _runtime.args.use_train_as_validation:
        _runtime.write_log("验证集正在复用训练集，仅适用于过拟合调试", level="WARNING")

    def split_count(parts: list[UltrasoundImageDataset]) -> str:
        counts = [len(part) for part in parts]
        total = sum(counts)
        return str(total) if len(counts) <= 1 else f"{total} ({' + '.join(map(str, counts))})"

    active_val_parts = train_parts if _runtime.args.use_train_as_validation else val_parts
    split_note = "validation=train（调试）" if _runtime.args.use_train_as_validation else "固定独立子集"
    unique_h5 = []
    for part in train_parts + val_parts + test_parts:
        entry = {
            "name": Path(part.h5_path).name,
            "target": part.target_algorithm,
            "iq": getattr(part, "_iq_target_keys", "H5字段"),
            "a": part.all_multi_I.shape[1],
            "c": part.num_channels,
        }
        if entry not in unique_h5:
            unique_h5.append(entry)
    source_desc = "; ".join(
        f"{entry['name']}(target={entry['target']}, IQ={entry['iq']}, A={entry['a']}, C={entry['c']})"
        for entry in unique_h5
    )
    _runtime.write_log(
        f"[DATA] {source_desc} | "
        f"split={_runtime.args.train_ratio}/{_runtime.args.validation_ratio}/{_runtime.args.test_ratio} | "
        f"{split_note} | train={split_count(train_parts)} | val={split_count(active_val_parts)} | "
        f"test={split_count(test_parts)} | split_file={_runtime.args.split_file or '自动'}",
        level="SUCCESS",
    )
    meta = train_parts[0]
    all_parts = train_parts + val_parts + test_parts
    reference_scalars = {
        "fs": meta.fs,
        "c": meta.c,
        "fc": meta.fc,
        "pitch": meta.pitch,
        "num_channels": meta.num_channels,
    }
    reference_tensors = {
        "z_grid": meta.z_grid,
        "x_grid": meta.x_grid,
        "angles": meta.angles,
    }
    reference_shapes = {
        "IQ": tuple(meta.all_multi_I.shape[1:]),
        "target_i": tuple(meta.target_i.shape[1:]),
        "target_q": tuple(meta.target_q.shape[1:]),
    }
    for part in all_parts[1:]:
        for field, expected in reference_scalars.items():
            actual = getattr(part, field)
            if not np.isclose(actual, expected, rtol=1.0e-6, atol=0.0):
                raise ValueError(f"混合 H5 的 {field} 不一致: {part.h5_path}, {actual} != {expected}")
        for field, expected in reference_tensors.items():
            actual = getattr(part, field)
            if actual.shape != expected.shape or not torch.allclose(actual, expected, rtol=1.0e-6, atol=1.0e-9):
                raise ValueError(f"混合 H5 的 {field} 不一致: {part.h5_path}")
        actual_shapes = {
            "IQ": tuple(part.all_multi_I.shape[1:]),
            "target_i": tuple(part.target_i.shape[1:]),
            "target_q": tuple(part.target_q.shape[1:]),
        }
        if actual_shapes != reference_shapes:
            raise ValueError(f"混合 H5 的数据形状不一致: {part.h5_path}, {actual_shapes} != {reference_shapes}")
    geometry_specs: list[tuple[torch.Tensor, int]] = []
    geometry_id_by_key: dict[tuple[int, bytes], int] = {}
    for part in all_parts:
        part_cache_ids = []
        for sample_t_start, sample_valid_time in zip(
            part.time_start,
            part.valid_time_samples,
            strict=True,
        ):
            valid_time = int(sample_valid_time)
            key = (valid_time, sample_t_start.detach().cpu().contiguous().numpy().tobytes())
            cache_id = geometry_id_by_key.get(key)
            if cache_id is None:
                cache_id = len(geometry_specs)
                geometry_id_by_key[key] = cache_id
                geometry_specs.append((sample_t_start, valid_time))
            part_cache_ids.append(cache_id)
        part.geometry_cache_ids = torch.tensor(part_cache_ids, dtype=torch.long)
    _, _, num_pixels, c_value, fc_value, fs_value = _set_geometry(meta)
    pitch = meta.pitch
    if _runtime.args.dynamic_aperture:
        network_channels, natural = derive_network_channels(
            meta.z_grid,
            pitch,
            _runtime.args.f_number,
            _runtime.NUM_CHANNELS,
            _runtime.args.expected_network_channels,
        )
    else:
        network_channels = _runtime.NUM_CHANNELS
    output_controls = network_channels if _runtime.args.output_controls == 0 else int(_runtime.args.output_controls)
    if not 1 <= output_controls <= network_channels:
        raise ValueError("output_controls 必须在 [1, 网络孔径宽度] 内")
    if output_controls < network_channels and _runtime.args.projection_controls > 0:
        raise ValueError("output_controls 与 projection_controls 不能同时启用")
    if _runtime.args.projection_controls > network_channels:
        raise ValueError("projection_controls 不能超过网络孔径上限")
    _runtime.write_log(
        f"[H5] C={_runtime.NUM_CHANNELS} | image={_runtime.IMG_H}×{_runtime.IMG_W} ({num_pixels:,} px) | "
        f"fc={fc_value / 1e6:.3f} MHz | fs={fs_value / 1e6:.3f} MHz | "
        f"c={c_value:.0f} m/s | pitch={pitch * 1e3:.3f} mm",
        level="INFO",
    )
    if _runtime.args.dynamic_aperture:
        aperture_m = natural
        _runtime.write_log(
            f"[APERTURE] dynamic=true | F={_runtime.args.f_number:g} | physical={_runtime.NUM_CHANNELS} | "
            f"network_max={network_channels} | active={int(aperture_m.min())}–{int(aperture_m.max())} | "
            f"K=⌊z/(F·pitch)⌋+1",
            level="INFO",
        )
        if _runtime.args.projection_controls > 0:
            compressed_fraction = float((aperture_m > _runtime.args.projection_controls).float().mean())
            _runtime.write_log(f"[APERTURE] projection 压缩像素比例={compressed_fraction:.2%}", level="INFO")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    singleton_zero_copy = _runtime.args.images_per_batch == 1 and _runtime.NUM_WORKERS == 0
    pin_memory = device.type == "cuda" and not singleton_zero_copy
    collate_fn = _collate_single_image if singleton_zero_copy else None
    train_loader_generator = torch.Generator().manual_seed(_runtime.args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=_runtime.args.images_per_batch,
        shuffle=True,
        num_workers=_runtime.NUM_WORKERS,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        generator=train_loader_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=_runtime.args.images_per_batch,
        shuffle=False,
        num_workers=_runtime.NUM_WORKERS,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    model = MBAN(
        num_channels=network_channels,
        dropout=_runtime.args.dropout,
        hidden_width=_runtime.args.hidden_width,
        activation=_runtime.args.activation,
        branch_mode=_runtime.args.branch_mode,
        hidden_layers=_runtime.args.hidden_layers,
        output_controls=output_controls,
        input_tile_features=_runtime.args.input_tile_features,
        centering=_runtime.args.centering,
        centering_layers=_runtime.args.centering_layers,
        normalization=_runtime.args.normalization,
        normalization_layers=_runtime.args.normalization_layers,
        bias_layers=_runtime.args.bias_layers,
        bias_implementation=_runtime.args.bias_implementation,
        weight_transform=_runtime.args.weight_transform,
        weight_transform_layers=_runtime.args.weight_transform_layers,
        weight_transform_epsilon=_runtime.args.weight_transform_epsilon,
        running_stat_momentum=_runtime.args.running_stat_momentum,
        running_stat_epsilon=_runtime.args.running_stat_epsilon,
        batch_renorm_rmax=_runtime.args.batch_renorm_rmax,
        batch_renorm_dmax=_runtime.args.batch_renorm_dmax,
        hardware_config=QATConfig.from_namespace(_runtime.args) if _runtime.args.qat_enabled else None,
    ).to(device)
    model_parameters = tuple(model.parameters())
    n_params = sum(parameter.numel() for parameter in model_parameters)
    tile_passes = (
        1
        if _runtime.args.input_tile_features <= 0
        else math.ceil(model.fc1.in_features / _runtime.args.input_tile_features)
    )
    _runtime.write_log(
        f"[MODEL] sensor={_runtime.NUM_CHANNELS} | network={network_channels} | controls={output_controls} | "
        f"hidden={model.hidden_width}×{model.hidden_layers} | fc_layers={model.total_fc_layers} | "
        f"output=fc{model.total_fc_layers} | params={n_params:,} | {model_detail}",
        level="SUCCESS",
    )
    _runtime.write_log(
        f"[INPUT] complex_features={model.input_channels} | fc1_inputs={model.fc1.in_features} | "
        f"tile={_runtime.args.input_tile_features or model.fc1.in_features} | passes={tile_passes}",
        level="INFO",
    )
    distillation_model = _load_distillation_model(network_channels, device)
    if model.hardware_qat is not None:
        _runtime.write_log(
            f"[QAT] inter_layer={_runtime.args.inter_layer} | bits "
            f"W={_runtime.args.weight_bits}/I={_runtime.args.input_bits}/C={_runtime.args.control_bits}/"
            f"Bias={_runtime.args.bias_bits}bit ({_runtime.args.bias_implementation}) | g∈[{_runtime.args.g_min:g},{_runtime.args.g_max:g}]",
            level="INFO",
        )
        qat_schedule = _runtime.args.qat_schedule or {}
        stages = qat_schedule.get("stages", []) if isinstance(qat_schedule, dict) else []
        if stages:
            stage_desc = " → ".join(
                f"{stage.get('name')}({stage.get('epochs')}ep,{stage.get('profile')})" for stage in stages
            )
            _runtime.write_log(f"[QAT] schedule={stage_desc}", level="INFO")
    _runtime.write_log(
        f"[SEMANTICS] requested_profile={_runtime.NONIDEAL_PROFILE_NAME} | hash={_runtime.SEMANTIC_HASH}",
        level="INFO",
    )
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        _runtime.write_log(
            f"[DEVICE] {torch.cuda.get_device_name(device_index)} | "
            f"memory={torch.cuda.get_device_properties(device_index).total_memory / 2**30:.1f} GiB | torch={torch.__version__}",
            level="INFO",
        )
    else:
        _runtime.write_log(f"[DEVICE] CPU | torch={torch.__version__}", level="INFO")
    _runtime.write_log(
        f"[CPU] intraop={torch.get_num_threads()} | interop={torch.get_num_interop_threads()} | "
        f"loader_workers={_runtime.NUM_WORKERS} | pin_memory={pin_memory} | "
        f"singleton_zero_copy={singleton_zero_copy}",
        level="INFO",
    )
    optimizer = torch.optim.Adam(model_parameters, lr=_runtime.args.learning_rate)
    amp_enabled = _amp_enabled(device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    _runtime.write_log(f"[AMP] enabled={amp_enabled}", level="INFO")
    train_pixels = (
        _runtime.NUM_PIXELS
        if _runtime.args.train_pixels_per_image == 0
        else min(_runtime.args.train_pixels_per_image, _runtime.NUM_PIXELS)
    )
    steps_per_epoch = sum(
        math.ceil(
            min(_runtime.args.images_per_batch, len(train_dataset) - i)
            * train_pixels
            / _runtime.args.optimization_batch_pixels
        )
        for i in range(0, len(train_dataset), _runtime.args.images_per_batch)
    )
    scheduler = build_learning_rate_scheduler(optimizer, steps_per_epoch) if _runtime.args.epochs > 0 else None
    _runtime.write_log(
        f"[SCHEDULE] train_pixels/image={train_pixels}/{_runtime.NUM_PIXELS} | "
        f"validation_pixels/image={_runtime.args.validation_pixels_per_image or _runtime.NUM_PIXELS}/"
        f"{_runtime.NUM_PIXELS} | steps/epoch={steps_per_epoch}",
        level="INFO",
    )

    depth_grid = meta.z_grid.view(-1).to(device)
    x_grid = meta.x_grid.to(device)
    angles_rad = meta.angles.to(device)
    offsets = torch.arange(0, 1, device=device)
    angle_idx = torch.arange(meta.all_multi_I.shape[1], device=device)
    geometry_caches: dict[int, dict[str, torch.Tensor] | None] = {}
    cache_capacity_exhausted = device.type != "cuda"
    for cache_id, (sample_t_start, valid_time) in enumerate(geometry_specs):
        if cache_capacity_exhausted:
            geometry_caches[cache_id] = None
            continue
        try:
            geometry_caches[cache_id] = build_fixed_geometry_cache(
                depth_grid,
                x_grid,
                angles_rad,
                angle_idx,
                pitch,
                fs_value,
                fc_value,
                sample_t_start,
                valid_time,
                dynamic_network_channels=network_channels if _runtime.args.dynamic_aperture else None,
                dynamic_control_count=output_controls if _runtime.args.dynamic_aperture else None,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            geometry_caches[cache_id] = None
        if geometry_caches[cache_id] is None:
            cache_capacity_exhausted = True
    cached_geometry_count = sum(cache is not None for cache in geometry_caches.values())
    cached_bytes = sum(
        int(cache[key].numel()) * int(cache[key].element_size())
        for cache in geometry_caches.values()
        if cache is not None
        for key in cache
    )
    _runtime.write_log(
        f"[CACHE] unique={len(geometry_specs)} | cached={cached_geometry_count} | "
        f"realtime={len(geometry_specs) - cached_geometry_count} | ~{cached_bytes / 2**20:.0f} MiB",
        level="INFO",
    )

    start_epoch = 0
    best_val = float("inf")
    best_val_reg = float("inf")
    best_train = float("inf")
    best_stage_index = None
    if _runtime.args.initial_checkpoint:
        checkpoint = torch.load(_runtime.args.initial_checkpoint, map_location=device, weights_only=False)
        _validate_training_checkpoint(
            checkpoint,
            _runtime.args.initial_checkpoint,
            expected_network_channels=network_channels,
            expected_output_controls=output_controls,
            expected_geometry={
                "f_number": float(_runtime.args.f_number),
                "max_depth": float(meta.z_grid.max().item()),
                "pitch": float(pitch),
            },
            for_resume=False,
        )
        state_dict = dict(checkpoint["model_state_dict"])
        if getattr(model, "array_bias", False):
            for name in [f"fc{i}" for i in range(1, model.total_fc_layers + 1)]:
                w_key = f"{name}.weight"
                b_key = f"{name}.bias"
                if w_key in state_dict and b_key in state_dict:
                    w = state_dict[w_key]
                    b = state_dict.pop(b_key)
                    layer = getattr(model, name, None)
                    if layer is not None and w.shape[1] == layer.weight.shape[1] - 1:
                        state_dict[w_key] = torch.cat([w, b.unsqueeze(1).to(dtype=w.dtype, device=w.device)], dim=1)
        model.load_state_dict(state_dict)
        if model.hardware_qat is not None and checkpoint.get("hardware_qat_state") is not None:
            model.hardware_qat.load_state_dict(checkpoint["hardware_qat_state"])
    elif not _runtime.args.start_fresh and os.path.exists(_runtime.LATEST_CHECKPOINT_PATH):
        checkpoint = torch.load(_runtime.LATEST_CHECKPOINT_PATH, map_location=device, weights_only=False)
        resume_fields = (
            (
                "optimizer_state_dict",
                "scaler_state_dict",
                "scheduler_state_dict",
                "epoch",
                "best_val_loss",
                "best_val_reg_loss",
                "best_train_loss",
            )
            if _runtime.args.resume == "all"
            else ()
        )
        _validate_training_checkpoint(
            checkpoint,
            _runtime.LATEST_CHECKPOINT_PATH,
            resume_fields,
            require_semantic_hash=_runtime.args.resume == "all",
            expected_network_channels=network_channels,
            expected_output_controls=output_controls,
            expected_geometry={
                "f_number": float(_runtime.args.f_number),
                "max_depth": float(meta.z_grid.max().item()),
                "pitch": float(pitch),
            },
            for_resume=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        if model.hardware_qat is not None and checkpoint.get("hardware_qat_state") is not None:
            model.hardware_qat.load_state_dict(checkpoint["hardware_qat_state"])
        if _runtime.args.resume == "all":
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if checkpoint["scaler_state_dict"]:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            restore_learning_rate_scheduler(scheduler, checkpoint["scheduler_state_dict"])
            restore_training_rng_state(checkpoint.get("rng_state"), train_loader_generator)
            start_epoch = int(checkpoint["epoch"]) + 1
            best_val = float(checkpoint["best_val_loss"])
            best_val_reg = float(checkpoint["best_val_reg_loss"])
            best_train = float(checkpoint["best_train_loss"])
            checkpoint_stage_index = checkpoint.get("best_stage_index")
            best_stage_index = None if checkpoint_stage_index is None else int(checkpoint_stage_index)

    resume_status = "从零初始化"
    if _runtime.args.initial_checkpoint:
        resume_status = f"warm-start {Path(_runtime.args.initial_checkpoint).name}（仅权重）"
    elif not _runtime.args.start_fresh and os.path.exists(_runtime.LATEST_CHECKPOINT_PATH):
        if _runtime.args.resume == "all":
            resume_status = f"恢复 latest.pth（epoch {start_epoch}/{_runtime.args.epochs}）"
        else:
            resume_status = "warm-start latest.pth（仅权重）"
    elif _runtime.args.resume == "all":
        resume_status = "从零初始化（latest.pth 不存在）"
    best_val_text = f"{best_val:.6f}" if math.isfinite(best_val) else "n/a"
    best_train_text = f"{best_train:.6f}" if math.isfinite(best_train) else "n/a"
    _runtime.write_log(
        f"[RESUME] {resume_status} | best_val={best_val_text} | best_train={best_train_text}",
        level="INFO",
    )

    if model.hardware_qat is not None and _runtime.args.mode == "qat":
        stage_index, _ = _qat_stage_for_epoch(start_epoch)
        apply_qat_stage(model, start_epoch)
        if start_epoch == 0:
            best_stage_index = stage_index
        else:
            checkpoint_stage_index = best_stage_index
            if checkpoint_stage_index is None:
                checkpoint_stage_index, _ = _qat_stage_for_epoch(start_epoch - 1)
            if checkpoint_stage_index != stage_index:
                best_val = float("inf")
                best_val_reg = float("inf")
                best_train = float("inf")
                _runtime.write_log(
                    f"[QAT] reset best checkpoint statistics at resumed stage={stage_index}"
                )
            best_stage_index = stage_index

    if model.hardware_qat is not None and _runtime.args.fixed_observer_calibration and start_epoch == 0:
        model.hardware_qat.clear_observer()
        model.hardware_qat.enable_observer()
        model.eval()
        with torch.no_grad():
            calibrated_samples = 0
            for batch in train_loader:
                if calibrated_samples >= _runtime.args.ptq_calibration_samples:
                    break
                rf_i, rf_q, _, target_i, target_q, t_starts, _, valid_times, cache_ids = batch
                sample_count = min(
                    rf_i.shape[0],
                    _runtime.args.ptq_calibration_samples - calibrated_samples,
                )
                rf_i = rf_i[:sample_count]
                rf_q = rf_q[:sample_count]
                target_i = target_i[:sample_count]
                target_q = target_q[:sample_count]
                t_starts = t_starts[:sample_count]
                valid_times = valid_times[:sample_count]
                cache_ids = cache_ids[:sample_count]
                b_img = rf_i.shape[0]
                calibrated_samples += b_img
                rf_i, rf_q = rf_i.to(device, non_blocking=True), rf_q.to(device, non_blocking=True)
                target_i = target_i.view(b_img, _runtime.NUM_PIXELS).to(device, non_blocking=True)
                target_q = target_q.view(b_img, _runtime.NUM_PIXELS).to(device, non_blocking=True)
                t_starts = t_starts.to(device, non_blocking=True)
                valid_times = valid_times.to(device, non_blocking=True)
                count = min(b_img * _runtime.NUM_PIXELS, _runtime.args.optimization_batch_pixels)
                indices = torch.linspace(0, b_img * _runtime.NUM_PIXELS - 1, steps=count, device=device).long()
                img_idx, pixel_idx = indices // _runtime.NUM_PIXELS, indices % _runtime.NUM_PIXELS
                forward_pixel_batch(
                    model,
                    rf_i,
                    rf_q,
                    t_starts,
                    img_idx,
                    pixel_idx,
                    target_i[img_idx, pixel_idx],
                    target_q[img_idx, pixel_idx],
                    rf_i.shape[2],
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
        digital_observer_warmup = (
            model.hardware_qat.config.inter_layer == "digital"
            and _runtime.args.observer_calibration_epochs > 0
        )
        if digital_observer_warmup:
            model.hardware_qat.clear_observer()
        else:
            model.hardware_qat.disable_observer()
        model.train()
    elif model.hardware_qat is not None and start_epoch == 0:
        model.hardware_qat.clear_observer()
        model.hardware_qat.enable_observer()

    if _runtime.args.epochs == 0:
        return
    current_scale = scaler.get_scale()
    for epoch in range(start_epoch, _runtime.args.epochs):
        if model.hardware_qat is not None:
            if _runtime.args.mode == "qat":
                stage_index, _ = _qat_stage_for_epoch(epoch)
                apply_qat_stage(model, epoch)
                if best_stage_index is None:
                    best_stage_index = stage_index
                elif stage_index != best_stage_index:
                    best_val = float("inf")
                    best_val_reg = float("inf")
                    best_train = float("inf")
                    best_stage_index = stage_index
                    _runtime.write_log(
                        f"[QAT] reset best checkpoint statistics at stage={stage_index}"
                    )
            observer_warmup = (
                model.hardware_qat.config.inter_layer == "digital"
                and epoch < _runtime.args.observer_calibration_epochs
            )
            if (
                not _runtime.args.fixed_observer_calibration
                or observer_warmup
            ):
                model.hardware_qat.enable_observer()
            else:
                model.hardware_qat.disable_observer()
        model.train()
        metric_names = ("loss", "data", "iq", "envelope", "unity", "aperture", "range", "distillation")
        train_metric_sums = torch.zeros(len(metric_names), dtype=torch.float64, device=device)
        train_count = 0
        train_grad_norm_sum = torch.zeros((), dtype=torch.float64, device=device)
        train_clip_count = torch.zeros((), dtype=torch.int64, device=device)
        train_nonfinite_grad_steps = torch.zeros((), dtype=torch.int64, device=device)
        train_nonfinite_grad_elements = torch.zeros((), dtype=torch.int64, device=device)
        train_gradient_elements = 0
        train_skipped_steps = 0
        train_nonfinite_param_elements = 0
        train_parameter_elements = 0
        first_nonfinite_grad_step = torch.full((), -1, dtype=torch.int64, device=device)
        started = time.time()
        track_weight_sums = _runtime.args.unity_constraint != "hard"
        if track_weight_sums:
            train_weight_sum_stats = init_weight_sum_stats(device)
            train_depth_weight_sum_stats = init_depth_weight_sum_stats(device)
        global_step = 0
        for rf_i, rf_q, _, target_i, target_q, t_starts, _, valid_times, cache_ids in train_loader:
            b_img = rf_i.shape[0]
            rf_i, rf_q = rf_i.to(device, non_blocking=True), rf_q.to(device, non_blocking=True)
            target_i = target_i.view(b_img, _runtime.NUM_PIXELS).to(device, non_blocking=True)
            target_q = target_q.view(b_img, _runtime.NUM_PIXELS).to(device, non_blocking=True)
            t_starts = t_starts.to(device, non_blocking=True)
            valid_times = valid_times.to(device, non_blocking=True)
            sampled = torch.cat(
                [
                    torch.randperm(_runtime.NUM_PIXELS, device=device)[:train_pixels] + image * _runtime.NUM_PIXELS
                    for image in range(b_img)
                ]
            )
            sampled = sampled[torch.randperm(sampled.numel(), device=device)]
            for offset_index in range(0, sampled.numel(), _runtime.args.optimization_batch_pixels):
                idx = sampled[offset_index : offset_index + _runtime.args.optimization_batch_pixels]
                img_idx, pixel_idx = idx // _runtime.NUM_PIXELS, idx % _runtime.NUM_PIXELS
                if epoch == start_epoch and global_step == 0:
                    model.begin_activation_diagnostics()
                optimizer.zero_grad(set_to_none=True)
                if model.hardware_qat is not None and model.hardware_qat.config.noise_enabled:
                    model.hardware_qat.begin_noise_realization()
                result = forward_pixel_batch(
                    model,
                    rf_i,
                    rf_q,
                    t_starts,
                    img_idx,
                    pixel_idx,
                    target_i[img_idx, pixel_idx],
                    target_q[img_idx, pixel_idx],
                    rf_i.shape[2],
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
                if epoch == start_epoch and global_step == 0:
                    report_activation_diagnostics(model)
                if track_weight_sums:
                    update_weight_sum_stats(train_weight_sum_stats, result["weights"])
                    update_depth_weight_sum_stats(train_depth_weight_sum_stats, result["weights"], pixel_idx)
                scaler.scale(result["loss"]).backward()
                scaler.unscale_(optimizer)
                train_gradient_elements += sum(
                    parameter.grad.numel() for parameter in model_parameters if parameter.grad is not None
                )
                if not amp_enabled:
                    nonfinite_elements, _ = count_nonfinite_gradients(
                        model_parameters, device=device
                    )
                    train_nonfinite_grad_elements += nonfinite_elements
                    has_nonfinite_gradient = nonfinite_elements > 0
                    train_nonfinite_grad_steps += has_nonfinite_gradient.to(dtype=torch.int64)
                    first_nonfinite_grad_step = torch.where(
                        (first_nonfinite_grad_step < 0) & has_nonfinite_gradient,
                        torch.full_like(first_nonfinite_grad_step, global_step),
                        first_nonfinite_grad_step,
                    )
                if _runtime.args.gradient_clip_norm > 0:
                    grad_norm_before = torch.nn.utils.clip_grad_norm_(
                        model_parameters, _runtime.args.gradient_clip_norm
                    ).detach()
                    train_grad_norm_sum += grad_norm_before.double()
                    train_clip_count += (grad_norm_before > _runtime.args.gradient_clip_norm).long()
                scale_before = current_scale
                scaler.step(optimizer)
                scaler.update()
                current_scale = scaler.get_scale()
                # GradScaler 发现非有限梯度时会跳过这次更新并下调 scale。
                if current_scale < scale_before:
                    train_skipped_steps += 1
                    if amp_enabled:
                        nonfinite_elements, _ = count_nonfinite_gradients(
                            model.parameters(), device=device
                        )
                        train_nonfinite_grad_elements += nonfinite_elements
                        train_nonfinite_grad_steps += 1
                        first_nonfinite_grad_step = torch.where(
                            first_nonfinite_grad_step < 0,
                            torch.full_like(first_nonfinite_grad_step, global_step),
                            first_nonfinite_grad_step,
                        )
                train_metric_sums += (
                    torch.stack([result[name].detach() for name in metric_names]).double() * idx.numel()
                )
                train_count += idx.numel()
                global_step += 1
                if (
                    _runtime.args.scheduler == "onecycle"
                    and scheduler is not None
                    and current_scale >= scale_before
                ):
                    scheduler.step()
        nonfinite_parameters, train_parameter_elements = count_nonfinite_parameters(
            model_parameters, device=device
        )
        train_nonfinite_grad_steps = int(train_nonfinite_grad_steps.item())
        train_nonfinite_grad_elements = int(train_nonfinite_grad_elements.item())
        train_nonfinite_param_elements = int(nonfinite_parameters.item())
        first_nonfinite_grad_step = int(first_nonfinite_grad_step.item())
        if first_nonfinite_grad_step < 0:
            first_nonfinite_grad_step = None
        (
            avg_train,
            avg_train_data,
            avg_train_iq,
            avg_train_envelope,
            avg_train_unity,
            avg_train_aperture,
            avg_train_range,
            avg_train_distill,
        ) = (train_metric_sums / max(train_count, 1)).cpu().tolist()
        if _runtime.args.gradient_clip_norm > 0:
            avg_train_grad_norm = (train_grad_norm_sum / max(global_step, 1)).item()
            clip_rate = (train_clip_count.float() / max(global_step, 1)).item()
        else:
            avg_train_grad_norm = None
            clip_rate = None
        throughput = train_count / max(time.time() - started, 1e-9)

        validation_noise_config = None
        validation_noise_seed = None
        training_noise_state = None
        if model.hardware_qat is not None:
            model.hardware_qat.disable_observer()
            if model.hardware_qat.config.noise_enabled:
                training_noise_state = model.hardware_qat.realization_state()
                validation_noise_config = model.hardware_qat.config
                validation_noise_seed = int(_runtime.args.seed + 1_000_000)
                model.hardware_qat.config = dataclasses.replace(
                    validation_noise_config,
                    noise_seed=validation_noise_seed,
                )
                model.hardware_qat.reset_noise_counter()
                model.hardware_qat.begin_noise_realization()
        model.eval()
        val_metric_sums = torch.zeros(len(metric_names), dtype=torch.float64, device=device)
        val_count = 0
        val_weight_sum_stats = init_weight_sum_stats(device) if track_weight_sums else None
        val_depth_weight_sum_stats = init_depth_weight_sum_stats(device) if track_weight_sums else None
        with torch.no_grad():
            for rf_i, rf_q, _, target_i, target_q, t_starts, _, valid_times, cache_ids in val_loader:
                b_img = rf_i.shape[0]
                rf_i, rf_q = rf_i.to(device, non_blocking=True), rf_q.to(device, non_blocking=True)
                target_i = target_i.view(b_img, _runtime.NUM_PIXELS).to(device, non_blocking=True)
                target_q = target_q.view(b_img, _runtime.NUM_PIXELS).to(device, non_blocking=True)
                t_starts = t_starts.to(device, non_blocking=True)
                valid_times = valid_times.to(device, non_blocking=True)
                indices = _training_validation_indices(b_img, device)
                count = indices.numel()
                for start in range(0, count, _runtime.args.optimization_batch_pixels):
                    idx = indices[start : start + _runtime.args.optimization_batch_pixels]
                    img_idx, pixel_idx = idx // _runtime.NUM_PIXELS, idx % _runtime.NUM_PIXELS
                    result = forward_pixel_batch(
                        model,
                        rf_i,
                        rf_q,
                        t_starts,
                        img_idx,
                        pixel_idx,
                        target_i[img_idx, pixel_idx],
                        target_q[img_idx, pixel_idx],
                        rf_i.shape[2],
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
                    val_metric_sums += (
                        torch.stack([result[name].detach() for name in metric_names]).double() * idx.numel()
                    )
                    val_count += idx.numel()
                    if track_weight_sums:
                        update_weight_sum_stats(val_weight_sum_stats, result["weights"])
                        update_depth_weight_sum_stats(val_depth_weight_sum_stats, result["weights"], pixel_idx)
        if validation_noise_config is not None:
            model.hardware_qat.config = validation_noise_config
            model.hardware_qat.restore_realization_state(training_noise_state)
        (
            avg_val,
            avg_val_data,
            avg_val_iq,
            avg_val_envelope,
            avg_val_unity,
            avg_val_aperture,
            avg_val_range,
            avg_val_distill,
        ) = (val_metric_sums / max(val_count, 1)).cpu().tolist()
        if scheduler is not None and _runtime.args.scheduler == "plateau":
            scheduler.step(avg_val)
        elif scheduler is not None and _runtime.args.scheduler in {"step", "cosine", "multistep"}:
            scheduler.step()
        curr_lr = optimizer.param_groups[0]["lr"]
        epoch_seconds = time.time() - started
        is_best_val_reg = avg_val < best_val_reg
        is_best_val = avg_val_data < best_val
        is_best_train = avg_train_data < best_train
        best_labels = []
        if is_best_val_reg:
            best_labels.append("val_reg")
        if is_best_val:
            best_labels.append("val")
        if is_best_train:
            best_labels.append("train")
        best_tag = f" | new_best={','.join(best_labels)}" if best_labels else ""
        gpu_peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        _runtime.write_log(
            f"[EPOCH] Ep {epoch + 1}/{_runtime.args.epochs} | Tr={avg_train:.6f} | Val={avg_val:.6f} "
            f"| LR={curr_lr:.2e} | 耗时={epoch_seconds:.1f}s "
            f"| GPU峰值={gpu_peak:.2f}GiB{best_tag}",
            level="SUCCESS",
        )
        _runtime.write_log(
            f"  监督损失 | Train Data={avg_train_data:.6f} "
            f"[IQ={avg_train_iq:.6f}×{loss_terms['iq']['weight']:g} + "
            f"Envelope={avg_train_envelope:.6f}×{loss_terms['envelope']['weight']:g}] | "
            f"Val Data={avg_val_data:.6f} "
            f"[IQ={avg_val_iq:.6f}×{loss_terms['iq']['weight']:g} + "
            f"Envelope={avg_val_envelope:.6f}×{loss_terms['envelope']['weight']:g}]",
            level="INFO",
        )
        performance = f"  吞吐 | {throughput:,.0f} px/s | 每epoch像素={train_count:,}"
        if avg_train_grad_norm is not None:
            performance += (
                f" | grad_norm={avg_train_grad_norm:.3e} "
                f"(clip={_runtime.args.gradient_clip_norm:g}, 触发率={clip_rate:.1%})"
            )
        _runtime.write_log(performance, level="INFO")
        if train_nonfinite_grad_steps or train_skipped_steps or train_nonfinite_param_elements:
            _runtime.write_log(
                f"  数值健康 | 非有限梯度步={train_nonfinite_grad_steps}/{global_step}"
                f"（元素 {train_nonfinite_grad_elements}/{train_gradient_elements}）| "
                f"跳过更新步={train_skipped_steps} | "
                f"非有限参数元素={train_nonfinite_param_elements}/{train_parameter_elements} | "
                f"首次非有限梯度步={first_nonfinite_grad_step}",
                level="WARNING",
            )
        if model.hardware_qat is not None:
            qat = model.hardware_qat
            stage_name = getattr(qat, "stage_name", "") or ""
            profile_name = getattr(qat, "profile_name", "") or ""
            _runtime.write_log(
                f"  QAT | stage={stage_name or 'config'} | stage_profile={profile_name} | "
                f"bits W={qat.config.weight_bits} I={qat.config.input_bits} C={qat.config.control_bits} "
                f"B={qat.config.bias_bits}",
                level="INFO",
            )
            if validation_noise_seed is not None:
                _runtime.write_log(f"  QAT validation_noise_seed={validation_noise_seed}", level="INFO")
            for name, values in qat_parameter_diagnostics(model).items():
                bias_range = "n/a" if values["bias_range"] is None else f"{values['bias_range']:.6f}"
                endpoints = (
                    "n/a/n/a"
                    if values["bias_endpoint_positive"] is None
                    else f"{values['bias_endpoint_positive']:.3f}/{values['bias_endpoint_negative']:.3f}"
                )
                _runtime.write_log(
                    f"  偏置/映射 | {name}: raw_b|max={values['raw_bias_max']:.6f} "
                    f"eff_b|max={values['effective_bias_max']:.6f} range={bias_range} "
                    f"end(+/-)={endpoints} "
                    f"wscale={values['weight_scale']:.6f} "
                    f"core_err(rms/max)={values['core_error_rms']:.6f}/{values['core_error_max']:.6f}",
                    level="INFO",
                )
        train_extra = []
        val_extra = []
        for name, train_value, val_value, weight in (
            ("Aperture", avg_train_aperture, avg_val_aperture, 1.0 if any(loss_terms[n]["weight"] for n in ("d1", "d2")) else 0.0),
            ("Lrange", avg_train_range, avg_val_range, loss_terms["lrange"]["weight"]),
            ("Distill", avg_train_distill, avg_val_distill, _runtime.args.distillation_weight),
        ):
            if weight:
                train_extra.append(f"{name}={train_value:.6f}×{weight:g}")
                val_extra.append(f"{name}={val_value:.6f}×{weight:g}")
        if _runtime.args.unity_constraint != "hard" and loss_terms["unity"]["weight"]:
            train_extra.append(f"Unity={avg_train_unity:.6f}×{loss_terms['unity']['weight']:g}")
            val_extra.append(f"Unity={avg_val_unity:.6f}×{loss_terms['unity']['weight']:g}")
        if train_extra:
            _runtime.write_log(
                f"  附加损失 | Train {', '.join(train_extra)} | Val {', '.join(val_extra)}",
                level="INFO",
            )
        if track_weight_sums:
            tr_w_mean, tr_w_std, tr_w_min, tr_w_max = finalize_weight_sum_stats(train_weight_sum_stats)
            va_w_mean, va_w_std, va_w_min, va_w_max = finalize_weight_sum_stats(val_weight_sum_stats)
            tr_depth_stats = [finalize_weight_sum_stats(stats) for stats in train_depth_weight_sum_stats]
            va_depth_stats = [finalize_weight_sum_stats(stats) for stats in val_depth_weight_sum_stats]
            _runtime.write_log(
                f"  Σw统计 | Train mean={tr_w_mean:.4f}, std={tr_w_std:.4f}, min={tr_w_min:.4f}, max={tr_w_max:.4f} | "
                f"Val mean={va_w_mean:.4f}, std={va_w_std:.4f}, min={va_w_min:.4f}, max={va_w_max:.4f}",
                level="INFO",
            )
            _runtime.write_log(
                "  Σw分深度带 | Train "
                + " / ".join(f"{mean:.4f}±{std:.4f}" for mean, std, _, _ in tr_depth_stats)
                + " | Val "
                + " / ".join(f"{mean:.4f}±{std:.4f}" for mean, std, _, _ in va_depth_stats),
                level="INFO",
            )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            layer_stats = []
            for name, param in model.named_parameters():
                if not name.endswith(".weight"):
                    continue
                flat = param.detach().float().flatten()
                if flat.numel() == 0:
                    continue
                q95 = torch.quantile(flat.abs(), 0.95).item()
                layer_stats.append(f"{name}={flat.mean().item():.4f}±{flat.std().item():.4f} |abs|q95={q95:.4f}")
            if layer_stats:
                _runtime.write_log(
                    "  逐层权重 | " + " ; ".join(layer_stats),
                    level="INFO",
                )
        metadata = checkpoint_metadata(
            network_channels,
            _runtime.NUM_CHANNELS,
            output_controls,
            float(meta.z_grid.max().item()),
            pitch,
        )
        hardware_state = hardware_checkpoint_state(model)
        if avg_val < best_val_reg:
            best_val_reg = avg_val
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "deployment_linear_state": model.deployment_linear_state(),
                    "epoch": epoch,
                    "best_val_reg_loss": best_val_reg,
                    "best_stage_index": best_stage_index,
                    "best_stage_name": getattr(model.hardware_qat, "stage_name", None),
                    **hardware_state,
                    **metadata,
                },
                _runtime.BEST_VAL_REG_WEIGHT_PATH,
            )
            _runtime.write_log(f"[SAVE] best_val_reg.pth | ValReg={avg_val:.6f}", level="SUCCESS")
        if avg_val_data < best_val:
            best_val = avg_val_data
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "deployment_linear_state": model.deployment_linear_state(),
                    "epoch": epoch,
                    "best_val_loss": best_val,
                    "best_stage_index": best_stage_index,
                    "best_stage_name": getattr(model.hardware_qat, "stage_name", None),
                    **hardware_state,
                    **metadata,
                },
                _runtime.BEST_VAL_WEIGHT_PATH,
            )
            _runtime.write_log(f"[SAVE] best_val.pth | Val={avg_val_data:.6f}", level="SUCCESS")
        if avg_train_data < best_train:
            best_train = avg_train_data
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "deployment_linear_state": model.deployment_linear_state(),
                    "epoch": epoch,
                    "best_train_loss": best_train,
                    "best_stage_index": best_stage_index,
                    "best_stage_name": getattr(model.hardware_qat, "stage_name", None),
                    **hardware_state,
                    **metadata,
                },
                _runtime.BEST_TRAIN_WEIGHT_PATH,
            )
            _runtime.write_log(f"[SAVE] best_train.pth | Tr={avg_train_data:.6f}", level="SUCCESS")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "deployment_linear_state": model.deployment_linear_state(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "best_val_loss": best_val,
                "best_val_reg_loss": best_val_reg,
                "best_train_loss": best_train,
                "best_stage_index": best_stage_index,
                "best_stage_name": getattr(model.hardware_qat, "stage_name", None),
                "rng_state": training_rng_state(train_loader_generator),
                **hardware_state,
                **metadata,
            },
            _runtime.LATEST_CHECKPOINT_PATH,
        )
        if _runtime.args.save_images_every_epochs > 0 and (epoch + 1) % _runtime.args.save_images_every_epochs == 0:
            visualization_hardware = getattr(model, "hardware_qat", None)
            visualization_config = None
            visualization_state = None
            visualization_observer = None
            if visualization_hardware is not None:
                visualization_config = visualization_hardware.config
                visualization_state = visualization_hardware.realization_state()
                visualization_observer = visualization_hardware.observer_enabled
                if visualization_config.noise_enabled:
                    visualization_hardware.config = dataclasses.replace(
                        visualization_config,
                        noise_seed=int(_runtime.args.seed) + VISUALIZATION_REALIZATION_SEED_OFFSET,
                    )
                    visualization_hardware.reset_noise_counter()
            try:
                save_reconstruction_image(
                    model,
                    visualization_dataset,
                    sample_idx=visualization_sample_idx,
                    epoch=epoch + 1,
                    device=device,
                    angle_idx=angle_idx,
                    offsets=offsets,
                    fs=fs_value,
                    pitch=pitch,
                    x_grid=x_grid,
                    depth_grid=depth_grid,
                    angles_rad=angles_rad,
                    geometry_caches=geometry_caches,
                )
            finally:
                if visualization_hardware is not None:
                    visualization_hardware.config = visualization_config
                    visualization_hardware.observer_enabled = visualization_observer
                    visualization_hardware.restore_realization_state(visualization_state)

    if test_dataset is not None and os.path.exists(_runtime.BEST_VAL_WEIGHT_PATH):
        checkpoint = torch.load(_runtime.BEST_VAL_WEIGHT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if model.hardware_qat is not None and checkpoint.get("hardware_qat_state") is not None:
            model.hardware_qat.load_state_dict(checkpoint["hardware_qat_state"], restore_config=True)
        if model.hardware_qat is not None:
            model.hardware_qat.disable_observer()
        test_loader = DataLoader(
            test_dataset,
            batch_size=_runtime.args.images_per_batch,
            shuffle=False,
            num_workers=_runtime.NUM_WORKERS,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
        test_realization = prepare_test_realization(model)
        if test_realization is not None:
            _runtime.write_log(f"[TEST] hardware realization seed={test_realization[1]}", level="INFO")
        try:
            test_metrics = evaluate_loss_components(
                model,
                test_loader,
                device,
                angle_idx,
                offsets,
                pitch,
                x_grid,
                depth_grid,
                angles_rad,
                distillation_model,
                geometry_caches,
            )
        finally:
            if test_realization is not None:
                original_config, _ = test_realization
                model.hardware_qat.config = original_config
                model.hardware_qat.reset_noise_counter()
        _runtime.write_log(
            f"[TEST] best_val 在独立测试集上的 loss={test_metrics['loss']:.6f}（{test_metrics['pixels']} px）",
            level="SUCCESS",
        )
        _runtime.write_log(
            "  [TEST] 分量（未加系数）| "
            + " ".join(f"{name}={test_metrics[name]:.6f}" for name in TEST_METRIC_NAMES)
            + f" | weighted_range={float(loss_terms['lrange']['weight']) * test_metrics['range']:.6f}"
            + f" | total={test_metrics['loss']:.6f}",
            level="SUCCESS",
        )
        _runtime.write_log("[DONE] 训练完成", level="SUCCESS")

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

TRAIN_DIR = Path(__file__).resolve().parents[1]
ROOT = TRAIN_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

import mban as training
import mban_core.beamforming as bf
from mban_core.config import configured_h5_files
from eval_core.inference import load_scene, metric_pair, prepare_noise_realization, read_split_indices, reconstruct
from eval_core.config import ModelSpec, load_evaluation_model, resolve_hardware_config, resolve_path
from mban_core.data import UltrasoundImageDataset, compute_aperture_mask, extract_windows_on_gpu, resolve_angle_indices
from mban_core.training_support import _set_geometry, forward_pixel_batch
from mban_core.naming import layer_name, output_controls_name, output_preactivation_name


DEVICE: torch.device
MICRO_BATCH = 8192
CONTROL_COUNT = 8


@dataclass(frozen=True)
class Stage:
    name: str
    kind: str
    checkpoint_candidates: tuple[Path, ...]


@dataclass(frozen=True)
class Locations:
    root: Path
    config: Path
    hardware_config: Path
    data_h5: Path
    simulation_h5: Path
    experiments_h5: Path
    output_dir: Path
    split_file: Path
    stages: tuple[Stage, ...]


def default_split_candidates(root: Path) -> tuple[Path, ...]:
    train = root / "train"
    return (
        train / "results" / "SI" / "shared_split_seed42_0p7-0p2-0p1.csv",
        train / "results" / "model_ablation" / "shared_split_seed42_0p7-0p2-0p1.csv",
    )


def build_stages(root: Path, fp32: Path | None, qat4: Path | None) -> tuple[Stage, ...]:
    train = root / "train"
    fp32_candidates = (fp32,) if fp32 is not None else (
        train / "results" / "SI" / "model" / "windows" / "FP32_FixedLR_30ep" / "best_val.pth",
    )
    qat4_candidates = (qat4,) if qat4 is not None else (
        train / "results" / "SI" / "model" / "windows" / "QAT4_from_FixedLR_30ep" / "best_val.pth",
    )
    return (
        Stage("FP32", "fp32", fp32_candidates),
        Stage("QAT4", "qat", qat4_candidates),
    )


def resolve_existing(candidates: tuple[Path, ...], label: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"{label} checkpoint 不存在: {', '.join(str(p) for p in candidates)}")


def read_split(path: Path, data_h5: Path) -> dict[str, list[int]]:
    result = {
        name: read_split_indices(path, data_h5, name)
        for name in ("train", "val", "test")
    }
    if not result["train"] or not result["val"]:
        raise ValueError(f"固定划分缺少 train 或 val: {path}")
    return result


def load_spec(checkpoint: Path, kind: str, locations: Locations) -> ModelSpec:
    return load_evaluation_model(
        checkpoint.stem,
        checkpoint,
        kind,
        DEVICE,
        config_path=locations.config,
        hardware_config_path=locations.hardware_config,
        profile="ideal" if kind == "fp32" else "common",
    )


class FixedK8(nn.Module):
    def __init__(
        self,
        hardware: object | None,
        network_channels: int,
        values: torch.Tensor,
        trainable_positive: bool,
        output_layer: str,
    ) -> None:
        super().__init__()
        self.hardware_qat = hardware
        self.network_channels = int(network_channels)
        self.output_control_range_name = output_controls_name(output_layer)
        self.output_preactivation_range_name = output_preactivation_name(output_layer)
        values = values.detach().float().reshape(1, -1).to(DEVICE)
        if values.shape[-1] != CONTROL_COUNT:
            raise ValueError(f"静态 K8 控制向量形状非法: {tuple(values.shape)}")
        self.trainable_positive = bool(trainable_positive)
        if self.trainable_positive:
            self.latent = nn.Parameter(values.clone())
        else:
            self.register_buffer("fixed_values", values)

    def controls(self) -> torch.Tensor:
        if self.trainable_positive:
            return F.relu(self.latent)
        return self.fixed_values

    def forward(self, rf_i: torch.Tensor, rf_q: torch.Tensor) -> torch.Tensor:
        return self.controls().to(device=rf_i.device, dtype=rf_i.dtype).expand(rf_i.shape[0], -1)

    def prepare_output_controls(
        self,
        controls: torch.Tensor,
        depth_index: torch.Tensor | None = None,
        depth_count: int | None = None,
        active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.hardware_qat is None:
            return controls
        runtime_args = training.args
        if self.hardware_qat.config.inter_layer == "analog":
            if runtime_args.output_domain != "nonnegative":
                return controls
            reference = (
                float(runtime_args.control_adc_full_scale)
                if runtime_args.control_adc_range == "full_scale"
                else None
            )
            return self.hardware_qat.output_activation_mismatch(controls, self.output_control_range_name, reference)
        if self.hardware_qat.config.inter_layer != "digital":
            return controls
        value_max = None
        if runtime_args.control_adc_range == "full_scale":
            value_max = controls.new_tensor(runtime_args.control_adc_full_scale)
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
        return F.relu(adc_controls) if runtime_args.output_domain == "nonnegative" else adc_controls

    def quantize_controls(
        self,
        controls: torch.Tensor,
        depth_index: torch.Tensor | None = None,
        depth_count: int | None = None,
    ) -> torch.Tensor:
        if self.hardware_qat is None:
            return controls
        runtime_args = training.args
        value_max = None
        if runtime_args.control_adc_range == "full_scale":
            value_max = controls.new_tensor(runtime_args.control_adc_full_scale)
        return self.hardware_qat.quantize_output_controls(
            controls,
            name=self.output_control_range_name,
            bits=self.hardware_qat.config.control_bits,
            unsigned=runtime_args.output_domain == "nonnegative",
            value_max=value_max,
            depth_index=depth_index,
            depth_count=depth_count,
        )


def model_metadata(spec: ModelSpec, checkpoint: Path) -> dict[str, object]:
    metadata: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "model_type": spec.model_type,
        "network_channels": int(spec.model.network_channels),
        "output_controls": int(getattr(spec.model, "output_controls", CONTROL_COUNT)),
        "runtime": {key: spec.runtime.get(key) for key in (
            "output_weights",
            "output_domain",
            "unity_constraint",
            "unity_scope",
            "angle_selection",
            "angle_reduction",
            "control_normalization",
            "control_adc_range",
            "control_adc_full_scale",
            "dynamic_aperture",
            "f_number",
            "output_interpolation",
            "interpolation_bits",
        )},
    }
    hardware = getattr(spec.model, "hardware_qat", None)
    if hardware is not None:
        metadata["hardware_config"] = asdict(hardware.config)
    effective_config = checkpoint.parent / "effective_config.json"
    if effective_config.exists():
        metadata["effective_config"] = json.loads(effective_config.read_text(encoding="utf-8"))
    return metadata


def make_dataset(frames: list[int], name: str, data_h5: Path) -> UltrasoundImageDataset:
    selection = getattr(training.args, "angle_selection", "center")
    with h5py.File(data_h5, "r") as handle:
        angles = np.asarray(handle["angles"][:], dtype=np.float32)
    selected_angles = resolve_angle_indices(angles, selection)
    angle_indices = None if selection == "all" else selected_angles.tolist()
    return UltrasoundImageDataset(
        str(data_h5),
        name=name,
        angle_indices=angle_indices,
        target_algorithm="mv",
        sample_indices=frames,
    )


def capture_raw_window(
    spec: ModelSpec,
    dataset: UltrasoundImageDataset,
    pixel_count: int,
    seed: int,
) -> torch.Tensor:
    _set_geometry(dataset)
    spec.model.eval()
    prepare_noise_realization(getattr(spec.model, "hardware_qat", None), seed)
    pixel_count = int(min(pixel_count or training.runtime.NUM_PIXELS, training.runtime.NUM_PIXELS))
    if pixel_count < 1:
        raise ValueError("冻结窗采样像素数必须为正数")
    angle_idx = torch.arange(dataset.all_multi_I.shape[1], device=DEVICE)
    offsets = torch.arange(0, 1, device=DEVICE)
    depth_grid = dataset.z_grid.view(-1).to(DEVICE)
    x_grid = dataset.x_grid.to(DEVICE)
    angles = dataset.angles.to(DEVICE)
    total = torch.zeros(CONTROL_COUNT, device=DEVICE, dtype=torch.float64)
    count = 0
    with torch.inference_mode():
        for frame_index in range(len(dataset)):
            rf_i = dataset.all_multi_I[frame_index : frame_index + 1].to(DEVICE)
            rf_q = dataset.all_multi_Q[frame_index : frame_index + 1].to(DEVICE)
            t_starts = dataset.time_start[frame_index : frame_index + 1].to(DEVICE)
            valid_times = dataset.valid_time_samples[frame_index : frame_index + 1].to(DEVICE)
            for offset in range(0, pixel_count, MICRO_BATCH):
                pixel = torch.arange(offset, min(offset + MICRO_BATCH, pixel_count), device=DEVICE)
                image_index = torch.zeros(pixel.numel(), dtype=torch.long, device=DEVICE)
                ext_i, ext_q, tof, tx_tof = extract_windows_on_gpu(
                    rf_i,
                    rf_q,
                    t_starts,
                    image_index,
                    pixel,
                    dataset.fs,
                    offsets,
                    angle_idx,
                    rf_i.shape[2],
                    valid_times,
                    dataset.pitch,
                    x_grid,
                    depth_grid,
                    angles,
                )
                center_t = ext_i.shape[-1] // 2
                i_sample, q_sample = ext_i[..., center_t], ext_q[..., center_t]
                phase = 2.0 * np.pi * dataset.fc * (tof - tx_tof.unsqueeze(2))
                i_aligned = i_sample * torch.cos(phase) - q_sample * torch.sin(phase)
                q_aligned = i_sample * torch.sin(phase) + q_sample * torch.cos(phase)
                mask, aperture_start, aperture_size = compute_aperture_mask(
                    pixel,
                    depth_grid,
                    x_grid,
                    dataset.pitch,
                    return_geometry=True,
                )
                i_model, local_active = bf.pack_dynamic_input(
                    i_aligned,
                    aperture_start,
                    aperture_size,
                    spec.model.network_channels,
                )
                q_model, _ = bf.pack_dynamic_input(
                    q_aligned,
                    aperture_start,
                    aperture_size,
                    spec.model.network_channels,
                )
                i_model, q_model, _, _ = bf.normalize_input_iq(i_model, q_model, local_active)
                raw = bf.predict_weights(spec.model, i_model, q_model)
                if raw.shape[-1] != CONTROL_COUNT:
                    raise ValueError(f"动态模型输出不是 K8: {tuple(raw.shape)}")
                total += raw.detach().double().sum(dim=(0, 1))
                count += int(raw.shape[0] * raw.shape[1])
    if count < 1:
        raise RuntimeError("冻结窗没有有效采样")
    return (total / count).float().cpu()


def batch_inputs(
    batch: tuple[torch.Tensor, ...],
    pixel_indices: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    rf_i, rf_q, _, target_i, target_q, t_starts, _, valid_times, cache_ids = batch
    batch_size = rf_i.shape[0]
    rf_i = rf_i.to(DEVICE)
    rf_q = rf_q.to(DEVICE)
    target_i = target_i.reshape(batch_size, training.runtime.NUM_PIXELS).to(DEVICE)
    target_q = target_q.reshape(batch_size, training.runtime.NUM_PIXELS).to(DEVICE)
    t_starts = t_starts.to(DEVICE)
    valid_times = valid_times.to(DEVICE)
    sampled = torch.cat(
        [pixel_indices + image_index * training.runtime.NUM_PIXELS for image_index in range(batch_size)]
    )
    return rf_i, rf_q, target_i, target_q, t_starts, valid_times, cache_ids.to(DEVICE), sampled


def select_pixel_indices(pixel_count: int, full_pixel_count: int, seed: int, train: bool) -> torch.Tensor:
    pixel_count = min(int(pixel_count), int(full_pixel_count))
    if pixel_count < 1:
        raise ValueError("像素采样数必须为正数")
    if pixel_count >= full_pixel_count:
        return torch.arange(full_pixel_count, device=DEVICE)
    if train:
        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
        return torch.randperm(full_pixel_count, generator=generator, device=DEVICE)[:pixel_count]
    if pixel_count == 1:
        return torch.tensor([full_pixel_count // 2], device=DEVICE, dtype=torch.long)
    positions = torch.arange(pixel_count, device=DEVICE, dtype=torch.long)
    return torch.div(positions * (full_pixel_count - 1), pixel_count - 1, rounding_mode="floor")


def static_epoch(
    static: FixedK8,
    dataset: UltrasoundImageDataset,
    pixel_count: int,
    optimizer: torch.optim.Optimizer | None,
    train: bool,
    seed: int,
) -> float:
    static.eval()
    angle_idx = torch.arange(dataset.all_multi_I.shape[1], device=DEVICE)
    offsets = torch.arange(0, 1, device=DEVICE)
    x_grid = dataset.x_grid.to(DEVICE)
    depth_grid = dataset.z_grid.view(-1).to(DEVICE)
    angles = dataset.angles.to(DEVICE)
    total_loss = 0.0
    total_count = 0
    loader = DataLoader(dataset, batch_size=1, shuffle=train, num_workers=0)
    full_pixel_count = int(training.runtime.NUM_PIXELS)
    fixed_pixel_indices = select_pixel_indices(pixel_count, full_pixel_count, seed, train=False)
    if static.hardware_qat is not None and not train:
        prepare_noise_realization(static.hardware_qat, seed)
    for batch_index, batch in enumerate(loader):
        if static.hardware_qat is not None and train:
            prepare_noise_realization(static.hardware_qat, seed + batch_index)
        pixel_indices = (
            select_pixel_indices(pixel_count, full_pixel_count, seed + batch_index, train=True)
            if train
            else fixed_pixel_indices
        )
        rf_i, rf_q, target_i, target_q, t_starts, valid_times, cache_ids, sampled = batch_inputs(
            batch, pixel_indices
        )
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        for start in range(0, sampled.numel(), MICRO_BATCH):
            selected = sampled[start : start + MICRO_BATCH]
            image_index = selected // training.runtime.NUM_PIXELS
            pixel_index = selected % training.runtime.NUM_PIXELS
            result = forward_pixel_batch(
                static,
                rf_i,
                rf_q,
                t_starts,
                image_index,
                pixel_index,
                target_i[image_index, pixel_index],
                target_q[image_index, pixel_index],
                rf_i.shape[2],
                valid_times,
                cache_ids,
                angle_idx,
                offsets,
                dataset.pitch,
                x_grid,
                depth_grid,
                angles,
                geometry_caches=None,
            )
            loss = result["loss"]
            if train:
                (loss * (selected.numel() / sampled.numel())).backward()
            total_loss += float(loss.detach()) * selected.numel()
            total_count += selected.numel()
        if optimizer is not None:
            torch.nn.utils.clip_grad_norm_([static.latent], float(training.args.gradient_clip_norm))
            optimizer.step()
    return total_loss / max(total_count, 1)


def train_static(
    spec: ModelSpec,
    train_dataset: UltrasoundImageDataset,
    val_dataset: UltrasoundImageDataset,
    stage_dir: Path,
    epochs: int,
    train_pixel_count: int,
    val_pixel_count: int,
    seed: int,
    initial_controls: torch.Tensor | None,
    initialization_source: str,
    checkpoint: Path,
    train_frames: list[int],
    val_frames: list[int],
    model_contract: dict[str, object],
) -> torch.Tensor:
    _set_geometry(train_dataset)
    init = (
        torch.full((CONTROL_COUNT,), 1.0 / CONTROL_COUNT, dtype=torch.float32)
        if initial_controls is None
        else torch.as_tensor(initial_controls, dtype=torch.float32).reshape(-1).cpu()
    )
    if init.numel() != CONTROL_COUNT or not torch.isfinite(init).all() or (init < 0).any():
        raise ValueError("静态 K8 初始控制向量必须为有限非负 8 维")
    static = FixedK8(
        getattr(spec.model, "hardware_qat", None),
        spec.model.network_channels,
        init,
        trainable_positive=True,
        output_layer=layer_name(spec.model.total_fc_layers),
    ).to(DEVICE)
    if sum(parameter.numel() for parameter in static.parameters()) != CONTROL_COUNT:
        raise RuntimeError("静态 K8 必须且只能训练 8 个参数")
    optimizer = torch.optim.Adam([static.latent], lr=float(training.args.learning_rate))
    best_val = math.inf
    best_controls = None
    history = []
    validation_seed = seed + 500000
    for epoch in range(int(epochs)):
        train_loss = static_epoch(static, train_dataset, train_pixel_count, optimizer, True, seed + epoch * 10000)
        val_loss = static_epoch(static, val_dataset, val_pixel_count, None, False, validation_seed)
        controls = static.controls().detach().cpu().reshape(-1)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_controls = controls.clone()
            torch.save(
                {
                    "controls": best_controls,
                    "val_loss": best_val,
                    "epoch": epoch + 1,
                    "stage": stage_dir.name,
                    "initialization": initialization_source,
                    "checkpoint": str(checkpoint.resolve()),
                    "train_frames": train_frames,
                    "val_frames": val_frames,
                    "train_pixel_count": train_pixel_count,
                    "val_pixel_count": val_pixel_count,
                    "model_contract": model_contract,
                },
                stage_dir / "best_static_k8.pt",
            )
        print(
            f"[{stage_dir.name}] static epoch={epoch + 1}/{epochs} "
            f"train={train_loss:.6f} val={val_loss:.6f}",
            flush=True,
        )
    if best_controls is None:
        raise RuntimeError("静态 K8 没有得到验证损失")
    (stage_dir / "static_k8_training.json").write_text(
        json.dumps(
            {
                "best_val_loss": best_val,
                "history": history,
                "train_pixel_count": train_pixel_count,
                "val_pixel_count": val_pixel_count,
                "only_trainable_parameters": ["latent[8]"],
                "checkpoint": str(checkpoint.resolve()),
                "train_frames": train_frames,
                "val_frames": val_frames,
                "initialization": initialization_source,
                "model_contract": model_contract,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return best_controls


def load_vector(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, dict):
        value = value.get("controls", value.get("raw_controls"))
    if value is None:
        raise ValueError(f"控制向量文件缺少 controls: {path}")
    value = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    if value.numel() != CONTROL_COUNT or not torch.isfinite(value).all():
        raise ValueError(f"控制向量必须为有限 8 维: {path}")
    return value


def render_model(
    spec: ModelSpec,
    model: nn.Module,
    data,
    seed: int,
    paired_state: dict[str, object] | None = None,
) -> np.ndarray:
    hardware = getattr(model, "hardware_qat", None)
    if paired_state is None:
        prepare_noise_realization(hardware, seed)
    else:
        if hardware is not None and paired_state is not None:
            hardware.restore_realization_state(paired_state)
    target_spec = replace(spec, model=model, display_name=spec.display_name)
    image, _, _, _ = reconstruct(target_spec, data, DEVICE, MICRO_BATCH, collect_weight_stats=False)
    return image


def evaluate_validation(
    spec: ModelSpec,
    frozen: torch.Tensor,
    static: torch.Tensor,
    frames: list[int],
    stage_dir: Path,
    seed: int,
    data_h5: Path,
    mc_runs: int = 1,
) -> None:
    methods = (
        ("dynamic_k8", spec.model),
        (
            "frozen_train_window",
            FixedK8(
                getattr(spec.model, "hardware_qat", None),
                spec.model.network_channels,
                frozen,
                False,
                layer_name(spec.model.total_fc_layers),
            ).to(DEVICE),
        ),
        (
            "static_k8",
            FixedK8(
                getattr(spec.model, "hardware_qat", None),
                spec.model.network_channels,
                static,
                False,
                layer_name(spec.model.total_fc_layers),
            ).to(DEVICE),
        ),
    )
    rows = []
    data_by_frame = {frame: load_scene(data_h5, frame) for frame in frames}
    for mc_run in range(max(1, int(mc_runs))):
        for frame in frames:
            data = data_by_frame[frame]
            noise_seed = seed + mc_run * 1000003 + frame
            paired_state = prepare_noise_realization(getattr(spec.model, "hardware_qat", None), noise_seed)
            for method, model in methods:
                image = render_model(spec, model, data, noise_seed, paired_state)
                ssim_db, ssim_env = metric_pair(image, data.gt_db)
                rows.append(
                    {
                        "stage": stage_dir.name,
                        "split": "val",
                        "frame": frame,
                        "mc_run": mc_run,
                        "noise_seed": noise_seed,
                        "method": method,
                        "SSIM_dB": float(ssim_db),
                        "SSIM": float(ssim_env),
                    }
                )
    with (stage_dir / "validation_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for method in ("dynamic_k8", "frozen_train_window", "static_k8"):
        selected = [row for row in rows if row["method"] == method]
        summary.append(
            {
                "stage": stage_dir.name,
                "split": "val",
                "method": method,
                "frames": len(frames),
                "mc_runs": max(1, int(mc_runs)),
                "observations": len(selected),
                "SSIM_dB_mean": float(np.mean([row["SSIM_dB"] for row in selected])),
                "SSIM_mean": float(np.mean([row["SSIM"] for row in selected])),
                "SSIM_std": float(np.std([row["SSIM"] for row in selected], ddof=1)) if len(selected) > 1 else 0.0,
            }
        )
    with (stage_dir / "validation_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


def evaluate_scenes(
    spec: ModelSpec,
    frozen: torch.Tensor,
    static: torch.Tensor,
    stage_dir: Path,
    seed: int,
    locations: Locations,
    mc_runs: int = 1,
) -> None:
    methods = (
        ("dynamic_k8", spec.model),
        (
            "frozen_train_window",
            FixedK8(
                getattr(spec.model, "hardware_qat", None),
                spec.model.network_channels,
                frozen,
                False,
                layer_name(spec.model.total_fc_layers),
            ).to(DEVICE),
        ),
        (
            "static_k8",
            FixedK8(
                getattr(spec.model, "hardware_qat", None),
                spec.model.network_channels,
                static,
                False,
                layer_name(spec.model.total_fc_layers),
            ).to(DEVICE),
        ),
    )
    scenes = (
        ("simulation_contrast_speckle", locations.simulation_h5, 0),
        ("simulation_resolution_distorsion", locations.simulation_h5, 1),
        ("experiments_contrast_speckle", locations.experiments_h5, 0),
        ("experiments_resolution_distorsion", locations.experiments_h5, 1),
    )
    mc_rows: list[dict[str, object]] = []
    for scene_name, h5_path, sample_idx in scenes:
        data = load_scene(h5_path, sample_idx)
        images = None
        for mc_run in range(max(1, int(mc_runs))):
            noise_seed = seed + mc_run * 1000003 + sample_idx
            paired_state = prepare_noise_realization(getattr(spec.model, "hardware_qat", None), noise_seed)
            realization_images = [data.gt_db]
            for method_name, model in methods:
                image = render_model(spec, model, data, noise_seed, paired_state)
                realization_images.append(image)
                ssim_db, ssim_env = metric_pair(image, data.gt_db)
                mc_rows.append(
                    {
                        "scene": scene_name,
                        "method": method_name,
                        "mc_run": mc_run + 1,
                        "noise_seed": noise_seed,
                        "ssim_db": float(ssim_db),
                        "ssim_env": float(ssim_env),
                    }
                )
            if images is None:
                images = realization_images
        assert images is not None
        if len(images) != 4:
            raise RuntimeError(f"场景 comparison 必须包含 GT+3 方法，实际为 {len(images)}")
        scene_dir = stage_dir / "scenes" / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        comparison = scene_dir / "comparison.npy"
        np.save(comparison, np.stack(images))
        import subprocess

        subprocess.run(
            [
                sys.executable,
                str(locations.root / "evaluation" / "evaluate.py"),
                "--comparison_npy",
                str(comparison),
                "--methods",
                "dynamic_k8,frozen_train_window,static_k8",
                "--method_labels",
                "dynamic_k8,frozen_train_window,static_k8",
                "--h5_path",
                str(h5_path),
                "--h5_sample_idx",
                str(sample_idx),
                "--dr",
                "60",
                "--out_dir",
                str(scene_dir / "metrics"),
                "--phantom_mode",
                "auto",
                "--phantom_source",
                "auto",
            ],
            cwd=locations.root,
            check=True,
        )
    if mc_rows:
        with (stage_dir / "monte_carlo_ssim_by_realization.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(mc_rows[0].keys()))
            writer.writeheader()
            writer.writerows(mc_rows)
        summaries = []
        for scene_name, method_name in ((row["scene"], row["method"]) for row in mc_rows):
            if any(item["scene"] == scene_name and item["method"] == method_name for item in summaries):
                continue
            values = [float(item["ssim_db"]) for item in mc_rows if item["scene"] == scene_name and item["method"] == method_name]
            env_values = [float(item["ssim_env"]) for item in mc_rows if item["scene"] == scene_name and item["method"] == method_name]
            summaries.append(
                {
                    "scene": scene_name,
                    "method": method_name,
                    "runs": len(values),
                    "ssim_db_mean": float(np.mean(values)),
                    "ssim_db_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "ssim_db_p05": float(np.quantile(values, 0.05)),
                    "ssim_db_p95": float(np.quantile(values, 0.95)),
                    "ssim_env_mean": float(np.mean(env_values)),
                    "ssim_env_std": float(np.std(env_values, ddof=1)) if len(env_values) > 1 else 0.0,
                }
            )
        with (stage_dir / "monte_carlo_ssim_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)


def load_or_capture(
    spec: ModelSpec,
    dataset: UltrasoundImageDataset,
    stage_dir: Path,
    frames: list[int],
    pixel_count: int,
    seed: int,
    checkpoint: Path,
    model_contract: dict[str, object],
) -> torch.Tensor:
    path = stage_dir / "frozen_train_window_raw_controls.pt"
    metadata_path = stage_dir / "frozen_train_window.json"
    expected = {
        "checkpoint": str(checkpoint.resolve()),
        "frames": frames,
        "pixel_count_per_frame": int(pixel_count),
        "control_count": CONTROL_COUNT,
        "model_contract": model_contract,
    }
    if path.exists():
        if not metadata_path.exists():
            raise RuntimeError(f"冻结窗缺少元数据，拒绝复用: {path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
        if mismatches:
            raise RuntimeError(f"冻结窗缓存不匹配 {mismatches}，请使用新的输出目录: {stage_dir}")
        vector = load_vector(path)
        if vector.numel() != CONTROL_COUNT:
            raise ValueError(f"已存在冻结窗不是 K8: {path}")
        return vector
    vector = capture_raw_window(spec, dataset, pixel_count, seed)
    torch.save(vector, path)
    metadata_path.write_text(
        json.dumps(
            {
                "source": "raw_control_before_output_domain_and_adc",
                "model_stage": stage_dir.name,
                "checkpoint": str(checkpoint.resolve()),
                "frames": frames,
                "pixel_count_per_frame": pixel_count,
                "angle_count": int(dataset.all_multi_I.shape[1]),
                "control_count": CONTROL_COUNT,
                "frozen": True,
                "model_contract": model_contract,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return vector


def load_static_cache(
    path: Path,
    stage_dir: Path,
    checkpoint: Path,
    train_frames: list[int],
    val_frames: list[int],
    initialization_source: str,
    model_contract: dict[str, object],
) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"静态窗缓存缺少训练合同，拒绝复用: {path}")
    expected = {
        "checkpoint": str(checkpoint.resolve()),
        "train_frames": train_frames,
        "val_frames": val_frames,
        "initialization": initialization_source,
        "model_contract": model_contract,
    }
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    if mismatches:
        raise RuntimeError(f"静态窗缓存不匹配 {mismatches}，请使用新的输出目录: {stage_dir}")
    return load_vector(path)


def run_stage(
    stage: Stage,
    checkpoint: Path,
    locations: Locations,
    train_frames: list[int],
    val_frames: list[int],
    args: argparse.Namespace,
    initial_controls: torch.Tensor | None,
    initialization_source: str,
) -> torch.Tensor:
    stage_dir = locations.output_dir / stage.name
    stage_dir.mkdir(parents=True, exist_ok=True)
    spec = load_spec(checkpoint, stage.kind, locations)
    train_dataset = make_dataset(train_frames, f"{stage.name}-static-train", locations.data_h5)
    val_dataset = make_dataset(val_frames, f"{stage.name}-static-val", locations.data_h5)
    _set_geometry(train_dataset)
    metadata = model_metadata(spec, checkpoint)
    (stage_dir / "model_contract.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    capture_pixels = min(
        args.smoke_pixels if args.smoke_test else training.runtime.NUM_PIXELS,
        training.runtime.NUM_PIXELS,
    )
    frozen = load_or_capture(
        spec,
        train_dataset,
        stage_dir,
        train_frames,
        capture_pixels,
        args.seed,
        checkpoint,
        metadata,
    )
    static_path = stage_dir / "best_static_k8.pt"
    if static_path.exists():
        static = load_static_cache(
            static_path,
            stage_dir,
            checkpoint,
            train_frames,
            val_frames,
            initialization_source,
            metadata,
        )
    else:
        train_pixels = args.smoke_pixels if args.smoke_test else int(training.args.train_pixels_per_image)
        val_pixels = args.smoke_pixels if args.smoke_test else int(training.args.validation_pixels_per_image)
        if train_pixels <= 0:
            train_pixels = training.runtime.NUM_PIXELS
        if val_pixels <= 0:
            val_pixels = training.runtime.NUM_PIXELS
        static = train_static(
            spec,
            train_dataset,
            val_dataset,
            stage_dir,
            args.smoke_epochs if args.smoke_test else int(training.args.epochs),
            min(train_pixels, training.runtime.NUM_PIXELS),
            min(val_pixels, training.runtime.NUM_PIXELS),
            args.seed + 1000,
            initial_controls,
            initialization_source,
            checkpoint,
            train_frames,
            val_frames,
            metadata,
        )
    eval_frames = val_frames[: args.eval_frame_limit] if args.eval_frame_limit else val_frames
    mc_runs = args.qat_mc_runs if stage.kind == "qat" else 1
    evaluate_validation(
        spec,
        frozen,
        static,
        eval_frames,
        stage_dir,
        args.seed + 200000,
        locations.data_h5,
        mc_runs,
    )
    if args.include_scenes and not args.smoke_test:
        evaluate_scenes(spec, frozen, static, stage_dir, args.seed + 300000, locations, mc_runs)
    del train_dataset, val_dataset
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return static


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="组A动态 K8 与静态 K8 窗口协议")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--hardware-config", type=Path)
    parser.add_argument("--simulation-h5", type=Path)
    parser.add_argument("--experiments-h5", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--fp32-checkpoint", type=Path)
    parser.add_argument("--qat4-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--smoke-pixels", type=int, default=64)
    parser.add_argument("--train-frame-limit", type=int)
    parser.add_argument("--val-frame-limit", type=int)
    parser.add_argument("--eval-frame-limit", type=int)
    parser.add_argument("--qat-mc-runs", type=int, default=1)
    parser.add_argument("--include-scenes", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前 PyTorch 没有可用 CUDA")
    return device


def main() -> None:
    global DEVICE
    args = parse_args()
    DEVICE = resolve_device(args.device)
    root = resolve_path(args.root, Path.cwd()) if args.root else ROOT
    config_path = (root / "train" / "config.yaml" if args.config is None else resolve_path(args.config, root)).resolve()
    selected_hardware = resolve_path(args.hardware_config, root) if args.hardware_config else None
    hardware_config = resolve_hardware_config(config_path, selected_hardware) or config_path.with_name("hardware_eval.yaml")
    split_path = resolve_path(args.split_file, root) if args.split_file else next(
        (path.resolve() for path in default_split_candidates(root) if path.exists()), None
    )
    if split_path is None:
        raise FileNotFoundError("找不到固定 train/val split CSV")
    locations = Locations(
        root=root,
        config=config_path,
        hardware_config=hardware_config,
        data_h5=configured_h5_files(config_path)[0],
        simulation_h5=resolve_path(args.simulation_h5, root) if args.simulation_h5 else root / "data" / "simulation.h5",
        experiments_h5=resolve_path(args.experiments_h5, root) if args.experiments_h5 else root / "data" / "experiments.h5",
        output_dir=(
            resolve_path(args.output_dir, root)
            if args.output_dir
            else root / "train" / "results" / "SI" / "evaluation" / "windows"
        ),
        split_file=split_path,
        stages=build_stages(
            root,
            resolve_path(args.fp32_checkpoint, root) if args.fp32_checkpoint else None,
            resolve_path(args.qat4_checkpoint, root) if args.qat4_checkpoint else None,
        ),
    )
    split = read_split(split_path, locations.data_h5)
    train_frames = split["train"][: args.train_frame_limit] if args.train_frame_limit else split["train"]
    val_frames = split["val"][: args.val_frame_limit] if args.val_frame_limit else split["val"]
    if args.smoke_test:
        train_frames = train_frames[:1]
        val_frames = val_frames[:1]
        output_root = locations.output_dir / "smoke"
    else:
        output_root = locations.output_dir
    locations = replace(locations, output_dir=output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_names = [str(stage.checkpoint_candidates[0]) for stage in locations.stages]
    if all("OneCycle" in name for name in checkpoint_names):
        checkpoint_group = "B"
    elif all("FixedLR" in name for name in checkpoint_names):
        checkpoint_group = "A"
    else:
        checkpoint_group = "custom"
    manifest = {
        "protocol": "dynamic_k8_vs_frozen_train_window_vs_static_k8",
        "group": checkpoint_group,
        "device": str(DEVICE),
        "split_file": str(split_path.resolve()),
        "root": str(locations.root),
        "config": str(locations.config),
        "hardware_config": str(locations.hardware_config),
        "data_h5": str(locations.data_h5),
        "simulation_h5": str(locations.simulation_h5),
        "experiments_h5": str(locations.experiments_h5),
        "output_dir": str(output_root),
        "train_frames": train_frames,
        "val_frames": val_frames,
        "control_count": CONTROL_COUNT,
        "smoke_test": bool(args.smoke_test),
        "stages": {},
    }
    for stage in locations.stages:
        checkpoint = resolve_existing(stage.checkpoint_candidates, stage.name)
        manifest["stages"][stage.name] = {"checkpoint": str(checkpoint), "kind": stage.kind}
    (output_root / "protocol_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fp32_static = None
    for stage in locations.stages:
        checkpoint = Path(manifest["stages"][stage.name]["checkpoint"])
        if stage.name == "FP32":
            initial_controls = None
            initialization_source = "uniform_positive_1_over_K"
        else:
            fp32_path = output_root / "FP32" / "best_static_k8.pt"
            if fp32_static is None:
                if not fp32_path.exists():
                    raise RuntimeError("QAT4 静态窗需要先生成 FP32 best_static_k8.pt")
                fp32_static = load_vector(fp32_path)
            initial_controls = fp32_static
            initialization_source = str(fp32_path.resolve())
        result = run_stage(
            stage,
            checkpoint,
            locations,
            train_frames,
            val_frames,
            args,
            initial_controls,
            initialization_source,
        )
        if stage.name == "FP32":
            fp32_static = result
        print(f"DONE {stage.name}", flush=True)


if __name__ == "__main__":
    main()

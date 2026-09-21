from __future__ import annotations

import dataclasses
import csv
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.ticker import FixedLocator

import mban as training
from eval_core.config import ModelSpec, ROOT, activate_model_runtime
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import run_one
from evaluation.evaluate import local_ssim
from evaluation.plot_metrics import MAX_CURVE_METHODS_PER_PAGE, build_page_method_colors
from mban_core.data import resolve_angle_indices

def prepare_noise_realization(hardware: object | None, seed: int) -> dict[str, object] | None:
    if hardware is None:
        return None
    hardware.config = dataclasses.replace(hardware.config, noise_seed=int(seed))
    hardware._realization_index = 0
    hardware._realization_seed_base = None
    hardware.reset_noise_counter()
    if hardware.config.noise_enabled:
        hardware.begin_noise_realization()
    return hardware.realization_state()
def reconstruct(
    spec: ModelSpec,
    data: SceneData,
    device: torch.device,
    micro_batch: int,
    collect_weight_stats: bool = True,
    path_collector: WeightPathCollector | None = None,
) -> tuple[np.ndarray, dict[str, float], np.ndarray, float]:
    if micro_batch <= 0:
        raise ValueError("micro_batch must be positive")
    backend = spec.training_module
    if collect_weight_stats:
        require_explicit_weight_evaluation([spec], "权重统计")
    activate_model_runtime(spec, backend)
    configure_geometry(data, backend)
    if backend.args.dynamic_aperture:
        derived_channels, _ = backend.derive_network_channels(
            data.z_grid,
            data.pitch,
            float(backend.args.f_number),
            data.channels,
            None,
        )
        if derived_channels > spec.model.network_channels:
            raise ValueError("验证网格所需动态孔径超过检查点训练时的最大孔径")
    elif data.channels != spec.model.network_channels:
        raise ValueError(
            f"固定孔径模型{spec.display_name}要求输入通道数等于 network_channels={spec.model.network_channels}，"
            f"当前数据为 {data.channels}"
        )
    rf_i, rf_q, t_start = data.rf_i.to(device), data.rf_q.to(device), data.t_start.to(device)
    valid_time_samples = torch.full((rf_i.shape[0],), rf_i.shape[2], device=device, dtype=torch.long)
    x_grid, z_grid, angles = data.x_grid.to(device), data.z_grid.view(-1).to(device), data.angles.to(device)
    angle_idx = torch.arange(angles.numel(), device=device)
    offsets = torch.arange(0, 1, device=device)
    out_i, out_q = [], []
    tv_sum = tv_count = d2_sum = d2_count = 0.0
    moment_sums = {
        key: [0.0, 0.0]
        for key in (
            "weight_mean_real",
            "weight_mean_imag",
            "weight_mean_magnitude",
            "weight_rms",
            "weight_total_energy",
            "E_non_dc",
            "E_low_ac",
            "E_high_ac",
            "high_freq_ratio_ac",
        )
    } if collect_weight_stats else {}
    cumulative_sum = None
    count = 0
    timing_events = []
    elapsed = 0.0
    timing_stream = torch.cuda.current_stream(device) if device.type == "cuda" else None
    with torch.inference_mode():
        for offset in range(0, data.pixels, micro_batch):
            if device.type == "cuda":
                timing_start = torch.cuda.Event(enable_timing=True)
                timing_end = torch.cuda.Event(enable_timing=True)
                timing_start.record(timing_stream)
            else:
                timing_started = time.perf_counter()
            pixel = torch.arange(offset, min(offset + micro_batch, data.pixels), device=device)
            image_index = torch.zeros(pixel.numel(), dtype=torch.long, device=device)
            ext_i, ext_q, tof, tx_tof = backend.extract_windows_on_gpu(
                rf_i,
                rf_q,
                t_start,
                image_index,
                pixel,
                data.fs,
                offsets,
                angle_idx,
                rf_i.shape[2],
                valid_time_samples,
                data.pitch,
                x_grid,
                z_grid,
                angles,
            )
            center_t = ext_i.shape[-1] // 2
            i_sample, q_sample = ext_i[..., center_t], ext_q[..., center_t]
            phase = 2.0 * np.pi * data.fc * (tof - tx_tof.unsqueeze(2))
            i_aligned = i_sample * torch.cos(phase) - q_sample * torch.sin(phase)
            q_aligned = i_sample * torch.sin(phase) + q_sample * torch.cos(phase)
            if backend.args.dynamic_aperture:
                mask, aperture_start, aperture_size = backend.compute_aperture_mask(
                    pixel,
                    z_grid,
                    x_grid,
                    data.pitch,
                    return_geometry=True,
                )
                if int(aperture_size.max()) > spec.model.network_channels:
                    raise ValueError("验证网格所需动态孔径超过检查点训练时的最大孔径")
            else:
                mask = aperture_start = aperture_size = None
            i_use, q_use, weights, controls, input_scale, input_mean, raw_controls = backend.predict_aperture_weights(
                spec.model,
                i_aligned,
                q_aligned,
                mask,
                aperture_start,
                aperture_size,
                depth_index=pixel // data.width,
                depth_count=data.height,
            )
            if path_collector is not None:
                path_collector.add(raw_controls, controls, weights, mask)
            pred_i, pred_q = backend.beamform_iq_with_tx_phase(
                weights,
                i_use,
                q_use,
                tx_tof,
                input_scale,
                input_mean,
                controls=controls,
                aperture_start=aperture_start,
                aperture_size=aperture_size,
                network_channels=spec.model.network_channels,
            )
            if device.type == "cuda":
                timing_end.record(timing_stream)
                timing_events.append((timing_start, timing_end))
            else:
                elapsed += time.perf_counter() - timing_started
            out_i.append(pred_i.cpu())
            out_q.append(pred_q.cpu())
            if collect_weight_stats:
                effective_weights = backend.effective_aperture_weights(weights)
                wr, wi = backend.split_complex_weights(effective_weights)
                active = torch.ones_like(wr) if mask is None else mask.to(wr.dtype).unsqueeze(1).expand_as(wr)
                pair = active[..., 1:] * active[..., :-1]
                triple = active[..., 2:] * active[..., 1:-1] * active[..., :-2]
                d1r, d1i = wr[..., 1:] - wr[..., :-1], wi[..., 1:] - wi[..., :-1]
                d2r = wr[..., 2:] - 2.0 * wr[..., 1:-1] + wr[..., :-2]
                d2i = wi[..., 2:] - 2.0 * wi[..., 1:-1] + wi[..., :-2]
                tv_sum += (torch.sqrt(d1r.square() + d1i.square()) * pair).sum().item()
                tv_count += pair.sum().item()
                d2_sum += ((d2r.square() + d2i.square()) * triple).sum().item()
                d2_count += triple.sum().item()
                active_count = active.sum(dim=-1).clamp_min(1.0)
                mean_real = (wr * active).sum(dim=-1) / active_count
                mean_imag = (wi * active).sum(dim=-1) / active_count
                energy = ((wr.square() + wi.square()) * active).sum(dim=-1)
                centered = torch.complex(
                    (wr - mean_real.unsqueeze(-1)) * active,
                    (wi - mean_imag.unsqueeze(-1)) * active,
                )
                spectrum = torch.fft.fft(centered, dim=-1, norm="ortho").abs().square()
                frequencies = torch.fft.fftfreq(wr.shape[-1], device=wr.device)
                order = torch.argsort(frequencies.abs())
                spectrum = spectrum[..., order].reshape(-1, spectrum.shape[-1])
                high = frequencies[order].abs() >= 0.375
                non_dc = spectrum.sum(dim=-1)
                e_low_ac = spectrum[:, ~high].sum(dim=-1)
                e_high_ac = spectrum[:, high].sum(dim=-1)
                high_freq_ratio_ac = torch.where(
                    non_dc > 1.0e-12,
                    e_high_ac / non_dc,
                    torch.zeros_like(non_dc),
                )
                metric_values = {
                    "weight_mean_real": mean_real.reshape(-1),
                    "weight_mean_imag": mean_imag.reshape(-1),
                    "weight_mean_magnitude": torch.sqrt(mean_real.square() + mean_imag.square()).reshape(-1),
                    "weight_rms": torch.sqrt(energy / active_count).reshape(-1),
                    "weight_total_energy": energy.reshape(-1),
                    "E_non_dc": non_dc,
                    "E_low_ac": e_low_ac,
                    "E_high_ac": e_high_ac,
                    "high_freq_ratio_ac": high_freq_ratio_ac,
                }
                for key, values in metric_values.items():
                    moment_sums[key][0] += values.sum().item()
                    moment_sums[key][1] += values.square().sum().item()
                cumulative = spectrum.sum(dim=0).cumsum(dim=0)
                cumulative_sum = cumulative if cumulative_sum is None else cumulative_sum + cumulative
                count += spectrum.shape[0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        elapsed = sum(start.elapsed_time(end) for start, end in timing_events) / 1000.0
    i_image = torch.cat(out_i).reshape(data.height, data.width).numpy()
    q_image = torch.cat(out_q).reshape(data.height, data.width).numpy()
    envelope = np.sqrt(i_image * i_image + q_image * q_image)
    if bool(spec.runtime["use_tgc"]):
        tgc = 10.0 ** (
            float(spec.runtime.get("tgc_alpha", 0.5))
            * (data.fc / 1.0e6)
            * (data.z_grid.numpy() * 100.0)
            * 2.0
            / 20.0
        )
        envelope = envelope * tgc[:, None]
    peak = float(np.max(envelope))
    if not np.isfinite(peak) or peak <= 0:
        raise FloatingPointError(f"模型 {spec.display_name} 输出包络无有效正峰值")
    db = (20.0 * np.log10(np.maximum(envelope / peak, 1.0e-12))).astype(np.float32)
    if collect_weight_stats:
        stats = {"weight_samples": count, "TV": tv_sum / max(tv_count, 1.0), "E2": d2_sum / max(d2_count, 1.0)}
        for key, (value_sum, square_sum) in moment_sums.items():
            mean = value_sum / max(count, 1)
            variance = max((square_sum - value_sum * value_sum / max(count, 1)) / max(count - 1, 1), 0.0)
            stats[key] = mean
            stats[f"{key}_std"] = math.sqrt(variance)
        cumulative = (cumulative_sum / max(count, 1)).cpu().numpy()
    else:
        stats = {}
        cumulative = np.empty(0, dtype=np.float32)
    return db, stats, cumulative, elapsed


def monte_carlo_reconstruct(
    spec: ModelSpec,
    data: SceneData,
    device: torch.device,
    micro_batch: int,
    runs: int,
    base_seed: int | None = None,
    return_realizations: bool = False,
    collect_weight_stats: bool = True,
) -> (
    tuple[np.ndarray, dict[str, float], np.ndarray, float, np.ndarray]
    | tuple[np.ndarray, dict[str, float], np.ndarray, float, np.ndarray, list[np.ndarray]]
):
    """Reconstruct one scene with hardware noise enabled and average ``runs`` realizations.

    This image-level helper is not the formal multi-frame MC protocol; use
    ``diagnostics.run_mc`` to keep one realization fixed across all test frames.

    Returns the mean dB image, statistics of the last run, cumulative spectrum,
    mean inference time, and per-pixel dB std across runs.
    """
    if runs < 1:
        raise ValueError("runs must be positive")
    if micro_batch <= 0:
        raise ValueError("micro_batch must be positive")
    hardware = getattr(spec.model, "hardware_qat", None)
    has_noise = hardware is not None and getattr(hardware.config, "noise_enabled", False)
    if runs > 1 and not has_noise:
        raise ValueError(f"模型 {spec.display_name} 未启用硬件噪声，不能执行 {runs}-run Monte Carlo；请启用 noise_enabled")
    if base_seed is not None and int(base_seed) < 0:
        raise ValueError("base_seed 必须非负")
    if hardware is not None and base_seed is not None:
        hardware.config = dataclasses.replace(hardware.config, noise_seed=int(base_seed))
    if runs <= 1:
        if hardware is not None and hardware.config.noise_enabled:
            hardware.reset_noise_counter()
            hardware.begin_noise_realization()
        image, stats, cumulative, elapsed = reconstruct(
            spec, data, device, micro_batch, collect_weight_stats=collect_weight_stats
        )
        base = (image, stats, cumulative, elapsed, np.zeros((data.height, data.width), dtype=np.float32))
        return (*base, [image]) if return_realizations else base
    images = []
    elapsed_sum = 0.0
    last = None
    hardware.reset_noise_counter()
    for _ in range(runs):
        hardware.begin_noise_realization()
        image, stats, cumulative, elapsed = reconstruct(
            spec, data, device, micro_batch, collect_weight_stats=collect_weight_stats
        )
        images.append(image)
        elapsed_sum += elapsed
        last = (image, stats, cumulative, elapsed)
    image, stats, cumulative, elapsed = last
    stack = np.stack(images)
    mean_image = stack.mean(axis=0).astype(np.float32)
    std_image = stack.std(axis=0, ddof=0).astype(np.float32)
    base = (mean_image, stats, cumulative, elapsed_sum / runs, std_image)
    return (*base, images) if return_realizations else base

def standard_baselines(data: SceneData, output: Path, f_number: float) -> tuple[np.ndarray, np.ndarray]:
    common = [
        "--h5_path",
        str(data.h5_path),
        "--h5_sample_idx",
        str(data.sample_idx),
        "--output_dir",
        str(output),
        "--select_angles",
        "1",
        "--f_number",
        str(f_number),
        "--window",
        "rect",
        "--interp",
        "linear",
        "--dr",
        "60",
    ]
    subprocess.run([sys.executable, "-m", "algorithms.das", *common], cwd=ROOT, check=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "algorithms.mv",
            *common,
            "--mv_dl",
            "0",
            "--subarray_ratio",
            "0.25",
            "--temporal_win",
            "3",
        ],
        cwd=ROOT,
        check=True,
    )
    return np.load(output / "das" / "das.npy"), np.load(output / "mv" / "mv.npy")


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean_ci(values: np.ndarray, seed: int, samples: int = 10000) -> tuple[float, float]:
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, values.size, size=(int(samples), values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def metric_pair(image_db: np.ndarray, gt_db: np.ndarray) -> tuple[float, float]:
    image_display = np.clip((image_db + 60.0) / 60.0, 0.0, 1.0)
    gt_display = np.clip((gt_db + 60.0) / 60.0, 0.0, 1.0)
    image_env = 10.0 ** (image_db / 20.0)
    gt_env = 10.0 ** (gt_db / 20.0)
    return (
        float(local_ssim(image_display, gt_display, data_range=1.0)),
        float(local_ssim(image_env, gt_env, data_range=1.0)),
    )




def require_explicit_weight_evaluation(specs: list[ModelSpec], purpose: str) -> None:
    factorized = [spec.display_name for spec in specs if str(spec.runtime.get("beamforming_implementation")) == "factorized"]
    if factorized:
        raise ValueError(f"{purpose}仅支持 E/explicit；F/factorized 不保留 M 维权重: {', '.join(factorized)}")

class _IndexedStats:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.count: torch.Tensor | None = None
        self.total: torch.Tensor | None = None
        self.total_sq: torch.Tensor | None = None
        self.negative: torch.Tensor | None = None
        self.minimum: torch.Tensor | None = None
        self.maximum: torch.Tensor | None = None
        self.sample_values = torch.empty((0, 0), dtype=torch.float32)
        self.sample_active = torch.empty((0, 0), dtype=torch.bool)
        self.sample_keys = torch.empty(0, dtype=torch.float64)
        self.rng = np.random.default_rng(1701)

    def add(self, value: torch.Tensor, active: torch.Tensor | None = None) -> None:
        matrix = value.detach().float().reshape(-1, value.shape[-1]).cpu()
        if matrix.numel() == 0:
            return
        valid = (
            torch.ones_like(matrix, dtype=torch.bool)
            if active is None
            else active.detach().bool().reshape(-1, matrix.shape[-1]).cpu()
        )
        if self.count is None:
            width = matrix.shape[-1]
            self.count = torch.zeros(width, dtype=torch.long)
            self.total = torch.zeros(width)
            self.total_sq = torch.zeros(width)
            self.negative = torch.zeros(width, dtype=torch.long)
            self.minimum = torch.full((width,), float("inf"))
            self.maximum = torch.full((width,), float("-inf"))
            self.sample_values = torch.empty((0, width), dtype=torch.float32)
            self.sample_active = torch.empty((0, width), dtype=torch.bool)
        self.count += valid.sum(dim=0)
        self.total += (matrix * valid).sum(dim=0)
        self.total_sq += (matrix.square() * valid).sum(dim=0)
        self.negative += ((matrix < 0.0) & valid).sum(dim=0)
        batch_min = torch.where(valid, matrix, torch.full_like(matrix, float("inf"))).amin(dim=0)
        batch_max = torch.where(valid, matrix, torch.full_like(matrix, float("-inf"))).amax(dim=0)
        has_values = valid.any(dim=0)
        self.minimum = torch.where(has_values, torch.minimum(self.minimum, batch_min), self.minimum)
        self.maximum = torch.where(has_values, torch.maximum(self.maximum, batch_max), self.maximum)
        if self.limit <= 0:
            return
        keys = torch.from_numpy(self.rng.random(matrix.shape[0]))
        pool_keys = torch.cat((self.sample_keys, keys))
        keep = min(self.limit, pool_keys.numel())
        selected_keys, selected = torch.topk(pool_keys, keep, sorted=False)
        self.sample_keys = selected_keys
        self.sample_values = torch.cat((self.sample_values, matrix))[selected]
        self.sample_active = torch.cat((self.sample_active, valid))[selected]

    def rows(self) -> list[dict[str, object]]:
        if self.count is None:
            return []
        result = []
        for index in range(self.count.numel()):
            count = int(self.count[index])
            if count == 0:
                continue
            mean = float(self.total[index] / count)
            variance = max(float(self.total_sq[index] / count) - mean * mean, 0.0)
            sample = self.sample_values[:, index][self.sample_active[:, index]]
            quantiles = (
                torch.quantile(sample, torch.tensor([0.05, 0.50, 0.95, 0.99])).tolist()
                if sample.numel()
                else [float("nan")] * 4
            )
            result.append(
                {
                    "index": index,
                    "samples": count,
                    "quantile_samples": int(sample.numel()),
                    "mean": mean,
                    "std": math.sqrt(variance),
                    "negative_fraction": int(self.negative[index]) / count,
                    "min": float(self.minimum[index]),
                    "p05": float(quantiles[0]),
                    "p50": float(quantiles[1]),
                    "p95": float(quantiles[2]),
                    "p99": float(quantiles[3]),
                    "max": float(self.maximum[index]),
                }
            )
        return result


class WeightPathCollector:
    def __init__(self, sample_limit: int = 100000) -> None:
        self.sample_limit = sample_limit
        self._stats: dict[tuple[str, str], _IndexedStats] = {}

    def _add(
        self,
        stage: str,
        component: str,
        value: torch.Tensor,
        active: torch.Tensor | None = None,
    ) -> None:
        stats = self._stats.setdefault((stage, component), _IndexedStats(self.sample_limit))
        stats.add(value, active)

    @staticmethod
    def _components(value: torch.Tensor) -> tuple[tuple[str, torch.Tensor], ...]:
        if training.runtime.args.output_weights == "complex":
            real, imag = training.split_complex_weights(value)
            return ("real", real), ("imag", imag)
        return (("real", value),)

    def add(
        self,
        raw_controls: torch.Tensor,
        controls: torch.Tensor,
        weights: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> None:
        for component, value in self._components(raw_controls):
            self._add("raw_control", component, value)
        for component, value in self._components(controls):
            self._add("adc_control", component, value)
        if weights is None:
            return
        effective = training.effective_aperture_weights(weights)
        wr, wi = training.split_complex_weights(effective)
        active = torch.ones_like(wr, dtype=torch.bool) if mask is None else mask.bool().unsqueeze(1).expand_as(wr)
        self._add("physical_weight", "real", wr, active)
        if training.runtime.args.output_weights == "complex":
            self._add("physical_weight", "imag", wi, active)
            self._add("physical_weight", "magnitude", torch.sqrt(wr.square() + wi.square()), active)

    def rows(self) -> list[dict[str, object]]:
        rows = []
        for (stage, component), stats in sorted(self._stats.items()):
            for row in stats.rows():
                rows.append({"stage": stage, "component": component, **row})
        return rows
def conventional_weights_at_pixel(
    data: SceneData,
    device: torch.device,
    z_index: int,
    x_index: int,
    f_number: float,
    subarray_ratio: float = 0.25,
    temporal_win: int = 9,
) -> dict[str, np.ndarray]:
    configure_geometry(data)
    pixel = torch.tensor([z_index * data.width + x_index], device=device)
    image_index = torch.zeros(1, dtype=torch.long, device=device)
    rf_i, rf_q, t_start = data.rf_i.to(device), data.rf_q.to(device), data.t_start.to(device)
    valid_time_samples = torch.full((rf_i.shape[0],), rf_i.shape[2], device=device, dtype=torch.long)
    x_grid, z_grid, angles = data.x_grid.to(device), data.z_grid.view(-1).to(device), data.angles.to(device)
    half = temporal_win // 2
    offsets = torch.arange(-half, half + 1, device=device)
    with torch.inference_mode():
        ext_i, ext_q, tof, tx_tof = training.extract_windows_on_gpu(
            rf_i,
            rf_q,
            t_start,
            image_index,
            pixel,
            data.fs,
            offsets,
            torch.arange(angles.numel(), device=device),
            rf_i.shape[2],
            valid_time_samples,
            data.pitch,
            x_grid,
            z_grid,
            angles,
        )
        phase = 2.0 * np.pi * data.fc * (tof - tx_tof.unsqueeze(2))
        i_aligned = ext_i * torch.cos(phase).unsqueeze(-1) - ext_q * torch.sin(phase).unsqueeze(-1)
        q_aligned = ext_i * torch.sin(phase).unsqueeze(-1) + ext_q * torch.cos(phase).unsqueeze(-1)
        depth = z_grid[z_index]
        lateral = x_grid[x_index]
        center = torch.round(lateral / data.pitch + (data.channels - 1) / 2.0).long().clamp(0, data.channels - 1)
        aperture_size = torch.floor(depth / (f_number * data.pitch)).long() + 1
        if int(aperture_size.item()) < 1 or int(aperture_size.item()) > data.channels:
            raise ValueError("传统基线动态孔径超出物理通道范围")
        aperture_start = torch.clamp(center - aperture_size // 2, min=0)
        aperture_start = torch.minimum(aperture_start, data.channels - aperture_size)
        start, size = int(aperture_start.item()), int(aperture_size.item())
        active = torch.zeros(data.channels, dtype=torch.bool, device=device)
        active[start : start + size] = True
        das = torch.zeros(data.channels, dtype=torch.float32, device=device)
        das[start : start + size] = 1.0 / size

        samples = torch.complex(i_aligned[0, :, start : start + size], q_aligned[0, :, start : start + size])
        subarray_size = min(max(int(size * subarray_ratio), 2), size)
        subarray_count = size - subarray_size + 1
        subarrays = samples.unfold(1, subarray_size, 1).permute(0, 3, 1, 2)
        snapshots = subarrays.reshape(angles.numel(), subarray_size, subarray_count * temporal_win)
        covariance_forward = snapshots @ snapshots.mH / max(subarray_count * temporal_win, 1)
        covariance = 0.5 * (covariance_forward + covariance_forward.mH)
        ones = torch.ones((angles.numel(), subarray_size, 1), dtype=torch.complex64, device=device)
        try:
            solved = torch.linalg.solve(covariance, ones)
        except RuntimeError:
            solved = torch.linalg.pinv(covariance) @ ones
        if not torch.isfinite(solved).all():
            solved = torch.linalg.pinv(covariance) @ ones
        mv_subarray = solved / (ones.mH @ solved + 1.0e-12)
        mv_active = torch.zeros((angles.numel(), size), dtype=torch.complex64, device=device)
        for shift in range(subarray_count):
            mv_active[:, shift : shift + subarray_size] += mv_subarray[..., 0] / subarray_count
        mv_physical = torch.zeros(data.channels, dtype=torch.complex64, device=device)
        mv_physical[start : start + size] = mv_active.mean(dim=0)
    return {
        "active": active.cpu().numpy(),
        "das_real": das.cpu().numpy(),
        "das_imag": np.zeros(data.channels, dtype=np.float32),
        "mv_real": mv_physical.real.cpu().numpy(),
        "mv_imag": mv_physical.imag.cpu().numpy(),
    }


def weight_at_pixel(
    spec: ModelSpec, data: SceneData, device: torch.device, z_index: int, x_index: int
) -> dict[str, np.ndarray]:
    backend = spec.training_module
    activate_model_runtime(spec, backend)
    configure_geometry(data, backend)
    pixel = torch.tensor([z_index * data.width + x_index], device=device)
    image_index = torch.zeros(1, dtype=torch.long, device=device)
    rf_i, rf_q, t_start = data.rf_i.to(device), data.rf_q.to(device), data.t_start.to(device)
    valid_time_samples = torch.full((rf_i.shape[0],), rf_i.shape[2], device=device, dtype=torch.long)
    x_grid, z_grid, angles = data.x_grid.to(device), data.z_grid.view(-1).to(device), data.angles.to(device)
    with torch.inference_mode():
        ext_i, ext_q, tof, tx_tof = backend.extract_windows_on_gpu(
            rf_i,
            rf_q,
            t_start,
            image_index,
            pixel,
            data.fs,
            torch.arange(0, 1, device=device),
            torch.arange(angles.numel(), device=device),
            rf_i.shape[2],
            valid_time_samples,
            data.pitch,
            x_grid,
            z_grid,
            angles,
        )
        i_sample, q_sample = ext_i[..., 0], ext_q[..., 0]
        phase = 2.0 * np.pi * data.fc * (tof - tx_tof.unsqueeze(2))
        i_aligned = i_sample * torch.cos(phase) - q_sample * torch.sin(phase)
        q_aligned = i_sample * torch.sin(phase) + q_sample * torch.cos(phase)
        if backend.args.dynamic_aperture:
            mask, aperture_start, aperture_size = backend.compute_aperture_mask(
                pixel,
                z_grid,
                x_grid,
                data.pitch,
                return_geometry=True,
            )
        else:
            mask = torch.ones((1, data.channels), device=device)
            aperture_start = aperture_size = None
        _, _, weights, controls, _, _, _ = backend.predict_aperture_weights(
            spec.model,
            i_aligned,
            q_aligned,
            mask if backend.args.dynamic_aperture else None,
            aperture_start if backend.args.dynamic_aperture else None,
            aperture_size if backend.args.dynamic_aperture else None,
        )
        wr, wi = backend.split_complex_weights(weights)
        control_r, control_i = backend.split_complex_weights(controls)
    physical_r = wr[0, 0].cpu().numpy()
    physical_i = wi[0, 0].cpu().numpy()
    active = mask[0].bool().cpu().numpy()
    if backend.args.dynamic_aperture and aperture_start is not None:
        aperture_start_int = int(aperture_start[0].item())
        aperture_size_int = int(aperture_size[0].item())
    else:
        aperture_start_int, aperture_size_int = 0, int(data.channels)
    return {
        "control_real": control_r[0, 0].cpu().numpy(),
        "control_imag": control_i[0, 0].cpu().numpy(),
        "physical_real": physical_r,
        "physical_imag": physical_i,
        "active": active,
        "aperture_start": aperture_start_int,
        "aperture_size": aperture_size_int,
    }


def _model_curve_colors(labels):
    color_map = build_page_method_colors(labels)
    return [color_map[label] for label in labels]


def save_weight_curves(
    scene_dir: Path,
    specs: list[ModelSpec],
    data: SceneData,
    device: torch.device,
    z_index: int,
    x_index: int,
    f_number: float,
) -> None:
    require_explicit_weight_evaluation(specs, "权重曲线")
    individual_dir = scene_dir / "weight_controls_individual"
    individual_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "weight_curves.png",
        "weight_controls.png",
        "weight_local_curves.csv",
        "weight_local_curves.png",
        "weight_differences.csv",
        "weight_differences.png",
        "weight_spectra.png",
        "weight_spectra_demeaned.png",
        "weight_cumulative_spectra.csv",
        "weight_cumulative_spectra.png",
    ):
        (scene_dir / name).unlink(missing_ok=True)
    for pattern in ("weight_curves_page*.png", "weight_controls_page*.png", "weight_controls_*.png"):
        for path in scene_dir.glob(pattern):
            path.unlink(missing_ok=True)
    for path in individual_dir.glob("weight_controls_*.png"):
        path.unlink(missing_ok=True)
    curves = {spec.display_name: weight_at_pixel(spec, data, device, z_index, x_index) for spec in specs}
    baseline = conventional_weights_at_pixel(data, device, z_index, x_index, f_number=f_number)
    effective_curves = {}
    for spec in specs:
        curve = curves[spec.display_name]
        physical_real = curve["physical_real"].copy()
        physical_imag = curve["physical_imag"].copy()
        if spec.runtime["output_domain"] == "nonnegative" and spec.runtime["unity_constraint"] == "hard":
            active = curve["active"].astype(bool)
            normalization = float(physical_real[active].sum())
            if abs(normalization) > 1.0e-12:
                physical_real /= normalization
                physical_imag /= normalization
        effective_curves[spec.display_name] = {
            **curve,
            "effective_real": physical_real,
            "effective_imag": physical_imag,
        }
    rows = []
    for channel in range(data.channels):
        row = {
            "channel": channel,
            "active": int(baseline["active"][channel]),
            "DAS_real": float(baseline["das_real"][channel]),
            "DAS_imag": 0.0,
            "MV_real": float(baseline["mv_real"][channel]),
            "MV_imag": float(baseline["mv_imag"][channel]),
        }
        for label, curve in effective_curves.items():
            raw_curve = curves[label]
            wr, wi, active = curve["effective_real"], curve["effective_imag"], curve["active"]
            row[f"{label}_active"] = int(active[channel])
            row[f"{label}_real"] = float(wr[channel])
            row[f"{label}_imag"] = float(wi[channel])
            row[f"{label}_raw_real"] = float(raw_curve["physical_real"][channel])
            row[f"{label}_raw_imag"] = float(raw_curve["physical_imag"][channel])
        rows.append(row)
    write_rows(scene_dir / "weight_curves.csv", rows)
    control_rows = []
    for index in range(max(len(curve["control_real"]) for curve in curves.values())):
        row = {"control_index": index}
        for label, curve in curves.items():
            if index < len(curve["control_real"]):
                row[f"{label}_real"] = float(curve["control_real"][index])
                row[f"{label}_imag"] = float(curve["control_imag"][index])
        control_rows.append(row)
    write_rows(scene_dir / "weight_controls.csv", control_rows)
    channels = np.arange(data.channels)
    curve_count = len(effective_curves)
    model_items = list(effective_curves.items())
    max_models_per_page = max(MAX_CURVE_METHODS_PER_PAGE - 2, 1)
    curve_page_count = math.ceil(curve_count / max_models_per_page)
    for page_idx, start in enumerate(range(0, curve_count, max_models_per_page), start=1):
        page_items = model_items[start : start + max_models_per_page]
        page_colors = _model_curve_colors([label for label, _ in page_items])
        page_suffix = "" if page_idx == 1 else f"_page{page_idx}"
        page_title_suffix = "" if curve_page_count == 1 else f" (page {page_idx}/{curve_page_count})"
        page_curve_count = len(page_items)
        curve_figsize = (
            max(11.5, 10.0 + 0.16 * page_curve_count),
            max(5.2, 5.0 + 0.02 * page_curve_count),
        )
        fig, ax = plt.subplots(figsize=curve_figsize)
        active_indices = np.flatnonzero(baseline["active"])
        if active_indices.size:
            ax.axvspan(
                active_indices[0] - 0.5,
                active_indices[-1] + 0.5,
                color="#BDBDBD",
                alpha=0.22,
                label=f"active M={active_indices.size}",
            )
        ax.plot(channels, baseline["das_real"], color="black", linewidth=1.5, label="DAS")
        ax.plot(channels, baseline["mv_real"], color="#D55E00", linewidth=1.35, label="MV real")
        if np.any(np.abs(baseline["mv_imag"]) > 1.0e-10):
            ax.plot(
                channels,
                baseline["mv_imag"],
                color="#D55E00",
                linestyle="--",
                linewidth=1.1,
                label="MV imag",
            )
        for page_index, (label, curve) in enumerate(page_items):
            wr, wi = curve["effective_real"], curve["effective_imag"]
            color = page_colors[page_index]
            real_label = label if not np.any(np.abs(wi) > 1.0e-12) else f"{label} real"
            ax.plot(channels, wr, color=color, label=real_label, linewidth=1.35)
            if np.any(np.abs(wi) > 1.0e-12):
                ax.plot(channels, wi, color=color, linestyle="--", label=f"{label} imag", linewidth=1.05)
        ax.set_title(f"{scene_dir.name}: z={z_index}, x={x_index}{page_title_suffix}")
        ax.set_xlabel("Physical channel")
        ax.set_ylabel("Effective aperture weight (sum=1 for nonnegative+hard)")
        ax.set_xlim(-0.5, data.channels - 0.5)
        ax.grid(alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.78, 0.5),
            ncol=2 if len(handles) > 4 else 1,
            fontsize=7,
        )
        fig.tight_layout(rect=(0.0, 0.0, 0.76, 1.0))
        fig.savefig(scene_dir / f"weight_curves{page_suffix}.png", dpi=220)
        plt.close(fig)
    anchor_indices = {}
    weight_values = []
    for label, curve in effective_curves.items():
        spec = next(s for s in specs if s.display_name == label)
        network = int(spec.model.network_channels)
        controls = curve["control_real"]
        start = int(curve["aperture_start"])
        size = int(curve["aperture_size"])
        if controls.size == network:
            packed_start = (network - size) // 2
            packed_idx = np.arange(packed_start, packed_start + size)
            anchors = start + (packed_idx - packed_start)
        elif controls.size > 1:
            anchors = start + np.arange(controls.size) * (size - 1) / max(controls.size - 1, 1)
        else:
            anchors = np.array([start])
        anchor_indices[label] = np.clip(np.round(anchors).astype(int), 0, data.channels - 1)
        weight_values.append(curve["effective_real"])
        if np.any(np.abs(curve["effective_imag"]) > 1.0e-12):
            weight_values.append(curve["effective_imag"])
    y_values = np.concatenate(weight_values)
    y_min = float(np.nanmin(y_values))
    y_max = float(np.nanmax(y_values))
    y_margin = max((y_max - y_min) * 0.05, 1.0e-6)
    shared_ylim = (y_min - y_margin, y_max + y_margin)
    for page_idx, start in enumerate(range(0, curve_count, max_models_per_page), start=1):
        page_items = model_items[start : start + max_models_per_page]
        page_colors = _model_curve_colors([label for label, _ in page_items])
        page_suffix = "" if page_idx == 1 else f"_page{page_idx}"
        page_title_suffix = "" if curve_page_count == 1 else f" (page {page_idx}/{curve_page_count})"
        page_curve_count = len(page_items)
        curve_figsize = (
            max(11.5, 10.0 + 0.16 * page_curve_count),
            max(5.2, 5.0 + 0.02 * page_curve_count),
        )
        fig, ax = plt.subplots(figsize=curve_figsize)
        for page_index, (label, curve) in enumerate(page_items):
            color = page_colors[page_index]
            wr = curve["effective_real"]
            ax.plot(channels, wr, color=color, label=f"{label} (controls={curve['control_real'].size})", linewidth=1.35)
            ax.plot(
                anchor_indices[label],
                wr[anchor_indices[label]],
                marker="o",
                linestyle="none",
                markersize=5,
                color=color,
            )
        ax.set_title(
            f"{scene_dir.name}: interpolated aperture weights with control anchors at z={z_index}, x={x_index}"
            f"{page_title_suffix}"
        )
        ax.set_xlabel("Physical channel")
        ax.set_ylabel("Effective aperture weight (unity-normalized)")
        ax.set_xlim(-0.5, data.channels - 0.5)
        ax.set_ylim(*shared_ylim)
        shared_xticks = ax.get_xticks()
        shared_yticks = ax.get_yticks()
        ax.xaxis.set_major_locator(FixedLocator(shared_xticks))
        ax.yaxis.set_major_locator(FixedLocator(shared_yticks))
        ax.grid(alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.78, 0.5),
            ncol=2 if len(handles) > 4 else 1,
            fontsize=7,
        )
        fig.tight_layout(rect=(0.0, 0.0, 0.76, 1.0))
        fig.savefig(scene_dir / f"weight_controls{page_suffix}.png", dpi=220)
        plt.close(fig)
    for label, curve in effective_curves.items():
        fig, ax = plt.subplots(figsize=(9.0, 5.2))
        controls = curve["control_real"]
        wr = curve["effective_real"]
        ax.plot(np.arange(wr.size), wr, color="#0072B2", label=f"{label} (controls={controls.size})", linewidth=1.4)
        anchor_idx = anchor_indices[label]
        ax.plot(
            anchor_idx,
            wr[anchor_idx],
            marker="o",
            linestyle="none",
            markersize=6,
            color="#D55E00",
            label="control anchors",
        )
        ax.set_title(f"{scene_dir.name}: {label} | z={z_index}, x={x_index}", fontsize=12)
        ax.set_xlabel("Physical channel")
        ax.set_ylabel("Effective aperture weight (unity-normalized)")
        ax.set_xlim(-0.5, data.channels - 0.5)
        ax.set_ylim(*shared_ylim)
        ax.xaxis.set_major_locator(FixedLocator(shared_xticks))
        ax.yaxis.set_major_locator(FixedLocator(shared_yticks))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9, loc="upper right")
        fig.tight_layout()
        fig.savefig(individual_dir / f"weight_controls_{label}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
def read_split_indices(split_file: Path, test_h5: Path, split_name: str) -> list[int]:
    if split_name not in {"train", "val", "test"}:
        raise ValueError(f"未知 split: {split_name!r}")
    target = str(test_h5.resolve())
    with split_file.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"split", "original_index", "source_h5"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"固定 split 缺少列 {sorted(missing)}: {split_file}")
        rows = list(reader)
    candidates = [row for row in rows if row.get("split") == split_name]

    def source_name(row: dict[str, str]) -> str:
        return str(row.get("source_h5") or "").strip()

    exact_rows = [
        row
        for row in candidates
        if source_name(row) and str(Path(source_name(row)).resolve()) == target
    ]
    if not exact_rows:
        target_name = target.replace("\\", "/").rsplit("/", 1)[-1]
        matching_sources = {
            source_name(row)
            for row in candidates
            if source_name(row).replace("\\", "/").rsplit("/", 1)[-1] == target_name
        }
        if len(matching_sources) == 1:
            exact_rows = [
                row
                for row in candidates
                if source_name(row).replace("\\", "/").rsplit("/", 1)[-1] == target_name
            ]
    try:
        indices = [int(row["original_index"]) for row in exact_rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"固定 split 的 original_index 无效: {split_file}") from exc
    if not indices:
        raise RuntimeError(f"split中没有找到 {target} 的 {split_name} 样本")
    if len(set(indices)) != len(indices):
        raise ValueError(f"split 的 {split_name} 样本包含重复帧: {split_file}")
    with h5py.File(test_h5.resolve(), "r") as handle:
        total = int(handle["all_multi_I"].shape[0])
    if any(frame < 0 or frame >= total for frame in indices):
        raise IndexError(f"split 中的帧号超出 {test_h5}: [0, {total - 1}]")
    return sorted(indices)


def read_test_indices(split_file: Path, test_h5: Path) -> list[int]:
    return read_split_indices(split_file, test_h5, "test")


def select_test_indices(indices: list[int], count: int) -> list[int]:
    if count <= 0:
        raise ValueError("test-frames 必须大于 0")
    if count > len(indices):
        raise ValueError(f"test-frames={count} 超过 split 中的 test 样本数 {len(indices)}")
    if count == len(indices):
        return list(indices)
    positions = np.rint(np.linspace(0, len(indices) - 1, count)).astype(int)
    return [indices[position] for position in positions]

@dataclass
class SceneData:
    h5_path: Path
    sample_idx: int
    rf_i: torch.Tensor
    rf_q: torch.Tensor
    t_start: torch.Tensor
    x_grid: torch.Tensor
    z_grid: torch.Tensor
    angles: torch.Tensor
    fs: float
    fc: float
    c: float
    pitch: float
    gt_db: np.ndarray | None
    ground_truth_f_number: float | None = None

    @property
    def height(self) -> int:
        return self.z_grid.numel()

    @property
    def width(self) -> int:
        return self.x_grid.numel()

    @property
    def pixels(self) -> int:
        return self.height * self.width

    @property
    def channels(self) -> int:
        return self.rf_i.shape[-1]

def gt_to_db(value: np.ndarray, dynamic_range: float = 60.0) -> np.ndarray:
    value = np.squeeze(value).astype(np.float32)
    if np.nanmin(value) < -1.0:
        return np.clip(value - np.nanmax(value), -dynamic_range, 0.0)
    return run_one.gt_norm_to_db(value, dynamic_range)

def read_ground_truth_f_number(handle: h5py.File) -> float | None:
    if "config_yaml" not in handle:
        return None
    raw = handle["config_yaml"][()]
    if isinstance(raw, (bytes, np.bytes_)):
        text = raw.decode("utf-8")
    elif np.isscalar(raw):
        text = str(raw)
    else:
        return None
    try:
        config = yaml.safe_load(text) or {}
    except yaml.YAMLError as error:
        raise ValueError("H5 config_yaml 不是有效 YAML") from error
    if not isinstance(config, dict):
        return None
    generation = config.get("generation", {})
    if not isinstance(generation, dict):
        return None
    ground_truth = generation.get("ground_truth", {})
    if not isinstance(ground_truth, dict) or ground_truth.get("f_number") is None:
        return None
    value = float(ground_truth["f_number"])
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("H5 generation.ground_truth.f_number 必须是有限正数")
    return value

def load_scene(
    h5_path: Path,
    sample_idx: int,
    gt_key: str | None = "auto",
    angle_indices: np.ndarray | None = None,
) -> SceneData:
    with h5py.File(h5_path, "r") as handle:
        iq_shape = handle["all_multi_I"].shape
        if handle["all_multi_Q"].shape != iq_shape or len(iq_shape) != 4:
            raise ValueError("all_multi_I/all_multi_Q 必须是形状一致的 [N,A,T,C]")
        n_samples = int(handle["all_multi_I"].shape[0])
        if not 0 <= sample_idx < n_samples:
            raise IndexError(f"sample_idx={sample_idx} 超出 [0,{n_samples - 1}]")
        angles = handle["angles"][:].astype(np.float32)
        if angles.ndim != 1 or angles.size != iq_shape[1] or not np.isfinite(angles).all():
            raise ValueError("angles 必须是非空有限一维数组")
        configured_selection = getattr(getattr(training, "args", None), "angle_selection", "center")
        if angle_indices is None:
            selected_angles = resolve_angle_indices(angles, configured_selection)
        else:
            selected_angles = np.asarray(angle_indices, dtype=np.int64)
        if selected_angles.ndim != 1 or selected_angles.size == 0:
            raise ValueError("angle_indices 必须是非空一维索引")
        if np.any(selected_angles < 0) or np.any(selected_angles >= angles.size):
            raise IndexError("angle_indices 超出角度范围")
        if np.unique(selected_angles).size != selected_angles.size:
            raise ValueError("angle_indices 不能包含重复索引")
        n_time = int(handle["all_multi_I"].shape[2])
        if handle["time_start_vector"].shape != iq_shape[:2]:
            raise ValueError("time_start_vector 必须匹配 IQ 的 [N,A]")
        valid_raw = handle["valid_time_samples"][sample_idx] if "valid_time_samples" in handle else n_time
        valid_time = int(valid_raw)
        if float(valid_raw) != valid_time:
            raise ValueError(f"valid_time_samples[{sample_idx}] 必须是整数")
        if not 2 <= valid_time <= n_time:
            raise ValueError(f"valid_time_samples[{sample_idx}]={valid_time} 超出 [2,{n_time}]")
        sample_i = handle["all_multi_I"][sample_idx, :, :valid_time].astype(np.float32)
        sample_q = handle["all_multi_Q"][sample_idx, :, :valid_time].astype(np.float32)
        sample_t_start = handle["time_start_vector"][sample_idx].astype(np.float32)
        rf_i = torch.from_numpy(sample_i[selected_angles][None])
        rf_q = torch.from_numpy(sample_q[selected_angles][None])
        t_start = torch.from_numpy(sample_t_start[selected_angles][None])
        if not torch.isfinite(rf_i).all() or not torch.isfinite(rf_q).all() or not torch.isfinite(t_start).all():
            raise ValueError("所选场景的 IQ/time_start_vector 包含 NaN/Inf")
        x_grid = torch.from_numpy(handle["x_grid"][:].astype(np.float32))
        z_grid = torch.from_numpy(handle["z_grid"][:].astype(np.float32))
        for name, grid in (("x_grid", x_grid), ("z_grid", z_grid)):
            grid_np = grid.numpy()
            if grid.ndim != 1 or grid.numel() < 2 or not np.isfinite(grid_np).all() or not np.all(np.diff(grid_np) > 0):
                raise ValueError(f"{name} 必须是至少含两个点的有限严格递增一维网格")
        gt_db = None
        if gt_key is not None:
            selected_gt = "all_envdb_norm" if gt_key == "auto" else gt_key
            if selected_gt not in handle:
                raise KeyError(f"{h5_path} 不包含 GT 字段 {selected_gt}")
            gt_db = gt_to_db(handle[selected_gt][sample_idx])
        fs, fc, c, pitch = (float(handle[key][()]) for key in ("fs", "fc", "c", "pitch"))
        ground_truth_f_number = read_ground_truth_f_number(handle)
        if gt_db is not None and (not np.isfinite(gt_db).all() or gt_db.shape != (z_grid.numel(), x_grid.numel())):
            raise ValueError(f"GT 形状或数值非法: {gt_db.shape}")
        if not all(np.isfinite(value) and value > 0 for value in (fs, fc, c, pitch)):
            raise ValueError("fs/fc/c/pitch 必须是有限正数")
    return SceneData(
        h5_path,
        sample_idx,
        rf_i,
        rf_q,
        t_start,
        x_grid,
        z_grid,
        torch.from_numpy(angles[selected_angles]),
        fs,
        fc,
        c,
        pitch,
        gt_db,
        ground_truth_f_number,
    )

def configure_geometry(data: SceneData, module: object = training) -> None:
    implementation = getattr(module, "_runtime", None)
    for target in (module, implementation):
        if target is None:
            continue
        target.IMG_H = data.height
        target.IMG_W = data.width
        target.NUM_PIXELS = data.pixels
        target.NUM_CHANNELS = data.channels
        target.c_global = data.c
        target.fc_global = data.fc
        target.fs_global = data.fs
    runtime_state = getattr(module, "runtime", None)
    if runtime_state is not None:
        runtime_state.IMG_H = data.height
        runtime_state.IMG_W = data.width
        runtime_state.NUM_PIXELS = data.pixels
        runtime_state.NUM_CHANNELS = data.channels
        runtime_state.c_global = data.c
        runtime_state.fc_global = data.fc
        runtime_state.fs_global = data.fs

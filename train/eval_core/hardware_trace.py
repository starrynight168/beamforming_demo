"""硬件内部信号追踪。"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import inspect
import json
import math
import re
import sys
import types
from pathlib import Path

import numpy as np
import torch

TRAIN_DIR = Path(__file__).resolve().parent.parent
ROOT = TRAIN_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

import mban as training
from eval_core.config import (
    DEFAULT_CONFIG,
    DEFAULT_HARDWARE_CONFIG,
    add_model_arguments,
    calibrate_models,
    load_specs,
    resolve_model_arguments,
)
from eval_core.inference import (
    WeightPathCollector,
    activate_model_runtime,
    load_scene,
    metric_pair,
    prepare_noise_realization,
    read_test_indices,
    reconstruct,
    select_test_indices,
)
from mban_core.config import configured_h5_files
from mban_core.hardware_backend import MemristorLinear
from mban_core.naming import bias_dac_name, bias_range_name

class Stats:
    def __init__(self, limit: int = 100000) -> None:
        if limit <= 0:
            raise ValueError("stats reservoir limit must be positive")
        self.limit = limit
        self.n = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self._reservoir = torch.empty(0, dtype=torch.float32)
        self._keys = torch.empty(0, dtype=torch.float32)
        self._rng = np.random.default_rng(1701)

    def add(self, value: torch.Tensor | None) -> None:
        if value is None or value.numel() == 0:
            return
        flat = value.detach().float().reshape(-1)
        summary = torch.stack((flat.sum(), flat.square().sum(), flat.min(), flat.max())).cpu().tolist()
        self.n += int(flat.numel())
        self.sum += float(summary[0])
        self.sum_sq += float(summary[1])
        self.minimum = min(self.minimum, float(summary[2]))
        self.maximum = max(self.maximum, float(summary[3]))
        if self.limit <= 0:
            return
        if flat.device.type == "cuda":
            keys = torch.rand(flat.numel(), device=flat.device, dtype=torch.float32)
        else:
            keys = torch.from_numpy(self._rng.random(flat.numel()).astype(np.float32))
        reservoir = self._reservoir.to(device=flat.device)
        reservoir_keys = self._keys.to(device=flat.device)
        pool_keys = torch.cat((reservoir_keys, keys))
        pool_size = pool_keys.numel()
        keep = min(self.limit, pool_size)
        selected_keys, selected = torch.topk(pool_keys, keep, sorted=False)
        old_count = reservoir_keys.numel()
        old_mask = selected < old_count
        old_values = reservoir[selected[old_mask]] if old_mask.any() else flat[:0]
        new_indices = selected[~old_mask] - old_count
        new_values = flat[new_indices] if new_indices.numel() else flat[:0]
        self._reservoir = torch.cat((old_values, new_values)).detach()
        self._keys = selected_keys.detach()

    def result(self) -> dict[str, float | int]:
        if self.n == 0:
            return {"samples": 0}
        values = self._reservoir.cpu() if self._reservoir.numel() else torch.zeros(1)
        mean = self.sum / self.n
        variance = max(self.sum_sq / self.n - mean * mean, 0.0)
        absolute = values.abs()
        quantiles = torch.quantile(absolute, torch.tensor([0.5, 0.95, 0.99, 0.999]))
        return {
            "samples": self.n,
            "min": self.minimum,
            "max": self.maximum,
            "mean": mean,
            "std": math.sqrt(variance),
            "rms": math.sqrt(self.sum_sq / self.n),
            "abs_p50": float(quantiles[0]),
            "abs_p95": float(quantiles[1]),
            "abs_p99": float(quantiles[2]),
            "abs_p999": float(quantiles[3]),
        }


class FeatureVariation:
    def __init__(self) -> None:
        self.runs: dict[str, dict[str, object]] = {}

    def add(self, run: str, value: torch.Tensor | None) -> None:
        if value is None or value.numel() == 0:
            return
        matrix = value.detach().float().reshape(-1, value.shape[-1])
        current = self.runs.get(run)
        count = matrix.shape[0]
        sums = matrix.sum(dim=0).cpu()
        sums_sq = matrix.square().sum(dim=0).cpu()
        if current is None:
            current = {
                "n": 0,
                "sum": torch.zeros_like(sums),
                "sum_sq": torch.zeros_like(sums_sq),
            }
            self.runs[run] = current
        current["n"] += count
        current["sum"] += sums
        current["sum_sq"] += sums_sq

    @staticmethod
    def _summary(values: torch.Tensor) -> dict[str, float]:
        return {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "p95": float(torch.quantile(values, 0.95)),
            "max": float(values.max()),
        }

    def result(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for run, current in self.runs.items():
            n = int(current["n"])
            mean = current["sum"] / max(n, 1)
            variance = (current["sum_sq"] / max(n, 1) - mean.square()).clamp_min(0.0)
            std = variance.sqrt()
            result[run] = {
                "n": n,
                "mean_by_neuron": mean.tolist(),
                "std_by_neuron": std.tolist(),
                "mean_summary": self._summary(mean),
                "std_summary": self._summary(std),
                "nonvarying_fraction": float((std <= 1.0e-12).float().mean()),
            }
        return result


class ADCStats:
    def __init__(self) -> None:
        self.calls = 0
        self.total = 0
        self.lower_outside = 0
        self.upper_outside = 0
        self.outside = 0
        self.zero_code = 0
        self.endpoint_code = 0
        self.input = Stats()
        self.output = Stats()
        self.bound = Stats()
        self.code_hist: Counter[int] = Counter()
        self.range_sources: Counter[str] = Counter()
        self.bits: int | None = None
        self.unsigned: bool | None = None

    def add(
        self,
        value: torch.Tensor,
        output: torch.Tensor,
        value_max: torch.Tensor | float,
        bits: int,
        unsigned: bool,
        range_source: str,
    ) -> None:
        if value.numel() == 0:
            return
        bound = torch.as_tensor(value_max, device=value.device, dtype=value.dtype).clamp_min(1.0e-12)
        lower = torch.zeros_like(bound) if unsigned else -bound
        upper = bound
        lower_outside = value < lower
        upper_outside = value > upper
        self.calls += 1
        self.total += value.numel()
        self.lower_outside += int(lower_outside.sum().item())
        self.upper_outside += int(upper_outside.sum().item())
        self.outside += int((lower_outside | upper_outside).sum().item())
        self.input.add(value)
        self.output.add(output)
        self.bound.add(bound)
        self.range_sources[range_source] += 1
        self.bits = bits
        self.unsigned = unsigned
        if bits >= 32:
            return
        qmax = (1 << bits) - 1 if unsigned else (1 << (bits - 1)) - 1
        scale = bound / qmax
        codes = torch.round(output / scale).clamp(0 if unsigned else -qmax, qmax).long()
        self.zero_code += int((codes == 0).sum().item())
        self.endpoint_code += int(
            (codes == qmax).sum().item() if unsigned else (codes.abs() == qmax).sum().item()
        )
        hist_size = qmax + 1 if unsigned else 2 * qmax + 1
        counts = torch.bincount(
            codes.reshape(-1) if unsigned else (codes.reshape(-1) + qmax),
            minlength=hist_size,
        ).cpu().tolist()
        offset = 0 if unsigned else qmax
        for index, count in enumerate(counts):
            if count:
                self.code_hist[index - offset] += int(count)

    def result(self) -> dict[str, object]:
        total = max(self.total, 1)
        result: dict[str, object] = {
            "calls": self.calls,
            "samples": self.total,
            "bits": self.bits,
            "unsigned": self.unsigned,
            "lower_bound": 0.0 if self.unsigned else None,
            "upper_bound": self.bound.result(),
            "input": self.input.result(),
            "output": self.output.result(),
            "bound": self.bound.result(),
            "lower_outside_fraction": self.lower_outside / total,
            "upper_outside_fraction": self.upper_outside / total,
            "out_of_range_fraction": self.outside / total,
            "zero_code_fraction": self.zero_code / total,
            "endpoint_code_fraction": self.endpoint_code / total,
            "code_hist": {str(code): count for code, count in sorted(self.code_hist.items())},
            "range_sources": dict(sorted(self.range_sources.items())),
        }
        if self.unsigned is False:
            bound = self.bound.result()
            result["lower_bound"] = {
                "min": -bound.get("max", 0.0),
                "max": -bound.get("min", 0.0),
            }
        return result


class TraceCollector:
    def __init__(self, hardware, array_bias_layers=()) -> None:
        self.hardware = hardware
        self.array_bias_layers = frozenset(array_bias_layers)
        self.enabled = True
        self.current_run = "unassigned"
        self.signals: dict[str, Stats] = {}
        self.weights_by_run: dict[str, dict[str, dict[str, object]]] = {}
        self.biases_by_run: dict[str, dict[str, dict[str, object]]] = {}
        self.adc_by_run: dict[str, dict[str, ADCStats]] = {}
        self.feature_variation: dict[str, FeatureVariation] = {}
        self.calls: dict[str, int] = {}
        self.run_meta: list[dict[str, object]] = []
        self._seen_weights: set[str] = set()
        self._seen_biases: set[str] = set()

    def begin_run(self, run: int | str, seed: int, realization_index: int) -> None:
        self.current_run = str(run)
        self._seen_weights.clear()
        self._seen_biases.clear()
        self.run_meta.append(
            {"run": self.current_run, "seed": seed, "realization_index": realization_index}
        )

    def reset_measurements(self) -> None:
        self.signals.clear()
        self.weights_by_run.clear()
        self.biases_by_run.clear()
        self.adc_by_run.clear()
        self.feature_variation.clear()
        self.calls.clear()
        self.run_meta.clear()
        self._seen_weights.clear()
        self._seen_biases.clear()

    def _recording(self) -> bool:
        return self.enabled

    def signal(self, name: str, value: torch.Tensor | None) -> None:
        if not self._recording():
            return
        self.signals.setdefault(name, Stats()).add(value)
        if re.fullmatch(r"fc\d+\.(?:linear_input|linear_output|tia\.output)", name):
            self.feature_variation.setdefault(name, FeatureVariation()).add(self.current_run, value)

    def record_weight(self, name: str, raw: torch.Tensor, state) -> None:
        if not self._recording() or name in self._seen_weights:
            return
        self._seen_weights.add(name)
        cfg = self.hardware.config
        step = (cfg.g_max - cfg.g_min) / ((1 << cfg.weight_bits) - 1) if cfg.weight_bits < 32 else 0.0
        result: dict[str, object] = {
            "calls": 1,
            "bias_path": "array_constant_input" if name in self.array_bias_layers else "external_or_none",
            "raw": self._stats(raw),
            "effective": self._stats(state.effective_weight),
            "weight_scale": self._stats(state.weight_scale),
            "g_plus": self._stats(state.g_plus),
            "g_minus": self._stats(state.g_minus),
        }
        if step > 0.0:
            plus_code = torch.round((state.g_plus.detach() - cfg.g_min) / step).clamp(0, (1 << cfg.weight_bits) - 1)
            minus_code = torch.round((state.g_minus.detach() - cfg.g_min) / step).clamp(0, (1 << cfg.weight_bits) - 1)
            states = 1 << cfg.weight_bits
            result["nearest_plus_code_hist"] = torch.bincount(
                plus_code.long().reshape(-1), minlength=states
            ).tolist()
            result["nearest_minus_code_hist"] = torch.bincount(
                minus_code.long().reshape(-1), minlength=states
            ).tolist()
            result["code_hist_basis"] = "nearest code derived from returned conductance"
            result["effective_zero_fraction"] = float(
                (state.effective_weight.detach().abs() <= max(cfg.epsilon, 1.0e-12)).float().mean()
            )
        if name in self.array_bias_layers:
            result["array_bias_column"] = {
                "raw": self._stats(raw[:, -1]),
                "effective": self._stats(state.effective_weight[:, -1]),
                "g_plus": self._stats(state.g_plus[:, -1]),
                "g_minus": self._stats(state.g_minus[:, -1]),
                "constant_input": 1.0,
                "input_dac": bias_dac_name(name),
            }
        self.weights_by_run.setdefault(self.current_run, {})[name] = result

    def record_bias(
        self,
        name: str,
        raw: torch.Tensor | None,
        effective: torch.Tensor | None,
        quantization_range: str,
    ) -> None:
        bias_key = bias_range_name(name)
        if not self._recording() or bias_key in self._seen_biases or raw is None or effective is None:
            return
        self._seen_biases.add(bias_key)
        cfg = self.hardware.config
        observed = self.hardware.activation_ranges.get(quantization_range)
        self.biases_by_run.setdefault(self.current_run, {})[bias_key] = {
            "raw": self._stats(raw),
            "effective": self._stats(effective),
            "observer_range": None if observed is None else float(observed),
            "raw_outside_observer_fraction": None
            if observed is None
            else float((raw.detach().abs() > float(observed)).float().mean()),
            "effective_endpoint_fraction": None
            if observed is None
            else float((effective.detach().abs() >= float(observed) - 1.0e-8).float().mean()),
            "bias_bits": int(cfg.bias_bits),
        }

    def record_adc(
        self,
        name: str,
        value: torch.Tensor,
        output: torch.Tensor,
        value_max: torch.Tensor | float,
        bits: int,
        unsigned: bool,
        range_source: str,
    ) -> None:
        if not self._recording():
            return
        by_name = self.adc_by_run.setdefault(self.current_run, {})
        stats = by_name.setdefault(name, ADCStats())
        stats.add(value, output, value_max, bits, unsigned, range_source)

    @staticmethod
    def _stats(value: torch.Tensor) -> dict[str, float | int]:
        stats = Stats()
        stats.add(value)
        return stats.result()

    def result(self) -> dict[str, object]:
        first_weights = next(iter(self.weights_by_run.values()), {})
        first_biases = next(iter(self.biases_by_run.values()), {})
        first_adc = next(iter(self.adc_by_run.values()), {})
        return {
            "signals": {name: value.result() for name, value in sorted(self.signals.items())},
            "weights": first_weights,
            "weights_by_run": self.weights_by_run,
            "biases": first_biases,
            "biases_by_run": self.biases_by_run,
            "adc_by_run": {
                run: {name: stats.result() for name, stats in sorted(values.items())}
                for run, values in self.adc_by_run.items()
            },
            "adc_value_max": {name: stats.bound.result() for name, stats in sorted(first_adc.items())},
            "feature_variation_by_signal": {
                name: variation.result() for name, variation in sorted(self.feature_variation.items())
            },
            "feature_variation_noise_enabled": bool(self.hardware.config.noise_enabled),
            "feature_variation_note": "按当前 realization 聚合；包含该 realization 配置启用的硬件噪声",
            "method_calls": self.calls,
            "runs": self.run_meta,
        }


class HardwareTrace:
    def __init__(self, model, collector: TraceCollector) -> None:
        self.model = model
        self.hardware = model.hardware_qat
        self.collector = collector
        self.originals: list[tuple[object, str, object]] = []
        self.hooks = []

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.remove()

    def install(self) -> None:
        hardware = self.hardware
        collector = self.collector

        def wrap(name: str):
            original = getattr(hardware, name)
            signature = inspect.signature(original)
            self.originals.append((hardware, name, original))

            def method(bound_self, *args, **kwargs):
                if name == "map_weight":
                    weight, layer_name = args[0], args[1]
                    state = original(*args, **kwargs)
                    collector.record_weight(layer_name, weight, state)
                    if collector._recording():
                        collector.calls[name] = collector.calls.get(name, 0) + 1
                    return state
                if name == "bias":
                    bound = signature.bind(*args, **kwargs)
                    bound.apply_defaults()
                    raw = bound.arguments["bias"]
                    layer_name = bound.arguments["name"]
                    quantization_range = bound.arguments["quantization_range"] or bias_range_name(layer_name)
                    effective = original(*args, **kwargs)
                    collector.record_bias(layer_name, raw, effective, quantization_range)
                    if collector._recording():
                        collector.calls[name] = collector.calls.get(name, 0) + 1
                    return effective
                if name in {"dac", "digital_dac", "tia"}:
                    value = args[0]
                    output = original(*args, **kwargs)
                    signal_name = args[1] if len(args) > 1 else kwargs.get("name", name)
                    collector.signal(f"{signal_name}.input", value)
                    collector.signal(f"{signal_name}.output", output)
                    if collector._recording():
                        collector.calls[name] = collector.calls.get(name, 0) + 1
                    return output
                if name == "output_activation_mismatch":
                    bound = signature.bind(*args, **kwargs)
                    bound.apply_defaults()
                    value = bound.arguments["x"]
                    layer_name = bound.arguments["layer"]
                    output = original(*args, **kwargs)
                    collector.signal(f"{layer_name}.output_relu.input", value)
                    collector.signal(f"{layer_name}.output_relu.output", output)
                    if collector._recording():
                        collector.calls[name] = collector.calls.get(name, 0) + 1
                    return output
                if name in {"adc_unsigned", "adc_signed"}:
                    bound = signature.bind(*args, **kwargs)
                    bound.apply_defaults()
                    value = bound.arguments["x"]
                    signal_name = bound.arguments["name"]
                    bits = int(bound.arguments["bits"])
                    unsigned = name == "adc_unsigned"
                    value_max = bound.arguments.get("value_max")
                    output = original(*args, **kwargs)
                    range_source = "explicit_argument"
                    if value_max is None:
                        value_max = hardware.activation_ranges.get(signal_name)
                        range_source = "hardware.activation_ranges"
                    if value_max is None and bits >= 32 and not hardware.observer_enabled:
                        value_max = value.detach().amax() if unsigned else value.detach().abs().amax()
                        range_source = "backend_current_input_observed"
                    if value_max is None:
                        raise RuntimeError(
                            f"{name}({signal_name}) 执行后没有可确认的 ADC 量程；"
                            "拒绝使用当前批次输入猜测量程"
                        )
                    collector.record_adc(signal_name, value, output, value_max, bits, unsigned, range_source)
                    if collector._recording():
                        collector.calls[name] = collector.calls.get(name, 0) + 1
                    return output
                return original(*args, **kwargs)

            setattr(hardware, name, types.MethodType(method, hardware))

        for name in (
            "map_weight",
            "bias",
            "dac",
            "digital_dac",
            "tia",
            "output_activation_mismatch",
            "adc_unsigned",
            "adc_signed",
        ):
            wrap(name)

        for module in self.model.modules():
            if not isinstance(module, MemristorLinear):
                continue
            layer_name = module.hardware_name

            def pre_hook(current, inputs, layer_name=layer_name):
                if inputs:
                    collector.signal(f"{layer_name}.linear_input", inputs[0])

            def forward_hook(current, inputs, output, layer_name=layer_name):
                collector.signal(f"{layer_name}.linear_output", output)

            self.hooks.append(module.register_forward_pre_hook(pre_hook))
            self.hooks.append(module.register_forward_hook(forward_hook))

    def remove(self) -> None:
        for hook in self.hooks:
            hook.remove()
        for target, name, original in reversed(self.originals):
            setattr(target, name, original)
        self.hooks.clear()
        self.originals.clear()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="正式 reconstruct 路径硬件内部信号追踪")
    add_model_arguments(parser, ("qat", "ptq"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--frames")
    parser.add_argument("--test-frames", type=int, default=10)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--eval-config", type=Path, default=DEFAULT_HARDWARE_CONFIG)
    parser.add_argument("--profile", default="common")
    parser.add_argument("--micro-batch", type=int, default=8192)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verification-atol", type=float, default=1.0e-6)
    parser.add_argument("--verification-ssim-atol", type=float, default=1.0e-8)
    args = parser.parse_args(argv)
    args.test_h5 = configured_h5_files(args.config)[0]
    return resolve_model_arguments(args)


def frame_list(args: argparse.Namespace) -> list[int]:
    if args.frames:
        return [int(item) for item in args.frames.split(",") if item.strip()]
    if args.split_file is None:
        raise ValueError("未提供 --frames 或 --split-file")
    indices = read_test_indices(args.split_file.resolve(), args.test_h5.resolve())
    return select_test_indices(indices, args.test_frames)


def verify_trace(
    spec,
    data,
    device: torch.device,
    micro_batch: int,
    trace: HardwareTrace,
    collector: TraceCollector,
    seed: int,
    output_atol: float,
    metric_atol: float,
) -> dict[str, object]:
    hardware = spec.model.hardware_qat
    activate_model_runtime(spec)
    state = prepare_noise_realization(hardware, seed)
    trace.remove()
    collector.enabled = False
    baseline, _, _, _ = reconstruct(spec, data, device, micro_batch, collect_weight_stats=False)
    hardware.restore_realization_state(state)
    with trace:
        collector.enabled = True
        collector.begin_run("trace_verification", seed, int(state["realization_index"]))
        hardware.begin_diagnostics()
        traced, _, _, _ = reconstruct(spec, data, device, micro_batch, collect_weight_stats=False)
        hardware.end_diagnostics(float(getattr(training.args, "public_gain_max", 16.0)))
    baseline_db, baseline_env = metric_pair(baseline, data.gt_db)
    traced_db, traced_env = metric_pair(traced, data.gt_db)
    if not np.isfinite(baseline).all() or not np.isfinite(traced).all():
        raise RuntimeError(f"{spec.display_name} 追踪一致性失败：基准或追踪输出包含非有限值")
    diff = np.abs(traced.astype(np.float64) - baseline.astype(np.float64))
    max_abs_diff = float(diff.max())
    mean_abs_diff = float(diff.mean())
    db_diff = abs(traced_db - baseline_db)
    env_diff = abs(traced_env - baseline_env)
    if max_abs_diff > output_atol or db_diff > metric_atol or env_diff > metric_atol:
        raise RuntimeError(
            f"{spec.display_name} 追踪一致性失败：max_abs={max_abs_diff:.3e}, "
            f"SSIM_dB_diff={db_diff:.3e}, SSIM_env_diff={env_diff:.3e}"
        )
    trace.install()
    return {
        "frame": int(data.sample_idx),
        "seed": seed,
        "realization_index": int(state["realization_index"]),
        "passed": True,
        "max_abs_output_diff": max_abs_diff,
        "mean_abs_output_diff": mean_abs_diff,
        "ssim_db_baseline": baseline_db,
        "ssim_db_traced": traced_db,
        "ssim_db_abs_diff": db_diff,
        "ssim_env_baseline": baseline_env,
        "ssim_env_traced": traced_env,
        "ssim_env_abs_diff": env_diff,
    }


def run(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.micro_batch <= 0 or args.runs <= 0:
        raise ValueError("micro-batch 和 runs 必须大于 0")
    if args.verification_atol < 0.0 or args.verification_ssim_atol < 0.0:
        raise ValueError("一致性校验容差必须非负")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    frames = frame_list(args)
    device = torch.device(args.device)
    data = {frame: load_scene(args.test_h5.resolve(), frame) for frame in frames}
    requests = args.model
    specs = load_specs(args, device, allowed_model_types={"qat", "ptq"})
    traces: dict[str, HardwareTrace] = {}
    collectors: dict[str, TraceCollector] = {}
    path_collectors: dict[str, WeightPathCollector] = {}
    try:
        for spec in specs:
            array_bias_layers = (
                spec.model.bias_layers if getattr(spec.model, "array_bias", False) else ()
            )
            collector = TraceCollector(spec.model.hardware_qat, array_bias_layers)
            trace = HardwareTrace(spec.model, collector)
            trace.install()
            collectors[spec.display_name] = collector
            traces[spec.display_name] = trace
            path_collectors[spec.display_name] = WeightPathCollector()
    except Exception:
        for trace in traces.values():
            trace.remove()
        raise

    model_type_by_model = {spec.display_name: spec.model_type for spec in specs}
    if any(spec.model_type == "ptq" for spec in specs):
        for collector in collectors.values():
            collector.enabled = False
        calibrate_models(args, specs, device)
        for collector in collectors.values():
            collector.reset_measurements()
            collector.enabled = True

    metric_rows: list[dict[str, object]] = []
    hardware_rows: dict[str, object] = {}
    verification_rows: dict[str, dict[str, object]] = {}
    try:
        for spec in specs:
            hardware = spec.model.hardware_qat
            activate_model_runtime(spec)
            collector = collectors[spec.display_name]
            verification_rows[spec.display_name] = verify_trace(
                spec,
                data[frames[0]],
                device,
                args.micro_batch,
                traces[spec.display_name],
                collector,
                args.seed,
                args.verification_atol,
                args.verification_ssim_atol,
            )
            collector.reset_measurements()
            collector.enabled = True
            hardware.begin_diagnostics()
            for run in range(args.runs):
                seed = args.seed + run
                realization_state = prepare_noise_realization(hardware, seed)
                collector.begin_run(run, seed, int(realization_state["realization_index"]))
                for frame in frames:
                    image, _, _, _ = reconstruct(
                        spec,
                        data[frame],
                        device,
                        args.micro_batch,
                        collect_weight_stats=False,
                        path_collector=path_collectors[spec.display_name],
                    )
                    ssim_db, ssim_env = metric_pair(image, data[frame].gt_db)
                    metric_rows.append(
                        {
                            "model": spec.display_name,
                            "run": run,
                            "seed": seed,
                            "frame": frame,
                            "ssim_db": ssim_db,
                            "ssim_env": ssim_env,
                        }
                    )
                    print(
                        f"[{spec.display_name}] run={run + 1}/{args.runs} "
                        f"frame={frame} SSIM={ssim_db:.6f}",
                        flush=True,
                    )
            hardware_rows[spec.display_name] = hardware.end_diagnostics(
                float(getattr(training.args, "public_gain_max", 16.0))
            )
    finally:
        for trace in traces.values():
            trace.remove()

    trace_rows = {
        label: {
            "checkpoint": str(next(request.path for request in requests if request.display_name == label)),
            "model_type": model_type_by_model[label],
            "frames": frames,
            "runs": args.runs,
            "hardware_diagnostics": hardware_rows[label],
            "trace_verification": verification_rows[label],
            "execution_trace": collectors[label].result(),
            "weight_path_diagnostics": path_collectors[label].rows(),
        }
        for label in collectors
    }
    (args.output / "trace.json").write_text(
        json.dumps(trace_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "hardware_diagnostics.json").write_text(
        json.dumps(hardware_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    (args.output / "protocol.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "display_name": request.display_name,
                        "checkpoint": str(request.path),
                        "model_type": model_type_by_model[request.display_name],
                    }
                    for request in requests
                ],
                "test_h5": str(args.test_h5.resolve()),
                "frames": frames,
                "runs": args.runs,
                "seed": args.seed,
                "micro_batch": args.micro_batch,
                "device": str(device),
                "path": "load_model -> load_scene -> reconstruct",
                "trace_verification": verification_rows,
                "verification_policy": {
                    "baseline": "wrappers and hooks removed",
                    "traced": "wrappers and hooks installed",
                    "output_atol": args.verification_atol,
                    "ssim_atol": args.verification_ssim_atol,
                    "failure": "raise and stop",
                },
                "trace_points": [
                    "MemristorLinear pre/post forward",
                    "HardwareQAT.map_weight",
                    "HardwareQAT.bias",
                    "HardwareQAT.dac",
                    "HardwareQAT.digital_dac",
                    "HardwareQAT.tia",
                    "HardwareQAT.output_activation_mismatch for analog nonnegative output",
                    "HardwareQAT.adc_unsigned/adc_signed",
                ],
                "weight_path_outputs": [
                    "raw_controls before output conversion",
                    "digital preactivation after signed ADC and Bias when inter_layer=digital",
                    "controls after final output stage: analog output ReLU/ADC or digital signed ADC plus float Bias/ReLU",
                    "effective physical weights after hard normalization",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "frames": frames, "models": list(collectors)}, ensure_ascii=False))


if __name__ == "__main__":
    run()

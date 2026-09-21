from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from pathlib import Path

import torch

TRAIN_DIR = Path(__file__).resolve().parents[1]
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from mban_core.hardware import QATConfig
from mban_core.hardware_backend import HardwareQAT


def parse_models(values: list[str]) -> list[tuple[str, Path]]:
    result = []
    for value in values:
        label, separator, raw_path = value.partition("=")
        path = Path(raw_path).expanduser().resolve()
        if not separator or not label.strip() or not path.is_file():
            raise ValueError(f"--model 必须是存在的 NAME=CHECKPOINT: {value}")
        result.append((label.strip(), path))
    if not result:
        raise ValueError("至少需要一个 --model")
    return result


def load_checkpoint(path: Path) -> tuple[dict, dict, QATConfig, HardwareQAT]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path} 不是 checkpoint 映射")
    hardware_state = checkpoint.get("hardware_qat_state")
    if not isinstance(hardware_state, dict) or not isinstance(hardware_state.get("config"), dict):
        raise ValueError(f"{path} 缺少 hardware_qat_state.config")
    config = QATConfig(**hardware_state["config"])
    quiet = dataclasses.replace(config, noise_enabled=False)
    hardware = HardwareQAT(quiet)
    hardware.activation_ranges = {
        str(key): torch.as_tensor(value, dtype=torch.float32)
        for key, value in (hardware_state.get("activation_ranges") or {}).items()
    }
    hardware.disable_observer()
    state = checkpoint.get("model_state_dict")
    model_config = checkpoint.get("model_config")
    if not isinstance(state, dict) or not isinstance(model_config, dict):
        raise ValueError(f"{path} 缺少 model_state_dict/model_config")
    return checkpoint, state, config, hardware


def layer_names(state: dict) -> list[str]:
    return sorted(
        key.removesuffix(".weight")
        for key in state
        if key.startswith("fc") and key.endswith(".weight")
    )


def split_layer(
    state: dict,
    layer: str,
    implementation: str,
    bias_layers: set[str],
) -> tuple[torch.Tensor, torch.Tensor | None, str]:
    weight = state[f"{layer}.weight"].detach().float()
    if layer not in bias_layers:
        return weight, None, "none"
    if implementation == "array":
        return weight[:, :-1].contiguous(), weight[:, -1].contiguous(), "array"
    bias = state.get(f"{layer}.bias")
    return weight, None if bias is None else bias.detach().float(), implementation


def audit_layer(
    state: dict,
    layer: str,
    implementation: str,
    bias_layers: set[str],
    hardware: HardwareQAT,
) -> dict[str, object]:
    core, raw_bias, bias_path = split_layer(state, layer, implementation, bias_layers)
    if raw_bias is None:
        mapped = hardware.map_weight(core, f"{layer}.shared")
        core_only = mapped
        effective_bias = None
        bias_range = None
        stacked = core
    else:
        effective_bias = hardware.bias(raw_bias, layer, bias_path)
        stacked = torch.cat((core, effective_bias[:, None]), dim=1)
        mapped = hardware.map_weight(stacked, f"{layer}.shared")
        core_only = hardware.map_weight(core, f"{layer}.core_only")
        bias_range = hardware.activation_ranges.get(f"{layer}.bias")
    mapped_core = mapped.effective_weight[:, : core.shape[1]]
    core_scale = float(core_only.weight_scale.max())
    shared_scale = float(mapped.weight_scale.max())
    qmax = (1 << hardware.config.weight_bits) - 1 if hardware.config.weight_bits < 32 else 0
    core_error = mapped_core - core
    effective_bias_max = float(effective_bias.abs().max()) if effective_bias is not None else None
    endpoint = float(bias_range) if bias_range is not None else effective_bias_max
    return {
        "layer": layer,
        "bias_path": bias_path,
        "core_elements": int(core.numel()),
        "core_absmax": float(core.abs().max()),
        "raw_bias_absmax": None if raw_bias is None else float(raw_bias.abs().max()),
        "effective_bias_absmax": effective_bias_max,
        "bias_range": endpoint,
        "shared_weight_scale": shared_scale,
        "core_only_weight_scale": core_scale,
        "shared_weight_lsb": shared_scale / qmax if qmax else 0.0,
        "core_only_weight_lsb": core_scale / qmax if qmax else 0.0,
        "weight_lsb_inflation": shared_scale / max(core_scale, 1.0e-12),
        "core_mean_abs_error": float(core_error.abs().mean()),
        "core_max_abs_error": float(core_error.abs().max()),
        "core_zero_fraction": float((mapped_core == 0).float().mean()),
        "bias_endpoint_positive": (
            None
            if effective_bias is None or endpoint is None
            else float((effective_bias >= endpoint - 1.0e-6 * max(endpoint, 1.0e-12)).float().mean())
        ),
        "bias_endpoint_negative": (
            None
            if effective_bias is None or endpoint is None
            else float((effective_bias <= -endpoint + 1.0e-6 * max(endpoint, 1.0e-12)).float().mean())
        ),
        "stacked_columns": int(stacked.shape[1]),
    }


def audit_model(label: str, path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    checkpoint, state, config, hardware = load_checkpoint(path)
    model_config = checkpoint["model_config"]
    implementation = str(model_config.get("bias_implementation", config.bias_implementation))
    bias_layers = set(model_config.get("bias_layers") or [])
    rows = []
    for layer in layer_names(state):
        row = audit_layer(state, layer, implementation, bias_layers, hardware)
        row.update(
            {
                "model": label,
                "checkpoint": str(path),
                "model_semantics_version": checkpoint.get("model_semantics_version"),
                "weight_bits": config.weight_bits,
                "bias_bits": config.bias_bits,
                "per_output_channel": config.per_output_channel,
            }
        )
        rows.append(row)
    manifest = {
        "model": label,
        "checkpoint": str(path),
        "model_semantics_version": checkpoint.get("model_semantics_version"),
        "bias_implementation": implementation,
        "bias_layers": sorted(bias_layers),
        "hardware_config": dataclasses.asdict(config),
        "observer_enabled": hardware.observer_enabled,
        "activation_ranges": {key: float(value) for key, value in hardware.activation_ranges.items()},
    }
    return rows, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="审计 QAT checkpoint 的 Bias 路径与权重量程")
    parser.add_argument("--model", action="append", required=True, metavar="NAME=CHECKPOINT")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    manifests = []
    for label, path in parse_models(args.model):
        model_rows, manifest = audit_model(label, path)
        rows.extend(model_rows)
        manifests.append(manifest)
    json_path = args.output / "qat_parameter_audit.json"
    json_path.write_text(
        json.dumps({"models": manifests, "layers": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (args.output / "qat_parameter_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"[{row['model']}] {row['layer']} | bias={row['bias_path']} | "
            f"scale={row['shared_weight_scale']:.6g}/{row['core_only_weight_scale']:.6g} | "
            f"lsb={row['weight_lsb_inflation']:.3f}x | "
            f"core_err={row['core_mean_abs_error']:.6g}"
        )
    print(f"[OUTPUT] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

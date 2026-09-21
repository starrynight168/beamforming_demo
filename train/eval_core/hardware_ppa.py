"""硬件映射、CrossSim 验证与 NeuroSim/PPA 账本。"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

TRAIN_DIR = Path(__file__).resolve().parent.parent
ROOT = TRAIN_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TRAIN_DIR))

import mban as training
from eval_core.config import validate_deployment_checkpoint
from eval_core.inference import write_rows
from eval_core.hardware import (
    checkpoint_bias_implementation,
    checkpoint_bias_layers,
    checkpoint_hidden_width,
    checkpoint_hidden_layers,
    checkpoint_layer_bias_implementation,
    checkpoint_layer_index,
    INTER_LAYER_MODES,
    parse_sweep_models,
)


def load_checkpoint(path: Path) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path} 不是有效训练检查点")
    expected = int(training.MODEL_SEMANTICS_VERSION)
    actual = checkpoint.get("model_semantics_version")
    if actual != expected:
        raise ValueError(f"{path} 的 model_semantics_version={actual!r}，当前要求 {expected!r}；请使用当前代码重新训练")
    return checkpoint


def folded_model_state(checkpoint: dict[str, object], path: Path) -> dict[str, torch.Tensor]:
    validate_deployment_checkpoint(checkpoint, path)
    state = checkpoint["model_state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError(f"{path} 缺少 model_state_dict")
    return state


def _linear_weight_specs(state: dict[str, torch.Tensor]) -> list[tuple[int, str, int, int]]:
    specs = []
    for key, weight in state.items():
        if not key.endswith(".weight"):
            continue
        layer_index = checkpoint_layer_index(key)
        if layer_index is None:
            continue
        out_dim, in_dim = map(int, weight.shape)
        specs.append((layer_index, key, out_dim, in_dim))
    return sorted(specs)


def _build_mapping_layout(
    layer_specs: list[tuple[int, str, int, int]],
    mapping_mode: str,
    tile_rows: int,
    tile_cols: int,
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    if mapping_mode not in {"custom_compact", "fixed_tile"}:
        raise ValueError(f"不支持的 mapping_mode: {mapping_mode!r}")
    groups: list[list[tuple[int, str, int, int]]] = []
    if mapping_mode == "custom_compact":
        by_input: dict[int, list[tuple[int, str, int, int]]] = {}
        for spec in layer_specs:
            by_input.setdefault(spec[3], []).append(spec)
        groups = list(by_input.values())
    else:
        groups = [[spec] for spec in layer_specs]

    layout: dict[str, dict[str, object]] = {}
    packing_groups: list[dict[str, object]] = []
    macro_cursor = 0
    for group_index, group_layers in enumerate(groups, start=1):
        input_dim = group_layers[0][3]
        col_blocks = int(np.ceil(input_dim / tile_cols))
        physical_cols = col_blocks * tile_cols
        logical_weights = sum(out_dim * in_dim for _, _, out_dim, in_dim in group_layers)
        if mapping_mode == "custom_compact":
            packed_rows = sum(out_dim for _, _, out_dim, _ in group_layers)
            row_blocks = 1
            macro_count = col_blocks
        else:
            row_blocks = max(int(np.ceil(out_dim / tile_rows)) for _, _, out_dim, _ in group_layers)
            packed_rows = row_blocks * tile_rows
            macro_count = row_blocks * col_blocks
        capacity_weights = packed_rows * physical_cols
        group_id = f"pack_{group_index}"
        row_offset = 0
        for _, key, out_dim, in_dim in group_layers:
            layer_capacity = out_dim * physical_cols if mapping_mode == "custom_compact" else capacity_weights
            layer_row_blocks = 1 if mapping_mode == "custom_compact" else int(np.ceil(out_dim / tile_rows))
            layer_tile_count = col_blocks if mapping_mode == "custom_compact" else layer_row_blocks * col_blocks
            layout[key] = {
                "tile_row_count": layer_row_blocks,
                "tile_col_count": col_blocks,
                "tile_count": layer_tile_count,
                "tile_capacity_weights": layer_capacity,
                "tile_utilization": (out_dim * in_dim) / max(layer_capacity, 1),
                "physical_macro_count": col_blocks if mapping_mode == "custom_compact" else layer_tile_count,
                "subarray_equivalent_count": layer_capacity / (tile_rows * tile_cols),
                "packed_group_id": group_id,
                "packed_group_layers": ",".join(item[1].removesuffix(".weight") for item in group_layers),
                "packed_macro_rows": packed_rows,
                "packed_macro_cols": tile_cols,
                "packed_macro_width": physical_cols,
                "packed_row_start": row_offset,
                "packed_row_end": row_offset + out_dim,
                "packed_macro_index_start": macro_cursor,
                "packed_macro_index_end": macro_cursor + macro_count,
                "packed_group_capacity_weights": capacity_weights,
                "packed_group_logical_weights": logical_weights,
                "packed_group_utilization": logical_weights / max(capacity_weights, 1),
            }
            row_offset += out_dim
        packing_groups.append(
            {
                "group_id": group_id,
                "layers": [key.removesuffix(".weight") for _, key, _, _ in group_layers],
                "input_dim": input_dim,
                "packed_rows": packed_rows,
                "packed_cols": physical_cols,
                "macro_cols": tile_cols,
                "macro_count": macro_count,
                "logical_weights": logical_weights,
                "capacity_weights": capacity_weights,
                "utilization": logical_weights / max(capacity_weights, 1),
                "subarray_equivalent_count": capacity_weights / (tile_rows * tile_cols),
                "macro_index_start": macro_cursor,
                "macro_index_end": macro_cursor + macro_count,
            }
        )
        macro_cursor += macro_count
    return layout, packing_groups


def _mapping_row_fields(mapping_mode: str, tile_geometry: dict[str, object]) -> dict[str, object]:
    return {
        "tile_count": tile_geometry["tile_count"],
        "tile_capacity_weights": tile_geometry["tile_capacity_weights"],
        "tile_utilization": tile_geometry["tile_utilization"],
        "physical_macro_count": tile_geometry["physical_macro_count"],
        "subarray_equivalent_count": tile_geometry["subarray_equivalent_count"],
        "mapping_mode": mapping_mode,
        "packed_group_id": tile_geometry["packed_group_id"],
        "packed_group_layers": tile_geometry["packed_group_layers"],
        "packed_macro_rows": tile_geometry["packed_macro_rows"],
        "packed_macro_cols": tile_geometry["packed_macro_cols"],
        "packed_macro_width": tile_geometry["packed_macro_width"],
        "packed_row_start": tile_geometry["packed_row_start"],
        "packed_row_end": tile_geometry["packed_row_end"],
        "packed_macro_index_start": tile_geometry["packed_macro_index_start"],
        "packed_macro_index_end": tile_geometry["packed_macro_index_end"],
        "packed_group_capacity_weights": tile_geometry["packed_group_capacity_weights"],
        "packed_group_logical_weights": tile_geometry["packed_group_logical_weights"],
        "packed_group_utilization": tile_geometry["packed_group_utilization"],
    }


def _mapping_summary_fields(
    model_rows: list[dict[str, object]], packing_groups: list[dict[str, object]]
) -> dict[str, object]:
    logical_total = sum(int(row["logical_weights"]) for row in model_rows)
    capacity_total = sum(int(group["capacity_weights"]) for group in packing_groups)
    physical_macro_total = sum(int(group["macro_count"]) for group in packing_groups)
    return {
        "logical_weights_total": logical_total,
        "differential_devices_total": sum(int(row["differential_devices"]) for row in model_rows),
        "tile_count_total": physical_macro_total,
        "physical_macro_count_total": physical_macro_total,
        "operation_tile_count_total": sum(int(row["tile_count"]) for row in model_rows),
        "tile_capacity_total": capacity_total,
        "overall_tile_utilization": logical_total / max(capacity_total, 1),
        "subarray_equivalent_count_total": sum(
            float(group["subarray_equivalent_count"]) for group in packing_groups
        ),
        "packing_groups": packing_groups,
        "packing_policy": "same-input layers are vertically packed and time-multiplexed",
    }


def run_mapping(args: argparse.Namespace) -> None:
    if args.tile_rows <= 0 or args.tile_cols <= 0:
        raise ValueError("tile size must be positive")
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for label, path in parse_sweep_models(args.model):
        checkpoint = load_checkpoint(path)
        model_config = checkpoint.get("model_config", {})
        evaluation_config = checkpoint.get("evaluation_config", {})
        hardware_config = (checkpoint.get("hardware_qat_state") or {}).get("config") or {}
        state = folded_model_state(checkpoint, path)
        hidden_layers = checkpoint_hidden_layers(model_config, state)
        output_layer_index = hidden_layers + 1
        bias_layers = checkpoint_bias_layers(model_config, hidden_layers)
        inter_layer = str(args.inter_layer or hardware_config.get("inter_layer", "none"))
        if inter_layer not in INTER_LAYER_MODES:
            raise ValueError(f"{label} 的 inter_layer 必须是 none、analog 或 digital")
        bias_implementation = checkpoint_bias_implementation(model_config, hardware_config)
        inter_layer_bits = int(args.inter_layer_bits or hardware_config.get("inter_layer_bits", 4))
        output_controls = int(model_config.get("output_controls", 0))
        output_components = (
            2
            if str(evaluation_config.get("output_weights", model_config.get("output_weights", "real"))) == "complex"
            else 1
        )
        output_conversion_dimension = output_controls * output_components
        physical_channels = int(model_config.get("physical_channels", model_config.get("network_channels", 0)))
        dual_rail = str(model_config.get("branch_mode", "single")) == "dual"
        hidden_boundary_layers = [f"fc{index}" for index in range(1, hidden_layers + 1)]
        activation_layers = set(model_config.get("activation", {}))
        dual_boundary_layers = [layer for layer in hidden_boundary_layers if dual_rail and layer in activation_layers]
        layer_specs = _linear_weight_specs(state)
        mapping_layout, packing_groups = _build_mapping_layout(
            layer_specs, args.mapping_mode, args.tile_rows, args.tile_cols
        )
        for layer_index, key, out_dim, in_dim in layer_specs:
            tile_geometry = mapping_layout[key]
            logical_weights = out_dim * in_dim
            tile_row_count = int(tile_geometry["tile_row_count"])
            tile_col_count = int(tile_geometry["tile_col_count"])
            is_first_layer = layer_index == 1
            is_output_layer = layer_index == output_layer_index
            layer_name = key.removesuffix(".weight")
            is_hidden_boundary = layer_name in hidden_boundary_layers
            is_dual_boundary = layer_name in dual_boundary_layers
            is_digital_boundary = inter_layer == "digital" and is_hidden_boundary
            has_bias = layer_name in bias_layers
            layer_bias_implementation = checkpoint_layer_bias_implementation(
                bias_implementation, inter_layer
            )
            signal_input_dim = in_dim - int(is_first_layer and has_bias and layer_bias_implementation == "array")
            signal_col_blocks = int(np.ceil(signal_input_dim / args.tile_cols)) if is_first_layer else 0
            input_dac_count = 0
            input_dac_passes = 0
            input_dac_banks = 0
            if is_first_layer:
                input_dac_count = (
                    signal_input_dim
                    if args.converter_schedule == "parallel"
                    else min(args.tile_cols, signal_input_dim)
                )
                input_dac_passes = 1 if args.converter_schedule == "parallel" else signal_col_blocks
                input_dac_banks = signal_col_blocks if args.converter_schedule == "parallel" else 1
            output_adc_count = 0
            if is_output_layer:
                output_adc_count = out_dim if args.differential_readout == "post_tia_subtractor" else 2 * out_dim
            rows.append(
                {
                    "model": label,
                    "checkpoint": str(path),
                    "layer": key.removesuffix(".weight"),
                    "matrix_rows": out_dim,
                    "matrix_cols": in_dim,
                    "logical_weights": logical_weights,
                    "differential_devices": logical_weights * 2,
                    "tile_rows": args.tile_rows,
                    "tile_cols": args.tile_cols,
                    "tile_row_count": tile_row_count,
                    "tile_col_count": tile_col_count,
                    **_mapping_row_fields(args.mapping_mode, tile_geometry),
                    "bias_parameters": out_dim if has_bias else 0,
                    "bias_implementation": (
                        "array constant-input line; included in matrix columns"
                        if has_bias and layer_bias_implementation == "array"
                        else "per-output analog peripheral bias"
                        if has_bias and layer_bias_implementation == "analog"
                        else "post-ADC digital bias"
                        if has_bias and layer_bias_implementation == "digital"
                        else "none"
                    ),
                    "dual_rail": is_dual_boundary,
                    "input_conversion_dimension": signal_input_dim if is_first_layer else 0,
                    "physical_dac_count": input_dac_count,
                    "array_bias_dac_count": int(has_bias and layer_bias_implementation == "array"),
                    "input_dac_passes": input_dac_passes,
                    "input_dac_parallel_banks": input_dac_banks,
                    "output_conversion_dimension": out_dim if is_output_layer else 0,
                    "physical_adc_count": output_adc_count,
                    "output_adc_passes": 1 if is_output_layer else 0,
                    "partial_sum_blocks": tile_col_count,
                    "partial_sum_passes": 1 if args.converter_schedule == "parallel" else tile_col_count,
                    "differential_weight_form": "G+/G− pair; row-concat equivalent is 2R×C",
                    "differential_readout": args.differential_readout,
                    "post_tia_peripheral": (
                        f"signed {inter_layer_bits}-bit ADC + {'digital dual/CReLU' if is_dual_boundary else 'digital signed transfer'} + {inter_layer_bits}-bit {'magnitude DAC + rail selector' if is_dual_boundary else 'signed DAC'}"
                        if is_digital_boundary
                        else "positive/negative dual-rail ReLU sign split + dual mismatch + analog inter-layer transfer"
                        if is_dual_boundary and inter_layer == "analog"
                        else "positive/negative dual-rail ReLU sign split + ideal inter-layer transfer"
                        if is_dual_boundary and inter_layer == "none"
                        else "analog signed inter-layer transfer"
                        if is_hidden_boundary and inter_layer == "analog"
                        else "none: ideal hidden inter-layer transfer"
                        if is_hidden_boundary and inter_layer == "none"
                        else "signed final-control ADC -> digital Bias (floating-point accumulation; overflow not modeled) -> activation -> direct ADC-scale decode"
                        if is_output_layer and inter_layer == "digital" and has_bias and layer_bias_implementation == "digital"
                        else "signed final-control ADC -> activation -> direct ADC-scale decode"
                        if is_output_layer and inter_layer == "digital"
                        else "analog output ReLU with threshold/gain channel mismatch (no independent ReLU noise) -> unsigned final-control ADC"
                        if is_output_layer and evaluation_config.get("output_domain", "signed") == "nonnegative"
                        else "signed final-control ADC"
                        if is_output_layer
                        else "none"
                    ),
                }
            )
        model_rows = [row for row in rows if row["model"] == label]
        first_layer = next((row for row in model_rows if row["layer"] == "fc1"), None)
        first_input_dim = (
            int(first_layer["input_conversion_dimension"])
            if first_layer
            else int(model_config.get("network_channels", 0)) * 2
        )
        hidden_boundary_count = len(hidden_boundary_layers)
        all_hidden_dual = bool(hidden_boundary_layers) and len(dual_boundary_layers) == len(hidden_boundary_layers)
        hidden_width = checkpoint_hidden_width(model_config, state)
        summaries.append(
            {
                "model": label,
                "checkpoint": str(path),
                "mapping_mode": args.mapping_mode,
                "inter_layer": inter_layer,
                "bias_implementation": bias_implementation,
                "bias_layers": sorted(bias_layers, key=lambda layer: int(layer[2:])),
                "bias_parameters_total": sum(int(row["bias_parameters"]) for row in model_rows),
                "physical_channels": physical_channels,
                "network_channels": int(model_config.get("network_channels", 0)),
                "output_controls": output_controls,
                "input_conversion_dimension": first_input_dim,
                "physical_dac_count": first_input_dim
                if args.converter_schedule == "parallel"
                else min(args.tile_cols, first_input_dim),
                "array_bias_dac_count_total": sum(int(row["array_bias_dac_count"]) for row in model_rows),
                "input_dac_passes": 1
                if args.converter_schedule == "parallel"
                else int(np.ceil(first_input_dim / args.tile_cols)),
                "input_dac_parallel_banks": int(
                    np.ceil(first_input_dim / args.tile_cols)
                )
                if args.converter_schedule == "parallel"
                else 1,
                "final_output_conversion_dimension": output_conversion_dimension,
                "physical_adc_count": output_conversion_dimension
                if args.differential_readout == "post_tia_subtractor"
                else output_conversion_dimension * 2,
                "final_adc_readout": args.differential_readout,
                "core_mvm_block_passes": len(model_rows)
                if args.converter_schedule == "parallel"
                else sum(int(row["tile_count"]) for row in model_rows),
                **_mapping_summary_fields(model_rows, packing_groups),
                "dual_rail": bool(dual_boundary_layers),
                "dual_activation": f"positive/negative dual-rail ReLU sign split ({hidden_width} -> {2 * hidden_width} rails) at {dual_boundary_layers}"
                if dual_boundary_layers
                else "single branch",
                "dual_hidden_boundary_layers": dual_boundary_layers,
                "dual_hidden_boundary_count": len(dual_boundary_layers),
                "analog_buffer_after_activation": inter_layer == "analog",
                "converter_schedule": args.converter_schedule,
                "block_rows": args.tile_rows,
                "block_cols": args.tile_cols,
                "inter_layer_bits": inter_layer_bits if inter_layer == "digital" else 32,
                "hidden_boundary_layers": hidden_boundary_layers,
                "hidden_layer_boundary_count": hidden_boundary_count,
                "hidden_layer_adc_count_per_boundary": hidden_width if inter_layer == "digital" else 0,
                "hidden_layer_dac_count_per_boundary": hidden_width if inter_layer == "digital" else 0,
                "hidden_layer_rail_selector_count_per_boundary": hidden_width
                if inter_layer == "digital" and all_hidden_dual
                else 0,
                "hidden_layer_adc_count_total": hidden_width * hidden_boundary_count if inter_layer == "digital" else 0,
                "hidden_layer_dac_count_total": hidden_width * hidden_boundary_count if inter_layer == "digital" else 0,
                "hidden_layer_rail_selector_count_total": hidden_width * len(dual_boundary_layers)
                if inter_layer == "digital"
                else 0,
                "interpolation": evaluation_config.get("output_interpolation", "linear"),
                "interpolation_mapping": f"K={output_controls} per weight component -> M physical channels",
                "aperture_weighting": "digital aperture weighting and summation",
                "implementation_status": {
                    "software_model": "behavioral QAT/PTQ and nonideality model",
                    "architecture_defined": "array mapping, standard CIM readout, conversion dimensions, interpolation",
                    "physical_circuit": "not designed; requires circuit-level implementation and validation",
                },
            }
        )
    if not rows:
        raise RuntimeError("checkpoint 中没有可部署的折叠线性权重")
    write_rows(args.output / "hardware_mapping_layers.csv", rows)
    write_rows(args.output / "hardware_mapping_summary.csv", summaries)
    (args.output / "hardware_mapping.json").write_text(
        json.dumps(
            {
                "mapping_mode": args.mapping_mode,
                "tile_rows": args.tile_rows,
                "tile_cols": args.tile_cols,
                "converter_schedule": args.converter_schedule,
                "differential_readout": args.differential_readout,
                "differential_devices_per_weight": 2,
                "input_policy": "signed 2×network_channels IQ input; no differential DAC split",
                "packing_policy": "same-input layers are vertically packed and time-multiplexed",
                "hidden_inter_layer": "none; no hidden inter-layer peripheral"
                if all(summary["inter_layer"] == "none" for summary in summaries)
                else "analog; no hidden ADC/DAC"
                if all(summary["inter_layer"] == "analog" for summary in summaries)
                else "digital; hidden ADC → activation → hidden DAC → rail selector",
                "layers": rows,
                "summary": summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def run_crosssim(args: argparse.Namespace) -> None:
    if args.crosssim_runs <= 0:
        raise ValueError("crosssim-runs must be positive")
    if args.tile_rows <= 0 or args.tile_cols <= 0:
        raise ValueError("tile size must be positive")
    unsupported = {
        "converter_schedule": args.converter_schedule != "parallel",
        "input_timing": args.input_timing != "analog_single_pulse",
        "differential_readout": args.differential_readout != "post_tia_subtractor",
        "inter_layer": args.inter_layer not in {None, "none"},
    }
    ignored = sorted(key for key, active in unsupported.items() if active)
    if ignored:
        raise ValueError(f"CrossSim 当前不建模这些参数，请恢复默认值: {', '.join(ignored)}")
    try:
        from applications.dnn.dnn_inference_params import dnn_inference_params
        import simulator
        from simulator import AnalogCore
    except ImportError as error:
        raise RuntimeError("CrossSim 未安装；请先 pip install -e external/cross-sim") from error
    cases = ("crossbar_stress",)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    for label, path in parse_sweep_models(args.model):
        model_row_start = len(rows)
        checkpoint = load_checkpoint(path)
        model_config = checkpoint.get("model_config", {})
        hardware_config = (checkpoint.get("hardware_qat_state") or {}).get("config") or {}
        hidden_layers = checkpoint_hidden_layers(model_config, checkpoint.get("model_state_dict", {}))
        bias_layers = checkpoint_bias_layers(model_config, hidden_layers)
        bias_implementation = checkpoint_bias_implementation(model_config, hardware_config)
        weight_bits = int(args.weight_bits if args.weight_bits is not None else hardware_config.get("weight_bits", 4))
        input_bits = int(args.input_bits if args.input_bits is not None else hardware_config.get("input_bits", 8))
        adc_bits = int(args.adc_bits if args.adc_bits is not None else hardware_config.get("adc_bits", 8))
        if not 2 <= weight_bits <= 32:
            raise ValueError(f"{label} 的 weight_bits 必须在 [2, 32] 内")
        if not 2 <= input_bits <= 32 or not 2 <= adc_bits <= 32:
            raise ValueError(f"{label} 的 input_bits 和 adc_bits 必须在 [2, 32] 内")
        state = folded_model_state(checkpoint, path)
        for layer_name, weight_tensor in sorted(state.items()):
            if not layer_name.endswith(".weight"):
                continue
            weight = weight_tensor.detach().cpu().numpy().astype(np.float32)
            layer_name = layer_name.removesuffix(".weight")
            bias_tensor = state.get(f"{layer_name}.bias")
            bias = None if bias_tensor is None else bias_tensor.detach().cpu().numpy().astype(np.float32)
            x = rng.normal(0.0, 1.0, size=(args.crosssim_runs, weight.shape[1])).astype(np.float32)
            if bias_implementation == "array" and layer_name in bias_layers:
                x[:, -1] = 1.0
            ideal_params = dnn_inference_params(
                drift_model="IdealDevice",
                weight_bits=weight_bits,
                input_bits=input_bits,
                adc_bits=adc_bits,
                core_style="BALANCED",
                NrowsMax=args.tile_rows,
                NcolsMax=args.tile_cols,
                input_range=[-4.0, 4.0],
                adc_range=[-100.0, 100.0],
            )
            ideal_core = AnalogCore(weight, ideal_params)
            ideal_output = np.asarray(ideal_core @ x.T).T
            if bias is not None:
                ideal_output = ideal_output + bias[None, :]
            for case in cases:
                kwargs: dict[str, object] = {
                    "drift_model": "IdealDevice",
                    "weight_bits": weight_bits,
                    "input_bits": input_bits,
                    "adc_bits": adc_bits,
                    "core_style": "BALANCED",
                    "NrowsMax": args.tile_rows,
                    "NcolsMax": args.tile_cols,
                    "input_range": [-4.0, 4.0],
                    "adc_range": [-100.0, 100.0],
                    "Rp_row": 0.35,
                    "Rp_col": 0.35,
                    "noise_model": "generic",
                    "alpha_noise": 0.033,
                }
                params = dnn_inference_params(**kwargs)
                core = AnalogCore(weight, params)
                output = np.asarray(core @ x.T).T
                if bias is not None:
                    output = output + bias[None, :]
                error = output - ideal_output
                rows.append(
                    {
                        "model": label,
                        "checkpoint": str(path),
                        "layer": layer_name,
                        "case": case,
                        "matrix_rows": int(weight.shape[0]),
                        "matrix_cols": int(weight.shape[1]),
                        "mvm_samples": args.crosssim_runs,
                        "weight_bits": weight_bits,
                        "input_bits": input_bits,
                        "adc_bits": adc_bits,
                        "ordinary_bias_included": bool(bias is not None),
                        "ideal_output_rms": float(np.sqrt(np.mean(np.square(ideal_output)))),
                        "crosssim_error_rmse": float(np.sqrt(np.mean(np.square(error)))),
                        "crosssim_relative_rmse": float(
                            np.sqrt(np.mean(np.square(error))) / max(np.sqrt(np.mean(np.square(ideal_output))), 1.0e-12)
                        ),
                    }
                )
        if len(rows) == model_row_start:
            raise RuntimeError(f"{label} 的 checkpoint 中没有可部署的折叠线性权重")
    if not rows:
        raise RuntimeError("checkpoint 中没有可部署的折叠线性权重")
    write_rows(args.output / "crosssim_layer_results.csv", rows)
    summary: list[dict[str, object]] = []
    for model in dict.fromkeys(str(row["model"]) for row in rows):
        for case in cases:
            selected = [row for row in rows if row["model"] == model and row["case"] == case]
            summary.append(
                {
                    "model": model,
                    "case": case,
                    "layers": len(selected),
                    "mean_relative_rmse": float(np.mean([float(row["crosssim_relative_rmse"]) for row in selected])),
                    "max_relative_rmse": float(np.max([float(row["crosssim_relative_rmse"]) for row in selected])),
                }
            )
    write_rows(args.output / "crosssim_summary.csv", summary)
    (args.output / "crosssim_protocol.json").write_text(
        json.dumps(
            {
                "cases": cases,
                "crosssim_version": getattr(simulator, "__version__", "unknown"),
                "mapping_mode": args.mapping_mode,
                "tile_rows": args.tile_rows,
                "tile_cols": args.tile_cols,
                "weight_bits": args.weight_bits if args.weight_bits is not None else "checkpoint_config_or_4",
                "input_bits": args.input_bits if args.input_bits is not None else "checkpoint_config_or_8",
                "adc_bits": args.adc_bits if args.adc_bits is not None else "checkpoint_config_or_8",
                "ir_drop_mapping": {"Rp_row": 0.35, "Rp_col": 0.35},
                "crossbar_stress_mapping": {
                    "ir_drop": True,
                    "crossbar_read_noise_std": 0.033,
                    "programming_error": "not included; CrossSim crossbar_stress is restricted to directly comparable crossbar effects",
                },
                "interpretation": "crossbar-level trend verification; not an end-to-end SSIM replacement",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def run_ppa(args: argparse.Namespace) -> None:
    if args.tile_rows <= 0 or args.tile_cols <= 0:
        raise ValueError("tile size must be positive")
    if args.weight_bits is not None and args.weight_bits <= 0:
        raise ValueError("weight-bits must be positive")
    if args.input_bits is not None and args.input_bits <= 0:
        raise ValueError("input-bits must be positive")
    if args.adc_bits is not None and args.adc_bits <= 0:
        raise ValueError("adc-bits must be positive")
    if args.technology_node <= 0:
        raise ValueError("technology-node must be positive")
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for label, path in parse_sweep_models(args.model):
        checkpoint = load_checkpoint(path)
        model_config = checkpoint.get("model_config", {})
        evaluation_config = checkpoint.get("evaluation_config", {})
        hardware_config = (checkpoint.get("hardware_qat_state") or {}).get("config") or {}
        state = folded_model_state(checkpoint, path)
        weight_bits = int(args.weight_bits if args.weight_bits is not None else hardware_config.get("weight_bits", 4))
        input_bits = int(args.input_bits if args.input_bits is not None else hardware_config.get("input_bits", 4))
        adc_bits = int(args.adc_bits if args.adc_bits is not None else hardware_config.get("control_bits", 4))
        if not 2 <= weight_bits <= 32 or not 2 <= input_bits <= 32 or not 2 <= adc_bits <= 32:
            raise ValueError(f"{label} 的 PPA 位宽必须在 [2, 32] 内")
        hidden_layers = checkpoint_hidden_layers(model_config, state)
        output_layer_index = hidden_layers + 1
        bias_layers = checkpoint_bias_layers(model_config, hidden_layers)
        inter_layer = str(args.inter_layer or hardware_config.get("inter_layer", "none"))
        if inter_layer not in INTER_LAYER_MODES:
            raise ValueError(f"{label} 的 inter_layer 必须是 none、analog 或 digital")
        bias_implementation = checkpoint_bias_implementation(model_config, hardware_config)
        output_controls = int(model_config.get("output_controls", 0))
        output_components = (
            2
            if str(evaluation_config.get("output_weights", model_config.get("output_weights", "real"))) == "complex"
            else 1
        )
        output_conversion_dimension = output_controls * output_components
        dual_rail = str(model_config.get("branch_mode", "single")) == "dual"
        inter_layer_bits = int(
            args.inter_layer_bits if args.inter_layer_bits is not None else hardware_config.get("inter_layer_bits", 4)
        )
        if not 2 <= inter_layer_bits <= 32:
            raise ValueError(f"{label} 的 inter_layer_bits 必须是 2 到 32")
        hidden_width = checkpoint_hidden_width(model_config, state)
        hidden_converter_count = hidden_layers * hidden_width * 2
        layer_specs = _linear_weight_specs(state)
        mapping_layout, packing_groups = _build_mapping_layout(
            layer_specs, args.mapping_mode, args.tile_rows, args.tile_cols
        )
        for layer_index, key, out_dim, in_dim in layer_specs:
            tile_geometry = mapping_layout[key]
            tile_col_count = int(tile_geometry["tile_col_count"])
            logical_weights = out_dim * in_dim
            is_first_layer = layer_index == 1
            is_output_layer = layer_index == output_layer_index
            layer_name = key.removesuffix(".weight")
            has_bias = layer_name in bias_layers
            layer_bias_implementation = checkpoint_layer_bias_implementation(
                bias_implementation, inter_layer
            )
            signal_input_dim = in_dim - int(is_first_layer and has_bias and layer_bias_implementation == "array")
            signal_col_blocks = int(np.ceil(signal_input_dim / args.tile_cols)) if is_first_layer else 0
            input_dac_count = 0
            if is_first_layer:
                input_dac_count = (
                    signal_input_dim
                    if args.converter_schedule == "parallel"
                    else min(args.tile_cols, signal_input_dim)
                )
            output_adc_count = 0
            if is_output_layer:
                output_adc_count = out_dim if args.differential_readout == "post_tia_subtractor" else 2 * out_dim
            rows.append(
                {
                    "model": label,
                    "layer": key.removesuffix(".weight"),
                    "matrix_rows": out_dim,
                    "matrix_cols": in_dim,
                    "logical_weights": logical_weights,
                    "differential_devices": logical_weights * 2,
                    "tile_rows": args.tile_rows,
                    "tile_cols": args.tile_cols,
                    **_mapping_row_fields(args.mapping_mode, tile_geometry),
                    "weight_bits": weight_bits,
                    "input_bits": input_bits,
                    "input_timing": args.input_timing,
                    "adc_bits": adc_bits,
                    "dual_rail": dual_rail,
                    "inter_layer": inter_layer,
                    "inter_layer_bits": inter_layer_bits if inter_layer == "digital" else 32,
                    "hidden_adc_count": out_dim if inter_layer == "digital" and not is_output_layer else 0,
                    "hidden_dac_count": out_dim if inter_layer == "digital" and not is_output_layer else 0,
                    "input_conversion_dimension": signal_input_dim if is_first_layer else 0,
                    "physical_dac_count": input_dac_count,
                    "array_bias_dac_count": int(has_bias and layer_bias_implementation == "array"),
                    "input_dac_passes": (1 if args.converter_schedule == "parallel" else signal_col_blocks)
                    if is_first_layer
                    else 0,
                    "output_conversion_dimension": out_dim if is_output_layer else 0,
                    "physical_adc_count": output_adc_count,
                    "output_adc_passes": 1 if is_output_layer else 0,
                    "partial_sum_passes": tile_col_count,
                    "differential_readout": args.differential_readout,
                    "inter_layer_conversion": (
                        f"{inter_layer_bits}-bit hidden ADC/DAC; custom MBAN peripheral excluded"
                        if inter_layer == "digital" and not is_output_layer
                        else "no hidden inter-layer converter"
                        if inter_layer == "none" and not is_output_layer
                        else "analog signal remains continuous"
                        if not is_output_layer
                        else "signed final-control ADC -> digital Bias (floating-point accumulation; overflow not modeled) -> activation -> direct ADC-scale decode"
                        if has_bias and inter_layer == "digital" and layer_bias_implementation == "digital"
                        else "signed final-control ADC -> activation -> direct ADC-scale decode"
                        if inter_layer == "digital"
                        else "analog output ReLU with threshold/gain channel mismatch (no independent ReLU noise) -> unsigned final-control ADC"
                        if is_output_layer and evaluation_config.get("output_domain", "signed") == "nonnegative"
                        else "signed final-control ADC"
                        if is_output_layer
                        else "final output conversion"
                    ),
                    "bias_parameters": out_dim if has_bias else 0,
                    "bias_outside_crossbar": has_bias and layer_bias_implementation != "array",
                    "bias_implementation": layer_bias_implementation if has_bias else "none",
                    "neurosim_scope": "array + built-in standard peripheral/interconnect models",
                    "custom_mban_peripheral_scope": "not included in adapter total",
                }
            )
        model_rows = [row for row in rows if row["model"] == label]
        first_layer = next((row for row in model_rows if row["layer"] == "fc1"), None)
        first_input_dim = (
            int(first_layer["input_conversion_dimension"])
            if first_layer
            else int(model_config.get("network_channels", 0)) * 2
        )
        summaries.append(
            {
                "model": label,
                "checkpoint": str(path),
                "mapping_mode": args.mapping_mode,
                "technology_node_nm": int(args.technology_node),
                "weight_bits": weight_bits,
                "input_bits": input_bits,
                "input_timing": args.input_timing,
                "adc_bits": adc_bits,
                "inter_layer": inter_layer,
                "bias_implementation": bias_implementation,
                "bias_layers": sorted(bias_layers, key=lambda layer: int(layer[2:])),
                "bias_parameters_total": sum(int(row["bias_parameters"]) for row in model_rows),
                "inter_layer_bits": inter_layer_bits if inter_layer == "digital" else 32,
                "tile_rows": int(args.tile_rows),
                "tile_cols": int(args.tile_cols),
                "input_conversion_dimension": first_input_dim,
                "physical_dac_count": first_input_dim
                if args.converter_schedule == "parallel"
                else min(args.tile_cols, first_input_dim),
                "array_bias_dac_count_total": sum(int(row["array_bias_dac_count"]) for row in model_rows),
                "input_dac_passes": 1
                if args.converter_schedule == "parallel"
                else int(np.ceil(first_input_dim / args.tile_cols)),
                "input_dac_parallel_banks": int(
                    np.ceil(first_input_dim / args.tile_cols)
                )
                if args.converter_schedule == "parallel"
                else 1,
                "final_output_conversion_dimension": output_conversion_dimension,
                "physical_adc_count": output_conversion_dimension
                if args.differential_readout == "post_tia_subtractor"
                else output_conversion_dimension * 2,
                "final_adc_readout": args.differential_readout,
                "layers": len(model_rows),
                **_mapping_summary_fields(model_rows, packing_groups),
                "dual_rail": dual_rail,
                "hidden_adc_count_total": hidden_converter_count // 2 if inter_layer == "digital" else 0,
                "hidden_dac_count_total": hidden_converter_count // 2 if inter_layer == "digital" else 0,
                "converter_schedule": args.converter_schedule,
                "block_rows": int(args.tile_rows),
                "block_cols": int(args.tile_cols),
            }
        )
    if not rows:
        raise RuntimeError("checkpoint 中没有可部署的折叠线性权重")
    write_rows(args.output / "neurosim_layers.csv", rows)
    write_rows(args.output / "neurosim_models.csv", summaries)
    precision_by_model = {
        str(summary["model"]): {
            "weight_bits": int(summary["weight_bits"]),
            "input_bits": int(summary["input_bits"]),
            "adc_bits": int(summary["adc_bits"]),
        }
        for summary in summaries
    }
    digital_summary = next((summary for summary in summaries if summary["inter_layer"] == "digital"), None)
    (args.output / "neurosim_input.json").write_text(
        json.dumps(
            {
                "framework": "NeuroSim",
                "official_repository": "https://github.com/neurosim/NeuroSim",
                "recommended_branch": "MLPInferenceV3.0",
                "purpose": "auditable layer/tile input preparation; no PPA values are fabricated",
                "technology_node_nm": int(args.technology_node),
                "mapping_mode": args.mapping_mode,
                "packing_policy": "same-input layers are vertically packed and time-multiplexed",
                "adapter_compatibility": (
                    "not compatible with bundled square-TILE adapter; custom compact PPA remains analytical"
                    if args.mapping_mode == "custom_compact"
                    else "compatible with bundled square-TILE adapter"
                ),
                "array": {
                    "tile_rows": int(args.tile_rows),
                    "tile_cols": int(args.tile_cols),
                    "differential_devices_per_weight": 2,
                },
                "precision_by_model": precision_by_model,
                "input_timing": args.input_timing,
                "conversion_mapping": {
                    "input_dimension": "2*network_channels at FC1",
                    "final_output_dimension": "output_controls (K)",
                    "converter_schedule": args.converter_schedule,
                    "block_rows": int(args.tile_rows),
                    "block_cols": int(args.tile_cols),
                    "physical_dac_count": "block_cols for time_multiplexed; input_dimension for parallel",
                    "array_bias_dac_count": "one independent Bias DAC per array-bias layer",
                    "input_dac_passes": "ceil(input_dimension / block_cols)",
                    "physical_adc_count": "output_dimension after differential TIA/subtractor",
                    "differential_readout": args.differential_readout,
                    "hidden_layer_converters": (
                        f"{digital_summary['inter_layer_bits']}-bit ADC/DAC per hidden boundary; custom MBAN peripheral excluded"
                        if digital_summary is not None
                        else "none when inter_layer=analog"
                    ),
                },
                "peripheral_scope": {
                    "neurosim_standard": [
                        "subarray",
                        "row/column readout",
                        "sensing",
                        "ADC-related cost",
                        "buffer/accumulation",
                        "interconnect",
                        "technology-node models",
                    ],
                    "custom_mban_requires_addon": [
                        "dual-rail sign split",
                        "analog hidden inter-layer",
                        "K-to-M interpolation",
                        "final normalization",
                        "IQ weighting and angle summation",
                        "program/verify/calibration controller",
                    ],
                },
                "nonidealities": {
                    "ir_drop": "evaluated separately in Python/CrossSim",
                    "programming_error": "evaluated separately in Python",
                    "drift": "evaluated separately in Python",
                    "stuck_at": "evaluated separately in Python",
                    "read_noise": "evaluated separately in Python",
                },
                "models": summaries,
                "layers": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    actual: list[dict[str, object]] = []
    adapter = ROOT / "external" / "neurosim" / "ppa_adapter"
    if not args.skip_neurosim and adapter.exists() and args.mapping_mode != "custom_compact":

        def wsl_path(path: Path) -> str:
            absolute = str(path.resolve()).replace("\\", "/")
            return f"/mnt/{absolute[0].lower()}{absolute[2:]}"

        adapter_wsl = wsl_path(adapter)
        for label, path in parse_sweep_models(args.model):
            checkpoint = load_checkpoint(path)
            hardware_config = (checkpoint.get("hardware_qat_state") or {}).get("config") or {}
            model_weight_bits = int(
                args.weight_bits if args.weight_bits is not None else hardware_config.get("weight_bits", 4)
            )
            model_input_bits = int(
                args.input_bits if args.input_bits is not None else hardware_config.get("input_bits", 4)
            )
            model_adc_bits = int(
                args.adc_bits if args.adc_bits is not None else hardware_config.get("control_bits", 4)
            )
            state = folded_model_state(checkpoint, path)
            dims = []
            for key, weight in sorted(state.items()):
                if key.endswith(".weight"):
                    dims.append(f"{int(weight.shape[0])}x{int(weight.shape[1])}")
            actual_path = args.output / f"neurosim_{label}.csv"
            command = [
                "wsl.exe",
                "-d",
                args.neurosim_distro,
                "--",
                adapter_wsl,
                label,
                wsl_path(actual_path),
                ",".join(dims),
                str(args.tile_rows),
                str(model_weight_bits),
                str(model_input_bits),
                str(model_adc_bits),
                str(args.technology_node),
                args.converter_schedule,
                args.input_timing,
            ]
            completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if completed.returncode != 0:
                raise RuntimeError(f"NeuroSim adapter failed for {label}: {completed.stdout}\n{completed.stderr}")
            match = re.search(
                r"area_m2=([0-9.eE+-]+) leakage_w=([0-9.eE+-]+) read_latency_s=([0-9.eE+-]+) read_energy_j=([0-9.eE+-]+)",
                completed.stdout,
            )
            if not match:
                raise RuntimeError(f"NeuroSim adapter returned no summary for {label}: {completed.stdout}")
            actual.append(
                {
                    "model": label,
                    "area_m2": float(match.group(1)),
                    "leakage_power_w": float(match.group(2)),
                    "read_latency_s": float(match.group(3)),
                    "read_energy_j": float(match.group(4)),
                    "dynamic_read_power_w": float(match.group(4)) / max(float(match.group(3)), 1.0e-30),
                    "read_plus_leakage_power_w": float(match.group(2))
                    + float(match.group(4)) / max(float(match.group(3)), 1.0e-30),
                    "scope": f"CIM core + NeuroSim standard peripherals; input_timing={args.input_timing}; custom MBAN blocks excluded",
                    "csv": str(actual_path),
                }
            )
        write_rows(args.output / "neurosim_ppa_summary.csv", actual)
    ledger: list[dict[str, object]] = []
    actual_by_model = {str(row["model"]): row for row in actual}
    for model_summary in summaries:
        label = str(model_summary["model"])
        actual_model = actual_by_model.get(label)
        if actual_model is not None:
            csv_path = Path(str(actual_model["csv"]))
            layer_pattern = re.compile(
                r"^# (fc\d+)_summary,tiles=(\d+),differential_devices=(\d+),"
                r"area_m2=([0-9.eE+-]+),leakage_w=([0-9.eE+-]+),"
                r"read_latency_s=([0-9.eE+-]+),read_energy_j=([0-9.eE+-]+)"
            )
            for line in csv_path.read_text(encoding="utf-8", errors="replace").splitlines():
                match = layer_pattern.match(line)
                if not match:
                    continue
                layer, tiles, devices, area, leakage, latency, energy = match.groups()
                latency_f = float(latency)
                energy_f = float(energy)
                ledger.append(
                    {
                        "model": label,
                        "stage": f"{layer}.array_standard_readout",
                        "status": "modeled",
                        "time_s": latency_f,
                        "energy_j": energy_f,
                        "dynamic_power_w": energy_f / max(latency_f, 1.0e-30),
                        "leakage_power_w": float(leakage),
                        "converter_count": "",
                        "parallel_or_passes": f"{tiles} parallel blocks"
                        if args.converter_schedule == "parallel"
                        else f"{tiles} sequential passes",
                        "physical_devices": devices,
                        "scope": "NeuroSim array + standard readout/peripheral model",
                    }
                )
        input_dim = int(model_summary["input_conversion_dimension"])
        output_dim = int(model_summary["final_output_conversion_dimension"])
        dac_count = int(model_summary["physical_dac_count"])
        dac_passes = int(model_summary["input_dac_passes"])
        model_inter_layer = str(model_summary["inter_layer"])
        hidden_converter_count = int(model_summary["hidden_adc_count_total"]) + int(
            model_summary["hidden_dac_count_total"]
        )
        modeled_rows = [row for row in ledger if row["model"] == label and row["status"] == "modeled"]
        modeled_time = sum(float(row["time_s"]) for row in modeled_rows)
        modeled_energy = sum(float(row["energy_j"]) for row in modeled_rows)
        ledger.extend(
            [
                {
                    "model": label,
                    "stage": "input_dac",
                    "status": "not_modeled",
                    "time_s": "",
                    "energy_j": "",
                    "dynamic_power_w": "",
                    "leakage_power_w": "",
                    "converter_count": dac_count,
                    "parallel_or_passes": dac_passes,
                    "scope": f"signed {input_dim}-D input; {dac_count} physical DACs, {dac_passes} passes; timing={args.input_timing}",
                },
                {
                    "model": label,
                    "stage": "hidden_inter_layer",
                    "status": "not_modeled",
                    "time_s": "",
                    "energy_j": "",
                    "dynamic_power_w": "",
                    "leakage_power_w": "",
                    "converter_count": hidden_converter_count or "",
                    "parallel_or_passes": "",
                    "scope": (
                        "dual split, mismatch, activation and analog buffer"
                        if model_inter_layer == "analog"
                        else "no hidden inter-layer peripheral"
                        if model_inter_layer == "none"
                        else "hidden ADC/DAC, activation and rail selector; custom MBAN blocks excluded"
                    ),
                },
                {
                    "model": label,
                    "stage": "array_bias_dac",
                    "status": "not_modeled" if int(model_summary["array_bias_dac_count_total"]) else "not_required",
                    "time_s": "",
                    "energy_j": "",
                    "dynamic_power_w": "",
                    "leakage_power_w": "",
                    "converter_count": int(model_summary["array_bias_dac_count_total"]) or "",
                    "parallel_or_passes": "1 per array-bias layer" if int(model_summary["array_bias_dac_count_total"]) else "",
                    "scope": "independent constant-input Bias DACs; included in array-bias hardware path",
                },
                {
                    "model": label,
                    "stage": "final_control_adc",
                    "status": "not_modeled",
                    "time_s": "",
                    "energy_j": "",
                    "dynamic_power_w": "",
                    "leakage_power_w": "",
                    "converter_count": int(model_summary["physical_adc_count"]),
                    "parallel_or_passes": 1,
                    "scope": (
                        f"{output_dim}-D signed ADC; floating-point digital Bias/ReLU follows ADC and decodes at ADC scale"
                        if model_inter_layer == "digital"
                        else f"{output_dim}-D {'unsigned ADC after analog output ReLU mismatch (no independent ReLU noise)' if evaluation_config.get('output_domain', 'signed') == 'nonnegative' else 'signed ADC'}"
                    ),
                },
                {
                    "model": label,
                    "stage": "digital_frontend",
                    "status": "not_modeled",
                    "time_s": "",
                    "energy_j": "",
                    "dynamic_power_w": "",
                    "leakage_power_w": "",
                    "converter_count": "",
                    "parallel_or_passes": "",
                    "scope": "TOF, time interpolation, phase alignment, aperture packing, RMS normalization; common upstream, outside adaptive-weight-generator hardware scope",
                },
                {
                    "model": label,
                    "stage": "digital_backend",
                    "status": "not_modeled",
                    "time_s": "",
                    "energy_j": "",
                    "dynamic_power_w": "",
                    "leakage_power_w": "",
                    "converter_count": "",
                    "parallel_or_passes": "",
                    "scope": "K-to-M aperture interpolation (MBAN-specific), IQ weighting, phase rotation, angle summation and B-mode; only K-to-M interpolation belongs to the adaptive-weight-generator boundary",
                },
                {
                    "model": label,
                    "stage": "modeled_cim_total",
                    "status": "partial_total" if modeled_rows else "not_executed",
                    "time_s": modeled_time if modeled_rows else "",
                    "energy_j": modeled_energy if modeled_rows else "",
                    "dynamic_power_w": modeled_energy / max(modeled_time, 1.0e-30) if modeled_rows else "",
                    "leakage_power_w": actual_model["leakage_power_w"] if actual_model is not None else "",
                    "converter_count": "",
                    "parallel_or_passes": "",
                    "scope": "not full-flow; excludes custom MBAN peripherals and digital front/back end",
                },
            ]
        )
    write_rows(args.output / "full_flow_resource_ledger.csv", ledger)
    status = {
        "status": "executed" if actual else "input_prepared_not_executed",
        "adapter": str(adapter),
        "distro": args.neurosim_distro,
        "official_branch_basis": "MLPInferenceV3.0",
        "differential_rails": 2,
        "technology_node_nm": int(args.technology_node),
        "converter_schedule": args.converter_schedule,
        "input_timing": args.input_timing,
        "block_rows": int(args.tile_rows),
        "block_cols": int(args.tile_cols),
        "differential_readout": args.differential_readout,
        "models": actual,
        "stock_mlp_main_consumes_this_json": False,
        "standard_neurosim_scope": [
            "subarray/crossbar",
            "row/column readout and sensing",
            "ADC-related cost",
            "standard buffer/accumulation",
            "interconnect and technology-node models",
        ],
        "custom_mban_scope_not_automatically_included": [
            "positive/negative dual-rail ReLU sign split",
            "analog hidden inter-layer implementation",
            "K-to-M interpolation and dynamic-aperture mapping",
            "final normalization and IQ weighting/angle summation",
            "program/verify/calibration controller",
        ],
        "notes": [
            "NeuroSim estimates the CIM array together with its built-in standard peripheral and interconnect models; it is not limited to raw crossbar cells.",
            "The adapter uses RealDevice cells and represents differential rails as two parallel cell arrays with shared modeled standard-peripheral accounting.",
            "MBAN-specific peripherals require an explicit NeuroSim primitive mapping, analytical PPA add-on, or circuit/SPICE validation.",
            "The Python model remains the source for end-to-end SSIM and nonideality behavior.",
        ],
    }
    if not actual:
        status["reason"] = (
            "custom_compact uses rectangular/variable-height packed macros; bundled adapter only accepts square TILE"
            if args.mapping_mode == "custom_compact"
            else "adapter not found or --skip-neurosim was specified"
        )
    (args.output / "ppa_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

"""固定 test、PTQ、Monte Carlo、权重和统计评估。"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
import time
from pathlib import Path

import mban as training
import numpy as np
import torch

TRAIN_DIR = Path(__file__).resolve().parent.parent
ROOT = TRAIN_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from eval_core.config import (
    DEFAULT_CONFIG,
    DEFAULT_HARDWARE_CONFIG,
    DEFAULT_SCENE_CONFIG,
    DEFAULT_SCENES,
    ModelSpec,
    add_model_arguments,
    calibrate_models,
    evaluation_stage,
    load_specs,
    prepare_test_specs,
    resolve_model_arguments,
)
from eval_core.inference import (
    SceneData,
    WeightPathCollector,
    activate_model_runtime,
    bootstrap_mean_ci,
    configure_geometry,
    load_scene,
    metric_pair,
    read_test_indices,
    reconstruct,
    require_explicit_weight_evaluation,
    select_test_indices,
    standard_baselines,
    write_rows,
)
from mban_core.config import configured_h5_files

import run_one

def parse_ssim_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="固定帧数据集的 SSIM 评估")
    add_model_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--eval-config", type=Path, default=DEFAULT_HARDWARE_CONFIG)
    parser.add_argument("--profile", help="硬件压力 profile；层间流程通过 inter_layer 选择")
    parser.add_argument("--test-frames", type=int, default=50, help="从固定 split 的 test 行中确定性抽取的帧数")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--micro-batch", type=int, default=8192)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    args.test_h5 = configured_h5_files(args.config)[0]
    return resolve_model_arguments(args)


def run_ptq(argv: list[str]) -> None:
    _run_ssim("ptq", argv)


def run_mc(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="固定 test split 的多次硬件实现 MC SSIM 评估")
    add_model_arguments(parser)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--eval-config", type=Path, default=DEFAULT_HARDWARE_CONFIG)
    parser.add_argument("--profile", help="硬件压力 profile；层间流程通过 inter_layer 选择")
    parser.add_argument("--test-frames", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--micro-batch", type=int, default=8192)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    args.test_h5 = configured_h5_files(args.config)[0]
    resolve_model_arguments(args)
    if args.runs <= 1:
        raise ValueError("runs must be greater than one")
    if args.micro_batch <= 0:
        raise ValueError("micro-batch must be positive")
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    device = torch.device(args.device)
    requests = args.model
    labels = [request.display_name for request in requests]
    specs = load_specs(args, device, allowed_model_types={"qat", "ptq"})
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    effective_hardware_config = {
        spec.display_name: dataclasses.asdict(spec.model.hardware_qat.config) for spec in specs
    }
    model_type_by_model = {spec.display_name: spec.model_type for spec in specs}
    (args.output / "mc_protocol.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "display_name": request.display_name,
                        "checkpoint": str(request.path),
                        "model_type": model_type_by_model[request.display_name],
                        "profile": request.profile or args.profile,
                        "overrides": list(request.overrides),
                    }
                    for request in requests
                ],
                "config": str(args.config.resolve()),
                "eval_config": str(args.eval_config.resolve()),
                "test_h5": str(args.test_h5.resolve()),
                "split_file": str(args.split_file.resolve()),
                "test_frames": args.test_frames,
                "effective_hardware_config": effective_hardware_config,
                "runs": args.runs,
                "seed": args.seed,
                "seed_mapping": "hardware.noise_seed=base_seed+run; default base_seed=0, so final seeds are 0..29",
                "realization_seeds": list(range(args.seed, args.seed + args.runs)),
                "deterministic": {
                    spec.display_name: not bool(spec.model.hardware_qat.config.noise_enabled) for spec in specs
                },
                "static_within_realization": [
                    "write_error",
                    "drift",
                    "stuck_at",
                    "configured_static_gain_offset_mismatch",
                ],
                "dynamic_per_forward_call": [
                    "dac_noise",
                    "read_noise",
                    "tia_noise",
                    "adc_noise",
                ],
                "deterministic_across_realizations": [
                    "configured_systematic_gain_offset_saturation",
                    "configured_systematic_activation_mismatch",
                    "configured_systematic_converter_gain_offset_nonlinearity",
                    "ir_drop_geometry_and_conductance_mapping",
                ],
                "common_mode_column_current_in_tia_saturation": True,
                "common_mode_saturation_scale": "column_current/(ir_drop_g_max*ir_drop_v_read)*tia_saturation_reference",
                "drift_semantics": "normalized_drift_stress; no time-temperature retention law",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    calibrate_models(args, specs, device)
    all_test_indices = read_test_indices(args.split_file.resolve(), args.test_h5.resolve())
    test_indices = (
        select_test_indices(all_test_indices, args.test_frames)
        if args.test_frames is not None
        else all_test_indices
    )
    test_data = {frame: load_scene(args.test_h5.resolve(), frame) for frame in test_indices}
    active_configs = {spec.display_name: spec.model.hardware_qat.config for spec in specs}
    diagnostics_active = {
        spec.display_name: (
            spec.runtime["output_domain"] == "nonnegative"
            and spec.runtime["unity_constraint"] == "hard"
        )
        for spec in specs
    }
    for spec in specs:
        if diagnostics_active[spec.display_name]:
            spec.model.hardware_qat.begin_diagnostics()
    print("=" * 72)
    print(
        f"[EVAL:mc] models={','.join(labels)} | runs={args.runs} | "
        f"frames/run={len(test_indices)} | device={device} | micro_batch={args.micro_batch}",
        flush=True,
    )
    rows: list[dict[str, object]] = []
    run_summaries: list[dict[str, object]] = []
    mc_started = time.perf_counter()
    for run in range(args.runs):
        run_started = time.perf_counter()
        realization_seed = args.seed + run
        print(
            f"[MC] run={run + 1}/{args.runs} start | seed={realization_seed}",
            flush=True,
        )
        for spec in specs:
            hardware = spec.model.hardware_qat
            active_config = active_configs[spec.display_name]
            hardware.config = dataclasses.replace(active_config, noise_seed=realization_seed)
            hardware.reset_noise_counter()
            if active_config.noise_enabled:
                hardware.begin_noise_realization()
            values_db: list[float] = []
            values_env: list[float] = []
            for frame in test_indices:
                data = test_data[frame]
                image, _, _, _ = reconstruct(spec, data, device, args.micro_batch, collect_weight_stats=False)
                db, env = metric_pair(image, data.gt_db)
                values_db.append(db)
                values_env.append(env)
                rows.append(
                    {
                        "model": spec.display_name,
                        "run": run,
                        "seed": realization_seed,
                        "frame": frame,
                        "ssim_db": db,
                        "ssim_env": env,
                    }
                )
            run_summaries.append(
                {
                        "model": spec.display_name,
                    "run": run,
                    "ssim_db_mean": float(np.mean(values_db)),
                    "ssim_env_mean": float(np.mean(values_env)),
                }
            )
        run_elapsed = time.perf_counter() - run_started
        total_elapsed = time.perf_counter() - mc_started
        eta = total_elapsed / (run + 1) * (args.runs - run - 1)
        print(
            f"[MC] run={run + 1}/{args.runs} done | "
            f"models={len(specs)} | "
            f"run_time={run_elapsed:.1f}s | elapsed={total_elapsed:.1f}s | eta={eta:.1f}s",
            flush=True,
        )
    for spec in specs:
        spec.model.hardware_qat.config = active_configs[spec.display_name]
    diagnostics = {
        spec.display_name: spec.model.hardware_qat.end_diagnostics(float(getattr(training.args, "public_gain_max", 16.0)))
        for spec in specs
        if diagnostics_active[spec.display_name]
    }
    if diagnostics:
        (args.output / "hardware_diagnostics.json").write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    with (args.output / "test_frame_ssim_by_realization.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "test_run_ssim_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(run_summaries[0]))
        writer.writeheader()
        writer.writerows(run_summaries)
    summary = []
    for spec in specs:
        model_runs = [row for row in run_summaries if row["model"] == spec.display_name]
        run_db = np.asarray([float(row["ssim_db_mean"]) for row in model_runs])
        run_env = np.asarray([float(row["ssim_env_mean"]) for row in model_runs])
        db_ci_low, db_ci_high = bootstrap_mean_ci(run_db, 100000 + args.seed + labels.index(spec.display_name))
        env_ci_low, env_ci_high = bootstrap_mean_ci(run_env, 200000 + args.seed + labels.index(spec.display_name))
        summary.append(
            {
                "model": spec.display_name,
                "runs": args.runs,
                "frames_per_run": len(test_indices),
                "ssim_db_mean": float(run_db.mean()),
                "ssim_db_std": float(run_db.std(ddof=1)),
                "ssim_db_p05": float(np.quantile(run_db, 0.05)),
                "ssim_db_p95": float(np.quantile(run_db, 0.95)),
                "ssim_db_ci95_low": db_ci_low,
                "ssim_db_ci95_high": db_ci_high,
                "ssim_env_mean": float(run_env.mean()),
                "ssim_env_std": float(run_env.std(ddof=1)),
                "ssim_env_p05": float(np.quantile(run_env, 0.05)),
                "ssim_env_p95": float(np.quantile(run_env, 0.95)),
                "ssim_env_ci95_low": env_ci_low,
                "ssim_env_ci95_high": env_ci_high,
            }
        )
    with (args.output / "test_mc_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    for result in summary:
        print(
            f"[RESULT] {result['model']} | SSIM_dB={result['ssim_db_mean']:.4f}±{result['ssim_db_std']:.4f} "
            f"[p05={result['ssim_db_p05']:.4f}, p95={result['ssim_db_p95']:.4f}] | "
            f"SSIM_env={result['ssim_env_mean']:.4f}±{result['ssim_env_std']:.4f}"
        )
    print(f"[OUTPUT] {args.output}")
    print("=" * 72)


def _evaluate_standard_baselines(
    data: SceneData, output: Path, f_number: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    das_img, mv_img = standard_baselines(data, output, f_number)
    return metric_pair(das_img, data.gt_db), metric_pair(mv_img, data.gt_db)


def _run_ssim(evaluation_mode: str = "test", argv: list[str] | None = None) -> None:
    args = parse_ssim_args(argv)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.micro_batch <= 0 or args.seed < 0:
        raise ValueError("micro-batch 必须为正数，seed 必须非负")
    test_indices = select_test_indices(
        read_test_indices(args.split_file.resolve(), args.test_h5.resolve()),
        args.test_frames,
    )
    device = torch.device(args.device)
    specs = prepare_test_specs(args, device)
    print("=" * 72)
    print(
        f"[EVAL:{evaluation_mode}] models={len(specs)} | frames/model={len(test_indices)} | "
        f"seed={args.seed} | device={device} | micro_batch={args.micro_batch}"
    )
    test_rows: list[dict[str, object]] = []
    stage_by_model: dict[str, str] = {}
    model_type_by_model: dict[str, str] = {}
    for spec in specs:
        stage_by_model[spec.display_name] = evaluation_stage(spec.model)
        model_type_by_model[spec.display_name] = spec.model_type
        print(f"[MODEL] {spec.display_name} | model_type={spec.model_type} | checkpoint={spec.path.name}")
        for frame in test_indices:
            data = load_scene(args.test_h5.resolve(), frame)
            image, _, _, _ = reconstruct(spec, data, device, args.micro_batch, collect_weight_stats=False)
            ssim_db, ssim_env = metric_pair(image, data.gt_db)
            test_rows.append(
                {
                    "model": spec.display_name,
                    "frame": frame,
                    "reference": "all_envdb_norm",
                    "ssim_db": ssim_db,
                    "ssim_env": ssim_env,
                }
            )

    # Evaluate standard baselines (DAS and MV) if testing FP32 models
    if any(spec.model_type == "fp32" for spec in specs):
        stage_by_model["DAS"] = "baseline"
        stage_by_model["MV"] = "baseline"
        f_number = float(getattr(specs[0].training_module.args, "f_number", 1.5)) if specs else 1.5
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            for frame in test_indices:
                data = load_scene(args.test_h5.resolve(), frame)
                (das_db, das_env), (mv_db, mv_env) = _evaluate_standard_baselines(data, tmp_path, f_number)
                test_rows.append(
                    {
                        "model": "DAS",
                        "frame": frame,
                        "reference": "all_envdb_norm",
                        "ssim_db": das_db,
                        "ssim_env": das_env,
                    }
                )
                test_rows.append(
                    {
                        "model": "MV",
                        "frame": frame,
                        "reference": "all_envdb_norm",
                        "ssim_db": mv_db,
                        "ssim_env": mv_env,
                    }
                )

    write_rows(args.output / "test_frame_ssim.csv", test_rows)
    summary: list[dict[str, object]] = []
    for label in dict.fromkeys(row["model"] for row in test_rows):
        rows = [row for row in test_rows if row["model"] == label]
        db = np.asarray([float(row["ssim_db"]) for row in rows])
        env = np.asarray([float(row["ssim_env"]) for row in rows])
        summary.append(
            {
                "model": label,
                "frames": len(rows),
                "ssim_db_mean": float(db.mean()),
                "ssim_db_std": float(db.std(ddof=1)) if len(db) > 1 else 0.0,
                "ssim_db_median": float(np.median(db)),
                "ssim_db_p05": float(np.quantile(db, 0.05)),
                "ssim_db_p95": float(np.quantile(db, 0.95)),
                "ssim_env_mean": float(env.mean()),
                "ssim_env_std": float(env.std(ddof=1)) if len(env) > 1 else 0.0,
                "reference": "all_envdb_norm",
            }
        )
    write_rows(args.output / "test_ssim_summary.csv", summary)
    (args.output / "run_params.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "display_name": request.display_name,
                        "checkpoint": str(request.path),
                        "model_type": model_type_by_model[request.display_name],
                        "profile": request.profile,
                        "overrides": list(request.overrides),
                    }
                    for request in args.model
                ],
                "test_h5": str(args.test_h5.resolve()),
                "split_file": str(args.split_file.resolve()),
                "test_indices": test_indices,
                "test_frames": args.test_frames,
                "test_frame_selection": "deterministic linspace over sorted split test indices",
                "reference_key": "all_envdb_norm",
                "config": str(args.config.resolve()),
                "profile": args.profile,
                "device": str(device),
                "micro_batch": args.micro_batch,
                "evaluation_mode": evaluation_mode,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    for row in summary:
        print(
            f"[RESULT] {row['model']} | stage={stage_by_model[str(row['model'])]} | "
            f"SSIM_dB={row['ssim_db_mean']:.4f}±{row['ssim_db_std']:.4f} "
            f"[p05={row['ssim_db_p05']:.4f}, p95={row['ssim_db_p95']:.4f}] | "
            f"SSIM_env={row['ssim_env_mean']:.4f}±{row['ssim_env_std']:.4f}"
        )
    print(f"[OUTPUT] {args.output}")
    print("=" * 72)
def tensor_percentile(values: torch.Tensor, q: float) -> float:
    flat = values.float().reshape(-1).cpu()
    if flat.numel() > 1_000_000:
        indices = torch.linspace(0, flat.numel() - 1, steps=1_000_000)
        indices = indices.round().long().clamp(0, flat.numel() - 1)
        flat = flat[indices]
    return float(torch.quantile(flat, torch.tensor(q)).item())


def collect_parameter_diagnostics(spec: ModelSpec) -> list[dict[str, object]]:
    rows = []
    with torch.inference_mode():
        for name, parameter in spec.model.named_parameters():
            if not name.endswith(".weight"):
                continue
            values = parameter.detach().float().reshape(-1)
            rows.append(
                {
                    "model": spec.display_name,
                    "checkpoint": str(spec.path),
                    "parameter": name,
                    "parameters": int(values.numel()),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "abs_p95": tensor_percentile(values.abs(), 0.95),
                }
            )
    return rows


def collect_weight_diagnostics(
    spec: ModelSpec,
    data: SceneData,
    device: torch.device,
    micro_batch: int,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    require_explicit_weight_evaluation([spec], "权重诊断")
    backend = spec.training_module
    activate_model_runtime(spec, backend)
    configure_geometry(data, backend)
    rf_i, rf_q, t_start = data.rf_i.to(device), data.rf_q.to(device), data.t_start.to(device)
    valid_time_samples = torch.full((rf_i.shape[0],), rf_i.shape[2], device=device, dtype=torch.long)
    x_grid, z_grid, angles = data.x_grid.to(device), data.z_grid.view(-1).to(device), data.angles.to(device)
    angle_idx = torch.arange(angles.numel(), device=device)
    offsets = torch.arange(0, 1, device=device)
    chtv_sum = 0.0
    chtv_count = 0.0
    chtv_shape_sum = 0.0
    w_abs_active: list[torch.Tensor] = []
    w_abs_norm_active: list[torch.Tensor] = []
    raw_ctrl_count = 0
    relu_zero_count = 0
    aperture_sizes: list[torch.Tensor] = []
    path_collector = WeightPathCollector()
    spec.model.begin_activation_diagnostics()
    spec.model.begin_signal_diagnostics()
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
    with torch.inference_mode():
        for offset in range(0, data.pixels, micro_batch):
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
            else:
                mask = aperture_start = aperture_size = None
            if aperture_size is None:
                aperture_size = torch.full((pixel.numel(),), data.channels, dtype=torch.long, device=device)
            aperture_sizes.append(aperture_size.detach().cpu())
            _, _, weights, controls, _, _, raw_controls = backend.predict_aperture_weights(
                spec.model,
                i_aligned,
                q_aligned,
                mask if backend.args.dynamic_aperture else None,
                aperture_start if backend.args.dynamic_aperture else None,
                aperture_size if backend.args.dynamic_aperture else None,
                depth_index=pixel // data.width,
                depth_count=data.height,
            )
            path_collector.add(raw_controls, controls, weights, mask)
            raw_controls_detached = raw_controls.reshape(-1).detach()
            raw_ctrl_count += int(raw_controls_detached.numel())
            if (
                getattr(backend.args, "output_domain", "") == "nonnegative"
                and getattr(backend.args, "unity_constraint", "") == "hard"
            ):
                relu_zero_count += int(
                    (torch.relu(raw_controls_detached) <= torch.finfo(raw_controls_detached.dtype).eps).sum().item()
                )
            effective_weights = backend.effective_aperture_weights(weights)
            wr, wi = backend.split_complex_weights(effective_weights)
            active_mask = (
                torch.ones_like(wr, dtype=torch.bool) if mask is None else mask.bool().unsqueeze(1).expand_as(wr)
            )
            w_abs = torch.sqrt(wr.square() + wi.square())
            w_abs_active.append(w_abs[active_mask].reshape(-1).detach())
            active_weight_sum = (w_abs * active_mask).sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
            w_bar = w_abs / active_weight_sum
            w_abs_norm_active.append(w_bar[active_mask].reshape(-1).detach())
            pair = active_mask[..., 1:] & active_mask[..., :-1]
            d1r = wr[..., 1:] - wr[..., :-1]
            d1i = wi[..., 1:] - wi[..., :-1]
            chtv_sum += float((torch.sqrt(d1r.square() + d1i.square()) * pair).sum())
            chtv_count += float(pair.sum())
            d1_bar = w_bar[..., 1:] - w_bar[..., :-1]
            chtv_shape_sum += float((d1_bar.abs() * pair).sum())
    aperture_values = torch.cat(aperture_sizes).float()
    w_abs_active_all = torch.cat(w_abs_active)
    w_abs_norm_active_all = torch.cat(w_abs_norm_active)
    if w_abs_active_all.numel() == 0:
        raise RuntimeError(f"模型{spec.display_name}在场景{data.h5_path}未统计到任何 active 权重")
    result: dict[str, float] = {
        "chtv_active": chtv_sum / max(chtv_count, 1.0),
        "chtv_shape": chtv_shape_sum / max(chtv_count, 1.0),
        "w_abs_min_active": float(w_abs_active_all.min().item()),
        "w_abs_p1_active": tensor_percentile(w_abs_active_all, 0.01),
        "w_abs_p5_active": tensor_percentile(w_abs_active_all, 0.05),
        "w_abs_p50_active": tensor_percentile(w_abs_active_all, 0.50),
        "w_abs_p95_active": tensor_percentile(w_abs_active_all, 0.95),
        "w_abs_p99_active": tensor_percentile(w_abs_active_all, 0.99),
        "w_abs_max_active": float(w_abs_active_all.max().item()),
        "w_abs_p5_norm": tensor_percentile(w_abs_norm_active_all, 0.05),
        "w_abs_near_zero_fraction_1e-6": float((w_abs_norm_active_all <= 1.0e-6).float().mean().item()),
        "relu_zero_fraction": relu_zero_count / max(raw_ctrl_count, 1),
        "network_channels": int(spec.model.network_channels),
        "output_controls": int(spec.model.output_controls),
        "physical_channels": int(data.channels),
        "M_min": int(aperture_values.min().item()),
        "M_max": int(aperture_values.max().item()),
        "M_mean": float(aperture_values.mean().item()),
        "M_over_K_mean": float((aperture_values / max(int(spec.model.output_controls), 1)).mean().item()),
        "fraction_M_lt_K": float((aperture_values < int(spec.model.output_controls)).float().mean().item()),
        "active_weight_samples": int(w_abs_active_all.numel()),
    }
    for layer, layer_stats in spec.model.end_activation_diagnostics().items():
        for metric, value in layer_stats.items():
            result[f"activation_{layer}_{metric}"] = float(value)
    for signal, signal_stats in spec.model.end_signal_diagnostics().items():
        for metric, value in signal_stats.items():
            result[f"signal_{signal}_{metric}"] = float(value)
    return result, path_collector.rows()


def run_weight_diagnostics(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="网络控制值、动态孔径和权重统计")
    add_model_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--eval-config", type=Path, default=DEFAULT_HARDWARE_CONFIG)
    parser.add_argument("--profile")
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--scene-config", type=Path, default=DEFAULT_SCENE_CONFIG)
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--micro-batch", type=int, default=8192)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    resolve_model_arguments(args)
    if args.micro_batch <= 0:
        raise ValueError("micro-batch 必须大于 0")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    specs = load_specs(args, device)
    calibrate_models(args, specs, device)
    scene_config = run_one.load_config(args.scene_config.resolve())
    scenes = [run_one.find_scene(scene_config, item.strip()) for item in args.scenes.split(",") if item.strip()]
    print("=" * 72)
    print(
        f"[EVAL:weights] models={len(specs)} | scenes={len(scenes)} | device={device} | micro_batch={args.micro_batch}"
    )
    parameter_rows = [parameter_row for spec in specs for parameter_row in collect_parameter_diagnostics(spec)]
    rows: list[dict[str, object]] = []
    path_rows: list[dict[str, object]] = []
    for scene in scenes:
        data = load_scene(run_one.resolve_path(scene["h5_path"]), int(scene["sample_idx"]))
        for spec in specs:
            print(f"[MODEL] {spec.display_name} | scene={scene['id']} | stage={evaluation_stage(spec.model)}")
            result, model_path_rows = collect_weight_diagnostics(spec, data, device, args.micro_batch)
            rows.append(
                {
                    "scene": scene["id"],
                    "model": spec.display_name,
                    "stage": evaluation_stage(spec.model),
                    **result,
                }
            )
            path_rows.extend(
                {"scene": scene["id"], "model": spec.display_name, **path_row} for path_row in model_path_rows
            )
    write_rows(args.output / "weight_diagnostics.csv", rows)
    write_rows(args.output / "weight_path_diagnostics.csv", path_rows)
    write_rows(args.output / "parameter_diagnostics.csv", parameter_rows)
    summary_keys = sorted({key for row in rows for key in row if key not in {"scene", "model", "stage"}})
    summary: list[dict[str, object]] = []
    for model in dict.fromkeys(str(row["model"]) for row in rows):
        model_rows = [row for row in rows if row["model"] == model]
        item: dict[str, object] = {"model": model, "stage": model_rows[0]["stage"], "scenes": len(model_rows)}
        for key in summary_keys:
            values = np.asarray([float(row[key]) for row in model_rows if key in row], dtype=float)
            if values.size:
                item[f"mean_{key}"] = float(values.mean())
                item[f"sd_{key}"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        summary.append(item)
    write_rows(args.output / "weight_diagnostics_summary.csv", summary)
    for row in summary:
        model_rows = [item for item in rows if item["model"] == row["model"]]
        m_min = min(int(item["M_min"]) for item in model_rows)
        m_max = max(int(item["M_max"]) for item in model_rows)
        print(
            f"[RESULT] {row['model']} | stage={row['stage']} | scenes={row['scenes']} | "
            f"physical={int(model_rows[0]['physical_channels'])} | K={int(model_rows[0]['output_controls'])} | "
            f"M={m_min}/{row['mean_M_mean']:.2f}/{m_max} (min/mean/max) | "
            f"mean(M/K)={row['mean_M_over_K_mean']:.3f}"
        )
        print(
            f"[WEIGHT] {row['model']} | CHTV={row['mean_chtv_active']:.4g} | "
            f"|w|_p05_norm={row['mean_w_abs_p5_norm']:.4g}"
        )
        activation_parts = [
            f"{key.removeprefix('mean_activation_')}={float(value):.4g}"
            for key, value in row.items()
            if key.startswith("mean_activation_")
            and (key.endswith("_q50") or key.endswith("_tail_ratio") or key.endswith("_mean_local_gain"))
        ]
        if activation_parts:
            print(f"[ACTIVATION] {row['model']} | " + " | ".join(activation_parts))
    print(f"[OUTPUT] {args.output}")
    print("=" * 72)


def bootstrap_ci_eval(values: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float, float]:
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def run_paired_stats_evaluation(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="配对SSIM差值和Bootstrap统计")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", default="H64")
    parser.add_argument("--models", default="PTQ8,PTQ6,PTQ4")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    with args.input.resolve().open(encoding="utf-8-sig", newline="") as handle:
        data: dict[str, dict[int, dict[str, float]]] = {}
        for row in csv.DictReader(handle):
            data.setdefault(str(row["model"]), {})[int(row["frame"])] = {
                "ssim_db": float(row["ssim_db"]),
                "ssim_env": float(row["ssim_env"]),
            }
    if args.reference not in data:
        raise KeyError(f"reference model not found: {args.reference}")
    selected_models = [item.strip() for item in args.models.split(",") if item.strip()]
    print("=" * 72)
    print(
        f"[EVAL:stats] reference={args.reference} | models={len(selected_models)} | "
        f"bootstrap={args.bootstrap_samples} | seed={args.seed}"
    )
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    for model in selected_models:
        if model not in data:
            raise KeyError(f"model not found: {model}")
        frames = sorted(set(data[args.reference]) & set(data[model]))
        if not frames:
            raise ValueError(f"no paired frames for {model}")
        for metric in ("ssim_db", "ssim_env"):
            delta = np.asarray(
                [data[model][frame][metric] - data[args.reference][frame][metric] for frame in frames], dtype=np.float64
            )
            ci_low, ci_high = bootstrap_ci_eval(delta, rng, args.bootstrap_samples)
            rows.append(
                {
                    "reference": args.reference,
                    "model": model,
                    "metric": metric,
                    "frames": int(delta.size),
                    "mean_delta": float(delta.mean()),
                    "std_delta": float(delta.std(ddof=1)) if delta.size > 1 else 0.0,
                    "median_delta": float(np.median(delta)),
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "model_better_frames": int(np.count_nonzero(delta > 0.0)),
                    "reference_better_frames": int(np.count_nonzero(delta < 0.0)),
                    "tie_frames": int(np.count_nonzero(delta == 0.0)),
                }
            )
    args.output.resolve().mkdir(parents=True, exist_ok=True)
    write_rows(args.output.resolve() / "paired_ssim_stats.csv", rows)
    (args.output.resolve() / "paired_ssim_stats.json").write_text(
        json.dumps(
            {
                "input": str(args.input.resolve()),
                "reference": args.reference,
                "models": selected_models,
                "bootstrap_samples": args.bootstrap_samples,
                "seed": args.seed,
                "difference_definition": "model - reference",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    for row in rows:
        print(
            f"[RESULT] {row['model']} | metric={row['metric']} | n={row['frames']} | "
            f"delta={row['mean_delta']:+.6f} | CI95=[{row['bootstrap_ci95_low']:+.6f}, "
            f"{row['bootstrap_ci95_high']:+.6f}]"
        )
    print(f"[OUTPUT] {args.output.resolve()}")
    print("=" * 72)

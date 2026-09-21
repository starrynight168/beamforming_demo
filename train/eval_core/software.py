from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
import torch

TRAIN_DIR = Path(__file__).resolve().parent.parent
ROOT = TRAIN_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TRAIN_DIR))

import mban as training
from eval_core.config import (
    DEFAULT_CONFIG,
    DEFAULT_HARDWARE_CONFIG,
    DEFAULT_SCENES,
    add_model_arguments,
    parse_args,
    resolve_model_arguments,
)
from eval_core.inference import (
    bootstrap_mean_ci,
    load_scene,
    metric_pair,
    monte_carlo_reconstruct,
    prepare_noise_realization,
    reconstruct,
    read_test_indices,
    save_weight_curves,
    standard_baselines,
    select_test_indices,
    write_rows,
)
from eval_core.config import ModelSpec, calibrate_models, evaluation_stage, load_specs
from mban_core.config import configured_h5_files

import run_one
from evaluation.evaluate import local_ssim
from evaluation.plot_metrics import MAX_CURVE_METHODS_PER_PAGE, build_page_method_colors, method_sort_key


def reference_only_scene(scene: dict[str, object]) -> bool:
    """Only the four controlled scenes use the full phantom metric engine."""
    return str(scene.get("id", "")) not in DEFAULT_SCENES


def save_ssim_summary_plot(
    output: Path,
    prefix: str,
    summary: list[dict[str, object]],
    reference_label: str,
) -> None:
    for stale in output.glob(f"{prefix}_ssim_mean_std_page*.png"):
        stale.unlink(missing_ok=True)
    if not summary:
        return
    summary = sorted(summary, key=lambda row: method_sort_key(row["model"]))
    max_models_per_page = MAX_CURVE_METHODS_PER_PAGE
    page_count = math.ceil(len(summary) / max_models_per_page)
    for page_idx, start in enumerate(range(0, len(summary), max_models_per_page), start=1):
        page_summary = summary[start : start + max_models_per_page]
        page_names = [str(row["model"]) for row in page_summary]
        page_suffix = "" if page_idx == 1 else f"_page{page_idx}"
        page_title_suffix = "" if page_count == 1 else f" (page {page_idx}/{page_count})"
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(14.0, max(6.0, 0.42 * len(page_summary) + 3.0)),
            sharey=True,
        )
        positions = np.arange(len(page_summary))
        page_colors = [build_page_method_colors(page_names)[name] for name in page_names]
        for axis, mean_key, std_key, title in (
            (axes[0], "ssim_db_mean", "ssim_db_std", "SSIM: dB display domain"),
            (axes[1], "ssim_env_mean", "ssim_env_std", "SSIM: envelope domain"),
        ):
            values = [float(row[mean_key]) for row in page_summary]
            errors = [float(row[std_key]) for row in page_summary]
            bars = axis.barh(
                positions,
                values,
                xerr=errors,
                capsize=4,
                color=page_colors,
                edgecolor="black",
                linewidth=0.7,
            )
            for bar, value in zip(bars, values, strict=True):
                axis.text(
                    min(value + 0.01, 1.04),
                    bar.get_y() + bar.get_height() / 2.0,
                    f"{value:.4f}",
                    ha="left",
                    va="center",
                    fontsize=8,
                )
            axis.set_title(f"{title}{page_title_suffix}")
            axis.set_xlim(0.0, 1.08)
            axis.grid(axis="x", alpha=0.25)
            axis.set_xlabel("SSIM")
        axes[0].set_yticks(positions)
        axes[0].set_yticklabels(page_names, fontsize=8)
        axes[0].invert_yaxis()
        axes[0].set_ylabel(f"SSIM vs {reference_label}")
        fig.tight_layout()
        fig.savefig(
            output / f"{prefix}_ssim_mean_std{page_suffix}.png",
            dpi=220,
            bbox_inches="tight",
            pad_inches=0.15,
        )
        plt.close(fig)


def evaluate_reference_only(
    args: argparse.Namespace,
    specs: list[ModelSpec],
    device: torch.device,
    h5_path: Path,
    frames: list[int] | np.ndarray,
    output: Path,
    prefix: str,
    reference_key: str,
    reference_label: str,
) -> None:
    """Evaluate datasets without an independent ground truth."""
    output.mkdir(parents=True, exist_ok=True)
    frames = [int(frame) for frame in frames]
    rows: list[dict[str, object]] = []
    mc_rows: list[dict[str, object]] = []
    complexity_rows: list[dict[str, object]] = []
    hardware_rows: list[dict[str, object]] = []
    first_images: list[np.ndarray] | None = None
    first_titles: list[str] | None = None
    first_extent: list[float] | None = None
    for frame in frames:
        data = load_scene(h5_path, frame, reference_key)
        frame_images = [data.gt_db]
        frame_titles = [reference_label]
        for spec in specs:
            hardware = getattr(spec.model, "hardware_qat", None)
            if (
                hardware is not None
                and spec.runtime["output_domain"] == "nonnegative"
                and spec.runtime["unity_constraint"] == "hard"
            ):
                hardware.begin_diagnostics()
            result = monte_carlo_reconstruct(
                spec,
                data,
                device,
                args.micro_batch,
                args.monte_carlo_runs,
                base_seed=args.seed,
                return_realizations=args.monte_carlo_runs > 1,
                collect_weight_stats=False,
            )
            if args.monte_carlo_runs > 1:
                image, _, _, elapsed, _, realizations = result
            else:
                image, _, _, elapsed, _ = result
                realizations = [image]
            if (
                hardware is not None
                and spec.runtime["output_domain"] == "nonnegative"
                and spec.runtime["unity_constraint"] == "hard"
            ):
                hardware_rows.append(
                    {
                        "frame": frame,
                        "model": spec.display_name,
                        **hardware.end_diagnostics(float(getattr(training.args, "public_gain_max", 16.0))),
                    }
                )
            if args.monte_carlo_runs > 1 and len(realizations) > 1:
                realization_metrics = [metric_pair(realization, data.gt_db) for realization in realizations]
                ssim_db = float(np.mean([item[0] for item in realization_metrics]))
                ssim_env = float(np.mean([item[1] for item in realization_metrics]))
            else:
                ssim_db, ssim_env = metric_pair(image, data.gt_db)
            rows.append(
                {
                    "frame": frame,
                    "model": spec.display_name,
                    "reference": reference_label,
                    "ssim_db": ssim_db,
                    "ssim_env": ssim_env,
                    "monte_carlo_runs": args.monte_carlo_runs,
                }
            )
            if args.monte_carlo_runs > 1 and len(realizations) > 1:
                for run, realization in enumerate(realizations):
                    run_ssim_db, run_ssim_env = metric_pair(realization, data.gt_db)
                    mc_rows.append(
                        {
                            "frame": frame,
                            "model": spec.display_name,
                            "run": run,
                            "seed": int(args.seed) + run,
                            "reference": reference_label,
                            "ssim_db": run_ssim_db,
                            "ssim_env": run_ssim_env,
                        }
                    )
            complexity_rows.append(
                {
                    "frame": frame,
                    "model": spec.display_name,
                    "stage": evaluation_stage(spec.model),
                    "parameters": sum(parameter.numel() for parameter in spec.model.parameters()),
                    "inference_seconds": elapsed,
                    "pixels": data.pixels,
                    "microseconds_per_pixel": elapsed * 1.0e6 / data.pixels,
                    "monte_carlo_runs": args.monte_carlo_runs,
                }
            )
            frame_images.append(image)
            frame_titles.append(spec.display_name)
        if first_images is None:
            first_images, first_titles = frame_images, frame_titles
            first_extent = [
                float(data.x_grid[0] * 1000.0),
                float(data.x_grid[-1] * 1000.0),
                float(data.z_grid[-1] * 1000.0),
                float(data.z_grid[0] * 1000.0),
            ]
    write_rows(output / f"{prefix}_ssim_by_frame.csv", rows)
    summary: list[dict[str, object]] = []
    for model in dict.fromkeys(row["model"] for row in rows):
        model_rows = [row for row in rows if row["model"] == model]
        db_values = np.asarray([float(row["ssim_db"]) for row in model_rows], dtype=float)
        env_values = np.asarray([float(row["ssim_env"]) for row in model_rows], dtype=float)
        summary.append(
            {
                "model": model,
                "reference": reference_label,
                "frames": len(model_rows),
                "statistical_unit": "frame_mean_image",
                "monte_carlo_runs": int(args.monte_carlo_runs),
                "ssim_db_mean": float(db_values.mean()),
                "ssim_db_std": float(db_values.std(ddof=1)) if len(db_values) > 1 else 0.0,
                "ssim_env_mean": float(env_values.mean()),
                "ssim_env_std": float(env_values.std(ddof=1)) if len(env_values) > 1 else 0.0,
            }
        )
    write_rows(output / f"{prefix}_ssim_mean_std.csv", summary)
    write_rows(output / f"{prefix}_complexity.csv", complexity_rows)
    if mc_rows:
        write_rows(output / f"{prefix}_mc_ssim_by_realization.csv", mc_rows)
        mc_summary: list[dict[str, object]] = []
        for model in dict.fromkeys(row["model"] for row in mc_rows):
            model_rows = [row for row in mc_rows if row["model"] == model]
            run_values: dict[int, list[dict[str, object]]] = {}
            for row in model_rows:
                run_values.setdefault(int(row["run"]), []).append(row)
            db_values = np.asarray(
                [np.mean([float(item["ssim_db"]) for item in run_rows]) for run_rows in run_values.values()], dtype=float
            )
            env_values = np.asarray(
                [np.mean([float(item["ssim_env"]) for item in run_rows]) for run_rows in run_values.values()], dtype=float
            )
            db_ci_low, db_ci_high = bootstrap_mean_ci(db_values, 100000 + len(db_values))
            env_ci_low, env_ci_high = bootstrap_mean_ci(env_values, 200000 + len(env_values))
            mc_summary.append(
                {
                    "model": model,
                    "reference": reference_label,
                    "frames": len(frames),
                    "runs": int(args.monte_carlo_runs),
                    "samples": len(db_values),
                    "statistical_unit": "per_frame_realization",
                    "ssim_db_mean": float(db_values.mean()),
                    "ssim_db_std": float(db_values.std(ddof=1)) if len(db_values) > 1 else 0.0,
                    "ssim_db_p05": float(np.quantile(db_values, 0.05)),
                    "ssim_db_p95": float(np.quantile(db_values, 0.95)),
                    "ssim_db_ci95_low": db_ci_low,
                    "ssim_db_ci95_high": db_ci_high,
                    "ssim_env_mean": float(env_values.mean()),
                    "ssim_env_std": float(env_values.std(ddof=1)) if len(env_values) > 1 else 0.0,
                    "ssim_env_p05": float(np.quantile(env_values, 0.05)),
                    "ssim_env_p95": float(np.quantile(env_values, 0.95)),
                    "ssim_env_ci95_low": env_ci_low,
                    "ssim_env_ci95_high": env_ci_high,
                }
            )
        write_rows(output / f"{prefix}_mc_ssim_mean_std.csv", mc_summary)
    if hardware_rows:
        (output / f"{prefix}_hardware_diagnostics.json").write_text(
            json.dumps(hardware_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if first_images is not None and first_titles is not None and first_extent is not None:
        run_one.save_comparison(first_images, first_titles, first_extent, output / "comparison.png", 60.0)
        run_one.save_individual_images(
            first_images,
            first_titles,
            ["reference", *(spec.display_name for spec in specs)],
            first_extent,
            output / "individual_images",
            60.0,
        )
    if summary:
        save_ssim_summary_plot(output, prefix, summary, reference_label)


def save_metrics_table_report(output: Path, rows: list[dict[str, object]]) -> None:
    available_scenes = list(dict.fromkeys(str(row["scene"]) for row in rows))
    scene_order = [scene for scene in DEFAULT_SCENES if scene in available_scenes]
    scene_order.extend(scene for scene in available_scenes if scene not in scene_order)
    if not scene_order:
        return
    common = [
        ("SSIM\ndB", "SSIM_dB_vs_GT", "max"),
        ("SSIM\nlinear", "SSIM_envelope_vs_GT", "max"),
        ("PSNR", "PSNR_dB_vs_GT", "max"),
        ("MAE", "MAE_dB_vs_GT", "min"),
    ]
    contrast = common + [
        ("Contrast score", "contrast_score_dB", "max"),
        ("CR", "CR_dB", "max"),
        ("CNR", "CNR", "max"),
        ("gCNR", "gCNR", "max"),
        ("Speckle\nSNR", "speckle_SNR", "max"),
        ("ENL", "ENL", "max"),
    ]
    resolution = common + [
        ("FWHM\naxial", "FWHM_axial_mm", "min"),
        ("FWHM\nlateral", "FWHM_lateral_mm", "min"),
        ("PSLR", "pslr_db", "min"),
        ("ISLR", "islr_db", "min"),
        ("Distortion", "distortion_mm", "min"),
    ]

    def numeric(value: object) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    page_width, page_height = landscape(A4)
    left, right = 42.0, page_width - 42.0
    table_width = right - left
    title_size, header_size, body_size = 12.0, 9.2, 9.4
    header_height, row_height = 27.0, 24.0

    def draw_centered_text(pdf, text: str, x: float, y: float, width: float, font: str, size: float) -> None:
        pdf.setFont(font, size)
        lines = text.split("\n")
        baseline = y + (len(lines) - 1) * 5.0
        for line in lines:
            pdf.drawString(x + (width - stringWidth(line, font, size)) / 2.0, baseline, line)
            baseline -= 10.0

    def draw_table(pdf, scene: str, top: float) -> float:
        scene_rows = [
            row
            for row in rows
            if row["scene"] == scene and str(row.get("method", "")).upper() not in {"GT", "GROUND TRUTH"}
        ]
        metrics = contrast if "contrast" in scene else resolution
        headers = ["Model"] + [title for title, _, _ in metrics]
        model_width = 118.0
        metric_width = (table_width - model_width) / max(len(metrics), 1)
        widths = [model_width, *([metric_width] * len(metrics))]

        pdf.setFillColor(colors.black)
        pdf.setFont("Times-Bold", title_size)
        pdf.drawString(left, top, scene)
        table_top = top - 22.0
        pdf.setStrokeColor(colors.HexColor("#222222"))
        pdf.setLineWidth(0.9)
        pdf.line(left, table_top, right, table_top)
        header_bottom = table_top - header_height
        x = left
        for index, (header, width) in enumerate(zip(headers, widths, strict=True)):
            if index == 0:
                pdf.setFont("Times-Roman", header_size)
                pdf.drawString(x + 8.0, header_bottom + 9.0, header)
            else:
                draw_centered_text(pdf, header, x, header_bottom + 9.0, width, "Times-Roman", header_size)
            x += width
        pdf.setLineWidth(0.65)
        pdf.line(left, header_bottom, right, header_bottom)

        best_values: dict[str, float] = {}
        for _, key, direction in metrics:
            values = np.asarray([numeric(row.get(key)) for row in scene_rows], dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size:
                best_values[key] = float(np.max(finite) if direction == "max" else np.min(finite))

        y = header_bottom
        for row in scene_rows:
            row_bottom = y - row_height
            pdf.setFillColor(colors.black)
            pdf.setFont("Times-Roman", body_size)
            pdf.drawString(left + 8.0, row_bottom + 8.0, str(row.get("method", "")))
            x = left + model_width
            for _, key, _ in metrics:
                value = numeric(row.get(key))
                text = "-" if not np.isfinite(value) else f"{value:.4f}"
                font = (
                    "Times-Bold"
                    if key in best_values and np.isclose(value, best_values[key], rtol=1.0e-9, atol=1.0e-12)
                    else "Times-Roman"
                )
                draw_centered_text(pdf, text, x, row_bottom + 8.0, metric_width, font, body_size)
                x += metric_width
            pdf.setStrokeColor(colors.HexColor("#C8C8C8"))
            pdf.setLineWidth(0.3)
            pdf.line(left, row_bottom, right, row_bottom)
            y = row_bottom
        pdf.setStrokeColor(colors.HexColor("#333333"))
        pdf.setLineWidth(0.7)
        pdf.line(left, y, right, y)
        return y - 28.0

    pdf_path = output / "metrics_tables.pdf"
    pdf = canvas.Canvas(str(pdf_path), pagesize=(page_width, page_height), pageCompression=1)
    pdf.setTitle("MBAN Evaluation Metrics")
    top = page_height - 38.0
    page_number = 1
    scenes_on_page = 0
    for scene in scene_order:
        scene_count = sum(
            1
            for row in rows
            if row["scene"] == scene and str(row.get("method", "")).upper() not in {"GT", "GROUND TRUTH"}
        )
        required_height = 22.0 + header_height + row_height * scene_count + 32.0
        if scenes_on_page >= 2 or top - required_height < 34.0:
            pdf.setFont("Times-Roman", 8.0)
            pdf.drawRightString(right, 18.0, str(page_number))
            pdf.showPage()
            page_number += 1
            top = page_height - 38.0
            scenes_on_page = 0
        top = draw_table(pdf, scene, top)
        scenes_on_page += 1
    pdf.setFont("Times-Roman", 8.0)
    pdf.drawRightString(right, 18.0, str(page_number))
    pdf.save()


def evaluate_scenes(args: argparse.Namespace, specs: list[ModelSpec], device: torch.device) -> None:
    scene_config = run_one.load_config(args.scene_config.resolve())
    scenes = [run_one.find_scene(scene_config, item.strip()) for item in args.scenes.split(",") if item.strip()]
    weight_rows: list[dict[str, object]] = []
    spectrum_rows: list[dict[str, object]] = []
    complexity_rows: list[dict[str, object]] = []
    monte_carlo_rows: list[dict[str, object]] = []
    for scene in scenes:
        h5_path = run_one.resolve_path(scene["h5_path"])
        if reference_only_scene(scene):
            scene_dir = args.output / scene["id"]
            evaluate_reference_only(
                args,
                specs,
                device,
                h5_path,
                [int(scene["sample_idx"])],
                scene_dir,
                "reference",
                "all_envdb_norm",
                "reference",
            )
            continue
        data = load_scene(h5_path, int(scene["sample_idx"]))
        scene_dir = args.output / scene["id"]
        scene_dir.mkdir(parents=True, exist_ok=True)
        f_numbers = [float(spec.runtime["f_number"]) for spec in specs]
        if any(not math.isclose(value, f_numbers[0], rel_tol=1.0e-9, abs_tol=1.0e-12) for value in f_numbers[1:]):
            raise ValueError("同一次场景评估的模型必须使用相同 f_number")
        f_number = f_numbers[0]
        if data.ground_truth_f_number is not None and not math.isclose(
            f_number, data.ground_truth_f_number, rel_tol=1.0e-9, abs_tol=1.0e-12
        ):
            raise ValueError(
                f"场景 {scene['id']} 的模型 f_number={f_number:g} 与 GT 生成值 "
                f"{data.ground_truth_f_number:g} 不一致"
            )
        images, names, titles = [data.gt_db], ["ground_truth"], ["Ground Truth"]
        hardware_results: dict[str, object] = {}
        if not args.skip_baselines:
            das, mv = standard_baselines(data, scene_dir, f_number)
            images.extend((das, mv))
            names.extend(("das", "mv"))
            titles.extend(("DAS", "MV"))
        for spec in specs:
            hardware = getattr(spec.model, "hardware_qat", None)
            if (
                hardware is not None
                and spec.runtime["output_domain"] == "nonnegative"
                and spec.runtime["unity_constraint"] == "hard"
            ):
                hardware.begin_diagnostics()
            result = monte_carlo_reconstruct(
                spec,
                data,
                device,
                args.micro_batch,
                args.monte_carlo_runs,
                base_seed=args.seed,
                return_realizations=args.monte_carlo_runs > 1,
                collect_weight_stats=args.collect_weight_stats,
            )
            if args.monte_carlo_runs > 1:
                image, stats, cumulative, elapsed, mc_std, realizations = result
            else:
                image, stats, cumulative, elapsed, mc_std = result
                realizations = [image]
            if (
                hardware is not None
                and spec.runtime["output_domain"] == "nonnegative"
                and spec.runtime["unity_constraint"] == "hard"
            ):
                hardware_results[spec.display_name] = hardware.end_diagnostics(
                    float(getattr(training.args, "public_gain_max", 16.0)),
                )
            images.append(image)
            names.append(spec.display_name)
            titles.append(spec.display_name)
            weight_rows.append({"scene": scene["id"], "method": spec.display_name, **stats})
            complexity_rows.append(
                {
                    "scene": scene["id"],
                    "method": spec.display_name,
                    "stage": evaluation_stage(spec.model),
                    "parameters": sum(p.numel() for p in spec.model.parameters()),
                    "inference_seconds": elapsed,
                    "pixels": data.pixels,
                    "microseconds_per_pixel": elapsed * 1.0e6 / data.pixels,
                    "monte_carlo_runs": args.monte_carlo_runs,
                }
            )
            spectrum_rows.extend(
                {"scene": scene["id"], "method": spec.display_name, "frequency_bin": index, "cumulative_energy": float(value)}
                for index, value in enumerate(cumulative)
            )
            if np.any(mc_std > 0.0):
                np.save(scene_dir / f"{spec.display_name}_mc_std.npy", mc_std)
            if args.monte_carlo_runs > 1:
                gt_display = np.clip((data.gt_db + 60.0) / 60.0, 0.0, 1.0)
                gt_linear_envelope = 10.0 ** (data.gt_db / 20.0)
                for realization_index, realization in enumerate(realizations):
                    display = np.clip((realization + 60.0) / 60.0, 0.0, 1.0)
                    linear_envelope = 10.0 ** (realization / 20.0)
                    monte_carlo_rows.append(
                        {
                            "scene": scene["id"],
                            "method": spec.display_name,
                            "realization": realization_index,
                            "ssim_db": float(local_ssim(display, gt_display, data_range=1.0)),
                            "ssim_env": float(local_ssim(linear_envelope, gt_linear_envelope, data_range=1.0)),
                        }
                    )
        comparison = np.stack(images)
        if hardware_results:
            (scene_dir / "hardware_diagnostics.json").write_text(
                json.dumps(hardware_results, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        comparison_path = scene_dir / "comparison.npy"
        np.save(comparison_path, comparison)
        extent, _ = run_one.load_grid_and_gt(h5_path, int(scene["sample_idx"]), True)
        run_one.save_comparison(images, titles, extent, scene_dir / "comparison.png", 60.0)
        run_one.save_individual_images(images, titles, names, extent, scene_dir / "individual_images", 60.0)
        methods, labels = names[1:], titles[1:]
        metrics_dir = scene_dir / "metrics"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "evaluation" / "evaluate.py"),
                "--comparison_npy",
                str(comparison_path),
                "--methods",
                ",".join(methods),
                "--method_labels",
                ",".join(labels),
                "--h5_path",
                str(h5_path),
                "--h5_sample_idx",
                str(scene["sample_idx"]),
                "--dr",
                "60",
                "--out_dir",
                str(metrics_dir),
                "--phantom_mode",
                "auto",
                "--phantom_source",
                "auto",
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "evaluation" / "plot_metrics.py"),
                "--metrics_dir",
                str(metrics_dir),
                "--dr",
                "60",
                "--reference-mode",
                "model",
            ],
            cwd=ROOT,
            check=True,
        )
        weight_position = None
        weight_curve_specs = [
            spec for spec in specs if str(spec.runtime.get("beamforming_implementation")) == "explicit"
        ]
        if not args.no_weight_curves:
            z_index = data.height // 2 if args.weight_z_index is None else args.weight_z_index
            x_index = data.width // 2 if args.weight_x_index is None else args.weight_x_index
            if not 0 <= z_index < data.height or not 0 <= x_index < data.width:
                raise IndexError("权重曲线像素索引超出图像范围")
            if weight_curve_specs:
                save_weight_curves(scene_dir, weight_curve_specs, data, device, z_index, x_index, f_number)
                weight_position = {
                    "z_index": z_index,
                    "x_index": x_index,
                "models": [spec.display_name for spec in weight_curve_specs],
                }
        (scene_dir / "run_params.json").write_text(
            json.dumps(
                {
                    "scene": scene,
                    "models": {spec.display_name: str(spec.path) for spec in specs},
                    "evaluation": {
                        "mode": args.mode,
                        "config": str(args.config.resolve()),
                        "profile": args.profile,
                        "device": str(device),
                        "seed": int(args.seed),
                        "micro_batch": args.micro_batch,
                        "skip_baselines": args.skip_baselines,
                        "monte_carlo_runs": args.monte_carlo_runs,
                        "collect_weight_stats": args.collect_weight_stats,
                        "weight_position": weight_position,
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    across_scene_rows = []
    summary_keys = (
        "TV",
        "E2",
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
    weight_stat_rows = []
    for row in weight_rows:
        if not row.get("method") or not any(key in row for key in summary_keys):
            continue
        weight_stat_rows.append(
            {
                "scene": row.get("scene", ""),
                "method": row["method"],
                **{key: row.get(key, float("nan")) for key in summary_keys},
            }
        )
    if weight_stat_rows:
        write_rows(args.output / "weight_metrics.csv", weight_stat_rows)
        for method in dict.fromkeys(str(row["method"]) for row in weight_stat_rows):
            method_rows = [row for row in weight_stat_rows if row["method"] == method]
            summary: dict[str, object] = {
                "method": method,
                "scenario_count": len(method_rows),
                "statistical_unit": "imaging_scenario",
            }
            for key in summary_keys:
                values = []
                for row in method_rows:
                    try:
                        value = float(row.get(key, float("nan")))
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(value):
                        values.append(value)
                numeric_values = np.asarray(values, dtype=float)
                summary[f"mean_{key}"] = float(numeric_values.mean()) if len(numeric_values) else float("nan")
                summary[f"sd_{key}"] = float(numeric_values.std(ddof=1)) if len(numeric_values) > 1 else 0.0
            across_scene_rows.append(summary)
        write_rows(args.output / "weight_metrics_across_scenes.csv", across_scene_rows)
    if spectrum_rows:
        write_rows(args.output / "cumulative_spectrum.csv", spectrum_rows)
    write_rows(args.output / "complexity.csv", complexity_rows)
    if monte_carlo_rows:
        write_rows(args.output / "monte_carlo_ssim_by_realization.csv", monte_carlo_rows)
        mc_summary: list[dict[str, object]] = []
        for scene_id, method in dict.fromkeys((str(row["scene"]), str(row["method"])) for row in monte_carlo_rows):
            method_rows = [row for row in monte_carlo_rows if row["scene"] == scene_id and row["method"] == method]
            db_values = np.asarray([float(row["ssim_db"]) for row in method_rows], dtype=float)
            linear_values = np.asarray([float(row["ssim_env"]) for row in method_rows], dtype=float)
            db_ci_low, db_ci_high = bootstrap_mean_ci(db_values, 100000 + len(db_values))
            env_ci_low, env_ci_high = bootstrap_mean_ci(linear_values, 200000 + len(linear_values))
            mc_summary.append(
                {
                    "scene": scene_id,
                    "method": method,
                    "runs": len(method_rows),
                    "ssim_db_mean": float(db_values.mean()),
                    "ssim_db_std": float(db_values.std(ddof=1)) if len(db_values) > 1 else 0.0,
                    "ssim_db_p05": float(np.quantile(db_values, 0.05)),
                    "ssim_db_p95": float(np.quantile(db_values, 0.95)),
                    "ssim_db_ci95_low": db_ci_low,
                    "ssim_db_ci95_high": db_ci_high,
                    "ssim_env_mean": float(linear_values.mean()),
                    "ssim_env_std": float(linear_values.std(ddof=1)) if len(linear_values) > 1 else 0.0,
                    "ssim_env_p05": float(np.quantile(linear_values, 0.05)),
                    "ssim_env_p95": float(np.quantile(linear_values, 0.95)),
                    "ssim_env_ci95_low": env_ci_low,
                    "ssim_env_ci95_high": env_ci_high,
                }
            )
        write_rows(args.output / "monte_carlo_ssim_summary.csv", mc_summary)
    stage_by_method = {spec.display_name: evaluation_stage(spec.model) for spec in specs}
    stage_by_method.update({"DAS": "baseline", "MV": "baseline"})
    metric_rows = []
    for path in args.output.glob("*/metrics/summary_metrics.csv"):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                metric_rows.append(
                    {
                        "scene": path.parent.parent.name,
                        "stage": stage_by_method.get(row.get("method", ""), "unknown"),
                        **row,
                    }
                )
    if args.reference_method:
        reference_rows = {row["scene"]: row for row in metric_rows if row.get("method") == args.reference_method}
        if not reference_rows:
            raise ValueError(f"未找到 reference-method={args.reference_method!r} 的场景指标")
        delta_keys = (
            "SSIM_dB_vs_GT",
            "SSIM_envelope_vs_GT",
            "PSNR_dB_vs_GT",
            "MAE_dB_vs_GT",
            "contrast_score_dB",
            "CR_dB",
            "CNR",
            "gCNR",
            "FWHM_axial_mm",
            "FWHM_lateral_mm",
            "pslr_db",
            "islr_db",
            "distortion_mm",
        )
        for row in metric_rows:
            reference = reference_rows.get(row["scene"])
            if reference is None:
                continue
            for key in delta_keys:
                try:
                    row[f"delta_{key}"] = float(row[key]) - float(reference[key])
                except (KeyError, TypeError, ValueError):
                    continue
    write_rows(args.output / "all_scene_metrics.csv", metric_rows)
    save_metrics_table_report(args.output, metric_rows)


def write_report(args: argparse.Namespace, specs: list[ModelSpec]) -> None:
    lines = ["# MBAN unified validation", "", "## Models", ""]
    lines.extend(f"- {spec.display_name}: `{spec.path}`" for spec in specs)
    if args.monte_carlo_runs > 1:
        lines.extend(
            (
                "",
                f"> Monte Carlo: {args.monte_carlo_runs} noise realizations averaged per image; "
                "per-pixel dB std saved as `<scene>/<label>_mc_std.npy`.",
                "",
            )
        )
    lines.extend(
        (
            "",
            "## Statistical unit",
            "",
            "Weight metrics are summarized separately within each imaging scenario. "
            "Across-scenario mean and SD are descriptive summaries with the imaging "
            "scenario as the statistical unit, not estimates of frame-level variability.",
            "",
        )
    )
    scene_images = sorted(args.output.glob("*/comparison.png"))
    if scene_images:
        lines.extend(("", "## Scene comparisons", ""))
        for path in scene_images:
            relative = path.relative_to(args.output).as_posix()
            lines.extend((f"### {path.parent.name}", "", f"![{path.parent.name}]({relative})", ""))
    reference_plots = sorted(args.output.glob("**/*_ssim_mean_std.png"))
    for plot in reference_plots:
        relative = plot.relative_to(args.output).as_posix()
        lines.extend(
            (
                "",
                f"## Reference-only SSIM: {plot.parent.name}",
                "",
                "Only SSIM in the dB-display and linear-envelope domains is reported; "
                "the reference is not treated as an independent ground truth.",
                "",
                f"![Reference-only SSIM]({relative})",
                "",
            )
        )
    lines.extend(
        (
            "## Outputs",
            "",
            "- `metrics_tables.pdf`: publication-style per-scene metric tables with best values in bold",
            "- `all_scene_metrics.csv`: per-scene image metrics with `stage` (FP32/PTQ/QAT); no cross-scene average",
            "- `<scene>/metrics/standard_metrics.png`: CR/CNR/gCNR, speckle, FWHM, PSLR/ISLR and distortion plots",
            "- `<scene>/metrics/auxiliary_metrics.png`: Gaussian-window SSIM, PSNR and MAE plots",
            "- `<scene>/weight_curves.png`: DAS, MV and all selected models on the physical-channel axis",
            "- `<scene>/weight_controls.png`: interpolated physical-channel weights with control anchors",
            "- `<scene>/weight_controls_individual/weight_controls_<label>.png`: per-model "
            "interpolated physical-channel weights with the same axes",
            "- `<scene>/weight_controls.csv`: raw network output controls",
            "- `<scene>/weight_curves.csv`: DAS, MV and model physical-channel weights",
            "- `weight_metrics.csv`: per-scene whole-image weight summaries and pixel-distribution descriptions",
            "- `weight_metrics_across_scenes.csv`: descriptive weight statistics only; not an image-quality average",
            "- `complexity.csv`: parameters and software inference time (not hardware latency)",
            "- `reference_ssim_mean_std.csv`: in-vivo/reference-only frame mean and standard deviation",
            "- `*_mc_ssim_mean_std.csv`: Monte Carlo SSIM mean and standard deviation when enabled",
            "",
        )
    )
    (args.output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def plot_training_curves(output: Path, specs: list[ModelSpec]) -> None:
    for stale in output.glob("training_curves_page*.png"):
        stale.unlink(missing_ok=True)
    pattern = re.compile(r"Ep\s+(\d+)/(?:\d+)\s+\|\s+Tr=([0-9.eE+-]+)\s+\|\s+Val=([0-9.eE+-]+)")
    curves = []
    rows: list[dict[str, object]] = []
    for spec in specs:
        log_path = spec.path.parent / "training_log.txt"
        if not log_path.exists():
            continue
        matches = pattern.findall(log_path.read_text(encoding="utf-8", errors="replace"))
        if not matches:
            continue
        epochs = np.asarray([int(item[0]) for item in matches])
        train_loss = np.asarray([float(item[1]) for item in matches])
        val_loss = np.asarray([float(item[2]) for item in matches])
        curves.append((spec.display_name, epochs, train_loss, val_loss))
        rows.extend(
            {
                "method": spec.display_name,
                "epoch": int(epoch),
                "train_loss": float(train_value),
                "validation_loss": float(val_value),
            }
            for epoch, train_value, val_value in zip(epochs, train_loss, val_loss, strict=True)
        )
    if not curves:
        return
    curves.sort(key=lambda item: method_sort_key(item[0]))
    write_rows(output / "training_curves.csv", rows)
    max_curves_per_page = MAX_CURVE_METHODS_PER_PAGE
    curve_pages = [
        curves[start : start + max_curves_per_page]
        for start in range(0, len(curves), max_curves_per_page)
    ]
    page_count = len(curve_pages)
    for page_idx, page_curves in enumerate(curve_pages, start=1):
        curve_count = len(page_curves)
        page_suffix = "" if page_idx == 1 else f"_page{page_idx}"
        page_title_suffix = "" if page_count == 1 else f" (page {page_idx}/{page_count})"
        fig_width = 15.0
        fig_height = max(6.2, 5.2 + 0.08 * curve_count)
        fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))
        color_map = build_page_method_colors([curve[0] for curve in page_curves])
        for label, epochs, train_loss, val_loss in page_curves:
            color = color_map[label]
            axes[0].plot(epochs, train_loss, label=label, color=color, linewidth=1.15)
            axes[1].plot(
                epochs,
                val_loss,
                label=f"{label} (best={val_loss.min():.4g})",
                color=color,
                linewidth=1.15,
            )
        axes[0].set_title(f"Training loss{page_title_suffix}")
        axes[1].set_title(f"Validation loss{page_title_suffix}")
        for axis in axes:
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Loss")
            axis.grid(alpha=0.25)
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=2,
            fontsize=8,
            frameon=True,
        )
        fig.tight_layout(rect=(0.0, 0.20, 1.0, 1.0))
        fig.savefig(
            output / f"training_curves{page_suffix}.png",
            dpi=220,
            bbox_inches="tight",
            pad_inches=0.15,
        )
        plt.close(fig)


def _run_validation(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.micro_batch <= 0:
        raise ValueError("micro_batch 必须大于 0")
    if args.monte_carlo_runs < 1:
        raise ValueError("monte_carlo_runs 必须大于等于 1")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    specs = load_specs(args, device)
    specs.sort(key=lambda spec: method_sort_key(spec.display_name))
    reserved_labels = {"DAS", "MV", "GT", "GROUND TRUTH"}
    if any(spec.display_name.upper() in reserved_labels for spec in specs):
        raise ValueError("模型标签不能使用保留名称 DAS、MV 或 GT")
    scope = []
    if args.mode == "scenes":
        scope.append(f"scenes={len([item for item in args.scenes.split(',') if item.strip()])}")
    model_text = ", ".join(f"{spec.display_name}:{spec.model_type}" for spec in specs)
    print("=" * 72)
    print(
        f"[EVAL:{args.mode}] models={len(specs)} ({model_text}) | {' | '.join(scope)} | "
        f"device={device} | micro_batch={args.micro_batch} | mc_runs={args.monte_carlo_runs}"
    )
    calibrate_models(args, specs, device)
    if args.mode == "scenes":
        evaluate_scenes(args, specs, device)
    plot_training_curves(args.output, specs)
    write_report(args, specs)
    print(f"[OUTPUT] {args.output}")
    print("=" * 72)


DEFAULT_FIGURES_OUTPUT = TRAIN_DIR / "results" / "figures"


def parse_figures_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed-test comparison figures and metrics")
    add_model_arguments(parser)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_FIGURES_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--eval-config", type=Path, default=DEFAULT_HARDWARE_CONFIG)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--micro-batch", type=int, default=8192)
    parser.add_argument("--test-frames", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--f-number", type=float, default=1.5)
    args = parser.parse_args(argv)
    args.test_h5 = configured_h5_files(args.config)[0]
    return resolve_model_arguments(args)


def run_single_angle_das(h5_path: Path, frame: int, f_number: float, output_dir: Path) -> np.ndarray:
    command = [
        sys.executable,
        "-m",
        "algorithms.das",
        "--h5_path",
        str(h5_path.resolve()),
        "--h5_sample_idx",
        str(frame),
        "--output_dir",
        str(output_dir),
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
    subprocess.run(command, cwd=ROOT, check=True)
    return np.load(output_dir / "das" / "das.npy").astype(np.float32)


def write_figure_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame_position", "frame", "method", "stage", "ssim_db", "ssim_env"],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_panel(
    images: list[np.ndarray],
    titles: list[str],
    extent: list[float],
    output_path: Path,
    frame: int,
    position: int,
    total: int,
) -> None:
    columns = min(len(images), 4)
    rows = (len(images) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.7 * columns, 4.9 * rows),
        dpi=300,
        squeeze=False,
        constrained_layout=True,
    )
    image_handle = None
    for index, (image, title) in enumerate(zip(images, titles, strict=True)):
        axis = axes[index // columns][index % columns]
        image_handle = axis.imshow(image, cmap="gray", vmin=-60.0, vmax=0.0, extent=extent, aspect="equal")
        axis.set_title(title, fontsize=11, pad=7, fontweight="bold")
        axis.set_xlabel("Lateral (mm)")
        if index % columns == 0:
            axis.set_ylabel("Depth (mm)")
        else:
            axis.set_yticklabels([])
        run_one.add_scale_bar(axis, extent)
    for index in range(len(images), rows * columns):
        axes[index // columns][index % columns].axis("off")
    fig.suptitle(f"Fixed-test frame {frame} ({position}/{total})", fontsize=14)
    fig.colorbar(image_handle, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="Amplitude (dB)")
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_figure_models(args: argparse.Namespace, device: torch.device):
    specs = load_specs(args, device)
    calibrate_models(args, specs, device)
    for spec in specs:
        spec.runtime["f_number"] = float(args.f_number)
        prepare_noise_realization(getattr(spec.model, "hardware_qat", None), args.seed)
    return specs


def run_figures(argv: list[str] | None = None) -> None:
    args = parse_figures_args(argv)
    if args.micro_batch <= 0 or args.test_frames <= 0:
        raise ValueError("micro-batch 和 test-frames 必须大于 0")
    device = torch.device(args.device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    test_h5 = args.test_h5.resolve()
    split_file = args.split_file.resolve()
    frames = select_test_indices(read_test_indices(split_file, test_h5), args.test_frames)
    specs = load_figure_models(args, device)
    models = [
        {
            "display_name": spec.display_name,
            "checkpoint": str(spec.path),
            "model_type": spec.model_type,
            "stage": evaluation_stage(spec.model),
        }
        for spec in specs
    ]
    (output / "figure_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "protocol": "fixed_test_comparison_figures",
                "command": sys.argv,
                "device": str(device),
                "config": str(args.config.resolve()),
                "hardware_config": str(args.eval_config.resolve()),
                "test_h5": str(test_h5),
                "split_file": str(split_file),
                "models": models,
                "frames": frames,
                "seed": args.seed,
                "f_number": args.f_number,
                "micro_batch": args.micro_batch,
                "baseline": "single-angle DAS",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    metrics: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="single_angle_das_") as temp_dir:
        das_root = Path(temp_dir)
        for position, frame in enumerate(frames, start=1):
            data = load_scene(test_h5, int(frame))
            das = run_single_angle_das(test_h5, int(frame), args.f_number, das_root / f"frame_{frame}")
            images = []
            titles = []
            frame_rows = []
            for spec in specs:
                image, *_ = reconstruct(spec, data, device, args.micro_batch, collect_weight_stats=False)
                score_db, score_env = metric_pair(image, data.gt_db)
                images.append(image)
                titles.append(f"{spec.display_name} ({spec.model_type}, SSIM={score_db:.4f})")
                frame_rows.append(
                    {
                        "frame_position": position,
                        "frame": frame,
                        "method": spec.display_name,
                        "stage": evaluation_stage(spec.model),
                        "ssim_db": score_db,
                        "ssim_env": score_env,
                    }
                )
            das_db, das_env = metric_pair(das, data.gt_db)
            images.extend((das, data.gt_db))
            titles.extend((f"DAS single-angle (SSIM={das_db:.4f})", "Ground Truth (MV)"))
            frame_rows.append(
                {
                    "frame_position": position,
                    "frame": frame,
                    "method": "DAS_single_angle",
                    "stage": "baseline",
                    "ssim_db": das_db,
                    "ssim_env": das_env,
                }
            )
            extent = [
                float(data.x_grid[0] * 1000.0),
                float(data.x_grid[-1] * 1000.0),
                float(data.z_grid[-1] * 1000.0),
                float(data.z_grid[0] * 1000.0),
            ]
            output_path = output / f"frame{position}_idx{frame}.png"
            save_panel(images, titles, extent, output_path, int(frame), position, len(frames))
            metrics.extend(frame_rows)
            print(f"[{position}/{len(frames)}] frame={frame} -> {output_path}", flush=True)
    write_figure_metrics(output / "metrics.csv", metrics)

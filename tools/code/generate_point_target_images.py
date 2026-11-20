import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms import das, mv
from algorithms.shared_core import common
from evaluation.context import load_picmus_context, read_sample_meta
from evaluation.metrics import build_square_label_map, target_resolution
from shared_core.plot_style import DIAGNOSTIC, apply_style, draw_bmode
from tools.shared.run_output import ToolOutput


from tools.shared.scenes import RESOLUTION_SCENE_SOURCES as SCENE_SOURCES, load_scenes as load_standard_scenes


def load_scenes(config_path):
    return load_standard_scenes(config_path, SCENE_SOURCES, sources=SCENE_SOURCES)


def beamform(sample_input, method, args, device):
    sample = sample_input.sample
    params = common.load_common_params(args.config)
    x_grid = np.linspace(
        float(sample.x_grid[0]),
        float(sample.x_grid[-1]),
        (len(sample.x_grid) - 1) * args.grid_factor + 1,
        dtype=np.float32,
    )
    sample = sample._replace(x_grid=x_grid)
    sample_input = sample_input._replace(sample=sample)
    if method == "das":
        options = das.build_parser(params=params).parse_args([])
        options.signal_domain = "iq"
        options.f_number = args.f_number
        options.tgc = True
        options.tgc_alpha = 0.5
        beamformer = common.build_beamformer(
            das.DASBeamformerIQ, sample.z_grid, sample.x_grid, sample.n_elem,
            sample.pitch, sample.c, sample.fc, sample.fs, sample_input.t0,
            sample.angles, options, device,
        )
    else:
        options = mv.build_parser(params=params).parse_args([])
        options.signal_domain = "iq"
        options.f_number = args.f_number
        options.mv_dl = args.mv_dl
        options.subarray_ratio = args.subarray_ratio
        options.temporal_win = args.temporal_win
        options.fbss = True
        beamformer = common.build_beamformer(
            mv.RowDynamicMVBeamformerIQ, sample.z_grid, sample.x_grid,
            sample.n_elem, sample.pitch, sample.c, sample.fc, sample.fs,
            sample_input.t0, sample.angles, options, device,
        )
    i_out, q_out, _ = common.run_beamformer(
        beamformer, sample_input.i_data, sample_input.q_data,
        sample_input.selected_angles, sample_input.t0, sample.fs, device,
    )
    image = common.envelope_to_db(
        i_out, q_out, sample.z_grid, sample.fc, True, 0.5,
    )
    return x_grid, image


def evaluate_scene(scene, args, device, output):
    meta = read_sample_meta(scene["h5_path"], scene["sample_idx"])
    if meta["phantom_mode"] != "resolution_distortion":
        raise ValueError(f"{scene['id']} metadata must declare resolution_distortion")
    if meta["phantom_source"] != scene["expected_source"]:
        raise ValueError(
            f"{scene['id']} metadata phantom_source={meta['phantom_source']!r}; "
            f"expected {scene['expected_source']!r}"
        )
    context = load_picmus_context(scene["h5_path"], scene["sample_idx"])
    targets = context["phantom"]["resolution_targets"]
    if not targets:
        raise ValueError(f"{scene['id']} has no annotated resolution targets")
    sample_input = common.prepare_beamforming_input(
        scene["h5_path"], scene["sample_idx"], "all", signal_domain="iq",
    )
    sample = sample_input.sample
    x_mm = None
    z_mm = sample.z_grid.astype(np.float64) * 1000
    outputs = {}
    rows = []
    for method in args.algorithms:
        x_grid, image = beamform(sample_input, method, args, device)
        x_mm = x_grid.astype(np.float64) * 1000
        label_map = build_square_label_map(x_mm, z_mm, targets, args.window_mm)
        for target_idx, target in enumerate(targets, start=1):
            metric = target_resolution(
                image, x_mm, z_mm, target, args.window_mm,
                target_idx=target_idx, label_map=label_map,
            )
            if metric is None:
                raise ValueError(f"{scene['id']} target {target_idx} has no measurable ROI")
            rows.append({
                "scene_id": scene["id"],
                "sample_idx": scene["sample_idx"],
                "source": meta["phantom_source"],
                "algorithm": method,
                "target_idx": target_idx,
                **metric,
            })
        outputs[method] = image
        output.log("progress", f"{scene['id']} {method.upper()} reconstruction completed")

    apply_style(DIAGNOSTIC)
    fig, axes = plt.subplots(1, len(outputs), figsize=(6 * len(outputs), 6), squeeze=False)
    for axis, (method, image) in zip(axes[0], outputs.items(), strict=True):
        display = np.clip(image - np.nanmax(image), -60, 0)
        draw_bmode(
            axis,
            display,
            (x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]),
            60.0,
            aspect="auto",
        )
        axis.scatter(
            [target["x_mm"] for target in targets],
            [target["z_mm"] for target in targets],
            marker="+", s=18, linewidths=0.7, color="red",
        )
        axis.set_title(f"{method.upper()} | {scene['id']}")
        axis.set_xlabel("Lateral position (mm)")
        axis.set_ylabel("Depth (mm)")
    fig.tight_layout()
    output.figure(f"point_targets_{scene['id']}.png", fig)
    np.savez_compressed(
        output.path("data", f"point_targets_{scene['id']}.npz"),
        x_mm=x_mm,
        z_mm=z_mm,
        target_x_mm=np.asarray([target["x_mm"] for target in targets]),
        target_z_mm=np.asarray([target["z_mm"] for target in targets]),
        **{f"image_db_{method}": image for method, image in outputs.items()},
    )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Reconstruct and measure the annotated point targets in the two resolution scenes.",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--algorithms", nargs="+", choices=("das", "mv"), default=("das", "mv"))
    parser.add_argument("--grid-factor", type=int, default=8)
    parser.add_argument("--window-mm", type=float, default=1.8)
    parser.add_argument("--f-number", type=float, default=1.5)
    parser.add_argument("--mv-dl", type=float, default=0.01)
    parser.add_argument("--subarray-ratio", type=float, default=0.25)
    parser.add_argument("--temporal-win", type=int, default=9)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.algorithms or len(set(args.algorithms)) != len(args.algorithms):
        parser.error("algorithms must be unique")
    numeric = (args.window_mm, args.f_number, args.mv_dl, args.subarray_ratio)
    if args.grid_factor < 1 or not np.isfinite(numeric).all() or args.window_mm <= 0 or args.f_number <= 0:
        parser.error("grid-factor, window-mm and f-number must be positive")
    if args.mv_dl < 0 or not 0 < args.subarray_ratio <= 1:
        parser.error("mv-dl must be nonnegative and subarray-ratio must be in (0,1]")
    if args.temporal_win < 1 or args.temporal_win % 2 != 1:
        parser.error("temporal-win must be a positive odd integer")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    config_path = args.config.expanduser()
    args.config = (config_path if config_path.is_absolute() else ROOT / config_path).resolve()
    scenes = load_scenes(args.config)
    output = ToolOutput("generate_point_target_images", args.output_dir)
    output.config({
        "tool": "generate_point_target_images",
        "scenes": [{key: str(value) if isinstance(value, Path) else value for key, value in scene.items()} for scene in scenes],
        "algorithms": args.algorithms,
        "signal_domain": "iq",
        "grid_factor": args.grid_factor,
        "window_mm": args.window_mm,
        "f_number": args.f_number,
        "mv_dl": args.mv_dl,
        "subarray_ratio": args.subarray_ratio,
        "temporal_win": args.temporal_win,
        "device": str(device),
    })
    rows = []
    for scene in scenes:
        rows.extend(evaluate_scene(scene, args, device, output))
    output.csv("metrics", "point_target_metrics.csv", rows)
    print(f"Results: {output.root.resolve()}")


if __name__ == "__main__":
    main()

"""
Template for adding a new beamforming algorithm.

Copy this file to algorithms/my_method.py, change METHOD_NAME, then implement
beamform(). The command-line interface and output format are kept compatible
with run_one.py and run_all.py.

This template is for ordinary reconstruction/comparison scripts. To use a new
method as an H5 pack-time teacher, also implement one unique *BeamformerIQ
class whose __call__ returns I_out, Q_out; see data/scripts/pack_EPFL.py.
"""

import argparse
import json
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from beamforming_utils import db_display_range, parse_selected_angles, resolve_project_path
from common_params import add_common_arguments, add_io_arguments


METHOD_NAME = "template_algorithm"


def parse_args():
    parser = argparse.ArgumentParser(description=f"{METHOD_NAME} beamforming template")
    add_io_arguments(parser)

    # Common options used by run_one.py. Keep them even if your method ignores some.
    add_common_arguments(parser)

    # Add your own method-specific parameters here.
    parser.add_argument("--demo_gain", type=float, default=1.0)
    return parser.parse_args()


def load_from_h5(h5_path, sample_idx):
    with h5py.File(h5_path, "r") as hf:
        data = {
            "I": hf["all_multi_I"][sample_idx].astype(np.float32),
            "Q": hf["all_multi_Q"][sample_idx].astype(np.float32),
            "t0": hf["time_start_vector"][sample_idx].astype(np.float32),
            "fs": float(np.asarray(hf["fs"]).squeeze()),
            "c": float(np.asarray(hf["c"]).squeeze()),
            "fc": float(np.asarray(hf["fc"]).squeeze()),
            "pitch": float(np.asarray(hf["pitch"]).squeeze()),
            "num_channels": int(np.asarray(hf["num_channels"]).squeeze()),
            "z_grid": hf["z_grid"][:].astype(np.float32),
            "x_grid": hf["x_grid"][:].astype(np.float32),
            "angles": hf["angles"][:].astype(np.float32),
        }
        gt = None
        if "all_envdb_norm" in hf:
            gt = hf["all_envdb_norm"][sample_idx, 0].astype(np.float32)
    return data, gt


def normalize_to_db(envelope, eps=1e-24):
    envelope = np.asarray(envelope, dtype=np.float64)
    power = envelope ** 2
    power /= np.max(power) + eps
    return (10.0 * np.log10(power + eps)).astype(np.float32)


def beamform(data, args):
    """
    Implement your algorithm here.

    Inputs:
        data["I"]: IQ real part, shape [angles, time, channels]
        data["Q"]: IQ imag part, shape [angles, time, channels]
        data["x_grid"], data["z_grid"]: image grid in meters
        data["angles"], data["fs"], data["c"], data["fc"], data["pitch"]: physics metadata
        args: command-line arguments

    Return:
        image_db: 2D B-mode image in dB, shape [len(z_grid), len(x_grid)].
                  The maximum should normally be normalized to 0 dB.

    If the method performs delayed channel sampling, mask out-of-range samples
    and renormalize the remaining aperture weights before summation.
    """
    raise NotImplementedError("Copy this file, set METHOD_NAME, and implement beamform().")


def add_scale_bar(ax, extent_mm):
    bar_length = 5.0
    bar_x = extent_mm[1] - bar_length - 2.0
    bar_y = extent_mm[2] - 2.0
    ax.add_patch(patches.Rectangle((bar_x, bar_y), bar_length, 0.5, color="white", zorder=5))
    ax.text(
        bar_x + bar_length / 2,
        bar_y - 1.0,
        "5 mm",
        color="white",
        fontsize=9,
        ha="center",
        va="bottom",
        fontweight="bold",
    )


def save_figure(db_img, x_grid, z_grid, out_path, title, dr):
    vmin, vmax = db_display_range(dr)
    extent_mm = [
        float(x_grid[0] * 1000.0),
        float(x_grid[-1] * 1000.0),
        float(z_grid[-1] * 1000.0),
        float(z_grid[0] * 1000.0),
    ]
    fig, ax = plt.subplots(figsize=(7, 8), dpi=300)
    im = ax.imshow(db_img, cmap="gray", vmin=vmin, vmax=vmax, extent=extent_mm, aspect="equal")
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("Lateral (mm)")
    ax.set_ylabel("Depth (mm)")
    add_scale_bar(ax, extent_mm)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Amplitude (dB)")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_outputs(image_db, data, gt, args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = resolve_project_path(args.output_dir, project_root)
    method_dir = os.path.join(output_dir, METHOD_NAME)
    os.makedirs(method_dir, exist_ok=True)

    npy_path = os.path.join(method_dir, f"{METHOD_NAME}.npy")
    png_path = os.path.join(method_dir, f"{METHOD_NAME}.png")
    params_path = os.path.join(method_dir, "params.json")

    np.save(npy_path, image_db.astype(np.float32))
    save_figure(image_db, data["x_grid"], data["z_grid"], png_path, METHOD_NAME, args.dr)

    params = vars(args).copy()
    params["method"] = METHOD_NAME
    params["output_dir"] = args.output_dir
    params["has_gt"] = gt is not None
    with open(params_path, "w", encoding="utf-8") as file:
        json.dump(params, file, ensure_ascii=False, indent=2)

    return method_dir


def main():
    args = parse_args()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h5_path = resolve_project_path(args.h5_path, project_root)
    data, gt = load_from_h5(h5_path, args.h5_sample_idx)
    selected, _ = parse_selected_angles(data["angles"], args.select_angles)
    data["selected_angle_indices"] = selected
    data["selected_angles"] = data["angles"][selected]

    print(f"{METHOD_NAME}: {h5_path} sample={args.h5_sample_idx}, angles={len(selected)}")
    image_db = beamform(data, args)
    expected_shape = (len(data["z_grid"]), len(data["x_grid"]))
    if image_db.shape != expected_shape:
        raise ValueError(f"beamform() returned {image_db.shape}, expected {expected_shape}")
    method_dir = save_outputs(image_db, data, gt, args)
    print(f"Done | Output: {method_dir}")


if __name__ == "__main__":
    main()

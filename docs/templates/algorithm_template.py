"""Template for adding a new beamforming algorithm.

Copy this file to algorithms/my_method.py, change METHOD_NAME, then implement
beamform(). The command-line interface and output format are kept compatible
with run_one.py and run_all.py.

This template is for ordinary reconstruction/comparison scripts. To use a new
method as an H5 pack-time teacher, also implement one unique *BeamformerIQ
class whose __call__ returns i_output, q_output; see data/pack_data.py.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from algorithms.common import (
    add_common_arguments,
    add_io_arguments,
    db_display_range,
    load_from_h5 as load_packed_sample,
    parse_selected_angles,
    resolve_project_path,
    validate_db_output,
)
from matplotlib import patches

METHOD_NAME = "template_algorithm"


def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser(description=f"{METHOD_NAME} beamforming template")
    add_io_arguments(parser)

    # Common options used by run_one.py. Keep them even if your method ignores some.
    add_common_arguments(parser)

    # Add your own method-specific parameters here.
    parser.add_argument("--demo_gain", type=float, default=1.0)
    return parser.parse_args()


def load_from_h5(h5_path, sample_idx):
    """Load from h5."""
    c, fc, fs, pitch, n_elem, angles, t0, z_grid, x_grid, i_data, q_data, gt, _ = load_packed_sample(
        h5_path,
        sample_idx,
    )
    data = {
        "I": i_data,
        "Q": q_data,
        "t0": t0,
        "fs": fs,
        "c": c,
        "fc": fc,
        "pitch": pitch,
        "num_channels": n_elem,
        "z_grid": z_grid,
        "x_grid": x_grid,
        "angles": angles,
    }
    if gt is not None and gt.ndim == 3:
        gt = gt[0]
    return data, gt


def beamform(data, args):
    """Implement your algorithm here.

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
    raise NotImplementedError(
        "Copy this file, set METHOD_NAME, and implement beamform().",
    )


def add_scale_bar(ax, extent_mm):
    """Execute add scale bar."""
    bar_length = 5.0
    bar_x = extent_mm[1] - bar_length - 2.0
    bar_y = extent_mm[2] - 2.0
    ax.add_patch(
        patches.Rectangle((bar_x, bar_y), bar_length, 0.5, color="white", zorder=5),
    )
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
    """Save figure."""
    vmin, vmax = db_display_range(dr)
    extent_mm = [
        float(x_grid[0] * 1000.0),
        float(x_grid[-1] * 1000.0),
        float(z_grid[-1] * 1000.0),
        float(z_grid[0] * 1000.0),
    ]
    fig, ax = plt.subplots(figsize=(7, 8), dpi=300)
    im = ax.imshow(
        db_img,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        extent=extent_mm,
        aspect="equal",
    )
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("Lateral (mm)")
    ax.set_ylabel("Depth (mm)")
    add_scale_bar(ax, extent_mm)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Amplitude (dB)")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_outputs(image_db, data, args):
    """Save outputs."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = resolve_project_path(args.output_dir, project_root)
    method_dir = os.path.join(output_dir, METHOD_NAME)
    os.makedirs(method_dir, exist_ok=True)

    npy_path = os.path.join(method_dir, f"{METHOD_NAME}.npy")
    png_path = os.path.join(method_dir, f"{METHOD_NAME}.png")
    params_path = os.path.join(method_dir, "params.json")

    np.save(npy_path, image_db.astype(np.float32))
    save_figure(
        image_db,
        data["x_grid"],
        data["z_grid"],
        png_path,
        METHOD_NAME,
        args.dr,
    )

    params = vars(args).copy()
    params["method"] = METHOD_NAME
    params["output_dir"] = args.output_dir
    with open(params_path, "w", encoding="utf-8") as file:
        json.dump(params, file, ensure_ascii=False, indent=2)

    return method_dir


def main():
    """Run the command-line workflow."""
    args = parse_args()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h5_path = resolve_project_path(args.h5_path, project_root)
    data, gt = load_from_h5(h5_path, args.h5_sample_idx)
    selected, _ = parse_selected_angles(data["angles"], args.select_angles)
    data["selected_angle_indices"] = selected
    data["selected_angles"] = data["angles"][selected]

    print(
        f"{METHOD_NAME}: {h5_path} sample={args.h5_sample_idx}, angles={len(selected)}",
    )
    image_db = beamform(data, args)
    expected_shape = (len(data["z_grid"]), len(data["x_grid"]))
    image_db = validate_db_output(image_db, expected_shape, METHOD_NAME)
    method_dir = save_outputs(image_db, data, args)
    print(f"Done | Output: {method_dir}")


if __name__ == "__main__":
    main()

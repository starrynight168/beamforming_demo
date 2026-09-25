"""Template for adding a new beamforming algorithm.

Copy this file to algorithms/my_method.py, set METHOD_NAME, then implement
beamform(). The command-line flags and the output layout are what run_one.py and
run_all.py expect, so keep them intact:

    results/<scene>/<method>/<method>.npy   # 2D B-mode image in dB
    results/<scene>/<method>/<method>.png   # rendered figure
    results/<scene>/<method>/params.json    # run parameters

Reusable pieces live in algorithms/common.py: input loading and angle selection
(prepare_beamforming_input), device handling and GPU timing (run_beamformer),
envelope-to-dB conversion (envelope_to_db), and standard artifact output
(save_algorithm_result). Use them instead of re-implementing the boilerplate.

To use this method as an H5 pack-time teacher as well, expose one uniquely named
*BeamformerIQ class whose __call__(i_data, q_data, selected_angles, t_starts, fs)
returns (i_output, q_output); see data/scripts/pack_EPFL.py for the discovery
contract.
"""

import argparse
from pathlib import Path

from algorithms.common import (
    add_common_arguments,
    add_io_arguments,
    prepare_beamforming_input,
    resolve_project_path,
    save_algorithm_result,
    validate_db_output,
)

METHOD_NAME = "template_algorithm"


def parse_args():
    """Build the command-line interface expected by run_one.py."""
    parser = argparse.ArgumentParser(description=f"{METHOD_NAME} beamforming template")
    add_io_arguments(parser)

    # Common options used by run_one.py. Keep them even if your method ignores some.
    add_common_arguments(parser)

    # Add your own method-specific parameters here.
    parser.add_argument("--demo_gain", type=float, default=1.0)
    return parser.parse_args()


def beamform(input_data, args):
    """Implement your algorithm here and return the dB image.

    Inputs:
        input_data.i_data, input_data.q_data: IQ, shape [n_angles, time, channels]
        input_data.selected_angles: selected steering angles (rad)
        input_data.t0: per-angle start time (s)
        input_data.sample: PackedSample with c, fc, fs, pitch, n_elem, z_grid,
            x_grid, angles, gt_data and has_gt
        args: parsed command-line arguments

    Return:
        image_db: 2D B-mode image in dB, shape [len(z_grid), len(x_grid)], with the
        peak normalized to about 0 dB.

    A delay-and-sum style method only needs a beamformer callable plus the shared
    helpers, which keeps device handling, GPU timing and the dB conversion
    consistent with the other algorithms:

        from algorithms.common import envelope_to_db, run_beamformer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        i_out, q_out, elapsed = run_beamformer(
            beamformer, input_data.i_data, input_data.q_data,
            input_data.selected_angles, input_data.t0, input_data.sample.fs, device,
        )
        return envelope_to_db(
            i_out, q_out, input_data.sample.z_grid, input_data.sample.fc,
            args.tgc, args.tgc_alpha,
        )

    If the method performs delayed channel sampling, mask out-of-range samples and
    renormalize the remaining aperture weights before summation.
    """
    raise NotImplementedError(
        "Copy this file, set METHOD_NAME, and implement beamform().",
    )


def save_outputs(image_db, input_data, args):
    """Write the image, the figure and the parameter record."""
    return save_algorithm_result(
        image_db,
        input_data,
        args,
        METHOD_NAME,
        METHOD_NAME,
        0.0,
    )


def main():
    """Run the command-line workflow."""
    args = parse_args()
    input_data = prepare_beamforming_input(
        resolve_project_path(args.h5_path, Path(__file__).resolve().parents[1]),
        args.h5_sample_idx,
        args.select_angles,
    )
    sample = input_data.sample

    print(
        f"{METHOD_NAME}: sample={args.h5_sample_idx}, "
        f"angles={len(input_data.selected_angles)}/{len(sample.angles)}",
    )
    image_db = beamform(input_data, args)
    image_db = validate_db_output(
        image_db,
        (len(sample.z_grid), len(sample.x_grid)),
        METHOD_NAME,
    )
    method_dir = save_outputs(image_db, input_data, args)
    print(f"Done | Output: {method_dir}")


if __name__ == "__main__":
    main()

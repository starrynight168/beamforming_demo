"""Cross-subarray MV using ordered pairs of distinct overlapping subarrays."""

from copy import deepcopy
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from algorithms import mv

METHOD_NAME = "ccmv"
device = mv.device
parser = deepcopy(mv.parser)
parser.description = "CC-MV Beamforming"
args = None


def cross_subarray_covariance(
    x_active, subarray_size, temporal_win, use_fbss, spatial_weights=None,
):
    if x_active.ndim != 3 or temporal_win < 1 or x_active.shape[-1] != temporal_win:
        raise ValueError("temporal_win must match the nonempty snapshot time dimension")
    if not 1 <= subarray_size <= x_active.shape[1]:
        raise ValueError("subarray_size must fit the active aperture")
    x_subarrays = x_active.unfold(1, subarray_size, 1).permute(0, 3, 1, 2)
    width, length, count, time_count = x_subarrays.shape
    if count < 2:
        raise ValueError("CC-MV requires at least two overlapping subarrays; reduce subarray_ratio")
    weighted = x_subarrays
    denominator = count * (count - 1) * time_count
    if spatial_weights is not None:
        if spatial_weights.shape != (count,):
            raise ValueError("spatial_weights must contain one weight per subarray")
        if not torch.isfinite(spatial_weights).all() or (spatial_weights < 0).any():
            raise ValueError("spatial_weights must be finite and nonnegative")
        denominator = 2 * (
            spatial_weights[1:] * spatial_weights[:-1].cumsum(dim=0)
        ).sum() * time_count
        if denominator <= 0:
            raise ValueError("CC-MV requires at least two nonzero subarray weights; use rect")
        weighted = x_subarrays * spatial_weights.view(1, 1, count, 1)
    previous = weighted.cumsum(dim=2)[:, :, :-1].reshape(width, length, -1)
    following = weighted[:, :, 1:].reshape(width, length, -1)
    pairs = previous @ following.mH
    covariance = (pairs + pairs.mH) / denominator
    if use_fbss:
        covariance = 0.5 * (covariance + covariance.conj().flip(-1, -2))
    return x_subarrays, covariance


class RowDynamicCCMVBeamformerIQ(mv.RowDynamicMVBeamformerIQ):
    def get_device(self):
        return device

    def estimate_covariance(self, x_active, subarray_size, temporal_win, use_fbss):
        return cross_subarray_covariance(
            x_active, subarray_size, temporal_win, use_fbss,
            spatial_weights=self._covariance_windows[x_active.shape[1]],
        )


def main():
    mv.main(
        beamformer_class=RowDynamicCCMVBeamformerIQ,
        method_name=METHOD_NAME,
        method_label="CC-MV",
        covariance_label="Cross-subarray covariance (i != j)",
        argument_parser=parser,
    )


if __name__ == "__main__":
    main()

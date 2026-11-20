"""Hasegawa 2020, Eq. (9), with full subarray sampling (Delta_n=1)."""

# ruff: noqa: E402

import torch

if __package__:
    from algorithms.bootstrap import initialize_project_path
else:
    from bootstrap import initialize_project_path

initialize_project_path(__file__)

from algorithms import mv
from algorithms.shared_core.beamforming import forward_backward_covariance

METHOD_NAME = "ccmv"
def build_parser(params=None):
    parser = mv.build_parser(params)
    parser.description = "CC-MV Beamforming"
    parser.set_defaults(mv_dl=0.1)
    return parser




def cross_subarray_covariance(
    x_active, subarray_size, temporal_win, use_fbss, spatial_weights=None,
):
    if x_active.ndim != 3 or temporal_win < 1 or x_active.shape[-1] != temporal_win:
        raise ValueError("temporal_win must match the nonempty snapshot time dimension")
    if not 1 <= subarray_size <= x_active.shape[1]:
        raise ValueError("subarray_size must fit the active aperture")
    x_subarrays = x_active.unfold(1, subarray_size, 1).permute(0, 3, 1, 2)
    count = x_subarrays.shape[2]
    mean_subarray = x_subarrays.mean(dim=2)
    if spatial_weights is not None:
        if spatial_weights.shape != (count,):
            raise ValueError("spatial_weights must contain one weight per subarray")
        if not torch.isfinite(spatial_weights).all() or (spatial_weights < 0).any():
            raise ValueError("spatial_weights must be finite and nonnegative")
        if spatial_weights.sum() <= 0:
            raise ValueError("CC-MV requires a positive subarray weight sum")
        weights = spatial_weights / spatial_weights.sum()
        mean_subarray = (x_subarrays * weights.view(1, 1, count, 1)).sum(dim=2)
    covariance = (mean_subarray @ mean_subarray.mH) / temporal_win
    return x_subarrays, forward_backward_covariance(covariance, use_fbss)


class RowDynamicCCMVBeamformerIQ(mv.RowDynamicMVBeamformerIQ):
    def __init__(
        self, z_grid, x_grid, n_elem, pitch, c, fc, fs, _t0_all, angles_rad,
        mv_dl=0.1, fbss=True, subarray_ratio=0.5, temporal_win=1, options=None,
        device=None,
    ):
        options = options if options is not None else build_parser().parse_args([])
        super().__init__(
            z_grid, x_grid, n_elem, pitch, c, fc, fs, _t0_all, angles_rad,
            mv_dl=mv_dl, fbss=fbss, subarray_ratio=subarray_ratio,
            temporal_win=temporal_win, options=options, device=device,
        )

    def estimate_covariance(self, x_active, subarray_size, temporal_win, use_fbss):
        return cross_subarray_covariance(
            x_active, subarray_size, temporal_win, use_fbss,
            spatial_weights=self._covariance_windows[x_active.shape[1]],
        )

    def estimate_joint_covariance(self, data, valid, axis, subarray_size, weights):
        return mv.joint_spatial_covariance(
            data, valid, axis, subarray_size, self.use_fbss, weights, cross=True,
        )


def main(argv=None):
    mv.main(
        argv,
        beamformer_class=RowDynamicCCMVBeamformerIQ,
        method_name=METHOD_NAME,
        method_label="CC-MV",
        covariance_label="Hasegawa 2020 Eq. (9), Delta_n=1 (including self-pairs)",
        argument_parser=build_parser(),
    )


if __name__ == "__main__":
    main()

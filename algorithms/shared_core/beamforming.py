"""Shared beamforming geometry and numerical kernels."""

import numpy as np
import torch

from shared_core.geometry import aperture_channel_count, element_positions, receive_distance


def aperture_half_width(depth, f_number, n_channels, pitch, dynamic_aperture):
    if dynamic_aperture:
        return depth / (2.0 * f_number)
    return torch.full_like(depth, (n_channels - 1) * pitch / 2.0)


def dynamic_aperture_channel_count(
    depth,
    f_number,
    pitch,
    n_channels,
    dynamic_aperture,
):
    if not dynamic_aperture:
        return n_channels
    count = int(aperture_channel_count(depth, f_number, pitch))
    if count < 1:
        raise ValueError("动态孔径必须包含至少一个物理通道")
    return min(count, n_channels)


def aperture_start_indices(x_positions, n_channels, aperture_size, pitch):
    starts = torch.round(
        x_positions / pitch + (n_channels - aperture_size) / 2
    )
    return starts.long().clamp(0, n_channels - aperture_size)


def _aperture_window(x_norm, window_type, tukey_alpha, kaiser_beta):
    if window_type == "rect":
        return torch.ones_like(x_norm)
    if window_type == "tukey":
        win = torch.ones_like(x_norm)
        transition = x_norm > (1.0 - tukey_alpha)
        val = 0.5 * (
            1.0 + torch.cos(torch.pi * (x_norm - (1.0 - tukey_alpha)) / tukey_alpha)
        )
        win[transition] = val[transition]
        return win
    if window_type == "hann":
        return 0.5 * (1.0 + torch.cos(torch.pi * x_norm))
    if window_type == "hamming":
        return 0.54 + 0.46 * torch.cos(torch.pi * x_norm)
    if window_type == "blackman":
        return (0.42 + 0.5 * torch.cos(torch.pi * x_norm) + 0.08 * torch.cos(2.0 * torch.pi * x_norm)).clamp_min(0)
    if window_type == "kaiser":
        beta = torch.as_tensor(kaiser_beta, dtype=x_norm.dtype, device=x_norm.device)
        arg = beta * torch.sqrt(torch.clamp(1.0 - x_norm.square(), min=0.0))
        return torch.i0(arg) / torch.i0(beta)
    raise ValueError(f"Unsupported window type: {window_type}")


def aperture_window_from_dx(dx, half_a, window_type, tukey_alpha=0.25, kaiser_beta=8.6):
    x_norm = dx.abs() / (half_a + 1e-9)
    in_aperture = (x_norm <= 1.0).float()
    if window_type == "rect":
        return in_aperture
    win = _aperture_window(x_norm, window_type, tukey_alpha, kaiser_beta)
    weighted = torch.where(in_aperture.bool(), win.clamp_min(0), torch.zeros_like(win))
    return torch.where(
        in_aperture.sum(-1, keepdim=True) <= 2,
        in_aperture,
        weighted,
    )


def aperture_window_1d(
    k,
    window_type,
    device,
    dtype=torch.float32,
    tukey_alpha=0.25,
    kaiser_beta=8.6,
):
    if k < 1:
        raise ValueError("aperture size must be positive")
    if window_type == "rect" or k <= 2:
        return torch.ones(k, dtype=dtype, device=device)
    x = torch.linspace(-1.0, 1.0, k, dtype=dtype, device=device)
    return _aperture_window(x.abs(), window_type, tukey_alpha, kaiser_beta)


def optional_aperture_window_1d(k, window_type, device, dtype=torch.float32):
    if window_type == "rect":
        return None
    return aperture_window_1d(k, window_type, device, dtype=dtype)


def build_row_dynamic_geometry(
    z_grid,
    x_grid,
    n_elem,
    pitch,
    c,
    fc,
    fs,
    f_number,
    dynamic_aperture,
    window,
    subarray_ratio,
    temporal_win,
    device,
):
    """Build shared geometry and row caches for MV-family beamformers."""
    if temporal_win < 1 or temporal_win % 2 != 1:
        raise ValueError("temporal_win must be a positive odd integer")
    if not np.isfinite(subarray_ratio) or not 0 < subarray_ratio <= 1:
        raise ValueError("subarray_ratio must be in (0,1]")
    height, width, n_channels = len(z_grid), len(x_grid), n_elem
    x_tensor = torch.from_numpy(x_grid).float().to(device)
    z_tensor = torch.from_numpy(z_grid).float().to(device)
    lateral_channel = element_positions(n_channels, pitch, device=device)
    x_mesh, z_mesh = torch.meshgrid(x_tensor, z_tensor, indexing="xy")
    drs = receive_distance(x_mesh[..., None], z_mesh[..., None], lateral_channel) * (fs / c)
    ch = torch.arange(n_channels, device=device).view(1, n_channels, 1)
    row_cache = []
    for hz in range(height):
        depth = float(z_tensor[hz].item())
        aperture_size = dynamic_aperture_channel_count(
            depth,
            f_number,
            pitch,
            n_channels,
            dynamic_aperture,
        )
        subarray_size = min(max(int(aperture_size * subarray_ratio), 2), aperture_size)
        subarray_count = aperture_size - subarray_size + 1
        idx_start = aperture_start_indices(
            x_tensor, n_channels, aperture_size, pitch
        )
        ch_idx = idx_start.unsqueeze(1) + torch.arange(
            aperture_size, device=device
        ).unsqueeze(0)
        row_cache.append(
            {
                "aperture_size": aperture_size,
                "subarray_size": subarray_size,
                "subarray_count": subarray_count,
                "ch_idx_t": ch_idx.unsqueeze(-1).expand(-1, -1, temporal_win),
                "window": optional_aperture_window_1d(subarray_count, window, device),
                "aperture_window": optional_aperture_window_1d(aperture_size, window, device),
                "eye": torch.eye(
                    subarray_size,
                    dtype=torch.complex64,
                    device=device,
                ).unsqueeze(0),
                "ones": torch.ones(
                    (width, subarray_size, 1),
                    dtype=torch.complex64,
                    device=device,
                ),
            },
        )
    return {
        "height": height,
        "width": width,
        "N": n_channels,
        "x_grid": x_tensor,
        "z_grid": z_tensor,
        "pitch": pitch,
        "fc": float(fc),
        "fs": fs,
        "sc": fs / c,
        "x_mesh": x_mesh,
        "z_mesh": z_mesh,
        "drs": drs,
        "ch": ch,
        "row_cache": row_cache,
    }


def normalized_mv_weights(covariance, steering):
    covariance = 0.5 * (covariance + covariance.mH)
    dtype = covariance.real.dtype
    scale = covariance.abs().amax(dim=(-2, -1), keepdim=True)
    normalized = covariance / scale.clamp_min(torch.finfo(dtype).tiny)
    size = covariance.shape[-1]
    eye = torch.eye(size, dtype=covariance.dtype, device=covariance.device)
    # A relative numerical floor also stabilizes zero user-specified loading.
    normalized = normalized + (10 * size * torch.finfo(dtype).eps) * eye
    solved = torch.linalg.solve(normalized, steering)
    denominator = (steering.mH @ solved).real
    nonzero = denominator.abs() > torch.finfo(dtype).tiny
    safe_denominator = torch.where(nonzero, denominator, torch.ones_like(denominator))
    weights = solved / safe_denominator
    return torch.where((scale > 0) & nonzero, weights, torch.zeros_like(weights))


def forward_backward_covariance(covariance, enabled):
    if enabled:
        return 0.5 * (covariance + covariance.conj().flip(-1, -2))
    return covariance


def spatial_temporal_covariance(
    x_active, subarray_size, temporal_win, use_fbss, spatial_weights=None, normalize=False,
):
    if x_active.shape[-1] != temporal_win:
        raise ValueError("temporal_win must match the snapshot time dimension")
    x_subarrays = x_active.unfold(1, subarray_size, 1).permute(0, 3, 1, 2)
    covariance_subarrays = x_subarrays
    if normalize:
        scale = x_active.abs().amax(dim=(-2, -1), keepdim=True)
        normalized = x_active / scale.clamp_min(torch.finfo(x_active.real.dtype).tiny)
        covariance_subarrays = normalized.unfold(1, subarray_size, 1).permute(0, 3, 1, 2).contiguous()
    snapshots = x_subarrays.shape[2] * temporal_win
    x_snapshots = covariance_subarrays.reshape(x_active.shape[0], subarray_size, snapshots)
    if spatial_weights is not None:
        snapshot_weights = spatial_weights.repeat_interleave(temporal_win)
        x_snapshots = x_snapshots * snapshot_weights.sqrt()
        snapshots = snapshot_weights.sum()
    covariance_forward = torch.matmul(x_snapshots, x_snapshots.mH) / snapshots
    covariance = forward_backward_covariance(covariance_forward, use_fbss)
    return x_subarrays, covariance


def mv_row_output(weights, x_subarrays, x_active, valid_mask, row):
    """Combine subarrays, falling back to valid-channel DAS for incomplete apertures."""
    center = x_subarrays.shape[-1] // 2
    values = x_subarrays[..., center]
    if row["window"] is None:
        mean = values.mean(dim=2)
    else:
        window = row["window"] / row["window"].sum()
        mean = (values * window).sum(dim=2)
    output = (weights.mH @ mean.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    valid = torch.gather(valid_mask[..., center], 1, row["ch_idx_t"][..., center])
    fallback_weights = valid.to(x_active.real.dtype)
    if row["aperture_window"] is not None:
        windowed_weights = fallback_weights * row["aperture_window"]
        windowed_sum = windowed_weights.sum(dim=1, keepdim=True)
        valid_sum = fallback_weights.sum(dim=1, keepdim=True)
        fallback_weights = torch.where(
            windowed_sum > 0,
            windowed_weights / windowed_sum.clamp_min(1e-12),
            fallback_weights / valid_sum.clamp_min(1e-12),
        )
    else:
        fallback_weights = fallback_weights / fallback_weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-12)
    fallback = (x_active[..., center] * fallback_weights).sum(dim=1)
    return torch.where(valid.bool().all(dim=1), output, fallback).view(-1, 1, 1)


def generalized_coherence_factor(data, low_bins):
    return generalized_coherence_factor_nd(data, (low_bins,))


def generalized_coherence_factor_nd(data, low_bins):
    dims = tuple(range(-len(low_bins), 0))
    scale = data.abs().amax(dim=dims, keepdim=True)
    data = data / scale.clamp_min(torch.finfo(data.real.dtype).tiny)
    spectrum = torch.fft.fftshift(torch.fft.fftn(data, dim=dims), dim=dims).abs().square()
    slices = tuple(
        slice(max(0, data.shape[dim] // 2 - bins), min(data.shape[dim], data.shape[dim] // 2 + bins + 1))
        for dim, bins in zip(dims, low_bins)
    )
    low = spectrum[(..., *slices)].sum(dim=dims)
    total = spectrum.sum(dim=dims)
    return (low / total.clamp_min(torch.finfo(total.dtype).tiny)).clamp(0, 1)


def _lagrange_weight(frac, offset, offsets):
    weight = torch.ones_like(frac)
    for other in offsets:
        if other != offset:
            weight = weight * (frac - other) / (offset - other)
    return weight


def _sinc_weight(frac, offset, radius):
    x = frac - offset
    sinc = torch.sinc(x)
    window_arg = x / (radius + 1.0)
    window = 0.5 * (1.0 + torch.cos(torch.pi * window_arg))
    return sinc * window


def interpolation_coefficients(fraction, mode):
    if mode == "linear":
        return (1.0 - fraction, fraction)
    if mode == "catmull_rom":
        t2 = fraction.square()
        t3 = t2 * fraction
        return (-0.5 * fraction + t2 - 0.5 * t3,
                1.0 - 2.5 * t2 + 1.5 * t3,
                0.5 * fraction + 2.0 * t2 - 1.5 * t3,
                -0.5 * t2 + 0.5 * t3)
    raise ValueError(f"Unsupported interpolation coefficient mode: {mode}")


def _interpolate_samples(sample, n_samples, interp, gather):
    sample = sample.clamp(0.0, float(n_samples - 1))

    if interp == "nearest":
        idx = sample.round().long().clamp(0, n_samples - 1)
        return gather(idx)

    if interp == "linear":
        idx0 = sample.floor().long()
        frac = sample - idx0.float()
        idx1 = torch.clamp(idx0 + 1, 0, n_samples - 1)
        idx0 = torch.clamp(idx0, 0, n_samples - 1)
        i0, q0 = gather(idx0)
        i1, q1 = gather(idx1)
        w0, w1 = interpolation_coefficients(frac, "linear")
        i = i0 * w0 + i1 * w1
        q = q0 * w0 + q1 * w1
        return i, q

    if interp == "catmull_rom":
        idx0 = sample.floor().long().clamp(0, n_samples - 2)
        frac = sample - idx0.float()
        weights = interpolation_coefficients(frac, "catmull_rom")
        pairs = [gather((idx0 + offset).clamp(0, n_samples - 1)) for offset in (-1, 0, 1, 2)]
        return tuple(sum(pair[part] * weight for pair, weight in zip(pairs, weights)) for part in (0, 1))

    idx0 = sample.floor().long()
    frac = sample - idx0.float()
    if interp == "cubic":
        offsets = [-1, 0, 1, 2]
    elif interp == "quintic":
        offsets = [-2, -1, 0, 1, 2, 3]
    elif interp in {"farrow", "sinc"}:
        offsets = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    else:
        raise ValueError(f"Unsupported interpolation type: {interp}")

    i_out = torch.zeros_like(sample)
    q_out = torch.zeros_like(sample)
    norm = torch.zeros_like(sample)
    for offset in offsets:
        idx = torch.clamp(idx0 + offset, 0, n_samples - 1)
        if interp == "sinc":
            weight = _sinc_weight(frac, offset, radius=4)
            norm = norm + weight
        else:
            weight = _lagrange_weight(frac, offset, offsets)
        i_part, q_part = gather(idx)
        i_out = i_out + i_part * weight
        q_out = q_out + q_part * weight

    if interp == "sinc":
        i_out = i_out / (norm + 1e-9)
        q_out = q_out / (norm + 1e-9)
    return i_out, q_out


def interpolate_channel_samples(i_data, q_data, sample, channel_index, interp):
    return _interpolate_samples(
        sample,
        i_data.shape[0],
        interp,
        lambda index: (i_data[index, channel_index], q_data[index, channel_index]),
    )


def interpolate_multi_angle_channel_samples(
    i_data,
    q_data,
    sample,
    angle_index,
    channel_index,
    interp,
):
    n_samples = i_data.shape[1]

    def gather(idx):
        idx = torch.clamp(idx, 0, n_samples - 1)
        return i_data[angle_index, idx, channel_index], q_data[
            angle_index,
            idx,
            channel_index,
        ]

    return _interpolate_samples(sample, n_samples, interp, gather)

"""Shared beamforming geometry and numerical kernels."""

import numpy as np
import torch


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
    count = int(depth / (f_number * pitch)) + 1
    if count < 1:
        raise ValueError("动态孔径必须包含至少一个物理通道")
    return min(count, n_channels)


def aperture_start_indices(x_positions, n_channels, aperture_size, pitch):
    starts = torch.round(
        x_positions / pitch + (n_channels - aperture_size) / 2
    )
    return starts.long().clamp(0, n_channels - aperture_size)


def aperture_window_from_dx(dx, half_a, window_type, tukey_alpha=0.25, kaiser_beta=8.6):
    x_norm = dx.abs() / (half_a + 1e-9)
    in_aperture = (x_norm <= 1.0).float()
    if window_type == "rect":
        return in_aperture

    if window_type == "tukey":
        win = torch.ones_like(x_norm)
        transition = (x_norm > (1.0 - tukey_alpha)) & (x_norm <= 1.0)
        val = 0.5 * (
            1.0 + torch.cos(torch.pi * (x_norm - (1.0 - tukey_alpha)) / tukey_alpha)
        )
        win[transition] = val[transition]
        win[x_norm > 1.0] = 0.0
        return torch.where(
            in_aperture.sum(-1, keepdim=True) <= 2,
            in_aperture,
            win,
        )

    if window_type == "hann":
        win = 0.5 * (1.0 + torch.cos(torch.pi * x_norm))
    elif window_type == "hamming":
        win = 0.54 + 0.46 * torch.cos(torch.pi * x_norm)
    elif window_type == "blackman":
        win = (
            0.42
            + 0.5 * torch.cos(torch.pi * x_norm)
            + 0.08 * torch.cos(2.0 * torch.pi * x_norm)
        )
    elif window_type == "kaiser":
        beta = torch.as_tensor(kaiser_beta, dtype=dx.dtype, device=dx.device)
        arg = beta * torch.sqrt(torch.clamp(1.0 - x_norm.square(), min=0.0))
        win = torch.i0(arg) / torch.i0(beta)
    else:
        raise ValueError(f"Unsupported window type: {window_type}")
    weighted = win.clamp_min(0) * in_aperture
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
    ax = x.abs()
    if window_type == "tukey":
        win = torch.ones_like(x)
        edge = ax > 1.0 - tukey_alpha
        win[edge] = 0.5 * (
            1.0 + torch.cos(torch.pi * (ax[edge] - (1.0 - tukey_alpha)) / tukey_alpha)
        )
        return win
    if window_type == "hann":
        return 0.5 * (1.0 + torch.cos(torch.pi * x))
    if window_type == "hamming":
        return 0.54 + 0.46 * torch.cos(torch.pi * x)
    if window_type == "blackman":
        return (
            0.42 + 0.5 * torch.cos(torch.pi * x) + 0.08 * torch.cos(2.0 * torch.pi * x)
        ).clamp_min(0)
    if window_type == "kaiser":
        beta = torch.as_tensor(kaiser_beta, dtype=dtype, device=device)
        arg = beta * torch.sqrt(torch.clamp(1.0 - ax.square(), min=0.0))
        return torch.i0(arg) / torch.i0(beta)
    raise ValueError(f"Unsupported window type: {window_type}")


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
    lateral_channel = (
        torch.arange(n_channels, device=device).float() - (n_channels - 1) / 2.0
    ) * pitch
    x_mesh, z_mesh = torch.meshgrid(x_tensor, z_tensor, indexing="xy")
    drs = torch.sqrt(
        (x_mesh[..., None] - lateral_channel) ** 2 + z_mesh[..., None] ** 2
    ) * (fs / c)
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
    if use_fbss:
        covariance = 0.5 * (
            covariance_forward + covariance_forward.conj().flip(-1, -2)
        )
    else:
        covariance = covariance_forward
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
    scale = data.abs().amax(dim=-1, keepdim=True)
    data = data / scale.clamp_min(torch.finfo(data.real.dtype).tiny)
    spectrum = torch.fft.fftshift(torch.fft.fft(data, dim=-1), dim=-1).abs().square()
    center = data.shape[-1] // 2
    low = spectrum[..., max(0, center - low_bins):min(data.shape[-1], center + low_bins + 1)].sum(dim=-1)
    total = spectrum.sum(dim=-1)
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
        i = i0 * (1.0 - frac) + i1 * frac
        q = q0 * (1.0 - frac) + q1 * frac
        return i, q

    if interp == "catmull_rom":
        idx0 = sample.floor().long().clamp(0, n_samples - 2)
        frac = sample - idx0.float()
        weights = (
            -.5 * frac**3 + frac**2 - .5 * frac,
            1.5 * frac**3 - 2.5 * frac**2 + 1,
            -1.5 * frac**3 + 2 * frac**2 + .5 * frac,
            .5 * frac**3 - .5 * frac**2,
        )
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

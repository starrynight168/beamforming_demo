import numpy as np
import os
import torch


WINDOW_CHOICES = ["rect", "tukey", "hann", "hamming", "blackman", "kaiser"]
INTERP_CHOICES = ["nearest", "linear", "cubic", "quintic", "farrow", "sinc"]


def resolve_project_path(path, project_root):
    if os.path.isabs(path):
        return path
    return os.path.join(project_root, path)


def parse_selected_angles(angles, select_str):
    select_str = str(select_str).strip().lower()
    if select_str == "all":
        return np.arange(len(angles)), angles
    if select_str == "center":
        center_idx = int(np.argmin(np.abs(angles)))
        return np.array([center_idx]), np.array([angles[center_idx]])
    if select_str.isdigit():
        count = max(1, min(int(select_str), len(angles)))
        if count == 1:
            center_idx = int(np.argmin(np.abs(angles)))
            return np.array([center_idx]), np.array([angles[center_idx]])
        sorted_indices = np.argsort(angles)
        selected = np.linspace(0, len(angles) - 1, count, dtype=int)
        indices = sorted_indices[selected]
        return indices, angles[indices]
    indices = [int(item) for item in select_str.split(",")]
    indices = [idx for idx in indices if 0 <= idx < len(angles)]
    return np.array(indices), angles[indices]


def aperture_half_width(depth, f_number, n_channels, pitch, dynamic_aperture):
    if dynamic_aperture:
        return depth / (2.0 * f_number)
    return torch.full_like(depth, (n_channels - 1) * pitch / 2.0)


def dynamic_aperture_channel_count(depth, f_number, pitch, n_channels, dynamic_aperture, min_channels=4):
    if not dynamic_aperture:
        return n_channels
    return min(max(int(depth / (f_number * pitch)) + 1, min_channels), n_channels)


def apply_tgc_image(env, z_grid, fc, tgc_alpha):
    return env * tgc_gain(z_grid, fc, tgc_alpha)[:, None]


def tgc_gain(z_grid, fc, tgc_alpha):
    return 10 ** (tgc_alpha * (fc / 1e6) * (z_grid * 100) * 2.0 / 20.0)


def aperture_window_from_dx(dx, half_a, window_type, tukey_alpha=0.25, kaiser_beta=8.6):
    x_norm = dx.abs() / (half_a + 1e-9)
    in_aperture = (x_norm <= 1.0).float()
    if window_type == "rect":
        return in_aperture

    if window_type == "tukey":
        win = torch.ones_like(x_norm)
        transition = (x_norm > (1.0 - tukey_alpha)) & (x_norm <= 1.0)
        val = 0.5 * (1.0 + torch.cos(torch.pi * (x_norm - (1.0 - tukey_alpha)) / tukey_alpha))
        win[transition] = val[transition]
        win[x_norm > 1.0] = 0.0
        return win

    if window_type == "hann":
        win = 0.5 * (1.0 + torch.cos(torch.pi * x_norm))
    elif window_type == "hamming":
        win = 0.54 + 0.46 * torch.cos(torch.pi * x_norm)
    elif window_type == "blackman":
        win = 0.42 + 0.5 * torch.cos(torch.pi * x_norm) + 0.08 * torch.cos(2.0 * torch.pi * x_norm)
    elif window_type == "kaiser":
        beta = torch.as_tensor(kaiser_beta, dtype=dx.dtype, device=dx.device)
        arg = beta * torch.sqrt(torch.clamp(1.0 - x_norm.square(), min=0.0))
        win = torch.i0(arg) / torch.i0(beta)
    else:
        raise ValueError(f"Unsupported window type: {window_type}")
    return win * in_aperture


def aperture_window_1d(k, window_type, device, dtype=torch.float32, tukey_alpha=0.25, kaiser_beta=8.6):
    if window_type == "rect":
        return torch.ones(k, dtype=dtype, device=device)
    x = torch.linspace(-1.0, 1.0, k, dtype=dtype, device=device)
    ax = x.abs()
    if window_type == "tukey":
        win = torch.ones_like(x)
        edge = ax > 1.0 - tukey_alpha
        win[edge] = 0.5 * (1.0 + torch.cos(torch.pi * (ax[edge] - (1.0 - tukey_alpha)) / tukey_alpha))
        return win
    if window_type == "hann":
        return 0.5 * (1.0 + torch.cos(torch.pi * x))
    if window_type == "hamming":
        return 0.54 + 0.46 * torch.cos(torch.pi * x)
    if window_type == "blackman":
        return 0.42 + 0.5 * torch.cos(torch.pi * x) + 0.08 * torch.cos(2.0 * torch.pi * x)
    if window_type == "kaiser":
        beta = torch.as_tensor(kaiser_beta, dtype=dtype, device=device)
        arg = beta * torch.sqrt(torch.clamp(1.0 - ax.square(), min=0.0))
        return torch.i0(arg) / torch.i0(beta)
    raise ValueError(f"Unsupported window type: {window_type}")


def optional_aperture_window_1d(k, window_type, device, dtype=torch.float32):
    if window_type == "rect":
        return None
    return aperture_window_1d(k, window_type, device, dtype=dtype)


def db_display_range(dynamic_range):
    return -float(dynamic_range), 0.0


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


def interpolate_channel_samples(i_data, q_data, sample, channel_index, interp):
    n_samples = i_data.shape[0]
    sample = sample.clamp(0.0, float(n_samples - 1))

    if interp == "nearest":
        idx = sample.round().long().clamp(0, n_samples - 1)
        return i_data[idx, channel_index], q_data[idx, channel_index]

    if interp == "linear":
        idx0 = sample.floor().long()
        frac = sample - idx0.float()
        idx1 = torch.clamp(idx0 + 1, 0, n_samples - 1)
        idx0 = torch.clamp(idx0, 0, n_samples - 1)
        i = i_data[idx0, channel_index] * (1.0 - frac) + i_data[idx1, channel_index] * frac
        q = q_data[idx0, channel_index] * (1.0 - frac) + q_data[idx1, channel_index] * frac
        return i, q

    idx0 = sample.floor().long()
    frac = sample - idx0.float()
    if interp == "cubic":
        offsets = [-1, 0, 1, 2]
    elif interp == "quintic":
        offsets = [-2, -1, 0, 1, 2, 3]
    elif interp == "farrow":
        offsets = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    elif interp == "sinc":
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
        i_out = i_out + i_data[idx, channel_index] * weight
        q_out = q_out + q_data[idx, channel_index] * weight

    if interp == "sinc":
        i_out = i_out / (norm + 1e-9)
        q_out = q_out / (norm + 1e-9)
    return i_out, q_out


def interpolate_multi_angle_channel_samples(i_data, q_data, sample, angle_index, channel_index, interp):
    n_samples = i_data.shape[1]
    sample = sample.clamp(0.0, float(n_samples - 1))

    def gather(idx):
        idx = torch.clamp(idx, 0, n_samples - 1)
        return i_data[angle_index, idx, channel_index], q_data[angle_index, idx, channel_index]

    if interp == "nearest":
        return gather(sample.round().long())

    if interp == "linear":
        idx0 = sample.floor().long()
        frac = sample - idx0.float()
        i0, q0 = gather(idx0)
        i1, q1 = gather(idx0 + 1)
        return i0 * (1.0 - frac) + i1 * frac, q0 * (1.0 - frac) + q1 * frac

    idx0 = sample.floor().long()
    frac = sample - idx0.float()
    if interp == "cubic":
        offsets = [-1, 0, 1, 2]
    elif interp == "quintic":
        offsets = [-2, -1, 0, 1, 2, 3]
    elif interp == "farrow":
        offsets = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    elif interp == "sinc":
        offsets = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    else:
        raise ValueError(f"Unsupported interpolation type: {interp}")

    i_out = torch.zeros_like(sample)
    q_out = torch.zeros_like(sample)
    norm = torch.zeros_like(sample)
    for offset in offsets:
        if interp == "sinc":
            weight = _sinc_weight(frac, offset, radius=4)
            norm = norm + weight
        else:
            weight = _lagrange_weight(frac, offset, offsets)
        i_part, q_part = gather(idx0 + offset)
        i_out = i_out + i_part * weight
        q_out = q_out + q_part * weight

    if interp == "sinc":
        i_out = i_out / (norm + 1e-9)
        q_out = q_out / (norm + 1e-9)
    return i_out, q_out

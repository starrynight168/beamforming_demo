import numpy as np
import torch


def receive_distance(lateral, depth, element):
    return torch.sqrt((lateral - element) ** 2 + depth ** 2)


def element_positions(n_channels, pitch, *, device=None, dtype=torch.float32):
    """Lateral position of each array element relative to the array centre."""
    positions = torch.arange(n_channels, device=device, dtype=dtype)
    return (positions - (n_channels - 1) / 2.0) * pitch


def plane_wave_distance(lateral, depth, cos_angle, sin_angle):
    return depth * cos_angle + lateral * sin_angle


def aperture_channel_count(depth, f_number, pitch):
    """Discrete dynamic-aperture channel count ``floor(depth / (F * pitch)) + 1``.

    NumPy scalars/arrays and torch tensors are both accepted so the classical
    beamformers, training, and diagnostics share one aperture-size rule.
    """
    ratio = depth / (f_number * pitch)
    floor = getattr(ratio, "floor", None)
    return (floor() if callable(floor) else np.floor(ratio)) + 1


def image_extent_mm(x_grid, z_grid):
    """``imshow`` extent ``(x_min, x_max, z_max, z_min)`` in millimetres.

    Grids are stored in metres; this is the single conversion used by every
    reconstruction preview and figure panel.
    """
    x = np.asarray(x_grid, dtype=np.float64).reshape(-1)
    z = np.asarray(z_grid, dtype=np.float64).reshape(-1)
    if x.size == 0 or z.size == 0:
        raise ValueError("image extent requires nonempty x/z grids")
    if not np.isfinite(x).all() or not np.isfinite(z).all():
        raise ValueError("image extent requires finite x/z grids")
    return [
        float(x[0] * 1000.0),
        float(x[-1] * 1000.0),
        float(z[-1] * 1000.0),
        float(z[0] * 1000.0),
    ]

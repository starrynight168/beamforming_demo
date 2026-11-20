"""Angle-selection primitives shared by generation, algorithms, and training.

Every entry point resolves ``all``/``center``/count selections through
:func:`select_angle_indices` so that a change to the sampling rule (for example
the endpoint-inclusive ``linspace`` convention) applies everywhere at once.
"""

import numpy as np


def select_angle_indices(angles, selection):
    """Return angle indices for ``all``, ``center``, or a positive count.

    ``all`` keeps acquisition order. ``center`` and a count of one select the
    acquisition closest to zero. Larger counts sample the sorted angle axis
    uniformly with both endpoints included.
    """
    values = np.asarray(angles, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("angles 不能为空")
    if isinstance(selection, str):
        mode = selection.strip().lower()
        if mode == "all":
            return np.arange(values.size, dtype=np.int64)
        if mode == "center":
            return np.asarray([int(np.argmin(np.abs(values)))], dtype=np.int64)
        if not mode.isdigit():
            raise ValueError(f"不支持的角度选择: {selection!r}")
        selection = int(mode)
    if isinstance(selection, (bool, np.bool_)):
        raise ValueError(f"角度选择必须为正整数: {selection!r}")
    if not isinstance(selection, (int, np.integer)):
        raise ValueError(f"角度选择必须是 all、center 或正整数: {selection!r}")
    count = int(selection)
    if count < 1:
        raise ValueError(f"角度数必须是正整数: {count}")
    if count > values.size:
        raise ValueError(f"角度数 {count} 超出范围 [1, {values.size}]")
    if count == 1:
        return np.asarray([int(np.argmin(np.abs(values)))], dtype=np.int64)
    order = np.argsort(values)
    return order[np.linspace(0, values.size - 1, count, dtype=int)].astype(np.int64)

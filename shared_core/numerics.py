import numpy as np


def component_magnitude(real, imag):
    try:
        import torch

        if torch.is_tensor(real):
            return torch.hypot(real, imag)
    except ImportError:
        pass
    return np.hypot(np.asarray(real), np.asarray(imag))

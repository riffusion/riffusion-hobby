import warnings

import numpy as np
import torch


def check_device(device: str, backup: str = "cpu") -> str:
    """
    Check that the device is valid and available. If not,
    """
    cuda_not_found = device.lower().startswith("cuda") and not torch.cuda.is_available()
    mps_not_found = device.lower().startswith("mps") and not torch.backends.mps.is_available()

    if cuda_not_found or mps_not_found:
        warnings.warn(f"WARNING: {device} is not available, using {backup} instead.", stacklevel=3)
        return backup

    return device


def slerp(
    t: float, v0: torch.Tensor, v1: torch.Tensor, dot_threshold: float = 0.9995
) -> torch.Tensor:
    """
    Helper function to spherically interpolate two arrays v1 v2.
    """
    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > dot_threshold:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2

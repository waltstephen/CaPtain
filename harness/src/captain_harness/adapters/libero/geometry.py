from __future__ import annotations

import math

import numpy as np


class DepthUnavailableError(RuntimeError):
    pass


def pixel_to_world(
    depth_m: np.ndarray,
    intrinsics_3x3: np.ndarray,
    camera_to_base_4x4: np.ndarray,
    *,
    x_norm: float,
    y_norm: float,
) -> tuple[float, float, float]:
    """Back-project one normalized image point using a finite 5x5 depth median."""
    if not math.isfinite(x_norm) or not math.isfinite(y_norm):
        raise ValueError("normalized image coordinates must be finite")
    if not 0.0 <= x_norm <= 1.0 or not 0.0 <= y_norm <= 1.0:
        raise ValueError("normalized image coordinates must be in [0, 1]")
    depth = np.asarray(depth_m, dtype=np.float64).squeeze()
    if depth.ndim != 2 or min(depth.shape) == 0:
        raise ValueError("depth must be a non-empty 2D array")
    intrinsics = np.asarray(intrinsics_3x3, dtype=np.float64).reshape(3, 3)
    camera_to_base = np.asarray(camera_to_base_4x4, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(intrinsics).all() or not np.isfinite(camera_to_base).all():
        raise ValueError("camera calibration must be finite")

    height, width = depth.shape
    x_pixel = int(round(x_norm * (width - 1)))
    y_pixel = int(round(y_norm * (height - 1)))
    window_width = min(5, width)
    window_height = min(5, height)
    x_start = min(max(0, x_pixel - 2), width - window_width)
    y_start = min(max(0, y_pixel - 2), height - window_height)
    x_stop = x_start + window_width
    y_stop = y_start + window_height
    patch = depth[y_start:y_stop, x_start:x_stop]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size == 0:
        raise DepthUnavailableError("no finite positive depth in the 5x5 image patch")
    depth_value = float(np.median(valid))

    pixel_h = np.array([float(x_pixel), float(y_pixel), 1.0])
    camera_ray = np.linalg.solve(intrinsics, pixel_h)
    camera_point = camera_ray * depth_value
    base_point = camera_to_base @ np.concatenate([camera_point, [1.0]])
    if not np.isfinite(base_point[:3]).all():
        raise DepthUnavailableError("back-projected point is not finite")
    return tuple(float(value) for value in base_point[:3])

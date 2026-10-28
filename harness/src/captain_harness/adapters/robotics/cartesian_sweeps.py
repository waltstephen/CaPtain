"""Task-independent line and pivoted-rotation setpoints in a declared frame.

These are geometric setpoints, not a collision-free plan or a contact claim.
Measured execution and force limits belong to the robot driver.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.spatial.transform import Rotation


def _vector(value, name):
    result = np.asarray(value, dtype=float).copy()
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(name + ' must be a finite three-vector')
    return result


def cartesian_sweep(start_pose, *, displacement_m=(0., 0., 0.),
                    axis=None, pivot_m=None, angle_rad=0.,
                    max_position_step_m=.004, max_orientation_step_rad=.02):
    """Rotate a full pose about a base-frame axis/pivot, then translate it.

    Both operations progress together. The caller supplies all geometry;
    no mechanism axis is inferred from task labels or environment state.
    """
    start = np.asarray(start_pose, dtype=float)
    if (start.shape != (4, 4) or not np.isfinite(start).all()
            or not np.allclose(start[3], [0., 0., 0., 1.], rtol=0., atol=1e-8)
            or not np.allclose(start[:3, :3].T @ start[:3, :3], np.eye(3), rtol=0., atol=1e-8)
            or not np.isclose(np.linalg.det(start[:3, :3]), 1., rtol=0., atol=1e-8)):
        raise ValueError('start_pose must be a finite rigid transform')
    displacement = _vector(displacement_m, 'displacement_m')
    if (not math.isfinite(angle_rad) or not math.isfinite(max_position_step_m)
            or not math.isfinite(max_orientation_step_rad)
            or max_position_step_m <= 0 or max_orientation_step_rad <= 0):
        raise ValueError('angle and positive setpoint resolutions must be finite')
    if axis is None:
        if angle_rad != 0 or pivot_m is not None:
            raise ValueError('a rotation requires an explicit axis and pivot')
        direction, pivot = np.zeros(3), start[:3, 3]
    else:
        direction = _vector(axis, 'axis')
        norm = float(np.linalg.norm(direction))
        if norm < 1e-10 or pivot_m is None:
            raise ValueError('a rotation requires a nonzero axis and explicit pivot')
        direction /= norm
        pivot = _vector(pivot_m, 'pivot_m')
    offset = start[:3, 3] - pivot
    radius = float(np.linalg.norm(offset - direction * float(offset @ direction)))
    travel_bound = float(np.linalg.norm(displacement)) + abs(angle_rad) * radius
    segments = max(1, int(math.ceil(travel_bound / max_position_step_m)),
                   int(math.ceil(abs(angle_rad) / max_orientation_step_rad)))
    if segments > 10000:
        raise ValueError('requested sweep exceeds the bounded setpoint count')
    path = []
    for fraction in np.linspace(0., 1., segments + 1):
        rotation = Rotation.from_rotvec(direction * (fraction * angle_rad)).as_matrix()
        pose = np.eye(4)
        pose[:3, :3] = rotation @ start[:3, :3]
        pose[:3, 3] = pivot + rotation @ offset + fraction * displacement
        path.append(pose)
    return np.asarray(path)

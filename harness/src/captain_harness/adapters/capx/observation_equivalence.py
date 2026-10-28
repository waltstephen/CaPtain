"""Prove exact public-snapshot equivalence without changing artifact identity."""
from __future__ import annotations

import math
import numpy as np


def same_public_snapshot(first, second) -> bool:
    """Alias geometry only at the same known time with identical sensor inputs.

    A repeated capture in a paused simulator may receive a new artifact handle.
    This is not permission to reuse a pose after time, camera, or robot motion.
    """
    if first.time_s is None or second.time_s is None:
        return False
    if not (math.isfinite(first.time_s) and math.isfinite(second.time_s)) or first.time_s != second.time_s:
        return False
    for field in ('joint_positions_rad', 'gripper_open_fraction',
                  'end_effector_position_m', 'end_effector_quaternion_wxyz',
                  'joint_velocity_norm_rad_s'):
        a, b = getattr(first.robot_state, field, None), getattr(second.robot_state, field, None)
        if not np.array_equal(a, b):
            return False
    left = {camera.camera_name: camera for camera in first.cameras}
    right = {camera.camera_name: camera for camera in second.cameras}
    if not left or left.keys() != right.keys() or len(left) != len(first.cameras) or len(right) != len(second.cameras):
        return False
    for name, a in left.items():
        b = right[name]
        for field in ('image', 'depth', 'intrinsics', 'extrinsics'):
            if not np.array_equal(getattr(a, field), getattr(b, field)):
                return False
    return True

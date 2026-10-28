"""Validate an external IK proposal with the independent robot hardware model."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def validate_returned_ik(model, joints, target_eef_pose, *, aperture,
                         position_tolerance_m, orientation_tolerance_rad):
    joints = np.asarray(joints, dtype=float)
    limits = np.asarray(model.limits, dtype=float)
    if (joints.ndim != 1 or limits.shape != (len(joints), 2)
            or not np.isfinite(joints).all() or not np.isfinite(limits).all()):
        raise ValueError('returned IK and robot limits must have matching finite dimensions')
    actual = model.eef_pose(joints, aperture=aperture)
    goal = np.asarray(target_eef_pose, dtype=float)
    position_error = float(np.linalg.norm(actual[:3, 3] - goal[:3, 3]))
    orientation_error = float(Rotation.from_matrix(goal[:3, :3] @ actual[:3, :3].T).magnitude())
    within_limits = bool(np.all(joints >= limits[:, 0]) and np.all(joints <= limits[:, 1]))
    valid = (within_limits and position_error <= position_tolerance_m
             and orientation_error <= orientation_tolerance_rad)
    return {'valid': valid,
        'reason': 'returned_ik_within_declared_limits' if valid else 'returned_ik_outside_declared_limits',
        'position_residual_m': position_error, 'orientation_residual_rad': orientation_error,
        'position_tolerance_m': position_tolerance_m, 'orientation_tolerance_rad': orientation_tolerance_rad,
        'joint_limits_satisfied': within_limits, 'check_scope': 'returned_endpoint_fk_and_joint_limits',
        'motion_path': 'not_checked',
        'limitation': 'Rejecting this numerical proposal does not prove physical unreachability. Endpoint accuracy does not establish collision clearance.'}

"""Bounded endpoint suggestions with explicit orientation freedom.

This unregistered component only consults a detached robot model. It neither
certifies paths nor executes an endpoint that happens to have an IK solution.
"""
from copy import deepcopy

import numpy as np

from captain_harness.adapters.capx.ik_validation import validate_returned_ik


def solve_pose_candidates(model, *, reference_position_m, eef_rotations,
                          reference_offset_in_eef_m, seed, aperture,
                          position_tolerance_m, orientation_tolerance_rad):
    def vector(value, count, name):
        result = np.asarray(value, dtype=float)
        if result.shape != (count,) or not np.isfinite(result).all():
            raise ValueError(f'{name} must be a finite {count}-vector')
        return result.copy()

    position = vector(reference_position_m, 3, 'reference position')
    offset = vector(reference_offset_in_eef_m, 3, 'EEF reference offset')
    limits = np.asarray(model.limits, dtype=float)
    if (limits.ndim != 2 or limits.shape[1] != 2 or not len(limits)
            or not np.isfinite(limits).all() or np.any(limits[:, 0] > limits[:, 1])):
        raise ValueError('model joint limits must be finite ordered pairs')
    measured = vector(seed, len(limits), 'robot seed')
    if np.any(measured < limits[:, 0]) or np.any(measured > limits[:, 1]):
        raise ValueError('robot seed must be within joint limits')
    if not np.isfinite(aperture) or not 0 <= aperture <= 1:
        raise ValueError('aperture must be finite and in [0, 1]')
    if any(not np.isfinite(t) or t <= 0 for t in
           (position_tolerance_m, orientation_tolerance_rad)):
        raise ValueError('endpoint tolerances must be finite and positive')
    rotations = np.asarray(eef_rotations, dtype=float)
    if (rotations.ndim != 3 or rotations.shape[1:] != (3, 3)
            or not 1 <= len(rotations) <= 12 or not np.isfinite(rotations).all()):
        raise ValueError('provide between 1 and 12 finite rotation matrices')
    for rotation in rotations:
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0)
                or not np.isclose(np.linalg.det(rotation), 1., atol=1e-7, rtol=0)):
            raise ValueError('candidate orientations must be proper rotations')

    entries = []
    for index, rotation in enumerate(rotations):
        target = np.eye(4)
        target[:3, :3] = rotation
        target[:3, 3] = position - rotation @ offset
        entry = {'candidate_index': index, 'target_eef_pose': target.tolist(),
                 'endpoint_valid': False, 'joints': None, 'joint_travel_rad': None,
                 'reason': 'no_numerical_solution'}
        entries.append(entry)
        try:
            reply = model.solve_ik(target.copy(), measured.copy(), aperture=aperture)
            entry['ik'] = deepcopy(getattr(model, 'last_ik', {}))
            if reply is None:
                continue
            checked = validate_returned_ik(model, reply, target, aperture=aperture,
                position_tolerance_m=position_tolerance_m,
                orientation_tolerance_rad=orientation_tolerance_rad)
            entry['validation'] = checked
            entry['reason'] = checked['reason']
            if checked['valid']:
                joints = np.asarray(reply, dtype=float)
                entry.update(endpoint_valid=True, joints=joints.tolist(),
                    joint_travel_rad=float(np.sum(np.abs(joints - measured))))
        except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
            entry.update(reason='invalid_numerical_reply', diagnostic=str(exc))

    ranked = sorted((entry for entry in entries if entry['endpoint_valid']),
                    key=lambda entry: (entry['joint_travel_rad'], entry['candidate_index']))
    return {'candidates': entries,
            'ranked_candidate_indices': [entry['candidate_index'] for entry in ranked],
            'reference_position_m': position.tolist(),
            'reference_offset_in_eef_m': offset.tolist(),
            'motion_path': 'not_checked', 'carried_object_collision': 'not_checked',
            'object_retention': 'not_checked', 'task_semantics': 'not_checked',
            'scope': 'endpoint_fk_and_joint_limits_only',
            'limitation': 'No numerical solution is not proof of physical unreachability. '
                          'Joint travel ranks endpoints, not path safety or grasp stability.'}

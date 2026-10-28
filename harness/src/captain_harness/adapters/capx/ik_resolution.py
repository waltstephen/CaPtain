"""Bounded IK recovery from unusable numerical replies.

No service calls or robot actions happen here. Callers must obtain a fresh
robot seed and distinguish transport failures from returned numerical data.
"""
from copy import deepcopy

import numpy as np

from .ik_validation import validate_returned_ik


def resolve_external_ik(model, proposal, goal, *, seed, aperture,
                        position_tolerance_m, orientation_tolerance_rad):
    evidence = {'motion_path': 'not_checked', 'check_scope': 'returned_endpoint_fk_and_joint_limits',
        'limitation': 'Failure of numerical IK is not proof of physical unreachability. No collision path was checked.'}
    def validate(joints):
        return validate_returned_ik(model, joints, goal, aperture=aperture,
            position_tolerance_m=position_tolerance_m,
            orientation_tolerance_rad=orientation_tolerance_rad)
    configuration = None
    if proposal is None:
        evidence.update(external_status='empty_reply', reason='external_ik_empty_reply')
    else:
        try:
            configuration = np.asarray(proposal, dtype=float)
            if configuration.shape != (8,) or not np.isfinite(configuration).all():
                raise ValueError('external IK configuration must contain eight finite values')
            checked = validate(configuration[:7])
            evidence.update(checked, external_status='validated_reply' if checked['valid'] else 'invalid_endpoint')
            if checked['valid']:
                return {'configuration': configuration.tolist(), 'used_fallback': False, 'evidence': evidence}
        except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
            evidence.update(external_status='malformed_reply', reason='external_ik_malformed_reply',
                external_diagnostic=str(exc))
    try:
        current = np.asarray(seed, dtype=float)
        seed_valid = (current.shape == (7,) and np.isfinite(current).all()
                      and np.isfinite(aperture) and 0 <= aperture <= 1)
    except (ValueError, TypeError):
        seed_valid = False
    if not seed_valid:
        evidence['recovery_status'] = 'fresh_robot_seed_unavailable'
    elif not callable(getattr(model, 'solve_ik', None)):
        evidence['recovery_status'] = 'robot_model_solver_unavailable'
    else:
        try:
            recovered = model.solve_ik(goal, current, aperture=aperture)
            evidence['independent_ik'] = deepcopy(getattr(model, 'last_ik', {}))
            evidence['recovery_status'] = 'no_local_solution'
            if recovered is not None:
                checked = validate(recovered)
                evidence['independent_validation'] = checked
                evidence['recovery_status'] = 'validated_endpoint' if checked['valid'] else 'invalid_local_endpoint'
                if checked['valid']:
                    return {'configuration': [*map(float, recovered), float(aperture)*.04],
                        'used_fallback': True, 'evidence': evidence}
        except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
            evidence.update(recovery_status='invalid_local_reply', recovery_diagnostic=str(exc))
    return {'configuration': None, 'used_fallback': False, 'evidence': evidence}

"""Bounded path simplification candidate; not wired into the running R6 planner.

Retains only shortcuts checked at the planner's declared joint resolution.
Sampled validity is not a continuous collision or physical tracking guarantee.
"""
import numpy as np

from .planning import interpolate_path


def shortcut_path(path, valid, *, max_attempts=64, max_checks=20000,
                  max_joint_step=.025, optimizer_seed=0):
    original = np.asarray(path, dtype=float)
    if (original.ndim != 2 or len(original) < 2 or len(original) > 10000
            or original.shape[1] < 1 or not np.isfinite(original).all()
            or not np.isfinite(max_joint_step) or max_joint_step <= 0
            or type(max_attempts) is not int or not 0 <= max_attempts <= 256
            or type(max_checks) is not int or not 1 <= max_checks <= 100000):
        raise ValueError('invalid finite path, resolution or bounded optimization budget')
    dense = interpolate_path(original, max_joint_step)
    checks, accepted, attempts = 0, 0, 0
    def checked(q):
        nonlocal checks
        checks += 1
        return bool(valid(q))
    if len(dense) > max_checks:
        return None, {'reason': 'input_validation_budget_exhausted', 'checks': 0}
    if not all(checked(q) for q in dense):
        return None, {'reason': 'input_path_invalid', 'checks': checks}
    length = lambda p: float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    before = length(dense)
    rng = np.random.default_rng(optimizer_seed)
    reason = 'attempt_budget_completed'
    for attempts in range(1, max_attempts + 1):
        i, j = sorted(rng.choice(len(dense), size=2, replace=False))
        if j <= i + 1 or np.linalg.norm(dense[j] - dense[i]) >= length(dense[i:j+1]) - 1e-9:
            continue
        edge = interpolate_path([dense[i], dense[j]], max_joint_step)
        if checks + len(edge) > max_checks:
            reason = 'shortcut_check_budget_exhausted'
            break
        if all(checked(q) for q in edge):
            dense = np.concatenate([dense[:i], edge, dense[j+1:]])
            accepted += 1
    return dense, {'reason': reason, 'checks': checks, 'attempts': attempts,
        'accepted_shortcuts': accepted, 'original_length_rad': before,
        'shortened_length_rad': length(dense), 'waypoints': len(dense),
        'max_joint_step_rad': max_joint_step, 'optimizer_seed': optimizer_seed,
        'limitation': 'Only sampled states were checked; execution remains untested.'}

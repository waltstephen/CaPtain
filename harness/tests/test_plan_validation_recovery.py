import numpy as np
import pytest
from captain_harness.adapters.robotics import planning


@pytest.mark.parametrize('recover', [True, False])
def test_invalid_resampled_path_is_never_returned_and_retries_are_bounded(monkeypatch, recover):
    start, goal = [-.8, 0], [.8, 0]
    limits = [[-1, 1], [-1, 1]]
    valid = lambda q: not (abs(q[0]) < .2 and q[1] < .5)
    calls = []
    def candidate(start, goal, limits, checked, *, timeout_s, max_joint_step):
        calls.append(timeout_s)
        if recover and len(calls) == 2:
            return np.array([start, [-.8,.8], [.8,.8], goal])
        return np.array([start, goal])
    monkeypatch.setattr(planning, '_rrt_candidate', candidate)
    path, info = planning.plan_joint_path(start, goal, limits, valid, timeout_s=1.)
    assert len(calls) == (2 if recover else 3)
    assert calls == sorted(calls, reverse=True)
    assert all(0 < t <= 1. for t in calls)
    assert info['rejected_candidate_paths'] == (1 if recover else 3)
    if recover:
        assert path is not None and all(valid(q) for q in path)
        assert np.max(np.abs(np.diff(path, axis=0))) <= .025 + 1e-9
    else:
        assert path is None and info['reason'] == 'dense_path_validation_failed'


def test_exhausted_budget_never_starts_another_solver(monkeypatch):
    now = [0.]
    monkeypatch.setattr(planning.time, 'monotonic', lambda: now[0])
    calls = []
    def candidate(start, goal, *args, **kwargs):
        calls.append(kwargs['timeout_s'])
        now[0] = 2.
        return np.array([start, goal])
    monkeypatch.setattr(planning, '_rrt_candidate', candidate)
    path, info = planning.plan_joint_path([-.8,0], [.8,0], [[-1,1],[-1,1]],
        lambda q: not (abs(q[0]) < .2 and q[1] < .5), timeout_s=1.)
    assert path is None and len(calls) == 1
    # A late candidate is not geometrically validated or classified as rejected.
    assert info['reason'] == 'deadline_expired'
    assert info['rejected_candidate_paths'] == 0

from types import SimpleNamespace

import numpy as np
import pytest

from captain_harness.adapters.capx.ik_resolution import resolve_external_ik


def fixture():
    attempts = []
    def fk(q, aperture):
        pose = np.eye(4)
        pose[0, 3] = q[0]
        return pose
    def solve(goal, seed, **kwargs):
        attempts.append(np.asarray(seed).copy())
        return np.zeros(7)
    return SimpleNamespace(limits=np.tile([-2., 2.], (7, 1)), eef_pose=fk,
        solve_ik=solve, last_ik={'seed_attempts': 1}), attempts


def resolve(model, proposal, seed=None):
    return resolve_external_ik(model, proposal, np.eye(4),
        seed=np.ones(7)*.1 if seed is None else seed, aperture=.5,
        position_tolerance_m=.03, orientation_tolerance_rad=.08)


@pytest.mark.parametrize('proposal', [None, [], [0.]*7, [float('nan')]*8, ['bad']*8, [.5]*8])
def test_unusable_external_reply_gets_exactly_one_validated_local_attempt(proposal):
    model, attempts = fixture()
    result = resolve(model, proposal)
    assert len(attempts) == 1
    assert result['configuration'][:7] == [0.]*7
    assert result['configuration'][7] == pytest.approx(.02)
    assert result['used_fallback']
    assert result['evidence']['independent_validation']['valid']
    assert result['evidence']['motion_path'] == 'not_checked'


def test_already_valid_reply_is_not_replaced_or_retried():
    model, attempts = fixture()
    proposal = [0.]*7 + [.015]
    result = resolve(model, proposal)
    assert not attempts and result['configuration'] == proposal
    assert not result['used_fallback']


@pytest.mark.parametrize('recovered', [None, [float('nan')]*7, [3.]*7, [.1]*7])
def test_invalid_local_reply_is_never_reported_as_a_solution(recovered):
    model, attempts = fixture()
    model.solve_ik = lambda *a, **kw: recovered
    result = resolve(model, None)
    assert result['configuration'] is None
    assert result['evidence']['external_status'] == 'empty_reply'


def test_missing_fresh_robot_seed_does_not_trigger_a_stale_recovery():
    model, attempts = fixture()
    result = resolve_external_ik(model, None, np.eye(4), seed=None, aperture=.5,
        position_tolerance_m=.03, orientation_tolerance_rad=.08)
    assert not attempts and result['configuration'] is None
    assert result['evidence']['recovery_status'] == 'fresh_robot_seed_unavailable'

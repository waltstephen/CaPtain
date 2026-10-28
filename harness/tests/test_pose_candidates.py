"""Explicit orientation freedom must preserve the intended reference point."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from captain_harness.adapters.robotics.pose_candidates import solve_pose_candidates


class CartesianModel:
    """Three independent translations, with a fixed EEF rotation."""
    limits = np.array([[-1., 1.]] * 3)

    def __init__(self):
        self.calls = []
        self.last_ik = {}

    def solve_ik(self, target, seed, *, aperture):
        self.calls.append((target.copy(), seed.copy(), aperture))
        self.last_ik = {'position_residual_m': 0., 'attempt': len(self.calls)}
        if not np.allclose(target[:3, :3], np.eye(3)):
            return None
        return target[:3, 3].copy()

    def eef_pose(self, joints, *, aperture):
        pose = np.eye(4)
        pose[:3, 3] = joints
        return pose


def run(model, rotations, **kwargs):
    inputs = dict(reference_position_m=[.3, .2, .4],
        eef_rotations=rotations, reference_offset_in_eef_m=[.1, 0., .2],
        seed=[0., 0., 0.], aperture=.4,
        position_tolerance_m=.003, orientation_tolerance_rad=.02)
    inputs.update(kwargs)
    return solve_pose_candidates(model, **inputs)


def test_rotating_an_offset_keeps_the_reference_point_and_original_indices():
    model = CartesianModel()
    rotations = [Rotation.from_euler('z', 90, degrees=True).as_matrix(), np.eye(3)]
    report = run(model, rotations)
    for entry, (target, seed, aperture) in zip(report['candidates'], model.calls):
        np.testing.assert_allclose(target[:3, 3] + target[:3, :3] @ [.1, 0., .2], [.3, .2, .4])
        np.testing.assert_array_equal(seed, [0., 0., 0.])
        assert aperture == .4
        np.testing.assert_allclose(entry['target_eef_pose'], target)
    assert report['ranked_candidate_indices'] == [1]
    assert not report['candidates'][0]['endpoint_valid']
    assert report['candidates'][1]['endpoint_valid']
    assert report['candidates'][0]['ik']['attempt'] == 1
    assert report['motion_path'] == report['carried_object_collision'] == 'not_checked'
    assert report['object_retention'] == report['task_semantics'] == 'not_checked'
    assert 'plan_handle' not in report


@pytest.mark.parametrize('reply', [[2., 0., 0.], [0., 0., 0.], [float('nan'), 0., 0.], [0., 0.]])
def test_never_accepts_a_solver_reply_without_independent_endpoint_validation(reply):
    model = CartesianModel()
    model.solve_ik = lambda *a, **kw: reply
    report = run(model, [np.eye(3)])
    assert report['ranked_candidate_indices'] == []
    assert not report['candidates'][0]['endpoint_valid']
    assert report['candidates'][0]['joints'] is None


@pytest.mark.parametrize('change', [
    {'eef_rotations': []}, {'eef_rotations': [np.eye(3)] * 13},
    {'eef_rotations': [np.diag([1., 1., -1.])]},
    {'eef_rotations': [np.eye(3) * 2.]},
    {'reference_position_m': [float('nan'), 0., 0.]},
    {'reference_offset_in_eef_m': [0., 0.]}, {'seed': [0., 0.]},
    {'seed': [2., 0., 0.]}, {'aperture': -1.},
    {'position_tolerance_m': 0.}, {'orientation_tolerance_rad': float('inf')},
])
def test_invalid_or_unbounded_queries_fail_before_any_numerical_solve(change):
    model = CartesianModel()
    with pytest.raises(ValueError):
        run(model, [np.eye(3)], **change) if 'eef_rotations' not in change else run(
            model, change['eef_rotations'])
    assert not model.calls


def test_shortest_valid_joint_change_is_ranked_without_dropping_failures():
    rotations = [np.eye(3), Rotation.from_euler('z', .2).as_matrix(), Rotation.from_euler('z', .4).as_matrix()]
    model = CartesianModel()
    goals = []
    def solve(target, seed, *, aperture):
        goals.append(target.copy())
        if len(goals) == 2:
            return None
        return np.full(3, .4 if len(goals) == 1 else .1)
    model.solve_ik = solve
    model.eef_pose = lambda *a, **kw: goals[-1]
    report = run(model, rotations)
    assert report['ranked_candidate_indices'] == [2, 0]
    assert len(report['candidates']) == 3
    assert report['candidates'][1]['reason'] == 'no_numerical_solution'


def test_real_detached_planar_robot_finds_an_allowed_orientation_without_relaxing_pose_accuracy():
    from test_robot_ik_restarts import XML
    from captain_harness.adapters.robotics.model import RobotCollisionModel
    model = RobotCollisionModel(XML, joints=['a', 'b', 'c'], gripper_joints=[],
        base='base', end_effector='eef', gripper_prefix='unused')
    result = run(model, [Rotation.from_euler('z', np.pi).as_matrix(), np.eye(3)],
        reference_position_m=[3.2, 0., 0.], reference_offset_in_eef_m=[.2, 0., 0.])
    assert result['ranked_candidate_indices'] == [1]
    assert not result['candidates'][0]['endpoint_valid']
    valid = result['candidates'][1]
    actual = model.eef_pose(valid['joints'], aperture=.4)
    np.testing.assert_allclose(actual[:3, 3] + actual[:3, :3] @ [.2, 0., 0.], [3.2, 0., 0.], atol=.003)
    assert valid['validation']['orientation_residual_rad'] <= .02

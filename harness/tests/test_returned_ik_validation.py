"""An external solver returning an array does not establish endpoint accuracy."""
from types import SimpleNamespace
import numpy as np
import pytest

from captain_harness.adapters.capx.motion_tools import pose_matrix
from captain_harness.ports.capability import FailureCategory
from test_tool_backend import _fixture


def test_endpoint_validation_is_independent_of_the_solver_success_flag():
    from captain_harness.adapters.capx.ik_validation import validate_returned_ik
    goal = np.eye(4)
    actual = goal.copy()
    actual[0, 3] = .077
    model = SimpleNamespace(limits=np.tile([-2., 2.], (7, 1)), eef_pose=lambda *a, **kw: actual)
    measured = validate_returned_ik(model, [0.] * 7, goal, aperture=1.,
        position_tolerance_m=.03, orientation_tolerance_rad=np.deg2rad(5))
    assert not measured['valid'] and measured['position_residual_m'] == pytest.approx(.077)
    actual[0, 3] = .002
    measured = validate_returned_ik(model, [0.] * 7, goal, aperture=1.,
        position_tolerance_m=.03, orientation_tolerance_rad=np.deg2rad(5))
    assert measured['valid'] and measured['motion_path'] == 'not_checked'


@pytest.mark.parametrize('bad', ['position', 'orientation', 'joint_limit'])
@pytest.mark.parametrize('command', ['solve_ik', 'goto_pose'])
def test_inaccurate_external_ik_is_rejected_before_motion(tmp_path, bad, command):
    f = _fixture(tmp_path, dehardcoded=True, v15_protocol=True)
    f.backend._sensor_motion = True
    target_position = [.4, .1, .3]
    target_quaternion = [1., 0., 0., 0.]
    hand = np.asarray(target_position) + f.backend._tcp_offset_local_m
    eef, eq = f.robot.eef_pose_from_external_ik_target(hand, target_quaternion)
    actual = pose_matrix(eef, eq)
    joints = np.zeros(8)
    if bad == 'position':
        actual[0, 3] += .08
    elif bad == 'orientation':
        actual[:3, :3] = pose_matrix([0, 0, 0], [np.cos(.2), 0, 0, np.sin(.2)])[:3, :3] @ actual[:3, :3]
    else:
        joints[2] = 3.
    f.services.solve_ik = lambda *a: joints
    f.robot.planning_model = lambda: SimpleNamespace(
        limits=np.tile([-2., 2.], (7, 1)), eef_pose=lambda *a, **kw: actual)
    f.robot.move_to_pose = lambda *a, **kw: pytest.fail('inaccurate IK reached motion')
    f.robot.move_to_joints = lambda *a, **kw: pytest.fail('inaccurate IK reached motion')
    previous = object()
    f.backend._aligned_pregrasp = previous
    result = getattr(f.backend, command)(target_position, target_quaternion)
    assert not result.success and not result.executed
    assert result.failure_category is FailureCategory.IK_FAILURE
    assert result.value['reason'] == 'returned_ik_outside_declared_limits'
    assert f.backend._aligned_pregrasp is previous
    if bad == 'position':
        assert result.value['position_residual_m'] == pytest.approx(.08)
    elif bad == 'orientation':
        assert result.value['orientation_residual_rad'] == pytest.approx(.4)
    else:
        assert result.value['joint_limits_satisfied'] is False


def test_valid_external_ik_keeps_the_public_joint_list_contract(tmp_path):
    f = _fixture(tmp_path, dehardcoded=True, v15_protocol=True)
    f.backend._sensor_motion = True
    p, q = [.4, .1, .3], [1., 0., 0., 0.]
    ep, eq = f.robot.eef_pose_from_external_ik_target(np.asarray(p) + f.backend._tcp_offset_local_m, q)
    joints = np.arange(8) / 20.
    f.services.solve_ik = lambda *a: joints
    f.robot.planning_model = lambda: SimpleNamespace(
        limits=np.tile([-2., 2.], (7, 1)), eef_pose=lambda *a, **kw: pose_matrix(ep, eq))
    result = f.backend.solve_ik(p, q)
    assert result.success and result.value == joints[:7].tolist()


@pytest.mark.parametrize('recoverable', [True, False])
def test_invalid_external_solution_has_one_bounded_local_recovery_with_independent_validation(tmp_path, recoverable):
    f = _fixture(tmp_path, dehardcoded=True, v15_protocol=True)
    f.backend._sensor_motion = True
    p, q = [.4, .1, .3], [1., 0., 0., 0.]
    ep, eq = f.robot.eef_pose_from_external_ik_target(np.asarray(p) + f.backend._tcp_offset_local_m, q)
    target = pose_matrix(ep, eq)
    proposals = []
    recovered = np.full(7, .1)
    def fk(joints, *args, **kwargs):
        result = target.copy()
        if not np.allclose(joints, recovered) or not recoverable:
            result[0, 3] += .08
        return result
    def solve(goal, seed, **kwargs):
        proposals.append(goal.copy())
        return recovered
    model = SimpleNamespace(limits=np.tile([-2., 2.], (7, 1)), eef_pose=fk, solve_ik=solve,
                            last_ik={'seed_attempts': 1})
    f.robot.planning_model = lambda: model
    f.services.solve_ik = lambda *a: np.zeros(8)
    result = f.backend.solve_ik(p, q)
    assert len(proposals) == 1
    assert result.success is recoverable
    if recoverable:
        assert result.value == recovered.tolist()
        assert 'Motion path not checked' in result.message
    else:
        assert not result.executed
        assert result.value['independent_validation']['valid'] is False


@pytest.mark.parametrize('proposal', [None, [0.]*7, ['invalid']*8])
def test_sensor_ik_recovers_empty_and_malformed_replies_without_another_policy_turn(tmp_path, proposal):
    f = _fixture(tmp_path, dehardcoded=True, v15_protocol=True)
    f.backend._sensor_motion = True
    p, q = [.4, .1, .3], [1., 0., 0., 0.]
    ep, eq = f.robot.eef_pose_from_external_ik_target(np.asarray(p) + f.backend._tcp_offset_local_m, q)
    calls = []
    def local(goal, seed, **kwargs):
        calls.append(1)
        return np.zeros(7)
    f.services.solve_ik = lambda *a: proposal
    f.robot.planning_model = lambda: SimpleNamespace(limits=np.tile([-2., 2.], (7, 1)),
        eef_pose=lambda *a, **kw: pose_matrix(ep, eq), solve_ik=local)
    f.robot.move_to_pose = lambda *a, **kw: pytest.fail('solve_ik must not execute')
    result = f.backend.solve_ik(p, q)
    assert result.success and result.value == [0.]*7
    assert len(calls) == 1
    assert 'Motion path not checked' in result.message

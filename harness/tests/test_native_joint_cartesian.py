"""Independent candidate: no live LIBERO imports or runtime registration."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from captain_harness.adapters.libero.joint_cartesian import NativeJointCartesian, plan_cartesian_joints


class Model:
    limits = np.tile([-3., 3.], (7, 1))

    def eef_pose(self, q, aperture):
        pose = np.eye(4)
        pose[:3, 3] = q[:3]
        pose[:3, :3] = Rotation.from_rotvec(q[3:6]).as_matrix()
        return pose

    def _solve_ik_seed(self, pose, seed, **kwargs):
        q = np.array(seed, copy=True)
        q[:3] = pose[:3, 3]
        q[3:6] = Rotation.from_matrix(pose[:3, :3]).as_rotvec()
        return q


class Robot:
    controller = 'JOINT_POSITION'
    _control_frequency = 20.
    _gripper_steps = 10
    _gripper_command_fraction = 1.
    _gripper_open_fraction = 1.
    simulation_steps = 0
    _last_step_rejection = None

    def __init__(self):
        self.q = np.zeros(7)
        self.model = Model()
        self.force = 0.
        self.actions = []
        self.obstructed = False

    def planning_model(self):
        return self.model

    def get_robot_state(self):
        return SimpleNamespace(value=SimpleNamespace(joint_positions_rad=self.q.copy(),
            end_effector_position_m=self.q[:3].copy(),
            end_effector_quaternion_wxyz=np.roll(Rotation.from_rotvec(self.q[3:6]).as_quat(), 1),
            gripper_open_fraction=self._gripper_open_fraction))

    def _eef_force_norm(self):
        return self.force

    def _step(self, dq, fraction):
        if self._last_step_rejection:
            return False
        self.actions.append((np.asarray(dq).copy(), fraction))
        self.q += dq
        self.simulation_steps += 1
        if self.obstructed:
            self.force = 20.
        return True

    def gripper_state(self):
        return SimpleNamespace(value={'open_fraction': .2, 'object_detected': True})


def test_plan_rejects_intermediate_ik_failure_before_any_action():
    robot = Robot()
    solve = robot.model._solve_ik_seed
    robot.model._solve_ik_seed = lambda pose, seed, **kw: (
        None if .009 < pose[0, 3] < .015 else solve(pose, seed, **kw))
    result = NativeJointCartesian(robot).move([.03, 0, 0], [1, 0, 0, 0])
    assert not result.success and not result.executed
    assert result.value['stopped_reason'] == 'cartesian_plan_rejected'
    assert not robot.actions


def test_plan_rejects_nullspace_branch_jump_despite_valid_cartesian_endpoint():
    model = Model()
    solve = model._solve_ik_seed
    def jump(pose, seed, **kw):
        q = solve(pose, seed, **kw)
        q[6] = 1.
        return q
    model._solve_ik_seed = jump
    goal = np.eye(4)
    goal[0, 3] = .02
    with pytest.raises(ValueError, match='joint branch'):
        plan_cartesian_joints(model, np.zeros(7), 1., np.eye(4), goal)


def test_force_stop_takes_precedence_over_reached_and_never_closes():
    robot = Robot()
    robot.obstructed = True
    result = NativeJointCartesian(robot).descend_and_close([.0025, 0, 0], [1, 0, 0, 0])
    assert not result.success and result.executed
    assert result.value['stopped_reason'] == 'contact_force'
    assert result.value['closure_executed'] is False
    assert len(robot.actions) == 1
    assert robot.actions[0][1] == 1.


@pytest.mark.parametrize('force', [None, float('nan'), float('inf')])
def test_missing_or_invalid_force_prevents_movement(force):
    robot = Robot()
    robot.force = force
    result = NativeJointCartesian(robot).move([.02, 0, 0], [1, 0, 0, 0])
    assert not result.success and not result.executed
    assert not robot.actions


def test_command_advances_and_hold_compensates_drift_without_moving_reference():
    robot = Robot()
    drive = NativeJointCartesian(robot)
    result = drive.move([.02, 0, 0], [2, 0, 0, 0])
    assert result.success and result.executed
    np.testing.assert_allclose(robot.q[:3], [.02, 0, 0], atol=.003)
    hold_q = robot.q.copy()
    raw = robot._step
    def drift(dq, fraction):
        ok = raw(dq, fraction)
        robot.q[2] -= .0001
        return ok
    robot._step = drift
    result = drive.hold(.5, gripper_fraction=0.)
    assert result.success
    assert np.linalg.norm(robot.q - hold_q) < .0002
    assert robot._gripper_command_fraction == 0.


def test_step_guard_rejection_is_neither_execution_nor_horizon_success():
    robot = Robot()
    robot._last_step_rejection = 'motion_guard_rejected'
    result = NativeJointCartesian(robot).move([.02, 0, 0], [1, 0, 0, 0])
    assert not result.success and not result.executed
    assert result.value['stopped_reason'] == 'motion_guard_rejected'
    assert not robot.actions


def test_invalid_quaternion_is_rejected_without_action():
    robot = Robot()
    result = NativeJointCartesian(robot).move([.02, 0, 0], [0, 0, 0, 0])
    assert not result.success and not result.executed
    assert not robot.actions


def test_closure_keeps_recent_commanded_posture_instead_of_freezing_tracking_lag():
    robot = Robot()
    raw = robot._step
    robot._step = lambda dq, fraction: raw(np.asarray(dq) * .25, fraction)
    drive = NativeJointCartesian(robot)
    moved = drive.move([.02, 0, 0], [1, 0, 0, 0])
    assert moved.success
    old_error = abs(robot.q[0] - .02)
    assert old_error > .0001
    assert drive.hold(1., gripper_fraction=0.).success
    assert abs(robot.q[0] - .02) < old_error / 10


def test_intervening_motion_invalidates_the_retained_hold_reference():
    robot = Robot()
    drive = NativeJointCartesian(robot)
    assert drive.move([.02, 0, 0], [1, 0, 0, 0]).success
    robot._step(np.array([.02, 0, 0, 0, 0, 0, 0]), 1.)
    current = robot.q.copy()
    assert drive.hold(.5).success
    np.testing.assert_allclose(robot.q, current)

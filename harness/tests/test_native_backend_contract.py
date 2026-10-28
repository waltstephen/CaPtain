"""Native robot method contract before production capability registration."""
import json

import numpy as np
import pytest

from captain_harness.adapters.libero.contact_motion import ContactMotionMixin
from captain_harness.adapters.libero.native_backend import NativeSensorMotionMixin
from captain_harness.ports.capability import CapabilityResult, FailureCategory
from test_native_joint_sweep import Robot


class ContractRobot(NativeSensorMotionMixin, ContactMotionMixin, Robot):
    _cartesian_max_steps = 480
    _home_max_steps = 480
    _home_eef_position = [.02, 0, 0]
    _home_eef_quaternion = [1, 0, 0, 0]

    def gripper_state(self):
        return CapabilityResult.ok({'object_detected': True, 'open_fraction': .3})

    def _pose_residuals(self, position, quaternion):
        return float(np.linalg.norm(self.q[:3]-position)), 0.


def test_pose_linear_arc_hold_and_home_use_native_commands():
    robot = ContractRobot()
    assert robot.move_to_pose([.02, 0, 0], [1, 0, 0, 0]).success
    assert robot.move_linear([0, 0, 1], .01).success
    assert robot.rotate_about_axis([0, 0, 1], [.06, 0, .01], .25).success
    result = robot.hold_position(.1)
    assert result.success and result.value['steps_executed'] == 2
    assert result.value['pose_hold']['within_tolerance']
    assert robot.home().success
    assert len(robot.actions) > 0


def test_descent_force_stop_does_not_close_and_retreat_uses_recorded_route():
    robot = ContractRobot()
    robot.force = lambda: 20. if robot.q[0] >= .005-1e-12 else 5.
    result = robot.descend_and_close([.03, 0, 0], [1, 0, 0, 0])
    assert not result.success and result.value['closure_executed'] is False
    assert all(fraction == 1. for _, fraction in robot.actions)
    assert robot.retreat_last_motion().success
    assert not robot.retreat_last_motion().executed


def test_successful_descent_exposes_measurements_and_closure_separately():
    robot = ContractRobot()
    result = robot.descend_and_close([.01, 0, 0], [1, 0, 0, 0])
    assert result.success and result.executed
    assert result.value['closure_executed'] and result.value['object_detected']
    assert result.value['grasp_outcome'] == 'object_detected'
    assert result.value['controller_execution']['control_steps'] > 0
    assert result.value['closure_execution']['control_steps'] == 2
    assert not robot.retreat_last_motion().executed


def test_hold_corrects_posture_instead_of_accumulating_zero_delta_drift():
    robot = ContractRobot()
    first = robot._step
    def disturbed(delta, fraction):
        result = first(delta, fraction)
        robot.q[0] += .001
        return result
    robot._step = disturbed
    result = robot.hold_position(1.)
    assert result.success
    assert robot.q[0] == pytest.approx(.001)


def test_public_contact_tolerance_cannot_be_relaxed_to_component_tolerance():
    robot = ContractRobot()
    drive = robot._native_drive()
    drive.execute_sweep = lambda *a, **kw: CapabilityResult.ok({
        'position_residual_m': .0025, 'orientation_residual_rad': .005, 'reached': True})
    result = robot.move_linear([1, 0, 0], .01)
    assert not result.success and result.message == 'native_endpoint_outside_public_tolerance'
    assert result.value['reached'] is False


@pytest.mark.parametrize('force', [float('nan'), float('inf'), True, 19., 'invalid'])
def test_invalid_descent_limit_is_json_safe_and_never_moves(force):
    robot = ContractRobot()
    result = robot.descend_and_close([.01, 0, 0], [1, 0, 0, 0], max_contact_force=force)
    assert not result.success and not result.executed and not robot.actions
    json.dumps(result.value, allow_nan=False)


def test_hold_driver_exception_is_an_explicit_executed_failure():
    robot = ContractRobot()
    original = robot._step
    def disconnected(delta, fraction):
        original(delta, fraction)
        raise RuntimeError('connection lost')
    robot._step = disconnected
    result = robot.hold_position(.2)
    assert not result.success and result.executed
    assert result.failure_category is FailureCategory.BACKEND_ERROR
    assert not robot.retreat_last_motion().executed


def test_invalid_hold_does_not_advance_simulation():
    robot = ContractRobot()
    assert not robot.hold_position(True).success
    assert not robot.actions


def test_retreat_passes_original_public_endpoint_tolerances_to_driver():
    robot = ContractRobot()
    received = {}
    def retreat(*args, **kwargs):
        received.update(kwargs)
        return CapabilityResult.ok({})
    robot._native_drive().retreat_last_motion = retreat
    robot.retreat_last_motion()
    assert received['position_tolerance_m'] == .002
    assert received['orientation_tolerance_rad'] == .01

import numpy as np

from captain_harness.adapters.libero.native_gripper_release import NativeGripperReleaseMixin
from test_native_backend_contract import ContractRobot


class Robot(NativeGripperReleaseMixin, ContractRobot):
    pass


def test_opening_can_unload_high_initial_force_while_holding_measured_posture():
    robot = Robot()
    robot.force = lambda: [25., 17., 5.][min(robot.simulation_steps, 2)]
    robot.q[:] = [.1, .2, .3, 0, 0, 0, 0]
    before = robot.q.copy()
    result = robot._set_gripper(1.)
    assert result.success and result.executed
    np.testing.assert_allclose(robot.q, before)
    assert all(fraction == 1. for _, fraction in robot.actions)
    report = robot._last_native_gripper_execution.value
    assert report['initial_force_n'] == 25. and report['observed_force_n'] == 5.


def test_opening_stops_on_force_reincrease_after_unloading():
    robot = Robot()
    robot._gripper_steps = 5
    robot.force = lambda: [25., 10., 16.][min(robot.simulation_steps, 2)]
    result = robot._set_gripper(1.)
    assert not result.success and robot.simulation_steps == 2
    assert result.message == 'release_force_increased'


def test_opening_does_not_ignore_missing_force_or_external_guard():
    robot = Robot()
    robot.force = lambda: None
    assert not robot._set_gripper(1.).executed
    robot.force = lambda: 25.
    robot._motion_guard = lambda: False
    result = robot._set_gripper(1.)
    assert not result.success and not result.executed
    assert not robot.actions


def test_high_force_opening_exception_does_not_enable_high_force_closing():
    robot = Robot()
    robot.force = lambda: 25.
    result = robot._set_gripper(0.)
    assert not result.success and not result.executed and not robot.actions


def test_still_loaded_after_opening_is_not_a_successful_release():
    robot = Robot()
    robot.force = lambda: 25.
    result = robot._set_gripper(1.)
    assert not result.success and result.executed
    assert result.message == 'release_still_loaded'

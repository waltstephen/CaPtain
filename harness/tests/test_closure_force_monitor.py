from types import SimpleNamespace

import pytest

from captain_harness.adapters.libero.closure_force_monitor import ClosureForceMonitorMixin
from captain_harness.ports.capability import CapabilityResult, FailureCategory


class Base:
    simulation_steps = 0
    _motion_guard = None

    def __init__(self):
        self.actions = []
        self.force = lambda: 5.

    def _eef_force_norm(self):
        return self.force()

    def _set_gripper(self, fraction):
        first = self.simulation_steps
        for _ in range(4):
            if self._motion_guard and not self._motion_guard():
                return CapabilityResult.failed(FailureCategory.CONTROLLER_REJECTED,
                    'motion_guard_rejected', executed=self.simulation_steps > first)
            self.actions.append(fraction)
            self.simulation_steps += 1
        return CapabilityResult.ok(None)

    def descend_and_close(self, *args, **kwargs):
        result = self._set_gripper(0.)
        # Reproduce stale pre-closure force from the existing descent method.
        if result.success:
            return CapabilityResult.ok({'observed_force_n': 5., 'closure_executed': True})
        return result


class Robot(ClosureForceMonitorMixin, Base):
    pass


def test_stops_closing_when_force_increases_and_reports_post_closure_measurement():
    robot = Robot()
    robot.force = lambda: 5. if robot.simulation_steps < 2 else 12.
    result = robot.descend_and_close([0, 0, 0], [1, 0, 0, 0])
    assert not result.success and result.executed
    assert result.value['closure_executed'] is True
    assert result.value['observed_force_n'] == 12.
    assert result.value['closure_force_monitor']['reason'] == 'closure_force_increase'
    assert robot.simulation_steps == 2


def test_final_closing_cycle_force_takes_precedence_over_success():
    robot = Robot()
    robot.force = lambda: 19. if robot.simulation_steps == 4 else 5.
    result = robot.descend_and_close([0, 0, 0], [1, 0, 0, 0])
    assert not result.success and result.executed
    assert len(result.value['closure_force_monitor']['samples']) == 5


def test_successful_closure_replaces_stale_force_without_changing_outcome():
    robot = Robot()
    robot.force = lambda: 5.+robot.simulation_steps*.1
    result = robot.descend_and_close([0, 0, 0], [1, 0, 0, 0])
    assert result.success and result.value['observed_force_n'] == pytest.approx(5.4)


def test_lower_explicit_force_limit_is_kept_for_closure():
    robot = Robot()
    robot.force = lambda: 10.+robot.simulation_steps
    result = robot.descend_and_close([0, 0, 0], [1, 0, 0, 0], max_contact_force=12.)
    assert not result.success and robot.simulation_steps == 2
    assert result.value['closure_force_monitor']['force_limit_n'] == 12.


def test_missing_force_blocks_closing_but_opening_under_load_remains_available():
    robot = Robot()
    robot.force = lambda: None
    assert not robot._set_gripper(0.).executed and not robot.actions
    robot.force = lambda: 25.
    assert robot._set_gripper(1.).success
    assert robot.actions == [1.]*4


def test_outer_guard_is_composed_and_restored():
    robot = Robot()
    outer = lambda: robot.simulation_steps < 1
    robot._motion_guard = outer
    result = robot.descend_and_close([0, 0, 0], [1, 0, 0, 0])
    assert not result.success and robot.simulation_steps == 1
    assert robot._motion_guard is outer


def test_no_closure_does_not_borrow_previous_closure_monitor():
    robot = Robot()
    assert robot.descend_and_close([0, 0, 0], [1, 0, 0, 0]).success
    robot.force = lambda: None
    result = robot.descend_and_close([0, 0, 0], [1, 0, 0, 0])
    assert not result.executed and result.value['closure_executed'] is False


def test_outer_guard_exception_is_an_explicit_failure_and_guard_is_restored():
    robot = Robot()
    def broken_guard():
        raise RuntimeError('sensor disconnected')
    robot._motion_guard = broken_guard
    result = robot.descend_and_close([0, 0, 0], [1, 0, 0, 0])
    assert not result.success and not result.executed
    assert robot._motion_guard is broken_guard
    assert result.value['closure_executed'] is False


def make_actual_monitored_robot(tmp_path):
    from test_libero_adapter import FakeOscEnv
    from captain_harness.adapters.libero.backend import LiberoActionBackend
    class ActualRobot(ClosureForceMonitorMixin, LiberoActionBackend):
        pass
    env=FakeOscEnv()
    robot=ActualRobot(env,observation_dir=tmp_path,initial_observation=env.reset(),
        horizon=1000,controller='OSC_POSE',gripper_steps=4)
    return env,robot


def test_force_stopped_closure_does_not_resume_closing_on_next_arm_hold(tmp_path):
    env,robot=make_actual_monitored_robot(tmp_path)
    robot._replace_gripper_target_with_measured_position=lambda: {'status':'replaced'}
    robot._eef_force_norm=lambda:5. if robot.simulation_steps<2 else 12.
    result=robot.close_gripper()
    assert not result.success and result.executed
    assert result.message=='closure_force_increase'
    assert len(env.actions)==2
    assert all(action[-1]==1. for action in env.actions)
    after_stop=len(env.actions)
    assert robot.hold_position(.1).success
    assert all(action[-1]==0. for action in env.actions[after_stop:])
    receipt=result.diagnostics['closure_force_monitor']
    assert receipt['post_stop_gripper_command']['native_action']==0.
    assert receipt['post_stop_gripper_command']['effect']=='hold_measured_joint_target'
    assert receipt['control_steps']==2
    # A later explicit opening request remains available under the same load.
    before_open=len(env.actions)
    assert robot.open_gripper().success
    assert all(action[-1]==-1. for action in env.actions[before_open:])


def test_successful_monitored_closure_keeps_closed_command_on_hold(tmp_path):
    env,robot=make_actual_monitored_robot(tmp_path)
    robot._eef_force_norm=lambda:5.
    assert robot.close_gripper().success
    before_hold=len(env.actions)
    assert robot.hold_position(.1).success
    assert all(action[-1]==1. for action in env.actions[before_hold:])


def test_zero_step_force_rejection_keeps_previous_gripper_command(tmp_path):
    env,robot=make_actual_monitored_robot(tmp_path)
    robot._eef_force_norm=lambda:25.
    result=robot.close_gripper()
    assert not result.executed and not env.actions
    robot._eef_force_norm=lambda:5.
    assert robot.hold_position(.1).success
    assert all(action[-1]==-1. for action in env.actions)

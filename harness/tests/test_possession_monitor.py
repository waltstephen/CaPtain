"""An endpoint reached with an emptied gripper is not preserved possession."""
import pytest

from captain_harness.adapters.libero.backend import LiberoActionBackend
from captain_harness.adapters.libero.possession_monitor import PossessionMonitorMixin
from captain_harness.ports.capability import CapabilityResult, FailureCategory
from test_libero_adapter import FakeOscEnv


class Robot(PossessionMonitorMixin, LiberoActionBackend):
    pass


def robot_fixture(tmp_path, *, lost_after=None, unavailable_after=None):
    env = FakeOscEnv()
    robot = Robot(env, observation_dir=tmp_path, initial_observation=env.reset(),
        horizon=1000, controller='OSC_POSE', cartesian_max_steps=480,
        guarded_cartesian_motion=True)
    def register():
        if unavailable_after is not None and len(env.actions) >= unavailable_after:
            return CapabilityResult.failed(FailureCategory.BACKEND_ERROR, 'register unavailable')
        detected = lost_after is None or len(env.actions) < lost_after
        return CapabilityResult.ok({'object_detected': detected, 'open_fraction': .4 if detected else .001})
    robot.gripper_state = register
    return robot, env


def test_loss_stops_the_next_motion_step_and_preserves_failure_evidence(tmp_path):
    robot, env = robot_fixture(tmp_path, lost_after=2)
    result = robot.with_possession_guard(lambda: robot.move_linear([1, 0, 0], .04))
    assert not result.success and result.executed
    assert len(env.actions) == 2
    assert result.failure_category is FailureCategory.CONTROLLER_REJECTED
    assert result.value['stopped_reason'] == 'gripper_possession_lost'
    assert result.value['possession_monitor']['status'] == 'lost'
    assert result.value['possession_monitor']['stop_sample']['open_fraction'] == .001
    assert len(result.value['trajectory']) == 2
    assert robot._possession_guard is None


@pytest.mark.parametrize('sensor_missing', [False, True])
def test_missing_initial_possession_executes_nothing(tmp_path, sensor_missing):
    robot, env = robot_fixture(tmp_path, lost_after=0,
        unavailable_after=0 if sensor_missing else None)
    result = robot.with_possession_guard(lambda: pytest.fail('unavailable initial possession reached motion'))
    assert not result.success and not result.executed
    assert result.value['possession_monitor']['status'] == ('unavailable' if sensor_missing else 'not_present')
    assert not env.actions


def test_sensor_failure_during_motion_stops_without_fabricating_loss(tmp_path):
    robot, env = robot_fixture(tmp_path, unavailable_after=2)
    result = robot.with_possession_guard(lambda: robot.move_linear([1, 0, 0], .04))
    assert not result.success and result.executed
    assert len(env.actions) == 2
    assert result.value['possession_monitor']['status'] == 'unavailable'


def test_maintained_possession_keeps_motion_and_unmonitored_moves_unchanged(tmp_path):
    robot, env = robot_fixture(tmp_path)
    result = robot.with_possession_guard(lambda: robot.move_linear([1, 0, 0], .01))
    assert result.success
    assert result.value['possession_monitor']['status'] == 'maintained'
    robot.gripper_state = lambda: pytest.fail('unmonitored motion queried possession')
    assert robot.move_linear([1, 0, 0], .01).success


def test_monitor_scope_is_cleared_when_a_driver_raises(tmp_path):
    robot, _ = robot_fixture(tmp_path)
    def broken():
        raise RuntimeError('driver failed')
    result = robot.with_possession_guard(broken)
    assert not result.success and not result.executed
    assert result.failure_category is FailureCategory.BACKEND_ERROR
    assert 'driver failed' in result.message
    assert robot._possession_guard is None


def test_raw_multistep_driver_is_sampled_through_existing_motion_guard(tmp_path):
    robot, env = robot_fixture(tmp_path, lost_after=2)
    robot._guarded_cartesian_motion = False
    result = robot.with_possession_guard(lambda: robot._move_to_pose_osc(
        [.9, .1, .5], [1, 0, 0, 0], max_steps=20, position_only=False))
    assert not result.success and result.executed
    assert len(env.actions) == 2
    assert result.value['possession_monitor']['status'] == 'lost'


def test_simultaneous_force_stop_keeps_primary_reason_and_reports_possession_loss(tmp_path):
    robot, env = robot_fixture(tmp_path, lost_after=2)
    robot._eef_force_norm = lambda: 20. if len(env.actions) >= 2 else 0.
    result = robot.with_possession_guard(lambda: robot.move_linear([1, 0, 0], .04))
    assert result.value['stopped_reason'] == 'contact_force'
    assert result.value['possession_monitor']['status'] == 'lost'
    assert len(env.actions) == 2


def test_normal_guarded_goto_dispatch_is_sampled_and_loss_cannot_be_soft_accepted(tmp_path):
    robot, env = robot_fixture(tmp_path, lost_after=2)
    current = robot.get_robot_state().value
    target = list(current.end_effector_position_m)
    target[0] += .05
    result = robot.with_possession_guard(lambda: robot.move_to_pose(target, current.end_effector_quaternion_wxyz))
    assert not result.success and result.failure_category is FailureCategory.CONTROLLER_REJECTED
    assert result.value['stopped_reason'] == 'gripper_possession_lost'
    assert len(env.actions) == 2


def test_stationary_hold_detects_loss_without_rebasing_the_hold_target(tmp_path):
    robot, env = robot_fixture(tmp_path, lost_after=2)
    result = robot.with_possession_guard(lambda: robot.hold_position(.5))
    assert not result.success and result.executed
    assert len(env.actions) == 2
    assert result.value['possession_monitor']['status'] == 'lost'
    monitor = result.value['possession_monitor']
    assert monitor['aperture_change_fraction'] == pytest.approx(.001 - .4)
    assert monitor['sampled_duration_s'] == pytest.approx(.1)
    assert len(monitor['aperture_samples']) == 3


def test_existing_motion_guard_is_composed_and_restored(tmp_path):
    robot, env = robot_fixture(tmp_path)
    original = lambda: False
    robot._motion_guard = original
    result = robot.with_possession_guard(lambda: robot.move_linear([1, 0, 0], .04))
    assert not result.success and not env.actions
    assert result.value['stopped_reason'] == 'motion_guard_rejected'
    assert robot._motion_guard is original


def test_driver_exception_after_loss_keeps_executed_flag_and_negative_sample(tmp_path):
    robot, env = robot_fixture(tmp_path, lost_after=1)
    def broken():
        robot._step_action([0, 0, 0, 0, 0, 0, 1])
        raise RuntimeError('driver disconnected')
    result = robot.with_possession_guard(broken)
    assert not result.success and result.executed
    assert result.failure_category is FailureCategory.BACKEND_ERROR
    assert 'driver disconnected' in result.message
    assert result.value['possession_monitor']['status'] == 'lost'
    assert robot._possession_guard is None


def test_register_exception_is_unknown_evidence_and_never_executes_motion(tmp_path):
    robot, env = robot_fixture(tmp_path)
    def broken_read():
        raise OSError('sensor disconnected')
    robot.gripper_state = broken_read
    result = robot.with_possession_guard(lambda: pytest.fail('must not execute'))
    assert not result.success and not result.executed
    assert result.value['possession_monitor']['status'] == 'unavailable'
    assert not env.actions

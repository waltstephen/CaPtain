"""Contact primitives stop from measured signals and retain a bounded retreat."""
import numpy as np
import pytest

from captain_harness.adapters.libero.backend import LiberoActionBackend
from captain_harness.adapters.libero.contact_motion import ContactMotionMixin
from test_libero_adapter import FakeOscEnv


class Robot(LiberoActionBackend):
    pass


def make_robot(tmp_path, env):
    return Robot(env, observation_dir=tmp_path, initial_observation=env.reset(),
                 horizon=1000, controller='OSC_POSE', cartesian_max_steps=480)


def test_linear_contact_motion_bounds_commands_and_preserves_the_gripper(tmp_path):
    env = FakeOscEnv()
    robot = make_robot(tmp_path, env)
    initial = robot.get_robot_state().value
    result = robot.move_linear([1., 0., 0.], .02, max_speed_m_s=.02)
    assert result.success and result.executed, result.message
    after = robot.get_robot_state().value
    assert after.end_effector_position_m[0] - initial.end_effector_position_m[0] == pytest.approx(.02, abs=.001)
    increments = np.array(env.actions)[:, :3] * .05
    assert np.max(np.linalg.norm(increments, axis=1)) <= .001 + 1e-12
    assert all(action[-1] == -1. for action in env.actions)
    assert result.value['reached'] and result.value['contact_identity'] == 'unknown'


def test_contact_force_stops_before_repeated_pressure_and_allows_recorded_retreat(tmp_path):
    class Obstacle(FakeOscEnv):
        def step(self, action):
            response = super().step(action)
            self.robots[0].ee_force = np.array([20. if self.joints[0] >= .005 else 0., 0., 0.])
            return response
    env = Obstacle()
    robot = make_robot(tmp_path, env)
    start = robot.get_robot_state().value
    result = robot.move_linear([1., 0., 0.], .04)
    assert not result.success and result.executed
    assert result.value['stopped_reason'] == 'contact_force'
    assert len(env.actions) <= 6
    failed_count = len(env.actions)
    retreat = robot.retreat_last_motion(max_distance_m=.03)
    assert retreat.success and retreat.executed, retreat.message
    assert len(env.actions) > failed_count
    assert robot.get_robot_state().value.end_effector_position_m[0] == pytest.approx(start.end_effector_position_m[0], abs=.001)
    assert robot._eef_force_norm() == 0.


def test_force_missing_is_an_unexecuted_capability_failure(tmp_path):
    env = FakeOscEnv()
    robot = make_robot(tmp_path, env)
    env.robots[0].ee_force = None
    result = robot.move_linear([1., 0., 0.], .02)
    assert not result.success and not result.executed
    assert result.value['stopped_reason'] == 'force_unavailable'
    assert not env.actions


def test_intervening_dynamics_invalidates_recorded_retreat(tmp_path):
    env = FakeOscEnv()
    robot = make_robot(tmp_path, env)
    assert robot.move_linear([1., 0., 0.], .01).success
    assert robot.hold_position(.1).success
    count = len(env.actions)
    result = robot.retreat_last_motion()
    assert not result.executed and not result.success
    assert len(env.actions) == count
    assert result.value['stopped_reason'] == 'recorded_motion_is_stale'


def test_initial_force_limit_rejects_forward_motion_without_erasing_retreat(tmp_path):
    env = FakeOscEnv()
    robot = make_robot(tmp_path, env)
    assert robot.move_linear([1., 0., 0.], .01).success
    env.robots[0].ee_force = [20., 0., 0.]
    previous = robot._last_guarded_motion
    result = robot.move_linear([1., 0., 0.], .01)
    assert not result.executed and not result.success
    assert robot._last_guarded_motion is previous


def test_invalid_contact_request_never_executes(tmp_path):
    robot = make_robot(tmp_path, FakeOscEnv())
    result = robot.move_linear([0., 0., 0.], .01)
    assert not result.success and not result.executed


def test_slow_but_progressing_robot_paces_the_command_without_relaxing_tracking_limit(tmp_path):
    class SlowRobot(FakeOscEnv):
        def step(self, action):
            slowed = np.array(action, copy=True)
            slowed[:3] *= .1
            return super().step(slowed)
    robot = make_robot(tmp_path, SlowRobot())
    result = robot.move_linear([0, 0, 1], .1, max_speed_m_s=.1)
    assert result.success, result.message
    trace = result.value['trajectory']
    measured = np.asarray([s['eef_position_m'] for s in trace])
    commanded = np.asarray([s['commanded_eef_position_m'] for s in trace])
    assert np.max(np.linalg.norm(commanded - measured, axis=1)) <= .02
    assert np.max(np.linalg.norm(np.diff(commanded, axis=0), axis=1)) <= .005 + 1e-10


def test_motion_monitor_rejection_is_not_mislabeled_as_horizon_exhaustion(tmp_path):
    from captain_harness.ports.capability import FailureCategory
    env = FakeOscEnv()
    robot = make_robot(tmp_path, env)
    robot._motion_guard = lambda: False
    result = robot.move_linear([0, 0, 1], .02)
    assert not result.success and not result.executed
    assert result.failure_category is FailureCategory.CONTROLLER_REJECTED
    assert result.value['stopped_reason'] == 'motion_guard_rejected'
    assert not env.actions


def test_monitored_pose_goal_preserves_original_osc_commands_when_unobstructed(tmp_path):
    original_env, guarded_env = FakeOscEnv(), FakeOscEnv()
    original = make_robot(tmp_path / 'original', original_env)
    guarded = make_robot(tmp_path / 'monitored', guarded_env)
    state = original.get_robot_state().value
    goal = np.asarray(state.end_effector_position_m) + [0, 0, .1]
    assert original.move_to_pose(goal, state.end_effector_quaternion_wxyz).success
    result = guarded.move_to_pose_monitored(goal, state.end_effector_quaternion_wxyz)
    assert result.success
    np.testing.assert_allclose(guarded_env.actions, original_env.actions)
    assert result.value['command_profile'] == 'fixed_pose_goal'
    assert result.value['max_command_translation_m_per_step'] is None


def test_monitored_pose_goal_stops_after_first_force_sample(tmp_path):
    class Contact(FakeOscEnv):
        def step(self, action):
            result = super().step(action)
            self.robots[0].ee_force = [20., 0., 0.]
            return result
    env = Contact()
    robot = make_robot(tmp_path, env)
    state = robot.get_robot_state().value
    result = robot.move_to_pose_monitored(np.asarray(state.end_effector_position_m) + [0, 0, .1],
                                          state.end_effector_quaternion_wxyz)
    assert not result.success and result.executed
    assert result.value['stopped_reason'] == 'contact_force'
    assert len(env.actions) == 1
    assert result.value['trajectory'][0]['force_n'] == 20.

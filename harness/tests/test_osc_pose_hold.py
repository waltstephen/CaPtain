"""A zero OSC increment follows drift; a hold must retain one pose target."""
import numpy as np
import pytest

from test_libero_adapter import FakeData, FakeOscEnv
from captain_harness.adapters.libero import LiberoActionBackend


class DisturbedOscEnv(FakeOscEnv):
    def step(self, action):
        super().step(action)
        # Persistent external disturbance, independent of task/object identity.
        self.joints[0] += .001
        return self._get_observations(), 0., False, {}


@pytest.mark.parametrize("operation", ["hold", "close", "open"])
def test_pose_is_held_against_disturbance_during_time_and_gripper_actions(tmp_path, operation):
    env = DisturbedOscEnv()
    robot = LiberoActionBackend(env, observation_dir=tmp_path,
        initial_observation=env.reset(), horizon=100, controller="OSC_POSE",
        gripper_steps=10)
    start = np.asarray(robot.get_robot_state().value.end_effector_position_m)
    result = {"hold": lambda: robot.hold_position(.5),
              "close": robot.close_gripper, "open": robot.open_gripper}[operation]()
    end = np.asarray(robot.get_robot_state().value.end_effector_position_m)
    assert result.success and result.executed
    assert len(env.actions) == 10
    # One disturbance step may remain; it must not accumulate across the hold.
    assert np.linalg.norm(end - start) <= .0011
    expected_gripper = 1. if operation == "close" else -1.
    assert all(action[-1] == expected_gripper for action in env.actions)


def test_hold_reports_pose_residual_without_claiming_object_stability(tmp_path):
    env = DisturbedOscEnv()
    robot = LiberoActionBackend(env, observation_dir=tmp_path,
        initial_observation=env.reset(), horizon=100, controller="OSC_POSE")
    result = robot.hold_position(.5)
    assert result.value["pose_hold"]["reference"] == "end_effector"
    assert result.value["pose_hold"]["position_residual_m"] <= .0011
    assert result.value["pose_hold"]["within_tolerance"] is True
    assert "object stability is not asserted" in result.message


def test_servo_iterations_count_this_call_only(tmp_path):
    env = FakeOscEnv()
    robot = LiberoActionBackend(env, observation_dir=tmp_path,
        initial_observation=env.reset(), horizon=100, controller="OSC_POSE")
    robot.hold_position(.5)
    before = robot.simulation_steps
    result = robot.servo_to((.43,.1,.5), (1.,0.,0.,0.),
        max_duration_s=1., tolerance_m=.004)
    assert result.success
    assert result.value["iterations"] == robot.simulation_steps - before


def test_hold_corrects_orientation_drift_as_well_as_translation(tmp_path):
    class RotatingData(FakeData):
        @property
        def xmat(self):
            c, s = np.cos(self.env.yaw), np.sin(self.env.yaw)
            return np.asarray([np.eye(3).ravel(), np.array([[c,-s,0],[s,c,0],[0,0,1]]).ravel()])

        @property
        def site_xmat(self):
            return self.xmat[1:]

    class RotatingEnv(FakeOscEnv):
        def __init__(self):
            super().__init__()
            self.yaw = .6
            self.sim.data = RotatingData(self)

        def step(self, action):
            super().step(action)
            self.yaw += float(action[5]) * .5 + .01
            return self._get_observations(), 0., False, {}

    env = RotatingEnv()
    robot = LiberoActionBackend(env, observation_dir=tmp_path,
        initial_observation=env.reset(), horizon=100, controller="OSC_POSE")
    result = robot.hold_position(.5)
    assert result.success
    assert result.value["pose_hold"]["orientation_residual_rad"] <= .0101
    assert abs(env.yaw - .6) <= .0101


def test_pregrasp_recheck_uses_original_goal_not_hold_start(tmp_path):
    env = FakeOscEnv()
    robot = LiberoActionBackend(env, observation_dir=tmp_path,
        initial_observation=env.reset(), horizon=100, controller="OSC_POSE")
    valid = robot.pregrasp_pose_status((.41,.1,.5), (1.,0.,0.,0.))
    invalid = robot.pregrasp_pose_status((.42,.1,.5), (1.,0.,0.,0.))
    assert valid.value["valid"] is True
    assert invalid.value["valid"] is False
    assert not env.actions

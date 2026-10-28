"""Small correct commands must not disappear inside an absolute tolerance."""
import numpy as np
import pytest
from test_libero_adapter import FakeOscEnv
from captain_harness.adapters.libero import LiberoActionBackend


class SlowOsc(FakeOscEnv):
    def step(self, action):
        scaled = np.array(action, copy=True)
        scaled[:3] *= .1
        return super().step(scaled)


def robot_at(tmp_path, env):
    return LiberoActionBackend(env, observation_dir=tmp_path,
        initial_observation=env.reset(), horizon=1000, controller="OSC_POSE")


@pytest.mark.parametrize("distance", [.000409, .001, .00285, .02])
@pytest.mark.parametrize("direction", [[1, 0, 0], [0, -1, 0], [2, -3, 4]])
def test_small_translation_converges_without_enlarging_command(tmp_path, distance, direction):
    robot = robot_at(tmp_path, SlowOsc())
    start = np.asarray(robot.get_robot_state().value.end_effector_position_m)
    delta = np.asarray(direction) / np.linalg.norm(direction) * distance
    result = robot.move_linear(direction, distance)
    assert result.success, result.message
    measured = np.asarray(robot.get_robot_state().value.end_effector_position_m) - start
    assert np.linalg.norm(measured - delta) <= min(.002, .1 * distance) + 1e-12
    trace = result.value["trajectory"]
    commanded = np.asarray([s["commanded_eef_position_m"] for s in trace]) - start
    assert max(np.linalg.norm(s) for s in commanded) <= distance + 1e-12
    np.testing.assert_allclose(result.value["requested_displacement_m"], delta, atol=1e-12)
    np.testing.assert_allclose(result.value["actual_displacement_m"], measured, atol=1e-12)
    assert result.value["position_tolerance_m"] <= .1 * distance + 1e-12


def test_small_motion_obstructed_is_failure_not_reached(tmp_path):
    class Stuck(FakeOscEnv):
        def step(self, action):
            blocked = np.array(action, copy=True)
            blocked[:6] = 0
            return super().step(blocked)
    robot = robot_at(tmp_path, Stuck())
    result = robot.move_linear([1, 0, 0], .000409)
    assert result.executed and not result.success
    assert not result.value["reached"]
    assert result.value["stopped_reason"] == "motion_stalled"
    assert result.value["control_steps"] <= 11
    assert result.value["position_residual_m"] == pytest.approx(.000409)


def test_small_motion_does_not_relax_force_or_duration(tmp_path):
    class Contact(SlowOsc):
        def step(self, action):
            result = super().step(action)
            self.robots[0].ee_force = [20, 0, 0]
            return result
    result = robot_at(tmp_path / "contact", Contact()).move_linear([1, 0, 0], .000409)
    assert not result.success and result.value["stopped_reason"] == "contact_force"
    assert result.value["control_steps"] == 1
    result = robot_at(tmp_path / "duration", SlowOsc()).move_linear([1, 0, 0], .000409, max_duration_s=.05)
    assert not result.success and result.value["stopped_reason"] == "duration_exhausted"
    assert result.value["control_steps"] == 1


def test_fine_rotation_uses_angular_accuracy_without_tightening_zero_translation():
    from scipy.spatial.transform import Rotation
    from captain_harness.adapters.robotics.motion_precision import motion_precision
    start, goal = np.eye(4), np.eye(4)
    goal[:3, :3] = Rotation.from_rotvec([0, 0, -.004]).as_matrix()
    precision = motion_precision(start, goal, .002, .01)
    assert precision["orientation_tolerance_rad"] == pytest.approx(.0004)
    assert precision["position_tolerance_m"] == .002
    np.testing.assert_allclose(precision["requested_rotation_vector_rad"], [0, 0, -.004])


def test_bounded_endpoint_does_not_finish_while_motion_is_still_large(tmp_path):
    robot = robot_at(tmp_path, SlowOsc())
    result = robot.move_linear([1, 0, 0], .02)
    assert result.success, result.message
    positions = np.asarray([step["eef_position_m"] for step in result.value["trajectory"]])
    # A following sub-mm correction cannot tolerate half-mm carryover per cycle.
    assert np.max(np.linalg.norm(np.diff(positions[-3:], axis=0), axis=1)) <= .00005
    assert result.value["endpoint_settled"] is True

import math
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_libero_adapter import FakeOscEnv
from test_motion_precision_contract import robot_at
from captain_harness.adapters.libero.contact_motion import _pose
from captain_harness.adapters.robotics.cartesian_sweeps import cartesian_sweep
from captain_harness.adapters.robotics.cartesian_path_timing import (
    advance_cartesian_reference, minimum_command_duration,
)


def traverse(path, speed=.05, angular=.5):
    command = path[0].copy()
    index = 1
    commands = [command]
    while index < len(path):
        command, index = advance_cartesian_reference(
            path, command, index, duration_s=.05,
            max_speed_m_s=speed, max_angular_speed_rad_s=angular)
        commands.append(command)
        assert len(commands) < 1000
    return commands


@pytest.mark.parametrize("resolution", [.004, .0007, .05])
def test_waypoint_density_does_not_consume_extra_ticks(resolution):
    path = cartesian_sweep(np.eye(4), displacement_m=[.35343, 0, 0],
                          max_position_step_m=resolution)
    commands = traverse(path)
    assert len(commands) - 1 == math.ceil(.35343 / .05 / .05)
    assert minimum_command_duration(path, .05, .5) == pytest.approx(7.0686)
    np.testing.assert_allclose(commands[-1], path[-1], atol=1e-12)
    assert max(np.linalg.norm(b[:3, 3]-a[:3, 3]) for a,b in zip(commands,commands[1:])) <= .0025+1e-12


def test_combined_translation_rotation_share_time_and_follow_path():
    path = cartesian_sweep(np.eye(4), displacement_m=[.03, 0, 0],
                          axis=[0, 0, 1], pivot_m=[0, 0, 0], angle_rad=.6,
                          max_position_step_m=.001)
    commands = traverse(path, speed=.05, angular=.2)
    assert len(commands)-1 == 60
    for a,b in zip(commands, commands[1:]):
        assert np.linalg.norm(b[:3,3]-a[:3,3]) <= .0025+1e-12
        assert Rotation.from_matrix(b[:3,:3]@a[:3,:3].T).magnitude() <= .01+1e-12
    for pose in commands:
        assert Rotation.from_matrix(pose[:3,:3]).magnitude() == pytest.approx(20*pose[0,3])


def test_real_execution_loop_no_longer_times_out_from_waypoint_quantization(tmp_path):
    robot = robot_at(tmp_path, FakeOscEnv())
    start = _pose(robot.get_robot_state().value)
    path = cartesian_sweep(start, displacement_m=[.35343, 0, 0])
    result = robot._execute_contact_sweep(
        path, max_speed_m_s=.05, max_angular_speed_rad_s=.5, max_duration_s=8,
        position_tolerance_m=.003, orientation_tolerance_rad=.03,
        precision_reference="absolute_pose_goal")
    assert result.success, result.value
    assert result.value["elapsed_s"] <= 8
    assert result.value["minimum_command_duration_s"] == pytest.approx(7.0686)
    assert result.value["endpoint_settled"]
    commands = np.asarray([r["commanded_eef_position_m"] for r in result.value["trajectory"]])
    np.testing.assert_allclose(commands[-1], path[-1,:3,3], atol=1e-12)


def test_duration_is_not_silently_extended(tmp_path):
    robot = robot_at(tmp_path, FakeOscEnv())
    start = _pose(robot.get_robot_state().value)
    path = cartesian_sweep(start, displacement_m=[.35343, 0, 0])
    result = robot._execute_contact_sweep(path, max_speed_m_s=.05, max_duration_s=1.01,
                                         precision_reference="absolute_pose_goal")
    assert not result.success and result.value["elapsed_s"] <= 1.01
    assert result.value["stopped_reason"] == "duration_exhausted"
    assert result.value["minimum_command_duration_s"] > result.value["effective_duration_limit_s"]

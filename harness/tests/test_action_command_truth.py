"""Correct commands must not be reversed by stale adapter state."""
import numpy as np
import pytest
from test_libero_adapter import FakeOscEnv
from captain_harness.adapters.libero import LiberoActionBackend


@pytest.mark.parametrize("initially_closed", [False, True])
def test_motion_preserves_interrupted_gripper_command(tmp_path, initially_closed):
    env=FakeOscEnv()
    robot=LiberoActionBackend(env,observation_dir=tmp_path,initial_observation=env.reset(),
        horizon=200,controller="OSC_POSE",gripper_steps=5)
    if initially_closed:
        assert robot.close_gripper().success
    start=robot.simulation_steps
    robot._motion_guard=lambda: robot.simulation_steps < start+1
    result=robot.open_gripper() if initially_closed else robot.close_gripper()
    assert not result.success and result.executed
    assert result.message == "motion_guard_rejected"
    first=len(env.actions)
    robot._motion_guard=None
    state=robot.get_robot_state().value
    moved=robot.move_to_pose(np.asarray(state.end_effector_position_m)+[0,0,.02],
        state.end_effector_quaternion_wxyz)
    assert moved.success
    assert len(env.actions)>first
    expected=-1. if initially_closed else 1.
    assert all(action[-1]==expected for action in env.actions[first:])


def test_noop_pose_does_not_report_simulation_execution(tmp_path):
    env=FakeOscEnv()
    robot=LiberoActionBackend(env,observation_dir=tmp_path,initial_observation=env.reset(),
        horizon=200,controller="OSC_POSE")
    state=robot.get_robot_state().value
    result=robot.move_to_pose(state.end_effector_position_m,state.end_effector_quaternion_wxyz)
    assert result.success
    assert "no simulation step required" in result.message
    assert not env.actions


@pytest.mark.parametrize("direction", [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],[2,-3,4]])
def test_linear_command_uses_base_frame_sign_and_normalized_distance(tmp_path,direction):
    env=FakeOscEnv()
    robot=LiberoActionBackend(env,observation_dir=tmp_path,initial_observation=env.reset(),
        horizon=200,controller="OSC_POSE")
    start=robot.get_robot_state().value
    r=robot.move_linear(direction,.02)
    assert r.success and r.executed
    after=robot.get_robot_state().value
    expected=np.asarray(direction)/np.linalg.norm(direction)*.02
    np.testing.assert_allclose(np.asarray(after.end_effector_position_m)-start.end_effector_position_m,expected,atol=.002)
    np.testing.assert_allclose(after.end_effector_quaternion_wxyz,start.end_effector_quaternion_wxyz)
    np.testing.assert_allclose(r.value["actual_displacement_m"],np.asarray(after.end_effector_position_m)-start.end_effector_position_m)


def test_rejected_gripper_does_not_change_next_motion_command(tmp_path):
    env=FakeOscEnv()
    robot=LiberoActionBackend(env,observation_dir=tmp_path,initial_observation=env.reset(),
        horizon=200,controller="OSC_POSE")
    robot._motion_guard=lambda:False
    result=robot.close_gripper()
    assert not result.success and not result.executed and not env.actions
    robot._motion_guard=None
    state=robot.get_robot_state().value
    assert robot.move_to_pose(np.asarray(state.end_effector_position_m)+[0,0,.02],state.end_effector_quaternion_wxyz).success
    assert all(action[-1]==-1. for action in env.actions)

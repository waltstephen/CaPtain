"""Hardware capability and nonexecution must agree with the published contract."""
import numpy as np
import pytest

from captain_harness.adapters.capx.manifest import build_capx_tool_manifest
from captain_harness.adapters.capx.runtime_manifest import runtime_manifest
from captain_harness.adapters.libero.backend import LiberoActionBackend
from captain_harness.ports.capability import CapabilityResult, FailureCategory
from test_libero_adapter import FakeOscEnv
from test_tool_backend import _fixture


def test_runtime_manifest_excludes_unexecutable_joint_command():
    base = build_capx_tool_manifest('S3', joint_count=7, service_aligned=True,
        dehardcoded=True, v15_protocol=True)
    osc = runtime_manifest(base, controller='OSC_POSE', motion_tools=False)
    joints = runtime_manifest(base, controller='JOINT_POSITION', motion_tools=False)
    assert 'move_to_joints' not in osc.names
    assert 'move_to_joints' in joints.names
    assert 'servo_to' not in joints.names
    assert 'align_pregrasp' not in joints.names
    assert 'descend_and_close' not in joints.names
    assert 'initial end-effector pose' in osc.require('home').description


def test_controller_rejection_preserves_previous_alignment(tmp_path):
    f = _fixture(tmp_path, dehardcoded=True, v15_protocol=True)
    previous = object()
    f.backend._aligned_pregrasp = previous
    f.robot.move_to_joints = lambda joints: CapabilityResult.failed(
        FailureCategory.CONTROLLER_REJECTED, 'not supported')
    result = f.backend.move_to_joints([0.] * 7)
    assert not result.executed
    assert f.backend._aligned_pregrasp is previous


@pytest.mark.parametrize('name, arguments', [
    ('goto_pose', {'position':[np.nan,0,0], 'quaternion':[1,0,0,0]}),
    ('servo_to', {'target_pose': {'position_m':[0,0,0], 'quaternion_wxyz':[1,0,0,0]},
                  'camera':'unknown', 'max_duration_s':1., 'tolerance_m':.01}),
    ('lift_probe', {'height': -1.}),
])
def test_invalid_motion_arguments_preserve_alignment(tmp_path, name, arguments):
    f = _fixture(tmp_path, dehardcoded=True, v15_protocol=True)
    previous = object(); f.backend._aligned_pregrasp = previous
    result = f.backend.invoke(name, arguments)
    assert not result.success and not result.executed
    assert f.backend._aligned_pregrasp is previous
    with pytest.raises(ValueError):
        f.backend.move_to_joints([np.nan] * 7)
    assert f.backend._aligned_pregrasp is previous


@pytest.mark.parametrize('value', [None, [np.nan, 0, 0], [1, 2]])
def test_force_missing_is_not_zero(tmp_path, value):
    env = FakeOscEnv()
    env.robots[0].ee_force = value
    robot = LiberoActionBackend(env, observation_dir=tmp_path,
        initial_observation=env.reset(), horizon=100, controller='OSC_POSE')
    result = robot.gripper_state()
    assert result.success and result.value['force_n'] is None
    assert result.value['force_available'] is False


def test_measured_zero_force_is_distinct_from_unavailable(tmp_path):
    env = FakeOscEnv()
    env.robots[0].ee_force = [0., 0., 0.]
    robot = LiberoActionBackend(env, observation_dir=tmp_path,
        initial_observation=env.reset(), horizon=100, controller='OSC_POSE')
    result = robot.gripper_state()
    assert result.value['force_n'] == 0.
    assert result.value['force_available'] is True
@pytest.mark.parametrize('with_depth', [False, True])
def test_single_camera_observation_and_missing_depth_are_explicit(tmp_path, with_depth):
    from test_libero_adapter import FakeEnv
    from captain_harness.adapters.libero.backend import LiberoActionBackend
    env = FakeEnv()
    observation = {k: v for k, v in env._get_observations().items()
                   if not k.startswith('robot0_eye_in_hand') and (with_depth or not k.endswith('_depth'))}
    robot = LiberoActionBackend(env, observation_dir=tmp_path,
                               initial_observation=observation, horizon=100)
    result = robot.get_observation_data()
    assert result.success, result.message
    assert [c.camera_name for c in result.value.cameras] == ['agentview']
    projected = robot.pixel_to_world('agentview', .5, .5)
    assert projected.success is with_depth
    if not with_depth:
        assert not projected.executed and 'no current depth' in projected.message

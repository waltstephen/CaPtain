from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from captain_harness.adapters.libero import LiberoActionBackend
from captain_harness.adapters.capx.backend import ToolCapabilityBackend
from captain_harness.adapters.capx.artifacts import EpisodeArtifactRegistry
from captain_harness.ports.capability import CapabilityResult
from test_libero_adapter import FakeOscEnv, RotationLockedAxisOscEnv


def fixture(tmp_path, drift):
    class Drifting(RotationLockedAxisOscEnv):
        def step(self, action):
            result = FakeOscEnv.step(self, action)
            self.yaw = drift
            return result
    env = Drifting()
    robot = LiberoActionBackend(env, observation_dir=tmp_path, initial_observation=env.reset(),
        horizon=3000, controller='OSC_POSE', cartesian_max_steps=80, guarded_cartesian_motion=True)
    tool = ToolCapabilityBackend(robot=robot, services=SimpleNamespace(),
        registry=EpisodeArtifactRegistry('compound-orientation', ttl_s=None),
        default_approach_frame='world', v15_protocol=True)
    tool._solve_ik_target = lambda *a, **kw:CapabilityResult.ok([0.]*7)
    return robot, tool


def test_compound_hold_does_not_tighten_incidental_orientation_drift(tmp_path):
    robot, tool = fixture(tmp_path, .003)
    public = tool._public_robot_state(robot.get_robot_state().value)
    result = tool.invoke('goto_pose', dict(position=(np.asarray(public['tool_position_m'])+[.02,0,0]).tolist(),
        quaternion=public['tool_quaternion_wxyz'], z_approach=.02, approach_frame='world', position_reference='tool'))
    assert result.success, result.message
    phases = result.value['phase_executions']
    for phase in ['approach', 'target']:
        actual = phases[phase]['value']['controller_execution']
        assert actual['reached'] is True, (phase, actual)
        assert actual['orientation_tolerance_rad'] == pytest.approx(.05)
        assert actual['control_steps'] < 80
    assert phases['target']['value']['controller_execution']['orientation_reference'] == 'compound_request'


@pytest.mark.parametrize('approach',[0.,.02])
@pytest.mark.parametrize('angle',[-.004,.004])
def test_explicit_small_rotation_failure_is_not_promoted_by_public_tolerance(tmp_path, approach, angle):
    robot, tool = fixture(tmp_path, 0.)
    public = tool._public_robot_state(robot.get_robot_state().value)
    requested = Rotation.from_rotvec([0,0,angle])*Rotation.from_quat(np.roll(public['tool_quaternion_wxyz'],-1))
    result = tool.invoke('goto_pose', dict(position=(np.asarray(public['tool_position_m'])+[.02,0,0]).tolist(),
        quaternion=np.roll(requested.as_quat(),1).tolist(), z_approach=approach, approach_frame='world', position_reference='tool'))
    assert result.executed and not result.success
    actual = result.value['controller_execution']
    assert actual['reached'] is False
    assert actual['orientation_tolerance_rad'] == pytest.approx(.0004)
    assert actual['orientation_residual_rad'] == pytest.approx(abs(angle))

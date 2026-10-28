from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from captain_harness.adapters.libero.recorded_backend import ClosingMonitoredPossessionActionBackend
from captain_harness.ports.capability import CapabilityResult, FailureCategory
from captain_harness.ports.deadline import deadline_scope
from test_libero_adapter import FakeOscEnv
from test_tool_backend import _fixture


@pytest.mark.parametrize('phase', ['descent', 'closure'])
@pytest.mark.parametrize('budget', [0., .25])
def test_compound_driver_preserves_shared_deadline_and_exact_physical_prefix(tmp_path, monkeypatch, phase, budget):
    import captain_harness.ports.deadline as mod
    clock = [100.]
    monkeypatch.setattr(mod.time, 'monotonic', lambda: clock[0])
    class ClockedEnv(FakeOscEnv):
        def step(self, action):
            result = super().step(action)
            if phase == 'closure':
                # Closing can disturb the arm even when no Cartesian motion
                # was commanded. Failure receipts must report the latest pose.
                self.joints[0] += .001
                result = (self._get_observations(), *result[1:])
            clock[0] += .1
            return result
    env = ClockedEnv()
    robot = ClosingMonitoredPossessionActionBackend(env, observation_dir=tmp_path,
        initial_observation=env.reset(), horizon=1000, controller='OSC_POSE')
    state = robot.get_robot_state().value
    target = np.asarray(state.end_effector_position_m) + ([0, 0, -.05] if phase=='descent' else [0, 0, 0])
    with deadline_scope('policy', expires_s=100.+budget):
        result = robot.descend_and_close(target, state.end_effector_quaternion_wxyz)
    assert not result.success and result.executed is (budget > 0)
    assert result.failure_category is FailureCategory.TIMEOUT
    assert result.diagnostics['deadline']['scope'] == 'policy'
    expected_steps = 3 if budget else 0
    assert robot.simulation_steps == expected_steps and len(env.actions) == expected_steps
    assert result.value['stopped_reason'] == 'deadline_expired'
    np.testing.assert_allclose(result.value['observed_eef_position_m'],
        robot.get_robot_state().value.end_effector_position_m, rtol=0, atol=1e-12)
    residual = target - np.asarray(robot.get_robot_state().value.end_effector_position_m)
    np.testing.assert_allclose(result.value['remaining_vector_m'], residual, rtol=0, atol=1e-12)
    assert result.value['failed_phase'] == phase
    assert result.value['closure_executed'] is (phase == 'closure' and budget > 0)
    if phase == 'descent':
        assert all(action[-1] == -1. for action in env.actions)
        assert result.value['controller_execution']['retreat_available'] is (budget > 0)
        assert result.value['position_residual_m'] > .003
    else:
        assert all(action[-1] == 1. for action in env.actions)
        assert not result.value['controller_execution']['retreat_available']


@pytest.mark.parametrize('tool', ['close_gripper', 'descend_and_close'])
@pytest.mark.parametrize('success', [False, True])
def test_compound_tool_wrapper_keeps_receipt_and_attempt_identity(tmp_path, tool, success):
    f = _fixture(tmp_path, v15_protocol=True)
    original = CapabilityResult(success, True, FailureCategory.NONE if success else FailureCategory.TIMEOUT,
        'completed' if success else 'deadline_expired', {'closure_executed': True, 'object_detected': True},
        attempt_identity={'attempt': 'physical-original'}, diagnostics={'driver_receipt': {'phase': 'closure'}})
    setattr(f.robot, tool, lambda *a, **kw: original)
    if tool == 'descend_and_close':
        f.backend._aligned_pregrasp = SimpleNamespace(eef_position=[0, 0, .5],
            eef_quaternion=[1, 0, 0, 0], tracking_handle=None, detection_id=None, candidate={})
        f.backend._pre_descent_centering = lambda aligned: (np.zeros(3), {'decision': 'retain'})
    result = getattr(f.backend, tool)()
    assert result.success is success and result.executed
    assert result.failure_category is original.failure_category
    assert result.message == original.message
    assert result.diagnostics == original.diagnostics
    assert result.attempt_identity == original.attempt_identity

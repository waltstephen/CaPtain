"""A relaxed intermediate-pose tolerance cannot override an execution stop."""
import pytest

from captain_harness.adapters.libero.backend import LiberoActionBackend
from captain_harness.ports.capability import CapabilityResult, FailureCategory
from test_libero_adapter import FakeOscEnv


@pytest.mark.parametrize('category', [FailureCategory.SAFETY_REJECTED,
    FailureCategory.CONTROLLER_REJECTED, FailureCategory.BACKEND_ERROR])
def test_non_timeout_stop_cannot_be_accepted_as_a_nearby_alignment_waypoint(tmp_path, category):
    env = FakeOscEnv()
    robot = LiberoActionBackend(env, observation_dir=tmp_path, initial_observation=env.reset(),
                               horizon=1000, controller='OSC_POSE')
    calls = []
    def stopped(*args, **kwargs):
        calls.append((args, kwargs))
        return CapabilityResult.failed(category, 'measured_stop', executed=True,
            value={'stopped_reason': 'contact_force', 'observed_force_n': 20.})
    robot._move_to_pose_osc = stopped
    robot._pose_residuals = lambda *args: (.001, .001)
    result = robot.align_pregrasp([.4, .1, .3], [1, 0, 0, 0], [0, 0, -1])
    assert not result.success and result.executed
    assert result.failure_category is category
    assert len(calls) == 1
    assert result.value['failed_phase'] == 'translate'

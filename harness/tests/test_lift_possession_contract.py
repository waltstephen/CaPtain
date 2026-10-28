"""Target possession and motion completion are independent public claims."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from captain_harness.adapters.capx.manifest import build_capx_tool_manifest
from captain_harness.adapters.capx.runtime_manifest import runtime_manifest
from captain_harness.ports.broker import ValidatedCapabilityBroker
from captain_harness.ports.capability import CapabilityResult, FailureCategory
from test_tool_backend import _fixture


def case(tmp_path, *, legacy=False):
    f = _fixture(tmp_path, dehardcoded=True, completion_evidence=True, v15_protocol=not legacy)
    f.backend._sensor_motion = not legacy
    f.robot.controller = 'OSC_POSE'
    f.robot.time_s = 0.
    f.robot.observation = replace(f.robot.observation,
        robot_state=replace(f.robot.observation.robot_state, gripper_open_fraction=.5))
    center = np.asarray(f.robot.get_robot_state().value.end_effector_position_m) - [0, 0, .01]
    track = SimpleNamespace(object_name='target', centroid=center, bounds_min=center-.03,
        bounds_max=center+.03, point_count=100, appearance_rgb_mean=np.zeros(3),
        attachment_verified=False, eef_to_object_transform=None,
        source_camera_name='agentview', source_detection_id='source')
    handle = f.backend.registry.put('entity_track', track)
    f.backend._pending_attachment_handle = handle
    f.backend._attachment_attempt_id = 1
    def probe(actual, success):
        def move(position, quaternion, **kwargs):
            state = f.robot.observation.robot_state
            final = np.asarray(state.end_effector_position_m) + [0, 0, actual]
            f.robot.observation = replace(f.robot.observation,
                robot_state=replace(state, end_effector_position_m=tuple(final)))
            f.robot.time_s += .35
            value = {'stopped_reason': 'target_reached' if success else 'contact_force',
                'reached': success, 'control_steps': 7, 'observed_force_n': 4. if success else 68.,
                'force_limit_n': 18., 'max_observed_force_n': 4. if success else 68.}
            return CapabilityResult(success, True,
                FailureCategory.NONE if success else FailureCategory.CONTROLLER_REJECTED,
                value['stopped_reason'], value,
                diagnostics={'physical_execution': {'control_steps': 7, 'status': 'executed'}})
        f.robot.move_to_pose_monitored = move
        f.robot.move_to_pose = move
    probe(.0185, True)
    first = f.backend.lift_probe(.02)
    assert first.success and first.value['supports_attachment']
    probe(.0322, False)
    second = f.backend.lift_probe(.05)
    assert not second.success and second.executed and not second.value['supports_attachment']
    state = f.robot.observation.robot_state
    obs = SimpleNamespace(robot_state=state, time_s=f.robot.time_s)
    observation_handle = f.backend.registry.put('observation', obs)
    lifted = center + [0, 0, .0507]
    mark = SimpleNamespace(centroid=lifted, bounds_min=lifted-.03, bounds_max=lifted+.03,
        point_count=100, appearance_rgb_mean=np.zeros(3), camera_name='agentview',
        observation_handle=observation_handle)
    mark_handle = f.backend.registry.put('marked_point', mark)
    f.backend.track_target = lambda *a: CapabilityResult.ok({'status': 'supported'})
    return f, track, handle, mark, mark_handle, second


def assert_held(f, handle, mark_handle):
    manifest = runtime_manifest(build_capx_tool_manifest('S3', joint_count=7,
        service_aligned=True, dehardcoded=True, completion_evidence=True, v15_protocol=True),
        controller='OSC_POSE', motion_tools=True)
    return ValidatedCapabilityBroker(manifest, f.backend.invoke).invoke('assert_entity_attached',
        {'tracking_handle': handle, 'current_point_handle': mark_handle})


def test_fresh_target_motion_can_confirm_possession_after_partial_force_stop(tmp_path):
    f, track, handle, mark, point, failed_lift = case(tmp_path)
    before_time, before_state = f.robot.time_s, f.robot.observation.robot_state
    f.robot._last_guarded_motion = saved = {'end_step': 14, 'poses': ['retained']}
    result = assert_held(f, handle, point)
    assert result.success, result.message
    value = result.value
    assert value['verified'] and value['pose_valid'] and value['pose_observation_accepted']
    assert value['reason'] == 'verified'
    assert value['attachment_evidence_basis'] == 'fresh_target_motion_and_gripper_register'
    assert value['lift_supports_attachment'] is False
    assert not value['lift_contradicts_attachment']
    assert value['lift_execution']['success'] is False
    assert value['lift_execution']['stopped_reason'] == 'contact_force'
    assert value['lift_execution']['observed_force_n'] == 68.
    assert value['motion_authorized'] is False
    assert 'contact' in value['motion_guidance'].lower()
    assert not failed_lift.success and failed_lift.executed
    assert f.robot.time_s == before_time and f.robot.observation.robot_state == before_state
    assert f.robot._last_guarded_motion is saved


@pytest.mark.parametrize('defect', ['stationary', 'wrong_appearance', 'empty_hand',
    'stale', 'incompatible_extent', 'different_view', 'contradiction'])
def test_partial_probe_does_not_override_negative_or_unresolved_evidence(tmp_path, defect):
    f, track, handle, mark, point, _ = case(tmp_path)
    if defect == 'stationary':
        mark.centroid = track.centroid.copy()
    elif defect == 'wrong_appearance':
        mark.appearance_rgb_mean = np.ones(3) * 200
    elif defect == 'empty_hand':
        f.backend._robot_gripper_state = lambda: CapabilityResult.ok({'object_detected': False, 'open_fraction': 0.})
    elif defect == 'stale':
        f.robot.time_s += .05
    elif defect == 'incompatible_extent':
        mark.bounds_min = mark.centroid - .001
        mark.bounds_max = mark.centroid + .001
    elif defect == 'different_view':
        mark.camera_name = 'robot0_eye_in_hand'
    else:
        f.backend._latest_lift_evidence = replace(f.backend._latest_lift_evidence,
            contradicts_attachment=True)
    result = assert_held(f, handle, point)
    assert not result.success or not result.value['verified']
    assert track.eef_to_object_transform is None and not track.attachment_verified


@pytest.mark.parametrize('defect', ['attempt', 'track'])
def test_unrelated_lift_is_not_claimed_as_evidence(tmp_path, defect):
    f, track, handle, mark, point, _ = case(tmp_path)
    changes = {'attachment_attempt_id': 0} if defect == 'attempt' else {'tracking_handle': 'another-track'}
    f.backend._latest_lift_evidence = replace(f.backend._latest_lift_evidence, **changes)
    # Without current target motion the unrelated lift cannot support possession.
    mark.centroid = track.centroid.copy()
    result = assert_held(f, handle, point)
    assert result.success and not result.value['verified']
    assert not result.value['lift_evidence_available']
    assert result.value.get('lift_execution') is None
    assert track.eef_to_object_transform is None


def test_legacy_inconclusive_lift_contract_is_unchanged(tmp_path):
    f, track, handle, mark, point, _ = case(tmp_path, legacy=True)
    result = f.backend.assert_entity_attached(handle, current_point_handle=point)
    assert result.success and not result.value['verified']
    assert result.value['reason'] == 'lift_probe_not_confirmed'


@pytest.mark.parametrize('executed', [False, True])
def test_lift_wrapper_keeps_lower_level_deadline_and_attempt_receipts(tmp_path, executed):
    f = _fixture(tmp_path, dehardcoded=True, completion_evidence=True, v15_protocol=True)
    f.backend._sensor_motion = True
    f.robot.controller = 'OSC_POSE'
    receipt = {'deadline': {'scope': 'episode', 'expires_s': 10.},
        'physical_execution': {'control_steps': 3 if executed else 0}}
    identity = {'attempt': 'original'}
    moved = CapabilityResult(False, executed, FailureCategory.TIMEOUT, 'deadline_expired',
        {'stopped_reason': 'deadline_expired', 'control_steps': 3 if executed else 0},
        attempt_identity=identity, diagnostics=receipt)
    f.robot.move_to_pose_monitored = lambda *a: moved
    result = f.backend.lift_probe(.05)
    assert not result.success and result.executed is executed
    assert result.diagnostics == receipt
    assert result.attempt_identity == identity
    assert result.value['controller_execution'] == moved.value


def test_possession_query_does_not_bypass_production_initial_force_guard(tmp_path):
    from types import MethodType
    from captain_harness.adapters.libero.contact_motion import ContactMotionMixin
    f, track, handle, mark, point, _ = case(tmp_path)
    assert assert_held(f, handle, point).value['verified']
    f.robot._eef_force_norm = lambda: 68.
    f.robot._execute_contact_sweep = MethodType(ContactMotionMixin._execute_contact_sweep, f.robot)
    before = f.robot.observation.robot_state
    result = ContactMotionMixin.move_linear(f.robot, [0, 0, 1], .01)
    assert not result.success and not result.executed
    assert result.value['stopped_reason'] == 'initial_force_limit'
    assert result.value['observed_force_n'] == 68.
    assert f.robot.observation.robot_state == before


def test_lift_initial_state_failure_keeps_receipt_and_dispatches_no_motion(tmp_path):
    f = _fixture(tmp_path)
    failure = CapabilityResult(False, False, FailureCategory.TIMEOUT, 'deadline_expired',
        diagnostics={'deadline': {'scope': 'episode'}})
    f.robot.get_robot_state = lambda: failure
    result = f.backend.lift_probe(.05)
    assert result.diagnostics == failure.diagnostics
    assert not result.success and not result.executed
    assert not f.robot.calls


def test_real_lift_driver_deadline_reaches_public_backend_without_late_steps(tmp_path, monkeypatch):
    import captain_harness.ports.deadline as deadline_module
    from captain_harness.ports.deadline import deadline_scope
    from captain_harness.adapters.libero import LiberoActionBackend
    from test_libero_adapter import FakeOscEnv
    clock = [100.]
    monkeypatch.setattr(deadline_module.time, 'monotonic', lambda: clock[0])
    class ClockedEnv(FakeOscEnv):
        def step(self, action):
            result = super().step(action)
            clock[0] += .1
            return result
    env = ClockedEnv()
    robot = LiberoActionBackend(env, observation_dir=tmp_path/'driver',
        initial_observation=env.reset(), horizon=1000, controller='OSC_POSE')
    f = _fixture(tmp_path/'tools', v15_protocol=True)
    f.backend.robot = robot
    f.backend._sensor_motion = True
    with deadline_scope('episode', timeout_s=.25):
        result = f.backend.invoke('lift_probe', {'height': .35})
    assert not result.success and result.executed
    assert result.failure_category is FailureCategory.TIMEOUT
    assert robot.simulation_steps == 3 and len(env.actions) == 3
    assert result.diagnostics['deadline']['scope'] == 'episode'
    assert result.value['controller_execution']['stopped_reason'] == 'deadline_expired'

"""Registry-owned grasps cross the validated RPC without copying native data."""
from copy import deepcopy

import pytest

from test_tool_backend import _fixture, _v9_selected_detection
from captain_harness.adapters.capx.manifest import build_capx_tool_manifest
from captain_harness.adapters.capx.runtime_manifest import runtime_manifest
from captain_harness.ports.broker import ValidatedCapabilityBroker


def issued(tmp_path, *, time_s=None):
    f = _fixture(tmp_path, dehardcoded=True, v15_protocol=True)
    if time_s is not None:
        from dataclasses import replace
        f.robot.observation = replace(f.robot.observation, time_s=time_s)
        f.robot.time_s = time_s
    obs, target = _v9_selected_detection(f, v15_protocol=True)
    manifest = runtime_manifest(build_capx_tool_manifest('S3', joint_count=7,
        service_aligned=True, dehardcoded=True, v15_protocol=True),
        controller='OSC_POSE', motion_tools=True)
    broker = ValidatedCapabilityBroker(manifest, f.backend.invoke)
    hand = broker.invoke('gripper_state', {})
    result = broker.invoke('grasp_pose_estimate', {
        'observation_handle': obs['observation_handle'], 'detection_id': target['detection_id'],
        'gripper_profile': hand.value['profile'], 'backend': 'auto', 'depth_source': 'sensor',
        'grasp_style': 'auto', 'top_k': 2})
    assert result.success, result.message
    return f, broker, result.value['candidates'][0], obs


@pytest.mark.parametrize('bare_string', [False, True])
def test_short_grasp_reference_executes_original_and_records_selected_descriptor(tmp_path, bare_string):
    f, broker, candidate, obs = issued(tmp_path)
    handle = candidate['grasp_handle']
    original = deepcopy(candidate)
    # Mutating a returned Python value cannot mutate the registry's pose.
    candidate['position_m'][0] += 1
    aligned = broker.invoke('align_pregrasp', {'candidate': handle if bare_string else {'grasp_handle': handle}})
    assert aligned.success, aligned.message
    closed = broker.invoke('descend_and_close', {})
    assert closed.success, closed.message
    tracked = f.backend.registry.get(original['tracking_handle'], expected_kind='entity_track')
    assert tracked.selected_grasp_descriptor == original
    assert original['source_observation_handle'] == obs['observation_handle']


def test_other_valid_candidate_on_same_track_cannot_bind_executed_placement(tmp_path):
    f, broker, original, _ = issued(tmp_path)
    assert broker.invoke('align_pregrasp', {'candidate': {'grasp_handle': original['grasp_handle']}}).success
    assert broker.invoke('descend_and_close', {}).success
    alternative = deepcopy(original)
    alternative.pop('grasp_handle')
    alternative['position_m'][0] += .01
    alternative = f.backend._issue_grasp_candidate(alternative)
    service_calls = []
    f.services.predict_placements = lambda **kw: service_calls.append(kw)
    result = f.backend.invoke('placement_estimate', {
        'tracking_handle': original['tracking_handle'],
        'target_detection_id': original['detection_id'],
        'destination_detection_id': original['detection_id'],
        'selected_grasp': {'grasp_handle': alternative['grasp_handle']},
        'reference_tolerance_m': .01,
    })
    assert not result.success and not result.executed
    assert 'executed_grasp_mismatch' in result.message
    assert not service_calls


def test_missing_attachment_is_reported_before_expensive_placement_service(tmp_path):
    f, broker, candidate, _ = issued(tmp_path)
    assert broker.invoke('align_pregrasp',{'candidate':candidate}).success
    assert broker.invoke('descend_and_close',{}).success
    calls = []
    f.services.predict_placements = lambda **kwargs:calls.append(kwargs)
    result = broker.invoke('placement_estimate',{
        'tracking_handle':candidate['tracking_handle'],
        'destination_detection_id':candidate['detection_id'], 'reference_tolerance_m':.01})
    assert not result.success and not result.executed
    assert 'attachment_pose_unknown' in result.message
    assert calls == []


def test_placement_can_reuse_executed_target_and_grasp_without_policy_copying(tmp_path):
    import numpy as np
    f, broker, candidate, _ = issued(tmp_path)
    assert broker.invoke('align_pregrasp', {'candidate': {'grasp_handle': candidate['grasp_handle']}}).success
    assert broker.invoke('descend_and_close', {}).success
    # This test supplies the independent attachment prerequisite explicitly;
    # closure alone must never manufacture an attachment transform.
    tracked = f.backend.registry.get(candidate['tracking_handle'], expected_kind='entity_track')
    tracked.attachment_verified = True
    tracked.eef_to_object_transform = np.eye(4)
    tracked.eef_to_object_transform[:3,3] = (
        tracked.centroid - np.asarray(f.robot.observation.robot_state.end_effector_position_m))
    recorded = []
    predict = f.services.predict_placements
    def capture(**arguments):
        recorded.append(arguments['selected_grasp'])
        return predict(**arguments)
    f.services.predict_placements = capture
    result = broker.invoke('placement_estimate', {
        'tracking_handle': candidate['tracking_handle'],
        'destination_detection_id': candidate['detection_id'],
        'reference_tolerance_m': .01})
    assert result.success, result.message
    assert recorded == [candidate['backend_candidate']]
    assert result.value['target_detection_id'] == candidate['detection_id']
    # This deliberately uses the same visible entity as destination: omitting
    # redundant bindings must not turn that into a correct placement claim.
    assert not result.value['identity_consistent']


def test_new_handle_for_identical_snapshot_can_supply_candidate_scene(tmp_path):
    from types import SimpleNamespace
    f, broker, candidate, old = issued(tmp_path, time_s=1.)
    new = f.backend.observe().value['observation_handle']
    assert new != old['observation_handle']
    f.backend._scene = lambda *args, **kwargs: SimpleNamespace(source_observation_handle=new)
    _, points = f.backend._candidate_scene_target(candidate, 'current_scene')
    assert len(points) >= 3
    captured = f.backend.registry.get(new, expected_kind='observation')
    # This changes the new camera only; an equal timestamp is insufficient.
    from dataclasses import replace
    camera = captured.cameras[0]
    changed = camera.image.copy()
    changed[0,0,0] ^= 1
    other = replace(captured, cameras=(replace(camera,image=changed), *captured.cameras[1:]))
    new = f.backend.registry.put('observation', other)
    with pytest.raises(ValueError, match='observation'):
        f.backend._candidate_scene_target(candidate, 'changed_scene')


@pytest.mark.parametrize('corruption', ['pose', 'diagnostic', 'foreign', 'wrong_kind', 'closed'])
def test_invalid_reference_cannot_move_or_replace_alignment(tmp_path, corruption):
    f, broker, candidate, _ = issued(tmp_path)
    ref = {'grasp_handle': candidate['grasp_handle']}
    if corruption == 'pose':
        ref = deepcopy(candidate)
        ref['position_m'][0] += .02
    elif corruption == 'diagnostic':
        ref = deepcopy(candidate)
        ref['ik_feasible'] = not ref['ik_feasible']
    elif corruption == 'foreign':
        parts = ref['grasp_handle'].split('/')
        parts[3] = 'different_episode'
        ref['grasp_handle'] = '/'.join(parts)
    elif corruption == 'wrong_kind':
        ref['grasp_handle'] = candidate['tracking_handle']
    else:
        f.backend.registry.close()
    previous = object()
    f.backend._aligned_pregrasp = previous
    before = list(f.robot.calls)
    result = broker.invoke('align_pregrasp', {'candidate': ref})
    assert not result.success and not result.executed
    assert f.robot.calls == before
    assert f.backend._aligned_pregrasp is previous


@pytest.mark.parametrize('pregrasp_feasible', [False, True])
@pytest.mark.parametrize('distance', [.06, .03])
def test_grasp_reachability_checks_execution_eef_and_pregrasp(tmp_path, pregrasp_feasible, distance):
    import numpy as np
    from types import SimpleNamespace
    from captain_harness.ports.capability import CapabilityResult, FailureCategory
    f, broker, candidate, _ = issued(tmp_path)
    if distance != .06:
        candidate = f.backend._issue_grasp_candidate({**candidate, 'approach_distance_m': distance})
    checked = []
    def solve(target, seed, **kwargs):
        checked.append(target.copy())
        return np.zeros(7) if len(checked) == 1 or pregrasp_feasible else None
    f.robot.planning_model = lambda: SimpleNamespace(solve_ik=solve, last_ik={'test': True})
    result = broker.invoke('check_reachable', {'pose': {'grasp_handle': candidate['grasp_handle']}})
    assert result.success, result.message
    assert result.value['reachable'] is pregrasp_feasible
    assert result.value['motion_path'] == 'not_checked'
    assert len(checked) == 2
    _, _, direction, ep, eq = f.backend._candidate_execution_pose(candidate)
    np.testing.assert_allclose(checked[0][:3, 3], ep)
    np.testing.assert_allclose(checked[1][:3, 3], np.asarray(ep) - np.asarray(direction) / np.linalg.norm(direction) * distance)
    planned = []
    f.backend._candidate_scene_target = lambda *_: (SimpleNamespace(aperture=1.), None)
    def plan(target, reference, scene_handle, **kwargs):
        planned.append((target, reference))
        return CapabilityResult.failed(FailureCategory.IK_FAILURE, 'fixture stops before execution')
    f.backend._plan_motion = plan
    f.backend._planned_alignment(candidate, 'fixture_scene')
    assert planned[0][1] == 'end_effector'
    np.testing.assert_allclose(planned[0][0]['position_m'], checked[1][:3, 3])


def test_grasp_inspection_uses_its_own_scene_and_keeps_freshness_checks(tmp_path):
    import numpy as np
    from dataclasses import replace
    from types import SimpleNamespace
    from captain_harness.adapters.robotics.planning import SensorScene
    from captain_harness.ports.capability import CapabilityResult
    f, broker, candidate, obs = issued(tmp_path)
    state = f.robot.observation.robot_state
    sources = []
    def build(observation_handle):
        sources.append(observation_handle)
        scene = SensorScene(np.zeros((3, 3)), observation_handle,
            state.joint_positions_rad, state.gripper_open_fraction, None, .003, 0)
        return CapabilityResult.ok({'scene_handle': f.backend.registry.put('sensor_scene', scene)})
    f.backend.build_scene = build
    f.robot.planning_model = lambda: SimpleNamespace(
        set_state=lambda *args: None, points_in_robot=lambda *args, **kwargs: (None, {}))
    result = broker.invoke('evaluate_grasp', {'candidate': {'grasp_handle': candidate['grasp_handle']}})
    assert result.success, result.message
    assert sources == [obs['observation_handle']]
    assert result.value['source_observation_handle'] == obs['observation_handle']
    assert result.value['scene_handle'].startswith('artifact://v1/')
    f.robot.observation = replace(f.robot.observation,
        robot_state=replace(state, joint_positions_rad=(.5,) * 7))
    stale = broker.invoke('evaluate_grasp', {'candidate': {'grasp_handle': candidate['grasp_handle']}})
    assert not stale.success and not stale.executed
    assert 'stale_scene' in stale.message

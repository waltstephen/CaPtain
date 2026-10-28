from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from test_v18_grasp_references import issued
from test_tool_backend import _v9_selected_detection
from captain_harness.adapters.robotics.planning import SensorScene
from captain_harness.ports.capability import CapabilityResult


def setup(tmp_path):
    f, broker, old, obs = issued(tmp_path, time_s=1.)
    f.robot.gripper_contact_envelope = lambda: {
        'pad_min_local_m':[-.01,-.04,.025], 'pad_max_local_m':[.01,.04,.055]}
    f.robot.observation = replace(f.robot.observation, time_s=2.)
    f.robot.time_s = 2.
    fresh, target = _v9_selected_detection(f, v15_protocol=True)
    state = f.robot.observation.robot_state
    scene = SensorScene(np.zeros((3,3)), fresh['observation_handle'],
        state.joint_positions_rad, state.gripper_open_fraction, 2., .003, 0)
    handle = f.backend.registry.put('sensor_scene', scene)
    return f, broker, old, fresh, target, handle


def test_refresh_uses_fresh_cloud_and_clears_old_authority(tmp_path):
    f, broker, old, fresh, target, scene = setup(tmp_path)
    before = deepcopy(old)
    seed, evidence = f.backend._refresh_grasp_seed(old, scene, target['detection_id'])
    assert old == before
    assert seed['detection_id'] == target['detection_id']
    assert seed['source_observation_handle'] == fresh['observation_handle']
    assert seed['tracking_handle'] != old['tracking_handle']
    assert seed.get('backend_candidate') is None and seed.get('model_score') is None
    assert 'feasibility_evidence' not in seed
    assert 'grasp_handle' not in seed
    assert seed['quaternion_wxyz'] == pytest.approx(old['quaternion_wxyz'])
    track = f.backend.registry.get(seed['tracking_handle'], expected_kind='entity_track')
    assert not track.attachment_verified
    from captain_harness.adapters.capx.motion_tools import pose_matrix
    _, local = f.backend._contact_local_points(pose_matrix(seed['position_m'], seed['quaternion_wxyz']), track.object_points)
    center = np.mean(np.percentile(local, [5,95], axis=0), axis=0)
    envelope = f.robot.gripper_contact_envelope()
    pad = (np.asarray(envelope['pad_min_local_m'])+envelope['pad_max_local_m'])/2
    pad[1] = 0
    np.testing.assert_allclose(center, pad, atol=1e-9)
    assert evidence['cross_observation_identity_verified'] is False
    assert evidence['prior_grasp_handle'] == old['grasp_handle']


def test_refresh_requires_issued_prior_and_matching_current_detection(tmp_path):
    f, broker, old, fresh, target, scene = setup(tmp_path)
    unissued = {k:v for k,v in old.items() if k != 'grasp_handle'}
    with pytest.raises(ValueError, match='issued'):
        f.backend._refresh_grasp_seed(unissued, scene, target['detection_id'])
    with pytest.raises(ValueError, match='observation'):
        f.backend._refresh_grasp_seed(old, scene, old['detection_id'])
    f.robot.time_s = 3.
    with pytest.raises(ValueError, match='stale_scene'):
        f.backend._refresh_grasp_seed(old, scene, target['detection_id'])


def test_broker_refresh_issues_even_when_seed_is_best_without_moving(tmp_path, monkeypatch):
    f, broker, old, fresh, target, scene = setup(tmp_path)
    monkeypatch.setattr('captain_harness.adapters.robotics.surfaces.support_plane', lambda *a: None)
    f.robot.planning_model = lambda: SimpleNamespace()
    f.backend._scene_collision_checker = lambda *a: (lambda q: {'collision':False}, {})
    f.backend._grasp_endpoint_reachability = lambda *a, **kw: {
        'reachable':True, 'joints':[0.]*7,
        'checks':{'contact':{'solution_returned':True}, 'pregrasp':{'solution_returned':True}}}
    f.backend.evaluate_grasp = lambda *a: CapabilityResult.ok({
        'contact':{'visible_point_count_in_pad_sweep':10}, 'approach':{'geometry_hits':{}}})
    calls = list(f.robot.calls)
    result = broker.invoke('refine_grasp', {'candidate':old['grasp_handle'],
        'scene_handle':scene, 'target_detection_id':target['detection_id']})
    assert result.success, result.message
    new = result.value['candidate']
    assert new['grasp_handle'] != old['grasp_handle']
    assert new['detection_id'] == target['detection_id']
    assert new['backend_candidate'] is None and new['model_score'] is None
    assert result.value['requires_decision'] is True
    assert result.value['target_refresh']['cross_observation_identity_verified'] is False
    assert f.backend._resolve_projected_grasp(new) == new
    assert f.backend._resolve_projected_grasp(old) == old
    assert f.robot.calls == calls
    stale = broker.invoke('refine_grasp', {'candidate':old['grasp_handle'], 'scene_handle':scene})
    assert not stale.success and not stale.executed
    assert 'observation' in stale.message

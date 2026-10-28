from types import SimpleNamespace

import numpy as np
import pytest

from captain_harness.ports.capability import CapabilityResult
from test_tool_backend import _fixture


def setup_case(tmp_path, *, partial=True, cross_view=False, stale=False):
    f = _fixture(tmp_path, dehardcoded=True, completion_evidence=True, v15_protocol=True)
    f.backend._sensor_motion = True
    f.robot.time_s = 2.
    state = f.robot.get_robot_state().value
    center = np.asarray(state.end_effector_position_m) - [0, 0, .03]
    track = SimpleNamespace(object_name='target', centroid=center, bounds_min=center-.05,
        bounds_max=center+.05, point_count=100, appearance_rgb_mean=np.zeros(3),
        attachment_verified=False, eef_to_object_transform=None,
        source_camera_name='agentview', source_detection_id='bound_detection')
    track_handle = f.backend.registry.put('entity_track', track)
    observation = SimpleNamespace(robot_state=state, time_s=1. if stale else 2.)
    observation_handle = f.backend.registry.put('observation', observation)
    scale = .25 if partial else 1.
    mark = SimpleNamespace(centroid=center.copy(), bounds_min=center-.05*scale,
        bounds_max=center+.05*scale, point_count=100, appearance_rgb_mean=np.zeros(3),
        camera_name='robot0_eye_in_hand' if cross_view else 'agentview', observation_handle=observation_handle)
    mark_handle = f.backend.registry.put('marked_point', mark)
    f.backend._robot_gripper_state = lambda: CapabilityResult.ok({'object_detected': True, 'open_fraction': .5})
    f.backend._lift_evidence_for = lambda *_: None
    # Strong apparent point motion is supplementary; it must not silently
    # authorize an object pose or possession from incompatible segmentations.
    f.backend.track_target = lambda *a: CapabilityResult.ok({'status': 'supported',
        'translation_m': [0, 0, .08], 'feature_count': 12, 'motion_spread_m': .001})
    return f, track, track_handle, mark_handle


@pytest.mark.parametrize('partial,cross_view', [(True, False), (False, True)])
def test_changed_visible_geometry_is_unresolved_and_never_auto_approved(tmp_path, partial, cross_view):
    f, track, handle, mark = setup_case(tmp_path, partial=partial, cross_view=cross_view)
    result = f.backend.assert_entity_attached(handle, current_point_handle=mark)
    assert result.success and not result.value['verified']
    assert result.value['reason'] == 'partial_geometry_motion_unresolved'
    assert result.value['target_motion_evidence']['status'] == 'supported'
    assert result.value['next_evidence_camera'] == 'agentview'
    assert track.eef_to_object_transform is None and not track.attachment_verified


def test_comparable_stationary_geometry_keeps_the_negative_target_motion_result(tmp_path):
    f, track, handle, mark = setup_case(tmp_path, partial=False)
    result = f.backend.assert_entity_attached(handle, current_point_handle=mark)
    assert not result.value['verified']
    assert result.value['reason'] == 'target_did_not_move'


def test_stale_mark_cannot_change_an_existing_attachment_transform(tmp_path):
    f, track, handle, mark = setup_case(tmp_path, stale=True)
    track.eef_to_object_transform = np.eye(4)
    track.attachment_verified = True
    result = f.backend.assert_entity_attached(handle, current_point_handle=mark)
    assert not result.success and not result.executed
    assert result.value['reason'] == 'stale_attachment_observation'
    np.testing.assert_array_equal(track.eef_to_object_transform, np.eye(4))
    assert track.attachment_verified


def test_partial_motion_feedback_survives_the_published_schema(tmp_path):
    from captain_harness.adapters.capx.manifest import build_capx_tool_manifest
    from captain_harness.adapters.capx.runtime_manifest import runtime_manifest
    from captain_harness.ports.broker import ValidatedCapabilityBroker
    f, track, handle, mark = setup_case(tmp_path)
    manifest = runtime_manifest(build_capx_tool_manifest('S3', joint_count=7,
        service_aligned=True, dehardcoded=True, completion_evidence=True, v15_protocol=True),
        controller='OSC_POSE', motion_tools=True)
    result = ValidatedCapabilityBroker(manifest, f.backend.invoke).invoke('assert_entity_attached',
        {'tracking_handle': handle, 'current_point_handle': mark})
    assert result.success, result.message
    assert not result.value['verified']
    assert result.value['reason'] == 'partial_geometry_motion_unresolved'


@pytest.mark.parametrize('motion', [[0.017,0,0.01855], [0.05,0,0.01855]])
def test_subthreshold_motion_is_not_reported_as_stationary_or_approved(tmp_path, motion):
    f, track, handle, mark = setup_case(tmp_path, partial=False)
    m=f.backend.registry.get(mark,expected_kind='marked_point')
    m.centroid=m.centroid+motion;m.bounds_min=m.bounds_min+motion;m.bounds_max=m.bounds_max+motion
    result=f.backend.assert_entity_attached(handle,current_point_handle=mark)
    assert result.success and not result.value['verified']
    assert result.value['reason'] in {'measured_motion_below_attachment_threshold','measured_lift_below_attachment_threshold'}
    assert result.value['motion_diagnosis']['stationary_proven'] is False
    assert result.value['motion_diagnosis']['retention_status']=='unverified'
    assert result.value['min_vertical_displacement_m']==.025
    assert result.value['min_displacement_m']>=.04

"""Counterexamples for placement corrections derived from partial public geometry."""
from dataclasses import replace

import numpy as np
import pytest


def _placement(tmp_path, offset=0.0):
    from test_tool_backend import _prepare_ordinary_placement_identity

    f, track, target, destination, tracked, mark = _prepare_ordinary_placement_identity(tmp_path)
    f.backend._dehardcoded = True
    f.backend._v15_protocol = True
    shift = np.full(3, offset)
    center = np.array([-.025, 0., .2]) + shift
    extent = np.array([.04, .04, .02])
    tracked.bounds_min = center - extent / 2
    tracked.bounds_max = center + extent / 2
    tracked.eef_to_object_transform[:3, 3] = center - np.array(f.robot.observation.robot_state.end_effector_position_m)
    mark.position = center.copy()
    mark.centroid = center.copy()
    mark.bounds_min = tracked.bounds_min.copy()
    mark.bounds_max = tracked.bounds_max.copy()
    dest = f.backend.registry.get(destination, expected_kind='marked_point')
    dest.bounds_min = np.array([0., -.1, .0]) + shift
    dest.bounds_max = np.array([.2, .1, .25]) + shift
    # A valid visible surface point is at the far edge, not the opening center.
    dest.position = np.array([.2, 0., .2]) + shift
    dest.camera_name = 'agentview'
    return f, track, target, destination, center, extent


@pytest.mark.parametrize('offset', [0., 1.5, -2.])
def test_container_edge_mark_yields_minimum_footprint_correction(tmp_path, offset):
    f, track, target, destination, center, extent = _placement(tmp_path, offset)
    result = f.backend.estimate_placement_relation(track, target, destination, relation_type='in')
    delta = np.array(result.value['recommended_eef_delta_m'])
    dest = f.backend.registry.get(destination, expected_kind='marked_point')
    assert delta == pytest.approx([.045, 0., 0.])
    assert np.all(center[:2] + delta[:2] - extent[:2] / 2 >= dest.bounds_min[:2] - 1e-9)
    assert np.all(center[:2] + delta[:2] + extent[:2] / 2 <= dest.bounds_max[:2] + 1e-9)
    assert result.value['correction_stage'] == 'align_xy'


def test_insufficient_visible_container_does_not_suggest_a_false_feasible_move(tmp_path):
    f, track, target, destination, _, _ = _placement(tmp_path)
    dest = f.backend.registry.get(destination, expected_kind='marked_point')
    dest.bounds_max[0] = dest.bounds_min[0] + .01
    result = f.backend.estimate_placement_relation(track, target, destination, relation_type='in')
    assert result.value['correction_stage'] == 'relocalize_required'
    assert result.value['correction_geometry_feasible'] is False
    assert result.value['relation_estimate'] is False
    assert result.value['recommended_eef_delta_m'] == pytest.approx([0, 0, 0])
    assert result.value['motion_allowed'] is True  # advisory, not a hidden policy gate


def test_target_identity_confidence_is_not_presented_as_placement_probability(tmp_path):
    f, track, target, destination, _, _ = _placement(tmp_path)
    result = f.backend.estimate_placement_relation(track, target, destination, relation_type='in')
    release = f.backend.authorize_safe_release(track, result.value['placement_estimate_handle']).value
    assert release['measurement_confidence'] == 1.
    assert 'target' in release['measurement_confidence_basis']
    assert release['placement_geometry_confidence'] is None
    assert release['confidence_calibrated'] is False
    assert release['destination_camera_name'] == 'agentview'


def test_release_refresh_reports_view_and_geometry_change_through_public_schema(tmp_path):
    from test_tool_backend import _invoke
    from captain_harness.adapters.capx.manifest import ToolApiLevel

    f, track, target, destination, _, _ = _placement(tmp_path)
    first = f.backend.estimate_placement_relation(track, target, destination, relation_type='in').value
    dest = f.backend.registry.get(destination, expected_kind='marked_point')
    new_obs = f.backend.observe().value['observation_handle']
    old_target = f.backend.registry.get(target, expected_kind='marked_point')
    target2 = f.backend.registry.put('marked_point', replace(old_target, observation_handle=new_obs))
    destination2 = f.backend.registry.put('marked_point', replace(dest,
        observation_handle=new_obs, camera_name='robot0_eye_in_hand',
        bounds_max=dest.bounds_max + [0., 0., .07]))
    result = _invoke(f, ToolApiLevel.S3_LOW_LEVEL, 'authorize_safe_release', {
        'tracking_handle': track, 'placement_estimate_handle': first['placement_estimate_handle'],
        'target_point_handle': target2, 'destination_point_handle': destination2,
    }, service_aligned=True, dehardcoded=True, completion_evidence=True, v15_protocol=True)
    assert result.success
    comparison = result.value['refresh_comparison']
    assert comparison['destination_camera_changed'] is True
    assert comparison['destination_bounds_max_change_m'] == pytest.approx(.07)
    assert comparison['estimated_target_displacement_m'] == pytest.approx(0.)
    assert comparison['previous_observation_handle'] != comparison['current_observation_handle']
    assert 'destination_view_changed' in result.value['quality_flags']


@pytest.mark.parametrize("change, camera, status", [
    ([.04, 0., 0.], "agentview", "below_observed_change"),
    ([0., 0., .07], "agentview", "exceeds_observed_change"),
    ([.04, 0., 0.], "robot0_eye_in_hand", "view_changed"),
])
def test_refresh_compares_correction_to_directional_geometry_change(tmp_path, change, camera, status):
    f, track, target, destination, _, _ = _placement(tmp_path)
    first = f.backend.estimate_placement_relation(track, target, destination, relation_type="in").value
    dest = f.backend.registry.get(destination, expected_kind="marked_point")
    obs = f.backend.observe().value["observation_handle"]
    target2 = f.backend.registry.put("marked_point", replace(
        f.backend.registry.get(target, expected_kind="marked_point"), observation_handle=obs))
    destination2 = f.backend.registry.put("marked_point", replace(dest,
        observation_handle=obs, camera_name=camera, bounds_min=dest.bounds_min-np.array(change)))
    from test_tool_backend import _invoke
    from captain_harness.adapters.capx.manifest import ToolApiLevel
    executed = _invoke(f, ToolApiLevel.S3_LOW_LEVEL, "authorize_safe_release", {
        "tracking_handle": track, "placement_estimate_handle": first["placement_estimate_handle"],
        "target_point_handle": target2, "destination_point_handle": destination2,
    }, service_aligned=True, dehardcoded=True, completion_evidence=True, v15_protocol=True)
    assert executed.success, executed.message
    result = executed.value
    assessment = result["correction_resolution"]
    assert assessment["status"] == status
    assert assessment["accuracy_guaranteed"] is False
    expected = .005 if change[0] else .045
    assert result["recommended_eef_delta_m"] == pytest.approx([expected,0,0])
    assert not result["safe_to_release"] and not result["semantic_ready"]
    if status == "below_observed_change":
        assert assessment["destination_change_along_correction_m"] == pytest.approx(.04)
        assert "correction_below_observed_geometry_change" in result["quality_flags"]


def test_no_comparison_does_not_claim_an_accurate_correction(tmp_path):
    f, track, target, destination, _, _ = _placement(tmp_path)
    result = f.backend.estimate_placement_relation(track, target, destination, relation_type="in").value
    assert result["correction_resolution"]["status"] == "not_assessed"
    assert result["correction_resolution"]["accuracy_guaranteed"] is False
    assert result["recommended_eef_delta_m"] == pytest.approx([.045,0,0])

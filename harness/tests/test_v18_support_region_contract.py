"""A support-surface mark denotes a point on the surface, not its center."""
import numpy as np
import pytest

from test_v17_6_placement_feedback import _placement


def support_fixture(tmp_path, *, center_x=.1, marker_x=.19, offset=0.):
    f, track, target, destination, _, extent = _placement(tmp_path, offset=offset)
    tracked = f.backend.registry.get(track, expected_kind='entity_track')
    mark = f.backend.registry.get(target, expected_kind='marked_point')
    center = np.array([center_x, 0., .27]) + offset
    tracked.bounds_min, tracked.bounds_max = center - extent/2, center + extent/2
    tracked.eef_to_object_transform[:3,3] = center - np.array(f.robot.observation.robot_state.end_effector_position_m)
    mark.position = mark.centroid = center.copy()
    mark.bounds_min, mark.bounds_max = tracked.bounds_min.copy(), tracked.bounds_max.copy()
    support = f.backend.registry.get(destination, expected_kind='marked_point')
    support.position = np.array([marker_x, 0., .25]) + offset
    return f, track, target, destination


@pytest.mark.parametrize('marker_x', [.01, .19])
@pytest.mark.parametrize('offset', [0., 1.5, -2.])
def test_on_relation_is_invariant_to_mark_location_on_same_support(tmp_path, marker_x, offset):
    f, track, target, destination = support_fixture(tmp_path, marker_x=marker_x, offset=offset)
    result = f.backend.estimate_placement_relation(track, target, destination, relation_type='on')
    assert result.success, result.message
    assert result.value['recommended_eef_delta_m'] == pytest.approx([0., 0., 0.])
    assert result.value['relation_estimate'] is True
    assert result.value['correction_stage'] == 'hold_for_release'


def test_on_correction_moves_only_enough_for_existing_half_footprint_requirement(tmp_path):
    f, track, target, destination = support_fixture(tmp_path, center_x=-.025)
    result = f.backend.estimate_placement_relation(track, target, destination, relation_type='on')
    # A 4 cm object needs its center at x=0 to obtain the existing 50% coverage.
    assert result.value['recommended_eef_delta_m'] == pytest.approx([.025, 0., 0.], abs=1e-7)
    assert result.value['relation_estimate'] is False


def test_insufficient_visible_support_cannot_propose_false_feasibility(tmp_path):
    f, track, target, destination = support_fixture(tmp_path)
    support = f.backend.registry.get(destination, expected_kind='marked_point')
    support.bounds_max[:2] = support.bounds_min[:2] + .01
    result = f.backend.estimate_placement_relation(track, target, destination, relation_type='on')
    assert result.value['correction_geometry_feasible'] is False
    assert result.value['correction_stage'] == 'relocalize_required'
    assert result.value['relation_estimate'] is False
    assert result.value['recommended_eef_delta_m'] == pytest.approx([0., 0., 0.])


@pytest.mark.parametrize('scale', [.1, 1., 10.])
def test_corner_correction_has_symmetric_minimum_and_scales_with_geometry(scale):
    from captain_harness.adapters.capx.support_geometry import minimum_support_translation

    # Square touches the support at one corner; equal axis displacements minimize
    # travel at the required product of overlaps. This tests the coupled XY case.
    delta = minimum_support_translation(np.array([-.02, -.02])*scale,
        np.array([.04, .04])*scale, np.zeros(2), np.ones(2)*scale)
    assert delta == pytest.approx(np.full(2, .04*np.sqrt(.5)*scale), abs=1e-7*scale)

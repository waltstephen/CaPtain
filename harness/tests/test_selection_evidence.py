"""Selection binds explicit public evidence independently of object wording."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from test_tool_backend import _fixture, _Detection, _MarkedPoint
from test_observation_equivalence import snapshot


def selection_fixture(tmp_path, prompt='small control beside a circular component', *, multiple=True):
    f = _fixture(tmp_path, dehardcoded=True, v15_protocol=True)
    registry = f.backend.registry
    observation = registry.put('observation', snapshot())
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:4, 1:4] = True
    target = _Detection(observation_handle=observation, camera_name='head', prompt=prompt,
        label='control', mask=mask, score=.9, box=None, backend='sam3_text',
        centroid=np.array([.4, .2, .05]), bounds_min=np.array([.35, .15, 0.]),
        bounds_max=np.array([.45, .25, .1]), point_count=100,
        appearance_rgb_mean=np.array([40., 40., 40.]))
    target_id = registry.put('detection', target)
    ids = [target_id]
    if multiple:
        peer_mask = np.zeros_like(mask)
        peer_mask[6:9, 6:9] = True
        peer = replace(target, mask=peer_mask, centroid=np.array([.6, -.1, .05]),
                       bounds_min=np.array([.55, -.15, 0.]), bounds_max=np.array([.65, -.05, .1]))
        ids.append(registry.put('detection', peer))
    f.backend._register_detection_groups(ids)
    point = _MarkedPoint(observation_handle=observation, camera_name='head',
        rgb_handle=registry.put('rgb', object()), depth_handle=registry.put('depth', np.ones((10,10))),
        x_norm=.2, y_norm=.2, pixel_xy=(2., 2.), position=target.centroid.copy(), label=prompt,
        mask=mask.copy(), segmentation_score=.9, mask_area_fraction=.09, depth_continuity=1.,
        geometry_diagonal_m=.1, appearance_rgb_mean=target.appearance_rgb_mean.copy(),
        centroid=target.centroid.copy(), bounds_min=target.bounds_min.copy(),
        bounds_max=target.bounds_max.copy(), point_count=100, grounding_status='grounded')
    return f.backend, target_id, target, point


@pytest.mark.parametrize('relation', ['beside', 'near', 'next to'])
def test_adjacent_description_does_not_require_nearest_reference(tmp_path, relation):
    backend, target_id, _, point = selection_fixture(tmp_path, f'control {relation} a circular component')
    handle = backend.registry.put('marked_point', point)
    result = backend.select_detection(target_id, identity_point_handle=handle)
    assert result.success, result.message
    assert result.value['identity_point_handle'] == handle
    assert result.value['instance_reference_distances_m'] is None


def test_plain_prompt_with_distinct_candidates_requires_identity(tmp_path):
    backend, target_id, _, _ = selection_fixture(tmp_path, 'control')
    result = backend.select_detection(target_id)
    assert not result.success
    assert result.value['reason'] == 'ambiguous_detection_requires_identity_provenance'


def test_single_candidate_still_binds_explicit_identity(tmp_path):
    backend, target_id, _, point = selection_fixture(tmp_path, 'control', multiple=False)
    handle = backend.registry.put('marked_point', point)
    result = backend.select_detection(target_id, identity_point_handle=handle)
    assert result.success and result.value['identity_point_handle'] == handle


@pytest.mark.parametrize('change', ['time', 'camera', 'pixel_and_mask', 'shape'])
def test_overlap_cannot_bypass_bad_identity_provenance(tmp_path, change):
    backend, target_id, target, point = selection_fixture(tmp_path, 'control on the panel')
    if change == 'time':
        other = snapshot()
        other.time_s += .05
        point = replace(point, observation_handle=backend.registry.put('observation', other))
    elif change == 'camera':
        point = replace(point, camera_name='wrist')
    elif change == 'shape':
        point = replace(point, pixel_xy=(9.,9.), mask=np.ones((3,3), dtype=bool))
    else:
        point = replace(point, pixel_xy=(8.,8.), mask=~target.mask)
    result = backend.select_detection(target_id, identity_point_handle=backend.registry.put('marked_point', point))
    assert not result.success
    assert target.identity_point_handle is None


def test_identical_public_capture_alias_accepts_same_point(tmp_path):
    backend, target_id, _, point = selection_fixture(tmp_path, 'control on the panel')
    other = deepcopy(backend.registry.get(point.observation_handle, expected_kind='observation'))
    # Pixel is valid while mark segmentation is too small to pass the IoU fallback.
    tiny = np.zeros_like(point.mask)
    tiny[2,2] = True
    point = replace(point, observation_handle=backend.registry.put('observation', other), mask=tiny)
    handle = backend.registry.put('marked_point', point)
    result = backend.select_detection(target_id, identity_point_handle=handle)
    assert result.success, result.message
    assert result.value['identity_point_handle'] == handle


def test_explicit_foreign_handle_is_never_ignored(tmp_path):
    from captain_harness.adapters.capx.artifacts import ArtifactHandleError
    backend, target_id, _, _ = selection_fixture(tmp_path/'a', 'control', multiple=False)
    other, _, _, point = selection_fixture(tmp_path/'b', 'control', multiple=False)
    foreign = other.registry.put('marked_point', point)
    with pytest.raises(ArtifactHandleError):
        backend.select_detection(target_id, identity_point_handle=foreign)


@pytest.mark.parametrize('prompt', ['control not closest to the disk',
    'control farther from the disk than its closest neighbor',
    'control beside the closest disk', 'control nearest to the disk or the tray'])
def test_negated_or_embedded_comparison_is_not_guessed(tmp_path, prompt):
    backend, target_id, _, point = selection_fixture(tmp_path, prompt)
    result = backend.select_detection(target_id, identity_point_handle=backend.registry.put('marked_point', point))
    assert result.success, result.message


def test_nearest_check_failure_does_not_bind_partial_selection(tmp_path):
    backend, target_id, target, point = selection_fixture(tmp_path, 'control closest to the disk')
    result = backend.select_detection(target_id, identity_point_handle=backend.registry.put('marked_point', point),
        selection_relation='nearest_xy')
    assert not result.success and result.value['reason'] == 'proximity_relation_requires_reference_point'
    assert not target.selected and target.identity_point_handle is None


def test_optional_adjacent_reference_on_another_mask_is_saved_without_argmin(tmp_path):
    backend, target_id, _, point = selection_fixture(tmp_path)
    handle = backend.registry.put('marked_point', point)
    reference = backend.registry.put('marked_point', replace(point, pixel_xy=(7.,7.)))
    result = backend.select_detection(target_id, identity_point_handle=handle, reference_point_handle=reference)
    assert result.success and result.value['reference_point_handle'] == reference
    assert result.value['selection_relation'] == 'identity'
    assert result.value['instance_reference_distances_m'] is None


def test_explicit_nearest_reference_accepts_an_identical_snapshot_alias(tmp_path):
    backend, target_id, _, point = selection_fixture(tmp_path, 'control closest to the disk')
    identity = backend.registry.put('marked_point', point)
    alias = backend.registry.put('observation', deepcopy(snapshot()))
    reference = backend.registry.put('marked_point', replace(point, observation_handle=alias,
        pixel_xy=(5.,5.), position=np.array([.45,.25,.05])))
    # This fixture has no semantic segmentation service for the reference;
    # only the separately requested public distance is measured here.
    result = backend.select_detection(target_id, identity_point_handle=identity,
        reference_point_handle=reference, selection_relation='nearest_xy')
    assert result.success, result.message
    assert result.value['instance_reference_distances_m'][target_id] < .08
    identity_only = backend.select_detection(target_id, identity_point_handle=identity)
    assert identity_only.success
    assert identity_only.value['instance_reference_distances_m'] is None
    assert identity_only.value['reference_point_handle'] is None

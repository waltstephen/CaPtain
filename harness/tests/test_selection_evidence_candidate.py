"""Exercise new public evidence checks before runtime integration."""
from dataclasses import replace

import pytest

from test_selection_evidence import selection_fixture
from test_observation_equivalence import snapshot
from captain_harness.adapters.capx.selection_evidence import bind_selection_evidence


@pytest.mark.parametrize('change', [None, 'time', 'camera'])
def test_candidate_binding_is_atomic(tmp_path, change):
    backend, target_id, target, point = selection_fixture(tmp_path)
    if change == 'time':
        other = snapshot()
        other.time_s += .05
        point = replace(point, observation_handle=backend.registry.put('observation', other))
    if change == 'camera':
        point = replace(point, camera_name='wrist')
    handle = backend.registry.put('marked_point', point)
    distractors, error = bind_selection_evidence(backend, target_id, target, handle, None)
    assert distractors
    if change is None:
        assert error is None and target.identity_point_handle == handle
    else:
        assert error is not None and target.identity_point_handle is None


def test_reference_failure_does_not_bind_good_identity_point(tmp_path):
    backend, target_id, target, point = selection_fixture(tmp_path)
    handle = backend.registry.put('marked_point', point)
    bad = backend.registry.put('marked_point', replace(point, camera_name='wrist'))
    _, error = bind_selection_evidence(backend, target_id, target, handle, bad)
    assert error.value['reason'] == 'reference_point_stale'
    assert target.identity_point_handle is None and target.reference_point_handle is None


def test_nearest_is_an_explicit_measurement_not_inferred_from_prompt(tmp_path):
    backend, target_id, target, point = selection_fixture(tmp_path, 'control closest to the disk')
    handle = backend.registry.put('marked_point', point)
    _, error = bind_selection_evidence(backend, target_id, target, handle, None, 'nearest_xy')
    assert error.value['reason'] == 'proximity_relation_requires_reference_point'
    assert target.identity_point_handle is None
    _, error = bind_selection_evidence(backend, target_id, target, handle, None, 'identity')
    assert error is None and target.identity_point_handle == handle

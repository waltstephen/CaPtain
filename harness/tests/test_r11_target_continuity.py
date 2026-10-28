from dataclasses import replace
from copy import deepcopy

import pytest

from test_selection_evidence import selection_fixture


def test_selection_relocalization_retains_original_anchor_and_correction(tmp_path):
    backend, first_id, first, _ = selection_fixture(tmp_path, multiple=False)
    backend._sensor_motion = True
    original = backend.select_detection(first_id)
    assert original.success
    latest = replace(first, selected=False, prompt='new visible part', label='new part')
    latest_id = backend.registry.put('detection', latest)
    result = backend.select_detection(latest_id, previous_detection_id=first_id,
        identity_update='corrected_target', identity_update_reason='The initial interpretation is contradicted by the wider view.')
    assert result.success, result.message
    record = result.value['target_continuity']
    assert record['initial_detection_id'] == first_id
    assert record['previous_detection_id'] == first_id
    assert record['identity_update'] == 'corrected_target'
    assert record['semantic_identity_status'] == 'unverified'
    assert len(result.value['recent_selections']) == 2


def test_invalid_target_revision_preserves_selection_and_history(tmp_path):
    backend, first_id, first, _ = selection_fixture(tmp_path, multiple=False)
    backend._sensor_motion = True
    assert backend.select_detection(first_id).success
    latest = replace(first, selected=False)
    latest_id = backend.registry.put('detection', latest)
    before = deepcopy(backend._selection_history)
    with pytest.raises(ValueError, match='reason'):
        backend.select_detection(latest_id, previous_detection_id=first_id,
            identity_update='corrected_target')
    assert not latest.selected
    assert backend._selection_history == before


def test_repeated_selection_is_not_a_new_identity_measurement(tmp_path):
    backend, first_id, _, _ = selection_fixture(tmp_path, multiple=False)
    backend._sensor_motion = True
    first = backend.select_detection(first_id)
    second = backend.select_detection(first_id)
    assert second.value['target_continuity'] == first.value['target_continuity']
    assert len(second.value['recent_selections']) == 1


def test_new_unlinked_target_keeps_prior_selection_visible(tmp_path):
    backend, first_id, first, _ = selection_fixture(tmp_path, multiple=False)
    backend._sensor_motion = True
    backend.select_detection(first_id)
    latest_id = backend.registry.put('detection', replace(first, selected=False, prompt='another part'))
    value = backend.select_detection(latest_id).value
    assert value['target_continuity']['previous_detection_id'] is None
    assert value['recent_selections'][0]['detection_id'] == first_id

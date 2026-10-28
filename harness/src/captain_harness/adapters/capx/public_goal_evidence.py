"""Goal-independent measurements from selected public visual observations.

Acquiring current and temporally separated evidence is a prerequisite for
verification, not a numeric task-success test. Open/closed, appliance states,
directional language and compound semantics remain visual verification questions.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from itertools import combinations
import math
import numpy as np


def _finite_time(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def measure_goal_observations(
    *, observation_handle: str, time_s: float | None, current_time_s: float | None,
    last_configuration_motion_time_s: float | None, subjects: Mapping,
    claims: Sequence[str], previous: Mapping | None = None,
    supporting_evidence: Sequence[Mapping] = (), min_sample_interval_s: float = .2,
) -> dict:
    if not (_finite_time(time_s) and _finite_time(current_time_s)) or time_s != current_time_s:
        raise ValueError('current public observation time is missing or stale; observe again')
    if not 1 <= len(subjects) <= 8:
        raise ValueError('provide one to eight selected visible subjects')
    if isinstance(claims, str) or not 1 <= len(claims) <= 8 or any(
        not isinstance(claim, str) or not claim.strip() or len(claim) > 512 for claim in claims
    ):
        raise ValueError('provide one to eight nonempty task-claim descriptions')
    if len(set(claims)) != len(claims) or len(supporting_evidence) > 8:
        raise ValueError('duplicate claims or too many historical evidence records')
    if not _finite_time(min_sample_interval_s) or min_sample_interval_s <= 0:
        raise ValueError('invalid public observation interval')
    records = []
    for handle, detection in subjects.items():
        if not detection.selected or detection.observation_handle != observation_handle:
            raise ValueError('all visible subjects must be selected from the current observation')
        centroid, lower, upper = (np.asarray(getattr(detection, field), float)
                                  for field in ('centroid', 'bounds_min', 'bounds_max'))
        valid = all(value.shape == (3,) and np.isfinite(value).all() for value in (centroid, lower, upper))
        valid = valid and bool(np.all(lower <= upper)) and detection.point_count >= 3
        records.append({'detection_id': handle, 'label': detection.label,
            'camera': detection.camera_name, 'source_observation_handle': observation_handle,
            'geometry_status': 'measured' if valid else 'insufficient',
            'visible_centroid_m': centroid.tolist() if valid else None,
            'visible_bounds_min_m': lower.tolist() if valid else None,
            'visible_bounds_max_m': upper.tolist() if valid else None,
            'measured_point_count': int(detection.point_count), 'frame': 'robot_base',
            'limitation': 'Selected visible surfaces only; these are not full object bounds or semantic identity proof.'})
    pairs = []
    for a, b in combinations(records, 2):
        if a['geometry_status'] != 'measured' or b['geometry_status'] != 'measured':
            continue
        pairs.append({'first_detection_id': a['detection_id'], 'second_detection_id': b['detection_id'],
            'frame': 'robot_base',
            'visible_centroid_delta_m': (np.asarray(a['visible_centroid_m'])-b['visible_centroid_m']).tolist(),
            'visible_separation_bounds_m': [
                (np.asarray(a['visible_bounds_min_m'])-b['visible_bounds_max_m']).tolist(),
                (np.asarray(a['visible_bounds_max_m'])-b['visible_bounds_min_m']).tolist()],
            'limitation': 'Signed XYZ geometry does not assign the task meaning of left/front/inside or a joint state.'})
    previous_time = previous.get('observation_time_s') if isinstance(previous, Mapping) else None
    previous_handle = previous.get('source_observation_handle') if isinstance(previous, Mapping) else None
    interval = time_s-previous_time if _finite_time(previous_time) else None
    sample_ready = bool(
        isinstance(previous_handle, str) and previous_handle != observation_handle
        and interval is not None and interval >= min_sample_interval_s
        and _finite_time(last_configuration_motion_time_s)
        and previous_time >= last_configuration_motion_time_s
    )
    history = []
    for item in supporting_evidence:
        if not isinstance(item, Mapping) or not isinstance(item.get('source_observation_handle'), str):
            raise ValueError('historical support must be an issued public evidence record')
        history.append({'scope': 'historical_only',
            **{key: deepcopy(item[key]) for key in ('source_observation_handle', 'observation_time_s',
                'evidence_handle', 'relation_type', 'relation_evidence', 'verification_requirements', 'evidence') if key in item},
            'measurements': {key: deepcopy(value) for key, value in item.get('public_evidence', {}).items()
                if key in ('subjects', 'pairwise_geometry')},
            'limitation': 'A previous relation may have changed after later actions; this record is not current confirmation.'})
    return {
        'evidence_contract': 'generic_acquisition_v1',
        'source_observation_handle': observation_handle, 'observation_time_s': time_s,
        'public_evidence': {'subjects': records, 'pairwise_geometry': pairs,
            'historical_support': history, 'previous_observation_handle': previous_handle,
            'sample_interval_s': interval, 'min_sample_interval_s': min_sample_interval_s,
            'last_configuration_motion_time_s': last_configuration_motion_time_s},
        'visual_evidence_sources': [
            *[{'source_id': item['source_observation_handle'], 'scope':'historical_support',
               'time_s':item.get('observation_time_s')} for item in supporting_evidence],
            *([{'source_id':previous_handle, 'scope':'previous_sample', 'time_s':previous_time}]
              if previous_handle is not None else []),
        ],
        'verification_requirements': [
            {'name': 'current_selected_visual_evidence', 'satisfied': True, 'status': 'supported'},
            {'name': 'separated_post_motion_samples', 'satisfied': sample_ready,
             'status': 'supported' if sample_ready else 'insufficient',
             'reason': 'Two timed samples acquired after the last configuration-changing action; this does not assert stability.'
                if sample_ready else 'Retain this sample, hold to advance dynamics, observe again and pass this evidence as the previous sample.'},
        ],
        'verification_questions': [
            {'name': 'full_task_satisfied', 'prompt':
             'Verify the entire original task, including every requested object, relation, device state and conjunction. '
             'The supplied claim list may be incomplete or mistaken. Use current images and any required historical evidence; '
             'acquisition prerequisites, selected labels and geometric overlap alone do not prove completion.'},
            *[{'name': f'goal_claim_{index+1}', 'prompt':
               f'Check this proposed task claim against the original task and raw evidence: {claim.strip()} '
               'Assess object identity, the relevant state/relation and whether it still holds now. '
               'If the relevant state is hidden or ambiguous, explain any temporal evidence or return insufficient.'}
              for index, claim in enumerate(claims)],
        ],
        'relation_evidence': {'status': 'unassessed', 'reason':
            'This tool measures public observations and acquisition prerequisites. Visual semantic verification is still required.'},
        'assessment_basis': 'public_rgbd_and_robot_time_only',
    }


def _goal_sample_scope(backend, subject_ids, claims):
    from .attempt_identity import _selection_reference
    # Claims remain exact; public correspondence only links issued references.
    if isinstance(claims, str) or any(not isinstance(c, str) for c in claims):
        return None
    return {"subject_references": sorted(_selection_reference(backend, handle, require_selection_overlap=True)["reference"]
                for handle in subject_ids), "claims": [claim.strip() for claim in claims]}


def _prior_goal_sample(backend, scope, current_time):
    from .artifacts import ArtifactHandleError
    last_motion = getattr(backend, '_last_configuration_motion_time_s', 0.)
    if scope is None or not _finite_time(current_time) or not _finite_time(last_motion):
        return None, None
    for row in reversed(getattr(backend, '_goal_sample_index', ())):
        if (row['scope'] != scope or not _finite_time(row['time_s'])
                or row['time_s'] < last_motion or current_time - row['time_s'] < .2):
            continue
        try:
            candidate = backend.registry.get(row['handle'], expected_kind='goal_evidence')
        except ArtifactHandleError:
            continue
        if candidate.get('source_observation_handle') not in getattr(backend, '_public_observation_history', {}):
            continue
        unloaded = any(item.get('name') == 'unloaded_gripper' and item.get('satisfied') is True
            for item in candidate.get('verification_requirements', ()))
        if not unloaded:
            continue
        return candidate, row['handle']
    return None, None


def _retain_goal_sample(backend, scope, time_s, handle):
    if scope is None or not _finite_time(time_s):
        return
    index = [row for row in getattr(backend, '_goal_sample_index', ())
        if row['scope'] != scope or row['time_s'] != time_s]
    index.append({'scope': deepcopy(scope), 'time_s': time_s, 'handle': handle})
    backend._goal_sample_index = index[-64:]


def collect_goal_evidence(
    backend, observation_handle, subject_detection_ids, claims,
    previous_evidence_handle=None, supporting_evidence_handles=(),
):
    """Resolve every input from the current episode registry, then measure.

    Policies provide handles, never fabricated previous evidence dictionaries.
    """
    from captain_harness.ports.capability import CapabilityResult
    if len(set(subject_detection_ids)) != len(subject_detection_ids):
        raise ValueError('subject detection handles must be unique')
    observation = backend.registry.get(observation_handle, expected_kind='observation')
    subjects = {handle: backend._require_selected_detection(handle, observation_handle=observation_handle)
                for handle in subject_detection_ids}
    scope = _goal_sample_scope(backend, subject_detection_ids, claims)
    if previous_evidence_handle is not None:
        previous = backend.registry.get(previous_evidence_handle, expected_kind='goal_evidence')
        resolution = {'mode': 'explicit', 'evidence_handle': previous_evidence_handle}
    else:
        previous, previous_handle = _prior_goal_sample(backend, scope, observation.time_s)
        resolution = {'mode': 'automatic' if previous is not None else 'unavailable',
            'evidence_handle': previous_handle,
            'basis': 'Same exact claims and issued subjects linked by supported public visual correspondence; '
                'unloaded timed samples after the last configuration motion. This does not prove task identity or stability.'}
    supporting = [backend.registry.get(handle, expected_kind='goal_evidence')
                  for handle in supporting_evidence_handles]
    value = measure_goal_observations(observation_handle=observation_handle,
        time_s=observation.time_s, current_time_s=getattr(backend.robot, 'time_s', None),
        last_configuration_motion_time_s=getattr(backend, '_last_configuration_motion_time_s', 0.),
        subjects=subjects, claims=claims, previous=previous, supporting_evidence=supporting)
    hand = backend.gripper_state()
    hand_value = hand.value if hand.success and isinstance(hand.value, Mapping) else {}
    unloaded = hand_value.get('object_detected') is False and not backend._active_attachment_handle
    value['verification_requirements'].append({
        'name': 'unloaded_gripper', 'satisfied': unloaded,
        'status': 'supported' if unloaded else 'insufficient',
        'reason': 'Public gripper state reports no retained object and no active attachment.' if unloaded
            else 'Release the manipulated item or fixture and recheck the public gripper state before final verification.',
    })
    value['public_evidence']['gripper'] = {
        key: hand_value.get(key) for key in ('open_fraction', 'object_detected', 'target_attachment_status')}
    from .target_continuity import add_goal_identity_evidence
    add_goal_identity_evidence(backend, value, subject_detection_ids)
    value['previous_evidence_resolution'] = resolution
    value['goal_sample_scope'] = scope
    value['evidence_handle'] = backend.registry.put('goal_evidence', deepcopy(value))
    _retain_goal_sample(backend, scope, observation.time_s, value['evidence_handle'])
    return CapabilityResult.ok(value)

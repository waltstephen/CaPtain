"""Episode-local selection provenance; labels and links are revisable claims."""
from copy import deepcopy


def prepare_selection(backend, detection_id, detection, previous_detection_id=None,
                      identity_update='same_target', identity_update_reason=''):
    if identity_update not in {'same_target', 'corrected_target'}:
        raise ValueError('identity_update must be same_target or corrected_target')
    if not isinstance(identity_update_reason, str) or len(identity_update_reason) > 512:
        raise ValueError('identity_update_reason must be text of at most 512 characters')
    if identity_update == 'corrected_target' and (
            not previous_detection_id or not identity_update_reason.strip()):
        raise ValueError('corrected_target requires a previous detection and a public-evidence reason')
    history = getattr(backend, '_selection_history', [])
    existing = next((r for r in history if r['detection_id'] == detection_id), None)
    if existing is not None:
        if previous_detection_id is not None and (
                previous_detection_id != existing['previous_detection_id']
                or identity_update != existing['identity_update']
                or identity_update_reason != existing['identity_update_reason']):
            raise ValueError('selection provenance is immutable; obtain a new detection to revise it')
        return deepcopy(existing)
    previous = None
    if previous_detection_id is not None:
        if previous_detection_id == detection_id:
            raise ValueError('previous detection must be a different issued selection')
        backend._require_selected_detection(previous_detection_id)
        previous = next((r for r in history if r['detection_id'] == previous_detection_id), None)
        if previous is None:
            raise ValueError('previous detection has no recorded selection; select it first')
    correspondence = {'status':'insufficient','reason':'no_previous_selection'}
    if previous is not None:
        old_detection = backend._require_selected_detection(previous_detection_id)
        if old_detection.camera_name == detection.camera_name:
            from .visual_tracking import track_rgbd
            old = backend.registry.get(old_detection.observation_handle, expected_kind='observation')
            new = backend.registry.get(detection.observation_handle, expected_kind='observation')
            correspondence = track_rgbd(backend._camera_by_name(old, detection.camera_name),
                backend._camera_by_name(new, detection.camera_name), old_detection.mask,
                target_mask=detection.mask)
        else:
            correspondence = {'status':'insufficient','reason':'different_cameras_require_visual_identity_assessment'}
    return {
        'detection_id': detection_id, 'source_observation_handle': detection.observation_handle,
        'camera': detection.camera_name, 'description': detection.prompt,
        'previous_detection_id': previous_detection_id,
        'initial_detection_id': previous['initial_detection_id'] if previous else detection_id,
        'initial_observation_handle': previous['initial_observation_handle'] if previous else detection.observation_handle,
        'identity_update': identity_update if previous else 'initial_selection',
        'identity_update_reason': identity_update_reason,
        'semantic_identity_status': 'unverified',
        'visual_correspondence': correspondence,
        'limitation': 'Descriptions and explicit links are policy claims, not visual identity proof. '
            'Compare the initial scene, distinguishing parts and current images; revise with public evidence.',
    }


def record_selection(backend, record):
    if not hasattr(backend, '_selection_history'):
        backend._selection_history = []
    if not any(r['detection_id'] == record['detection_id'] for r in backend._selection_history):
        backend._selection_history.append(deepcopy(record))
    return {'target_continuity': deepcopy(record),
            'recent_selections': deepcopy(backend._selection_history[-8:])}


def add_goal_identity_evidence(backend, value, subject_ids):
    history = getattr(backend, '_selection_history', [])
    selected = [r for r in history if r['detection_id'] in subject_ids]
    value['public_evidence']['selection_history'] = deepcopy(history[-16:])
    value['public_evidence']['subject_identity_links'] = deepcopy(selected)
    value['visual_evidence_sources'] = [
        *[{'source_id': r['initial_observation_handle'], 'scope': 'target_identity_anchor'}
          for r in selected], *value['visual_evidence_sources']]
    value['verification_questions'].insert(0, {
        'name': 'task_target_identity',
        'required_evidence_roles': ['initial_identity_reference', 'current'],
        'prompt': 'Identify the exact object or part requested by the ORIGINAL TASK in the initial '
            'scene, then match it to the current manipulated part. Cite initial and current images '
            'and explain distinguishing neighboring parts or appearance. For visually similar parts, '
            'enumerate the visible alternatives and their relative arrangement before identifying the '
            'requested one; state which parts are occluded. Do not infer ordinal position from a cropped view. The selection history '
            'may contain unlinked relocalizations, corrected targets or changed descriptions; '
            'these are claims, not proof. Do not infer identity from the latest label. '
            'If correspondence is ambiguous, return insufficient; if a different part was used, contradicted.',
    })
    return value

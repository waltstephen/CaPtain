"""Public attachment evidence gaps, never a task plan or relaxed verifier."""


def attachment_evidence_gaps(*, current_mark, source_camera, extent_comparable,
        appearance_consistent, appearance_available, identity_checks, gripper_detected, translation_m,
        vertical_m, distance_m, min_translation_m, min_vertical_m, max_distance_m):
    same_view = (current_mark.camera_name == source_camera
        if current_mark is not None and source_camera is not None else None)
    comparable = bool(current_mark is not None and same_view and extent_comparable and appearance_consistent)
    def check(name, passed, **values):
        return {'name': name, 'status': 'unknown' if passed is None else 'met' if passed else 'unmet', **values}
    checks = [check('current_target_mark', current_mark is not None),
        check('same_camera_coverage', same_view if current_mark is not None else None),
        check('visible_extent_comparable', extent_comparable if current_mark is not None else None),
        check('appearance_consistent', appearance_consistent if current_mark is not None and appearance_available else None),
        check('public_identity_checks', identity_checks if current_mark is not None else None),
        check('gripper_register_detects_object', gripper_detected),
        check('target_translation', translation_m >= min_translation_m if comparable else None,
              measured_m=translation_m, threshold_m=min_translation_m),
        check('target_vertical_translation', vertical_m >= min_vertical_m if comparable else None,
              measured_m=vertical_m, threshold_m=min_vertical_m),
        check('target_eef_distance', distance_m <= max_distance_m if comparable else None,
              measured_m=distance_m, threshold_m=max_distance_m)]
    return {'checks': checks, 'unmet_checks': [c['name'] for c in checks if c['status']=='unmet'],
        'unknown_checks': [c['name'] for c in checks if c['status']=='unknown'],
        'source_camera': source_camera,
        'current_camera': current_mark.camera_name if current_mark is not None else None,
        'current_observation_handle': current_mark.observation_handle if current_mark is not None else None,
        'sustained_retention_verified': False, 'task_recovery_verified': False,
        'scope': 'Explains current public measurements and unchanged acceptance thresholds. Uncomparable surface centroids cannot distinguish failed target motion from missing evidence. Gripper detection and public identity checks are not independent semantic identity proof; no prescribed next action.'}

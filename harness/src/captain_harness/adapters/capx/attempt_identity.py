"""Public issued-reference association for attempt memory, never action authorization."""
from collections.abc import Mapping
from .artifacts import ArtifactHandleError
from captain_harness.ports.capability import SideEffect


def _selection_reference(backend, detection_id, *, require_selection_overlap=False):
    detection = backend.registry.get(detection_id, expected_kind="detection")
    if not detection.selected:
        raise ValueError("attempt target must be an issued selected detection")
    history = {row["detection_id"]: row for row in getattr(backend, "_selection_history", [])}
    reference = detection_id
    visited = set()
    while reference in history and reference not in visited:
        visited.add(reference)
        row = history[reference]
        previous = row.get("previous_detection_id")
        if (row.get("identity_update") != "same_target" or not previous or previous not in history
                or row.get("visual_correspondence", {}).get("status") != "supported"
                or (require_selection_overlap
                    and row.get("visual_correspondence", {}).get("association_status") != "supported")):
            break
        reference = previous
    return {"reference": reference, "source_reference": detection_id,
        "basis": "issued_selection" if reference == detection_id else "explicit_link_with_public_visual_correspondence",
        "semantic_identity_status": "unverified"}


def _track_reference(backend, handle):
    tracked = backend.registry.get(handle, expected_kind="entity_track")
    detection = getattr(tracked, "source_detection_id", None)
    if detection:
        return _selection_reference(backend, detection)
    source = getattr(tracked, "source_point_handle", None)
    if source:
        backend.registry.get(source, expected_kind="marked_point")
    return {"reference": source or handle, "source_reference": handle,
        "basis": "issued_point_track" if source else "issued_entity_track",
        "semantic_identity_status": "unverified"}


def _with_execution_parameters(identity, candidate):
    if not isinstance(candidate, Mapping):
        return identity
    from copy import deepcopy
    # Whitelist public selected geometry. Handles, scores and native payload
    # are deliberately not treated as an execution strategy change.
    fields = ('position_m', 'quaternion_wxyz', 'approach_direction_world', 'approach_distance_m')
    values = {key: deepcopy(candidate[key]) for key in fields if key in candidate}
    return {**identity, 'execution_parameters': values,
        'execution_parameter_scope': 'Selected candidate geometry before execution and centering; not proof of executed contact.'}


def resolve_attempt_identity(backend, spec, arguments):
    if spec.side_effect not in {SideEffect.ROBOT_MOTION, SideEffect.ACTUATOR}:
        return None
    try:
        if arguments.get("target_detection_id") is not None:
            return _selection_reference(backend, arguments["target_detection_id"])
        if arguments.get("tracking_handle") is not None:
            return _track_reference(backend, arguments["tracking_handle"])
        if spec.name in {"align_pregrasp", "approach_and_close"}:
            candidate = arguments.get("grasp_handle") or arguments.get("candidate")
            if isinstance(candidate, str):
                candidate = backend.registry.get(candidate, expected_kind="grasp_candidate")
            elif isinstance(candidate, Mapping) and candidate.get("grasp_handle"):
                candidate = backend.registry.get(candidate["grasp_handle"], expected_kind="grasp_candidate")
            else:
                return None
            if isinstance(candidate, Mapping):
                if candidate.get("detection_id"):
                    return _selection_reference(backend, candidate["detection_id"])
                if candidate.get("tracking_handle"):
                    return _track_reference(backend, candidate["tracking_handle"])
            return None
        if spec.name == "descend_and_close":
            aligned = getattr(backend, "_aligned_pregrasp", None)
            if aligned is not None:
                if aligned.detection_id:
                    return _with_execution_parameters(_selection_reference(backend, aligned.detection_id),
                        getattr(aligned, 'candidate', None))
                if aligned.tracking_handle:
                    return _with_execution_parameters(_track_reference(backend, aligned.tracking_handle),
                        getattr(aligned, 'candidate', None))
            return None
        if spec.name in {"lift_probe", "goto_pose", "move_linear", "rotate_about_axis", "servo_to",
                         "hold_position", "execute_motion", "open_gripper", "retreat_last_motion"}:
            handle = getattr(backend, "_pending_attachment_handle", None) or getattr(backend, "_active_attachment_handle", None)
            if handle:
                return _track_reference(backend, handle)
    except (ArtifactHandleError, ValueError, TypeError, KeyError):
        # An invalid/ambiguous reference must not borrow the previously active target.
        return None
    return None

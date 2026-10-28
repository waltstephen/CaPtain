"""Public source checks and conservative parsing for instance selection.

These checks bind evidence; they do not establish general language semantics.
"""
from __future__ import annotations

import numpy as np

from .observation_equivalence import same_public_snapshot
from captain_harness.ports.capability import CapabilityResult, FailureCategory


def point_source_error(registry, detection, marked) -> str | None:
    """Neither overlapping masks nor 3-D proximity can waive source identity."""
    if marked.camera_name != detection.camera_name:
        return 'camera_mismatch'
    if marked.observation_handle != detection.observation_handle:
        source = registry.get(detection.observation_handle, expected_kind='observation')
        point_source = registry.get(marked.observation_handle, expected_kind='observation')
        if not same_public_snapshot(source, point_source):
            return 'public_snapshot_mismatch'
    return None


def identity_point_error(registry, detection, marked) -> tuple[str | None, float]:
    source_error = point_source_error(registry, detection, marked)
    if source_error is not None:
        return source_error, 0.0
    mask = np.asarray(detection.mask, dtype=bool)
    point_mask = np.asarray(marked.mask, dtype=bool)
    pixel = np.asarray(marked.pixel_xy, dtype=float)
    if mask.ndim != 2 or point_mask.shape != mask.shape or pixel.shape != (2,) or not np.isfinite(pixel).all():
        return 'invalid_point_geometry', 0.0
    x, y = np.rint(pixel).astype(int)
    union = np.count_nonzero(mask | point_mask)
    iou = float(np.count_nonzero(mask & point_mask) / union) if union else 0.0
    matches = 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]
    return (None if matches or iou >= .20 else 'point_mask_mismatch'), iou


def selection_recovery(backend, detection_id, detection, marked, error):
    """Explain binding failures using only issued same-source geometry; never auto-select."""
    matching = []
    if error == "point_mask_mismatch":
        group = (detection.observation_handle, detection.camera_name, detection.prompt.casefold())
        for peer_id in backend._detection_groups.get(group, ()):
            peer = backend.registry.get(peer_id, expected_kind="detection")
            if identity_point_error(backend.registry, peer, marked)[0] is None:
                matching.append(peer_id)
    source = lambda item: {"observation_handle": item.observation_handle, "camera_name": item.camera_name}
    return {"evidence_error": error, "selection_required": True,
        "detection_source": source(detection), "point_source": source(marked),
        "point_pixel_xy": list(marked.pixel_xy), "matching_detection_ids": matching[:8],
        "guidance": ("These candidates geometrically match the point in the same public snapshot/camera. "
            "Inspect the image and explicitly select the intended instance; geometric agreement is not "
            "semantic identity. Do not repeat the rejected point/detection pair." if matching else
            "Use one public snapshot and camera for both point and segmentation. If the point is correct "
            "but no candidate covers it, inspect point-guided geometry or resegment the intended part. "
            "If the service point is wrong, mark_point from the visible image. Do not invent a pixel "
            "when the target is hidden or repeat the same mismatched pair.")}


def bind_selection_evidence(backend, detection_id, detection, identity_point_handle, reference_point_handle,
                            selection_relation="identity"):
    """Validate the whole selection before committing any identity binding."""
    if selection_relation not in {"identity", "nearest_xy"}:
        raise ValueError("selection_relation must be identity or nearest_xy")
    distractors = backend._identity_distractors(detection_id, detection) if backend._v15_protocol else ()

    def rejected(reason, message, **detail):
        return distractors, CapabilityResult.failed(
            FailureCategory.PERCEPTION_UNCERTAIN, f'{reason}: {message}', executed=True,
            value={'reason': reason, 'detection_id': detection_id, **detail})

    if distractors and identity_point_handle is None:
        return rejected('ambiguous_detection_requires_identity_provenance',
            'point or mark_point the intended instance, then pass identity_point_handle',
            candidate_detection_ids=[detection_id, *distractors],
            required_call="select_detection(detection_id, identity_point_handle=point.value['point_handle'])")
    if identity_point_handle is not None:
        marked = backend.registry.get(identity_point_handle, expected_kind='marked_point')
        error, iou = identity_point_error(backend.registry, detection, marked)
        if error is not None:
            return rejected('identity_point_does_not_match_detection',
                'the point/mark must match the selected instance in the same public snapshot and camera',
                identity_point_handle=identity_point_handle, evidence_error=error, mask_iou=iou,
                selection_recovery=selection_recovery(backend, detection_id, detection, marked, error))
    if reference_point_handle is not None:
        reference = backend.registry.get(reference_point_handle, expected_kind='marked_point')
        error = point_source_error(backend.registry, detection, reference)
        if error is not None:
            return rejected('reference_point_stale',
                'use the same public snapshot and camera for the reference point',
                reference_point_handle=reference_point_handle, evidence_error=error)
    measured_nearest = selection_relation == 'nearest_xy'
    if measured_nearest:
        failure = backend._proximity_reference_check(detection_id, detection, distractors, reference_point_handle)
        if failure is not None:
            return distractors, failure
    # All fallible checks precede these mutations. Explicit evidence is retained
    # even when segmentation supplied only one candidate or no relational words.
    if identity_point_handle is not None:
        detection.identity_point_handle = identity_point_handle
    detection.distractor_detection_id = distractors[0] if distractors else None
    if not measured_nearest:
        detection.reference_point_handle = reference_point_handle
        detection.instance_reference_distances_m = None
        detection.instance_selection_margin_ratio = None
    return distractors, None

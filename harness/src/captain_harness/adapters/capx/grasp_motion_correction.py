"""Bounded grasp translation from public corresponding RGB-D surface samples.

Segmentation-centroid changes are deliberately not an input: an occluded
surface can move its visible centroid while the object stays still.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np


def assess_grasp_correction(
    evidence: Mapping[str, object], *, min_translation_m: float = .005,
    max_translation_m: float = .03, max_spread_m: float = .003,
    max_vertical_m: float = .01,
) -> dict[str, object]:
    result: dict[str, object] = {
        "decision": "retain", "reason": "motion_evidence_insufficient",
        "applied_m": None,
        "basis": "corresponding_public_rgbd_surface_points",
        "motion_evidence": dict(evidence),
    }
    if evidence.get("status") != "supported":
        return result
    try:
        delta = np.asarray(evidence.get("translation_m"), dtype=np.float64)
        spread = float(evidence["motion_spread_m"])
    except (KeyError, TypeError, ValueError):
        return result
    if delta.shape != (3,) or not np.isfinite(delta).all() or not np.isfinite(spread) or spread < 0:
        return result
    if spread > max_spread_m:
        return {**result, "reason": "motion_spread_exceeds_control_precision"}
    magnitude = float(np.linalg.norm(delta[:2]))
    result.update(measured_m=magnitude, measured_translation_m=delta.tolist())
    if abs(float(delta[2])) > max_vertical_m:
        return {**result, "decision": "reacquire", "reason": "vertical_target_motion"}
    if magnitude > max_translation_m:
        return {**result, "decision": "reacquire", "reason": "target_motion_exceeds_correction_range"}
    if magnitude < max(min_translation_m, 2 * spread):
        return {**result, "reason": "below_measurement_resolution"}
    return {**result, "decision": "translate", "reason": "applied",
            "applied_m": [float(delta[0]), float(delta[1]), 0.]}


def measure_pre_descent_correction(backend, aligned):
    """Compare the selected surface with a fresh, calibrated observation."""
    zero = np.zeros(3, dtype=np.float64)
    if not backend._v15_protocol:
        return zero, {"decision": "retain", "reason": "disabled", "applied_m": None}
    try:
        observation = backend.observe()
        if not observation.success or not isinstance(observation.value, Mapping):
            return zero, {"decision": "retain", "reason": "observation_failed", "applied_m": None}
        tracked = backend.track_target(aligned.detection_id, observation.value["observation_handle"])
        evidence = tracked.value if tracked.success and isinstance(tracked.value, Mapping) else {
            "status": "insufficient", "reason": tracked.message}
        diagnostic = assess_grasp_correction(evidence)
        diagnostic["source_detection_id"] = aligned.detection_id
        diagnostic["observation_handle"] = observation.value["observation_handle"]
        delta = diagnostic["applied_m"]
        return (zero if delta is None else np.asarray(delta, dtype=np.float64)), diagnostic
    except (ValueError, LookupError, RuntimeError) as exc:
        return zero, {"decision": "retain", "reason": f"tracking_unavailable: {type(exc).__name__}",
                      "applied_m": None}

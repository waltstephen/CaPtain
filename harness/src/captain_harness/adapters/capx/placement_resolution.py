"""Qualify nominal placement deltas using public estimate changes only."""
from collections.abc import Mapping
import numpy as np


def correction_resolution(correction_m, comparison=None):
    delta = np.asarray(correction_m, dtype=float).reshape(3)
    distance = float(np.linalg.norm(delta))
    result = {
        "status": "not_assessed",
        "nominal_correction_m": distance,
        "destination_change_along_correction_m": None,
        "accuracy_guaranteed": False,
        "basis": "partial visible geometry; decimal places and stopping tolerances do not guarantee accuracy",
    }
    if not np.isfinite(delta).all():
        result["nominal_correction_m"] = None
        return result
    if distance <= 1e-12:
        result["status"] = "no_nominal_correction"
        return result
    if not isinstance(comparison, Mapping):
        return result
    if comparison.get("destination_camera_changed") is True:
        result["status"] = "view_changed"
        return result
    if (comparison.get("destination_camera_changed") is not False
            or not comparison.get("previous_observation_handle")
            or not comparison.get("current_observation_handle")
            or comparison["previous_observation_handle"] == comparison["current_observation_handle"]):
        return result
    try:
        low = np.asarray(comparison["destination_bounds_min_delta_m"], dtype=float).reshape(3)
        high = np.asarray(comparison["destination_bounds_max_delta_m"], dtype=float).reshape(3)
    except (KeyError, ValueError, TypeError):
        return result
    if not np.isfinite(low).all() or not np.isfinite(high).all():
        return result
    direction = delta / distance
    change = max(abs(float(np.where(direction >= 0, low, high) @ direction)),
                 abs(float(np.where(direction >= 0, high, low) @ direction)))
    result["destination_change_along_correction_m"] = change
    result["status"] = "below_observed_change" if distance <= change else "exceeds_observed_change"
    result["basis"] = ("two public destination-bound estimates projected onto the correction direction; "
        "change may be scene motion, segmentation, or viewpoint, not calibrated noise")
    if distance <= change:
        result["guidance"] = ("This exact correction is smaller than the observed geometry change. "
            "Inspect existing views before deciding to correct; do not multiply the command to chase the estimate.")
    return result

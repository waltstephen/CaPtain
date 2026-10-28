"""Public grasp-choice evidence; no simulator, controller, or task dependencies."""
from __future__ import annotations

import math
from typing import Sequence


def _unit(vector: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("direction must be a finite three-vector")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        raise ValueError("direction must be nonzero")
    return tuple(value / norm for value in values)


def grasp_style_evidence(
    approach: Sequence[float],
    requested_style: str,
    *,
    reference_up: Sequence[float],
    min_top_down_alignment: float,
    max_side_vertical_alignment: float,
) -> dict[str, object]:
    """Assess a direction relative to the adapter's declared frame convention.

    The caller supplies its existing angular criteria. This is a request match,
    not a contact, reachability, or controller-convergence prediction.
    """
    vertical = max(-1., min(1., sum(a * b for a, b in zip(_unit(approach), _unit(reference_up)))))
    if requested_style == "auto":
        return {"match": "not_requested", "score": 1.}
    if requested_style == "top_down":
        score = max(0., -vertical)
        matches = score >= min_top_down_alignment
    elif requested_style == "side":
        score = 1. - abs(vertical)
        matches = abs(vertical) <= max_side_vertical_alignment
    else:
        raise ValueError("requested_style must be auto, top_down, or side")
    return {"match": "matched" if matches else "mismatched", "score": score}


def summarize_grasp_choices(
    matches: Sequence[str],
    *,
    returned_count: int,
    requested_style: str,
    source_observation_handle: str,
) -> dict[str, object]:
    """Keep a successful service call distinct from a satisfied style request."""
    returned = matches[:returned_count]
    matching_indices = [i for i, match in enumerate(returned) if match == "matched"]
    alternative_indices = [i for i, match in enumerate(returned) if match == "mismatched"]
    matching_count = sum(match == "matched" for match in matches)
    requires_decision = requested_style != "auto" and not matching_indices
    if requested_style == "auto":
        reason = "no_explicit_style_requested"
    elif matching_indices:
        reason = "requested_style_available"
    elif matching_count:
        reason = "matching_candidates_outside_top_k"
    else:
        reason = "requested_style_unavailable"
    return {
        "requires_decision": requires_decision,
        "reason": reason,
        "selection_feedback": {
            "requested_style": requested_style,
            "considered_count": len(matches),
            "matching_count": matching_count,
            "returned_matching_count": len(matching_indices),
            "matching_candidate_indices": matching_indices,
            "alternative_candidate_indices": alternative_indices,
            "source_observation_handle": source_observation_handle,
            "guidance": (
                "No returned candidate matches the requested style. End this code cell, "
                "inspect the alternatives and contact evidence, then explicitly choose an "
                "alternative or request another observation/backend/top_k. "
                if requires_decision else "Inspect contact evidence before choosing a candidate. "
            ) + (
                "Style match and a returned endpoint IK solution do not verify the motion path "
                "or a secure grasp. Missing IK is not proof of physical unreachability. "
                "Width estimates use partial visible geometry; unknown width remains unknown. "
                "convergence_risk is an uncalibrated heuristic, not a success probability."
            ),
        },
    }

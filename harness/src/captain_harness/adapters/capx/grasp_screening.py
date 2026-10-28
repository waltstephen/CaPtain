"""Compose public geometric refinement into normal candidate estimation.

No physical actions, task labels, or hidden scoring are used. Original network
candidates remain available for inspection. A screened candidate still needs
measured motion execution and a subsequent attachment check.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace


def _admissible(value):
    after = value.get("after", {})
    candidate = value.get("candidate", {})
    hits = after.get("approach", {}).get("geometry_hits")
    return bool(
        isinstance(candidate, Mapping)
        and candidate.get("width_feasible") is True
        and after.get("reachability", {}).get("reachable") is True
        and after.get("pregrasp_collision", {}).get("checked") is True
        and after.get("pregrasp_collision", {}).get("collision") is False
        and isinstance(hits, Mapping) and all(count == 0 for count in hits.values())
        and after.get("contact", {}).get("visible_point_count_in_pad_sweep", 0) > 0
    )


def screen_grasp_result(backend, result, *, local_search=False, orientation_search=False):
    if not result.success or not isinstance(result.value, Mapping) or not result.value.get("candidates"):
        return result
    value = deepcopy(dict(result.value))
    original = value["candidates"]
    evidence = {
        "original_candidates": deepcopy(original), "promoted": False,
        "selected_checks_passed": False,
        "scope": "first network candidate and bounded public-geometry alternatives",
        "motion_path": "not_checked",
        "limitation": "Visible surfaces and endpoint IK do not prove a free path or a stable grasp.",
    }
    try:
        checked = backend.refine_grasp(original[0])
        if local_search and checked.success:
            from .local_grasp_search import expand_grasp_refinement
            checked = expand_grasp_refinement(backend, original[0], checked)
        if orientation_search and checked.success:
            from .contact_orientation import expand_contact_orientation
            checked = expand_contact_orientation(backend, checked)
        if not checked.success or not isinstance(checked.value, Mapping):
            evidence.update(status="unavailable", reason=checked.message)
        else:
            refined = checked.value
            evidence.update(status="measured", refinement=deepcopy(refined))
            admissible = _admissible(refined)
            candidate = refined.get("candidate", {})
            style_compatible = value.get("grasp_style") == "auto"
            reference = value.get("selection_feedback", {}).get("style_reference", {})
            if not style_compatible and reference.get("frame") == "robot_base":
                from .grasp_feedback import grasp_style_evidence
                try:
                    style = grasp_style_evidence(candidate["approach_direction_world"], value["grasp_style"],
                        reference_up=reference["up_axis"],
                        min_top_down_alignment=reference["min_top_down_alignment"],
                        max_side_vertical_alignment=reference["max_side_vertical_alignment"])
                    evidence["requested_style_evidence"] = style
                    style_compatible = style["match"] == "matched"
                except (KeyError, TypeError, ValueError):
                    evidence["requested_style_evidence"] = {"match": "unassessed"}
            changed = candidate.get("grasp_handle") != original[0].get("grasp_handle")
            evidence['geometry_choice'] = {
                'grasp_handle': candidate.get('grasp_handle'),
                'measured_checks_passed': admissible,
                'requested_style_match': evidence.get('requested_style_evidence', {}).get(
                    'match', 'not_requested' if value.get('grasp_style') == 'auto' else 'unassessed'),
                'requires_style_change_decision': bool(changed and not style_compatible),
                'motion_path': 'not_checked',
                'guidance': ('This alternative passed the measured geometry checks but differs from the requested style. '
                    'If that style was only a preference, explicitly choose this issued handle in the next turn; '
                    'another estimation request is not required. This does not prove a stable grasp.'
                    if admissible and changed and not style_compatible else
                    'Inspect the measured checks and current views before choosing this handle.')}
            evidence["selected_checks_passed"] = admissible and (style_compatible or not changed)
            if admissible and style_compatible and changed:
                value["candidates"] = [deepcopy(candidate), *original[1:]]
                evidence.update(promoted=True, reason="public_geometry_candidate_passed_measured_checks")
            else:
                evidence.update(reason=("requested_style_requires_policy_choice" if not style_compatible
                    else "original_preserved" if admissible else "no_admissible_refinement"))
    except (ValueError, RuntimeError) as exc:
        evidence.update(status="unavailable", reason=f"{type(exc).__name__}: {exc}")
    value["sensor_screening"] = evidence
    if local_search:
        from .grasp_comparison import compare_grasps
        refined = evidence.get('refinement', {}).get('candidate')
        comparison_candidates = [*value['candidates'], *original]
        if isinstance(refined, Mapping):
            comparison_candidates.insert(0, refined)
        evidence['candidate_comparison'] = compare_grasps(backend, comparison_candidates)
    # The policy must read this new comparison before issuing an alignment,
    # especially when a native network score no longer describes the pose.
    value["requires_decision"] = True
    value["selection_feedback"] = {
        **value.get("selection_feedback", {}),
        "sensor_screening_status": evidence["status"],
        "sensor_screening_reason": evidence["reason"],
        "geometry_candidate_promoted": evidence["promoted"],
        "required_decision": "Inspect the measured geometry and original candidates, then choose an immutable grasp reference for alignment.",
    }
    return replace(result, value=value)

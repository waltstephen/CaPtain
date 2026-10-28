"""Host-side completion gate: reads the agent's own public measurements at declaration.

V1.2: one measured pair per placement goal, knob rotation evidence for turn on/off, 0.04 m alignment.

At COMPLETE_CANDIDATE the host looks at the measure_object_geometry / compare_object_observations results the
agent recorded during the episode and checks them against the declared placement standard. It never reads
simulator state; the goal predicates come from the task definition (the same information as the instruction).
A rejection returns the numbers to the agent. After MAX_REJECTIONS the declaration passes to the environment.
"""
from __future__ import annotations

import re

import numpy as np

PLACEMENT_PREDICATES = {"on", "in"}
ARTICULATED_PREDICATES = {"open", "close", "turnon", "turnoff"}
SIDE_WORDS = ("left", "right", "front", "back", "behind")
from captain_harness.core.placement_standard import CENTER_STANDARD_M, LOW_CONFIDENCE_SCORE, MAX_UNCERTAINTY_CREDIT_M
ADJACENT_MAX_M = 0.35             # 'beside X' means within this distance
ALIGN_TOL_M = 0.04                # and aligned with X along the other axis within this (V1.2: the BDDL *_region boxes
                                  # are 0.08 m wide across the side axis; 0.06 accepted a 0.051 m miss, both successes were < 0.01)
TURN_MIN_ROTATION_RAD = 0.35      # V1.2: turn on/off needs a measured knob rotation; LIBERO's turn-on threshold is 0.5 rad,
                                  # successes measured 0.49-0.64 rad, failures that only pushed the knob measured ~0
IDENTITY_DIM_TOL = 0.30           # relative size difference tolerated between pre-grasp and final object
IDENTITY_RGB_TOL = 80.0           # mean-RGB distance tolerated (0-255 scale)
MAX_REJECTIONS = 2
ARTICULATED_FAILURES = {"completion_state_change_unmeasured", "completion_rotation_unmeasured", "completion_rotation_insufficient"}
# side word -> (robot_base XY axis, sign) as seen from the head camera that faces the robot:
# +x is toward the camera ('front'), +y is the camera's right.
SIDE_DIRECTION = {"right": (1, +1.0), "left": (1, -1.0), "front": (0, +1.0), "back": (0, -1.0), "behind": (0, -1.0)}


def goals_from_bddl(goal_state):
    """Task-definition goal predicates -> gate goals. Pure; accepts the parsed BDDL goal_state list."""
    goals = []
    for item in goal_state or ():
        if not item:
            continue
        pred = str(item[0]).lower()
        args = [str(a) for a in item[1:]]
        if pred in PLACEMENT_PREDICATES and len(args) >= 2:
            dest = args[1]
            kind = "containment" if pred == "in" else "centering"
            side = None
            if dest.endswith("_region"):
                for word in SIDE_WORDS:
                    if re.search(rf"_{word}_region$", dest):
                        side, kind = word, "relation"
                        break
                if kind != "relation" and dest.endswith("_contain_region"):
                    kind = "containment"
            goals.append({"predicate": pred, "target": args[0], "destination": dest, "kind": kind, "side": side})
        elif pred in ARTICULATED_PREDICATES and args:
            goals.append({"predicate": pred, "target": args[0], "destination": None, "kind": "articulated", "side": None})
    return goals


def _dims(obj):
    fp = obj.get("footprint") or {}
    if fp.get("model") in ("cylinder", "disc"):
        xy = [2.0 * float(fp["radius_m"])] * 2
    else:
        xy = [float(v) for v in (fp.get("extent_xy_m") or [0.0, 0.0])]
    return sorted(xy + [float(obj.get("height_m") or 0.0)])  # orientation-invariant


def identity_mismatch(latest_obj, references):
    """Compare the object measured at the destination with the objects measured before the first grasp."""
    if not latest_obj or not references:
        return None
    d1 = np.asarray(_dims(latest_obj))
    best = None
    for ref in references:
        d0 = np.asarray(_dims(ref))
        rel = np.abs(d1 - d0) / np.maximum(np.maximum(d0, d1), 0.01)
        rgb = None
        if latest_obj.get("appearance_rgb_mean") and ref.get("appearance_rgb_mean"):
            rgb = float(np.linalg.norm(np.asarray(latest_obj["appearance_rgb_mean"], float) - np.asarray(ref["appearance_rgb_mean"], float)))
        score = float(rel.max()) + (rgb / 255.0 if rgb is not None else 0.0)
        if best is None or score < best[0]:
            best = (score, rel, rgb, d0)
    _, rel, rgb, d0 = best
    if rel.max() <= IDENTITY_DIM_TOL and (rgb is None or rgb <= IDENTITY_RGB_TOL):
        return None
    return {"latest_dims_m": [round(float(v), 3) for v in d1], "closest_pre_grasp_dims_m": [round(float(v), 3) for v in d0],
            "max_relative_dim_difference": round(float(rel.max()), 2), "rgb_mean_distance": None if rgb is None else round(rgb, 1)}


def _fmt(off):
    return f"x {float(off[0]):+.3f}, y {float(off[1]):+.3f} m"


def _placement_problem(goal, label, rec):
    """The rejection a single (object, destination) record earns against a placement goal, or None when it satisfies it."""
    if rec.get("low_confidence_detection"):
        return {"failure": "completion_geometry_low_confidence", "goal": goal, "measurement": rec, "message": (
            f"{label}: the latest measurement rests on a detection with score {float(rec.get('min_detection_score') or 0):.4f} "
            f"(below {LOW_CONFIDENCE_SCORE}); re-detect with a prompt naming the whole object or container and use a candidate "
            "with a real score.")}
    if rec.get("destination_smaller_than_object"):
        return {"failure": "completion_geometry_destination_fragment", "goal": goal, "measurement": rec, "message": (
            f"{label}: the destination detection in your latest measurement is smaller than the object itself (footprints "
            f"{rec.get('destination_span_m')} vs {rec.get('object_span_m')} m); it cannot support or contain the object, so it is a "
            "fragment or the wrong detection. Re-detect the whole plate / container / surface and measure again.")}
    off = rec.get("object_to_destination_xy_offset_m") or [0.0, 0.0]
    dist = float(rec.get("xy_distance_m") or np.hypot(*off))
    credit = min(float(rec.get("xy_offset_uncertainty_m") or 0.0), MAX_UNCERTAINTY_CREDIT_M)
    if goal["kind"] == "centering":
        if dist > CENTER_STANDARD_M + credit:
            return {"failure": "completion_geometry_offset", "goal": goal, "measurement": rec, "message": (
                f"{label}: object-minus-destination XY offset {_fmt(off)} (norm {dist:.3f} m) is above the {CENTER_STANDARD_M:.2f} m "
                f"centering standard (+{credit:.3f} m uncertainty credit). Move the object by the negative offset and re-measure.")}
    elif goal["kind"] == "containment":
        if not rec.get("object_center_inside_destination_footprint"):
            return {"failure": "completion_geometry_not_contained", "goal": goal, "measurement": rec, "message": (
                f"{label}: the object center is outside the destination footprint; offset from the container center {_fmt(off)}, "
                f"container extents {rec.get('destination_span_m')} m. The object is not inside the container.")}
    elif goal["kind"] == "relation":
        axis, sign = SIDE_DIRECTION[goal["side"]]
        other = 1 - axis
        ospan = rec.get("object_span_m") or [0.0, 0.0]
        dspan = rec.get("destination_span_m") or [0.0, 0.0]
        along = float(off[axis]) * sign
        across = abs(float(off[other]))
        half = 0.5 * (float(ospan[axis]) + float(dspan[axis]))
        if along < 0.8 * half or along > ADJACENT_MAX_M or across > ALIGN_TOL_M:
            return {"failure": "completion_relation_unmet", "goal": goal, "measurement": rec, "message": (
                f"{label}: '{goal['side']} of' means resting beside the reference on its {goal['side']} side as seen in the head camera "
                "(front = toward the camera, right = the camera's right), adjacent to it (beyond the two half-widths, within "
                f"{ADJACENT_MAX_M:.2f} m) and aligned along the other axis (within {ALIGN_TOL_M:.2f} m). Measured object-minus-reference "
                f"offset {_fmt(off)}; along-side {along:+.3f} m, across {across:.3f} m.")}
    return None


def _match_placement_goal(goal, label, candidates, claimed):
    """Assign one unclaimed (object, destination) record to this goal.

    Records are the latest per detection pair, in measurement order. The goal takes the last unclaimed record that
    satisfies it; when none does, the rejection quotes the last unclaimed record (the agent's latest claim) so a
    single-goal task behaves exactly as before. Multi-goal tasks therefore need one measured pair per goal.
    Detection IDs do not establish distinct physical instances across observations; this public gate cannot
    certify task identity. The environment's final predicate remains authoritative.
    """
    free = [(i, r) for i, r in enumerate(candidates) if i not in claimed]
    if not free:
        return None, {"failure": "completion_geometry_unmeasured", "goal": goal, "message": (
            f"{label}: every fresh measurement is already accounted for by another goal of this task. Each placed object "
            "needs its own measure_object_geometry(target, destination) against its own destination after your last physical action.")}
    for i, rec in reversed(free):
        if _placement_problem(goal, label, rec) is None:
            claimed.add(i)
            return rec, None
    i, rec = free[-1]
    claimed.add(i)
    return rec, _placement_problem(goal, label, rec)


def evaluate(goals, geometry_records, comparison_records, *, last_physical_step, first_grasp_step, rejections):
    """Verdict for a COMPLETE_CANDIDATE declaration. Pure function over recorded public measurements."""
    if not goals:
        return {"verdict": "not_applicable"}
    if rejections >= MAX_REJECTIONS:
        return {"verdict": "bypass", "reason": f"{rejections} rejections already issued; declaration passed to the environment"}

    def fresh(step):
        return last_physical_step is None or step is None or step >= last_physical_step

    fresh_geo = [r for r in geometry_records if r.get("has_destination") and fresh(r.get("step"))]
    # V1.2: one record per (object, destination) detection pair, the latest of each (repeats of the same pair cannot be
    # cherry-picked); every placement goal must be satisfied by its own pair.
    latest_per_pair = {}
    for r in fresh_geo:
        latest_per_pair[(r.get("target_detection_id"), r.get("destination_detection_id"))] = r
    candidate_records = list(latest_per_pair.values())
    claimed = set()
    fresh_state = [r for r in geometry_records if r.get("has_before") and fresh(r.get("step"))]
    fresh_cmp = [r for r in comparison_records if fresh(r.get("step"))]
    pre_grasp = [r["object"] for r in geometry_records
                 if r.get("object") and first_grasp_step is not None and (r.get("step") or 0) < first_grasp_step]
    problems = []
    advisories = []
    for goal in goals:
        label = f"{goal['predicate']} {goal['target']}" + (f" -> {goal['destination']}" if goal.get("destination") else "")
        if goal["kind"] == "articulated":
            if goal["predicate"] in ("turnon", "turnoff"):
                # V1.2: the before/after geometry tool reports the knob's yaw change; require a real rotation, not a push
                rotations = [abs(float(r.get("part_rotation_rad") or 0.0)) for r in fresh_state]
                if not rotations:
                    problems.append({"failure": "completion_rotation_unmeasured", "goal": goal, "message": (
                        f"{label}: no before/after measurement of the knob after your last physical action. Detect the knob in an "
                        "observation taken before turning and in a fresh one, call measure_object_geometry(after_id, "
                        f"before_detection_id=before_id) and print part_rotation_rad; the knob must rotate by at least {TURN_MIN_ROTATION_RAD:.2f} rad.")})
                elif max(rotations) < TURN_MIN_ROTATION_RAD:
                    problems.append({"failure": "completion_rotation_insufficient", "goal": goal, "message": (
                        f"{label}: the measured knob rotation is only {max(rotations):.2f} rad (part_rotation_rad); pushing the knob "
                        f"sideways does not turn it. Rotate it by at least {TURN_MIN_ROTATION_RAD:.2f} rad and re-measure.")})
                continue
            measured = (any(r.get("part_motion_significant") for r in fresh_state)
                        or any(r.get("comparable") for r in fresh_cmp))
            if not measured:
                stale = [r for r in fresh_state if not r.get("part_motion_significant")]
                detail = (f" The latest before/after measurement moved only {float(stale[-1].get('part_distance_m') or 0):.3f} m." if stale else "")
                problems.append({"failure": "completion_state_change_unmeasured", "goal": goal, "message": (
                    f"{label}: no significant state change measured after your last physical action.{detail} Detect the same "
                    "handle or door front in an observation taken before the action and in a fresh one, call "
                    "measure_object_geometry(after_id, before_detection_id=before_id) and print part_displacement_m / part_rotation_rad.")})
            continue
        if not fresh_geo:
            problems.append({"failure": "completion_geometry_unmeasured", "goal": goal, "message": (
                f"{label}: no successful measure_object_geometry(target, destination) after your last physical action. Take a fresh "
                "observe(), select the placed object and its destination surface or container (a different detection than the "
                "object), measure, and print object_to_destination_xy_offset_m.")})
            continue
        rec, problem = _match_placement_goal(goal, label, candidate_records, claimed)
        if problem is not None:
            problems.append(problem)
            continue
        mismatch = identity_mismatch(rec.get("object"), pre_grasp)
        if mismatch:
            # Advisory only: objects standing inside containers are partly occluded, so size differences of
            # 30-40 % occur on correct placements (two false rejections in the first six episodes), while SAM3 mislabels
            # move the same wrong object throughout and are invisible to this check. Logged, never rejected.
            advisories.append({"kind": "target_identity_mismatch", "goal": goal, "detail": mismatch})
    if problems and rejections >= 1 and all(p["failure"] in ARTICULATED_FAILURES for p in problems):
        # articulated evidence gets one rejection only; the tools cannot certify every mechanism
        return {"verdict": "bypass", "reason": "articulated state evidence was already rejected once; declaration passed to the environment",
                "advisories": advisories, "problems": problems}
    if problems:
        return {"verdict": "reject", "failure": problems[0]["failure"],
                "message": "completion gate: " + " | ".join(p["message"] for p in problems), "problems": problems,
                "advisories": advisories}
    return {"verdict": "pass", "checked": [g["kind"] for g in goals], "advisories": advisories,
            "latest_measurement_step": fresh_geo[-1]["step"] if fresh_geo else None}


def geometry_record(value, arguments, step):
    """Compact host-side record of one successful measure_object_geometry result (never model input)."""
    obj = dict(value.get("object") or {})
    dst = value.get("destination")
    def span(g):
        fp = (g or {}).get("footprint") or {}
        if fp.get("model") in ("cylinder", "disc"):
            return [2.0 * float(fp["radius_m"])] * 2
        if fp.get("extent_xy_m"):
            w, h = (float(v) for v in fp["extent_xy_m"]); yaw = float(fp.get("yaw_rad") or 0.0)
            return [abs(w * np.cos(yaw)) + abs(h * np.sin(yaw)), abs(w * np.sin(yaw)) + abs(h * np.cos(yaw))]
        return [0.0, 0.0]
    scores = [s for s in (obj.get("detection_score"), (dst or {}).get("detection_score")) if s is not None]
    return {"step": step, "has_destination": dst is not None, "has_before": "before" in value,
            "part_motion_significant": value.get("part_motion_significant"), "part_distance_m": value.get("part_distance_m"),
            "part_rotation_rad": value.get("part_rotation_rad"), "before_detection_id": arguments.get("before_detection_id"),
            "part_rotation_basis": value.get("part_rotation_basis"),
            "part_rotation_symmetry_period_rad": value.get("part_rotation_symmetry_period_rad"),
            "target_detection_id": arguments.get("target_detection_id"), "destination_detection_id": arguments.get("destination_detection_id"),
            "object": {"footprint": obj.get("footprint"), "height_m": obj.get("height_m"), "appearance_rgb_mean": obj.get("appearance_rgb_mean"),
                       "detection_score": obj.get("detection_score"), "center_m": obj.get("center_m"), "quality": obj.get("quality")},
            "object_span_m": span(obj), "destination_span_m": span(dst),
            "xy_distance_m": value.get("xy_distance_m"), "object_to_destination_xy_offset_m": value.get("object_to_destination_xy_offset_m"),
            "xy_offset_uncertainty_m": value.get("xy_offset_uncertainty_m"),
            "object_center_inside_destination_footprint": value.get("object_center_inside_destination_footprint"),
            "resting_height_consistent": value.get("resting_height_consistent"),
            "destination_smaller_than_object": bool(value.get("destination_smaller_than_object")),
            "min_detection_score": min(scores) if scores else None,
            "low_confidence_detection": bool(scores) and min(scores) < LOW_CONFIDENCE_SCORE}


def comparison_record(value, step):
    tracking = value.get("tracking") or {}
    comparable = tracking.get("status") == "supported" and tracking.get("association_status") == "supported"
    return {"step": step, "comparable": bool(comparable), "effect": value.get("effect"),
            "measured_object_displacement_m": value.get("measured_object_displacement_m")}

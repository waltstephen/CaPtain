"""Host-side completion gate over recorded public measurements; no environment, no model."""
import numpy as np
from types import SimpleNamespace
from captain_harness.adapters.capx import completion_gate as cg
from captain_harness.adapters.capx.object_geometry import footprint_area_m2, point_in_footprint, placement_relation, same_surface


def box(cx, cy, w, h, yaw=0.0):
    return {"model": "box", "center_xy_m": [cx, cy], "extent_xy_m": [w, h], "yaw_rad": yaw}


def circ(cx, cy, r):
    return {"model": "cylinder", "center_xy_m": [cx, cy], "radius_m": r}


def obj(fp, height=0.1, rgb=(30, 30, 30), score=0.5):
    return {"footprint": fp, "height_m": height, "appearance_rgb_mean": list(rgb), "detection_score": score}


def rec(step, off, *, inside=True, unc=0.003, ospan=(0.1, 0.03), dspan=(0.4, 0.14), o=None, scores=(0.5, 0.5)):
    return {"step": step, "has_destination": True, "object": o or obj(box(0, 0, *ospan)), "object_span_m": list(ospan),
            "destination_span_m": list(dspan), "xy_distance_m": float(np.hypot(*off)), "object_to_destination_xy_offset_m": list(off),
            "xy_offset_uncertainty_m": unc, "object_center_inside_destination_footprint": inside,
            "min_detection_score": min(scores), "low_confidence_detection": min(scores) < cg.LOW_CONFIDENCE_SCORE}


def test_goals_from_bddl_classifies_predicates():
    goals = cg.goals_from_bddl([["On", "bowl_1", "plate_1"], ["In", "book_1", "desk_caddy_1_front_contain_region"],
                                ["On", "mug_1", "study_table_desk_caddy_right_region"], ["Turnon", "flat_stove_1"],
                                ["On", "pan_1", "flat_stove_1_cook_region"]])
    assert [g["kind"] for g in goals] == ["centering", "containment", "relation", "articulated", "centering"]
    assert goals[2]["side"] == "right"


def test_unmeasured_and_stale_measurements_are_rejected():
    goals = cg.goals_from_bddl([["On", "bowl_1", "plate_1"]])
    v = cg.evaluate(goals, [], [], last_physical_step=100, first_grasp_step=10, rejections=0)
    assert v["verdict"] == "reject" and v["failure"] == "completion_geometry_unmeasured"
    v = cg.evaluate(goals, [rec(50, (0.0, 0.0))], [], last_physical_step=100, first_grasp_step=10, rejections=0)
    assert v["failure"] == "completion_geometry_unmeasured"


def test_centering_uses_latest_record_not_the_best_one():
    goals = cg.goals_from_bddl([["On", "bowl_1", "plate_1"]])
    records = [rec(120, (0.0004, 0.0)), rec(121, (0.147, -0.001))]
    v = cg.evaluate(goals, records, [], last_physical_step=100, first_grasp_step=10, rejections=0)
    assert v["failure"] == "completion_geometry_offset" and "0.147" in v["message"]
    v = cg.evaluate(goals, [rec(122, (0.012, 0.009))], [], last_physical_step=100, first_grasp_step=10, rejections=0)
    assert v["verdict"] == "pass"


def test_containment_relation_and_low_confidence():
    goals = cg.goals_from_bddl([["In", "book_1", "desk_caddy_1_front_contain_region"]])
    v = cg.evaluate(goals, [rec(120, (0.147, 0.0), inside=False)], [], last_physical_step=100, first_grasp_step=10, rejections=0)
    assert v["failure"] == "completion_geometry_not_contained"
    goals = cg.goals_from_bddl([["On", "mug_1", "study_table_desk_caddy_right_region"]])
    # mug placed in front of the caddy (+x) instead of to its right (+y)
    v = cg.evaluate(goals, [rec(120, (0.15, 0.08), ospan=(0.13, 0.13), dspan=(0.14, 0.44))], [], last_physical_step=100, first_grasp_step=10, rejections=0)
    assert v["failure"] == "completion_relation_unmet"
    # correct side, adjacent and aligned
    v = cg.evaluate(goals, [rec(120, (0.01, 0.29), ospan=(0.13, 0.13), dspan=(0.14, 0.44))], [], last_physical_step=100, first_grasp_step=10, rejections=0)
    assert v["verdict"] == "pass"
    goals = cg.goals_from_bddl([["On", "mug_1", "plate_2"]])
    v = cg.evaluate(goals, [rec(120, (0.01, 0.0), scores=(0.9, 0.003))], [], last_physical_step=100, first_grasp_step=10, rejections=0)
    assert v["failure"] == "completion_geometry_low_confidence"


def test_identity_mismatch_against_pre_grasp_measurement():
    goals = cg.goals_from_bddl([["In", "tomato_sauce_1", "basket_1_contain_region"]])
    pre = {"step": 5, "has_destination": False, "object": obj(circ(0.5, 0.2, 0.035), height=0.13, rgb=(180, 40, 30))}
    final_same = rec(120, (0.003, 0.002), o=obj(circ(0.6, 0.2, 0.036), height=0.125, rgb=(170, 45, 35)))
    final_other = rec(120, (0.003, 0.002), o=obj(box(0.6, 0.2, 0.08, 0.05), height=0.03, rgb=(90, 60, 40)))
    v = cg.evaluate(goals, [pre, final_same], [], last_physical_step=100, first_grasp_step=10, rejections=0)
    assert v["verdict"] == "pass" and v["advisories"] == []
    # a different object at the destination is reported as an advisory, never as a rejection (occlusion inside containers)
    v = cg.evaluate(goals, [pre, final_other], [], last_physical_step=100, first_grasp_step=10, rejections=0)
    assert v["verdict"] == "pass" and v["advisories"][0]["kind"] == "target_identity_mismatch"
    assert cg.identity_mismatch(final_other["object"], [pre["object"]]) is not None


def test_articulated_requires_fresh_comparison_and_bypass_after_two_rejections():
    goals = cg.goals_from_bddl([["Open", "drawer_1"]])
    v = cg.evaluate(goals, [], [], last_physical_step=100, first_grasp_step=None, rejections=0)
    assert v["failure"] == "completion_state_change_unmeasured"
    v = cg.evaluate(goals, [], [{"step": 130, "comparable": True}], last_physical_step=100, first_grasp_step=None, rejections=0)
    assert v["verdict"] == "pass"
    assert cg.evaluate(goals, [], [], last_physical_step=100, first_grasp_step=None, rejections=2)["verdict"] == "bypass"


def test_geometry_helpers_and_same_surface():
    assert abs(footprint_area_m2(circ(0, 0, 0.05)) - np.pi * 0.0025) < 1e-9
    assert point_in_footprint((0.02, 0.0), box(0, 0, 0.1, 0.05)) and not point_in_footprint((0.06, 0.0), box(0, 0, 0.1, 0.05))
    rel = placement_relation({"center_xy_m": [0.493, -0.134], "bottom_z_m": -0.03}, {"footprint": box(0.346, -0.133, 0.137, 0.426), "top_z_m": 0.056, "support_surface_z_m": -0.03})
    assert rel["object_center_inside_destination_footprint"] is False
    mask = np.zeros((8, 8), bool); mask[2:5, 2:5] = True
    det = SimpleNamespace(camera_name="a", observation_handle="o", mask=mask, bounds_min=(0, 0, 0), bounds_max=(0.1, 0.03, 0.14), centroid=(0.5, 0.1, 0.05))
    other = SimpleNamespace(camera_name="a", observation_handle="o", mask=np.zeros((8, 8), bool), bounds_min=(0, 0, 0), bounds_max=(0.4, 0.14, 0.06), centroid=(0.35, 0.1, 0.03))
    assert same_surface(det, det) and same_surface(det, SimpleNamespace(**{**det.__dict__})) and not same_surface(det, other)

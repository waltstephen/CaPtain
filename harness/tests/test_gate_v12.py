"""V1.2 gate: one measured pair per placement goal, knob rotation evidence, tighter side alignment."""
from captain_harness.adapters.capx import completion_gate as cg
from test_completion_gate import rec


def _pair(step, off, target, dest, **kw):
    r = rec(step, off, **kw)
    r["target_detection_id"] = target
    r["destination_detection_id"] = dest
    return r


def test_single_goal_keeps_latest_record_semantics():
    goals = cg.goals_from_bddl([["On", "bowl_1", "plate_1"]])
    records = [_pair(120, (0.0004, 0.0), "b", "p"), _pair(121, (0.147, -0.001), "b", "p")]
    v = cg.evaluate(goals, records, [], last_physical_step=100, first_grasp_step=10, rejections=0)
    assert v["failure"] == "completion_geometry_offset" and "0.147" in v["message"]


def test_two_placement_goals_need_two_measured_pairs():
    goals = cg.goals_from_bddl([["On", "white_mug_1", "plate_1"], ["On", "yellow_mug_1", "plate_2"]])
    one = [_pair(120, (0.004, 0.002), "m1", "p1")]
    v = cg.evaluate(goals, one, [], last_physical_step=100, first_grasp_step=10, rejections=0)
    assert v["verdict"] == "reject" and v["failure"] == "completion_geometry_unmeasured" and "own destination" in v["message"]
    two = one + [_pair(121, (0.006, -0.003), "m2", "p2")]
    assert cg.evaluate(goals, two, [], last_physical_step=100, first_grasp_step=10, rejections=0)["verdict"] == "pass"
    # a repeated measurement of the same pair replaces the earlier one instead of counting as a second object
    repeat = one + [_pair(121, (0.006, -0.003), "m1", "p1")]
    assert cg.evaluate(goals, repeat, [], last_physical_step=100, first_grasp_step=10, rejections=0)["failure"] == "completion_geometry_unmeasured"


def test_centering_plus_relation_goals_are_matched_independently():
    goals = cg.goals_from_bddl([["On", "mug_1", "plate_1"], ["On", "pudding_1", "plate_1_right_region"]])
    records = [_pair(120, (0.003, 0.001), "mug", "plate"),
               _pair(121, (0.01, 0.20), "pudding", "plate", ospan=(0.08, 0.08), dspan=(0.2, 0.2))]
    assert cg.evaluate(goals, records, [], last_physical_step=100, first_grasp_step=10, rejections=0)["verdict"] == "pass"
    # the old gate judged both goals against the last record and rejected correct placements
    bad = [_pair(120, (0.003, 0.001), "mug", "plate"), _pair(121, (0.01, 0.055), "pudding", "plate", ospan=(0.08, 0.08), dspan=(0.2, 0.2))]
    v = cg.evaluate(goals, bad, [], last_physical_step=100, first_grasp_step=10, rejections=0)
    assert v["failure"] == "completion_relation_unmet"


def test_side_alignment_tolerance_is_the_region_half_width():
    goals = cg.goals_from_bddl([["On", "plate_1", "main_table_stove_front_region"]])
    ok = _pair(120, (0.177, -0.002), "plate", "stove", ospan=(0.14, 0.14), dspan=(0.19, 0.19))
    miss = _pair(120, (0.179, -0.051), "plate", "stove", ospan=(0.14, 0.14), dspan=(0.19, 0.19))
    assert cg.evaluate(goals, [ok], [], last_physical_step=100, first_grasp_step=10, rejections=0)["verdict"] == "pass"
    v = cg.evaluate(goals, [miss], [], last_physical_step=100, first_grasp_step=10, rejections=0)
    assert v["failure"] == "completion_relation_unmet" and "0.051" in v["message"]


def test_turn_on_requires_measured_rotation_and_bypasses_after_one_rejection():
    goals = cg.goals_from_bddl([["Turnon", "flat_stove_1"]])
    v = cg.evaluate(goals, [], [], last_physical_step=100, first_grasp_step=None, rejections=0)
    assert v["verdict"] == "reject" and v["failure"] == "completion_rotation_unmeasured" and "part_rotation_rad" in v["message"]
    pushed = {"step": 130, "has_destination": False, "has_before": True, "part_motion_significant": True, "part_distance_m": 0.027, "part_rotation_rad": 0.0}
    v = cg.evaluate(goals, [pushed], [], last_physical_step=100, first_grasp_step=None, rejections=0)
    assert v["failure"] == "completion_rotation_insufficient" and "0.00 rad" in v["message"]
    turned = {**pushed, "part_rotation_rad": -0.54}
    assert cg.evaluate(goals, [turned], [], last_physical_step=100, first_grasp_step=None, rejections=0)["verdict"] == "pass"
    assert cg.evaluate(goals, [pushed], [], last_physical_step=100, first_grasp_step=None, rejections=1)["verdict"] == "bypass"


def test_destination_smaller_than_object_is_refused_as_a_placement_destination_but_recorded():
    goals = cg.goals_from_bddl([["On", "bowl_1", "plate_1"]])
    fragment = _pair(120, (0.002, 0.001), "bowl", "rim_fragment")
    fragment["destination_smaller_than_object"] = True
    v = cg.evaluate(goals, [fragment], [], last_physical_step=100, first_grasp_step=10, rejections=0)
    assert v["failure"] == "completion_geometry_destination_fragment" and "smaller than the object" in v["message"]
    rec = cg.geometry_record({"object": {}, "destination": {}, "destination_smaller_than_object": True,
                              "object_to_destination_xy_offset_m": [0.0, 0.0]}, {"target_detection_id": "a", "destination_detection_id": "b"}, 5)
    assert rec["destination_smaller_than_object"] is True and rec["has_destination"]

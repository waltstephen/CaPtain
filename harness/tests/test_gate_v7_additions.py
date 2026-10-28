"""v7 gate additions: articulated geometry evidence, turnon advisory, single articulated rejection; part_motion."""
import numpy as np
from captain_harness.adapters.capx import completion_gate as cg
from captain_harness.adapters.capx.object_geometry import part_motion


def geo(center, yaw=0.0, model="box", unc=0.003):
    fp = {"model": model, "center_xy_m": center[:2], "extent_xy_m": [0.1, 0.02], "yaw_rad": yaw} if model == "box" else {"model": model, "center_xy_m": center[:2], "radius_m": 0.03}
    return {"center_m": list(center), "footprint": fp, "center_uncertainty_m": [unc, unc, 0.003]}


def test_part_motion_displacement_rotation_and_significance():
    m = part_motion(geo([0.6, 0.1, 0.2]), geo([0.6, 0.22, 0.2]))
    assert abs(m["part_distance_m"] - 0.12) < 1e-9 and m["part_motion_significant"] and m["part_rotation_rad"] == 0.0
    swapped = geo([0.6, 0.1, 0.2], yaw=np.pi / 2 + np.radians(20))
    swapped["footprint"]["extent_xy_m"] = [0.02, 0.1]
    m = part_motion(geo([0.6, 0.1, 0.2], yaw=0.0), swapped)
    assert abs(np.degrees(m["part_rotation_rad"]) - 20) < 1e-6 and m["part_motion_significant"]
    m = part_motion(geo([0.6, 0.1, 0.2]), geo([0.604, 0.1, 0.2]))
    assert not m["part_motion_significant"]
    assert part_motion(geo([0, 0, 0], model="cylinder"), geo([0, 0.05, 0], model="cylinder"))["part_rotation_rad"] is None


def test_quarter_turn_of_elongated_part_is_not_discarded_by_the_gate():
    motion = part_motion(geo([0, 0, 0]), geo([0, 0, 0], yaw=np.pi / 2))
    assert abs(abs(motion["part_rotation_rad"]) - np.pi / 2) < 1e-9
    assert motion["part_motion_significant"]
    assert motion["part_rotation_basis"] == "rectangle_long_axis"
    measured = {"step": 10, "has_before": True, **motion}
    goals = cg.goals_from_bddl([["Turnon", "flat_stove_1"]])
    assert cg.evaluate(goals, [measured], [], last_physical_step=10,
                       first_grasp_step=None, rejections=0)["verdict"] == "pass"


def test_axis_relabeling_and_half_turn_keep_rectangle_symmetry():
    before = geo([0, 0, 0])
    relabeled = geo([0, 0, 0], yaw=np.pi / 2)
    relabeled["footprint"]["extent_xy_m"] = [0.02, 0.1]
    for after in (relabeled, geo([0, 0, 0], yaw=np.pi)):
        motion = part_motion(before, after)
        assert abs(motion["part_rotation_rad"]) < 1e-9
        assert not motion["part_motion_significant"]
        assert motion["part_rotation_symmetry_period_rad"] == np.pi


def test_square_quarter_turn_is_ambiguous_and_circular_rotation_is_unavailable():
    before = geo([0, 0, 0])
    before["footprint"]["extent_xy_m"] = [0.04, 0.04]
    after = geo([0, 0, 0], yaw=np.pi / 2)
    after["footprint"]["extent_xy_m"] = [0.04, 0.04]
    motion = part_motion(before, after)
    assert abs(motion["part_rotation_rad"]) < 1e-9
    assert motion["part_rotation_basis"] == "square_footprint"
    assert motion["part_rotation_symmetry_period_rad"] == np.pi / 2
    unknown = part_motion(before, geo([0, 0, 0]))
    assert unknown["part_rotation_rad"] is None and unknown["part_rotation_basis"] == "unavailable"
    circle = part_motion(geo([0, 0, 0], model="cylinder"), geo([0, 0, 0], model="cylinder"))
    assert circle["part_rotation_rad"] is None


def test_articulated_goal_accepts_geometry_state_evidence_and_bypasses_after_one_rejection():
    goals = cg.goals_from_bddl([["Open", "wooden_cabinet_1_middle_region"]])
    state_ok = {"step": 130, "has_destination": False, "has_before": True, "part_motion_significant": True, "part_distance_m": 0.09}
    state_small = {"step": 130, "has_destination": False, "has_before": True, "part_motion_significant": False, "part_distance_m": 0.002}
    assert cg.evaluate(goals, [state_ok], [], last_physical_step=100, first_grasp_step=None, rejections=0)["verdict"] == "pass"
    v = cg.evaluate(goals, [state_small], [], last_physical_step=100, first_grasp_step=None, rejections=0)
    assert v["verdict"] == "reject" and "0.002" in v["message"] and "before_detection_id" in v["message"]
    assert cg.evaluate(goals, [state_small], [], last_physical_step=100, first_grasp_step=None, rejections=1)["verdict"] == "bypass"


def test_placement_goals_keep_two_rejections():
    goals = cg.goals_from_bddl([["On", "bowl_1", "plate_1"]])
    assert cg.evaluate(goals, [], [], last_physical_step=100, first_grasp_step=None, rejections=1)["verdict"] == "reject"

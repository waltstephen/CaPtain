"""Private goal audit: BDDL identity, region geometry and articulated progress."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from captain_harness.adapters.libero.hidden_object_audit import HiddenObjectAudit, SCHEMA_VERSION


ROOT = Path(__file__).resolve().parents[1]
LIBERO = ROOT / "third_party/libero/libero/libero"


@pytest.fixture(scope="module")
def parse_problem():
    pytest.importorskip("bddl.parsing")
    spec = importlib.util.spec_from_file_location("audit_bddl_parser", LIBERO / "envs/bddl_utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return lambda relative: module.robosuite_parse_problem(str(LIBERO / "bddl_files" / relative))


def audit_env(goals, *, regions=None, objects=("object_1",), fixtures=(), sites=None, base_rotation=None):
    names = ["robot0_base", "gripper0_eef", *objects, *fixtures]
    body_ids = {name: index for index, name in enumerate(names)}
    positions = np.zeros((len(names), 3))
    rotations = np.tile(np.eye(3).ravel(), (len(names), 1))
    if base_rotation is not None:
        rotations[0] = np.asarray(base_rotation).ravel()
    site_positions = {name: np.asarray(site.position, dtype=float) for name, site in (sites or {}).items()}
    site_rotations = {name: np.asarray(getattr(site, "rotation", np.eye(3)), dtype=float) for name, site in (sites or {}).items()}
    joint_names = list(dict.fromkeys(joint for site in (sites or {}).values() for joint in site.joints))
    data = SimpleNamespace(xpos=positions, body_xpos=positions, xmat=rotations, qpos=np.zeros(20),
                           get_site_xpos=lambda name: site_positions[name],
                           get_site_xmat=lambda name: site_rotations[name])
    model = SimpleNamespace(body_name2id=lambda name: body_ids[name],
                            get_joint_qpos_addr=lambda name: joint_names.index(name))
    inner = SimpleNamespace(sim=SimpleNamespace(model=model, data=data),
                            parsed_problem={"goal_state": goals, "regions": regions or {}},
                            obj_body_id={name: body_ids[name] for name in [*objects, *fixtures]},
                            objects_dict={name: SimpleNamespace(joints=[]) for name in objects},
                            fixtures_dict={name: SimpleNamespace(joints=[]) for name in fixtures},
                            object_sites_dict=sites or {}, _eval_predicate=lambda goal: False)
    inner.test_body_ids = body_ids
    inner.test_joint_names = joint_names
    inner.test_site_positions = site_positions
    return inner


def site(position, size=(.1, .1, .01), *, parent=None, joints=(), rotation=None):
    return SimpleNamespace(position=position, size=size, site_type="box", parent_name=parent,
                           joints=list(joints), rotation=np.eye(3) if rotation is None else rotation)


def test_all_40_actual_bddl_goals_keep_argument_identity(parse_problem):
    region_tasks = 0
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        paths = sorted((LIBERO / "bddl_files" / suite).glob("*.bddl"))
        assert len(paths) == 10
        for path in paths:
            problem = parse_problem(path.relative_to(LIBERO / "bddl_files"))
            objects = [name for names in problem["objects"].values() for name in names]
            fixtures = [name for category, names in problem["fixtures"].items()
                        if category not in {"table", "coffee_table", "living_room_table", "study_table", "kitchen_table"}
                        for name in names]
            env = audit_env(problem["goal_state"], regions=problem["regions"], objects=objects, fixtures=fixtures)
            header = HiddenObjectAudit(env).header()
            assert header["schema_version"] == SCHEMA_VERSION
            assert len(header["goals"]) == len(problem["goal_state"])
            for actual, goal in zip(header["goals"], problem["goal_state"]):
                assert actual["arguments"] == goal[1:]
                assert actual["target"]["name"] == goal[1]
                if len(goal) == 3:
                    assert actual["destination"]["name"] == goal[2]
                    if goal[2] in problem["regions"]:
                        assert actual["destination"]["kind"] == "site"
            region_tasks += any(arg in problem["regions"] for goal in problem["goal_state"] for arg in goal[1:])
    assert region_tasks == 26


def test_actual_pushplate_region_survives_missing_arena_body(parse_problem):
    problem = parse_problem("libero_goal/push_the_plate_to_the_front_of_the_stove.bddl")
    name = "main_table_stove_front_region"
    env = audit_env(problem["goal_state"], regions=problem["regions"], objects=("plate_1",),
                    sites={name: site([.5, .1, .2])})
    reader = HiddenObjectAudit(env)
    assert reader.header()["goals"][0]["destination"]["name"] == name
    env.sim.data.xpos[env.obj_body_id["plate_1"]] = [.2, .1, .24]
    far = reader.sample()["goal_states"]["goal_0"]
    env.sim.data.xpos[env.obj_body_id["plate_1"]] = [.5, .1, .24]
    near = reader.sample()["goal_states"]["goal_0"]
    assert far["distance_to_goal"] == pytest.approx(.2)
    assert near["distance_to_goal"] == 0
    assert near["destination_position_m"] == [.5, .1, .2]
    assert not near["requires_contact"]


def test_elevated_region_geometry_uses_destination_height_and_rotation():
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    env = audit_env([["on", "object_1", "cabinet_top"]], fixtures=("cabinet",),
                    regions={"cabinet_top": {"target": "cabinet"}},
                    sites={"cabinet_top": site([1., 2., .8], parent="cabinet")}, base_rotation=rotation)
    env.sim.data.xpos[0] = [.5, 1., .1]
    env.sim.data.xpos[env.obj_body_id["object_1"]] = [1., 2., .85]
    row = HiddenObjectAudit(env).sample()["goal_states"]["goal_0"]
    assert row["distance_to_goal"] == 0
    assert row["target_position_m"] == [1., -.5, .75]
    assert row["destination_position_m"] == [1., -.5, .7]
    assert row["requires_contact"]
    env.sim.data.xpos[env.obj_body_id["object_1"], 2] = 1.4
    assert HiddenObjectAudit(env).sample()["goal_states"]["goal_0"]["distance_to_goal"] == pytest.approx(.49)


def test_containment_rotated_site_uses_libero_predicate_box():
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    env = audit_env([["in", "object_1", "box_inside"]],
                    regions={"box_inside": {"target": "box"}},
                    sites={"box_inside": site([0., 0., .5], size=(.1, .2, .1), rotation=rotation)})
    env.sim.data.xpos[env.obj_body_id["object_1"]] = [.15, 0., .5]
    row = HiddenObjectAudit(env).sample()["goal_states"]["goal_0"]
    assert row["distance_to_goal"] == 0
    assert row["region_geometry"]["half_size_m"] == [.2, .1, .105]
    env.sim.data.xpos[env.obj_body_id["object_1"], 0] = .25
    assert HiddenObjectAudit(env).sample()["goal_states"]["goal_0"]["distance_to_goal"] == pytest.approx(.05)


def test_actual_two_target_goal_keeps_distinct_destinations(parse_problem):
    problem = parse_problem("libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate.bddl")
    env = audit_env(problem["goal_state"], regions=problem["regions"],
                    objects=("porcelain_mug_1", "plate_1", "chocolate_pudding_1"))
    goals = HiddenObjectAudit(env).header()["goals"]
    assert [(g["target"]["name"], g["destination"]["name"]) for g in goals] == [
        ("porcelain_mug_1", "plate_1"), ("chocolate_pudding_1", "living_room_table_plate_right_region")]


@pytest.mark.parametrize("predicate,initial,closer,complete", [
    ("open", 0., -.07, -.15), ("close", -.15, -.07, .002)])
def test_drawer_joint_progress_uses_selected_drawer_not_cabinet(predicate, initial, closer, complete):
    env = audit_env([[predicate, "cabinet_middle"]], objects=(), fixtures=("cabinet",),
                    regions={"cabinet_middle": {"target": "cabinet"}},
                    sites={"cabinet_middle": site([.5, 0., .3], parent="cabinet", joints=("middle",))})
    env.fixtures_dict["cabinet"].object_properties = {"articulation": {
        "default_open_ranges": [-.16, -.14], "default_close_ranges": [0., .005]}}
    env._eval_predicate = lambda goal: bool(env.sim.data.qpos[0] < -.14 if predicate == "open" else env.sim.data.qpos[0] > 0.)
    reader = HiddenObjectAudit(env)
    measurements = []
    for qpos in (initial, closer, complete):
        env.sim.data.qpos[0] = qpos
        measurements.append(reader.sample()["goal_states"]["goal_0"])
    assert measurements[0]["distance_to_goal"] > measurements[1]["distance_to_goal"] > measurements[2]["distance_to_goal"]
    assert measurements[2]["satisfied"] is True
    assert measurements[2]["progress_kind"] == "joint_range"
    assert not measurements[2]["target_approach_supported"]
    assert measurements[2]["target_position_m"] == [.5, 0., .3]


@pytest.mark.parametrize("predicate,positions,expected", [
    ("turnon", [.25, 0.], .5), ("turnoff", [.25, 0.], .5),
    ("turnon", [.5, 0.], 0.), ("turnoff", [.5, 0.], 1.)])
def test_stove_joint_any_all_semantics(predicate, positions, expected):
    env = audit_env([[predicate, "stove"]], objects=(), fixtures=("stove",))
    env.test_joint_names.extend(["button_a", "button_b"])
    stove = env.fixtures_dict["stove"]
    stove.joints = ["button_a", "button_b"]
    stove.object_properties = {"articulation": {
        "default_turnon_ranges": [.5, 2.1], "default_turnoff_ranges": [-.005, 0.]}}
    env.sim.data.qpos[:2] = positions
    row = HiddenObjectAudit(env).sample()["goal_states"]["goal_0"]
    assert row["distance_to_goal"] == pytest.approx(expected)
    assert row["distance_unit"] == "normalized"
    assert not row["target_approach_supported"]


def test_object_on_requires_truth_without_inventing_contact_distance():
    env = audit_env([["on", "object_1", "plate"]], objects=("object_1", "plate"))
    env.sim.data.xpos[env.obj_body_id["object_1"], 2] = .6
    row = HiddenObjectAudit(env).sample()["goal_states"]["goal_0"]
    assert row["progress_kind"] == "predicate_only"
    assert row["distance_to_goal"] is None and row["requires_contact"]
    assert row["satisfied"] is False and "error" not in row
    assert row["unsupported_reason"]


def test_unavailable_predicate_or_geometry_is_an_explicit_audit_error():
    env = audit_env([["on", "object_1", "missing_site"]], regions={"missing_site": {"target": "table"}})
    row = HiddenObjectAudit(env).sample()["goal_states"]["goal_0"]
    assert row["error"].startswith("goal_geometry_unavailable")
    env = audit_env([["on", "object_1", "plate"]], objects=("object_1", "plate"))
    env._eval_predicate = lambda goal: None
    row = HiddenObjectAudit(env).sample()["goal_states"]["goal_0"]
    assert row["satisfied"] is None and row["error"].startswith("predicate_unavailable")

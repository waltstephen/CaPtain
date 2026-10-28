"""Predicate-specific reward contracts; no simulator or live services required."""
from __future__ import annotations

import copy
import importlib
import json
import os
from pathlib import Path

import pytest

from test_reward_v11e import call, robot, write_episode
import reward as R


@pytest.fixture(autouse=True)
def reward_version():
    os.environ["RL_REWARD_VERSION"] = "v12a"
    importlib.reload(R)


def state(distance=0.30, *, satisfied=False, kind="region_distance", approach=True, target=None):
    row = {"satisfied": satisfied, "distance_to_goal": distance,
           "distance_unit": "normalized" if kind == "joint_range" else "m",
           "distance_scale": 1.0 if kind == "joint_range" else 0.10,
           "progress_kind": kind, "requires_contact": False, "contact_gap_m": None,
           "target_position_m": target or [0.0, 0.0, 0.0], "destination_position_m": [0.3, 0.0, 0.15],
           "target_approach_supported": approach, "joints": [], "region_geometry": None}
    if kind == "predicate_only":
        row.update(distance_to_goal=None, distance_unit=None, distance_scale=None, requires_contact=True,
                   unsupported_reason="exact_object_contact_distance_unavailable")
    return row


def move(ci=12, *, p=None, grip=1.0):
    p = p or [0.5, 0.5, 0.5]
    return call("move_to", ci, args={"position": p, "quaternion": [1, 0, 0, 0]},
                before=robot([0.5, 0.5, 0.5], grip), after=robot(p, grip))


def episode(tmp_path, initial=None, final=None, *, calls=None, termination="request_cap_exhausted",
            success=False, predicates=None, initial_eef=None, final_eef=None, intermediate=None):
    initial = copy.deepcopy(initial or [state()])
    final = copy.deepcopy(final if final is not None else initial)
    calls = calls if calls is not None else [move()]
    predicates = predicates or ["in"] * len(initial)
    d = write_episode(tmp_path, calls=calls, scripts=["move_to()" for _ in calls], success=success,
                      termination=termination, finish_source="agent" if termination == "agent_finished" else "driver_exit")
    descriptors = [{"goal_id": f"goal_{i}", "predicate": predicate, "arguments": [f"target_{i}", f"region_{i}"],
                    "target": {"name": f"target_{i}", "kind": "object" if initial[i]["target_approach_supported"] else "site",
                               "movable": initial[i]["target_approach_supported"]},
                    "destination": {"name": f"region_{i}", "kind": "site"}}
                   for i, predicate in enumerate(predicates)]

    def snapshot(ci, capability, states, eef):
        return {"schema_version": R.GOAL_AUDIT_SCHEMA, "call_index": ci, "capability": capability,
                "eef": eef or [0.5, 0.5, 0.5], "goal_states": {f"goal_{i}": s for i, s in enumerate(states)}}

    head = snapshot(0, "reset", initial, initial_eef)
    head.update(coordinate_frame="robot_base", goals=descriptors)
    rows = [head]
    for c in calls:
        if c.get("consumes_action") and c.get("executed"):
            rows.append(snapshot(c["call_index"], c["name"], copy.deepcopy((intermediate or {}).get(c["call_index"], final)), final_eef))
    rows.append(snapshot(max([c["call_index"] for c in calls], default=0), "finish", final, final_eef))
    write_audit(d, rows)
    return d, rows


def write_audit(d, rows):
    (d / "hidden_object_audit.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_activity_and_approach_alone_never_net_positive(tmp_path):
    calls = [move(12 + 10*k, p=[0.30 + 0.02*k, 0.3, 0.3]) for k in range(5)]
    d, _ = episode(tmp_path, calls=calls, initial_eef=[0.3, 0, 0], final_eef=[0, 0, 0])
    result = R.score_episode(d)
    assert result["verdict"] == "request_cap_exhausted"
    assert result["reward"] < 0
    assert result["components"]["approach"] == pytest.approx(R.W_APPROACH_A)
    for key in ("physical", "interaction", "first_move", "closed_on_material", "lifted", "transported", "lowered", "released"):
        assert result["components"][key] == 0


def test_capped_exploration_beats_premature_quitting(tmp_path):
    q, _ = episode(tmp_path / "quit", termination="agent_stopped_without_finish")
    c, _ = episode(tmp_path / "cap", calls=[move(12 + 10*k) for k in range(40)])
    qr, cr = R.score_episode(q), R.score_episode(c)
    assert cr["reward"] > qr["reward"]
    max_process_cost = sum((R.ACTION_COST_CAP_A, R.NOOP_COST_CAP_A, R.DUPLICATE_COST_CAP_A,
                            R.TOOL_ERROR_COST_CAP_A, R.GATE_COST_CAP_A))
    assert R.W_EXHAUSTED_A - max_process_cost > R.W_NO_FINISH_A + R.W_APPROACH_A


def test_success_dominates_all_failure_rewards_and_process_costs(tmp_path):
    d, _ = episode(tmp_path, final=[state(0.0, satisfied=True)], success=True,
                   calls=[move(12 + 10*k) for k in range(40)], termination="agent_finished")
    result = R.score_episode(d)
    assert result["verdict"] == "hidden_success"
    assert result["components"]["hidden"] == 2.0
    max_process_cost = sum((R.ACTION_COST_CAP_A, R.NOOP_COST_CAP_A, R.DUPLICATE_COST_CAP_A,
                            R.TOOL_ERROR_COST_CAP_A, R.GATE_COST_CAP_A))
    failure_upper_bound = R.W_GOAL_DISTANCE_A + R.W_GOAL_SATISFIED_A + R.W_APPROACH_A + R.W_EXHAUSTED_A
    assert R.W_HIDDEN - max_process_cost - R.W_APPROACH_A > failure_upper_bound
    assert result["reward"] > failure_upper_bound


def test_no_action_is_policy_failure_with_valid_reset_and_finish(tmp_path):
    d, _ = episode(tmp_path, calls=[], termination="agent_stopped_without_finish")
    result = R.score_episode(d)
    assert result["verdict"] == "no_action"
    assert result["diagnostics"]["reward_valid"]
    assert result["reward"] == pytest.approx(R.W_NO_ACTION_A + R.W_NO_FINISH_A)


def test_progress_undo_removes_credit_and_success_ever_bonus(tmp_path):
    calls = [move(12), move(22)]
    d, _ = episode(tmp_path, calls=calls, intermediate={12: [state(0, satisfied=True)]})
    server = json.loads((d / "result.json").read_text())
    server["host_success_ever"] = True
    (d / "result.json").write_text(json.dumps(server))
    result = R.score_episode(d)
    for key in ("goal_progress", "goal_satisfied", "success_ever", "hidden"):
        assert result["components"][key] == 0


def test_predicate_goal_pairing_does_not_transfer_progress(tmp_path):
    initial = [state(), state()]
    # Goal 0's target rises, but ends at goal 1's destination: neither predicate
    # is satisfied and each exact goal's measured distance remains unchanged.
    final = [state(target=[0.8, 0, 0.10]), state(target=[0.8, 0, 0])]
    d, _ = episode(tmp_path, initial, final, calls=[move(12, grip=0.4), move(22, grip=1)])
    result = R.score_episode(d)
    assert result["components"]["goal_progress"] == 0
    assert result["components"]["goal_satisfied"] == 0
    assert result["diagnostics"]["final_goals_satisfied"] == 0
    assert all(result["components"][key] == 0 for key in ("closed_on_material", "lifted", "transported", "lowered", "released"))


def test_partial_multiobject_completion_is_averaged_per_exact_goal(tmp_path):
    initial = [state(), state()]
    final = [state(0, satisfied=True), state()]
    d, _ = episode(tmp_path, initial, final)
    result = R.score_episode(d)
    assert result["components"]["goal_satisfied"] == pytest.approx(R.W_GOAL_SATISFIED_A / 2)
    assert result["components"]["goal_progress"] == pytest.approx(R.W_GOAL_DISTANCE_A * 0.75 / 2)
    goals = result["diagnostics"]["goal_progress"]
    assert goals[0]["satisfied_delta"] == 1 and goals[1]["satisfied_delta"] == 0


def test_correct_elevated_placement_uses_region_not_initial_height(tmp_path):
    d, _ = episode(tmp_path, [state(target=[0, 0, 0])], [state(0, satisfied=True, target=[0.3, 0, 0.15])],
                   success=True, termination="agent_finished")
    result = R.score_episode(d)
    assert result["components"]["goal_satisfied"] == R.W_GOAL_SATISFIED_A
    assert result["components"]["goal_progress"] > 0
    assert result["verdict"] == "hidden_success"


def test_rising_diagonal_motion_does_not_invent_placement(tmp_path):
    d, _ = episode(tmp_path, final=[state(0.08, target=[0.2, 0, 0.04])])
    result = R.score_episode(d)
    assert result["components"]["goal_progress"] > 0
    assert result["components"]["goal_satisfied"] == 0
    assert result["components"]["lowered"] == 0 and result["components"]["released"] == 0


def test_push_gets_region_progress_without_grasp_or_lift(tmp_path):
    d, _ = episode(tmp_path, [state(0.30)], [state(0.10, target=[0.2, 0, 0])], predicates=["on"])
    result = R.score_episode(d)
    assert result["components"]["goal_progress"] == pytest.approx(R.W_GOAL_DISTANCE_A * 0.25)
    assert result["components"]["lifted"] == 0


@pytest.mark.parametrize("predicate", ["open", "close", "turnon", "turnoff"])
def test_fixture_joint_progress_needs_no_perception_label(tmp_path, predicate):
    initial, final = state(1.0, kind="joint_range", approach=False), state(0.25, kind="joint_range", approach=False)
    d, _ = episode(tmp_path, [initial], [final], predicates=[predicate])
    result = R.score_episode(d)
    assert result["components"]["goal_progress"] == pytest.approx(R.W_GOAL_DISTANCE_A * 0.3)
    assert result["components"]["approach"] == 0
    assert result["diagnostics"]["goal_progress"][0]["predicate"] == predicate


def test_unknown_contact_geometry_uses_exact_predicate_only(tmp_path):
    initial, final = state(kind="predicate_only"), state(kind="predicate_only", satisfied=True)
    d, _ = episode(tmp_path, [initial], [final], predicates=["on"], success=True, termination="agent_finished")
    result = R.score_episode(d)
    assert result["components"]["goal_progress"] == 0
    assert result["components"]["goal_satisfied"] == R.W_GOAL_SATISFIED_A
    progress = result["diagnostics"]["goal_progress"][0]
    assert progress["dense_supported"] is False and progress["unsupported_reason"]


def test_aperture_body_rise_and_future_perception_cannot_create_grasp_credit(tmp_path):
    p = [0.5, 0.5, 0.5]
    calls = [call("move_to", 12, args={"gripper": "close"}, before=robot(p, 1), after=robot(p, 0.4)),
             call("mark_point", 22, consumes=False, value={"label": "target_0", "position_m": p, "quality_flags": []})]
    d, _ = episode(tmp_path, calls=calls, final=[state(target=[0, 0, 0.04])])
    result = R.score_episode(d)
    for key in ("closed_on_material", "lifted", "transported", "lowered", "released", "close_near", "goal_satisfied", "goal_progress"):
        assert result["components"][key] == 0


@pytest.mark.parametrize("failure", ["missing", "v1", "missing_finish", "missing_physical", "missing_goal",
                                    "null_predicate", "nan", "changed_scale", "duplicate_id", "bad_final",
                                    "predicate_error", "stale_finish", "unknown_kind", "invalid_target"])
def test_invalid_private_audit_is_infrastructure_not_policy_failure(tmp_path, failure):
    d, rows = episode(tmp_path)
    if failure == "missing":
        (d / "hidden_object_audit.jsonl").unlink()
    else:
        end = rows[-1]["goal_states"]["goal_0"]
        if failure == "v1":
            rows[0]["schema_version"] = "hidden-object-audit/v1"
        elif failure == "missing_finish":
            rows.pop()
        elif failure == "missing_physical":
            rows.pop(1)
        elif failure == "missing_goal":
            rows[-1]["goal_states"] = {}
        elif failure == "null_predicate":
            end["satisfied"] = None
        elif failure == "nan":
            end["distance_to_goal"] = float("nan")
        elif failure == "changed_scale":
            end["distance_scale"] = 0.2
        elif failure == "duplicate_id":
            rows[0]["goals"].append(copy.deepcopy(rows[0]["goals"][0]))
        elif failure == "bad_final":
            end["satisfied"] = True
        elif failure == "predicate_error":
            end["error"] = "predicate unavailable"
        elif failure == "stale_finish":
            rows[-1]["call_index"] = 0
        elif failure == "unknown_kind":
            end["progress_kind"] = "aperture"
        elif failure == "invalid_target":
            rows[0]["goals"][0]["target"] = "wrong_descriptor_type"
        write_audit(d, rows)
    result = R.score_episode(d)
    assert result["verdict"] == "infrastructure_error"
    assert result["reward"] == 0 and result["diagnostics"]["reward_valid"] is False
    assert result["diagnostics"]["error"].startswith("goal_audit_")


def test_null_host_final_predicate_is_not_counted_as_failure(tmp_path):
    d, _ = episode(tmp_path)
    p = d / "result.json"
    server = json.loads(p.read_text())
    server.update(host_final_success=None, finalize_error="final scorer failed")
    p.write_text(json.dumps(server))
    result = R.score_episode(d)
    assert result["verdict"] == "infrastructure_error" and result["reward"] == 0
    assert result["diagnostics"]["error"] == "host_final_predicate_unavailable"


def test_unknown_reward_version_is_rejected():
    os.environ["RL_REWARD_VERSION"] = "v12-typo"
    with pytest.raises(ValueError, match="unsupported RL_REWARD_VERSION"):
        importlib.reload(R)

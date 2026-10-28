"""staged_progress_v1: furthest manipulation stage per goal from host goal marks, and its curriculum wiring."""
from __future__ import annotations

import pytest

import reward_segment as RS
from segment_curriculum import SegmentCurriculum


def mark(eef, target, dest=(0.6, 0.2, 0.0), satisfied=False, kind="region_distance", d=None):
    return {"eef": list(eef), "goal_states": {"goal_0": {"target_position_m": list(target), "destination_position_m": list(dest),
                                                          "progress_kind": kind, "satisfied": satisfied, "distance_to_goal": d}}}


def test_stages_are_the_furthest_state_reached():
    start = mark((0.4, -0.2, 0.40), (0.5, -0.2, 0.02))
    assert RS.staged_progress([start], start)["score"] == 0.0
    near = mark((0.5, -0.2, 0.10), (0.5, -0.2, 0.02))                         # 8 cm above the target
    touch = mark((0.5, -0.2, 0.05), (0.5, -0.2, 0.02))                        # 3 cm
    lifted = mark((0.5, -0.2, 0.15), (0.5, -0.2, 0.12))                       # target 10 cm up
    carried = mark((0.6, 0.15, 0.15), (0.6, 0.15, 0.12))                      # within 10 cm (xy) of the destination
    dropped = mark((0.4, 0.0, 0.40), (0.5, -0.1, 0.02))                       # later dropped far away
    assert RS.staged_progress([near], start)["score"] == 0.1
    assert RS.staged_progress([near, touch], start)["score"] == 0.2
    assert RS.staged_progress([touch, lifted], start)["score"] == 0.4
    assert RS.staged_progress([touch, lifted, carried, dropped], start)["score"] == 0.6   # the furthest stage counts
    assert RS.staged_progress([dropped], start)["score"] == 0.0


def test_joint_goals_score_partial_progress_and_goals_average():
    start = {"eef": [0.3, 0.0, 0.4], "goal_states": {
        "a": {"target_position_m": [0.5, 0.0, 0.2], "progress_kind": "joint_range", "distance_to_goal": 1.0, "satisfied": False},
        "b": {"target_position_m": [0.6, 0.1, 0.02], "destination_position_m": [0.7, 0.2, 0.0], "progress_kind": "region_distance",
              "satisfied": False}}}
    half = {"eef": [0.5, 0.0, 0.22], "goal_states": {
        "a": {"target_position_m": [0.5, 0.0, 0.2], "progress_kind": "joint_range", "distance_to_goal": 0.5, "satisfied": False},
        "b": {"target_position_m": [0.6, 0.1, 0.02], "destination_position_m": [0.7, 0.2, 0.0], "progress_kind": "region_distance",
              "satisfied": False}}}
    got = RS.staged_progress([half], start)
    assert got["goals"] == {"a": 0.3, "b": 0.0} and got["score"] == 0.15     # 0.6 x 50% joint progress; b untouched


def test_satisfied_goal_counts_as_carried_and_empty_marks_score_zero():
    start = mark((0.4, -0.2, 0.40), (0.5, -0.2, 0.02))
    assert RS.staged_progress([mark((0.1, 0.1, 0.5), (0.6, 0.2, 0.02), satisfied=True)], start)["score"] == 0.6
    assert RS.staged_progress([], None) == {"score": 0.0, "goals": {}}


def bank():
    trajectories = {f"t{i}": {"trajectory_id": f"t{i}", "suite": "libero_object", "task_id": i, "seed": 1, "requests": 0, "source_dir": "x"}
                    for i in range(3)}
    starts = {f"s{i}": {"start_id": f"s{i}", "trajectory_id": f"t{i}", "K": 0, "suite": "libero_object", "task_id": i, "seed": 1}
              for i in range(3)}
    return {"sha256": "abc", "trajectories": trajectories, "starts": starts}


def test_fresh_only_curriculum_entries_carry_the_reward_mode():
    c = SegmentCurriculum(bank(), episode_fraction=1.0, budget=40, reward_mode=RS.STAGED_REWARD, train_all_on_success=True)
    entries = c.next_batch(3)
    assert all(e["K"] == 0 and e["horizon"] == 40 for e in entries)
    assert all(e["reward_mode"] == RS.STAGED_REWARD and e["train_all_on_success"] for e in entries)
    plain = SegmentCurriculum(bank(), episode_fraction=1.0, budget=40)
    assert all("reward_mode" not in e and "train_all_on_success" not in e for e in plain.next_batch(2))
    state = c.state_dict()
    with pytest.raises(ValueError, match="reward mode"):
        plain.load_state_dict(state)
    fresh = SegmentCurriculum(bank(), episode_fraction=1.0, budget=40, reward_mode=RS.STAGED_REWARD, train_all_on_success=True)
    fresh.load_state_dict(state)
    assert fresh.cursor == c.cursor


def test_a_goal_satisfied_at_the_start_counts_only_if_it_still_holds():
    def two(a_sat, eef=(0.3, 0.0, 0.4)):
        return {"eef": list(eef), "goal_states": {
            "a": {"target_position_m": [0.5, 0.0, 0.2], "progress_kind": "joint_range", "distance_to_goal": 0.0, "satisfied": a_sat},
            "b": {"target_position_m": [0.6, 0.1, 0.02], "destination_position_m": [0.7, 0.2, 0.0], "progress_kind": "region_distance",
                  "satisfied": False}}}
    start = two(True)
    assert RS.staged_progress([start, two(True)], start)["goals"] == {"a": 0.6, "b": 0.0}
    assert RS.staged_progress([start, two(False)], start)["goals"] == {"a": 0.0, "b": 0.0}    # broken: no credit

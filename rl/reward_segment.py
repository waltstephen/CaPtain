"""Reward v13s: one python tool call (a segment) scored from host-only goal marks (python-runtime/v1).

A segment rollout replays the first K recorded model responses of a source episode, lets the policy produce
request K+1 (one python call, i.e. a program or a single step), runs it, and stops. The driver marks the private
goal state before every request's tools run (hidden_request_marks.jsonl); the finish row of
hidden_object_audit.jsonl is the state after the last executed call. So:

    before = mark of request K+1        after = mark of request K+2, else the finish row

and the segment reward is the v12a goal potential difference between the two (gamma = 1, so the segment terms of
one trajectory add up to its episode potential difference), plus:

    success       +W_HIDDEN   all goals satisfied after and not before
    finish        +0.5        the policy called finish() in this segment and the task is solved after it
    false_finish  -0.4        finish() in this segment while the task is not solved
    python_error  -0.05       the segment's python call failed, timed out or restarted
    format        -0.2        per live response without a valid python call (unparsed tool-call text, empty input, wrong tool)
    stop          -0.3        the policy ended its turn without a tool call and without finish()

With ``entry["horizon"] = H > 1`` the window is requests K+1 .. K+H (the policy continues; H large = to the end of the
episode): after = mark K+H+1, else the finish row. The python/format terms count bad calls over the window at the weights
above for a window of up to PENALTY_REFERENCE_H requests, scaled by PENALTY_REFERENCE_H / H beyond it (window_penalty), so
they keep their resolution over a 40-request episode. The v4 caps (3 errors, 2 malformed responses) made every fresh
episode's error term identical (it saturated on every episode).
    action_cost   -0.001 per physical request of the segment (cap 0.04)
    gate          -0.02 per finish() rejection in the segment (cap 0.04)

Replay fidelity: the before-state must match the bank's expected mark (object/EEF positions within tol_m, same
satisfied flags, same action and call counts) and the replayed calls must have the recorded names and outcomes;
otherwise the rollout is `replay_infidelity` and is not scored (never a policy example).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from reward import GOAL_AUDIT_SCHEMA, W_HIDDEN, _read_json, _read_jsonl, audited_tool_errors, goal_potentials, load_items, read_goal_audit, read_tool_audit

REWARD_VERSION = "v13s"
MARK_SCHEMA = "hidden-request-mark/v1"
W_FINISH = 0.5
W_FALSE_FINISH = -0.4
W_PYTHON_ERROR = -0.05
W_FORMAT = -0.2
PENALTY_REFERENCE_H = 4
PROGRESS_MIN = 0.01          # a probe rollout "made progress" above this task value (not any positive drift)
W_STOP = -0.3
ACTION_COST_PER_REQUEST = 0.001
ACTION_COST_CAP = 0.04
GATE_COST_PER_REJECTION = 0.02
GATE_COST_CAP = 0.04
DEFAULT_TOL_M = 0.01
UNPARSED_MARKERS = ("<tool_call>", "<function=", "<parameter=")
COMPONENT_KEYS = ("hidden", "goal_progress", "goal_satisfied", "approach", "finish", "false_finish", "python_error", "format", "stop",
                  "action_cost", "gate")


def _infra(error: str, **diag: Any) -> dict[str, Any]:
    return {"reward": 0.0, "verdict": "infrastructure_error", "components": {},
            "diagnostics": {"reward_version": REWARD_VERSION, "reward_valid": False, "infrastructure": True, "error": error, **diag}}


def read_marks(episode_dir: Path) -> dict[int, dict[str, Any]]:
    rows = _read_jsonl(Path(episode_dir) / "hidden_request_marks.jsonl")
    marks: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.get("schema_version") != MARK_SCHEMA or row.get("state_schema") != GOAL_AUDIT_SCHEMA or row.get("error"):
            raise ValueError("segment_mark_invalid")
        index = row.get("request_index")
        if type(index) is not int or index < 1 or index in marks:
            raise ValueError("segment_mark_index")
        marks[index] = row
    return marks


def _state(row: Mapping[str, Any], goals: list[dict]) -> dict[str, Any]:
    states = row.get("goal_states")
    if not isinstance(states, dict) or set(states) != {g["goal_id"] for g in goals}:
        raise ValueError("segment_mark_incomplete_goal_states")
    for goal_id, state in states.items():
        if not isinstance(state, dict) or type(state.get("satisfied")) is not bool:
            raise ValueError(f"segment_mark_invalid_predicate:{goal_id}")
    return row


def _close(a: Any, b: Any, tol: float) -> bool:
    if not (isinstance(a, list) and isinstance(b, list) and len(a) == len(b) == 3):
        return False
    return math.dist([float(x) for x in a], [float(x) for x in b]) <= tol


def fidelity(before: Mapping[str, Any], calls: list[dict], expected: Mapping[str, Any] | None, *, tol_m: float = DEFAULT_TOL_M) -> dict[str, Any]:
    """Does the replayed state equal the source episode's state at the same boundary?"""
    if not expected:
        return {"ok": True, "checked": False}
    mark = expected.get("mark") or {}
    problems = []
    if before.get("action_index") != mark.get("action_index") or before.get("call_index") != mark.get("call_index"):
        problems.append("counts")
    if not _close(before.get("eef"), mark.get("eef"), tol_m):
        problems.append("eef")
    objects, source_objects = before.get("objects") or {}, mark.get("objects") or {}
    if set(objects) != set(source_objects) or any(not _close(objects[k], source_objects[k], tol_m) for k in objects):
        problems.append("objects")
    satisfied = {k: v.get("satisfied") for k, v in (before.get("goal_states") or {}).items()}
    if satisfied != {k: v.get("satisfied") for k, v in (mark.get("goal_states") or {}).items()}:
        problems.append("satisfied")
    replayed = [[c.get("name"), bool(c.get("success"))] for c in calls if type(c.get("call_index")) is int and c["call_index"] <= (before.get("call_index") or 0)]
    if replayed != [list(x) for x in expected.get("calls") or []]:
        problems.append("calls")
    return {"ok": not problems, "checked": True, "problems": problems}


def _segment_response(episode_dir: Path, request: int) -> dict[str, Any]:
    for row in _read_jsonl(Path(episode_dir) / "driver" / "responses.jsonl"):
        if row.get("request") == request:
            return row
    return {}


def score_segment(episode_dir: Path, entry: Mapping[str, Any], *, tol_m: float = DEFAULT_TOL_M) -> dict[str, Any]:
    """Score one segment rollout; ``entry`` is the segment-bank start (K, expected mark/calls)."""
    episode_dir = Path(episode_dir)
    k = int(entry.get("K", 0))
    er = _read_json(episode_dir / "episode_result.json")
    server = _read_json(episode_dir / "result.json")
    calls = _read_jsonl(episode_dir / "calls.jsonl")
    diag: dict[str, Any] = {"reward_version": REWARD_VERSION, "start_id": entry.get("start_id"), "K": k,
                            "termination_reason": er.get("termination_reason"), "episode_dir": str(episode_dir)}
    if not er or not server or er.get("termination_reason") == "infrastructure_error":
        return _infra(er.get("error") or "episode_infrastructure_error", **diag)
    if er.get("environment_error") is not None:
        return _infra("verified_environment_error", **diag)
    if type(server.get("host_final_success")) is not bool or server.get("finalize_error"):
        return _infra("host_final_predicate_unavailable", **diag)
    try:
        audit = read_goal_audit(episode_dir, calls)
        marks = read_marks(episode_dir)
        items = load_items(episode_dir)
        tools = audited_tool_errors(read_tool_audit(episode_dir, items, er), items)
    except ValueError as exc:
        return _infra(str(exc), **diag)
    goals = audit["goals"]
    replayed_requests = [r for r in _read_jsonl(episode_dir / "driver" / "requests.jsonl") if r.get("replayed")]
    if len(replayed_requests) != k:
        return _infra(f"segment_replay_count:{len(replayed_requests)}!={k}", **diag)
    if entry.get("handoff") == "clean" and k > 0:
        # the live requests must have started from a clean handoff at K + 1 (fresh session, workspace and conversation),
        # taken at the teacher's last call and without a physical step in between (the fidelity check below reads mark K + 1)
        handoff, rows = er.get("handoff") or {}, server.get("handoffs") or []
        expected_call = ((entry.get("expected") or {}).get("mark") or {}).get("call_index")
        if (handoff.get("mode") != "clean" or handoff.get("at_request") != k + 1 or len(rows) != 1
                or (expected_call is not None and rows[0].get("call_index") != expected_call)):
            return _infra("segment_handoff_missing", **diag, handoff=handoff, server_handoffs=rows)
        diag["handoff"] = {"at_request": k + 1, "call_index": rows[0].get("call_index"), "python": handoff.get("python"),
                           "workspace": handoff.get("workspace"), "budget": handoff.get("budget")}
    if (k + 1) not in marks and server.get("finish_source") == "agent" and k > 0:
        # the driver marks only before running tools: a live text-only reply that ends the episode has no mark
        first_live = _segment_response(episode_dir, k + 1).get("items")
        if not first_live or any(it.get("type") in ("custom_tool_call", "function_call") for it in first_live):
            return _infra("segment_missing_boundary_mark", **diag)
    horizon = max(1, int(entry.get("horizon") or 1))
    end_mark = k + horizon + 1              # the mark that closes the window, if the episode got that far
    try:
        before = _state(marks[k + 1], goals) if (k + 1) in marks else _state(audit["final"], goals)
        after = _state(marks[end_mark], goals) if end_mark in marks else _state(audit["final"], goals)
    except ValueError as exc:
        return _infra(str(exc), **diag)
    check = fidelity(before, calls, entry.get("expected"), tol_m=tol_m)
    diag["fidelity"] = check
    if not check["ok"]:
        return {"reward": 0.0, "verdict": "replay_infidelity", "components": {},
                "diagnostics": {**diag, "reward_valid": False, "infrastructure": False}}
    potentials = goal_potentials({"goals": goals, "initial": before, "final": after})
    solved_before = all(s["satisfied"] for s in before["goal_states"].values())
    solved_after = all(s["satisfied"] for s in after["goal_states"].values())
    closed_by_finish = end_mark not in marks
    if closed_by_finish and solved_after != server["host_final_success"]:
        return _infra("segment_final_predicate_mismatch", **diag)
    boundary_call = before.get("call_index") or 0
    window_calls = [c for c in calls if type(c.get("call_index")) is int and c["call_index"] > boundary_call
                    and (closed_by_finish or c["call_index"] <= after.get("call_index", 0))]
    physical = sum(1 for c in window_calls if c.get("consumes_action"))
    finished_here = server.get("finish_source") == "agent" and closed_by_finish
    attempts = _read_jsonl(episode_dir / "finish_attempts.jsonl")
    started = float(before.get("t_s") or 0.0)
    rejections = sum(1 for a in attempts if a.get("verdict") == "reject" and float(a.get("t_s") or 0) >= started)
    live = [_segment_response(episode_dir, k + 1 + i) for i in range(horizon)]
    live = [row for row in live if row]
    per_request = tools.get("per_request") or {}
    python_errors = sum(int((per_request.get(i) or per_request.get(str(i)) or {}).get("python_errors") or 0)
                        for i in range(k + 1, k + 1 + horizon))
    malformed = python_calls = 0
    for row in live:
        response_items = row.get("items") or []
        valid = [it for it in response_items if it.get("type") == "custom_tool_call" and it.get("name") == "python"
                 and str(it.get("input") or "").strip()]
        python_calls += len(valid)
        any_call = [it for it in response_items if it.get("type") in ("custom_tool_call", "function_call")]
        text = " ".join(str(c.get("text", "")) for it in response_items if it.get("type") == "message" for c in it.get("content") or [])
        if not valid and (any_call or any(marker in text for marker in UNPARSED_MARKERS) or row.get("status") == "incomplete"):
            malformed += 1
    last_items = (live[-1].get("items") or []) if live else []
    ended_turn = bool(live) and not any(it.get("type") in ("custom_tool_call", "function_call") for it in last_items)
    stopped = (ended_turn and er.get("termination_reason") == "agent_stopped_without_finish" and closed_by_finish
               and not finished_here and not malformed)
    comps = dict.fromkeys(COMPONENT_KEYS, 0.0)
    comps.update(goal_progress=potentials["goal_progress"], goal_satisfied=potentials["goal_satisfied"], approach=potentials["approach"],
                 action_cost=-min(ACTION_COST_CAP, ACTION_COST_PER_REQUEST * physical),
                 gate=-min(GATE_COST_CAP, GATE_COST_PER_REJECTION * rejections))
    if solved_after and not solved_before:
        comps["hidden"] = W_HIDDEN
    if finished_here:
        comps["finish" if solved_after else "false_finish"] = W_FINISH if solved_after else W_FALSE_FINISH
    if python_errors:
        comps["python_error"] = window_penalty(W_PYTHON_ERROR, python_errors, horizon)
    if malformed:
        comps["format"] = window_penalty(W_FORMAT, malformed, horizon)
    if stopped:
        comps["stop"] = W_STOP
    if comps["hidden"]:
        verdict = "segment_success"
    elif comps["finish"]:
        verdict = "segment_finish"
    elif comps["false_finish"]:
        verdict = "false_finish"
    elif comps["format"]:
        verdict = "format_error"
    elif comps["stop"]:
        verdict = "stopped"
    elif potentials["goal_progress"] + potentials["goal_satisfied"] > 0:
        verdict = "progress"
    elif potentials["goal_progress"] + potentials["goal_satisfied"] < 0:
        verdict = "regress"
    else:
        verdict = "no_progress"
    # success is the final state (goals satisfied when the window closes, finish() or not); ever_solved also counts a state
    # that satisfied every goal at a request boundary and was later broken
    ever_solved = solved_after or any(all(bool((g or {}).get("satisfied")) for g in (marks[i].get("goal_states") or {}).values())
                                      and marks[i].get("goal_states") for i in marks if k + 1 < i <= end_mark)
    diag.update({"reward_valid": True, "infrastructure": False, "horizon": horizon, "live_requests": len(live),
                 "final_success": bool(solved_after),   # the environment's goal predicate when the window closed (the eval label)
                 "ever_solved": bool(ever_solved), "finish_attempts": sum(1 for a in attempts if float(a.get("t_s") or 0) >= started),
                 "request_use": request_use(marks, calls, per_request, k, len(live)),
                 "solved_before": solved_before, "solved_after": solved_after,
                 "host_final_success": server["host_final_success"], "segment_physical_requests": physical,
                 "segment_calls": len(window_calls), "finished_here": finished_here, "gate_rejections": rejections,
                 "python_calls": python_calls, "python_errors": python_errors, "malformed_responses": malformed,
                 "goal_progress": potentials["goals"], "goal_count": potentials["goal_count"], "teacher_delta": entry.get("teacher_delta")})
    reward = sum(comps.values())
    if entry.get("reward_mode") in (STAGED_REWARD, STAGED_V2):
        # staged_progress_v1: 1 on final success, else the mean over goals of the furthest manipulation stage the window
        # reached (host goal marks only). v1 reads only the pre-tool marks, so the
        # last request's own effect is never staged; v2 also stages the state that closes the window (the finish row or
        # the next mark)
        if entry["reward_mode"] == STAGED_REWARD:
            window = [marks[i] for i in sorted(marks) if i > k]
        else:
            window = [marks[i] for i in sorted(marks) if k < i < end_mark] + [after]
        stage = staged_progress(window, marks.get(min(marks)) if marks else None)
        reward = 1.0 if (solved_after and not solved_before) else stage["score"]
        diag.update(reward_mode=entry["reward_mode"], stage_score=stage["score"], stage_goals=stage["goals"])
    if entry.get("reward_mode") == BINARY_REWARD:
        # final_success_binary_v1: R = 1 when every original goal holds at the protocol's final state (the evaluation's
        # label), else 0; the shaping components stay as diagnostics only
        reward = 1.0 if (solved_after and not solved_before) else 0.0
        diag["reward_mode"] = BINARY_REWARD
    return {"reward": reward, "verdict": verdict, "components": comps, "diagnostics": diag}


BINARY_REWARD = "final_success_binary_v1"
STAGED_REWARD = "staged_progress_v1"
STAGED_V2 = "staged_progress_v2"
# furthest stage per goal (success itself is 1.0): gripper within 10 cm / 5 cm of the target, target lifted, target lifted
# within 10 cm (xy) of its destination; articulated goals: approach stages, then 0.6 x joint progress toward the goal
STAGE_VALUES = {"approach_10cm": 0.1, "approach_5cm": 0.2, "lifted": 0.4, "carried_near": 0.6}
LIFT_M, NEAR_M = 0.03, 0.10


def staged_progress(window: list[Mapping[str, Any]], initial: Mapping[str, Any] | None) -> dict[str, Any]:
    """Mean over the task's goals of the furthest stage reached at any request boundary of the window (a stage is a state
    predicate of that boundary; success is scored separately). ``initial`` is the episode's first mark (lift reference)."""
    import math
    goals0 = (initial or {}).get("goal_states") or {}
    per_goal: dict[str, float] = {}
    for gid in sorted(goals0):
        g0 = goals0[gid] or {}
        tp0, d0 = g0.get("target_position_m"), g0.get("distance_to_goal")
        if g0.get("satisfied"):
            # already satisfied at the start: no progress to credit, only whether it still holds at the window's end
            last = ((window[-1].get("goal_states") or {}).get(gid) or {}) if window else g0
            per_goal[gid] = STAGE_VALUES["carried_near"] if last.get("satisfied") else 0.0
            continue
        best = 0.0
        for m in window:
            g = (m.get("goal_states") or {}).get(gid) or {}
            tp, dest, eef = g.get("target_position_m"), g.get("destination_position_m"), m.get("eef")
            value = 0.0
            if tp and eef:
                gap = math.dist(eef, tp)
                value = max(value, STAGE_VALUES["approach_10cm"] if gap < 0.10 else 0.0, STAGE_VALUES["approach_5cm"] if gap < 0.05 else 0.0)
            kind = str(g.get("progress_kind") or "")
            if "joint" in kind:
                d = g.get("distance_to_goal")
                if isinstance(d, (int, float)) and isinstance(d0, (int, float)) and d0 > 1e-9:
                    value = max(value, STAGE_VALUES["carried_near"] * min(1.0, max(0.0, 1.0 - d / d0)))
            elif tp and tp0 and dest and tp[2] - tp0[2] > LIFT_M:
                value = max(value, STAGE_VALUES["lifted"])
                if math.dist(tp[:2], dest[:2]) < NEAR_M:
                    value = max(value, STAGE_VALUES["carried_near"])
            if g.get("satisfied"):
                value = max(value, STAGE_VALUES["carried_near"])
            best = max(best, value)
        per_goal[gid] = round(best, 4)
    return {"score": round(sum(per_goal.values()) / len(per_goal), 4) if per_goal else 0.0, "goals": per_goal}
REQUEST_USE_KEYS = ("physical_ok", "physical_failed", "perception_only", "no_robot_call", "python_error")


def solved_suffix_requests(episode_dir: Path, proxy_requests: list[dict], final_success: bool) -> dict[str, Any]:
    """Proxy request indices issued when every goal already held and kept holding to the end of a SUCCESSFUL trajectory
    (the state before request r's tools is its mark r; a request without a mark inherits the next known state, the final
    state being solved). The driver's live request rows join the proxy records by response id; unmatched requests keep
    their credit (counted as unmapped). Failed trajectories: none (their negative advantage stays on every request)."""
    if not final_success:
        return {"proxy_request_indices": set(), "unmapped": 0}
    episode_dir = Path(episode_dir)
    marks = read_marks(episode_dir)
    live = [r for r in _read_jsonl(episode_dir / "driver" / "requests.jsonl") if not r.get("replayed")]

    def solved(row: Mapping[str, Any]) -> bool:
        states = row.get("goal_states") or {}
        return bool(states) and all(bool((g or {}).get("satisfied")) for g in states.values())

    suffix_ids: set[str] = set()
    state = True                                   # the final state of a successful trajectory
    for row in sorted(live, key=lambda r: int(r.get("index") or 0), reverse=True):
        index = int(row.get("index") or 0)
        if index in marks:
            state = solved(marks[index])
        if not state:
            break
        if row.get("response_id"):
            suffix_ids.add(str(row["response_id"]))
    by_id = {str(q.get("response_id")): int(q["request_index"]) for q in proxy_requests if q.get("response_id")}
    return {"proxy_request_indices": {by_id[r] for r in suffix_ids if r in by_id},
            "unmapped": sum(1 for r in suffix_ids if r not in by_id)}


def request_use(marks: Mapping[int, Mapping[str, Any]], calls: list[dict], per_request: Mapping[Any, Any], k: int, live: int) -> dict[str, int]:
    """What the live requests were spent on. Each request's robot calls are the calls between its boundary mark and the next
    one: physical_ok (a physical action succeeded), physical_failed (physical actions, none succeeded), perception_only
    (robot calls, none physical), no_robot_call. python_error counts requests whose cell raised, timed out or restarted
    (it overlaps the other four)."""
    out = dict.fromkeys(REQUEST_USE_KEYS, 0)
    indexed = [c for c in calls if type(c.get("call_index")) is int]
    for r in range(k + 1, k + 1 + live):
        if r not in marks:
            continue
        lo = int(marks[r].get("call_index") or 0)
        hi = int(marks[r + 1].get("call_index") or 0) if (r + 1) in marks else None
        mine = [c for c in indexed if c["call_index"] > lo and (hi is None or c["call_index"] <= hi)]
        physical = [c for c in mine if c.get("consumes_action")]
        if any(c.get("success") for c in physical):
            out["physical_ok"] += 1
        elif physical:
            out["physical_failed"] += 1
        elif mine:
            out["perception_only"] += 1
        else:
            out["no_robot_call"] += 1
        if int((per_request.get(r) or per_request.get(str(r)) or {}).get("python_errors") or 0):
            out["python_error"] += 1
    return out


def _use_rates(rows: list[Mapping[str, Any]]) -> dict[str, float]:
    totals = dict.fromkeys(REQUEST_USE_KEYS, 0)
    for r in rows:
        for key, value in ((r.get("diagnostics") or {}).get("request_use") or {}).items():
            if key in totals:
                totals[key] += int(value)
    live = sum(totals[key] for key in REQUEST_USE_KEYS[:4])
    return {key: totals[key] / max(1, live) for key in REQUEST_USE_KEYS}


def window_penalty(weight: float, count: int, horizon: int) -> float:
    """``weight`` per bad call in a window of up to PENALTY_REFERENCE_H requests, scaled down for longer windows; bounded
    by the weight of a window where every call is bad, monotonic in ``count`` over the whole window."""
    scale = PENALTY_REFERENCE_H / max(PENALTY_REFERENCE_H, int(horizon))
    return max(weight * PENALTY_REFERENCE_H, weight * int(count) * scale)


def reached(result: Mapping[str, Any], teacher_delta: float | None) -> bool:
    """Curriculum criterion for one rollout: it solved the task or finished correctly; with a one-call horizon it may
    instead make half the teacher's progress, or (where the teacher's call made no measurable progress, e.g. perception)
    run a valid python call without regressing."""
    if result.get("verdict") in ("segment_success", "segment_finish"):
        return True
    diag = result.get("diagnostics") or {}
    if diag.get("horizon", 1) != 1 or result.get("verdict") in ("infrastructure_error", "replay_infidelity"):
        return False
    comps = result.get("components") or {}
    progress = float(comps.get("goal_progress", 0.0)) + float(comps.get("goal_satisfied", 0.0))
    if teacher_delta is not None and teacher_delta > 0:
        return progress >= 0.5 * teacher_delta
    return result.get("verdict") in ("no_progress", "progress") and not comps.get("python_error")


TASK_COMPONENTS = ("hidden", "goal_progress", "goal_satisfied", "finish", "false_finish")
_INVALID = ("infrastructure_error", "replay_infidelity")


def _task_value(result: Mapping[str, Any]) -> float:
    comps = result.get("components") or {}
    return sum(float(comps.get(k, 0.0)) for k in TASK_COMPONENTS)


def group_signal_stats(results: list[Mapping[str, Any]]) -> dict[str, float]:
    """What contrast each scored group carries. A nonzero reward spread alone overstates the signal: one python error
    makes a group "informative" without saying anything about the task. Classes, in order: a success/failure mix; a
    difference in task components (progress, predicates, finish); only shaping or penalty differences; identical rewards."""
    groups: dict[Any, list[Mapping[str, Any]]] = {}
    for r in results:
        if r.get("verdict") not in _INVALID:
            groups.setdefault(r.get("group"), []).append(r)
    counts = dict.fromkeys(("mixed_success", "substantive", "weak_only", "flat"), 0)
    scored = [rows for rows in groups.values() if len(rows) >= 2]
    for rows in scored:
        success = [r.get("verdict") == "segment_success" for r in rows]
        task = [_task_value(r) for r in rows]
        reward = [float(r.get("reward") or 0.0) for r in rows]
        if any(success) and not all(success):
            counts["mixed_success"] += 1
        elif max(task) - min(task) > 1e-3:
            counts["substantive"] += 1
        elif max(reward) - min(reward) > 1e-6:
            counts["weak_only"] += 1
        else:
            counts["flat"] += 1
    valid = [r for rows in groups.values() for r in rows]
    return {"signal/groups": len(scored), **{f"signal/{k}": v / max(1, len(scored)) for k, v in counts.items()},
            "reward_valid": sum(float(r.get("reward") or 0.0) for r in valid) / max(1, len(valid))}


def probe_stats(results: list[Mapping[str, Any]], sets: Mapping[str, list[str]]) -> dict[str, float]:
    """Fixed-start probe (no gradient): per named set, the rollout success and progress rates, the mean reward and the
    share of starts solved at least once."""
    name_of = {start: name for name, ids in sets.items() for start in ids}
    out: dict[str, float] = {}
    for name, ids in sets.items():
        rows = [r for r in results if name_of.get(r.get("start_id")) == name and r.get("verdict") not in _INVALID]
        n = len(rows)
        solved = {r.get("start_id") for r in rows if r.get("verdict") == "segment_success"}
        out.update({f"probe/{name}/n": n,
                    f"probe/{name}/success": sum(r.get("verdict") == "segment_success" for r in rows) / max(1, n),
                    f"probe/{name}/progress": sum(_task_value(r) > PROGRESS_MIN for r in rows) / max(1, n),
                    f"probe/{name}/reward": sum(float(r.get("reward") or 0.0) for r in rows) / max(1, n),
                    f"probe/{name}/starts_solved": len(solved) / max(1, len(ids)),
                    f"probe/{name}/ever_solved": sum(bool((r.get("diagnostics") or {}).get("ever_solved")) for r in rows) / max(1, n),
                    **{f"probe/{name}/use_{key}": v for key, v in _use_rates(rows).items()}})
    return out


def branch_stats(results: list[Mapping[str, Any]]) -> dict[str, float]:
    """Per branch (fresh = K 0 episodes, window = teacher-state starts): new-success rate, python error rate per call,
    and the branch's share of the batch's completion tokens (the objective is token-normalized)."""
    valid = [r for r in results if r.get("verdict") not in _INVALID]
    tokens_all = sum(float((r.get("diagnostics") or {}).get("full_completion_tokens") or 0) for r in valid)
    out: dict[str, float] = {}
    for name, rows in (("fresh", [r for r in valid if not r.get("K")]), ("window", [r for r in valid if r.get("K")])):
        diag = [r.get("diagnostics") or {} for r in rows]
        calls = sum(int(d.get("python_calls") or 0) for d in diag)
        out.update({f"branch/{name}/n": len(rows),
                    f"branch/{name}/success": sum(r.get("verdict") == "segment_success" for r in rows) / max(1, len(rows)),
                    f"branch/{name}/python_error_rate": sum(int(d.get("python_errors") or 0) for d in diag) / max(1, calls),
                    f"branch/{name}/token_share": sum(float(d.get("full_completion_tokens") or 0) for d in diag) / max(1.0, tokens_all),
                    f"branch/{name}/ever_solved": sum(bool(d.get("ever_solved")) for d in diag) / max(1, len(rows)),
                    **{f"branch/{name}/use_{key}": v for key, v in _use_rates(rows).items()}})
    return out


__all__ = ["COMPONENT_KEYS", "REWARD_VERSION", "branch_stats", "fidelity", "group_signal_stats", "probe_stats", "reached",
           "read_marks", "request_use", "score_segment", "solved_suffix_requests", "staged_progress", "window_penalty"]

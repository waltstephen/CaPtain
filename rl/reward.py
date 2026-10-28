#!/usr/bin/env python3
"""Episode reward / verdict scorer for CaPtain-Harness ReAct rollouts.

The final CaPtain pipeline trains on the binary host outcome plus the simulator-critic shaping (reward_mode
final_success_binary_v1, advantage_mode physcritic); this scorer still produces the episode verdict and the logged
shaped score (RL_REWARD_VERSION v12b in the reference run). The earlier shaped versions (v11a-v11i, v12a) are kept
for reproducibility of ablations.

    score_episode(episode_dir) -> {"reward": float, "verdict": str, "components": {...}, "diagnostics": {...}}

Reads only host-side files written after the episode: episode_result.json (driver + host summary),
result.json (server), calls.jsonl (every capability call with its receipt), driver/transcript.jsonl
(tool calls / outputs, for the error count). Nothing here is served to the agent.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
from pathlib import Path
from typing import Any

REWARD_VERSION = os.environ.get("RL_REWARD_VERSION", "v11a")
if REWARD_VERSION not in {f"v11{x}" for x in "abcdefghi"} | {"v12a", "v12b"}:
    raise ValueError(f"unsupported RL_REWARD_VERSION: {REWARD_VERSION!r}")
V12A = REWARD_VERSION == "v12a"
V12B = REWARD_VERSION == "v12b"

W_HIDDEN = 2.00
W_SUCCESS_EVER = 0.50
W_FALSE_FINISH = -1.00
W_GATE_REJECTION = -0.10
GATE_REJECTION_CAP = 3
W_PHYSICAL = 0.30
PHYSICAL_FULL = 5
W_FIRST_MOVE = 0.10
W_RETAIN = 0.25
W_ALL_PHYSICAL_OK = 0.10
W_NO_ACTION = -0.80
W_TOOL_ERROR = -0.05
TOOL_ERROR_CAP = 6
W_NO_FINISH = -0.30
W_EXHAUSTED = -0.20
LENGTH_FREE_TOKENS = 20000
LENGTH_PENALTY_PER = 0.10
LENGTH_PENALTY_UNIT = 4000
# v11b: verbosity of the tool outputs the student asks for. Text tokens added to the context per
# model request (context growth between consecutive requests minus the previous completion, image pad tokens excluded),
# averaged over the episode; -0.05 per 1k tokens above 1.5k/request, capped at -0.6. Measured from the proxy record.
W_VERBOSITY_PER_K = -0.05
VERBOSITY_FREE_PER_REQ = 1500
VERBOSITY_CAP = 0.6
IMAGE_PAD_ID = 248056
# v11c: (1) the driver's `request cap N reached` is a policy outcome, termination request_cap_exhausted,
# same -0.20 as the other exhaustions, trained with weight 1 (was infrastructure_error -> weight 0); (2) verbosity slope -0.15/1k
# above 1.5k tool tokens per request (cap -0.6) plus -0.10 per single tool output above 8k tokens (cap -0.6); (3) grasp shaping
# +0.15 for the first executed gripper close within 5 cm of a detected object or with measured contact; (4) first_move +0.05.
V11C = REWARD_VERSION not in ("v11a", "v11b")
W_VERBOSITY_PER_K_C = -0.15
W_BIG_OUTPUT = -0.10
BIG_OUTPUT_TOKENS = 8000
BIG_OUTPUT_CAP = 0.6
W_GRASP = 0.15
GRASP_RADIUS_M = 0.05
CONTACT_FORCE_INCREASE_N = 2.0
W_FIRST_MOVE_C = 0.05
_REQUEST_CAP_RE = re.compile(r"request cap \d+ reached", re.I)
# v11d: staged manipulation ladder (the teacher's sequence) replaces the proximity grasp term and retain;
# empty closes earn nothing. Plus a charge on the perception/setup preamble before the first physical attempt.
V11D = REWARD_VERSION not in ("v11a", "v11b", "v11c")
W_CLOSED_ON_MATERIAL = 0.10
W_LIFTED = 0.15
W_TRANSPORTED = 0.20
W_LOWERED_NEAR_TARGET = 0.20
W_RELEASED_AFTER_PLACE = 0.10
LIFT_M = 0.04
TRANSPORT_M = 0.10
LOWER_M = 0.04
PLACE_RADIUS_M = 0.08
W_PREAMBLE_PER_REQUEST = -0.02
PREAMBLE_FREE_REQUESTS = 12
PREAMBLE_CAP = 0.3
_PHYSICAL_SCRIPT_RE = re.compile(r"move_to|gripper|\brecover\s*\(|\brelease\s*\(|\bhold\s*\(|['\"](?:recover|release|hold)['\"]")
# v11e (RL_REWARD_VERSION=v11e): interaction shaping with bounded anti-spin terms.
# (1) interaction: +0.03 for each of the first 6 model requests that execute >= 1 clean physical action (move_to/hold/release; recover
#     excluded), +0.01 per further such request, cap +0.30; (2) noop: after the first request that requests a physical action, each request
#     that neither requests a physical action nor observes (perception capability or view_image) is a no-op (previews, recover-only, file
#     edits, pure text); the final request and finish() requests are exempt; -0.03 per no-op beyond 4, cap -0.24; (3) duplicate: a physical
#     request identical to an earlier one (same capability + gripper command, position within 1 cm; non-pose actions: identical arguments)
#     repeated more than twice costs -0.05 per extra repeat, cap -0.20. Calls are attributed to requests by wall time (server.json
#     started_wall_time + calls.jsonl t_s vs driver/requests.jsonl t); items come from driver/items.jsonl when present (transcript.jsonl
#     is the compacted final history) else driver/transcript.jsonl; tool_output_tokens sums per-transition growth (compaction-robust).
V11E = REWARD_VERSION not in ("v11a", "v11b", "v11c", "v11d")
# v11f: evaluations stayed at 0 success while the ladder was reached in <4 % of
# rollouts and ~0.5 of the reward came from mere activity (physical + interaction). v11f adds dense shaping toward the object
# (approach: EEF-to-perceived-centroid distance; close_near: a gripper close within CLOSE_NEAR_M of a perceived object, even if
# empty) and halves the activity credit so progress, not motion, carries the reward.
V11F = REWARD_VERSION not in ("v11a", "v11b", "v11c", "v11d", "v11e")
W_APPROACH = 0.15          # * clip(1 - min_dist / APPROACH_FULL_M, 0, 1)
APPROACH_FULL_M = 0.15
W_CLOSE_NEAR = 0.10        # a gripper close with the EEF within CLOSE_NEAR_M (3D) of a perceived object centroid
CLOSE_NEAR_M = 0.05
# v11g: v11f made the student perceive more but act less (median actions 6.5 vs 11 under
# v11e, 6 no_action, still 0/40; closed/lifted flat at 0-3 of 48 for 24 steps). v11g keeps the v11f shaping but restores the v11e
# activity credit (physical / interaction caps back to 0.30), shrinks the one-off perceive bonus (perception saturated at 40+/48)
# and makes the grasp rungs worth more than any amount of looking: close_near 0.15, closed_on_material 0.20, lifted 0.25.
V11G = REWARD_VERSION not in ("v11a", "v11b", "v11c", "v11d", "v11e", "v11f")
W_PERCEIVE_G = 0.02
W_CLOSE_NEAR_G = 0.15
W_CLOSED_ON_MATERIAL_G = 0.20
W_LIFTED_G = 0.25
# v11h: progress must be VERIFIED before it is
# paid, and activity alone must not net positive.
#  (1) activity credit cut so that "act cleanly, never progress, exhaust the budget" nets <= 0: physical 0.10 (was 0.30), interaction cap 0.05
#      (was 0.30), first_move 0.02, all_physical_ok 0 (exhausted stays -0.20);
#  (2) target grounding: perception points count only if the call succeeded, carries no quality flag (mark_point/point: no
#      label_point_disagreement / implausible_* / sparse_geometry / depth_discontinuity) and its label names the task's target object
#      (words of the BDDL target names from hidden_object_audit.jsonl, else of the task text); `point` outputs are accepted like mark_point;
#  (3) privileged ladder from hidden_object_audit.jsonl (robot-base positions of the goal's target/destination
#      bodies + EEF after every executed physical action): approach = EEF-to-true-target distance, close_near = a successful close within
#      CLOSE_NEAR_M of the true target, closed = that close also stops the jaws on material, lifted = the target body itself rises
#      >= TRUE_LIFT_M while the jaws hold, transported = target xy moves >= TRUE_TRANSPORT_M, lowered = target within PLACE_RADIUS_M (xy)
#      of a destination body and back down, released = jaws open afterwards. Jaw aperture alone never counts as a grasp any more; a
#      close with success=false never counts. Without hidden data (fixture tasks: open/close/turn) the same rungs run on trusted
#      perception points with the close gated on success + proximity.
V11H = REWARD_VERSION not in ("v11a", "v11b", "v11c", "v11d", "v11e", "v11f", "v11g")
W_PHYSICAL_H = 0.10
INTERACT_CAP_H = 0.05
W_FIRST_MOVE_H = 0.02
W_APPROACH_H = 0.15
APPROACH_FULL_H_M = 0.30      # true-target distance is verified, so the ramp can be wide: batch median closest approach was ~20 cm
CLOSE_NEAR_TRUE_M = 0.08      # hidden mode: EEF to the target BODY CENTRE (a rim grasp of a bowl is ~6 cm from it); perception mode keeps CLOSE_NEAR_M
W_CLOSE_NEAR_H = 0.10
W_CLOSED_H = 0.20
W_LIFTED_H = 0.30
W_TRANSPORTED_H = 0.20
W_PLACED_H = 0.20
W_RELEASED_H = 0.10
TRUE_LIFT_M = 0.03
TRUE_TRANSPORT_M = 0.10
PLACE_Z_SLACK_M = 0.05
# v11i: under v11h quitting became the cheapest policy - stop-early episodes (no_finish + no_action) were up to
# 27 of 48 per step and true lifts per 4 steps fell from 8 to 2.
# Stopping without finish (-0.30) plus nothing else beat acting to the request cap (-0.20 exhausted, up to -0.24 noop, -0.30 tool_error).
# v11i keeps every v11h term and re-orders the terminal outcomes: no_finish -0.60, request cap -0.10 (acting to the cap is the desired
# fallback), noop cap 0.12.
V11I = REWARD_VERSION not in ("v11a", "v11b", "v11c", "v11d", "v11e", "v11f", "v11g", "v11h")
W_NO_FINISH_I = -0.60
W_REQUEST_CAP_I = -0.10
NOOP_CAP_I = 0.12
# v12a uses gamma=1 potential differences on exact, private goal predicates.
# All process costs together are <= 0.20; terminal success (+2) dominates every
# unsuccessful outcome. Approach alone (<= 0.04) cannot offset exhaustion (-0.05).
GOAL_AUDIT_SCHEMA = "hidden-object-audit/v2"
W_GOAL_DISTANCE_A = 0.60
W_GOAL_SATISFIED_A = 0.40
W_APPROACH_A = 0.04
APPROACH_SCALE_A_M = 0.30
W_NO_FINISH_A = -0.30
W_FALSE_FINISH_A = -0.40
W_EXHAUSTED_A = -0.05
W_NO_ACTION_A = -0.40
ACTION_COST_CAP_A = 0.04
NOOP_COST_CAP_A = 0.04
DUPLICATE_COST_CAP_A = 0.04
TOOL_ERROR_COST_CAP_A = 0.05
GATE_COST_CAP_A = 0.03
# v12b changes only the error evidence, not the v12a weights or goal potentials.
# Complete driver audit is mandatory; historical printed JSON is diagnostic-only.
TOOL_AUDIT_SCHEMA = "tool-execution-audit/v1"
TOOL_RUNTIME_CONTRACT = "code-mode-runtime/v2"
PYTHON_RUNTIME_CONTRACT = "python-runtime/v1"
TOOL_RUNTIME_CONTRACTS = frozenset((TOOL_RUNTIME_CONTRACT, PYTHON_RUNTIME_CONTRACT))
PYTHON_CELL_STATUSES = frozenset(("ok", "error", "timeout", "restarted", "rejected", "unavailable"))
ENVIRONMENT_ERROR_SCHEMA = "harness-environment-error/v1"
ENVIRONMENT_PREFLIGHT_SCHEMA = "harness-toolchain-preflight/v1"
_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_SHELL_TOOLS = frozenset(("exec_command", "write_stdin"))
_NORMAL_EXIT_ONE = frozenset(("rg", "grep", "egrep", "fgrep", "test", "[", "diff", "cmp"))
_TASK_STOPWORDS = frozenset(("the", "a", "an", "and", "it", "its", "of", "to", "in", "on", "into", "onto", "up", "down", "from", "with", "both", "then",
                             "pick", "place", "put", "open", "close", "turn", "push", "move", "take", "grab", "get", "stack", "at", "next", "between",
                             "front", "back", "side", "region", "area", "table", "object", "objects", "can", "box", "bottle", "handle"))
_TASK_SPLIT_RE = re.compile(r"\s(?:and place|and put|on the|in the|into the|onto the|to the|on top of)\s", re.I)
_WORD_RE = re.compile(r"[a-z]+")
W_PERCEIVE = 0.05          # v11f: once, when a perception call returned a 3D object point (student mostly only calls observe)
W_PHYSICAL_F = 0.15        # v11f: half of W_PHYSICAL
INTERACT_CAP_F = 0.15      # v11f: half of INTERACT_CAP
W_INTERACT_FIRST = 0.03
INTERACT_FIRST_N = 6
W_INTERACT_AFTER = 0.01
INTERACT_CAP = 0.30
W_NOOP = -0.03
NOOP_FREE = 4
NOOP_CAP = 0.24
W_DUP = -0.05
DUP_FREE_REPEATS = 2
DUP_CAP = 0.20
DUP_RADIUS_M = 0.01
PERCEPTION_CALLS = frozenset(("observe", "segment", "select_detection", "point", "mark_point", "track_target", "read_motion_trajectory",
                              "get_robot_state", "gripper_state", "measure_image_points", "inspect_contact_geometry", "compare_object_observations",
                              "inspect_release_geometry", "measure_object_geometry"))
INTERACTION_CALLS = frozenset(("move_to", "hold", "release"))   # physical calls that touch the scene; recover only retreats
_OBSERVE_SCRIPT_RE = re.compile(r"view_image|observe\s*\(|\.png\b")
_FINISH_SCRIPT_RE = re.compile(r"\bfinish\s*\(|/finish\b")

_FORCE_STOP_RE = re.compile(r"force|contact|collision|stall|timeout|guard|limit|emergency|blocked|reject|interrupt|ik_unavailable|unavailable", re.I)
_TOOL_ERROR_RE = re.compile(r"^(unsupported call|.* failed: |failed to parse function arguments)", re.S)
_EXIT_RE = re.compile(r"^Process exited with code (-?\d+)$", re.M)
_UNPARSED_MARKERS = ("<tool_call>", "<function=")


def _read_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return rows


def _result(reward: float, verdict: str, components: dict[str, float] | None = None, **diag: Any) -> dict:
    return {"reward": float(reward), "verdict": verdict, "components": components or {}, "diagnostics": diag}


def _gripper(value: Any, key: str) -> float | None:
    if not isinstance(value, dict):
        return None
    robot = value.get(key)
    if isinstance(robot, dict):
        g = robot.get("gripper_open_fraction")
        if isinstance(g, (int, float)):
            return float(g)
    return None


def physical_summary(calls: list[dict]) -> dict[str, Any]:
    """Executed / clean / retention statistics over the physical calls of calls.jsonl."""
    executed = 0
    clean = 0
    ok = 0
    requested = 0
    first_move = False
    retain = False
    holding = False
    hold_open: float | None = None
    for row in calls:
        if not row.get("consumes_action"):
            continue
        requested += 1
        if not row.get("executed"):
            continue
        executed += 1
        value = row.get("value") if isinstance(row.get("value"), dict) else {}
        reason = str(value.get("stopped_reason") or value.get("limitation") or "")
        if row.get("success"):
            ok += 1
            if not _FORCE_STOP_RE.search(reason):
                clean += 1
        if row.get("name") == "move_to":
            first_move = True
        before, after = _gripper(value, "robot_before"), _gripper(value, "robot_after")
        if before is None or after is None:
            continue
        if holding and hold_open is not None and 0.03 < after < 0.90 and abs(after - hold_open) < 0.05:
            retain = True  # jaws still closed on material through a later physical call
        if 0.03 < after < 0.90 and after < before - 0.10:
            holding, hold_open = True, after
        elif not (0.03 < after < 0.90):
            holding, hold_open = False, None
    return {"requested": requested, "executed": executed, "ok": ok, "clean": clean, "first_move": first_move, "retain": retain,
            "all_ok": executed > 0 and ok == requested}


def _vec3(v: Any) -> list[float] | None:
    return [float(x) for x in v] if isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v) else None


def _detected_points(name: str, value: Any) -> list[tuple[str, list[float]]]:
    """3D points of detected objects reported by perception calls (world frame, metres)."""
    if not isinstance(value, dict):
        return []
    pts: list[tuple[str, list[float]]] = []
    label = str(value.get("label") or value.get("prompt") or name)
    if name == "mark_point":
        for k in ("position_m", "point_centroid_m"):
            if (p := _vec3(value.get(k))) is not None:
                pts.append((label, p))
    elif name == "segment":
        for d in value.get("detections") or []:
            if isinstance(d, dict) and (p := _vec3(d.get("centroid_m"))) is not None:
                pts.append((str(d.get("label") or label), p))
    elif name == "select_detection":
        if (p := _vec3(value.get("centroid_m"))) is not None:
            pts.append((label, p))
    elif name == "measure_object_geometry":
        obj = value.get("object")
        if isinstance(obj, dict) and (p := _vec3(obj.get("center_m"))) is not None:
            pts.append((str(obj.get("label") or label), p))
    return pts


def _is_gripper_close(args: Any) -> bool:
    if not isinstance(args, dict):
        return False
    g = args.get("gripper")
    if g is None and isinstance(args.get("arguments"), dict):
        g = args["arguments"].get("gripper")
    if isinstance(g, str):
        return g.lower() in ("close", "closed")
    if isinstance(g, (int, float)):
        return float(g) <= 0.5  # commanded open fraction
    return False


def grasp_attempt(calls: list[dict]) -> dict[str, Any]:
    """First executed gripper close within GRASP_RADIUS_M of a detected object, or with measured contact (jaws stopped on
    material, force increase, or retention reported)."""
    points: list[tuple[str, list[float]]] = []
    for row in calls:
        name = str(row.get("name") or "")
        if not row.get("consumes_action"):
            points.extend(_detected_points(name, row.get("value")))
            continue
        if not row.get("executed") or not _is_gripper_close(row.get("arguments")):
            continue
        value = row.get("value") if isinstance(row.get("value"), dict) else {}
        diag = row.get("diagnostics") if isinstance(row.get("diagnostics"), dict) else {}
        before, after = _gripper(value, "robot_before"), _gripper(value, "robot_after")
        contact = None
        if before is not None and after is not None and 0.03 < after < 0.90 and after < before - 0.10:
            contact = "jaws_stopped_on_material"
        f0, f1 = value.get("initial_force_n"), value.get("max_observed_force_n")
        if contact is None and isinstance(f0, (int, float)) and isinstance(f1, (int, float)) and f1 - f0 >= CONTACT_FORCE_INCREASE_N:
            contact = "force_increase"
        cfm = diag.get("closure_force_monitor") if isinstance(diag.get("closure_force_monitor"), dict) else {}
        cm = cfm.get("closure_motion") if isinstance(cfm.get("closure_motion"), dict) else {}
        if contact is None and (cm.get("physical_retention_verified") is True or value.get("object_retention") == "verified"):
            contact = "retention_verified"
        eefs = []
        for key in ("robot_after", "robot_before"):
            r = value.get(key)
            if isinstance(r, dict) and (p := _vec3(r.get("end_effector_position_m"))) is not None:
                eefs.append(p)
        for key in ("initial_pose", "final_pose"):
            r = cm.get(key)
            if isinstance(r, dict) and (p := _vec3(r.get("eef_position_m"))) is not None:
                eefs.append(p)
        best: tuple[float, str] | None = None
        for e in eefs:
            for label, q in points:
                d = sum((a - b) ** 2 for a, b in zip(e, q)) ** 0.5
                if best is None or d < best[0]:
                    best = (d, label)
        near = best is not None and best[0] <= GRASP_RADIUS_M
        if contact or near:
            return {"attempt": True, "how": contact or "near", "nearest_m": round(best[0], 3) if best else None,
                    "nearest_label": best[1] if best else None, "call_index": row.get("call_index")}
        # first close that neither touched nor was near anything: keep looking (a later close may qualify)
    return {"attempt": False, "how": None}


def _eef(value: dict, key: str) -> list[float] | None:
    r = value.get(key) if isinstance(value, dict) else None
    return _vec3(r.get("end_effector_position_m")) if isinstance(r, dict) else None


def manipulation_ladder(calls: list[dict]) -> dict[str, Any]:
    """Milestones of the teacher's sequence, each at most once: closed_on_material (jaws stop in (0.03, 0.90) after dropping > 0.10),
    lifted (EEF z + LIFT_M while holding), transported (EEF xy > TRANSPORT_M from the grasp point while holding), lowered_near_target
    (EEF drops > LOWER_M while holding, within PLACE_RADIUS_M (xy) of a point reported by a perception call), released_after_place
    (jaws open > 0.90 after lowered_near_target). Empty closes (jaws to ~0) count for nothing."""
    got = {"closed_on_material": False, "lifted": False, "transported": False, "lowered_near_target": False, "released_after_place": False}
    events: list[tuple[str, Any]] = []
    points: list[tuple[str, list[float]]] = []
    min_dist: float | None = None      # v11f: closest the EEF got (3D) to any perceived object centroid known at that time
    close_near_object = False          # v11f: any gripper close within CLOSE_NEAR_M of a perceived centroid
    holding = False
    hold_open: float | None = None
    grasp_pos: list[float] | None = None
    for row in calls:
        name = str(row.get("name") or "")
        if not row.get("consumes_action"):
            points.extend(_detected_points(name, row.get("value")))
            continue
        if not row.get("executed"):
            continue
        value = row.get("value") if isinstance(row.get("value"), dict) else {}
        before, after = _gripper(value, "robot_before"), _gripper(value, "robot_after")
        p0, p1 = _eef(value, "robot_before"), _eef(value, "robot_after")
        if p1 is not None and points:
            d = min(((p1[0] - q[0]) ** 2 + (p1[1] - q[1]) ** 2 + (p1[2] - q[2]) ** 2) ** 0.5 for _, q in points)
            min_dist = d if min_dist is None else min(min_dist, d)
            if _is_gripper_close(row.get("arguments")) and d <= CLOSE_NEAR_M and not close_near_object:
                close_near_object = True
                events.append(("close_near_object", row.get("call_index"), round(d, 3)))
        if _is_gripper_close(row.get("arguments")):
            if before is not None and after is not None and 0.03 < after < 0.90 and after < before - 0.10:
                holding, hold_open, grasp_pos = True, after, p1
                if not got["closed_on_material"]:
                    got["closed_on_material"] = True
                    events.append(("closed_on_material", row.get("call_index"), round(after, 3)))
            else:
                holding, hold_open, grasp_pos = False, None, None
                events.append(("empty_close", row.get("call_index"), None if after is None else round(after, 3)))
            continue
        if not holding:
            continue
        if after is None or not (0.03 < after < 0.90 and hold_open is not None and abs(after - hold_open) < 0.08):
            if after is not None and after > 0.90 and got["lowered_near_target"] and not got["released_after_place"]:
                got["released_after_place"] = True
                events.append(("released_after_place", row.get("call_index"), round(after, 3)))
            else:
                events.append(("lost_hold", row.get("call_index"), None if after is None else round(after, 3)))
            holding, hold_open, grasp_pos = False, None, None
            continue
        if p1 is None or grasp_pos is None:
            continue
        if not got["lifted"] and p1[2] > grasp_pos[2] + LIFT_M:
            got["lifted"] = True
            events.append(("lifted", row.get("call_index"), round(p1[2] - grasp_pos[2], 3)))
        dxy = ((p1[0] - grasp_pos[0]) ** 2 + (p1[1] - grasp_pos[1]) ** 2) ** 0.5
        if not got["transported"] and dxy > TRANSPORT_M:
            got["transported"] = True
            events.append(("transported", row.get("call_index"), round(dxy, 3)))
        if got["transported"] and not got["lowered_near_target"] and p0 is not None and p1[2] < p0[2] - LOWER_M:
            near = any(((p1[0] - q[0]) ** 2 + (p1[1] - q[1]) ** 2) ** 0.5 <= PLACE_RADIUS_M for _, q in points)
            if near:
                got["lowered_near_target"] = True
                events.append(("lowered_near_target", row.get("call_index"), round(p0[2] - p1[2], 3)))
    return {**got, "events": events[:10], "min_dist_to_point": None if min_dist is None else round(min_dist, 4), "close_near_object": close_near_object,
            "perceived_points": len(points)}


# ----------------------------------------------------------------------------- v11h: verified grounding + privileged ladder
def _words(text: Any) -> set[str]:
    return {w for w in _WORD_RE.findall(str(text or "").lower()) if w not in _TASK_STOPWORDS}


def _entity_words(name: str) -> set[str]:
    """BDDL body name -> words: akita_black_bowl_1 -> {akita, black, bowl}."""
    return {w for w in re.split(r"[_\W]+", name.lower()) if w and not w.isdigit() and w not in _TASK_STOPWORDS}


def target_words(task: Any, hidden: dict | None) -> set[str]:
    """Words that a perception label must share to count as the task's target object."""
    if hidden and hidden.get("targets"):
        words: set[str] = set()
        for name in hidden["targets"]:
            words |= _entity_words(name)
        if words:
            return words
    text = str(task or "").strip()
    if not text:
        return set()
    head = _TASK_SPLIT_RE.split(text, maxsplit=1)[0]
    words = _words(head)
    return words or _words(text)


def hidden_objects(episode_dir: Path) -> dict[str, Any] | None:
    """hidden_object_audit.jsonl (harness server 23629ea): goal entities + robot-base positions per executed physical call."""
    rows = _read_jsonl(Path(episode_dir) / "hidden_object_audit.jsonl")
    if not rows:
        return None
    head = next((r for r in rows if r.get("call_index") == 0 and isinstance(r.get("objects"), dict)), None)
    if head is None:
        return None
    movable = set(head.get("movable") or [])
    targets = [str(t) for t in (head.get("targets") or [])]
    by_call: dict[int, dict] = {}
    for r in rows:
        if isinstance(r.get("objects"), dict) and r.get("call_index") is not None:
            by_call[int(r["call_index"])] = r
    return {"targets": targets, "movable_targets": [t for t in targets if t in movable], "destinations": [str(d) for d in (head.get("destinations") or [])],
            "initial": {k: _vec3(v) for k, v in head["objects"].items()}, "by_call": by_call, "goal_state": head.get("goal_state"), "rows": len(rows)}


def trusted_points(calls: list[dict], words: set[str]) -> dict[str, Any]:
    """Perception points that may stand for the target: successful call, no quality flag (mark_point/point), label shares a target word."""
    points: list[tuple[str, list[float]]] = []
    others: list[tuple[str, list[float]]] = []   # trusted but not the target (destinations etc.)
    rejected = {"failed": 0, "flagged": 0, "label": 0}
    for row in calls:
        if row.get("consumes_action"):
            continue
        name = str(row.get("name") or "")
        if name not in PERCEPTION_CALLS:
            continue
        value = row.get("value") if isinstance(row.get("value"), dict) else {}
        pts = _detected_points("mark_point" if name == "point" else name, value)
        if not pts:
            continue
        if row.get("success") is not True:
            rejected["failed"] += len(pts)
            continue
        if name in ("mark_point", "point") and (value.get("quality_flags") or (value.get("grounding_status") not in (None, "grounded"))):
            rejected["flagged"] += len(pts)
            continue
        for label, p in pts:
            if words and not (_words(label) & words):
                rejected["label"] += 1
                others.append((label, p))
                continue
            points.append((label, p))
    return {"points": points, "others": others, "rejected": rejected, "words": sorted(words)}


def _dist(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def _dist_xy(a: list[float], b: list[float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def ladder_h(calls: list[dict], trusted: dict[str, Any], hidden: dict[str, Any] | None) -> dict[str, Any]:
    """v11h milestones, each at most once. mode "hidden": the target body's own motion decides lifted/transported/placed; mode "perception":
    EEF motion while holding after a VERIFIED close (success + jaws on material + within CLOSE_NEAR_M of a trusted target point)."""
    got = {"close_near": False, "closed_on_material": False, "lifted": False, "transported": False, "lowered_near_target": False, "released_after_place": False}
    events: list[tuple] = []
    movable = list((hidden or {}).get("movable_targets") or [])
    mode = "hidden" if movable else "perception"
    initial = (hidden or {}).get("initial") or {}
    by_call = (hidden or {}).get("by_call") or {}
    destinations = list((hidden or {}).get("destinations") or [])
    target_points = [p for _, p in trusted.get("points") or []]
    place_points = target_points + [p for _, p in trusted.get("others") or []]
    last_objects: dict[str, list[float]] = {k: v for k, v in initial.items() if v is not None}
    min_dist: float | None = None
    holding = False
    hold_open: float | None = None
    grasp_pos: list[float] | None = None
    closes = empty_closes = 0
    for row in calls:
        if not row.get("consumes_action") or not row.get("executed"):
            continue
        value = row.get("value") if isinstance(row.get("value"), dict) else {}
        before, after = _gripper(value, "robot_before"), _gripper(value, "robot_after")
        p0, p1 = _eef(value, "robot_before"), _eef(value, "robot_after")
        ci = row.get("call_index")
        hrow = by_call.get(int(ci)) if ci is not None else None
        if hrow:
            for k, v in (hrow.get("objects") or {}).items():
                if (q := _vec3(v)) is not None:
                    last_objects[k] = q
            if p1 is None:
                p1 = _vec3(hrow.get("eef"))
        refs = [last_objects[t] for t in movable if t in last_objects] if mode == "hidden" else target_points
        d = min((_dist(p1, q) for q in refs), default=None) if p1 is not None else None
        if d is not None:
            min_dist = d if min_dist is None else min(min_dist, d)
        if _is_gripper_close(row.get("arguments")):
            closes += 1
            ok_call = row.get("success") is True
            aperture = before is not None and after is not None and 0.03 < after < 0.90 and after < before - 0.10
            near = d is not None and d <= (CLOSE_NEAR_TRUE_M if mode == "hidden" else CLOSE_NEAR_M)
            if near and ok_call and not got["close_near"]:
                got["close_near"] = True
                events.append(("close_near", ci, round(d, 3)))
            if aperture and ok_call and near:
                holding, hold_open, grasp_pos = True, after, p1
                if not got["closed_on_material"]:
                    got["closed_on_material"] = True
                    events.append(("closed_verified", ci, round(after, 3)))
            else:
                holding, hold_open, grasp_pos = False, None, None
                if not aperture:
                    empty_closes += 1
                events.append(("close_unverified" if aperture else "empty_close", ci, {"success": ok_call, "near": near, "after": None if after is None else round(after, 3)}))
            continue
        if mode == "hidden":
            jaws_hold = after is not None and 0.03 < after < 0.90
            for t in movable:
                now, init = last_objects.get(t), initial.get(t)
                if now is None or init is None:
                    continue
                dz = now[2] - init[2]
                if not got["lifted"] and dz >= TRUE_LIFT_M and jaws_hold:
                    got["lifted"] = True
                    if not got["closed_on_material"]:
                        got["closed_on_material"] = True   # the body is in the air between the jaws: that is a grasp
                        events.append(("closed_by_lift", ci, round(dz, 3)))
                    events.append(("lifted_true", ci, round(dz, 3)))
                if got["lifted"] and not got["transported"] and _dist_xy(now, init) >= TRUE_TRANSPORT_M:
                    got["transported"] = True
                    events.append(("transported_true", ci, round(_dist_xy(now, init), 3)))
                if got["transported"] and not got["lowered_near_target"] and dz <= PLACE_Z_SLACK_M:
                    dests = [last_objects[x] for x in destinations if x in last_objects]
                    if any(_dist_xy(now, q) <= PLACE_RADIUS_M for q in dests):
                        got["lowered_near_target"] = True
                        events.append(("placed_true", ci, round(dz, 3)))
                if got["lowered_near_target"] and not got["released_after_place"] and after is not None and after > 0.90:
                    got["released_after_place"] = True
                    events.append(("released_after_place", ci, round(after, 3)))
            continue
        # perception mode: EEF motion while holding after a verified close
        if not holding:
            continue
        if after is None or not (0.03 < after < 0.90 and hold_open is not None and abs(after - hold_open) < 0.08):
            if after is not None and after > 0.90 and got["lowered_near_target"] and not got["released_after_place"]:
                got["released_after_place"] = True
                events.append(("released_after_place", ci, round(after, 3)))
            else:
                events.append(("lost_hold", ci, None if after is None else round(after, 3)))
            holding, hold_open, grasp_pos = False, None, None
            continue
        if p1 is None or grasp_pos is None:
            continue
        if not got["lifted"] and p1[2] > grasp_pos[2] + LIFT_M:
            got["lifted"] = True
            events.append(("lifted", ci, round(p1[2] - grasp_pos[2], 3)))
        if not got["transported"] and _dist_xy(p1, grasp_pos) > TRANSPORT_M:
            got["transported"] = True
            events.append(("transported", ci, round(_dist_xy(p1, grasp_pos), 3)))
        if got["transported"] and not got["lowered_near_target"] and p0 is not None and p1[2] < p0[2] - LOWER_M:
            if any(_dist_xy(p1, q) <= PLACE_RADIUS_M for q in place_points):
                got["lowered_near_target"] = True
                events.append(("lowered_near_target", ci, round(p0[2] - p1[2], 3)))
    return {**got, "mode": mode, "events": events[:12], "min_dist_to_target": None if min_dist is None else round(min_dist, 4), "closes": closes,
            "empty_closes": empty_closes, "targets": movable if mode == "hidden" else None, "trusted_points": len(target_points),
            "rejected_points": trusted.get("rejected"), "target_words": trusted.get("words")}


def preamble_requests(items: list[dict]) -> dict[str, int]:
    """Model requests up to and including the first whose tool-call input tries a physical action (regex on the script text);
    if none, every request counts. In code mode each request is one exec tool call."""
    req = 0
    last_kind = None
    first_physical = None
    for it in items:
        kind = it.get("type")
        if kind in ("function_call", "custom_tool_call"):
            if last_kind not in ("function_call", "custom_tool_call"):
                req += 1
            text = str(it.get("arguments") or it.get("input") or "")
            if first_physical is None and _PHYSICAL_SCRIPT_RE.search(text):
                first_physical = req
        last_kind = kind
    return {"requests": req, "first_physical_request": first_physical, "preamble": first_physical if first_physical is not None else req}


def tool_error_count(items: list[dict]) -> dict[str, int]:
    """Tool-side failures visible to the model: non-zero exits, unsupported/failed tool calls, unparsed tool-call text."""
    nonzero = unsupported = unparsed = 0
    for item in items:
        kind = item.get("type")
        if kind in ("function_call_output", "custom_tool_call_output"):
            out = item.get("output")
            if isinstance(out, list):  # code-mode part list: "Script completed|failed\nWall time ...", "Script error:\n..."
                text = "\n".join(str(p.get("text", "")) for p in out if isinstance(p, dict))
                if "Script failed" in text or "Script error:" in text:
                    nonzero += 1
                continue
            text = out if isinstance(out, str) else ""
            m = _EXIT_RE.search(text)
            if m and m.group(1) != "0":
                nonzero += 1
            elif text and _TOOL_ERROR_RE.match(text):
                unsupported += 1
        elif kind == "message" and item.get("role") == "assistant":
            text = "".join(str(c.get("text", "")) for c in (item.get("content") or []) if isinstance(c, dict))
            if any(mk in text for mk in _UNPARSED_MARKERS):
                unparsed += 1
    return {"nonzero_exit": nonzero, "unsupported": unsupported, "unparsed": unparsed, "total": nonzero + unsupported + unparsed}


def legacy_structured_tool_error_count(items: list[dict]) -> dict[str, Any]:
    """Offline compatibility only: parse complete printed ExecResult objects, never fragments.

    Printed results cannot prove command identity, environment health, or omitted
    output. v12b never uses this fallback for training.
    """
    counts = tool_error_count(items)
    parsed_failures = unreadable = 0
    for item in items:
        output = item.get("output")
        if item.get("type") not in ("function_call_output", "custom_tool_call_output") or not isinstance(output, list):
            continue
        failures = set()
        outer_failed = False
        for part in output:
            text = part.get("text") if isinstance(part, dict) else None
            if not isinstance(text, str):
                continue
            outer_failed |= "Script failed" in text or "Script error:" in text
            try:
                value = json.loads(text)
            except ValueError:
                unreadable += int(text.lstrip().startswith(("{", "[")))
                continue
            values = value if isinstance(value, list) else [value]
            for result in values:
                if (isinstance(result, dict) and isinstance(result.get("chunk_id"), str)
                        and isinstance(result.get("output"), str) and type(result.get("exit_code")) is int
                        and result["exit_code"] != 0 and _finite_number(result.get("wall_time_seconds"))
                        and type(result.get("original_token_count")) is int):
                    failures.add(result["chunk_id"])
        parsed_failures += len(failures)
        counts["nonzero_exit"] += max(0, len(failures) - int(outer_failed))
    counts["total"] = counts["nonzero_exit"] + counts["unsupported"] + counts["unparsed"]
    return {**counts, "authoritative": False, "source": "legacy_printed_json", "parsed_nested_failures": parsed_failures,
            "unreadable_json_parts": unreadable}


def _environment_error(value: Any) -> dict[str, Any]:
    codes = {"required_executable_unavailable", "robot_stub_import_failed", "sandbox_preflight_failed"}
    if (not isinstance(value, dict) or value.get("schema_version") != ENVIRONMENT_ERROR_SCHEMA
            or value.get("verified") is not True or not isinstance(value.get("code"), str) or value["code"] not in codes
            or value.get("source") not in ("tool_preflight", "tool_runtime")):
        raise ValueError("tool_audit_unverified_environment_error")
    if value["code"] == "required_executable_unavailable" and value.get("executable") not in ("rg", "python3", "bash", "node"):
        raise ValueError("tool_audit_missing_environment_executable")
    return value


def _simple_command_exit_is_normal(command: Any, exit_code: int) -> bool:
    """Exit 1 is normal for a simple search/test/difference query, not arbitrary shell code."""
    if exit_code != 1 or not isinstance(command, str) or "$" in command or "`" in command:
        return False
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()<>\n")
        lexer.whitespace = " \t\r"
        lexer.whitespace_split = True
        words = list(lexer)
    except ValueError:
        return False
    if not words or any(word and all(c in ";&|()<>\n" for c in word) for word in words):
        return False
    while words and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", words[0], re.S):
        words.pop(0)
    return bool(words and Path(words[0]).name in _NORMAL_EXIT_ONE)


def read_tool_audit(episode_dir: Path, items: list[dict], episode_result: dict) -> dict[str, Any]:
    """Require complete host-recorded outer/nested calls, independent of text() and truncation."""
    try:
        source = (Path(episode_dir) / "driver/tool_audit.jsonl").read_text(encoding="utf-8")
        rows = [json.loads(line) for line in source.splitlines() if line.strip()]
    except (OSError, ValueError) as exc:
        raise ValueError(f"tool_audit_unreadable:{type(exc).__name__}") from exc
    if len(rows) < 2 or any(not isinstance(row, dict) for row in rows):
        raise ValueError("tool_audit_missing_header_or_footer")
    header, footer = rows[0], rows[-1]
    if (header.get("type") != "header" or header.get("schema_version") != TOOL_AUDIT_SCHEMA
            or header.get("contract_version") not in TOOL_RUNTIME_CONTRACTS
            or not isinstance(header.get("contract_sha256"), str) or not _HASH_RE.fullmatch(header["contract_sha256"])):
        raise ValueError("tool_audit_invalid_runtime_contract")
    preflight = header.get("environment_preflight")
    from toolchain_preflight import validate_receipt
    try:
        validate_receipt(preflight, contract=header["contract_version"])
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"tool_audit_missing_verified_preflight:{exc}") from exc
    if episode_result.get("environment_preflight") is not None and episode_result["environment_preflight"] != preflight:
        raise ValueError("tool_audit_preflight_identity_mismatch")
    if footer.get("type") != "footer" or footer.get("complete") is not True or footer.get("unresolved_sessions") != []:
        raise ValueError("tool_audit_incomplete_footer")
    outers, outer_ends, tools, tool_ends, commands, sessions = {}, {}, {}, {}, {}, []
    environment_errors = []
    for holder in (episode_result, header, footer):
        if holder.get("environment_error") is not None:
            environment_errors.append(_environment_error(holder["environment_error"]))

    def identifier(value, label):
        if not isinstance(value, str) or not value:
            raise ValueError(f"tool_audit_invalid_{label}")
        return value

    for row in rows[1:-1]:
        if row.get("schema_version", TOOL_AUDIT_SCHEMA) != TOOL_AUDIT_SCHEMA:
            raise ValueError("tool_audit_changed_schema")
        kind = row.get("type")
        if row.get("environment_error") is not None:
            environment_errors.append(_environment_error(row["environment_error"]))
        if kind == "outer_start":
            key = identifier(row.get("outer_call_id"), "outer_id")
            if (key in outers or type(row.get("request_index")) is not int or row["request_index"] < 1
                    or row.get("call_kind") not in ("function_call", "custom_tool_call") or not isinstance(row.get("tool_name"), str)):
                raise ValueError("tool_audit_invalid_outer_start")
            outers[key] = row
        elif kind == "outer_end":
            key = identifier(row.get("outer_call_id"), "outer_id")
            if key not in outers or key in outer_ends or row.get("status") not in ("completed", "failed"):
                raise ValueError("tool_audit_invalid_outer_end")
            for field in ("request_index", "tool_name", "call_kind"):
                if field in row and row[field] != outers[key][field]:
                    raise ValueError("tool_audit_outer_identity_mismatch")
            if row.get("error") is not None and not isinstance(row["error"], str):
                raise ValueError("tool_audit_invalid_outer_error")
            if (row["status"] == "failed") != bool(row.get("error")):
                raise ValueError("tool_audit_inconsistent_outer_status")
            outer_ends[key] = row
        elif kind == "tool_start":
            key = identifier(row.get("tool_call_id"), "tool_id")
            parent = identifier(row.get("parent_call_id"), "parent_id")
            if (key in tools or parent not in outers or type(row.get("request_index")) is not int
                    or row["request_index"] != outers[parent]["request_index"]
                    or not isinstance(row.get("tool_name"), str) or "args" not in row):
                raise ValueError("tool_audit_invalid_tool_start")
            tools[key] = row
        elif kind == "tool_end":
            key = identifier(row.get("tool_call_id"), "tool_id")
            if key not in tools or key in tool_ends:
                raise ValueError("tool_audit_invalid_tool_end")
            original = tools[key]
            for field in ("parent_call_id", "request_index", "tool_name", "args"):
                if field in row and row[field] != original[field]:
                    raise ValueError("tool_audit_tool_identity_mismatch")
            error = row.get("error")
            if error is not None and not isinstance(error, str):
                raise ValueError("tool_audit_invalid_tool_error")
            if row.get("error_class") not in (None, "policy", "environment"):
                raise ValueError("tool_audit_invalid_error_class")
            if row.get("error_class") == "environment" and row.get("environment_error") is None:
                raise ValueError("tool_audit_unverified_environment_error")
            if original["tool_name"] not in _SHELL_TOOLS and row.get("command_id") is not None:
                raise ValueError("tool_audit_nonprocess_command_id")
            if original["tool_name"] == "python":
                result = row.get("result")
                if (header["contract_version"] != PYTHON_RUNTIME_CONTRACT or not isinstance(result, dict)
                        or not isinstance(result.get("output"), str) or result.get("status") not in PYTHON_CELL_STATUSES):
                    raise ValueError("tool_audit_invalid_python_result")
                if (result["status"] != "ok") != bool(error):
                    raise ValueError("tool_audit_inconsistent_python_status")
            if original["tool_name"] in _SHELL_TOOLS and (row.get("command_id") is not None or not error):
                result = row.get("result")
                if not isinstance(result, dict) or not isinstance(result.get("output"), str):
                    raise ValueError("tool_audit_missing_raw_process_result")
                code, session = result.get("exit_code"), result.get("session_id")
                if ((code is not None and type(code) is not int)
                        or (session is not None and (type(session) is not int or session < 0))
                        or (code is None and session is None)):
                    raise ValueError("tool_audit_invalid_process_status")
                command_id = identifier(row.get("command_id"), "command_id")
                if original["tool_name"] == "exec_command":
                    if command_id != key or not isinstance(original["args"], dict) or not isinstance(original["args"].get("cmd"), str):
                        raise ValueError("tool_audit_invalid_command_origin")
                    commands[command_id] = {"call": original, "session_id": session, "pending": code is None}
                elif command_id not in commands:
                    raise ValueError("tool_audit_unknown_command_session")
                else:
                    try:
                        requested_session = int(original["args"]["session_id"])
                    except (KeyError, TypeError, ValueError, OverflowError) as exc:
                        raise ValueError("tool_audit_command_session_mismatch") from exc
                    if requested_session != commands[command_id]["session_id"] or session != requested_session:
                        raise ValueError("tool_audit_command_session_mismatch")
            tool_ends[key] = row
        elif kind == "tool_processes_closed":
            if not isinstance(row.get("stats"), list):
                raise ValueError("tool_audit_invalid_process_close")
        elif kind == "session_end":
            command_id = identifier(row.get("command_id"), "command_id")
            if (command_id not in commands or row.get("source") != "episode_cleanup"
                    or row.get("status") not in ("completed", "cancelled") or not isinstance(row.get("output"), str)
                    or (row.get("status") == "completed" and type(row.get("exit_code")) is not int)
                    or (row.get("exit_code") is not None and type(row["exit_code"]) is not int)):
                raise ValueError("tool_audit_invalid_session_end")
            if type(row.get("session_id")) is not int or row["session_id"] != commands[command_id]["session_id"]:
                raise ValueError("tool_audit_command_session_mismatch")
            if row.get("error_class") == "environment" and row.get("environment_error") is None:
                raise ValueError("tool_audit_unverified_environment_error")
            if row.get("error_class") not in (None, "policy", "environment"):
                raise ValueError("tool_audit_invalid_error_class")
            sessions.append(row)
        else:
            raise ValueError(f"tool_audit_unknown_event:{kind}")
    if set(outers) != set(outer_ends) or set(tools) != set(tool_ends):
        raise ValueError("tool_audit_unfinished_calls")
    for key, count in (("outer_started", len(outers)), ("outer_finished", len(outer_ends)), ("tool_started", len(tools)), ("tool_finished", len(tool_ends))):
        if type(footer.get(key)) is not int or footer[key] != count:
            raise ValueError("tool_audit_footer_count_mismatch")
    item_calls = [item for item in items if item.get("type") in ("function_call", "custom_tool_call")]
    item_ids = [item.get("call_id") for item in item_calls]
    if any(not isinstance(key, str) for key in item_ids) or len(set(item_ids)) != len(item_ids) or set(item_ids) != set(outers):
        raise ValueError("tool_audit_driver_call_coverage")
    for item in item_calls:
        if item["type"] != outers[item["call_id"]]["call_kind"] or item.get("name") != outers[item["call_id"]]["tool_name"]:
            raise ValueError("tool_audit_driver_call_identity")
    finished_commands = {row["command_id"] for row in tool_ends.values()
                         if row.get("command_id") and (row.get("result") or {}).get("exit_code") is not None}
    finished_commands.update(row["command_id"] for row in sessions)
    if any(command["pending"] and key not in finished_commands for key, command in commands.items()):
        raise ValueError("tool_audit_unresolved_process")
    return {"header": header, "outers": outers, "outer_ends": outer_ends, "tools": tools, "tool_ends": tool_ends,
            "commands": commands, "sessions": sessions, "environment_errors": environment_errors,
            "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest()}


def audited_tool_errors(audit: dict[str, Any], items: list[dict]) -> dict[str, Any]:
    """Count executions, not printed copies: terminal command IDs are unique failure units."""
    terminal, failure_parents, unsupported = {}, set(), 0
    benign = cancelled = python_errors = 0
    per_request: dict[int, dict[str, int]] = {}

    def bump(request_index: int, key: str) -> None:
        row = per_request.setdefault(int(request_index), {"calls": 0, "python_errors": 0, "unsupported": 0, "outer_errors": 0})
        row[key] += 1

    for start in audit["outers"].values():
        bump(start["request_index"], "calls")
    for key, end in audit["tool_ends"].items():
        start = audit["tools"][key]
        if start["tool_name"] == "python":
            if end.get("error") and end.get("error_class") != "environment":
                python_errors += 1
                failure_parents.add(start["parent_call_id"])
                bump(start["request_index"], "python_errors")
            elif end.get("error"):
                failure_parents.add(start["parent_call_id"])
        elif start["tool_name"] in _SHELL_TOOLS and end.get("command_id") is not None and end["result"].get("exit_code") is not None:
            command_id = end["command_id"]
            code = end["result"]["exit_code"]
            if command_id in terminal and terminal[command_id] != code:
                raise ValueError("tool_audit_conflicting_process_exit")
            terminal[command_id] = code
        elif end.get("error"):
            unsupported += 1
            failure_parents.add(start["parent_call_id"])
            bump(start["request_index"], "unsupported")
    seen_sessions = set()
    for row in audit["sessions"]:
        command_id = row["command_id"]
        if command_id in seen_sessions:
            raise ValueError("tool_audit_duplicate_session_end")
        seen_sessions.add(command_id)
        if row["status"] == "cancelled":
            if command_id in terminal:
                raise ValueError("tool_audit_cancelled_terminal_command")
            cancelled += 1
            continue
        code = row["exit_code"]
        if command_id in terminal and terminal[command_id] != code:
            raise ValueError("tool_audit_conflicting_process_exit")
        terminal[command_id] = code
    nonzero = 0
    for command_id, code in terminal.items():
        if code == 0:
            continue
        original = audit["commands"][command_id]["call"]
        if _simple_command_exit_is_normal(original["args"].get("cmd"), code):
            benign += 1
            continue
        nonzero += 1
        failure_parents.add(original["parent_call_id"])
        for key, end in audit["tool_ends"].items():
            if end.get("command_id") == command_id and (end.get("result") or {}).get("exit_code") == code:
                failure_parents.add(audit["tools"][key]["parent_call_id"])
    outer_errors = 0
    for key, end in audit["outer_ends"].items():
        if end["status"] == "failed" and key not in failure_parents:
            outer_errors += 1
            bump(audit["outers"][key]["request_index"], "outer_errors")
    unparsed = tool_error_count([item for item in items if item.get("type") == "message"])["unparsed"]
    return {"nonzero_exit": nonzero + outer_errors, "unsupported": unsupported, "unparsed": unparsed,
            "total": nonzero + outer_errors + unsupported + unparsed + python_errors, "authoritative": True,
            "source": TOOL_AUDIT_SCHEMA, "failed_commands": nonzero, "outer_errors": outer_errors,
            "benign_nonzero_commands": benign, "cancelled_commands": cancelled, "python_errors": python_errors,
            "per_request": per_request}


def tool_output_tokens(proxy_requests: list[dict] | None, image_pad_id: int = IMAGE_PAD_ID) -> dict[str, float]:
    """Text tokens the tool outputs (and template glue) added to the context per model request, from the proxy record.

    Works on the full record (`prompt_token_ids` / `token_ids`, image pads excluded exactly) and on the stripped copy kept in
    rollout.json (`prompt_len` / `completion_len`, image pads then counted as text)."""
    rows = []
    for q in proxy_requests or []:
        ids, comp = q.get("prompt_token_ids"), q.get("token_ids")
        if ids is not None and comp is not None:
            rows.append((len(ids) - ids.count(image_pad_id), len(comp)))
        elif q.get("prompt_len"):
            rows.append((int(q["prompt_len"]), int(q.get("completion_len") or 0)))
    if len(rows) < 2:
        return {"text_tokens": 0, "transitions": max(0, len(rows) - 1), "per_request": 0.0, "big_outputs": 0, "max_output": 0}
    per_step = [max(0, rows[i + 1][0] - rows[i][0] - rows[i][1]) for i in range(len(rows) - 1)]  # tool output(s) between request i and i+1
    if V11E:
        # compaction shrinks the prompt between two requests: sum the per-transition growth instead of last - first
        added = sum(per_step)
    else:
        added = max(0, rows[-1][0] - rows[0][0] - sum(c for _, c in rows[:-1]))
    return {"text_tokens": int(added), "transitions": len(rows) - 1, "per_request": added / (len(rows) - 1),
            "big_outputs": sum(1 for x in per_step if x > BIG_OUTPUT_TOKENS), "max_output": max(per_step, default=0)}


def load_items(episode_dir: Path) -> list[dict]:
    """Every history item the driver registered, in order: driver/items.jsonl (survives compaction) else driver/transcript.jsonl."""
    rows = _read_jsonl(Path(episode_dir) / "driver" / "items.jsonl")
    if rows:
        return [r.get("item") for r in rows if isinstance(r.get("item"), dict)]
    return _read_jsonl(Path(episode_dir) / "driver" / "transcript.jsonl")


def request_scripts(items: list[dict]) -> list[str]:
    """Tool-call input text per model request that made tool calls (code mode: one exec script per request), in order."""
    scripts: list[str] = []
    last_kind = None
    for it in items:
        kind = it.get("type")
        if kind in ("function_call", "custom_tool_call"):
            text = str(it.get("arguments") or it.get("input") or "")
            if last_kind in ("function_call", "custom_tool_call"):
                scripts[-1] += "\n" + text
            else:
                scripts.append(text)
        last_kind = kind
    return scripts


def calls_by_request(calls: list[dict], request_rows: list[dict], started_wall: float | None) -> list[list[dict]] | None:
    """calls_by_request(...)[k] = capability calls executed while serving model request k (1-based); None if the timeline is unusable.
    Request k's tools run after response k was received (requests.jsonl t of row k) and before response k+1."""
    times = [r.get("t") for r in request_rows]
    if started_wall is None or not times or any(not isinstance(t, (int, float)) for t in times):
        return None
    per: list[list[dict]] = [[] for _ in range(len(times) + 1)]
    for c in calls:
        ts = c.get("t_s")
        if not isinstance(ts, (int, float)):
            return None
        w = float(started_wall) + float(ts)
        k = len(times)
        for i, t in enumerate(times):
            if w < t:
                k = i
                break
        per[k].append(c)
    return per


def _position_of(args: Any) -> list[float] | None:
    if not isinstance(args, dict):
        return None
    for key in ("position", "position_m", "target_position", "target_position_m"):
        if (p := _vec3(args.get(key))) is not None:
            return p
    inner = args.get("arguments")
    return _position_of(inner) if isinstance(inner, dict) else None


def duplicate_requests(calls: list[dict]) -> dict[str, Any]:
    """Physical requests (executed or rejected) that repeat an earlier one: move_to with the same gripper command and a position within
    DUP_RADIUS_M of a previous move_to; other physical calls with identical arguments. Repeats beyond DUP_FREE_REPEATS per cluster count."""
    clusters: list[dict[str, Any]] = []
    for row in calls:
        if not row.get("consumes_action"):
            continue
        name = str(row.get("name") or "")
        args = row.get("arguments")
        pos = _position_of(args) if name == "move_to" else None
        grip = "close" if _is_gripper_close(args) else "open"
        key = None if pos is not None else (name, json.dumps(args, sort_keys=True, default=str))
        hit = None
        for cl in clusters:
            if pos is not None and cl["name"] == name and cl["grip"] == grip and cl["pos"] is not None:
                if sum((a - b) ** 2 for a, b in zip(pos, cl["pos"])) ** 0.5 <= DUP_RADIUS_M:
                    hit = cl
                    break
            elif pos is None and cl["key"] == key:
                hit = cl
                break
        if hit is None:
            clusters.append({"name": name, "grip": grip, "pos": pos, "key": key, "count": 1, "call_index": row.get("call_index")})
        else:
            hit["count"] += 1
    extra = sum(max(0, cl["count"] - DUP_FREE_REPEATS) for cl in clusters)
    worst = max(clusters, key=lambda cl: cl["count"], default=None)
    return {"extra_repeats": extra, "clusters": len(clusters),
            "worst": None if worst is None else {"name": worst["name"], "grip": worst["grip"], "pos": [round(x, 3) for x in worst["pos"]] if worst["pos"] else None,
                                                  "count": worst["count"], "first_call_index": worst["call_index"]}}


def _clean_interaction(row: dict) -> bool:
    """Executed, successful move_to/hold/release whose stop reason is not a force/contact/limit event."""
    if not (row.get("consumes_action") and row.get("executed") and row.get("success") and str(row.get("name")) in INTERACTION_CALLS):
        return False
    value = row.get("value") if isinstance(row.get("value"), dict) else {}
    return not _FORCE_STOP_RE.search(str(value.get("stopped_reason") or ""))


def interaction_terms(calls: list[dict], items: list[dict], request_rows: list[dict], started_wall: float | None) -> dict[str, Any]:
    """Per-request classification (act / observe / no-op) -> interaction bonus and bounded no-op charge (v11e)."""
    scripts = request_scripts(items)
    per = calls_by_request(calls, request_rows, started_wall)
    n_req = len(request_rows) if per is not None else len(scripts)
    kinds: list[str] = []
    for k in range(1, n_req + 1):
        text = scripts[k - 1] if k - 1 < len(scripts) else ""
        if per is not None:
            cs = per[k] if k < len(per) else []
            requested = any(c.get("consumes_action") and str(c.get("name")) != "recover" for c in cs)
            clean = any(_clean_interaction(c) for c in cs)
            observed = any(not c.get("consumes_action") and str(c.get("name")) in PERCEPTION_CALLS for c in cs) or bool(_OBSERVE_SCRIPT_RE.search(text))
        else:  # no timeline: script text only
            requested = bool(_PHYSICAL_SCRIPT_RE.search(text))
            clean = False
            observed = bool(_OBSERVE_SCRIPT_RE.search(text)) or any(name in text for name in PERCEPTION_CALLS)
        finish = bool(_FINISH_SCRIPT_RE.search(text))
        kinds.append("C" if clean else "A" if requested else "F" if finish else "O" if observed else "-")
    first_act = next((i for i, k in enumerate(kinds) if k in "CA"), None)
    clean_requests = sum(1 for k in kinds if k == "C")
    noop = 0
    if first_act is not None:
        for i, k in enumerate(kinds):
            if i > first_act and i < len(kinds) - 1 and k == "-":   # final request (end of turn) exempt
                noop += 1
    bonus = min(INTERACT_CAP, W_INTERACT_FIRST * min(clean_requests, INTERACT_FIRST_N) + W_INTERACT_AFTER * max(0, clean_requests - INTERACT_FIRST_N))
    noop_pen = max(-NOOP_CAP, W_NOOP * max(0, noop - NOOP_FREE))
    return {"kinds": "".join(kinds), "requests": n_req, "first_act_request": None if first_act is None else first_act + 1, "clean_requests": clean_requests,
            "noop_requests": noop, "timeline": per is not None, "interaction": bonus, "noop": noop_pen}


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _finite_point(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 3 and all(_finite_number(x) for x in value)


def read_goal_audit(episode_dir: Path, calls: list[dict]) -> dict[str, Any]:
    """Validate the complete private v2 audit; never substitute perception points."""
    path = Path(episode_dir) / "hidden_object_audit.jsonl"
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError) as exc:
        raise ValueError(f"goal_audit_unreadable: {type(exc).__name__}") from exc
    if len(rows) < 2 or any(not isinstance(row, dict) for row in rows):
        raise ValueError("goal_audit_missing_reset_or_finish")
    head, final = rows[0], rows[-1]
    if head.get("schema_version") != GOAL_AUDIT_SCHEMA:
        raise ValueError("goal_audit_requires_v2")
    if head.get("capability") != "reset" or head.get("call_index") != 0 or final.get("capability") != "finish":
        raise ValueError("goal_audit_missing_reset_or_finish")
    if head.get("coordinate_frame") != "robot_base":
        raise ValueError("goal_audit_coordinate_frame")
    goals = head.get("goals")
    if not isinstance(goals, list) or not goals:
        raise ValueError("goal_audit_missing_goals")
    descriptors = {}
    for goal in goals:
        if not isinstance(goal, dict) or not isinstance(goal.get("goal_id"), str) or not goal["goal_id"]:
            raise ValueError("goal_audit_invalid_goal_id")
        if goal["goal_id"] in descriptors or not isinstance(goal.get("predicate"), str) or not goal["predicate"]:
            raise ValueError("goal_audit_duplicate_or_invalid_goal")
        if goal.get("target") is not None and not isinstance(goal["target"], dict):
            raise ValueError("goal_audit_invalid_target_descriptor")
        descriptors[goal["goal_id"]] = goal
    expected = []
    for call_row in calls:
        if call_row.get("consumes_action") and call_row.get("executed"):
            ci = call_row.get("call_index")
            if type(ci) is not int or ci <= 0:
                raise ValueError("goal_audit_invalid_physical_call_index")
            expected.append(ci)
    if len(set(expected)) != len(expected):
        raise ValueError("goal_audit_duplicate_physical_call_index")
    actual = [row.get("call_index") for row in rows[1:-1]]
    if (any(type(ci) is not int for ci in actual) or actual != expected
            or any(row.get("capability") in ("reset", "finish") for row in rows[1:-1])):
        raise ValueError("goal_audit_physical_coverage")
    if type(final.get("call_index")) is not int or final["call_index"] < max(expected, default=0):
        raise ValueError("goal_audit_stale_finish")
    contracts = {}
    for row in rows:
        if row.get("error") or row.get("schema_version") != GOAL_AUDIT_SCHEMA:
            raise ValueError("goal_audit_snapshot_error_or_schema")
        states = row.get("goal_states")
        if not isinstance(states, dict) or set(states) != set(descriptors):
            raise ValueError("goal_audit_incomplete_goal_states")
        for goal_id, state in states.items():
            if not isinstance(state, dict) or state.get("error") or type(state.get("satisfied")) is not bool:
                raise ValueError(f"goal_audit_invalid_predicate:{goal_id}")
            kind, distance, scale = state.get("progress_kind"), state.get("distance_to_goal"), state.get("distance_scale")
            if kind == "predicate_only":
                if distance is not None or not state.get("unsupported_reason"):
                    raise ValueError(f"goal_audit_invalid_predicate_only:{goal_id}")
            elif kind in ("region_distance", "joint_range"):
                expected_unit = "m" if kind == "region_distance" else "normalized"
                if (not _finite_number(distance) or distance < 0 or not _finite_number(scale) or scale <= 0
                        or state.get("distance_unit") != expected_unit):
                    raise ValueError(f"goal_audit_invalid_distance:{goal_id}")
            else:
                raise ValueError(f"goal_audit_unsupported_progress:{goal_id}")
            approach = state.get("target_approach_supported")
            if type(approach) is not bool:
                raise ValueError(f"goal_audit_invalid_approach_support:{goal_id}")
            if approach:
                target = descriptors[goal_id].get("target") or {}
                if (target.get("kind") != "object" or target.get("movable") is not True
                        or not _finite_point(state.get("target_position_m")) or not _finite_point(row.get("eef"))):
                    raise ValueError(f"goal_audit_invalid_approach_geometry:{goal_id}")
            contract = (kind, state.get("distance_unit"), scale, approach)
            if goal_id in contracts and contracts[goal_id] != contract:
                raise ValueError(f"goal_audit_changed_contract:{goal_id}")
            contracts[goal_id] = contract
    return {"goals": goals, "initial": head, "final": final, "rows": len(rows), "physical_rows": len(actual)}


def goal_potentials(audit: dict[str, Any]) -> dict[str, Any]:
    """Undiscounted final-minus-initial potentials, paired by exact predicate id."""
    initial, final = audit["initial"], audit["final"]
    progress = []
    distance_total = satisfied_total = approach_total = 0.0
    for goal in audit["goals"]:
        goal_id = goal["goal_id"]
        before, after = initial["goal_states"][goal_id], final["goal_states"][goal_id]
        dense = before["progress_kind"] != "predicate_only"

        def distance_potential(state):
            if not dense:
                return 0.0
            if state["satisfied"]:
                return 1.0
            return 1.0 / (1.0 + state["distance_to_goal"] / state["distance_scale"])

        def approach_potential(row, state):
            if not state["target_approach_supported"]:
                return 0.0
            return max(0.0, 1.0 - _dist(row["eef"], state["target_position_m"]) / APPROACH_SCALE_A_M)

        p0, p1 = distance_potential(before), distance_potential(after)
        a0, a1 = approach_potential(initial, before), approach_potential(final, after)
        distance_delta = p1 - p0
        satisfied_delta = int(after["satisfied"]) - int(before["satisfied"])
        approach_delta = a1 - a0
        distance_total += distance_delta
        satisfied_total += satisfied_delta
        approach_total += approach_delta
        progress.append({"goal_id": goal_id, "predicate": goal["predicate"], "arguments": goal.get("arguments"),
                         "progress_kind": before["progress_kind"], "dense_supported": dense,
                         "unsupported_reason": before.get("unsupported_reason"),
                         "initial_satisfied": before["satisfied"], "final_satisfied": after["satisfied"],
                         "initial_distance": before["distance_to_goal"], "final_distance": after["distance_to_goal"],
                         "distance_unit": before.get("distance_unit"), "distance_scale": before.get("distance_scale"),
                         "distance_potential_delta": distance_delta, "satisfied_delta": satisfied_delta,
                         "approach_potential_delta": approach_delta,
                         "initial_joints": before.get("joints"), "final_joints": after.get("joints")})
    count = len(progress)
    return {"goal_progress": W_GOAL_DISTANCE_A * distance_total / count,
            "goal_satisfied": W_GOAL_SATISFIED_A * satisfied_total / count,
            "approach": W_APPROACH_A * approach_total / count, "goals": progress,
            "initial_satisfied": sum(p["initial_satisfied"] for p in progress),
            "final_satisfied": sum(p["final_satisfied"] for p in progress), "goal_count": count}


def score_v12a(episode_dir: Path, server: dict, calls: list[dict], items: list[dict], diag: dict,
               *, tool_errors: dict[str, Any] | None = None) -> dict:
    """Goal progress only; process rewards never pay for mere activity."""
    if type(server.get("host_final_success")) is not bool or server.get("finalize_error"):
        return _result(0.0, "infrastructure_error", infrastructure=True, reward_valid=False,
                       error="host_final_predicate_unavailable", **diag)
    try:
        audit = read_goal_audit(episode_dir, calls)
        progress = goal_potentials(audit)
        success = server["host_final_success"]
        if (progress["final_satisfied"] == progress["goal_count"]) != success:
            raise ValueError("goal_audit_final_predicate_mismatch")
    except ValueError as exc:
        return _result(0.0, "infrastructure_error", infrastructure=True, reward_valid=False, error=str(exc), **diag)
    phys = physical_summary(calls)
    errs = tool_error_count(items) if tool_errors is None else tool_errors
    request_rows = _read_jsonl(episode_dir / "driver" / "requests.jsonl")
    started_wall = _read_json(episode_dir / "server.json").get("started_wall_time") or server.get("started_wall_time")
    inter = interaction_terms(calls, items, request_rows, started_wall)
    dups = duplicate_requests(calls)
    termination = diag["termination_reason"]
    agent_finished = server.get("finish_source") == "agent"
    false_finish = agent_finished and not success
    gate_rejections = max(0, int(server.get("gate_rejections") or 0))
    comps = dict.fromkeys(COMPONENT_KEYS, 0.0)
    comps.update(hidden=W_HIDDEN if success else 0.0, goal_progress=progress["goal_progress"],
                 goal_satisfied=progress["goal_satisfied"], approach=progress["approach"],
                 action_cost=-min(ACTION_COST_CAP_A, 0.001 * phys["requested"]),
                 noop=-min(NOOP_COST_CAP_A, 0.005 * max(0, inter["noop_requests"] - NOOP_FREE)),
                 duplicate=-min(DUPLICATE_COST_CAP_A, 0.005 * dups["extra_repeats"]),
                 tool_error=-min(TOOL_ERROR_COST_CAP_A, 0.01 * errs["total"]),
                 gate=-min(GATE_COST_CAP_A, 0.01 * gate_rejections),
                 no_action=W_NO_ACTION_A if not success and phys["executed"] == 0 else 0.0)
    if success:
        verdict = "hidden_success"
    elif false_finish:
        comps["false_finish"] = W_FALSE_FINISH_A
        verdict = "false_finish"
    elif termination == "agent_stopped_without_finish":
        comps["no_finish"] = W_NO_FINISH_A
        verdict = "no_action" if phys["executed"] == 0 else "no_finish"
    else:
        comps["exhausted"] = W_EXHAUSTED_A
        verdict = "no_action" if phys["executed"] == 0 else (termination if termination in VERDICTS else "acted")
    diag.update({"reward_valid": True, "host_final_success": success, "host_success_ever": server.get("host_success_ever"),
                 "agent_finished": agent_finished, "gate_rejections": gate_rejections, "physical": phys, "tool_errors": errs,
                 "action_requests": server.get("action_requests"), "tool_requests": server.get("tool_requests"),
                 "interaction": inter, "duplicates": dups,
                 "goal_progress": progress["goals"], "initial_goals_satisfied": progress["initial_satisfied"],
                 "final_goals_satisfied": progress["final_satisfied"], "goal_count": progress["goal_count"],
                 "goal_audit": {"schema_version": GOAL_AUDIT_SCHEMA, "rows": audit["rows"],
                                "physical_rows": audit["physical_rows"], "gamma": 1.0},
                 "ladder": {"mode": "goal_predicates", "events": []}})
    return _result(sum(comps.values()), verdict, comps, **diag)


def score_v12b(episode_dir: Path, episode_result: dict, server: dict, calls: list[dict], items: list[dict], diag: dict) -> dict:
    """v12a goal reward with authoritative host-side process accounting."""
    try:
        audit = read_tool_audit(episode_dir, items, episode_result)
        diag["tool_audit"] = {"schema_version": TOOL_AUDIT_SCHEMA, "sha256": audit["sha256"],
                              "contract_version": audit["header"]["contract_version"],
                              "contract_sha256": audit["header"]["contract_sha256"],
                              "preflight_fingerprint": audit["header"]["environment_preflight"]["fingerprint"],
                              "outer_calls": len(audit["outers"]), "nested_calls": len(audit["tools"])}
        if audit["environment_errors"]:
            return _result(0.0, "infrastructure_error", infrastructure=True, reward_valid=False,
                           error="verified_environment_error", environment_errors=audit["environment_errors"], **diag)
        errors = audited_tool_errors(audit, items)
    except ValueError as exc:
        return _result(0.0, "infrastructure_error", infrastructure=True, reward_valid=False, error=str(exc), **diag)
    return score_v12a(episode_dir, server, calls, items, diag, tool_errors=errors)


def score_episode(episode_dir: Path, proxy_requests: list[dict] | None = None) -> dict:
    episode_dir = Path(episode_dir)
    er = _read_json(episode_dir / "episode_result.json")
    server = _read_json(episode_dir / "result.json")
    calls = _read_jsonl(episode_dir / "calls.jsonl")
    items = load_items(episode_dir) if V11E else _read_jsonl(episode_dir / "driver" / "transcript.jsonl")
    driver = er.get("driver") or {}
    termination = er.get("termination_reason")
    request_cap = bool(_REQUEST_CAP_RE.search(str(er.get("error") or "") + " " + str(driver.get("loop_reason") or "")))
    if V11C and termination == "infrastructure_error" and request_cap and server:
        termination = "request_cap_exhausted"  # policy outcome (too many model requests), not infrastructure
    diag: dict[str, Any] = {"reward_version": REWARD_VERSION, "termination_reason": termination, "request_cap": request_cap,
                            "loop_termination": driver.get("loop_termination"), "requests": driver.get("requests"),
                            "tool_calls": driver.get("tool_calls"), "output_tokens": (driver.get("usage") or {}).get("output_tokens"),
                            "max_context_tokens": driver.get("max_context_tokens"), "wall_time_s": er.get("wall_time_s"),
                            "episode_dir": str(episode_dir)}
    if V12B:
        try:
            environment_errors = [_environment_error(holder["environment_error"])
                                  for holder in (er, er.get("environment_preflight") or {})
                                  if isinstance(holder, dict) and holder.get("environment_error") is not None]
        except ValueError as exc:
            return _result(0.0, "infrastructure_error", infrastructure=True, reward_valid=False, error=str(exc), **diag)
        if environment_errors:
            return _result(0.0, "infrastructure_error", infrastructure=True, reward_valid=False,
                           error="verified_environment_error", environment_errors=environment_errors, **diag)
    if not er or termination == "infrastructure_error" or not server:
        diag["error"] = er.get("error") or driver.get("loop_reason")
        if V12B:
            diag["reward_valid"] = False
        return _result(0.0, "infrastructure_error", infrastructure=True, **diag)
    if V12B:
        return score_v12b(episode_dir, er, server, calls, items, diag)
    if V12A:
        return score_v12a(episode_dir, server, calls, items, diag)
    success = bool(server.get("host_final_success"))
    success_ever = bool(server.get("host_success_ever"))
    agent_finished = server.get("finish_source") == "agent"
    false_finish = bool(agent_finished and not success)
    gate_rejections = int(server.get("gate_rejections") or 0)
    phys = physical_summary(calls)
    errs = tool_error_count(items)
    out_tokens = int((driver.get("usage") or {}).get("output_tokens") or 0)
    verb = tool_output_tokens(proxy_requests)
    verbosity = 0.0
    if V11C:
        verbosity = max(-VERBOSITY_CAP, W_VERBOSITY_PER_K_C * max(0.0, (verb["per_request"] - VERBOSITY_FREE_PER_REQ) / 1000.0))
        verbosity += max(-BIG_OUTPUT_CAP, W_BIG_OUTPUT * verb["big_outputs"])
    elif REWARD_VERSION != "v11a":
        verbosity = max(-VERBOSITY_CAP, W_VERBOSITY_PER_K * max(0.0, (verb["per_request"] - VERBOSITY_FREE_PER_REQ) / 1000.0))
    grasp = grasp_attempt(calls) if (V11C and not V11D) else {"attempt": False, "how": None}
    w_first_move = W_FIRST_MOVE_C if V11C else W_FIRST_MOVE
    ladder = manipulation_ladder(calls) if V11D else {}
    hidden = None
    if V11H:
        hidden = hidden_objects(episode_dir)
        trusted = trusted_points(calls, target_words(server.get("task"), hidden))
        ladder = ladder_h(calls, trusted, hidden)
    pre = preamble_requests(items) if V11D else {}
    preamble_pen = 0.0
    if V11D:
        preamble_pen = max(-PREAMBLE_CAP, W_PREAMBLE_PER_REQUEST * max(0, pre["preamble"] - PREAMBLE_FREE_REQUESTS))
    inter: dict[str, Any] = {}
    dups: dict[str, Any] = {}
    if V11E:
        request_rows = _read_jsonl(episode_dir / "driver" / "requests.jsonl")
        started_wall = _read_json(episode_dir / "server.json").get("started_wall_time") or server.get("started_wall_time")
        inter = interaction_terms(calls, items, request_rows, started_wall)
        dups = duplicate_requests(calls)
    shaping_fg = V11F and not V11H   # v11f/v11g perception-proximity shaping (superseded by the verified terms of v11h)
    if V11H:
        w_physical = W_PHYSICAL_H
        interaction = min(INTERACT_CAP_H, inter.get("interaction", 0.0))
        w_first_move = W_FIRST_MOVE_H
        w_all_ok = 0.0
        approach = (W_APPROACH_H * max(0.0, 1.0 - ladder["min_dist_to_target"] / APPROACH_FULL_H_M)) if ladder.get("min_dist_to_target") is not None else 0.0
        close_near = W_CLOSE_NEAR_H if ladder.get("close_near") else 0.0
        w_closed, w_lifted, w_transported, w_lowered, w_released = W_CLOSED_H, W_LIFTED_H, W_TRANSPORTED_H, W_PLACED_H, W_RELEASED_H
        perceive = 0.0
    else:
        w_physical = W_PHYSICAL_F if (V11F and not V11G) else W_PHYSICAL
        interaction = (min(INTERACT_CAP_F, inter.get("interaction", 0.0)) if (V11F and not V11G) else inter.get("interaction", 0.0)) if V11E else 0.0
        w_all_ok = W_ALL_PHYSICAL_OK
        approach = (W_APPROACH * max(0.0, 1.0 - ladder["min_dist_to_point"] / APPROACH_FULL_M)) if (shaping_fg and ladder.get("min_dist_to_point") is not None) else 0.0
        close_near = (W_CLOSE_NEAR_G if V11G else W_CLOSE_NEAR) if (shaping_fg and ladder.get("close_near_object")) else 0.0
        w_closed = W_CLOSED_ON_MATERIAL_G if V11G else W_CLOSED_ON_MATERIAL
        w_lifted = W_LIFTED_G if V11G else W_LIFTED
        w_transported, w_lowered, w_released = W_TRANSPORTED, W_LOWERED_NEAR_TARGET, W_RELEASED_AFTER_PLACE
        perceive = (W_PERCEIVE_G if V11G else W_PERCEIVE) if (shaping_fg and ladder.get("perceived_points", 0) > 0) else 0.0
    comps = {
        "hidden": W_HIDDEN if success else 0.0,
        "success_ever": W_SUCCESS_EVER if (success_ever and not success) else 0.0,
        "false_finish": W_FALSE_FINISH if false_finish else 0.0,
        "gate": W_GATE_REJECTION * min(GATE_REJECTION_CAP, gate_rejections),
        "physical": w_physical * min(1.0, phys["clean"] / PHYSICAL_FULL),
        "approach": approach,
        "close_near": close_near,
        "perceive": perceive,
        "first_move": w_first_move if phys["first_move"] else 0.0,
        "grasp": W_GRASP if grasp["attempt"] else 0.0,
        "closed_on_material": w_closed if ladder.get("closed_on_material") else 0.0,
        "lifted": w_lifted if ladder.get("lifted") else 0.0,
        "transported": w_transported if ladder.get("transported") else 0.0,
        "lowered": w_lowered if ladder.get("lowered_near_target") else 0.0,
        "released": w_released if ladder.get("released_after_place") else 0.0,
        "preamble": preamble_pen,
        "interaction": interaction,
        "noop": (max(-NOOP_CAP_I, inter.get("noop", 0.0)) if V11I else inter.get("noop", 0.0)) if V11E else 0.0,
        "duplicate": max(-DUP_CAP, W_DUP * dups.get("extra_repeats", 0)) if V11E else 0.0,
        "retain": W_RETAIN if (phys["retain"] and not V11D) else 0.0,
        "all_physical_ok": w_all_ok if phys["all_ok"] else 0.0,
        "no_action": W_NO_ACTION if phys["executed"] == 0 else 0.0,
        "tool_error": W_TOOL_ERROR * min(TOOL_ERROR_CAP, errs["total"]),
        "no_finish": (W_NO_FINISH_I if V11I else W_NO_FINISH) if (termination == "agent_stopped_without_finish" and not success) else 0.0,
        "exhausted": ((W_REQUEST_CAP_I if (V11I and termination == "request_cap_exhausted") else W_EXHAUSTED)
                      if termination in ("wall_time_exhausted", "context_exhausted", "request_cap_exhausted") else 0.0),
        "length_penalty": -LENGTH_PENALTY_PER * max(0.0, (out_tokens - LENGTH_FREE_TOKENS) / LENGTH_PENALTY_UNIT),
        "verbosity": verbosity,
    }
    reward = sum(comps.values())
    if success:
        verdict = "hidden_success"
    elif false_finish:
        verdict = "false_finish"
    elif phys["executed"] == 0:
        verdict = "no_action"
    elif termination == "agent_stopped_without_finish":
        verdict = "no_finish"
    elif termination in ("wall_time_exhausted", "context_exhausted", "request_cap_exhausted"):
        verdict = termination
    else:
        verdict = "acted"
    diag.update({"host_final_success": success, "host_success_ever": success_ever, "agent_finished": agent_finished,
                 "gate_rejections": gate_rejections, "physical": phys, "tool_errors": errs,
                 "action_requests": server.get("action_requests"), "tool_requests": server.get("tool_requests"),
                 "view_image_calls": driver.get("view_image_calls"), "finish_note": (server.get("finish_note") or "")[:200],
                 "tool_output_tokens_per_request": round(verb["per_request"]), "tool_output_tokens": verb["text_tokens"],
                 "big_outputs": verb["big_outputs"], "max_output_tokens": verb["max_output"], "grasp": grasp, "ladder": ladder, "preamble": pre,
                 "interaction": inter, "duplicates": dups,
                 "hidden_objects": None if hidden is None else {"targets": hidden["targets"], "movable_targets": hidden["movable_targets"],
                                                                "destinations": hidden["destinations"], "rows": hidden["rows"]}})
    return _result(reward, verdict, comps, **diag)


COMPONENT_KEYS = ("hidden", "success_ever", "false_finish", "gate", "physical", "first_move", "retain", "all_physical_ok",
                  "no_action", "tool_error", "no_finish", "exhausted", "length_penalty", "verbosity", "grasp",
                  "closed_on_material", "lifted", "transported", "lowered", "released", "preamble", "interaction", "noop", "duplicate", "approach", "close_near", "perceive",
                  "goal_progress", "goal_satisfied", "action_cost")
VERDICTS = ("hidden_success", "false_finish", "no_action", "no_finish", "wall_time_exhausted", "context_exhausted", "request_cap_exhausted",
            "acted", "infrastructure_error")


if __name__ == "__main__":
    import sys
    for d in sys.argv[1:]:
        rj = Path(d) / "rollout.json"
        reqs = (_read_json(rj).get("requests") if rj.is_file() else None)
        r = score_episode(Path(d), proxy_requests=reqs)
        print(json.dumps({"dir": d, "reward": round(r["reward"], 3), "verdict": r["verdict"], "components": {k: round(v, 3) for k, v in r["components"].items()},
                          "physical": r["diagnostics"].get("physical"), "tool_errors": r["diagnostics"].get("tool_errors"),
                          "termination": r["diagnostics"].get("termination_reason"),
                          "tool_output_tokens_per_request": r["diagnostics"].get("tool_output_tokens_per_request"),
                          "ladder": r["diagnostics"].get("ladder"), "preamble": r["diagnostics"].get("preamble"),
                          "interaction": r["diagnostics"].get("interaction"), "duplicates": r["diagnostics"].get("duplicates")}, ensure_ascii=False))

#!/usr/bin/env python3
"""Synthetic edge cases for reward v11e (rl/reward.py). Run: RL_REWARD_VERSION=v11e python -m pytest -q rl/test_reward_v11e.py
(or plain `python rl/test_reward_v11e.py`)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["RL_REWARD_VERSION"] = "v11e"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import importlib  # noqa: E402

import pytest  # noqa: E402

import reward as R  # noqa: E402


@pytest.fixture(autouse=True)
def _reward_version():
    """reward reads RL_REWARD_VERSION at import: re-import per test so this file can run in one session with the v11h tests."""
    os.environ["RL_REWARD_VERSION"] = "v11e"
    importlib.reload(R)
    yield

T0 = 1_000_000.0


def robot(pos, grip):
    return {"end_effector_position_m": list(pos), "gripper_open_fraction": grip}


def call(name, t_s, *, consumes=True, executed=True, success=True, args=None, before=None, after=None, reason=None, value=None):
    v = dict(value or {})
    if before is not None:
        v["robot_before"] = before
    if after is not None:
        v["robot_after"] = after
    if reason:
        v["stopped_reason"] = reason
    return {"name": name, "t_s": t_s, "consumes_action": consumes, "executed": executed, "success": success,
            "arguments": args or {}, "value": v, "call_index": int(t_s)}


def write_episode(tmp: Path, *, calls, scripts, success=False, finish_source="driver_exit", termination="agent_stopped_without_finish",
                  error=None, items_file="transcript.jsonl", request_times=None):
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "driver").mkdir(exist_ok=True)
    (tmp / "episode_result.json").write_text(json.dumps({"termination_reason": termination, "error": error,
                                                         "driver": {"requests": len(scripts) + 1, "usage": {"output_tokens": 1000}}}))
    (tmp / "result.json").write_text(json.dumps({"host_final_success": success, "host_success_ever": success, "finish_source": finish_source,
                                                 "gate_rejections": 0, "started_wall_time": T0}))
    (tmp / "server.json").write_text(json.dumps({"started_wall_time": T0}))
    (tmp / "calls.jsonl").write_text("\n".join(json.dumps(c) for c in calls) + ("\n" if calls else ""))
    # request k's response arrives at T0 + 10k; its tools run in (10k, 10k+10)
    n = len(scripts) + 1
    times = request_times or [T0 + 10.0 * k for k in range(1, n + 1)]
    (tmp / "driver" / "requests.jsonl").write_text("\n".join(json.dumps({"index": k + 1, "t": t}) for k, t in enumerate(times)) + "\n")
    items = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "task"}]}]
    for k, sc in enumerate(scripts):
        items.append({"type": "custom_tool_call", "name": "exec", "input": sc, "call_id": f"c{k}"})
        items.append({"type": "custom_tool_call_output", "call_id": f"c{k}", "output": [{"type": "input_text", "text": "Script completed"}]})
    items.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]})
    if items_file == "items.jsonl":
        (tmp / "driver" / "items.jsonl").write_text("\n".join(json.dumps({"id": i, "item": it}) for i, it in enumerate(items)) + "\n")
        (tmp / "driver" / "transcript.jsonl").write_text(json.dumps(items[-1]) + "\n")   # compacted: only the tail survives
    else:
        (tmp / "driver" / "transcript.jsonl").write_text("\n".join(json.dumps(it) for it in items) + "\n")
    return tmp


def teacher_calls():
    """observe -> approach -> close on material -> lift -> transport -> lower near target -> release (requests 1..8)."""
    p = [0.5, 0.0, 0.10]
    return [
        call("observe", 12, consumes=False, value={}),
        call("segment", 13, consumes=False, value={"detections": [{"label": "bowl", "centroid_m": [0.5, 0.0, 0.05]}, {"label": "plate", "centroid_m": [0.5, 0.3, 0.02]}]}),
        call("move_to", 22, args={"position": [0.5, 0.0, 0.10], "gripper": "open"}, before=robot([0.4, 0, 0.3], 1.0), after=robot(p, 1.0)),
        call("move_to", 32, args={"position": [0.5, 0.0, 0.10], "gripper": "close"}, before=robot(p, 1.0), after=robot(p, 0.45)),
        call("move_to", 42, args={"position": [0.5, 0.0, 0.20]}, before=robot(p, 0.45), after=robot([0.5, 0.0, 0.20], 0.45)),
        call("move_to", 52, args={"position": [0.5, 0.3, 0.20]}, before=robot([0.5, 0, 0.2], 0.45), after=robot([0.5, 0.3, 0.20], 0.45)),
        call("move_to", 62, args={"position": [0.5, 0.3, 0.10]}, before=robot([0.5, 0.3, 0.2], 0.45), after=robot([0.5, 0.3, 0.10], 0.45)),
        call("release", 72, before=robot([0.5, 0.3, 0.1], 0.45), after=robot([0.5, 0.3, 0.10], 1.0)),
    ]


TEACHER_SCRIPTS = ["observe(); segment('bowl')", "move_to(open)", "move_to(close)", "move_to(lift)", "move_to(transport)", "move_to(lower)", "release()",
                   "finish('placed')"]


def test_teacher_like_success(tmp_path):
    d = write_episode(tmp_path / "t", calls=teacher_calls(), scripts=TEACHER_SCRIPTS, success=True, finish_source="agent", termination="agent_finished")
    r = R.score_episode(d)
    c = r["components"]
    assert r["verdict"] == "hidden_success"
    for k in ("closed_on_material", "lifted", "transported", "lowered", "released"):
        assert c[k] > 0, k
    assert abs(c["interaction"] - 6 * R.W_INTERACT_FIRST) < 1e-9          # 6 clean requests (approach, close, lift, transport, lower, release)
    assert c["noop"] == 0 and c["duplicate"] == 0
    assert 3.0 <= r["reward"] <= 3.4, r["reward"]
    assert r["diagnostics"]["interaction"]["kinds"] == "OCCCCCCF-"


def test_empty_close_earns_no_ladder(tmp_path):
    calls = teacher_calls()[:4]
    calls[3] = call("move_to", 32, args={"position": [0.5, 0.0, 0.10], "gripper": "close"}, before=robot([0.5, 0, 0.1], 1.0), after=robot([0.5, 0, 0.1], 0.01))
    d = write_episode(tmp_path / "e", calls=calls, scripts=TEACHER_SCRIPTS[:3])
    r = R.score_episode(d)
    lad = r["diagnostics"]["ladder"]
    assert not lad["closed_on_material"] and not lad["lifted"]
    assert r["components"]["interaction"] > 0            # the attempts themselves are still rewarded as interaction
    real = R.score_episode(write_episode(tmp_path / "r", calls=teacher_calls()[:4], scripts=TEACHER_SCRIPTS[:3]))
    assert real["reward"] > r["reward"] + 0.09           # closed_on_material separates a real grasp from an empty close


def test_noop_spin_is_charged_and_bounded(tmp_path):
    calls = [call("observe", 12, consumes=False), call("move_to", 22, args={"position": [0.5, 0, 0.1]}, before=robot([0.4, 0, 0.3], 1.0), after=robot([0.5, 0, 0.1], 1.0))]
    spin = ["preview_motion(...)"] * 30                   # 30 preview-only requests after the first action
    for k in range(30):
        calls.append(call("preview_motion", 32 + 10 * k, consumes=False, args={"action": "move_to"}))
    d = write_episode(tmp_path / "n", calls=calls, scripts=["observe()", "move_to()"] + spin)
    r = R.score_episode(d)
    inter = r["diagnostics"]["interaction"]
    assert inter["noop_requests"] == 30
    assert abs(r["components"]["noop"] + R.NOOP_CAP) < 1e-9                 # bounded at the cap
    assert R.NOOP_CAP + R.DUP_CAP < R.W_CLOSED_ON_MATERIAL + R.W_LIFTED + R.W_TRANSPORTED + R.W_LOWERED_NEAR_TARGET + R.W_RELEASED_AFTER_PLACE


def test_noop_free_allowance_and_exemptions(tmp_path):
    # 4 no-ops after the first action are free; the final request and finish() requests are never no-ops; recover-only requests are no-ops
    calls = [call("move_to", 12, args={"position": [0.5, 0, 0.1]}, before=robot([0.4, 0, 0.3], 1.0), after=robot([0.5, 0, 0.1], 1.0))]
    scripts = ["move_to()"]
    for k in range(3):
        calls.append(call("preview_motion", 22 + 10 * k, consumes=False))
        scripts.append("preview")
    calls.append(call("recover", 52, args={"max_distance_m": 0.02}, before=robot([0.5, 0, 0.1], 1.0), after=robot([0.5, 0, 0.12], 1.0)))
    scripts.append("recover()")
    scripts.append("finish('x')")
    d = write_episode(tmp_path / "f", calls=calls, scripts=scripts)
    r = R.score_episode(d)
    inter = r["diagnostics"]["interaction"]
    assert inter["kinds"] == "C----F-", inter
    assert inter["noop_requests"] == 4 and r["components"]["noop"] == 0


def test_observing_requests_are_not_noops(tmp_path):
    calls = [call("move_to", 12, args={"position": [0.5, 0, 0.1]}, before=robot([0.4, 0, 0.3], 1.0), after=robot([0.5, 0, 0.1], 1.0))]
    scripts = ["move_to()"]
    for k in range(6):
        calls.append(call("mark_point", 22 + 10 * k, consumes=False, value={"position_m": [0.5, 0, 0.05]}))
        scripts.append("mark_point()")
    scripts.append("tools.view_image('images/x.png')")   # view_image is a driver tool: only visible in the script text
    d = write_episode(tmp_path / "o", calls=calls, scripts=scripts)
    r = R.score_episode(d)
    assert r["diagnostics"]["interaction"]["noop_requests"] == 0
    assert r["diagnostics"]["interaction"]["kinds"] == "C" + "O" * 7 + "-"


def test_duplicate_pose_penalty_bounded(tmp_path):
    calls = []
    scripts = []
    for k in range(9):   # same pose within 1 cm, 9 times (7 extra repeats)
        calls.append(call("move_to", 12 + 10 * k, args={"position": [0.5 + 0.004 * (k % 2), 0.0, 0.1], "gripper": "close"}, executed=False, success=False))
        scripts.append("move_to(close)")
    d = write_episode(tmp_path / "d", calls=calls, scripts=scripts)
    r = R.score_episode(d)
    assert r["diagnostics"]["duplicates"]["extra_repeats"] == 7
    assert abs(r["components"]["duplicate"] + R.DUP_CAP) < 1e-9
    # open vs close at the same pose are different requests; 2 cm apart is a different pose
    calls2 = [call("move_to", 12, args={"position": [0.5, 0, 0.1], "gripper": "open"}), call("move_to", 22, args={"position": [0.5, 0, 0.1], "gripper": "close"}),
              call("move_to", 32, args={"position": [0.52, 0, 0.1], "gripper": "close"}), call("move_to", 42, args={"position": [0.5, 0, 0.1], "gripper": "close"})]
    assert R.duplicate_requests(calls2)["extra_repeats"] == 0


def test_interaction_bonus_diminishing_and_capped():
    assert R.interaction_terms([], [], [], None)["interaction"] == 0.0
    calls = []
    scripts = []
    for k in range(40):
        calls.append(call("move_to", 12 + 10 * k, args={"position": [0.5, 0.01 * k, 0.1]}, before=robot([0.4, 0, 0.3], 1.0), after=robot([0.5, 0.01 * k, 0.1], 1.0)))
        scripts.append("move_to()")
    rows = [{"t": T0 + 10.0 * k} for k in range(1, 42)]
    items = []
    for sc in scripts:
        items += [{"type": "custom_tool_call", "input": sc}, {"type": "custom_tool_call_output", "output": "ok"}]
    t = R.interaction_terms(calls, items, rows, T0)
    assert t["clean_requests"] == 40
    assert abs(t["interaction"] - R.INTERACT_CAP) < 1e-9
    t7 = R.interaction_terms(calls[:7], items[:14], rows[:8], T0)
    assert abs(t7["interaction"] - (6 * R.W_INTERACT_FIRST + R.W_INTERACT_AFTER)) < 1e-9
    # many actions inside ONE request count once (no farming through a single script)
    one = [call("move_to", 12 + 0.1 * k, args={"position": [0.5, 0.01 * k, 0.1]}, before=robot([0.4, 0, 0.3], 1.0), after=robot([0.5, 0.01 * k, 0.1], 1.0)) for k in range(20)]
    t1 = R.interaction_terms(one, items[:2], rows[:2], T0)
    assert t1["clean_requests"] == 1 and abs(t1["interaction"] - R.W_INTERACT_FIRST) < 1e-9


def test_force_stopped_action_is_not_clean():
    c = call("move_to", 12, args={"position": [0.5, 0, 0.1]}, reason="contact force limit", before=robot([0.4, 0, 0.3], 1.0), after=robot([0.5, 0, 0.1], 1.0))
    assert not R._clean_interaction(c)
    assert R._clean_interaction(call("hold", 12, before=robot([0.5, 0, 0.1], 1.0), after=robot([0.5, 0, 0.1], 0.4)))
    assert not R._clean_interaction(call("recover", 12))


def test_items_jsonl_preferred_over_compacted_transcript(tmp_path):
    d = write_episode(tmp_path / "i", calls=teacher_calls(), scripts=TEACHER_SCRIPTS, success=True, finish_source="agent", termination="agent_finished",
                      items_file="items.jsonl")
    r = R.score_episode(d)
    assert r["diagnostics"]["interaction"]["kinds"] == "OCCCCCCF-"
    assert r["diagnostics"]["preamble"]["first_physical_request"] == 2


def test_timeline_fallback_to_script_text(tmp_path):
    d = write_episode(tmp_path / "s", calls=teacher_calls()[:3], scripts=["observe()", "move_to(open)", "cat notes.txt"], request_times=[None] * 4)
    r = R.score_episode(d)
    inter = r["diagnostics"]["interaction"]
    assert inter["timeline"] is False and inter["kinds"] == "OA-"


def test_request_cap_is_a_policy_outcome(tmp_path):
    d = write_episode(tmp_path / "c", calls=teacher_calls()[:3], scripts=TEACHER_SCRIPTS[:2], termination="infrastructure_error", error="request cap 40 reached")
    r = R.score_episode(d)
    assert r["verdict"] == "request_cap_exhausted" and r["components"]["exhausted"] == R.W_EXHAUSTED


def test_tool_output_tokens_compaction_robust():
    # prompts: 20k -> 23k -> 26k -> compaction to 15k -> 18k; completions 200 each; growth per transition 2.8k, 2.8k, 0, 2.8k
    reqs = [{"prompt_len": 20000, "completion_len": 200}, {"prompt_len": 23000, "completion_len": 200}, {"prompt_len": 26000, "completion_len": 200},
            {"prompt_len": 15000, "completion_len": 200}, {"prompt_len": 18000, "completion_len": 200}]
    v = R.tool_output_tokens(reqs)
    assert v["text_tokens"] == 3 * 2800 and v["transitions"] == 4
    assert abs(v["per_request"] - 2100) < 1e-6


if __name__ == "__main__":
    import inspect
    import tempfile
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                if "tmp_path" in inspect.signature(fn).parameters:
                    with tempfile.TemporaryDirectory() as td:
                        fn(Path(td))
                else:
                    fn()
                print("ok  ", name)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("FAIL", name, repr(exc))
    sys.exit(1 if failed else 0)

"""V1.2 context management: turn-atomic cuts, programmatic drop notice, incremental summary, pre-request compaction,
tool-output spill, exact per-request views."""
from __future__ import annotations

import json
import os

import pytest

from captain_harness.react import context as ctx
from captain_harness.react.catalog import ModelProfile, profile_for, small_context_profile
from captain_harness.react.loop import Loop
from captain_harness.react.sandbox import Sandbox
from captain_harness.react.tools import ToolRouter
from captain_harness.react.unified_exec import SessionManager, child_environment
from test_react_llm_and_loop import ScriptedClient, _usage, harness, message  # noqa: F401


def _call(cid, cmd):
    return {"type": "function_call", "call_id": cid, "name": "exec_command", "arguments": json.dumps({"cmd": cmd})}


def _out(cid, text):
    return {"type": "function_call_output", "call_id": cid, "output": text}


def test_split_turns_keeps_outputs_with_their_calls():
    items = [_call("a", "ls"), _call("b", "pwd"), _out("a", "x"), _out("b", "y"), message("note"), _call("c", "date"), _out("c", "z"),
             ctx.user_message("steer"), message("final")]
    turns = ctx.split_turns(items)
    assert [len(t) for t in turns] == [4, 3, 1, 1]
    assert turns[0][-1]["type"] == "function_call_output" and turns[1][0]["type"] == "message"
    assert all(not (t[0].get("type") == "function_call_output") for t in turns)


def test_choose_cut_keeps_newest_turns_within_budget():
    turns = [[_call(str(i), "x" * 400), _out(str(i), "y" * 400)] for i in range(6)]
    per_turn = ctx.estimate_tokens(turns[0])
    cut = ctx.choose_cut(turns, per_turn * 2 + 5)
    assert cut == 4
    assert ctx.choose_cut(turns, 1) == 5   # at least the newest turn survives


def test_drop_notice_carries_programmatic_facts_and_summary(tmp_path):
    (tmp_path / "notes.txt").write_text("n")
    state = ctx.CompactionState()
    dropped = [[_call("a", "python3 - <<'PY'\nprint(1)\nPY"), _out("a", "1"), message("Bowl grasped; moving to plate.")],
               [{"type": "custom_tool_call", "call_id": "b", "name": "exec", "input": "await tools.exec_command({cmd: 'echo hi'});"},
                {"type": "custom_tool_call_output", "call_id": "b", "output": []}]]
    state.absorb(dropped)
    assert state.omitted_turns == 2 and state.omitted_tool_calls == 2 and state.commands == ["python3 - <<'PY'\nprint(1)\nPY", "echo hi"]
    notice = ctx.drop_notice(state, workspace=tmp_path, summary="SUMMARY TEXT")
    text = ctx.text_of(notice)
    assert text.startswith(ctx.NOTICE_HEADER) and text.rstrip().endswith(ctx.NOTICE_FOOTER)
    assert "2 of your turns with 2 tool calls were removed" in text and "echo hi" in text and "Bowl grasped" in text
    assert "notes.txt" in text and ctx.SUMMARY_PREFIX in text and "SUMMARY TEXT" in text and ctx.is_notice(notice)


def test_summary_request_is_incremental_and_toolless():
    body = {"model": "m", "tools": [1], "tool_choice": "auto", "parallel_tool_calls": True, "store": False}
    req = ctx.summary_request(body, [ctx.user_message("task")], "OLD SUMMARY", [[_call("a", "ls"), _out("a", "x")]], 2_000)
    assert "tools" not in req and req["store"] is False
    texts = [ctx.text_of(it) for it in req["input"] if it.get("type") == "message"]
    assert texts[0] == "task" and texts[1].startswith("<previous-summary>\nOLD SUMMARY") and "<conversation>" in texts[2]
    assert ctx.SUMMARIZATION_PROMPT in texts[-1] and "1500 words" in texts[-1]
    assert ctx.extract_summary([message("a" * 100)], 10).endswith("…") and ctx.extract_summary([], 10) == ctx.NO_SUMMARY


def test_loop_compacts_before_request_with_summary_strategy(harness, tmp_path):
    ws, router = harness
    # limit 2000 estimated tokens, keep the newest ~600 tokens of turns verbatim
    profile = ModelProfile("tiny", "gpt-5.5.request.json", "function", 10_000, 4_000, 95, 2_000, True, "summary", 600, 500)
    client = ScriptedClient([
        ([_call("c1", "python3 -c \"print('a'*1600)\"")], _usage(900)),     # est before r2: 900 + ~420 < 2000
        ([_call("c2", "python3 -c \"print('b'*3200)\"")], _usage(1_200)),  # est before r3: 1200 + ~820 >= 2000 -> compact
        ([message("HANDOFF: printed a's, then b's")], _usage(300)),          # summary request
        ([message("done")], _usage(700)),
    ])
    initial = [ctx.user_message("go")]
    loop = Loop(client=client, body={"model": "tiny", "tools": [{"type": "function", "name": "exec_command"}], "store": False},
                history=list(initial), router=router, profile=profile, output_dir=tmp_path / "d", prompt_cache_key="k", workspace=ws)
    result = loop.run()
    assert result.termination == "completed" and result.compactions == 1 and result.requests == 3 and len(client.bodies) == 4
    summary_body = client.bodies[2]
    assert "tools" not in summary_body and "<conversation>" in json.dumps(summary_body["input"]) and "print('a'*1600)" in json.dumps(summary_body["input"])
    after = client.bodies[3]["input"]
    assert after[0] == initial[0] and ctx.is_notice(after[1])
    notice = ctx.text_of(after[1])
    assert "HANDOFF" in notice and "print('a'*1600)" in notice and "1 of your turns with 1 tool calls" in notice
    # the newest turn (c2 call + its output) is kept verbatim, the older one is gone, and a continuation cue closes the history
    assert [it["type"] for it in after[2:]] == ["function_call", "function_call_output", "message"] and after[2]["call_id"] == "c2"
    assert "b" * 100 in after[3]["output"] and ctx.is_continue(after[-1])
    rows = [json.loads(l) for l in (tmp_path / "d" / "compactions.jsonl").read_text().splitlines()]
    assert rows[0]["strategy"] == "summary" and rows[0]["dropped_turns"] == 1 and rows[0]["kept_turns"] == 1 and not rows[0]["forced"]
    # views.jsonl + items.jsonl reconstruct exactly what request 3 saw
    views = [json.loads(l) for l in (tmp_path / "d" / "views.jsonl").read_text().splitlines()]
    items = {r["id"]: r["item"] for r in (json.loads(l) for l in (tmp_path / "d" / "items.jsonl").read_text().splitlines())}
    reconstructed = [items[i] for i in views[2]["items"]]
    assert [it["type"] for it in reconstructed] == [it["type"] for it in after] and reconstructed[1]["content"] == after[1]["content"]
    assert views[2]["compactions"] == 1 and views[1]["compactions"] == 0


def test_loop_window_strategy_makes_no_summary_request(harness, tmp_path):
    ws, router = harness
    small = small_context_profile(profile_for("gpt-5.5"), context_window=8_000, truncation_tokens=500)
    assert small.context_strategy == "window" and small.spill_tool_output and small.context_limit == 8_000 - max(800, 3 * 500 + 4_000)
    profile = ModelProfile("tiny", "gpt-5.5.request.json", "function", 10_000, 4_000, 95, 2_000, True, "window", 600, 500)
    client = ScriptedClient([
        ([_call("c1", "python3 -c \"print('a'*1600)\"")], _usage(900)),
        ([_call("c2", "python3 -c \"print('b'*3200)\"")], _usage(1_200)),
        ([message("done")], _usage(700)),
    ])
    loop = Loop(client=client, body={"model": "tiny"}, history=[ctx.user_message("go")], router=router, profile=profile,
                output_dir=tmp_path / "d", prompt_cache_key="k", workspace=ws)
    result = loop.run()
    assert result.termination == "completed" and result.compactions == 1 and result.requests == 3 and len(client.bodies) == 3
    notice = ctx.text_of(client.bodies[2]["input"][1])
    assert ctx.NOTICE_HEADER in notice and ctx.SUMMARY_PREFIX not in notice and "print('a'*1600)" in notice


def test_single_oversized_turn_is_dropped_as_last_resort(harness, tmp_path):
    ws, router = harness
    profile = ModelProfile("tiny", "gpt-5.5.request.json", "function", 10_000, 4_000, 95, 1_000, True, "window", 300, 500)
    client = ScriptedClient([
        ([_call("c1", "python3 -c \"print('a'*6000)\"")], _usage(500)),   # output ~1500 tokens alone exceeds the limit
        ([message("done")], _usage(400)),
    ])
    loop = Loop(client=client, body={"model": "tiny"}, history=[ctx.user_message("go")], router=router, profile=profile,
                output_dir=tmp_path / "d", prompt_cache_key="k", workspace=ws)
    result = loop.run()
    assert result.termination == "completed" and result.compactions == 1
    after = client.bodies[1]["input"]
    assert len(after) == 3 and ctx.is_notice(after[1]) and "print('a'*6000)" in ctx.text_of(after[1]) and ctx.is_continue(after[2])


def test_spill_writes_full_output_and_points_to_it(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    m = SessionManager(sandbox=Sandbox(workspace=ws, bwrap=None), env=child_environment(os.environ), default_cwd=ws,
                       output_limit_tokens=50, spill_dir=ws / ".tool_output")
    try:
        r = m.exec_command(cmd="seq 1 400")
    finally:
        m.close_all()
    assert r.output.startswith("Warning: truncated output") and "[Full output saved to .tool_output/" in r.output
    files = list((ws / ".tool_output").glob("*.txt"))
    assert len(files) == 1 and files[0].read_text().splitlines()[-1] == "400" and f"{files[0].name}" in r.output
    m2 = SessionManager(sandbox=Sandbox(workspace=ws, bwrap=None), env=child_environment(os.environ), default_cwd=ws, output_limit_tokens=50)
    try:
        r2 = m2.exec_command(cmd="seq 1 400")
    finally:
        m2.close_all()
    assert "Full output saved" not in r2.output  # Codex-faithful profiles keep the plain truncation


def test_item_registry_never_reuses_ids_of_dropped_items(tmp_path):
    """The registry keys items by id(); dropped turns are freed after a compaction and CPython reuses their addresses,
    so without a strong reference a later response item would silently map to an old id (corrupt views.jsonl)."""
    import gc
    from captain_harness.react.loop import Loop

    class _Router:
        commands: list = []

    loop = Loop.__new__(Loop)
    loop.output_dir = tmp_path
    loop.items_path = tmp_path / "items.jsonl"
    loop.router = _Router()
    loop._item_ids, loop._item_refs, loop._next_item_id = {}, [], 0
    first = [{"type": "function_call_output", "call_id": str(i), "output": "x" * 100} for i in range(200)]
    ids = [loop._register(it) for it in first]
    del first
    gc.collect()
    later = [{"type": "function_call_output", "call_id": f"n{i}", "output": "y" * 100} for i in range(200)]
    new_ids = [loop._register(it) for it in later]
    assert min(new_ids) > max(ids) and len(set(new_ids)) == 200
    rows = [json.loads(line) for line in (tmp_path / "items.jsonl").read_text().splitlines()]
    assert len(rows) == 400 and rows[-1]["item"]["call_id"] == "n199"


def test_usage_normalisation_accepts_chat_completions_shape():
    u = ctx.normalize_usage({"prompt_tokens": 1200, "completion_tokens": 50, "prompt_tokens_details": {"cached_tokens": 800}})
    assert u["input_tokens"] == 1200 and u["output_tokens"] == 50 and u["total_tokens"] == 1250 and u["input_tokens_details"]["cached_tokens"] == 800
    assert ctx.normalize_usage({"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})["total_tokens"] == 15


def test_estimator_calibrates_to_the_provider_tokenizer():
    est = ctx.TokenEstimator()
    items = [ctx.user_message("x" * 12000)]          # raw estimate ~3000 tokens
    raw = ctx.estimate_tokens(items)
    est.calibrate(raw, int(raw * 1.18))               # provider counts 18% more
    assert abs(est.scale - 1.18) < 0.01 and est.estimate(items) == int(round(raw * est.scale))
    est.calibrate(raw, int(raw * 1.18))
    assert abs(est.scale - 1.18) < 0.01
    est.calibrate(raw, raw * 50)                       # absurd report is clamped
    assert est.scale <= est.max_scale
    tiny = ctx.TokenEstimator()
    tiny.calibrate(200, 9000)                          # too small to calibrate on
    assert tiny.scale == 1.0 and tiny.samples == 0


def test_loop_trigger_uses_calibrated_estimate_when_usage_is_chat_shaped(harness, tmp_path):
    ws, router = harness
    # limit 6000; the provider counts 1.5x our raw estimate and reports chat-completions usage keys. Uncalibrated, request 3
    # would be sent at ~5300 estimated and ~7900 real tokens; calibrated, the loop compacts before it.
    profile = ModelProfile("tiny", "gpt-5.5.request.json", "function", 10_000, 20_000, 95, 6_000, True, "window", 1_500, 500)
    from captain_harness.react.llm import ResponseResult

    class ChatShapedClient(ScriptedClient):
        def respond(self, body):
            self.bodies.append(json.loads(json.dumps(body)))
            items, _ = self.responses.pop(0)
            raw = ctx.estimate_tokens([{k: v for k, v in body.items() if k != "input"}]) + ctx.estimate_tokens(body["input"])
            return ResponseResult(response_id=f"r{len(self.bodies)}", items=items, status="completed", latency_s=0.0, attempts=1,
                                  usage={"prompt_tokens": int(raw * 1.5), "completion_tokens": 20})

    client = ChatShapedClient([
        ([_call("c1", "python3 -c \"print('a'*8000)\"")], None),
        ([_call("c2", "python3 -c \"print('b'*8000)\"")], None),
        ([message("done")], None),
    ])
    loop = Loop(client=client, body={"model": "tiny", "instructions": "s" * 4000}, history=[ctx.user_message("go")], router=router,
                profile=profile, output_dir=tmp_path / "d", prompt_cache_key="k", workspace=ws)
    result = loop.run()
    assert result.termination == "completed" and result.compactions >= 1
    views = [json.loads(l) for l in (tmp_path / "d" / "views.jsonl").read_text().splitlines()]
    assert views[-1]["estimate_scale"] > 1.4
    for body in client.bodies:   # every request the provider saw stayed under the limit in real (1.5x) tokens
        real = 1.5 * (ctx.estimate_tokens([{k: v for k, v in body.items() if k != "input"}]) + ctx.estimate_tokens(body["input"]))
        assert real <= profile.context_limit, real


def test_hard_limit_derives_trigger():
    from captain_harness.react.catalog import one_turn_tokens, profile_for, small_context_profile, trigger_for_hard_limit
    student = small_context_profile(profile_for("gpt-5.5"), context_window=81_920, truncation_tokens=4_000)
    fitted = trigger_for_hard_limit(student, 65_536)
    assert fitted.context_limit == 65_536 - one_turn_tokens(student) == 65_536 - (3 * 4_000 + 2_000 + 2_000)
    with pytest.raises(ValueError):
        trigger_for_hard_limit(student, 20_000)

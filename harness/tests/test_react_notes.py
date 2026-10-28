"""Notes-strategy compaction (checkpoint turn) and the server /history route."""
from __future__ import annotations

import json
import os

from captain_harness.react import context as ctx
from captain_harness.react.catalog import ModelProfile
from captain_harness.react.loop import Loop
from test_react_llm_and_loop import ScriptedClient, _usage, harness, message  # noqa: F401


def _call(cid, cmd):
    return {"type": "function_call", "call_id": cid, "name": "exec_command", "arguments": json.dumps({"cmd": cmd})}


def test_notes_strategy_runs_a_checkpoint_turn_and_carries_the_file(harness, tmp_path):
    ws, router = harness
    profile = ModelProfile("tiny", "gpt-5.5.request.json", "function", 10_000, 4_000, 95, 2_000, True, "notes", 600, 500)
    client = ScriptedClient([
        ([_call("c1", "python3 -c \"print('a'*1600)\"")], _usage(900)),
        ([_call("c2", "python3 -c \"print('b'*3200)\"")], _usage(1_200)),       # est >= 2000 before request 3 -> checkpoint turn
        ([message("Saving notes."), _call("ck", "printf 'GOAL: demo\\nNEXT: run c3\\n' > NOTES.md")], _usage(1_300)),  # checkpoint
        ([message("done")], _usage(700)),
    ])
    loop = Loop(client=client, body={"model": "tiny"}, history=[ctx.user_message("go")], router=router, profile=profile,
                output_dir=tmp_path / "d", prompt_cache_key="k", workspace=ws)
    result = loop.run()
    assert result.termination == "completed" and result.compactions == 1 and len(client.bodies) == 4
    checkpoint_body = client.bodies[2]["input"]
    assert "<context_checkpoint>" in ctx.text_of(checkpoint_body[-1]) and "NOTES.md" in ctx.text_of(checkpoint_body[-1])
    assert (ws / "NOTES.md").read_text() == "GOAL: demo\nNEXT: run c3\n"
    after = client.bodies[3]["input"]
    notice = ctx.text_of(after[1])
    assert ctx.is_notice(after[1]) and "Contents of your NOTES.md" in notice and "NEXT: run c3" in notice and ctx.SUMMARY_PREFIX not in notice
    # the checkpoint exchange is not kept; the newest real turn is, and the history ends with the continuation cue
    assert all("<context_checkpoint>" not in ctx.text_of(it) for it in after if it.get("type") == "message")
    assert all(it.get("call_id") != "ck" for it in after)
    assert after[-2]["type"] == "function_call_output" and after[-2]["call_id"] == "c2" and ctx.is_continue(after[-1])
    rows = [json.loads(l) for l in (tmp_path / "d" / "compactions.jsonl").read_text().splitlines()]
    assert rows[0]["strategy"] == "notes"
    events = [json.loads(l) for l in (tmp_path / "d" / "events.jsonl").read_text().splitlines()]
    assert any(e.get("item", {}).get("type") == "checkpoint" for e in events)


def test_notice_without_notes_asks_for_one(tmp_path):
    state = ctx.CompactionState()
    state.absorb([[_call("a", "ls"), {"type": "function_call_output", "call_id": "a", "output": "x"}]])
    text = ctx.text_of(ctx.drop_notice(state, workspace=tmp_path, summary=None, notes=None))
    assert "You had not written NOTES.md" in text

"""python-runtime/v1 through the real loop: one tool, private audit per cell, replay and boundary hooks."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from captain_harness.react.catalog import profile_for
from captain_harness.react.chat_llm import chat_message_to_items, convert_tools
from captain_harness.react.llm import ResponseResult
from captain_harness.react.loop import Loop
from captain_harness.react.python_repl import (
    PYTHON_RUNTIME_CONTRACT_VERSION,
    RUNTIME_NOTE,
    PythonRepl,
    python_runtime_contract,
    run_python_cell,
)
from captain_harness.react.replay import ReplayClient, load_responses
from captain_harness.react.request import render_request
from captain_harness.react.sandbox import Sandbox
from captain_harness.react.tools import ToolRouter
from captain_harness.react.unified_exec import SessionManager, child_environment

PREFLIGHT = {"schema_version": "harness-toolchain-preflight/v1", "ok": True, "fingerprint": "test"}


def call(n, code, name="python"):
    return {"type": "custom_tool_call", "call_id": f"call_{n}", "name": name, "input": code}


class ScriptedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = 0
        self.bodies = []

    def respond(self, body):
        self.requests += 1
        self.bodies.append(body)
        items = self.responses[self.requests - 1]
        return ResponseResult(response_id=f"response_{self.requests}", items=items,
                              usage={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}, status="completed",
                              latency_s=0, attempts=1)


def make_loop(tmp_path, responses, *, profile="captain", client=None, output_limit_tokens=10_000, marks=None, max_requests=None):
    ws = tmp_path / "episode" / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    sandbox = Sandbox(workspace=ws.resolve())
    env = child_environment(os.environ)
    router = ToolRouter(sessions=SessionManager(sandbox=sandbox, env=env, default_cwd=ws), workspace=ws)
    repl = PythonRepl(sandbox=sandbox, env=env, python=sys.executable, startup="helper = 7\n")
    repl.start()
    router.closers.append(repl.close)
    body, items = render_request(profile_for(profile), model="student", workspace=ws, prompt="test")
    loop = Loop(client=client or ScriptedClient(responses), body=body, history=items, router=router, profile=profile_for(profile),
                output_dir=ws.parent / "driver", workspace=ws, prompt_cache_key="test",
                max_requests=max_requests if max_requests is not None else len(responses),
                python_runner=lambda source: run_python_cell(source, router=router, repl=repl, output_limit_tokens=output_limit_tokens),
                runtime_tool_contract=PYTHON_RUNTIME_CONTRACT_VERSION, environment_preflight=PREFLIGHT,
                tool_description="test python tool", before_tools=(marks.append if marks is not None else None))
    return loop, router


def audit(loop):
    return [json.loads(line) for line in (loop.output_dir / "tool_audit.jsonl").read_text().splitlines()]


def outputs(loop):
    return [item for item in loop.history if item.get("type") == "custom_tool_call_output"]


@pytest.mark.parametrize("profile", ["captain", "gpt-5.5"])
def test_contract_exposes_exactly_the_python_tool_and_leaves_templates_untouched(profile):
    body, items = render_request(profile_for(profile), model="student", workspace="/workspace", prompt="test")
    original = json.dumps([body, items])
    adapted_body, adapted_items, contract = python_runtime_contract(body, items, PYTHON_RUNTIME_CONTRACT_VERSION, description="d")
    assert json.dumps([body, items]) == original
    tools, custom = convert_tools(adapted_body, adapted_items, allow=None)
    assert [tool["function"]["name"] for tool in tools] == ["python"] and custom == {"python"}
    assert list(tools[0]["function"]["parameters"]["properties"]) == ["input"]
    assert adapted_body["parallel_tool_calls"] is False
    assert contract["sha256"] == hashlib.sha256(json.dumps(tools, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    notes = [it for it in adapted_items if it.get("role") == "developer" and RUNTIME_NOTE in json.dumps(it)]
    assert len(notes) == 1
    first_user = next(i for i, it in enumerate(adapted_items) if it.get("role") == "user")
    assert adapted_items.index(notes[0]) == first_user - 1


def test_chat_round_trip_of_a_python_call():
    items = chat_message_to_items({"tool_calls": [{"id": "c1", "function": {"name": "python", "arguments": json.dumps({"input": "print(1)"})}}]},
                                  {"python"})
    assert items[0]["type"] == "custom_tool_call" and items[0]["name"] == "python" and items[0]["input"] == "print(1)"


def test_every_cell_is_audited_with_its_untruncated_output_and_errors(tmp_path):
    loop, router = make_loop(tmp_path, [[call(1, "print('z' * 30000)")], [call(2, "helper + 1")], [call(3, "1/0")],
                                        [call(4, "print(1)", name="exec")]], output_limit_tokens=200)
    result = loop.run()
    assert result.requests == 4
    rows = audit(loop)
    header, footer = rows[0], rows[-1]
    assert header["contract_version"] == PYTHON_RUNTIME_CONTRACT_VERSION and header["contract_sha256"] == loop.tool_contract["sha256"]
    ends = [row for row in rows if row["type"] == "tool_end"]
    assert [row["tool_name"] for row in ends] == ["python"] * 3
    assert ends[0]["result"]["output"].count("z") == 30000 and ends[0]["error"] is None
    assert ends[1]["result"]["display"] == "8" and ends[1]["result"]["status"] == "ok"
    assert ends[2]["result"]["status"] == "error" and ends[2]["error_class"] == "policy" and "ZeroDivisionError" in ends[2]["error"]
    outer = [row for row in rows if row["type"] == "outer_end"]
    assert [row["status"] for row in outer] == ["completed", "completed", "failed", "failed"]
    assert "unsupported" in (outer[3]["error"] or "")
    assert any(row["type"] == "tool_processes_closed" for row in rows)
    assert footer["type"] == "footer" and footer["complete"] is True and footer["outer_started"] == footer["outer_finished"] == 4
    shown = outputs(loop)[0]["output"][0]["text"]
    assert shown.count("z") < 2000 and "truncated" in shown


def test_duplicate_call_in_one_response_is_not_executed_and_hooks_see_request_indices(tmp_path):
    marks = []
    loop, router = make_loop(tmp_path, [[call(1, "counter = globals().get('counter', 0) + 1"), call(2, "counter = globals().get('counter', 0) + 1")],
                                        [call(3, "counter")]], marks=marks)
    loop.run()
    texts = [o["output"] if isinstance(o["output"], str) else o["output"][0]["text"] for o in outputs(loop)]
    assert "not executed" in texts[1] and "Out: 1" in texts[2]
    assert marks == [1, 2]
    assert [row["tool_name"] for row in audit(loop) if row["type"] == "tool_end"] == ["python", "python"]


def test_prefix_replay_reissues_recorded_responses_then_hands_over(tmp_path):
    source, _ = make_loop(tmp_path / "source", [[call(1, "x = 40")], [call(2, "x += 1")], [call(3, "x")]])
    source.run()
    recorded = load_responses(source.output_dir / "responses.jsonl", limit=2)
    assert [row["request"] for row in recorded] == [1, 2]
    replay = ReplayClient(ScriptedClient([[call(9, "x + 1")]]), recorded)
    loop, _ = make_loop(tmp_path / "replay", [], client=replay, max_requests=3)
    loop.run()
    requests = [json.loads(line) for line in (loop.output_dir / "requests.jsonl").read_text().splitlines()]
    assert [r["replayed"] for r in requests] == [True, True, False]
    assert "Out: 42" in outputs(loop)[-1]["output"][0]["text"]
    assert replay.served == 2 and replay.remaining == 0


def test_python_contract_requires_a_runner(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    body, items = render_request(profile_for("captain"), model="student", workspace=ws, prompt="test")
    router = ToolRouter(sessions=SessionManager(sandbox=Sandbox(workspace=ws), env={}, default_cwd=ws), workspace=ws)
    with pytest.raises(ValueError, match="python runner"):
        Loop(client=ScriptedClient([]), body=body, history=items, router=router, profile=profile_for("captain"),
             output_dir=tmp_path / "driver", workspace=ws, prompt_cache_key="t", runtime_tool_contract=PYTHON_RUNTIME_CONTRACT_VERSION,
             environment_preflight=PREFLIGHT)


def test_clean_handoff_replaces_session_workspace_and_conversation_but_keeps_counting(tmp_path):
    from captain_harness.server import python_session
    from captain_harness.server.episode_launch import restore_workspace, snapshot_workspace

    source, _ = make_loop(tmp_path / "source", [[call(1, "plan = 'teacher plan'")], [call(2, "def helper_fn():\n    return 1")], [call(3, "x")]])
    source.run()
    recorded = load_responses(source.output_dir / "responses.jsonl", limit=2)
    student = ScriptedClient([[call(9, "print(sorted(k for k in ('plan', 'helper_fn') if k in globals()))")],
                              [call(10, "import os\nprint(sorted(os.listdir('.')))")]])
    ws = tmp_path / "live" / "episode" / "workspace"
    ws.mkdir(parents=True)
    (ws / "TASK.md").write_text("task")
    (ws / "NOTES.md").write_text("")
    sandbox = Sandbox(workspace=ws.resolve())
    env = child_environment(os.environ)
    router = ToolRouter(sessions=SessionManager(sandbox=sandbox, env=env, default_cwd=ws), workspace=ws)
    repls = {"current": PythonRepl(sandbox=sandbox, env=env, python=sys.executable, startup="")}
    repls["current"].start()
    router.closers.append(repls["current"].close)
    body, items = render_request(profile_for("captain"), model="student", workspace=ws, prompt="test")
    loop = Loop(client=ReplayClient(student, recorded), body=body, history=items, router=router, profile=profile_for("captain"),
                output_dir=ws.parent / "driver", workspace=ws, prompt_cache_key="t", max_requests=4,
                python_runner=lambda s: run_python_cell(s, router=router, repl=repls["current"]),
                runtime_tool_contract=PYTHON_RUNTIME_CONTRACT_VERSION, environment_preflight=PREFLIGHT, tool_description="t")
    snapshot, initial = snapshot_workspace(ws), list(loop._initial_items)
    calls = []

    def handoff(loop_, index):
        calls.append(index)
        if index != 3:
            return
        (ws / "NOTES.md").write_text("teacher notes")   # what the earlier actor left behind
        (ws / "teacher_plan.py").write_text("x = 1")
        (ws / "images").mkdir()
        (ws / "images" / "obs_000_head.png").write_bytes(b"png")
        repls["current"].close()
        repls["current"] = PythonRepl(sandbox=sandbox, env=env, python=sys.executable, startup="")
        repls["current"].start()
        router.closers.append(repls["current"].close)
        assert restore_workspace(ws, snapshot, keep=("images/obs_000_head.png",)) == {"removed": 1, "restored": 1}
        loop_.reset_history(initial + [python_session.handoff_message([], requests=2, budget={"physical_actions_remaining": 5,
                                                                                               "tool_calls_remaining": 7})])

    loop.before_request = handoff
    result = loop.run()
    assert calls == [1, 2, 3, 4] and result.requests == 4 and result.termination == "request_cap"
    texts = [o["output"][0]["text"] if isinstance(o["output"], list) else o["output"] for o in outputs(loop)]
    assert "[]" in texts[0]                                               # the teacher's plan and helper are gone
    assert "'NOTES.md', 'TASK.md', 'images'" in texts[1] and "teacher_plan.py" not in texts[1]
    assert (ws / "NOTES.md").read_text() == ""
    # the first live request carries the initial prompt, the handoff message and nothing of the teacher's turns
    first_live = student.bodies[0]["input"]
    assert not any(item.get("type") == "custom_tool_call" and item.get("call_id") in ("call_1", "call_2") for item in first_live)
    assert "Remaining budget: 2 model requests, 5 physical actions, 7 tool calls" in json.dumps(first_live)
    requests = [json.loads(line) for line in (loop.output_dir / "requests.jsonl").read_text().splitlines()]
    assert [r["replayed"] for r in requests] == [True, True, False, False]


def test_restore_workspace_removes_new_files_and_empty_dirs_and_restores_changes(tmp_path):
    from captain_harness.server.episode_launch import restore_workspace, snapshot_workspace

    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "initial_head.png").write_bytes(b"a")
    (tmp_path / "TASK.md").write_text("task")
    snapshot = snapshot_workspace(tmp_path)
    (tmp_path / "TASK.md").write_text("edited")
    (tmp_path / "images" / "initial_head.png").unlink()
    (tmp_path / "scratch" / "deep").mkdir(parents=True)
    (tmp_path / "scratch" / "deep" / "f.py").write_text("x")
    (tmp_path / "images" / "obs_003_head.png").write_bytes(b"b")
    assert restore_workspace(tmp_path, snapshot, keep=("images/obs_003_head.png",)) == {"removed": 1, "restored": 2}
    assert snapshot_workspace(tmp_path) == {**snapshot, "images/obs_003_head.png": b"b"}
    assert not (tmp_path / "scratch").exists()


STUDENT_PYTHON = Path(os.environ.get("CAPTAIN_STUDENT_PYTHON", "/nonexistent/student-py311/bin/python"))


@pytest.mark.skipif(not STUDENT_PYTHON.exists(), reason="the pinned student interpreter is not on this host")
def test_student_runtime_probe_runs_the_declared_libraries_and_the_sdk_in_the_real_session(tmp_path):
    import shutil
    from captain_harness.react.toolchain_probe import probe_student_runtime
    from captain_harness.server import python_session

    ws = tmp_path / "ws"
    ws.mkdir()
    shutil.copyfile(Path(python_session.__file__).with_name("robot.py"), ws / "robot.py")
    receipt = probe_student_runtime(Sandbox(workspace=ws.resolve()), child_environment(os.environ), str(STUDENT_PYTHON),
                                    startup=python_session.OFFLINE_PROBE_STARTUP, cells=python_session.STUDENT_RUNTIME_CELLS)
    assert receipt["ok"], receipt
    assert receipt["versions"].startswith("('3.11.16', '1.26.4', '1.13.1', '10.4.0')")
    assert sorted(ws.iterdir()) == [ws / "robot.py"]


def test_driver_can_close_the_audit_of_a_loop_blocked_at_the_deadline_once(tmp_path):
    loop, router = make_loop(tmp_path, [[call(1, "x = 1")]])
    audit_rows = lambda: [json.loads(line) for line in (loop.output_dir / "tool_audit.jsonl").read_text().splitlines()]  # noqa: E731
    router.finish_audit(complete=True)                     # what react_episode does when the loop thread is still alive
    router.finish_audit(complete=False)                    # the loop's own finally, reached later: a no-op
    footer = audit_rows()[-1]
    assert footer["type"] == "footer" and footer["complete"] is True and len([r for r in audit_rows() if r["type"] == "footer"]) == 1
    with pytest.raises(RuntimeError):
        router.audit.write({"type": "outer_start"})       # a late tool call cannot append after the footer
    busy, busy_router = make_loop(tmp_path / "busy", [[call(1, "x = 1")]])
    busy_router.audit.start_outer({"outer_call_id": "c1", "request_index": 1, "tool_name": "python", "call_kind": "custom_tool_call"})
    busy_router.finish_audit(complete=True)
    assert [json.loads(line) for line in (busy.output_dir / "tool_audit.jsonl").read_text().splitlines()][-1]["complete"] is False

"""The opt-in runtime contract audits execution independently of model output."""
from __future__ import annotations

import json
import hashlib
import base64
import io
import os
import shlex
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from captain_harness.react.apply_patch import apply_patch
from captain_harness.react.catalog import profile_for
from captain_harness.react.chat_llm import convert_tools
from captain_harness.react.code_mode import RUNTIME_CONTRACT_VERSION, node_binary, run_exec_script, runtime_tool_contract
from captain_harness.react.llm import ResponseResult
from captain_harness.react.loop import Loop
from captain_harness.react.request import render_request
from captain_harness.react.sandbox import Sandbox
from captain_harness.react.tools import ExecResult, ToolRouter
from captain_harness.react.unified_exec import SessionManager, child_environment

pytestmark = pytest.mark.skipif(node_binary() is None, reason="node not installed")


class ScriptedClient:
    def __init__(self, scripts):
        self.scripts = iter(scripts)
        self.requests = 0

    def respond(self, body):
        self.requests += 1
        return ResponseResult(response_id=f"response_{self.requests}", items=[{
            "type": "custom_tool_call", "call_id": f"call_{self.requests}", "name": "exec", "input": next(self.scripts),
        }], usage={"input_tokens": 100, "output_tokens": 20}, status="completed", latency_s=0, attempts=1)


def make_loop(tmp_path, scripts, *, sandbox=None):
    ws = tmp_path / "episode" / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    sandbox = sandbox or Sandbox(workspace=ws)
    sessions = SessionManager(sandbox=sandbox, env=child_environment(os.environ), default_cwd=ws)
    router = ToolRouter(sessions=sessions, workspace=ws, apply_patch=lambda patch: apply_patch(patch, ws))
    body, items = render_request(profile_for("captain"), model="student", workspace=ws, prompt="test")
    loop = Loop(client=ScriptedClient(scripts), body=body, history=items, router=router, profile=profile_for("captain"),
                output_dir=ws.parent / "driver", workspace=ws, prompt_cache_key="test", max_requests=len(scripts),
                code_mode_runner=lambda source: run_exec_script(source, router=router, sandbox=sandbox, env=child_environment(os.environ)),
                runtime_tool_contract=RUNTIME_CONTRACT_VERSION,
                environment_preflight={"schema_version": "harness-toolchain-preflight/v1", "ok": True, "fingerprint": "test"})
    return loop, router


def events(loop):
    return [json.loads(line) for line in (loop.output_dir / "tool_audit.jsonl").read_text().splitlines()]


def model_outputs(loop):
    return [item["output"] for item in loop.history if item.get("type") == "custom_tool_call_output"]


def test_runtime_schema_is_explicit_accurate_and_hashes_actual_tools():
    body, items = render_request(profile_for("captain"), model="student", workspace="/workspace", prompt="test")
    original = json.dumps([body, items])
    adapted_body, adapted_items, contract = runtime_tool_contract(body, items, RUNTIME_CONTRACT_VERSION)
    assert json.dumps([body, items]) == original
    tools, _ = convert_tools(adapted_body, adapted_items, allow=None)
    assert [tool["function"]["name"] for tool in tools] == ["exec"]
    description = tools[0]["function"]["description"]
    assert "tools.apply_patch" in description and "across exec calls" in description
    assert "tools.read_mcp_resource" not in description and "tools.create_goal" not in description
    assert "There are no deferred tools" in description and "accepted for compatibility but ignored" in description
    assert len(contract["sha256"]) == 64
    assert contract["sha256"] == hashlib.sha256(json.dumps(tools, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    body2, items2 = render_request(profile_for("captain"), model="student", workspace="/different", prompt="different")
    assert runtime_tool_contract(body2, items2, RUNTIME_CONTRACT_VERSION)[2] == contract


def test_store_patch_and_tool_descriptions_work_across_calls(tmp_path):
    patch = "*** Begin Patch\n*** Add File: saved.txt\n+saved\n*** End Patch"
    loop, router = make_loop(tmp_path, [
        'store("k", {value: 42}); text(await tools.apply_patch(' + json.dumps(patch) + '));',
        'text(load("k")); text(ALL_TOOLS.filter(x => /shell command/.test(x.description)).map(x => x.name));',
    ])
    loop.run()
    assert (router.workspace / "saved.txt").read_text() == "saved\n"
    output = json.dumps(model_outputs(loop))
    assert '\\"value\\":42' in output and 'exec_command' in output
    audit = events(loop)
    assert audit[-1]["complete"] is True
    assert audit[-1]["outer_started"] == audit[-1]["outer_finished"] == 2
    assert audit[0]["contract_sha256"] == loop.tool_contract["sha256"]


def test_audit_captures_silent_and_truncated_shell_failures(tmp_path):
    source = ('await tools.exec_command({cmd:"printf silent; exit 3", login:false});'
              'text(await tools.exec_command(' + json.dumps({"cmd": "python3 -c 'print(\"x\"*2000)'; exit 4", "login": False, "max_output_tokens": 1}) + '));')
    loop, _ = make_loop(tmp_path, [source])
    loop.run()
    audit = events(loop)
    ended = [event for event in audit if event["type"] == "tool_end"]
    assert [event["result"]["exit_code"] for event in ended] == [3, 4]
    assert ended[0]["result"]["output"] == "silent"
    assert ended[1]["result"]["output"] == "x" * 2000 + "\n"
    assert ended[0]["command_id"] != ended[1]["command_id"]
    assert "x" * 2000 not in json.dumps(model_outputs(loop))
    assert "raw_output" not in json.dumps(model_outputs(loop))
    assert audit[-1]["complete"] is True and audit[-1]["tool_started"] == 2
    assert {event["parent_call_id"] for event in ended} == {"call_1"}
    assert {event["request_index"] for event in ended} == {1}


def test_poll_final_failure_keeps_original_command_identity(tmp_path):
    loop, _ = make_loop(tmp_path, [
        'const r=await tools.exec_command({cmd:"sleep 0.4; printf final; exit 7",login:false,yield_time_ms:250}); store("sid",r.session_id);',
        'await tools.write_stdin({session_id:load("sid"),chars:"",yield_time_ms:5000});',
    ])
    loop.run()
    ended = [event for event in events(loop) if event["type"] == "tool_end"]
    assert [event["result"]["exit_code"] for event in ended] == [None, 7]
    assert ended[0]["command_id"] == ended[1]["command_id"]
    assert ended[1]["result"]["output"] == "final"


def test_unpolled_natural_completion_is_audited_once(tmp_path):
    loop, _ = make_loop(tmp_path, [
        'await tools.exec_command({cmd:"sleep 0.35; printf final; exit 9",login:false,yield_time_ms:250});'
        'await new Promise(resolve=>setTimeout(resolve,250));',
    ])
    loop.run()
    final = next(event for event in events(loop) if event["type"] == "session_end")
    assert final["status"] == "completed" and final["exit_code"] == 9 and final["output"] == "final"
    assert events(loop)[-1]["unresolved_sessions"] == []


def test_host_cleanup_is_not_a_model_process_failure(tmp_path):
    loop, _ = make_loop(tmp_path, ['await tools.exec_command({cmd:"sleep 30",login:false,yield_time_ms:250});'])
    loop.run()
    final = next(event for event in events(loop) if event["type"] == "session_end")
    assert final["status"] == "cancelled"
    assert events(loop)[-1]["complete"] is True


def test_outer_error_and_nested_failure_are_both_recorded(tmp_path):
    loop, _ = make_loop(tmp_path, ['await tools.exec_command({cmd:"exit 5",login:false}); throw new Error("independent JS error");'])
    loop.run()
    audit = events(loop)
    assert next(event for event in audit if event["type"] == "outer_end")["error"] == "independent JS error"
    assert next(event for event in audit if event["type"] == "tool_end")["result"]["exit_code"] == 5
    assert audit[-1]["complete"] is True


@pytest.mark.parametrize("source", ['yield_control(); text("done");', '// @exec: {"yield_time_ms":1}\nnotify("before"); await new Promise(r=>setTimeout(r,20)); text("done");'])
def test_yield_compatibility_is_synchronous_and_does_not_break_old_calls(tmp_path, source):
    loop, _ = make_loop(tmp_path, [source])
    loop.run()
    end = next(event for event in events(loop) if event["type"] == "outer_end")
    assert end["status"] == "completed" and end["error"] is None
    assert "done" in json.dumps(model_outputs(loop))


def test_failed_audit_write_cannot_produce_complete_footer(tmp_path, monkeypatch):
    loop, router = make_loop(tmp_path, ['await tools.exec_command({cmd:"exit 0",login:false});'])
    original = router.audit.end_tool

    def fail(event):
        raise OSError("audit disk failure")

    monkeypatch.setattr(router.audit, "end_tool", fail)
    loop.run()
    assert events(loop)[-1]["complete"] is False
    monkeypatch.setattr(router.audit, "end_tool", original)


def test_private_audit_requires_hidden_sandbox_path(tmp_path):
    ws = tmp_path / "episode" / "workspace"
    with pytest.raises(ValueError, match="not hidden"):
        make_loop(tmp_path, ['text(1);'], sandbox=Sandbox(workspace=ws, bwrap="unused-bwrap"))


def test_host_preflight_path_is_not_passed_to_model_processes():
    env = child_environment({"PATH": "/usr/bin", "FRESH_ENV_PREFLIGHT_PATH": "/private/receipt.json"})
    assert "FRESH_ENV_PREFLIGHT_PATH" not in env


def test_raw_output_is_private_even_without_audit(tmp_path):
    sandbox = Sandbox(workspace=tmp_path)
    sessions = SessionManager(sandbox=sandbox, env=child_environment(os.environ), default_cwd=tmp_path)
    try:
        result = sessions.exec_command(cmd="printf abcdefghijklmnop", login=False, max_output_tokens=1)
        assert result.raw_output == "abcdefghijklmnop"
        assert "raw_output" not in result.as_object() and "audit_session_id" not in result.as_object()
        assert "abcdefghijklmnop" not in result.as_text()
    finally:
        sessions.close_all()


@pytest.mark.parametrize("background", [False, True])
def test_missing_required_executable_is_machine_verified(tmp_path, monkeypatch, background):
    from captain_harness.react import toolchain_probe

    probes = []

    def probe(sessions, name, **kwargs):
        probes.append(name)
        return {"ok": name != "rg", "executable": name, "exit_code": 127 if name == "rg" else 0}

    monkeypatch.setattr(toolchain_probe, "probe_executable", probe)
    source = ('await tools.exec_command({cmd:"sleep 0.35; exit 127",login:false,yield_time_ms:250});'
              'await new Promise(r=>setTimeout(r,250));') if background else 'await tools.exec_command({cmd:"exit 127",login:false});'
    loop, router = make_loop(tmp_path, [source])
    loop.run()
    assert "rg" in probes and router.environment_errors[0]["verified"] is True
    kind = "session_end" if background else "tool_end"
    event = next(event for event in events(loop) if event["type"] == kind)
    assert event["error_class"] == "environment" and event["environment_error"]["executable"] == "rg"


def test_forged_command_not_found_text_is_not_environment_evidence(tmp_path, monkeypatch):
    from captain_harness.react import toolchain_probe

    def probe(*args, **kwargs):
        raise AssertionError("output text must not trigger environment classification")

    monkeypatch.setattr(toolchain_probe, "probe_executable", probe)
    loop, router = make_loop(tmp_path, ['await tools.exec_command({cmd:"printf \'rg: command not found\'; exit 1",login:false});'])
    loop.run()
    assert not router.environment_errors
    event = next(event for event in events(loop) if event["type"] == "tool_end")
    assert event["error_class"] == "policy"


def test_original_execution_exception_has_a_complete_error_record(tmp_path, monkeypatch):
    loop, router = make_loop(tmp_path, ['await tools.exec_command({cmd:"unused"});'])

    def fail(**kwargs):
        raise OSError("original execution failure")

    monkeypatch.setattr(router.sessions, "exec_command", fail)
    loop.run()
    event = next(event for event in events(loop) if event["type"] == "tool_end")
    assert "original execution failure" in event["error"]
    assert "untruncated" not in event["error"] and events(loop)[-1]["complete"] is True


def test_missing_raw_output_fails_audit_closed(tmp_path, monkeypatch):
    loop, router = make_loop(tmp_path, ['await tools.exec_command({cmd:"unused"});'])
    monkeypatch.setattr(router.sessions, "exec_command", lambda **kwargs: ExecResult("chunk", 0, "truncated", 100, exit_code=0))
    loop.run()
    assert not any(event["type"] == "tool_end" for event in events(loop))
    assert events(loop)[-1]["complete"] is False


def test_invalid_nested_args_remain_policy_errors(tmp_path):
    loop, _ = make_loop(tmp_path, ['await tools.exec_command(42);'])
    loop.run()
    event = next(event for event in events(loop) if event["type"] == "tool_end")
    assert event["args"] == 42 and event["error_class"] == "policy"
    assert "expected a JSON object" in event["error"] and events(loop)[-1]["complete"] is True


def test_parallel_nested_calls_have_independent_execution_records(tmp_path):
    loop, _ = make_loop(tmp_path, ['await Promise.all([tools.exec_command({cmd:"exit 2",login:false}),tools.exec_command({cmd:"exit 3",login:false})]);'])
    loop.run()
    ended = [event for event in events(loop) if event["type"] == "tool_end"]
    assert sorted(event["result"]["exit_code"] for event in ended) == [2, 3]
    assert len({event["tool_call_id"] for event in ended}) == 2 and events(loop)[-1]["complete"] is True


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")
def test_shell_cannot_read_private_tool_audit(tmp_path):
    ws = tmp_path / "episode" / "workspace"
    sandbox = Sandbox.detect(ws, hidden_dirs=[ws.parent])
    audit_path = ws.parent / "driver" / "tool_audit.jsonl"
    source = 'text(await tools.exec_command(' + json.dumps({"cmd": "test ! -e " + shlex.quote(str(audit_path)), "login": False}) + '));'
    loop, _ = make_loop(tmp_path, [source], sandbox=sandbox)
    loop.run()
    end = next(event for event in events(loop) if event["type"] == "tool_end")
    assert end["result"]["exit_code"] == 0 and audit_path.is_file()
    assert audit_path.stat().st_mode & 0o777 == 0o600


def test_external_images_are_policy_errors_without_image_prompt_parts(tmp_path):
    loop, _ = make_loop(tmp_path, ['image("https://unresolvable.invalid/forbidden.png");'])
    loop.run()
    assert not any(part.get("type") == "input_image" for output in model_outputs(loop) for part in output)
    end = next(event for event in events(loop) if event["type"] == "outer_end")
    assert end["status"] == "failed" and "external images are unavailable" in end["error"]
    assert events(loop)[-1]["complete"] is True


def test_python_image_validation_rejects_external_runner_payload(tmp_path, monkeypatch):
    from captain_harness.react import code_mode

    loop, _ = make_loop(tmp_path, ['text("unused");'])
    payload = {"status": "completed", "error": None, "parts": [{"type": "input_image", "image_url": "https://unresolvable.invalid/a.png"}]}
    monkeypatch.setattr(code_mode.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(payload).encode(), stderr=b"", returncode=0))
    loop.run()
    output = model_outputs(loop)[0]
    assert not any(part.get("type") == "input_image" for part in output)
    assert output[0]["text"].startswith("Script failed") and "external images" in output[-1]["text"]


def test_valid_inline_data_image_remains_supported(tmp_path):
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    loop, _ = make_loop(tmp_path, ['image(' + json.dumps(url) + ');'])
    loop.run()
    assert any(part.get("image_url") == url for part in model_outputs(loop)[0])
    assert next(event for event in events(loop) if event["type"] == "outer_end")["status"] == "completed"


def test_corrupt_image_with_valid_signature_is_a_policy_error(tmp_path):
    url = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8).decode("ascii")
    loop, _ = make_loop(tmp_path, ['image(' + json.dumps(url) + ');'])
    loop.run()
    assert not any(part.get("type") == "input_image" for part in model_outputs(loop)[0])
    end = next(event for event in events(loop) if event["type"] == "outer_end")
    assert end["status"] == "failed" and "invalid image data" in end["error"]


def test_corrupt_local_image_does_not_reach_model_image_parts(tmp_path):
    loop, router = make_loop(tmp_path, ['image(await tools.view_image({path:"broken.png"}));'])
    (router.workspace / "broken.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
    loop.run()
    assert not any(part.get("type") == "input_image" for part in model_outputs(loop)[0])
    event = next(event for event in events(loop) if event["type"] == "tool_end")
    assert event["error_class"] == "policy" and "invalid image data" in event["error"]


@pytest.mark.parametrize("source,error", [
    ('setTimeout(()=>{throw new Error("late timer error")},0);', "late timer error"),
    ('Promise.reject(new Error("late promise error"));', "late promise error"),
])
def test_real_node_late_async_failure_overrides_completed_payload(tmp_path, source, error):
    loop, _ = make_loop(tmp_path, [source])
    loop.run()
    audit = events(loop)
    end = next(event for event in audit if event["type"] == "outer_end")
    assert end["status"] == "failed"
    assert "code-mode runner exited with code 1" in end["error"] and error in end["error"]
    assert model_outputs(loop)[0][0]["text"].startswith("Script failed")
    assert audit[-1]["complete"] is True and audit[-1]["tool_started"] == 0


def test_late_failure_preserves_executed_nested_audit_and_output(tmp_path):
    source = ('text(await tools.exec_command({cmd:"printf executed",login:false}));'
              'store("saved",42); Promise.reject(new Error("after nested execution"));')
    loop, router = make_loop(tmp_path, [source])
    loop.run()
    audit = events(loop)
    tool = next(event for event in audit if event["type"] == "tool_end")
    assert tool["result"]["exit_code"] == 0 and tool["result"]["output"] == "executed"
    assert tool["error"] is None
    outer = next(event for event in audit if event["type"] == "outer_end")
    assert outer["status"] == "failed" and "after nested execution" in outer["error"]
    assert "executed" in json.dumps(model_outputs(loop)) and router.code_mode_store == {"saved": 42}
    assert audit[-1]["complete"] is True and audit[-1]["tool_finished"] == 1


@pytest.mark.parametrize("returncode", [1, -9])
def test_nonzero_runner_exit_with_complete_json_is_not_success(tmp_path, monkeypatch, returncode):
    from captain_harness.react import code_mode

    loop, _ = make_loop(tmp_path, ['text("unused");'])
    payload = {"status": "completed", "error": None, "parts": [{"type": "input_text", "text": "completed output"}], "store": {}}
    monkeypatch.setattr(code_mode.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        stdout=json.dumps(payload).encode(), stderr=b"late failure detail", returncode=returncode))
    loop.run()
    outer = next(event for event in events(loop) if event["type"] == "outer_end")
    assert outer["status"] == "failed" and f"exited with code {returncode}" in outer["error"]
    assert "late failure detail" in outer["error"] and "completed output" in json.dumps(model_outputs(loop))


def test_legacy_runner_result_handling_is_unchanged_for_late_error(tmp_path):
    router = ToolRouter(sessions=None, workspace=tmp_path)
    parts = run_exec_script('Promise.reject(new Error("legacy late error"));', router=router, sandbox=Sandbox(workspace=tmp_path))
    assert parts[0]["text"].startswith("Script completed")

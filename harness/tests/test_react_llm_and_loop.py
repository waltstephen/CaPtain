"""Responses client against a fake SSE server; the bare loop against scripted responses."""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from captain_harness.react.catalog import profile_for
from captain_harness.react.llm import ContextWindowExceeded, ResponsesClient
from captain_harness.react.loop import Loop
from captain_harness.react.sandbox import Sandbox
from captain_harness.react.tools import ToolRouter
from captain_harness.react.unified_exec import SessionManager, child_environment


def _sse(events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


class FakeResponses:
    """Scripted Responses endpoint: each POST pops the next scripted reply."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        handler_self = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                return

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                handler_self.requests.append(body)
                status, payload, headers = handler_self.replies.pop(0)
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/openai/responses"


def completed(items, usage=None, rid="resp_1"):
    usage = usage or {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 4}, "output_tokens": 5, "output_tokens_details": {"reasoning_tokens": 1}, "total_tokens": 15}
    events = [{"type": "response.created", "response": {"id": rid}}]
    events += [{"type": "response.output_item.done", "item": it} for it in items]
    events.append({"type": "response.completed", "response": {"id": rid, "status": "completed", "output": items, "usage": usage}})
    return 200, _sse(events), {"Content-Type": "text/event-stream"}


def message(text, role="assistant"):
    return {"type": "message", "role": role, "status": "completed", "content": [{"type": "output_text", "text": text}]}


def test_client_streams_items_and_retries_429():
    fake = FakeResponses([
        (429, json.dumps({"error": {"code": "rate_limit_exceeded", "message": "slow down. Please try again in 10ms."}}).encode(), {"Content-Type": "application/json"}),
        completed([{"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "abc"}, message("OK")]),
    ])
    client = ResponsesClient(fake.url, auth="none")
    result = client.respond({"model": "m", "input": [], "store": False})
    assert result.attempts == 2 and result.response_id == "resp_1" and [i["type"] for i in result.items] == ["reasoning", "message"]
    assert result.usage["total_tokens"] == 15
    assert fake.requests[-1]["stream"] is True and fake.requests[-1]["store"] is False


def test_client_raises_on_context_length():
    fake = FakeResponses([(400, json.dumps({"error": {"code": "context_length_exceeded", "message": "too long"}}).encode(), {"Content-Type": "application/json"})])
    with pytest.raises(ContextWindowExceeded):
        ResponsesClient(fake.url, auth="none").respond({"model": "m", "input": []})


class ScriptedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.bodies = []

    def respond(self, body):
        self.bodies.append(json.loads(json.dumps(body)))
        items, usage = self.responses.pop(0)
        from captain_harness.react.llm import ResponseResult
        return ResponseResult(response_id=f"resp_{len(self.bodies)}", items=items, usage=usage, status="completed", latency_s=0.01, attempts=1)


def _usage(total):
    return {"input_tokens": total - 5, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 5, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": total}


@pytest.fixture
def harness(tmp_path):
    ws = tmp_path / "ep" / "workspace"
    ws.mkdir(parents=True)
    sandbox = Sandbox(workspace=ws, bwrap=None)
    sessions = SessionManager(sandbox=sandbox, env=child_environment(os.environ), default_cwd=ws)
    router = ToolRouter(sessions=sessions, workspace=ws)
    yield ws, router
    sessions.close_all()


def test_loop_runs_parallel_function_calls_in_order_and_stops_on_final(harness, tmp_path):
    ws, router = harness
    profile = profile_for("gpt-5.5")
    calls = [
        {"type": "function_call", "call_id": "c1", "name": "exec_command", "arguments": json.dumps({"cmd": "sleep 0.3; echo one"})},
        {"type": "function_call", "call_id": "c2", "name": "exec_command", "arguments": json.dumps({"cmd": "echo two"})},
        {"type": "function_call", "call_id": "c3", "name": "view_image", "arguments": json.dumps({"path": "missing.png"})},
    ]
    client = ScriptedClient([
        ([{"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "enc"}, message("working"), *calls], _usage(100)),
        ([message("done")], _usage(200)),
    ])
    loop = Loop(client=client, body={"model": "gpt-5.5", "store": False}, history=[message("go", role="user")], router=router,
                profile=profile, output_dir=tmp_path / "ep" / "driver", prompt_cache_key="k1")
    result = loop.run()
    assert result.termination == "completed" and result.requests == 2 and result.tool_calls == 3 and result.final_message == "done"
    second = client.bodies[1]["input"]
    types = [i["type"] for i in second]
    assert types == ["message", "reasoning", "message", "function_call", "function_call", "function_call", "function_call_output", "function_call_output", "function_call_output"]
    outputs = second[6:]
    assert [o["call_id"] for o in outputs] == ["c1", "c2", "c3"]
    assert outputs[0]["output"].endswith("Output:\none\n") and outputs[1]["output"].endswith("Output:\ntwo\n")
    assert outputs[2]["output"].startswith("unable to locate image")
    assert second[1]["encrypted_content"] == "enc" and client.bodies[1]["prompt_cache_key"] == "k1" and client.bodies[1]["store"] is False
    events = [json.loads(l) for l in (tmp_path / "ep" / "driver" / "events.jsonl").read_text().splitlines()]
    kinds = [e["type"] for e in events]
    assert kinds[0] == "thread.started" and kinds[1] == "turn.started" and kinds[-1] == "turn.completed"
    assert sum(1 for e in events if e["type"] == "item.completed" and e["item"]["type"] == "command_execution") == 2
    assert result.usage_total["input_tokens"] == 95 + 195
    transcript = (tmp_path / "ep" / "driver" / "transcript.jsonl").read_text().splitlines()
    assert len(transcript) == len(second) + 1


def test_loop_code_mode_uses_runner_and_context_guard(harness, tmp_path):
    ws, router = harness
    profile = profile_for("captain")
    client = ScriptedClient([
        ([{"type": "custom_tool_call", "call_id": "x1", "name": "exec", "input": "text('hi')"}], _usage(900_000)),
    ])
    seen = []

    def runner(source):
        seen.append(source)
        return [{"type": "input_text", "text": "Script completed\nWall time 0.1 seconds\nOutput:\n"}, {"type": "input_text", "text": "hi"}]

    loop = Loop(client=client, body={"model": "captain-4b"}, history=[], router=router, profile=profile,
                output_dir=tmp_path / "ep" / "driver", prompt_cache_key="k2", code_mode_runner=runner, compaction=False)
    result = loop.run()
    assert seen == ["text('hi')"] and result.termination == "context_exhausted" and result.tool_calls == 1
    output_item = loop.history[-1]
    assert output_item["type"] == "custom_tool_call_output" and output_item["call_id"] == "x1" and output_item["output"][1]["text"] == "hi"


def test_loop_stops_at_deadline(harness, tmp_path):
    ws, router = harness
    client = ScriptedClient([([message("never")], _usage(10))])
    loop = Loop(client=client, body={}, history=[], router=router, profile=profile_for("gpt-5.5"), output_dir=tmp_path / "d",
                prompt_cache_key="k", deadline_monotonic=0.0)
    assert loop.run().termination == "deadline" and client.bodies == []

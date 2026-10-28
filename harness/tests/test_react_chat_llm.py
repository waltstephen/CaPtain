"""Chat Completions backend: Responses items <-> chat messages, and the bare loop against a fake vLLM."""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from captain_harness.react.catalog import profile_for, with_context_window
from captain_harness.react.chat_llm import (
    CUSTOM_INPUT_PARAMETER,
    HOISTED_IMAGES_NOTE,
    INCOMPLETE_NUDGE_TEXT,
    ChatCompletionsClient,
    cap_images,
    chat_message_to_items,
    convert_tools,
    looks_like_unparsed_tool_call,
    responses_to_chat_messages,
)
from captain_harness.react.llm import ContextWindowExceeded, ModelRequestError
from captain_harness.react.loop import Loop
from captain_harness.react.request import render_request
from captain_harness.react.sandbox import Sandbox
from captain_harness.react.tools import ToolRouter
from captain_harness.react.unified_exec import SessionManager, child_environment

PNG = "data:image/png;base64,iVBORw0KGgo="


def _user(text):
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def _developer(text):
    return {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": text}]}


def _assistant(text):
    return {"type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": text}]}


# --------------------------------------------------------------------------- translation
def test_history_translation_groups_one_response_into_one_assistant_message():
    items = [
        _developer("perm"), _user("env"), _user("go"),
        {"type": "reasoning", "id": "rs1", "summary": [{"type": "summary_text", "text": "think"}], "encrypted_content": "x"},
        _assistant("I will look."),
        {"type": "function_call", "call_id": "c1", "name": "exec_command", "arguments": json.dumps({"cmd": "ls"})},
        {"type": "function_call", "call_id": "c2", "name": "view_image", "arguments": json.dumps({"path": "a.png"})},
        {"type": "function_call_output", "call_id": "c1", "output": "Chunk ID: 1\nOutput:\nfile\n"},
        {"type": "function_call_output", "call_id": "c2", "output": [{"type": "input_image", "image_url": PNG, "detail": "high"}]},
        {"type": "reasoning", "id": "rs2", "summary": [], "encrypted_content": "y"},
        _assistant("done"),
    ]
    messages = responses_to_chat_messages("BASE", items)
    assert [m["role"] for m in messages] == ["system", "user", "user", "assistant", "tool", "tool", "assistant"]
    assert messages[0]["content"] == "BASE\n\nperm"
    assistant = messages[3]
    assert assistant["content"] == "I will look." and assistant["reasoning_content"] == "think"
    assert [c["function"]["name"] for c in assistant["tool_calls"]] == ["exec_command", "view_image"]
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"cmd": "ls"}
    assert messages[4] == {"role": "tool", "tool_call_id": "c1", "content": "Chunk ID: 1\nOutput:\nfile\n"}
    assert messages[5]["content"] == [{"type": "image_url", "image_url": {"url": PNG}}]
    assert messages[6] == {"role": "assistant", "content": "done"}  # empty summary -> no reasoning_content
    dropped = responses_to_chat_messages("BASE", items, history_reasoning=False)
    assert "reasoning_content" not in dropped[3]


def test_hoisting_moves_tool_images_into_one_following_user_message():
    items = [_user("go"),
             {"type": "function_call", "call_id": "c1", "name": "view_image", "arguments": "{}"},
             {"type": "function_call", "call_id": "c2", "name": "exec_command", "arguments": "{}"},
             {"type": "function_call", "call_id": "c3", "name": "view_image", "arguments": "{}"},
             {"type": "function_call_output", "call_id": "c1", "output": [{"type": "input_image", "image_url": PNG, "detail": "high"}]},
             {"type": "function_call_output", "call_id": "c2", "output": "text"},
             {"type": "custom_tool_call_output", "call_id": "c3", "output": [{"type": "input_text", "text": "Script completed\n"}, {"type": "input_image", "image_url": PNG}, {"type": "input_image", "image_url": PNG}]},
             _assistant("done")]
    messages = responses_to_chat_messages(None, items, hoist_tool_images=True)
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "tool", "tool", "user", "assistant"]
    assert messages[2]["content"] == HOISTED_IMAGES_NOTE.format(n=1)
    assert messages[4]["content"] == [{"type": "text", "text": "Script completed\n"}, {"type": "text", "text": HOISTED_IMAGES_NOTE.format(n=2)}]
    hoisted = messages[5]["content"]
    assert hoisted[0]["text"] == "Images returned by tool call(s) c1, c3:" and [p["type"] for p in hoisted[1:]] == ["image_url"] * 3
    # default: images stay in the tool messages
    inline = responses_to_chat_messages(None, items)
    assert [m["role"] for m in inline] == ["user", "assistant", "tool", "tool", "tool", "assistant"]
    assert inline[2]["content"][0]["type"] == "image_url"


def test_custom_tool_calls_round_trip_through_the_input_parameter():
    items = [_user("go"), {"type": "custom_tool_call", "call_id": "x1", "name": "exec", "input": "text('hi')"},
             {"type": "custom_tool_call_output", "call_id": "x1", "output": [{"type": "input_text", "text": "Script completed\nOutput:\n"}, {"type": "input_text", "text": "hi"}]}]
    messages = responses_to_chat_messages(None, items)
    call = messages[1]["tool_calls"][0]
    assert call["function"]["name"] == "exec" and json.loads(call["function"]["arguments"]) == {CUSTOM_INPUT_PARAMETER: "text('hi')"}
    assert messages[2]["content"] == [{"type": "text", "text": "Script completed\nOutput:\n"}, {"type": "text", "text": "hi"}]
    back = chat_message_to_items({"role": "assistant", "content": None, "reasoning": "r",
                                  "tool_calls": [{"id": "call_a", "type": "function", "function": {"name": "exec", "arguments": json.dumps({"input": "text(1)"})}},
                                                 {"id": "call_b", "type": "function", "function": {"name": "exec_command", "arguments": json.dumps({"cmd": "ls"})}}]},
                                 custom_names={"exec", "apply_patch"})
    assert [i["type"] for i in back] == ["reasoning", "custom_tool_call", "function_call"]
    assert back[0]["summary"][0]["text"] == "r"
    assert back[1]["input"] == "text(1)" and back[1]["call_id"] == "call_a"
    assert json.loads(back[2]["arguments"]) == {"cmd": "ls"}


def test_tools_are_converted_and_filtered_to_what_the_driver_implements():
    body, items = render_request(profile_for("gpt-5.5"), model="m", workspace="/w", prompt="p")
    tools, custom = convert_tools(body, items)
    assert [t["function"]["name"] for t in tools] == ["exec_command", "write_stdin", "apply_patch", "view_image"]
    assert custom == {"apply_patch"}
    patch = next(t for t in tools if t["function"]["name"] == "apply_patch")
    assert patch["function"]["parameters"]["required"] == [CUSTOM_INPUT_PARAMETER] and "*** Begin Patch" in patch["function"]["description"]
    every, _ = convert_tools(body, items, allow=None)
    assert len(every) > len(tools) and all(t["type"] == "function" for t in every)  # web_search / tool_search dropped, MCP stubs kept
    body, items = render_request(profile_for("captain"), model="m", workspace="/w", prompt="p")
    tools, custom = convert_tools(body, items)
    assert [t["function"]["name"] for t in tools] == ["exec"] and custom == {"exec"}
    assert "await tools.exec_command" in tools[0]["function"]["description"]


def test_cap_images_keeps_the_most_recent_ones():
    messages = [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {"url": PNG}}]},
                {"role": "tool", "tool_call_id": "1", "content": [{"type": "image_url", "image_url": {"url": PNG}}]},
                {"role": "tool", "tool_call_id": "2", "content": [{"type": "image_url", "image_url": {"url": PNG}}]}]
    assert cap_images(messages, 2) == 1
    assert messages[0]["content"][1]["type"] == "text" and messages[1]["content"][0]["type"] == "image_url"
    assert cap_images(messages, 2) == 0 and cap_images(messages, None) == 0


def test_unparsed_tool_call_text_is_detected():
    assert looks_like_unparsed_tool_call({"content": "<tool_call>\n<function=exec_command>..."})
    assert not looks_like_unparsed_tool_call({"content": "<tool_call>", "tool_calls": [{"id": "x"}]})
    assert not looks_like_unparsed_tool_call({"content": "plain"})


# --------------------------------------------------------------------------- fake server
class FakeChat:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        owner = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                return

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append((self.path, body))
                status, payload = owner.replies.pop(0)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"


def _completion(message, finish="stop", usage=None):
    usage = usage or {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "completion_tokens_details": {"reasoning_tokens": 7}}
    return 200, {"id": "chatcmpl-1", "model": "student", "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": finish}], "usage": usage}


def test_client_posts_chat_request_and_normalises_usage():
    fake = FakeChat([_completion({"content": "hello", "reasoning": "hmm"})])
    client = ChatCompletionsClient(fake.url, max_output_tokens=512, temperature=0.6, enable_thinking=True)
    body, items = render_request(profile_for("gpt-5.5"), model="student", workspace="/w", prompt="p")
    body["input"] = items
    result = client.respond(body)
    path, request = fake.requests[0]
    assert path == "/v1/chat/completions" and request["model"] == "student" and request["stream"] is False
    assert request["messages"][0]["role"] == "system" and request["messages"][0]["content"].startswith(body["instructions"][:40])
    assert request["tool_choice"] == "auto" and request["parallel_tool_calls"] is True and request["max_tokens"] == 512
    assert request["temperature"] == 0.6 and request["chat_template_kwargs"] == {"enable_thinking": True}
    assert "prompt_cache_key" not in request and "store" not in request
    assert [i["type"] for i in result.items] == ["reasoning", "message"]
    assert result.usage["input_tokens"] == 100 and result.usage["output_tokens_details"]["reasoning_tokens"] == 7 and result.usage["total_tokens"] == 120
    assert result.response_id == "chatcmpl-1" and result.raw["finish_reason"] == "stop"


def test_client_retries_on_prompt_budget_then_raises_when_prompt_alone_exceeds():
    msg = "This model's maximum context length is 1000 tokens. However, you requested 1300 tokens (800 in the messages, 500 in the completion). Please reduce the length."
    fake = FakeChat([(400, {"error": {"message": msg, "type": "BadRequestError"}}), _completion({"content": "ok"})])
    client = ChatCompletionsClient(fake.url, max_output_tokens=500)
    client.respond({"model": "m", "input": [_user("p")]})
    assert fake.requests[0][1]["max_tokens"] == 500 and fake.requests[1][1]["max_tokens"] == 1000 - 800 - 16
    msg2 = "This model's maximum context length is 1000 tokens. However, you requested 1500 tokens (1200 in the messages, 300 in the completion)."
    fake = FakeChat([(400, {"error": {"message": msg2}})])
    with pytest.raises(ContextWindowExceeded):
        ChatCompletionsClient(fake.url).respond({"model": "m", "input": [_user("p")]})
    fake = FakeChat([(400, {"error": {"message": "At most 16 image(s) may be provided in one request."}})])
    with pytest.raises(ModelRequestError):
        ChatCompletionsClient(fake.url).respond({"model": "m", "input": [_user("p")]})


@pytest.fixture
def harness(tmp_path):
    ws = tmp_path / "ep" / "workspace"
    ws.mkdir(parents=True)
    sandbox = Sandbox(workspace=ws, bwrap=None)
    sessions = SessionManager(sandbox=sandbox, env=child_environment(os.environ), default_cwd=ws)
    router = ToolRouter(sessions=sessions, workspace=ws)
    yield ws, router
    sessions.close_all()


def test_loop_runs_end_to_end_against_a_fake_chat_server(harness, tmp_path):
    ws, router = harness
    (ws / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    fake = FakeChat([
        _completion({"content": "Listing.", "reasoning_content": "step 1",
                     "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "exec_command", "arguments": json.dumps({"cmd": "echo one"})}},
                                    {"id": "call_2", "type": "function", "function": {"name": "view_image", "arguments": json.dumps({"path": "a.png"})}}]}, finish="tool_calls"),
        _completion({"content": "<tool_call>\n<function=exec_command>\n<parameter=cmd>\nbroken\n</parameter>\n</function>\n</tool_call>", "reasoning": "step 2"}),
    ])
    events = []
    client = ChatCompletionsClient(fake.url, event_sink=events.append)
    profile = with_context_window(profile_for("Qwen/Student", fallback="gpt-5.5"), 131072)
    body, items = render_request(profile, model="Qwen/Student", workspace=ws, prompt="go")
    loop = Loop(client=client, body=body, history=items, router=router, profile=profile, output_dir=tmp_path / "ep" / "driver", prompt_cache_key="k")
    result = loop.run()
    assert result.termination == "completed" and result.requests == 2 and result.tool_calls == 2
    assert result.final_message.startswith("<tool_call>")
    second = fake.requests[1][1]["messages"]
    roles = [m["role"] for m in second]
    assert roles == ["system", "user", "user", "assistant", "tool", "tool"]
    assert second[3]["reasoning_content"] == "step 1" and [c["id"] for c in second[3]["tool_calls"]] == ["call_1", "call_2"]
    assert second[4]["tool_call_id"] == "call_1" and second[4]["content"].endswith("Output:\none\n")
    assert second[5]["content"][0]["type"] == "image_url" and second[5]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    responses = [e for e in events if e["event"] == "response"]
    assert responses[1]["unparsed_tool_call"] is True and responses[0]["unparsed_tool_call"] is False
    transcript = [json.loads(l) for l in (tmp_path / "ep" / "driver" / "transcript.jsonl").read_text().splitlines()]
    assert [t["type"] for t in transcript[-4:]] == ["function_call_output", "function_call_output", "reasoning", "message"]
    assert "chars omitted" in transcript[-3]["output"][0]["image_url"]


def test_length_truncated_reply_without_tool_call_gets_a_nudge_then_ends(harness, tmp_path):
    ws, router = harness
    fake = FakeChat([
        _completion({"content": "I am thinking out loud for a very long time"}, finish="length"),
        _completion({"content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "exec_command", "arguments": json.dumps({"cmd": "echo two"})}}]}, finish="tool_calls"),
        _completion({"content": "still rambling"}, finish="length"),
        _completion({"content": "more rambling"}, finish="length"),
    ])
    client = ChatCompletionsClient(fake.url)
    profile = with_context_window(profile_for("Qwen/Student", fallback="gpt-5.5"), 131072)
    body, items = render_request(profile, model="Qwen/Student", workspace=ws, prompt="go")
    loop = Loop(client=client, body=body, history=items, router=router, profile=profile, output_dir=tmp_path / "ep" / "driver",
                prompt_cache_key="k", incomplete_nudges=1, incomplete_nudge_text=INCOMPLETE_NUDGE_TEXT)
    result = loop.run()
    assert result.termination == "completed" and result.requests == 3 and result.tool_calls == 1
    second = fake.requests[1][1]["messages"]
    assert second[-1] == {"role": "user", "content": INCOMPLETE_NUDGE_TEXT} and second[-2]["role"] == "assistant"
    events = [json.loads(l) for l in (tmp_path / "ep" / "driver" / "events.jsonl").read_text().splitlines()]
    assert sum(1 for e in events if e["type"] == "incomplete_nudge") == 1
    assert len(fake.replies) == 1  # the second truncated reply ended the turn: only one nudge allowed


def test_transcript_annotates_viewed_images_with_their_path(harness, tmp_path):
    ws, router = harness
    (ws / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 40)
    fake = FakeChat([
        _completion({"content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "view_image", "arguments": json.dumps({"path": "a.png"})}}]}, finish="tool_calls"),
        _completion({"content": "seen"}),
    ])
    profile = with_context_window(profile_for("Qwen/Student", fallback="gpt-5.5"), 131072)
    body, items = render_request(profile, model="Qwen/Student", workspace=ws, prompt="go")
    Loop(client=ChatCompletionsClient(fake.url), body=body, history=items, router=router, profile=profile,
         output_dir=tmp_path / "ep" / "driver", prompt_cache_key="k").run()
    transcript = [json.loads(l) for l in (tmp_path / "ep" / "driver" / "transcript.jsonl").read_text().splitlines()]
    output = next(t for t in transcript if t["type"] == "function_call_output")
    assert output["output"][0]["path"] == "a.png" and "chars omitted" in output["output"][0]["image_url"]


def test_stop_when_hook_ends_the_loop_before_the_next_request(harness, tmp_path):
    ws, router = harness
    call = {"id": "call_1", "type": "function", "function": {"name": "exec_command", "arguments": json.dumps({"cmd": "true"})}}
    fake = FakeChat([_completion({"content": "", "tool_calls": [call]}, finish="tool_calls")] * 3)
    profile = with_context_window(profile_for("Qwen/Student", fallback="gpt-5.5"), 131072)
    body, items = render_request(profile, model="Qwen/Student", workspace=ws, prompt="go")
    seen = {"n": 0}

    def stop():
        seen["n"] += 1
        return seen["n"] >= 2  # first check (before request 2) says go on, second says stop

    result = Loop(client=ChatCompletionsClient(fake.url), body=body, history=items, router=router, profile=profile,
                  output_dir=tmp_path / "ep" / "driver", prompt_cache_key="k", stop_when=stop).run()
    assert result.termination == "stopped" and result.requests == 2 and result.tool_calls == 2


def test_calls_to_tools_the_request_did_not_expose_are_refused(harness, tmp_path):
    ws, router = harness
    fake = FakeChat([
        _completion({"content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "exec_command", "arguments": json.dumps({"cmd": "echo leaked"})}}]}, finish="tool_calls"),
        _completion({"content": "", "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "exec", "arguments": json.dumps({"input": "text('ok')"})}}]}, finish="tool_calls"),
        _completion({"content": "done"}),
    ])
    profile = with_context_window(profile_for("Qwen/Student", fallback="captain"), 131072)
    body, items = render_request(profile, model="Qwen/Student", workspace=ws, prompt="go")
    exposed, custom = convert_tools(body, items)
    router.allowed_tools = {t["function"]["name"] for t in exposed}
    assert router.allowed_tools == {"exec"} and custom == {"exec"}
    result = Loop(client=ChatCompletionsClient(fake.url), body=body, history=items, router=router, profile=profile,
                  output_dir=tmp_path / "ep" / "driver", prompt_cache_key="k",
                  code_mode_runner=lambda src: [{"type": "input_text", "text": "Script completed\nOutput:\n"}, {"type": "input_text", "text": "ok"}]).run()
    assert result.termination == "completed" and result.tool_calls == 2
    transcript = [json.loads(l) for l in (tmp_path / "ep" / "driver" / "transcript.jsonl").read_text().splitlines()]
    refused = next(t for t in transcript if t.get("call_id") == "call_1" and t["type"] == "function_call_output")
    assert refused["output"] == "unsupported call: tool exec_command is not available"
    ran = next(t for t in transcript if t.get("call_id") == "call_2" and t["type"] == "custom_tool_call_output")
    assert ran["output"][1]["text"] == "ok"
    assert not (ws / "leaked").exists() and "leaked" not in json.dumps(router.commands)


def test_chat_request_is_the_drivers_view_translated_one_to_one():
    """The client must not window or edit the history: a compaction notice and a spilled-output note produced by
    react.context appear verbatim in the chat messages, and every view item maps to exactly one message/tool call."""
    from captain_harness.react import context as ctx
    from captain_harness.react.chat_llm import ChatCompletionsClient

    state = ctx.CompactionState()
    dropped = [[{"type": "custom_tool_call", "call_id": "c0", "name": "exec", "input": "await tools.exec_command({cmd: 'ls'})"},
                {"type": "custom_tool_call_output", "call_id": "c0", "output": [{"type": "input_text", "text": "Script completed\nOutput:\n"}]}]]
    state.absorb(dropped)
    notice = ctx.drop_notice(state, workspace=None, summary=None)
    spilled = {"type": "custom_tool_call_output", "call_id": "c1", "output": [
        {"type": "input_text", "text": "Warning: truncated output (original token count: 9000)\n…8000 tokens truncated…\nFull output saved to .tool_output/abc.txt"}]}
    items = [{"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "SYSTEM"}]},
             {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "task"}]},
             notice,
             {"type": "custom_tool_call", "call_id": "c1", "name": "exec", "input": "await tools.exec_command({cmd: 'cat big'})"},
             spilled]
    body = {"model": "m", "input": items, "tools": [{"type": "custom", "name": "exec", "description": "js"}]}
    request, custom, stats = ChatCompletionsClient("http://x/v1", max_images=16).build_request(body)
    messages = request["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "user", "assistant", "tool"]
    assert messages[2]["content"] == ctx.text_of(notice) and messages[2]["content"].startswith(ctx.NOTICE_HEADER)
    assert "1 of your turns with 1 tool calls were removed" in messages[2]["content"] and "ls" in messages[2]["content"]
    assert messages[4]["content"] == ctx.text_of(spilled) or messages[4]["content"] == spilled["output"][0]["text"]
    assert messages[3]["tool_calls"][0]["function"]["name"] == "exec" and custom == {"exec"}
    assert stats["view_items"] == len(items)


def test_chat_parts_drop_invalid_image_urls():
    from captain_harness.react.chat_llm import _chat_parts
    parts = _chat_parts([{"type": "input_text", "text": "a"},
                         {"type": "input_image", "image_url": "rw_head.png", "detail": "high"},
                         {"type": "input_image", "image_url": "data:text/markdown;base64,QUJD", "detail": "high"},
                         {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgoAAAAAAAAAAAA=", "detail": "high"}])
    assert [p["type"] for p in parts] == ["text", "text", "text", "image_url"]
    assert parts[1]["text"].startswith("[image dropped") and parts[3]["image_url"]["url"].startswith("data:image/png")

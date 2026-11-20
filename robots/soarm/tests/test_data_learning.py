"""Data learning: forced/parallel scratchpad, round cap, merged reasoning, and the image cap that matches chat."""
from __future__ import annotations

import json

import pytest

from captain_harness.react.chat_llm import IMAGE_OMITTED_TEXT, cap_images, responses_to_chat_messages
from soarm.data_learning import ACK, PARALLEL_HINT, SCRATCHPAD_NAME, DataLearningClient, cap_response_images


def _usage(reasoning=0):
    return {"input_tokens": 50, "input_tokens_details": {"cached_tokens": 10}, "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": reasoning}, "total_tokens": 70}


def think(call_id, step):
    return {"type": "function_call", "call_id": call_id, "name": SCRATCHPAD_NAME, "arguments": json.dumps({"step": step})}


def python_call(call_id, source):
    return {"type": "custom_tool_call", "call_id": call_id, "name": "python", "input": source}


def message(text, role="assistant"):
    return {"type": "message", "role": role, "status": "completed", "content": [{"type": "output_text", "text": text}]}


class Scripted:
    """Responses transport double: pops one recorded reply per call and records the request bodies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.bodies = []

    def __call__(self, body):
        self.bodies.append(json.loads(json.dumps(body)))
        items, reasoning = self.replies.pop(0)
        return {"id": f"r{len(self.bodies)}", "output": items, "usage": _usage(reasoning), "status": "completed"}


BODY = {"model": "gpt-6-astra", "tools": [{"type": "function", "name": "python", "description": "run code",
                                           "parameters": {"type": "object"}}],
        "tool_choice": "auto", "input": [message("go", role="user")], "store": False}


def test_forced_first_call_then_acknowledged_continuation():
    inner = Scripted([([think("t1", "the bowl is left of the plate; approach from above")], 3),
                      ([python_call("p1", "robot.observe()")], 0)])
    result = DataLearningClient(inner).respond(dict(BODY))
    first, second = inner.bodies
    assert first["tool_choice"] == {"type": "function", "name": SCRATCHPAD_NAME}
    assert [t["name"] for t in first["tools"]] == ["python", SCRATCHPAD_NAME]
    assert second["tool_choice"] == "auto" and [t["name"] for t in second["tools"]] == ["python", SCRATCHPAD_NAME]
    assert [i["type"] for i in second["input"]] == ["message", "function_call", "function_call_output"]
    assert second["input"][2] == {"type": "function_call_output", "call_id": "t1", "output": ACK}
    # the scratchpad call and its ack never reach the teacher's visible reply; the steps become one reasoning item
    assert result["output"] == [{"type": "reasoning", "summary": [],
                                 "content": [{"type": "reasoning_text", "text": "the bowl is left of the plate; approach from above"}]},
                                python_call("p1", "robot.observe()")]
    assert result["usage"]["input_tokens"] == 100 and result["usage"]["output_tokens"] == 40
    assert result["usage"]["input_tokens_details"]["cached_tokens"] == 20
    assert result["usage"]["output_tokens_details"]["reasoning_tokens"] == 3
    meta = result["data_learning"]
    assert meta["steps"] == 1 and meta["visible_words"] == 10 and meta["thought"] and meta["think_first"]
    assert meta["native_reasoning_tokens"] == 3 and [c["forced"] for c in meta["calls"]] == [True, False]
    assert BODY["tools"] == [{"type": "function", "name": "python", "description": "run code", "parameters": {"type": "object"}}]


def test_round_cap_withdraws_the_scratchpad():
    inner = Scripted([([think("t1", "a")], 0), ([think("t2", "b")], 0), ([python_call("p1", "x=1")], 0)])
    result = DataLearningClient(inner, max_rounds=2).respond(dict(BODY))
    assert [t["name"] for t in inner.bodies[2]["tools"]] == ["python"]
    assert result["data_learning"]["rounds"] == 2
    assert [p["text"] for p in result["output"][0]["content"]] == ["a", "b"]
    assert [i["call_id"] for i in result["output"] if i.get("call_id")] == ["p1"]


def test_think_and_act_in_one_response_stops():
    inner = Scripted([([think("t1", "plan"), python_call("p1", "x=1")], 0)])
    result = DataLearningClient(inner).respond(dict(BODY))
    assert len(inner.bodies) == 1
    assert [i["type"] for i in result["output"]] == ["reasoning", "custom_tool_call"]
    assert result["data_learning"]["think_first"] is True


def test_parallel_mode_offers_without_forcing():
    inner = Scripted([([think("t1", "reach above the bowl"), python_call("p1", "robot.move()")], 0)])
    result = DataLearningClient(inner, mode="parallel").respond(dict(BODY))
    (body,) = inner.bodies
    assert body["tool_choice"] == "auto" and body["parallel_tool_calls"] is True
    assert [t["name"] for t in body["tools"]] == ["python", SCRATCHPAD_NAME]
    assert body["instructions"] == PARALLEL_HINT
    assert result["data_learning"]["mode"] == "parallel" and result["data_learning"]["thought"]


def test_parallel_mode_records_a_turn_without_thinking():
    inner = Scripted([([python_call("p1", "robot.observe()")], 0)])
    result = DataLearningClient(inner, mode="parallel").respond(dict(BODY))
    assert len(inner.bodies) == 1 and not result["data_learning"]["thought"]
    assert result["output"] == [python_call("p1", "robot.observe()")]


def test_steps_become_the_reasoning_content_the_sft_exporter_supervises():
    inner = Scripted([([think("t1", "look first")], 0), ([python_call("p1", "print(1)")], 0)])
    merged = DataLearningClient(inner).respond(dict(BODY))
    messages = responses_to_chat_messages(None, merged["output"])
    assistant = [m for m in messages if m["role"] == "assistant"]
    assert len(assistant) == 1 and assistant[0]["reasoning_content"] == "look first"
    assert [c["function"]["name"] for c in assistant[0]["tool_calls"]] == ["python"]


PNG = {tag: f"http://soarm-image.invalid/{tag}.png" for tag in "ABCD"}


def img(tag):
    return {"type": "input_image", "image_url": PNG[tag]}


ITEMS = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}, img("A")]},
         {"type": "function_call", "call_id": "c1", "name": "observe", "arguments": "{}"},
         {"type": "function_call_output", "call_id": "c1", "output": [{"type": "input_text", "text": "o"}, img("B"), img("C")]},
         {"type": "function_call", "call_id": "c2", "name": "observe", "arguments": "{}"},
         {"type": "function_call_output", "call_id": "c2", "output": [img("D")]}]


def test_image_cap_keeps_the_most_recent_and_matches_the_chat_cap():
    capped, n = cap_response_images(ITEMS, 2)
    assert n == 2 and ITEMS[0]["content"][1] == img("A")                    # the input is not mutated
    assert capped[0]["content"][1] == {"type": "input_text", "text": IMAGE_OMITTED_TEXT.format(n=2)}
    assert capped[2]["output"][1]["type"] == "input_text"
    assert capped[2]["output"][2] == img("C") and capped[4]["output"][0] == img("D")
    chat = responses_to_chat_messages(None, ITEMS)
    cap_images(chat, 2)
    assert responses_to_chat_messages(None, capped) == chat                 # same rendering as the chat backend's cap


def test_image_cap_rejects_a_nonpositive_cap():
    with pytest.raises(ValueError):
        cap_response_images(ITEMS, 0)

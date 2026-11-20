"""Data learning: make a Responses teacher write its working out where the record can keep it.

Why: a Responses teacher may answer a tool-calling turn without returning its reasoning (gpt-6.1-terra only returns a
one-line summary title), so the exported SFT rows would supervise the action but not the thinking that produced it.
Following a published scratchpad protocol (arXiv:2609.26637, Box 5), this module offers one extra
function tool, ``log_reasoning_step``, whose description asks the model to work the problem out *inside its argument*.
Mode ``forced`` names it in ``tool_choice`` on the first call of every request; each call is acknowledged with the fixed
output ``"Received"`` and the conversation continues in the same request until the model calls a real tool or answers.
Mode ``parallel`` offers it next to the real tools with ``parallel_tool_calls`` and a use-it-first instruction, so
thinking and acting usually arrive in one call (one request instead of two, but the teacher may skip thinking).

The steps are folded into one ``reasoning`` item of the merged response, so ``responses_to_chat`` hands them to the ReAct
driver as ``message.reasoning_content`` -- the same channel native DeepSeek reasoning arrives on -- and
``soarm.export_sft`` supervises them as the ``<think>`` block of the row (``sft/data/FORMAT.md``). The scratchpad calls
themselves never reach the teacher's visible reply: they are consumed here.

Cost: ``forced`` issues two full-history Responses calls per model request (the second one is a prefix-cache hit);
``parallel`` issues one. Both are opt-in at the proxy (``--data-learning``).
"""
from __future__ import annotations

import copy
import json
from typing import Any, Callable

SCRATCHPAD_NAME = "log_reasoning_step"
SCRATCHPAD_TEXT = ("Scratchpad for working through the problem. Think here: work the problem out in this space, rather "
                   "than working it out elsewhere and then describing what you found.")
SCRATCHPAD_TOOL = {"type": "function", "name": SCRATCHPAD_NAME, "description": SCRATCHPAD_TEXT,
                   "parameters": {"type": "object", "properties": {"step": {"type": "string", "description": SCRATCHPAD_TEXT}},
                                  "required": ["step"], "additionalProperties": False}}
ACK = "Received"
PARALLEL_HINT = (f"In every turn, first call {SCRATCHPAD_NAME} with your reasoning for this step, then call the tool that "
                 "acts, in the same response.")
MODES = ("forced", "parallel")

Transport = Callable[[dict[str, Any]], dict[str, Any]]


def _is_scratchpad(item: dict[str, Any]) -> bool:
    return item.get("type") == "function_call" and item.get("name") == SCRATCHPAD_NAME


def _step_text(item: dict[str, Any]) -> str:
    try:
        return str((json.loads(item.get("arguments") or "{}") or {}).get("step") or "")
    except (TypeError, ValueError):
        return str(item.get("arguments") or "")


def _add_usage(total: dict[str, Any], usage: dict[str, Any] | None) -> None:
    """Sum token counters across chained calls (top-level integers and one level of *_details)."""
    for key, value in (usage or {}).items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            total[key] = total.get(key, 0) + value
        elif isinstance(value, dict):
            sub = total.setdefault(key, {})
            for k, v in value.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    sub[k] = sub.get(k, 0) + v


def _omitted_text(n: int) -> str:
    from captain_harness.react.chat_llm import IMAGE_OMITTED_TEXT      # single source with the chat backend's cap

    return IMAGE_OMITTED_TEXT.format(n=n)


def cap_response_images(items: list[dict[str, Any]], max_images: int) -> tuple[list[dict[str, Any]], int]:
    """Copy of ``items`` with all but the ``max_images`` most recent images replaced by the chat placeholder.

    The same rule the chat backend applies to vLLM's ``--limit-mm-per-prompt`` and the SFT exporter reproduces, so the
    Responses teacher sees exactly the images a capped student would (and request bodies stay under gateway byte limits).
    """
    if max_images < 1:
        raise ValueError("max_images must be >= 1")
    items = copy.deepcopy(items)
    slots: list[tuple[list, int]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in ("content", "output"):
            parts = item.get(key)
            if isinstance(parts, list):
                for index, part in enumerate(parts):
                    if isinstance(part, dict) and part.get("type") in ("input_image", "image_url"):
                        slots.append((parts, index))
    excess = len(slots) - max_images
    for parts, index in slots[:max(0, excess)]:
        parts[index] = {"type": "input_text", "text": _omitted_text(max_images)}
    return items, max(0, excess)


class DataLearningClient:
    """Wraps a Responses transport. ``respond`` runs the scratchpad rounds and returns one merged Responses reply."""

    def __init__(self, transport: Transport, *, max_rounds: int = 3, mode: str = "forced") -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        if mode not in MODES:
            raise ValueError(f"unknown data-learning mode {mode!r}")
        self.transport = transport
        self.max_rounds = max_rounds
        self.mode = mode

    def _call(self, body: dict[str, Any], history: list[dict[str, Any]], *, force: bool,
              offer: bool) -> dict[str, Any]:
        request = dict(body)
        request["input"] = history
        if offer:
            request["tools"] = [*copy.deepcopy(body.get("tools") or []), copy.deepcopy(SCRATCHPAD_TOOL)]
            if self.mode == "parallel":
                request["parallel_tool_calls"] = True
                request["instructions"] = "\n\n".join(t for t in (body.get("instructions"), PARALLEL_HINT) if t)
        if force:
            request["tool_choice"] = {"type": "function", "name": SCRATCHPAD_NAME}
        return self.transport(request)

    def respond(self, body: dict[str, Any]) -> dict[str, Any]:
        history = list(body.get("input") or [])
        items: list[dict[str, Any]] = []
        steps: list[str] = []
        scratch_ids: set[str] = set()
        kinds: list[str] = []
        calls: list[dict[str, Any]] = []
        usage: dict[str, Any] = {}
        last: dict[str, Any] = {}
        rounds = 0
        while True:
            force = rounds == 0 and self.mode == "forced"
            offer = rounds < self.max_rounds
            last = self._call(body, history + items, force=force, offer=offer)
            output = [it for it in (last.get("output") or []) if isinstance(it, dict)]
            thinks = [it for it in output if _is_scratchpad(it)]
            acted = [it for it in output if it.get("type") in ("function_call", "custom_tool_call") and not _is_scratchpad(it)]
            u = last.get("usage") or {}
            _add_usage(usage, u)
            calls.append({"forced": force, "offered": offer, "response_id": last.get("id"),
                          "output_tokens": int(u.get("output_tokens") or 0),
                          "reasoning_tokens": int((u.get("output_tokens_details") or {}).get("reasoning_tokens") or 0)})
            items.extend(output)
            kinds.extend("think" if _is_scratchpad(it) else "act" for it in output
                         if it.get("type") in ("function_call", "custom_tool_call"))
            for call in thinks:
                steps.append(_step_text(call))
                scratch_ids.add(str(call.get("call_id")))
                items.append({"type": "function_call_output", "call_id": call.get("call_id"), "output": ACK})
            rounds += len(thinks)
            if not thinks or acted:
                break
        reply = [it for it in items if not _is_scratchpad(it) and str(it.get("call_id")) not in scratch_ids]
        merged = dict(last)
        if steps:
            merged["output"] = [{"type": "reasoning", "summary": [],
                                 "content": [{"type": "reasoning_text", "text": s} for s in steps]}] + reply
        else:
            merged["output"] = reply
        if usage:
            merged["usage"] = usage
        merged["data_learning"] = {
            "mode": self.mode, "max_rounds": self.max_rounds, "rounds": rounds, "steps": len(steps),
            "visible_words": sum(len(s.split()) for s in steps),
            "thought": bool(steps),
            "think_first": bool(kinds) and kinds[0] == "think",
            "native_reasoning_tokens": sum(c["reasoning_tokens"] for c in calls),
            "calls": calls,
        }
        return merged


__all__ = ["ACK", "MODES", "PARALLEL_HINT", "SCRATCHPAD_NAME", "SCRATCHPAD_TEXT", "SCRATCHPAD_TOOL",
           "DataLearningClient", "cap_response_images"]

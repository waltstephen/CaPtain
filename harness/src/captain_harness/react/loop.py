"""The bare loop: send the history, run every tool call the model made, append outputs, repeat
until a response carries no tool call. No turn cap; the caller bounds the episode by deadline.

Events are written in the `codex exec --json` shape (events.jsonl); the full item history and
per-request usage go to transcript.jsonl / requests.jsonl.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from captain_harness.react import context as ctx
from captain_harness.react.catalog import ModelProfile
from captain_harness.react.llm import ContextWindowExceeded, ResponsesClient
from captain_harness.react.tools import ToolExecutionAudit, ToolRouter

DATA_URL = re.compile(r"^data:([^;]+);base64,")


class EventLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def write(self, event: dict[str, Any]) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def item(self, kind: str, item_type: str, **fields: Any) -> None:
        self.write({"type": kind, "item": {"id": f"item_{uuid.uuid4().hex[:12]}", "type": item_type, **fields}})


@dataclass
class LoopResult:
    termination: str                    # completed | context_exhausted | deadline | model_error | request_cap | stopped
    requests: int
    tool_calls: int
    usage_total: dict[str, int]
    final_message: str | None
    reason: str | None = None
    max_context_tokens: int = 0
    reasoning_items: int = 0
    assistant_messages: int = 0
    compactions: int = 0


@dataclass
class Loop:
    client: ResponsesClient
    body: dict[str, Any]                # request body without input
    history: list[dict[str, Any]]       # initial input items
    router: ToolRouter
    profile: ModelProfile
    output_dir: Path
    prompt_cache_key: str
    code_mode_runner: Callable[[str], list[dict[str, Any]]] | None = None
    deadline_monotonic: float | None = None
    max_requests: int | None = None
    parallel_workers: int = 8
    compaction: bool = True            # compact before a request whose estimated size reaches profile.context_limit
    max_compactions: int = 50
    workspace: Path | None = None      # listed in the drop notice
    incomplete_nudges: int = 0          # chat backends: user nudges after a length-truncated reply without tool calls
    incomplete_nudge_text: str = ""
    stop_when: Callable[[], bool] | None = None   # caller's hook, checked before each request (e.g. "the environment is done")
    runtime_tool_contract: str | None = None
    environment_preflight: dict[str, Any] | None = None
    python_runner: Callable[[str], list[dict[str, Any]]] | None = None   # python-runtime/v1: runs one `python` call
    tool_description: str | None = None                                   # python-runtime/v1: the tool's description
    before_tools: Callable[[int], None] | None = None                     # called with the request index before its tools run
    before_request: Callable[["Loop", int], None] | None = None           # called with the next request's index before it is built
    events: EventLog = field(init=False)
    usage_total: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "cached_input_tokens": 0, "cache_write_input_tokens": 0, "output_tokens": 0, "reasoning_output_tokens": 0})

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tool_contract = None
        self._request_index = 0
        if self.runtime_tool_contract is not None:
            from captain_harness.react.code_mode import runtime_tool_contract
            from captain_harness.react.python_repl import PYTHON_RUNTIME_CONTRACT_VERSION, python_runtime_contract

            python_mode = self.runtime_tool_contract == PYTHON_RUNTIME_CONTRACT_VERSION
            if python_mode and self.python_runner is None:
                raise ValueError("python runtime contract requires a python runner")
            if not python_mode and self.profile.tool_mode != "code_mode":
                raise ValueError("runtime tool contract requires a code-mode profile")
            workspace = Path(self.workspace or self.router.workspace).resolve()
            audit_dir = self.output_dir.resolve()
            if audit_dir == workspace or workspace in audit_dir.parents:
                raise ValueError("private tool audit must be outside the model workspace")
            sandbox = getattr(self.router.sessions, "sandbox", None)
            if sandbox is not None and sandbox.enabled:
                hidden = [Path(path).resolve() for path in sandbox.hidden_dirs]
                if not any((path == audit_dir or path in audit_dir.parents) and path != workspace and workspace not in path.parents for path in hidden):
                    raise ValueError("private tool audit is not hidden by the workspace sandbox")
            if python_mode:
                self.body, self.history, self.tool_contract = python_runtime_contract(
                    self.body, self.history, self.runtime_tool_contract, description=self.tool_description)
            else:
                self.body, self.history, self.tool_contract = runtime_tool_contract(self.body, self.history, self.runtime_tool_contract)
            self.router.runtime_contract = self.runtime_tool_contract
            self.router.audit = ToolExecutionAudit(self.output_dir / "tool_audit.jsonl", contract=self.tool_contract,
                                                   environment_preflight=self.environment_preflight)
        self.events = EventLog(self.output_dir / "events.jsonl")
        self.requests_path = self.output_dir / "requests.jsonl"
        self.transcript_path = self.output_dir / "transcript.jsonl"
        self.router._on_command = self._on_command  # type: ignore[attr-defined]
        self._thread_id = f"thread_{uuid.uuid4()}"
        self._initial_items = list(self.history)
        self._state = ctx.CompactionState()
        self._estimator = ctx.TokenEstimator()
        # instructions + tool schemas are sent with every request and count in usage.input_tokens
        self._fixed_raw = ctx.estimate_tokens([{k: v for k, v in self.body.items() if k != "input"}])
        self._last_raw_estimate = 0          # bytes/4 estimate of the last request (fixed part + history), for calibration
        self._last_total_tokens = 0
        self._items_since_response: list[dict[str, Any]] = []
        self._item_ids: dict[int, int] = {}
        self._item_refs: list[dict[str, Any]] = []   # strong references: id(item) must never be reused by a later item
        self._next_item_id = 0
        self.views_path = self.output_dir / "views.jsonl"
        self.items_path = self.output_dir / "items.jsonl"
        for item in self.history:
            self._register(item)

    @property
    def compactions(self) -> int:
        return self._state.compactions

    # ------------------------------------------------------------- item registry (exact per-request views for SFT export)
    def _register(self, item: dict[str, Any]) -> int:
        key = id(item)
        if key not in self._item_ids:
            self._item_ids[key] = self._next_item_id
            self._item_refs.append(item)
            with self.items_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"id": self._next_item_id, "item": _strip_images(item, self._image_sources())}, ensure_ascii=False) + "\n")
            self._next_item_id += 1
        return self._item_ids[key]

    def _record_view(self, index: int, estimated_tokens: int) -> None:
        self._request_index = index
        self._last_raw_estimate = self._fixed_raw + ctx.estimate_tokens(self.history)
        with self.views_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"request": index, "items": [self._register(it) for it in self.history],
                                     "estimated_tokens": estimated_tokens, "estimate_scale": round(self._estimator.scale, 3),
                                     "compactions": self.compactions}) + "\n")

    def _estimated_context(self) -> int:
        """Codex's active_context_tokens: last reported real total + calibrated estimate of everything appended since.
        Without a real total (first request, after a compaction, or a provider that reports no usage) the whole
        history is estimated with the calibrated scale."""
        if self._last_total_tokens > 0:
            return self._last_total_tokens + self._estimator.estimate(self._items_since_response)
        return self._estimator.estimate(self.history, self._fixed_raw)

    # ------------------------------------------------------------------ run
    def reset_history(self, items: list[dict[str, Any]]) -> None:
        """Start the conversation over from ``items`` (a new initial prompt): the compaction state and the per-response
        bookkeeping restart with it. Requests already made keep their indices; the loop's budgets are unchanged."""
        self.history = list(items)
        self._initial_items = list(self.history)
        self._state = ctx.CompactionState()
        self._items_since_response = list(self.history)
        self._last_total_tokens = 0          # the size of the old conversation no longer describes the next request
        for item in self.history:
            self._register(item)
        self.events.write({"type": "history.reset", "history_items": len(self.history)})

    def run(self) -> LoopResult:
        self.events.write({"type": "thread.started", "thread_id": self._thread_id})
        self.events.write({"type": "turn.started"})
        requests = 0
        tool_calls = 0
        final_message = None
        termination, reason = "completed", None
        max_context = 0
        reasoning_items = 0
        assistant_messages = 0
        nudges = 0
        audit_complete = True
        try:
            while True:
                if self.deadline_monotonic is not None and time.monotonic() >= self.deadline_monotonic:
                    termination, reason = "deadline", "wall time budget exhausted before the next request"
                    break
                if self.max_requests is not None and requests >= self.max_requests:
                    termination, reason = "request_cap", f"request cap {self.max_requests} reached"
                    break
                if self.before_request is not None:
                    self.before_request(self, requests + 1)
                estimated = self._estimated_context()
                if estimated >= self.profile.context_limit:
                    if not self.compaction or self.compactions >= self.max_compactions:
                        termination, reason = "context_exhausted", f"estimated {estimated} tokens >= limit {self.profile.context_limit}"
                        break
                    if not self._compact(estimated, forced=False):
                        termination, reason = "context_exhausted", f"compaction failed at estimated {estimated} tokens"
                        break
                    estimated = self._estimated_context()
                if self.stop_when is not None and requests > 0 and self.stop_when():
                    termination, reason = "stopped", "caller asked the loop to stop"
                    break
                body = dict(self.body)
                body["input"] = self.history
                body["prompt_cache_key"] = self.prompt_cache_key
                requests += 1
                self._record_view(requests, estimated)
                try:
                    result = self.client.respond(body)
                except ContextWindowExceeded as exc:
                    if self.compaction and self.compactions < self.max_compactions and self._compact(estimated, forced=True):
                        continue
                    termination, reason = "context_exhausted", str(exc)[:500]
                    break
                except Exception as exc:  # noqa: BLE001 - any provider/client failure ends the turn like Codex's turn.failed
                    termination, reason = "model_error", f"{type(exc).__name__}: {exc}"[:1000]
                    self.events.write({"type": "error", "message": reason})
                    break
                result.usage = ctx.normalize_usage(result.usage)
                self._account(result.usage)
                self._estimator.calibrate(self._last_raw_estimate, int(result.usage.get("input_tokens") or 0))
                self._last_total_tokens = int(result.usage.get("total_tokens") or 0)
                self._items_since_response = []
                max_context = max(max_context, self._last_total_tokens)
                self._record_request(requests, result, len(result.items))
                calls: list[dict[str, Any]] = []
                for item in result.items:
                    self.history.append(item)
                    self._register(item)
                    kind = item.get("type")
                    if kind == "reasoning":
                        reasoning_items += 1
                        summary = " ".join(str(s.get("text", "")) for s in (item.get("summary") or []) if isinstance(s, dict)).strip()
                        if summary:
                            self.events.item("item.completed", "reasoning", text=summary)
                    elif kind == "message" and item.get("role") == "assistant":
                        assistant_messages += 1
                        text = "".join(str(c.get("text", "")) for c in (item.get("content") or []) if isinstance(c, dict))
                        final_message = text
                        self.events.item("item.completed", "agent_message", text=text)
                    elif kind in ("function_call", "custom_tool_call"):
                        calls.append(item)
                if not calls:
                    if result.status == "incomplete" and nudges < self.incomplete_nudges:
                        nudges += 1
                        nudge = ctx.user_message(self.incomplete_nudge_text)
                        self.history.append(nudge)
                        self._register(nudge)
                        self._items_since_response.append(nudge)
                        self.events.write({"type": "incomplete_nudge", "count": nudges, "response_id": result.response_id})
                        self._write_transcript()
                        continue
                    break
                tool_calls += len(calls)
                if self.before_tools is not None:
                    self.before_tools(requests)
                for output in self._execute(calls):
                    self.history.append(output)
                    self._register(output)
                    self._items_since_response.append(output)
                self._write_transcript()
        except BaseException:
            audit_complete = False
            raise
        finally:
            self._write_transcript()
            self.events.write({"type": "turn.completed", "usage": dict(self.usage_total)})
            self.router.finish_audit(complete=audit_complete)
        return LoopResult(termination=termination, requests=requests, tool_calls=tool_calls, usage_total=dict(self.usage_total),
                          final_message=final_message, reason=reason, max_context_tokens=max_context,
                          reasoning_items=reasoning_items, assistant_messages=assistant_messages, compactions=self.compactions)

    # -------------------------------------------------------------- compaction
    def _compact(self, estimated: int, *, forced: bool) -> bool:
        """Turn-atomic compaction: initial items + drop notice (facts [+ incremental summary]) + recent turns.

        ``forced`` is the provider-error path: the recent-turn budget is halved so the retry fits."""
        keep = self.profile.keep_recent_tokens // (2 if forced else 1)
        history_before = len(self.history)
        summary = None
        notes = None
        if self.profile.context_strategy == "notes":
            if not forced:
                # one turn to update the notes file; the checkpoint exchange itself is not kept (its instruction
                # "end your turn" must not be the last thing the model sees after compaction)
                mark = len(self.history)
                self._checkpoint_turn()
                self.history = self.history[:mark]
            notes = ctx.read_notes(self.workspace)
        dropped, kept = ctx.plan_compaction(self._initial_items, self.history, keep)
        if not dropped:
            # nothing older to drop: as a last resort drop the newest turn too (its commands are listed in the notice)
            if not kept:
                self.events.write({"type": "error", "message": "compaction impossible: no turns to drop"})
                return False
            dropped, kept = kept, []
        if self.profile.context_strategy == "summary":
            request = ctx.summary_request(dict(self.body, prompt_cache_key=self.prompt_cache_key), self._initial_items,
                                          self._state.summary, dropped, self.profile.summary_budget_tokens)
            try:
                result = self.client.respond(request)
                self._account(ctx.normalize_usage(result.usage))
                summary = ctx.extract_summary(result.items, self.profile.summary_budget_tokens)
            except Exception as exc:  # noqa: BLE001 - fall back to facts only
                self.events.write({"type": "error", "message": f"compaction summary failed, keeping facts only: {exc}"[:500]})
                summary = self._state.summary
        self._state.absorb(dropped)
        self._state.summary = summary
        notice = ctx.drop_notice(self._state, workspace=self.workspace, summary=summary, notes=notes)
        self.history = ctx.rebuild(self._initial_items, kept, notice)
        for item in self.history:
            self._register(item)
        self._last_total_tokens = 0
        self._items_since_response = list(self.history[len(self._initial_items):])
        self._write_transcript()
        with (self.output_dir / "compactions.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"index": self.compactions, "t": time.time(), "forced": forced, "trigger_estimated_tokens": estimated,
                                     "history_items_before": history_before, "history_items_after": len(self.history),
                                     "dropped_turns": len(dropped), "kept_turns": len(kept), "strategy": self.profile.context_strategy,
                                     "notice": ctx.text_of(notice)}, ensure_ascii=False) + "\n")
        self.events.item("item.completed", "compaction", dropped_turns=len(dropped), kept_turns=len(kept), history_items=len(self.history),
                         strategy=self.profile.context_strategy)
        return True

    def _checkpoint_turn(self) -> None:
        """One extra turn in which the model updates its notes file (a compact checkpoint turn, without a schema)."""
        prompt = ctx.user_message(ctx.CHECKPOINT_PROMPT.format(notes=ctx.DEFAULT_NOTES_FILE))
        self.history.append(prompt)
        self._register(prompt)
        body = dict(self.body)
        body["input"] = self.history
        body["prompt_cache_key"] = self.prompt_cache_key
        try:
            result = self.client.respond(body)
        except Exception as exc:  # noqa: BLE001 - checkpoint is best effort
            self.events.write({"type": "error", "message": f"checkpoint turn failed: {exc}"[:500]})
            return
        self._account(ctx.normalize_usage(result.usage))
        calls = []
        for item in result.items:
            self.history.append(item)
            self._register(item)
            if item.get("type") in ("function_call", "custom_tool_call"):
                calls.append(item)
        for output in self._execute(calls):
            self.history.append(output)
            self._register(output)
        self.events.item("item.completed", "checkpoint", tool_calls=len(calls))

    # -------------------------------------------------------------- tool calls
    def _execute(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.python_runner is not None:
            outputs, seen = [], set()
            for call in calls:
                key = (call.get("name"), call.get("input"), call.get("arguments"))
                outputs.append(self._duplicate_output(call) if key in seen else self._run_one(call))
                seen.add(key)
            return outputs
        if self.profile.tool_mode == "code_mode" or len(calls) == 1:
            return [self._run_one(call) for call in calls]
        with ThreadPoolExecutor(max_workers=min(self.parallel_workers, len(calls))) as pool:
            return list(pool.map(self._run_one, calls))

    def _run_one(self, call: dict[str, Any]) -> dict[str, Any]:
        with self.router.outer_call(str(call.get("call_id") or call.get("id") or ""), self._request_index,
                                    str(call.get("name") or ""), str(call.get("type") or "")):
            return self._run_one_impl(call)

    def _duplicate_output(self, call: dict[str, Any]) -> dict[str, Any]:
        """A second identical call in the same response is not executed (models sometimes emit the same call twice)."""
        call_id = call.get("call_id") or call.get("id") or ""
        text = "not executed: identical to the previous call in the same response"
        with self.router.outer_call(str(call_id), self._request_index, str(call.get("name") or ""), str(call.get("type") or "")):
            self.router.record_outer_error("duplicate call in one response")
        kind = "custom_tool_call_output" if call.get("type") == "custom_tool_call" else "function_call_output"
        return {"type": kind, "call_id": call_id, "output": text}

    def _run_one_impl(self, call: dict[str, Any]) -> dict[str, Any]:
        call_id = call.get("call_id") or call.get("id") or ""
        name = str(call.get("name") or "")
        if call.get("type") == "custom_tool_call":
            raw_input = str(call.get("input") or "")
            try:
                if name == "python" and self.python_runner is not None:
                    output: Any = self.python_runner(raw_input)
                elif name == "exec" and self.code_mode_runner is not None:
                    output = self.code_mode_runner(raw_input)
                else:
                    output = self.router.run_custom(call_id, name, raw_input)
            except Exception as exc:  # noqa: BLE001
                self.router.record_outer_error(f"{name} failed: {exc}")
                output = f"{name} failed: {exc}"
            return {"type": "custom_tool_call_output", "call_id": call_id, "output": output}
        try:
            output = self.router.run_function(call_id, name, call.get("arguments"))
        except Exception as exc:  # noqa: BLE001 - a tool failure is reported to the model, never raised
            self.router.record_outer_error(str(exc))
            output = f"{name} failed: {exc}" if not isinstance(exc, ValueError) else str(exc)
        return {"type": "function_call_output", "call_id": call_id, "output": output}

    def _on_command(self, record: dict[str, Any]) -> None:
        self.events.item("item.completed", "command_execution", command=f"/bin/bash -lc {json.dumps(record.get('command', ''))}",
                         exit_code=record.get("exit_code"), status="completed" if record.get("exit_code") is not None else "in_progress",
                         aggregated_output_chars=record.get("output_chars"))

    # -------------------------------------------------------------- accounting
    def _account(self, usage: dict[str, Any]) -> None:
        details_in = usage.get("input_tokens_details") or {}
        details_out = usage.get("output_tokens_details") or {}
        self.usage_total["input_tokens"] += int(usage.get("input_tokens") or 0)
        self.usage_total["cached_input_tokens"] += int(details_in.get("cached_tokens") or 0)
        self.usage_total["cache_write_input_tokens"] += int(details_in.get("cache_write_tokens") or 0)
        self.usage_total["output_tokens"] += int(usage.get("output_tokens") or 0)
        self.usage_total["reasoning_output_tokens"] += int(details_out.get("reasoning_tokens") or 0)

    def _record_request(self, index: int, result, new_items: int) -> None:
        replayed = bool((getattr(result, "raw", None) or {}).get("replayed"))
        with self.requests_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"index": index, "new_items": new_items, "history_items": len(self.history), "response_id": result.response_id,
                                     "status": result.status, "usage": result.usage, "latency_s": round(result.latency_s, 2),
                                     "attempts": result.attempts, "estimate_scale": round(self._estimator.scale, 3),
                                     "raw_estimate": self._last_raw_estimate, "replayed": replayed, "t": time.time()}) + "\n")
        # The model's own output items per request: what a prefix replay (react.replay) re-issues verbatim.
        with (self.output_dir / "responses.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"request": index, "response_id": result.response_id, "status": result.status, "usage": result.usage,
                                     "replayed": replayed, "items": [_strip_images(item, self._image_sources()) for item in result.items]},
                                    ensure_ascii=False) + "\n")
        self.events.write({"type": "token_count", "info": {"last_token_usage": result.usage, "total_token_usage": dict(self.usage_total)}})

    def _image_sources(self) -> dict[str, str]:
        return getattr(self.router, "image_sources", None) or {}

    def _write_transcript(self) -> None:
        sources = self._image_sources()
        with self.transcript_path.open("w", encoding="utf-8") as stream:
            for item in self.history:
                stream.write(json.dumps(_strip_images(item, sources), ensure_ascii=False) + "\n")


def _strip_images(item: dict[str, Any], sources: dict[str, str] | None = None) -> dict[str, Any]:
    """Replace base64 image payloads (tool outputs and message content) with their mime type and
    length so transcripts stay small; ``sources`` (data URL -> file path, kept by the router) adds
    a ``path`` so exporters can re-attach the file."""
    stripped = dict(item)
    for key in ("output", "content"):
        parts = item.get(key)
        if not isinstance(parts, list):
            continue
        new_parts = []
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "input_image" and isinstance(part.get("image_url"), str) \
                    and part["image_url"].startswith("data:"):
                url = part["image_url"]
                match = DATA_URL.match(url)
                replaced = {**part, "image_url": f"data:{match.group(1) if match else 'image'};base64,<{len(url)} chars omitted>"}
                path = (sources or {}).get(url)
                if path and "path" not in replaced:
                    replaced["path"] = path
                new_parts.append(replaced)
            else:
                new_parts.append(part)
        stripped[key] = new_parts
    return stripped


__all__ = ["EventLog", "Loop", "LoopResult"]

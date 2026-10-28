"""Chat Completions backend for the bare loop (vLLM and other OpenAI-compatible servers).

The loop builds Responses-API bodies (``instructions``, ``input`` items, ``tools``) and expects a
``ResponseResult`` with Responses output items back. This client translates both directions so
``Loop`` stays unchanged:

* history: developer messages and ``instructions`` become one leading ``system`` message; user
  messages keep their text / image parts; ``reasoning`` + assistant ``message`` + ``function_call`` /
  ``custom_tool_call`` items of one model response merge into one assistant message with
  ``reasoning_content``, ``content`` and ``tool_calls``; ``*_call_output`` items become ``tool``
  messages. Image parts stay inside the tool message by default (vLLM and the Qwen3.5 chat template
  render them there, and a real user message would reset the template's "last user query" and drop
  the ``<think>`` blocks of earlier turns); ``hoist_tool_images`` moves them into one user message
  right after the batch of tool messages for servers that only accept text in ``tool`` messages.
* tools: Responses function tools become chat function tools; ``custom`` (freeform) tools such as
  ``apply_patch`` and code-mode ``exec`` become function tools with a single string parameter
  ``input``. Only the tools the driver implements are exposed unless ``all_tools`` is set.
* reply: ``reasoning_content`` / ``reasoning`` -> a ``reasoning`` item with a summary; ``tool_calls`` ->
  ``function_call`` items (or ``custom_tool_call`` for the custom tool names); text -> assistant message.

The client never edits the history it is given: ``body["input"]`` is the driver's per-request VIEW
(react.context: initial items, compaction notice, kept turns, spilled-output notes; recorded in
driver/views.jsonl), so the chat request is that view translated one-to-one. The only extra
transformation is ``cap_images`` (vLLM's ``--limit-mm-per-prompt``), recorded in episode_result.json
so the exporter can apply it identically. The same translation is used offline to export views as SFT
samples, so what a student is trained on is exactly what it is served.
"""
from __future__ import annotations

import http.client
import json
import os
import re
import socket
import ssl
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from captain_harness.react.tools import valid_image_url
from captain_harness.react.llm import (
    ContextWindowExceeded,
    ModelRequestError,
    ResponseResult,
    RetryableModelError,
    _retry_hint,
    backoff_delay,
)

CUSTOM_INPUT_PARAMETER = "input"
IMPLEMENTED_TOOLS: tuple[str, ...] = ("exec_command", "write_stdin", "view_image", "apply_patch", "exec", "python")
UNPARSED_TOOL_CALL_MARKERS = ("<tool_call>", "<function=")
# vLLM: "This model's maximum context length is 131072 tokens. However, you requested 140000 tokens
# (125000 in the messages, 15000 in the completion). Please reduce ..."
VLLM_CONTEXT_RE = re.compile(r"maximum context length is (\d+) tokens.*?requested (\d+) tokens?\s*\((\d+) in the messages", re.S)
IMAGE_OMITTED_TEXT = "[image omitted: older than the {n} most recent images]"
HOISTED_IMAGES_NOTE = "[{n} image(s) returned by this tool call are attached in the next user message]"
HOISTED_IMAGES_HEADER = "Images returned by tool call(s) {ids}:"
INCOMPLETE_NUDGE_TEXT = ("Your previous reply was cut off by the output token limit before it made a tool call. "
                         "Continue from where you stopped; keep replies short and act through tool calls.")
MIN_COMPLETION_TOKENS = 64


# ----------------------------------------------------------------------------- content helpers
def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    texts = []
    for part in content or ():
        if isinstance(part, dict) and part.get("type") in ("input_text", "output_text", "text") and "text" in part:
            texts.append(str(part["text"]))
    return "".join(texts)


def _image_url(part: dict[str, Any]) -> str | None:
    url = part.get("image_url")
    if isinstance(url, dict):
        url = url.get("url")
    return str(url) if url else None


def _chat_parts(content: Any, *, keep_detail: bool = False) -> list[dict[str, Any]]:
    """Responses content parts -> chat content parts (text / image_url)."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    parts: list[dict[str, Any]] = []
    for part in content or ():
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind in ("input_text", "output_text", "text"):
            parts.append({"type": "text", "text": str(part.get("text", ""))})
        elif kind in ("input_image", "image_url"):
            url = _image_url(part)
            if url is None:
                continue
            if not valid_image_url(url):
                # never forward a non-image URL: the provider would reject the whole request (HTTP 400)
                parts.append({"type": "text", "text": "[image dropped: not a valid image URL]"})
                continue
            image: dict[str, Any] = {"url": url}
            if keep_detail and part.get("detail"):
                image["detail"] = part["detail"]
            parts.append({"type": "image_url", "image_url": image})
    return parts


def _compact(parts: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    """A single text part collapses to a plain string (what chat servers expect for text-only)."""
    if len(parts) == 1 and parts[0].get("type") == "text":
        return parts[0]["text"]
    if not parts:
        return ""
    return parts


# ----------------------------------------------------------------------------- tools
# --------------------------------------------------------------------------------------------- image-blind control
# CAPTAIN_GRAY_IMAGES=1 (eval-only causal check): every image the MODEL receives is replaced by a uniform gray PNG of the
# same size right before the request is sent. The environment, depth, SAM3/Molmo tools and mark_point back-projection keep using
# the real RGB-D (they never read these request images). CAPTAIN_GRAY_AUDIT=<file> appends one line per request with the check.
_GRAY_CACHE: dict[tuple[int, int], str] = {}


def _gray_url(url: str) -> str:
    import base64 as _b64
    import io as _io
    from PIL import Image as _Image
    head, _, data = url.partition(",")
    size = _Image.open(_io.BytesIO(_b64.b64decode(data))).size
    if size not in _GRAY_CACHE:
        buf = _io.BytesIO()
        _Image.new("RGB", size, (128, 128, 128)).save(buf, format="PNG")
        _GRAY_CACHE[size] = "data:image/png;base64," + _b64.b64encode(buf.getvalue()).decode("ascii")
    return _GRAY_CACHE[size]


def is_uniform_gray(url: str) -> bool:
    import base64 as _b64
    import io as _io
    from PIL import Image as _Image
    im = _Image.open(_io.BytesIO(_b64.b64decode(url.partition(",")[2]))).convert("RGB")
    return all(lo == hi == 128 for lo, hi in im.getextrema())


def gray_out_images(messages: list[dict[str, Any]]) -> int:
    n = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for p in c:
                if p.get("type") == "image_url" and str(p["image_url"].get("url", "")).startswith("data:image/"):
                    p["image_url"]["url"] = _gray_url(p["image_url"]["url"]); n += 1
    urls = [p["image_url"]["url"] for m in messages if isinstance(m.get("content"), list) for p in m["content"] if p.get("type") == "image_url"]
    ok = all(u.startswith("data:image/") and is_uniform_gray(u) for u in urls)
    audit = os.environ.get("CAPTAIN_GRAY_AUDIT")
    if audit:
        with open(audit, "a") as f:
            f.write(json.dumps({"images": len(urls), "grayed": n, "all_gray": ok}) + "\n")
    if not ok:
        raise RuntimeError("CAPTAIN_GRAY_IMAGES: a request image is not uniform gray")
    return n


def _function_tool(name: str, description: str, parameters: dict[str, Any] | None) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description or "",
                                             "parameters": parameters or {"type": "object", "properties": {}}}}


def _custom_as_function(tool: dict[str, Any]) -> dict[str, Any]:
    description = str(tool.get("description") or "")
    fmt = tool.get("format") or {}
    if isinstance(fmt, dict) and fmt.get("syntax") == "lark" and fmt.get("definition"):
        description += f"\n\nThe `{CUSTOM_INPUT_PARAMETER}` parameter is the raw tool input (no JSON quoting) and must match this grammar:\n{fmt['definition']}"
    return _function_tool(str(tool["name"]), description, {
        "type": "object",
        "properties": {CUSTOM_INPUT_PARAMETER: {"type": "string", "description": "Raw input text for the tool."}},
        "required": [CUSTOM_INPUT_PARAMETER],
        "additionalProperties": False,
    })


def convert_tools(body: dict[str, Any], items: Iterable[dict[str, Any]], *, allow: Iterable[str] | None = IMPLEMENTED_TOOLS
                  ) -> tuple[list[dict[str, Any]], set[str]]:
    """Return (chat function tools, names that are custom/freeform and must round-trip as custom_tool_call)."""
    allowed = set(allow) if allow is not None else None
    raw: list[dict[str, Any]] = list(body.get("tools") or [])
    for item in items:
        if isinstance(item, dict) and item.get("type") == "additional_tools":
            for namespace in item.get("tools") or []:
                prefix = "" if namespace.get("name") == "functions" else f"{namespace.get('name')}__"
                for tool in namespace.get("tools") or []:
                    raw.append({**tool, "name": prefix + str(tool.get("name"))} if prefix else tool)
    tools: list[dict[str, Any]] = []
    custom: set[str] = set()
    seen: set[str] = set()
    for tool in raw:
        if not isinstance(tool, dict):
            continue
        kind = tool.get("type")
        name = tool.get("name")
        if kind not in ("function", "custom") or not name or name in seen:
            continue
        if allowed is not None and name not in allowed:
            continue
        seen.add(name)
        if kind == "custom":
            custom.add(str(name))
            tools.append(_custom_as_function(tool))
        else:
            tools.append(_function_tool(str(name), str(tool.get("description") or ""), tool.get("parameters")))
    return tools, custom


# ----------------------------------------------------------------------------- history
def _tool_call(item: dict[str, Any]) -> dict[str, Any]:
    call_id = str(item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:24]}")
    if item.get("type") == "custom_tool_call":
        arguments = json.dumps({CUSTOM_INPUT_PARAMETER: str(item.get("input") or "")}, ensure_ascii=False)
    else:
        raw = item.get("arguments")
        arguments = raw if isinstance(raw, str) else json.dumps(raw or {}, ensure_ascii=False)
    return {"id": call_id, "type": "function", "function": {"name": str(item.get("name") or ""), "arguments": arguments}}


def _reasoning_text(item: dict[str, Any]) -> str:
    texts = []
    for part in item.get("summary") or []:
        if isinstance(part, dict) and part.get("text"):
            texts.append(str(part["text"]))
    for part in item.get("content") or []:
        if isinstance(part, dict) and part.get("type") in ("reasoning_text", "text") and part.get("text"):
            texts.append(str(part["text"]))
    return "\n\n".join(t for t in texts if t.strip())


def responses_to_chat_messages(instructions: str | None, items: Iterable[dict[str, Any]], *, history_reasoning: bool = True,
                               keep_detail: bool = False, turn_groups: list[tuple[int, int]] | None = None,
                               hoist_tool_images: bool = False) -> list[dict[str, Any]]:
    """Translate the Responses history into chat messages.

    ``turn_groups`` (optional, from requests.jsonl: (first_index, count) per model response) lets the
    offline exporter split adjacent responses exactly; online, consecutive assistant-side items are
    one group anyway because every response with tool calls is followed by their outputs.

    ``hoist_tool_images``: tool outputs keep their text plus a note, and the image parts of one batch
    of tool outputs are appended as a single user message right after the batch.
    """
    items = list(items)
    boundaries = {start for start, _ in (turn_groups or [])} - {0}
    system_texts: list[str] = [instructions] if instructions else []
    messages: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    hoisted: list[dict[str, Any]] = []
    hoisted_ids: list[str] = []

    def flush_hoisted() -> None:
        if hoisted:
            messages.append({"role": "user", "content": [{"type": "text", "text": HOISTED_IMAGES_HEADER.format(ids=", ".join(hoisted_ids))}, *hoisted]})
            hoisted.clear()
            hoisted_ids.clear()

    def flush() -> None:
        nonlocal pending
        if pending is None:
            return
        message: dict[str, Any] = {"role": "assistant", "content": "\n\n".join(pending["texts"]) if pending["texts"] else ""}
        reasoning = "\n\n".join(pending["reasoning"])
        if reasoning and history_reasoning:
            message["reasoning_content"] = reasoning
        if pending["calls"]:
            message["tool_calls"] = pending["calls"]
        messages.append(message)
        pending = None

    def ensure_pending() -> dict[str, Any]:
        nonlocal pending
        if pending is None:
            pending = {"texts": [], "reasoning": [], "calls": []}
        return pending

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind not in ("function_call_output", "custom_tool_call_output"):
            flush_hoisted()
        if index in boundaries and pending is not None and pending["calls"]:
            flush()
        if kind == "additional_tools":
            continue
        if kind == "message":
            role = item.get("role")
            if role == "developer" or role == "system":
                if not messages and pending is None:
                    system_texts.append(_text_of(item.get("content")))
                else:
                    flush()
                    messages.append({"role": "user", "content": _compact(_chat_parts(item.get("content"), keep_detail=keep_detail))})
            elif role == "user":
                flush()
                messages.append({"role": "user", "content": _compact(_chat_parts(item.get("content"), keep_detail=keep_detail))})
            else:  # assistant
                text = _text_of(item.get("content"))
                if text:
                    ensure_pending()["texts"].append(text)
                else:
                    ensure_pending()
        elif kind == "reasoning":
            text = _reasoning_text(item)
            group = ensure_pending()
            if text:
                group["reasoning"].append(text)
        elif kind in ("function_call", "custom_tool_call"):
            ensure_pending()["calls"].append(_tool_call(item))
        elif kind in ("function_call_output", "custom_tool_call_output"):
            flush()
            output = item.get("output")
            call_id = str(item.get("call_id") or "")
            if isinstance(output, list):
                parts = _chat_parts(output, keep_detail=keep_detail)
                if hoist_tool_images and any(p.get("type") == "image_url" for p in parts):
                    images = [p for p in parts if p.get("type") == "image_url"]
                    texts = [p for p in parts if p.get("type") == "text"]
                    texts.append({"type": "text", "text": HOISTED_IMAGES_NOTE.format(n=len(images))})
                    hoisted.extend(images)
                    hoisted_ids.append(call_id)
                    parts = texts
                content: Any = _compact(parts)
            elif isinstance(output, dict):
                content = json.dumps(output, ensure_ascii=False)
            else:
                content = str(output if output is not None else "")
            messages.append({"role": "tool", "tool_call_id": call_id, "content": content})
    flush()
    flush_hoisted()
    system = "\n\n".join(t for t in system_texts if t)
    if system:
        messages.insert(0, {"role": "system", "content": system})
    return messages


def cap_images(messages: list[dict[str, Any]], max_images: int | None) -> int:
    """Keep only the ``max_images`` most recent image parts (in place); returns how many were replaced."""
    if not max_images or max_images <= 0:
        return 0
    positions: list[tuple[int, int]] = []
    for m_index, message in enumerate(messages):
        content = message.get("content")
        if isinstance(content, list):
            for p_index, part in enumerate(content):
                if isinstance(part, dict) and part.get("type") == "image_url":
                    positions.append((m_index, p_index))
    excess = len(positions) - max_images
    if excess <= 0:
        return 0
    for m_index, p_index in positions[:excess]:
        messages[m_index]["content"][p_index] = {"type": "text", "text": IMAGE_OMITTED_TEXT.format(n=max_images)}
    return excess


# ----------------------------------------------------------------------------- reply
def chat_message_to_items(message: dict[str, Any], custom_names: Iterable[str] = ()) -> list[dict[str, Any]]:
    custom = set(custom_names)
    items: list[dict[str, Any]] = []
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        items.append({"type": "reasoning", "id": f"rs_{uuid.uuid4().hex}", "summary": [{"type": "summary_text", "text": reasoning}], "content": []})
    content = message.get("content")
    text = _text_of(content) if isinstance(content, list) else (content if isinstance(content, str) else "")
    if text and text.strip():
        items.append({"type": "message", "id": f"msg_{uuid.uuid4().hex}", "role": "assistant", "status": "completed",
                      "content": [{"type": "output_text", "text": text, "annotations": []}]})
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(function.get("name") or call.get("name") or "")
        raw_arguments = function.get("arguments", call.get("arguments", "{}"))
        call_id = str(call.get("id") or f"call_{uuid.uuid4().hex[:24]}")
        if name in custom:
            parsed: Any = None
            if isinstance(raw_arguments, dict):
                parsed = raw_arguments
            else:
                try:
                    parsed = json.loads(raw_arguments or "{}")
                except json.JSONDecodeError:
                    parsed = None
            if isinstance(parsed, dict) and CUSTOM_INPUT_PARAMETER in parsed:
                raw_input = str(parsed[CUSTOM_INPUT_PARAMETER])
            elif isinstance(parsed, dict) and len(parsed) == 1:
                raw_input = str(next(iter(parsed.values())))
            else:
                raw_input = raw_arguments if isinstance(raw_arguments, str) else json.dumps(raw_arguments)
            items.append({"type": "custom_tool_call", "id": f"ctc_{uuid.uuid4().hex}", "status": "completed", "call_id": call_id, "name": name, "input": raw_input})
        else:
            arguments = raw_arguments if isinstance(raw_arguments, str) else json.dumps(raw_arguments or {}, ensure_ascii=False)
            items.append({"type": "function_call", "id": f"fc_{uuid.uuid4().hex}", "status": "completed", "call_id": call_id, "name": name, "arguments": arguments})
    return items


def looks_like_unparsed_tool_call(message: dict[str, Any]) -> bool:
    if message.get("tool_calls"):
        return False
    content = message.get("content")
    text = _text_of(content) if isinstance(content, list) else (content or "")
    return any(marker in text for marker in UNPARSED_TOOL_CALL_MARKERS)


def normalize_usage(usage: dict[str, Any] | None) -> dict[str, Any]:
    usage = usage or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    return {
        "input_tokens": prompt,
        "input_tokens_details": {"cached_tokens": int(prompt_details.get("cached_tokens") or 0), "cache_write_tokens": 0},
        "output_tokens": completion,
        "output_tokens_details": {"reasoning_tokens": int(completion_details.get("reasoning_tokens") or 0)},
        "total_tokens": int(usage.get("total_tokens") or (prompt + completion)),
    }


# ----------------------------------------------------------------------------- client
@dataclass
class ChatCompletionsClient:
    endpoint: str                                   # ".../v1" or a full ".../chat/completions" URL
    api_key: str | None = None
    max_output_tokens: int | None = 16_384
    temperature: float | None = None
    top_p: float | None = None
    enable_thinking: bool | None = None             # chat_template_kwargs.enable_thinking (Qwen-style servers)
    max_images: int | None = 16                     # vLLM --limit-mm-per-prompt image cap
    history_reasoning: bool = True
    hoist_tool_images: bool = False
    keep_image_detail: bool = False
    all_tools: bool = False
    extra_body: dict[str, Any] = field(default_factory=dict)
    request_max_retries: int = 8
    stream_max_retries: int = 20
    timeout_s: float = 900.0
    connect_timeout_s: float = 60.0
    event_sink: Callable[[dict[str, Any]], None] | None = None

    @property
    def url(self) -> str:
        endpoint = self.endpoint.rstrip("/")
        return endpoint if endpoint.endswith("/chat/completions") else endpoint + "/chat/completions"

    # ------------------------------------------------------------------ translation
    def build_request(self, body: dict[str, Any]) -> tuple[dict[str, Any], set[str], dict[str, int]]:
        items = list(body.get("input") or [])
        tools, custom = convert_tools(body, items, allow=None if self.all_tools else IMPLEMENTED_TOOLS)
        messages = responses_to_chat_messages(body.get("instructions"), items, history_reasoning=self.history_reasoning,
                                              keep_detail=self.keep_image_detail, hoist_tool_images=self.hoist_tool_images)
        dropped = cap_images(messages, self.max_images)
        grayed = gray_out_images(messages) if os.environ.get("CAPTAIN_GRAY_IMAGES") == "1" else None
        request: dict[str, Any] = {"model": body.get("model"), "messages": messages, "stream": False}
        if tools:
            request["tools"] = tools
            choice = body.get("tool_choice", "auto")
            request["tool_choice"] = choice if choice in ("auto", "required", "none") else "auto"
            if isinstance(body.get("parallel_tool_calls"), bool):
                request["parallel_tool_calls"] = body["parallel_tool_calls"]
        if self.max_output_tokens:
            request["max_tokens"] = int(self.max_output_tokens)
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.top_p is not None:
            request["top_p"] = self.top_p
        if self.enable_thinking is not None:
            request["chat_template_kwargs"] = {"enable_thinking": bool(self.enable_thinking)}
        for key, value in self.extra_body.items():
            request.setdefault(key, value)
        stats = {"messages": len(messages), "images_dropped": dropped,
                 "images": sum(1 for m in messages if isinstance(m.get("content"), list) for p in m["content"] if p.get("type") == "image_url"),
                 "view_items": len(items)}
        if grayed is not None:
            stats["images_grayed"] = grayed
        return request, custom, stats

    # ------------------------------------------------------------------ public
    def respond(self, body: dict[str, Any]) -> ResponseResult:
        started = time.monotonic()
        request, custom, stats = self.build_request(body)
        if stats["images_dropped"]:
            self._emit({"event": "images_dropped", "count": stats["images_dropped"], "kept": self.max_images})
        attempt = 0
        http_failures = 0
        context_retry_done = False
        last_error: Exception | None = None
        while attempt < max(1, self.stream_max_retries):
            attempt += 1
            try:
                payload = self._once(request)
            except _ContextBudget as exc:
                if context_retry_done or exc.messages_tokens <= 0 or exc.max_context - exc.messages_tokens < MIN_COMPLETION_TOKENS:
                    raise ContextWindowExceeded(exc.message)
                context_retry_done = True
                request["max_tokens"] = exc.max_context - exc.messages_tokens - 16
                self._emit({"event": "max_tokens_adjusted", "max_tokens": request["max_tokens"], "messages_tokens": exc.messages_tokens})
                continue
            except RetryableModelError as exc:
                last_error = exc
                if exc.kind == "http":
                    http_failures += 1
                    if http_failures > self.request_max_retries:
                        break
                delay = exc.retry_after_s if exc.retry_after_s is not None else backoff_delay(attempt)
                self._emit({"event": "retry", "attempt": attempt, "delay_s": round(delay, 2), "error": str(exc)[:300]})
                time.sleep(delay)
                continue
            result = self._to_result(payload, custom)
            result.latency_s = time.monotonic() - started
            result.attempts = attempt
            self._emit({"event": "response", "attempt": attempt, "status": result.status, "latency_s": result.latency_s,
                        "usage": result.usage, "response_id": result.response_id, "finish_reason": result.raw.get("finish_reason"),
                        "unparsed_tool_call": result.raw.get("unparsed_tool_call"), **stats})
            return result
        raise ModelRequestError(f"model request failed after {attempt} attempts: {last_error}")

    # ------------------------------------------------------------------ helpers
    def _emit(self, event: dict[str, Any]) -> None:
        if self.event_sink:
            try:
                self.event_sink(event)
            except Exception:
                pass

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _once(self, request: dict[str, Any]) -> dict[str, Any]:
        url = urllib.parse.urlsplit(self.url)
        path = url.path + (f"?{url.query}" if url.query else "")
        conn_cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
        kwargs: dict[str, Any] = {"timeout": self.connect_timeout_s}
        if url.scheme == "https":
            kwargs["context"] = ssl.create_default_context()
        conn = conn_cls(url.hostname, url.port, **kwargs)
        payload = json.dumps(request).encode("utf-8")
        try:
            try:
                conn.request("POST", path, body=payload, headers=self._headers())
                conn.sock.settimeout(self.timeout_s)
                response = conn.getresponse()
                text = response.read().decode("utf-8", errors="replace")
            except (socket.timeout, TimeoutError):
                raise RetryableModelError("timeout waiting for the chat completion", kind="stream")
            except (OSError, http.client.HTTPException) as exc:
                raise RetryableModelError(f"connection failed: {exc}", kind="http")
            if response.status != 200:
                self._raise_for_status(response.status, text, response.headers)
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                raise RetryableModelError(f"invalid JSON from server: {exc}", kind="stream")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _raise_for_status(self, status: int, text: str, headers: Any) -> None:
        code = ""
        try:
            error = json.loads(text).get("error") or {}
            code = str(error.get("code") or error.get("type") or "")
            message = str(error.get("message") or text)
        except Exception:
            message = text
        if status == 400 and ("context_length" in code or "context length" in message or "maximum context" in message):
            match = VLLM_CONTEXT_RE.search(message)
            if match:
                raise _ContextBudget(message, max_context=int(match.group(1)), messages_tokens=int(match.group(3)))
            raise ContextWindowExceeded(message)
        if status == 429:
            raise RetryableModelError(f"HTTP 429: {message[:500]}", retry_after_s=_retry_hint(message, headers), kind="rate_limit")
        if status >= 500 or status in (408, 409):
            raise RetryableModelError(f"HTTP {status}: {message[:500]}", retry_after_s=_retry_hint(message, headers), kind="http")
        raise ModelRequestError(f"HTTP {status}: {message[:1000]}")

    def _to_result(self, payload: dict[str, Any], custom: set[str]) -> ResponseResult:
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ModelRequestError("chat completion has no choices")
        choice = choices[0]
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        finish_reason = choice.get("finish_reason")
        items = chat_message_to_items(message, custom)
        unparsed = looks_like_unparsed_tool_call(message)
        status = "completed"
        if finish_reason == "length" and not message.get("tool_calls"):
            status = "incomplete"
        raw = {"id": payload.get("id"), "model": payload.get("model"), "finish_reason": finish_reason, "unparsed_tool_call": unparsed,
               "reasoning_chars": len(str(message.get("reasoning_content") or message.get("reasoning") or ""))}
        return ResponseResult(response_id=str(payload.get("id") or f"chatcmpl-{uuid.uuid4().hex}"), items=items,
                              usage=normalize_usage(payload.get("usage")), status=status, latency_s=0.0, attempts=1, raw=raw)


class _ContextBudget(Exception):
    """vLLM rejected prompt+max_tokens; the prompt alone may still fit."""

    def __init__(self, message: str, *, max_context: int, messages_tokens: int) -> None:
        super().__init__(message)
        self.message = message
        self.max_context = max_context
        self.messages_tokens = messages_tokens


__all__ = ["CUSTOM_INPUT_PARAMETER", "HOISTED_IMAGES_HEADER", "HOISTED_IMAGES_NOTE", "IMAGE_OMITTED_TEXT",
           "IMPLEMENTED_TOOLS", "INCOMPLETE_NUDGE_TEXT", "ChatCompletionsClient", "cap_images", "chat_message_to_items", "convert_tools",
           "looks_like_unparsed_tool_call", "normalize_usage", "responses_to_chat_messages"]

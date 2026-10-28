"""Responses-API client with Codex CLI's transport behaviour.

* ``store: false`` + full-history replay is the caller's business; this client only posts a body.
* streaming SSE: collects ``response.output_item.done`` items, returns on ``response.completed``.
* retries: connection errors / 429 / 5xx / stream failures with exponential backoff
  (200 ms * 2^(n-1) * jitter), honouring Retry-After and "try again in Xs" hints.
* auth: none (a local proxy or server without auth) or bearer (``Authorization: Bearer $CAPTAIN_API_KEY``,
  falling back to ``$OPENAI_API_KEY``).
"""
from __future__ import annotations

import http.client
import json
import os
import random
import re
import socket
import ssl
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable

RETRY_AFTER_RE = re.compile(r"try again in ([0-9.]+)\s*(ms|s)", re.IGNORECASE)


class ModelRequestError(RuntimeError):
    pass


class ContextWindowExceeded(ModelRequestError):
    pass


class RetryableModelError(ModelRequestError):
    """kind: "rate_limit" (429 / rate-limit failures), "stream" (SSE errors, idle timeouts), "http" (5xx, auth),
    "connection" (the endpoint could not be reached: refused / reset / connect timeout)."""

    def __init__(self, message: str, *, retry_after_s: float | None = None, kind: str = "stream") -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s
        self.kind = kind


@dataclass
class ResponseResult:
    response_id: str | None
    items: list[dict[str, Any]]
    usage: dict[str, Any]
    status: str
    latency_s: float
    attempts: int
    raw: dict[str, Any] = field(default_factory=dict)


def backoff_delay(attempt: int, base_s: float = 0.2, cap_s: float = 60.0) -> float:
    return min(cap_s, base_s * (2 ** (attempt - 1)) * random.uniform(0.9, 1.1))


def _retry_hint(message: str, headers: Any) -> float | None:
    for key in ("retry-after-ms", "x-ms-retry-after-ms"):
        value = headers.get(key) if headers else None
        if value:
            try:
                return float(value) / 1000.0
            except ValueError:
                pass
    value = headers.get("retry-after") if headers else None
    if value:
        try:
            return float(value)
        except ValueError:
            pass
    match = RETRY_AFTER_RE.search(message or "")
    if match:
        number = float(match.group(1))
        return number / 1000.0 if match.group(2).lower() == "ms" else number
    return None


class EnvTokenProvider:
    """Static bearer token read from the environment (CAPTAIN_API_KEY, else OPENAI_API_KEY). Never written to logs."""

    def __init__(self, names: tuple[str, ...] = ("CAPTAIN_API_KEY", "OPENAI_API_KEY")) -> None:
        import os

        self._token = next((os.environ[n] for n in names if os.environ.get(n)), None)
        if not self._token:
            raise ValueError(f"--auth bearer needs one of {', '.join(names)} in the environment")

    def get(self) -> str:
        return self._token

    def invalidate(self) -> None:  # static token: nothing to refresh
        pass


class ResponsesClient:
    def __init__(self, endpoint: str, *, auth: str = "none", token_provider: Any | None = None,
                 request_max_retries: int = 8, stream_max_retries: int = 20, stream_idle_timeout_s: float = 900.0,
                 connect_timeout_s: float = 60.0, event_sink: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.endpoint = endpoint
        self.auth = auth
        self.token_provider = token_provider
        self.request_max_retries = request_max_retries
        self.stream_max_retries = stream_max_retries
        self.stream_idle_timeout_s = stream_idle_timeout_s
        self.connect_timeout_s = connect_timeout_s
        self.event_sink = event_sink
        if auth not in ("none", "bearer"):
            raise ValueError(f"unknown auth mode {auth!r} (use 'none' or 'bearer')")
        if auth == "bearer" and token_provider is None:
            self.token_provider = EnvTokenProvider()

    # ------------------------------------------------------------------ public
    def respond(self, body: dict[str, Any]) -> ResponseResult:
        started = time.monotonic()
        attempt = 0
        http_failures = 0
        last_error: Exception | None = None
        # infra only (env, default off): a separate, longer budget for rate limits with a backoff floor - very short
        # Retry-After hints otherwise burn the attempt budget within a minute of throttling
        rl_budget = int(os.environ.get("REACT_RATE_LIMIT_MAX_RETRIES", "0"))
        rl_floor = float(os.environ.get("REACT_RATE_LIMIT_MIN_DELAY_S", "0"))
        rate_limited = 0
        while attempt < max(1, self.stream_max_retries) + rate_limited:
            attempt += 1
            try:
                result = self._once(body)
                result.latency_s = time.monotonic() - started
                result.attempts = attempt
                self._emit({"event": "response", "attempt": attempt, "status": result.status, "latency_s": result.latency_s,
                            "usage": result.usage, "response_id": result.response_id})
                return result
            except ContextWindowExceeded:
                raise
            except RetryableModelError as exc:
                last_error = exc
                if exc.kind == "http":
                    http_failures += 1
                    if http_failures > self.request_max_retries:
                        break
                if exc.kind == "connection":
                    # an unreachable endpoint (e.g. a reverse tunnel being re-established) is waited out for the whole
                    # attempt budget (~7 min at 30 s per retry) instead of failing the episode after the HTTP budget (~50 s)
                    delay = backoff_delay(attempt, cap_s=30.0)
                else:
                    delay = exc.retry_after_s if exc.retry_after_s is not None else backoff_delay(attempt)
                if exc.kind == "rate_limit" and rl_budget and rate_limited < rl_budget:
                    rate_limited += 1  # does not use the normal attempt budget
                    delay = max(delay, min(60.0, rl_floor * (1.5 ** min(rate_limited - 1, 10))) * random.uniform(0.8, 1.2))
                self._emit({"event": "retry", "attempt": attempt, "delay_s": round(delay, 2), "error": str(exc)[:300]})
                time.sleep(delay)
            except ModelRequestError:
                raise
        raise ModelRequestError(f"model request failed after {attempt} attempts: {last_error}")

    # ----------------------------------------------------------------- helpers
    def _emit(self, event: dict[str, Any]) -> None:
        if self.event_sink:
            try:
                self.event_sink(event)
            except Exception:
                pass

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.auth == "bearer" and self.token_provider is not None:
            headers["Authorization"] = f"Bearer {self.token_provider.get()}"
        return headers

    def _once(self, body: dict[str, Any]) -> ResponseResult:
        url = urllib.parse.urlsplit(self.endpoint)
        path = url.path + (f"?{url.query}" if url.query else "")
        conn_cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
        kwargs: dict[str, Any] = {"timeout": self.connect_timeout_s}
        if url.scheme == "https":
            kwargs["context"] = ssl.create_default_context()
        conn = conn_cls(url.hostname, url.port, **kwargs)
        payload = json.dumps({**body, "stream": True}).encode("utf-8")
        try:
            try:
                conn.request("POST", path, body=payload, headers=self._headers())
                response = conn.getresponse()
            except (OSError, http.client.HTTPException) as exc:
                raise RetryableModelError(f"connection failed: {exc}", kind="connection")
            if response.status in (401, 403) and self.auth == "bearer":
                text = response.read().decode("utf-8", errors="replace")
                raise ModelRequestError(f"auth rejected ({response.status}): {text[:300]}")
            if response.status != 200:
                text = response.read().decode("utf-8", errors="replace")
                self._raise_for_status(response.status, text, response.headers)
            conn.sock.settimeout(self.stream_idle_timeout_s)
            return self._read_stream(response)
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
        if status == 400 and ("context_length" in code or "context_length" in message or "maximum context length" in message):
            raise ContextWindowExceeded(message)
        if status == 429:
            raise RetryableModelError(f"HTTP 429: {message[:500]}", retry_after_s=_retry_hint(message, headers), kind="rate_limit")
        if status >= 500:
            raise RetryableModelError(f"HTTP {status}: {message[:500]}", retry_after_s=_retry_hint(message, headers), kind="http")
        raise ModelRequestError(f"HTTP {status}: {message[:1000]}")

    def _read_stream(self, response: http.client.HTTPResponse) -> ResponseResult:
        items: list[dict[str, Any]] = []
        completed: dict[str, Any] | None = None
        event_type = None
        data_lines: list[str] = []

        def handle(evt: dict[str, Any]) -> None:
            nonlocal completed
            kind = evt.get("type")
            if kind == "response.output_item.done":
                items.append(evt.get("item") or {})
            elif kind == "response.completed":
                completed = evt.get("response") or {}
            elif kind in ("response.failed", "error"):
                err = (evt.get("response") or {}).get("error") or evt.get("error") or evt
                code = str(err.get("code") or err.get("type") or "")
                message = str(err.get("message") or err)
                if "context_length" in code or "context_length" in message:
                    raise ContextWindowExceeded(message)
                if "rate_limit" in code or "rate limit" in message.lower():
                    raise RetryableModelError(message, retry_after_s=_retry_hint(message, None), kind="rate_limit")
                if "server_error" in code or "overloaded" in message.lower():
                    raise RetryableModelError(message, retry_after_s=_retry_hint(message, None), kind="stream")
                raise ModelRequestError(f"response failed: {message[:1000]}")
            elif kind == "response.incomplete":
                reason = ((evt.get("response") or {}).get("incomplete_details") or {}).get("reason")
                if reason == "max_output_tokens":
                    completed = evt.get("response") or {}
                    return
                raise RetryableModelError(f"incomplete response: {reason}")

        try:
            while True:
                raw = response.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:
                    if data_lines:
                        data = "\n".join(data_lines)
                        data_lines = []
                        if data.strip() and data.strip() != "[DONE]":
                            try:
                                evt = json.loads(data)
                            except json.JSONDecodeError:
                                evt = None
                            if isinstance(evt, dict):
                                handle(evt)
                                if completed is not None:
                                    break
                    event_type = None
                    continue
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
        except (socket.timeout, TimeoutError):
            raise RetryableModelError("idle timeout waiting for SSE")
        except (OSError, http.client.HTTPException) as exc:
            raise RetryableModelError(f"stream error: {exc}")
        if completed is None:
            raise RetryableModelError("stream ended without response.completed")
        output = completed.get("output") if isinstance(completed.get("output"), list) else None
        final_items = output if output else items
        return ResponseResult(response_id=completed.get("id"), items=final_items, usage=completed.get("usage") or {},
                              status=str(completed.get("status") or "completed"), latency_s=0.0, attempts=1, raw=completed)


__all__ = ["ContextWindowExceeded", "ModelRequestError", "ResponseResult", "ResponsesClient", "RetryableModelError", "backoff_delay"]

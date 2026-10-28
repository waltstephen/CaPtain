"""Tool execution and output formatting exactly as Codex CLI 0.154 presents them to the model.

* token accounting: ``(bytes + 3) // 4``; budget ``min(max_output_tokens, 10000)`` tokens
* middle truncation keeping half the byte budget at each end, marker ``…N tokens truncated…``
* exec_command / write_stdin envelope: Chunk ID / Wall time / Process ... / Original token count / Output:
* view_image: an ``input_image`` data URL part, or the exact "unable to locate image" error text
"""
from __future__ import annotations

import base64
import functools
import binascii
import io
import json
import mimetypes
import os
import re
import secrets
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

DEFAULT_MAX_OUTPUT_TOKENS = 10_000
OUTPUT_TOKEN_LIMIT = 10_000
TRUNCATION_MARKER = "…{n} tokens truncated…"


def approx_tokens(data: bytes | str) -> int:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return (len(data) + 3) // 4


def _decode_head(data: bytes) -> str:
    return data.decode("utf-8", errors="ignore")


def _decode_tail(data: bytes) -> str:
    return data.decode("utf-8", errors="ignore")


def truncate_output(text: str, max_output_tokens: int | None = None, *, limit: int = OUTPUT_TOKEN_LIMIT) -> tuple[str, int]:
    """Return (output as the model sees it, original token count)."""
    data = text.encode("utf-8")
    original = approx_tokens(data)
    requested = int(max_output_tokens) if max_output_tokens else DEFAULT_MAX_OUTPUT_TOKENS
    kept = max(1, min(requested, limit))
    budget = 4 * kept
    if len(data) <= budget:
        return text, original
    head_len = budget // 2
    tail_len = budget - head_len
    head = data[:head_len]
    tail = data[len(data) - tail_len:]
    removed_tokens = (len(data) - len(head) - len(tail) + 3) // 4
    total_lines = text.count("\n") + (0 if (not text or text.endswith("\n")) else 1)
    body = _decode_head(head) + TRUNCATION_MARKER.format(n=removed_tokens) + _decode_tail(tail)
    return f"Warning: truncated output (original token count: {original})\nTotal output lines: {total_lines}\n\n{body}", original


def new_chunk_id() -> str:
    return secrets.token_hex(3)


@dataclass
class ExecResult:
    chunk_id: str
    wall_time_s: float
    output: str                    # already truncated
    original_token_count: int
    exit_code: int | None = None
    session_id: int | None = None
    raw_output: str | None = field(default=None, repr=False)
    audit_session_id: int | None = field(default=None, repr=False)

    def as_text(self) -> str:
        """function_call_output body for exec_command / write_stdin."""
        lines = [f"Chunk ID: {self.chunk_id}", f"Wall time: {self.wall_time_s:.4f} seconds"]
        if self.exit_code is not None:
            lines.append(f"Process exited with code {self.exit_code}")
        elif self.session_id is not None:
            lines.append(f"Process running with session ID {self.session_id}")
        lines.append(f"Original token count: {self.original_token_count}")
        lines.append("Output:")
        return "\n".join(lines) + "\n" + self.output

    def as_object(self) -> dict[str, Any]:
        """Nested-tool result for code mode (key order matters: JSON.stringify keeps it)."""
        result: dict[str, Any] = {"chunk_id": self.chunk_id, "wall_time_seconds": self.wall_time_s}
        if self.exit_code is not None:
            result["exit_code"] = self.exit_code
        else:
            result["session_id"] = self.session_id
        result["original_token_count"] = self.original_token_count
        result["output"] = self.output
        return result


@dataclass
class PythonCellResult:
    """One call of the persistent `python` tool (react.python_repl); ``output`` is untruncated."""
    cell: int
    status: str                    # ok | error | timeout | restarted | rejected
    output: str
    display: str | None
    exception: str | None
    images: list[dict[str, Any]]
    epilogue: str | None
    changed: list[dict[str, Any]]
    wall_s: float
    generation: int


def view_image(path: str, workspace: Path, detail: str | None = None) -> list[dict[str, Any]] | str:
    """Return the input_image part list, or Codex's error text when the file is missing."""
    detail = detail if detail in ("high", "original") else "high"
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(workspace) / candidate
    try:
        data = candidate.read_bytes()
    except FileNotFoundError:
        return f"unable to locate image at `{path}`: No such file or directory (os error 2)"
    except IsADirectoryError:
        return f"unable to locate image at `{path}`: Is a directory (os error 21)"
    mime = sniff_image_mime(data)
    if mime is None:
        # a text/markdown/json file rendered as an image makes the provider reject the whole request (vLLM 400
        # "cannot identify image file"); the model gets a tool error instead, like a missing file
        guessed = mimetypes.guess_type(str(candidate))[0] or "unknown type"
        return f"unable to view image at `{path}`: not an image file ({guessed}); view_image accepts PNG/JPEG/GIF/WEBP/BMP files only"
    url = f"data:{mime};base64," + base64.b64encode(data).decode("ascii")
    return [{"type": "input_image", "image_url": url, "detail": detail}]


IMAGE_SIGNATURES = ((b"\x89PNG\r\n\x1a\n", "image/png"), (b"\xff\xd8\xff", "image/jpeg"), (b"GIF87a", "image/gif"), (b"GIF89a", "image/gif"), (b"BM", "image/bmp"))


def sniff_image_mime(data: bytes) -> str | None:
    """MIME type from the file signature (PNG/JPEG/GIF/WEBP/BMP), or None when the bytes are not an image."""
    for magic, mime in IMAGE_SIGNATURES:
        if data.startswith(magic):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


VALID_IMAGE_URL = re.compile(r"^(data:image/[a-zA-Z0-9.+-]+;base64,|https?://)")


@functools.lru_cache(maxsize=4096)
def _valid_data_image(url: str) -> bool:
    payload = url.split(",", 1)[1] if "," in url else ""
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        return False
    return sniff_image_mime(data) is not None


def valid_image_url(url: Any) -> bool:
    """True for the only image URLs a chat/Responses provider accepts: http(s), or data:image/...;base64 whose payload is
    valid base64 AND decodes to a real image (a model-built or edited data URL fails the provider with HTTP 400
    "Non-base64 digit found" / "cannot identify image file", which would otherwise abort the whole episode)."""
    if not isinstance(url, str) or not VALID_IMAGE_URL.match(url):
        return False
    if url.startswith("http"):
        return True
    return _valid_data_image(url)


def image_data_error(url: Any) -> str | None:
    """Validate local image bytes without fetching a URL or changing legacy image handling."""
    if not isinstance(url, str) or not url.startswith("data:image/") or not VALID_IMAGE_URL.match(url):
        return "image() requires a valid data-image URL; external images are unavailable"
    from PIL import Image

    try:
        raw = base64.b64decode(url.split(",", 1)[1], validate=True)
        with Image.open(io.BytesIO(raw)) as decoded:
            decoded.load()
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        return f"image() received invalid image data ({exc})"
    return None


def parse_arguments(raw: str | Mapping[str, Any] | None) -> tuple[dict[str, Any] | None, str | None]:
    if raw is None:
        return {}, None
    if isinstance(raw, Mapping):
        return dict(raw), None
    if not isinstance(raw, str):
        return None, "failed to parse function arguments: expected a JSON object"
    if not str(raw).strip():
        return {}, None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"failed to parse function arguments: {exc}"
    if not isinstance(value, dict):
        return None, "failed to parse function arguments: expected a JSON object"
    return value, None


class ToolExecutionAudit:
    """Private, append-only execution facts, never part of a model tool response."""

    SCHEMA_VERSION = "tool-execution-audit/v1"

    def __init__(self, path: Path, *, contract: Mapping[str, Any], environment_preflight: Any = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._closed = False
        self.outer_started = self.outer_finished = self.tool_started = self.tool_finished = 0
        self._active_outer: set[str] = set()
        self._active_tools: set[str] = set()
        # A new driver must never append to an older episode's audit.
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
        self.write({"type": "header", "schema_version": self.SCHEMA_VERSION,
                    "contract_version": contract["version"], "contract_sha256": contract["sha256"],
                    "environment_preflight": environment_preflight})

    @property
    def closed(self) -> bool:
        return self._closed

    def write(self, event: Mapping[str, Any]) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("tool execution audit is already closed")
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(event), ensure_ascii=False) + "\n")
                stream.flush()

    def start_outer(self, context: dict[str, Any]) -> None:
        with self._lock:
            call_id = context["outer_call_id"]
            if call_id in self._active_outer:
                raise ValueError("duplicate active outer tool call id")
            self.write({"type": "outer_start", **context})
            self.outer_started += 1
            self._active_outer.add(call_id)

    def end_outer(self, context: dict[str, Any], error: str | None) -> None:
        with self._lock:
            self.write({"type": "outer_end", **context, "status": "failed" if error else "completed", "error": error})
            self.outer_finished += 1
            self._active_outer.remove(context["outer_call_id"])

    def start_tool(self, event: dict[str, Any]) -> None:
        with self._lock:
            self.write({"type": "tool_start", **event})
            self.tool_started += 1
            self._active_tools.add(event["tool_call_id"])

    def end_tool(self, event: dict[str, Any]) -> None:
        with self._lock:
            self.write({"type": "tool_end", **event})
            self.tool_finished += 1
            self._active_tools.remove(event["tool_call_id"])

    def finalize(self, *, complete: bool, unresolved_sessions: list[int]) -> None:
        with self._lock:
            self.write({"type": "footer", "complete": bool(complete and not self._active_outer and not self._active_tools and not unresolved_sessions),
                        "outer_started": self.outer_started, "outer_finished": self.outer_finished,
                        "tool_started": self.tool_started, "tool_finished": self.tool_finished,
                        "unresolved_sessions": unresolved_sessions})
            self._closed = True


class ToolRouter:
    """Dispatch model tool calls (function mode and code-mode nested calls) to implementations."""

    STUB_MESSAGES = {
        "request_user_input": "request_user_input is unavailable in Default mode",
        "request_user_input_async": "request_user_input_async is unavailable in Default mode",
        "list_mcp_resources": "no MCP servers are configured",
        "list_mcp_resource_templates": "no MCP servers are configured",
        "read_mcp_resource": "no MCP servers are configured",
        "get_goal": "no goal is set for this thread",
        "create_goal": "goals are disabled in this session",
        "update_goal": "goals are disabled in this session",
        "tool_search": "no deferred tools matched the query",
        "wait": "no background work to wait for",
        "sleep": "sleep is disabled in this session",
    }

    def __init__(self, *, sessions, workspace: Path, apply_patch: Callable[[str], str] | None = None,
                 on_command: Callable[[Mapping[str, Any]], None] | None = None) -> None:
        self.sessions = sessions
        self.workspace = Path(workspace)
        self._apply_patch = apply_patch
        self._on_command = on_command
        self.image_paths: dict[str, list[str]] = {}
        self.image_sources: dict[str, str] = {}      # data URL -> path as the model gave it (transcript annotation)
        self.allowed_tools: set[str] | None = None    # chat backends: names the request exposed; others get "unsupported call"
        self.view_image_calls = 0
        self.commands: list[dict[str, Any]] = []
        self.audit: ToolExecutionAudit | None = None
        self.runtime_contract: str | None = None
        self.code_mode_store: dict[str, Any] = {}
        self.environment_errors: list[dict[str, Any]] = []
        self._audit_context = threading.local()
        self._session_commands: dict[int, str] = {}
        self._session_arguments: dict[int, dict[str, Any]] = {}
        self._audit_lock = threading.RLock()
        self.closers: list[Callable[[], Mapping[str, Any]]] = []   # long-lived tool processes, closed before the audit footer
        self.python_cells: list[dict[str, Any]] = []

    def audit_context(self) -> dict[str, Any] | None:
        return getattr(self._audit_context, "current", None)

    @contextmanager
    def outer_call(self, call_id: str, request_index: int, name: str, kind: str):
        context = {"outer_call_id": call_id, "request_index": request_index, "tool_name": name, "call_kind": kind}
        previous = self.audit_context()
        self._audit_context.current = context
        self._audit_context.error = None
        if self.audit is not None:
            self.audit.start_outer(context)
        try:
            yield
        except BaseException as exc:
            self._audit_context.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if self.audit is not None:
                self.audit.end_outer(context, getattr(self._audit_context, "error", None))
            self._audit_context.current = previous

    def record_outer_error(self, error: str) -> None:
        self._audit_context.error = error

    def _record_tool(self, name: str, arguments: Any, invoke: Callable[[], Any], context: dict[str, Any] | None = None) -> Any:
        if self.audit is None:
            return invoke()
        context = context or self.audit_context()
        if context is None:
            raise RuntimeError("audited tool execution has no outer call")
        args = dict(arguments) if isinstance(arguments, Mapping) else arguments
        call_id = "tool_" + secrets.token_hex(12)
        event = {"tool_call_id": call_id, "parent_call_id": context["outer_call_id"],
                 "request_index": context["request_index"], "tool_name": name, "args": args}
        self.audit.start_tool(event)
        result = None
        error = None
        environment_error = None
        command_id = None
        try:
            result = invoke()
            if name == "apply_patch" and isinstance(result, str) and result.startswith("apply_patch verification failed:"):
                error = result
            if isinstance(result, ExecResult):
                with self._audit_lock:
                    session_id = result.audit_session_id or result.session_id
                    if name == "exec_command":
                        command_id = call_id
                        if session_id is not None:
                            self._session_commands[session_id] = command_id
                            self._session_arguments[session_id] = dict(args or {})
                    else:
                        session_id = int((args or {}).get("session_id", session_id))
                        command_id = self._session_commands.get(session_id)
                    command_args = self._session_arguments.get(session_id, dict(args or {}))
                if result.exit_code == 127:
                    environment_error = self._verify_environment(command_args)
                if result.exit_code not in (None, 0):
                    error = f"process exited with code {result.exit_code}"
            return result
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if isinstance(result, ExecResult):
                if result.raw_output is None:
                    raise RuntimeError("audited shell result is missing untruncated output")
                value = {"exit_code": result.exit_code, "session_id": result.audit_session_id or result.session_id,
                         "output": result.raw_output}
            elif isinstance(result, PythonCellResult):
                if result.status == "unavailable":
                    environment_error = {"schema_version": "harness-environment-error/v1", "code": "python_session_unavailable",
                                         "executable": "python3", "source": "tool_runtime", "verified": True,
                                         "details": {"error": (result.exception or "")[-2000:]}}
                    with self._audit_lock:
                        self.environment_errors.append(environment_error)
                if result.status != "ok" and error is None:
                    last_line = ((result.exception or "").strip().splitlines() or [""])[-1]
                    error = f"python cell {result.status}: {last_line}"[:500]
                value = {"exit_code": None, "session_id": None, "output": result.output, "status": result.status,
                         "display": result.display, "exception": result.exception,
                         "images": [str(i.get("path") or "") for i in result.images], "wall_s": result.wall_s,
                         "cell": result.cell, "generation": result.generation}
            else:
                value = {"exit_code": None, "session_id": None, "output": result if isinstance(result, str) else ""}
            self.audit.end_tool({**event, "command_id": command_id, "result": value, "error": error,
                                 "error_class": "environment" if environment_error else ("policy" if error else None),
                                 "environment_error": environment_error})

    def _verify_environment(self, command_args: Mapping[str, Any]) -> dict[str, Any] | None:
        from captain_harness.react.toolchain_probe import REQUIRED_EXECUTABLES, probe_executable

        for executable in REQUIRED_EXECUTABLES:
            if executable == "node":
                continue  # The code runner resolves its entrypoint outside the login shell.
            probe = probe_executable(self.sessions, executable, login=command_args.get("login", True), shell=command_args.get("shell"))
            if not probe["ok"]:
                error = {"schema_version": "harness-environment-error/v1", "code": "required_executable_unavailable",
                         "executable": executable, "source": "tool_runtime", "verified": True, "details": probe}
                with self._audit_lock:
                    self.environment_errors.append(error)
                return error
        return None

    def record_python(self, arguments: Mapping[str, Any], invoke: Callable[[], PythonCellResult]) -> PythonCellResult:
        """Audited execution of one `python` tool call (the cell result keeps the untruncated output)."""
        result = self._record_tool("python", dict(arguments), invoke)
        self.python_cells.append({"cell": result.cell, "status": result.status, "wall_s": result.wall_s,
                                  "generation": result.generation, "code_chars": len(str(arguments.get("code") or ""))})
        return result

    def close_tools(self) -> list[dict[str, Any]]:
        stats = []
        for closer in self.closers:
            try:
                stats.append(dict(closer()))
            except Exception as exc:  # noqa: BLE001 - closing must not block the audit footer
                stats.append({"error": f"{type(exc).__name__}: {exc}"})
        self.closers = []
        return stats

    def finish_audit(self, *, complete: bool) -> None:
        """Close tool processes and write the footer once; later calls are no-ops (the driver closes the audit of a loop
        still blocked at the wall-time deadline, and the loop thread may reach its own finally afterwards)."""
        if self.audit is not None and self.audit.closed:
            return
        tool_stats = self.close_tools()
        if self.audit is None:
            return
        if tool_stats:
            self.audit.write({"type": "tool_processes_closed", "stats": tool_stats})
        unresolved = []
        for final in self.sessions.close_all():
            session_id = final["session_id"]
            command_id = self._session_commands.get(session_id)
            if not final["completed"]:
                unresolved.append(session_id)
            if command_id is not None:
                environment_error = None
                if final["exit_code"] == 127 and not final["cancelled"]:
                    environment_error = self._verify_environment(self._session_arguments.get(session_id, {}))
                failed = final["exit_code"] not in (None, 0) and not final["cancelled"]
                self.audit.write({"type": "session_end", "command_id": command_id, "session_id": session_id,
                                  "exit_code": final["exit_code"], "output": final["output"],
                                  "status": "cancelled" if final["cancelled"] else "completed", "source": "episode_cleanup",
                                  "error_class": "environment" if environment_error else ("policy" if failed else None),
                                  "environment_error": environment_error})
        self.audit.finalize(complete=complete, unresolved_sessions=unresolved)

    # --- nested/code-mode entry point: returns a JSON-able object or raises ValueError(message)
    def nested_call(self, name: str, arguments: Mapping[str, Any] | str | None, *, audit_context: dict[str, Any] | None = None) -> Any:
        if isinstance(arguments, str) and name in ("exec_command", "write_stdin", "view_image"):
            arguments = {"cmd": arguments} if name == "exec_command" else {"chars": arguments} if name == "write_stdin" else {"path": arguments}
        result = self._record_tool(name, arguments, lambda: self._nested_call(name, arguments), audit_context)
        return result.as_object() if isinstance(result, ExecResult) else result

    def _nested_call(self, name: str, arguments: Mapping[str, Any] | str | None) -> Any:
        if name == "apply_patch":
            if self._apply_patch is None:
                raise ValueError("apply_patch is not available")
            if not isinstance(arguments, str):
                raise ValueError("apply_patch expects raw patch text")
            return self._apply_patch(arguments)
        if isinstance(arguments, str) and name in ("exec_command", "write_stdin", "view_image"):
            arguments = {"cmd": arguments} if name == "exec_command" else {"chars": arguments} if name == "write_stdin" else {"path": arguments}
        args, error = parse_arguments(arguments)
        if error:
            raise ValueError(error)
        if name == "exec_command":
            return self._exec_command(args)
        if name == "write_stdin":
            return self._write_stdin(args)
        if name == "view_image":
            parts = self._view_image_parts(args)
            return {"detail": parts[0]["detail"], "image_url": parts[0]["image_url"]}
        if name == "clock__curr_time":
            from datetime import datetime, timezone
            return {"current_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}
        if name in self.STUB_MESSAGES:
            raise ValueError(self.STUB_MESSAGES[name])
        raise ValueError(f"unsupported call: tool {name} is not available")

    # --- function mode: function_call -> output (str or content parts)
    def run_function(self, call_id: str, name: str, raw_arguments: str | None) -> str | list[dict[str, Any]]:
        if self.allowed_tools is not None and name not in self.allowed_tools:
            error = f"unsupported call: tool {name} is not available"
            self.record_outer_error(error)
            return error
        args, error = parse_arguments(raw_arguments)
        if error:
            self.record_outer_error(error)
            return error
        if name == "exec_command":
            return self._record_tool(name, args, lambda: self._exec_command(args)).as_text()
        if name == "write_stdin":
            return self._record_tool(name, args, lambda: self._write_stdin(args)).as_text()
        if name == "view_image":
            try:
                parts = self._record_tool(name, args, lambda: self._view_image_parts(args))
            except ValueError as exc:
                self.record_outer_error(str(exc))
                return str(exc)
            self.image_paths[call_id] = [str(args.get("path", ""))]
            return parts
        if name in self.STUB_MESSAGES:
            self.record_outer_error(self.STUB_MESSAGES[name])
            return self.STUB_MESSAGES[name]
        error = f"unsupported call: tool {name} is not available"
        self.record_outer_error(error)
        return error

    def run_custom(self, call_id: str, name: str, raw_input: str) -> str | list[dict[str, Any]]:
        if self.allowed_tools is not None and name not in self.allowed_tools:
            error = f"unsupported call: tool {name} is not available"
            self.record_outer_error(error)
            return error
        if name == "apply_patch":
            if self._apply_patch is None:
                self.record_outer_error("apply_patch is not available")
                return "apply_patch is not available"
            return self._record_tool(name, raw_input, lambda: self._apply_patch(raw_input))
        error = f"unsupported call: tool {name} is not available"
        self.record_outer_error(error)
        return error

    # --- implementations
    def _view_image_parts(self, args: Mapping[str, Any]) -> list[dict[str, Any]]:
        parts = view_image(str(args.get("path", "")), self.workspace, args.get("detail"))
        if isinstance(parts, str):
            raise ValueError(parts)
        if self.runtime_contract is not None:
            error = image_data_error(parts[0]["image_url"])
            if error:
                raise ValueError(error)
        self.view_image_calls += 1
        self.image_sources[parts[0]["image_url"]] = str(args.get("path", ""))
        return parts

    def _exec_command(self, args: Mapping[str, Any]) -> ExecResult:
        cmd = args.get("cmd")
        if not isinstance(cmd, str) or not cmd:
            raise ValueError("exec_command failed: `cmd` is required")
        result = self.sessions.exec_command(
            cmd=cmd, workdir=args.get("workdir"), tty=bool(args.get("tty")),
            yield_time_ms=args.get("yield_time_ms"), max_output_tokens=args.get("max_output_tokens"),
            login=args.get("login", True), shell=args.get("shell"),
        )
        record = {"command": cmd, "exit_code": result.exit_code, "session_id": result.session_id,
                  "output_chars": len(result.output), "wall_time_s": result.wall_time_s}
        self.commands.append(record)
        if self._on_command:
            self._on_command(record)
        return result

    def _write_stdin(self, args: Mapping[str, Any]) -> ExecResult:
        try:
            session_id = int(args.get("session_id"))
        except (TypeError, ValueError):
            raise ValueError("write_stdin failed: `session_id` must be a number")
        return self.sessions.write_stdin(
            session_id=session_id, chars=str(args.get("chars") or ""),
            yield_time_ms=args.get("yield_time_ms"), max_output_tokens=args.get("max_output_tokens"),
        )


__all__ = ["sniff_image_mime", "valid_image_url", "DEFAULT_MAX_OUTPUT_TOKENS", "ExecResult", "PythonCellResult", "OUTPUT_TOKEN_LIMIT", "ToolRouter", "approx_tokens",
           "new_chunk_id", "parse_arguments", "truncate_output", "view_image"]

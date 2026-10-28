"""The `python` tool: one persistent Python session per episode (python-runtime/v1).

The model sends raw Python source; it runs in a long-lived worker process inside the sandbox
(react.python_worker), so variables, imports and helper functions persist across calls. A call can
be a single statement or a whole program. What the model gets back, as content parts:

    Cell N completed|failed|timed out|restarted in X.X s
    Output:
    <captured stdout/stderr, then "Out: <final expression>", truncated like every tool output>
    Error:
    <traceback trimmed to the cell's own frames>
    <epilogue: a per-cell summary printed by the session's start-up code, e.g. one line per API call>
    Changed: <names the cell created or rebound>
    [images attached with image(path) inside the cell]

The start-up code (what to import, the epilogue hook) and the tool description are supplied by the
caller; this module knows nothing about the task behind the session.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import queue
import secrets
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from captain_harness.react.sandbox import Sandbox
from captain_harness.react.tools import (
    OUTPUT_TOKEN_LIMIT,
    PythonCellResult,
    ToolRouter,
    image_data_error,
    new_chunk_id,
    truncate_output,
    view_image,
)

PYTHON_RUNTIME_CONTRACT_VERSION = "python-runtime/v1"
PYTHON_TOOL_NAME = "python"
WORKER = Path(__file__).with_name("python_worker.py")
MAX_IMAGES_PER_CELL = 4
DEFAULT_CELL_TIMEOUT_S = 600.0
KILL_GRACE_S = 15.0
START_TIMEOUT_S = 180.0
CAPTURE_CAP_BYTES = 1024 * 1024
MIN_CELL_TIMEOUT_S = 5.0
DEADLINE_MARGIN_S = 30.0
DEFAULT_DESCRIPTION = ("Run Python in a persistent Python 3 session whose working directory is the workspace. "
                       "The input is raw Python source (not JSON, not Markdown). Variables, imports and functions persist "
                       "across calls. image(path) attaches an image file to this call's result. This is the only tool.")
RUNTIME_NOTE = ("Tool note for this session: the only tool is `python`, a persistent Python 3 session in the workspace. "
                "Instructions above that mention exec, exec_command, shell commands, rg or apply_patch do not apply here; "
                "do everything in Python (for example open() to read or write files).")
RESTART_TEXT = "Python session restarted: the previous session stopped responding or exited; its variables were lost and the start-up code was re-run."


class ReplStartError(RuntimeError):
    """The worker could not be started or its start-up code failed."""


class PythonRepl:
    """Host side of one persistent worker; restarts it (losing the namespace) when it stops responding."""

    def __init__(self, *, sandbox: Sandbox, env: Mapping[str, str], python: str, startup: str = "",
                 kill_grace_s: float = KILL_GRACE_S, start_timeout_s: float = START_TIMEOUT_S) -> None:
        self.sandbox = sandbox
        self.env = dict(env)
        self.python = python
        self.startup = startup
        self.kill_grace_s = kill_grace_s
        self.start_timeout_s = start_timeout_s
        self.process: subprocess.Popen | None = None
        self.generation = 0
        self.cells = 0
        self.restarts = 0
        self.timeouts = 0
        self.closed = False
        self._nonce = ""
        self._replies: queue.Queue = queue.Queue()
        self._stderr_tail = b""
        self._lock = threading.Lock()

    # ------------------------------------------------------------ process
    def start(self) -> dict[str, Any]:
        if self.closed:
            raise ReplStartError("python session is closed")
        argv = self.sandbox.wrap([self.python, "-u", "-X", "utf8", str(WORKER)], cwd=self.sandbox.workspace)
        self._replies = queue.Queue()
        self._nonce = secrets.token_hex(16)
        self.process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        env=self.env, cwd=str(self.sandbox.workspace), start_new_session=True, close_fds=True)
        self.generation += 1
        threading.Thread(target=self._pump, args=(self.process, self._replies), daemon=True).start()
        threading.Thread(target=self._drain_stderr, args=(self.process,), daemon=True).start()
        self._send({"op": "init", "nonce": self._nonce, "startup": self.startup, "capture_cap": CAPTURE_CAP_BYTES})
        ready = self._wait(lambda m: m.get("op") == "ready", self.start_timeout_s)
        if ready is None or not ready.get("ok"):
            detail = (ready or {}).get("error") or self._stderr_tail.decode("utf-8", "replace")[-2000:] or "no reply"
            self._kill()
            raise ReplStartError(f"python session failed to start: {detail}")
        return ready

    def _pump(self, process: subprocess.Popen, replies: queue.Queue) -> None:
        assert process.stdout is not None
        for raw in process.stdout:
            try:
                message = json.loads(raw.decode("utf-8", "replace"))
            except ValueError:
                continue
            if isinstance(message, dict):
                replies.put(message)
        replies.put({"op": "eof"})

    def _drain_stderr(self, process: subprocess.Popen) -> None:
        assert process.stderr is not None
        for chunk in iter(lambda: process.stderr.read(4096), b""):
            self._stderr_tail = (self._stderr_tail + chunk)[-8192:]

    def _send(self, message: dict[str, Any]) -> None:
        assert self.process is not None and self.process.stdin is not None
        self.process.stdin.write((json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8"))
        self.process.stdin.flush()

    def _wait(self, accept: Callable[[dict[str, Any]], bool], timeout_s: float) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                message = self._replies.get(timeout=remaining)
            except queue.Empty:
                return None
            if message.get("op") == "eof":
                return None
            if message.get("nonce") == self._nonce and accept(message):
                return message

    def _kill(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass

    # ------------------------------------------------------------ cells
    def run(self, code: str, *, timeout_s: float = DEFAULT_CELL_TIMEOUT_S) -> PythonCellResult:
        with self._lock:
            self.cells += 1
            cell_id = self.cells
            if self.process is None or self.process.poll() is not None:
                restarted = self.process is not None or self.generation > 0
                self._kill()
                try:
                    self.start()
                except ReplStartError as exc:
                    return PythonCellResult(cell=cell_id, status="unavailable", output="", display=None, exception=str(exc),
                                            images=[], epilogue=None, changed=[], wall_s=0.0, generation=self.generation)
                if restarted:
                    self.restarts += 1
            started = time.monotonic()
            try:
                self._send({"op": "run", "id": cell_id, "nonce": self._nonce, "code": code, "timeout_s": float(timeout_s)})
                reply = self._wait(lambda m: m.get("op") == "result" and m.get("id") == cell_id, float(timeout_s) + self.kill_grace_s)
            except (BrokenPipeError, OSError):
                reply = None
            if reply is None:
                generation = self.generation
                self._kill()
                self.restarts += 1
                error = None
                try:
                    self.start()
                except ReplStartError as exc:
                    error = str(exc)
                return PythonCellResult(cell=cell_id, status="restarted", output="", display=None,
                                        exception=RESTART_TEXT + (f"\n{error}" if error else ""), images=[], epilogue=None,
                                        changed=[], wall_s=round(time.monotonic() - started, 3), generation=generation)
            if reply.get("status") == "timeout":
                self.timeouts += 1
            output = str(reply.get("output") or "")
            if reply.get("omitted_bytes"):
                output += f"\n[{reply['omitted_bytes']} bytes of output omitted by the session]"
            return PythonCellResult(cell=cell_id, status=str(reply.get("status") or "error"), output=output,
                                    display=reply.get("display"), exception=reply.get("exception"),
                                    images=[i for i in reply.get("images") or [] if isinstance(i, dict)],
                                    epilogue=reply.get("epilogue"), changed=[c for c in reply.get("changed") or [] if isinstance(c, dict)],
                                    wall_s=float(reply.get("wall_s") or 0.0), generation=self.generation)

    def close(self) -> dict[str, Any]:
        if not self.closed:
            self.closed = True
            if self.process is not None and self.process.poll() is None:
                try:
                    self._send({"op": "shutdown"})
                    self.process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            self._kill()
        return {"cells": self.cells, "restarts": self.restarts, "timeouts": self.timeouts, "generation": self.generation, "closed": True}


# ---------------------------------------------------------------- tool output
STATUS_WORDS = {"ok": "completed", "error": "failed", "timeout": "timed out", "restarted": "restarted", "rejected": "rejected",
                "unavailable": "could not run (the Python session is unavailable)"}


def strip_fence(source: str) -> tuple[str, str | None]:
    """A single Markdown fence around the whole input is removed (models trained on chat often add one)."""
    text = source.strip()
    if text.startswith("```") and text.endswith("```") and text.count("```") == 2:
        body = text[3:-3]
        first, _, rest = body.partition("\n")
        return (rest if first.strip().isidentifier() or not first.strip() else body).strip("\n") + "\n", "fence_stripped"
    return source, None


def _spill(spill_dir: Path | None, workspace: Path, raw: str, truncated: str) -> str:
    if spill_dir is None:
        return truncated
    try:
        spill_dir.mkdir(parents=True, exist_ok=True)
        path = spill_dir / f"{new_chunk_id()}.txt"
        path.write_text(raw, encoding="utf-8")
    except OSError:
        return truncated
    try:
        shown = path.relative_to(workspace).as_posix()
    except ValueError:
        shown = str(path)
    return (f"{truncated}\n[Full output saved to {shown} ({len(raw.encode('utf-8'))} bytes). Read only the part you need, "
            f"e.g. print(open('{shown}').read()[:4000])]")


def _images(result: PythonCellResult, router: ToolRouter, workspace: Path) -> tuple[list[dict[str, Any]], list[str]]:
    parts: list[dict[str, Any]] = []
    notes: list[str] = []
    root = workspace.resolve()
    for request in result.images:
        path = str(request.get("path") or "")
        if len(parts) >= MAX_IMAGES_PER_CELL:
            notes.append(f"image({path!r}) not attached: at most {MAX_IMAGES_PER_CELL} images per call")
            continue
        candidate = Path(path) if Path(path).is_absolute() else root / path
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved != root and root not in resolved.parents:
            notes.append(f"image({path!r}) not attached: only files inside the workspace can be attached")
            continue
        found = view_image(str(resolved), root, request.get("detail"))
        if isinstance(found, str):
            notes.append(found.replace("unable to locate image", "image() could not attach").replace("view_image accepts", "image() accepts"))
            continue
        error = image_data_error(found[0]["image_url"])
        if error:
            notes.append(f"image({path!r}) not attached: {error}")
            continue
        router.image_sources[found[0]["image_url"]] = path
        router.view_image_calls += 1
        parts.append(found[0])
    return parts, notes


def render_cell(result: PythonCellResult, *, router: ToolRouter, workspace: Path, output_limit_tokens: int = OUTPUT_TOKEN_LIMIT,
                spill_dir: Path | None = None) -> list[dict[str, Any]]:
    """Content parts for one python call; the audit keeps the untruncated output separately."""
    body = result.output
    if result.display is not None:
        body += ("" if not body or body.endswith("\n") else "\n") + f"Out: {result.display}"
    shown, _ = truncate_output(body, None, limit=output_limit_tokens)
    if shown != body:
        shown = _spill(spill_dir, workspace, body, shown)
    image_parts, notes = _images(result, router, workspace)
    header = ("Python call rejected" if result.status == "rejected"
              else f"Cell {result.cell} {STATUS_WORDS.get(result.status, result.status)} in {result.wall_s:.1f} s")
    shown = shown.rstrip("\n")
    lines = [header, "Output:" + ("\n" + shown if shown else " (none)")]
    if result.exception:
        lines += ["Error:", result.exception.rstrip("\n")]
    lines += notes
    if result.epilogue:
        lines.append(result.epilogue.rstrip("\n"))
    if result.changed:
        shown_changed = [f"{row.get('name')}={row.get('preview')}" for row in result.changed[:12]]
        more = len(result.changed) - len(shown_changed)
        lines.append("Changed: " + "; ".join(shown_changed) + (f"; +{more} more" if more > 0 else ""))
    return [{"type": "input_text", "text": "\n".join(line for line in lines if line is not None)}, *image_parts]


def run_python_cell(source: str, *, router: ToolRouter, repl: PythonRepl, timeout_s: float = DEFAULT_CELL_TIMEOUT_S,
                    output_limit_tokens: int = OUTPUT_TOKEN_LIMIT, spill_dir: Path | None = None,
                    deadline_monotonic: float | None = None) -> list[dict[str, Any]]:
    """Execute one `python` tool call (audited) and return its custom_tool_call_output parts."""
    code, normalized = strip_fence(source)
    effective = float(timeout_s)
    if deadline_monotonic is not None:
        effective = min(effective, deadline_monotonic - time.monotonic() - DEADLINE_MARGIN_S)
    arguments = {"code": source, "timeout_s": round(max(effective, 0.0), 1), "normalized": normalized}

    def invoke() -> PythonCellResult:
        if not code.strip():
            return PythonCellResult(cell=repl.cells, status="rejected", output="", display=None,
                                    exception="python needs Python source code as its input", images=[], epilogue=None,
                                    changed=[], wall_s=0.0, generation=repl.generation)
        if effective < MIN_CELL_TIMEOUT_S:
            return PythonCellResult(cell=repl.cells, status="rejected", output="", display=None,
                                    exception="the episode's wall-clock budget is used up; no further code runs", images=[],
                                    epilogue=None, changed=[], wall_s=0.0, generation=repl.generation)
        return repl.run(code, timeout_s=effective)

    result = router.record_python(arguments, invoke)
    if result.status != "ok":
        router.record_outer_error(f"python cell {result.status}")
    return render_cell(result, router=router, workspace=router.workspace, output_limit_tokens=output_limit_tokens, spill_dir=spill_dir)


# ---------------------------------------------------------------- request contract
def python_tool(description: str) -> dict[str, Any]:
    return {"type": "custom", "name": PYTHON_TOOL_NAME, "description": description, "format": {"type": "text"}}


def python_runtime_contract(body: dict[str, Any], items: list[dict[str, Any]], version: str, *, description: str | None = None,
                            note: str | None = RUNTIME_NOTE) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    """Adapt a request copy so `python` is the only tool; the captured templates stay untouched."""
    if version != PYTHON_RUNTIME_CONTRACT_VERSION:
        raise ValueError(f"unsupported runtime tool contract: {version}")
    body, items = copy.deepcopy(body), copy.deepcopy(items)
    tool = python_tool(description or DEFAULT_DESCRIPTION)
    placed = False
    for item in items:
        if item.get("type") == "additional_tools":
            functions = next((ns for ns in item.get("tools") or [] if ns.get("name") == "functions"),
                             {"type": "namespace", "name": "functions", "description": ""})
            item["tools"] = [] if placed else [{**functions, "tools": [tool]}]
            placed = True
    if placed:
        body.pop("tools", None)
    else:
        body["tools"] = [tool]
    body["parallel_tool_calls"] = False
    if note:
        first_user = next((i for i, it in enumerate(items) if it.get("type") == "message" and it.get("role") == "user"), len(items))
        items.insert(first_user, {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": note}]})
    from captain_harness.react.chat_llm import convert_tools

    payload, _ = convert_tools(body, items, allow=None)
    if [t["function"]["name"] for t in payload] != [PYTHON_TOOL_NAME]:
        raise ValueError("python runtime contract requires exactly one python tool")
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    return body, items, {"version": version, "sha256": digest}


__all__ = ["DEFAULT_CELL_TIMEOUT_S", "PYTHON_RUNTIME_CONTRACT_VERSION", "PYTHON_TOOL_NAME", "PythonRepl", "ReplStartError",
           "python_runtime_contract", "python_tool", "render_cell", "run_python_cell", "strip_fence"]

"""Codex code mode: the `exec` custom tool runs model-written JavaScript that calls nested tools.

The script runs in Node (inside the sandbox); nested tool calls come back to Python over a
loopback HTTP endpoint and are answered by the ToolRouter. The custom_tool_call_output is a
list of content parts: a header part ("Script completed\nWall time X seconds\nOutput:\n"),
then every text()/image() item, then a "Script error:" part when the script threw.
"""
from __future__ import annotations

import json
import copy
import hashlib
import re
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from captain_harness.react.sandbox import Sandbox
from captain_harness.react.tools import OUTPUT_TOKEN_LIMIT, ToolRouter, image_data_error, truncate_output, valid_image_url

RUNNER = Path(__file__).with_name("code_mode_runner.js")
PRAGMA = re.compile(r"^[ \t]*// @exec:(.*)$")
DEFAULT_SCRIPT_TIMEOUT_S = 900.0
NESTED_TOOLS = ("exec_command", "write_stdin", "view_image", "clock__curr_time", "wait", "request_user_input", "request_user_input_async")
RUNTIME_CONTRACT_VERSION = "code-mode-runtime/v2"
RUNTIME_TOOL_DESCRIPTIONS = {
    "exec_command": "Run a shell command in the workspace sandbox; return output, exit_code, or a running session_id.",
    "write_stdin": "Write to or poll a running shell session; return new output and its final exit_code when complete.",
    "view_image": "Read a local PNG/JPEG/GIF/WEBP/BMP image; return image_url and detail for image().",
    "apply_patch": "Apply raw *** Begin Patch / *** End Patch text to files inside the workspace.",
    "clock__curr_time": "Return the current UTC time.",
}
RUNTIME_EXEC_DESCRIPTION = """Run JavaScript as an async function to call the supported nested tools.
The input is raw JavaScript, not Python, JSON, or a Markdown code block.
Only the tools listed below exist. There are no deferred tools, MCP resource tools, or goal tools.
Each call runs to completion before returning; there are no background exec cells or wait tool.
The yield_time_ms exec pragma is accepted for compatibility but ignored; it does not return early.
To poll a shell process use tools.write_stdin.
The optional first-line pragma // @exec: {"max_output_tokens": 1000} limits each text output item.

Helpers:
- text(value): append a string or JSON-serialized value to the result.
- image(value, detail?): append a valid data-image URL or image object, usually from tools.view_image.
  Image bytes must decode successfully. External HTTP(S) images are unavailable.
- exit(): stop this script successfully.
- store(key, value), load(key): save and retrieve JSON-serializable values across exec calls in this episode.
  Local JavaScript variables do not persist. Stored values are copied as JSON, not shared objects.
- notify(value): same as text(value); output is buffered until this script finishes.
- yield_control(): a compatibility no-op; output remains buffered until this script finishes.
- ALL_TOOLS: the complete list of supported nested tools, with name and description fields.
- setTimeout and clearTimeout: timer helpers. Await asynchronous work before returning.

Nested tools (call these through tools, not as top-level function calls):
```ts
tools.exec_command({cmd: string, workdir?: string, shell?: string, login?: boolean,
  tty?: boolean, yield_time_ms?: number, max_output_tokens?: number})
// Returns {output, exit_code?, session_id?, chunk_id, wall_time_seconds, original_token_count}.
// A session_id means the process is still running; poll it with write_stdin.
tools.write_stdin({session_id: number, chars?: string, yield_time_ms?: number, max_output_tokens?: number})
// Returns the same result shape, containing only new output.
tools.view_image({path: string, detail?: "high" | "original"})
// Returns {image_url, detail}; display it with image(await tools.view_image(...)).
tools.apply_patch(patch: string)
// Raw patch text. Files must remain inside the workspace. Returns a success or verification-failure message.
tools.clock__curr_time({})
// Returns {current_time: string} in UTC.
```
Shell/file example: text(await tools.exec_command({cmd: "ls"}));
"""


def runtime_tool_contract(body: dict[str, Any], items: list[dict[str, Any]], version: str
                          ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    """Adapt a request copy, leaving the captured templates and legacy rendering intact."""
    if version != RUNTIME_CONTRACT_VERSION:
        raise ValueError(f"unsupported runtime tool contract: {version}")
    body, items = copy.deepcopy(body), copy.deepcopy(items)
    exec_tools = []
    for item in items:
        if item.get("type") == "additional_tools":
            namespaces = []
            for namespace in item.get("tools") or []:
                if namespace.get("name") != "functions":
                    continue
                supported = [tool for tool in namespace.get("tools") or [] if tool.get("name") == "exec"]
                if supported:
                    namespaces.append({**namespace, "tools": supported})
                    exec_tools.extend(supported)
            item["tools"] = namespaces
    if body.get("tools"):
        body["tools"] = [tool for tool in body["tools"] if tool.get("name") == "exec"]
        exec_tools.extend(body["tools"])
    if len(exec_tools) != 1:
        raise ValueError("runtime code-mode contract requires exactly one exec tool")
    exec_tools[0]["description"] = RUNTIME_EXEC_DESCRIPTION
    # Include the custom-input grammar suffix exactly as the Chat backend exposes it.
    from captain_harness.react.chat_llm import convert_tools

    payload, _ = convert_tools(body, items, allow=None)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    return body, items, {"version": version, "sha256": digest}


def node_binary() -> str | None:
    return shutil.which("node")


class _Endpoint:
    """Loopback HTTP endpoint the Node runner calls for nested tools."""

    def __init__(self, router: ToolRouter) -> None:
        router_ref = router
        audit_context = router.audit_context()

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # quiet
                return

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                try:
                    result = router_ref.nested_call(str(body.get("name")), body.get("args"), audit_context=audit_context)
                    payload = {"result": result, "error": None}
                except ValueError as exc:
                    payload = {"result": None, "error": str(exc)}
                except Exception as exc:  # noqa: BLE001 - surfaced to the script as an error
                    payload = {"result": None, "error": f"{type(exc).__name__}: {exc}"}
                data = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def parse_pragma(source: str) -> dict[str, Any]:
    first = source.split("\n", 1)[0]
    match = PRAGMA.match(first)
    if not match:
        return {}
    try:
        value = json.loads(match.group(1).strip() or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def run_exec_script(source: str, *, router: ToolRouter, sandbox: Sandbox, node: str | None = None,
                    timeout_s: float = DEFAULT_SCRIPT_TIMEOUT_S, env: dict[str, str] | None = None,
                    output_limit_tokens: int = OUTPUT_TOKEN_LIMIT) -> list[dict[str, Any]]:
    """Execute one `exec` tool call; returns the content-part list for custom_tool_call_output."""
    node = node or node_binary()
    if not node:
        router.record_outer_error("node is not available on this host")
        return [{"type": "input_text", "text": "Script failed\nWall time 0.0 seconds\nOutput:\n"},
                {"type": "input_text", "text": "Script error:\nnode is not available on this host"}]
    pragma = parse_pragma(source)
    modern = router.runtime_contract == RUNTIME_CONTRACT_VERSION
    max_output_tokens = pragma.get("max_output_tokens")
    endpoint = _Endpoint(router)
    started = time.monotonic()
    status, parts, error = "failed", [], None
    try:
        child_env = dict(env or {})
        child_env.update({"CODE_MODE_ENDPOINT": endpoint.url, "CODE_MODE_TOOLS": json.dumps(list(RUNTIME_TOOL_DESCRIPTIONS) if modern else list(NESTED_TOOLS))})
        if modern:
            child_env["CODE_MODE_PROTOCOL"] = "2"
            child_env["CODE_MODE_TOOL_DESCRIPTIONS"] = json.dumps(RUNTIME_TOOL_DESCRIPTIONS)
        stdin = json.dumps({"source": source, "store": router.code_mode_store}) if modern else source
        argv = sandbox.wrap([node, str(RUNNER)], cwd=sandbox.workspace)
        try:
            completed = subprocess.run(argv, input=stdin.encode("utf-8"), capture_output=True, timeout=timeout_s,
                                       env=child_env, cwd=str(sandbox.workspace))
            stdout = completed.stdout.decode("utf-8", errors="replace").strip()
            payload = json.loads(stdout.splitlines()[-1]) if stdout else {"status": "failed", "parts": [], "error": completed.stderr.decode("utf-8", errors="replace")[-2000:] or "runner produced no output"}
            if modern and completed.returncode != 0:
                # Late timer errors and unhandled rejections can occur after main() wrote its JSON.
                stderr = completed.stderr.decode("utf-8", errors="replace")[-2000:].strip()
                errors = [str(payload["error"])] if payload.get("error") else []
                errors.append(f"code-mode runner exited with code {completed.returncode}")
                if stderr:
                    errors.append(stderr)
                payload.update(status="failed", error="\n".join(errors))
        except subprocess.TimeoutExpired:
            payload = {"status": "failed", "parts": [], "error": f"script exceeded {timeout_s:.0f} s and was terminated"}
        status = payload.get("status", "failed")
        parts = payload.get("parts") or []
        error = payload.get("error")
        if modern and isinstance(payload.get("store"), dict):
            router.code_mode_store = payload["store"]
    finally:
        endpoint.close()
    wall = time.monotonic() - started
    header = {"type": "input_text", "text": f"Script {'completed' if status == 'completed' else 'failed'}\nWall time {wall:.1f} seconds\nOutput:\n"}
    rendered: list[dict[str, Any]] = [header]
    for part in parts:
        if part.get("type") == "input_text":
            text, _ = truncate_output(str(part.get("text", "")), max_output_tokens, limit=output_limit_tokens)
            rendered.append({"type": "input_text", "text": text})
        elif part.get("type") == "input_image" and modern:
            image_error = image_data_error(part.get("image_url"))
            if image_error:
                error = image_error
            else:
                rendered.append(part)
        elif part.get("type") == "input_image" and not valid_image_url(part.get("image_url")):
            rendered.append({"type": "input_text", "text": "[image dropped: not a valid image URL; use the image_url returned by tools.view_image on a PNG/JPEG file]"})
        else:
            rendered.append(part)
    if error:
        router.record_outer_error(str(error))
        rendered[0]["text"] = f"Script failed\nWall time {wall:.1f} seconds\nOutput:\n"
        rendered.append({"type": "input_text", "text": f"Script error:\n{error}"})
    return rendered


__all__ = ["NESTED_TOOLS", "RUNNER", "node_binary", "parse_pragma", "run_exec_script"]

"""Persistent Python session worker (runs inside the sandbox; stdlib only, Python >= 3.8).

The host (react.python_repl.PythonRepl) starts this file with the sandbox's python3 and talks to it
over newline-delimited JSON on the original stdin/stdout, which the worker moves to private file
descriptors at start-up. fd 0 becomes /dev/null and fd 1/2 point at an anonymous capture file, so
everything a cell prints (including subprocesses and C-level writes) is captured and can never
be mistaken for a protocol reply.

    host -> worker  {"op": "init", "nonce", "startup", "capture_cap"}
                    {"op": "run", "id", "nonce", "code", "timeout_s"}
                    {"op": "shutdown"}
    worker -> host  {"op": "ready", "nonce", "ok", "error"}
                    {"op": "result", "id", "nonce", "status", "output", "omitted_bytes", "display",
                     "exception", "images", "epilogue", "changed", "wall_s"}

Cells share one namespace for the whole session. A final expression's value is displayed; an
exception ends the cell with a traceback trimmed to the cell's own frames; a timeout raises
CellTimeout (a BaseException, so ``except Exception`` in model code cannot swallow it) and keeps
the namespace. The host kills and restarts the worker if a cell does not come back at all.
"""
from __future__ import annotations

import ast
import builtins
import inspect
import json
import linecache
import os
import signal
import sys
import tempfile
import time
import traceback
import types

DISPLAY_LIMIT = 2000
EPILOGUE_LIMIT = 2000
TRACEBACK_LINES = 20
MAX_IMAGES = 4
CHANGED_LIMIT = 12
PREVIEW_LIMIT = 60


class CellTimeout(BaseException):
    """Raised inside a cell when its time budget is used up."""


def _capture_fd() -> int:
    if hasattr(os, "memfd_create"):
        try:
            return os.memfd_create("cell-output")
        except OSError:
            pass
    fd, path = tempfile.mkstemp(prefix=".cell-", dir="/tmp")
    os.unlink(path)
    return fd


class Worker:
    def __init__(self) -> None:
        self.commands = os.fdopen(os.dup(0), "r", encoding="utf-8", buffering=1)
        self.replies = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
        null = os.open(os.devnull, os.O_RDONLY)
        os.dup2(null, 0)
        os.close(null)
        self.capture = _capture_fd()
        os.dup2(self.capture, 1)
        os.dup2(self.capture, 2)
        self.nonce = ""
        self.capture_cap = 1024 * 1024
        self.cell = 0
        self.images: list[dict] = []
        self.namespace: dict = {"__name__": "__main__", "__builtins__": builtins}
        self.baseline: dict[str, int] = {}
        sys.path[0] = os.getcwd()

    # ------------------------------------------------------------ protocol
    def reply(self, payload: dict) -> None:
        payload["nonce"] = self.nonce
        self.replies.write(json.dumps(payload, ensure_ascii=False, default=repr) + "\n")
        self.replies.flush()

    def serve(self) -> None:
        for line in self.commands:
            try:
                message = json.loads(line)
            except ValueError:
                continue
            op = message.get("op")
            if op == "init":
                self.nonce = str(message.get("nonce") or "")
                self.capture_cap = int(message.get("capture_cap") or self.capture_cap)
                self.reply({"op": "ready", **self.startup(str(message.get("startup") or ""))})
            elif op == "run" and message.get("nonce") == self.nonce:
                if isinstance(message.get("id"), int):
                    self.cell = message["id"] - 1  # the host numbers cells across restarts
                self.reply({"op": "result", "id": message.get("id"),
                            **self.run(str(message.get("code") or ""), float(message.get("timeout_s") or 600.0))})
            elif op == "shutdown":
                return

    # ------------------------------------------------------------ namespace
    def image(self, path, detail: str = "high"):
        """Attach a workspace image file (PNG/JPEG/...) to this cell's result; also accepts {"path": ...}."""
        if isinstance(path, dict):
            path = path.get("path")
        if isinstance(path, (list, tuple)):
            for item in path:
                self.image(item, detail)
            return None
        self.images.append({"path": os.fspath(path) if path is not None else "", "detail": detail if detail in ("high", "original") else "high"})
        return None

    def _inject(self) -> None:
        self.namespace["image"] = self.image
        self.namespace.setdefault("__builtins__", builtins)

    def _names(self) -> dict[str, int]:
        return {name: id(value) for name, value in self.namespace.items()
                if not name.startswith("_") and not isinstance(value, types.ModuleType)}

    def startup(self, code: str) -> dict:
        self._inject()
        error = None
        if code:
            result = self.run(code, 120.0, startup=True)
            if result["status"] != "ok":
                error = (result.get("exception") or "") + (result.get("output") or "")
        self.images = []
        self.baseline = self._names()
        return {"ok": error is None, "error": error}

    # ------------------------------------------------------------ one cell
    def _reset_capture(self) -> None:
        sys.stdout.flush()
        sys.stderr.flush()
        os.ftruncate(self.capture, 0)
        os.lseek(self.capture, 0, os.SEEK_SET)

    def _read_capture(self) -> tuple[str, int]:
        sys.stdout.flush()
        sys.stderr.flush()
        size = os.fstat(self.capture).st_size
        if size <= self.capture_cap:
            data = os.pread(self.capture, size, 0)
            omitted = 0
        else:
            half = self.capture_cap // 2
            data = os.pread(self.capture, half, 0) + b"\n...\n" + os.pread(self.capture, half, size - half)
            omitted = size - 2 * half
        return data.decode("utf-8", errors="replace"), omitted

    def _compile(self, source: str, filename: str):
        tree = ast.parse(source, filename=filename, mode="exec")
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
        body = compile(tree, filename, "exec")
        expression = compile(last, filename, "eval") if last is not None else None
        return body, expression

    def _format_exception(self, exc: BaseException) -> str:
        if isinstance(exc, SyntaxError):
            return "".join(traceback.format_exception_only(type(exc), exc)).rstrip()
        frames = [frame for frame in traceback.extract_tb(exc.__traceback__) if frame.filename.startswith("<cell ")]
        lines = ["Traceback (most recent call last):"] if frames else []
        for frame in frames:
            lines.append(f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}')
            if frame.line:
                lines.append(f"    {frame.line.strip()}")
        lines.extend("".join(traceback.format_exception_only(type(exc), exc)).rstrip().splitlines())
        if len(lines) > TRACEBACK_LINES:
            lines = lines[:2] + ["  ..."] + lines[-(TRACEBACK_LINES - 3):]
        return "\n".join(lines)

    def _display(self, value) -> str | None:
        if value is None:
            return None
        try:
            text = repr(value)
        except Exception as exc:  # noqa: BLE001 - a broken __repr__ must not fail the cell
            text = f"<unrepresentable {type(value).__name__}: {exc}>"
        return text if len(text) <= DISPLAY_LIMIT else text[:DISPLAY_LIMIT] + f"... [{len(text) - DISPLAY_LIMIT} more characters]"

    def _preview(self, value) -> str:
        if inspect.isfunction(value):
            try:
                return f"function{inspect.signature(value)}"
            except (TypeError, ValueError):
                return "function"
        try:
            text = repr(value)
        except Exception:  # noqa: BLE001
            text = "?"
        text = " ".join(text.split())
        kind = type(value).__name__
        if isinstance(value, (list, tuple, dict, set)):
            kind += f"[{len(value)}]"
        return f"{kind} {text[:PREVIEW_LIMIT]}{'...' if len(text) > PREVIEW_LIMIT else ''}"

    def _changed(self, before: dict[str, int]) -> list[dict]:
        rows = []
        for name, identity in self._names().items():
            if name == "image" or before.get(name) == identity:
                continue
            if name in self.baseline and self.baseline[name] == identity:
                continue
            rows.append({"name": name, "preview": self._preview(self.namespace[name])})
        return rows

    def _epilogue(self) -> str | None:
        hook = self.namespace.get("_cell_epilogue")
        if not callable(hook):
            return None
        try:
            text = hook()
        except Exception as exc:  # noqa: BLE001
            return f"(cell epilogue failed: {type(exc).__name__}: {exc})"
        if not text:
            return None
        text = str(text)
        return text if len(text) <= EPILOGUE_LIMIT else text[:EPILOGUE_LIMIT] + "..."

    def _on_alarm(self, signum, frame):
        raise CellTimeout()

    def run(self, source: str, timeout_s: float, *, startup: bool = False) -> dict:
        self._inject()
        self.images = []
        if not startup:
            self.cell += 1
        filename = "<startup>" if startup else f"<cell {self.cell}>"
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
        before = self._names()
        self._reset_capture()
        started = time.monotonic()
        status, exception, display = "ok", None, None
        previous = signal.signal(signal.SIGALRM, self._on_alarm)
        signal.setitimer(signal.ITIMER_REAL, max(0.05, timeout_s))
        try:
            try:
                body, expression = self._compile(source, filename)
                exec(body, self.namespace, self.namespace)  # noqa: S102 - this is the tool
                if expression is not None:
                    display = self._display(eval(expression, self.namespace, self.namespace))  # noqa: S307
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
        except CellTimeout:
            status, exception = "timeout", f"CellTimeout: the cell was interrupted after {timeout_s:.0f} s (variables are kept)"
        except KeyboardInterrupt:
            status, exception = "error", "KeyboardInterrupt"
        except SystemExit as exc:
            status, exception = "error", f"SystemExit: {exc.code!r} (the session keeps running)"
        except BaseException as exc:  # noqa: BLE001 - every model error is reported, never raised
            status, exception = "error", self._format_exception(exc)
        finally:
            signal.signal(signal.SIGALRM, previous)
        wall = time.monotonic() - started
        output, omitted = self._read_capture()
        self._inject()
        images, self.images = self.images[:], []
        return {"status": status, "output": output, "omitted_bytes": omitted, "display": display, "exception": exception,
                "images": images, "epilogue": None if startup else self._epilogue(),
                "changed": [] if startup else self._changed(before), "wall_s": round(wall, 3), "cell": self.cell}


def main() -> int:
    Worker().serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())

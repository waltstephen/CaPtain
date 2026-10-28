"""Session-based command execution mirroring Codex CLI's unified exec (exec_command / write_stdin).

A command runs under the sandbox as ``bash -lc <cmd>``; exec_command waits up to
``yield_time_ms`` (default 10 s, 250-30000 ms) and returns everything collected so far. If the
process is still running the caller gets a session id and polls with write_stdin, which returns
only the output produced since the previous read. Idle background sessions are killed after
300 s. stdout and stderr are merged; a PTY is used when ``tty`` is requested.
"""
from __future__ import annotations

import os
import pty
import secrets
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Mapping

from captain_harness.react.sandbox import Sandbox
from captain_harness.react.tools import ExecResult, OUTPUT_TOKEN_LIMIT, approx_tokens, new_chunk_id, truncate_output

MIN_YIELD_MS = 250
MAX_YIELD_MS = 30_000
DEFAULT_YIELD_MS = 10_000
DEFAULT_WRITE_YIELD_MS = 250
MIN_EMPTY_POLL_MS = 5_000
MAX_EMPTY_POLL_MS = 300_000
BACKGROUND_TIMEOUT_S = 300.0
MAX_SESSIONS = 64
OUTPUT_CAP_BYTES = 1024 * 1024
FORCED_ENV = {
    "NO_COLOR": "1", "TERM": "dumb", "LANG": "C.UTF-8", "LC_CTYPE": "C.UTF-8", "LC_ALL": "C.UTF-8",
    "COLORTERM": "", "PAGER": "cat", "GIT_PAGER": "cat", "GH_PAGER": "cat", "CODEX_CI": "1",
}
SECRET_PREFIXES = ("OPENAI_", "ANTHROPIC_", "HF_", "HUGGING_FACE_", "AWS_", "GOOGLE_", "WANDB_")
SECRET_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "CREDENTIAL")


def child_environment(base: Mapping[str, str], extra: Mapping[str, str] | None = None) -> dict[str, str]:
    env = {k: v for k, v in base.items() if k != "FRESH_ENV_PREFLIGHT_PATH" and not (k.startswith(SECRET_PREFIXES) or any(m in k for m in SECRET_MARKERS))}
    env.update(FORCED_ENV)
    if extra:
        env.update(extra)
    return env


class _HeadTailBuffer:
    """Keeps the first and last half of a byte cap; tracks how many bytes were dropped."""

    def __init__(self, cap: int = OUTPUT_CAP_BYTES) -> None:
        self.cap = cap
        self.head = bytearray()
        self.tail = bytearray()
        self.omitted = 0
        self.total = 0

    def append(self, data: bytes) -> None:
        self.total += len(data)
        half = self.cap // 2
        if len(self.head) < half:
            take = min(half - len(self.head), len(data))
            self.head += data[:take]
            data = data[take:]
        if data:
            self.tail += data
            if len(self.tail) > half:
                self.omitted += len(self.tail) - half
                del self.tail[: len(self.tail) - half]

    def snapshot(self) -> bytes:
        if not self.omitted:
            return bytes(self.head + self.tail)
        return bytes(self.head) + f"\n... {self.omitted} bytes omitted ...\n".encode() + bytes(self.tail)


class ExecSession:
    def __init__(self, session_id: int, process: subprocess.Popen, read_fd: int, write_fd: int | None) -> None:
        self.session_id = session_id
        self.process = process
        self.read_fd = read_fd
        self.write_fd = write_fd
        self.buffer = _HeadTailBuffer()
        self.unread = bytearray()
        self.lock = threading.Lock()
        self.new_output = threading.Event()
        self.exited = threading.Event()
        self.exit_code: int | None = None
        self.started = time.monotonic()
        self.last_touch = time.monotonic()
        self.reader = threading.Thread(target=self._pump, daemon=True)
        self.reader.start()

    def _pump(self) -> None:
        try:
            while True:
                try:
                    chunk = os.read(self.read_fd, 65536)
                except OSError:
                    chunk = b""
                if not chunk:
                    break
                with self.lock:
                    self.buffer.append(chunk)
                    self.unread += chunk
                self.new_output.set()
        finally:
            try:
                os.close(self.read_fd)
            except OSError:
                pass
            self.exit_code = self.process.wait()
            self.exited.set()
            self.new_output.set()

    def take_unread(self) -> bytes:
        with self.lock:
            data = bytes(self.unread)
            self.unread.clear()
            self.new_output.clear()
            return data

    def write(self, chars: str) -> None:
        if self.write_fd is None:
            raise ValueError("write_stdin failed: session has no stdin")
        try:
            os.write(self.write_fd, chars.encode("utf-8"))
        except OSError as exc:  # process already exited / pipe closed
            raise ValueError(f"write_stdin failed: {exc.strerror or exc}")

    def kill(self) -> None:
        try:
            os.killpg(self.process.pid, 9)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass


class SessionManager:
    def __init__(self, *, sandbox: Sandbox, env: Mapping[str, str], default_cwd: Path, default_shell: str = "/bin/bash",
                 output_limit_tokens: int = OUTPUT_TOKEN_LIMIT, spill_dir: Path | None = None) -> None:
        self.sandbox = sandbox
        self.env = dict(env)
        self.default_cwd = Path(default_cwd)
        self.default_shell = default_shell
        self.output_limit_tokens = int(output_limit_tokens)
        self.spill_dir = Path(spill_dir) if spill_dir is not None else None
        self.sessions: dict[int, ExecSession] = {}
        self._lock = threading.Lock()
        self._janitor = threading.Thread(target=self._reap_idle, daemon=True)
        self._janitor.start()

    # ------------------------------------------------------------------ public
    def exec_command(self, *, cmd: str, workdir: str | None = None, tty: bool = False, yield_time_ms=None,
                     max_output_tokens=None, login=True, shell: str | None = None) -> ExecResult:
        wait_ms = _clamp(yield_time_ms, DEFAULT_YIELD_MS, MIN_YIELD_MS, MAX_YIELD_MS)
        cwd = Path(workdir) if workdir else self.default_cwd
        if not cwd.is_absolute():
            cwd = self.default_cwd / cwd
        shell_path = shell or self.default_shell
        flag = "-lc" if login in (True, None) else "-c"
        argv = self.sandbox.wrap([shell_path, flag, cmd], cwd=cwd)
        session = self._spawn(argv, tty=tty, cwd=cwd)
        started = time.monotonic()
        self._wait(session, wait_ms / 1000.0, require_output=False)
        wall = time.monotonic() - started
        raw = session.take_unread().decode("utf-8", errors="replace")
        return self._result(session, raw, wall, max_output_tokens)

    def write_stdin(self, *, session_id: int, chars: str = "", yield_time_ms=None, max_output_tokens=None) -> ExecResult:
        session = self.sessions.get(int(session_id))
        if session is None:
            raise ValueError(f"write_stdin failed: unknown session id {session_id}")
        started = time.monotonic()
        if chars:
            session.write(chars)
            wait_ms = _clamp(yield_time_ms, DEFAULT_WRITE_YIELD_MS, MIN_YIELD_MS, MAX_YIELD_MS)
            self._wait(session, wait_ms / 1000.0, require_output=False)
        else:
            wait_ms = _clamp(yield_time_ms, MIN_EMPTY_POLL_MS, MIN_EMPTY_POLL_MS, MAX_EMPTY_POLL_MS)
            self._wait(session, wait_ms / 1000.0, require_output=True, minimum_s=MIN_EMPTY_POLL_MS / 1000.0)
        wall = time.monotonic() - started
        raw = session.take_unread().decode("utf-8", errors="replace")
        return self._result(session, raw, wall, max_output_tokens)

    def close_all(self) -> list[dict]:
        with self._lock:
            sessions = list(self.sessions.values())
            self.sessions.clear()
        results = []
        for session in sessions:
            # Reader completion also waits for descendants to close inherited output pipes.
            # Cleaning up those descendants must not erase the main command's natural exit.
            cancelled = session.process.poll() is None
            if not session.exited.is_set():
                session.kill()
            session.reader.join(timeout=5.0)
            results.append({"session_id": session.session_id, "exit_code": session.process.poll(),
                            "output": session.take_unread().decode("utf-8", errors="replace"),
                            "completed": session.exited.is_set(), "cancelled": cancelled})
        return results

    # ----------------------------------------------------------------- helpers
    def _spawn(self, argv, *, tty: bool, cwd: Path) -> ExecSession:
        with self._lock:
            if len(self.sessions) >= MAX_SESSIONS:
                raise ValueError("exec_command failed: too many live sessions")
        if tty:
            master, slave = pty.openpty()
            process = subprocess.Popen(argv, stdin=slave, stdout=slave, stderr=slave, env=self.env, cwd=str(cwd),
                                       start_new_session=True, close_fds=True)
            os.close(slave)
            session = ExecSession(process.pid, process, master, master)
        else:
            read_fd, write_fd = os.pipe()
            process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=write_fd, stderr=write_fd, env=self.env, cwd=str(cwd),
                                       start_new_session=True, close_fds=True)
            os.close(write_fd)
            session = ExecSession(process.pid, process, read_fd, process.stdin.fileno() if process.stdin else None)
        with self._lock:
            self.sessions[session.session_id] = session
        return session

    @staticmethod
    def _wait(session: ExecSession, timeout_s: float, *, require_output: bool, minimum_s: float = 0.0) -> None:
        deadline = time.monotonic() + timeout_s
        floor = time.monotonic() + minimum_s
        while True:
            now = time.monotonic()
            if session.exited.is_set():
                time.sleep(0.05)  # drain trailing output
                return
            remaining = deadline - now
            if remaining <= 0:
                return
            if require_output:
                if session.new_output.is_set() and now >= floor:
                    return
                session.new_output.wait(timeout=min(remaining, max(floor - now, 0.05)))
            else:
                session.exited.wait(timeout=remaining)
                if session.exited.is_set():
                    time.sleep(0.05)
                    return
                return

    def _result(self, session: ExecSession, raw: str, wall: float, max_output_tokens) -> ExecResult:
        session.last_touch = time.monotonic()
        output, original = truncate_output(raw, max_output_tokens, limit=self.output_limit_tokens)
        chunk = new_chunk_id()
        if self.spill_dir is not None and output != raw:
            output = self._spill(chunk, raw, output)
        if session.exited.is_set():
            with self._lock:
                self.sessions.pop(session.session_id, None)
            return ExecResult(chunk, wall, output, original, exit_code=session.exit_code,
                              raw_output=raw, audit_session_id=session.session_id)
        return ExecResult(chunk, wall, output, original, session_id=session.session_id,
                          raw_output=raw, audit_session_id=session.session_id)

    def _spill(self, chunk: str, raw: str, truncated: str) -> str:
        """Pi-style: the full output goes to a file the model can read; the truncated preview points at it."""
        try:
            self.spill_dir.mkdir(parents=True, exist_ok=True)
            path = self.spill_dir / f"{chunk}.txt"
            path.write_text(raw, encoding="utf-8")
        except OSError:
            return truncated
        lines = raw.count("\n") + (0 if raw.endswith("\n") or not raw else 1)
        try:
            shown = path.relative_to(self.default_cwd).as_posix()
        except ValueError:
            shown = str(path)
        return f"{truncated}\n[Full output saved to {shown} ({lines} lines, {len(raw.encode('utf-8'))} bytes). Read the part you need, e.g. sed -n '1,80p' {shown}; do not cat it whole.]"

    def _reap_idle(self) -> None:
        while True:
            time.sleep(5.0)
            now = time.monotonic()
            with self._lock:
                stale = [s for s in self.sessions.values() if not s.exited.is_set() and now - s.last_touch > BACKGROUND_TIMEOUT_S]
            for session in stale:
                session.kill()


def _clamp(value, default: int, low: int, high: int) -> int:
    try:
        number = int(float(value)) if value is not None else default
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def shell_command_string(cmd: str, shell: str = "/bin/bash", login: bool = True) -> str:
    """Render the command the way Codex logs it in `command_execution` items."""
    return f"{shell} {'-lc' if login else '-c'} {shlex.quote(cmd)}"


__all__ = ["SessionManager", "ExecSession", "child_environment", "shell_command_string", "FORCED_ENV"]

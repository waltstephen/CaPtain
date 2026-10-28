"""Atomic rollout publication and bounded reads across shared-filesystem clients."""
from __future__ import annotations

import errno
import json
import math
import os
import tempfile
import time
from pathlib import Path


class RolloutVisibilityTimeout(TimeoutError):
    pass


def write_rollout_json_atomic(path: str | Path, value: dict) -> None:
    """Publish only complete, fsynced JSON, then flush the directory entry."""
    path = Path(path)
    payload = json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def read_rollout_json(path: str | Path, *, timeout_s: float = 90.0,
                      initial_delay_s: float = 0.25, max_delay_s: float = 2.0) -> dict:
    """Retry only ENOENT or incomplete JSON until one absolute visibility deadline.

Schema failures and other I/O errors are not retried. A persistently corrupt
document receives the same explicit timeout as a persistently missing file.
"""
    for name, value in (("timeout_s", timeout_s), ("initial_delay_s", initial_delay_s), ("max_delay_s", max_delay_s)):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (name != "timeout_s" and value == 0):
            raise ValueError(f"invalid {name}: {value!r}")
    path = Path(path)
    started = time.monotonic()
    deadline = started + timeout_s
    delay = min(initial_delay_s, max_delay_s)
    attempts = 0
    last_error = None
    while attempts == 0 or time.monotonic() < deadline:
        attempts += 1
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            if exc.errno != errno.ENOENT:
                raise
            last_error = exc
        except json.JSONDecodeError as exc:
            last_error = exc
        else:
            if not isinstance(value, dict):
                raise ValueError(f"rollout JSON must be an object: {path}")
            return value
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(delay, remaining))
        delay = min(max_delay_s, delay * 2)
    elapsed = time.monotonic() - started
    detail = f"{type(last_error).__name__}: {last_error}" if last_error is not None else "visibility deadline exhausted"
    raise RolloutVisibilityTimeout(
        f"rollout unavailable or incomplete at {path} after {elapsed:.3f}s / {timeout_s:.3f}s deadline "
        f"({attempts} attempts; last error: {detail})"
    ) from last_error

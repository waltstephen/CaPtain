"""Prefix replay: re-issue a recorded episode's first K model responses, then hand over to a live model.

Used to restore a mid-episode state without simulator snapshots: the driver runs the recorded
responses (driver/responses.jsonl of the source episode) through the normal loop, so every tool call
executes again against a fresh environment, and request K+1 is the first one the live client sees.
Each replayed result carries the source usage (so compaction decisions repeat) and ``raw.replayed``.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable

from captain_harness.react.llm import ResponseResult


def load_responses(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if isinstance(row, dict) and isinstance(row.get("items"), list)]
    rows.sort(key=lambda row: int(row.get("request") or 0))
    return rows[:limit] if limit is not None else rows


class ReplayClient:
    """Wraps a model client; the first ``len(responses)`` calls return the recorded responses."""

    def __init__(self, inner: Any, responses: Iterable[dict[str, Any]]) -> None:
        self.inner = inner
        self.responses = [copy.deepcopy(r) for r in responses]
        self.served = 0

    @property
    def remaining(self) -> int:
        return len(self.responses) - self.served

    def respond(self, body: dict[str, Any]) -> ResponseResult:
        if self.served < len(self.responses):
            row = self.responses[self.served]
            self.served += 1
            return ResponseResult(response_id=row.get("response_id") or f"replay-{self.served}", items=copy.deepcopy(row["items"]),
                                  usage=copy.deepcopy(row.get("usage") or {}), status=str(row.get("status") or "completed"),
                                  latency_s=0.0, attempts=0, raw={"replayed": True, "source_request": row.get("request")})
        return self.inner.respond(body)


__all__ = ["ReplayClient", "load_responses"]

"""Host-only interface audit: structured facts from the validators that already decide a capability call.

The broker and adapters report what their existing checks concluded (which validator, which rule, which argument
path, accepted or rejected); a host observer bound for the duration of one call collects them. Without an observer
every ``record`` is a no-op, so validation behaviour and the student's visible results never change. This module
imports nothing from the server or the robot adapters.
"""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Mapping

SCHEMA_VERSION = "interface-audit/v1"
_EVENTS: ContextVar[list[dict[str, Any]] | None] = ContextVar("interface_audit_events", default=None)


def schema_sha256(schema: Mapping[str, Any] | None) -> str | None:
    if schema is None:
        return None
    return hashlib.sha256(json.dumps(schema, sort_keys=True, default=repr).encode("utf-8")).hexdigest()


def record(*, validation_source: str, phase: str, outcome: str, rule_id: str | None = None,
           argument_path: list[Any] | None = None, tool: str | None = None, validated_action: str | None = None,
           schema: Mapping[str, Any] | None = None, detail: str | None = None) -> None:
    """Report one validator conclusion. phase: input | output; outcome: accepted | rejected."""
    events = _EVENTS.get()
    if events is None:
        return
    events.append({"validation_source": validation_source, "validation_phase": phase, "outcome": outcome,
                   "rule_id": rule_id, "argument_path": [p if isinstance(p, (int, str)) else repr(p) for p in (argument_path or [])],
                   "tool": tool, "validated_action": validated_action, "schema_sha256": schema_sha256(schema),
                   "detail": (detail or "")[:300] or None})


def record_validation_error(exc: Any, *, validation_source: str, tool: str | None, schema: Mapping[str, Any] | None,
                            validated_action: str | None = None, path_prefix: list[Any] | None = None) -> None:
    """A jsonschema ValidationError from an input check: rule jsonschema.<keyword> at the failing argument path."""
    record(validation_source=validation_source, phase="input", outcome="rejected",
           rule_id=f"jsonschema.{getattr(exc, 'validator', 'unknown')}",
           argument_path=list(path_prefix or []) + list(getattr(exc, "absolute_path", []) or []),
           tool=tool, validated_action=validated_action, schema=schema, detail=getattr(exc, "message", str(exc)))


@contextmanager
def capture() -> Iterator[list[dict[str, Any]]]:
    """Collect the validator events raised while the body runs (one host capability call)."""
    events: list[dict[str, Any]] = []
    token = _EVENTS.set(events)
    try:
        yield events
    finally:
        _EVENTS.reset(token)

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ModelRequest:
    input_items: tuple[Mapping[str, object], ...]
    metadata: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CompactionRequest:
    previous_response_id: str
    task_state: str
    unresolved_failures: tuple[str, ...]
    retained_anchor_ids: tuple[str, ...]
    pending_tool_outputs: tuple[Mapping[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class ModelToolCall:
    call_id: str
    name: str
    arguments: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ModelResponse:
    response_id: str
    text: str
    metadata: Mapping[str, object]
    tool_calls: tuple[ModelToolCall, ...] = ()


@runtime_checkable
class ModelBackend(Protocol):
    def respond(
        self, request: ModelRequest, *, previous_response_id: str | None = None
    ) -> ModelResponse: ...

    def compact(self, request: CompactionRequest) -> ModelResponse: ...


class ModelRequestTimeout(TimeoutError):
    """A bounded model request ended, independently of the enclosing episode."""
    scope = "model_request"

    def __init__(self, message: str, *, attempts=(), timeout_s=None, elapsed_s=None):
        super().__init__(message)
        self.attempts = tuple(dict(item) for item in attempts)
        self.timeout_s = timeout_s
        self.elapsed_s = elapsed_s


class ModelOutputBudgetExhausted(RuntimeError):
    """An admitted generation ended at its output limit; no partial output is executable."""
    scope = "model_generation"

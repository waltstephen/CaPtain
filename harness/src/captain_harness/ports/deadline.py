"""Cooperative, context-local deadlines shared across synchronous boundaries."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
import math
import time


@dataclass(frozen=True)
class Deadline:
    scope: str
    started_s: float
    expires_s: float

    def remaining(self):
        return max(0., self.expires_s - time.monotonic())

    def receipt(self):
        now = time.monotonic()
        return {"scope": self.scope, "limit_s": self.expires_s - self.started_s,
            "elapsed_s": now - self.started_s, "remaining_s": max(0., self.expires_s - now),
            "expired": now >= self.expires_s}


class DeadlineExceeded(TimeoutError):
    def __init__(self, deadline, message="execution deadline expired", *, attempts=()):
        super().__init__(message)
        self.scope = deadline.scope
        self.deadline_receipt = deadline.receipt()
        self.attempts = tuple(dict(item) for item in attempts)


_ACTIVE = ContextVar("captain_execution_deadline", default=None)


def current_deadline():
    return _ACTIVE.get()


@contextmanager
def deadline_scope(scope, *, timeout_s=None, expires_s=None):
    now = time.monotonic()
    parent = current_deadline()
    if timeout_s is not None:
        if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        expires_s = min(expires_s, now + timeout_s) if expires_s is not None else now + timeout_s
    if expires_s is not None and not math.isfinite(expires_s):
        raise ValueError("deadline must be finite")
    selected = parent
    if expires_s is not None and (parent is None or expires_s < parent.expires_s):
        selected = Deadline(scope, now, expires_s)
    token = _ACTIVE.set(selected)
    try:
        yield selected
    finally:
        _ACTIVE.reset(token)


def check_deadline():
    deadline = current_deadline()
    if deadline is not None and deadline.remaining() <= 0:
        raise DeadlineExceeded(deadline)
    return deadline


def deadline_failure(deadline, *, executed=False, value=None):
    from .capability import CapabilityResult, FailureCategory
    return replace(CapabilityResult.failed(FailureCategory.TIMEOUT, "deadline_expired",
        executed=executed, value=value), diagnostics={"deadline": deadline.receipt()})


def retain_deadline_status(result, deadline):
    """Keep completed/partial execution truth when a synchronous call returns late."""
    from .capability import CapabilityResult, FailureCategory
    if deadline is None or deadline.remaining() > 0 or not isinstance(result, CapabilityResult):
        return result
    diagnostics = dict(result.diagnostics or {})
    diagnostics.setdefault("deadline", deadline.receipt())
    diagnostics.setdefault("result_before_deadline_classification", {
        "success": result.success, "executed": result.executed,
        "failure_category": result.failure_category.value, "message": result.message})
    if result.success:
        return replace(result, success=False, failure_category=FailureCategory.TIMEOUT,
            message="capability completed after its deadline", diagnostics=diagnostics)
    return replace(result, diagnostics=diagnostics)

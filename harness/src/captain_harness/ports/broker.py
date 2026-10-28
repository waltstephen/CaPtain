"""Manifest-validated capability invocation (moved from policy.python_executor in V1.1).

Both sides of a host capability call are validated against the manifest: the arguments
against ``arguments_schema`` before the invoker runs, the value against ``result_schema``
after it succeeds. Timeouts and backend exceptions become explicit failure receipts.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Callable, Mapping

from jsonschema import Draft202012Validator, ValidationError

from captain_harness.ports.capability import (
    CapabilityManifest,
    CapabilityResult,
    FailureCategory,
    UnknownCapabilityError,
)
from captain_harness.ports.deadline import deadline_failure, deadline_scope, retain_deadline_status
from captain_harness.ports import interface_audit

Invoker = Callable[[str, Mapping[str, object]], CapabilityResult]


class ValidatedCapabilityBroker:
    """Validate both sides of a manifest-declared host capability call."""

    def __init__(self, manifest: CapabilityManifest, invoker: Invoker) -> None:
        self.manifest = manifest
        self._invoker = invoker

    def invoke(self, name: str, arguments: Mapping[str, object]) -> CapabilityResult:
        try:
            spec = self.manifest.require(name)
        except UnknownCapabilityError as exc:
            interface_audit.record(validation_source="broker.manifest", phase="input", outcome="rejected",
                                   rule_id="unknown_capability", tool=name, detail=str(exc))
            return CapabilityResult.failed(FailureCategory.NOT_EXECUTED, str(exc))
        try:
            require_json_safe(arguments)
            Draft202012Validator(spec.arguments_schema).validate(dict(arguments))
        except (TypeError, ValueError, ValidationError) as exc:
            message = exc.message if isinstance(exc, ValidationError) else str(exc)
            if isinstance(exc, ValidationError):
                interface_audit.record_validation_error(exc, validation_source="broker.arguments_schema", tool=name,
                                                        schema=spec.arguments_schema)
            else:
                interface_audit.record(validation_source="broker.json_safe", phase="input", outcome="rejected",
                                       rule_id=f"json_safe.{type(exc).__name__}", tool=name, detail=message)
            return CapabilityResult.failed(
                FailureCategory.SCHEMA_ERROR,
                f"arguments schema validation failed: {message}",
            )
        interface_audit.record(validation_source="broker.arguments_schema", phase="input", outcome="accepted", tool=name,
                               schema=spec.arguments_schema)
        try:
            with deadline_scope("tool", timeout_s=spec.timeout_s) as deadline:
                if deadline.remaining() <= 0:
                    return deadline_failure(deadline)
                result = self._invoker(name, dict(arguments))
                result = retain_deadline_status(result, deadline)
        except TimeoutError as exc:
            return replace(
                CapabilityResult.failed(FailureCategory.TIMEOUT, str(exc) or "capability timed out", executed=True),
                diagnostics={
                    "deadline": getattr(exc, "deadline_receipt", None),
                    "execution_state": "unknown",
                    "physical_execution": "unknown; generic callback did not return a partial execution receipt",
                },
            )
        except Exception as exc:  # noqa: BLE001 - every backend failure must become a receipt
            return replace(
                CapabilityResult.failed(FailureCategory.BACKEND_ERROR, f"{type(exc).__name__}: {exc}", executed=True),
                diagnostics={"execution_state": "unknown"},
            )
        if not isinstance(result, CapabilityResult):
            return CapabilityResult.failed(
                FailureCategory.BACKEND_ERROR,
                "capability backend did not return CapabilityResult",
                executed=True,
            )
        if result.success:
            try:
                require_json_safe(result.value)
                Draft202012Validator(spec.result_schema).validate(result.value)
            except (TypeError, ValueError, ValidationError) as exc:
                message = exc.message if isinstance(exc, ValidationError) else str(exc)
                interface_audit.record(validation_source="broker.result_schema", phase="output", outcome="rejected",
                                       rule_id=f"jsonschema.{getattr(exc, 'validator', type(exc).__name__)}", tool=name,
                                       schema=spec.result_schema, detail=message)
                from captain_harness.ports.result_validation import result_validation_failure

                return result_validation_failure(result, message)
        return result


def require_json_safe(value: object, *, path: str = "$") -> None:
    """Raise TypeError/ValueError unless ``value`` is plain JSON (finite numbers, string keys)."""
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string object key")
            require_json_safe(item, path=f"{path}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            require_json_safe(item, path=f"{path}[{index}]")
        return
    raise TypeError(f"{path} contains non-JSON value {type(value).__name__}")


_require_json_safe = require_json_safe

__all__ = ["Invoker", "ValidatedCapabilityBroker", "require_json_safe"]

"""Retain the separately issued public envelope when a result value is invalid."""
from copy import deepcopy
from dataclasses import replace
from collections.abc import Mapping
import math

from .capability import FailureCategory


def _safe(value):
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(isinstance(k, str) and _safe(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return all(_safe(v) for v in value)
    return False


def result_validation_failure(result, message):
    # Never publish the malformed value or infer its physical outcome. These
    # envelope fields were issued independently by the trusted adapter.
    diagnostics = (deepcopy(dict(result.diagnostics))
        if isinstance(result.diagnostics, Mapping) and _safe(result.diagnostics) else {})
    diagnostics['result_validation'] = {
        'status': 'failed', 'phase': 'after_backend_return',
        'backend_reported_success': result.success,
        'backend_reported_executed': result.executed,
        'target_effect_status': 'unknown',
        'invalid_value_withheld': True,
        'scope': 'Result contract failure; execution and target effect are separate. '
                 'Use retained execution receipts and current public observations before deciding whether to retry.'}
    return replace(result, success=False, failure_category=FailureCategory.BACKEND_ERROR,
        message='result schema validation failed: '+message, value=None,
        diagnostics=diagnostics,
        attempt_identity=deepcopy(result.attempt_identity) if _safe(result.attempt_identity) else None,
        attempt_evidence=deepcopy(result.attempt_evidence) if _safe(result.attempt_evidence) else None)

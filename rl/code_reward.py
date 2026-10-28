"""python_exec_cost_v1: per-request code-execution cost from host audit records.

Every live model request of an episode is classified from what the host recorded when its Python cell(s) ran:

    syntax_invalid      1.0   SyntaxError / IndentationError / TabError: the cell never started executing
    runtime_invalid     0.5   the cell raised (NameError, KeyError, TypeError, AttributeError on the robot module, ...)
                              and no host tool call (observation, measurement or motion) ran during the request
    mixed_effect_error  0.25  the cell raised after at least one host tool call ran during the request
    ok                  0     the cell finished (tool failures handled by the program are not code errors)
    physical_failure    0     a RuntimeError/AssertionError carrying a robot motion/planning failure (collision, contact,
                              unreachable, joint limit, ...): judged by the task reward
    environment         -     verified environment / interpreter error: not scored
    unknown             -     no audit row, no traceback or an unrecognised status: not scored

A request with several cells takes the most severe cell; duplicate audit layers are not double counted. Correct code
earns nothing (no positive code reward). ``no_effect`` marks the pure errors (syntax/runtime invalid) whose positive
task credit the trainer withholds. Classification uses only host records: the driver's
tool audit (exception text and status), the executed-call log and the pre-tool request marks - never the model's
own claims. Requests are joined to the proxy's training records by response id, never by index.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

VERSION = "python_exec_cost_v1"
VERSION_V2 = "python_exec_contract_v2"
VERSIONS = (VERSION, VERSION_V2)
INTERFACE_COST = 0.5
# validator conclusions that decide an input contract violation;
# budget/finish/wall-time refusals and transport errors are recorded but never scored
CONTRACT_SOURCES = ("broker.arguments_schema", "broker.json_safe", "broker.manifest", "motion_argument_validator")
CONTRACT_EARLY = ("unknown_capability", "arguments_not_object")
COSTS = {"syntax_invalid": 1.0, "runtime_invalid": 0.5, "mixed_effect_error": 0.25, "ok": 0.0}
SEVERITY = ["ok", "physical_failure", "mixed_effect_error", "runtime_invalid", "syntax_invalid"]
COMPILE_ERRORS = ("SyntaxError", "IndentationError", "TabError")
NO_EFFECT = ("syntax_invalid", "runtime_invalid")
PHYSICAL_MARKERS = ("collision", "unreachable", "joint_limit", "contact_detected", "contact_stall", "controller_rejected",
                    "workspace", "inverse_kinematics", "ik_fail", "emergency_limit")
_EXC = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*)(?::|$)")


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def exception_type(text: str | None) -> str | None:
    for line in reversed((text or "").strip().splitlines()):
        m = _EXC.match(line.strip())
        if m:
            return m.group(1).rsplit(".", 1)[-1]
    return None


def classify_cell(end: dict, effect: bool) -> dict[str, Any]:
    """One audited python tool_end row -> {category, cost, exception_type, phase, reason}."""
    result = end.get("result") or {}
    status = result.get("status")
    if end.get("environment_error") is not None or end.get("error_class") == "environment":
        return {"category": "environment", "cost": None, "exception_type": None, "phase": None, "reason": "environment_error"}
    if status == "ok":
        return {"category": "ok", "cost": 0.0, "exception_type": None, "phase": None, "reason": "cell_completed"}
    if status != "error":
        return {"category": "unknown", "cost": None, "exception_type": None, "phase": None, "reason": f"status:{status}"}
    exc = exception_type(result.get("exception")) or exception_type(end.get("error"))
    if exc is None:
        return {"category": "unknown", "cost": None, "exception_type": None, "phase": None, "reason": "no_exception_text"}
    if exc in COMPILE_ERRORS:
        return {"category": "syntax_invalid", "cost": COSTS["syntax_invalid"], "exception_type": exc, "phase": "compile",
                "reason": "compile_failed_before_exec"}
    message = ((result.get("exception") or end.get("error") or "").strip().splitlines() or [""])[-1].lower()
    if exc in ("RuntimeError", "AssertionError") and any(k in message for k in PHYSICAL_MARKERS):
        # a motion/planning failure the robot reported (raised by the adapter or re-raised by the program): judged by the
        # task reward, never a code error
        return {"category": "physical_failure", "cost": 0.0, "exception_type": exc, "phase": "runtime", "reason": "physical_failure"}
    if exc in ("CellTimeout", "TimeoutError", "KeyboardInterrupt", "SystemExit", "MemoryError"):
        return {"category": "unknown", "cost": None, "exception_type": exc, "phase": "runtime", "reason": "interrupt_or_timeout"}
    category = "mixed_effect_error" if effect else "runtime_invalid"
    return {"category": category, "cost": COSTS[category], "exception_type": exc, "phase": "runtime",
            "reason": "raised_after_host_tool_call" if effect else "raised_without_host_tool_call"}


def request_windows(marks: list[dict], calls: list[dict]) -> dict[int, list[dict]]:
    """Executed host calls owned by each driver request: after its pre-tool mark's call_index, up to the next mark's."""
    bounds = sorted((int(m["request_index"]), int(m.get("call_index") or 0)) for m in marks
                    if type(m.get("request_index")) is int)
    owned: dict[int, list[dict]] = {}
    for i, (req, lo) in enumerate(bounds):
        hi = bounds[i + 1][1] if i + 1 < len(bounds) else 10 ** 9
        owned[req] = [c for c in calls if type(c.get("call_index")) is int and lo < c["call_index"] <= hi
                      and c.get("executed", True)]
    return owned


def episode_request_costs(episode_dir: Path) -> dict[int, dict[str, Any]]:
    """Driver request index -> classification of that live request (most severe of its cells)."""
    episode_dir = Path(episode_dir)
    audit = _jsonl(episode_dir / "driver" / "tool_audit.jsonl")
    marks = _jsonl(episode_dir / "hidden_request_marks.jsonl")
    calls = _jsonl(episode_dir / "calls.jsonl")
    windows = request_windows(marks, calls)
    ends: dict[int, list[dict]] = {}
    seen: set[str] = set()
    for row in audit:
        if row.get("type") != "tool_end" or row.get("tool_name") != "python":
            continue
        key = str(row.get("tool_call_id"))
        if key in seen:                                   # a duplicate audit line is one cell
            continue
        seen.add(key)
        ends.setdefault(int(row.get("request_index") or 0), []).append(row)
    out: dict[int, dict[str, Any]] = {}
    for req in sorted({int(r["index"]) for r in _jsonl(episode_dir / "driver" / "requests.jsonl") if not r.get("replayed")}):
        cells = ends.get(req, [])
        effect = bool(windows.get(req))
        if not cells:
            out[req] = {"category": "no_cell", "cost": 0.0, "scored": True, "no_effect": False, "cells": 0,
                        "reason": "no_python_cell", "exception_type": None, "phase": None}
            continue
        labels = [classify_cell(c, effect) for c in cells]
        if any(l["category"] in ("environment", "unknown") for l in labels) and not any(l["category"] in NO_EFFECT for l in labels):
            worst = next(l for l in labels if l["category"] in ("environment", "unknown"))
            out[req] = {**worst, "cost": 0.0, "scored": False, "no_effect": False, "cells": len(cells)}
            continue
        scored = [l for l in labels if l["cost"] is not None]
        worst = max(scored, key=lambda l: SEVERITY.index(l["category"]))
        out[req] = {**worst, "scored": True, "no_effect": worst["category"] in NO_EFFECT, "cells": len(cells),
                    "effect_calls": len(windows.get(req) or [])}
    return out


def proxy_request_costs(episode_dir: Path, proxy_requests: list[dict], version: str = VERSION) -> dict[str, Any]:
    """Proxy request index -> code label, joined through the driver's response ids; unmatched proxy requests are unscored.
    version python_exec_contract_v2 adds the host interface audit (confirmed input-contract rejections)."""
    episode_dir = Path(episode_dir)
    if version not in VERSIONS:
        raise ValueError(f"unknown code reward version {version}")
    driver = {str(r["response_id"]): int(r["index"]) for r in _jsonl(episode_dir / "driver" / "requests.jsonl")
              if r.get("response_id") and not r.get("replayed")}
    by_driver = episode_request_costs(episode_dir)
    interface = None
    if (episode_dir / "interface_audit.jsonl").exists() or version == VERSION_V2:
        interface = interface_attempts(episode_dir)          # measured for both versions; scored only by v2
        if version == VERSION_V2:
            by_driver = {req: v2_label(label, interface["requests"].get(req)) for req, label in by_driver.items()}
    labels: dict[int, dict[str, Any]] = {}
    unmapped = 0
    for q in proxy_requests:
        idx = q.get("request_index")
        rid = str(q.get("response_id") or "")
        if rid in driver and driver[rid] in by_driver:
            labels[int(idx)] = {**by_driver[driver[rid]], "driver_request_index": driver[rid]}
        else:
            unmapped += 1
            labels[int(idx)] = {"category": "unmapped", "cost": 0.0, "scored": False, "no_effect": False}
    counts: dict[str, int] = {}
    for label in by_driver.values():
        counts[label["category"]] = counts.get(label["category"], 0) + 1
    out = {"version": version, "labels": labels, "unmapped": unmapped, "category_counts": counts,
           "episode_cost_sum": sum(float(l["cost"] or 0.0) for l in by_driver.values() if l.get("scored"))}
    if interface is not None:
        reqs = interface["requests"]
        out["interface"] = {"attempts": interface["attempts"], "unattributed": interface["unattributed"],
                            "contract_rejections": sum(r["contract_rejections"] for r in reqs.values()),
                            "requests_with_rejection": sum(1 for r in reqs.values() if r["contract_rejections"]),
                            "recovered_within_response": sum(1 for r in reqs.values() if r["recovered_within_response"]),
                            "delta_cost_requests": sum(1 for l in by_driver.values() if l.get("delta_cost", 0) > 0)}
    return out


def interface_attempts(episode_dir: Path) -> dict[str, Any]:
    """interface-audit/v1 attempts of an episode grouped by driver request: confirmed input-contract rejections (one per
    attempt, however many validator layers reported it), all attempts and unattributable ones."""
    path = Path(episode_dir) / "interface_audit.jsonl"
    if not path.exists():
        raise ValueError("interface_audit_missing")
    by_request: dict[int, dict[str, Any]] = {}
    unattributed = 0
    seen: set[int] = set()
    for row in _jsonl(path):
        if row.get("schema_version") != "interface-audit/v1" or type(row.get("attempt_seq")) is not int:
            raise ValueError("interface_audit_invalid_row")
        if row["attempt_seq"] in seen:
            continue
        seen.add(row["attempt_seq"])
        req = row.get("driver_request_index")
        rejected = [e for e in row.get("events") or [] if e.get("validation_phase") == "input" and e.get("outcome") == "rejected"
                    and e.get("validation_source") in CONTRACT_SOURCES]
        contract = bool(rejected) or row.get("early_return") in CONTRACT_EARLY
        if type(req) is not int:
            unattributed += 1
            continue
        slot = by_request.setdefault(req, {"attempts": 0, "contract_rejections": 0, "rules": [], "tools": [],
                                           "effects": 0, "recovered_within_response": False})
        slot["attempts"] += 1
        if contract:
            slot["contract_rejections"] += 1
            slot["rules"].append((rejected[0].get("rule_id") if rejected else row.get("early_return")))
            slot["tools"].append(row.get("tool"))
        elif row.get("outcome") == "ok":
            slot["effects"] += 1
            if slot["contract_rejections"]:
                slot["recovered_within_response"] = True
    return {"requests": by_request, "unattributed": unattributed, "attempts": len(seen)}


def v2_label(v1: dict[str, Any], interface: dict[str, Any] | None) -> dict[str, Any]:
    """c_new = max(c_python, 0.5 if a confirmed input-contract rejection); the v1 no_effect mask is kept unchanged."""
    rejections = int((interface or {}).get("contract_rejections") or 0)
    c_python = float(v1.get("cost") or 0.0) if v1.get("scored") else 0.0
    c_interface = INTERFACE_COST if rejections else 0.0
    cost = max(c_python, c_interface) if v1.get("scored") or rejections else 0.0
    return {**v1, "cost": cost, "scored": bool(v1.get("scored") or rejections), "python_cost": c_python,
            "interface_cost": c_interface, "delta_cost": cost - c_python, "interface_rejections": rejections,
            "interface_rules": list((interface or {}).get("rules") or []),
            "recovered_within_response": bool((interface or {}).get("recovered_within_response"))}

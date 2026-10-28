#!/usr/bin/env python3
"""Check the episode shell toolchain in its real sandbox without a physical environment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import time


SCHEMA_VERSION = "harness-toolchain-preflight/v1"
RUNTIME_CONTRACT = "code-mode-runtime/v2"
PYTHON_RUNTIME_CONTRACT = "python-runtime/v1"
REQUIRED_EXECUTABLES = ("bash", "python3", "rg", "node")
# Mirrors captain_harness.react.toolchain_probe.REQUIRED_EXECUTABLES_BY_CONTRACT (asserted by a test).
REQUIRED_EXECUTABLES_BY_CONTRACT = {RUNTIME_CONTRACT: REQUIRED_EXECUTABLES, PYTHON_RUNTIME_CONTRACT: ("bash", "python3")}


def fingerprint(receipt):
    fields = ("schema_version", "runtime_tool_contract", "hostname", "required_executables", "executables", "sandbox",
              "workspace_import", "path_environment")
    if receipt.get("runtime_tool_contract") == PYTHON_RUNTIME_CONTRACT:
        fields += ("python_worker",)
    if "student_runtime" in receipt:
        fields += ("student_runtime",)   # receipts without a pinned student interpreter keep their fingerprint
    identity = {key: receipt.get(key) for key in fields}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_receipt(receipt, *, hostname=None, contract=None):
    if not isinstance(receipt, dict) or receipt.get("schema_version") != SCHEMA_VERSION or receipt.get("ok") is not True:
        raise ValueError("toolchain preflight is missing or failed")
    version = receipt.get("runtime_tool_contract")
    if version not in REQUIRED_EXECUTABLES_BY_CONTRACT or (contract is not None and version != contract):
        raise ValueError("toolchain preflight has a different runtime contract")
    required = REQUIRED_EXECUTABLES_BY_CONTRACT[version]
    if receipt.get("required_executables") != list(required):
        raise ValueError("toolchain preflight has a different executable requirement")
    if version == PYTHON_RUNTIME_CONTRACT and (receipt.get("python_worker") or {}).get("ok") is not True:
        raise ValueError("toolchain preflight did not verify the python session worker")
    if "student_runtime" in receipt and (receipt.get("student_runtime") or {}).get("ok") is not True:
        raise ValueError("toolchain preflight did not verify the student interpreter")
    if hostname is not None and receipt.get("hostname") != hostname:
        raise ValueError("toolchain preflight belongs to a different node")
    checks = receipt.get("executables") or {}
    for name in required:
        row = checks.get(name) or {}
        if (row.get("ok") is not True or row.get("exit_code") != 0 or not row.get("version")
                or not isinstance(row.get("resolved"), str) or not row["resolved"].startswith("/")):
            raise ValueError(f"toolchain preflight did not execute {name} successfully")
    if (receipt.get("sandbox") or {}).get("enabled") is not True or (receipt.get("workspace_import") or {}).get("ok") is not True:
        raise ValueError("sandbox or workspace import preflight did not pass")
    if receipt.get("fingerprint") != fingerprint(receipt):
        raise ValueError("toolchain preflight fingerprint is invalid")
    return receipt


def load_preflight_receipt(path, *, contract=None):
    if not path:
        raise ValueError("FRESH_ENV_PREFLIGHT_PATH is required")
    return validate_receipt(json.loads(Path(path).read_text()), contract=contract)


def probe(harness, contract=RUNTIME_CONTRACT, student_python=None):
    if contract not in REQUIRED_EXECUTABLES_BY_CONTRACT:
        raise ValueError(f"unknown runtime contract {contract!r}")
    harness = Path(harness).resolve()
    sys.path.insert(0, str(harness / "src"))
    from captain_harness.react.code_mode import node_binary
    from captain_harness.react.sandbox import Sandbox
    from captain_harness.react.toolchain_probe import SHELL_EXECUTABLES, probe_executable, probe_entrypoint, run_probe_command, environment_error
    from captain_harness.react.unified_exec import SessionManager, child_environment

    required = REQUIRED_EXECUTABLES_BY_CONTRACT[contract]
    receipt = {"schema_version": SCHEMA_VERSION, "runtime_tool_contract": contract,
               "hostname": socket.gethostname().split(".")[0], "checked_at": time.time(), "ok": False,
               "required_executables": list(required), "executables": {},
               "sandbox": {"enabled": False}, "workspace_import": {"ok": False},
               "path_environment": {name: os.environ.get(name) for name in ("PATH", "HOME", "SHELL")}}
    manager = None
    try:
        with tempfile.TemporaryDirectory(prefix="captain-toolchain-") as directory:
            episode = Path(directory)
            workspace = episode / "workspace"
            workspace.mkdir()
            stub = harness / "src/captain_harness/server/robot.py"
            shutil.copyfile(stub, workspace / "robot.py")
            sandbox = Sandbox.detect(workspace, hidden_dirs=[episode], require=True)
            receipt["sandbox"] = {"enabled": sandbox.enabled, "binary": sandbox.bwrap,
                                  "shell": "/bin/bash", "login": True}
            env = child_environment(os.environ, {"ROBOT_SERVER_URL": "http://127.0.0.1:1/not-a-service"})
            manager = SessionManager(sandbox=sandbox, env=env, default_cwd=workspace)
            for name in [n for n in SHELL_EXECUTABLES if n in required]:
                check = probe_executable(manager, name)
                receipt["executables"][name] = check
                if not check["ok"]:
                    receipt["environment_error"] = environment_error(name, check, source="tool_preflight")
                    return receipt
            if "node" in required:
                check = probe_entrypoint(sandbox, "node", node_binary(), env=env)
                receipt["executables"]["node"] = check
                if not check["ok"]:
                    receipt["environment_error"] = environment_error("node", check, source="tool_preflight")
                    return receipt
            command = """python3 - <<'CAPTAIN_IMPORT_PROBE'
import io, json
from unittest.mock import patch
payload = {"manifest": {"capabilities": [{"name": "observe", "argument_order": [], "arguments_schema": {"required": []}}]}}
def offline_manifest(*args, **kwargs):
    return io.BytesIO(json.dumps(payload).encode())
with patch("urllib.request.urlopen", side_effect=offline_manifest):
    import robot
    assert robot.capabilities() == ["observe"]
    assert callable(robot.observe)
print("CAPTAIN_WORKSPACE_IMPORT_OK")
CAPTAIN_IMPORT_PROBE"""
            result = run_probe_command(manager, command)
            passed = result["exit_code"] == 0 and "CAPTAIN_WORKSPACE_IMPORT_OK" in result["output"].splitlines()
            receipt["workspace_import"] = {"ok": passed, "exit_code": result["exit_code"],
                                             "source_sha256": hashlib.sha256(stub.read_bytes()).hexdigest(),
                                             "network": "urllib.request.urlopen replaced by in-memory manifest"}
            if not passed:
                receipt["environment_error"] = {"schema_version": "harness-environment-error/v1",
                    "code": "robot_stub_import_failed", "source": "tool_preflight", "verified": True,
                    "details": result}
                return receipt
            if contract == PYTHON_RUNTIME_CONTRACT:
                from captain_harness.react.toolchain_probe import probe_python_worker
                from captain_harness.server.python_session import OFFLINE_PROBE_STARTUP

                worker = probe_python_worker(sandbox, env, receipt["executables"]["python3"]["resolved"], startup=OFFLINE_PROBE_STARTUP)
                receipt["python_worker"] = worker
                if not worker["ok"]:
                    receipt["environment_error"] = {"schema_version": "harness-environment-error/v1",
                        "code": "python_session_unavailable", "source": "tool_preflight", "verified": True, "details": worker}
                    return receipt
                if student_python:
                    from captain_harness.react.toolchain_probe import probe_student_runtime
                    from captain_harness.server.python_session import STUDENT_RUNTIME_CELLS

                    student = probe_student_runtime(sandbox, env, str(student_python), startup=OFFLINE_PROBE_STARTUP,
                                                    cells=STUDENT_RUNTIME_CELLS)
                    receipt["student_runtime"] = {"python": str(student_python), "ok": student["ok"], "versions": student["versions"],
                                                  "cells": student["cells"], "worker_ok": bool(student["worker"].get("ok")),
                                                  "error": student["error"]}
                    if not student["ok"]:
                        receipt["environment_error"] = {"schema_version": "harness-environment-error/v1",
                            "code": "student_runtime_unavailable", "source": "tool_preflight", "verified": True,
                            "details": receipt["student_runtime"]}
                        return receipt
            receipt["ok"] = True
            receipt["fingerprint"] = fingerprint(receipt)
            return receipt
    except Exception as exc:
        receipt["environment_error"] = {"schema_version": "harness-environment-error/v1",
            "code": "sandbox_preflight_failed", "source": "tool_preflight", "verified": True,
            "details": {"error": f"{type(exc).__name__}: {exc}"}}
        return receipt
    finally:
        if manager is not None:
            manager.close_all()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--contract", choices=sorted(REQUIRED_EXECUTABLES_BY_CONTRACT), default=RUNTIME_CONTRACT)
    parser.add_argument("--student-python", default=None, help="python-runtime/v1: also verify this pinned student interpreter "
                        "(persistent session, NumPy/SciPy/Pillow, SDK failure result) in the real sandbox")
    args = parser.parse_args()
    receipt = probe(args.harness, args.contract, student_python=args.student_python)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0 if receipt["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Verify executable availability using the same shell and session interface as tools."""
from __future__ import annotations

import re
import time
import hashlib
import json
from pathlib import Path
import subprocess


REQUIRED_EXECUTABLES = ("bash", "python3", "rg", "node")
SHELL_EXECUTABLES = ("bash", "python3", "rg")
REQUIRED_EXECUTABLES_BY_CONTRACT = {
    "code-mode-runtime/v2": REQUIRED_EXECUTABLES,
    # python-runtime/v1: the worker is launched with the python3 that the login shell resolves
    "python-runtime/v1": ("bash", "python3"),
}
PREFLIGHT_SCHEMA = "harness-toolchain-preflight/v1"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")
_MARKER = "__CAPTAIN_EXECUTABLE__="


def preflight_fingerprint(receipt):
    fields = ("schema_version", "runtime_tool_contract", "hostname", "required_executables",
              "executables", "sandbox", "workspace_import", "path_environment")
    if receipt.get("runtime_tool_contract") == "python-runtime/v1":
        fields += ("python_worker",)   # code-mode receipts keep their original fingerprint
    if "student_runtime" in receipt:
        fields += ("student_runtime",)  # so do receipts without a pinned student interpreter
    identity = {key: receipt.get(key) for key in fields}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_preflight_receipt(receipt, contract=None):
    if not isinstance(receipt, dict) or receipt.get("schema_version") != PREFLIGHT_SCHEMA or receipt.get("ok") is not True:
        raise ValueError("toolchain preflight is missing, failed, or has an unsupported contract")
    version = receipt.get("runtime_tool_contract")
    if version not in REQUIRED_EXECUTABLES_BY_CONTRACT or (contract is not None and version != contract):
        raise ValueError("toolchain preflight is missing, failed, or has an unsupported contract")
    required = REQUIRED_EXECUTABLES_BY_CONTRACT[version]
    if receipt.get("required_executables") != list(required):
        raise ValueError("toolchain executable requirements differ")
    if version == "python-runtime/v1" and (receipt.get("python_worker") or {}).get("ok") is not True:
        raise ValueError("the python session worker was not verified")
    for name in required:
        row = (receipt.get("executables") or {}).get(name) or {}
        if (row.get("ok") is not True or row.get("exit_code") != 0 or not row.get("version")
                or not isinstance(row.get("resolved"), str) or not row["resolved"].startswith("/")):
            raise ValueError(f"toolchain did not execute {name} successfully")
    if (receipt.get("sandbox") or {}).get("enabled") is not True or (receipt.get("workspace_import") or {}).get("ok") is not True:
        raise ValueError("sandbox or workspace import was not verified")
    if receipt.get("fingerprint") != preflight_fingerprint(receipt):
        raise ValueError("toolchain preflight fingerprint is invalid")
    return receipt


def run_probe_command(sessions, command, *, login=True, shell=None, timeout_s=20):
    """Run a bounded probe without invoking tool callbacks or adding policy-visible history."""
    deadline = time.monotonic() + timeout_s
    result = sessions.exec_command(cmd=command, login=login, shell=shell,
                                   yield_time_ms=min(1000, timeout_s * 1000), max_output_tokens=1024)
    output = [result.output]
    while result.session_id is not None and time.monotonic() < deadline:
        result = sessions.write_stdin(session_id=result.session_id, chars="", yield_time_ms=1000,
                                      max_output_tokens=1024)
        output.append(result.output)
    if result.session_id is not None:
        session = sessions.sessions.get(result.session_id)
        if session is not None:
            session.kill()
        return {"exit_code": None, "output": "".join(output), "error_code": "probe_timeout"}
    return {"exit_code": result.exit_code, "output": "".join(output), "error_code": None}


def probe_executable(sessions, name, *, login=True, shell=None, timeout_s=20):
    """Resolve and actually execute a binary; a PATH hit alone is not a successful probe."""
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError("invalid executable name")
    command = (f"captain_probe_executable=$(command -v {name}) || exit 127\n"
               "case \"$captain_probe_executable\" in /*) ;; *) exit 126 ;; esac\n"
               f"printf '{_MARKER}%s\\n' \"$captain_probe_executable\"\n"
               '"$captain_probe_executable" --version')
    receipt = {"executable": name, "ok": False, "resolved": None, "version": None,
               "exit_code": None, "error_code": None, "probe_command": command,
               "login": bool(login), "shell": shell or sessions.default_shell}
    try:
        result = run_probe_command(sessions, command, login=login, shell=shell, timeout_s=timeout_s)
        receipt.update(exit_code=result["exit_code"], error_code=result["error_code"])
        lines = result["output"].splitlines()
        marker = next((i for i, line in enumerate(lines) if line.startswith(_MARKER)), None)
        if marker is not None:
            receipt["resolved"] = lines[marker][len(_MARKER):]
            receipt["version"] = "\n".join(lines[marker + 1:]).strip()[:4096]
        receipt["ok"] = bool(result["exit_code"] == 0 and receipt["resolved"] and receipt["version"])
        if not receipt["ok"]:
            receipt["error_code"] = receipt["error_code"] or "required_executable_unavailable"
            receipt["detail"] = result["output"][-4096:]
    except (OSError, ValueError, RuntimeError) as exc:
        receipt.update(error_code="required_executable_unavailable", detail=f"{type(exc).__name__}: {exc}")
    return receipt


def probe_entrypoint(sandbox, name, path, *, env, timeout_s=20):
    """Execute a host-resolved entrypoint directly, matching a sandboxed runner launch."""
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError("invalid executable name")
    receipt = {"executable": name, "ok": False, "resolved": path, "version": None,
               "exit_code": None, "error_code": "required_executable_unavailable",
               "resolution": "host_process_path", "probe_argv": [path, "--version"],
               "login": False, "shell": None}
    if not isinstance(path, str) or not Path(path).is_absolute():
        receipt["detail"] = "entrypoint was not resolved to an absolute path"
        return receipt
    try:
        argv = sandbox.wrap([path, "--version"], cwd=sandbox.workspace)
        result = subprocess.run(argv, capture_output=True, timeout=timeout_s, env=env,
                                cwd=str(sandbox.workspace), text=True)
        version = result.stdout.strip()[:4096]
        receipt.update(exit_code=result.returncode, version=version,
                       ok=result.returncode == 0 and bool(version))
        if receipt["ok"]:
            receipt["error_code"] = None
        else:
            receipt["detail"] = (result.stdout + result.stderr)[-4096:]
    except subprocess.TimeoutExpired:
        receipt.update(error_code="probe_timeout", detail="entrypoint probe timed out")
    except (OSError, ValueError, RuntimeError) as exc:
        receipt["detail"] = f"{type(exc).__name__}: {exc}"
    return receipt


def environment_error(executable, details, *, source="tool_runtime"):
    return {"schema_version": "harness-environment-error/v1", "code": "required_executable_unavailable",
            "executable": executable, "source": source, "verified": True, "details": details}


def probe_python_worker(sandbox, env, python, *, startup, timeout_s=60):
    """Start the real persistent session (python-runtime/v1) with the caller's offline start-up code and run four
    cells: printing, attaching an image, a timeout, and a read that proves the timeout kept the namespace."""
    import zlib
    from captain_harness.react.python_repl import PythonRepl

    receipt = {"ok": False, "python": python, "cells": [], "error": None}
    workspace = Path(sandbox.workspace)
    # 1x1 PNG, written by the host so the check does not depend on imaging libraries in the sandbox
    raw = b"\x00\xff\x00\x00"
    png = (b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", (1).to_bytes(4, "big") * 2 + b"\x08\x02\x00\x00\x00")
           + _png_chunk(b"IDAT", zlib.compress(raw)) + _png_chunk(b"IEND", b""))
    (workspace / "probe.png").write_bytes(png)
    repl = PythonRepl(sandbox=sandbox, env=env, python=python, startup=startup, kill_grace_s=5, start_timeout_s=timeout_s)
    try:
        repl.start()
        first = repl.run("probe_value = 41\nprint('CAPTAIN_PYTHON_OK')", timeout_s=timeout_s)
        second = repl.run("image('probe.png')\nprobe_value + 1", timeout_s=timeout_s)
        third = repl.run("import time\ntime.sleep(30)", timeout_s=1)
        fourth = repl.run("probe_value", timeout_s=timeout_s)
        receipt["cells"] = [{"status": c.status, "display": c.display, "images": len(c.images)} for c in (first, second, third, fourth)]
        receipt["ok"] = bool(first.status == "ok" and "CAPTAIN_PYTHON_OK" in first.output and second.status == "ok"
                             and second.display == "42" and len(second.images) == 1 and third.status == "timeout"
                             and fourth.status == "ok" and fourth.display == "41")
        receipt["startup_sha256"] = hashlib.sha256(startup.encode()).hexdigest()
    except Exception as exc:  # noqa: BLE001 - reported in the receipt
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        repl.close()
        (workspace / "probe.png").unlink(missing_ok=True)
    return receipt


def probe_student_runtime(sandbox, env, python, *, startup, cells, timeout_s=60):
    """A pinned interpreter in the real sandbox: the persistent-session checks of probe_python_worker, then the caller's
    ``cells`` - (name, source, expected display or None) run in order in one session, e.g. the declared libraries doing
    real work. The versions cell's display is recorded. A receipt that is not ok means the environment is broken."""
    from captain_harness.react.python_repl import PythonRepl

    receipt = {"ok": False, "python": python, "worker": probe_python_worker(sandbox, env, python, startup=startup, timeout_s=timeout_s),
               "cells": [], "versions": None, "error": None}
    workspace = Path(sandbox.workspace)
    present = set(workspace.iterdir())
    repl = PythonRepl(sandbox=sandbox, env=env, python=python, startup=startup, kill_grace_s=5, start_timeout_s=timeout_s)
    try:
        repl.start()
        ok = bool(receipt["worker"].get("ok"))
        for name, source, expected in cells:
            cell = repl.run(source, timeout_s=timeout_s)
            passed = cell.status == "ok" and (expected is None or cell.display == expected)
            receipt["cells"].append({"name": name, "status": cell.status, "display": cell.display, "ok": passed,
                                     "error": None if passed else (cell.output or "")[-500:]})
            if name == "versions" and cell.status == "ok":
                receipt["versions"] = cell.display
            ok = ok and passed
        receipt["ok"] = ok
    except Exception as exc:  # noqa: BLE001 - reported in the receipt
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        repl.close()
        for path in set(workspace.iterdir()) - present:   # files the cells wrote
            if path.is_file():
                path.unlink()
    return receipt


def _png_chunk(kind, data):
    import zlib

    return len(data).to_bytes(4, "big") + kind + data + zlib.crc32(kind + data).to_bytes(4, "big")

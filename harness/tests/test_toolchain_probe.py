from types import SimpleNamespace
import subprocess

import pytest

from captain_harness.react.toolchain_probe import probe_executable, probe_entrypoint, run_probe_command


class Sessions:
    default_shell = "/bin/bash"

    def __init__(self, output, exit_code=0):
        self.output = output
        self.exit_code = exit_code
        self.calls = []

    def exec_command(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output=self.output, exit_code=self.exit_code, session_id=None)


def test_probe_runs_version_in_the_selected_login_shell():
    sessions = Sessions("__CAPTAIN_EXECUTABLE__=/usr/bin/rg\nripgrep 14.1.1\n")
    row = probe_executable(sessions, "rg")
    assert row["ok"] and row["resolved"] == "/usr/bin/rg"
    assert row["version"] == "ripgrep 14.1.1"
    assert sessions.calls[0]["login"] is True
    assert '"$captain_probe_executable" --version' in sessions.calls[0]["cmd"]
    assert "command -v rg" in sessions.calls[0]["cmd"]


def test_path_hit_is_not_enough_when_the_binary_fails():
    sessions = Sessions("__CAPTAIN_EXECUTABLE__=/opt/bin/rg\nerror while loading shared libraries\n", 127)
    row = probe_executable(sessions, "rg", login=False, shell="/bin/sh")
    assert not row["ok"] and row["error_code"] == "required_executable_unavailable"
    assert sessions.calls[0]["login"] is False and sessions.calls[0]["shell"] == "/bin/sh"


def test_probe_does_not_accept_shell_error_text_as_a_success():
    assert not probe_executable(Sessions("bash: rg: command not found\n", 127), "rg")["ok"]
    assert not probe_executable(Sessions("__CAPTAIN_EXECUTABLE__=/usr/bin/rg\n", 0), "rg")["ok"]
    with pytest.raises(ValueError, match="executable name"):
        probe_executable(Sessions(""), "rg; exit 0")


def test_timeout_kills_only_the_probe_session(monkeypatch):
    killed = []
    sessions = Sessions("")
    sessions.sessions = {10: SimpleNamespace(kill=lambda: killed.append(10)),
                         11: SimpleNamespace(kill=lambda: killed.append(11))}
    sessions.exec_command = lambda **kwargs: SimpleNamespace(output="", exit_code=None, session_id=10)
    monkeypatch.setattr("captain_harness.react.toolchain_probe.time.monotonic", iter([0., 21.]).__next__)
    result = run_probe_command(sessions, "rg --version")
    assert result["error_code"] == "probe_timeout" and killed == [10]


def test_entrypoint_uses_the_host_resolved_path_without_a_login_shell(tmp_path, monkeypatch):
    calls = []
    sandbox = SimpleNamespace(workspace=tmp_path, wrap=lambda argv, cwd: ["sandbox", *argv])
    env = {"PATH": "/does-not-contain-node"}
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout="v22.16.0\n", stderr="")
    monkeypatch.setattr("captain_harness.react.toolchain_probe.subprocess.run", run)
    row = probe_entrypoint(sandbox, "node", "/opt/node/bin/node", env=env)
    assert row["ok"] and row["resolved"] == "/opt/node/bin/node"
    assert row["login"] is False and row["shell"] is None
    assert calls[0][0] == ["sandbox", "/opt/node/bin/node", "--version"]
    assert calls[0][1]["env"] == env and calls[0][1]["cwd"] == str(tmp_path)


def test_absolute_node_entrypoint_runs_when_child_path_cannot_resolve_node(tmp_path):
    from captain_harness.react.code_mode import node_binary
    from captain_harness.react.sandbox import Sandbox

    node = node_binary()
    if not node:
        pytest.skip("host node is unavailable")
    row = probe_entrypoint(Sandbox(workspace=tmp_path), "node", node, env={"PATH": "/no-executables"})
    assert row["ok"] and row["version"].startswith("v")


@pytest.mark.parametrize("path", [None, "node", "tools/node"])
def test_entrypoint_rejects_unresolved_paths(path, tmp_path):
    row = probe_entrypoint(SimpleNamespace(workspace=tmp_path), "node", path, env={})
    assert not row["ok"] and row["error_code"] == "required_executable_unavailable"


@pytest.mark.parametrize("failure", ["exit", "version", "timeout", "missing"])
def test_entrypoint_requires_successful_execution(failure, tmp_path, monkeypatch):
    sandbox = SimpleNamespace(workspace=tmp_path, wrap=lambda argv, cwd: argv)
    def run(argv, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        if failure == "missing":
            raise FileNotFoundError("entrypoint is absent")
        return SimpleNamespace(returncode=127 if failure == "exit" else 0, stdout="", stderr="failed")
    monkeypatch.setattr("captain_harness.react.toolchain_probe.subprocess.run", run)
    row = probe_entrypoint(sandbox, "node", "/opt/node/bin/node", env={})
    assert not row["ok"]
    assert row["error_code"] == ("probe_timeout" if failure == "timeout" else "required_executable_unavailable")


def _receipt(contract, executables, **extra):
    from captain_harness.react.toolchain_probe import preflight_fingerprint

    receipt = {"schema_version": "harness-toolchain-preflight/v1", "runtime_tool_contract": contract, "ok": True, "hostname": "h",
               "required_executables": list(executables),
               "executables": {n: {"ok": True, "exit_code": 0, "version": "1", "resolved": f"/usr/bin/{n}"} for n in executables},
               "sandbox": {"enabled": True}, "workspace_import": {"ok": True}, "path_environment": {}, **extra}
    receipt["fingerprint"] = preflight_fingerprint(receipt)
    return receipt


def test_receipts_are_validated_against_their_own_contract():
    from captain_harness.react.toolchain_probe import REQUIRED_EXECUTABLES_BY_CONTRACT, validate_preflight_receipt

    code = _receipt("code-mode-runtime/v2", REQUIRED_EXECUTABLES_BY_CONTRACT["code-mode-runtime/v2"])
    assert validate_preflight_receipt(code) is code
    python = _receipt("python-runtime/v1", ("bash", "python3"), python_worker={"ok": True})
    assert validate_preflight_receipt(python, contract="python-runtime/v1") is python
    with pytest.raises(ValueError, match="unsupported contract"):
        validate_preflight_receipt(python, contract="code-mode-runtime/v2")
    without_worker = _receipt("python-runtime/v1", ("bash", "python3"), python_worker={"ok": False})
    with pytest.raises(ValueError, match="python session worker"):
        validate_preflight_receipt(without_worker)
    python["python_worker"] = {"ok": True, "changed": 1}
    with pytest.raises(ValueError, match="fingerprint"):
        validate_preflight_receipt(python)


def test_code_mode_fingerprint_is_unchanged_by_the_python_contract():
    import hashlib
    import json

    from captain_harness.react.toolchain_probe import preflight_fingerprint

    receipt = _receipt("code-mode-runtime/v2", ("bash", "python3", "rg", "node"), python_worker={"ignored": True})
    fields = ("schema_version", "runtime_tool_contract", "hostname", "required_executables", "executables", "sandbox",
              "workspace_import", "path_environment")
    expected = hashlib.sha256(json.dumps({k: receipt.get(k) for k in fields}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert preflight_fingerprint(receipt) == expected


@pytest.mark.skipif(__import__("shutil").which("bwrap") is None or __import__("shutil").which("python3") is None,
                    reason="bwrap/python3 not available")
def test_python_worker_probe_runs_the_real_session_in_the_sandbox(tmp_path):
    import os
    import shutil

    from captain_harness.react.sandbox import Sandbox
    from captain_harness.react.toolchain_probe import probe_python_worker
    from captain_harness.react.unified_exec import child_environment
    from captain_harness.server.python_session import OFFLINE_PROBE_STARTUP

    workspace = tmp_path / "episode" / "workspace"
    workspace.mkdir(parents=True)
    shutil.copyfile(os.path.join(os.path.dirname(__file__), "..", "src", "captain_harness", "server", "robot.py"), workspace / "robot.py")
    sandbox = Sandbox.detect(workspace, hidden_dirs=[tmp_path / "episode"], require=True)
    receipt = probe_python_worker(sandbox, child_environment(os.environ, {"ROBOT_SERVER_URL": "http://127.0.0.1:1/x"}),
                                  shutil.which("python3"), startup=OFFLINE_PROBE_STARTUP)
    assert receipt["ok"], receipt
    assert [c["status"] for c in receipt["cells"]] == ["ok", "ok", "timeout", "ok"]

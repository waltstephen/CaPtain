"""Unified exec sessions, tool router, apply_patch and sandbox argv."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from captain_harness.react.apply_patch import apply_patch
from captain_harness.react.sandbox import Sandbox
from captain_harness.react.tools import ToolRouter, parse_arguments, view_image
from captain_harness.react.unified_exec import FORCED_ENV, SessionManager, child_environment


@pytest.fixture
def manager(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    sandbox = Sandbox(workspace=ws, bwrap=None)
    m = SessionManager(sandbox=sandbox, env=child_environment(os.environ), default_cwd=ws)
    yield m
    m.close_all()


def test_exec_command_returns_exit_code_and_merged_output(manager):
    result = manager.exec_command(cmd="echo out; echo err 1>&2; exit 3")
    assert result.exit_code == 3 and result.session_id is None
    assert "out\n" in result.output and "err\n" in result.output
    text = result.as_text()
    assert text.startswith(f"Chunk ID: {result.chunk_id}\nWall time: ") and "\nProcess exited with code 3\nOriginal token count: 2\nOutput:\n" in text


def test_exec_command_runs_in_workspace_with_forced_env(manager):
    result = manager.exec_command(cmd="pwd; echo $TERM $CODEX_CI $NO_COLOR")
    assert result.output == f"{manager.default_cwd}\n{FORCED_ENV['TERM']} 1 1\n"


def test_long_running_command_yields_session_then_polls(manager):
    result = manager.exec_command(cmd="echo first; sleep 1.2; echo second", yield_time_ms=300)
    assert result.exit_code is None and result.session_id is not None
    assert result.output == "first\n"
    poll = manager.write_stdin(session_id=result.session_id, chars="", yield_time_ms=5000)
    assert poll.exit_code == 0 and poll.output == "second\n"
    with pytest.raises(ValueError):
        manager.write_stdin(session_id=result.session_id)


def test_write_stdin_delivers_input(manager):
    result = manager.exec_command(cmd="read line; echo got:$line", yield_time_ms=250)
    assert result.session_id is not None
    reply = manager.write_stdin(session_id=result.session_id, chars="hello\n", yield_time_ms=2000)
    assert reply.exit_code == 0 and reply.output == "got:hello\n"


def test_write_stdin_to_exited_process_is_a_tool_error(manager):
    result = manager.exec_command(cmd="sleep 0.4", yield_time_ms=250)
    assert result.session_id is not None
    import time as _t
    _t.sleep(0.8)
    with pytest.raises(ValueError, match="write_stdin failed"):
        manager.write_stdin(session_id=result.session_id, chars="x\n", yield_time_ms=250)


def test_tty_mode_allocates_a_terminal(manager):
    result = manager.exec_command(cmd="test -t 1 && echo tty || echo notty", tty=True)
    assert result.output.strip() == "tty"


def test_secrets_are_dropped_from_child_environment():
    env = child_environment({"PATH": "/bin", "CLOUD_OPENAI_API_KEY": "x", "MY_TOKEN": "y", "HOME": "/h"}, {"ROBOT_SERVER_URL": "u"})
    assert "CLOUD_OPENAI_API_KEY" not in env and "MY_TOKEN" not in env
    assert env["PATH"] == "/bin" and env["ROBOT_SERVER_URL"] == "u" and env["PAGER"] == "cat"


def test_view_image_and_parse_arguments(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
    parts = view_image("images/a.png", tmp_path)
    assert parts[0]["type"] == "input_image" and parts[0]["image_url"].startswith("data:image/png;base64,") and parts[0]["detail"] == "high"
    assert view_image("images/nope.png", tmp_path) == "unable to locate image at `images/nope.png`: No such file or directory (os error 2)"
    assert parse_arguments('{"cmd": "ls"}') == ({"cmd": "ls"}, None)
    args, error = parse_arguments("{bad json")
    assert args is None and error.startswith("failed to parse function arguments: ")


def test_router_function_calls_and_stubs(manager):
    router = ToolRouter(sessions=manager, workspace=manager.default_cwd)
    out = router.run_function("c1", "exec_command", json.dumps({"cmd": "echo hi", "max_output_tokens": 100}))
    assert out.endswith("Output:\nhi\n") and router.commands[0]["command"] == "echo hi"
    assert router.run_function("c2", "request_user_input", "{}") == "request_user_input is unavailable in Default mode"
    assert router.run_function("c3", "teleport", "{}") == "unsupported call: tool teleport is not available"
    nested = router.nested_call("exec_command", {"cmd": "printf abc"})
    assert list(nested) == ["chunk_id", "wall_time_seconds", "exit_code", "original_token_count", "output"] and nested["output"] == "abc"
    with pytest.raises(ValueError, match="unable to locate image"):
        router.nested_call("view_image", {"path": "missing.png"})
    assert "UTC" in router.nested_call("clock__curr_time", {})["current_time"]


def test_apply_patch_add_update_move_delete(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    out = apply_patch("*** Begin Patch\n*** Add File: a/hello.py\n+print('hi')\n+print('there')\n*** End Patch\n", ws)
    assert out == "Success. Updated the following files:\nA a/hello.py" and (ws / "a/hello.py").read_text() == "print('hi')\nprint('there')\n"
    out = apply_patch("*** Begin Patch\n*** Update File: a/hello.py\n@@\n print('hi')\n-print('there')\n+print('world')\n*** End Patch\n", ws)
    assert out.endswith("M a/hello.py") and (ws / "a/hello.py").read_text() == "print('hi')\nprint('world')\n"
    out = apply_patch("*** Begin Patch\n*** Update File: a/hello.py\n*** Move to: b/hello.py\n@@\n-print('hi')\n+print('bye')\n*** End Patch\n", ws)
    assert "A b/hello.py" in out and not (ws / "a/hello.py").exists() and (ws / "b/hello.py").read_text() == "print('bye')\nprint('world')\n"
    assert apply_patch("*** Begin Patch\n*** Delete File: b/hello.py\n*** End Patch\n", ws) == "Success. Updated the following files:\nD b/hello.py"
    assert apply_patch("*** Begin Patch\n*** Add File: ../escape.txt\n+x\n*** End Patch\n", ws).startswith("apply_patch verification failed: ")
    assert apply_patch("*** Begin Patch\n*** Update File: missing.py\n@@\n-x\n*** End Patch\n", ws).startswith("apply_patch verification failed: ")
    assert apply_patch("no patch", ws).startswith("apply_patch verification failed: ")


def test_sandbox_argv_layout(tmp_path):
    out = tmp_path / "ep"
    ws = out / "workspace"
    ws.mkdir(parents=True)
    sandbox = Sandbox(workspace=ws, hidden_dirs=(out,), bwrap="/usr/bin/bwrap")
    argv = sandbox.wrap(["/bin/bash", "-lc", "ls"], cwd=ws)
    assert argv[:7] == ["/usr/bin/bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc"][:7]
    assert argv.index("--tmpfs") < argv.index("--bind", argv.index("--tmpfs"))  # hidden dir before workspace bind
    assert argv[argv.index("--tmpfs") + 1] == str(out)
    assert "--share-net" in argv and "--unshare-all" in argv and argv[-3:] == ["/bin/bash", "-lc", "ls"]
    assert Sandbox(workspace=ws, bwrap=None).wrap(["true"]) == ["true"]


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")
def test_bwrap_sandbox_hides_output_dir_and_keeps_network(tmp_path):
    out = tmp_path / "ep"
    ws = out / "workspace"
    ws.mkdir(parents=True)
    (out / "secret.txt").write_text("hidden")
    sandbox = Sandbox.detect(ws, hidden_dirs=[out])

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"ok")

        def log_message(self, *a):
            return

    server = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    cmd = f"ls {out} ; touch {ws}/w.txt && echo wrote; touch /usr/forbidden 2>/dev/null || echo ro; curl -s http://127.0.0.1:{port}/ || python3 -c \"import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:{port}/').read().decode())\""
    proc = subprocess.run(sandbox.wrap(["/bin/bash", "-lc", cmd]), capture_output=True, text=True, timeout=60)
    server.shutdown()
    if proc.returncode != 0 and "setting up uid map" in proc.stderr:
        pytest.skip(f"bwrap cannot create user namespaces here: {proc.stderr.strip()[:200]}")
    assert "secret.txt" not in proc.stdout and "wrote" in proc.stdout and "ro" in proc.stdout and "ok" in proc.stdout, proc.stdout + proc.stderr
    assert (ws / "w.txt").exists()


def test_view_image_rejects_non_image_files(tmp_path):
    from captain_harness.react.tools import sniff_image_mime, valid_image_url
    (tmp_path / "TASK.md").write_text("# task\nnot an image")
    msg = view_image("TASK.md", tmp_path)
    assert isinstance(msg, str) and msg.startswith("unable to view image at `TASK.md`: not an image file")
    (tmp_path / "shot.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)
    parts = view_image("shot.jpg", tmp_path)
    assert parts[0]["image_url"].startswith("data:image/jpeg;base64,")
    assert sniff_image_mime(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp" and sniff_image_mime(b"{\"a\": 1}") is None
    import base64 as _b64
    png_url = "data:image/png;base64," + _b64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8).decode()
    assert valid_image_url(png_url) and valid_image_url("https://x/y.png")
    assert not valid_image_url("rw_head.png") and not valid_image_url("data:text/markdown;base64,AAAA") and not valid_image_url(None)


def test_valid_image_url_checks_the_base64_payload():
    import base64
    from captain_harness.react.tools import valid_image_url
    png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8).decode()
    assert valid_image_url("data:image/png;base64," + png)
    assert not valid_image_url("data:image/png;base64," + png[:-3] + "$$$"), "non-base64 digit"
    assert not valid_image_url("data:image/png;base64," + base64.b64encode(b"# TASK.md").decode()), "valid base64 but not an image"
    assert not valid_image_url("data:image/png;base64,")

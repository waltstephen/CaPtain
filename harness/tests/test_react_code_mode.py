"""Code mode: model JavaScript calling nested tools through the Node runner (skipped without node)."""
from __future__ import annotations

import os
import shutil

import pytest

from captain_harness.react.code_mode import node_binary, parse_pragma, run_exec_script
from captain_harness.react.sandbox import Sandbox
from captain_harness.react.tools import ToolRouter
from captain_harness.react.unified_exec import SessionManager, child_environment

pytestmark = pytest.mark.skipif(node_binary() is None, reason="node not installed")


@pytest.fixture
def env(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "images").mkdir()
    (ws / "images" / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
    sandbox = Sandbox(workspace=ws, bwrap=None)
    sessions = SessionManager(sandbox=sandbox, env=child_environment(os.environ), default_cwd=ws)
    router = ToolRouter(sessions=sessions, workspace=ws)
    yield ws, sandbox, router, child_environment(os.environ)
    sessions.close_all()


def test_exec_script_collects_text_and_images(env):
    ws, sandbox, router, child_env = env
    script = ('text(await tools.exec_command({cmd:"echo hello","max_output_tokens":500}));'
              'image((await tools.view_image({path:"images/a.png"})).image_url);'
              'store("k", 1); text(load("k"));')
    parts = run_exec_script(script, router=router, sandbox=sandbox, env=child_env)
    assert parts[0]["type"] == "input_text" and parts[0]["text"].startswith("Script completed\nWall time ") and parts[0]["text"].endswith(" seconds\nOutput:\n")
    import json
    obj = json.loads(parts[1]["text"])
    assert list(obj) == ["chunk_id", "wall_time_seconds", "exit_code", "original_token_count", "output"] and obj["output"] == "hello\n"
    assert parts[2]["type"] == "input_image" and parts[2]["image_url"].startswith("data:image/png;base64,") and parts[2]["detail"] == "high"
    assert parts[3] == {"type": "input_text", "text": "1"}
    assert router.commands[0]["command"] == "echo hello" and router.view_image_calls == 1


def test_exec_script_error_and_pragma(env):
    ws, sandbox, router, child_env = env
    parts = run_exec_script('// @exec: {"max_output_tokens": 50}\ntext("before"); image((await tools.view_image({path:"images/none.png"})).image_url);', router=router, sandbox=sandbox, env=child_env)
    assert parts[0]["text"].startswith("Script failed\n")
    assert parts[1] == {"type": "input_text", "text": "before"}
    assert parts[-1]["text"] == "Script error:\nunable to locate image at `images/none.png`: No such file or directory (os error 2)"
    assert parse_pragma('// @exec: {"yield_time_ms": 10000, "max_output_tokens": 1000}\nfoo') == {"yield_time_ms": 10000, "max_output_tokens": 1000}
    assert parse_pragma("text(1)") == {}


def test_exit_helper_ends_script_successfully(env):
    ws, sandbox, router, child_env = env
    parts = run_exec_script('text("a"); exit(); text("b");', router=router, sandbox=sandbox, env=child_env)
    assert parts[0]["text"].startswith("Script completed") and [p["text"] for p in parts[1:]] == ["a"]


def test_image_helper_rejects_plain_strings_and_non_image_files(env):
    ws, sandbox, router, child_env = env
    (ws / "robot.py").write_text("def observe(): pass\n")
    parts = run_exec_script('image("rw_head.png");', router=router, sandbox=sandbox, env=child_env)
    assert parts[0]["text"].startswith("Script failed\n") and "image() requires a data:image/" in parts[-1]["text"] and "rw_head.png" in parts[-1]["text"]
    assert not any(p["type"] == "input_image" for p in parts)
    parts = run_exec_script('image((await tools.view_image({path:"robot.py"})).image_url);', router=router, sandbox=sandbox, env=child_env)
    assert parts[0]["text"].startswith("Script failed\n") and "not an image file" in parts[-1]["text"]
    assert not any(p["type"] == "input_image" for p in parts)


def test_image_helper_rejects_corrupt_base64_payload(env):
    ws, sandbox, router, child_env = env
    parts = run_exec_script('image("data:image/png;base64,iVBORw0KGgo$$$$");', router=router, sandbox=sandbox, env=child_env)
    assert parts[0]["text"].startswith("Script failed\n") and "base64 payload is invalid" in parts[-1]["text"]
    assert not any(p["type"] == "input_image" for p in parts)

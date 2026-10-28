"""TASK.md for the python tool (python-runtime/v1) and the per-cell robot receipts."""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from captain_harness.server.task_markdown import task_markdown

INFO = {"task": "put the bowl on the plate", "initial_images": [{"path": "/ep/workspace/images/obs_000_head.png", "role": "head", "camera": "agentview"}]}
MANIFEST = {"python_interface": "def observe() -> CapabilityResult[dict]:\n    \"\"\"Capture frames.\"\"\""}


def test_python_runtime_task_text():
    text = task_markdown(INFO, MANIFEST, wall_time_s=720, max_actions=150, max_tool_calls=600, runtime="python")
    assert "persistent Python 3 session" in text and "image(path)" in text and "There is no shell" in text
    assert "Run your code with `python3`" not in text and "image viewing tool" not in text
    assert "**Task: put the bowl on the plate**" in text and "def observe() -> dict:" in text


def test_shell_runtime_text_is_unchanged():
    text = task_markdown(INFO, MANIFEST, wall_time_s=720, max_actions=150, max_tool_calls=600)
    assert "Run your code with `python3` from this directory" in text
    assert "`{success, executed, failure_category, message, value,\n  diagnostics, budget}`" in text
    with pytest.raises(ValueError):
        task_markdown(INFO, MANIFEST, wall_time_s=720, max_actions=150, max_tool_calls=600, runtime="bash")


def _robot_module():
    payload = {"manifest": {"capabilities": [{"name": "move_to", "argument_order": ["position"], "arguments_schema": {"required": []}}]}}
    path = Path(__file__).resolve().parents[1] / "src" / "captain_harness" / "server" / "robot.py"
    spec = importlib.util.spec_from_file_location("robot_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    with patch("urllib.request.urlopen", side_effect=lambda *a, **k: io.BytesIO(json.dumps(payload).encode())), \
            patch.dict("os.environ", {"ROBOT_SERVER_URL": "http://127.0.0.1:1"}):
        spec.loader.exec_module(module)
    return module


def test_robot_client_collects_one_receipt_line_per_call():
    robot = _robot_module()
    replies = iter([{"success": True, "receipt": "move_to: ok; reached"}, {"success": False, "failure_category": "schema_error"},
                    {"finished": False, "rejected": True, "message": "measure the placement first"}])
    with patch.object(robot, "_post", side_effect=lambda route, payload: next(replies)):
        robot.move_to([0, 0, 0.1])
        robot.call("teleport")
        robot.finish("done")
    text = robot._cell_receipts()
    assert text.splitlines() == ["Robot calls in this cell:", "- move_to: ok; reached", "- teleport: FAILED schema_error",
                                 "- finish: rejected: measure the placement first"]
    assert robot._cell_receipts() is None


def test_robot_client_sends_array_like_arguments_as_plain_json():
    robot = _robot_module()

    class Array:          # the NumPy protocol the client relies on, without importing NumPy here
        def __init__(self, values):
            self.values = values

        def tolist(self):
            return list(self.values)

    class Scalar:
        def item(self):
            return 3

    sent = []

    def capture(request, *a, **k):
        sent.append(json.loads(request.data))
        return io.BytesIO(json.dumps({"success": True}).encode())

    with patch("urllib.request.urlopen", side_effect=capture), patch.dict("os.environ", {"ROBOT_SERVER_URL": "http://127.0.0.1:1"}):
        robot.move_to(Array([0.5, 0.25, 0.1]))
        robot.call("move_to", position=(1, 2, 3), steps=Scalar())
    assert sent == [{"name": "move_to", "arguments": {"position": [0.5, 0.25, 0.1]}},
                    {"name": "move_to", "arguments": {"position": [1, 2, 3], "steps": 3}}]
    with pytest.raises(TypeError, match="not JSON serializable"):
        robot.call("move_to", position=object())

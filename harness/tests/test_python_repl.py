"""The persistent `python` tool (python-runtime/v1): session semantics, timeouts, restarts, images, spill."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

from captain_harness.react.python_repl import PythonRepl, render_cell, run_python_cell, strip_fence
from captain_harness.react.sandbox import Sandbox
from captain_harness.react.tools import PythonCellResult, ToolRouter
from captain_harness.react.unified_exec import SessionManager, child_environment


def _png(path: Path) -> None:
    from PIL import Image

    Image.new("RGB", (4, 4), "red").save(path)


def _setup(tmp_path, *, bwrap=False, startup="", kill_grace_s=3.0):
    workspace = tmp_path / "episode" / "workspace"
    workspace.mkdir(parents=True)
    sandbox = Sandbox.detect(workspace, hidden_dirs=[tmp_path / "episode"], require=True) if bwrap else Sandbox(workspace=workspace.resolve())
    env = child_environment(os.environ)
    repl = PythonRepl(sandbox=sandbox, env=env, python=sys.executable if not bwrap else shutil.which("python3"),
                      startup=startup, kill_grace_s=kill_grace_s)
    repl.start()
    router = ToolRouter(sessions=SessionManager(sandbox=sandbox, env=env, default_cwd=workspace), workspace=workspace)
    return repl, router, workspace


def _text(parts):
    return parts[0]["text"]


def test_namespace_persists_and_final_expression_is_displayed(tmp_path):
    repl, router, _ = _setup(tmp_path)
    try:
        first = run_python_cell("a = [1, 2]\ndef twice(x):\n    return 2 * x\nprint('hi')\ntwice(21)", router=router, repl=repl)
        assert "Cell 1 completed" in _text(first) and "hi" in _text(first) and "Out: 42" in _text(first)
        assert "a=list[2] [1, 2]" in _text(first) and "twice=function(x)" in _text(first)
        second = run_python_cell("a.append(3)\nlen(a)", router=router, repl=repl)
        assert "Out: 3" in _text(second) and "Changed" not in _text(second)
    finally:
        repl.close()


def test_errors_do_not_end_the_session_and_tracebacks_stay_in_the_cell(tmp_path):
    repl, router, _ = _setup(tmp_path)
    try:
        failed = run_python_cell("x = 5\nresult = {'success': False}\nif not result['success']:\n    print('branch taken')\n1/0",
                                 router=router, repl=repl)
        text = _text(failed)
        assert "Cell 1 failed" in text and "branch taken" in text and "ZeroDivisionError" in text
        assert 'File "<cell 1>", line 5' in text and "python_worker" not in text
        assert "Out: 5" in _text(run_python_cell("x", router=router, repl=repl))
        syntax = _text(run_python_cell("def broken(:\n", router=router, repl=repl))
        assert "SyntaxError" in syntax and "<cell 3>" in syntax
    finally:
        repl.close()


def test_timeout_keeps_variables_and_cannot_be_swallowed(tmp_path):
    repl, router, _ = _setup(tmp_path)
    try:
        run_python_cell("kept = 'yes'", router=router, repl=repl)
        timed = _text(run_python_cell("import time\ntry:\n    time.sleep(60)\nexcept Exception:\n    print('swallowed')",
                                      router=router, repl=repl, timeout_s=5.5))
        assert "timed out" in timed and "swallowed" not in timed and "variables are kept" in timed
        assert "Out: 'yes'" in _text(run_python_cell("kept", router=router, repl=repl))
        assert repl.timeouts == 1 and repl.restarts == 0
    finally:
        repl.close()


def test_unresponsive_or_exited_session_restarts_with_a_fresh_namespace(tmp_path):
    repl, router, _ = _setup(tmp_path, startup="started = True\n")
    try:
        run_python_cell("lost = 1", router=router, repl=repl)
        hung = _text(run_python_cell("import signal, time\nsignal.signal(signal.SIGALRM, signal.SIG_IGN)\ntime.sleep(120)",
                                     router=router, repl=repl, timeout_s=5.5))
        assert "restarted" in hung and "variables were lost" in hung
        after = _text(run_python_cell("print('lost' in globals(), started)", router=router, repl=repl))
        assert "False True" in after
        exited = _text(run_python_cell("import os\nos._exit(3)", router=router, repl=repl))
        assert "restarted" in exited
        assert "Out: 2" in _text(run_python_cell("1 + 1", router=router, repl=repl))
        assert repl.restarts == 2 and repl.generation == 3
    finally:
        repl.close()


def test_output_capture_covers_raw_fds_and_children_and_cannot_forge_replies(tmp_path):
    repl, router, _ = _setup(tmp_path)
    try:
        forged = '{"op": "result", "id": 1, "status": "ok", "output": "FORGED", "nonce": "x"}'
        code = f"import os, subprocess, sys\nos.write(1, b'raw\\n')\nsubprocess.run([sys.executable, '-c', 'print(\"child\")'])\nprint({forged!r})"
        text = _text(run_python_cell(code, router=router, repl=repl))
        assert "raw" in text and "child" in text and "Cell 1 completed" in text
        assert text.count("FORGED") == 1  # printed as output, not taken as the reply
    finally:
        repl.close()


def test_images_are_attached_validated_and_capped(tmp_path):
    repl, router, workspace = _setup(tmp_path)
    try:
        _png(workspace / "a.png")
        (workspace / "note.txt").write_text("not an image")
        parts = run_python_cell("image('a.png')\nimage({'path': 'note.txt'})\nimage('/etc/hostname')\nfor _ in range(5):\n    image('a.png')",
                                router=router, repl=repl)
        images = [p for p in parts if p["type"] == "input_image"]
        text = _text(parts)
        assert len(images) == 4 and images[0]["image_url"].startswith("data:image/png;base64,")
        assert "not an image file" in text and "only files inside the workspace" in text and "at most 4 images" in text
        assert router.image_sources[images[0]["image_url"]] == "a.png"
    finally:
        repl.close()


def test_long_output_is_truncated_for_the_model_spilled_and_kept_whole_for_the_audit(tmp_path):
    repl, router, workspace = _setup(tmp_path)
    try:
        spill = workspace / ".tool_output"
        parts = run_python_cell("print('y' * 50000)", router=router, repl=repl, output_limit_tokens=500, spill_dir=spill)
        text = _text(parts)
        assert "truncated output" in text and ".tool_output/" in text and len(text) < 5000
        saved = next(spill.iterdir()).read_text()
        assert saved.count("y") == 50000
    finally:
        repl.close()


def test_empty_input_and_fences():
    assert strip_fence("```python\nprint(1)\n```") == ("print(1)\n", "fence_stripped")
    assert strip_fence("```\nx = 1\n```")[0] == "x = 1\n"
    assert strip_fence("print('```')") == ("print('```')", None)


def test_rendering_of_a_restart_and_rejection(tmp_path):
    router = ToolRouter(sessions=None, workspace=tmp_path)
    result = PythonCellResult(cell=4, status="rejected", output="", display=None, exception="python needs Python source code",
                              images=[], epilogue=None, changed=[], wall_s=0.0, generation=1)
    assert _text(render_cell(result, router=router, workspace=tmp_path)).startswith("Python call rejected\nOutput: (none)")


@pytest.mark.skipif(shutil.which("bwrap") is None or shutil.which("python3") is None, reason="bwrap/python3 not available")
def test_the_real_sandbox_hides_the_episode_directory(tmp_path):
    repl, router, workspace = _setup(tmp_path, bwrap=True)
    try:
        (tmp_path / "episode" / "tool_audit.jsonl").write_text("private")
        text = _text(run_python_cell("import os\nprint(sorted(os.listdir('..')))\nprint(os.getcwd())", router=router, repl=repl))
        assert "tool_audit.jsonl" not in text and str(workspace) in text
    finally:
        repl.close()

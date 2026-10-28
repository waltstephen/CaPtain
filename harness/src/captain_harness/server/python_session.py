"""Task side of the `python` tool (python-runtime/v1): start-up code, tool description, prompt.

react.python_repl runs a generic persistent Python session; this module tells it what the session
is for. The start-up code imports the workspace's robot.py client (every capability becomes a bare
function) and installs robot._cell_receipts as the per-cell epilogue, so each python call ends with
one line per robot call it made.
"""
from __future__ import annotations

STARTUP = """\
import robot
from robot import *
if not robot.capabilities():
    raise RuntimeError("robot capability manifest is unavailable")
_cell_epilogue = robot._cell_receipts
"""

# The start-up code with the capability manifest served from memory (no environment server): the toolchain
# preflight runs the real session worker with it to prove the sandboxed interpreter can import the client.
OFFLINE_PROBE_STARTUP = """\
import io, json
from unittest.mock import patch
_payload = {"manifest": {"capabilities": [{"name": "observe", "argument_order": [], "arguments_schema": {"required": []}}]}}
with patch("urllib.request.urlopen", side_effect=lambda *a, **k: io.BytesIO(json.dumps(_payload).encode())):
    import robot
    from robot import *
assert robot.capabilities() == ["observe"] and callable(observe)
_cell_epilogue = robot._cell_receipts
del io, json, patch
"""

# The student interpreter's declared libraries doing real work, and a structured SDK failure result, checked in the real
# sandbox before an episode (react.toolchain_probe.probe_student_runtime).
STUDENT_RUNTIME_CELLS = (
    ("versions", "import sys, numpy, scipy, PIL\n(sys.version.split()[0], numpy.__version__, scipy.__version__, PIL.__version__)", None),
    ("numpy_scipy", "import numpy as np\nfrom scipy.spatial.transform import Rotation\n"
                    "q = Rotation.from_euler('xyz', [0, 0, 90], degrees=True).as_quat()\n"
                    "(round(float(np.linalg.det(np.array([[1.0, 2.0], [3.0, 4.0]]))), 6), [round(float(v), 6) for v in q])",
     "(-2.0, [0.0, 0.0, 0.707107, 0.707107])"),
    ("pillow", "from PIL import Image\nImage.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save('student_probe.png')\n"
               "Image.open('student_probe.png').size", "(5, 4)"),
    ("sdk_failure", "import io, json\nfrom unittest.mock import patch\n"
                    "_fail = {'success': False, 'executed': False, 'failure_category': 'invalid_argument', 'message': 'probe'}\n"
                    "with patch('urllib.request.urlopen', side_effect=lambda *a, **k: io.BytesIO(json.dumps(_fail).encode())):\n"
                    "    _r = robot.call('observe')\n"
                    "(_r['success'], _r['failure_category'])", "(False, 'invalid_argument')"),
    ("sdk_numpy_arguments", "_sent = []\n"
                            "def _capture(request, *a, **k):\n"
                            "    _sent.append(json.loads(request.data))\n"
                            "    return io.BytesIO(json.dumps(_fail).encode())\n"
                            "with patch('urllib.request.urlopen', side_effect=_capture):\n"
                            "    robot.call('observe', position=np.array([0.5, 0.25]), steps=np.int64(3), scale=np.float32(0.5))\n"
                            "_sent[-1]['arguments']", "{'position': [0.5, 0.25], 'steps': 3, 'scale': 0.5}"),
)


PYTHON_PROMPT = ("Read TASK.md and complete the robot task it describes. Act only through the python tool: "
                 "`robot` is already imported in its persistent session. Call robot.finish() when you are done.")


def tool_description(cell_timeout_s: float) -> str:
    return (
        "Run Python in a persistent Python 3 session whose working directory is the task workspace. The input is raw "
        "Python source (not JSON, not Markdown). Variables, imports and functions persist across calls, so define helpers "
        "once and reuse them. `robot` is already imported and every capability is also a bare function "
        "(e.g. observe(), segment(...), move_to(...)). A call can be one statement or a whole program: compute poses from "
        "the measured numbers in variables instead of retyping them. Robot calls never raise on failure: they return a "
        "dict; check r[\"success\"] and branch in code. image(path) (or image(obs[\"value\"][\"images\"])) attaches up to 4 "
        "workspace images to this call's result. The result shows printed output, the value of a final expression, one "
        "receipt line per robot call made in the call, and the names it changed. A call is interrupted after "
        f"{cell_timeout_s:.0f} s (variables kept); if the session crashes it restarts and variables are lost. "
        "This is the only tool: there is no shell."
    )


HANDOFF_TEXT = (
    "Handoff: another operator started this episode and has already acted on the scene. The camera images above show the "
    "scene at the start of the episode; the images below show it now. You continue with a fresh Python session: `robot` is "
    "imported, but none of the earlier code, variables, notes or robot call history is available. Remaining budget: "
    "{requests} model requests, {actions} physical actions, {tool_calls} tool calls. Complete the whole task in TASK.md "
    "(every goal must hold at the end), then call robot.finish().")


def handoff_message(image_files, *, requests: int, budget: dict) -> dict:
    """The user message a clean handoff appends to the fresh initial prompt: the rule above and the current frames."""
    import base64
    import mimetypes
    from pathlib import Path

    text = HANDOFF_TEXT.format(requests=requests, actions=budget.get("physical_actions_remaining", "?"),
                               tool_calls=budget.get("tool_calls_remaining", "?"))
    content = [{"type": "input_text", "text": text}]
    for path in image_files:
        mime = mimetypes.guess_type(str(path))[0] or "image/png"
        content.append({"type": "input_image", "image_url": f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii"),
                        "detail": "high"})
    return {"type": "message", "role": "user", "content": content}


__all__ = ["HANDOFF_TEXT", "OFFLINE_PROBE_STARTUP", "STUDENT_RUNTIME_CELLS", "PYTHON_PROMPT", "STARTUP", "handoff_message", "tool_description"]

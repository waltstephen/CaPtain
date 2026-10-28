"""Render TASK.md for one episode from the public manifest (task text + API + scoring standards).

Ported from the Codex-CLI reference runner; the protocol-sentence filter is unchanged. The
"Standards" section carries the physical/scoring facts that the v6/v7 contract established
(placement 2 cm, containment, relation frame, articulated evidence, identity by RGB, handle
alignment, gated finish()).
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Mapping

PROTOCOL_SENTENCE = re.compile(
    r"suspend|resume\s*\(|continuation id|current continuation|last_action_result|checkpoint\s*\(|control block|Python stack|"
    r"persistent namespace|Functions, variables and loop locals persist|emit only|replacement Python|"
    r"under the host|fresh images to YOU|supplies fresh images|control\.|completion review|reflection|"
    r"\bHarness\b|image budget|revisable notes|Python workspace|frame IDs|host resolves|memory tools|"
    r"Explain the expected visual effect",
    re.IGNORECASE,
)


def _filter_sentences(body: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", body)
    return " ".join(s for s in sentences if s and not PROTOCOL_SENTENCE.search(s))


def strip_protocol(text: str) -> str:
    """Drop sentences that describe the retired episode-loop protocol, keep API semantics.

    Consecutive `# ` comment lines (the manifest usage instructions, hard-wrapped) are joined into
    paragraphs first so multi-line sentences are filtered whole, then re-wrapped."""
    lines = text.splitlines()
    out_lines: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("# "):
            block = []
            while i < len(lines) and lines[i].startswith("# "):
                block.append(lines[i][2:].strip())
                i += 1
            filtered = _filter_sentences(" ".join(block))
            if filtered.strip():
                out_lines.extend("# " + w for w in textwrap.wrap(filtered, width=108))
            continue
        out_lines.append(lines[i])
        i += 1
    text = "\n".join(out_lines)
    out_lines = []
    for line in text.splitlines():
        # Only the free-text description line of each docstring is filtered; signatures,
        # argument constraints, examples and return shapes are kept verbatim.
        m = re.match(r'^(\s+""")(.+)$', line)
        if m and m.group(2).strip():
            out_lines.append(m.group(1) + _filter_sentences(m.group(2)))
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


def rewrite_interface(python_interface: str) -> str:
    text = python_interface
    text = text.replace(
        "# Every call returns CapabilityResult: .success, .executed, .failure_category, .message, .value.\n"
        "# Optional .diagnostics retains deadline scope, service attempts and measured partial execution.\n"
        "# A timed-out operation may have partially executed; inspect its receipt before retrying.",
        "# Every call returns a dict with keys success, executed, failure_category, message, value, budget.\n"
        "# Optional diagnostics retains deadline scope, service attempts and measured partial execution.\n"
        "# A timed-out operation may have partially executed; inspect its result before retrying.",
    )
    text = re.sub(r"-> CapabilityResult\[[^\]]*\]:", "-> dict:", text)
    text = text.replace("Returns .value:", "Returns value:")
    return strip_protocol(text)


STANDARDS = """## Standards the final scene is judged by
- Placement ("on"): the object's footprint center must rest within 0.02 m (XY) of the destination's
  footprint center. After the final release, call `measure_object_geometry(target_detection_id,
  destination_detection_id)` on a fresh observation and read `object_to_destination_xy_offset_m`;
  a detection score below 0.02 makes the measurement untrustworthy.
- Containment ("in" / "into"): the object's center must lie inside the container's footprint with a
  consistent resting height (`object_center_inside_destination_footprint`, `resting_height_consistent`).
- Relations "left / right / front / back of X" are judged in the head camera (`agentview`) view of the
  table: front = +x (toward the camera), right = +y, within about 0.10 m of the reference object.
- Articulated targets (open / close a drawer or door): detect the moving part before acting, then after
  acting measure `measure_object_geometry(after_detection_id, before_detection_id=before_detection_id)`
  on the same camera and check `part_motion_significant`. Knobs (turn on / off the stove) are judged by
  rotation, not displacement: the same before/after measurement must show `part_rotation_rad` of at
  least 0.5 rad; pushing the knob sideways does not turn it.
- Tasks with several goals ("put A on X and B on Y"): every placed object needs its own fresh
  `measure_object_geometry(object, its destination)` after your last physical action; one measurement
  cannot vouch for two goals.
- Object identity is your call from the RGB images (read printed labels, colors, shapes); text-prompted
  detections are hypotheses that confuse similar packages. Pin the instance you mean with
  `mark_point(observation_handle, camera_name, x_norm, y_norm, label=...)`, which returns a selected
  `detection_id` you can measure and act on; for packaged foods confirm the identity in the wrist
  camera before grasping.
- Verify the instance before you grasp it. The task names the target through a reference ("the bowl
  on the ramekin", "the bowl between the plate and the ramekin", "the middle drawer"): detect the
  reference too and measure the relation with `measure_object_geometry(target_detection_id,
  reference_detection_id)` (offset ~0 for "on", roughly equal distances for "between", rank by height
  among all handles for top/middle/bottom). Print the numbers; if they contradict the task, you have
  the wrong instance and must re-select before any physical action. A measurement of the wrong
  instance is self-consistent all the way to `finish()` and is the most common way to fail.
- Handles: close the jaws across the handle's short axis (closing axis perpendicular to the footprint's
  long axis, `yaw_rad`), at handle height.
- `robot.finish()` checks your own recorded measurements against these standards. If it returns
  `{"finished": false, "rejected": true, "message": ...}` the episode continues: read the message, fix
  the scene or take the missing measurement, then call `finish()` again (after two rejections the
  next `finish()` is accepted as-is).
"""


SHELL_ACT = """- All interaction goes through `robot.py` in this directory (Python 3 standard library only, no
  extra packages needed): `import robot` then e.g. `robot.observe()`, or `from robot import *`.
  Run your code with `python3` from this directory (the server address is in `.robot_server` and
  in the `ROBOT_SERVER_URL` environment variable).
- Every capability returns a plain dict `{success, executed, failure_category, message, value,
  diagnostics, budget}`. Always check `success` before using `value`; a failed physical request
  returns `success=False` with a `failure_category`, a `message` and the measured receipt.
- `observe()` captures the current head (`agentview`) and wrist (`robot0_eye_in_hand`) RGB-D
  frames. `value["images"]` lists 512x512 PNG files written into `images/` that you can look at
  with your image viewing tool; `value` also contains camera intrinsics/extrinsics, depth handles
  and the measured robot state (end-effector pose, `gripper_open_fraction`, ...)."""

INTRO = """You are operating a simulated Franka Panda arm (LIBERO benchmark, MuJoCo physics) through a
small Python API. There is no other way to act."""

PYTHON_ACT = """- You act through the `python` tool: a persistent Python 3 session running in this directory.
  `robot` is already imported and every capability is also a bare function (`observe()`,
  `robot.move_to(...)`, ...). Variables, functions and imports persist between calls.
- A call can be a single step or a whole program (detect the target, compute the grasp pose from the
  measured numbers, move, close, lift, verify). Compute coordinates from variables in code; do not
  retype numbers from earlier output. Define a helper once and reuse it.
- Every capability returns a plain dict `{success, executed, failure_category, message, value,
  diagnostics, budget}`. Robot calls never raise on failure: a failed physical request returns
  `success=False` with a `failure_category`, a `message` and the measured receipt, and your program
  keeps running. Check `success` in code and branch on it.
- After each call you see what it printed, the value of its final expression, one receipt line per
  robot call it made (outcome and measured summary) and the names it changed.
- `observe()` captures the current head (`agentview`) and wrist (`robot0_eye_in_hand`) RGB-D
  frames. `value["images"]` lists 512x512 PNG files in `images/`; to look at them call
  `image(path)` inside a python call, e.g. `image(obs["value"]["images"])` (at most 4 per call).
  `value` also contains camera intrinsics/extrinsics, depth handles and the measured robot state
  (end-effector pose, `gripper_open_fraction`, ...).
- There is no shell: use Python for files (e.g. `open("NOTES.md", "w")`)."""


def task_markdown(info: Mapping[str, object], manifest: Mapping[str, object], *, wall_time_s: float,
                  max_actions: int, max_tool_calls: int, standards: bool = True, runtime: str = "shell") -> str:
    if runtime not in ("shell", "python"):
        raise ValueError(f"unknown runtime {runtime!r}")
    interface = rewrite_interface(str(manifest["python_interface"]))
    images = "\n".join(
        f"- `{Path(i['path']).relative_to(Path(i['path']).parents[1])}` ({i['role']} camera `{i['camera']}`)"
        for i in info["initial_images"])
    overrides = dict(info.get("task_markdown_overrides") or {})
    standards_text = STANDARDS
    for old, new in (overrides.get("standards_replace") or {}).items():
        if old not in standards_text:
            raise ValueError(f"standards override target not found: {old[:60]!r}")
        standards_text = standards_text.replace(old, new)
    standards_block = ("\n" + standards_text) if standards else ""
    if overrides.get("guide"):  # benchmark guide: short generic procedures (robotwin2 R5), after the standards
        standards_block += "\n" + str(overrides["guide"]).rstrip() + "\n"
    intro = overrides.get("intro")
    if intro is None:  # CaP-Bench (capgym_* suites) names its benchmark; everything else is LIBERO's text
        benchmark = "CaP-Gym Robosuite benchmark" if str(info.get("suite") or "").startswith("capgym_") else "LIBERO benchmark"
        intro = INTRO.replace("LIBERO benchmark", benchmark)
    act = PYTHON_ACT if runtime == "python" else SHELL_ACT
    return f"""# Robot manipulation task

{intro}

**Task: {info['task']}**

## How to act
{act}
- Initial camera images (taken before any action):
{images}
- Coordinates are meters in the `robot_base` frame, quaternions are `wxyz`.
- `robot.status()` returns the remaining budgets at any time; `robot.history(start, end, limit)` returns the
  objective records of your past calls (what you called, whether it succeeded, the message, image paths).
- Keep `NOTES.md` in this directory up to date: the goal, the detection ids / poses / numbers you rely on,
  what is done, what failed and why, the current plan and the next step. Long conversations are compacted;
  afterwards you keep only the task files, the images, `NOTES.md`, `robot.history()` and your latest turns.
- When you believe the task is complete, call `robot.finish()`. This ends the episode: the
  environment then scores the final scene state with a hidden success predicate. You will NOT
  receive the verdict, and no further calls are accepted afterwards. Tool-call success is not
  task success; only the final physical state of the scene counts.
- Budgets: {max_actions} physical actions (`move_to`, `hold`, `recover`, `release`; every request
  counts, including rejected ones), {max_tool_calls} perception/compute calls, and
  {wall_time_s / 60:.0f} minutes of wall-clock time from the moment the episode started (the
  `budget` field of every result shows what remains). When a budget is exhausted the calls fail;
  when the wall budget expires the episode is scored as it stands.
- Only the API is available: do not modify `robot.py`, do not try to reach the simulator, the
  server process, or files outside this directory.
{standards_block}
## Capability API (Python signatures; call them as `robot.<name>(...)`)

```python
{interface}
```
"""


__all__ = ["PROTOCOL_SENTENCE", "PYTHON_ACT", "SHELL_ACT", "STANDARDS", "rewrite_interface", "strip_protocol", "task_markdown"]

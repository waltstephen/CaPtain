"""Intro v5 = harness prompt profile compact_v1 (short TASK.md, API docs in the workspace, summarised large results), the
same overrides in the real-arm config fragment, and the prompt record in batch rows."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from soarm import episode

ROOT = Path(__file__).resolve().parents[1]
INTERFACE = '''# General rules of the API.

def observe() -> CapabilityResult[dict]:
    """Capture current public RGB-D. More text.
    """
'''


def _render(version: str) -> str:
    from captain_harness.server.task_markdown import task_markdown
    info = {"task": "Put the cube in the box.", "initial_images": [],
            "task_markdown_overrides": episode.task_markdown_overrides(version)}
    return task_markdown(info, {"python_interface": INTERFACE}, wall_time_s=3600, max_actions=150, max_tool_calls=600,
                         runtime="python")


def test_v5_is_the_compact_profile_with_a_sim_neutral_intro():
    o = episode.task_markdown_overrides("v5")
    assert o["profile"] == "compact_v1" and o["standards"] == episode.STANDARDS_V5
    assert "simulated" not in o["intro"] and "MuJoCo" not in o["intro"] and "SO-101 arm" in o["intro"]
    assert o["intro"].replace("an SO-101 arm (", "a simulated SO-101 arm (").replace(
        "opens to about 7 cm)", "opens to about 7 cm; MuJoCo physics)") == episode.INTROS["v4"]
    md = _render("v5")
    assert "Articulated targets" not in md and "Knobs" not in md and "whatever the head camera's viewpoint" in md
    assert "def observe() -> dict  # Capture current public RGB-D." in md and "General rules" not in md
    v4 = _render("v4")
    assert "General rules" in v4 and "Articulated targets" in v4 and len(md) < len(v4)
    assert episode.task_markdown_overrides("v4") == {"intro": episode.INTROS["v4"],
                                                      "standards_replace": {episode.RELATION_LIBERO: episode.RELATION_TABLE}}
    assert episode.prompt_record("v5") == {"intro": "v5", "prompt_profile": "compact_v1",
                                           "task_md_version": "soarm-v5/compact_v1"}
    assert episode.prompt_record("v4")["prompt_profile"] == "full"


def test_real_arm_fragment_matches_the_sim_twin():
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "real_arm_config.py"), "--check"], capture_output=True,
                         text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    frag = json.loads((ROOT / "configs" / "hardware" / "soarm101_prompt_depth.json").read_text())
    sim = episode.task_markdown_overrides("v5")
    budget = f"at most {episode.REQUEST_BUDGET} model requests"
    assert frag["task_markdown_overrides"] == {**sim, "intro": sim["intro"].replace(budget, "at most 100 model requests")}
    from soarm import perception
    assert frag["mono_depth"] == {"model": "moge2", "support_plane_z_m": perception.SUPPORT_PLANE_Z_M,
                                  "table_footprint_m": perception.TABLE_FOOTPRINT_M}


def test_batch_rows_record_the_prompt(tmp_path):
    spec = importlib.util.spec_from_file_location("soarm_batch", ROOT / "tools" / "batch.py")
    batch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(batch)
    ep = tmp_path / "soarm_sim_1_seed-1"
    (ep / "soarm").mkdir(parents=True)
    (ep / "workspace").mkdir()
    assert batch.prompt_summary(ep) == {"intro": None, "prompt_profile": "full", "task_md_version": None,
                                        "task_md_sha256": None}
    (ep / "soarm" / "prompt.json").write_text(json.dumps(episode.prompt_record("v5")))
    (ep / "workspace" / "TASK.md").write_text("# t\n")
    row = batch.prompt_summary(ep)
    assert row["prompt_profile"] == "compact_v1" and row["intro"] == "v5" and len(row["task_md_sha256"]) == 16

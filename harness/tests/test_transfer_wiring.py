"""CPU-only checks of the transfer-benchmark wiring (no simulator, no GPU, no model)."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRANSFER = ROOT / "eval" / "transfer"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_suite_dispatch_and_arguments() -> None:
    from captain_harness.server import env_server

    names = {suite: getattr(env_server.episode_builder(suite), "__name__", None)
             for suite in ("robotwin2_sim", "robodojo_sim", "simplerenv_sim", "capgym_stack", "hardware_real", "libero_goal")}
    assert names == {"robotwin2_sim": "build_robotwin_episode", "robodojo_sim": "build_robodojo_episode",
                     "simplerenv_sim": "build_simplerenv_episode", "capgym_stack": "build_capgym_episode",
                     "hardware_real": "build_hardware_episode", "libero_goal": None}
    args = env_server.build_parser().parse_args(
        ["--suite", "robotwin2_sim", "--task-id", "0", "--output-dir", "/tmp/x", "--robotwin-config", "c.json",
         "--robotwin-task", "t.json", "--hardware-driver", "pkg.mod:make"])
    assert str(args.robotwin_config) == "c.json" and args.hardware_driver == "pkg.mod:make"


def _task_markdown(**info):
    from captain_harness.server.task_markdown import task_markdown

    base = {"task": "put the bowl on the plate", "initial_images": [], **info}
    return task_markdown(base, {"python_interface": "def observe(): ..."}, wall_time_s=1080, max_actions=150,
                         max_tool_calls=600, runtime="python")


def test_task_markdown_intro_and_overrides() -> None:
    from captain_harness.server.task_markdown import INTRO

    libero = _task_markdown(suite="libero_goal")
    assert INTRO in libero
    capgym = _task_markdown(suite="capgym_lift")
    assert "CaP-Gym Robosuite benchmark" in capgym and capgym.replace("CaP-Gym Robosuite benchmark", "LIBERO benchmark") == libero
    guided = _task_markdown(suite="robotwin2_sim", task_markdown_overrides={"intro": "You are operating two arms.",
                                                                            "guide": "## Guide\n- step"})
    assert guided.startswith("# Robot manipulation task\n\nYou are operating two arms.") and "## Guide\n- step" in guided
    with pytest.raises(ValueError):
        _task_markdown(task_markdown_overrides={"standards_replace": {"no such text": "x"}})


def test_render_config(tmp_path: Path) -> None:
    render = _load("render_config", TRANSFER / "render_config.py")
    os.environ["CAPTAIN_TEST_ROOT"] = "/opt/sim"
    try:
        out = json.loads(render.render('{"a": "${CAPTAIN_TEST_ROOT}/x", "b": "${CAPTAIN_TEST_UNSET_VAR:-d}"}'))
    finally:
        del os.environ["CAPTAIN_TEST_ROOT"]
    assert out == {"a": "/opt/sim/x", "b": "d"}
    with pytest.raises(SystemExit):
        render.render('{"a": "${CAPTAIN_TEST_UNSET_VAR}"}')
    for template in (ROOT / "configs" / "benchmarks").glob("*.json"):
        json.loads(template.read_text())


def test_task_lists() -> None:
    seeds = [line.split("\t") for line in (TRANSFER / "robotwin2" / "official_hard_seeds.tsv").read_text().splitlines()
             if line and not line.startswith("#")]
    assert len(seeds) == 5000 and len({s[0] for s in seeds}) == 50
    assert sum(1 for s in seeds if s[2] == "single") == 3300
    assert min(int(s[1]) for s in seeds) == 100000
    bridge = [json.loads(line) for line in (TRANSFER / "simplerenv" / "bridge_vm_96.jsonl").read_text().splitlines()]
    assert len(bridge) == 96 and {b["group"] for b in bridge} == {"spoon_on_towel", "carrot_on_plate", "stack_cube",
                                                                  "eggplant_in_basket"}
    dojo = [json.loads(line) for line in (TRANSFER / "robodojo" / "official_2100.jsonl").read_text().splitlines()]
    assert len(dojo) == 2100 and len({d["leaderboard_task"] for d in dojo}) == 42
    cap = [json.loads(line) for line in (TRANSFER / "capbench" / "capbench_5x20.jsonl").read_text().splitlines()]
    assert len(cap) == 100


def test_round_robin_order(tmp_path: Path) -> None:
    jobs = _load("run_jobs", TRANSFER / "run_jobs.py")
    lines = [{"name": f"capgym_{t}_0_seed-{s}", "suite": f"capgym_{t}", "seed": s} for t in ("lift", "stack") for s in range(3)]
    path = tmp_path / "t.jsonl"
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    order = [name for name, _ in jobs.load_tasks("capbench", path)]
    assert order[:4] == ["capgym_lift_0_seed-0", "capgym_stack_0_seed-0", "capgym_lift_0_seed-1", "capgym_stack_0_seed-1"]


def test_robotwin2_summary_on_synthetic_rows(tmp_path: Path) -> None:
    rows = [{"instance": "click_bell_s100000", "task": "click_bell", "ever": True, "host": True, "strict": True,
             "finished": True, "false_finish": False},
            {"instance": "click_bell_s100001", "task": "click_bell", "ever": True, "host": False, "strict": False,
             "finished": False, "false_finish": False}]
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(rows))
    out = subprocess.run([sys.executable, str(TRANSFER / "summarize.py"), "robotwin2", "--rows", str(path)],
                         capture_output=True, text=True, check=True)
    summary = json.loads(out.stdout)
    assert summary["all"]["n"] == 2 and summary["all"]["ever"] == 100.0 and summary["all"]["host"] == 50.0


@pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="mujoco not installed")
def test_transfer_modules_import() -> None:
    import mujoco

    if not hasattr(mujoco, "MjSpec"):
        pytest.skip("the kinematic twins need MuJoCo >= 3 (requirements-transfer.txt)")
    for module in ("captain_harness.adapters.robotwin2.episode", "captain_harness.adapters.simplerenv.episode",
                   "captain_harness.adapters.robodojo.episode", "captain_harness.adapters.robodojo.dual_arm",
                   "captain_harness.adapters.hardware.episode"):
        __import__(module)

#!/usr/bin/env python3
"""Replay the completion gate offline over finished batches: which false finishes would it reject on the agent's
first finish(), and which true successes would it send back at least once. Uses only the public measurements the
agent recorded (calls.jsonl) and the task goals from BDDL; no simulator.

    .venv-libero/bin/python scripts/gate_replay.py outputs/<batch> [outputs/<batch> ...] [--expect-catch N --expect-hurt M]

Run this before changing a gate rule and after: the two counts are the self-consistency check.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _goals(suite: str, task_id: int, cache: dict):
    key = (suite, task_id)
    if key not in cache:
        os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".libero-config"))
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs.bddl_utils import robosuite_parse_problem
        from captain_harness.adapters.capx.completion_gate import goals_from_bddl

        task = benchmark.get_benchmark_dict()[suite]().get_task(task_id)
        problem = robosuite_parse_problem(os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file))
        cache[key] = goals_from_bddl(problem["goal_state"])
    return cache[key]


def episode_records(ep: Path):
    from captain_harness.adapters.capx.completion_gate import comparison_record, geometry_record

    calls = [json.loads(l) for l in (ep / "calls.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    geo, cmp_, last_phys, first_grasp = [], [], None, None
    for c in calls:
        idx = c["call_index"]
        if c["consumes_action"] and c.get("executed"):
            last_phys = idx
            if first_grasp is None and c["name"] == "hold":
                first_grasp = idx
        if not c.get("success"):
            continue
        if c["name"] == "measure_object_geometry":
            geo.append(geometry_record(c["value"] or {}, c["arguments"] or {}, idx))
        elif c["name"] == "compare_object_observations":
            cmp_.append(comparison_record(c["value"] or {}, idx))
    return geo, cmp_, last_phys, first_grasp


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("batches", nargs="+", type=Path)
    p.add_argument("--expect-catch", type=int, default=None)
    p.add_argument("--expect-hurt", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    from captain_harness.adapters.capx.completion_gate import evaluate

    cache: dict = {}
    conf = collections.Counter()
    caught, hurt, missed = [], [], []
    for batch in args.batches:
        for ep in sorted(batch.glob("episodes/*_seed-*")):
            if ".infra-" in ep.name or ".partial-" in ep.name or not (ep / "episode_result.json").is_file() or not (ep / "calls.jsonl").is_file():
                continue
            r = json.loads((ep / "episode_result.json").read_text(encoding="utf-8"))
            if not r.get("agent_finished") or r.get("host_final_success") is None:
                continue
            goals = _goals(r["suite"], r["task_id"], cache)
            geo, cmp_, last_phys, first_grasp = episode_records(ep)
            v = evaluate(goals, geo, cmp_, last_physical_step=last_phys, first_grasp_step=first_grasp, rejections=0)
            verdict = v["verdict"]
            success = bool(r["host_final_success"])
            conf[(verdict, success)] += 1
            tag = (batch.name, r["suite"], r["task_id"], v.get("failure"))
            if verdict == "reject" and not success:
                caught.append(tag)
            elif verdict == "reject" and success:
                hurt.append(tag)
            elif verdict != "reject" and not success:
                missed.append(tag)
    n_ff = conf[("reject", False)] + conf[("pass", False)] + conf[("bypass", False)] + conf[("not_applicable", False)]
    n_ok = sum(v for (verdict, s), v in conf.items() if s)
    print(f"gate replay: catches {len(caught)}/{n_ff} false finishes on the first finish(); would send back {len(hurt)}/{n_ok} true successes; misses {len(missed)}")
    if args.verbose:
        print("caught:", *sorted(caught), sep="\n  ")
        print("hurt:", *sorted(hurt), sep="\n  ")
        print("missed:", *sorted(missed), sep="\n  ")
    ok = True
    if args.expect_catch is not None and len(caught) != args.expect_catch:
        print(f"EXPECTED {args.expect_catch} catches"); ok = False
    if args.expect_hurt is not None and len(hurt) != args.expect_hurt:
        print(f"EXPECTED {args.expect_hurt} hurt"); ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

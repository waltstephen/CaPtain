#!/usr/bin/env python3
"""Run a transfer-benchmark task list against one served policy, resumably and in parallel.

    python3 eval/transfer/run_jobs.py robotwin2  runs/robotwin2_tasks/          --out runs/captain-4b/robotwin2 --parallel 8
    python3 eval/transfer/run_jobs.py simplerenv eval/transfer/simplerenv/bridge_vm_96.jsonl --out runs/captain-4b/simplerenv
    python3 eval/transfer/run_jobs.py robodojo   eval/transfer/robodojo/official_2100.jsonl  --out runs/captain-4b/robodojo
    python3 eval/transfer/run_jobs.py capbench   eval/transfer/capbench/capbench_5x20.jsonl  --out runs/captain-4b/capbench

TASKS is either a .jsonl file (one task object per line; `name` is the episode directory name) or a directory of task
.json files (the RoboTwin task files written by eval/transfer/robotwin2/build_official_tasks.py). Each task becomes one
call of eval/transfer/run_episode.sh. Jobs are ordered round-robin over tasks (episode k of every task before k+1), so an
interrupted run is balanced. Episodes that already have episode_result.json are skipped; episodes that end with
termination_reason "infrastructure_error" are moved aside and retried up to --retry-infra times and never count as
policy failures. The environment of run_episode.sh (POLICY_URL, STUDENT_PYTHON, SIM_GPU, benchmark paths) is passed through.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCHMARKS = ("robotwin2", "simplerenv", "robodojo", "capbench")


def task_name(bench: str, task: dict) -> str:
    if task.get("name"):
        return str(task["name"])
    if bench == "robotwin2":
        return f"{task['robotwin_task']}_s{task['robotwin_seed']}"
    if bench == "simplerenv":
        return str(task["episode_key"])
    if bench == "robodojo":
        return f"{task['robodojo_task']}_L{task['layout_id']}"
    return f"{task['suite']}_0_seed-{task['seed']}"


def group_of(bench: str, task: dict) -> str:
    if bench == "robotwin2":
        return str(task["robotwin_task"])
    if bench == "simplerenv":
        return f"{task['setting']}/{task['group']}"
    if bench == "robodojo":
        return str(task["robodojo_task"])
    return str(task["suite"])


def load_tasks(bench: str, source: Path) -> list[tuple[str, dict]]:
    if source.is_dir():
        items = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(source.glob("*.json")) if not p.name.startswith("_")]
    else:
        items = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_group: dict[str, list[dict]] = defaultdict(list)
    for task in items:
        by_group[group_of(bench, task)].append(task)
    for tasks in by_group.values():   # within a task: official order (seed / layout id / episode key)
        tasks.sort(key=lambda t: (t.get("robotwin_seed", 0), t.get("layout_id", 0), t.get("seed", 0), task_name(bench, t)))
    ordered = []
    for rank in range(max((len(v) for v in by_group.values()), default=0)):
        for group in sorted(by_group):
            if rank < len(by_group[group]):
                ordered.append(by_group[group][rank])
    return [(task_name(bench, t), t) for t in ordered]


def result_of(path: Path) -> dict | None:
    try:
        return json.loads((path / "episode_result.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("benchmark", choices=BENCHMARKS)
    p.add_argument("tasks", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--parallel", type=int, default=8, help="episodes in flight (each needs a simulator process and ~1 CPU core)")
    p.add_argument("--launch-interval", type=float, default=20.0, help="seconds between launches (simulator start-up bursts)")
    p.add_argument("--retry-infra", type=int, default=2)
    p.add_argument("--limit", type=int, default=None, help="only the first N jobs of the round-robin order (smoke test / milestone)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    jobs = load_tasks(args.benchmark, args.tasks)[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    task_dir = args.out / "_tasks"
    task_dir.mkdir(exist_ok=True)
    launch_lock = threading.Lock()
    last_launch = [0.0]

    def run(job: tuple[str, dict]) -> str:
        name, task = job
        task_file = task_dir / f"{name}.json"
        task_file.write_text(json.dumps(task, indent=1), encoding="utf-8")
        target = args.out / name
        for _ in range(args.retry_infra + 1):
            result = result_of(target)
            if result is not None and result.get("termination_reason") != "infrastructure_error":
                return "done"
            if target.exists():
                shutil.move(str(target), str(args.out / f"_{name}.infra-{int(time.time())}"))
            command = ["bash", str(HERE / "run_episode.sh"), args.benchmark, str(task_file), str(args.out)]
            if args.dry_run:
                print(" ".join(command))
                return "dry"
            with launch_lock:
                wait = args.launch_interval - (time.monotonic() - last_launch[0])
                if wait > 0:
                    time.sleep(wait)
                last_launch[0] = time.monotonic()
            log = args.out / "_logs" / f"{name}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", encoding="utf-8") as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=os.environ.copy(), check=False)
        result = result_of(target)
        return "done" if result and result.get("termination_reason") != "infrastructure_error" else "infra"

    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        outcomes = list(pool.map(run, jobs))
    print(json.dumps({k: outcomes.count(k) for k in sorted(set(outcomes))}), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

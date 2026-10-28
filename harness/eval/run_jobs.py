#!/usr/bin/env python3
"""Run an evaluation task list (eval/tasks/*.tsv) against one served policy, resumably and in parallel.

    python3 eval/run_jobs.py eval/tasks/libero_official_240.tsv --out runs/captain-4b/official --parallel 8
    python3 eval/run_jobs.py eval/tasks/libero_pro_postask_800.tsv --out runs/captain-4b/pro_postask \
        --artifacts /path/to/CaPtain-artifacts/eval --parallel 8
    python3 eval/summarize.py runs/captain-4b/official

Each row becomes one call of eval/run_episode.sh (same protocol for every benchmark); benchmark-specific settings are passed to
the environment server only (--libero-variant, --init-state-index, --init-state-file/--init-state-sha256, --max-case/--max-arm,
--instruction-text). The model-facing tool API and TASK.md format are identical across benchmarks.

Episodes that already have episode_result.json are skipped. Episodes that end with termination_reason "infrastructure_error"
are moved aside and retried up to --retry-infra times; they are never counted as failures of the policy.
Environment variables of eval/run_episode.sh (POLICY_URL, STUDENT_PYTHON, EGL_DEVICE, ...) are passed through.
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLS = ["id", "benchmark", "cell", "variant", "suite", "task_id", "seed", "init_index", "init_state_file", "init_state_sha256",
        "max_case", "max_arm", "instruction"]


def read_jobs(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        values = line.split("\t")
        if len(values) != len(COLS):
            raise ValueError(f"{path}: expected {len(COLS)} tab-separated columns, got {len(values)}: {line[:120]}")
        rows.append({k: (None if v == "-" else v) for k, v in zip(COLS, values)})
    return rows


def episode_root(out: Path, job: dict) -> Path:
    if job["max_case"]:
        return out / job["benchmark"] / Path(job["max_case"]).stem
    return out / job["benchmark"]


def episode_dir(out: Path, job: dict) -> Path:
    return episode_root(out, job) / f"{job['suite']}_{job['task_id']}_seed-{job['seed']}"


def server_args(job: dict, artifacts: Path | None) -> list[str]:
    args: list[str] = []
    if job["variant"] and job["variant"] != "libero":
        args += ["--libero-variant", job["variant"]]

    def server(*values: str) -> None:
        for value in values:
            args.append(f"--server-arg={value}")

    if job["instruction"]:
        server("--instruction-text", job["instruction"])
    if job["init_index"] is not None:
        server("--init-state-index", job["init_index"])
    for key, flag in (("init_state_file", "--init-state-file"), ("max_case", "--max-case")):
        if job[key]:
            if artifacts is None:
                raise SystemExit(f"job {job['id']} needs --artifacts (CaPtain-artifacts eval/ directory) for {job[key]}")
            path = (artifacts / job[key]).resolve()
            if not path.is_file():
                raise SystemExit(f"missing artifact {path}")
            server(flag, str(path))
    if job["init_state_sha256"]:
        server("--init-state-sha256", job["init_state_sha256"])
    if job["max_arm"]:
        server("--max-arm", job["max_arm"])
    return args


def result_of(path: Path) -> dict | None:
    try:
        return json.loads((path / "episode_result.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tasks", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--artifacts", type=Path, default=None, help="CaPtain-artifacts eval/ directory (PRO frozen states, MAX cases)")
    p.add_argument("--parallel", type=int, default=8, help="episodes in flight against the policy server (paper: <= 8 per server)")
    p.add_argument("--launch-interval", type=float, default=5.0, help="seconds between launches (EGL context bursts)")
    p.add_argument("--retry-infra", type=int, default=2)
    p.add_argument("--limit", type=int, default=None, help="only the first N jobs (smoke test)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    jobs = read_jobs(args.tasks)[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    launch_lock = threading.Lock()
    last_launch = [0.0]

    def run(job: dict) -> str:
        target = episode_dir(args.out, job)
        for attempt in range(args.retry_infra + 1):
            result = result_of(target)
            if result is not None and result.get("termination_reason") != "infrastructure_error":
                return "done"
            if target.exists():
                shutil.move(str(target), f"{target}.infra-{int(time.time())}")
            command = ["bash", str(HERE / "run_episode.sh"), job["suite"], job["task_id"], job["seed"], str(episode_root(args.out, job)),
                       *server_args(job, args.artifacts)]
            if args.dry_run:
                print(" ".join(command))
                return "dry"
            with launch_lock:
                wait = args.launch_interval - (time.monotonic() - last_launch[0])
                if wait > 0:
                    time.sleep(wait)
                last_launch[0] = time.monotonic()
            log = args.out / "logs" / f"{job['benchmark']}_{job['id']}.log"
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

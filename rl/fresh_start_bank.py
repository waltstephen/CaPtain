#!/usr/bin/env python3
"""Fresh start pool: a segment bank of prompt-only K = 0 starts on FRESH LIBERO initial states only.

The harness keeps per task ``<fresh_dir>/<problem_folder>/<task_name>.fresh_init`` (+ a
``.json`` sidecar with the generating reset seeds and the file's sha256) and maps ``seed = 1000 + k`` to fresh state k;
seeds below 1000 are official states (the official evaluation uses 11-16). This tool turns those files into

  * a segment bank (rl/segment_bank.py schema) with one prompt-only trajectory per (suite, task, 1000 + k), which the
    trainer samples without replacement per epoch (SegmentCurriculum: shuffled order, reshuffled every epoch), and
  * a pool manifest ``fresh-start-pool/v1``: per task the file path, its sha256 and the state count, plus the bank's sha256
    (fresh_run hashes the manifest with the run's sources).

Checks: every sidecar matches its file; every task of the requested suites has at least --min-per-task states; no seed
below 1000 can be produced. Run with the harness venv (LIBERO importable) or give --task-map {"suite:task": "folder/name"}.

  python rl/fresh_start_bank.py --fresh-dir <harness>/data/libero_fresh_init --out-bank bank.json --out-manifest pool.json \
      [--suites libero_spatial libero_object libero_goal libero_10] [--min-per-task 50] [--max-per-task N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from segment_bank import SCHEMA_VERSION, bank_sha256, prompt_starts
from segment_curriculum import FRESH_SEED_BASE, check_fresh_population

POOL_SCHEMA = "fresh-start-pool/v1"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def libero_task_map(suites) -> dict[str, str]:
    from libero.libero import benchmark  # harness venv

    out = {}
    for suite in suites:
        bench = benchmark.get_benchmark_dict()[suite]()
        for i in range(bench.n_tasks):
            task = bench.get_task(i)
            out[f"{suite}:{i}"] = f"{task.problem_folder}/{task.name}"
    return out


def read_task(fresh_dir: Path, rel: str) -> dict:
    path = fresh_dir / f"{rel}.fresh_init"
    meta = json.loads(path.with_name(path.name + ".json").read_text(encoding="utf-8"))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if meta.get("sha256") != sha:
        raise ValueError(f"{path}: content does not match its sidecar sha256")
    n = len(meta["reset_seeds"])
    if n <= 0:
        raise ValueError(f"{path}: no states")
    return {"path": str(path), "sha256": sha, "n": n}


def build(task_map: dict[str, str], fresh_dir: Path, *, min_per_task: int, max_per_task: int | None = None) -> tuple[dict, dict]:
    files, prompts, short = {}, [], []
    for key in sorted(task_map, key=lambda k: (k.split(":")[0], int(k.split(":")[1]))):
        suite, task = key.split(":")
        info = read_task(fresh_dir, task_map[key])
        n = min(info["n"], max_per_task) if max_per_task else info["n"]
        if n < min_per_task:
            short.append((key, n))
        info["used"] = n
        files[key] = info
        prompts += [(suite, int(task), FRESH_SEED_BASE + k) for k in range(n)]
    if short:
        raise ValueError(f"tasks with fewer than {min_per_task} fresh states: {short}")
    trajectories, starts = prompt_starts(prompts)
    bank = {"schema_version": SCHEMA_VERSION, "trajectories": {t["trajectory_id"]: t for t in sorted(trajectories, key=lambda t: t["trajectory_id"])},
            "starts": {s["start_id"]: s for s in sorted(starts, key=lambda s: s["start_id"])}, "skipped_sources": 0}
    bank["sha256"] = bank_sha256(bank)
    summary = check_fresh_population(bank["trajectories"], min_tasks=len(task_map))
    manifest = {"schema": POOL_SCHEMA, "fresh_dir": str(fresh_dir), "seed_base": FRESH_SEED_BASE, "files": files,
                "bank_sha256": bank["sha256"], **summary}
    return bank, manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fresh-dir", type=Path, required=True)
    ap.add_argument("--suites", nargs="+", default=list(SUITES))
    ap.add_argument("--task-map", type=Path, default=None, help='{"suite:task": "problem_folder/task_name"} (default: from LIBERO)')
    ap.add_argument("--min-per-task", type=int, default=50)
    ap.add_argument("--max-per-task", type=int, default=None)
    ap.add_argument("--out-bank", type=Path, required=True)
    ap.add_argument("--out-manifest", type=Path, required=True)
    a = ap.parse_args()
    task_map = json.loads(a.task_map.read_text()) if a.task_map else libero_task_map(a.suites)
    task_map = {k: v for k, v in task_map.items() if k.split(":")[0] in a.suites}
    bank, manifest = build(task_map, a.fresh_dir.resolve(), min_per_task=a.min_per_task, max_per_task=a.max_per_task)
    a.out_bank.parent.mkdir(parents=True, exist_ok=True)
    a.out_bank.write_text(json.dumps(bank, indent=1), encoding="utf-8")
    manifest["bank_path"] = str(a.out_bank.resolve())
    a.out_manifest.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("states", "tasks", "per_task_min", "per_task_max", "bank_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

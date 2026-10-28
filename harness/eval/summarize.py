#!/usr/bin/env python3
"""Summarise an eval/run_jobs.py output directory: host and strict success per benchmark and per cell.

    python3 eval/summarize.py eval/tasks/libero_official_240.tsv runs/captain-4b/official

host success   = the hidden LIBERO goal predicate holds at the end of the episode (episode_result.json host_final_success);
strict success = host success AND the agent itself called finish() (agent_finished).
For LIBERO-PRO Pos+Task the 6-cell mean (spatial/object/goal x swap/task) and the 8-cell mean (+ libero_10) are reported
as in the paper. Jobs without a result, or whose result is an infrastructure_error, are listed as missing and excluded.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_jobs import episode_dir, read_jobs  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tasks", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = p.parse_args()

    cells: dict[tuple[str, str], list[tuple[bool, bool]]] = defaultdict(list)
    missing = []
    terminations: dict[str, int] = defaultdict(int)
    for job in read_jobs(args.tasks):
        path = episode_dir(args.out, job) / "episode_result.json"
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            missing.append(job["id"])
            continue
        if result.get("termination_reason") == "infrastructure_error":
            missing.append(job["id"])
            continue
        host = bool(result.get("host_final_success"))
        cells[(job["benchmark"], job["cell"])].append((host, host and bool(result.get("agent_finished"))))
        terminations[str(result.get("termination_reason"))] += 1

    def rate(values: list[tuple[bool, bool]], k: int) -> float:
        return 100.0 * sum(v[k] for v in values) / max(1, len(values))

    summary = {"cells": {}, "benchmarks": {}, "missing": missing, "terminations": dict(terminations)}
    by_bench: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for (bench, cell), values in sorted(cells.items()):
        summary["cells"][f"{bench}:{cell}"] = {"n": len(values), "host": round(rate(values, 0), 1), "strict": round(rate(values, 1), 1)}
        by_bench[bench].extend(values)
    for bench, values in by_bench.items():
        summary["benchmarks"][bench] = {"n": len(values), "host": round(rate(values, 0), 1), "strict": round(rate(values, 1), 1)}
    postask = {cell: v for (bench, cell), v in cells.items() if bench == "libero_pro_postask"}
    if postask:
        six = [c for c in postask if not c.startswith("libero_10")]
        for name, chosen in (("6-cell", six), ("8-cell", list(postask))):
            summary["benchmarks"][f"libero_pro_postask {name}"] = {
                "cells": len(chosen),
                "host": round(sum(rate(postask[c], 0) for c in chosen) / max(1, len(chosen)), 1),
                "strict": round(sum(rate(postask[c], 1) for c in chosen) / max(1, len(chosen)), 1)}
    if args.json:
        print(json.dumps(summary, indent=2))
        return 0
    print(f"{'benchmark / cell':58s} {'n':>5s} {'host':>6s} {'strict':>6s}")
    for name, row in summary["cells"].items():
        print(f"  {name:56s} {row['n']:5d} {row['host']:6.1f} {row['strict']:6.1f}")
    for name, row in summary["benchmarks"].items():
        print(f"{name:58s} {row.get('n', row.get('cells')):5d} {row['host']:6.1f} {row['strict']:6.1f}")
    print(f"terminations: {dict(terminations)}; missing/infra: {len(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

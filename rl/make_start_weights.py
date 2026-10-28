"""Prioritized start sampling (optional; unused by the reference config): weight each official initial
state (suite:task:seed) by the outcome variance of the current student there, so starts where it always fails (V ~ 0, no
signal) or always succeeds are drawn less often.

  p = (successes + 0.5) / (episodes + 1)        w = max(floor, 4 p (1 - p))        unseen starts: default (1.0)

Usage: python rl/make_start_weights.py --rollouts RUN/rollouts.jsonl [...] --out start_weights.json [--floor 0.1]
Episodes: the rows' host success (``success``); infrastructure errors are skipped. Write the weights from rollouts of the
policy the run starts from.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from segment_curriculum import MIN_START_WEIGHT, START_WEIGHTS_SCHEMA


def start_weights(rows, *, floor: float = 0.1, default: float = 1.0) -> dict:
    if not MIN_START_WEIGHT <= floor <= 1.0 or not MIN_START_WEIGHT <= default <= 1.0:
        raise ValueError(f"floor and default must be in [{MIN_START_WEIGHT}, 1]")
    count = defaultdict(lambda: [0, 0])
    for r in rows:
        if r.get("verdict") == "infrastructure_error" or not isinstance(r.get("success"), bool) or int(r.get("K") or 0):
            continue
        c = count[f'{r["suite"]}:{int(r["task_id"])}:{int(r["seed"])}']
        c[0] += int(r["success"])
        c[1] += 1
    weights, stats = {}, {}
    for key, (s, n) in sorted(count.items()):
        p = (s + 0.5) / (n + 1)
        weights[key] = round(max(floor, 4 * p * (1 - p)), 4)
        stats[key] = [s, n]
    return {"schema": START_WEIGHTS_SCHEMA, "weights": weights, "default": default, "floor": floor, "counts": stats}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--floor", type=float, default=0.1)
    ap.add_argument("--default", type=float, default=1.0)
    args = ap.parse_args()
    rows = [json.loads(line) for path in args.rollouts for line in open(path) if line.strip()]
    spec = start_weights(rows, floor=args.floor, default=args.default)
    Path(args.out).write_text(json.dumps(spec, indent=1))
    w = list(spec["weights"].values())
    print(json.dumps({"starts": len(w), "mean_weight": sum(w) / max(1, len(w)), "at_floor": sum(x <= args.floor for x in w)}))


if __name__ == "__main__":
    main()

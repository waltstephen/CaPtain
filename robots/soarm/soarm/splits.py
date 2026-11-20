"""Train / eval split of the SO-101 twin tasks (fixed before any SFT run).

- Held-out families: never in SFT data; evaluated to measure transfer to unseen task types. Defined by the task specs'
  `split` field (legacy gather/lay_down/swap; cube swap/gather/count_place/unstack); task.json `family` = spec id.
- Cube-domain eval: the fixed suites in soarm/specs/suites (taskspec.suite); eval_plan below is the legacy eval set.
- Layout seeds: teacher/SFT data use seeds 100-8999; evaluation uses 9000+ (layouts, objects, colours, lighting never
  seen in training, for every family).
"""
from __future__ import annotations

import json
from pathlib import Path

from . import taskspec
from .episode import FAMILY_ORDER

HELD_OUT_FAMILIES = tuple(s["id"] for s in taskspec.select(split="heldout"))
TRAIN_FAMILIES = tuple(f for f in FAMILY_ORDER if f not in HELD_OUT_FAMILIES)
TRAIN_SEEDS = range(100, 9000)
EVAL_SEED0 = 9000


def eval_plan(seen_per_family: int = 3, heldout_per_family: int = 5) -> list[dict]:
    jobs = [{"family": f, "seed": EVAL_SEED0 + k, "split": "seen_family"} for k in range(seen_per_family)
            for f in TRAIN_FAMILIES]
    jobs += [{"family": f, "seed": EVAL_SEED0 + k, "split": "heldout_family"} for k in range(heldout_per_family)
             for f in HELD_OUT_FAMILIES if f in FAMILY_ORDER]
    return jobs


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1])
    out.write_text("\n".join(json.dumps(j) for j in eval_plan()) + "\n")
    print(f"{len(eval_plan())} eval jobs -> {out}")

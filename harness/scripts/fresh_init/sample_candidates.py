"""Sample candidate initial states for one task with LIBERO's own placement samplers (env.seed(s); env.reset()) and
score each one exactly like the harness loads it (set_init_state + 5 hold steps, then 50 more hold steps).

    sample_candidates.py SUITE TASK_ID OUT.npz --seeds START:STOP[:STEP]

Writes OUT.npz with states (N, D) float64 (raw reset state, time 0, the official file layout), reset_seeds (N,), and a
metrics JSON string per candidate. Filtering/dedup happens in build_fresh_init.py.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import fresh_init_lib as L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("suite")
    ap.add_argument("task_id", type=int)
    ap.add_argument("out")
    ap.add_argument("--seeds", required=True, help="START:STOP[:STEP] reset seeds (all >= 1000)")
    ap.add_argument("--with-official", action="store_true", help="also score the official .pruned_init states (reference)")
    ap.add_argument("--fresh-env-per-seed", action="store_true",
                    help="construct a new env for every candidate (= what one harness episode does); needed for scenes whose "
                         "seeded reset depends on the env object's earlier resets (articulated-fixture samplers)")
    args = ap.parse_args()
    parts = [int(x) for x in args.seeds.split(":")]
    seeds = list(range(*parts))
    if min(seeds) < 1000:
        raise SystemExit("candidate reset seeds must be >= 1000")
    L.setup_from_env()
    task, bddl, init_dir = L.task_info(args.suite, args.task_id)
    env = L.make_env(bddl)
    env.reset()
    lay = L.layout(env)
    official_metrics = []
    if args.with_official:
        for i, state in enumerate(L.load_official(init_dir, task.name)["pruned_init"]):
            env.seed(i)
            env.reset()
            m = L.evaluate_state(env, state, lay)
            m["idx"] = i
            official_metrics.append(json.dumps(m))
    states, metrics = [], []
    t0 = time.time()
    for s in seeds:
        if args.fresh_env_per_seed:
            env.close()
            env = L.make_env(bddl)
        state = L.env_seed_reset(env, s)
        m = L.evaluate_state(env, state, lay)
        m["reset_seed"] = s
        states.append(state)
        metrics.append(json.dumps(m))
    np.savez(args.out, states=np.stack(states), reset_seeds=np.asarray(seeds, dtype=np.int64),
             metrics=np.asarray(metrics), official_metrics=np.asarray(official_metrics), task=task.name, problem_folder=task.problem_folder,
             objects=np.asarray(lay.free_names), articulations=np.asarray(lay.art_names), nq=lay.nq, nv=lay.nv,
             seconds=time.time() - t0, fresh_env_per_seed=bool(args.fresh_env_per_seed))


if __name__ == "__main__":
    main()

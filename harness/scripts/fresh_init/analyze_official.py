"""Study the official LIBERO init files of one task: what separates the kept (.pruned_init) from the dropped (.init only)
states, how far official states are from each other, how far raw resets with seeds >= 1000 land from them, and whether
seed+reset is reproducible.  Usage: analyze_official.py SUITE TASK_ID OUT_JSON [N_FRESH]"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

import fresh_init_lib as L


def main() -> None:
    suite, task_id, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    n_fresh = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    L.setup_from_env()
    task, bddl, init_dir = L.task_info(suite, task_id)
    env = L.make_env(bddl)
    env.reset()
    lay = L.layout(env)
    off = L.load_official(init_dir, task.name)
    pruned = off["pruned_init"]
    full = off.get("init")
    rep: dict = {"suite": suite, "task_id": task_id, "task": task.name, "dim": int(pruned.shape[1]),
                 "objects": list(lay.free_names), "articulations": list(lay.art_names),
                 "n_pruned": int(len(pruned)), "n_init": int(len(full)) if full is not None else None}
    robot_cols = [i for i in range(1, 1 + lay.nq) if i - 1 not in
                  {a + k for a in lay.free_adr for k in range(7)} | set(lay.art_adr)]
    rep["robot_qpos_std_max"] = float(pruned[:, robot_cols].std(0).max())
    rep["qvel_abs_max"] = float(np.abs(pruned[:, 1 + lay.nq:]).max())
    pos, quat, art = L.poses(pruned, lay)
    rep["pos_span_cm"] = {n: [round(float(v), 2) for v in (pos[:, i].max(0) - pos[:, i].min(0)) * 100] for i, n in enumerate(lay.free_names)}
    rep["yaw_span_deg"] = {n: round(float(np.degrees(np.ptp(np.unwrap(L.yaw_of(quat[:, i]))))), 2) for i, n in enumerate(lay.free_names)}
    rep["art_values"] = {n: [round(float(art[:, i].min()), 4), round(float(art[:, i].max()), 4)] for i, n in enumerate(lay.art_names)}
    # official vs official (leave one out) nearest-neighbour distances
    d = L.pairwise(pruned, pruned, lay)
    np.fill_diagonal(d["pos"], np.inf)
    nn = d["pos"].min(1)
    rep["official_nn_pos_cm"] = [round(float(np.quantile(nn, q)) * 100, 3) for q in (0, 0.1, 0.5, 0.9, 1)]
    # kept vs dropped
    t0 = time.time()
    kept_metrics, dropped_metrics = [], []
    if full is not None and len(full) > len(pruned):
        kept_idx = set(int(np.argmin(np.abs(full - x).max(1))) for x in pruned)
        last = max(kept_idx)
        rep["kept_idx_max"] = last
        for i in range(len(full)):
            m = L.evaluate_state(env, full[i], lay)
            m["idx"] = i
            (kept_metrics if i in kept_idx else dropped_metrics).append(m)
        rep["dropped_idx"] = [m["idx"] for m in dropped_metrics]
    else:
        for i in range(len(pruned)):
            m = L.evaluate_state(env, pruned[i], lay)
            m["idx"] = i
            kept_metrics.append(m)
    rep["kept"] = kept_metrics
    rep["dropped"] = dropped_metrics
    rep["t_eval_s"] = round(time.time() - t0, 1)
    # reproducibility + fresh resets
    a = L.env_seed_reset(env, 1000)
    b = L.env_seed_reset(env, 1000)
    rep["reset_deterministic_maxdiff"] = float(np.abs(a - b).max())
    officials = full if full is not None else pruned
    small = [L.env_seed_reset(env, s) for s in range(0, 5)]
    dd = L.pairwise(np.stack(small), officials, lay)["pos"]
    rep["seed0_4_min_dist_to_official_cm"] = [round(float(x) * 100, 3) for x in dd.min(1)]
    t0 = time.time()
    fresh = np.stack([L.env_seed_reset(env, 1000 + k) for k in range(n_fresh)])
    rep["t_reset_s"] = round((time.time() - t0) / n_fresh, 3)
    df = L.pairwise(fresh, pruned, lay)
    j = df["pos"].argmin(1)
    rep["fresh_nn_pos_cm"] = [round(float(x) * 100, 3) for x in df["pos"].min(1)]
    rep["fresh_nn_rot_deg"] = [round(float(np.degrees(df["rot"][i, j[i]])), 2) for i in range(len(fresh))]
    rep["fresh_nn_art"] = [round(float(df["art"][i, j[i]]), 4) for i in range(len(fresh))]
    dfi = L.pairwise(fresh, officials, lay)["pos"].min(1)
    rep["fresh_nn_pos_vs_all_init_cm"] = [round(float(x) * 100, 3) for x in dfi]
    Path_out = out
    with open(Path_out, "w") as fh:
        json.dump(rep, fh, indent=1)


if __name__ == "__main__":
    main()

"""Filter sampled candidates into per-task fresh init files.

    build_fresh_init.py --candidates DIR --out data/libero_fresh_init --report report.json [--max-states 100]

Per task (candidates DIR/<suite>_<task>_{a,b}.npz from sample_candidates.py):

1. validity (LIBERO's own pruning, reproduced; see INIT_STATES.md): the task is not satisfied at load, after the
   harness's 5 settle steps or during 50 more hold steps; while settling+holding no movable object slides more than
   XY_MAX in xy, rotates more than ROT_MAX or drops more than Z_DROP_MAX (falls off a support); articulations stay
   put (ART_MAX); after holding, no contact penetrates deeper than PEN_MAX and nothing is still moving (SPEED_MAX).
2. official near-duplicates: rejected when, against any official state of the task (.pruned_init and .init),
   every movable object is within tau_task in position AND within ROT_DUP (5 deg) in rotation AND every hinge
   articulation (stove knob, microwave door) within ROT_DUP AND every slide articulation (drawer) within tau_task.
   tau_task = min(1 cm, median nearest-neighbour distance among the 50 official states).
3. the same rule among the accepted fresh states (greedy in reset-seed order), so no two fresh states are
   near-duplicates of each other either.
4. the first --max-states survivors in reset-seed order are written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import fresh_init_lib as L

XY_MAX = 0.05          # m; LIBERO-kept spatial states move <= 0.06, all LIBERO-dropped ones fail this or ROT_MAX
ROT_MAX = 0.35         # rad (20 deg); libero_spatial_5 bowl-on-ramekin settles by 0.285 in every official state
Z_DROP_MAX = 0.20      # m; objects spawn <= ~16 cm above their support
ART_MAX = 0.01         # rad / m
PEN_MAX = 0.005        # m, after settling
SPEED_MAX = 0.05       # max |qvel| after settling
TAU_CAP = 0.01         # m
ROT_DUP = np.radians(5.0)


def valid(m: dict) -> tuple[bool, str]:
    if m["success_raw"] or m["success_settled"] or m["success_any"]:
        return False, "solved_at_reset"
    if m["z_drop_m"] > Z_DROP_MAX:
        return False, "fell"
    if m["xy_move_m"] > XY_MAX or m["total_rot_rad"] > ROT_MAX:
        return False, "unstable"
    if m["art_move"] > ART_MAX:
        return False, "articulation_moved"
    if m["penetration_held_m"] > PEN_MAX:
        return False, "penetration"
    if m["max_speed"] > SPEED_MAX:
        return False, "not_settled"
    return True, "ok"


def norm_dist(d: dict, tau: float) -> np.ndarray:
    """max of the per-component distances in units of their duplicate thresholds (duplicate iff < 1)."""
    return np.maximum.reduce([d["pos"] / tau, d["rot"] / ROT_DUP, d["art_hinge"] / ROT_DUP, d["art_slide"] / tau])


def dup_mask(d: dict, tau: float) -> np.ndarray:
    return (d["pos"] < tau) & (d["rot"] < ROT_DUP) & (d["art_hinge"] < ROT_DUP) & (d["art_slide"] < tau)


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def build_task(suite: str, task_id: int, cand_dir: Path, out_dir: Path, max_states: int, exclude: set[int] = frozenset()) -> dict:
    import torch

    task, bddl, init_dir = L.task_info(suite, task_id)
    env = L.make_env(bddl)
    env.reset()
    lay = L.layout(env)
    env.close()
    files = sorted(cand_dir.glob(f"{suite}_{task_id}_*.npz"))
    states, seeds, metrics, official_metrics = [], [], [], []
    for f in files:
        z = np.load(f)
        if str(z["task"]) != task.name:
            raise SystemExit(f"{f}: task mismatch")
        states.append(z["states"]); seeds.append(z["reset_seeds"]); metrics += [json.loads(x) for x in z["metrics"]]
        official_metrics += [json.loads(x) for x in z["official_metrics"]]
    states = np.concatenate(states); seeds = np.concatenate(seeds)
    order = np.argsort(seeds, kind="stable")
    states, seeds, metrics = states[order], seeds[order], [metrics[i] for i in order]
    if len(set(seeds.tolist())) != len(seeds) or seeds.min() < 1000:
        raise SystemExit(f"{suite}_{task_id}: bad candidate seeds")
    off = L.load_official(init_dir, task.name)
    pruned = off["pruned_init"]
    officials = np.concatenate([pruned] + ([off["init"]] if "init" in off else []))
    if states.shape[1] != pruned.shape[1]:
        raise SystemExit(f"{suite}_{task_id}: state dim {states.shape[1]} != official {pruned.shape[1]}")
    oo = L.pairwise(pruned, pruned, lay)["pos"]
    np.fill_diagonal(oo, np.inf)
    off_nn = oo.min(1)
    tau = float(min(TAU_CAP, np.median(off_nn)))
    reasons: dict[str, int] = {}
    ok_valid = []
    for i, m in enumerate(metrics):
        good, why = valid(m)
        if good and int(seeds[i]) in exclude:
            good, why = False, "recheck_failed"
        reasons[why] = reasons.get(why, 0) + 1
        ok_valid.append(good)
    ok_valid = np.asarray(ok_valid)
    d_off = L.pairwise(states, officials, lay)
    dup_off = dup_mask(d_off, tau).any(1)
    nn_off_pos = d_off["pos"].min(1)
    accepted: list[int] = []
    dup_fresh = 0
    for i in np.flatnonzero(ok_valid & ~dup_off):
        if accepted:
            d = L.pairwise(states[i:i + 1], states[accepted], lay)
            if dup_mask(d, tau).any():
                dup_fresh += 1
                continue
        accepted.append(int(i))
    chosen = accepted[:max_states]
    # guard: exact equality with any official vector is impossible after the filter, check anyway
    exact = int(sum(np.any(np.all(np.isclose(officials, states[i], atol=1e-9, rtol=0), axis=1)) for i in chosen))
    if exact:
        raise SystemExit(f"{suite}_{task_id}: {exact} chosen states equal an official state")
    out = out_dir / task.problem_folder / f"{task.name}.fresh_init"
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(states[chosen], dtype=np.float64)
    torch.save(arr, out)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    chosen_nn = nn_off_pos[chosen]
    chosen_norm = norm_dist(d_off, tau)[chosen].min(1) if chosen else np.zeros(0)
    d_oo = L.pairwise(pruned, pruned, lay)
    oo_norm = norm_dist(d_oo, tau)
    np.fill_diagonal(oo_norm, np.inf)
    off_norm = oo_norm.min(1)
    official_valid = [valid(m)[0] for m in official_metrics]
    meta = {
        "schema": "libero-fresh-init/v1", "suite": suite, "task_id": task_id, "task": task.name,
        "problem_folder": task.problem_folder, "n": len(chosen), "dim": int(arr.shape[1]), "sha256": sha,
        "reset_seeds": [int(seeds[i]) for i in chosen],
        "harness_seed_of_state_k": "1000 + k",
        "official_files_sha256": {k: hashlib.sha256((init_dir / f"{task.name}.{k}").read_bytes()).hexdigest() for k in off},
        "generator": {"commit": git_head(), "procedure": "env.seed(s); env.reset(); env.get_sim_state() (ControlEnv, OSC_POSE)",
                      "candidate_seeds": [int(seeds.min()), int(seeds.max())]},
        "filters": {"xy_max_m": XY_MAX, "rot_max_rad": ROT_MAX, "z_drop_max_m": Z_DROP_MAX, "art_max": ART_MAX,
                    "penetration_max_m": PEN_MAX, "speed_max": SPEED_MAX, "tau_pos_m": tau, "rot_dup_rad": float(ROT_DUP),
                    "hinge_dup_rad": float(ROT_DUP), "slide_dup_m": tau},
        "nn_to_official_pos_m": [round(float(x), 5) for x in chosen_nn],
    }
    out.with_name(out.name + ".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    q = lambda x: [round(float(np.quantile(x, p)) * 100, 3) for p in (0, 0.1, 0.5)] if len(x) else None
    return {"suite": suite, "task_id": task_id, "task": task.name, "path": str(out), "sha256": sha, "n": len(chosen),
            "candidates": int(len(seeds)), "invalid": {k: v for k, v in reasons.items() if k != "ok"},
            "valid": int(ok_valid.sum()), "dup_official": int((ok_valid & dup_off).sum()), "dup_fresh": dup_fresh,
            "accepted_total": len(accepted), "tau_cm": round(tau * 100, 3),
            "official_nn_cm_min_p10_med": q(off_nn), "fresh_nn_to_official_cm_min_p10_med": q(chosen_nn),
            "official_nn_norm_min_p10_med": [round(float(np.quantile(off_norm, p)), 3) for p in (0, 0.1, 0.5)],
            "fresh_nn_norm_min_p10_med": [round(float(np.quantile(chosen_norm, p)), 3) for p in (0, 0.1, 0.5)] if chosen else None,
            "official_pass_filter": f"{sum(official_valid)}/{len(official_valid)}",
            "official_fail_reasons": sorted({valid(m)[1] for m in official_metrics if not valid(m)[0]}),
            "reset_seed_range": [int(seeds[chosen].min()), int(seeds[chosen].max())] if chosen else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--max-states", type=int, default=100)
    ap.add_argument("--only", default=None, help="suite:task to build a single task")
    ap.add_argument("--exclude-seeds", type=Path, default=None,
                    help='json {"suite:task": [reset_seed, ...]} of candidates that failed recheck_all.py')
    args = ap.parse_args()
    L.setup_from_env()
    rows = []
    for suite in L.COMMUNITY_SUITES:
        for task_id in range(10):
            if args.only and args.only != f"{suite}:{task_id}":
                continue
            excl = json.loads(args.exclude_seeds.read_text()) if args.exclude_seeds else {}
            row = build_task(suite, task_id, args.candidates, args.out, args.max_states,
                             {int(x) for x in excl.get(f"{suite}:{task_id}", [])})
            print(json.dumps(row), flush=True)
            rows.append(row)
    args.report.write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())

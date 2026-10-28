"""Re-check every state of the written fresh init files through the harness reset path (no rendering).

    recheck_all.py --root <harness> SUITE:TASK [SUITE:TASK ...] --out DIR

Per fresh state k (seed 1000 + k): env.seed(reset_seed); env.reset() must reproduce the stored vector bit-for-bit
(fixture poses = generation); backend.reset_to_seeded_init_state(seed=1000+k) must return source "fresh"/index k;
then 50 hold steps: predicate never true, no MuJoCo instability warning, objects stay put (same limits as the
builder). Also: the normalized distance to every official state (.pruned_init and .init) is >= 1, and official seeds
0..49 (with eval seeds 11..16) still return init_states[seed].
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import fresh_init_lib as L
import build_fresh_init as B


def warnings_total(env) -> int:
    d = env.sim.data._data
    return int(sum(int(w.number) for w in d.warning))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("tasks", nargs="+")
    ap.add_argument("--fresh-env-regen", action="store_true",
                    help="check the regeneration in a new env per state (exactly one harness episode's history)")
    ap.add_argument("--k-range", default=None, help="START:STOP subset of fresh indices (sharding)")
    args = ap.parse_args()
    import os
    if (args.root / "third_party" / "libero").is_dir():
        os.environ["FRESH_LIBERO_ROOT"] = str(args.root)  # else FRESH_LIBERO_ROOT / this harness checkout provides LIBERO
    L.setup_from_env()
    sys.path.insert(0, str(args.root / "src"))
    from captain_harness.adapters.libero.backend import reset_to_seeded_init_state
    from captain_harness.adapters.libero.fresh_init import DEFAULT_FRESH_DIR, fresh_init_path, load_fresh_init_states
    args.out.mkdir(parents=True, exist_ok=True)
    for spec in args.tasks:
        suite, task_id = spec.split(":")
        task, bddl, init_dir = L.task_info(suite, int(task_id))
        off = L.load_official(init_dir, task.name)
        official = off["pruned_init"]
        officials = np.concatenate([official] + ([off["init"]] if "init" in off else []))
        fresh = load_fresh_init_states(fresh_init_path(args.root / DEFAULT_FRESH_DIR, task.problem_folder, task.name),
                                       expected_dim=official.shape[1])
        meta = json.loads(Path(fresh.path + ".json").read_text())
        tau = float(meta["filters"]["tau_pos_m"])
        env = L.make_env(bddl)
        env.reset()
        lay = L.layout(env)
        fails: dict[str, list[int]] = {}
        def fail(why, k):
            fails.setdefault(why, []).append(k)
        ks = range(len(fresh)) if not args.k_range else range(*[min(int(x), len(fresh)) for x in args.k_range.split(":")])
        for k in ks:
            if args.fresh_env_regen:
                env.close()
                env = L.make_env(bddl)
            pre = L.env_seed_reset(env, fresh.reset_seeds[k])
            if not np.array_equal(pre, fresh.states[k]):
                fail("regen_mismatch", k)
            res = reset_to_seeded_init_state(env, official, seed=1000 + k, fresh_init_states=fresh)
            if res.init_state_source != "fresh" or res.init_state_index != k:
                fail("wrong_selection", k)
            raw = fresh.states[k]
            solved = bool(env.check_success())
            act = L.hold_action(env, res.observation)
            for _ in range(50):
                obs, _, _, _ = env.step(act)
                solved = solved or bool(env.check_success())
            if solved:
                fail("solved", k)
            if warnings_total(env):
                fail("mujoco_warning", k)
            p0, q0, _ = L.poses(raw, lay)
            p2, q2, _ = L.poses(env.get_sim_state(), lay)
            if p0.size and (np.linalg.norm((p2 - p0)[..., :2], axis=-1).max() > B.XY_MAX
                            or L.rot_angle(q2, q0).max() > B.ROT_MAX or (p0[..., 2] - p2[..., 2]).max() > B.Z_DROP_MAX):
                fail("moved", k)
        norm = B.norm_dist(L.pairwise(fresh.states, officials, lay), tau).min(1)
        for k in np.flatnonzero(norm < 1.0):
            fail("near_duplicate", int(k))
        bad_official = []
        for seed in (range(50) if not args.k_range else ()):
            res = reset_to_seeded_init_state(env, official, seed=seed, fresh_init_states=fresh, settling_steps=0)
            if res.init_state_source != "official" or np.abs(env.get_sim_state()[1:] - official[seed][1:]).max() > 1e-12:
                bad_official.append(seed)
        row = {"suite": suite, "task_id": int(task_id), "n": len(fresh), "k_range": args.k_range,
               "fresh_env_regen": bool(args.fresh_env_regen), "fails": fails,
               "min_norm_dist_to_official": round(float(norm.min()), 4), "official_seed_mismatch": bad_official}
        tag = f"_{args.k_range.replace(':', '-')}" if args.k_range else ""
        (args.out / f"{suite}_{task_id}{tag}.json").write_text(json.dumps(row))
        print(json.dumps(row), flush=True)
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

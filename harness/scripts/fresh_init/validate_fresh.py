"""Load fresh states through the harness reset path in a rendering LIBERO env and check them.

    MUJOCO_GL=osmesa validate_fresh.py --root <harness> --out DIR --cases libero_spatial:0:0,3 libero_10:5:0 ...

Per case (suite:task:k[,k...]): reset with seed 1000+k via backend.reset_to_seeded_init_state (the code the env server
runs), then check (a) the reset with the recorded generation seed reproduces the stored state bit-for-bit, so fixture
poses match generation; (b) the task predicate is false after the harness settle; (c) no deep penetration, objects
above the floor; (d) the hidden success predicate flips to True when the first goal object is teleported onto/into its
goal target (On/In goals); (e) official seeds still map to init_states[seed % 50]. Saves agentview PNGs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

import fresh_init_lib as L


def teleport_check(env, lay) -> dict:
    """Move the first goal's object above its target, settle, read the predicate."""
    import mujoco

    goals = list((env.env.parsed_problem or {}).get("goal_state") or [])
    if not goals:
        return {"skipped": "no goals"}
    g = [str(x).lower() for x in goals[0]]
    if g[0] not in ("on", "in") or len(g) != 3:
        return {"skipped": f"goal {g}"}
    obj, target = g[1], g[2]
    objects = {**env.env.objects_dict, **env.env.fixtures_dict}
    base = target
    for suffix in ("_contain_region",):
        base = base.removesuffix(suffix)
    if obj not in env.env.objects_dict:
        return {"skipped": f"object {obj} not movable"}
    m, d = env.sim.model._model, env.sim.data._data
    if base in objects:
        tpos = np.array(d.xpos[m.body(objects[base].root_body).id])
    else:
        site = target
        tpos = np.array(d.site_xpos[m.site(site).id]) if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site) >= 0 else None
        if tpos is None:
            return {"skipped": f"target {target} not found"}
    joint = env.env.objects_dict[obj].joints[-1]
    q = env.sim.data.get_joint_qpos(joint).copy()
    q[:3] = tpos + np.array([0, 0, 0.10])
    env.sim.data.set_joint_qpos(joint, q)
    env.sim.data.set_joint_qvel(joint, np.zeros(6))
    env.sim.forward()
    before = bool(env.check_success())
    act = np.zeros(int(env.robots[0].action_dim)); act[-1] = -1.0
    after = False
    for _ in range(60):
        env.step(act)
        after = after or bool(env.check_success())
    return {"goal": g, "success_after_teleport": after, "success_right_after_move": before}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="harness checkout with data/libero_fresh_init and third_party")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cases", nargs="+", required=True)
    args = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
    if (args.root / "third_party" / "libero").is_dir():
        os.environ["FRESH_LIBERO_ROOT"] = str(args.root)  # else FRESH_LIBERO_ROOT / this harness checkout provides LIBERO
    L.setup_from_env()
    sys.path.insert(0, str(args.root / "src"))
    from PIL import Image
    from libero.libero.envs import OffScreenRenderEnv
    from captain_harness.adapters.libero.backend import reset_to_seeded_init_state
    from captain_harness.adapters.libero.fresh_init import DEFAULT_FRESH_DIR, fresh_init_path, load_fresh_init_states
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in args.cases:
        suite, task_id, ks = case.split(":")
        task, bddl, init_dir = L.task_info(suite, int(task_id))
        official = L.load_official(init_dir, task.name)["pruned_init"]
        fresh = load_fresh_init_states(fresh_init_path(args.root / DEFAULT_FRESH_DIR, task.problem_folder, task.name),
                                       expected_dim=official.shape[1])
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), controller="OSC_POSE", horizon=45000,
                                 camera_names=["agentview", "robot0_eye_in_hand"], camera_heights=256, camera_widths=256,
                                 camera_depths=False, control_freq=20)
        env.reset()
        lay = L.layout(env)
        for k in [int(x) for x in ks.split(",")]:
            seed = 1000 + k
            row = {"suite": suite, "task_id": int(task_id), "k": k, "seed": seed, "reset_seed": fresh.reset_seeds[k]}
            pre = L.env_seed_reset(env, fresh.reset_seeds[k])
            row["regen_maxdiff"] = float(np.abs(pre - fresh.states[k]).max())
            res = reset_to_seeded_init_state(env, official, seed=seed, fresh_init_states=fresh, settling_steps=5)
            row.update(source=res.init_state_source, index=res.init_state_index, success_after_settle=bool(env.check_success()),
                       penetration_m=L.penetration(env))
            p, _, _ = L.poses(env.get_sim_state(), lay)
            row["min_obj_z"] = float(p[..., 2].min()) if p.size else None
            dist = L.pairwise(env.get_sim_state()[None], official, lay)["pos"].min()
            row["settled_nn_to_official_cm"] = round(float(dist) * 100, 3)
            img = np.asarray(res.observation["agentview_image"])[::-1]
            path = args.out / f"{suite}_{task_id}_fresh{k}.png"
            Image.fromarray(img).save(path)
            row["image"] = str(path)
            row["predicate"] = teleport_check(env, lay)
            rows.append(row)
            print(json.dumps(row), flush=True)
        # official seeds keep their mapping (eval seeds 11-16 included)
        for seed in (11, 16):
            res = reset_to_seeded_init_state(env, official, seed=seed, fresh_init_states=fresh, settling_steps=0)
            same = float(np.abs(env.get_sim_state()[1:] - official[seed % len(official)][1:]).max())
            img = np.asarray(res.observation["agentview_image"])[::-1]
            Image.fromarray(img).save(args.out / f"{suite}_{task_id}_official{seed}.png")
            row = {"suite": suite, "task_id": int(task_id), "seed": seed, "source": res.init_state_source,
                   "index": res.init_state_index, "official_state_maxdiff": same}
            rows.append(row)
            print(json.dumps(row), flush=True)
        env.close()
    (args.out / "validation.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Physics gate of registry objects: drop-settle + scripted top-down grasp/lift/hold through SoarmDriver.command.

    python tools/gate_assets.py [--ids 'cube_*'] [--tries 6] [--workers 12] [--out gate.jsonl] [--update-registry]

Per object `--tries` independent trials (layout in the reachable annulus, random yaw, side jitter); the first four use
the corners of the row's mass x friction ranges (light/slippery ... heavy/grippy), the rest random draws. A trial:
1. drop: the object is released 3-5 cm above the table in a random orientation; after 2 s it must rest (|v| < 1 cm/s,
   |w| < 0.2 rad/s), on a face (cube: some axis within 5 deg of vertical), on the table, inside the arm's reach.
2. grasp (fresh settle at the layout): open, top-down over the object, close across a face pair (meshes: across the
   narrow horizontal extent, at mid height), lift the grip point to 7.5 cm (taller objects: grip + 5 cm, <= 11.5 cm),
   hold 1.5 s; success = held + bottom >= 4 cm (the `lifted` predicate) after the hold, slip < 1 cm.
   The SO-101 jaw is a FIXED jaw (inner face 1.8 cm from the grip point on the -closing side) + a hinged jaw, so an
   object wider than ~3.6 cm is grasped off-centre: grip point = centre - closing_axis * (w/2 - 1.8 cm + 4 mm).
   Layout r 0.15-0.25 m, |angle| <= 60 deg: where an exact top-down pose at 7.5 cm is reachable (reach map); the gate measures the object, not the arm's workspace edge.
graspable = every drop ok, >= 5 valid grasp trials, grasp success rate >= 0.8 over the valid trials (a layout whose
pre-grasp pose the arm cannot reach is redrawn, up to 3 times, and excluded). Uses privileged poses (a test, not a policy).
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import multiprocessing as mp
import time
from dataclasses import replace
import zlib
from pathlib import Path

from OpenGL import GL  # noqa: F401  (load OSMesa before mujoco)
import mujoco
import numpy as np

from scripted_pick import move, topdown  # tools/ sibling

from soarm import asset_mj  # noqa: E402
from soarm.objects import REGISTRY_PATH, asset_spec, load_registry  # noqa: E402
from soarm.predicates import lifted  # noqa: E402
from soarm.scene import Randomization  # noqa: E402
from soarm.sim import SoarmSim  # noqa: E402
from soarm.tasks import Placement, TaskInstance  # noqa: E402

PASS_RATE = 0.8
HOME_XY = np.array([0.18, 0.0])
GATE_R, GATE_ANG = (0.15, 0.25), np.deg2rad(60.0)
FIXED_JAW = 0.018            # fixed jaw inner face, distance from the grip point along -closing axis (twin.py)
JAW_CLEARANCE = 0.004
ARM_UNREACHABLE = 0.010      # pre-grasp pose error above which the layout is an arm limit (redrawn, not counted)

asset_mj.install_scene_hook()


class GateSim(SoarmSim):
    def __init__(self, task: TaskInstance, rand: Randomization) -> None:      # noqa: super().__init__ samples a family
        self._build(task, rand)


def _trial_spec(row, k, rng):
    spec = asset_spec(rng, row, "obj_0")
    corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
    if k < len(corners):
        im, jf = corners[k]
        spec.attrs["friction"] = float(row["friction_range"][jf])
        lo, hi = row["mass_range_kg"]
        spec = replace(spec, mass=float(hi if im else lo))
    return spec


def _layout(rng):
    while True:
        r, a = rng.uniform(*GATE_R), rng.uniform(-GATE_ANG, GATE_ANG)
        xy = np.array([r * np.cos(a), r * np.sin(a)])
        if np.linalg.norm(xy - HOME_XY) > 0.09:      # drop clear of the gripper at home
            return xy


def _free_adr(sim, name):
    j = sim.model.joint(f"{name}:free")
    return int(sim.model.jnt_qposadr[j.id]), int(sim.model.jnt_dofadr[j.id])


def drop_test(sim: GateSim, name: str, rng) -> dict:
    m, d = sim.model, sim.data
    qa, va = _free_adr(sim, name)
    h = 0.5 * float(np.linalg.norm(sim.specs[name].size))
    d.qpos[qa:qa + 3] = [*d.qpos[qa:qa + 2], h + rng.uniform(0.03, 0.05)]
    q = rng.normal(size=4)
    d.qpos[qa + 3:qa + 7] = q / np.linalg.norm(q)
    d.qvel[va:va + 6] = 0
    mujoco.mj_forward(m, d)
    tilt0 = float(np.degrees(np.arccos(min(1.0, float(np.abs(d.xmat[sim.body_of[name]].reshape(3, 3)[2, :]).max())))))
    sim.tick(None, None, ticks=40)
    p = sim.poses()[name]
    v = float(np.linalg.norm(d.qvel[va:va + 3]))
    w = float(np.linalg.norm(d.qvel[va + 3:va + 6]))
    tilt = float(np.degrees(np.arccos(min(1.0, float(np.abs(p.rot[2, :]).max())))))
    r = float(np.hypot(*p.pos[:2]))
    bottom = sim.world().bottom(name)
    cube = sim.specs[name].kind == "cube"
    # cubes land on a face, flush with the table; meshes: any stable pose (the world's bottom() is a bounding-box
    # estimate, which a tilted mesh pushes below the table), centre above the table
    pose_ok = tilt < 5.0 and abs(bottom) < 0.003 if cube else float(p.pos[2]) > 0.0
    ok = v < 0.01 and w < 0.2 and pose_ok and 0.10 <= r <= 0.32
    return dict(ok=bool(ok), v=round(v, 4), w=round(w, 3), tilt0_deg=round(tilt0, 1), tilt_deg=round(tilt, 2),
                bottom_mm=round(1000 * bottom, 2), r=round(r, 3))


def grasp_test(sim: GateSim, name: str, png: str | None = None) -> dict:
    from soarm.driver import SoarmDriver
    from soarm.twin import build_twin_xml
    drv = SoarmDriver(sim, build_twin_xml(), image_size=(256, 256) if png else (64, 64))
    try:
        p = sim.poses()[name]
        o = sim.specs[name]
        yaw = float(np.arctan2(p.rot[1, 0], p.rot[0, 0]))
        if o.kind == "cube":
            yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4      # cube: any face pair, least wrist roll
        else:                                                     # close across the narrow horizontal extent
            yaw = (yaw + (np.pi / 2 if o.size[0] > o.size[1] else 0.0) + np.pi / 2) % np.pi - np.pi / 2
        h = float(o.size[2])
        z_grip = float(np.clip(p.pos[2], 0.016, max(0.016, h - 0.012)))
        z_pre = max(0.075, h + 0.012)
        z_lift = min(0.115, max(z_pre, z_grip + 0.05))
        width = float(min(o.size[:2]))
        off = max(0.0, width / 2 - FIXED_JAW + JAW_CLEARANCE)
        xy = p.pos[:2] - off * np.array([np.cos(yaw), np.sin(yaw)])     # eef x (closing axis) = (cos yaw, sin yaw)
        err1 = move(drv, topdown([*xy, z_pre], yaw), -1.0)
        err2 = move(drv, topdown([*xy, z_grip], yaw), -1.0, step=0.006)
        for _ in range(15):
            drv.command(topdown([*xy, z_grip], yaw), 1.0)
        held = name in sim.held()
        move(drv, topdown([*xy, z_lift], yaw), 1.0)
        z0 = float(sim.poses()[name].pos[2])
        for _ in range(30):
            drv.command(topdown([*xy, z_lift], yaw), 1.0)
        z1 = float(sim.poses()[name].pos[2])
        ok = lifted(sim.world(), name)
        if png:
            from PIL import Image
            Image.fromarray(drv.frames(["agentview"])["agentview"].rgb).save(png)
        return dict(ok=bool(ok and z0 - z1 < 0.01), held=bool(held), slip_mm=round(1000 * (z0 - z1), 2),
                    pregrasp_err_mm=round(1000 * err1, 1), descend_err_mm=round(1000 * err2, 1),
                    arm_unreachable=bool(err1 > ARM_UNREACHABLE), jaw=round(sim.open_fraction(), 3))
    finally:
        drv.close()


def trial(args) -> dict:
    """One gate trial; a layout where the arm cannot reach the pre-grasp pose (an arm-workspace limit, independent of
    the object) is redrawn up to 3 times and counted in `redraws`."""
    row, k, seed, png = args
    rng = np.random.default_rng([seed, zlib.crc32(row["id"].encode()), k])
    t0 = time.time()
    spec = _trial_spec(row, k, rng)
    out = dict(id=row["id"], trial=k, mass_g=round(1000 * spec.mass, 1), friction=round(spec.attrs["friction"], 3),
               side_cm=round(100 * spec.size[0], 2), redraws=0)
    for attempt in range(4):
        xy = _layout(rng)
        yaw = float(rng.uniform(-np.pi, np.pi))
        task = TaskInstance(family="asset_gate", seed=seed, language="", objects=[spec],
                            layout={spec.name: Placement(xy=(float(xy[0]), float(xy[1])), yaw=yaw)},
                            atoms=[("lifted", (spec.name,), {})], roles={"target": spec.name})
        out.update(xy=[round(float(v), 3) for v in xy], redraws=attempt)
        try:
            sim = GateSim(task, Randomization.sample(np.random.default_rng([seed, k, attempt, 7])))
            out["drop"] = drop_test(sim, spec.name, rng)
            sim.reset()
            out["grasp"] = grasp_test(sim, spec.name, png and f"{png}/{row['id']}_{k}.png")
        except Exception as exc:          # a crash is a failed trial, reported
            out["error"] = f"{type(exc).__name__}: {exc}"
            break
        if not out["grasp"]["arm_unreachable"]:
            break
    out["wall_s"] = round(time.time() - t0, 1)
    return out


def summarize(rows: list[dict], results: list[dict]) -> dict[str, dict]:
    by: dict[str, list[dict]] = {}
    for r in results:
        by.setdefault(r["id"], []).append(r)
    out = {}
    for rid, rs in by.items():
        valid = [r for r in rs if not r.get("grasp", {}).get("arm_unreachable")]
        n = len(rs)
        drop_ok = sum(bool(r.get("drop", {}).get("ok")) for r in rs)
        grasp_ok = sum(bool(r.get("grasp", {}).get("ok")) for r in valid)
        rate = grasp_ok / max(1, len(valid))
        out[rid] = dict(graspable=bool(drop_ok == n and len(valid) >= 5 and rate >= PASS_RATE),
                        scripted_grasp_success=round(rate, 3),
                        gate=dict(trials=n, valid=len(valid), drop_ok=drop_ok, grasp_ok=grasp_ok,
                                  redraws=sum(r.get("redraws", 0) for r in rs), errors=sum("error" in r for r in rs),
                                  date=time.strftime("%Y-%m-%d"), tool="tools/gate_assets.py"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="*", help="fnmatch pattern(s), comma-separated")
    ap.add_argument("--tries", type=int, default=6)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="gate_results.jsonl")
    ap.add_argument("--png", default=None, help="dir: save the agentview after each lift")
    ap.add_argument("--update-registry", action="store_true")
    a = ap.parse_args()
    pats = a.ids.split(",")
    rows = [r for r in load_registry() if any(fnmatch.fnmatch(r["id"], p) for p in pats)]
    if a.png:
        Path(a.png).mkdir(parents=True, exist_ok=True)
    jobs = [(r, k, a.seed, a.png) for r in rows for k in range(a.tries)]
    print(f"{len(rows)} objects x {a.tries} trials = {len(jobs)} on {a.workers} workers", flush=True)
    results = []
    with open(a.out, "w") as f, mp.get_context("fork").Pool(a.workers, maxtasksperchild=8) as pool:
        for i, res in enumerate(pool.imap_unordered(trial, jobs)):
            results.append(res)
            f.write(json.dumps(res) + "\n")
            f.flush()
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)
    summ = summarize(rows, results)
    good = sum(s["graspable"] for s in summ.values())
    print(f"graspable {good}/{len(summ)}; trials grasp ok {sum(s['gate']['grasp_ok'] for s in summ.values())}"
          f"/{len(results)}, drop ok {sum(s['gate']['drop_ok'] for s in summ.values())}/{len(results)}")
    for rid, s in sorted(summ.items()):
        if not s["graspable"]:
            print("  FAIL", rid, s["scripted_grasp_success"], s["gate"])
    if a.update_registry:
        allrows = load_registry()
        for r in allrows:
            if r["id"] in summ:
                r.update(summ[r["id"]])
        REGISTRY_PATH.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in allrows))
        print(f"registry updated: {REGISTRY_PATH}")


if __name__ == "__main__":
    main()

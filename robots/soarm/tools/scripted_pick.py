"""Physics sanity check: scripted top-down pick (and optional place) through SoarmDriver.command only.

    python tools/scripted_pick.py --family cube_pick_lift --seeds 0-9 [--png /tmp/pick]   (task spec id or tid)
Prints per seed: grasped/lifted, final success, sim time. Uses privileged object poses (this is a test, not a policy).
"""
from __future__ import annotations

import argparse
import time

from OpenGL import GL  # noqa: F401  (load OSMesa before mujoco)
import numpy as np

from soarm.driver import SoarmDriver
from soarm.sim import SoarmSim
from soarm.twin import build_twin_xml

# Wide objects: the fixed jaw face sits ~1.8 cm from the grip point, so a centred grip lands it on objects wider than
# ~3.5 cm; shift the grip point by (width/2 - margin) along -closing direction (intro v3 tells the agent the same).
FIXED_JAW_MARGIN = 0.014
DOWN = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])   # eef x (closing) = world x, z (approach) = world -z


def topdown(xyz, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    g = np.eye(4)
    g[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ DOWN
    g[:3, 3] = xyz
    return g


def move(drv, goal, grip, step=0.01, settle=10):
    cur = drv.sim.grip_pose()
    start = cur[:3, 3].copy()
    n = max(1, int(np.ceil(np.linalg.norm(goal[:3, 3] - start) / step)))
    for k in range(1, n + 1):
        g = goal.copy()
        g[:3, 3] = start + (goal[:3, 3] - start) * k / n
        drv.command(g, grip)
    for _ in range(settle):
        drv.command(goal, grip)
    return float(np.linalg.norm(drv.sim.grip_pose()[:3, 3] - goal[:3, 3]))


def grasp_yaw(spec, rot):
    """Close across the narrow side: closing axis (eef x) along the object's short horizontal axis."""
    yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
    if spec.kind == "block":
        yaw += np.pi / 2       # block long axis = object x -> close along object y
    yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
    return yaw


def run(family, seed, png=None):
    sim = SoarmSim(family, seed)
    drv = SoarmDriver(sim, build_twin_xml())
    t0 = time.time()
    target = sim.task.roles["target"]
    spec = sim.specs[target]
    p = sim.poses()[target]
    yaw = grasp_yaw(spec, p.rot)
    xy = p.pos[:2] - max(0.0, min(spec.size[:2]) / 2 - FIXED_JAW_MARGIN) * np.array([np.cos(yaw), np.sin(yaw)])
    z_grip = max(0.016, float(p.pos[2]))
    err1 = move(drv, topdown([*xy, 0.075], yaw), -1.0)
    err2 = move(drv, topdown([*xy, z_grip], yaw), -1.0, step=0.006)
    for _ in range(15):
        drv.command(topdown([*xy, z_grip], yaw), 1.0)
    held = target in sim.held()
    move(drv, topdown([*xy, 0.075], yaw), 1.0)
    lifted = target in sim.held() and sim.world().bottom(target) > 0.04
    dest = sim.task.roles.get("destination")
    if dest is not None:      # place: over the destination, lower until the object is just above its support, open
        w = sim.world()
        dpos = w.poses[dest].pos.copy()
        off = w.poses[target].pos[:2] - sim.grip_pose()[:2, 3]          # object centre relative to the grip point
        delta = 0.0
        if sim.specs[dest].attrs.get("slot"):     # tight slot: turn the cube's faces parallel to the slot walls
            yaw_of = lambda r: float(np.arctan2(r[1, 0], r[0, 0]))    # noqa: E731
            delta = (yaw_of(w.poses[dest].rot) - yaw_of(w.poses[target].rot) + np.pi / 4) % (np.pi / 2) - np.pi / 4
        c, s = np.cos(delta), np.sin(delta)
        dpos[:2] -= np.array([[c, -s], [s, c]]) @ off                   # object (not grip point) over the target
        yaw = yaw + delta
        support = w.top(dest)
        below = float(sim.grip_pose()[2, 3]) - w.bottom(target)      # grip point height above the object's bottom
        z_rel = support + below + 0.008
        move(drv, topdown([*dpos[:2], max(0.075, z_rel)], yaw), 1.0)
        move(drv, topdown([*dpos[:2], z_rel], yaw), 1.0, step=0.006)
        for _ in range(10):
            drv.command(topdown([*dpos[:2], z_rel], yaw), -1.0)
        move(drv, topdown([*dpos[:2], max(0.075, z_rel + 0.03)], yaw), -1.0)
        for _ in range(20):
            drv.hold()
    out = dict(seed=seed, target=spec.kind, width=round(min(spec.size[:2]), 3), reach_err=round(max(err1, err2), 4),
               held=held, lifted=lifted, success=sim.success(), jaw=round(sim.open_fraction(), 3),
               wall_s=round(time.time() - t0, 2), sim_s=round(sim.time_s, 1),
               force=np.round(drv._wrist_force(), 2).tolist())
    if png:
        from PIL import Image
        Image.fromarray(drv.frames(["agentview"])["agentview"].rgb).save(f"{png}_{family}_{seed}.png")
    drv.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="pick_lift")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--png", default=None)
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-")
    rows = [run(a.family, s, a.png) for s in range(int(lo), int(hi or lo) + 1)]
    for r in rows:
        print(r)
    print("lifted", sum(r["lifted"] for r in rows), "/", len(rows), "success", sum(r["success"] for r in rows))


if __name__ == "__main__":
    main()

"""Physics solvability: run a spec instance in the real twin physics (soarm.sim + soarm.driver, unchanged) and execute the
planner's steps with scripted top-down pick/place through SoarmDriver.command (privileged object poses: a test of the
task, not a policy). Also the physics start check (settle: nothing moved/tipped, no goal atom true), as SoarmSim does.
"""
from __future__ import annotations

import time

from OpenGL import GL  # noqa: F401  (load OSMesa before mujoco)
import numpy as np

from soarm.driver import SoarmDriver
from soarm.scene import Randomization
from soarm.sim import SoarmSim
from soarm.tasks import Reject, SceneConfig
from soarm.twin import build_twin_xml

from .planner import LIFT_BOTTOM, NoPlan, Step, plan, reach_band

DOWN = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])   # eef x (closing) = world x, z (approach) = world -z
_TWIN = None
FIXED_JAW_FACE = 0.018         # fixed jaw inner face, -x_eef from the grip point (soarm.twin GRIP_OFFSET_SITE)
TRANSIT_Z = 0.075              # empty-handed transit height of the grip point (lowered where reach ends)


class SpecSim(SoarmSim):
    """SoarmSim built from a given TaskInstance (SoarmSim samples its own from a family name)."""

    def __init__(self, task, seed: int, randomize: bool = True, config: SceneConfig | None = None) -> None:
        config = config or SceneConfig()
        self._build(task, Randomization.sample(np.random.default_rng([seed, 0, 7]), randomize, config))
        self.check_start()


def topdown(xyz, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    g = np.eye(4)
    g[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ DOWN
    g[:3, 3] = xyz
    return g


def _move(drv, xyz, yaw, grip, step=0.01, settle=10):
    start = drv.sim.grip_pose()[:3, 3].copy()
    goal = topdown(xyz, yaw)
    n = max(1, int(np.ceil(np.linalg.norm(goal[:3, 3] - start) / step)))
    for k in range(1, n + 1):
        g = goal.copy()
        g[:3, 3] = start + (goal[:3, 3] - start) * k / n
        drv.command(g, grip)
    for _ in range(settle):
        drv.command(goal, grip)
    return float(np.linalg.norm(drv.sim.grip_pose()[:3, 3] - goal[:3, 3]))


def _grasp_yaw(spec, rot):
    yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
    if spec.kind == "block" or (spec.kind == "registry" and spec.size[0] > spec.size[1]):
        yaw += np.pi / 2           # close across the narrow horizontal extent (eef x = closing axis)
    period = np.pi / 2 if spec.kind == "cube" else np.pi
    return (yaw + period / 2) % period - period / 2


def _safe_z(xy, z):
    """Highest grip height <= z that is still inside the reach band at xy (retreat after a release)."""
    r = float(np.hypot(*xy))
    while z > 0.02:
        b = reach_band(z)
        if b and b[0] <= r <= b[1]:
            return z
        z -= 0.005
    return z


def _transit(drv, xy, yaw, z_low):
    """Empty-handed move to above xy: up to the transit height, across, down to z_low (never sweep low through objects)."""
    cur = drv.sim.grip_pose()[:3, 3].copy()
    z_t = min(_safe_z(cur[:2], TRANSIT_Z), _safe_z(xy, TRANSIT_Z))
    if cur[2] < z_t - 0.005:
        _move(drv, [*cur[:2], z_t], yaw, -1.0)
    _move(drv, [*xy, max(z_t, z_low)], yaw, -1.0)
    _move(drv, [*xy, z_low], yaw, -1.0, step=0.006)


def _attempt(drv: SoarmDriver, st: Step) -> dict:
    """One scripted pick (+ place) of `st.obj` from wherever it is now; returns what happened."""
    sim = drv.sim
    w = sim.world()
    p = w.poses[st.obj]
    spec = sim.specs[st.obj]
    yaw = _grasp_yaw(spec, p.rot)
    centre, z_grip = p.pos[:2], max(0.016, float(p.pos[2]))
    if spec.kind == "registry":   # meshes (as tools/gate_assets.py): aim at the centre of mass, >= 1.2 cm below the top
        com = sim.data.xipos[sim.body_of[st.obj]]
        centre = com[:2].copy()
        z_grip = float(np.clip(com[2], 0.016, max(0.016, w.top(st.obj) - 0.012)))
    # the fixed jaw's inner face is 1.8 cm from the grip point on the -x_eef side (soarm.twin): objects wider than
    # ~3.5 cm land on it when the grip point is centred, so shift the grip point toward the hinged jaw, which is
    # -(cos yaw, sin yaw) in the world for topdown(yaw) (measured 11-10 on 4.0-5.0 cm cubes, 6 seeds each: centred
    # 1/18 lifted, shifted the other way 3/18, this way 18/18)
    shift = max(0.0, 0.5 * min(spec.size[:2]) - FIXED_JAW_FACE + 0.004)
    gxy = centre - shift * np.array([np.cos(yaw), np.sin(yaw)])
    _transit(drv, gxy, yaw, max(st.carry_z, z_grip + 0.03))
    _move(drv, [*gxy, z_grip], yaw, -1.0, step=0.006)
    for _ in range(15):
        drv.command(topdown([*gxy, z_grip], yaw), 1.0)
    _move(drv, [*gxy, st.carry_z], yaw, 1.0)
    held_off = sim.poses()[st.obj].pos[:2] - sim.grip_pose()[:2, 3]     # object centre relative to the grip point
    held = st.obj in sim.held()
    if st.hold:
        z_hold = st.carry_z
        lack = LIFT_BOTTOM - sim.world().bottom(st.obj)
        if spec.kind == "registry" and lack > 0:     # a mesh tilted in the jaw: its bounding box reaches lower
            z_hold = max(z_hold, _safe_z(gxy, st.carry_z + lack))
            _move(drv, [*gxy, z_hold], yaw, 1.0)
        for _ in range(20):
            drv.command(topdown([*gxy, z_hold], yaw), 1.0)
        return {"held": held, "ok": st.obj in sim.held()}
    if not held:                                  # missed: open and let the next attempt re-read the pose
        _move(drv, [*gxy, _safe_z(gxy, st.carry_z + 0.02)], yaw, -1.0)
        return {"held": False, "ok": False}
    w = sim.world()
    if st.support is None:
        target, support_top = np.asarray(st.xy), 0.0
    else:
        target = w.poses[st.support].pos[:2] + np.asarray(st.offset)
        support_top = w.top(st.support)
    below = float(sim.grip_pose()[2, 3]) - w.bottom(st.obj)
    z_rel = support_top + below + 0.008
    dxy = target - held_off
    _move(drv, [*dxy, max(st.carry_z, z_rel)], yaw, 1.0)
    _move(drv, [*dxy, z_rel], yaw, 1.0, step=0.006)
    for _ in range(10):
        drv.command(topdown([*dxy, z_rel], yaw), -1.0)
    _move(drv, [*dxy, _safe_z(dxy, z_rel + 0.03)], yaw, -1.0)
    for _ in range(10):
        drv.hold()
    w = sim.world()
    err = float(np.linalg.norm(w.poses[st.obj].pos[:2] - target))
    resting = abs(w.bottom(st.obj) - (support_top if st.support is not None else 0.0)) < 0.012
    tol = 0.012 if st.support is None or sim.specs[st.support].category == "object" else 0.02
    return {"held": True, "err_xy": round(err, 4), "ok": err <= tol and resting and st.obj not in sim.held()}


def execute(drv: SoarmDriver, steps: list[Step], attempts: int = 3) -> dict:
    """Steps in order; a step whose object was missed or landed off target is retried (closed loop on privileged poses)."""
    sim = drv.sim
    log = []
    for st in steps:
        for k in range(attempts):
            res = _attempt(drv, st)
            if res["ok"]:
                break
        log.append({"obj": st.obj, "why": st.why, "attempts": k + 1, **res})
    for _ in range(20):
        drv.hold() if not any(s.hold for s in steps) else drv.command(None, 1.0)
    return {"steps": log, "atoms": [bool(v) for v in sim.atom_values()], "success": sim.success()}


def run_instance(task, seed: int, config: SceneConfig | None = None) -> dict:
    """Physics start check + scripted solve of one instance. Returns a row for the verifier."""
    global _TWIN
    t0 = time.time()
    try:
        steps = plan(task)
    except NoPlan as exc:
        return {"stage": "plan", "ok": False, "detail": str(exc)}
    try:
        sim = SpecSim(task, seed, config=config)
    except Reject as exc:
        return {"stage": "start", "ok": False, "detail": str(exc)}
    _TWIN = _TWIN or build_twin_xml()
    drv = SoarmDriver(sim, _TWIN)
    try:
        out = execute(drv, steps)
    finally:
        drv.close()
    return {"stage": "solve", "ok": bool(out["success"]), "n_steps": len(steps), "detail": out,
            "wall_s": round(time.time() - t0, 1), "sim_s": round(sim.time_s, 1)}

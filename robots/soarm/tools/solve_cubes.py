"""Scripted proofs for the cube skills (soarm/skills/specs.py) through the UNCHANGED CaPtain tool surface.

Every action is a tool call on an in-process harness env server (captain_harness EpisodeServerState: the same
move_to / hold / release code, IK path checks, contact guard and receipts the agent gets). Object poses are read from
the simulator (privileged: this proves the arm + tools can do the skill, it is not a policy); --pose-noise-mm adds
Gaussian noise to every pose read, standing in for measurement error.

    python tools/solve_cubes.py --spec cube_precise_stack --seeds 0-9 [--pose-noise-mm 1.5] [--out DIR] [--png]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import traceback
from pathlib import Path

from OpenGL import GL  # noqa: F401  (load OSMesa before mujoco)
import numpy as np

from soarm.skills.sim import SkillSim
from soarm.skills.specs import SPECS
from soarm.predicates import aligned_on, on_top

FIXED_JAW = -0.018      # fixed jaw inner face along the closing axis (eef x), relative to the grip point (twin.py)
JAW_CLEAR = 0.004       # keep the fixed jaw this far off the object while descending
HOVER = 0.075           # top-down travel height of the grip point


def quat(theta: float) -> list[float]:
    """Top-down, jaws closing along table direction theta (TASK.md intro formula)."""
    return [0.0, float(np.cos(theta / 2)), float(np.sin(theta / 2)), 0.0]


def yaw_of(rot) -> float:
    return float(np.arctan2(rot[1, 0], rot[0, 0]))


def wrap_quarter(a: float) -> float:
    """Cube face direction closest to 0 (cubes are 90-degree symmetric)."""
    return float((a + np.pi / 4) % (np.pi / 2) - np.pi / 4)


class Failed(Exception):
    pass


class Robot:
    """Tool calls on the episode server + privileged reads of the sim."""

    def __init__(self, state, sim: SkillSim, noise_m: float, rng) -> None:
        self.st, self.sim, self.noise, self.rng = state, sim, noise_m, rng
        self.calls = 0
        self.log: list[str] = []

    def call(self, name: str, **args):
        self.calls += 1
        r = self.st.call(name, args)
        ok = bool(r.get("success"))
        poses = " ".join(f"{n}=({p.pos[0]:.3f},{p.pos[1]:.3f},{p.pos[2]:.3f} t{np.degrees(np.arccos(min(1, np.abs(p.rot[2]).max()))):.0f})"
                         for n, p in self.sim.poses().items() if self.sim.specs[n].kind == "cube")
        e = self.sim.grip_pose()[:3, 3]
        self.log.append(f"{name}({json.dumps(args)[:120]}) -> {ok} {str(r.get('failure_category') or '')} "
                        f"{str(r.get('receipt') or r.get('message') or '')[:160]} || eef=({e[0]:.3f},{e[1]:.3f},{e[2]:.3f}) "
                        f"held={sorted(self.sim.held())} {poses}")
        return r

    def move(self, xyz, theta, *, must=True, **kw):
        return self.move_q(xyz, quat(theta), must=must, **kw)

    def move_q(self, xyz, q, *, must=True, **kw):
        r = self.call("move_to", position=[float(v) for v in xyz], quaternion=q, **kw)
        if not r.get("success") and r.get("message") == "tracking_error_limit":
            # fast wrist rotation / long sweep outran the servos: the same goal again, slower (the receipt says so)
            r = self.call("move_to", position=[float(v) for v in xyz], quaternion=q,
                          **{**kw, "max_speed_m_s": 0.02, "max_angular_speed_rad_s": 0.25})
        if not r.get("success") and not r.get("executed") and r.get("message") in (
                "requested_cartesian_path_ik_unavailable", "requested_cartesian_path_self_collision"):
            # a straight path that rotates the wrist while translating left the reachable set: split it
            e = self.sim.grip_pose()
            from scipy.spatial.transform import Rotation
            q0 = Rotation.from_matrix(e[:3, :3]).as_quat()
            q_now = [float(q0[3]), float(q0[0]), float(q0[1]), float(q0[2])]
            for first in ({"position": [float(v) for v in e[:3, 3]], "quaternion": q},       # rotate in place
                          {"position": [float(v) for v in xyz], "quaternion": q_now}):       # or translate first
                r1 = self.call("move_to", **first, **kw)
                if r1.get("success"):
                    r = self.call("move_to", position=[float(v) for v in xyz], quaternion=q, **kw)
                    break
        if must and not r.get("success"):
            raise Failed(self.log[-1])
        return r

    def grip(self, what: str):
        return self.call("move_to", gripper=what)

    # ------------------------------------------------------------- privileged state (+ measurement noise)
    def pos(self, name: str) -> np.ndarray:
        p = self.sim.poses()[name].pos.copy()
        return p + np.r_[self.rng.normal(0, self.noise, 2), 0.0]

    def yaw(self, name: str) -> float:
        return yaw_of(self.sim.poses()[name].rot) + float(self.rng.normal(0, np.radians(2.0) if self.noise else 0))

    def size(self, name: str) -> float:
        return float(self.sim.specs[name].size[0])

    def eef(self) -> np.ndarray:
        return self.sim.grip_pose()

    def held(self, name: str) -> bool:
        return name in self.sim.held()


# ------------------------------------------------------------------ primitives
def axis(theta: float) -> np.ndarray:
    return np.array([np.cos(theta), np.sin(theta), 0.0])


def grasp_theta(rb: Robot, name: str) -> float:
    """Close across two faces; of the two face directions take the one nearer the radial direction (wrist friendlier)."""
    y = wrap_quarter(rb.yaw(name))
    p = rb.pos(name)
    radial = np.arctan2(p[1], p[0])
    cands = [y + k * np.pi / 2 for k in range(-2, 3)]
    cands = [c for c in cands if -np.pi / 2 - 1e-6 <= c <= np.pi / 2 + 1e-6]
    return min(cands, key=lambda c: abs(np.angle(np.exp(1j * (c - radial)))))


# Top-down reach of the grip point: highest z at which the twin IK (the same IK move_to checks along its path) reaches
# a top-down pose. Coarse map at y = 0, closing along x (11-10): r 0.12 -> 5 cm, 0.14 -> 7, 0.16 -> 8, 0.18-0.26 -> 9,
# 0.28 -> 8, 0.30 -> 6, 0.32 -> 2. The solver queries the twin per (xy, theta) instead of the map.
_TWIN: dict = {}


def _reachable(xyz, theta: float) -> bool:
    import mujoco
    drv = _TWIN["driver"]
    g = np.eye(4)
    c, s = np.cos(theta), np.sin(theta)
    g[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ np.diag([1.0, -1.0, -1.0])
    g[:3, 3] = xyz
    for seed in (drv.sim.arm_qpos(), np.array([0.0, -0.54, 0.63, 1.49, 0.0])):
        q = drv._ik(g, seed, iterations=150)
        d = drv._data
        d.qpos[drv._q] = np.r_[q, 0.0, 0.0]
        mujoco.mj_kinematics(drv.model, d)
        R = d.site_xmat[drv._site].reshape(3, 3)
        if (np.linalg.norm(d.site_xpos[drv._site] - xyz) < 0.0015
                and np.arccos(np.clip((np.trace(g[:3, :3] @ R.T) - 1) / 2, -1, 1)) < 0.02):
            return True
    return False


def zmax(xy, theta: float | None = None) -> float:
    """Highest reachable top-down grip height at xy (minus 6 mm), for closing direction theta (default: the hand's).
    Evaluated on a 5 mm x 0.1 rad grid (cached; the twin IK is deterministic)."""
    if theta is None:
        e = _TWIN["driver"].sim.grip_pose()
        theta = float(np.arctan2(e[1, 0], e[0, 0]))
    gx, gy = round(float(xy[0]) / 0.005) * 0.005, round(float(xy[1]) / 0.005) * 0.005
    gth = round(float(np.angle(np.exp(1j * theta))), 1)      # full turn: theta and theta+pi use other wrist rolls
    key = (round(gx, 3), round(gy, 3), gth)
    xy, theta = (gx, gy), gth
    cache = _TWIN.setdefault("cache", {})
    if key not in cache:
        lo, hi = 0.0, 0.14
        if not _reachable(np.array([xy[0], xy[1], 0.02]), theta):
            cache[key] = 0.0
        else:
            lo = 0.02
            for _ in range(7):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if _reachable(np.array([xy[0], xy[1], mid]), theta) else (lo, mid)
            cache[key] = lo - 0.006
    return cache[key]


def obstacle_top(rb: Robot, a, b, skip: tuple[str, ...], band: float = 0.03) -> float:
    """Highest top of objects whose footprint comes within `band` (hand half-width) of the segment a-b (XY)."""
    a, b = np.asarray(a[:2], float), np.asarray(b[:2], float)
    top = 0.0
    w = rb.sim.world()
    for n, p in w.poses.items():
        if n in skip:
            continue
        q = p.pos[:2]
        t = np.clip(np.dot(q - a, b - a) / max(np.dot(b - a, b - a), 1e-9), 0, 1)
        if np.linalg.norm(q - (a + t * (b - a))) < band + rb.sim.specs[n].footprint_radius:
            top = max(top, w.top(n) if rb.sim.specs[n].category != "container"
                      else float(p.pos[2]) + 0.5 * rb.sim.specs[n].height)
    return top


def _leg(rb: Robot, a, b, theta: float, below: float, skip, blocked=()) -> tuple[float, float]:
    """(highest reachable z along a->b, lowest z that clears the non-blocked objects near it)."""
    pts = [np.asarray(a[:2]) + (np.asarray(b[:2]) - np.asarray(a[:2])) * t for t in np.linspace(0, 1, 9)]
    return min(zmax(p, theta) for p in pts), obstacle_top(rb, a, b, tuple(skip) + tuple(blocked)) + below + 0.005


def travel(rb: Robot, xy, theta: float, below_grip: float, skip: tuple[str, ...] = (), z: float | None = None,
           **kw) -> float:
    """Up at the current spot, across, to above xy at a reachable height that clears the objects on the way by 5 mm
    (`below_grip` = how far the hand / carried object reaches below the grip point). Objects too tall to pass over
    within reach (towers, the tray) are bypassed through one waypoint. Returns the travel z."""
    e = rb.eef()[:3, 3]
    xy = np.asarray(xy, float)
    kw = {"contact_mode": "manipulation", "load_budget_n": 20, **kw}
    legs: list[tuple[np.ndarray, float]] = []        # (waypoint, height of the leg that ends there)
    if z is not None:
        legs = [(xy, z)]
    else:
        z_top, need = _leg(rb, e, xy, theta, below_grip, skip)
        if need <= z_top:
            legs = [(xy, float(min(z_top, max(need + 0.01, 0.05))))]
        else:
            # Too tall to pass straight over within reach (e.g. carrying a cube from the arm's low-reach rim to a
            # tower): one waypoint, each leg at its own height; objects too tall to pass over anywhere are bypassed.
            w = rb.sim.world()
            tops = {n: (w.top(n) if rb.sim.specs[n].category != "container"
                        else float(w.poses[n].pos[2]) + 0.5 * rb.sim.specs[n].height) for n in w.poses if n not in skip}
            near = lambda n, p: np.linalg.norm(w.poses[n].pos[:2] - p[:2]) < rb.sim.specs[n].footprint_radius + 0.01
            under_hand = {n for n in tops if near(n, e) and tops[n] + below_grip <= e[2] + 0.002}
            skip = tuple(skip) + tuple(under_hand)       # already above them: leave sideways at the current height
            cap = 0.09
            blocked = [n for n, h in tops.items() if h + below_grip + 0.005 > cap - 0.01
                       and n not in under_hand and not near(n, xy)]
            best = None
            for r in np.arange(0.13, 0.285, 0.015):
                for a in np.radians(np.arange(-75, 76, 7.5)):
                    p = r * np.array([np.cos(a), np.sin(a)])
                    if any(min(seg_dist(w.poses[n].pos, e, p), seg_dist(w.poses[n].pos, p, xy))
                           < rb.sim.specs[n].footprint_radius + 0.035 for n in blocked):
                        continue
                    z1, n1 = _leg(rb, e, p, theta, below_grip, skip, blocked)
                    z2, n2 = _leg(rb, p, xy, theta, below_grip, skip, blocked)
                    if under_hand:
                        n1 = max(n1, min(e[2], z1) - 0.01)
                    if n1 > z1 or n2 > z2:
                        continue
                    cost = np.linalg.norm(p - e[:2]) + np.linalg.norm(xy - p)
                    if best is None or cost < best[0]:
                        best = (cost, [(p, float(min(z1, max(n1 + 0.01, 0.05)))),
                                       (xy, float(min(z2, max(n2 + 0.01, 0.05))))])
            if best is None:
                raise Failed(f"no reachable travel height: need {need:.3f} > reach {z_top:.3f}, no detour")
            legs = best[1]
    for i, (p, zl) in enumerate(legs):
        cur = rb.eef()[:3, 3]
        if cur[2] < zl - 0.002:                          # rise before a leg, descend along it
            rb.move([cur[0], cur[1], zl], theta, **kw)
        rb.move([p[0], p[1], zl], theta, **kw)
    return legs[-1][1]


def seg_dist(p, a, b) -> float:
    a, b, p = (np.asarray(v[:2], float) for v in (a, b, p))
    t = np.clip(np.dot(p - a, b - a) / max(np.dot(b - a, b - a), 1e-12), 0, 1)
    return float(np.linalg.norm(p - (a + t * (b - a))))


def route_around(rb: Robot, xy, z: float, theta: float, avoid, clearance: float) -> None:
    """Go to (xy, z) at height z, keeping the hand `clearance` (XY) away from `avoid` (a tower too tall to pass over):
    straight if the line clears it, else via one waypoint on a circle around it."""
    e = rb.eef()[:3, 3]
    kw = dict(contact_mode="manipulation", load_budget_n=15)
    legs = [np.asarray(xy, float)]
    if seg_dist(avoid, e, xy) < clearance:
        best = None
        for a in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            w = np.asarray(avoid) + (clearance + 0.015) * np.array([np.cos(a), np.sin(a)])
            if zmax(w, theta) < z or seg_dist(avoid, e, w) < clearance or seg_dist(avoid, w, xy) < clearance:
                continue
            cost = np.linalg.norm(w - e[:2]) + np.linalg.norm(np.asarray(xy) - w)
            if best is None or cost < best[0]:
                best = (cost, w)
        if best is None:
            raise Failed("no route around the tower")
        legs = [best[1], np.asarray(xy, float)]
    if abs(e[2] - z) > 0.002:
        rb.move([e[0], e[1], max(z, min(e[2], zmax(e[:2], theta)))], theta, **kw)
    for p in legs:
        rb.move([p[0], p[1], z], theta, **kw)


def pick(rb: Robot, name: str, theta: float | None = None, grip_z: float | None = None, depth: float = 0.010) -> float:
    """Top-down grasp of a cube `depth` below its top (shallow grasps hold heavy cubes best: the closure guard leaves
    the jaw at its contact position), with the fixed jaw kept JAW_CLEAR off its face. If the cube top is too high to
    come down onto (towers), the open jaws slide in sideways at grasp height. Returns the closing direction."""
    theta = grasp_theta(rb, name) if theta is None else theta
    s = rb.size(name)
    c = rb.pos(name)
    g = c - (FIXED_JAW + s / 2 + JAW_CLEAR) * axis(theta)
    top = c[2] + s / 2
    z = grip_z if grip_z is not None else max(top - depth, 0.019)
    rb.grip("open")
    if top + 0.016 + 0.005 <= zmax(g, theta):
        travel(rb, g[:2], theta, 0.016)
        rb.move([g[0], g[1], z], theta, contact_mode="manipulation", load_budget_n=8, max_speed_m_s=0.03)
    else:
        # The hand slides in sideways at grasp height. The physical hand's camera mount sticks out ~7.7 cm on the
        # -y_eef side, so the hand must move along +y_eef (mount trailing), starting on the base side of the cube:
        # choose the face direction whose +y_eef points away from the base.
        faces = [wrap_quarter(rb.yaw(name)) + k * np.pi / 2 for k in range(4)]
        faces = [float(np.angle(np.exp(1j * f))) for f in faces]
        faces = [f for f in faces if abs(f) < 2.6]
        theta = max(faces, key=lambda f: float(np.dot([-np.sin(f), np.cos(f)], c[:2])))
        g = c - (FIXED_JAW + s / 2 + JAW_CLEAR) * axis(theta)
        if z > zmax(g, theta):
            raise Failed(f"grasp height {z:.3f} out of reach {zmax(g, theta):.3f}")
        side = -np.array([-np.sin(theta), np.cos(theta)])
        start = g[:2] + side * 0.07
        route_around(rb, start, z + 0.004, theta, c[:2], 0.5 * s * 1.42 + 0.035)
        rb.move([start[0], start[1], z], theta, contact_mode="manipulation", load_budget_n=8, max_speed_m_s=0.03)
        rb.move([g[0], g[1], z], theta, contact_mode="manipulation", load_budget_n=8, max_speed_m_s=0.02)
    rb.grip("close")
    if not rb.held(name):
        raise Failed(f"grasp of {name} failed")
    return theta


def carry_depth(s: float, support_top: float, *xys, minimum: float = 0.010) -> float:
    """Grasp depth that lets a cube of side s be carried over a support whose top is support_top at dest_xy."""
    reach = min(zmax(p) for xy in xys[:1] for p in [xy + (np.asarray(xys[-1]) - xy) * k for k in np.linspace(0, 1, 6)])
    need = support_top + s + 0.005 + 0.012 - reach
    return float(min(max(minimum, need), s - 0.008))


def place_at(rb: Robot, name: str, xy, support_z: float, theta: float, drop: float = 0.002,
             target_yaw: float | None = None, skip: tuple[str, ...] = ()) -> None:
    """Carry the held cube so its CENTRE lands at xy with its bottom `drop` above support_z, release, retreat.
    Uses the measured in-hand offset (cube centre - grip point) so off-centre grasps are corrected."""
    e = rb.eef()
    s = rb.size(name)
    below = e[2, 3] - (rb.sim.poses()[name].pos[2] - s / 2)          # grip point height above the cube's bottom
    lift = min(zmax(e[:2, 3]), e[2, 3] + 0.03)
    rb.move([e[0, 3], e[1, 3], lift], np.arctan2(e[1, 0], e[0, 0]), contact_mode="manipulation", load_budget_n=20)
    if target_yaw is not None:      # rotate about vertical so the cube's faces end up at target_yaw (mod 90)
        theta = theta + np.angle(np.exp(1j * 4 * (target_yaw - rb.yaw(name)))) / 4
    e = rb.eef()
    off = rb.pos(name) - e[:3, 3]
    cur_theta = np.arctan2(e[1, 0], e[0, 0])
    dth = float(np.angle(np.exp(1j * (theta - cur_theta))))
    c, sn = np.cos(dth), np.sin(dth)
    off_xy = np.array([c * off[0] - sn * off[1], sn * off[0] + c * off[1]])
    g = np.asarray(xy, float) - off_xy
    z = support_z + below + drop
    travel(rb, g, theta, below, skip=(name, *skip))
    rb.move([g[0], g[1], z], theta, contact_mode="manipulation", load_budget_n=10, max_speed_m_s=0.02,
            position_tolerance_m=0.0015)
    rb.grip("open")
    rb.move([g[0], g[1], min(zmax(g, theta), z + 0.03)], theta, contact_mode="manipulation", load_budget_n=10,
            max_speed_m_s=0.03)


def free_spot(rb: Robot, avoid: list[str], s: float, rng) -> np.ndarray:
    """A table spot in the annulus at least 4.5 cm + footprints away from every object in `avoid` (and the eef path)."""
    poses = rb.sim.poses()
    for _ in range(500):
        r = rng.uniform(0.15, 0.27)
        a = rng.uniform(-1.0, 1.0)
        xy = np.array([r * np.cos(a), r * np.sin(a)])
        if all(np.linalg.norm(xy - poses[n].pos[:2]) > rb.sim.specs[n].footprint_radius + s * 0.71 + 0.035 for n in avoid):
            return xy
    raise Failed("no free spot")


# ------------------------------------------------------------------ skills
def solve_stack(rb: Robot, task, rng) -> None:
    roles = task.roles
    pairs = [("top", "base")]
    aligned = SPECS[task.family]["goal"][0][2].get("yaw_tol_deg") is not None
    # Stacking needs the carried cube's bottom above the base's top: only r ~0.17-0.25 m reaches high enough
    # (top-down grip <= ~9 cm). A base elsewhere is first moved into that band (goal does not fix the location).
    b0 = roles["base"]
    p = rb.pos(b0)
    if not 0.17 <= float(np.hypot(p[0], p[1])) <= 0.25:
        a = np.arctan2(p[1], p[0])
        spot = None
        for rr in (0.21, 0.19, 0.23):
            for da in (0.0, 0.25, -0.25, 0.5, -0.5):
                xy = rr * np.array([np.cos(a + da), np.sin(a + da)])
                if all(np.linalg.norm(xy - q.pos[:2]) > 0.075 for n, q in rb.sim.poses().items() if n != b0):
                    spot = xy
                    break
            if spot is not None:
                break
        if spot is None:
            raise Failed("no reachable spot for the stack")
        th = pick(rb, b0)
        place_at(rb, b0, spot, 0.0, th)
    goal = dict(((a[1][0], a[1][1]), a[2]) for a in task.atoms if a[0] == "aligned_on")
    for top, base in pairs:
        t, b = roles[top], roles[base]
        bias = np.zeros(2)
        for attempt in range(3):
            # measure -> place -> measure: a miss of a few mm (release shifts the cube by ~2-5 mm) is corrected by
            # re-grasping the cube where it sits and aiming the opposite way
            bp = rb.pos(b)
            base_top = bp[2] + rb.size(b) / 2
            th = pick(rb, t, depth=carry_depth(rb.size(t), base_top, rb.pos(t)[:2], bp[:2]))
            place_at(rb, t, bp[:2] - bias, base_top, th, target_yaw=rb.yaw(b) if aligned else None)
            rb.call("hold", duration_s=0.5)
            w = rb.sim.world()
            if aligned_on(w, t, b, **goal.get((t, b), {})):
                break
            if not on_top(w, t, b):
                bias = np.zeros(2)            # fell off: start over from where it lies
                continue
            bias = bias + 0.9 * (rb.pos(t)[:2] - rb.pos(b)[:2])


def solve_unstack(rb: Robot, task, rng) -> None:
    roles = task.roles
    order = [roles["top"]]
    for name in order:
        th = pick(rb, name)
        spot = free_spot(rb, [n for n in rb.sim.specs if n != name], rb.size(name), rng)
        place_at(rb, name, spot, 0.0, th)


def solve_slot(rb: Robot, task, rng) -> None:
    """Shallow grasp (fingertips must stay above the holder rim), square the cube to the opening, lower until its
    bottom is 3 mm inside the opening, release (it drops in), then press it down with the closed jaws if it wedged."""
    t, h = task.roles["target"], task.roles["holder"]
    s = rb.size(t)
    c = rb.pos(t)
    th = pick(rb, t, grip_z=c[2] + s / 2 - min(0.012, s - 0.019))
    hp = rb.pos(h)
    rim = float(rb.sim.poses()[h].pos[2]) + 0.5 * rb.sim.specs[h].height
    place_at(rb, t, hp[:2], rim - 0.003, th, drop=0.0, target_yaw=rb.yaw(h))
    rb.call("hold", duration_s=0.5)
    w = rb.sim.world()
    if w.bottom(t) - w.top(h) > 0.004:                 # wedged: press on the top face with the closed jaws
        rb.grip("close")
        p = rb.pos(t)
        top = w.poses[t].pos[2] + w.specs[t].z_half_extent(w.poses[t].rot)
        rb.move([p[0], p[1], top + 0.016], th, contact_mode="manipulation", load_budget_n=15)
        rb.move([p[0], p[1], top + 0.006], th, contact_mode="manipulation", load_budget_n=15, max_speed_m_s=0.01)
        rb.move([p[0], p[1], min(zmax(p), top + 0.04)], th, contact_mode="manipulation", load_budget_n=15)


def solve_push(rb: Robot, task, rng, max_pushes: int = 8) -> None:
    """Closed jaws, fingertips ~0.8 cm above the table, 2.2 cm behind the cube on the cube->mat-centre line; push in
    <= 7 cm strokes (contact_mode='manipulation'), back off, re-measure, re-aim. Never opens around the cube."""
    t, mat = task.roles["target"], task.roles["region"]
    rb.grip("close")
    for _ in range(max_pushes):
        if all(rb.sim.atom_values()):
            return
        c, m = rb.pos(t), rb.pos(mat)
        d = m[:2] - c[:2]
        dist = float(np.linalg.norm(d))
        u = d / max(dist, 1e-9)
        s = rb.size(t)
        # closing axis anti-parallel to the push: the FIXED jaw (flat, vertical outer face, at -x_eef) leads. With the
        # hinged jaw leading, its slanted back lifted the cube's edge (tilts of 15-40 deg, contact stalls).
        theta = float(np.angle(np.exp(1j * (np.arctan2(u[1], u[0]) + np.pi))))
        start = c[:2] - u * (s / 2 + 0.05)           # physical fixed jaw's outer face: 4.2 cm from the grip point
        z = 0.022
        travel(rb, start, theta, 0.016, skip=(mat,), z=min(max(0.05, s + 0.022), zmax(start, theta), zmax(rb.eef()[:2, 3])))
        rb.move([start[0], start[1], z], theta, contact_mode="manipulation", load_budget_n=10, max_speed_m_s=0.03)
        end = start + u * min(0.07, dist + 0.012)
        rb.move([end[0], end[1], z], theta, contact_mode="manipulation", load_budget_n=15, max_speed_m_s=0.03,
                must=False)
        back = end - u * 0.015
        rb.move([back[0], back[1], z + 0.03], theta, contact_mode="manipulation", load_budget_n=15, must=False)
        rb.call("hold", duration_s=0.3)
    if not all(rb.sim.atom_values()):
        raise Failed("push budget exhausted")


SOLVERS = {"stack": solve_stack, "unstack": solve_unstack, "slot": solve_slot, "push": solve_push}


# ------------------------------------------------------------------ runner
def run(spec_id: str, seed: int, out: Path, noise_mm: float, png: bool) -> dict:
    from captain_harness.server import env_server
    from soarm.skills.episode import make_builder
    d = out / f"{spec_id}_{seed}"
    shutil.rmtree(d, ignore_errors=True)
    args = env_server.build_parser().parse_args([
        "--suite", "soarm_skills", "--task-id", "0", "--seed", str(seed), "--output-dir", str(d),
        "--mujoco-gl", os.environ.get("SOARM_GL", "osmesa"), "--receipts", "--max-actions", "150"])
    holder = {}
    builder = make_builder(spec_id, on_sim=lambda s: holder.setdefault("sim", s))
    st = env_server.EpisodeServerState(args, builder=builder)
    t0 = time.time()
    st.build()
    sim = holder["sim"]
    _TWIN["driver"] = st.env._driver
    rng = np.random.default_rng(seed + 77)
    rb = Robot(st, sim, noise_mm / 1000.0, rng)
    err = None
    try:
        SOLVERS[sim.spec["skill"]](rb, sim.task, rng)
        rb.call("hold", duration_s=1.0)
    except Failed as e:
        err = str(e)[:300]
    except Exception:
        err = traceback.format_exc()[-600:]
    values = sim.atom_values()
    w = sim.world()
    # proof = the goal holds AND the scripted plan ran to its end (a tower knocked over by a bad path can satisfy an
    # unstack goal by accident: not counted)
    row = dict(spec=spec_id, seed=seed, success=bool(all(values)) and err is None, goal_holds=bool(all(values)),
               atoms=[bool(v) for v in values], error=err,
               calls=rb.calls, wall_s=round(time.time() - t0, 1), sim_s=round(sim.time_s, 1),
               sizes={n: round(s.size[0], 4) for n, s in sim.specs.items()},
               tilt={n: round(float(np.degrees(np.arccos(min(1, np.abs(w.poses[n].rot[2, :]).max())))), 1)
                     for n, s in sim.specs.items() if s.kind == "cube"},
               language=sim.task.language, roles=dict(sim.task.roles),
               start={n: np.round(p.pos, 3).tolist() for n, p in sim.initial.items()},
               final={n: np.round(p.pos, 3).tolist() for n, p in w.poses.items()})
    roles = sim.task.roles
    if "top" in roles and ("base" in roles or "middle" in roles):
        b = roles.get("base") or roles["middle"]
        row["xy_err_mm"] = round(1000 * float(np.linalg.norm(w.poses[roles["top"]].pos[:2] - w.poses[b].pos[:2])), 1)
    (d / "solver_log.txt").write_text("\n".join(rb.log))
    if png:
        from PIL import Image
        drv = st.env.driver if hasattr(st.env, "driver") else None
        if drv is not None:
            Image.fromarray(drv.frames(["agentview"])["agentview"].rgb).save(d / "final.png")
    try:
        st.close_env()
    except Exception:
        pass
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--pose-noise-mm", type=float, default=0.0)
    ap.add_argument("--out", default=os.path.expandvars("$SOARM_ROOT/skills_runs/solve"))
    ap.add_argument("--png", action="store_true")
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in range(int(lo), int(hi or lo) + 1):
        r = run(a.spec, s, out, a.pose_noise_mm, a.png)
        rows.append(r)
        print(json.dumps(r), flush=True)
        with (out / "results.jsonl").open("a") as f:
            f.write(json.dumps({**r, "noise_mm": a.pose_noise_mm}) + "\n")
    print(f"SUMMARY {a.spec} noise={a.pose_noise_mm}mm success {sum(r['success'] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()

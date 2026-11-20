"""Geometric success predicates over object states (pure numpy, no MuJoCo; unit-tested in tests/test_predicates.py).

A goal is a list of atoms `(name, args, params)`; the task succeeds when every atom holds. `World` carries the object
specs, their current poses, their poses at reset and the set of objects the gripper holds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np

from .objects import ObjSpec

TABLE_Z = 0.0
TABLE_HALF = (0.40, 0.45)       # table top half extents around (TABLE_CENTER_X, 0)
TABLE_CENTER_X = 0.15
REST_TOL = 0.012                # bottom within this of the support surface counts as resting on it


@dataclass
class Pose:
    pos: np.ndarray        # (3,)
    rot: np.ndarray        # (3, 3)


@dataclass
class World:
    specs: Mapping[str, ObjSpec]
    poses: Mapping[str, Pose]
    initial: Mapping[str, Pose] = field(default_factory=dict)
    held: frozenset[str] = frozenset()

    def bottom(self, name: str) -> float:
        p = self.poses[name]
        return float(p.pos[2]) - self.specs[name].z_half_extent(p.rot)

    def top(self, name: str) -> float:
        """Top support surface: containers -> their floor top; others -> their top face."""
        p, s = self.poses[name], self.specs[name]
        if s.category == "container":
            return float(p.pos[2]) - 0.5 * s.height + 0.004
        return float(p.pos[2]) + s.z_half_extent(p.rot)

    def local_xy(self, name: str, ref: str) -> np.ndarray:
        """XY of `name`'s centre in `ref`'s frame (yaw only)."""
        r = self.poses[ref]
        d = self.poses[name].pos[:2] - r.pos[:2]
        yaw = float(np.arctan2(r.rot[1, 0], r.rot[0, 0]))
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])


def _inside_footprint(world: World, name: str, ref: str, margin: float = 0.0) -> bool:
    spec = world.specs[ref]
    hx, hy = spec.inner_half_xy()
    q = world.local_xy(name, ref)
    if spec.round:
        return float(np.linalg.norm(q)) <= hx - margin
    return abs(q[0]) <= hx - margin and abs(q[1]) <= hy - margin


def on_table(world: World, a: str) -> bool:
    x, y = world.poses[a].pos[:2]
    return (abs(x - TABLE_CENTER_X) <= TABLE_HALF[0] and abs(y) <= TABLE_HALF[1]
            and world.bottom(a) > TABLE_Z - 0.02)


def resting_on_table(world: World, a: str) -> bool:
    return on_table(world, a) and world.bottom(a) <= TABLE_Z + REST_TOL and a not in world.held


def on_top(world: World, a: str, b: str) -> bool:
    """`a` rests on `b`: a's centre over b's top face, a's bottom at b's top, not held."""
    if a in world.held:
        return False
    sb = world.specs[b]
    hx, hy = 0.5 * sb.size[0], 0.5 * sb.size[1]
    q = world.local_xy(a, b)
    over = (float(np.linalg.norm(q)) <= hx) if sb.round else (abs(q[0]) <= hx and abs(q[1]) <= hy)
    gap = world.bottom(a) - world.top(b)
    return bool(over and -REST_TOL <= gap <= REST_TOL)


def inside(world: World, a: str, c: str) -> bool:
    """`a` is in container `c`: centre within the inner footprint, below the rim + half its height, above the floor - 1 cm,
    not held."""
    if a in world.held or world.specs[c].category != "container":
        return False
    pc = world.poses[c]
    rim = float(pc.pos[2]) + 0.5 * world.specs[c].height
    floor = float(pc.pos[2]) - 0.5 * world.specs[c].height
    za = float(world.poses[a].pos[2])
    return bool(_inside_footprint(world, a, c) and floor - 0.01 <= world.bottom(a) and za <= rim + 0.01)


def in_region(world: World, a: str, region: str) -> bool:
    """`a` rests on the flat region (mat): centre inside the mat, bottom at the mat surface, not held."""
    if a in world.held:
        return False
    return bool(_inside_footprint(world, a, region) and world.bottom(a) <= world.top(region) + REST_TOL)


def placed_at(world: World, a: str, dest: str) -> bool:
    """Generic destination: container -> inside, region -> in_region, object -> on_top."""
    cat = world.specs[dest].category
    if cat == "container":
        return inside(world, a, dest)
    if cat == "region":
        return in_region(world, a, dest)
    return on_top(world, a, dest)


# LIBERO convention (front camera faces the robot): front = +x (toward the camera), right = +y as seen in the camera image.
DIRECTIONS = {"left": np.array([0.0, -1.0]), "right": np.array([0.0, 1.0]),
              "front": np.array([1.0, 0.0]), "behind": np.array([-1.0, 0.0])}


def relation(world: World, a: str, ref: str, direction: str, near: float = 0.03, far: float = 0.16,
             lateral: float = 0.06) -> bool:
    """`a` rests on the table `direction` of `ref` (DIRECTIONS: LIBERO camera convention): the gap between
    footprints along the direction is in [~0, far], the lateral offset is below `lateral`."""
    if not resting_on_table(world, a):
        return False
    d = world.poses[a].pos[:2] - world.poses[ref].pos[:2]
    u = DIRECTIONS[direction]
    along = float(d @ u)
    perp = abs(float(d @ np.array([-u[1], u[0]])))
    clear = world.specs[a].footprint_radius + world.specs[ref].footprint_radius
    return bool(clear + near - 0.03 <= along <= clear + far and perp <= lateral)


def at_initial_of(world: World, a: str, b: str, tol: float = 0.035) -> bool:
    """`a` rests on the table within `tol` (XY) of where `b` started (swap)."""
    if not resting_on_table(world, a):
        return False
    return bool(np.linalg.norm(world.poses[a].pos[:2] - world.initial[b].pos[:2]) <= tol)


def lifted(world: World, a: str, height: float = 0.04) -> bool:
    return bool(a in world.held and world.bottom(a) >= TABLE_Z + height)


def lying(world: World, a: str) -> bool:
    """Cylinder on its side: axis within ~25 deg of horizontal, resting on the table."""
    return bool(abs(float(world.poses[a].rot[2, 2])) < 0.42 and resting_on_table(world, a))


def off_region(world: World, a: str, region: str) -> bool:
    """`a` rests on the table with no part of its footprint over the region."""
    return bool(resting_on_table(world, a) and not _touching_region(world, a, region))


def _touching_region(world: World, a: str, region: str) -> bool:
    """Part of a's footprint over the region (centre within region half extents + 0.7 x a's footprint radius)."""
    spec = world.specs[region]
    hx, hy = 0.5 * spec.size[0], 0.5 * spec.size[1]
    q = world.local_xy(a, region)
    r = world.specs[a].footprint_radius * 0.7
    return abs(q[0]) <= hx + r and abs(q[1]) <= hy + r


def in_line(world: World, *names: str, order: str = "", max_residual: float = 0.015, min_gap: float = 0.0,
            max_gap: float = 0.12) -> bool:
    """All objects rest on the table on one straight line (perpendicular residual <= max_residual), neighbours spaced
    <= max_gap between footprints; order "left_to_right" (camera view) requires y to increase in the given name order."""
    if not all(resting_on_table(world, n) for n in names):
        return False
    pts = np.array([world.poses[n].pos[:2] for n in names])
    centred = pts - pts.mean(0)
    _, _, vt = np.linalg.svd(centred)
    direction = vt[0]
    residual = np.abs(centred @ np.array([-direction[1], direction[0]]))
    if residual.max() > max_residual:
        return False
    t = centred @ direction
    idx = np.argsort(t)
    for i, j in zip(idx[:-1], idx[1:]):
        gap = float(t[j] - t[i]) - world.specs[names[i]].footprint_radius - world.specs[names[j]].footprint_radius
        if gap < min_gap - 0.01 or gap > max_gap:
            return False
    if order == "left_to_right":
        ys = pts[:, 1]
        return bool(np.all(np.diff(ys) > 0.01))
    if order == "front_to_back":
        xs = pts[:, 0]
        return bool(np.all(np.diff(xs) < -0.01))
    return True


def grouped(world: World, *names: str, radius: float = 0.09) -> bool:
    """All objects rest on the table within `radius` (XY) of their common centroid."""
    if not all(resting_on_table(world, n) for n in names):
        return False
    pts = np.array([world.poses[n].pos[:2] for n in names])
    return bool(np.linalg.norm(pts - pts.mean(0), axis=1).max() <= radius)


def stacked(world: World, *names: str) -> bool:
    """names[0] on the table (or anywhere), names[1] on names[0], names[2] on names[1], ..."""
    return all(on_top(world, b, a) for a, b in zip(names[:-1], names[1:]))


PREDICATES: dict[str, Callable[..., bool]] = {
    "on_top": on_top, "inside": inside, "in_region": in_region, "placed_at": placed_at, "relation": relation,
    "at_initial_of": at_initial_of, "lifted": lifted, "lying": lying, "off_region": off_region, "in_line": in_line,
    "grouped": grouped, "stacked": stacked, "resting_on_table": resting_on_table,
}


def evaluate(world: World, atoms) -> list[bool]:
    out = []
    for name, args, params in atoms:
        out.append(bool(PREDICATES[name](world, *args, **dict(params or {}))))
    return out

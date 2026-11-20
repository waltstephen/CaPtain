"""Scripted symbolic planner: goal atoms -> top-down pick/place steps with target poses, on an ideal geometric state.

The plan is checked twice: geometrically here (every pick/place/carry point inside the measured top-down reach band at its
grip height, every target free, tasks.evaluate_goal true on the planned final state) and in physics by taskgen.solve,
which executes the same steps through SoarmDriver.command with privileged object poses (a test, not a policy).
Unsupported predicates raise NoPlan: the family then needs a teacher attempt to show solvability.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from soarm.objects import ObjSpec
from soarm.predicates import DIRECTIONS, Pose, World
from soarm.tasks import ANG_MAX, Reject, evaluate_goal

# top-down reach of the grip point vs its height above the table, measured with the driver's own IK
# (`python -m taskgen.measure_reach`, armhost 11-10: position error < 4 mm, approach within 15 deg of vertical; the same
# band at 35 deg, so position is the limit). The report's phase-1 table (8 cm: 0.14-0.28) is close.
REACH = [(0.00, 0.100, 0.320), (0.01, 0.100, 0.320), (0.02, 0.105, 0.320), (0.03, 0.110, 0.300),
         (0.04, 0.115, 0.295), (0.05, 0.120, 0.290), (0.06, 0.125, 0.285), (0.07, 0.135, 0.275),
         (0.08, 0.150, 0.260), (0.09, 0.165, 0.245), (0.10, 0.195, 0.215), (0.105, 0.205, 0.205)]
REACH_MARGIN = 0.005
CARRY_Z = 0.03              # carry floor; raised to pick/place height + 6 mm and over obstacles on the path
LIFT_BOTTOM = 0.05          # lifted: object bottom >= 4 cm (predicates.lifted) + 1 cm
PLACE_CLEAR = 0.008         # release with the object's bottom 8 mm above its support
TARGET_GAP = 0.02           # free space around a table target (the opening jaw sweeps sideways)
PLANNED = ("count", "lifted", "on_top", "stacked", "inside", "in_region", "placed_at", "relation", "at_initial_of", "in_line",
           "grouped", "off_region", "resting_on_table")


class NoPlan(Reject):
    """No scripted plan (a Reject, so instance sampling retries with the next draw)."""


def reach_band(z: float) -> tuple[float, float] | None:
    if z > REACH[-1][0]:
        return None
    for (z0, a0, b0), (z1, a1, b1) in zip(REACH[:-1], REACH[1:]):
        if z <= z1:
            f = (max(z, z0) - z0) / (z1 - z0)
            return a0 + f * (a1 - a0) + REACH_MARGIN, b0 + f * (b1 - b0) - REACH_MARGIN
    return None


def reachable(xy, z: float) -> bool:
    band = reach_band(z)
    if band is None:
        return False
    r = float(np.hypot(*xy))
    return band[0] <= r <= band[1] and abs(float(np.arctan2(xy[1], xy[0]))) <= np.deg2rad(90)


@dataclass
class Step:
    obj: str
    xy: tuple[float, float]
    support: str | None             # None = table
    offset: tuple[float, float] = (0.0, 0.0)   # target relative to the support's centre (containers/mats)
    hold: bool = False              # lifted: keep it in the jaw at carry height, no place
    pick_z: float = 0.0
    place_z: float = 0.0
    carry_z: float = CARRY_Z
    why: str = ""


@dataclass
class State:
    specs: dict[str, ObjSpec]
    xy: dict[str, np.ndarray]
    yaw: dict[str, float]
    support: dict[str, str | None]
    initial_xy: dict[str, np.ndarray] = field(default_factory=dict)
    held: str | None = None

    def bottom(self, n: str) -> float:
        s = self.support[n]
        if s is None:
            return 0.0 if n != self.held else 0.05
        return self.surface(s)

    def surface(self, n: str) -> float:
        """Height objects rest at when placed on `n`."""
        o = self.specs[n]
        if o.category == "container":
            return self.bottom(n) + 0.004
        return self.bottom(n) + o.size[2]

    def top(self, n: str) -> float:
        return self.bottom(n) + self.specs[n].size[2]

    def on(self, n: str) -> list[str]:
        return [m for m, s in self.support.items() if s == n]

    def world(self) -> World:
        poses = {}
        for n, o in self.specs.items():
            z = self.bottom(n) + 0.5 * o.size[2]
            c, s = np.cos(self.yaw[n]), np.sin(self.yaw[n])
            poses[n] = Pose(pos=np.array([*self.xy[n], z]), rot=np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]))
        init = {n: Pose(pos=np.array([*self.initial_xy[n], 0.0]), rot=np.eye(3)) for n in self.specs}
        return World(specs=self.specs, poses=poses, initial=init, held=frozenset([self.held] if self.held else []))


def state_from(task) -> State:
    specs = {o.name: o for o in task.objects}
    xy = {n: np.asarray(p.xy, dtype=float) for n, p in task.layout.items()}
    return State(specs=specs, xy=xy, yaw={n: p.yaw for n, p in task.layout.items()},
                 support={n: p.on for n, p in task.layout.items()}, initial_xy={k: v.copy() for k, v in xy.items()})


class Planner:
    def __init__(self, task) -> None:
        self.task = task
        self.st = state_from(task)
        self.steps: list[Step] = []
        self.load: dict[str, int] = {}          # destination -> how many objects the goal puts into it
        for name, args, params in task.atoms:
            if name in ("inside", "in_region", "placed_at"):
                self.load[args[1]] = self.load.get(args[1], 0) + 1
            elif name == "count":
                dest = next((a for a in params["args"] if a != "$"), None)
                if dest:
                    self.load[dest] = self.load.get(dest, 0) + params.get("eq", params.get("min", len(args)))

    # ------------------------------------------------------------------ geometry helpers
    def _fr(self, n):
        return self.st.specs[n].footprint_radius

    def _table_free(self, xy, n, ignore=()) -> bool:
        for m, o in self.st.specs.items():
            if m == n or m in ignore or self.st.support[m] is not None or self.st.held == m:
                continue
            gap = TARGET_GAP if o.category == "object" else 0.012
            if np.linalg.norm(xy - self.st.xy[m]) < self._fr(n) + self._fr(m) + gap:
                return False
        return True

    def _free_spot(self, n, near, avoid_region=None, ignore=(), keep_clear=()) -> np.ndarray:
        """Nearest reachable free table spot to `near`; `keep_clear` = [(xy, radius)] targets that must stay free."""
        best, best_d = None, 1e9
        h = self.st.specs[n].size[2]
        for r in np.arange(0.14, 0.281, 0.02):
            for a in np.deg2rad(np.arange(-64, 65, 6)):
                xy = np.array([r * np.cos(a), r * np.sin(a)])
                if not reachable(xy, 0.5 * h) or not self._table_free(xy, n, ignore):
                    continue
                if any(np.linalg.norm(xy - np.asarray(p)) < self._fr(n) + r + TARGET_GAP for p, r in keep_clear):
                    continue
                if avoid_region is not None:
                    reg = self.st.specs[avoid_region]
                    if np.linalg.norm(xy - self.st.xy[avoid_region]) < 0.5 * np.hypot(*reg.size[:2]) + self._fr(n) + 0.02:
                        continue
                d = float(np.linalg.norm(xy - near))
                if d < best_d:
                    best, best_d = xy, d
        if best is None:
            raise NoPlan(f"no free table spot for {n}")
        return best

    def _slot_in(self, n, dest) -> tuple[np.ndarray, tuple[float, float]]:
        o, d = self.st.specs[n], self.st.specs[dest]
        hx, hy = d.inner_half_xy() if d.category != "object" else (0.0, 0.0)
        c = self.st.xy[dest]
        others = [m for m in self.st.on(dest) if m != n]
        lim = min(hx, hy) - self._fr(n) * (0.75 if d.category == "container" else 0.5) - 0.004
        ring = [] if lim <= 0 else [rr * np.array([np.cos(t), np.sin(t)]) for rr in (lim, 0.6 * lim)
                                    for t in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
        # several items: start at the rim so the next ones fit (two 4 cm cubes fit an 11 cm bowl only side by side)
        offs = ring + [np.zeros(2)] if self.load.get(dest, 1) > 1 else [np.zeros(2)] + ring
        side = lambda m: 0.6 * float(np.mean(self.st.specs[m].size[:2]))    # half side + rotation slack
        for off in offs:
            xy = c + off
            if all(np.linalg.norm(xy - self.st.xy[m]) >= side(n) + side(m) + 0.004 for m in others):
                return xy, (float(off[0]), float(off[1]))
        raise NoPlan(f"no room for {n} in {dest}")

    # ------------------------------------------------------------------ actions
    def move(self, n, xy, support=None, offset=(0.0, 0.0), why="") -> None:
        keep = [(np.asarray(xy, dtype=float), self._fr(n))]
        for m in self.st.on(n):                      # clear what sits on n first
            self.move(m, self._free_spot(m, self.st.xy[m], ignore=(n,), keep_clear=keep), why=f"clear {n}")
        if support is None and not self._table_free(np.asarray(xy), n):
            for m, o in list(self.st.specs.items()):  # move blockers aside (swap / line slots)
                if m != n and self.st.support[m] is None and o.category == "object" and \
                        np.linalg.norm(np.asarray(xy) - self.st.xy[m]) < self._fr(n) + self._fr(m) + TARGET_GAP:
                    self.move(m, self._free_spot(m, self.st.xy[m], ignore=(n,), keep_clear=keep), why=f"unblock {n}")
            if not self._table_free(np.asarray(xy), n):
                raise NoPlan(f"target of {n} blocked")
        o = self.st.specs[n]
        pick_z = max(0.016, self.st.bottom(n) + 0.5 * o.size[2])
        place_base = 0.0 if support is None else self.st.surface(support)
        place_z = max(0.016, place_base + 0.5 * o.size[2] + PLACE_CLEAR)
        carry_z = max(CARRY_Z, pick_z + 0.006, place_z + 0.006,
                      self._obstacle_top(self.st.xy[n], np.asarray(xy), n) + 0.5 * o.size[2] + 0.012)
        for p, z in ((self.st.xy[n], pick_z), (np.asarray(xy), place_z), (self.st.xy[n], carry_z),
                     (np.asarray(xy), carry_z)):
            if not reachable(p, z):
                raise NoPlan(f"{why or n}: ({p[0]:.3f}, {p[1]:.3f}) z {z:.3f} outside top-down reach")
        self.steps.append(Step(obj=n, xy=(float(xy[0]), float(xy[1])), support=support, offset=offset,
                               pick_z=pick_z, place_z=place_z, carry_z=carry_z, why=why))
        self.st.xy[n] = np.asarray(xy, dtype=float)
        self.st.support[n] = support

    def _obstacle_top(self, a, b, n) -> float:
        """Highest top (container rims included) near the straight carry path of `n` from a to b."""
        top = 0.0
        ab = b - a
        L = float(np.linalg.norm(ab)) or 1e-9
        for m in self.st.specs:
            if m == n:
                continue
            p = self.st.xy[m]
            t = np.clip(float((p - a) @ ab) / L ** 2, 0, 1)
            if np.linalg.norm(a + t * ab - p) < self._fr(m) + self._fr(n) + 0.01:
                top = max(top, self.st.top(m))
        return top

    # ------------------------------------------------------------------ atoms
    def atom(self, name, args, params) -> None:
        st = self.st
        if name == "lifted":
            n = args[0]
            for m in st.on(n):
                self.move(m, self._free_spot(m, st.xy[m], ignore=(n,)), why=f"clear {n}")
            o = st.specs[n]
            pick_z = max(0.016, st.bottom(n) + 0.5 * o.size[2])
            lift_z = LIFT_BOTTOM + 0.5 * o.size[2]
            for z in (pick_z, lift_z):
                if not reachable(st.xy[n], z):
                    raise NoPlan(f"lift {n} outside reach")
            self.steps.append(Step(obj=n, xy=tuple(st.xy[n]), support=None, hold=True, pick_z=pick_z, place_z=lift_z,
                                   carry_z=lift_z, why=name))
            st.held, st.support[n] = n, None
        elif name == "on_top":
            self.move(args[0], st.xy[args[1]], support=args[1], why=name)
        elif name == "stacked":
            for base, top in zip(args[:-1], args[1:]):
                self.move(top, st.xy[base], support=base, why=name)
        elif name in ("inside", "in_region", "placed_at"):
            a, dest = args
            if st.specs[dest].category == "object":
                self.move(a, st.xy[dest], support=dest, why=name)
            else:
                xy, off = self._slot_in(a, dest)
                self.move(a, xy, support=dest, offset=off, why=name)
        elif name == "relation":
            a, ref, side = args
            u = DIRECTIONS[side]
            base = st.xy[ref] + u * (self._fr(a) + self._fr(ref) + 0.05)
            for lat in (0.0, 0.02, -0.02, 0.04, -0.04):
                xy = base + lat * np.array([-u[1], u[0]])
                for extra in (0.0, 0.03, 0.06):
                    cand = xy + u * extra
                    if self._table_free(cand, a) and reachable(cand, 0.5 * st.specs[a].size[2] + PLACE_CLEAR):
                        self.move(a, cand, why=name)
                        return
            raise NoPlan(f"no free spot {side} of {ref}")
        elif name == "at_initial_of":
            self.move(args[0], st.initial_xy[args[1]].copy(), why=name)
        elif name in ("off_region", "resting_on_table"):
            n = args[0]
            region = args[1] if name == "off_region" else None
            self.move(n, self._free_spot(n, st.xy[n], avoid_region=region), why=name)
        elif name == "grouped":
            self._group(args)
        elif name == "count":
            # CONTRACT 2.2 combinator: make the first `eq` (else `min`) members satisfy the inner atom; members that
            # already satisfy it count (the start check guarantees the combinator itself is false at the start)
            n = params.get("eq", params.get("min", 0))
            inner = [(params["pred"], tuple(m if a == "$" else a for a in params["args"]), params.get("params") or {})
                     for m in args]
            have = sum(evaluate_goal(st.world(), inner))
            for at in inner:
                if have >= n:
                    break
                if not evaluate_goal(st.world(), [at])[0]:
                    self.atom(*at)
                    have += 1
        elif name == "in_line":
            self._line(args, params.get("order", ""))
        else:
            raise NoPlan(f"no scripted plan for predicate {name!r}")

    def _group(self, names) -> None:
        anchor = names[0]
        c = self.st.xy[anchor]
        for n in names[1:]:
            ring = self._fr(anchor) + self._fr(n) + TARGET_GAP + 0.002
            for t in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                xy = c + ring * np.array([np.cos(t), np.sin(t)])
                if self._table_free(xy, n) and reachable(xy, 0.5 * self.st.specs[n].size[2] + PLACE_CLEAR):
                    self.move(n, xy, why="grouped")
                    break
            else:
                raise NoPlan("no free spot next to the group")

    def _line(self, names, order) -> None:
        frs = [self._fr(n) for n in names]
        sp = max(frs) * 2 + TARGET_GAP + 0.005
        k = len(names)
        best = None
        for x in np.arange(0.17, 0.251, 0.02):
            for yc in np.arange(-0.10, 0.101, 0.02):
                if order == "front_to_back":
                    slots = [np.array([x + 0.04 + (k - 1) / 2 * sp - i * sp, yc]) for i in range(k)]
                else:
                    slots = [np.array([x, yc + (i - (k - 1) / 2) * sp]) for i in range(k)]
                if not all(reachable(s, 0.03) for s in slots):
                    continue
                if not all(self._table_free(s, n, ignore=names) for s, n in zip(slots, names)):
                    continue
                cost = sum(float(np.linalg.norm(s - self.st.xy[n])) for s, n in zip(slots, names))
                if best is None or cost < best[0]:
                    best = (cost, slots)
        if best is None:
            raise NoPlan("no free line")
        slots = best[1]
        if not order:      # unordered: assign slots greedily by distance
            rest, assign = list(names), {}
            for s in slots:
                n = min(rest, key=lambda m: float(np.linalg.norm(self.st.xy[m] - s)))
                assign[n] = s
                rest.remove(n)
        else:
            assign = dict(zip(names, slots))
        for n in sorted(names, key=lambda m: float(np.linalg.norm(self.st.xy[m] - assign[m]))):
            if np.linalg.norm(self.st.xy[n] - assign[n]) > 0.004 or self.st.support[n] is not None:
                self.move(n, assign[n], why="in_line")


def plan(task) -> list[Step]:
    """Steps that reach every atom of `task`, or NoPlan. Atoms are handled in order; `lifted` must come last."""
    p = Planner(task)
    for name, args, params in task.atoms:
        if name not in PLANNED:
            raise NoPlan(f"no scripted plan for predicate {name!r}")
        p.atom(name, args, params)
    vals = evaluate_goal(p.st.world(), task.atoms)
    if not all(vals):
        raise NoPlan(f"planned final state misses atoms {[a for a, v in zip(task.atoms, vals) if not v]}")
    return p.steps


def check_start_reach(task) -> None:
    """Every graspable object can be grasped top-down where it starts (otherwise distractors could never be moved)."""
    st = state_from(task)
    for n, o in st.specs.items():
        if o.category == "object" and not reachable(st.xy[n], max(0.016, st.bottom(n) + 0.5 * o.size[2])):
            raise NoPlan(f"{n} starts outside top-down reach")
    for n in st.specs:
        if abs(float(np.arctan2(st.xy[n][1], st.xy[n][0]))) > ANG_MAX + 1e-6:
            raise NoPlan(f"{n} outside the workspace sector")

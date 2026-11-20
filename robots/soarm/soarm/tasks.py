"""Procedural task families for the SO-101 twin: object sets, initial layout, language and success atoms.

`sample_task(family, seed)` -> TaskInstance (pure numpy: no MuJoCo). The scene builder places the objects at
`TaskInstance.layout` and lets them settle; the success atoms are evaluated by `predicates.evaluate`.
Family generators live in `families.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .objects import COLORS, ObjSpec, describe, sample_spec

# Workspace annulus (robot base frame): top-down grasps reach r in [0.10, 0.32] m at table height.
R_MIN, R_MAX = 0.13, 0.29
ANG_MAX = np.deg2rad(70.0)


@dataclass
class Placement:
    xy: tuple[float, float]
    yaw: float
    on: str | None = None          # name of a mat/object it starts on (None = table)
    lying: bool = False            # cylinder laid on its side


@dataclass
class TaskInstance:
    family: str
    seed: int
    language: str
    objects: list[ObjSpec]
    layout: dict[str, Placement]
    atoms: list[tuple[str, tuple, dict]]
    roles: dict[str, str] = field(default_factory=dict)      # role -> object name (audit)
    platforms: list[dict] = field(default_factory=list)      # static boxes (scene_random.Platform fields), not objects

    def to_json(self) -> dict:
        out = {"family": self.family, "seed": self.seed, "language": self.language,
               "objects": [{"name": o.name, "kind": o.kind, "color": o.color, "size": list(o.size), "mass": o.mass,
                            **({"attrs": dict(o.attrs)} if o.attrs else {})} for o in self.objects],
               "layout": {k: {"xy": list(p.xy), "yaw": p.yaw, "on": p.on, "lying": p.lying} for k, p in self.layout.items()},
               "atoms": [[n, list(a), dict(p)] for n, a, p in self.atoms], "roles": dict(self.roles)}
        if self.platforms:
            out["platforms"] = [dict(p) for p in self.platforms]
        return out


class Reject(Exception):
    """Sample rejected (non-unique reference, no room): the sampler retries with the next sub-seed."""


class Builder:
    """Accumulates objects for one task and lays them out without overlap."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.objects: list[ObjSpec] = []
        self.layout: dict[str, Placement] = {}
        self._count: dict[str, int] = {}

    def add(self, kind: str, color: str | None = None, avoid_colors: tuple[str, ...] = ()) -> ObjSpec:
        if color is None:
            choices = [c for c in COLORS if c not in avoid_colors]
            color = str(self.rng.choice(choices))
        k = self._count.get(kind, 0)
        self._count[kind] = k + 1
        spec = sample_spec(self.rng, kind, color, f"{kind}_{k}")
        self.objects.append(spec)
        return spec

    def colors(self, n: int, exclude: tuple[str, ...] = ()) -> list[str]:
        pool = [c for c in COLORS if c not in exclude]
        return [str(c) for c in self.rng.choice(pool, size=n, replace=False)]

    def ref(self, spec: ObjSpec) -> str:
        text = describe(spec, self.objects)
        if text is None:
            raise Reject(f"no unique description for {spec.name}")
        return text

    def _free(self, xy: np.ndarray, radius: float, placed: list[tuple[np.ndarray, float]], gap: float = 0.035) -> bool:
        """Footprints at least `gap` apart: room for the open hinged jaw (it sweeps ~3 cm beside the object)."""
        return all(np.linalg.norm(xy - p) >= radius + r + gap for p, r in placed)

    def place(self, fixed: dict[str, Placement] | None = None, r_range=(R_MIN, R_MAX), tries: int = 400,
              skip: set[str] = frozenset(), ang_max: float = ANG_MAX, sampler=None) -> None:
        """Lay out every object not in `fixed` on the table in the annulus, largest first, no footprint overlap.
        Objects with `on=<mat>` in `fixed` are placed inside that mat after it is placed; `skip` names are left to the
        caller (stacked starts, taskspec). `sampler(placed) -> xy` replaces the uniform polar draw (taskspec layout
        styles); without it the draws are the pre-style ones (legacy golden scenes)."""
        fixed = dict(fixed or {})
        placed: list[tuple[np.ndarray, float]] = []
        on_mats = {k: v for k, v in fixed.items() if v.on is not None}
        for name, p in fixed.items():
            if p.on is None:
                self.layout[name] = p
                placed.append((np.asarray(p.xy), self._spec(name).footprint_radius))
        order = sorted((o for o in self.objects if o.name not in fixed and o.name not in skip),
                       key=lambda o: -o.footprint_radius)
        for spec in order:
            for _ in range(tries):
                if sampler is None:
                    r = self.rng.uniform(*r_range)
                    a = self.rng.uniform(-ang_max, ang_max)
                    xy = np.array([r * np.cos(a), r * np.sin(a)])
                else:
                    xy = sampler(placed)
                    if xy is None:
                        continue
                if self._free(xy, spec.footprint_radius, placed):
                    break
            else:
                raise Reject("no room")
            self.layout[spec.name] = Placement(xy=(float(xy[0]), float(xy[1])), yaw=float(self.rng.uniform(-np.pi, np.pi)))
            placed.append((xy, spec.footprint_radius))
        # objects that start on a mat: inside the mat's footprint, apart from each other
        inner: dict[str, list[tuple[np.ndarray, float]]] = {}
        for name, p in on_mats.items():
            mat = self._spec(p.on)
            centre = np.asarray(self.layout[p.on].xy)
            spec = self._spec(name)
            half = 0.5 * min(mat.size[0], mat.size[1]) - spec.footprint_radius
            for _ in range(tries):
                xy = centre + self.rng.uniform(-half, half, size=2)
                if self._free(xy, spec.footprint_radius, inner.setdefault(p.on, []), gap=0.012):
                    break
            else:
                raise Reject("no room on mat")
            inner[p.on].append((xy, spec.footprint_radius))
            self.layout[name] = Placement(xy=(float(xy[0]), float(xy[1])), yaw=float(self.rng.uniform(-np.pi, np.pi)),
                                          on=p.on)

    def _spec(self, name: str) -> ObjSpec:
        return next(o for o in self.objects if o.name == name)

    def distractors(self, n: int, kinds=("cube", "block", "cylinder", "ball")) -> list[ObjSpec]:
        return [self.add(str(self.rng.choice(kinds))) for _ in range(n)]


def sample_task(family: str, seed: int, *, max_tries: int = 50) -> TaskInstance:
    from .families import FAMILIES
    gen = FAMILIES[family]
    for k in range(max_tries):
        rng = np.random.default_rng([seed, k])
        b = Builder(rng)
        try:
            language, atoms, roles, fixed = gen(b)
            b.place(fixed)
        except Reject:
            continue
        return TaskInstance(family=family, seed=seed, language=language, objects=b.objects, layout=b.layout,
                            atoms=atoms, roles=roles)
    raise RuntimeError(f"could not sample {family} seed {seed}")


@dataclass(frozen=True)
class SceneConfig:
    """Scene distribution (CONTRACT 5). Defaults = the training scenes; legacy specs ignore the cube fields and their
    scenes are bit-identical to the pre-spec ones. Eval suites override fields (held-out cubes, scene shift).
    Cubes are registry rows (assets/registry.jsonl, soarm.objects.select_assets) filtered by these pools."""
    cube_split: str = "train"                    # registry split of the cubes: train | heldout
    cube_sizes: tuple[float, ...] = (3.0, 3.5, 4.0, 4.5, 5.0)
    cube_textures: tuple[str, ...] = ("plain", "wood")      # registry: plain | wood (train); letters | printed (heldout)
    randomize: bool = True                       # legacy lights, table tone, +-1.5 cm camera jitter, table friction
    # scene_random appearance (all default = no extra draws): camera shift, textured table/floor/wall, random lights
    camera_jitter_m: float = 0.015
    camera_rot_jitter_deg: float = 0.0
    table_textures: str | tuple[str, ...] = "plain"          # plain | all | texture ids (assets/textures.jsonl)
    lights: str = "default"                                  # default | random
    # div2 scene distribution (all default = the pre-div2 scenes and draws)
    table_mount: str = "center"      # center: base mid-table (legacy) | edge: base clamped at the back edge, random table
                                     # | edge3: edge mount with the div3 table ranges (size, height, offset)
    camera: str = "jitter"           # jitter: agentview + the two jitter fields | train | heldout | one view style
    layout: str = "band"             # band: uniform polar draws (legacy) | mixed: div2 layout styles | div3: div3 styles
    clutter: tuple[int, ...] = (0, 0)                        # [lo, hi] extra non-goal objects (declarative specs)
    clutter_kinds: tuple[str, ...] = ("cylinder", "block")
    platforms: float = 0.0           # probability of static platforms (1-4 cm boxes) in the band (declarative specs)
    # div3 additions (all default = the div2 draws)
    platforms_max: int = 1           # platforms per scene when present: 1..platforms_max
    distractor_stacks: float = 0.0   # probability that two "distractors" cubes start stacked (no goal refers to them)
    cube_size_weights: tuple[float, ...] = ()               # relative draw weight per cube_sizes entry (empty = uniform)

    def scene_is_default(self) -> bool:
        d = SceneConfig()
        keys = ("camera_jitter_m", "camera_rot_jitter_deg", "table_textures", "lights", "table_mount", "camera")
        return all(getattr(self, k) == getattr(d, k) for k in keys)

    @staticmethod
    def from_json(d: dict | None) -> "SceneConfig":
        d = dict(d or {})
        return SceneConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})

    def to_json(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in self.__dict__.items()}


def evaluate_goal(world, atoms) -> list[bool]:
    """Atom values: predicates.evaluate plus the `count` combinator (CONTRACT 2.2): ("count", members, {"pred", "args"
    with "$" for the member, "params", "eq"|"min"|"max"})."""
    from .predicates import evaluate
    out = []
    for name, args, params in atoms:
        if name != "count":
            out.append(evaluate(world, [(name, args, params)])[0])
            continue
        inner = [(params["pred"], tuple(m if a == "$" else a for a in params["args"]), params.get("params") or {})
                 for m in args]
        n = sum(evaluate(world, inner))
        out.append(("eq" not in params or n == params["eq"]) and n >= params.get("min", 0)
                   and n <= params.get("max", len(args)))
    return out

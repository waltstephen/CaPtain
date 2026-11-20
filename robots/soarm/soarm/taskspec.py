"""Task specs (CONTRACT.md 2): JSON task families, schema + semantic validation, registry by id/tid, sampling.

    spec = taskspec.get(101)            # or taskspec.get("cube_stack_two")
    task = taskspec.sample(spec, seed)  # -> tasks.TaskInstance (pure numpy)

Builders: "legacy:<family>" runs the hand-written generator in families.py (the 16 legacy families, bit-identical to
the pre-spec sampler, tests/test_taskspec.py); "declarative" interprets objects/layout/goal/instructions (cube specs,
taskgen specs).
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import jsonschema
import numpy as np

from .objects import COLORS, KINDS, WALL, ObjSpec, add_mesh, asset_spec, sample_spec, select_assets
from .predicates import PREDICATES
from .tasks import Builder, Placement, Reject, SceneConfig, TaskInstance

SPEC_DIR = Path(__file__).resolve().parent / "specs"
# Declarative layouts use the reliable top-down band, not the legacy annulus r 0.13-0.29 m, |angle| <= 70 deg: scripted
# picks of registry cubes failed on reach at r 0.13-0.16 and r >= 0.27 at large angles (5/20; soarm-assets and
# soarm-taskgen reach maps 11-10: reliable band r 0.15-0.25).
R_MIN, R_MAX = 0.15, 0.26
ANG_MAX = float(np.deg2rad(60.0))
# div2 layouts (SceneConfig.layout "mixed"): a mid-radius wing beyond ANG_MAX (11-10 reach map: 0-4 mm error at
# r 0.17-0.25 out to 70 deg), styles drawn per episode, clutter in the wide annulus (never grasped).
WING = (0.17, 0.24, float(np.deg2rad(70.0)))
LAYOUT_STYLES = {"uniform": 0.30, "cluster": 0.25, "side": 0.15, "row": 0.15, "spread": 0.15}
# div3 (SceneConfig.layout "div3"): the div2 styles plus near/far sub-bands, wide angles, a jittered grid and two clusters
LAYOUT_STYLES3 = {"uniform": 0.16, "cluster": 0.12, "side": 0.10, "row": 0.10, "spread": 0.10, "near": 0.10,
                  "far": 0.10, "wide": 0.08, "grid": 0.07, "two_clusters": 0.07}
# div3 receptacle placement (containers/mats/trays first, p RECEPTACLE_P): one sub-band of the reach band
RECEPTACLE_P = 0.5
RECEPTACLE_BANDS = ("near", "far", "left", "right", "centre")
# div3 yaw modes: random (div2), axis-aligned to the table, one shared yaw
YAW_MODES = {"random": 0.6, "aligned": 0.25, "same": 0.15}
CLUTTER_R, CLUTTER_ANG = (0.13, 0.29), float(np.deg2rad(70.0))      # the legacy annulus (taskgen start-reach check)
# Layout seeds per machine (CONTRACT 1): data never duplicates across machines; eval seeds are disjoint from data seeds.
# datahost: data 1,000,000-1,499,999 (pre-div2 cube blocks 1,000,200-1,008,999; div2 mix blocks 1,100,000+ per model),
# ad-hoc eval 1,500,000+.
SEED_RANGES = {"armhost": {"data": (0, 8999), "eval": (9000, 999_999)},
               "datahost": {"data": (1_000_000, 1_499_999), "eval": (1_500_000, 1_999_999)}}
TID_RANGES = {"legacy": (0, 99), "cube": (100, 999), "gen": (1000, 9999), "broad": (10000, 10**9)}

# div3 size variants: spec kind -> (scene kind, noun); sizes in _pseudo_size
PSEUDO_KINDS = {"tray": ("box", "tray"), "big_bowl": ("bowl", "bowl"), "big_mat": ("pad", "mat"),
                "big_cylinder": ("cylinder", "cylinder"), "big_block": ("block", "block")}
_KINDS = sorted(set(KINDS) | {"bin", "slot", "registry"} | set(PSEUDO_KINDS))
_ATTR = {"oneOf": [{"type": ["string", "number"]}, {"type": "array", "items": {"type": ["string", "number"]}},
                   {"type": "object", "additionalProperties": False, "minProperties": 1,
                    "properties": {"not": {"type": "array"}, "min": {"type": "number"}, "max": {"type": "number"},
                                   "same_role": {"type": "string"}, "not_role": {"type": "string"},
                                   "max_role": {"type": "string"}, "min_role": {"type": "string"},
                                   "distinct": {"type": "boolean"}}}]}
_COUNT = {"oneOf": [{"type": "integer", "minimum": 0},
                    {"type": "array", "items": {"type": "integer", "minimum": 0}, "minItems": 2, "maxItems": 2}]}
_ATOM = {"type": "object", "additionalProperties": False, "required": ["pred", "args"],
         "properties": {"pred": {"type": "string"}, "args": {"type": "array"}, "params": {"type": "object"},
                        "for_each": {"type": "string"}}}
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["id", "tid", "domain", "split", "provenance", "difficulty", "builder", "objects", "goal", "instructions"],
    "properties": {
        "id": {"type": "string", "pattern": "^[a-z][a-z0-9_]*$"}, "tid": {"type": "integer", "minimum": 0},
        "version": {"type": "integer"}, "domain": {"enum": ["cube", "broad", "legacy"]},
        "split": {"enum": ["train", "heldout"]}, "provenance": {"enum": ["hand", "llm"]},
        "difficulty": {"type": "integer", "minimum": 1, "maximum": 5},
        "builder": {"type": "string", "pattern": "^(declarative|legacy:[a-z_]+)$"},
        "objects": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["role", "kind"],
            "properties": {"role": {"type": "string", "pattern": "^[a-z][a-z0-9_]*$"},
                           "kind": {"oneOf": [{"enum": _KINDS}, {"type": "array", "items": {"enum": _KINDS}}]},
                           "count": _COUNT,
                           "where": {"type": "object", "additionalProperties": _ATTR}}}},
        "layout": {"type": "array", "items": {"type": "object", "required": ["type"],
                                              "properties": {"type": {"enum": ["on", "stack", "distance", "region"]}}}},
        "goal": {"type": "array", "minItems": 1, "items": {"oneOf": [_ATOM, {
            "type": "object", "additionalProperties": False, "required": ["count", "over"],
            "properties": {"count": _ATOM, "over": {"type": "string"}, "eq": {"type": "integer"},
                           "min": {"type": "integer"}, "max": {"type": "integer"}}}]}},
        "instructions": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "paraphrase_of": {"type": "array", "items": {"type": "integer"}},
        "workspace": {"type": "object", "additionalProperties": False,
                      "properties": {"r": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                                     "angle_deg": {"type": "number"}}},
        "vars": {"type": "object", "additionalProperties": {"type": "object", "minProperties": 1,
                                                            "additionalProperties": {"type": "string"}}},
        "notes": {"type": "string"},
    },
}
_REF = re.compile(r"^@([a-z][a-z0-9_]*)(\[(\d+)\]|\*)?$")
_SLOT = re.compile(r"\{(%?[a-z][a-z0-9_]*)(?:\.([a-z_]+))?\}")


def validate(spec: dict) -> None:
    """Raise ValueError when the spec breaks the schema or refers to unknown roles/predicates/kinds."""
    try:
        jsonschema.validate(spec, SCHEMA)
    except jsonschema.ValidationError as exc:
        raise ValueError(f"{spec.get('id')}: schema: {exc.message} at {list(exc.absolute_path)}") from None
    sid = spec["id"]
    roles = [o["role"] for o in spec["objects"]]
    if len(set(roles)) != len(roles):
        raise ValueError(f"{sid}: duplicate roles")
    lo, hi = TID_RANGES[{"legacy": "legacy", "cube": "cube", "broad": "broad"}[spec["domain"]]] \
        if spec["provenance"] == "hand" else TID_RANGES["gen"]
    if not lo <= spec["tid"] <= hi:
        raise ValueError(f"{sid}: tid {spec['tid']} outside {spec['domain']}/{spec['provenance']} range {lo}-{hi}")
    if spec["builder"].startswith("legacy:"):
        from .families import FAMILIES
        if spec["builder"][7:] not in FAMILIES:
            raise ValueError(f"{sid}: unknown legacy family")
    for o in spec["objects"]:
        for key, cons in (o.get("where") or {}).items():
            if isinstance(cons, dict):
                for rel in ("same_role", "not_role", "max_role", "min_role"):
                    if rel in cons and _role_of(cons[rel]) not in roles:
                        raise ValueError(f"{sid}: {o['role']}.{key} refers to unknown role {cons[rel]}")

    def check_atom(a: dict, member_ok: bool) -> None:
        if a["pred"] not in PREDICATES:
            raise ValueError(f"{sid}: unknown predicate {a['pred']}")
        for arg in a["args"]:
            if isinstance(arg, str) and arg.startswith("@"):
                m = _REF.match(arg)
                if not m or m.group(1) not in roles:
                    raise ValueError(f"{sid}: bad role reference {arg}")
            elif isinstance(arg, str) and arg.startswith("%"):
                if arg[1:] not in (spec.get("vars") or {}):
                    raise ValueError(f"{sid}: unknown var {arg}")
            elif arg == "$" and not member_ok:
                raise ValueError(f"{sid}: '$' outside for_each/count")
        if a.get("for_each") and a["for_each"] not in roles:
            raise ValueError(f"{sid}: for_each unknown role {a['for_each']}")

    for g in spec["goal"]:
        if "count" in g:
            if g["over"] not in roles or not any(k in g for k in ("eq", "min", "max")):
                raise ValueError(f"{sid}: count needs a known 'over' role and eq/min/max")
            check_atom(g["count"], True)
        else:
            check_atom(g, bool(g.get("for_each")))
    for t in spec["instructions"]:
        for role, _attr in _SLOT.findall(t):
            if role.startswith("%"):
                if role[1:] not in (spec.get("vars") or {}):
                    raise ValueError(f"{sid}: instruction slot {{{role}}} is not a var")
            elif role not in roles and role != "count":
                raise ValueError(f"{sid}: instruction slot {{{role}}} is not a role")
    for c in spec.get("layout") or []:
        for key in ("role", "a", "b", "support"):
            if key in c and _role_of(c[key]) not in roles:
                raise ValueError(f"{sid}: layout {c['type']} unknown role {c[key]}")
        for r in c.get("roles", []):
            if _role_of(r) not in roles:
                raise ValueError(f"{sid}: layout {c['type']} unknown role {r}")


def _role_of(ref: str) -> str:
    m = _REF.match(ref if ref.startswith("@") else "@" + ref)
    return m.group(1) if m else ref


@lru_cache(maxsize=1)
def registry() -> dict[str, dict]:
    """All specs under soarm/specs/**.json (one spec per file), validated; ids and tids unique."""
    out: dict[str, dict] = {}
    tids: dict[int, str] = {}
    for path in sorted(SPEC_DIR.rglob("*.json")):
        if path.name.startswith("_") or path.parent.name in ("suites", "mixes"):
            continue
        spec = json.loads(path.read_text())
        validate(spec)
        if spec["id"] in out or spec["tid"] in tids:
            raise ValueError(f"{path}: duplicate id/tid {spec['id']}/{spec['tid']}")
        out[spec["id"]] = spec
        tids[spec["tid"]] = spec["id"]
    return out


def get(ref: int | str) -> dict:
    """Spec by tid (int or digit string) or id."""
    reg = registry()
    if isinstance(ref, int) or str(ref).isdigit():
        tid = int(ref)
        for spec in reg.values():
            if spec["tid"] == tid:
                return spec
        raise KeyError(f"no spec with tid {tid}")
    return reg[ref]


def select(domain: str | list[str] | None = None, split: str | None = None,
           tids: tuple[int, int] | list[int] | None = None) -> list[dict]:
    domains = [domain] if isinstance(domain, str) else domain
    return sorted((s for s in registry().values() if (domains is None or s["domain"] in domains)
                   and (split is None or s["split"] == split) and (tids is None or tids[0] <= s["tid"] <= tids[1])),
                  key=lambda s: s["tid"])


def suite(name: str) -> dict:
    """Fixed eval suite (soarm/specs/suites/<name>.json, CONTRACT 6): {"id", "description", "jobs": [(spec id, seed)],
    "config": SceneConfig}. Seeds are always in the eval range of armhost (9000+). Specs: "select" (one selector or a
    list whose union is taken; keys domain/split/tids) or "mix" (the tasks of soarm/specs/mixes/<mix>.json); "config"
    is a SceneConfig, or {"mix": name, **overrides} = that mix's scene config with overrides."""
    d = json.loads((SPEC_DIR / "suites" / f"{name}.json").read_text())
    lo, _, hi = d["seeds"].partition("-")
    if "mix" in d:
        from . import mix
        chosen = [get(t) for t in mix.load(d["mix"])["weights"]]
    else:
        sels = d["select"] if isinstance(d["select"], list) else [d["select"]]
        chosen = sorted({s["id"]: s for sel in sels for s in select(**sel)}.values(), key=lambda s: s["tid"])
    specs = [s for s in chosen if s["id"] not in d.get("exclude", [])]
    jobs = [(s["id"], seed) for seed in range(int(lo), int(hi or lo) + 1) for s in specs]
    cfg = dict(d["config"])
    if "mix" in cfg:
        from . import mix
        cfg = {**mix.load(cfg.pop("mix"))["config"].to_json(), **cfg}
    return {"id": d["id"], "description": d["description"], "jobs": jobs, "config": SceneConfig.from_json(cfg)}


def suites() -> list[str]:
    return sorted(p.stem for p in (SPEC_DIR / "suites").glob("*.json"))


# ---------------------------------------------------------------------------------------------------------- sampling
def sample(spec: dict, seed: int, config: SceneConfig | None = None, *, max_tries: int = 50) -> TaskInstance:
    """Deterministic task instance of `spec` for layout seed `seed` (same rng scheme as tasks.sample_task)."""
    if spec["builder"].startswith("legacy:"):
        from .tasks import sample_task
        return sample_task(spec["builder"][7:], seed, max_tries=max_tries)
    config = config or SceneConfig()
    last = None
    for k in range(max_tries):
        rng = np.random.default_rng([seed, k])
        try:
            return _declarative(spec, seed, rng, config)
        except Reject as exc:
            last = exc
    raise RuntimeError(f"could not sample {spec['id']} seed {seed} ({last})")


def _count(rng: np.random.Generator, c) -> int:
    if c is None:
        return 1
    return int(c) if isinstance(c, int) else int(rng.integers(c[0], c[1] + 1))


def _attr(o: ObjSpec, key: str):
    return o.color if key == "color" else o.attrs.get(key)


def _values(members: dict[str, list[ObjSpec]], ref: str, attr: str) -> list:
    m = _REF.match(ref if ref.startswith("@") else "@" + ref)
    objs = members.get(m.group(1), [])
    if m.group(3) is not None:
        idx = int(m.group(3))
        objs = objs[idx:idx + 1]
    return [_attr(o, attr) for o in objs]


def _candidates(pool: list, cons, attr: str, members, mine: list[ObjSpec]) -> list:
    """Values of `attr` in `pool` allowed by a CONTRACT 2.1 constraint."""
    if cons is None:
        return list(pool)
    if isinstance(cons, list):
        return [v for v in pool if v in cons] if pool else list(cons)
    if not isinstance(cons, dict):
        return [v for v in pool if v == cons] if pool else [cons]
    cand = list(pool)
    if "same_role" in cons:      # first member of a self-referencing role ("same as items[0]") is unconstrained
        same = _values(members, cons["same_role"], attr)
        cand = [v for v in cand if v in same] if same else cand
    if "not" in cons:
        cand = [v for v in cand if v not in cons["not"]]
    if "not_role" in cons:
        used = set(_values(members, cons["not_role"], attr))
        cand = [v for v in cand if v not in used]
    if "min" in cons:
        cand = [v for v in cand if v >= cons["min"]]
    if "max" in cons:
        cand = [v for v in cand if v <= cons["max"]]
    if "max_role" in cons:
        vals = _values(members, cons["max_role"], attr)
        cand = [v for v in cand if vals and v <= min(vals)]
    if "min_role" in cons:
        vals = _values(members, cons["min_role"], attr)
        cand = [v for v in cand if vals and v >= max(vals)]
    if cons.get("distinct"):
        cand = [v for v in cand if v not in {_attr(o, attr) for o in mine}]
    return cand


def _resolve(rng, pool: list, cons, attr: str, members, mine: list[ObjSpec]):
    """Draw one value of `attr` from `pool` under a CONTRACT 2.1 constraint."""
    cand = _candidates(pool, cons, attr, members, mine)
    if not cand:
        raise Reject(f"no value for {attr}")
    v = cand[int(rng.integers(len(cand)))]
    return v.item() if hasattr(v, "item") else v


ROW_ATTRS = ("size_class", "color", "texture", "letter")      # registry columns a cube selector may constrain
SLOT_CLEARANCE = 0.008


def _cube_row(rng, where: dict, members, mine: list[ObjSpec], config: SceneConfig) -> dict:
    """One registry cube row: the scene config's pools, then the selector's constraints, attribute by attribute."""
    rows = select_assets(category="cube", split=config.cube_split, size_class=tuple(config.cube_sizes),
                         texture=tuple(config.cube_textures))
    if "color" in where and "texture" not in where:
        rows = [r for r in rows if r["texture"] != "wood"]       # colour-defined roles stay coloured
    for key in ROW_ATTRS:
        if key in where:
            allowed = _candidates(sorted({r.get(key) for r in rows if r.get(key) is not None}), where[key], key,
                                  members, mine)
            rows = [r for r in rows if r.get(key) in allowed]
    if not rows:
        raise Reject("no registry cube matches the selector")
    if config.cube_size_weights:        # div3: size classes drawn with relative weights (bias toward 4-5 cm)
        wmap = dict(zip(config.cube_sizes, config.cube_size_weights))
        p = np.array([wmap.get(r["size_class"], 1.0) for r in rows], float)
        return rows[int(rng.choice(len(rows), p=p / p.sum()))]
    return rows[int(rng.integers(len(rows)))]


def _pseudo_size(rng, kind: str) -> tuple[tuple[float, float, float], float]:
    """(size, mass) of a div3 size variant (PSEUDO_KINDS)."""
    u = rng.uniform
    if kind == "tray":                  # wide shallow box: 14-18 x 11-14 cm, 2.2-3 cm rim
        return (u(0.14, 0.18), u(0.11, 0.14), u(0.022, 0.03)), 0.15
    if kind == "big_bowl":              # 13-15 cm bowl
        d = u(0.13, 0.15)
        return (d, d, u(0.035, 0.045)), 0.14
    if kind == "big_mat":               # 12-15 cm mat
        return (u(0.12, 0.15), u(0.12, 0.15), 0.002), 0.0
    if kind == "big_cylinder":          # 3.6-4.2 cm wide, 5-7 cm tall
        d, h = u(0.036, 0.042), u(0.05, 0.07)
        return (d, d, h), 500 * d * d * h
    if kind == "big_block":             # 6-7 x 3-3.5 x 3-3.5 cm
        s = (u(0.06, 0.07), u(0.03, 0.035), u(0.03, 0.035))
        return s, 500 * float(np.prod(s))
    raise ValueError(kind)


def _make_object(b: Builder, sel: dict, members, mine: list[ObjSpec], config: SceneConfig) -> ObjSpec:
    where = sel.get("where") or {}
    kind = sel["kind"] if isinstance(sel["kind"], str) else str(sel["kind"][int(b.rng.integers(len(sel["kind"])))])
    if kind == "registry":      # GSO/YCB mesh of the config's split, one per noun group (soarm.objects.add_mesh)
        return add_mesh(b, where, config.cube_split)
    if kind == "cube":
        row = _cube_row(b.rng, where, members, mine, config)
        k = b._count.get("cube", 0)
        b._count["cube"] = k + 1
        spec = asset_spec(b.rng, row, f"cube_{k}")
        b.objects.append(spec)
        return spec
    if kind == "slot":      # tight holder for one cube: inner side = cube side (+3 % jitter) + 8 mm, 3 cm walls
        size = float(_resolve(b.rng, list(config.cube_sizes), where.get("size_class"), "size_class", members, mine))
        color = _resolve(b.rng, list(COLORS), where.get("color"), "color", members, mine)
        k = b._count.get("slot", 0)
        b._count["slot"] = k + 1
        side = 0.01 * size * (1 + 0.03) + SLOT_CLEARANCE + 2 * WALL     # 0.03 = registry scale_jitter
        spec = ObjSpec(name=f"slot_{k}", kind="box", color=str(color), size=(side, side, 0.03), mass=0.12,
                       attrs={"slot": True, "size_class": size})
        b.objects.append(spec)
        return spec
    if kind in PSEUDO_KINDS:   # div3 size variants of a base kind (own noun for the tray)
        real, noun = PSEUDO_KINDS[kind]
        color = _resolve(b.rng, list(COLORS), where.get("color"), "color", members, mine)
        k = b._count.get(kind, 0)
        b._count[kind] = k + 1
        size, mass = _pseudo_size(b.rng, kind)
        spec = ObjSpec(name=f"{kind}_{k}", kind=real, color=str(color), size=tuple(float(v) for v in size),
                       mass=float(max(mass, 0.01)) if mass else 0.0, attrs={"variant": kind, "noun": noun})
        b.objects.append(spec)
        return spec
    real = "box" if kind == "bin" else kind
    color = _resolve(b.rng, list(COLORS), where.get("color"), "color", members, mine)
    k = b._count.get(kind, 0)
    b._count[kind] = k + 1
    spec = sample_spec(b.rng, real, str(color), f"{kind}_{k}")
    if kind == "bin":       # sorting bin: open box, low walls
        spec = ObjSpec(name=spec.name, kind="box", color=spec.color, size=(spec.size[0], spec.size[1], 0.03),
                       mass=spec.mass, attrs={"bin": True})
    b.objects.append(spec)
    return spec


def _names(members, ref: str) -> list[str]:
    m = _REF.match(ref if ref.startswith("@") else "@" + ref)
    objs = members[m.group(1)]
    if m.group(3) is not None:
        objs = objs[int(m.group(3)):int(m.group(3)) + 1]
    return [o.name for o in objs]


def _polar(rng, r=(R_MIN, R_MAX), ang=(-ANG_MAX, ANG_MAX)) -> np.ndarray:
    rr, a = rng.uniform(*r), rng.uniform(*ang)
    return np.array([rr * np.cos(a), rr * np.sin(a)])


def layout_style(rng: np.random.Generator, styles: dict[str, float] = LAYOUT_STYLES) -> str:
    names = sorted(styles)
    p = np.array([styles[n] for n in names])
    return names[int(rng.choice(len(names), p=p / p.sum()))]


def in_band(xy, r=(R_MIN, R_MAX), ang_max: float = ANG_MAX, wing: bool = True) -> bool:
    rr, a = float(np.hypot(xy[0], xy[1])), abs(float(np.arctan2(xy[1], xy[0])))
    if r[0] <= rr <= r[1] and a <= ang_max:
        return True
    return wing and WING[0] <= rr <= WING[1] and a <= WING[2]


def layout_sampler(rng: np.random.Generator, style: str, r=(R_MIN, R_MAX), ang_max: float = ANG_MAX,
                   wing: bool = True):
    """Candidate positions `f(placed) -> xy | None` of one layout style (SceneConfig.layout "mixed"); Builder.place
    keeps the footprint/gap check, so every style still leaves room for the open jaw."""
    amax = max(ang_max, WING[2]) if wing else ang_max

    def ok(xy):
        return xy if in_band(xy, r, ang_max, wing) else None

    def uniform():                      # area-uniform over the band (+ wing)
        for _ in range(100):
            a = rng.uniform(-amax, amax)
            xy = ok(float(np.sqrt(rng.uniform(r[0] ** 2, r[1] ** 2))) * np.array([np.cos(a), np.sin(a)]))
            if xy is not None:
                return xy
        raise Reject("no band point")

    def annulus(r_lo, r_hi, a_lo, a_hi):   # area-uniform in a sub-annulus (a_lo > 0: both signs), kept only in the band
        for _ in range(100):
            a = rng.uniform(a_lo, a_hi) * (1.0 if rng.random() < 0.5 else -1.0) if a_lo > 0 else rng.uniform(a_lo, a_hi)
            xy = ok(float(np.sqrt(rng.uniform(r_lo ** 2, r_hi ** 2))) * np.array([np.cos(a), np.sin(a)]))
            if xy is not None:
                return xy
        return None

    if style == "uniform":
        return lambda placed: uniform()
    if style == "cluster":              # everything near one random centre
        centre, sigma = uniform(), float(rng.uniform(0.035, 0.06))
        return lambda placed: ok(centre + rng.normal(0.0, sigma, 2))
    if style == "side":                 # one side of the arm only
        sgn, lo = (1.0 if rng.random() < 0.5 else -1.0), float(rng.uniform(-0.15, 0.1))

        def side(placed):
            a = sgn * rng.uniform(lo, amax)
            return ok(float(np.sqrt(rng.uniform(r[0] ** 2, r[1] ** 2))) * np.array([np.cos(a), np.sin(a)]))
        return side
    if style == "near":                 # div3: inner part of the band, close to the base
        r_hi = r[0] + float(rng.uniform(0.5, 0.65)) * (r[1] - r[0])
        return lambda placed: annulus(r[0], r_hi, -amax, amax)
    if style == "far":                  # div3: outer part of the band (the wing included)
        r_lo = r[0] + float(rng.uniform(0.35, 0.5)) * (r[1] - r[0])
        return lambda placed: annulus(r_lo, r[1], -amax, amax)
    if style == "wide":                 # div3: large angles on both sides, little in front of the base
        a_lo = float(rng.uniform(0.35, 0.5)) * amax
        return lambda placed: annulus(r[0], r[1], a_lo, amax)
    if style == "grid":                 # div3: jittered grid with a random orientation and 6-9 cm pitch
        p0, phi, pitch = uniform(), float(rng.uniform(-np.pi, np.pi)), float(rng.uniform(0.06, 0.09))
        u, v = np.array([np.cos(phi), np.sin(phi)]), np.array([-np.sin(phi), np.cos(phi)])
        return lambda placed: ok(p0 + pitch * (int(rng.integers(-2, 3)) * u + int(rng.integers(-2, 3)) * v)
                                 + rng.normal(0.0, 0.004, 2))
    if style == "two_clusters":         # div3: two groups
        centres, sigma = [uniform(), uniform()], float(rng.uniform(0.03, 0.05))
        return lambda placed: ok(centres[int(rng.integers(2))] + rng.normal(0.0, sigma, 2))
    if style == "row":                  # along an arc around the base or a straight line
        if rng.random() < 0.5:
            r0 = float(rng.uniform(r[0] + 0.02, r[1] - 0.02))

            def arc(placed):
                a = rng.uniform(-amax, amax)
                return ok((r0 + rng.normal(0.0, 0.006)) * np.array([np.cos(a), np.sin(a)]))
            return arc
        p0, phi = uniform(), float(rng.uniform(-np.pi, np.pi))
        u = np.array([np.cos(phi), np.sin(phi)])
        return lambda placed: ok(p0 + rng.uniform(-0.16, 0.16) * u + rng.normal(0.0, 0.006) * np.array([-u[1], u[0]]))
    if style == "spread":               # far apart: best of 10 candidates by distance to the placed footprints
        def spread(placed):
            cands = [uniform() for _ in range(10)]
            if not placed:
                return cands[0]
            return max(cands, key=lambda c: min(float(np.linalg.norm(c - p)) - pr for p, pr in placed))
        return spread
    raise ValueError(f"layout style {style!r}")


def _layout(spec: dict, b: Builder, members, config: SceneConfig | None = None,
            clutter: list[ObjSpec] = ()) -> list[dict]:
    """Places every object (b.layout); returns the static platforms (SceneConfig.platforms)."""
    config = config or SceneConfig()
    fixed: dict[str, Placement] = {}
    stacks: list[list[str]] = []
    spec_of = {o.name: o for o in b.objects}
    placed: list[tuple[np.ndarray, float]] = []
    for c in spec.get("layout") or []:
        t = c["type"]
        if t == "on":
            support = _names(members, c["support"])[0]
            for n in _names(members, c["role"]):
                fixed[n] = Placement(xy=(0.0, 0.0), yaw=0.0, on=support)
        elif t == "stack":
            stacks.append([n for r in c["roles"] for n in _names(members, r)])
        elif t == "distance":
            a, bb = _names(members, c["a"])[0], _names(members, c["b"])[0]
            for _ in range(100):
                pa = _polar(b.rng, (R_MIN + 0.01, R_MAX - 0.01), (-ANG_MAX + 0.2, ANG_MAX - 0.2))
                phi = b.rng.uniform(-np.pi, np.pi)
                pb = pa + b.rng.uniform(c.get("min", 0.0), c.get("max", 0.2)) * np.array([np.cos(phi), np.sin(phi)])
                rb, ab = float(np.hypot(*pb)), float(np.arctan2(pb[1], pb[0]))
                if R_MIN <= rb <= R_MAX and abs(ab) <= ANG_MAX and b._free(pa, spec_of[a].footprint_radius, placed) \
                        and b._free(pb, spec_of[bb].footprint_radius, placed):
                    break
            else:
                raise Reject("no distance layout")
            for n, p in ((a, pa), (bb, pb)):
                fixed[n] = Placement(xy=(float(p[0]), float(p[1])), yaw=float(b.rng.uniform(-np.pi, np.pi)))
                placed.append((p, spec_of[n].footprint_radius))
        elif t == "region":
            ang = tuple(np.deg2rad(v) for v in c.get("angle_deg", (-70, 70)))
            for n in _names(members, c["role"]):
                for _ in range(200):
                    p = _polar(b.rng, tuple(c.get("r", (R_MIN, R_MAX))), ang)
                    if b._free(p, spec_of[n].footprint_radius, placed):
                        break
                else:
                    raise Reject("no room in region")
                fixed[n] = Placement(xy=(float(p[0]), float(p[1])), yaw=float(b.rng.uniform(-np.pi, np.pi)))
                placed.append((p, spec_of[n].footprint_radius))
    div3 = config.layout == "div3"
    if div3 and config.distractor_stacks > 0:        # two distractor cubes start stacked (larger one below)
        dis = [o for o in members.get("distractors", []) if o.kind == "cube" and o.name not in fixed
               and not any(o.name in s for s in stacks)]
        if len(dis) >= 2 and b.rng.random() < config.distractor_stacks:
            stacks.append([o.name for o in sorted(dis[:2], key=lambda o: -o.size[0])])
    upper = {n for s in stacks for n in s[1:]} | {o.name for o in clutter}
    ws = spec.get("workspace") or {}      # default band for unconstrained objects (wider for mats-heavy scenes)
    r_ws = tuple(ws.get("r", (R_MIN, R_MAX)))
    ang_ws = float(np.deg2rad(ws["angle_deg"])) if "angle_deg" in ws else ANG_MAX
    default_band = "r" not in ws and "angle_deg" not in ws
    sampler = None
    if config.layout == "mixed":
        sampler = layout_sampler(b.rng, layout_style(b.rng), r_ws, ang_ws, wing=default_band)
    elif div3:
        sampler = layout_sampler(b.rng, layout_style(b.rng, LAYOUT_STYLES3), r_ws, ang_ws, wing=default_band)
        if b.rng.random() < RECEPTACLE_P:
            _place_receptacles(b, fixed, placed, upper, r_ws, ang_ws, default_band)
    elif config.layout != "band":
        raise ValueError(f"layout={config.layout!r}")
    b.place(fixed, skip=upper, r_range=r_ws, ang_max=ang_ws, sampler=sampler)
    if div3:
        _yaw_mode(b)
    for s in stacks:
        for below, above in zip(s[:-1], s[1:]):
            if spec_of[above].size[0] > spec_of[below].size[0] + 0.005:
                raise Reject("stack: upper cube wider than the one below")
            base = np.asarray(b.layout[below].xy)
            xy = base + b.rng.uniform(-0.003, 0.003, size=2)
            yaw = b.layout[below].yaw + float(b.rng.uniform(-0.15, 0.15))
            b.layout[above] = Placement(xy=(float(xy[0]), float(xy[1])), yaw=yaw, on=below)
    foot = [(np.asarray(p.xy), spec_of[n].footprint_radius) for n, p in b.layout.items() if p.on is None]
    for o in clutter:           # non-goal objects last: wide annulus, clear of everything; one that does not fit is dropped
        for _ in range(100):
            p = _polar(b.rng, CLUTTER_R, (-CLUTTER_ANG, CLUTTER_ANG))
            if b._free(p, o.footprint_radius, foot):
                b.layout[o.name] = Placement(xy=(float(p[0]), float(p[1])), yaw=float(b.rng.uniform(-np.pi, np.pi)))
                foot.append((p, o.footprint_radius))
                break
        else:
            b.objects.remove(o)
    if config.platforms > 0 and b.rng.random() < config.platforms:
        from .scene_random import sample_platforms
        n = 1 if config.platforms_max <= 1 else int(b.rng.integers(1, config.platforms_max + 1))
        return [dict(p.__dict__) for p in sample_platforms(b.rng, n, foot)]
    return []


def _place_receptacles(b: Builder, fixed: dict[str, Placement], placed: list, skip: set[str], r, ang_max: float,
                       wing: bool) -> None:
    """div3: containers/mats/trays not placed by the spec go first into one sub-band of the reach band (near the base,
    far, left, right or centre); one that finds no room there is left to the layout sampler."""
    band = RECEPTACLE_BANDS[int(b.rng.integers(len(RECEPTACLE_BANDS)))]
    amax = max(ang_max, WING[2]) if wing else ang_max
    span = r[1] - r[0]
    r_lo, r_hi = {"near": (r[0], r[0] + 0.45 * span), "far": (r[0] + 0.55 * span, r[1])}.get(band, r)
    a_lo, a_hi = {"left": (0.3, amax), "right": (-amax, -0.3), "centre": (-0.45, 0.45)}.get(band, (-amax, amax))
    recs = sorted((o for o in b.objects if o.category in ("container", "region") and o.name not in fixed
                   and o.name not in skip), key=lambda o: -o.footprint_radius)
    for o in recs:
        for _ in range(200):
            p = _polar(b.rng, (r_lo, r_hi), (a_lo, a_hi))
            if in_band(p, r, ang_max, wing) and b._free(p, o.footprint_radius, placed):
                fixed[o.name] = Placement(xy=(float(p[0]), float(p[1])), yaw=float(b.rng.uniform(-np.pi, np.pi)))
                placed.append((p, o.footprint_radius))
                break


def _yaw_mode(b: Builder) -> None:
    """div3: object yaws random (as div2), axis-aligned with the table, or one shared yaw (+ a few degrees)."""
    names = sorted(YAW_MODES)
    p = np.array([YAW_MODES[n] for n in names])
    mode = names[int(b.rng.choice(len(names), p=p / p.sum()))]
    if mode == "random":
        return
    yaw0 = float(b.rng.uniform(-np.pi, np.pi))
    for name in sorted(b.layout):
        pl = b.layout[name]
        if mode == "aligned":
            yaw = 0.5 * np.pi * int(b.rng.integers(4)) + float(b.rng.normal(0.0, 0.05))
        else:
            yaw = yaw0 + float(b.rng.normal(0.0, 0.07))
        b.layout[name] = Placement(xy=pl.xy, yaw=float(yaw), on=pl.on, lying=pl.lying)


def _args(args: list, members, member: str | None, var: dict[str, str]) -> tuple:
    out: list = []
    for a in args:
        if a == "$":
            out.append(member)
        elif isinstance(a, str) and a.startswith("@"):
            m = _REF.match(a)
            names = _names(members, a)
            if m.group(2) == "*":
                out += names
            elif len(names) != 1:
                raise ValueError(f"{a} has {len(names)} members: use {a}[i], {a}* or for_each")
            else:
                out.append(names[0])
        elif isinstance(a, str) and a.startswith("%"):
            out.append(var[a[1:]])
        else:
            out.append(a)
    return tuple(out)


def _atoms(spec: dict, members, var: dict[str, str]) -> list[tuple[str, tuple, dict]]:
    atoms = []
    for g in spec["goal"]:
        if "count" in g:
            inner = g["count"]
            over = tuple(o.name for o in members[g["over"]])
            params = {"pred": inner["pred"], "params": dict(inner.get("params") or {}),
                      "args": [x if x == "$" else _args([x], members, None, var)[0] for x in inner["args"]]}
            params.update({k: g[k] for k in ("eq", "min", "max") if k in g})
            atoms.append(("count", over, params))
        elif g.get("for_each"):
            for o in members[g["for_each"]]:
                atoms.append((g["pred"], _args(g["args"], members, o.name, var), dict(g.get("params") or {})))
        else:
            atoms.append((g["pred"], _args(g["args"], members, None, var), dict(g.get("params") or {})))
    return atoms


_NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def _join(words: list[str]) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def _language(spec: dict, b: Builder, members, var: dict[str, str]) -> str:
    template = spec["instructions"][int(b.rng.integers(len(spec["instructions"])))]
    count = next((g.get("eq", g.get("min", g.get("max"))) for g in spec["goal"] if "count" in g), None)

    def fill(m: re.Match) -> str:
        role, attr = m.group(1), m.group(2)
        if role.startswith("%"):
            return spec["vars"][role[1:]][var[role[1:]]]
        if role == "count":
            return _NUM.get(count, str(count))
        objs = members[role]
        if attr is None:
            return _join([b.ref(o) for o in objs])
        vals = [_attr(o, attr) for o in objs]
        if attr == "size_class":
            vals = [f"{v:g} cm" for v in vals]
        return _join([str(v) for v in dict.fromkeys(vals)])

    return _SLOT.sub(fill, template)


def _declarative(spec: dict, seed: int, rng: np.random.Generator, config: SceneConfig) -> TaskInstance:
    b = Builder(rng)
    var = {name: str(sorted(opts)[int(rng.integers(len(opts)))]) for name, opts in sorted((spec.get("vars") or {}).items())}
    members: dict[str, list[ObjSpec]] = {}
    for sel in spec["objects"]:
        mine: list[ObjSpec] = []
        members[sel["role"]] = mine
        for _ in range(_count(rng, sel.get("count"))):
            mine.append(_make_object(b, sel, members, mine, config))
    clutter = _clutter(spec, b, rng, config)
    platforms = _layout(spec, b, members, config, clutter)
    language = _language(spec, b, members, var)
    atoms = _atoms(spec, members, var)
    roles = {r: ",".join(o.name for o in objs) for r, objs in members.items() if objs}
    return TaskInstance(family=spec["id"], seed=seed, language=language, objects=b.objects, layout=b.layout,
                        atoms=atoms, roles=roles, platforms=platforms)


def _clutter(spec: dict, b: Builder, rng: np.random.Generator, config: SceneConfig) -> list[ObjSpec]:
    """SceneConfig.clutter: extra objects no goal or instruction refers to, of kinds the spec does not use (so
    "the cylinder" / "all the cubes" keep their meaning); placed by _layout in the wide annulus."""
    lo, hi = config.clutter
    if hi <= 0:
        return []
    used = {k for o in spec["objects"] for k in ([o["kind"]] if isinstance(o["kind"], str) else o["kind"])}
    used |= {PSEUDO_KINDS[k][0] for k in used if k in PSEUDO_KINDS}
    kinds = [k for k in config.clutter_kinds if k not in used]
    n = int(rng.integers(lo, hi + 1))
    if not kinds or not n:
        return []
    return [b.add(str(kinds[int(rng.integers(len(kinds)))])) for _ in range(n)]

"""Object catalog of the SO-101 twin: small graspable objects, containers and flat regions sized for its gripper.

Frames: world = robot base frame (x forward, y left, z up), table top at z = 0. All sizes are full extents in metres.
The SO-101 jaw opens 1.6 cm at 0 rad and ~8 cm at 0.9 rad, so graspable objects are 2–5 cm wide.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

COLORS: dict[str, tuple[float, float, float]] = {
    "red": (0.85, 0.12, 0.10), "green": (0.15, 0.62, 0.20), "blue": (0.12, 0.30, 0.85), "yellow": (0.95, 0.85, 0.10),
    "orange": (0.95, 0.50, 0.08), "purple": (0.52, 0.20, 0.70), "pink": (0.96, 0.52, 0.72), "cyan": (0.10, 0.78, 0.85),
    "white": (0.93, 0.93, 0.90), "black": (0.07, 0.07, 0.07), "brown": (0.50, 0.30, 0.12), "gray": (0.52, 0.52, 0.52),
}

# kind -> (category, noun)
KINDS: dict[str, tuple[str, str]] = {
    "cube": ("object", "cube"), "block": ("object", "block"), "cylinder": ("object", "cylinder"),
    "ball": ("object", "ball"), "tile": ("object", "tile"),
    "bowl": ("container", "bowl"), "box": ("container", "box"), "cup": ("container", "cup"),
    "pad": ("region", "mat"),
    "registry": ("object", "object"),     # broad registry objects (asset_spec; noun from attrs["noun"])
}
GRASPABLE = ("cube", "block", "cylinder", "ball")
WALL = 0.004          # container wall thickness
FLOOR = 0.004         # container floor thickness


@dataclass(frozen=True)
class ObjSpec:
    name: str                          # unique id, e.g. "cube_0"
    kind: str
    color: str
    size: tuple[float, float, float]   # full extents x, y, z in the object frame (cylinder/ball/bowl/cup: diameter, diameter, height)
    mass: float = 0.03
    attrs: dict = field(default_factory=dict, compare=False, hash=False)   # e.g. {"size_word": "small"}

    @property
    def category(self) -> str:
        return KINDS[self.kind][0]

    @property
    def noun(self) -> str:
        return self.attrs.get("noun") or KINDS[self.kind][1]

    @property
    def round(self) -> bool:
        return self.kind in ("cylinder", "ball", "bowl", "cup")

    @property
    def footprint_radius(self) -> float:
        """Radius of the upright footprint (layout clearance)."""
        sx, sy, _ = self.size
        return 0.5 * sx if self.round else 0.5 * float(np.hypot(sx, sy))

    @property
    def height(self) -> float:
        return self.size[2]

    def inner_half_xy(self) -> tuple[float, float]:
        """Containers/regions: half extents of the usable inner area (round: radius twice)."""
        sx, sy, _ = self.size
        wall = WALL if self.category == "container" else 0.0
        return 0.5 * sx - wall, 0.5 * sy - wall

    def z_half_extent(self, rot: np.ndarray) -> float:
        """Half extent along world z of the object rotated by `rot` (3x3)."""
        sx, sy, sz = self.size
        if self.kind == "ball":
            return 0.5 * sx
        if self.round:   # cylinder-like, axis = object z
            c = abs(float(rot[2, 2]))
            return 0.5 * sz * c + 0.5 * sx * float(np.sqrt(max(0.0, 1.0 - c * c)))
        return 0.5 * float(np.abs(rot[2, :]) @ np.array([sx, sy, sz]))


def sample_spec(rng: np.random.Generator, kind: str, color: str, name: str) -> ObjSpec:
    u = rng.uniform
    if kind == "cube":
        s = u(0.025, 0.035); size, mass = (s, s, s), 600 * s ** 3
    elif kind == "block":
        size = (u(0.045, 0.06), u(0.022, 0.028), u(0.022, 0.03)); mass = 500 * float(np.prod(size))
    elif kind == "cylinder":
        d = u(0.025, 0.034); size = (d, d, u(0.035, 0.055)); mass = 500 * d * d * size[2]
    elif kind == "ball":
        d = u(0.028, 0.035); size, mass = (d, d, d), 400 * d ** 3
    elif kind == "tile":
        # square 2.5 cm slab, too wide to grasp comfortably: pushed. Pilot used a round 1.6 cm puck: it was pressed into
        # the table or spun off the push line (0/20 for both teachers, 0/6 scripted)
        d = u(0.065, 0.08); size, mass = (d, d, 0.025), 0.05
    elif kind == "bowl":
        d = u(0.10, 0.12); size, mass = (d, d, u(0.035, 0.045)), 0.12
    elif kind == "box":
        size, mass = (u(0.10, 0.12), u(0.085, 0.10), u(0.03, 0.04)), 0.12
    elif kind == "cup":
        d = u(0.07, 0.08); size, mass = (d, d, u(0.055, 0.065)), 0.08
    elif kind == "pad":
        size, mass = (u(0.085, 0.10), u(0.085, 0.10), 0.002), 0.0
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return ObjSpec(name=name, kind=kind, color=color, size=tuple(float(v) for v in size),
                   mass=float(max(mass, 0.01)) if mass else 0.0)


def describe(obj: ObjSpec, scene: list[ObjSpec]) -> str | None:
    """Shortest unique referring expression ("the red cube", "the small blue cylinder"); None if not unique."""
    same_kind = [o for o in scene if o.noun == obj.noun]
    if len(same_kind) == 1:
        return f"the {obj.noun}"
    same_color = [o for o in same_kind if o.color == obj.color]
    if len(same_color) == 1:
        return f"the {obj.color} {obj.noun}"
    if len(same_color) == 2:
        other = same_color[0] if same_color[1] is obj else same_color[1]
        a, b = _size_key(obj), _size_key(other)
        if abs(a - b) > 0.25 * max(a, b):
            return f"the {'larger' if a > b else 'smaller'} {obj.color} {obj.noun}"
    return None


def _size_key(o: ObjSpec) -> float:
    return float(np.prod(o.size)) ** (1 / 3)


# ---------------------------------------------------------------------------------------------------- asset registry
# assets/registry.jsonl: one row per real/procedural object (assets/build_cubes.py; gate fields: tools/gate_assets.py).
# Mesh/texture paths in a row are relative to asset_root() ($SOARM_ASSETS, default $SOARM_ROOT/assets).
REGISTRY_PATH = Path(__file__).resolve().parent.parent / "assets" / "registry.jsonl"


def asset_root() -> Path:
    root = os.environ.get("SOARM_ASSETS") or str(Path(os.environ.get("SOARM_ROOT", "~/soarm_work")) / "assets")
    return Path(root).expanduser()


@lru_cache(maxsize=4)
def _load(path: str) -> tuple[dict, ...]:
    return tuple(json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())


def load_registry(path: str | Path | None = None) -> list[dict]:
    return list(_load(str(path or REGISTRY_PATH)))


def registry_row(asset_id: str, path: str | Path | None = None) -> dict:
    for r in _load(str(path or REGISTRY_PATH)):
        if r["id"] == asset_id:
            return r
    raise KeyError(f"asset {asset_id!r} not in {path or REGISTRY_PATH}")


def select_assets(rows: list[dict] | None = None, *, graspable: bool | None = True, **attrs) -> list[dict]:
    """Rows whose attributes match: a scalar matches equal values or list membership (colours), a tuple/list/set is a
    set of allowed values. `graspable=True` (default) keeps rows that passed the physics gate; None ignores the gate.
        select_assets(category="cube", split="train", texture=("plain", "wood"), size_class=(3.0, 4.0))"""
    out = []
    for r in (load_registry() if rows is None else rows):
        if graspable is not None and bool(r.get("graspable")) != graspable:
            continue
        ok = True
        for k, want in attrs.items():
            have = r.get(k)
            allowed = set(want) if isinstance(want, (tuple, list, set, frozenset)) else {want}
            values = set(have) if isinstance(have, list) else {have}
            if not values & allowed:
                ok = False
                break
        if ok:
            out.append(r)
    return out


def asset_spec(rng: np.random.Generator, row: dict, name: str) -> ObjSpec:
    """ObjSpec of one registry object with per-episode physical randomisation. Cubes: side x (1 +- scale_jitter),
    density and friction uniform in the row's ranges, mass clipped to mass_range_kg; attrs carry the CONTRACT 2.1 cube
    attributes (size_class, color, texture, density, friction, letter). Meshes (GSO/YCB): scale uniform in scale_range
    (relative to the scan), mass from the row's nominal mass x scale^3, kind "registry", noun from MESH_NOUNS
    (fallback: category). Rendered/collided by soarm.asset_mj (attrs["scale"] = mesh scale factor)."""
    friction = float(rng.uniform(*row["friction_range"]))
    if row["shape"] == "mesh":
        s = float(rng.uniform(*row["scale_range"]))
        size = tuple(float(v) * s for v in row["scan_dims_m"])
        mass = float(np.clip(row["mass_kg"] * (s / row["scale"]) ** 3, *row["mass_range_kg"]))
        color = next((c for c in row["colours"] if c in COLORS), "gray")
        noun, group = MESH_NOUNS.get(row["id"], (row["category"], row["category"]))
        return ObjSpec(name=name, kind="registry", color=color, size=size, mass=mass,
                       attrs={"asset": row["id"], "noun": noun, "group": group, "category": row["category"],
                              "display_name": row["name"], "friction": friction, "scale": s,
                              "colours": list(row["colours"]), "split": row["split"]})
    if row["shape"] != "cube":
        raise NotImplementedError(f"asset_spec: shape {row['shape']!r}")
    j = float(row.get("scale_jitter", 0.0))
    scale = float(rng.uniform(1 - j, 1 + j))
    size = tuple(float(v) * scale for v in row["dims_m"])
    density = float(rng.uniform(*row["density_range_kg_m3"]))
    mass = float(np.clip(density * float(np.prod(size)), *row["mass_range_kg"]))
    return ObjSpec(name=name, kind="cube", color=row["color"], size=size, mass=mass,
                   attrs={"asset": row["id"], "size_class": row["size_class"], "color": row["color"],
                          "texture": row["texture"], "density": mass / float(np.prod(size)), "friction": friction,
                          "letter": row.get("letter"), "material": row["material"], "colours": list(row["colours"]),
                          "scale": scale, "split": row["split"]})


# GSO/YCB meshes that passed the physics gate: id -> (referring noun, confusion group). The noun is what language,
# describe() and the SAM3/Molmo prompts use, so it is a short concrete phrase (checked on renders, not the scan title),
# never the vague category ("toy", "object") nor a primitive kind's noun ("cube", "block", ...). Objects of one group
# (all dinosaurs, all vehicles, lego and the foam brick, ...) would be confused by a noun prompt, so a scene holds at
# most one mesh per group (mesh_rows). Gate-passed rows not listed are never sampled: the ABC block, the coloured wood
# block and the moisturizer (a plain white box) are cubes in all but name; the bird rattle moves while settling upright;
# the dancing alligator, dollhouse swing and N64 cartridge failed the taskgen scripted lift (<= 4/8 layouts, 11-12).
MESH_NOUNS: dict[str, tuple[str, str]] = {
    # train
    "gso_schleich_allosaurus": ("toy dinosaur", "dinosaur"),
    "gso_great_dinos_triceratops_toy": ("android figure", "figure"),     # the scan is a green Android robot figure
    "gso_schleich_african_black_rhino": ("toy rhino", "animal"),
    "gso_schleich_hereford_bull": ("toy bull", "animal"),
    "gso_toysmith_windem_up_flippin_animals_dog": ("toy dog", "animal"),
    "gso_my_first_wiggle_crocodile": ("toy crocodile", "reptile"),
    "gso_snail_measuring_tape": ("snail tape measure", "snail"),
    "gso_ladybug_bead": ("toy ladybug", "ladybug"),
    "gso_whale_whistle_6pcs_set": ("whale whistle", "whale"),
    "gso_magnifying_glassassrt": ("magnifying glass", "magnifier"),
    "gso_grandmother": ("toy grandmother", "doll"),
    "gso_gopro_hero3_composite_cable": ("cable", "cable"),
    "gso_nickelodeon_the_spongebob_movie_popapart_spongebob": ("spongebob toy", "spongebob"),
    "gso_thomas_friends_woodan_railway_henry": ("toy train", "vehicle"),
    "gso_thomas_friends_wooden_railway_porter_5jzrhmm3a9o": ("toy train", "vehicle"),
    "gso_thomas_friends_wooden_railway_talking_thomas_z7yi7ufhjrj": ("toy train", "vehicle"),
    "gso_car_carrier_train": ("toy train", "vehicle"),
    "gso_school_bus": ("toy bus", "vehicle"),
    "gso_fire_engine": ("toy fire truck", "vehicle"),
    "gso_bunny_racer": ("toy car", "vehicle"),
    "gso_chicken_racer": ("toy car", "vehicle"),
    "gso_baby_car": ("toy car", "vehicle"),
    "gso_lalaloopsy_peanut_big_top_tricycle": ("toy doll", "doll"),
    "gso_turboprop_airplane_with_pilot": ("toy airplane", "vehicle"),
    "gso_coast_guard_boat": ("toy boat", "vehicle"),
    "ycb_072_d_toy_airplane": ("toy bolt", "bolt"),
    "ycb_061_foam_brick": ("foam brick", "lego"),
    "ycb_062_dice": ("dice", "dice"),
    "ycb_073_a_lego_duplo": ("lego brick", "lego"),
    "ycb_073_b_lego_duplo": ("lego brick", "lego"),
    "ycb_073_c_lego_duplo": ("lego brick", "lego"),
    "ycb_073_e_lego_duplo": ("lego brick", "lego"),
    "ycb_073_f_lego_duplo": ("lego brick", "lego"),
    # heldout (categories figure, tool)
    "gso_dino_3": ("toy dinosaur", "dinosaur"),
    "gso_dino_4": ("toy dinosaur", "dinosaur"),
    "gso_dino_5": ("toy dinosaur", "dinosaur"),
    "gso_nintendo_yoshi_action_figure": ("yoshi figure", "figure"),
    "gso_android_figure_panda": ("panda figure", "figure"),
    "gso_nickelodeon_teenage_mutant_ninja_turtles_raphael": ("ninja turtle figure", "figure"),
    "gso_modern_doll_family": ("doll family", "doll"),
    "ycb_040_large_marker": ("marker", "marker"),
}
# selector keys of a registry-mesh object (spec "where"); footprint = diagonal of the horizontal bounding box (m) at the
# nominal (largest) scale, to keep objects that must fit a bowl/box/mat small enough
MESH_KEYS = ("category", "noun", "group", "asset", "footprint")


def mesh_rows(split: str = "train", where: dict | None = None, scene: list[ObjSpec] = ()) -> list[dict]:
    """Gate-passed curated mesh rows of `split` that may join `scene`: group and noun not yet in the scene (so every
    mesh has a unique noun and SAM3/Molmo noun prompts are unambiguous) and matching `where` (keys MESH_KEYS; a value,
    a list of allowed values or {"not": [...], "min": x, "max": y})."""
    groups = {o.attrs.get("group") for o in scene if o.kind == "registry"}
    nouns = {o.noun for o in scene}
    out = []
    for r in select_assets(shape="mesh", split=split):
        if r["id"] not in MESH_NOUNS:
            continue
        noun, group = MESH_NOUNS[r["id"]]
        if group in groups or noun in nouns:
            continue
        have = {"category": r["category"], "noun": noun, "group": group, "asset": r["id"],
                "footprint": float(np.hypot(*r["dims_m"][:2]))}
        ok = True
        for key, cons in (where or {}).items():
            if key not in MESH_KEYS:
                raise ValueError(f"registry object selector: unknown key {key!r} (allowed {MESH_KEYS})")
            v = have[key]
            if isinstance(cons, dict):
                ok = (v not in cons.get("not", ()) and v >= cons.get("min", v) and v <= cons.get("max", v))
            else:
                ok = v in (cons if isinstance(cons, list) else [cons])
            if not ok:
                break
        if ok:
            out.append(r)
    return out


def add_mesh(b, where: dict | None, split: str = "train") -> ObjSpec:
    """Append one registry mesh object `mesh_<k>` to builder `b` (soarm.tasks.Builder): asset_spec of a row of
    mesh_rows(split, where, b.objects), drawn uniformly over noun groups, then over the group's rows (so the 5 lego
    bricks or 12 vehicles do not dominate). No qualifying row -> tasks.Reject (the sampler retries)."""
    from .tasks import Reject
    rows = mesh_rows(split, where, b.objects)
    if not rows:
        raise Reject("no registry mesh matches the selector")
    groups = sorted({MESH_NOUNS[r["id"]][1] for r in rows})
    group = groups[int(b.rng.integers(len(groups)))]
    rows = [r for r in rows if MESH_NOUNS[r["id"]][1] == group]
    k = b._count.get("registry", 0)
    b._count["registry"] = k + 1
    spec = asset_spec(b.rng, rows[int(b.rng.integers(len(rows)))], f"mesh_{k}")
    b.objects.append(spec)
    return spec

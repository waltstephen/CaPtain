"""Scene appearance randomisation (table/floor/wall textures, lights, agentview extrinsics) and height platforms.

Applied to the MjSpec that scene.build_scene returns (it only edits the table/floor materials, the `key` light and the
`agentview` camera, and adds a wall, fill lights and platforms), so the default scene stays byte-identical when
SceneConfig asks for nothing:

    app = sample_appearance(rng, table_textures="all", lights="random", camera_jitter_m=0.08, camera_rot_jitter_deg=10)
    apply_appearance(spec, app)

Textures: assets/textures.jsonl (assets/fetch_textures.py, ambientCG CC0), images under asset_root().
Camera: position jitter per axis, rotation jitter about the camera's own axes, fovy 64-74 deg when jittered; a sample
is kept only if the whole reachable workspace (annulus r 0.13-0.29 m, |angle| <= 70 deg, z 0..6 cm) projects inside the
image with a 3 % margin, so the agentview always sees every object the task can place.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

from .objects import asset_root
from .scene import AGENTVIEW_FOVY, AGENTVIEW_POS, AGENTVIEW_TARGET, TABLE_TONES, look_at_quat

TEXTURES_PATH = Path(__file__).resolve().parent.parent / "assets" / "textures.jsonl"
WORKSPACE_R, WORKSPACE_ANG, WORKSPACE_Z = (0.13, 0.29), np.deg2rad(70.0), (0.0, 0.06)
VIEW_MARGIN = 0.03


@lru_cache(maxsize=1)
def textures() -> tuple[dict, ...]:
    if not TEXTURES_PATH.exists():
        return ()
    return tuple(json.loads(line) for line in TEXTURES_PATH.read_text().splitlines() if line.strip())


@dataclass
class Appearance:
    table_texture: str | None = None          # textures.jsonl id; None = plain tone
    table_rgb: tuple = TABLE_TONES[0]         # plain tone, or the tint multiplied into the texture
    floor_texture: str | None = None
    wall_texture: str | None = None           # back wall behind the robot (None = no wall)
    lights: list[dict] = field(default_factory=list)   # extra lights: {dir, diffuse(rgb), castshadow}
    key_dir: np.ndarray | None = None         # None = keep the scene's key light
    key_diffuse: tuple | None = None
    ambient: float | None = None
    cam_pos: np.ndarray | None = None         # None = keep the scene's agentview camera
    cam_quat: list | None = None
    cam_fovy: float | None = None
    view: str | None = None                   # camera view style (VIEW_STYLES) when SceneConfig.camera picks one
    table: dict | None = None                 # edge mount: back (x of the back edge), depth, width, y, thick, floor_z
    wall_x: float | None = None               # x of the backdrop wall (edge mount: behind the table and the camera)

    def to_json(self) -> dict:
        out = {}
        for k, v in self.__dict__.items():
            out[k] = v.tolist() if isinstance(v, np.ndarray) else v
        return out


# ---------------------------------------------------------------------------------------------------- camera
def _rot(axis: int, a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    i, j = [(1, 2), (0, 2), (0, 1)][axis]
    r = np.eye(3)
    r[i, i], r[i, j], r[j, i], r[j, j] = c, -s, s, c
    return r


def _quat_to_mat(q) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


def workspace_points() -> np.ndarray:
    r = np.linspace(*WORKSPACE_R, 3)
    a = np.linspace(-WORKSPACE_ANG, WORKSPACE_ANG, 7)
    rr, aa, zz = np.meshgrid(r, a, WORKSPACE_Z, indexing="ij")
    return np.stack([rr * np.cos(aa), rr * np.sin(aa), zz], -1).reshape(-1, 3)


def sees_workspace(pos, quat, fovy: float, margin: float = VIEW_MARGIN) -> bool:
    """All workspace points inside the (square) image of a MuJoCo camera (x right, y up, looking along -z)."""
    pc = (workspace_points() - np.asarray(pos)) @ _quat_to_mat(quat)
    depth = -pc[:, 2]
    if (depth <= 0.02).any():
        return False
    t = np.tan(np.deg2rad(fovy) / 2) * (1 - 2 * margin)
    return bool((np.abs(pc[:, 0] / depth) <= t).all() and (np.abs(pc[:, 1] / depth) <= t).all())


def sample_camera(rng: np.random.Generator, jitter_m: float, rot_deg: float, tries: int = 200):
    """(pos, quat, fovy) of a jittered agentview that still sees the workspace; returns the unjittered camera if no
    sample out of `tries` does (counted by the caller via `jittered`)."""
    base_q = look_at_quat(AGENTVIEW_POS, AGENTVIEW_TARGET)
    for _ in range(tries):
        pos = AGENTVIEW_POS + rng.uniform(-jitter_m, jitter_m, 3)
        rot = _quat_to_mat(look_at_quat(pos, AGENTVIEW_TARGET))
        if rot_deg > 0:
            pitch, yaw, roll = np.deg2rad(rng.uniform(-rot_deg, rot_deg, 3))
            rot = rot @ _rot(0, pitch) @ _rot(1, yaw) @ _rot(2, roll)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, rot.reshape(-1))
        fovy = float(rng.uniform(64.0, 74.0)) if (jitter_m > 0.02 or rot_deg > 0) else AGENTVIEW_FOVY
        if sees_workspace(pos, q, fovy):
            return pos, q.tolist(), fovy, True
    return AGENTVIEW_POS.copy(), list(base_q), AGENTVIEW_FOVY, False


# View styles of the agentview (SceneConfig.camera): spherical pose around a jittered workspace centre. az = degrees
# from +x (0 = in front of the robot, facing it; the sign is drawn), el = elevation above the table, dist = metres to
# the look-at point. fovy is fitted so the whole workspace (WORKSPACE_*) is in view, plus 2-12 deg of slack.
# TRAIN and HELD-OUT ranges are disjoint (held-out: |az| > 130 or el > 75); held-out views are for eval only.
VIEW_STYLES = {
    "front": {"az": (0, 20), "el": (35, 55), "dist": (0.50, 0.70)},
    "front_high": {"az": (0, 25), "el": (55, 72), "dist": (0.45, 0.65)},
    "oblique": {"az": (25, 60), "el": (30, 55), "dist": (0.50, 0.75)},
    "side": {"az": (60, 100), "el": (30, 55), "dist": (0.50, 0.75)},
    "low": {"az": (0, 45), "el": (18, 30), "dist": (0.55, 0.80)},
    "rear": {"az": (130, 180), "el": (40, 65), "dist": (0.45, 0.65)},
    "overhead": {"az": (0, 180), "el": (75, 88), "dist": (0.55, 0.75)},
}
VIEW_SETS = {"train": {"front": 0.30, "front_high": 0.15, "oblique": 0.25, "side": 0.15, "low": 0.15},
             "heldout": {"rear": 0.5, "overhead": 0.5}}
VIEW_FOVY = (40.0, 80.0)


def fit_fovy(pos, quat, margin: float = VIEW_MARGIN) -> float | None:
    """Smallest square-image fovy (deg) that shows every workspace point with `margin`; None if a point is behind."""
    pc = (workspace_points() - np.asarray(pos)) @ _quat_to_mat(quat)
    depth = -pc[:, 2]
    if (depth <= 0.05).any():
        return None
    t = float(np.max(np.abs(pc[:, :2]) / depth[:, None])) / (1 - 2 * margin)
    return float(np.rad2deg(2 * np.arctan(t)))


def sample_view(rng: np.random.Generator, views: str, tries: int = 200):
    """(pos, quat, fovy, style) of an agentview drawn from a view set (VIEW_SETS key) or one VIEW_STYLES name; every
    returned camera sees the whole workspace (sees_workspace)."""
    weights = VIEW_SETS.get(views) or ({views: 1.0} if views in VIEW_STYLES else None)
    if weights is None:
        raise ValueError(f"camera={views!r}: one of {sorted(VIEW_SETS)} or {sorted(VIEW_STYLES)}")
    names = sorted(weights)
    p = np.array([weights[n] for n in names], float)
    style = names[int(rng.choice(len(names), p=p / p.sum()))]
    s = VIEW_STYLES[style]
    for _ in range(tries):
        target = np.array([rng.uniform(0.17, 0.23), rng.uniform(-0.04, 0.04), rng.uniform(0.0, 0.03)])
        az = np.deg2rad(rng.uniform(*s["az"])) * (1 if rng.random() < 0.5 else -1)
        el = np.deg2rad(rng.uniform(*s["el"]))
        pos = target + rng.uniform(*s["dist"]) * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        rot = _quat_to_mat(look_at_quat(pos, target)) @ _rot(2, np.deg2rad(rng.uniform(-5, 5)))
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, rot.reshape(-1))
        need = fit_fovy(pos, q)
        if need is None or need > VIEW_FOVY[1] - 2:
            continue
        fovy = float(np.clip(need + rng.uniform(2, 12), max(VIEW_FOVY[0], need + 1), VIEW_FOVY[1]))
        if sees_workspace(pos, q, fovy):
            return pos, q.tolist(), fovy, style
    raise RuntimeError(f"no {style} view sees the workspace in {tries} draws")


# Edge-mount table ranges: back edge x, depth, width, base offset along the edge (|y| cap), top thickness, floor below
# the top. div3 widens every range; its table still covers the reach (x up to 0.40 m in front, |y| <= 0.33 m).
TABLE_RANGES = {
    "edge": {"back": (0.06, 0.10), "depth": (0.55, 0.85), "width": (0.75, 1.40), "cover_y": 0.36, "y_max": 0.2,
             "thick": (0.025, 0.045), "floor": (0.65, 0.80)},
    "edge3": {"back": (0.05, 0.11), "depth": (0.50, 0.95), "width": (0.70, 1.50), "cover_y": 0.33, "y_max": 0.25,
              "thick": (0.02, 0.05), "floor": (0.58, 0.92)},
}


# SceneConfig.lights: key light elevation (deg), colour-temperature shift, key intensity, ambient, max fill lights and
# their intensity. "random" = div2, "random3" = div3 (darker/brighter, lower key light, one more fill light).
LIGHT_RANGES = {
    "random": {"key_el": (35, 85), "warm": 0.12, "key": (0.35, 0.9), "ambient": (0.12, 0.45), "fills": 2,
               "fill": (0.08, 0.3)},
    "random3": {"key_el": (25, 88), "warm": 0.18, "key": (0.25, 1.0), "ambient": (0.08, 0.5), "fills": 3,
                "fill": (0.05, 0.35)},
}


def sample_table(rng: np.random.Generator, mount: str = "edge") -> dict:
    """Edge mount: the arm is clamped near the table's back edge; the table extends in front and to both sides and
    always covers the reach (TABLE_RANGES[mount]: |y| <= cover_y m, x up to back + depth)."""
    t = TABLE_RANGES[mount]
    back = -float(rng.uniform(*t["back"]))
    depth = float(rng.uniform(*t["depth"]))
    width = float(rng.uniform(*t["width"]))
    slack = max(0.0, width / 2 - t["cover_y"])
    return {"back": back, "depth": depth, "width": width,
            "y": float(rng.uniform(-min(slack, t["y_max"]), min(slack, t["y_max"]))),
            "thick": float(rng.uniform(*t["thick"])), "floor_z": -float(rng.uniform(*t["floor"]))}

# ---------------------------------------------------------------------------------------------------- sampling
def _pick_texture(rng, use: str, allowed) -> str | None:
    pool = [t["id"] for t in textures() if t["use"] == use and (allowed == "all" or t["id"] in allowed)]
    return str(rng.choice(pool)) if pool else None


def sample_appearance(rng: np.random.Generator, *, table_textures="plain", lights: str = "default",
                      camera_jitter_m: float = 0.015, camera_rot_jitter_deg: float = 0.0,
                      background: bool | None = None, table_mount: str = "center", views: str = "jitter") -> Appearance:
    """table_textures: "plain" (tones only), "all", or a list of texture ids (plain tones stay at 25 %).
    lights: "default" (scene key light) | "random" (key direction/colour/intensity + 0-2 fill lights + ambient) |
    "random3" (div3: wider ranges, 0-3 fill lights; LIGHT_RANGES).
    table_mount "edge"/"edge3": random table around a base at its back edge (TABLE_RANGES); views != "jitter":
    sample_view (jitter ignored)."""
    app = Appearance(table_rgb=TABLE_TONES[int(rng.integers(len(TABLE_TONES)))])
    if table_textures != "plain" and rng.random() > 0.25:
        app.table_texture = _pick_texture(rng, "table", table_textures)
        app.table_rgb = tuple(float(v) for v in rng.uniform(0.8, 1.0, 3))      # tint
    if (table_textures != "plain") if background is None else background:
        app.floor_texture = _pick_texture(rng, "floor", "all")
        app.wall_texture = _pick_texture(rng, "wall", "all") if rng.random() < 0.7 else None
    if lights in LIGHT_RANGES:
        L = LIGHT_RANGES[lights]
        el, az = np.deg2rad(rng.uniform(*L["key_el"])), rng.uniform(-np.pi, np.pi)
        d = -np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        warm = rng.uniform(-L["warm"], L["warm"])                                # colour temperature
        tint = np.array([1 + warm, 1.0, 1 - warm])
        app.key_dir, app.key_diffuse = d, tuple((rng.uniform(*L["key"]) * tint).tolist())
        app.ambient = float(rng.uniform(*L["ambient"]))
        for _ in range(int(rng.integers(0, L["fills"] + 1))):
            el, az = np.deg2rad(rng.uniform(20, 70)), rng.uniform(-np.pi, np.pi)
            fd = -np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
            app.lights.append({"dir": fd.tolist(), "diffuse": [float(rng.uniform(*L["fill"]))] * 3, "castshadow": False})
    elif lights != "default":
        raise ValueError(f"lights={lights!r}")
    if views != "jitter":
        app.cam_pos, app.cam_quat, app.cam_fovy, app.view = sample_view(rng, views)
    elif camera_jitter_m > 0 or camera_rot_jitter_deg > 0:
        app.cam_pos, app.cam_quat, app.cam_fovy, _ = sample_camera(rng, camera_jitter_m, camera_rot_jitter_deg)
    if table_mount in TABLE_RANGES:
        app.table = sample_table(rng, table_mount)
        if app.wall_texture:      # behind the table's back edge and behind the camera
            x = app.table["back"] - float(rng.uniform(0.15, 0.6))
            if app.cam_pos is not None:
                x = min(x, float(app.cam_pos[0]) - 0.25)
            app.wall_x = x
    elif table_mount != "center":
        raise ValueError(f"table_mount={table_mount!r}")
    return app


# ---------------------------------------------------------------------------------------------------- apply
def _texture_material(spec: mujoco.MjSpec, name: str, tex_id: str, rgb, repeat_per_m: float) -> str:
    row = next(t for t in textures() if t["id"] == tex_id)
    tex = spec.add_texture(name=f"tex:{name}", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(asset_root() / row["file"]))
    tex.colorspace = mujoco.mjtColorSpace.mjCOLORSPACE_LINEAR
    mat = spec.add_material(name=f"mat:{name}:{tex_id}")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    mat.texuniform = True
    mat.texrepeat = [repeat_per_m, repeat_per_m]
    mat.rgba = [*rgb, 1.0]
    mat.specular = 0.1
    return mat.name


def _repeat(tex_id: str, default_tile_m: float) -> float:
    tile = next(t for t in textures() if t["id"] == tex_id).get("tile_m") or default_tile_m
    return 1.0 / float(np.clip(tile, 0.2, 2.0))


def apply_appearance(spec: mujoco.MjSpec, app: Appearance) -> None:
    table = spec.geom("table")
    if app.table_texture:
        table.material = _texture_material(spec, "table", app.table_texture, app.table_rgb,
                                           _repeat(app.table_texture, 0.6))
    else:
        spec.material("table").rgba = [*app.table_rgb, 1.0]
    if app.floor_texture:
        spec.geom("floor").material = _texture_material(spec, "floor", app.floor_texture, (0.8, 0.8, 0.8),
                                                        _repeat(app.floor_texture, 1.0))
    if app.wall_texture:
        wall = spec.worldbody.add_geom(name="backdrop_wall", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 1.6, 1.0],
                                       pos=[-0.75 if app.wall_x is None else app.wall_x, 0, 0.1])
        wall.material = _texture_material(spec, "wall", app.wall_texture, (0.85, 0.85, 0.85),
                                          _repeat(app.wall_texture, 1.0))
        wall.contype = wall.conaffinity = 0
    if app.table is not None:      # edge mount: resize/move the table top (top stays at z = 0), legs, floor
        t = app.table
        table.size = [t["depth"] / 2, t["width"] / 2, t["thick"] / 2]
        table.pos = [t["back"] + t["depth"] / 2, t["y"], -t["thick"] / 2]
        floor_z = t["floor_z"]
        spec.geom("floor").pos = [0, 0, floor_z]
        leg_h = -t["thick"] + -floor_z
        for k, (sx, sy) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            x = t["back"] + 0.04 + sx * (t["depth"] - 0.08)
            y = t["y"] - t["width"] / 2 + 0.04 + sy * (t["width"] - 0.08)
            leg = spec.worldbody.add_geom(name=f"table_leg_{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                          size=[0.02, 0.02, leg_h / 2], pos=[x, y, -t["thick"] - leg_h / 2],
                                          material=table.material)
            leg.contype = leg.conaffinity = 0
    key = spec.light("key")
    if app.key_dir is not None:
        key.dir = np.asarray(app.key_dir).tolist()
        key.diffuse = list(app.key_diffuse)
    if app.ambient is not None:
        spec.visual.headlight.ambient = [app.ambient] * 3
    for k, li in enumerate(app.lights):
        light = spec.worldbody.add_light(name=f"fill_{k}", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                                         pos=[0, 0, 1.5], dir=li["dir"])
        light.diffuse = li["diffuse"]
        light.castshadow = li["castshadow"]
    if app.cam_pos is not None:
        cam = spec.camera("agentview")
        cam.pos = np.asarray(app.cam_pos).tolist()
        cam.quat = list(app.cam_quat)
        cam.fovy = app.cam_fovy


# ---------------------------------------------------------------------------------------------------- platforms
@dataclass
class Platform:
    name: str
    xy: tuple[float, float]
    size: tuple[float, float, float]        # full extents; top at z = size[2]
    yaw: float
    rgb: tuple


def sample_platforms(rng: np.random.Generator, n: int, placed: list[tuple[np.ndarray, float]] | None = None,
                     height=(0.01, 0.04), side=(0.06, 0.10), gap: float = 0.035) -> list[Platform]:
    """Up to n static boxes (height variation) inside the reachable annulus, clear of `placed` footprints (xy, radius).
    Heights <= 4 cm: a cube on top keeps the grip point inside the top-down reach (~12 cm)."""
    placed = list(placed or [])
    out = []
    for k in range(n):
        for _ in range(200):
            sx, sy = rng.uniform(*side, 2)
            rad = 0.5 * float(np.hypot(sx, sy))
            r, a = rng.uniform(0.15, 0.26), rng.uniform(-np.deg2rad(60), np.deg2rad(60))
            xy = np.array([r * np.cos(a), r * np.sin(a)])
            if all(np.linalg.norm(xy - p) >= rad + pr + gap for p, pr in placed):
                break
        else:
            break
        tone = TABLE_TONES[int(rng.integers(len(TABLE_TONES)))]
        rgb = tuple(float(v) for v in np.clip(np.asarray(tone) * rng.uniform(0.7, 1.2), 0, 1))
        out.append(Platform(f"platform_{k}", (float(xy[0]), float(xy[1])), (float(sx), float(sy),
                            float(rng.uniform(*height))), float(rng.uniform(-np.pi, np.pi)), rgb))
        placed.append((xy, rad))
    return out


def add_platforms(spec: mujoco.MjSpec, platforms: list[Platform], friction: float = 0.8) -> None:
    for p in platforms:
        mat = spec.add_material(name=f"mat:{p.name}")
        mat.rgba = [*p.rgb, 1.0]
        mat.specular = 0.1
        g = spec.worldbody.add_geom(name=p.name, type=mujoco.mjtGeom.mjGEOM_BOX,
                                    size=[p.size[0] / 2, p.size[1] / 2, p.size[2] / 2],
                                    pos=[p.xy[0], p.xy[1], p.size[2] / 2],
                                    quat=[np.cos(p.yaw / 2), 0, 0, np.sin(p.yaw / 2)], material=mat.name)
        g.friction = [friction, 0.005, 0.0001]

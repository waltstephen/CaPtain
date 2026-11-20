"""Physics scene of one task: Menagerie SO-101 + table + task objects + cameras + randomised lights/materials.

World frame = robot base frame (x forward, y left, z up); table top at z = 0. Object geoms are named `obj:<name>:<k>`
so a segmentation render maps pixels back to task objects.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .objects import COLORS, FLOOR, WALL, ObjSpec
from .predicates import TABLE_CENTER_X, TABLE_HALF
from .tasks import SceneConfig, TaskInstance

MENAGERIE_COMMIT = "98a70c569c4c1b6da467a20012f766ee1f8dc521"
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
CAMERAS = ("agentview", "robot0_eye_in_hand")
# agentview: in front of the robot, facing it and looking down over the workspace -- LIBERO's agentview geometry, so
# TASK.md's LIBERO relation line holds unchanged (front = +x toward the camera, right = +y in the camera image).
AGENTVIEW_POS = np.array([0.46, 0.0, 0.36])
AGENTVIEW_TARGET = np.array([0.11, 0.0, 0.0])
AGENTVIEW_FOVY = 68.0          # 58 deg clipped objects at the workspace's lateral edges (r 0.29 m, 70 deg)
TABLE_TONES = [(0.62, 0.48, 0.34), (0.75, 0.72, 0.68), (0.40, 0.42, 0.45), (0.85, 0.83, 0.78), (0.55, 0.40, 0.28)]


def menagerie_dir() -> Path:
    path = Path(os.environ.get("SOARM_MENAGERIE", "")).expanduser()
    if not (path / "so101.xml").is_file():
        raise FileNotFoundError("set SOARM_MENAGERIE to mujoco_menagerie/robotstudio_so101 "
                                f"(commit {MENAGERIE_COMMIT}; robots/soarm/fetch_assets.sh)")
    return path


def look_at_quat(pos: np.ndarray, target: np.ndarray) -> list[float]:
    """MuJoCo camera quaternion (wxyz) looking from pos at target with world z up (camera looks along its -z)."""
    f = (target - pos) / np.linalg.norm(target - pos)
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    rot = np.stack([right, up, -f], axis=1)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, rot.reshape(-1))
    return q.tolist()


@dataclass
class Randomization:
    light_dir: np.ndarray
    light_diffuse: float
    ambient: float
    table_rgb: tuple
    cam_jitter: np.ndarray       # xyz metres
    friction: float
    appearance: object = None    # scene_random.Appearance (non-default SceneConfig scene fields), applied last

    @staticmethod
    def sample(rng: np.random.Generator, enabled: bool = True, config: SceneConfig | None = None) -> "Randomization":
        """Legacy draws first (default SceneConfig reproduces the pre-config scenes bit for bit), then the
        scene_random appearance (textured table/floor/wall, random lights, camera shift) when the config asks for it."""
        config = config or SceneConfig()
        if not enabled:
            return Randomization(np.array([0.2, 0.1, -1.0]), 0.65, 0.35, TABLE_TONES[0], np.zeros(3), 0.7)
        d = np.array([rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), -1.0])
        r = Randomization(d / np.linalg.norm(d), float(rng.uniform(0.45, 0.85)), float(rng.uniform(0.2, 0.45)),
                          TABLE_TONES[int(rng.integers(len(TABLE_TONES)))], rng.uniform(-0.015, 0.015, 3),
                          float(rng.uniform(0.5, 0.9)))
        if config.scene_is_default():
            return r
        from .scene_random import sample_appearance
        camera = config.camera_jitter_m != SceneConfig.camera_jitter_m or config.camera_rot_jitter_deg > 0
        r.appearance = sample_appearance(rng, table_textures=config.table_textures, lights=config.lights,
                                         camera_jitter_m=config.camera_jitter_m if camera else 0.0,
                                         camera_rot_jitter_deg=config.camera_rot_jitter_deg,
                                         table_mount=config.table_mount, views=config.camera)
        return r


def _material(spec: mujoco.MjSpec, name: str, rgb, specular: float = 0.2) -> str:
    m = spec.add_material(name=name)
    m.rgba = [*rgb, 1.0]
    m.specular = specular
    return name


def _add_object(spec: mujoco.MjSpec, o: ObjSpec, friction: float) -> None:
    """One task object as a body named `o.name` (free joint unless it is a flat mat). Registry objects (cube domain,
    attrs["asset"]) are built by asset_mj: textured mesh + box collision with their own friction."""
    if o.attrs.get("asset"):
        from . import asset_mj
        asset_mj.add_object(spec, o, friction)
        return
    mat = _material(spec, f"mat:{o.name}", COLORS[o.color])
    body = spec.worldbody.add_body(name=o.name)
    if o.kind != "pad":
        body.add_freejoint(name=f"{o.name}:free")
    sx, sy, sz = o.size
    G = mujoco.mjtGeom
    geoms: list[tuple] = []           # (type, size, pos, quat)
    if o.kind in ("cube", "block", "tile"):
        geoms.append((G.mjGEOM_BOX, [sx / 2, sy / 2, sz / 2], [0, 0, 0], [1, 0, 0, 0]))
    elif o.kind == "cylinder":
        geoms.append((G.mjGEOM_CYLINDER, [sx / 2, sz / 2, 0], [0, 0, 0], [1, 0, 0, 0]))
    elif o.kind == "ball":
        geoms.append((G.mjGEOM_SPHERE, [sx / 2, 0, 0], [0, 0, 0], [1, 0, 0, 0]))
    elif o.kind == "pad":
        geoms.append((G.mjGEOM_BOX, [sx / 2, sy / 2, sz / 2], [0, 0, 0], [1, 0, 0, 0]))
    elif o.kind == "box":
        geoms.append((G.mjGEOM_BOX, [sx / 2, sy / 2, FLOOR / 2], [0, 0, -sz / 2 + FLOOR / 2], [1, 0, 0, 0]))
        for sgn in (-1, 1):
            geoms.append((G.mjGEOM_BOX, [WALL / 2, sy / 2, sz / 2], [sgn * (sx / 2 - WALL / 2), 0, 0], [1, 0, 0, 0]))
            geoms.append((G.mjGEOM_BOX, [sx / 2, WALL / 2, sz / 2], [0, sgn * (sy / 2 - WALL / 2), 0], [1, 0, 0, 0]))
    elif o.kind in ("bowl", "cup"):
        r = sx / 2
        geoms.append((G.mjGEOM_CYLINDER, [r, FLOOR / 2, 0], [0, 0, -sz / 2 + FLOOR / 2], [1, 0, 0, 0]))
        n = 20 if o.kind == "bowl" else 16
        seg = 2 * np.pi * r / n * 0.55
        for k in range(n):
            a = 2 * np.pi * k / n
            pos = [(r - WALL / 2) * np.cos(a), (r - WALL / 2) * np.sin(a), 0]
            geoms.append((G.mjGEOM_BOX, [WALL / 2, seg, sz / 2], pos, [np.cos(a / 2), 0, 0, np.sin(a / 2)]))
    else:
        raise ValueError(o.kind)
    total = sum(1 for _ in geoms)
    for k, (gtype, size, pos, quat) in enumerate(geoms):
        g = body.add_geom(name=f"obj:{o.name}:{k}", type=gtype, size=size, pos=pos, quat=quat, material=mat)
        # balls: rolling friction (condim 6) -- without it a released ball rolled off the table and kept going (pilot)
        g.friction = [friction, 0.005, 0.003 if o.kind == "ball" else 0.0001]
        g.condim = 6 if o.kind == "ball" else 3
        g.solref = [0.01, 1.0]
        if o.kind != "pad":
            g.mass = o.mass / total


def build_scene(task: TaskInstance, rand: Randomization) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(menagerie_dir() / "so101.xml"))
    spec.modelname = f"soarm_{task.family}_{task.seed}"
    spec.visual.global_.offwidth = 640
    spec.visual.global_.offheight = 640
    spec.visual.headlight.ambient = [rand.ambient] * 3
    spec.visual.headlight.diffuse = [0.25] * 3
    spec.visual.headlight.specular = [0.0] * 3
    wb = spec.worldbody
    light = wb.add_light(name="key", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=[0, 0, 1.5],
                         dir=rand.light_dir.tolist())
    light.diffuse = [rand.light_diffuse] * 3
    light.castshadow = True
    table_mat = _material(spec, "table", rand.table_rgb, specular=0.1)
    table = wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[TABLE_HALF[0], TABLE_HALF[1], 0.02],
                        pos=[TABLE_CENTER_X, 0, -0.02], material=table_mat)
    table.friction = [rand.friction, 0.005, 0.0001]
    floor_mat = _material(spec, "floor", (0.25, 0.25, 0.27))
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], pos=[0, 0, -0.75], material=floor_mat)
    pos = AGENTVIEW_POS + rand.cam_jitter
    wb.add_camera(name="agentview", pos=pos.tolist(), quat=look_at_quat(pos, AGENTVIEW_TARGET), fovy=AGENTVIEW_FOVY)
    spec.camera("wrist_cam").name = "robot0_eye_in_hand"
    by_name = {o.name: o for o in task.objects}
    init = object_qpos(task, by_name)
    for o in task.objects:
        _add_object(spec, o, rand.friction)
        body = spec.body(o.name)
        body.pos = init[o.name][0].tolist()          # body pose = qpos0 of its free joint (pads: fixed)
        body.quat = init[o.name][1].tolist()
    if task.platforms:
        from .scene_random import Platform, add_platforms
        add_platforms(spec, [Platform(**p) for p in task.platforms], rand.friction)
    if rand.appearance is not None:
        from .scene_random import apply_appearance
        apply_appearance(spec, rand.appearance)
    return spec


def object_qpos(task: TaskInstance, spec_by_name: dict[str, ObjSpec]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Initial (pos, quat wxyz) of every object from the task layout (bottom on the table or on its mat)."""
    out = {}

    def top(name: str) -> float:       # support surface height of `name` (stacks, objects on mats or in containers)
        p, o = task.layout[name], spec_by_name[name]
        below = top(p.on) if p.on is not None else 0.0
        if o.category == "container":
            return below + FLOOR
        return 0.0 + o.size[2] if o.kind == "pad" else below + o.size[2]

    for name, p in task.layout.items():
        o = spec_by_name[name]
        base = 0.0
        if p.on is not None:
            base = top(p.on)
        z = base + 0.5 * o.size[2] + 0.001
        quat = np.array([np.cos(p.yaw / 2), 0, 0, np.sin(p.yaw / 2)])
        if p.lying:
            z = base + 0.5 * o.size[0] + 0.001
            q2 = np.zeros(4)
            mujoco.mju_mulQuat(q2, quat, np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]))
            quat = q2
        if o.kind == "pad":
            z = 0.5 * o.size[2]
        out[name] = (np.array([p.xy[0], p.xy[1], z]), quat)
    return out

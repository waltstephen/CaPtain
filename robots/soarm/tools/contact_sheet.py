"""Contact sheets of random cube-domain scenes (registry cubes + scene_random appearance) for visual review.

    python tools/contact_sheet.py --out $SOARM_ROOT/assets_preview [--n 16] [--level strong|default] [--split train]

Per scene: 1-5 gate-passed cubes of the split (PIVOT domain), a bowl/box with p 0.4, a mat with p 0.3, 0-2 platforms
(one cube may start on top) at level strong; table/floor/wall textures, random lights and camera jitter (+-8 cm,
+-10 deg) at level strong. Renders the agentview after a 0.5 s settle and writes a labelled grid PNG + scenes.jsonl.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from OpenGL import GL  # noqa: F401  (load OSMesa before mujoco)
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from soarm import asset_mj, scene_random
from soarm.objects import asset_spec, select_assets
from soarm.scene import ARM_JOINTS, Randomization, build_scene
from soarm.sim import HOME_Q
from soarm.twin import JAW_OPEN_RAD
from soarm.tasks import Builder, Reject, TaskInstance

LEVELS = {"default": dict(table_textures="plain", lights="default", camera_jitter_m=0.015, camera_rot_jitter_deg=0.0),
          "strong": dict(table_textures="all", lights="random", camera_jitter_m=0.08, camera_rot_jitter_deg=10.0)}


def make_scene(seed: int, level: str, split: str):
    rng = np.random.default_rng([seed, 4242])
    rows = select_assets(category="cube", split=split)
    for _ in range(50):
        b = Builder(rng)
        for k in range(int(rng.integers(1, 6))):
            b.objects.append(asset_spec(rng, rows[int(rng.integers(len(rows)))], f"cube_{k}"))
        if rng.random() < 0.4:
            b.add(str(rng.choice(["bowl", "box"])))
        if rng.random() < 0.3:
            b.add("pad")
        try:
            b.place()
            break
        except Reject:
            continue
    task = TaskInstance("preview", seed, "", b.objects, b.layout, [], {})
    spec = build_scene(task, Randomization.sample(rng))
    app = scene_random.sample_appearance(rng, **LEVELS[level])
    scene_random.apply_appearance(spec, app)
    platforms = []
    if level == "strong":
        placed = [(np.asarray(p.xy), next(o for o in b.objects if o.name == n).footprint_radius)
                  for n, p in b.layout.items()]
        cubes = [o for o in b.objects if o.attrs.get("asset")]
        lift = cubes[0] if len(cubes) > 1 and rng.random() < 0.5 else None
        if lift is not None:              # free its footprint: it moves onto the first platform
            placed = [pl for pl, n in zip(placed, b.layout) if n != lift.name]
        platforms = scene_random.sample_platforms(rng, int(rng.integers(0, 3)) or (1 if lift else 0), placed)
        scene_random.add_platforms(spec, platforms)
        if lift is not None and platforms:
            p = platforms[0]
            body = spec.body(lift.name)
            body.pos = [p.xy[0], p.xy[1], p.size[2] + 0.5 * lift.size[2] + 0.001]
    return task, spec, app, platforms


def render(spec: mujoco.MjSpec, size: int) -> np.ndarray:
    """Agentview after a 0.5 s settle with the arm held at the episode home pose (as SoarmSim.reset)."""
    m = spec.compile()
    d = mujoco.MjData(m)
    arm = [m.jnt_qposadr[m.joint(n).id] for n in ARM_JOINTS]
    d.qpos[arm] = HOME_Q
    d.qpos[m.jnt_qposadr[m.joint("gripper").id]] = JAW_OPEN_RAD
    d.ctrl[:5], d.ctrl[5] = HOME_Q, JAW_OPEN_RAD
    mujoco.mj_forward(m, d)
    for _ in range(int(0.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    r = mujoco.Renderer(m, size, size)
    r.update_scene(d, camera="agentview")
    img = r.render().copy()
    r.close()
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", choices=sorted(LEVELS), default="strong")
    ap.add_argument("--split", default="train")
    ap.add_argument("--size", type=int, default=320)
    a = ap.parse_args()
    asset_mj.install_scene_hook()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tiles, meta = [], []
    for s in range(a.seed, a.seed + a.n):
        task, spec, app, platforms = make_scene(s, a.level, a.split)
        img = Image.fromarray(render(spec, a.size))
        cubes = [o for o in task.objects if o.attrs.get("asset")]
        label = f"{s}: " + " ".join(f"{o.attrs['color']}/{o.attrs['texture'][:2]}{o.attrs['size_class']:g}" for o in cubes)
        ImageDraw.Draw(img).rectangle([0, 0, a.size, 14], fill=(0, 0, 0))
        ImageDraw.Draw(img).text((3, 1), label[:60], fill=(255, 255, 255))
        tiles.append(np.asarray(img))
        meta.append({"seed": s, "level": a.level, "objects": [{"name": o.name, "kind": o.kind, "asset": o.attrs.get("asset"),
                     "size": o.size, "mass": o.mass} for o in task.objects], "appearance": app.to_json(),
                     "platforms": [p.__dict__ for p in platforms]})
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    grid = np.zeros((rows * a.size, cols * a.size, 3), np.uint8)
    for i, t in enumerate(tiles):
        grid[(i // cols) * a.size:(i // cols + 1) * a.size, (i % cols) * a.size:(i % cols + 1) * a.size] = t
    name = f"sheet_{a.level}_{a.split}_{a.seed:04d}"
    Image.fromarray(grid).save(out / f"{name}.png")
    (out / f"{name}.jsonl").write_text("".join(json.dumps(m) + "\n" for m in meta))
    print(out / f"{name}.png")


if __name__ == "__main__":
    main()

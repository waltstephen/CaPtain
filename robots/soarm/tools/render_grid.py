"""Render check of the scene distribution: agentview images of N layouts x camera view styles, as one labelled grid PNG
per call, plus per-style visibility of the task objects (segmentation pixels in the agentview at the home pose).

    python tools/render_grid.py --out DIR --mix div2 --views front,front_high,oblique,side,low,rear,overhead \
        --specs cube_stack_two,cube_sort_by_color,... --seed 1009000 --size 224

Every scene is a real SoarmSim start (settled, start-checked) with SceneConfig.camera set to one style; rows = layouts
(same task sample for every style), columns = styles. An object counts as visible with >= --min-pixels pixels.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from OpenGL import GL  # noqa: F401  (load OSMesa before mujoco)
import numpy as np
from PIL import Image, ImageDraw

from soarm.camera import Cameras
from soarm.sim import SoarmSim
from soarm.tasks import SceneConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mix", default=None, help="take the scene config from soarm/specs/mixes/<name>.json")
    ap.add_argument("--config", default=None, help="SceneConfig JSON (overrides --mix)")
    ap.add_argument("--views", default="front,front_high,oblique,side,low,rear,overhead")
    ap.add_argument("--specs", required=True, help="comma-separated spec ids, one row each (cycled with seeds)")
    ap.add_argument("--rows", type=int, default=0, help="rows (default: one per spec)")
    ap.add_argument("--seed", type=int, default=1_009_000)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--min-pixels", type=int, default=20)
    ap.add_argument("--name", default="grid")
    a = ap.parse_args()
    if a.config:
        config = SceneConfig.from_json(json.loads(a.config))
    elif a.mix:
        from soarm import mix
        config = mix.load(a.mix)["config"]
    else:
        config = SceneConfig()
    views = a.views.split(",")
    specs = a.specs.split(",")
    rows = a.rows or len(specs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s = a.size
    grid = Image.new("RGB", (s * len(views), s * rows + 18), (20, 20, 20))
    draw = ImageDraw.Draw(grid)
    for j, v in enumerate(views):
        draw.text((j * s + 4, 3), v, fill=(255, 255, 255))
    stats = {v: {"scenes": 0, "objects": 0, "visible": 0, "all_visible": 0} for v in views}
    records = []
    for i in range(rows):
        sid, seed = specs[i % len(specs)], a.seed + i
        for j, v in enumerate(views):
            sim = SoarmSim(sid, seed, config=dataclasses.replace(config, camera=v))
            cams = Cameras(sim.model, sim.geom_object, list(sim.specs), size=(s, s))
            f = cams.render(sim.data, "agentview")
            cams.close()
            px = {n: int((f.seg_object == k).sum()) for k, n in enumerate(sim.specs)}
            vis = {n: c >= a.min_pixels for n, c in px.items()}
            st = stats[v]
            st["scenes"] += 1
            st["objects"] += len(vis)
            st["visible"] += sum(vis.values())
            st["all_visible"] += all(vis.values())
            app = sim.rand.appearance
            records.append({"spec": sid, "seed": seed, "view": v, "style": getattr(app, "view", None),
                            "language": sim.task.language, "pixels": px,
                            "cam_pos": None if app is None or app.cam_pos is None else np.asarray(app.cam_pos).tolist(),
                            "fovy": None if app is None else app.cam_fovy,
                            "table": None if app is None else app.table, "platforms": len(sim.task.platforms)})
            tile = Image.fromarray(f.rgb)
            ImageDraw.Draw(tile).text((3, s - 12), f"{sid[5:] if sid.startswith('cube_') else sid} {seed}"[:34],
                                      fill=(255, 255, 0) if all(vis.values()) else (255, 60, 60))
            grid.paste(tile, (j * s, 18 + i * s))
            print(f"{sid} {seed} {v}: {sum(vis.values())}/{len(vis)} visible", flush=True)
    grid.save(out / f"{a.name}.png")
    for v, st in stats.items():
        st["visible_frac"] = round(st["visible"] / max(1, st["objects"]), 3)
        st["all_visible_frac"] = round(st["all_visible"] / max(1, st["scenes"]), 3)
    (out / f"{a.name}.json").write_text(json.dumps({"stats": stats, "scenes": records}, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()

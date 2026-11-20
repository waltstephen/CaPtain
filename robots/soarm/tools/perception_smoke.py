"""Smoke the real perception services on SO-101 rendered frames through the harness's own client.

    python tools/perception_smoke.py --mix div2 --scenes 4 --out /tmp/perc_smoke [--repeat 3]

For each scene (a div2 training layout, agentview + wrist frames), every object is queried by "<colour> <kind>" with
CapXHttpToolServices.segment_text (SAM3 text), .point (Molmo2 "Point at") and .segment_point (SAM3 at Molmo's point),
using the replica URLs of SOARM_SAM3_URLS / SOARM_MOLMO_URLS (all replicas in turn). Host-side MuJoCo segmentation is
used here ONLY to score the answers (best-mask IoU, point inside the object); episodes never render it. Writes
overlay PNGs + smoke.jsonl, prints per-endpoint latency and accuracy.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
    from captain_harness.adapters.capx.services import CapXHttpToolServices, CapXServiceEndpoints
    from soarm import mix as mixes, taskspec
    from soarm.camera import Cameras
    from soarm.perception import LoggedHttpTransport
    from soarm.scene import CAMERAS
    from soarm.sim import SoarmSim

    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default="div2")
    ap.add_argument("--scenes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1_059_000)
    ap.add_argument("--repeat", type=int, default=1, help="query each object this many times (latency sample)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sam3s = [u for u in os.environ.get("SOARM_SAM3_URLS", "http://127.0.0.1:8114").split(",") if u]
    molmos = [u for u in os.environ.get("SOARM_MOLMO_URLS", "http://127.0.0.1:8122/v1").split(",") if u]
    m = mixes.load(a.mix)
    ids = sorted(m["weights"])
    rng = np.random.default_rng(a.seed)
    log = out / "perception.jsonl"
    rows = []
    for k in range(a.scenes):
        sid = ids[int(rng.integers(len(ids)))]
        seed = a.seed + k
        sim = SoarmSim(taskspec.get(sid), seed, config=dataclasses.replace(m["config"]))
        cams = Cameras(sim.model, sim.geom_object, list(sim.specs))
        svc = CapXHttpToolServices(CapXServiceEndpoints(sam3=sam3s[k % len(sam3s)], molmo=molmos[k % len(molmos)]),
                                   transport=LoggedHttpTransport(log), attempts=1, timeout_s=180)
        for cam in CAMERAS:
            f = cams.render(sim.data, cam)
            img = Image.fromarray(f.rgb)
            draw = ImageDraw.Draw(img)
            for i, (name, spec) in enumerate(sim.specs.items()):
                gt = f.seg_object == i
                if gt.sum() < 30:
                    continue
                text = f"{spec.color} {spec.kind}"
                for _ in range(a.repeat):
                    t0 = time.monotonic()
                    masks = svc.segment_text(f.rgb, text)
                    t1 = time.monotonic()
                    pt = svc.point(f.rgb, text)
                    t2 = time.monotonic()
                    iou = max((float((c.mask & gt).sum() / max(1, (c.mask | gt).sum())) for c in masks), default=0.0)
                    top_iou = float((masks[0].mask & gt).sum() / max(1, (masks[0].mask | gt).sum())) if masks else 0.0
                    inside = bool(pt is not None and gt[min(gt.shape[0] - 1, max(0, int(pt[1]))),
                                                        min(gt.shape[1] - 1, max(0, int(pt[0])))])
                    near = None
                    pt_iou = None
                    if pt is not None:
                        ys, xs = np.nonzero(gt)
                        near = float(np.min(np.hypot(xs - pt[0], ys - pt[1])))
                        pm = svc.segment_point(f.rgb, pt)
                        pt_iou = max((float((c.mask & gt).sum() / max(1, (c.mask | gt).sum())) for c in pm), default=0.0)
                    t3 = time.monotonic()
                    rows.append({"scene": k, "spec": sid, "seed": seed, "camera": cam, "object": name, "text": text,
                                 "pixels": int(gt.sum()), "sam3_n": len(masks), "sam3_best_iou": round(iou, 3),
                                 "sam3_top_iou": round(top_iou, 3), "molmo_point": pt, "molmo_inside": inside,
                                 "molmo_px_from_object": near, "sam3_point_iou": pt_iou,
                                 "lat_segment_s": round(t1 - t0, 3), "lat_point_s": round(t2 - t1, 3),
                                 "lat_segment_point_s": round(t3 - t2, 3)})
                if masks:
                    ys, xs = np.nonzero(masks[0].mask)
                    if xs.size:
                        draw.rectangle([xs.min(), ys.min(), xs.max(), ys.max()], outline=(255, 255, 0))
                if pt is not None:
                    draw.ellipse([pt[0] - 4, pt[1] - 4, pt[0] + 4, pt[1] + 4], outline=(255, 0, 255), width=2)
                    draw.text((pt[0] + 6, pt[1] - 6), text, fill=(255, 0, 255))
            img.save(out / f"scene{k}_{sid}_{cam}.png")
        cams.close()
    with (out / "smoke.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    by = {}
    for c in calls:
        ep = c["url"].rsplit("/", 1)[-1]
        by.setdefault(ep, []).append(c["latency_s"])
    summary = {"objects": len(rows),
               "sam3_best_iou_mean": round(float(np.mean([r["sam3_best_iou"] for r in rows])), 3) if rows else None,
               "sam3_top_iou_ge_0.5": round(float(np.mean([r["sam3_top_iou"] >= 0.5 for r in rows])), 3) if rows else None,
               "molmo_inside_rate": round(float(np.mean([r["molmo_inside"] for r in rows])), 3) if rows else None,
               "calls": len(calls), "errors": sum(1 for c in calls if not c["ok"]),
               "privileged_calls": sum(1 for c in calls if c.get("privileged") is not False),
               "latency_s": {ep: {"n": len(v), "p50": round(float(np.median(v)), 3),
                                  "p90": round(float(np.percentile(v, 90)), 3)} for ep, v in by.items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

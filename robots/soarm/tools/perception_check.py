"""Real SAM3 / Molmo2 on twin start frames vs the sim segmentation (soarm-speed realistic-perception check).

    python tools/perception_check.py [--families ...] [--seeds 9000-9004] [--cameras agentview]

For every object, prompt "<colour> <noun>" (the task language's form): Molmo point hit rate (point inside the object's
true mask) and SAM3 segment_text best-mask IoU, with service latencies. Masks are host-side ground truth here only.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    from batch import parse_range
    from captain_harness.adapters.capx import CapXHttpToolServices, CapXServiceEndpoints
    from soarm import taskspec
    from soarm.driver import SoarmDriver
    from soarm.sim import SoarmSim
    from soarm.twin import build_twin_xml

    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="pick_lift,pick_place_mat,put_in_container")
    ap.add_argument("--seeds", default="9000-9004")
    ap.add_argument("--cameras", default="agentview,robot0_eye_in_hand")
    ap.add_argument("--sam3", default="http://127.0.0.1:8114")
    ap.add_argument("--molmo", default="http://127.0.0.1:8122/v1")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    svc = CapXHttpToolServices(CapXServiceEndpoints(sam3=a.sam3, molmo=a.molmo, openeta_revision="x"),
                               attempts=1, timeout_s=120)
    xml = build_twin_xml()
    rows = []
    for family in a.families.split(","):
        for seed in parse_range(a.seeds):
            sim = SoarmSim(taskspec.get(family), seed)
            drv = SoarmDriver(sim, xml, force_seed=seed)
            names = list(sim.specs)
            for cam in a.cameras.split(","):
                f = drv.cameras.render(sim.data, cam)
                rgb, seg = np.ascontiguousarray(f.rgb[::-1]), f.seg_object[::-1]   # image rows as the policy sees them
                for i, name in enumerate(names):
                    gt = seg == i
                    if gt.sum() < 30:
                        continue
                    spec = sim.specs[name]
                    text = f"the {spec.color} {spec.noun}"
                    t0 = time.perf_counter()
                    p = svc.point(rgb, text)
                    tp = time.perf_counter() - t0
                    hit = bool(p is not None and gt[min(int(p[1]), gt.shape[0] - 1), min(int(p[0]), gt.shape[1] - 1)])
                    t0 = time.perf_counter()
                    masks = svc.segment_text(rgb, f"{spec.color} {spec.noun}")
                    ts = time.perf_counter() - t0
                    iou = max((float((m.mask & gt).sum() / max((m.mask | gt).sum(), 1)) for m in masks), default=0.0)
                    rows.append({"family": family, "seed": seed, "camera": cam, "object": text, "px": int(gt.sum()),
                                 "molmo_hit": hit, "molmo_s": round(tp, 3), "sam3_iou": round(iou, 3),
                                 "sam3_s": round(ts, 3)})
                    print(json.dumps(rows[-1]), flush=True)
            drv.close()
    for cam in a.cameras.split(","):
        r = [x for x in rows if x["camera"] == cam]
        if r:
            print(json.dumps({"tag": a.tag, "camera": cam, "n": len(r),
                              "molmo_hit_rate": round(sum(x["molmo_hit"] for x in r) / len(r), 3),
                              "molmo_s_med": st.median(x["molmo_s"] for x in r),
                              "sam3_iou_med": st.median(x["sam3_iou"] for x in r),
                              "sam3_iou>0.5": round(sum(x["sam3_iou"] > 0.5 for x in r) / len(r), 3),
                              "sam3_s_med": st.median(x["sam3_s"] for x in r)}))


if __name__ == "__main__":
    main()

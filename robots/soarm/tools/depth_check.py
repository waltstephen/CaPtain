"""MoGe-2 (+ table-plane scale) vs rendered depth on twin start frames (soarm-speed realistic-perception check).

    python tools/depth_check.py --families pick_lift,pick_place_mat,put_in_container --seeds 9000-9004 [--url ...]

Per camera: median |error| (m) on object pixels and on table pixels, and the error of the object top height
(robot_base z, 95th percentile of the object's back-projected points) vs rendered depth. The object masks come from
the sim segmentation: host-side diagnostic only; nothing here reaches a policy.
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


def base_z(depth: np.ndarray, intr: np.ndarray, ext: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v, u = np.nonzero(mask)
    z = depth[v, u]
    ok = np.isfinite(z) & (z > 0)
    u, v, z = u[ok], v[ok], z[ok]
    pts = np.stack([(u - intr[0, 2]) / intr[0, 0] * z, (v - intr[1, 2]) / intr[1, 1] * z, z, np.ones_like(z)])
    return (ext @ pts)[2]


def main() -> None:
    from batch import parse_range
    from captain_harness.adapters.hardware.mono_depth import MonoDepthSource
    from soarm import taskspec
    from soarm.driver import SoarmDriver
    from soarm.scene import CAMERAS
    from soarm.sim import SoarmSim
    from soarm.twin import build_twin_xml

    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="pick_lift,pick_place_mat,put_in_container")
    ap.add_argument("--seeds", default="9000-9004")
    ap.add_argument("--url", default="http://127.0.0.1:8130")
    a = ap.parse_args()
    mono = MonoDepthSource(a.url, model="moge2", align="plane", support_plane_z_m=0.0)
    xml = build_twin_xml()
    rows = []
    for family in a.families.split(","):
        spec = taskspec.get(family)
        for seed in parse_range(a.seeds):
            sim = SoarmSim(spec, seed)
            drv = SoarmDriver(sim, xml, force_seed=seed)
            for cam in CAMERAS:
                f = drv.cameras.render(sim.data, cam)
                intr = np.asarray(f.intrinsics, float).reshape(3, 3)
                ext = np.asarray(f.optical_in_base, float).reshape(4, 4)
                t0 = time.perf_counter()
                est = mono(camera=cam, rgb=f.rgb, raw_sim_depth=None, intrinsics_3x3=intr, camera_to_base_4x4=ext)
                dt = time.perf_counter() - t0
                obj = f.seg_object >= 0
                table = (~obj) & (~f.seg_robot) & np.isfinite(f.depth_m)
                err = np.abs(est - f.depth_m)
                row = {"family": family, "seed": seed, "camera": cam, "latency_s": round(dt, 3),
                       "scale": (mono.last_record or {}).get("align", {}).get("scale"),
                       "obj_px": int(obj.sum()),
                       "obj_err_m": float(np.nanmedian(err[obj])) if obj.any() else None,
                       "table_err_m": float(np.nanmedian(err[table])) if table.any() else None}
                tops = []
                for oid in np.unique(f.seg_object[obj]):
                    m = f.seg_object == oid
                    if m.sum() < 50:
                        continue
                    zt = np.percentile(base_z(f.depth_m, intr, ext, m), 95)
                    ze = np.percentile(base_z(est, intr, ext, m), 95)
                    tops.append(abs(float(ze - zt)))
                row["top_z_err_m"] = round(st.median(tops), 4) if tops else None
                rows.append(row)
                print(json.dumps(row), flush=True)
            drv.close()
    for cam in CAMERAS:
        r = [x for x in rows if x["camera"] == cam]
        def m(k):
            v = [x[k] for x in r if x[k] is not None]
            return round(st.median(v), 4) if v else None
        print(json.dumps({"camera": cam, "n": len(r), **{k: m(k) for k in ("latency_s", "obj_err_m", "table_err_m",
                                                                         "top_z_err_m")}}))


if __name__ == "__main__":
    main()

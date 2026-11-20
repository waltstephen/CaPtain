"""MoGe-2 depth (the episodes' SOARM_DEPTH=moge2 path) vs MuJoCo's rendered depth on div2 frames, and its effect on what the
tools compute from depth.

    python tools/depth_eval.py --mix div2 --scenes-per-view 6 --out /tmp/depth_eval [--depth-url http://127.0.0.1:8130]

Scenes: div2 training layouts (spec drawn from the mix, settled start), one per (train view style, k). Frames: agentview,
wrist camera at the home pose, and wrist camera hovering 10 cm above the first task object (arm posed by the driver's
IK). Depth: perception.mono_depth_estimators(...) -- the same harness MonoDepthSource (MoGe-2, intrinsics, table-plane
scale) the episodes use; the raw MoGe-2 metric depth (before the plane scale) and an oracle-scale diagnostic
(best per-frame scale from the true table pixels: plane-fit error vs MoGe-2 shape error) are reported too. Host-side MuJoCo
segmentation and rendered depth are used here ONLY as ground truth for scoring.

Per frame: median |error| (cm) and AbsRel over all valid pixels and over object pixels. Per visible object (>= 150 px):
pixel errors, and the tool-level quantities from the harness's own back-projection (services._depth_to_points with the
object's mask): 3D centroid (mean point) and top height (95th-percentile z = top-down grasp height reference), MoGe-2 vs
rendered depth. Writes frames.jsonl, objects.jsonl, summary.json and a few rgb|rendered|MoGe-2|error PNG strips.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

from OpenGL import GL  # noqa: F401  (load OSMesa before mujoco)
import mujoco
import numpy as np
from PIL import Image

TRAIN_VIEWS = ("front", "front_high", "oblique", "side", "low")
MIN_OBJECT_PIXELS = 150


def _err(est: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> dict:
    valid = mask & np.isfinite(est) & (est > 0) & np.isfinite(ref) & (ref > 0)
    n = int(valid.sum())
    if not n:
        return {"pixels": 0}
    e, r = est[valid], ref[valid]
    ae = np.abs(e - r)
    return {"pixels": n, "abs_cm_median": round(float(np.median(ae)) * 100, 3),
            "abs_cm_mean": round(float(ae.mean()) * 100, 3), "abs_cm_p90": round(float(np.percentile(ae, 90)) * 100, 3),
            "absrel": round(float(np.mean(ae / r)), 4), "delta_1.05": round(float(np.mean(np.maximum(e / r, r / e) < 1.05)), 4),
            "signed_cm_median": round(float(np.median(e - r)) * 100, 3)}


def _colour(depth: np.ndarray, lo: float, hi: float) -> np.ndarray:
    v = np.clip((np.nan_to_num(depth, nan=hi) - lo) / max(hi - lo, 1e-6), 0, 1)
    return (np.stack([v, 1 - np.abs(2 * v - 1), 1 - v], -1) * 255).astype(np.uint8)


def main() -> None:
    from captain_harness.adapters.capx.services import _depth_to_points
    from captain_harness.adapters.hardware.mono_depth import plane_depth
    from soarm import mix as mixes
    from soarm import perception
    from soarm.driver import SoarmDriver
    from soarm.sim import SoarmSim
    from soarm.twin import build_twin_xml

    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default="div2")
    ap.add_argument("--scenes-per-view", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1_059_500, help="layout seeds (smoke range; nothing is collected)")
    ap.add_argument("--depth-url", default=None, help="MoGe-2 depth service (default: SOARM_DEPTH_URLS[0] / :8130)")
    ap.add_argument("--strips", type=int, default=10, help="PNG strips to write")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    url = a.depth_url or ((os.environ.get("SOARM_DEPTH_URLS") or "").split(",")[0].strip()
                          or "http://127.0.0.1:8130")
    health = perception.require_depth_healthy(url)
    perc = {"depth": "moge2", "depth_url": url, "depth_align": "plane",
            "depth_support_plane_z_m": perception.SUPPORT_PLANE_Z_M,
            "depth_table_footprint_m": perception.TABLE_FOOTPRINT_M}
    transport = perception.LoggedHttpTransport(out / "perception.jsonl")
    m = mixes.load(a.mix)
    ids = sorted(m["weights"])
    rng = np.random.default_rng(a.seed)
    frames_out, objects_out = [], []
    strips = 0
    k = 0
    for view in TRAIN_VIEWS:
        for _ in range(a.scenes_per_view):
            sid, seed = ids[int(rng.integers(len(ids)))], a.seed + k
            k += 1
            sim = SoarmSim(sid, seed, config=dataclasses.replace(m["config"], camera=view))
            drv = SoarmDriver(sim, build_twin_xml(), force_seed=seed, privileged_frames=True)   # host-side ground truth
            estimators = perception.mono_depth_estimators(perc, ("agentview", "robot0_eye_in_hand"), transport)
            source = estimators["agentview"].source
            shots = [("agentview", "agentview", None), ("robot0_eye_in_hand", "wrist_home", None)]
            target = next(iter(sim.specs))
            goal = np.eye(4)
            goal[:3, :3] = np.diag([1.0, -1.0, -1.0])            # top-down, quaternion_wxyz [0, 1, 0, 0]
            goal[:3, 3] = sim.poses()[target].pos + [0, 0, 0.10]
            shots.append(("robot0_eye_in_hand", "wrist_hover", goal))
            for cam, shot, goal_pose in shots:
                if goal_pose is not None:
                    q = drv._ik(goal_pose, sim.arm_qpos())
                    sim.data.qpos[sim.arm_q] = q
                    mujoco.mj_forward(sim.model, sim.data)
                f = drv.cameras.render(sim.data, cam)
                est = estimators[cam](camera=cam, rgb=f.rgb, raw_sim_depth=None, intrinsics_3x3=f.intrinsics,
                                      camera_to_base_4x4=f.optical_in_base)
                rec = source.last_record or {}
                align = rec.get("align") or {}
                scale = float(align.get("scale", 1.0))
                raw = est / scale
                ref = f.depth_m.astype(np.float64)
                obj = f.seg_object >= 0
                # oracle: the best single scale for this frame (median rendered/raw over the true table pixels): separates
                # the plane fit's scale error from MoGe-2's shape error (diagnostic only; episodes cannot know it)
                table = (~obj & ~f.seg_robot & np.isfinite(raw) & (raw > 0)
                         & (np.abs(ref - plane_depth(f.intrinsics, f.optical_in_base, perception.SUPPORT_PLANE_Z_M,
                                                     ref.shape)) < 0.003))
                oracle = float(np.median(ref[table] / raw[table])) if table.sum() > 500 else float("nan")
                row = {"spec": sid, "seed": seed, "view": view, "style": getattr(sim.rand.appearance, "view", None),
                       "camera": cam, "shot": shot, "plane_scale": scale, "plane_fit": align.get("status"),
                       "plane_inliers": align.get("inlier_pixels"), "oracle_scale": oracle,
                       "scale_err": round(scale / oracle - 1, 4) if np.isfinite(oracle) else None,
                       "moge_valid_ratio": (rec.get("service") or {}).get("valid_ratio"),
                       "service_s": (rec.get("service") or {}).get("elapsed_s"),
                       "all": _err(est, ref, np.ones_like(obj)), "objects": _err(est, ref, obj),
                       "raw_all": _err(raw, ref, np.ones_like(obj)), "raw_objects": _err(raw, ref, obj),
                       "oracle_objects": _err(raw * oracle, ref, obj) if np.isfinite(oracle) else {"pixels": 0}}
                frames_out.append(row)
                for idx, name in enumerate(sim.specs):
                    mask = f.seg_object == idx
                    if int(mask.sum()) < MIN_OBJECT_PIXELS:
                        continue
                    p_ref = _depth_to_points(ref, f.intrinsics, f.optical_in_base, mask=mask)
                    p_est = _depth_to_points(est, f.intrinsics, f.optical_in_base, mask=mask)
                    if len(p_ref) < 20 or len(p_est) < 20:
                        continue
                    c_ref, c_est = p_ref.mean(0), p_est.mean(0)
                    top_ref, top_est = np.percentile(p_ref[:, 2], 95), np.percentile(p_est[:, 2], 95)
                    objects_out.append({
                        "spec": sid, "seed": seed, "view": view, "camera": cam, "shot": shot, "object": name,
                        "pixels": int(mask.sum()), "depth": _err(est, ref, mask),
                        "centroid_err_cm": round(float(np.linalg.norm(c_est - c_ref)) * 100, 3),
                        "centroid_xy_err_cm": round(float(np.linalg.norm((c_est - c_ref)[:2])) * 100, 3),
                        "centroid_z_err_cm": round(float(c_est[2] - c_ref[2]) * 100, 3),
                        "top_z_err_cm": round(float(top_est - top_ref) * 100, 3),
                        "top_z_ref_cm": round(float(top_ref) * 100, 2)})
                if strips < a.strips and shot != "wrist_home":
                    lo, hi = np.nanpercentile(ref, 1), np.nanpercentile(ref, 99)
                    err = np.clip(np.abs(np.nan_to_num(est - ref)) / 0.03, 0, 1)   # white = 3 cm error
                    strip = np.concatenate([f.rgb, _colour(ref, lo, hi), _colour(est, lo, hi),
                                            (np.stack([err] * 3, -1) * 255).astype(np.uint8)], 1)
                    Image.fromarray(strip).save(out / f"strip_{view}_{seed}_{shot}.png")
                    strips += 1
                print(json.dumps({k2: row[k2] for k2 in ("spec", "view", "shot", "plane_scale")}
                                 | {"all_cm": row["all"].get("abs_cm_median"), "obj_cm": row["objects"].get("abs_cm_median")}),
                      flush=True)
            drv.close()
    with (out / "frames.jsonl").open("w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in frames_out)
    with (out / "objects.jsonl").open("w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in objects_out)

    def agg(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return None if not vals else {"median": round(float(np.median(vals)), 3), "mean": round(float(np.mean(vals)), 3),
                                      "p90": round(float(np.percentile(np.abs(vals), 90)), 3), "n": len(vals)}

    def frame_table(rows):
        return {"frames": len(rows),
                "all_abs_cm_median": agg([r["all"] for r in rows], "abs_cm_median"),
                "all_absrel": agg([r["all"] for r in rows], "absrel"),
                "obj_abs_cm_median": agg([r["objects"] for r in rows], "abs_cm_median"),
                "obj_absrel": agg([r["objects"] for r in rows], "absrel"),
                "obj_signed_cm_median": agg([r["objects"] for r in rows], "signed_cm_median"),
                "raw_all_absrel": agg([r["raw_all"] for r in rows], "absrel"),
                "raw_obj_abs_cm_median": agg([r["raw_objects"] for r in rows], "abs_cm_median"),
                "oracle_obj_abs_cm_median": agg([r["oracle_objects"] for r in rows], "abs_cm_median"),
                "plane_scale_err": agg(rows, "scale_err"),
                "plane_scale_err_gt_5pct": sum(1 for r in rows if r.get("scale_err") is not None and abs(r["scale_err"]) > 0.05),
                "plane_scale": agg(rows, "plane_scale"),
                "plane_fit_failed": sum(1 for r in rows if r["plane_fit"] != "ok")}

    def object_table(rows):
        return {"objects": len(rows), "centroid_err_cm": agg(rows, "centroid_err_cm"),
                "centroid_xy_err_cm": agg(rows, "centroid_xy_err_cm"), "centroid_z_err_cm": agg(rows, "centroid_z_err_cm"),
                "top_z_err_cm": agg(rows, "top_z_err_cm"),
                "top_z_abs_err_cm": agg([{"v": abs(r["top_z_err_cm"])} for r in rows], "v"),
                "obj_abs_cm_median": agg([r["depth"] for r in rows], "abs_cm_median")}

    summary = {"depth_url": url, "health": health, "revision": perception.MOGE2_REVISION, "mix": a.mix,
               "by_shot": {s: frame_table([r for r in frames_out if r["shot"] == s])
                           for s in ("agentview", "wrist_home", "wrist_hover")},
               "agentview_by_view": {v: frame_table([r for r in frames_out if r["shot"] == "agentview" and r["view"] == v])
                                     for v in TRAIN_VIEWS},
               "objects_by_shot": {s: object_table([r for r in objects_out if r["shot"] == s])
                                   for s in ("agentview", "wrist_home", "wrist_hover")},
               "objects_agentview_by_view": {v: object_table([r for r in objects_out if r["shot"] == "agentview"
                                                              and r["view"] == v]) for v in TRAIN_VIEWS},
               "latency_s": agg([r for r in frames_out], "service_s")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

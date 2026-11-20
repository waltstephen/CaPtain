"""Fixed-camera calibration from the checkerboard lying flat on the table: the arm localizes it, nobody holds it.

Per placement (camstream.py's record button runs flatrec.py, which chains these stages):
  plan   from rest: a fixed survey of high look-down wrist poses (SURVEY, chosen in sim to cover board placements in
         front of the arm; every pose and transfer re-checked in sim), a snapshot of both cameras at each, back to rest
         (torque on; home releases it). No fixed-camera calibration is needed to plan it.
  add    board corners in the base frame from the wrist snapshots that see the whole board (FK with the kinematic
         corrections + hand-eye + wrist intrinsics; views must agree to 4 mm), paired with the board corners in the
         fixed snapshots that see the whole board (the arm is out of the way in some of them; they must agree to 1.5 px)
  solve  all placements at once: focal length, radial distortion (k1, k2) and T_base_cam of the fixed camera by
         reprojection least squares. Principal point and aspect ratio stay at the wrist camera's (same camera model):
         a single plane orientation cannot determine them. Validation: leave one placement out, fit the rest, map the
         held-out fixed-image corners onto the table and compare with the arm-measured corners (mm). Baseline: the
         borrowed wrist intrinsics with only the pose fitted, same hold-out. --write updates <cameras.json>["fixed"]
         (backup first) from 3 placements on, when the hold-out error beats the baseline.
  home   sim-checked return to rest from wherever the arm is (after the survey, an abort or a refusal), torque off

usage (venv-srv, cwd $SOARM_ROOT/armcal with snaps/):
  flatcal.py plan  <cameras.json> '<rest raw json>' <plan_out.json> <prefix>
  flatcal.py add   <cameras.json> <survey log> <flat dir> <prefix>
  flatcal.py solve <cameras.json> <flat dir> [--write]
  flatcal.py home  <cameras.json> '<raw json>' '<rest raw json>' <plan_out.json>
Safety (corrected model, mesh vertices, every transfer sampled joint by joint; calib_views.Scene rules): survey poses
keep the arm >= 80 mm and the gripper >= 60 mm above the table (above the 4 cm blocks and the board), transfers >= 50 /
30 mm, 15 cm around the stored camera stand position, and no arm vertex at y > 0.10 m below z = 0.35 m (the raised
shelf on the +y side).
"""
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np
from scipy.optimize import least_squares

import calib_views as V
import pickplace as P
import so101_ik as ik
from calib_kinematics import project

# printed A4 ChArUco boards: 7x5 squares of 35 mm, 26 mm markers, 5X5_100. Board 0 (charuco_7x5_35mm_A4.pdf) uses
# marker ids 0-16, board 1 (charuco_7x5_35mm_A4_b.pdf) ids 17-33, so both can lie in view at once. Their 24 inner
# corners have ids, so a partly hidden board still gives known corners.
_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
BOARDS = [cv2.aruco.CharucoBoard((7, 5), 0.035, 0.026, _DICT, np.arange(17 * b, 17 * b + 17)) for b in range(2)]
_DP = cv2.aruco.DetectorParameters()  # small markers in the fixed view: default params find 10 of 17, these all
_DP.adaptiveThreshWinSizeMax, _DP.minMarkerPerimeterRate, _DP.errorCorrectionRate = 53, 0.01, 0.8
_CP = cv2.aruco.CharucoParameters()
_CP.minMarkers = 1
DETECTORS = [cv2.aruco.CharucoDetector(B, _CP, _DP) for B in BOARDS]
OBJ = np.asarray(BOARDS[0].getChessboardCorners(), float).reshape(-1, 3)  # 24 x 3, rows of 6 (same for both)
MIN_CORNERS = 10
SAME_PLACE_MM = 5.0  # a board that did not move since an earlier placement adds nothing new
SHELF_Y, SHELF_Z = 0.10, 0.35
SURVEY_CLEAR = 0.03  # on top of calib_views' 50 / 30 mm floors
VIEW_AGREE_MM, FIXED_AGREE_PX = 4.0, 1.5
# 11-13: a board centered 20 cm from the base came out 6 / 5 / 9 mm (x / y / z) off: the close, steep views are
# outside the kinematic calibration's range. Closer boards are skipped.
MIN_REACH_M = 0.25
# flatcal.py survey (11-13, ChArUco): 12 poses; 96 % of board placements (center 18-50 cm in front) seen by >= 2
SURVEY = [{"1": 2221, "2": 1562, "3": 2208, "4": 3026, "5": 3063}, {"1": 2104, "2": 1546, "3": 2172, "4": 3089, "5": 2192},
          {"1": 2043, "2": 1382, "3": 2343, "4": 3117, "5": 2857}, {"1": 2423, "2": 1387, "3": 2199, "4": 2945, "5": 3913},
          {"1": 2299, "2": 1328, "3": 2566, "4": 2830, "5": 3035}, {"1": 1997, "2": 1480, "3": 2332, "4": 3041, "5": 2831},
          {"1": 2426, "2": 1320, "3": 2594, "4": 2908, "5": 3537}, {"1": 2313, "2": 1498, "3": 2335, "4": 3117, "5": 2920},
          {"1": 2370, "2": 1488, "3": 1880, "4": 3073, "5": 2068}, {"1": 1940, "2": 1498, "3": 2366, "4": 3065, "5": 3180},
          {"1": 2254, "2": 1660, "3": 2169, "4": 2966, "5": 3384}, {"1": 2113, "2": 2095, "3": 1511, "4": 3068, "5": 2825}]


def board_px(img):
    """{board index: (corner ids, px)} for every board with >= MIN_CORNERS detected inner corners."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = {}
    for b, det in enumerate(DETECTORS):
        cc, ci, _, _ = det.detectBoard(g)
        if ci is not None and len(ci) >= MIN_CORNERS:
            out[b] = (ci.ravel().astype(int), cc.reshape(-1, 2).astype(np.float64))
    return out


def to_T(rv, tv):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = cv2.Rodrigues(np.asarray(rv, float))[0], np.asarray(tv, float).ravel()
    return T


def raw_json(s):
    return {k: int(v) for k, v in json.loads(s).items()}


class Scene(V.Scene):
    """calib_views.Scene (table floors, colored-block boxes, camera stand) + the shelf keep-out."""

    def __init__(self, cams, blocks_xy=()):
        kin = {"T_gripperframe_cam": cams["wrist"]["T_gripperframe_cam"],
               "T_base_tag": cams.get("tag", {}).get("T_base_tag", np.eye(4).tolist())}
        super().__init__(kin, cams, list(blocks_xy))
        # calib_views puts a 15 cm keep-out cylinder (all heights) under the fixed camera. flatcal's camera position
        # (x 0.08, y -0.20, z 0.67 m) is right above the arm's reach, so that cylinder blocks every survey pose; the
        # stand itself is not under the lens. Keep the old cylinder position (0.249, -0.265), with which all 12
        # survey poses ran on the real arm 4 times without contact (11-13).
        self.stand = np.array([0.249, -0.265])

    def clearance(self, q):
        worst = super().clearance(q)  # leaves the model posed at q
        P_ = np.concatenate([v @ ik.d.geom_xmat[g].reshape(3, 3).T + ik.d.geom_xpos[g] for g, _, v in V._GEOMS])
        low = P_[P_[:, 2] < SHELF_Z]
        return min(worst, SHELF_Y - low[:, 1].max()) if len(low) else worst



def moves(order, b):
    return [{"id": int(j), "goal": b[j], "torque_limit": V.TL[j], "return_after": False,
             "speed": 180 if j == "5" else 120} for j in order]


def transfer(sc, a, b, what):
    mg, order = V.best_path(sc, a, b)
    if order is None or mg <= 0:
        raise SystemExit(f"REFUSE no clear joint order for {what} (margin {mg * 1000:.0f} mm)")
    return moves(order, b)


def stage_plan(cams_path, rest_s, out_path, prefix):
    """rest -> every SURVEY pose that is still clear in sim (nearest first, joint order checked) with a snapshot of
    both cameras -> rest, torque left on for home."""
    sc, rest = Scene(json.load(open(cams_path))), raw_json(rest_s)
    left = [r for r in SURVEY if sc.clearance(V.raw_q(r)) >= SURVEY_CLEAR]
    steps, cur, n = [], {k: rest[k] for k in "12345"}, 0
    while left:
        left.sort(key=lambda v: sum(abs(v[j] - cur[j]) / (900 if j == "5" else 300) for j in "12345"))
        for v in list(left):
            mg, order = V.best_path(sc, cur, v)
            if order is not None and mg > 0:
                break
        else:
            break
        left.remove(v)
        steps += moves(order, v) + [{"snap": f"{prefix}_v{n:02d}"}]
        cur, n = dict(v), n + 1
    if n < 4:
        raise SystemExit(f"REFUSE only {n} of {len(SURVEY)} survey poses are clear in sim")
    steps += transfer(sc, cur, {k: rest[k] for k in "12345"}, "return to rest")
    print(f"{n} of {len(SURVEY)} survey poses planned, {len(steps)} steps")
    json.dump({"torque_limit": 400, "speed": 120, "steps": steps, "return": False, "release": False},
              open(out_path, "w"))


def board_view_ok(sc, q, T_bb):
    """>= 16 of the 24 corners in the usable wrist image (right of the jaws, 25 px margins), board normal within
    67 deg of the optical axis, >= 17 px per square (the conditions of the 11-12 views that detected a board)."""
    T = np.linalg.inv(ik.fk(q) @ sc.T_gc) @ T_bb
    Pc = OBJ @ T[:3, :3].T + T[:3, 3]
    if (Pc[:, 2] < 0.10).any() or abs(T[2, 2]) < 0.38:
        return False
    uv = project(Pc, sc.K, sc.dist)
    inside = (uv[:, 0] >= 470) & (uv[:, 0] <= V.IMG_W - 25) & (uv[:, 1] >= 25) & (uv[:, 1] <= V.IMG_H - 25)
    if inside.sum() < 16:
        return False
    g = uv.reshape(4, 6, 2)
    sq = min(np.linalg.norm(np.diff(g, axis=0), axis=2).min(), np.linalg.norm(np.diff(g, axis=1), axis=2).min())
    return sq >= 17


def stage_survey(cams_path, n_poses=12, n_samples=20000):
    """Offline: choose SURVEY. Random clear poses scored on 300 random board placements in front of the arm (center
    x 0.18-0.50 m, y -0.30-0.08 m, any yaw); greedy, each placement counted until two poses see it."""
    sc, rng = Scene(json.load(open(cams_path))), np.random.default_rng(0)
    boards = []
    for _ in range(300):
        cx, cy, yaw = rng.uniform(0.18, 0.50), rng.uniform(-0.30, 0.08), rng.uniform(-np.pi, np.pi)
        R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]) @ np.diag([1, -1, -1])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, np.array([cx, cy, 0.004]) - R @ OBJ.mean(0)
        boards.append(T)
    poses = []
    for _ in range(n_samples):
        r = {k: int(v) for k, v in zip("12345", rng.uniform([1500, 1300, 1000, 1500, 1955],
                                                              [2700, 2900, 3100, 3400, 3963]))}
        q = V.raw_q(r)
        try:
            ik.q_to_raw(q)
        except ValueError:
            continue
        seen = np.array([board_view_ok(sc, q, b) for b in boards])
        if seen.any() and sc.clearance(q) >= SURVEY_CLEAR:
            poses.append((r, seen))
    cnt, chosen = np.zeros(len(boards)), []
    for _ in range(min(n_poses, len(poses))):
        i = max(range(len(poses)), key=lambda i: np.minimum(cnt + poses[i][1], 2).sum())
        r, seen = poses.pop(i)
        cnt += seen
        chosen.append(r)
    ctr = np.array([b[:3, 3] + b[:3, :3] @ OBJ.mean(0) for b in boards])
    print(f"{len(poses)} clear poses see a board; chosen {len(chosen)}: placements seen whole by >= 2 poses "
          f"{(cnt >= 2).mean():.2f}, by >= 1 {(cnt >= 1).mean():.2f}")
    for lo, hi in ((0.18, 0.28), (0.28, 0.38), (0.38, 0.50)):
        m = (ctr[:, 0] >= lo) & (ctr[:, 0] < hi)
        print(f"  board center x {lo:.2f}-{hi:.2f} m: seen by >= 2 poses {(cnt[m] >= 2).mean():.2f}")
    print("SURVEY = " + json.dumps(chosen))


def stage_home(cams_path, raw_s, rest_s, out_path):
    sc, raw, rest = Scene(json.load(open(cams_path))), raw_json(raw_s), raw_json(rest_s)
    a, b = {k: raw[k] for k in "12345"}, {k: rest[k] for k in "12345"}
    steps = transfer(sc, a, b, "home") if a != b else []
    json.dump({"torque_limit": 400, "speed": 120, "steps": steps, "return": False, "release": True},
              open(out_path, "w"))


def stage_add(cams_path, log_path, flat_dir, prefix):
    """Arm-measured board corners (base frame) from the wrist snaps, paired with the fixed-image corners; every
    board that moved since the earlier placements counts."""
    cams = json.load(open(cams_path))
    w = cams["wrist"]
    K, dist, T_gc = np.array(w["K"]), np.array(w["dist"]), np.array(w["T_gripperframe_cam"])
    views, fixed = {}, {}  # board -> [(snap, 24x3 base)], board -> corner id -> [px]
    for e in map(json.loads, open(log_path)):
        if e.get("event") != "snap" or not e["name"].startswith(prefix + "_v"):
            continue
        for b, (ids, px) in board_px(cv2.imread(f"snaps/{e['name']}_fixed.png", cv2.IMREAD_GRAYSCALE)).items():
            for i, p in zip(ids, px):
                fixed.setdefault(b, {}).setdefault(int(i), []).append(p)
        T_bc = ik.fk(ik.raw_to_q({str(k): v for k, v in e["raw"].items()})) @ T_gc
        for b, (ids, px) in board_px(cv2.imread(f"snaps/{e['name']}_wrist.png", cv2.IMREAD_GRAYSCALE)).items():
            ok, rv, tv = cv2.solvePnP(OBJ[ids], px, K, dist)
            T = T_bc @ to_T(rv, tv)
            views.setdefault(b, []).append((e["name"], OBJ @ T[:3, :3].T + T[:3, 3]))
    pl_path = os.path.join(flat_dir, "placements.json")
    pls = json.load(open(pl_path)) if os.path.exists(pl_path) else []
    entry, notes = {"prefix": prefix, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "boards": {}}, []
    for b in sorted(set(views) | set(fixed)):
        vs, fx = views.get(b, []), fixed.get(b, {})
        if len(vs) < 2 or len(fx) < MIN_CORNERS:
            notes.append(f"board {b}: {len(vs)} wrist views, {len(fx)} fixed-camera corners, not enough")
            continue
        med = np.median([c for _, c in vs], axis=0)
        dev = {n: round(float(np.linalg.norm(c - med, axis=1).mean() * 1000), 1) for n, c in vs}
        good = [c for n, c in vs if dev[n] <= VIEW_AGREE_MM]
        print(f"board {b}: wrist views mm from the median {dev}")
        if len(good) < 2:
            notes.append(f"board {b}: wrist views disagree on its position (was it bumped?)")
            continue
        # per corner: median over the snapshots; corners that jump (partly hidden by the arm) are dropped
        med_px = {i: np.median(fx[i], axis=0) for i in fx}
        ids = sorted(i for i in fx if np.abs(np.array(fx[i]) - med_px[i]).max() <= FIXED_AGREE_PX)
        print(f"board {b}: {len(ids)} of {len(fx)} fixed corners consistent within {FIXED_AGREE_PX} px")
        if len(ids) < MIN_CORNERS:
            notes.append(f"board {b}: only {len(ids)} corners stable in the fixed camera (was the board bumped?)")
            continue
        fpx = np.array([med_px[i] for i in ids])
        full = np.mean(good, axis=0)
        if np.linalg.norm(full.mean(0)[:2]) < MIN_REACH_M:
            notes.append(f"board {b}: only {np.linalg.norm(full.mean(0)[:2]) * 100:.0f} cm from the base, wrist localization is biased there, skipped")
            continue
        if any(str(b) in p["boards"] and np.linalg.norm(np.array(p["boards"][str(b)]["center_m"]) - full.mean(0))
               * 1000 < SAME_PLACE_MM for p in pls):
            notes.append(f"board {b}: same position as before, skipped")
            continue
        entry["boards"][str(b)] = {"ids": ids, "fixed_px": fpx.tolist(), "base_m": full[ids].tolist(),
                                   "center_m": full.mean(0).tolist(), "n_views": len(good), "view_dev_mm": dev}
        print(f"board {b}: center {np.round(full.mean(0) * 1000, 1)} mm, {len(good)} wrist views, "
              f"{len(ids)} fixed corners")
    if not entry["boards"]:
        raise SystemExit("REFUSE no usable new board position this time: " + "; ".join(notes))
    pls.append(entry)
    os.makedirs(flat_dir, exist_ok=True)
    json.dump(pls, open(pl_path, "w"), indent=1)
    print(f"placement {len(pls)} saved (boards {sorted(entry['boards'])})" + ("; " + "; ".join(notes) if notes else ""))


def fit(obj, img, K0):
    """f, k1, k2, pose of the fixed camera from base-frame points; cx, cy, fx/fy fixed at K0's."""
    aspect = K0[0, 0] / K0[1, 1]

    def unpack(x):
        K = np.array([[x[0] * aspect, 0, K0[0, 2]], [0, x[0], K0[1, 2]], [0, 0, 1]])
        return K, np.array([x[1], x[2], 0, 0, 0]), to_T(x[3:6], x[6:9])

    def resid(x):
        K, dist, T = unpack(x)
        return (project(obj @ T[:3, :3].T + T[:3, 3], K, dist) - img).ravel()

    ok, rv, tv = cv2.solvePnP(obj, img, K0, np.zeros(5), flags=cv2.SOLVEPNP_SQPNP)
    s = least_squares(resid, np.r_[K0[1, 1], 0, 0, rv.ravel(), tv.ravel()], loss="soft_l1", f_scale=1.0,
                      x_scale="jac")
    K, dist, T_cb = unpack(s.x)
    r = s.fun.reshape(-1, 2)
    return K, dist, np.linalg.inv(T_cb), float(np.sqrt((r ** 2).sum(1).mean()))


def points(sel):
    """(base m N x 3, fixed px N x 2) of every board in the placements sel."""
    bs = [bd for p in sel for bd in p["boards"].values()]
    return (np.concatenate([np.array(bd["base_m"]) for bd in bs]),
            np.concatenate([np.array(bd["fixed_px"]) for bd in bs]))


def table_err_mm(K, dist, T_base_cam, pl):
    """Fixed-image corners mapped onto the board's plane height vs the arm-measured corners: median / max mm."""
    base, px = points([pl])
    xyz = np.array([P.px_to_plane(np.array([uv]), K, dist, T_base_cam, z)[0] for uv, z in zip(px, base[:, 2])])
    e = np.linalg.norm(xyz[:, :2] - base[:, :2], axis=1) * 1000
    return float(np.median(e)), float(e.max())


def stage_solve(cams_path, flat_dir, write):
    cams = json.load(open(cams_path))
    pls = json.load(open(os.path.join(flat_dir, "placements.json")))
    K0, d0 = np.array(cams["wrist"]["K"]), np.array(cams["wrist"]["dist"])
    stack = points

    def baseline(sel):  # borrowed wrist intrinsics, only the pose fitted
        o, i = stack(sel)
        ok, rv, tv = cv2.solvePnP(o, i, K0, d0, flags=cv2.SOLVEPNP_SQPNP)
        return np.linalg.inv(to_T(rv, tv))
    K, dist, T_bc, rms = fit(*stack(pls), K0)
    res = {"n_placements": len(pls), "f_px": float(K[1, 1]), "k1": float(dist[0]), "k2": float(dist[1]),
           "rms_px": rms, "cam_pos_mm": (T_bc[:3, 3] * 1000).round(1).tolist()}
    print(f"{len(pls)} placements: f {K[1, 1]:.1f} px (wrist {K0[1, 1]:.1f}), k1 {dist[0]:+.3f} k2 {dist[1]:+.3f}, "
          f"rms {rms:.2f} px, camera at {res['cam_pos_mm']} mm")
    if len(pls) >= 3:
        res["holdout_err_mm"], res["old_err_mm"] = [], []
        for i in range(len(pls)):
            rest_ = pls[:i] + pls[i + 1:]
            Ki, di, Ti, _ = fit(*stack(rest_), K0)
            res["holdout_err_mm"].append(table_err_mm(Ki, di, Ti, pls[i]))
            res["old_err_mm"].append(table_err_mm(K0, d0, baseline(rest_), pls[i]))
        new = float(np.median([m for m, _ in res["holdout_err_mm"]]))
        old = float(np.median([m for m, _ in res["old_err_mm"]]))
        res.update(holdout_median_mm=new, old_median_mm=old)
        print("held-out placement errors on the table, median/max mm: "
              + ", ".join(f"{m:.1f}/{x:.1f}" for m, x in res["holdout_err_mm"])
              + "; borrowed intrinsics: " + ", ".join(f"{m:.1f}/{x:.1f}" for m, x in res["old_err_mm"]))
        res["written"] = bool(write and new < old)
        if res["written"]:
            bak = f"{cams_path}.bak-{time.strftime('%m%d-%H%M%S')}"
            shutil.copy(cams_path, bak)
            cams["fixed"].update(
                K=K.tolist(), dist=dist.tolist(), T_base_cam=T_bc.tolist(), rms_px=rms, n_placements=len(pls),
                holdout_err_mm_median=new, pnp_reproj_px=rms,
                intrinsics_source=f"flatcal.py {time.strftime('%Y-%m-%d')}: board flat on the table at {len(pls)} "
                                  f"places, corners located by the arm (wrist views); f, k1, k2 and pose fitted, "
                                  f"principal point / aspect from the wrist camera",
                note="pickplace.py still re-locates T_base_cam from the AprilTag when it is in view")
            cams["fixed"].pop("n_views", None)
            json.dump(cams, open(cams_path, "w"), indent=1)
            print(f"wrote {cams_path} (backup {bak})")
        else:
            print("not written: " + ("hold-out error not below the borrowed intrinsics'" if write else "no --write"))
    json.dump(res, open(os.path.join(flat_dir, "result.json"), "w"), indent=1)


if __name__ == "__main__":
    cmd, a = sys.argv[1:2], sys.argv[2:]
    if cmd == ["plan"]:
        stage_plan(*a[:4])
    elif cmd == ["survey"]:
        stage_survey(a[0])
    elif cmd == ["add"]:
        stage_add(*a[:4])
    elif cmd == ["solve"]:
        stage_solve(a[0], a[1], "--write" in a)
    elif cmd == ["home"]:
        stage_home(*a[:4])
    else:
        raise SystemExit(__doc__)

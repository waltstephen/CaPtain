"""Where does the gripper really go? Hover the grip point 30 mm above ChArUco board-0 corners and measure.

plan:    grip_check.py plan <cameras.json> '<rest raw>' <fixed.png> <plan_out.json>
measure: grip_check.py measure <cameras.json> <log> <fixed.png>
The board pose comes from the fixed camera (flatcal calibration). At each hover the wrist camera sees the board;
PnP gives the real camera pose, so the real grip point = that pose x inv(T_gripperframe_cam) x GRIP_LOCAL, compared
with the target corner (mm). The wrist image also gets the target corner marked: the jaws should straddle it.
Transfers: pickplace.go(big=True) (joint order maximizing the lowest point, sim), hovers 30 mm (lowest gripper mesh).
"""
import json
import sys

import cv2
import numpy as np

import calib_low as L
import flatcal as F
import pickplace as P
import so101_ik as ik
from calib_kinematics import project

CORNERS = [0, 5, 18, 23, 8]  # ChArUco inner corner ids: the four extremes + one inside
HOVER_Z = 0.03


def plan(cams_path, rest_s, fixed_path, out):
    cams, rest = json.load(open(cams_path)), F.raw_json(rest_s)
    T_bb = L.board_pose(cams, cv2.imread(fixed_path))
    P.GRIP_LOCAL = P.grip_local(0.0325)
    close = T_bb[:3, 0] * [1, 1, 0]
    close /= np.linalg.norm(close)
    steps, cur = [], {k: rest[k] for k in "12345"}
    for c in CORNERS:
        t = (T_bb @ np.r_[F.OBJ[c], 1])[:3]
        q = P.topdown_ik(np.r_[t[:2], 0.08], close, P.grip_q(P.GRIP_OPEN))
        q = P.at_height(q, t[:2], HOVER_Z, close)
        cur = P.go(steps, cur, P.arm_raw(q), f"gc_{c:02d}", big=True)
        print(f"corner {c}: target {np.round(t * 1000, 1)} mm")
    P.go(steps, cur, {k: rest[k] for k in "12345"}, big=True)
    steps.insert(0, {"id": 6, "goal": P.GRIP_OPEN, "torque_limit": 300, "return_after": False})
    steps.append({"id": 6, "goal": rest["6"], "torque_limit": 300, "return_after": False})
    json.dump({"torque_limit": 500, "speed": 120, "steps": steps, "return": False, "release": False}, open(out, "w"))


def measure(cams_path, log, fixed_path):
    cams = json.load(open(cams_path))
    w = cams["wrist"]
    K, dist, T_gc = np.array(w["K"]), np.array(w["dist"]), np.array(w["T_gripperframe_cam"])
    T_bb = L.board_pose(cams, cv2.imread(fixed_path))
    gl = P.grip_local(0.0325)
    for e in map(json.loads, open(log)):
        if e.get("event") != "snap":
            continue
        c = int(e["name"].split("_")[1])
        t = (T_bb @ np.r_[F.OBJ[c], 1])[:3]
        T_fk = ik.fk(ik.raw_to_q({str(k): v for k, v in e["raw"].items()}))
        g_fk = T_fk[:3, :3] @ gl + T_fk[:3, 3]
        img = cv2.imread(f"snaps/{e['name']}_wrist.png")
        det = F.board_px(img)
        line = f"corner {c:2d}: FK grip point - target {np.round((g_fk - t)[:2] * 1000, 1)} mm"
        if 0 in det and len(det[0][0]) >= 6:
            ids, px = det[0]
            ok, rv, tv = cv2.solvePnP(F.OBJ[ids], px, K, dist)
            T_bc = T_bb @ np.linalg.inv(F.to_T(rv, tv))
            T_real = T_bc @ np.linalg.inv(T_gc)
            g = T_real[:3, :3] @ gl + T_real[:3, 3]
            line += f"; camera-measured grip point - target {np.round((g - t)[:2] * 1000, 1)} mm ({len(ids)} corners)"
            Tc = F.to_T(rv, tv)
            uv = project((F.OBJ[c] @ Tc[:3, :3].T + Tc[:3, 3])[None], K, dist)[0]
            cv2.drawMarker(img, tuple(int(v) for v in uv), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
        print(line)
        cv2.imwrite(f"/tmp/gc_{c:02d}.jpg", img)


if __name__ == "__main__":
    {"plan": plan, "measure": measure}[sys.argv[1]](*sys.argv[2:])

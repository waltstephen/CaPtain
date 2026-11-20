"""Find the AprilTag again after it was moved: wrist-camera survey from high look-down poses, then tag pose in base.

usage (cwd with so101_ik / pickplace / arm_probe and the kinematics + cameras json, snaps/):
  tag_survey.py plan '<rest raw json>' <plan_out.json>      -> high look-down survey poses (pan sweep), sim-checked
  tag_survey.py solve <survey_log> <cameras.json>          -> T_base_tag from every view with the full tag
The poses keep every arm vertex >= 100 mm above the table (nothing on the table can be touched); the solve step
reports the spread of the per-view tag poses (mm / deg) and writes the median pose into <cameras.json>["tag"].
"""
import json
import sys

import cv2
import numpy as np

import pickplace as P
import so101_ik as ik

TAG_ID, TAG = 25, 0.150
OBJ = np.array([[-TAG / 2, TAG / 2, 0], [TAG / 2, TAG / 2, 0], [TAG / 2, -TAG / 2, 0], [-TAG / 2, -TAG / 2, 0]])
DET = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                              cv2.aruco.DetectorParameters())


def plan(rest_json, out):
    rest = {k: int(v) for k, v in json.loads(rest_json).items()}
    poses = []
    for pan in np.linspace(-0.6, 0.6, 7):
        for lift, elbow, wrist in ((-0.68, 0.11, 1.40), (-0.45, -0.25, 1.30)):  # high, camera looking down-forward
            q = np.array([pan, lift, elbow, wrist, 0.0, P.grip_q(2080)])
            if ik.body_min_z(q) < 0.10 or P.lowest_gripper_z(q) < 0.10:
                continue
            poses.append(q)
    steps, cur = [], {k: rest[k] for k in "12345"}
    for n, q in enumerate(poses):
        r = ik.q_to_raw(np.r_[q[:5], P.grip_q(2650)])
        r = {str(k): v for k, v in r.items() if k != 6}
        order = P.best_order(cur, r)
        if P.body_min_z_path(cur, r, order) < 0.08:
            raise SystemExit(f"survey move {n} would go below 80 mm")
        for j in order:
            steps.append({"id": int(j), "goal": r[j], "torque_limit": P.TL[j], "return_after": False})
        steps.append({"snap": f"sv{n:02d}"})
        cur = r
    order = P.best_order(cur, {k: rest[k] for k in "12345"})
    steps += [{"id": int(j), "goal": rest[j], "torque_limit": P.TL[j], "return_after": False} for j in order]
    json.dump({"torque_limit": 500, "speed": 150, "steps": steps, "return": False, "release": True}, open(out, "w"))
    print(f"{len(poses)} survey poses, {len(steps)} steps -> {out}")


def solve(log, cal_path):
    cal = json.load(open(cal_path))
    w = cal["wrist"]
    K, dist, T_gc = np.array(w["K"]), np.array(w["dist"]), np.array(w["T_gripperframe_cam"])
    Ts = []
    for e in map(json.loads, open(log)):
        if e.get("event") != "snap":
            continue
        g = cv2.imread(f"snaps/{e['name']}_wrist.png", cv2.IMREAD_GRAYSCALE)
        c, ids, _ = DET.detectMarkers(g)
        if ids is None or TAG_ID not in ids.ravel():
            continue
        pts = c[list(ids.ravel()).index(TAG_ID)].reshape(-1, 2)
        if pts[:, 0].min() < 480:  # left third of the wrist image is behind the jaws
            continue
        ok, rv, tv = cv2.solvePnP(OBJ, pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        T_ct = np.eye(4)
        T_ct[:3, :3], T_ct[:3, 3] = cv2.Rodrigues(rv)[0], tv.ravel()
        Ts.append((e["name"], ik.fk(ik.raw_to_q(e["raw"])) @ T_gc @ T_ct))
    if not Ts:
        raise SystemExit("tag not fully seen in any survey view")
    pos = np.array([T[:3, 3] for _, T in Ts])
    med = np.median(pos, axis=0)
    ref = min(Ts, key=lambda t: np.linalg.norm(t[1][:3, 3] - med))[1].copy()
    ref[:3, 3] = med
    for name, T in Ts:
        ang = np.degrees(np.arccos(np.clip((np.trace(ref[:3, :3].T @ T[:3, :3]) - 1) / 2, -1, 1)))
        print(f"{name}: tag center {np.round(T[:3, 3] * 1000, 1)} mm, {np.linalg.norm(T[:3, 3] - med) * 1000:.1f} mm / "
              f"{ang:.1f} deg from the median")
    old = cal.get("tag", {}).get("T_base_tag")
    if old is not None:
        print(f"moved {np.linalg.norm(med - np.array(old)[:3, 3]) * 1000:.0f} mm since the stored tag pose")
    cal["tag"] = {**cal.get("tag", {}), "family": "36h11", "id": TAG_ID, "size_m": TAG, "T_base_tag": ref.tolist(),
                  "source": f"tag_survey.py: median of {len(Ts)} wrist views ({log})"}
    json.dump(cal, open(cal_path, "w"), indent=1)
    print(f"tag center now {np.round(med * 1000, 1)} mm (z should be ~0); wrote {cal_path}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["plan"]:
        plan(sys.argv[2], sys.argv[3])
    elif sys.argv[1:2] == ["solve"]:
        solve(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)

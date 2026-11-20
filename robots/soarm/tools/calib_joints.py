"""Joint-zero + hand-eye calibration of the real SO-101 from wrist-camera views of known flat targets.

usage: calib_joints.py <cameras.json> <tag_log> <out_prefix> [board_log ...]
  tag_log: arm_probe log whose snaps see the AprilTag 36h11 (150 mm black square, flat on the table)
  board_log: arm_probe logs whose snaps see the 11x8 / 30 mm checkerboard (all from one board placement)
Fits T_gripperframe_cam, T_base_tag, T_base_board and zero offsets of shoulder_lift / elbow_flex / wrist_flex by
minimizing corner reprojection (wrist intrinsics fixed). shoulder_pan and wrist_roll offsets are not observable here
(pan rotates every target about z; roll only rotates the camera, which T_gripperframe_cam absorbs).
Checks: per-offset 1-sigma, error on held-out views, and target heights (both lie on the table, z ~ 0 in the base
frame because the base also stands on it). Writes <out_prefix>_joint_map.json and <out_prefix>_cameras.json.
"""
import json
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

import so101_ik as ik

TAG = 0.150
TAG_OBJ = np.array([[-TAG / 2, TAG / 2, 0], [TAG / 2, TAG / 2, 0], [TAG / 2, -TAG / 2, 0], [-TAG / 2, -TAG / 2, 0]])
SQ = 0.030
BOARD_OBJ = np.zeros((88, 3))
BOARD_OBJ[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2) * SQ
OFF_JOINTS = [1, 2, 3]  # q indices: shoulder_lift, elbow_flex, wrist_flex
DET = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                              cv2.aruco.DetectorParameters())


def to_T(r, t):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(r, float))[0]
    T[:3, 3] = t
    return T


def logr(R):
    return cv2.Rodrigues(R)[0].ravel()


def snaps(path):
    return [e for e in map(json.loads, open(path)) if e.get("event") == "snap" and e.get("ok")]


def tag_corners(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    corners, ids, _ = DET.detectMarkers(img)
    if ids is None or 25 not in ids.ravel():
        return None
    c = corners[list(ids.ravel()).index(25)].reshape(-1, 2)
    cv2.cornerSubPix(img, c, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01))
    return c


def board_corners(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    ok, pts = cv2.findChessboardCornersSB(img, (11, 8), flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    return pts.reshape(-1, 2) if ok else None


class Problem:
    def __init__(self, views, K, dist):
        self.views, self.K, self.dist = views, K, dist  # views: (q, target 'tag'|'board', obj, pts)

    def unpack(self, x):
        dq = np.zeros(6)
        dq[OFF_JOINTS] = x[18:21]
        return to_T(x[0:3], x[3:6]), {"tag": to_T(x[6:9], x[9:12]), "board": to_T(x[12:15], x[15:18])}, dq

    def view_resid(self, x, v):
        T_gc, T_t, dq = self.unpack(x)
        q, tgt, obj, pts = v
        T = np.linalg.inv(ik.fk(q + dq) @ T_gc) @ T_t[tgt]
        proj, _ = cv2.projectPoints(obj, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], self.K, self.dist)
        return (proj.reshape(-1, 2) - pts).ravel()

    def resid(self, x, idx):
        return np.concatenate([self.view_resid(x, self.views[i]) for i in idx])


def init_target(views, tgt, T_gc, K, dist):
    Ts = []
    for q, t, obj, pts in views:
        if t != tgt:
            continue
        ok, rv, tv = cv2.solvePnP(obj, pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE if t == "tag" else 0)
        Ts.append(ik.fk(q) @ T_gc @ to_T(rv.ravel(), tv.ravel()))
    T = np.median(np.stack(Ts), axis=0)
    U, _, Vt = np.linalg.svd(T[:3, :3])
    T[:3, :3] = U @ Vt
    return T


def main():
    cal = json.load(open(sys.argv[1]))
    w = cal["wrist"]
    K, dist, T_gc0 = np.array(w["K"]), np.array(w["dist"]), np.array(w["T_gripperframe_cam"])
    views = []
    for e in snaps(sys.argv[2]):
        c = tag_corners(f"snaps/{e['name']}_wrist.png")
        if c is not None:
            views.append((ik.raw_to_q(e["raw"]), "tag", TAG_OBJ, c))
    n_tag = len(views)
    for path in sys.argv[4:]:
        for e in snaps(path):
            c = board_corners(f"snaps/{e['name']}_wrist.png")
            if c is not None:
                views.append((ik.raw_to_q(e["raw"]), "board", BOARD_OBJ, c))
    print(f"views: tag {n_tag}, board {len(views) - n_tag}")
    T_tag0, T_board0 = init_target(views, "tag", T_gc0, K, dist), init_target(views, "board", T_gc0, K, dist)
    x0 = np.r_[logr(T_gc0[:3, :3]), T_gc0[:3, 3], logr(T_tag0[:3, :3]), T_tag0[:3, 3],
               logr(T_board0[:3, :3]), T_board0[:3, 3], np.zeros(3)]
    pb = Problem(views, K, dist)
    allv = list(range(len(views)))

    def fit(idx, free_offsets):
        if free_offsets:
            f = lambda x: pb.resid(x, idx)
            return least_squares(f, x0, loss="huber", f_scale=2.0, x_scale="jac", max_nfev=300)
        f = lambda x: pb.resid(np.r_[x, 0, 0, 0], idx)
        s = least_squares(f, x0[:18], loss="huber", f_scale=2.0, x_scale="jac", max_nfev=300)
        s.x = np.r_[s.x, 0, 0, 0]
        return s

    def rms(x, idx):
        r = pb.resid(x, idx).reshape(-1, 2)
        return float(np.sqrt(np.mean(np.sum(r ** 2, axis=1))))

    def heights(x):
        _, T_t, _ = pb.unpack(x)
        out = {}
        for t, obj in (("tag", TAG_OBJ), ("board", BOARD_OBJ)):
            z = (T_t[t][:3, :3] @ obj.T).T[:, 2] + T_t[t][2, 3]
            out[t] = (round(float(z.min()) * 1000, 1), round(float(z.max()) * 1000, 1))
        return out

    rng = np.random.default_rng(0)
    hold = sorted(rng.choice(len(views), size=max(4, len(views) // 5), replace=False).tolist())
    train = [i for i in allv if i not in hold]
    report = {}
    for name, free in (("rigid", False), ("joint_zeros", True)):
        s_tr = fit(train, free)
        s = fit(allv, free)
        J = s.jac
        sd = np.sqrt(np.diag(np.linalg.pinv(J.T @ J) * (2 * s.cost / max(1, J.shape[0] - J.shape[1]))))
        report[name] = {"x": s.x, "rms_px": rms(s.x, allv), "heldout_rms_px": rms(s_tr.x, hold), "heights_mm": heights(s.x),
                        "offsets_deg": np.degrees(s.x[18:21]), "offsets_sd_deg": np.degrees(sd[18:21]) if free else None}
        r = report[name]
        print(f"{name:12s} rms {r['rms_px']:.2f} px | held-out rms {r['heldout_rms_px']:.2f} px | heights mm {r['heights_mm']}"
              + ("" if not free else f" | offsets lift/elbow/wrist deg {np.round(r['offsets_deg'], 2)}"
                 f" +- {np.round(r['offsets_sd_deg'], 2)}"))

    # physical check decides: the paper tag lies on the table the base stands on, so its corners must be at z ~ 0.
    # A lower pixel error that puts the tag centimetres into the table means the offsets absorbed something else.
    tag_z = {k: max(abs(z) for z in v["heights_mm"]["tag"]) for k, v in report.items()}
    choice = "joint_zeros" if tag_z["joint_zeros"] <= 8 and tag_z["joint_zeros"] <= tag_z["rigid"] else "rigid"
    print(f"chosen: {choice} (tag max |z| mm: rigid {tag_z['rigid']}, joint_zeros {tag_z['joint_zeros']})")
    T_gc, T_t, dq = pb.unpack(report[choice]["x"])
    jm = json.loads(json.dumps(ik.MAP))
    if choice == "joint_zeros":
        for qi, name in zip(OFF_JOINTS, ("shoulder_lift", "elbow_flex", "wrist_flex")):
            j = jm["joints"][name]
            old = j["raw_at_mjcf_zero"]
            j["raw_at_mjcf_zero"] = round(old - j["sign"] * dq[qi] / j["rad_per_step"], 1)
            j["zero_calibration"] = (f"raw_at_mjcf_zero {old} -> {j['raw_at_mjcf_zero']} ({np.degrees(dq[qi]):+.2f} deg) "
                                     f"from {n_tag} AprilTag + {len(views) - n_tag} checkerboard views (tools/calib_joints.py)")
    json.dump(jm, open(sys.argv[3] + "_joint_map.json", "w"), indent=1)
    cam = json.loads(json.dumps(cal))
    cam["wrist"]["T_gripperframe_cam"] = T_gc.tolist()
    cam["wrist"]["hand_eye_method"] = f"tools/calib_joints.py ({choice}): {n_tag} AprilTag + {len(views) - n_tag} board views"
    cam["tag"] = {"family": "36h11", "id": 25, "size_m": TAG, "T_base_tag": T_t["tag"].tolist()}
    cam["board"]["T_base_board"] = T_t["board"].tolist()
    cam["joint_calibration"] = {"chosen": choice, **{k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                                                          for kk, vv in v.items() if kk != "x"} for k, v in report.items()}}
    json.dump(cam, open(sys.argv[3] + "_cameras.json", "w"), indent=1)
    print("wrote", sys.argv[3] + "_joint_map.json", sys.argv[3] + "_cameras.json")


if __name__ == "__main__":
    main()

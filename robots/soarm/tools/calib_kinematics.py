"""Kinematic calibration of the real SO-101 (armhost) from wrist-camera views of flat targets on the table.

usage: calib_kinematics.py fit <cameras.json> --tag LOG.. [--board LOG..] [--hold-far] [--out kin.json]
       calib_kinematics.py eval <kin.json> <cameras.json> --tag LOG..     (prediction error of a saved fit)
Model: the Menagerie robotstudio_so101 chain (shoulder -> upper_arm -> lower_arm -> wrist -> gripper -> gripperframe)
with per-body frame corrections (translation t and small rotation r, in the nominal body frame, applied before the
body's hinge), zero offsets of shoulder_lift / elbow_flex / wrist_flex, optional gravity compliance, plus free
hand-eye (T_gripperframe_cam), AprilTag pose and checkerboard pose. Gaussian priors on every kinematic parameter;
tag corners on the table plane z = 0 (sigma 1 mm); board horizontal (it lay on a ~1 cm glass plate, z free).
Unobservable directions are not parameters: pan zero, roll zero (hand-eye absorbs it), gripper z shift along the
roll axis, rotations about the hinge axes of upper_arm/lower_arm/wrist (= zero offsets), z shifts of lower_arm/wrist
(add up with the upper_arm z shift because the three hinges are parallel).
"""
import argparse
import json
import os
import sys

import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
MJCF = os.environ.get("SOARM_MJCF", os.path.expandvars("$SOARM_ROOT/menagerie/robotstudio_so101/so101.xml"))
_MAP = os.path.join(HERE, "..", "configs", "hardware", "so101_armhost_joint_map.json")
MAP = json.load(open(_MAP if os.path.exists(_MAP) else os.path.join(HERE, "so101_armhost_joint_map.json")))
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CHAIN = ["shoulder", "upper_arm", "lower_arm", "wrist", "gripper"]  # body k carries hinge q[k] about its local z

TAG = 0.150
TAG_OBJ = np.array([[-TAG / 2, TAG / 2, 0], [TAG / 2, TAG / 2, 0], [TAG / 2, -TAG / 2, 0], [-TAG / 2, -TAG / 2, 0]])
SQ = 0.030
BOARD_OBJ = np.zeros((88, 3))
BOARD_OBJ[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2) * SQ
DET = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                              cv2.aruco.DetectorParameters())

MM, DEG = 1e-3, np.pi / 180
# (name, body index in CHAIN or None, component, prior sigma). t* in m, r* in rad (axis-angle about the body frame).
KIN = [("shoulder.rx", 0, "rx", 2 * DEG), ("shoulder.ry", 0, "ry", 2 * DEG), ("shoulder.tz", 0, "tz", 5 * MM),
       ("upper_arm.tx", 1, "tx", 5 * MM), ("upper_arm.ty", 1, "ty", 5 * MM), ("upper_arm.tz", 1, "tz", 5 * MM),
       ("upper_arm.rx", 1, "rx", 2 * DEG), ("upper_arm.ry", 1, "ry", 2 * DEG),
       ("lower_arm.tx", 2, "tx", 5 * MM), ("lower_arm.ty", 2, "ty", 5 * MM),
       ("lower_arm.rx", 2, "rx", 2 * DEG), ("lower_arm.ry", 2, "ry", 2 * DEG),
       ("wrist.tx", 3, "tx", 5 * MM), ("wrist.ty", 3, "ty", 5 * MM),
       ("wrist.rx", 3, "rx", 2 * DEG), ("wrist.ry", 3, "ry", 2 * DEG),
       ("gripper.tx", 4, "tx", 5 * MM), ("gripper.ty", 4, "ty", 5 * MM),
       ("gripper.rx", 4, "rx", 2 * DEG), ("gripper.ry", 4, "ry", 2 * DEG),
       ("zero.shoulder_lift", 1, "zero", 5 * DEG), ("zero.elbow_flex", 2, "zero", 5 * DEG),
       ("zero.wrist_flex", 3, "zero", 5 * DEG),
       # gravity compliance: q_eff = q + c * tau_gravity(q) (rad per N m, tau from the nominal model)
       ("comp.shoulder_lift", 1, "comp", 0.03), ("comp.elbow_flex", 2, "comp", 0.03),
       ("comp.wrist_flex", 3, "comp", 0.03)]
KIN_NAMES = [k[0] for k in KIN]


def raw_to_q(raw):
    """Encoder raw -> nominal MJCF q (joint map as measured, before any kinematic correction)."""
    return np.array([MAP["joints"][n]["sign"] * (raw[str(MAP["joints"][n]["id"])] - MAP["joints"][n]["raw_at_mjcf_zero"])
                     * MAP["joints"][n]["rad_per_step"] for n in NAMES])


def rodrigues(r):
    return cv2.Rodrigues(np.asarray(r, float).reshape(3))[0]


def to_T(r, t):
    T = np.eye(4)
    T[:3, :3] = rodrigues(r)
    T[:3, 3] = t
    return T


def logr(R):
    return cv2.Rodrigues(np.asarray(R, float))[0].ravel()


def quat_R(qt):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(qt, float))
    return R.reshape(3, 3)


def R_quat(R):
    qt = np.zeros(4)
    mujoco.mju_mat2Quat(qt, np.asarray(R, float).ravel())
    return qt


class Chain:
    """Batched FK of the gripperframe site with body-frame corrections; nominal geometry from the MJCF."""

    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(MJCF)
        self.d = mujoco.MjData(self.m)
        self.bid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n) for n in CHAIN]
        for k, b in enumerate(self.bid):
            j = self.m.body_jntadr[b]
            assert self.m.jnt_qposadr[j] == k and np.allclose(self.m.jnt_axis[j], [0, 0, 1]) \
                and np.allclose(self.m.jnt_pos[j], 0), CHAIN[k]
        self.pos = self.m.body_pos[self.bid].copy()
        self.R = np.stack([quat_R(self.m.body_quat[b]) for b in self.bid])
        sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.site = to_T(np.zeros(3), self.m.site_pos[sid])
        self.site[:3, :3] = quat_R(self.m.site_quat[sid])

    def gravity(self, Q):
        """Gravity torque at each joint (N m) for each row of Q (nominal model, zero velocity)."""
        out = np.zeros((len(Q), self.m.nv))
        for i, q in enumerate(Q):
            self.d.qpos[:] = q
            self.d.qvel[:] = 0
            mujoco.mj_forward(self.m, self.d)
            out[i] = self.d.qfrc_bias
        return out

    def corrections(self, p):
        """Per-body (t, R) corrections and q offsets from the KIN parameter vector p."""
        t, r = np.zeros((5, 3)), np.zeros((5, 3))
        dz, comp = np.zeros(6), np.zeros(6)
        for v, (_, k, c, _) in zip(p, KIN):
            if c[0] == "t":
                t[k, "xyz".index(c[1])] = v
            elif c[0] == "r":
                r[k, "xyz".index(c[1])] = v
            elif c == "zero":
                dz[k] = v
            else:
                comp[k] = v
        return t, np.stack([rodrigues(x) for x in r]), dz, comp

    def fk(self, Q, p, G):
        """(N,4,4) gripperframe poses for joint rows Q, KIN params p, gravity torques G."""
        t, Rc, dz, comp = self.corrections(p)
        Qe = Q + dz + comp * G
        N = len(Q)
        T = np.tile(np.eye(4), (N, 1, 1))
        for k in range(5):
            B = np.eye(4)
            B[:3, :3] = self.R[k] @ Rc[k]
            B[:3, 3] = self.pos[k] + self.R[k] @ t[k]
            c, s = np.cos(Qe[:, k]), np.sin(Qe[:, k])
            J = np.tile(np.eye(4), (N, 1, 1))
            J[:, 0, 0], J[:, 0, 1], J[:, 1, 0], J[:, 1, 1] = c, -s, s, c
            T = T @ B @ J
        return T @ self.site

    def corrected_bodies(self, p):
        """Corrected body_pos / body_quat for each CHAIN body (for writing into an MjModel)."""
        t, Rc, _, _ = self.corrections(p)
        return {n: {"pos": (self.pos[k] + self.R[k] @ t[k]).tolist(), "quat": R_quat(self.R[k] @ Rc[k]).tolist()}
                for k, n in enumerate(CHAIN)}


def project(Pc, K, dist):
    """Pinhole + OpenCV radial/tangential distortion, Pc (..., 3) in camera frame -> (..., 2) pixels."""
    k1, k2, p1, p2, k3 = (list(dist) + [0] * 5)[:5]
    x, y = Pc[..., 0] / Pc[..., 2], Pc[..., 1] / Pc[..., 2]
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]], -1)


def tag_corners(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
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


def charuco_corners(path):
    """Board-0 ChArUco corners (flatcal.board_px) as (obj, px), or None."""
    import flatcal  # flatcal imports this module
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    d = flatcal.board_px(img) if img is not None else {}
    return (flatcal.OBJ[d[0][0]], d[0][1]) if 0 in d else None


def load_views(tag_logs, board_logs, charuco_logs=(), snapdir="snaps", skip=()):
    views = []
    for tgt, logs, det, obj in (("tag", tag_logs, tag_corners, TAG_OBJ), ("board", board_logs, board_corners, BOARD_OBJ),
                                ("charuco", charuco_logs, charuco_corners, None)):
        for path in logs:
            for e in map(json.loads, open(path)):
                if e.get("event") != "snap" or not e.get("ok"):
                    continue
                c = det(f"{snapdir}/{e['name']}_wrist.png")
                if tgt == "charuco" and c is not None:
                    views.append({"name": e["name"], "log": path, "raw": e["raw"], "q": raw_to_q(e["raw"]),
                                  "tgt": tgt, "obj": c[0], "pts": c[1].astype(float)})
                    continue
                if c is not None:
                    # tag corners next to the gripper edge (x < 535 px) or the image border are often clipped / shaded
                    mask = np.ones(len(c), bool) if tgt == "board" else \
                        (c[:, 0] >= 535) & (c[:, 0] <= 1235) & (c[:, 1] >= 45) & (c[:, 1] <= 675)
                    for prefix, k in skip:  # tag corners a block stands next to (it hides them from some views)
                        if tgt == "tag" and path.startswith(prefix):
                            mask[k] = False
                    if mask.sum() >= 3:
                        views.append({"name": e["name"], "log": path, "raw": e["raw"], "q": raw_to_q(e["raw"]),
                                      "tgt": tgt, "obj": obj[mask], "pts": c[mask].astype(float)})
    return views


SIG_PX = 1.0                   # tag corner (cornerSubPix) noise
SIG_PX_BOARD = np.sqrt(88 / 16)  # board corners are correlated within a view: one board view ~ 4 tag views
SIG_TAG_Z = 1 * MM             # tag paper on the table plane
SIG_BOARD_TILT = 1 * DEG       # board on a flat glass plate on the table


class Problem:
    """x = [KIN params (free subset), hand-eye r t, tag r t, board r t]."""

    def __init__(self, chain, views, K, dist, free=None):
        self.ch, self.views, self.K, self.dist = chain, views, K, dist
        self.free = np.array([True] * len(KIN) if free is None else free)
        self.nk = int(self.free.sum())
        self.Q = np.stack([v["q"] for v in views])
        self.G = chain.gravity(self.Q)
        self.prior = np.array([k[3] for k in KIN])

    def unpack(self, x):
        p = np.zeros(len(KIN))
        p[self.free] = x[:self.nk]
        o = self.nk
        return p, to_T(x[o:o + 3], x[o + 3:o + 6]), {"tag": to_T(x[o + 6:o + 9], x[o + 9:o + 12]),
                                                     "board": to_T(x[o + 12:o + 15], x[o + 15:o + 18]),
                                                     "charuco": to_T(x[o + 18:o + 21], x[o + 21:o + 24])}

    def pack(self, p, T_gc, T_t):
        return np.r_[p[self.free], logr(T_gc[:3, :3]), T_gc[:3, 3], logr(T_t["tag"][:3, :3]), T_t["tag"][:3, 3],
                     logr(T_t["board"][:3, :3]), T_t["board"][:3, 3],
                     logr(T_t["charuco"][:3, :3]), T_t["charuco"][:3, 3]]

    def cams(self, x, idx):
        p, T_gc, T_t = self.unpack(x)
        return self.ch.fk(self.Q[idx], p, self.G[idx]) @ T_gc, T_t

    def pix_resid(self, x, idx):
        """Per-view pixel residual arrays (n_pts, 2) for views idx."""
        T_bc, T_t = self.cams(x, idx)
        out = []
        for T, i in zip(T_bc, idx):
            v = self.views[i]
            Tct = np.linalg.inv(T) @ T_t[v["tgt"]]
            Pc = v["obj"] @ Tct[:3, :3].T + Tct[:3, 3]
            out.append(project(Pc, self.K, self.dist) - v["pts"])
        return out

    def resid(self, x, idx):
        p, _, T_t = self.unpack(x)
        rs = [r.ravel() / (SIG_PX if self.views[i]["tgt"] == "tag" else SIG_PX_BOARD)
              for r, i in zip(self.pix_resid(x, idx), idx)]
        zt = TAG_OBJ @ T_t["tag"][:3, :3].T[:, 2] + T_t["tag"][2, 3]
        rs += [p[self.free] / self.prior[self.free], zt / SIG_TAG_Z, T_t["board"][:2, 2] / SIG_BOARD_TILT,
               T_t["charuco"][:2, 2] / SIG_BOARD_TILT]  # the printed sheet lies flat on the table
        return np.concatenate(rs)

    def plane_err_mm(self, x, idx):
        """Per-view corner errors (mm) on the target plane: observed pixel rays from the predicted camera pose,
        intersected with the fitted target plane, vs the true corner positions."""
        T_bc, T_t = self.cams(x, idx)
        out = []
        for T, i in zip(T_bc, idx):
            v = self.views[i]
            Ttc = np.linalg.inv(T_t[v["tgt"]]) @ T
            n = cv2.undistortPoints(v["pts"].reshape(-1, 1, 2), self.K, self.dist).reshape(-1, 2)
            rays = np.c_[n, np.ones(len(n))] @ Ttc[:3, :3].T
            s = -Ttc[2, 3] / rays[:, 2]
            P = Ttc[:3, 3] + s[:, None] * rays
            out.append(np.linalg.norm(P[:, :2] - v["obj"][:, :2], axis=1) / MM)
        return out


def init_targets(views, T_gc, K, dist, chain, p):
    out = {}
    for tgt in ("tag", "board", "charuco"):
        Ts = []
        for v in views:
            if v["tgt"] != tgt or len(v["pts"]) < 4:
                continue
            ok, rv, tv = cv2.solvePnP(v["obj"], v["pts"], K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE if tgt == "tag" else 0)
            G = chain.gravity(v["q"][None])
            Ts.append(chain.fk(v["q"][None], p, G)[0] @ T_gc @ to_T(rv.ravel(), tv.ravel()))
        if not Ts:
            out[tgt] = np.eye(4)
            continue
        T = np.median(np.stack(Ts), axis=0)
        U, _, Vt = np.linalg.svd(T[:3, :3])
        T[:3, :3] = U @ Vt
        out[tgt] = T
    return out


def solve(pb, x0, idx):
    s = least_squares(lambda x: pb.resid(x, idx), x0, loss="soft_l1", f_scale=3.0, x_scale="jac", max_nfev=400)
    J = s.jac
    cov = np.linalg.pinv(J.T @ J)
    s.sd = np.sqrt(np.diag(cov))
    s.corr = cov / np.outer(s.sd, s.sd)
    return s


def summary(pb, x, idx):
    if not idx:
        return {}
    r = np.concatenate(pb.pix_resid(x, idx))
    e = np.concatenate(pb.plane_err_mm(x, idx))
    return {"n_views": len(idx), "rms_px": float(np.sqrt(np.mean(np.sum(r ** 2, 1)))),
            "plane_err_mm_mean": float(e.mean()), "plane_err_mm_p90": float(np.percentile(e, 90)),
            "plane_err_mm_max": float(e.max())}


def heights_mm(pb, x):
    _, _, T_t = pb.unpack(x)
    out = {}
    import flatcal
    for t, obj in (("tag", TAG_OBJ), ("board", BOARD_OBJ), ("charuco", flatcal.OBJ)):
        z = obj @ T_t[t][:3, :3].T[:, 2] + T_t[t][2, 3]
        out[t] = [round(float(z.min()) / MM, 1), round(float(z.max()) / MM, 1)]
    return out


MODELS = {"rigid": [], "zeros": [n for n in KIN_NAMES if n.startswith("zero.")],
          "geom": [n for n in KIN_NAMES if not n.startswith("comp.")], "full": KIN_NAMES}
# "full" fitted on 73 tag + 38 board views (2026-11-12): upper_arm.tx duplicates shoulder.tz (both shift the arm along
# the pan axis), upper_arm.tz ~ gripper.ty (lateral shift of the planar chain), and upper_arm.rx, lower_arm.tx, wrist.tx,
# gripper.ry kept posterior sd > 0.8 x prior sd (data cannot see them; lower_arm.tx also correlates 0.99 with the lift
# zero). "final" drops these six.
UNOBSERVABLE = ["upper_arm.tx", "gripper.ty", "upper_arm.rx", "lower_arm.tx", "wrist.tx", "gripper.ry"]
MODELS["final"] = [n for n in KIN_NAMES if n not in UNOBSERVABLE]


def fit_model(chain, views, K, dist, T_gc0, names, train, test):
    free = [n in names for n in KIN_NAMES]
    pb = Problem(chain, views, K, dist, free)
    p0 = np.zeros(len(KIN))
    tv = [views[i] for i in train]
    x0 = pb.pack(p0, T_gc0, init_targets(tv, T_gc0, K, dist, chain, p0))
    s = solve(pb, x0, train)
    return pb, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cams")
    ap.add_argument("--tag", nargs="*", default=[])
    ap.add_argument("--board", nargs="*", default=[])
    ap.add_argument("--charuco", nargs="*", default=[], help="logs of wrist views of ChArUco board 0 (calib_low.py)")
    ap.add_argument("--hold", default=None, help="python expr over r (raw dict, int keys as str): held-out cluster")
    ap.add_argument("--hold-log", default=None, help="hold out every view of this log (e.g. a half of the low views)")
    ap.add_argument("--models", default="rigid,zeros,geom,full,final")
    ap.add_argument("--skip-corner", nargs="*", default=[], help="LOGPREFIX=IDX: drop tag corner IDX in those logs")
    ap.add_argument("--choose", default=None, help="model whose fit on ALL views is written to --out")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cal = json.load(open(a.cams))
    w = cal["wrist"]
    K, dist, T_gc0 = np.array(w["K"]), np.array(w["dist"]), np.array(w["T_gripperframe_cam"])
    chain = Chain()
    views = load_views(a.tag, a.board, a.charuco, skip=[(x.split("=")[0], int(x.split("=")[1])) for x in a.skip_corner])
    allv = list(range(len(views)))
    hold = [i for i in allv if (a.hold and eval(a.hold, {}, {"r": {k: int(v) for k, v in views[i]["raw"].items()}}))
            or views[i]["log"] == a.hold_log]
    train = [i for i in allv if i not in hold]
    print(f"views: {sum(v['tgt'] == 'tag' for v in views)} tag, {sum(v['tgt'] == 'board' for v in views)} board, "
          f"{sum(v['tgt'] == 'charuco' for v in views)} charuco; "
          f"held out {len(hold)} ({a.hold})")
    res = {}
    for name in a.models.split(","):
        pb, s = fit_model(chain, views, K, dist, T_gc0, MODELS[name], train, hold)
        r = {"train": summary(pb, s.x, [i for i in train if views[i]["tgt"] == "tag"]),
             "train_board": summary(pb, s.x, [i for i in train if views[i]["tgt"] == "board"]),
             "train_charuco": summary(pb, s.x, [i for i in train if views[i]["tgt"] == "charuco"]),
             "heldout": summary(pb, s.x, hold), "heights_mm": heights_mm(pb, s.x)}
        if hold:  # per held-out view
            r["heldout_views"] = {views[i]["name"]: round(float(e.max()), 2) for i, e in zip(hold, pb.plane_err_mm(s.x, hold))}
        worst = sorted(((float(e.max()), views[i]["name"]) for i, e in zip(train, pb.plane_err_mm(s.x, train))),
                       reverse=True)[:6]
        r["worst_train_views_mm"] = {n: round(e, 2) for e, n in worst}
        res[name] = r
        f = lambda d: (f"{d['rms_px']:.2f}px {d['plane_err_mm_mean']:.2f}/{d['plane_err_mm_max']:.2f}mm" if d else "-")
        print(f"{name:6s} train tag {f(r['train'])} | board {f(r['train_board'])} | charuco {f(r['train_charuco'])} "
              f"| held {f(r['heldout'])} | z {r['heights_mm']} | worst train {r['worst_train_views_mm']}", flush=True)
    if a.choose and a.out:
        names = MODELS[a.choose]
        pb, s = fit_model(chain, views, K, dist, T_gc0, names, allv, [])
        p, T_gc, T_t = pb.unpack(s.x)
        sd = np.zeros(len(KIN))
        sd[pb.free] = s.sd[:pb.nk]
        params = {n: {"value": float(p[k]), "sd": float(sd[k]), "prior_sd": KIN[k][3],
                      "unit": "m" if KIN[k][2][0] == "t" else ("rad/Nm" if KIN[k][2] == "comp" else "rad")}
                  for k, n in enumerate(KIN_NAMES) if n in names}
        fn = [n for n in KIN_NAMES if n in names]
        pairs = [(fn[i], fn[j], round(float(s.corr[i, j]), 3)) for i in range(len(fn)) for j in range(i)
                 if abs(s.corr[i, j]) > 0.95]
        print("strongly correlated parameter pairs (|r| > 0.95):", pairs)
        for n, v in params.items():
            print(f"  {n:20s} {v['value'] / (MM if v['unit'] == 'm' else (DEG if v['unit'] == 'rad' else 1)):+8.3f} "
                  f"+- {v['sd'] / (MM if v['unit'] == 'm' else (DEG if v['unit'] == 'rad' else 1)):.3f} "
                  f"({'mm' if v['unit'] == 'm' else ('deg' if v['unit'] == 'rad' else 'rad/Nm')}, "
                  f"sd/prior {v['sd'] / v['prior_sd']:.2f})")
        out = {"schema_version": "captain-so101-kinematics/v1", "mjcf": "robotstudio_so101/so101.xml (Menagerie)",
               "model": a.choose, "bodies": chain.corrected_bodies(p),
               "joint_zero_offset_rad": {n: float(p[KIN_NAMES.index("zero." + n)]) if "zero." + n in names else 0.0
                                         for n in NAMES[1:4]},
               "compliance_rad_per_Nm": {n: float(p[KIN_NAMES.index("comp." + n)]) if "comp." + n in names else 0.0
                                         for n in NAMES[1:4]},
               "params": params, "correlated_pairs": pairs, "T_gripperframe_cam": T_gc.tolist(), "T_base_tag": T_t["tag"].tolist(),
               "T_base_board": T_t["board"].tolist(), "T_base_charuco": T_t["charuco"].tolist(),
               "heights_mm": heights_mm(pb, s.x),
               "fit_all": {"tag": summary(pb, s.x, [i for i in allv if views[i]["tgt"] == "tag"]),
                           "board": summary(pb, s.x, [i for i in allv if views[i]["tgt"] == "board"]),
                           "charuco": summary(pb, s.x, [i for i in allv if views[i]["tgt"] == "charuco"])},
               "data": {"tag_logs": a.tag, "board_logs": a.board, "charuco_logs": a.charuco, "n_views": len(views)},
               "validation": {"holdout_expr": a.hold, **res},
               "method": "tools/calib_kinematics.py: " + __doc__.split("\n")[3].strip()}
        json.dump(out, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()

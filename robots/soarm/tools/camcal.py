"""Camera calibration for the real SO-101 rig from arm_probe snapshots (run with a venv that has mujoco + opencv).

usage: camcal.py <arm_probe.log> [<more.log> ...] <out.json>
Views = the log's "snap" events ({"name", "raw"}); images snaps/<name>_wrist.png and snaps/<name>_fixed.png.

Board: 12x9-square checkerboard, 11x8 inner corners, 30 mm squares (asymmetric, so the corner order is unambiguous).
  The square size is checked against the arm: FK is metric, so the fit can estimate it (2026-11-12: 29.62 mm; with
  25 mm assumed the board landed 6 cm above the table and the chain error was 5.3 px instead of 1.5 px).
Solves
  wrist camera intrinsics (all wrist views),
  T_gripper_cam (eye-in-hand: Park-Martin AX=XB init, then a joint reprojection fit with T_base_board),
  T_base_board and its spread across views,
  fixed camera: intrinsics from its own views when they vary, else the wrist intrinsics (same camera model 1e45:8022),
  T_base_fixedcam = T_base_board * inv(T_fixedcam_board).
Frames: base = MJCF robot base (x forward at pan 0, z up), gripper = MJCF gripperframe site (x = approach).
"""
import json
import sys

import cv2
import numpy as np

import so101_ik as ik

PATTERN = (11, 8)
SQUARE = 0.030
OBJ = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
OBJ[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE


def corners(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    ok, pts = cv2.findChessboardCornersSB(img, PATTERN, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    return (pts.reshape(-1, 2) if ok else None), img.shape[::-1]


def to_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.ravel(t)
    return T


def pnp(pts, K, dist):
    ok, rvec, tvec = cv2.solvePnP(OBJ, pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    proj, _ = cv2.projectPoints(OBJ, rvec, tvec, K, dist)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - pts) ** 2, axis=1))))
    return to_T(cv2.Rodrigues(rvec)[0], tvec), err


def rot_deg(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def intrinsics(pts_list, size):
    rms, K, dist, _, _ = cv2.calibrateCamera([OBJ] * len(pts_list), [p.astype(np.float32) for p in pts_list],
                                             size, None, None, flags=cv2.CALIB_FIX_K3)
    return float(rms), K, dist


def _log_rot(R):
    return cv2.Rodrigues(R)[0].ravel()


def park_hand_eye(T_bg, T_cb):
    """Park & Martin closed form for AX = XB (X = T_gripper_cam) over all view pairs."""
    A, B = [], []
    for i in range(len(T_bg)):
        for j in range(i + 1, len(T_bg)):
            A.append(np.linalg.inv(T_bg[i]) @ T_bg[j])
            B.append(T_cb[i] @ np.linalg.inv(T_cb[j]))
    M = sum(np.outer(_log_rot(b[:3, :3]), _log_rot(a[:3, :3])) for a, b in zip(A, B))
    w, V = np.linalg.eigh(M.T @ M)
    R = V @ np.diag(w ** -0.5) @ V.T @ M.T
    C = np.vstack([a[:3, :3] - np.eye(3) for a in A])
    d = np.concatenate([R @ b[:3, 3] - a[:3, 3] for a, b in zip(A, B)])
    t = np.linalg.lstsq(C, d, rcond=None)[0]
    return to_T(R, t)


def refine_hand_eye(T_gc0, T_bg, T_cb, pts_list, K, dist):
    """Jointly fit T_gripper_cam and T_base_board (12 params) minimizing corner reprojection through FK."""
    from scipy.optimize import least_squares

    T_board0 = np.median(np.stack([Tb @ T_gc0 @ Tc for Tb, Tc in zip(T_bg, T_cb)]), axis=0)
    U, _, Vt = np.linalg.svd(T_board0[:3, :3])  # median of rotations is not a rotation: re-orthonormalize
    T_board0[:3, :3] = U @ Vt

    def unpack(x):
        return (to_T(cv2.Rodrigues(x[0:3])[0], x[3:6]), to_T(cv2.Rodrigues(x[6:9])[0], x[9:12]))

    def resid(x):
        T_gc, T_board = unpack(x)
        out = []
        for Tb, p in zip(T_bg, pts_list):
            T = np.linalg.inv(Tb @ T_gc) @ T_board
            proj, _ = cv2.projectPoints(OBJ, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, dist)
            out.append((proj.reshape(-1, 2) - p).ravel())
        return np.concatenate(out)

    x0 = np.concatenate([_log_rot(T_gc0[:3, :3]), T_gc0[:3, 3], _log_rot(T_board0[:3, :3]), T_board0[:3, 3]])
    sol = least_squares(resid, x0, loss="huber", f_scale=2.0, x_scale="jac")
    return unpack(sol.x)


def fit_square(T_gc, T_board, T_bg, pts_list, K, dist):
    """Square size that best explains the views given FK (metric): the check on the assumed SQUARE."""
    from scipy.optimize import least_squares

    unit = OBJ / SQUARE

    def resid(x):
        Tg, Tb_, sq = to_T(cv2.Rodrigues(x[0:3])[0], x[3:6]), to_T(cv2.Rodrigues(x[6:9])[0], x[9:12]), x[12]
        out = []
        for Tb, p in zip(T_bg, pts_list):
            T = np.linalg.inv(Tb @ Tg) @ Tb_
            proj, _ = cv2.projectPoints(unit * sq, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, dist)
            out.append((proj.reshape(-1, 2) - p).ravel())
        return np.concatenate(out)

    x0 = np.r_[_log_rot(T_gc[:3, :3]), T_gc[:3, 3], _log_rot(T_board[:3, :3]), T_board[:3, 3], SQUARE]
    return float(least_squares(resid, x0, max_nfev=400).x[12])


def main():
    # arm_probe logs: every "snap" event carries the measured joint positions at the moment of the shot
    views = [e for path in sys.argv[1:-1] for e in map(json.loads, open(path))
             if e.get("event") == "snap" and e.get("ok")]
    wrist, fixed = [], []
    for v in views:
        pw, size_w = corners(f"snaps/{v['name']}_wrist.png")
        pf, size_f = corners(f"snaps/{v['name']}_fixed.png")
        if pw is not None:
            wrist.append((v, pw))
        if pf is not None:
            fixed.append((v, pf))
    print(f"board found: wrist {len(wrist)}/{len(views)}, fixed {len(fixed)}/{len(views)}")
    if len(wrist) < 6:
        raise SystemExit("need >= 6 wrist views with the full board")

    rms_w, Kw, dw = intrinsics([p for _, p in wrist], size_w)
    print(f"wrist intrinsics rms {rms_w:.3f} px  fx {Kw[0, 0]:.1f} fy {Kw[1, 1]:.1f} cx {Kw[0, 2]:.1f} cy {Kw[1, 2]:.1f}")

    T_bg, T_cb, names = [], [], []
    for v, p in wrist:
        T_bg.append(ik.fk(ik.raw_to_q(v["raw"])))
        T_cb.append(pnp(p, Kw, dw)[0])
        names.append(v["name"])
    T_gc, T_board = refine_hand_eye(park_hand_eye(T_bg, T_cb), T_bg, T_cb, [p for _, p in wrist], Kw, dw)
    method = "park_martin_init+joint_reprojection_ls"
    T_bb = [Tb @ T_gc @ Tc for Tb, Tc in zip(T_bg, T_cb)]  # board pose seen independently from each view
    pos = np.array([T[:3, 3] for T in T_bb])
    med = T_board[:3, 3]
    pos_err = np.linalg.norm(pos - med, axis=1) * 1000
    rot_err = [rot_deg(T[:3, :3], T_board[:3, :3]) for T in T_bb]
    # reprojection through the whole chain: board in base -> FK -> hand-eye -> wrist camera
    chain = []
    for (v, p), Tb in zip(wrist, T_bg):
        T_cb_pred = np.linalg.inv(Tb @ T_gc) @ T_board
        rvec = cv2.Rodrigues(T_cb_pred[:3, :3])[0]
        proj, _ = cv2.projectPoints(OBJ, rvec, T_cb_pred[:3, 3], Kw, dw)
        chain.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - p) ** 2, axis=1)))))
    print(f"hand-eye ({method}): camera in gripperframe t = {np.round(T_gc[:3, 3] * 1000, 1)} mm")
    print(f"board in base: {np.round(med * 1000, 1)} mm; per-view spread median {np.median(pos_err):.1f} mm "
          f"max {pos_err.max():.1f} mm, rot median {np.median(rot_err):.2f} deg max {max(rot_err):.2f} deg")
    print(f"chain reprojection (FK + hand-eye + board) median {np.median(chain):.1f} px max {max(chain):.1f} px")
    sq_fit = fit_square(T_gc, T_board, T_bg, [p for _, p in wrist], Kw, dw)
    print(f"square size fitted against FK: {sq_fit * 1000:.2f} mm (assumed {SQUARE * 1000:.0f} mm)")

    out = {"pattern_inner_corners": list(PATTERN), "square_m": SQUARE,
           "wrist": {"device": "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:8:1.0-video-index0", "image_size": list(size_w),
                     "K": Kw.tolist(), "dist": dw.ravel().tolist(), "rms_px": rms_w, "n_views": len(wrist),
                     "T_gripperframe_cam": T_gc.tolist(), "hand_eye_method": method},
           "board": {"T_base_board": T_board.tolist(), "square_fit_mm": sq_fit * 1000,
                     "spread_mm_median": float(np.median(pos_err)),
                     "spread_mm_max": float(pos_err.max()), "spread_deg_median": float(np.median(rot_err)),
                     "chain_reproj_px_median": float(np.median(chain)), "chain_reproj_px_max": float(max(chain)),
                     "per_view": {n: {"mm": float(e), "deg": float(r), "chain_px": float(c)}
                                  for n, e, r, c in zip(names, pos_err, rot_err, chain)}}}

    if fixed:
        size_f = cv2.imread(f"snaps/{fixed[0][0]['name']}_fixed.png").shape[1::-1]
        moved = np.ptp(np.array([p for _, p in fixed]), axis=0).max() if len(fixed) > 1 else 0.0
        if moved > 20 and len(fixed) >= 6:
            rms_f, Kf, df = intrinsics([p for _, p in fixed], size_f)
            src = f"own views (rms {rms_f:.3f} px)"
        else:
            Kf, df, src = Kw, dw, "wrist intrinsics (same camera model; board static in the fixed view)"
        errs, Ts = [], []
        for _, p in fixed:
            T_fb, e = pnp(p, Kf, df)
            Ts.append(T_board @ np.linalg.inv(T_fb))
            errs.append(e)
        T_bf = Ts[int(np.argmin(errs))]
        print(f"fixed camera: intrinsics from {src}; PnP reprojection {np.median(errs):.2f} px; "
              f"camera at {np.round(T_bf[:3, 3] * 1000, 1)} mm in base, looking along {np.round(T_bf[:3, 2], 2)}")
        out["fixed"] = {"device": "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:12:1.0-video-index0",
                        "image_size": list(size_f), "K": Kf.tolist(), "dist": df.ravel().tolist(),
                        "intrinsics_source": src, "pnp_reproj_px": float(np.median(errs)),
                        "T_base_cam": T_bf.tolist(), "n_views": len(fixed)}
    json.dump(out, open(sys.argv[-1], "w"), indent=1)
    print("wrote", sys.argv[-1])


if __name__ == "__main__":
    main()

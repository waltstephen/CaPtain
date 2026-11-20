"""Low, grasp-like wrist views of the ChArUco board for the kinematic calibration (calib_kinematics.py --charuco).

The 11-12 kinematic fit has no data with the gripper near the table, where pick & place grasps (11-13: two grasps
missed by ~1.5 cm in opposite directions with a fixed camera good to ~1 mm). These views put the gripper 15-60 mm
above the table, approach axis within ~35 deg of straight down, over the workspace in front of the arm, with >= 12
ChArUco corners of board 0 in the usable wrist image. The board pose for planning comes from the fixed camera
(flatcal calibration); the fit treats it as a free target.

usage (venv-srv, cwd $SOARM_ROOT/armcal):
  calib_low.py <cameras.json> '<rest raw json>' <fixed.png> <plan_out.json> <prefix> [n_views=24]
Safety (sim, corrected model, mesh vertices): views keep the gripper >= 15 mm and the arm >= 50 mm above the table,
inflated boxes around the colored blocks seen by the fixed camera, the flatcal shelf keep-out; each transfer goes up
to a clear pose 80 mm higher first (joint order checked in sim, every joint move sampled). Ends at rest, torque on.
"""
import json
import sys

import cv2
import numpy as np

import calib_views as V
import flatcal as F
import pickplace as P
import so101_ik as ik
from calib_kinematics import project


def board_pose(cams, img):
    det = F.board_px(img)
    if 0 not in det:
        raise SystemExit("REFUSE the fixed camera does not see board 0")
    ids, px = det[0]
    f = cams["fixed"]
    ok, rv, tv = cv2.solvePnP(F.OBJ[ids], px, np.array(f["K"]), np.array(f["dist"]))
    return np.array(f["T_base_cam"]) @ F.to_T(rv, tv)


def seen(sc, q, T_bb):
    T = np.linalg.inv(ik.fk(q) @ sc.T_gc) @ T_bb
    Pc = F.OBJ @ T[:3, :3].T + T[:3, 3]
    if (Pc[:, 2] < 0.06).any():
        return 0
    uv = project(Pc, sc.K, sc.dist)
    return int(((uv[:, 0] >= 520) & (uv[:, 0] <= 1240) & (uv[:, 1] >= 40) & (uv[:, 1] <= 680)).sum())


def main():
    cams, rest = json.load(open(sys.argv[1])), F.raw_json(sys.argv[2])
    img, out, prefix = cv2.imread(sys.argv[3]), sys.argv[4], sys.argv[5]
    n = int(sys.argv[6]) if len(sys.argv) > 6 else 24
    T_bb = board_pose(cams, img)
    blocks = []
    f = cams["fixed"]
    for c, _ in P.color_blobs(img):
        blocks.append(P.px_to_plane(c.mean(0)[None], np.array(f["K"]), np.array(f["dist"]),
                                    np.array(f["T_base_cam"]), 0.02)[0][:2])
    print(f"board 0 center {np.round((T_bb @ np.r_[F.OBJ.mean(0), 1])[:3] * 1000)} mm; blocks "
          f"{[np.round(b * 1000).tolist() for b in blocks]}")
    sc = F.Scene(cams, blocks)
    V.GRIP_Z = 0.015  # views only: gripper may come down to 15 mm (the board is paper)
    rng = np.random.default_rng(2)
    cands = []
    for _ in range(40000):
        r = {k: int(v) for k, v in zip("12345", rng.uniform([1600, 2300, 1300, 2400, 1955],
                                                              [2600, 3000, 2600, 3500, 3963]))}
        q = V.raw_q(r)
        try:
            ik.q_to_raw(q)
        except ValueError:
            continue
        T = ik.fk(q)
        if T[2, 0] > -0.8 or P.lowest_gripper_z(q) > 0.06:
            continue
        if seen(sc, q, T_bb) >= 12 and sc.clearance(q) > 0.0:
            cands.append(r)
    views = V.pick_diverse(cands, n, [])
    print(f"{len(cands)} candidate low views, {len(views)} picked")
    V.GRIP_Z = 0.030
    steps, cur, k = [], {j: rest[j] for j in "12345"}, 0
    left = list(views)
    while left:
        left.sort(key=lambda v: sum(abs(v[j] - cur[j]) / (900 if j == "5" else 300) for j in "12345"))
        v = left.pop(0)
        # above the view: same pan / roll, lift and elbow raised until the gripper is 80 mm higher
        q_v = V.raw_q(v)
        q_up, _, _ = ik.look_ik(ik.fk(q_v)[:3, 3] + [0, 0, 0.08], ik.fk(q_v)[:3, 3] + [0, 0, -0.3], q_v,
                                roll=q_v[4], n=12)
        try:
            up = {str(a): b for a, b in ik.q_to_raw(q_up).items() if a != 6}
        except ValueError:
            continue
        up["1"], up["5"] = v["1"], v["5"]
        if sc.clearance(V.raw_q(up)) <= 0:
            continue
        mg, order = V.best_path(sc, cur, up)
        if order is None or mg <= 0:
            continue
        V.GRIP_Z = 0.015
        mg_d, down = V.best_path(sc, up, v)  # the descent at the views' floor; the ascent retraces it
        V.GRIP_Z = 0.030
        if down is None or mg_d <= 0:
            continue
        steps += F.moves(order, up)
        steps += F.moves(down, v) + [{"snap": f"{prefix}_{k:02d}"}]
        steps += F.moves(list(reversed(down)), up)
        cur, k = dict(up), k + 1
    if k < 6:
        raise SystemExit(f"REFUSE only {k} low views reachable")
    steps += F.transfer(sc, cur, {j: rest[j] for j in "12345"}, "return to rest")
    print(f"{k} low views planned, {len(steps)} steps")
    json.dump({"torque_limit": 400, "speed": 100, "steps": steps, "return": False, "release": False}, open(out, "w"))


if __name__ == "__main__":
    main()

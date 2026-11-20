"""Corner touch test: put the gripper's lowest vertex a few mm above a known point of the AprilTag and measure, in the
wrist image, how far the real point is from where the model predicts it.

usage: SOARM_KINEMATICS=<kin.json> calib_touch.py plan <kin.json> <cameras.json> <spec.json> <plan_out.json>
       SOARM_KINEMATICS=<kin.json> calib_touch.py measure <kin.json> <cameras.json> <log> [<log> ...]
spec = {"blocks_xy": [...], "rest": {...}, "prefix": "t1", "h": 0.004,
        "targets": [{"cell": [col, row], "lift": 2800 | [..], "roll": 3945 | [..], "name": "far_a"}, ...]}
Targets are vertices of the tag's cell grid (col, row in 0..8 from the top-left corner of the black square, 18.75 mm
cells; [0, 0] .. [8, 8] are the outer corners). The pose is solved with shoulder_lift fixed at the requested raw
(far / low vs mid configurations reach the same point differently), the descent is done in 1 cm steps, every pose and
transfer is checked in sim (calib_views rules; during the descent the gripper may go down to 1 mm above the paper within 6 cm of
the target (10 cm: the gripper housing is long), other arm vertices stay >= 30 mm up, the wrist camera housing >= 20 mm; blocks: 3 cm / 55 mm).
Measure: render the tag as the model predicts it (undistorted image), register it to the real wrist image with ECC
(homography, all visible tag pixels right of the gripper); the registered target pixel's ray meets the tag plane at the
real target position; error = its distance to the target. Views where the tag is seen edge-on are not measurable.
"""
import json
import sys

import cv2
import numpy as np

import calib_views as cv_
import so101_ik as ik
from calib_kinematics import TAG, project

CELL = TAG / 8
TOUCH_BLOCK_MARGIN, TOUCH_BLOCK_Z = 0.03, 0.055  # touches: within 3 cm of a block (positions good to ~1 cm) stay
# >= 15 mm above its top
MARKER = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), 25, 8)


def cell_point(T_base_tag, col, row):
    p = np.array([-TAG / 2 + col * CELL, TAG / 2 - row * CELL, 0.0])
    return T_base_tag[:3, :3] @ p + T_base_tag[:3, 3]


def lowest_vertex(q):
    """Lowest gripper/jaw mesh vertex (base frame) at q."""
    ik.set_q(q)
    best = None
    for g, grip, v in cv_._GEOMS:
        if not grip:
            continue
        P = v @ ik.d.geom_xmat[g].reshape(3, 3).T + ik.d.geom_xpos[g]
        i = int(np.argmin(P[:, 2]))
        if best is None or P[i, 2] < best[2]:
            best = P[i]
    return best


def touch_clearance(sc, q, target, h, why=None):
    """Like Scene.clearance, but the gripper may be down to 1 mm above the paper within 10 cm (xy) of the target.
    why: optional list that receives (margin, body, rule) for every rule."""
    ik.set_q(q)
    worst = 1.0
    for g, grip, v in cv_._GEOMS:
        P = v @ ik.d.geom_xmat[g].reshape(3, 3).T + ik.d.geom_xpos[g]
        cam = ik.m.body(int(ik.m.geom_bodyid[g])).name == "camera_mount"
        near = np.linalg.norm(P[:, :2] - target[:2], axis=1) < 0.10
        floor = np.where(near, target[2] + 0.001, cv_.GRIP_Z) if grip else np.full(len(P), 0.020 if cam else 0.030)
        ms = [((P[:, 2] - floor).min(), "floor"),
              (np.linalg.norm(P[:, :2] - sc.stand, axis=1).min() - cv_.STAND_R, "stand")]
        for k, b in enumerate(sc.blocks):
            dxy = np.abs(P[:, :2] - b).max(1) - (cv_.BLOCK_HALF + TOUCH_BLOCK_MARGIN)
            ms.append((np.maximum(dxy, P[:, 2] - TOUCH_BLOCK_Z).min(), f"block{k}"))
        for mg, rule in ms:
            worst = min(worst, mg)
            if why is not None:
                why.append((round(float(mg) * 1000, 1), ik.m.body(int(ik.m.geom_bodyid[g])).name, rule))
    return float(worst)


def target_px(sc, q, p):
    T = np.linalg.inv(ik.fk(q) @ sc.T_gc)
    Pc = T[:3, :3] @ p + T[:3, 3]
    return project(Pc[None], sc.K, sc.dist)[0] if Pc[2] > 0.03 else None


def solve(sc, target, lift, h, roll=3945):
    """Raw pose with shoulder_lift = lift whose lowest gripper vertex is h above target and the target is visible
    (right of the gripper in the wrist image). Grid seed over pan / elbow / wrist_flex, then coordinate descent."""
    goal = target + [0, 0, h]

    def cost(r):
        q = cv_.raw_q(r)
        c = np.linalg.norm(lowest_vertex(q) - goal)
        uv = target_px(sc, q, target)
        if uv is None:
            return c + 1.0
        T_bc = ik.fk(q) @ sc.T_gc
        ray = target - T_bc[:3, 3]
        dist = np.linalg.norm(ray)
        incid = np.degrees(np.arccos(min(1.0, abs(ray[2]) / dist)))  # 0 = looking straight down at the target
        c += 1e-3 * max(0.0, (0.09 - dist) * 100) + 1e-3 * max(0.0, incid - 60)  # measurable: not too close / edge-on
        return (c + 2e-5 * max(0.0, cv_.U_MIN - uv[0]) + 2e-5 * max(0.0, abs(uv[1] - 360) - 300)
                + 1e-5 * max(0.0, abs(r["2"] - lift) - 150))

    pan0 = round(2048 + np.arctan2(target[1], target[0]) / ik.MAP["joints"]["shoulder_pan"]["rad_per_step"])
    best = None
    for dp in (-60, 0, 60):
        for el in range(1000, 2400, 100):
            for wr in range(1900, 3400, 100):
                r = {"1": pan0 + dp, "2": lift, "3": el, "4": wr, "5": roll}
                c = cost(r)
                if best is None or c < best[0]:
                    best = (c, r)
    c, r = best
    for step in (32, 16, 8, 4, 2, 1):
        improved = True
        while improved:
            improved = False
            for j in "1234":
                for s in (-step, step):
                    rr = dict(r, **{j: r[j] + s})
                    if not in_limits(rr):
                        continue
                    cc = cost(rr)
                    if cc < c:
                        c, r, improved = cc, rr, True
    return r, c


def in_limits(r):
    j = ik.MAP["joints"]
    return all(j[n]["soft_limit_raw"]["min"] is None or j[n]["soft_limit_raw"]["min"] <= r[k] <= j[n]["soft_limit_raw"]["max"]
               for k, n in zip("12345", ik.NAMES))


def refine(sc, target, r, h):
    """Continuity solve: from pose r, coordinate descent (pan / elbow / wrist_flex) to the lowest vertex h above,
    inside the soft limits (2026-11-12: an unchecked rung asked the elbow for raw 746 and it stalled on its stop)."""
    goal = target + [0, 0, h]
    c = np.linalg.norm(lowest_vertex(cv_.raw_q(r)) - goal)
    for step in (16, 8, 4, 2, 1):
        improved = True
        while improved:
            improved = False
            for j in "134":
                for s in (-step, step):
                    rr = dict(r, **{j: r[j] + s})
                    if not in_limits(rr):
                        continue
                    cc = np.linalg.norm(lowest_vertex(cv_.raw_q(rr)) - goal)
                    if cc < c:
                        c, r, improved = cc, rr, True
    return r, c


def seg_margin(sc, a, b, target, h, n=6):
    """Joint-by-joint (order 1, 3, 4: small local moves) touch clearance from raw a to raw b."""
    cur, worst = dict(a), 1.0
    for j in "134":
        x0, x1 = cur[j], b[j]
        for t in np.linspace(0, 1, n)[1:]:
            worst = min(worst, touch_clearance(sc, cv_.raw_q(dict(cur, **{j: round(x0 + t * (x1 - x0))})), target, h))
        cur[j] = x1
    return worst


def plan(kin, cams, spec, out):
    cv_.BLOCK_XY_MARGIN, cv_.BLOCK_Z = TOUCH_BLOCK_MARGIN, TOUCH_BLOCK_Z  # transfers of a touch run: same block rule
    sc = cv_.Scene(kin, cams, spec.get("blocks_xy", []))
    T_tag = np.array(kin["T_base_tag"])
    h = spec.get("h", 0.004)
    rest = {k: int(v) for k, v in spec["rest"].items()}
    steps, cur, meta = [], {k: rest[k] for k in "12345"}, []
    mv = lambda r, order, sp: [{"id": int(j), "goal": r[j], "torque_limit": cv_.TL[j], "return_after": False,
                                "speed": sp} for j in order]
    for t in spec["targets"]:
        p = cell_point(T_tag, *t["cell"])
        rolls = t.get("roll", 3945)
        lifts = t["lift"] if isinstance(t["lift"], list) else [t["lift"]]
        for lift, roll in [(lf, rl) for lf in lifts for rl in (rolls if isinstance(rolls, list) else [rolls])]:
            bot, err = solve(sc, p, lift, h, roll)  # first (lift, roll) whose touch is clear wins
            if err > 0.002:
                print(f"{t['name']} lift {lift} roll {roll}: no pose (err {err * 1000:.1f} mm)")
                continue
            ladder = [bot]
            for hh in (0.005, 0.010, 0.020, 0.030, 0.045, 0.060):
                ladder.append(refine(sc, p, dict(ladder[-1]), h + hh)[0])
            hover = ladder[-1]
            if not all(in_limits(r) for r in ladder):
                print(f"{t['name']} lift {lift} roll {roll}: outside soft limits")
                continue
            mg, order = cv_.best_path(sc, cur, hover)
            segs = [seg_margin(sc, ladder[k + d], ladder[k + 1 - d], p, h)
                    for k in range(len(ladder) - 1) for d in (0, 1)]
            static = touch_clearance(sc, cv_.raw_q(bot), p, h)
            uv = target_px(sc, cv_.raw_q(bot), p)
            ray = p - (ik.fk(cv_.raw_q(bot)) @ sc.T_gc)[:3, 3]
            look = (np.linalg.norm(ray), np.degrees(np.arccos(abs(ray[2]) / np.linalg.norm(ray))))
            print(f"{t['name']} lift {lift} roll {roll}: cell {t['cell']} at {np.round(p * 1000, 1)} mm, bottom raw "
                  f"{[bot[j] for j in '12345']}, transfer margin {mg * 1000:.0f} mm ({''.join(order or '')}), descent "
                  f"margin {min(segs) * 1000:.1f} mm, bottom {static * 1000:.1f} mm, target px "
                  f"{None if uv is None else np.round(uv)}, camera {look[0] * 1000:.0f} mm at {look[1]:.0f} deg")
            if look[0] < 0.08 or look[1] > 65:
                continue
            if mg > 0 and min(segs) > 0 and static > 0 and uv is not None:
                break
        else:
            print("  skipped (no clear / visible roll)")
            continue
        steps += mv(hover, order, 120)
        name = f"{spec['prefix']}_{t['name']}"
        steps.append({"snap": name + "_hover"})
        for r in reversed(ladder[:-1]):
            steps += mv(r, "134", 60)
        steps.append({"snap": name})
        for r in ladder[1:]:
            steps += mv(r, "134", 60)
        meta.append({"name": name, "cell": t["cell"], "point": p.tolist(), "bottom_raw": bot})
        cur = dict(hover)
    mg, order = cv_.best_path(sc, cur, {k: rest[k] for k in "12345"})
    if mg <= 0:
        raise SystemExit(f"no clear return to rest from {cur}")
    steps += mv(rest, order, 120)
    json.dump({"torque_limit": 400, "speed": 100, "steps": steps, "return": True, "release": True}, open(out, "w"))
    json.dump(meta, open(out.replace(".json", "_meta.json"), "w"), indent=1)
    print(f"{len(meta)} touches, {len(steps)} steps -> {out}")


def measure_one(sc, T_tag, img, raw, cell):
    """(error mm on the tag plane, observed px, predicted px, px per mm) for one snapshot, or None."""
    p = cell_point(T_tag, *cell)
    q = ik.raw_to_q({**{k: int(v) for k, v in raw.items()}})
    T_bc = ik.fk(q) @ sc.T_gc
    Tcb = np.linalg.inv(T_bc)
    und = cv2.undistort(img, sc.K, sc.dist)
    g = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)
    # render the tag (with its white quiet zone) in the undistorted image: homography tag plane -> pixels
    m = cv2.resize(MARKER, (800, 800), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((1200, 1200), 255, np.uint8)
    canvas[200:1000, 200:1000] = m
    s = TAG / 800  # m per canvas pixel; canvas (u, v) -> tag (x, y) = ((u - 600) s, (600 - v) s)
    src = np.array([[0, 0], [1200, 0], [1200, 1200], [0, 1200]], float)
    P = np.c_[(src[:, 0] - 600) * s, (600 - src[:, 1]) * s, np.zeros(4)]
    Pb = P @ T_tag[:3, :3].T + T_tag[:3, 3]
    Pc = Pb @ Tcb[:3, :3].T + Tcb[:3, 3]
    if (Pc[:, 2] <= 0).any():
        return None
    dst = (Pc[:, :2] / Pc[:, 2:]) @ sc.K[:2, :2].T + sc.K[:2, 2]
    H = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
    rend = cv2.warpPerspective(canvas, H, (g.shape[1], g.shape[0]), borderValue=200)
    pc = Tcb[:3, :3] @ p + Tcb[:3, 3]
    up = pc[:2] / pc[2] * np.diag(sc.K)[:2] + sc.K[:2, 2]
    # register the rendered tag to the real (undistorted) image with ECC (homography), using every visible tag pixel
    # right of the gripper; the registered position of the predicted target pixel is where the target really is
    valid = cv2.warpPerspective(np.full((1200, 1200), 255, np.uint8), H, (g.shape[1], g.shape[0]))
    valid[:, :cv_.U_MIN] = 0
    valid = cv2.erode(valid, np.ones((9, 9), np.uint8))
    if valid.sum() / 255 < 20000:
        return None
    a = cv2.GaussianBlur(rend, (0, 0), 3).astype(np.float32)
    b = cv2.GaussianBlur(g, (0, 0), 3).astype(np.float32)
    W = np.eye(3, dtype=np.float32)
    try:
        score, W = cv2.findTransformECC(a, b, W, cv2.MOTION_HOMOGRAPHY,
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 300, 1e-6), valid, 5)
    except cv2.error:
        return None
    o = W @ np.r_[up, 1.0]
    obs = o[:2] / o[2]
    ray = Tcb[:3, :3].T @ np.r_[(obs - sc.K[:2, 2]) / np.diag(sc.K)[:2], 1.0]
    o = T_bc[:3, 3]
    n, z0 = T_tag[:3, 2], T_tag[:3, 3]
    hit = o + ray * ((z0 - o) @ n / (ray @ n))
    p1 = Tcb[:3, :3] @ (p + T_tag[:3, 0] * 0.001) + Tcb[:3, 3]
    ppm = np.linalg.norm(p1[:2] / p1[2] * np.diag(sc.K)[:2] - pc[:2] / pc[2] * np.diag(sc.K)[:2])
    return float(np.linalg.norm(hit - p) * 1000), obs, up, float(ppm), float(score)


def measure(kin, cams, logs, meta):
    sc = cv_.Scene(kin, cams, [])
    T_tag = np.array(kin["T_base_tag"])
    cells = {m["name"]: m["cell"] for m in meta}
    out = []
    for path in logs:
        for e in map(json.loads, open(path)):
            if e.get("event") != "snap":
                continue
            base = e["name"][:-6] if e["name"].endswith("_hover") else e["name"]
            if base not in cells:
                continue
            img = cv2.imread(f"snaps/{e['name']}_wrist.png")
            r = measure_one(sc, T_tag, img, {k: v for k, v in e["raw"].items()}, cells[base])
            if r is None:
                print(f"{e['name']}: target not measurable")
                continue
            err, obs, up, ppm, score = r
            vis = img.copy()
            cv2.drawMarker(vis, tuple(int(v) for v in up), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
            cv2.circle(vis, tuple(int(v) for v in obs), 12, (0, 255, 0), 2)
            cv2.imwrite(f"snaps/{e['name']}_meas.jpg", vis)
            print(f"{e['name']}: raw {[e['raw'][k] for k in '12345']} error {err:.1f} mm ({np.linalg.norm(obs - up):.0f} px,"
                  f" {ppm:.1f} px/mm, match {score:.2f})")
            out.append({"name": e["name"], "raw": e["raw"], "cell": cells[base], "error_mm": err,
                        "px": float(np.linalg.norm(obs - up)), "px_per_mm": ppm, "match": score})
    return out


if __name__ == "__main__":
    kin, cams = json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
    if sys.argv[1] == "plan":
        plan(kin, cams, json.load(open(sys.argv[4])), sys.argv[5])
    elif sys.argv[1] == "measure":
        logs = sys.argv[4:]
        meta = []
        for lg in logs:
            mp = lg.replace(".log", "_meta.json")
            meta += json.load(open(mp)) if __import__("os").path.exists(mp) else []
        print(json.dumps(measure(kin, cams, logs, meta)))

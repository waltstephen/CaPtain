"""Top-down pick & place planner for the real SO-101 on armhost (fixed camera + camera calibration + MuJoCo FK).

usage (venv with mujoco + opencv, cwd = dir holding pp/, snaps/ and the calibration json):
  pickplace.py hover <scene.png> <red|yellow|blue|any> <cameras.json> '<start raw json>' <plan_out.json>
  pickplace.py approach <cameras.json> '<raw json>' <wrist.png> <plan_out.json>
  pickplace.py grasp <cameras.json> '<raw json>' <plan_out.json>
Finds a colored block in the fixed camera (saturated pixels; the black/white tag and the white arm are ignored),
re-locates the fixed camera from the AprilTag, flies above the block, re-locates it with the wrist camera, then plans
the top-down grasp (roll closes the jaws across the narrow axis), a place spot clear of the tag and other blocks, and
the return to rest. run_pickplace.sh chains the stages.
"""
import json
import sys

import cv2
import mujoco
import numpy as np

import so101_ik as ik

TABLE_Z = 0.0         # table plane in the FK frame: corrected kinematics (so101_armhost_kinematics.json) put the tag at 0
OBJ_H = 0.04          # assumed object height (the 4 cm toy blocks): top-face plane for the wrist camera
CLEAR = 0.006         # lowest gripper mesh this far above the table at the grasp
HOVER = 0.08          # pre-grasp / pre-place height of the lowest gripper mesh above the object top / table
GRIP_OPEN = 2650      # raw; closed on itself = 2027
GRIP_CLOSE = 1950     # raw goal when closing (the object or the jaws stop it first; torque-limited)


HUES = {"red": (0, 10, 170, 180), "yellow": (18, 35), "blue": (95, 130)}  # OpenCV hue ranges (0-180)


def color_blobs(scene, min_area=300, max_area=40000):
    """Saturated blobs (the toy blocks): list of (contour px, median hue). Tag (black/white) and arm (white) drop out;
    thin red/black servo cables are removed by the opening."""
    hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 1] > 110) & (hsv[..., 2] > 60)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        if min_area < cv2.contourArea(c) < max_area:
            mask = np.zeros(m.shape, np.uint8)
            cv2.fillPoly(mask, [c], 1)
            out.append((c.reshape(-1, 2).astype(float), float(np.median(hsv[..., 0][mask > 0]))))
    return out


def _hue_match(h, name):
    r = HUES[name]
    return any(r[i] <= h <= r[i + 1] for i in range(0, len(r), 2))


def detect(scene, color="any"):
    """The block to pick: largest saturated blob of the requested color. Returns (contour px, all blob contours)."""
    blobs = color_blobs(scene)
    cands = [c for c, h in blobs if color == "any" or _hue_match(h, color)]
    if not cands:
        raise SystemExit(f"no {color} block found")
    return max(cands, key=lambda c: cv2.contourArea(c.astype(np.float32))), [c for c, _ in blobs]


def fixed_pose_from_tag(scene, cal):
    """Re-locate the fixed camera from the AprilTag (pose in the base frame from the kinematic calibration), so a
    moved fixed camera does not break the coarse localization. Intrinsics stay borrowed: coarse only."""
    tag = cal.get("tag")
    if tag is None:
        return False
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                                  cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY))
    if ids is None or tag["id"] not in ids.ravel():
        return False
    s = tag["size_m"]
    obj = np.array([[-s / 2, s / 2, 0], [s / 2, s / 2, 0], [s / 2, -s / 2, 0], [-s / 2, -s / 2, 0]])
    f = cal["fixed"]
    ok, rv, tv = cv2.solvePnP(obj, corners[list(ids.ravel()).index(tag["id"])].reshape(-1, 2), np.array(f["K"]),
                              np.array(f["dist"]), flags=cv2.SOLVEPNP_IPPE_SQUARE)
    T_ct = np.eye(4)
    T_ct[:3, :3], T_ct[:3, 3] = cv2.Rodrigues(rv)[0], tv.ravel()
    f["T_base_cam"] = (np.array(tag["T_base_tag"]) @ np.linalg.inv(T_ct)).tolist()
    return True


def px_to_plane(uv, K, dist, T_base_cam, z):
    """Intersect the camera rays of pixels uv (N,2) with the horizontal plane z (base frame)."""
    n = cv2.undistortPoints(uv.reshape(-1, 1, 2).astype(np.float64), K, dist).reshape(-1, 2)
    rays = (T_base_cam[:3, :3] @ np.c_[n, np.ones(len(n))].T).T
    o = T_base_cam[:3, 3]
    s = (z - o[2]) / rays[:, 2]
    return o + s[:, None] * rays


def lowest_gripper_z(q):
    """Lowest vertex of the gripper and moving-jaw meshes (what touches the table first)."""
    ik.set_q(q)
    zs = []
    for g in range(ik.m.ngeom):
        if ik.m.body(int(ik.m.geom_bodyid[g])).name in ("gripper", "moving_jaw_so101_v1") \
                and ik.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = ik.m.geom_dataid[g]
            v = ik.m.mesh_vert[ik.m.mesh_vertadr[mid]: ik.m.mesh_vertadr[mid] + ik.m.mesh_vertnum[mid]]
            zs.append(((ik.d.geom_xmat[g].reshape(3, 3) @ v.T).T + ik.d.geom_xpos[g])[:, 2].min())
    return float(min(zs))


ROLL_RANGE = (0.06, 3.14)  # wrist_roll raw 1955..3963 (zero raw 1915 since 11-13); a stop just above 3963, free below


def grip_q(raw):
    j = ik.MAP["joints"]["gripper"]
    return (raw - j["raw_at_mjcf_zero"]) * j["rad_per_step"]


def grip_local(width):
    """Grip point in gripperframe-site coordinates (soarm/twin.py): 0.8 cm behind the tips, z = -0.002; wider objects
    shift so the fixed jaw (inner face at site z = -0.020) clears the object edge by 4 mm on the way down."""
    return np.array([-0.008, 0.0, max(-0.002, -0.020 + width / 2 + 0.004)])


GRIP_LOCAL = grip_local(0.03)  # set per object in main()


def _cost(q, target, close_dir):
    T = ik.fk(q)
    p = T[:3, :3] @ GRIP_LOCAL + T[:3, 3]
    c = T[:3, 2].copy()  # closing axis = gripperframe z
    c[2] = 0.0
    yaw = 1.0 - abs(c @ close_dir) / (np.linalg.norm(c) + 1e-9) if close_dir is not None else 0.0
    return np.linalg.norm(p - target) + 0.05 * (1.0 + T[2, 0]) + 0.02 * yaw


def _refine(q, target, close_dir, joints=(0, 1, 2, 3, 4), step0=(0.04, 0.05, 0.05, 0.08, 0.15)):
    q = q.copy()
    best = _cost(q, target, close_dir)
    step = np.array(step0, float)
    lo, hi = ik.LO.copy(), ik.HI.copy()
    lo[4], hi[4] = ROLL_RANGE
    for _ in range(9):
        for _ in range(40):
            improved = False
            for k, j in enumerate(joints):
                for s in (-1, 1):
                    qq = q.copy()
                    qq[j] = np.clip(qq[j] + s * step[k], lo[j], hi[j])
                    c = _cost(qq, target, close_dir)
                    if c < best:
                        best, q, improved = c, qq, True
            if not improved:
                break
        step /= 2
    return q


def topdown_ik(target, close_dir, grip):
    """gripperframe at target, approach straight down, closing axis along close_dir (horizontal unit vector)."""
    az = np.arctan2(target[1], target[0])
    best = None
    for pn in np.linspace(az - 0.3, az + 0.3, 7):
        for l in np.linspace(-0.6, 1.4, 21):
            for e in np.linspace(-1.4, 1.4, 25):
                for w in np.linspace(0.6, 1.66, 9):
                    q = np.array([pn, l, e, w, 0.0, grip])
                    c = _cost(q, target, None)
                    if best is None or c < best[0]:
                        best = (c, q)
    q = _refine(best[1], target, None, joints=(0, 1, 2, 3))
    if close_dir is None:
        return q
    # the closing-axis yaw has a basin per roll; seed the whole roll range so the refinement can reach any of them
    seeds = [np.r_[q[:4], r, q[5]] for r in np.linspace(ROLL_RANGE[0], ROLL_RANGE[1], 13)]
    sols = [_refine(s, target, close_dir) for s in seeds]
    return min(sols, key=lambda s: _cost(s, target, close_dir))


WORKSPACE = {"r": (0.12, 0.42), "x_min": 0.05, "y_abs": 0.30}  # table area the gripper may work in (base frame, m)
IK_TOL = 0.005  # refuse any pose whose grip point misses its target by more than this (no silent clamping)


def check_workspace(xy, what):
    r = float(np.hypot(*xy[:2]))
    if not (WORKSPACE["r"][0] <= r <= WORKSPACE["r"][1] and xy[0] >= WORKSPACE["x_min"]
            and abs(xy[1]) <= WORKSPACE["y_abs"]):
        raise SystemExit(f"REFUSED: {what} at {np.round(np.asarray(xy[:2]) * 1000)} mm is outside the table workspace")


def at_height(q, xy, low_z, close_dir):
    """Solve so the lowest gripper mesh sits at low_z with the grip point over xy, starting from q (continuity).
    Refuses (exits, no plan) when the target is outside the workspace or the solution misses it by > IK_TOL."""
    check_workspace(xy, "target")
    for _ in range(3):
        T = ik.fk(q)
        grip_minus_low = (T[:3, :3] @ GRIP_LOCAL + T[:3, 3])[2] - lowest_gripper_z(q)
        q = _refine(q, np.r_[xy, low_z + grip_minus_low], close_dir, step0=(0.02, 0.03, 0.03, 0.04, 0.05))
    T = ik.fk(q)
    g = T[:3, :3] @ GRIP_LOCAL + T[:3, 3]
    err_xy, err_z = float(np.linalg.norm(g[:2] - np.asarray(xy)[:2])), abs(lowest_gripper_z(q) - low_z)
    if err_xy > IK_TOL or err_z > IK_TOL:
        raise SystemExit(f"REFUSED: IK misses target {np.round(np.asarray(xy[:2]) * 1000)} mm by "
                         f"{err_xy * 1000:.1f} mm (xy) / {err_z * 1000:.1f} mm (z)")
    return q


def object_on_table(contour_px, cal):
    """Object center, narrow (closing) axis and width/length in the base frame from its fixed-camera blob."""
    f = cal["fixed"]
    K, dist, T_bc = np.array(f["K"]), np.array(f["dist"]), np.array(f["T_base_cam"])
    P = px_to_plane(contour_px, K, dist, T_bc, TABLE_Z + OBJ_H / 2)[:, :2]
    center = P.mean(0)
    w, V = np.linalg.eigh(np.cov((P - center).T))
    narrow = np.r_[V[:, 0], 0.0]  # smallest spread = the direction to close the jaws along (horizontal, base frame)
    ext = (P - center) @ V
    return center, narrow, float(np.ptp(ext[:, 0])), float(np.ptp(ext[:, 1]))


def tag_poly_xy(cal, margin=0.03):
    """Tag paper outline (black square + margin) in the base frame: never place on it, the tag must stay put."""
    T, h = np.array(cal["tag"]["T_base_tag"]), cal["tag"]["size_m"] / 2 + margin
    corners = np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]])
    return (T[:3, :3] @ corners.T).T[:, :2] + T[:2, 3]


def choose_place(center, cal, scene, contour, blobs):
    """A free table spot 8-12 cm from the pick: reachable, off the tag, on white table, clear of the other blocks."""
    f = cal["fixed"]
    K, dist, T_bc = np.array(f["K"]), np.array(f["dist"]), np.array(f["T_base_cam"])
    poly = tag_poly_xy(cal).astype(np.float32)
    gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    occupied = np.zeros(gray.shape, np.uint8)
    for b in blobs:
        if b is not contour:
            cv2.fillPoly(occupied, [b.astype(np.int32)], 1)
    r0 = np.hypot(*center)
    best = None
    for dist_m in (0.10, 0.08, 0.12):
        for ang in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            p = center + dist_m * np.array([np.cos(ang), np.sin(ang)])
            r = np.hypot(*p)
            if not 0.18 <= r <= 0.36 or cv2.pointPolygonTest(poly, tuple(map(float, p)), False) >= 0:
                continue
            Pc = np.linalg.inv(T_bc) @ np.r_[p, TABLE_Z, 1.0]
            uv, _ = cv2.projectPoints(Pc[None, :3], np.zeros(3), np.zeros(3), K, dist)
            u, v = uv.ravel().astype(int)
            if not (60 <= u < gray.shape[1] - 60 and 60 <= v < gray.shape[0] - 60):
                continue
            if gray[v - 25:v + 25, u - 25:u + 25].mean() < 150:  # dark = off the table / shadow
                continue
            if occupied[v - 55:v + 55, u - 55:u + 55].any():  # ~5 cm clear of every other block
                continue
            score = abs(r - r0)
            if best is None or score < best[0]:
                best = (score, p, (u, v))
    if best is None:
        raise SystemExit("no free place spot found")
    return best[1], best[2]


def arm_raw(q):
    r = ik.q_to_raw(np.r_[q[:5], grip_q(GRIP_OPEN)])
    r.pop(6)
    return {str(k): v for k, v in r.items()}


def body_min_z_path(raw_a, raw_b, order):
    """Lowest point (gripper meshes and arm bodies) along a joint-by-joint move a -> b in the given joint order."""
    cur, worst = dict(raw_a), 9.0
    for j in order:
        a, b = cur[j], raw_b[j]
        for t in np.linspace(0, 1, 12):
            r = dict(cur, **{j: round(a + t * (b - a))})
            q = ik.raw_to_q({**r, "6": GRIP_OPEN})
            worst = min(worst, lowest_gripper_z(q), ik.body_min_z(q))
        cur[j] = b
    return worst


def best_order(raw_a, raw_b):
    from itertools import permutations
    js = [j for j in "12345" if raw_a[j] != raw_b[j]]
    return max(permutations(js), key=lambda o: body_min_z_path(raw_a, raw_b, o)) if js else ()


SIZE_RANGE = (0.025, 0.060)  # plausible block footprint side (m): a cable, a reflection or half a block falls outside
MAX_DEV = 0.06               # wrist estimate may differ this much from the fixed-camera estimate, else refuse


def wrist_locate(cal, raw, img, bgr_ref, expect):
    """Object center / narrow axis / width from a wrist-camera image taken looking down from the hover pose.
    Segments pixels close in hue to the object's color in the fixed camera (the blob is mostly the top face), keeps
    blobs of block size, takes the one nearest the fixed-camera estimate `expect`, refuses if none is within MAX_DEV."""
    w = cal["wrist"]
    K, dist, T_gc = np.array(w["K"]), np.array(w["dist"]), np.array(w["T_gripperframe_cam"])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(int)
    ref = cv2.cvtColor(np.uint8([[bgr_ref]]), cv2.COLOR_BGR2HSV)[0, 0].astype(int)
    dh = np.minimum(abs(hsv[..., 0] - ref[0]), 180 - abs(hsv[..., 0] - ref[0]))
    m = ((dh < 12) & (hsv[..., 1] > 80) & (hsv[..., 2] > 50)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    T_bc = ik.fk(ik.raw_to_q(raw)) @ T_gc
    found, rejected = [], []
    for c in cs:
        P = px_to_plane(c.reshape(-1, 2).astype(float), K, dist, T_bc, TABLE_Z + OBJ_H)[:, :2]
        center = P.mean(0)
        _, V = np.linalg.eigh(np.cov((P - center).T))
        ext = (P - center) @ V
        width, length = float(np.ptp(ext[:, 0])), float(np.ptp(ext[:, 1]))
        dev = float(np.linalg.norm(center - np.asarray(expect)))
        if SIZE_RANGE[0] <= width and length <= SIZE_RANGE[1]:
            found.append((dev, center, np.r_[V[:, 0], 0.0], width, length))
        else:
            rejected.append(f"{width * 1000:.0f}x{length * 1000:.0f} mm at {np.round(center * 1000)}")
    if not found:
        raise SystemExit(f"REFUSED: no block-sized blob in the wrist image (rejected: {rejected or 'none'})")
    dev, center, narrow, width, length = min(found, key=lambda f: f[0])
    if dev > MAX_DEV:
        raise SystemExit(f"REFUSED: wrist estimate {np.round(center * 1000)} mm is {dev * 1000:.0f} mm from the "
                         f"fixed-camera estimate {np.round(np.asarray(expect) * 1000)} mm")
    return center, narrow, width, length


TL = {"1": 400, "2": 650, "3": 500, "4": 400, "5": 300}


def go(steps, raw_a, raw_b, snap=None, big=False):
    order = best_order(raw_a, raw_b) if big else [j for j in "23415" if raw_a[j] != raw_b[j]]
    if big:
        print(f"  transfer order {order}: path min z {body_min_z_path(raw_a, raw_b, order) * 1000:.0f} mm")
    for j in order:
        steps.append({"id": int(j), "goal": raw_b[j], "torque_limit": TL[j], "return_after": False,
                      "speed": 150 if big else 100})
    if snap:
        steps.append({"snap": snap})
    return raw_b


def vertical(q0, xy, z_from, z_to, close_dir):
    """IK solutions every 1 cm for the lowest gripper mesh going z_from -> z_to above xy (continuous from q0)."""
    zs = np.arange(z_from, z_to, -0.01 if z_to < z_from else 0.01).tolist() + [z_to]
    path = [at_height(q0, xy, zs[0], close_dir)]
    for z in zs[1:]:
        path.append(at_height(path[-1], xy, z, close_dir))
    return path


def stage_hover(scene_path, color, cal_path, start_json, out_path):
    """Fixed camera -> rough object pose -> plan: open gripper, fly above it, wrist snapshot. Torque stays on."""
    global GRIP_LOCAL
    scene = cv2.imread(scene_path)
    cal = json.load(open(cal_path))
    start = {k: int(v) for k, v in json.loads(start_json).items()}
    print("fixed camera re-located from the tag" if fixed_pose_from_tag(scene, cal)
          else "tag not visible: using the stored fixed-camera pose")
    contour, blobs = detect(scene, color)
    center, narrow, width, length = object_on_table(contour, cal)
    mask = np.zeros(scene.shape[:2], np.uint8)
    cv2.fillPoly(mask, [contour.astype(np.int32)], 1)
    hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
    sat = (mask > 0) & (hsv[..., 1] > 80)  # object color: saturated pixels only (not shadow / table)
    bgr_ref = np.median(scene[sat if sat.sum() > 50 else mask > 0], axis=0)
    place, _ = choose_place(center, cal, scene, contour, blobs)
    print(f"fixed camera: object {np.round(center * 1000, 1)} mm, {width * 1000:.0f} x {length * 1000:.0f} mm; "
          f"place {np.round(place * 1000, 1)} mm")
    GRIP_LOCAL = grip_local(width)
    q = topdown_ik(np.r_[center, TABLE_Z + OBJ_H + HOVER + 0.03], narrow, grip_q(GRIP_OPEN))
    q = at_height(q, center, TABLE_Z + OBJ_H + HOVER, narrow)
    steps = [{"id": 6, "goal": GRIP_OPEN, "torque_limit": 300, "return_after": False}]
    go(steps, {k: start[k] for k in "12345"}, arm_raw(q), "pp_hover", big=True)
    json.dump({"torque_limit": 500, "speed": 150, "steps": steps, "return": False, "release": False},
              open(out_path, "w"))
    json.dump({"start": start, "place": place.tolist(), "bgr_ref": bgr_ref.tolist(), "fixed_est": center.tolist()},
              open("pp/target.json", "w"))
    print(f"hover plan: {len(steps)} steps -> {out_path}")


NEAR = 0.03  # visual-correction height: lowest gripper mesh this far above the object top


def stage_approach(cal_path, raw_json, img_path, out_path):
    """Re-locate the object in the hover wrist image (checked against the fixed-camera estimate) and move (small,
    local IK) to NEAR above it; snapshot 'pp_near'. Run once, from the hover image (closer views are partly hidden
    by the jaws and drift)."""
    global GRIP_LOCAL
    cal, tgt = json.load(open(cal_path)), json.load(open("pp/target.json"))
    raw = {k: int(v) for k, v in json.loads(raw_json).items()}
    center, narrow, width, length = wrist_locate(cal, raw, cv2.imread(img_path), tgt["bgr_ref"], tgt["fixed_est"])
    dev = float(np.linalg.norm(center - np.array(tgt["fixed_est"]))) * 1000
    print(f"wrist estimate {np.round(center * 1000, 1)} mm, {width * 1000:.0f} x {length * 1000:.0f} mm, "
          f"{dev:.0f} mm from the fixed-camera estimate")
    GRIP_LOCAL = grip_local(width)
    q0 = ik.raw_to_q({**raw, "6": GRIP_OPEN})
    q = _refine(q0, np.r_[center, ik.fk(q0)[2, 3]], narrow)
    q = at_height(q, center, TABLE_Z + OBJ_H + NEAR, narrow)
    steps = []
    go(steps, {k: raw[k] for k in "12345"}, arm_raw(q), "pp_near")
    json.dump({"torque_limit": 500, "speed": 120, "steps": steps, "return": False, "release": False},
              open(out_path, "w"))
    json.dump({**tgt, "est": center.tolist(), "narrow": narrow.tolist(), "width": width}, open("pp/target.json", "w"))


def stage_home(raw_json, out_path):
    """Plan back to the start (rest) pose of this pick from wherever the arm is, joint order checked in sim, and
    release torque there. Used by run_pickplace.sh when a later stage refuses."""
    tgt = json.load(open("pp/target.json"))
    raw = {k: int(v) for k, v in json.loads(raw_json).items()}
    steps = []
    go(steps, {k: raw[k] for k in "12345"}, {k: tgt["start"][k] for k in "12345"}, big=True)
    steps.append({"id": 6, "goal": tgt["start"]["6"], "torque_limit": 300, "return_after": False})
    json.dump({"torque_limit": 500, "speed": 150, "steps": steps, "return": False, "release": True},
              open(out_path, "w"))


def stage_grasp(cal_path, raw_json, out_path):
    """From the NEAR pose with the last estimate: descend straight, grasp, lift, place, release, go to rest."""
    global GRIP_LOCAL
    cal = json.load(open(cal_path))
    tgt = json.load(open("pp/target.json"))
    hover = {k: int(v) for k, v in json.loads(raw_json).items()}
    center, narrow, width = np.array(tgt["est"]), np.array(tgt["narrow"]), tgt["width"]
    print(f"grasping at {np.round(center * 1000, 1)} mm, width {width * 1000:.0f} mm")
    if width > 0.06:
        raise SystemExit("object wider than the 6-7 cm jaw opening")
    GRIP_LOCAL = grip_local(width)
    place = np.array(tgt["place"])
    top = TABLE_Z + OBJ_H
    q_near = ik.raw_to_q({**hover, "6": GRIP_OPEN})  # continue from where the arm is: no re-solve, no jump
    pick = vertical(q_near, center, top + NEAR, TABLE_Z + CLEAR, narrow)
    up = vertical(pick[-1], center, TABLE_Z + CLEAR, top + HOVER, narrow)
    q = _refine(up[-1], np.r_[place, ik.fk(up[-1])[2, 3]], None, joints=(0, 1, 2, 3))
    drop = vertical(q, place, top + HOVER, TABLE_Z + CLEAR + 0.004, None)
    T = ik.fk(pick[-1])
    c = T[:3, 2] * [1, 1, 0]
    print(f"grasp: grip point {np.round((T[:3, :3] @ GRIP_LOCAL + T[:3, 3]) * 1000, 1)} mm, lowest gripper "
          f"{lowest_gripper_z(pick[-1]) * 1000:.1f} mm, closing axis vs narrow axis "
          f"{np.degrees(np.arccos(min(1.0, abs(c @ narrow) / np.linalg.norm(c)))):.0f} deg")
    steps = []
    cur = go(steps, {k: hover[k] for k in "12345"}, arm_raw(pick[0]))
    for q in pick[1:]:
        cur = go(steps, cur, arm_raw(q))
    steps.append({"snap": "pp_pregrasp"})
    steps.append({"id": 6, "goal": GRIP_CLOSE, "torque_limit": 350, "return_after": False, "squeeze": True,
                  "snap": "pp_closed"})
    for q in up[1:]:
        cur = go(steps, cur, arm_raw(q))
    steps.append({"snap": "pp_lifted"})
    cur = go(steps, cur, arm_raw(drop[0]), big=True)
    for q in drop[1:]:
        cur = go(steps, cur, arm_raw(q))
    steps.append({"id": 6, "goal": GRIP_OPEN, "torque_limit": 300, "return_after": False, "snap": "pp_released"})
    for q in reversed(drop[:-1]):
        cur = go(steps, cur, arm_raw(q))
    steps.append({"snap": "pp_done"})
    go(steps, cur, {k: tgt["start"][k] for k in "12345"}, big=True)
    steps.append({"id": 6, "goal": tgt["start"]["6"], "torque_limit": 300, "return_after": False})
    json.dump({"torque_limit": 500, "speed": 150, "steps": steps, "return": False, "release": True},
              open(out_path, "w"))
    print(f"grasp plan: {len(steps)} steps -> {out_path}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["hover"]:
        stage_hover(*sys.argv[2:7])
    elif sys.argv[1:2] == ["approach"]:
        stage_approach(*sys.argv[2:6])
    elif sys.argv[1:2] == ["grasp"]:
        stage_grasp(*sys.argv[2:5])
    elif sys.argv[1:2] == ["home"]:
        stage_home(*sys.argv[2:4])
    else:
        raise SystemExit(__doc__)

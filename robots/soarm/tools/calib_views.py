"""Plan a batch of wrist-camera AprilTag views for kinematic calibration, with sim-checked joint-by-joint transfers.

usage: SOARM_KINEMATICS=<kin.json> calib_views.py <kin.json> <cameras.json> <spec.json> <plan_out.json>
spec = {"blocks_xy": [[x, y], ...], "n": 20, "prefix": "k1", "rest": {"1": 2085, ...}, "seed": 0,
        "quota": {"far": 8, "steep": 4, "any": 8}, "avoid_raw": [[pan, lift, elbow, wrist, roll], ...]}
Views: tag (T_base_tag of kin.json) fully inside the wrist image region x >= 480 + 40 px, 40 px margins elsewhere,
predicted with the current model (so101_ik with the corrections, T_gripperframe_cam of kin.json).
Safety (corrected model, every mesh vertex of upper_arm and below, every transfer sampled joint by joint):
arm vertices z >= 50 mm, gripper/jaw vertices z >= 30 mm, nothing inside a block box inflated by 6 cm in xy and up to
70 mm in z, nothing within 15 cm (horizontal) of the fixed camera stand. Start and end at rest; torque released.
"""
import json
import sys
from itertools import permutations

import mujoco
import numpy as np

import so101_ik as ik
from calib_kinematics import TAG_OBJ, project

TL = {"1": 400, "2": 650, "3": 500, "4": 400, "5": 300}
IMG_W, IMG_H = 1280, 720
U_MIN, U_MAX, V_MIN, V_MAX = 480 + 40, IMG_W - 40, 40, IMG_H - 40
ARM_Z, GRIP_Z = 0.050, 0.030
BLOCK_HALF, BLOCK_XY_MARGIN, BLOCK_Z = 0.02, 0.06, 0.07
STAND_R = 0.15

_GEOMS = []  # (geom id, is_gripper, vertices subsampled)
for _g in range(ik.m.ngeom):
    _b = ik.m.body(int(ik.m.geom_bodyid[_g])).name
    if ik.m.geom_type[_g] != mujoco.mjtGeom.mjGEOM_MESH or _b in ("world", "base", "shoulder"):
        continue
    _mid = ik.m.geom_dataid[_g]
    _v = ik.m.mesh_vert[ik.m.mesh_vertadr[_mid]: ik.m.mesh_vertadr[_mid] + ik.m.mesh_vertnum[_mid]]
    _GEOMS.append((_g, _b in ("gripper", "moving_jaw_so101_v1"), _v[:: max(1, len(_v) // 300)].copy()))


class Scene:
    def __init__(self, kin, cams, blocks_xy):
        self.T_gc = np.array(kin["T_gripperframe_cam"])
        self.tag = TAG_OBJ @ np.array(kin["T_base_tag"])[:3, :3].T + np.array(kin["T_base_tag"])[:3, 3]
        w = cams["wrist"]
        self.K, self.dist = np.array(w["K"]), np.array(w["dist"])
        self.stand = np.array(cams["fixed"]["T_base_cam"])[:2, 3]
        self.blocks = np.array(blocks_xy, float).reshape(-1, 2)

    def clearance(self, q):
        """Margin (m, >= 0 means safe) of pose q against all the rules; the worst rule wins."""
        ik.set_q(q)
        worst = 1.0
        for g, grip, v in _GEOMS:
            P = v @ ik.d.geom_xmat[g].reshape(3, 3).T + ik.d.geom_xpos[g]
            worst = min(worst, P[:, 2].min() - (GRIP_Z if grip else ARM_Z))
            worst = min(worst, np.linalg.norm(P[:, :2] - self.stand, axis=1).min() - STAND_R)
            for b in self.blocks:
                dxy = np.abs(P[:, :2] - b).max(1) - (BLOCK_HALF + BLOCK_XY_MARGIN)
                worst = min(worst, np.maximum(dxy, P[:, 2] - BLOCK_Z).min())
        return float(worst)

    def tag_px(self, q):
        T = np.linalg.inv(ik.fk(q) @ self.T_gc)
        Pc = self.tag @ T[:3, :3].T + T[:3, 3]
        if (Pc[:, 2] < 0.08).any():
            return None
        return project(Pc, self.K, self.dist)

    def view_ok(self, q):
        uv = self.tag_px(q)
        if uv is None:
            return False
        if not ((uv[:, 0] >= U_MIN).all() and (uv[:, 0] <= U_MAX).all() and (uv[:, 1] >= V_MIN).all()
                and (uv[:, 1] <= V_MAX).all()):
            return False
        side = np.linalg.norm(np.roll(uv, 1, 0) - uv, axis=1)
        return side.min() >= 110  # corner accuracy: tag edges >= 110 px


def raw_q(r):
    return ik.raw_to_q({**{k: int(v) for k, v in r.items()}, "6": 2079})


def path_margin(sc, a, b, order, n=8):
    cur, worst = dict(a), 1.0
    for j in order:
        x0, x1 = cur[j], b[j]
        for t in np.linspace(0, 1, n)[1:]:
            worst = min(worst, sc.clearance(raw_q(dict(cur, **{j: round(x0 + t * (x1 - x0))}))))
            if worst < 0:
                return worst
        cur[j] = x1
    return worst


def best_path(sc, a, b):
    js = [j for j in "12345" if a[j] != b[j]]
    best = (-1.0, None)
    for o in permutations(js):
        mg = path_margin(sc, a, b, o)
        if mg > best[0]:
            best = (mg, o)
        if mg > 0.03:
            break
    return best


def candidates(sc, rng, n, kind):
    """Random joint configurations of the requested kind that see the whole tag and are statically clear."""
    lo = {"far": [1700, 2450, 950, 1800, 1955], "steep": [1800, 1500, 1500, 2600, 1955], "any": [1550, 1450, 1000, 1800, 1955]}[kind]
    hi = {"far": [2400, 2950, 1500, 3300, 3963], "steep": [2300, 2500, 2600, 3400, 3963], "any": [2550, 2950, 2600, 3400, 3963]}[kind]
    out = []
    for _ in range(n):
        r = {k: int(v) for k, v in zip("12345", rng.uniform(lo, hi))}
        q = raw_q(r)
        try:
            ik.q_to_raw(q)
        except ValueError:
            continue
        if kind == "steep" and ik.fk(q)[2, 0] > -0.85:  # approach axis close to straight down
            continue
        if sc.view_ok(q) and sc.clearance(q) > 0.01:
            out.append(r)
    return out


def pick_diverse(cands, k, taken):
    """Greedy max-min over normalized joints (roll weighted down a bit)."""
    if not cands:
        return []
    X = np.array([[c[j] for j in "12345"] for c in cands], float) / [300, 300, 300, 300, 900]
    T = [np.array([t[j] for j in "12345"], float) / [300, 300, 300, 300, 900] for t in taken]
    chosen = []
    dmin = np.full(len(X), np.inf)
    for t in T:
        dmin = np.minimum(dmin, np.linalg.norm(X - t, axis=1))
    for _ in range(min(k, len(X))):
        i = int(np.argmax(dmin)) if np.isfinite(dmin).any() else 0
        chosen.append(cands[i])
        dmin = np.minimum(dmin, np.linalg.norm(X - X[i], axis=1))
    return chosen


def main():
    kin, cams, spec = (json.load(open(p)) for p in sys.argv[1:4])
    sc = Scene(kin, cams, spec.get("blocks_xy", []))
    rng = np.random.default_rng(spec.get("seed", 0))
    rest = {k: int(v) for k, v in spec["rest"].items()}
    taken = [dict(zip("12345", r)) for r in spec.get("avoid_raw", [])]
    views = []
    for kind, k in spec["quota"].items():
        c = candidates(sc, rng, spec.get("samples", 6000), kind)
        got = pick_diverse(c, k, taken + views)
        print(f"{kind}: {len(c)} candidates, picked {len(got)}", flush=True)
        views += got
    # order: nearest neighbour from rest; each transfer needs a clear joint order (else the view is dropped)
    steps, cur, left, i = [], {k: rest[k] for k in "12345"}, list(views), 0
    while left:
        left.sort(key=lambda v: sum(abs(v[j] - cur[j]) / (900 if j == "5" else 300) for j in "12345"))
        for v in list(left):
            mg, order = best_path(sc, cur, v)
            if mg > 0:
                break
        else:
            print(f"dropped {len(left)} views: no clear transfer from {cur}")
            break
        left.remove(v)
        for j in order:
            steps.append({"id": int(j), "goal": v[j], "torque_limit": TL[j], "return_after": False, "speed": 180 if j == "5" else 120})
        name = f"{spec['prefix']}_{i:02d}"
        steps.append({"snap": name})
        q = raw_q(v)
        print(f"{name} raw {[v[j] for j in '12345']} order {''.join(order)} path margin {mg * 1000:.0f} mm, "
              f"tag px x {sc.tag_px(q)[:, 0].min():.0f}-{sc.tag_px(q)[:, 0].max():.0f}", flush=True)
        cur, i = dict(v), i + 1
    mg, order = best_path(sc, cur, {k: rest[k] for k in "12345"})
    if mg <= 0:
        raise SystemExit(f"no clear return to rest from {cur}")
    for j in order:
        steps.append({"id": int(j), "goal": rest[j], "torque_limit": TL[j], "return_after": False, "speed": 180 if j == "5" else 120})
    print(f"return order {''.join(order)} margin {mg * 1000:.0f} mm; {i} views, {len(steps)} steps")
    json.dump({"torque_limit": 400, "speed": 120, "steps": steps, "return": True, "release": True},
              open(sys.argv[4], "w"))


if __name__ == "__main__":
    main()

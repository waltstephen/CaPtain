"""Sim-side helpers for the real SO-101: joint map <-> MJCF q, FK of gripperframe, look-at IK (pan..wrist_flex).

If configs/hardware/so101_armhost_kinematics.json (or so101_armhost_kinematics.json next to this file in a flat copy) exists,
it is applied on import: corrected body_pos / body_quat of the arm bodies, joint zero offsets (folded into the joint
map's raw_at_mjcf_zero) and gravity compliance (set_q). SOARM_KINEMATICS=<path> overrides, SOARM_KINEMATICS=none
gives the plain Menagerie model. Fitted by tools/calib_kinematics.py.
"""
import json
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MJCF = os.environ.get("SOARM_MJCF", os.path.expandvars("$SOARM_ROOT/menagerie/robotstudio_so101/so101.xml"))
_MAP_REPO = os.path.join(HERE, "..", "configs", "hardware", "so101_armhost_joint_map.json")
MAP = json.load(open(_MAP_REPO if os.path.exists(_MAP_REPO) else os.path.join(HERE, "so101_armhost_joint_map.json")))
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
m = mujoco.MjModel.from_xml_path(MJCF)
d = mujoco.MjData(m)
SID = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
_KIN_REPO = os.path.join(HERE, "..", "configs", "hardware", "so101_armhost_kinematics.json")
KIN_PATH = os.environ.get("SOARM_KINEMATICS") or (
    _KIN_REPO if os.path.exists(_KIN_REPO) else os.path.join(HERE, "so101_armhost_kinematics.json"))
KIN = json.load(open(KIN_PATH)) if KIN_PATH != "none" and os.path.exists(KIN_PATH) else None
COMP = np.zeros(m.nv)  # gravity compliance (rad per N m): the arm sits at q + COMP * tau_gravity(q)
if KIN is not None:
    for _name, _b in KIN["bodies"].items():
        _bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, _name)
        m.body_pos[_bid], m.body_quat[_bid] = _b["pos"], _b["quat"]
    for _k, _name in enumerate(NAMES):
        _j = MAP["joints"][_name]
        _j["raw_at_mjcf_zero"] -= _j["sign"] * KIN["joint_zero_offset_rad"].get(_name, 0.0) / _j["rad_per_step"]
        COMP[_k] = KIN["compliance_rad_per_Nm"].get(_name, 0.0)


def set_q(q):
    """Pose the model at joint angles q (from raw_to_q) including the fitted gravity sag; runs mj_forward."""
    d.qpos[:] = q
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    if COMP.any():
        d.qpos[:] = np.asarray(q) + COMP * d.qfrc_bias
        mujoco.mj_forward(m, d)


def _soft_range():
    """Joint ranges in MJCF rad from the measured soft limits (the real stops are wider than the MJCF ranges)."""
    lo, hi = m.jnt_range[:, 0].copy(), m.jnt_range[:, 1].copy()
    for k, n in enumerate(NAMES):
        j = MAP["joints"][n]
        for arr, key in ((lo, "min"), (hi, "max")):
            raw = j["soft_limit_raw"][key]
            if raw is not None:
                arr[k] = j["sign"] * (raw - j["raw_at_mjcf_zero"]) * j["rad_per_step"]
    return lo, hi


LO, HI = _soft_range()


def raw_to_q(raw):
    q = []
    for n in NAMES:
        j = MAP["joints"][n]
        q.append(j["sign"] * (raw[str(j["id"])] - j["raw_at_mjcf_zero"]) * j["rad_per_step"])
    return np.array(q)


def q_to_raw(q):
    out = {}
    for n, v in zip(NAMES, q):
        j = MAP["joints"][n]
        raw = round(j["raw_at_mjcf_zero"] + j["sign"] * v / j["rad_per_step"])
        lo, hi = j["soft_limit_raw"]["min"], j["soft_limit_raw"]["max"]
        if lo is not None and raw < lo or raw > hi:
            raise ValueError(f"{n} raw {raw} outside soft limits [{lo}, {hi}]")
        out[j["id"]] = raw
    return out


def fk(q):
    """4x4 pose of gripperframe in the MJCF base frame (x = approach axis of the hand)."""
    set_q(q)
    T = np.eye(4)
    T[:3, :3] = d.site_xmat[SID].reshape(3, 3)
    T[:3, 3] = d.site_xpos[SID]
    return T


def body_min_z(q):
    """Lowest of the moving bodies below the shoulder (lift onward) and the gripperframe site: table clearance proxy."""
    set_q(q)
    return float(min(d.xpos[3:, 2].min(), d.site_xpos[SID][2]))


def look_ik(pos, target, q0, roll=0.0, n=36, fix_pan=True, aim_w=0.3):
    """Grid search + local refinement over (lift, elbow, wrist_flex) (and pan unless fix_pan): gripperframe at pos with
    its x (approach) axis pointing at target. Returns (q, position error m, aim error deg)."""
    q0 = np.array(q0, float)
    lo, hi = LO, HI
    ks = [1, 2, 3] if fix_pan else [0, 1, 2, 3]

    def cost(q):
        T = fk(q)
        v = target - T[:3, 3]
        aim = np.arccos(np.clip(T[:3, 0] @ v / np.linalg.norm(v), -1, 1))
        return np.linalg.norm(pos - T[:3, 3]) + aim_w * aim

    best, bq = np.inf, None
    grids = [np.linspace(lo[k], hi[k], n) for k in ks]
    for vals in np.stack(np.meshgrid(*grids, indexing="ij"), -1).reshape(-1, len(ks)):
        q = q0.copy()
        q[4] = roll
        q[ks] = vals
        c = cost(q)
        if c < best:
            best, bq = c, q
    step = (hi - lo) / n
    for _ in range(6):  # coordinate refinement, halving the step
        for _ in range(20):
            improved = False
            for k in ks:
                for s_ in (-1, 1):
                    q = bq.copy()
                    q[k] = np.clip(q[k] + s_ * step[k], lo[k], hi[k])
                    c = cost(q)
                    if c < best:
                        best, bq, improved = c, q, True
            if not improved:
                break
        step = step / 2
    T = fk(bq)
    v = target - T[:3, 3]
    aim = np.degrees(np.arccos(np.clip(T[:3, 0] @ v / np.linalg.norm(v), -1, 1)))
    return bq, float(np.linalg.norm(pos - T[:3, 3])), float(aim)

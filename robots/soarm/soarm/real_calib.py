"""Calibrated kinematics of the real SO-101 on armhost (armcal: so101_armhost_{joint_map,kinematics,cameras}.json).

Joint convention: q_raw = joint map applied to the servo positions (with the fitted joint zero offsets folded in);
q_eff = q_raw + compliance * gravity torque(q_raw) = where the links really are (fitted gravity sag). The harness twin
sees q_eff, so twin FK of the reported joints is the calibrated pose of the hand.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import mujoco
import numpy as np

from .scene import menagerie_dir
from .twin import build_twin_xml

NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
HAND_BODIES = ("gripper", "moving_jaw_so101_v1")
TWIN_RANGE_MARGIN_RAD = 0.1


def _apply_bodies(spec: mujoco.MjSpec, kin: dict) -> None:
    for name, body in kin["bodies"].items():
        b = spec.body(name)
        b.pos, b.quat = list(body["pos"]), list(body["quat"])


class Calibration:
    def __init__(self, joint_map: str, kinematics: str, cameras: str) -> None:
        joint_map, kinematics, cameras = (os.path.expandvars(p) for p in (joint_map, kinematics, cameras))   # "$SOARM_CODE/..."
        self.map = json.loads(Path(joint_map).read_text())["joints"]
        kin = json.loads(Path(kinematics).read_text())
        self.cameras = json.loads(Path(cameras).read_text())
        self.ids = [int(self.map[n]["id"]) for n in NAMES]
        self.sign = np.array([self.map[n]["sign"] for n in NAMES], float)
        self.rps = np.array([self.map[n]["rad_per_step"] for n in NAMES], float)
        self.zero = np.array([self.map[n]["raw_at_mjcf_zero"] for n in NAMES], float)
        self.zero -= self.sign * np.array([kin["joint_zero_offset_rad"].get(n, 0.0) for n in NAMES]) / self.rps
        self.comp = np.array([kin["compliance_rad_per_Nm"].get(n, 0.0) for n in NAMES], float)
        self.soft_lo = np.array([self.map[n]["soft_limit_raw"]["min"] or 0 for n in NAMES], float)
        self.soft_hi = np.array([self.map[n]["soft_limit_raw"]["max"] or 4095 for n in NAMES], float)
        spec = mujoco.MjSpec.from_file(str(menagerie_dir() / "so101.xml"))
        _apply_bodies(spec, kin)
        self.full = spec.compile()                      # physics MJCF + calibrated bodies: gravity, meshes, gripperframe
        self.fd = mujoco.MjData(self.full)
        self.gripperframe = self.full.site("gripperframe").id
        twin = mujoco.MjSpec.from_string(build_twin_xml())
        _apply_bodies(twin, kin)
        # the harness plans on the twin's joint ranges: the calibrated bodies put straight top-down poses near the base
        # ~1 deg past the wrist_flex range, so the twin gets TWIN_RANGE_MARGIN_RAD more; the driver still clips every
        # goal to the measured soft limits (the hand then stays within ~1-2 deg / 3 mm of the request)
        for k in range(1, 6):
            j = twin.joint(f"robot0_joint{k}")
            j.range = [j.range[0] - TWIN_RANGE_MARGIN_RAD, j.range[1] + TWIN_RANGE_MARGIN_RAD]
        self.twin_xml = twin.to_xml()   # the driver's IK / force model
        # the harness plans on the uncalibrated twin of the training episodes (its IK accepts 1 mm / 0.01 rad, which
        # the calibrated bodies miss for exact top-down poses); the driver reports joints for which this nominal twin's
        # grip site sits where the calibrated model puts the real one
        self.nominal_xml = build_twin_xml()
        self._hand_geoms = [g for g in range(self.full.ngeom)
                            if self.full.body(int(self.full.geom_bodyid[g])).name in HAND_BODIES
                            and self.full.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]

    # --------------------------------------------------------------- joints
    def raw_to_q(self, raw: np.ndarray) -> np.ndarray:
        return self.sign * (np.asarray(raw, float) - self.zero) * self.rps

    def q_to_raw(self, q: np.ndarray) -> np.ndarray:
        raw = self.zero + self.sign * np.asarray(q, float) / self.rps
        return np.clip(np.round(raw), self.soft_lo, self.soft_hi)

    def _bias(self, q6: np.ndarray) -> np.ndarray:
        d = self.fd
        d.qpos[:] = q6
        d.qvel[:] = 0
        mujoco.mj_forward(self.full, d)
        return d.qfrc_bias.copy()

    def q_eff(self, q_raw6: np.ndarray) -> np.ndarray:
        return q_raw6 + self.comp * self._bias(q_raw6)

    def q_raw_for(self, q_eff6: np.ndarray) -> np.ndarray:
        q = np.asarray(q_eff6, float).copy()
        for _ in range(2):
            q = q_eff6 - self.comp * self._bias(q)
        return q

    # --------------------------------------------------------------- poses (q_eff)
    def _pose(self, q_eff6):
        d = self.fd
        d.qpos[:] = q_eff6
        mujoco.mj_kinematics(self.full, d)

    def gripperframe_pose(self, q_eff6: np.ndarray) -> np.ndarray:
        self._pose(q_eff6)
        t = np.eye(4)
        t[:3, :3] = self.fd.site_xmat[self.gripperframe].reshape(3, 3)
        t[:3, 3] = self.fd.site_xpos[self.gripperframe]
        return t

    def lowest_hand_z(self, q_eff6: np.ndarray) -> float:
        """Lowest vertex of the gripper + moving-jaw meshes (what touches the table first)."""
        self._pose(q_eff6)
        m, d = self.full, self.fd
        zs = []
        for g in self._hand_geoms:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            zs.append(((d.geom_xmat[g].reshape(3, 3) @ v.T).T + d.geom_xpos[g])[:, 2].min())
        return float(min(zs))

    def lowest_body_z(self, q_eff6: np.ndarray) -> float:
        """Lowest moving body origin below the shoulder (path clearance proxy for joint-space moves)."""
        self._pose(q_eff6)
        return float(min(self.fd.xpos[3:, 2].min(), self.fd.site_xpos[self.gripperframe][2]))

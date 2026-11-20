"""XR1Driver for the real SO-101 on armhost (harness adapters/hardware/facade.py protocol), the real-arm counterpart of
soarm/driver.py: same twin (calibrated bodies), same IK, same jaw mapping, cameras from tools/camstream.py.

Safety: every arm goal is clipped to the measured joint soft limits; a goal whose lowest hand vertex would go below
TABLE_GUARD_M is not sent (the arm holds); torque limits per joint; a closing jaw that stalls on an object keeps only a
small squeeze (holding the full torque limit trips the STS3215 overload protection and the jaw stops answering);
reset() moves the arm from rest to the twin's home pose and close() takes it back to rest and releases torque.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from captain_harness.adapters.hardware.facade import CameraFrame, DriverState

from .real_bus import A_ACC, A_SPEED, A_STATUS, A_TEMP, A_TLIM, A_TORQUE, Bus
from .real_calib import Calibration
from .sim import HOME_Q
from .twin import JAW_CLOSED_RAD, JAW_OPEN_RAD, gripper_action_to_jaw, jaw_to_open_fraction

TWIN_ARM = tuple(f"robot0_joint{k}" for k in range(1, 8))
# lowest hand vertex above z = TABLE_GUARD_M. The calibrated arm frame puts the real table at z = -5.5 mm (six
# arm-measured flat boards, 11-14: -5.1..-6.3 mm), so -3 mm keeps the fingertips ~2.5 mm above the table. The old
# +2 mm stopped them ~7.5 mm above it: a 14 mm cube grasp was held short and the harness read that as contact
# (web_1114_225452), after which the policy closed the jaws in the air.
TABLE_GUARD_M = -0.003
GRIP_SQUEEZE = 50            # raw steps past the stalled jaw position while holding an object
GRIP_RAW_MIN = 2000          # jaws closed on themselves ~2027
GRIP_STALL_TICKS = 3
JOINT_MOVE_RAD_S = 0.3       # reset / close joint-space moves
# hand force estimate (no force sensor): servo position error beyond the gravity / friction deadband x a nominal
# STS3215 stiffness, plus a virtual table spring; the per-episode free-space calibration removes bias and lag terms
SERVO_DEADBAND_RAW = 50
# while the jaws hold an object and the hand is being lowered (placing), a support under the held object is felt at a
# smaller servo error: with 50 steps a held cube lowered onto a light cube pushed it away unnoticed (web_1114_232106,
# 0 N reported), where the twin's contact truth would have stopped the descent on top of it
HOLD_DESCENT_DEADBAND_RAW = 20
HOLD_JAW_MIN_RAW = 2100      # jaw stalled above this while closing = an object between the jaws (empty closes ~2040)
GOAL_LAG_TICKS = 6            # 300 ms of goals
FEEDFORWARD_TICKS = 1.0
# integral trim on every tick a joint's goal is still. 10 still ticks (11-14 smoothing change) almost never happened
# under the harness, which steps each goal from the measured pose: the uncompensated gravity sag was re-based every
# tick and the hand sank 10-12 cm towards the base (web_1114_170231; harness-like loop: 1 tick 3/3 runs within 2 mm,
# 10 ticks up to 66 mm off, no trim 98-121 mm off)
TRIM_STILL_TICKS = 1
IK_POS_PRIORITY_M = 0.003    # driver IK: re-solve position-first when the balanced solve misses by more
SUBSTEPS = 5                 # servo goals per control tick, linearly interpolated (100 Hz at 20 Hz control)
TRIM_GAIN = 0.3
TRIM_MAX_RAW = 40
SERVO_NM_PER_RAD = 20.0
TABLE_K_N_PER_M = 2000.0
FORCE_DEADBAND_N = 0.6       # as the twin driver (soarm/driver.py)
MAX_GOAL_STEP_RAW = 50       # per tick (50 steps / 50 ms = 1.5 rad/s); the servos run without their own speed profile


def make_driver(config: dict) -> "RealSoarmDriver":
    return RealSoarmDriver(config)


class RealSoarmDriver:
    reread_after_hold = True

    def __init__(self, config: dict) -> None:
        self.config = config
        c = config["calibration"]
        self.cal = Calibration(c["joint_map"], c["kinematics"], c["cameras"])
        self.control_frequency = int(config["control"]["control_frequency_hz"])
        self._dt = 1.0 / self.control_frequency
        self.model = mujoco.MjModel.from_xml_string(self.cal.twin_xml)
        self._data = mujoco.MjData(self.model)
        m = self.model
        self._q = np.array([m.jnt_qposadr[m.joint(n).id] for n in TWIN_ARM])
        self._v = np.array([m.jnt_dofadr[m.joint(n).id] for n in TWIN_ARM])
        # twin joint ranges cut to the measured soft limits (wrist_roll turns 0..3.17 rad only, not the MJCF's +-2.8)
        soft = np.sort(np.stack([self.cal.raw_to_q(self.cal.soft_lo), self.cal.raw_to_q(self.cal.soft_hi)]), axis=0)
        twin = m.jnt_range[[m.joint(n).id for n in TWIN_ARM]][:5]
        self._limits = np.stack([np.maximum(twin[:, 0], soft[0, :5]), np.minimum(twin[:, 1], soft[1, :5])], axis=1)
        self._site = m.site("gripper0_grip_site").id
        self.nominal = mujoco.MjModel.from_xml_string(self.cal.nominal_xml)
        self._nom_data = mujoco.MjData(self.nominal)
        self._nom_q: np.ndarray | None = None
        # reported joints stay inside the harness twin's ranges (shrunk 1 mrad): the harness refuses every motion while
        # a measured joint is past them (web_1114_180916: wrist_flex reported 3 mrad over 1.658 rad, all moves refused)
        self._nom_limits = self.nominal.jnt_range[[self.nominal.joint(n).id for n in TWIN_ARM]][:5] + [0.001, -0.001]
        self.log_path = Path(config["log_path"]) if config.get("log_path") else None
        self.bus = Bus(config.get("port", "/dev/ttyACM0"), self.cal.ids)
        servo = config.get("servo", {})
        self._tl = [int(v) for v in servo.get("torque_limit", [500, 800, 650, 500, 400, 500])]
        for k, i in enumerate(self.cal.ids):
            self.bus.w2(i, A_TLIM, self._tl[k])
            self.bus.w2(i, A_SPEED, int(servo.get("speed", 800)))
            self.bus.w1(i, A_ACC, int(servo.get("acceleration", 40)))
        raw = self._raw()
        self.rest_raw = raw.copy()
        self.bus.goals(dict(zip(self.cal.ids, raw.astype(int))))   # hold where it stands, then torque on
        for i in self.cal.ids:
            self.bus.w1(i, A_TORQUE, 1)
        self._torque = True
        self._t0 = time.monotonic()
        self._next_tick = time.monotonic()
        self._q_cmd: np.ndarray | None = None
        self._arm_goal_raw = raw[:5].copy()
        self._penetration = 0.0
        self._goal_hist: list[np.ndarray] = []
        self._trim = np.zeros(5)
        self._still_ticks = np.zeros(5)
        self._sent_arm = raw[:5].copy()
        self._jaw_target_raw = float(raw[5])
        self._jaw_goal_raw = float(raw[5])
        self._jaw_hist: list[float] = []
        self._last = None
        self._ticks = 0
        self.timing = {"command_calls": 0, "guard_holds": 0, "jaw_squeeze_ticks": 0, "jaw_torque_rearms": 0,
                       "frame_calls": 0, "frame_wall_s": 0.0}
        self._log(event="start", rest_raw=raw.tolist())
        if config.get("standard_rest_raw"):   # every episode ends at the same rest pose (tools/real_arm_web.py reset)
            self.rest_raw = np.array(config["standard_rest_raw"], float)
        self._squeeze = float(config.get("grip_squeeze_raw", GRIP_SQUEEZE))
        # real grip point = commanded grip site shifted in the EEF frame (x = jaw closing axis, towards the moving
        # jaw): the twin leaves the fixed jaw only 3 mm clear of a 3 cm cube centred on the grip point, less than the
        # real position error, so the fixed jaw landed on cube edges (11-13); closing pushes the cube back
        self._tool = np.eye(4)
        self._tool[:3, 3] = np.asarray(config.get("grip_offset_eef_m", (0.0, 0.0, 0.0)), float)
        self._tool_inv = np.linalg.inv(self._tool)
        self._state_now()

    # ------------------------------------------------------------------ io
    def _log(self, **row) -> None:
        if self.log_path is not None:
            with self.log_path.open("a") as f:
                f.write(json.dumps({"t": round(time.monotonic() - getattr(self, "_t0", time.monotonic()), 3), **row})
                        + "\n")

    def _raw(self) -> np.ndarray:
        pos = self.bus.positions()
        return np.array([pos[i] for i in self.cal.ids], float)

    def _jaw_raw(self, jaw_rad: float) -> float:
        k = 5
        return float(np.clip(self.cal.zero[k] + self.cal.sign[k] * jaw_rad / self.cal.rps[k], GRIP_RAW_MIN,
                             self.cal.soft_hi[k]))

    def _state_now(self) -> DriverState:
        raw = self._raw()
        now = time.monotonic()
        q_eff = self.cal.q_eff(self.cal.raw_to_q(raw))
        vel = np.zeros(5)
        if self._last is not None and now > self._last[0]:
            vel = (q_eff[:5] - self._last[1][:5]) / (now - self._last[0])
        self._last = (now, q_eff, raw)
        jaw = float(self.cal.raw_to_q(raw)[5])
        return DriverState(joints=np.r_[self._nominal_q(q_eff), 0.0, 0.0], joint_velocities=np.r_[vel, 0.0, 0.0],
                           gripper_open_fraction=jaw_to_open_fraction(jaw), wrist_force_world=self._force(q_eff, raw),
                           time_s=now - self._t0)

    def _force(self, q_eff: np.ndarray, raw: np.ndarray) -> np.ndarray:
        """External force on the hand (robot_base) as the twin driver reports it (deadband FORCE_DEADBAND_N): the joint
        torques implied by the servos' position errors beyond SERVO_DEADBAND_RAW (STS3215 P control), mapped through
        the grip-site Jacobian, plus a stiff virtual table under goals the table guard refused."""
        # a servo following a moving goal lags it by up to GOAL_LAG_TICKS: only the distance from the measured position
        # to the span of the recent goals counts (a blocked joint falls out of that span as the goals move on)
        hist = np.array(self._goal_hist or [self._arm_goal_raw])
        lo, hi = hist.min(axis=0), hist.max(axis=0)
        err = np.where(raw[:5] < lo, lo - raw[:5], np.where(raw[:5] > hi, hi - raw[:5], 0.0))
        # placing (an object held, the commanded grip site going down): the smaller deadband
        placing = (getattr(self, "_goal_descending", False) and self._jaw_target_raw < float(raw[5]) - 30
                   and float(raw[5]) > HOLD_JAW_MIN_RAW)
        band = HOLD_DESCENT_DEADBAND_RAW if placing else SERVO_DEADBAND_RAW
        err = np.sign(err) * np.maximum(np.abs(err) - band, 0.0)
        tau = SERVO_NM_PER_RAD * err * self.cal.rps[:5] * self.cal.sign[:5]
        m, d = self.model, self._data
        d.qpos[self._q] = np.r_[q_eff[:5], 0.0, 0.0]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        jp = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, None, self._site)
        force = -np.linalg.pinv(jp[:, self._v[:5]].T) @ tau
        force[2] += TABLE_K_N_PER_M * self._penetration
        return np.zeros(3) if float(np.linalg.norm(force)) < FORCE_DEADBAND_N else force

    def _tick(self) -> None:
        now = time.monotonic()
        if self._next_tick - now > 0:
            time.sleep(self._next_tick - now)
        self._next_tick = max(self._next_tick + self._dt, time.monotonic() - self._dt)
        self._ticks += 1

    def _jaw_goal(self, jaw_meas: float) -> float:
        """Jaw goal: the commanded opening, except that a closing jaw stalled on an object only squeezes lightly."""
        self._jaw_hist = (self._jaw_hist + [jaw_meas])[-(GRIP_STALL_TICKS + 1):]
        closing = self._jaw_target_raw < jaw_meas - 30
        stalled = len(self._jaw_hist) > GRIP_STALL_TICKS and max(self._jaw_hist) - min(self._jaw_hist) < 4
        if closing and stalled:
            self.timing["jaw_squeeze_ticks"] += 1
            return max(self._jaw_target_raw, jaw_meas - self._squeeze)
        return self._jaw_target_raw

    def _check_jaw_torque(self) -> None:
        jaw_id = self.cal.ids[5]
        if self.bus.r1(jaw_id, A_TORQUE) == 0:
            self.timing["jaw_torque_rearms"] += 1
            self._log(event="jaw_torque_rearm", status=self.bus.r1(jaw_id, A_STATUS), temp=self.bus.r1(jaw_id, A_TEMP))
            self.bus.w1(jaw_id, A_TORQUE, 1)

    def _send(self) -> None:
        prev = self._goal_hist[-1] if self._goal_hist else self._arm_goal_raw
        self._goal_hist = (self._goal_hist + [self._arm_goal_raw.copy()])[-GOAL_LAG_TICKS:]
        # velocity feed-forward: the P-controlled servos lag a moving goal by a few ticks
        # integral trim: removes the servos' steady-state error under gravity (15-25 steps on elbow / shoulder) while
        # the goal stands still; frozen during motion, bounded so a blocked joint never pushes much harder
        meas = self._last[2][:5]
        self._still_ticks = np.where(np.abs(self._arm_goal_raw - prev) < 2, self._still_ticks + 1, 0)
        err = self._arm_goal_raw - meas
        update = (self._still_ticks >= TRIM_STILL_TICKS) & (np.abs(err) < TRIM_MAX_RAW)
        self._trim = np.where(update, np.clip(self._trim + TRIM_GAIN * err, -TRIM_MAX_RAW, TRIM_MAX_RAW), self._trim)
        arm = np.clip(self._arm_goal_raw + self._trim + FEEDFORWARD_TICKS * (self._arm_goal_raw - prev),
                      self.cal.soft_lo[:5], self.cal.soft_hi[:5])
        jaw_meas = float(self._last[2][5])
        self._jaw_goal_raw = self._jaw_goal(jaw_meas)
        # the 20 Hz goal is spread over SUBSTEPS servo writes inside the tick: the P-controlled servos follow a ramp
        # instead of a 50 ms staircase (the stutter seen on 11-13)
        start, t0 = self._sent_arm, time.monotonic()
        for k in range(1, SUBSTEPS + 1):
            goal = start + (arm - start) * k / SUBSTEPS
            self.bus.goals(dict(zip(self.cal.ids, [*np.round(goal).astype(int), int(round(self._jaw_goal_raw))])))
            if k < SUBSTEPS:
                time.sleep(max(0.0, t0 + k * self._dt / SUBSTEPS - time.monotonic()))
        self._sent_arm = arm
        if self._ticks % 20 == 0:
            self._check_jaw_torque()

    # ------------------------------------------------------------------ IK (soarm/driver.py, on the calibrated twin)
    def _nominal_q(self, q_eff: np.ndarray) -> np.ndarray:
        """Joints reported to the harness: nominal-twin joints whose grip-site pose equals the calibrated one."""
        d = self._data
        d.qpos[self._q] = np.r_[q_eff[:5], 0.0, 0.0]
        mujoco.mj_kinematics(self.model, d)
        pose = np.eye(4)
        pose[:3, :3] = d.site_xmat[self._site].reshape(3, 3)
        pose[:3, 3] = d.site_xpos[self._site]
        pose = pose @ self._tool_inv   # the harness sees the commanded grip point, not the shifted real one
        # position first (the harness tracks positions to mm), orientation as far as the nominal 5 DoF allow. The last
        # answer is tried first; when it does not reach the pose within 0.5 mm the measured joints are tried too and
        # the better one wins (random joint walks on 11-14: max reported-vs-real 10 mm, a warm start alone 14 mm).
        best = None
        for seed in ([self._nom_q] if self._nom_q is not None else []) + [q_eff[:5]]:
            q = self._ik(pose, seed, iterations=60, rot_weight=0.2, nominal=True, limits=self._nom_limits)
            if self._nom_pos_err(pose, q) > 0.001:
                # far / tilted poses: match the position first; the balanced solve reported the hand 2-3 cm off
                # at (0.18, 0.24, 0.08) on 11-14
                q = self._ik(pose, q, iterations=150, rot_weight=0.02, nominal=True, limits=self._nom_limits)
            e = self._nom_pos_err(pose, q)
            if best is None or e < best[0] - 1e-4:
                best = (e, q)
            if e < 0.0005:
                break
        self._nom_q = best[1]
        return self._nom_q

    def _nom_pos_err(self, goal: np.ndarray, q: np.ndarray) -> float:
        d = self._nom_data
        d.qpos[self._q] = np.r_[q, 0.0, 0.0]
        mujoco.mj_kinematics(self.nominal, d)
        return float(np.linalg.norm(goal[:3, 3] - d.site_xpos[self._site]))

    def _ik(self, goal: np.ndarray, q: np.ndarray, iterations: int = 60, rot_weight: float = 0.35,
            nominal: bool = False, limits: np.ndarray | None = None) -> np.ndarray:
        m, d = (self.nominal, self._nom_data) if nominal else (self.model, self._data)
        if limits is None:
            limits = m.jnt_range[[m.joint(n).id for n in TWIN_ARM]][:5] if nominal else self._limits
        q = q.copy()
        for _ in range(iterations):
            d.qpos[self._q] = np.r_[q, 0.0, 0.0]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            dp = goal[:3, 3] - d.site_xpos[self._site]
            dr = Rotation.from_matrix(goal[:3, :3] @ d.site_xmat[self._site].reshape(3, 3).T).as_rotvec()
            if np.linalg.norm(dp) < 2e-4 and np.linalg.norm(dr) < 2e-3:
                break
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, jp, jr, self._site)
            jac = np.vstack([jp[:, self._v[:5]], rot_weight * jr[:, self._v[:5]]])
            err = np.r_[dp, rot_weight * dr]
            step = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), err)
            step *= min(1.0, 0.1 / max(np.abs(step).max(), 1e-9))
            q = np.clip(q + step, limits[:, 0] + 1e-4, limits[:, 1] - 1e-4)
        return q

    def _pos_err(self, goal: np.ndarray, q: np.ndarray) -> float:
        d = self._data
        d.qpos[self._q] = np.r_[q, 0.0, 0.0]
        mujoco.mj_kinematics(self.model, d)
        return float(np.linalg.norm(goal[:3, 3] - d.site_xpos[self._site]))

    def _arm_raw_for(self, q_eff5: np.ndarray) -> np.ndarray:
        q6 = np.r_[q_eff5, self._last[1][5]]
        return self.cal.q_to_raw(self.cal.q_raw_for(q6))[:5]

    # ------------------------------------------------------------------ XR1Driver
    def read(self) -> DriverState:
        return self._state_now()

    def command(self, grip_site_goal: np.ndarray | None, gripper_action: float) -> DriverState:
        self.timing["command_calls"] += 1
        # gripper action exactly 0 = the harness's neutral step after a stopped move: keep the jaw where it was. In
        # sim that one tick barely moves the jaw; here it stays the target while the harness thinks (~2 s), and on 11-14
        # it opened a held cube's jaws by 7 mm twice (web_1114_213339, raw 2251 -> 2328), so the cube slid out on the lift
        if gripper_action != 0.0:
            self._jaw_target_raw = self._jaw_raw(gripper_action_to_jaw(gripper_action))
        if grip_site_goal is not None:
            gz = float(np.asarray(grip_site_goal, float)[2, 3])
            self._goal_descending = gz < getattr(self, "_last_goal_z", gz) - 1e-4
            self._last_goal_z = gz
            measured = self._last[1][:5]
            seed = self._q_cmd if self._q_cmd is not None and np.abs(self._q_cmd - measured).max() < 0.05 else measured
            goal = np.asarray(grip_site_goal, float) @ self._tool
            target = self._ik(goal, seed)
            if self._pos_err(goal, target) > IK_POS_PRIORITY_M:
                # orientation out of reach (e.g. top-down beyond ~0.29 m): keep the position, give up orientation;
                # the balanced solve put the hand 3-4 cm away from (0.18, 0.23, 0.075) on 11-14
                target = self._ik(goal, target, rot_weight=0.03)
            low = self.cal.lowest_hand_z(np.r_[target, self._last[1][5]])
            self._penetration = max(0.0, TABLE_GUARD_M - low)
            if low < TABLE_GUARD_M:
                self.timing["guard_holds"] += 1
                if self.timing["guard_holds"] % 20 == 1:
                    self._log(event="table_guard", lowest_hand_z=round(low, 4), goal=np.round(grip_site_goal[:3, 3], 4).tolist())
            else:
                self._q_cmd = target.copy()
                goal = self._arm_raw_for(target)   # at most MAX_GOAL_STEP_RAW per tick ahead of the last goal
                self._arm_goal_raw = self._arm_goal_raw + np.clip(goal - self._arm_goal_raw, -MAX_GOAL_STEP_RAW,
                                                                  MAX_GOAL_STEP_RAW)
        self._send()
        self._tick()
        self._last_motion = time.monotonic()
        state = self._state_now()
        # per-tick trace (web_1114_170231 could not be replayed without it): grip-site goal, measured raw, sent goal
        self._log(event="tick", goal=None if grip_site_goal is None else np.round(grip_site_goal[:3, 3], 4).tolist(),
                  raw=self._last[2].astype(int).tolist(), sent=np.round(self._sent_arm).astype(int).tolist(),
                  grip=round(float(gripper_action), 2))
        return state

    def hold(self) -> None:
        self._q_cmd = None
        self._penetration = 0.0
        self._goal_descending = False
        self._arm_goal_raw = self._last[2][:5].copy()
        self._send()
        self._tick()

    def _home_raw(self) -> np.ndarray:
        return self._arm_raw_for(HOME_Q)

    def reset(self) -> DriverState:
        """Rest -> home: arm first with roll and jaw as they are (a roll swing near the table catches the jaw on it),
        then roll and jaw open at home height."""
        start = self._raw()
        home = np.r_[self._home_raw(), self._jaw_raw(JAW_OPEN_RAD)]
        self._joint_move(np.r_[home[:4], start[4:]], "home_arm")
        self._joint_move(home, "home_roll")
        self._jaw_target_raw = home[5]
        return self._state_now()

    def _joint_move(self, target_raw: np.ndarray, what: str) -> None:
        """Linear move in raw joint space at JOINT_MOVE_RAD_S, refused if the path brings any body or the hand
        meshes toward the table."""
        start = self._raw()
        n = max(1, int(np.ceil(np.abs((target_raw - start) * self.cal.rps)[:5].max() / JOINT_MOVE_RAD_S / self._dt)))
        path = [start + (target_raw - start) * k / n for k in range(1, n + 1)]
        z = []
        for r in [start, *path[::max(1, n // 25)], path[-1]]:
            q = self.cal.q_eff(self.cal.raw_to_q(r))
            z.append(min(self.cal.lowest_body_z(q), self.cal.lowest_hand_z(q)))
        floor = min(0.005, min(z[0], z[-1]) - 0.003)
        self._log(event="joint_move", what=what, start=start.tolist(), target=target_raw.tolist(), ticks=n,
                  path_min_z=round(min(z), 4), floor=round(floor, 4))
        if min(z) < floor:
            raise RuntimeError(f"{what}: joint path dips to {min(z):.3f} m (floor {floor:.3f} m)")
        for r in path:
            self._arm_goal_raw = np.round(r[:5])
            self.bus.goals(dict(zip(self.cal.ids, np.round(r).astype(int))))
            self._tick()
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline and np.abs(self._raw() - target_raw).max() > 25:
            time.sleep(0.05)
        self._state_now()
        err = np.abs(self._last[2][:5] - target_raw[:5])
        if err.max() > 60:   # a joint that does not follow (blocked, or in STS overload protection)
            self._log(event="joint_move_short", what=what, err=err.tolist())
            if what.startswith("home"):
                raise RuntimeError(f"{what}: joints did not reach the target, error {err.tolist()} raw steps")
        self._arm_goal_raw = target_raw[:5].copy()
        self._sent_arm = target_raw[:5].copy()
        self._jaw_target_raw = float(target_raw[5])
        self._q_cmd = None

    def _lift(self, z_min: float = 0.10) -> None:
        """Straight up (5 mm per tick) until the grip site is z_min above the table."""
        for _ in range(60):
            q6 = self._last[1]
            d = self._data
            d.qpos[self._q] = np.r_[q6[:5], 0.0, 0.0]
            mujoco.mj_kinematics(self.model, d)
            pose = np.eye(4)
            pose[:3, :3] = d.site_xmat[self._site].reshape(3, 3)
            pose[:3, 3] = d.site_xpos[self._site]
            pose = pose @ self._tool_inv
            if pose[2, 3] >= z_min:
                return
            pose[2, 3] = min(z_min, pose[2, 3] + 0.005)
            jaw = (self._jaw_target_raw - self.cal.zero[5]) * self.cal.rps[5] * self.cal.sign[5]
            self.command(pose, 2.0 * (jaw - JAW_OPEN_RAD) / (JAW_CLOSED_RAD - JAW_OPEN_RAD) - 1.0)   # keep the jaw

    # ------------------------------------------------------------------ cameras
    def _rectify(self, name: str, bgr: np.ndarray):
        cam = self.config["cameras"][name]
        calib = self.cal.cameras[cam["calibration"]]
        maps = getattr(self, "_maps", {})
        if name not in maps:   # undistort to the largest all-valid image (no black borders for SAM3 / MoGe-2)
            h, w = bgr.shape[:2]
            K0, dist = np.array(calib["K"], float), np.array(calib["dist"], float)
            K, _ = cv2.getOptimalNewCameraMatrix(K0, dist, (w, h), 0.0, (w, h))
            K[0, 0] = K[1, 1] = max(K[0, 0], K[1, 1])   # square pixels, as rendered in the twin (MoGe-2 reads one fov)
            maps[name] = (*cv2.initUndistortRectifyMap(K0, dist, None, K, (w, h), cv2.CV_16SC2), K)
            self._maps = maps
        map1, map2, K = maps[name]
        img = cv2.remap(bgr, map1, map2, cv2.INTER_LINEAR)
        x0, y0, size = cam["crop"]
        out = int(self.config.get("image_size", [512, 512])[0])
        img = cv2.resize(img[y0:y0 + size, x0:x0 + size], (out, out), interpolation=cv2.INTER_AREA)
        s = out / size
        K2 = np.array([[K[0, 0] * s, 0, (K[0, 2] - x0 + 0.5) * s - 0.5],
                       [0, K[1, 1] * s, (K[1, 2] - y0 + 0.5) * s - 0.5], [0, 0, 1.0]])
        return np.ascontiguousarray(img[..., ::-1]), K2

    def frames(self, camera_names: Sequence[str]) -> Mapping[str, CameraFrame]:
        t0 = time.monotonic()
        settle = 0.25 - (t0 - getattr(self, "_last_motion", 0.0))   # the stream lags a few frames behind the arm
        if settle > 0:
            time.sleep(settle)
        self._state_now()
        q6 = self._last[1]
        out = {}
        for name in camera_names:
            cam = self.config["cameras"][name]
            buf = urllib.request.urlopen(self.config["camstream_url"].format(cam["stream"]), timeout=5).read()
            rgb, K = self._rectify(name, cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR))
            calib = self.cal.cameras[cam["calibration"]]
            if "T_base_cam" in calib:
                pose = np.array(calib["T_base_cam"], float)
                # measured residual of the head-camera object estimates (11-15, offline): against wrist-camera truth
                # for 39 cube placements the policy's depth + box centre sat (-3.4, +6.7) mm off (95 % CI x -5.0..-1.9,
                # y 5.1..8.2); a horizontal shift of the reported camera pose moves every head-camera point by the same
                # amount and keeps the table plane at z = 0 (leave-one-episode-out median error 9.5 -> 7.1 mm)
                pose[:2, 3] += np.asarray(cam.get("xy_correction_m", (0.0, 0.0)), float)
            else:
                pose = self.cal.gripperframe_pose(q6) @ np.array(calib["T_gripperframe_cam"], float)
            out[name] = CameraFrame(rgb=rgb, depth_m=None, intrinsics=K, optical_in_robot_base=pose)
        self.timing["frame_calls"] += 1
        self.timing["frame_wall_s"] += time.monotonic() - t0
        return out

    # ------------------------------------------------------------------ end
    def park(self) -> None:
        """Lift, home, back to the rest pose the episode started from; torque off."""
        if not self._torque:
            return
        try:
            self._state_now()
            self._lift()
            cur = self._raw()
            home = np.r_[self._home_raw(), cur[5]]
            home[4] = cur[4]
            self._joint_move(home, "park_home")
            home[4:] = self.rest_raw[4:]
            self._joint_move(home, "park_roll")
            self._joint_move(self.rest_raw.copy(), "park_rest")
        finally:
            for i in self.cal.ids:
                self.bus.w1(i, A_TORQUE, 0)
            self._torque = False
            self._log(event="park", timing=self.timing)

    def close(self) -> None:
        try:
            self.park()
        finally:
            self.bus.close()

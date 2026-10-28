"""Harness side of the RoboTwin 2.0 simulator: one SAPIEN process, one XR1Driver-protocol view per arm.

RoboTwinSim owns the subprocess (RoboTwin python env) and the shared clock; ArmDriver is what each
arm's XR1FacadeEnv talks to: `command` turns an absolute grip-site goal (robot_base) into joint
targets with damped IK on that arm's twin and advances the whole scene one control tick (the other
arm keeps its last joint/jaw targets). robot_base is the primary arm's panda_link0 frame.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from captain_harness.adapters.hardware.facade import ARM_JOINTS, GRIP_SITE, CameraFrame, DriverState

from . import wire

HERE = Path(__file__).resolve().parent
FINGER_STROKE_M = 0.04
NON_TASK_ACTORS = {"wall", "table"}


class RoboTwinSim:
    """The SAPIEN scene of one RoboTwin task + seed, reached over a pipe."""

    def __init__(self, config: Mapping, task: Mapping, *, seed: int, log_path: Path | None = None) -> None:
        self.config = config
        self.task = task
        rt = config["robotwin"]
        python = os.path.expanduser(rt["python"])
        root = os.path.expanduser(rt["root"])
        to_child_r, to_child_w = os.pipe()
        from_child_r, from_child_w = os.pipe()
        env = dict(os.environ, ROBOTWIN_ROOT=root, ROBOTWIN_SHADER=rt.get("shader", "default"), ROBOTWIN_NO_PLANNER="1")
        if task.get("allow_unstable") or rt.get("allow_unstable"):
            env["ROBOTWIN_ALLOW_UNSTABLE"] = "1"
        env.pop("PYTHONPATH", None)
        if rt.get("vulkan_device") is not None:
            env["VK_DEVICE_INDEX"] = str(rt["vulkan_device"])
        log = open(log_path, "ab") if log_path else subprocess.DEVNULL
        self._proc = subprocess.Popen([python, str(HERE / "sapien_server.py"), str(to_child_r), str(from_child_w)],
                                      pass_fds=(to_child_r, from_child_w), stdin=subprocess.DEVNULL,
                                      stdout=log, stderr=subprocess.STDOUT, env=env, cwd=root)
        os.close(to_child_r)
        os.close(from_child_w)
        self._w = os.fdopen(to_child_w, "wb")
        self._r = os.fdopen(from_child_r, "rb")
        self.control_frequency = int(config["control"]["control_frequency_hz"])
        cams = {}
        for name, cam in config["cameras"].items():
            cams[cam["sapien_name"]] = {k: v for k, v in cam.items() if k not in ("link", "depth", "sapien_name")}
        for name, cam in (config.get("extra_sapien_cameras") or {}).items():
            cams[name] = dict(cam)
        init_seed = int(task.get("robotwin_seed", seed))
        info = self.rpc("init", task=task["robotwin_task"], seed=init_seed,
                        embodiment=rt.get("embodiment", "franka-panda"), embodiment_dis=float(rt.get("embodiment_dis", 0.6)),
                        cameras=cams, home=None, settle_s=0.0, gripper_rate_m_s=float(rt.get("gripper_rate_m_s", 0.05)),
                        domain_randomization=task.get("domain_randomization"),
                        robot_pose_offset=(np.asarray(rt.get("robot_pose_offset_world_m", [0.0, 0.0, 0.0]), float)
                                           + np.asarray(task.get("robot_pose_offset_extra_world_m", [0.0, 0.0, 0.0]), float)).tolist(),
                        layout_embodiment=task.get("layout_embodiment") or rt.get("layout_embodiment"),
                        task_config=task.get("task_config", "demo_clean"),
                        gripper_links=rt.get("gripper_links"), wrist_camera_links=rt.get("wrist_camera_links"),
                        arm_base_links=rt.get("arm_base_links"),
                        **({"arm_drive": tuple(float(v) for v in rt["arm_drive"])} if rt.get("arm_drive") else {}))
        self.timestep = float(info["timestep"])
        self.joint_limits = {arm: np.asarray(v, float) for arm, v in (info.get("joint_limits") or {}).items()}
        self.unstable_objects = list(info.get("unstable_objects") or [])
        self.primary = config.get("primary_arm", "right")
        self.world_to_base = np.linalg.inv(info["bases"][self.primary])
        self.bases = {arm: self.world_to_base @ pose for arm, pose in info["bases"].items()}
        self.state = info["state"]
        # cameras given in the robot_base frame (training-aligned LIBERO-like agentview): convert to world now
        base_to_world = np.linalg.inv(self.world_to_base)
        for name, cam in config["cameras"].items():
            rb = cam.get("robot_base_pose")
            if rb:
                pos = base_to_world @ np.r_[np.asarray(rb["position"], float), 1.0]
                fwd = base_to_world[:3, :3] @ np.asarray(rb["forward"], float)
                left = base_to_world[:3, :3] @ np.asarray(rb["left"], float)
                self.rpc("set_camera_pose", name=cam["sapien_name"], position=pos[:3], forward=fwd, left=left)
        self._substep_debt = 0.0
        self._frames_cache: tuple[float, dict] | None = None

    def rpc(self, cmd: str, **kwargs):
        wire.send(self._w, {"cmd": cmd, **kwargs})
        reply = wire.recv(self._r)
        if not reply["ok"]:
            raise RuntimeError(f"sapien server {cmd}: {reply['error']}\n{reply.get('traceback', '')}")
        return reply.get("result")

    # ------------------------------------------------------------------ control
    def home(self, joints: Mapping[str, Sequence[float]], settle_s: float) -> None:
        for arm, q in joints.items():
            self.rpc("teleport", arm=arm, q=np.asarray(q, float))
        self.tick({}, ticks=max(1, int(settle_s * self.control_frequency)))
        self.rpc("reset_clock")
        self.state = self.rpc("read")

    companions: dict = {}

    def tick(self, targets: Mapping[str, Mapping], ticks: int = 1) -> dict:
        # move_both: an arm with an active companion advances along its own path in the same control tick
        targets = dict(targets)
        for arm, comp in list(self.companions.items()):
            if arm not in targets and not comp.finished:
                cmd = comp.next_command(self.state)
                if cmd:
                    targets[arm] = cmd
        self._substep_debt += ticks / self.control_frequency / self.timestep
        n = int(math.floor(self._substep_debt + 1e-9))
        self._substep_debt -= n
        self.state = self.rpc("step", targets=dict(targets), n=n)
        self._frames_cache = None
        return self.state

    def set_target(self, arm: str, q: np.ndarray) -> None:
        self.rpc("step", targets={arm: {"q": np.asarray(q, float)}}, n=0)

    def frames(self, names: Sequence[str]) -> dict:
        key = self.state["time_s"]
        if self._frames_cache is not None and self._frames_cache[0] == key and all(n in self._frames_cache[1] for n in names):
            return self._frames_cache[1]
        raw = self.rpc("frames", names=list(names))
        if self._frames_cache is not None and self._frames_cache[0] == key:
            raw = {**self._frames_cache[1], **raw}
        self._frames_cache = (key, raw)
        return raw

    # ------------------------------------------------------------------ host-only
    def success(self) -> bool:
        return bool(self.rpc("success"))

    def objects(self) -> dict:
        out = {}
        for name, obj in self.rpc("objects").items():
            if name in NON_TASK_ACTORS:
                continue
            pose = np.eye(4)
            w, x, y, z = obj["q_wxyz"]
            pose[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
            pose[:3, 3] = obj["p"]
            out[name] = {"pose": self.world_to_base @ pose, "model": obj.get("model")}
        return out

    def close(self) -> None:
        try:
            self.rpc("close")
        except Exception:  # noqa: BLE001
            pass
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()


class Companion:
    """The second arm of a move_both call: straight Cartesian path (position lerp, rotation slerp) from its measured
    EEF to its goal at its own speed limits, one IK step per shared control tick, while the leading arm executes
    its own guarded move_to. Stops (holding the measured pose) on the legacy force rule (+5 N over its start or 18 N).
    A gripper command is applied at the first tick (both jaws move together)."""

    def __init__(self, driver: "ArmDriver", goal: np.ndarray | None, *, open_cmd: float | None,
                 max_speed_m_s: float, max_angular_speed_rad_s: float) -> None:
        from scipy.spatial.transform import Slerp
        self.driver, self.goal, self.open_cmd = driver, goal, open_cmd
        st = driver.sim.state[driver.arm]
        self.start = driver.fk(driver._pad(st["q"]))
        f0 = st.get("force_world")
        self.f0 = float(np.linalg.norm(f0)) if f0 is not None else 0.0
        self.k, self.finished, self.stopped_reason, self.peak_force = 0, False, None, self.f0
        if goal is not None:
            dist = float(np.linalg.norm(goal[:3, 3] - self.start[:3, 3]))
            ang = float(np.linalg.norm(Rotation.from_matrix(goal[:3, :3] @ self.start[:3, :3].T).as_rotvec()))
            dur = max(dist / max_speed_m_s, ang / max_angular_speed_rad_s, 1.0 / driver.control_frequency)
            self.n = max(1, int(math.ceil(dur * driver.control_frequency)))
            self._slerp = Slerp([0, 1], Rotation.from_matrix([self.start[:3, :3], goal[:3, :3]]))
        else:
            self.n = 1

    def next_command(self, state) -> dict | None:
        st = state[self.driver.arm]
        f = st.get("force_world")
        fn = float(np.linalg.norm(f)) if f is not None else 0.0
        self.peak_force = max(self.peak_force, fn)
        cmd = {}
        if self.k == 0 and self.open_cmd is not None:
            cmd["open"] = self.open_cmd
        if self.goal is not None and (fn >= 18.0 or fn - self.f0 >= 5.0):
            self.finished, self.stopped_reason = True, "contact_force"
            cmd["q"] = np.asarray(st["q"], float)
            return cmd
        self.k += 1
        if self.goal is not None:
            s = min(1.0, self.k / self.n)
            g = np.eye(4)
            g[:3, 3] = self.start[:3, 3] + s * (self.goal[:3, 3] - self.start[:3, 3])
            g[:3, :3] = self._slerp([s]).as_matrix()[0]
            cmd["q"] = self.driver._ik(g, self.driver._pad(st["q"]))[:self.driver._dof]
        if self.k >= self.n:
            self.finished, self.stopped_reason = True, "path_completed"
        return cmd or None


class ArmDriver:
    """XR1Driver protocol for one arm of a RoboTwinSim."""

    def __init__(self, sim: RoboTwinSim, arm: str, twin_xml: str, *, home_joints: Sequence[float],
                 camera_map: Mapping[str, str], depth_cameras: Sequence[str], arm_dof: int = 7,
                 finger_stroke_m: float = FINGER_STROKE_M, ik_limit_margin_rad: float = 1e-4) -> None:
        # arm_dof 6 (aloha ARX5): the twin's 7th joint is a locked placeholder; the sim gets the first 6
        self._dof = int(arm_dof)
        self._stroke = float(finger_stroke_m)
        self.sim = sim
        self.arm = arm
        self.control_frequency = sim.control_frequency
        self._camera_map = dict(camera_map)        # harness camera name -> sapien camera name
        self._depth = set(depth_cameras)
        self.model = mujoco.MjModel.from_xml_string(twin_xml)
        self._data = mujoco.MjData(self.model)
        m = self.model
        self._q = np.asarray([m.jnt_qposadr[m.joint(n).id] for n in ARM_JOINTS])
        self._v = np.asarray([m.jnt_dofadr[m.joint(n).id] for n in ARM_JOINTS])
        self._limits = m.jnt_range[[m.joint(n).id for n in ARM_JOINTS]].copy()
        # driver IK stays a margin inside the twin (= planner) limits, so tracking overshoot never reads as "exceeded"
        margin = float(ik_limit_margin_rad)
        self._ik_limits = np.c_[self._limits[:, 0] + np.minimum(margin, 0.45 * np.ptp(self._limits, axis=1)),
                                self._limits[:, 1] - np.minimum(margin, 0.45 * np.ptp(self._limits, axis=1))]
        self._site = m.site(GRIP_SITE).id
        self._home = np.asarray(home_joints, float)
        self._open = 1.0

    # ------------------------------------------------------------------ helpers
    def _ik(self, goal: np.ndarray, q: np.ndarray, iterations: int = 20) -> np.ndarray:
        """Damped least-squares IK on the twin (same solver as the XR1 sim driver's emulated MPC)."""
        m, d = self.model, self._data
        q = q.copy()
        for _ in range(iterations):
            d.qpos[self._q] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            dp = goal[:3, 3] - d.site_xpos[self._site]
            dr = Rotation.from_matrix(goal[:3, :3] @ d.site_xmat[self._site].reshape(3, 3).T).as_rotvec()
            if np.linalg.norm(dp) < 2e-4 and np.linalg.norm(dr) < 2e-3:
                break
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, jp, jr, self._site)
            jac = np.vstack([jp[:, self._v], jr[:, self._v]])
            step = jac.T @ np.linalg.solve(jac @ jac.T + 0.002 * np.eye(6), np.r_[dp, dr])
            step *= min(1.0, 0.1 / max(np.abs(step).max(), 1e-9))
            q = np.clip(q + step, self._ik_limits[:, 0], self._ik_limits[:, 1])
        return q

    def _state(self) -> DriverState:
        st = self.sim.state[self.arm]
        force = st.get("force_world")
        force_base = None if force is None else self.sim.world_to_base[:3, :3] @ np.asarray(force, float)
        return DriverState(joints=self._pad(st["q"]), joint_velocities=self._pad(st["dq"]),
                           gripper_open_fraction=float(np.clip(np.mean(st["fingers"]) / self._stroke, 0.0, 1.0)),
                           wrist_force_world=force_base if force_base is not None else np.zeros(3),
                           time_s=float(self.sim.state["time_s"]))

    def fk(self, q) -> np.ndarray:
        m, d = self.model, self._data
        d.qpos[self._q] = np.asarray(q, float)
        mujoco.mj_kinematics(m, d)
        pose = np.eye(4)
        pose[:3, :3] = d.site_xmat[self._site].reshape(3, 3)
        pose[:3, 3] = d.site_xpos[self._site]
        return pose

    def _pad(self, q) -> np.ndarray:
        q = np.asarray(q, float).reshape(-1).copy()
        return np.r_[q, np.zeros(7 - len(q))] if len(q) < 7 else q

    # ------------------------------------------------------------------ driver API
    def reset(self) -> DriverState:
        return self._state()

    def read(self) -> DriverState:
        return self._state()

    def command(self, grip_site_goal: np.ndarray | None, gripper_action: float) -> DriverState:
        cmd = {"open": float(np.clip((1.0 - gripper_action) / 2.0, 0.0, 1.0))}
        if grip_site_goal is not None:
            # 6-DoF arms (aloha ARX5) converge slower near the edge of reach: more damped-LS iterations per control tick
            cmd["q"] = self._ik(np.asarray(grip_site_goal, float), self._pad(self.sim.state[self.arm]["q"]),
                                iterations=20 if self._dof >= 7 else 60)[:self._dof]
        self.sim.tick({self.arm: cmd})
        return self._state()

    def hold(self) -> None:
        self.sim.set_target(self.arm, np.asarray(self.sim.state[self.arm]["q"], float))

    # R5 F2 (config r5_settle_on_hold): the wrist force is the contact impulse summed over the last control tick, so after a
    # stop it still read the impact spike (60-280 N), and a stiff joint drive held at the measured (pressed-in) posture kept
    # pressing (~20 N); every later call was refused as initial_force_limit / emergency_limit without stepping (deadlock).
    # Compliant hold: re-target the measured posture each tick until the contact unloads (LIBERO's OSC hold does not keep
    # pressing after a stop), at most 10 ticks (0.5 s); the measured pose moves < 1 mm (probe).
    settle_on_hold = False
    SETTLE_FORCE_N = 2.0
    SETTLE_MAX_TICKS = 10

    def settle(self) -> DriverState:
        for _ in range(self.SETTLE_MAX_TICKS):
            self.hold()
            self.sim.tick({}, 1)
            st = self._state()
            if st.wrist_force_world is None or float(np.linalg.norm(st.wrist_force_world)) < self.SETTLE_FORCE_N:
                break
        return st

    def frames(self, camera_names: Sequence[str]) -> Mapping[str, CameraFrame]:
        raw = self.sim.frames([self._camera_map[n] for n in camera_names])
        out = {}
        for name in camera_names:
            f = raw[self._camera_map[name]]
            out[name] = CameraFrame(rgb=np.asarray(f["rgb"]), depth_m=np.asarray(f["depth"], np.float32) if name in self._depth else None,
                                    intrinsics=np.asarray(f["K"], float),
                                    optical_in_robot_base=self.sim.world_to_base @ np.asarray(f["optical_world"], float))
        return out

    def hidden_object_audit(self):
        return RoboTwinObjectAudit(self.sim, self)

    def success(self) -> bool:
        return self.sim.success()

    def close(self) -> None:
        if self.arm == self.sim.primary:  # the primary facade owns the episode's SAPIEN process
            self.sim.close()


class RoboTwinObjectAudit:
    """hidden-object-audit/v2 rows (robot_base): task actor poses, both EEFs, the task's own check_success."""

    def __init__(self, sim: RoboTwinSim, driver: ArmDriver) -> None:
        self._sim = sim
        self._driver = driver
        self._names = list(sim.objects())

    def header(self) -> dict:
        goal = ["robotwin_check_success", self._sim.task["robotwin_task"]]
        return {"schema_version": "hidden-object-audit/v2", "coordinate_frame": "robot_base", "goal_state": [goal],
                "goals": [{"goal_id": "goal_0", "predicate": goal[0], "arguments": goal[1:], "rule": {"type": "robotwin_check_success"}}],
                "entities": self._names, "targets": list(self._sim.task.get("targets", [])),
                "destinations": list(self._sim.task.get("destinations", [])), "movable": self._names, "fixtures": []}

    def sample(self) -> dict:
        objs = self._sim.objects()
        objects = {n: o["pose"][:3, 3].round(6).tolist() for n, o in objs.items()}
        yaws = {n: round(float(math.atan2(o["pose"][1, 0], o["pose"][0, 0])), 4) for n, o in objs.items()}
        try:
            satisfied = self._sim.success()
        except Exception:  # noqa: BLE001
            satisfied = None
        eef = {}
        for arm in ("left", "right"):
            hand = self._sim.world_to_base @ np.asarray(self._sim.state[arm]["hand"], float)
            eef[arm] = hand[:3, 3].round(6).tolist()
        return {"schema_version": "hidden-object-audit/v2", "objects": objects, "object_yaw_rad": yaws,
                "eef": eef[self._sim.primary], "hand_positions_by_arm": eef,
                "gripper_open_fraction": round(float(self._driver.read().gripper_open_fraction), 4),
                "goal_states": {"goal_0": {"satisfied": satisfied, "progress_kind": "predicate_only"}}}

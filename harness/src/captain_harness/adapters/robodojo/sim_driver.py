"""Harness side of the RoboDojo simulator: one Isaac Sim process, one XR1Driver-protocol view per X5 arm.

RoboDojoSim owns the subprocess (RoboDojo's Isaac python, started through rd_server.sh) and the shared clock;
ArmDriver is what each arm's XR1FacadeEnv talks to: `command` turns an absolute grip-site goal (robot_base) into
joint targets with damped IK on that arm's twin and advances the scene by ONE official RoboDojo control tick
(take_action at the benchmark's 25 Hz; the other arm keeps its last joint/jaw targets). robot_base is the primary
(right) arm's base_link frame.
"""
from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from captain_harness.adapters.hardware.facade import ARM_JOINTS, GRIP_SITE, CameraFrame, DriverState

from . import wire
from .twin import FINGER_STROKE_M

HERE = Path(__file__).resolve().parent


class RoboDojoSim:
    """The Isaac scene of one RoboDojo task + official eval layout, reached over a pipe."""

    def __init__(self, config: Mapping, task: Mapping, *, log_path: Path | None = None, out_dir: Path | None = None) -> None:
        self.config = config
        self.task = task
        rd = config["robodojo"]
        self.rpc_counts: dict[str, int] = {}
        self.client_pid = os.getpid()
        variant = os.environ.get("RD_LAYOUT_VARIANT", task.get("layout_variant", rd.get("layout_variant", "official")))
        try:
            info = self._start(config, task, variant, log_path, out_dir)
        except RuntimeError as exc:
            # a priori rule (ROUND 4): a reachable layout that fails RoboDojo's own stability check falls back to the
            # official layout for that episode, labelled official_fallback
            if variant != "reachable" or "UnStableError" not in str(exc):
                raise
            self._kill()
            info = self._start(config, task, "official", log_path, out_dir)
            variant = "official_fallback"
        info["layout_variant_used"] = variant
        self.info = info
        self.timestep = float(info["timestep"])
        self.control_frequency = int(round(1.0 / self.timestep))
        self.primary = config.get("primary_arm", "right")
        self.world_to_base = np.linalg.inv(info["bases"][self.primary])
        self.bases = {arm: self.world_to_base @ pose for arm, pose in info["bases"].items()}
        self.robots = info["robots"]
        self.state = info["state"]
        self._frames_cache: tuple[float, dict] | None = None
        self.final_status: dict | None = None

    def _start(self, config, task, variant, log_path, out_dir):
        rd = config["robodojo"]
        to_child_r, to_child_w = os.pipe()
        from_child_r, from_child_w = os.pipe()
        env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "VIRTUAL_ENV", "MUJOCO", "PYOPENGL", "CUDA_VISIBLE", "__EGL", "LD_LIBRARY_PATH"))}
        env["RD_ROOT"] = os.path.expanduser(rd["root"])
        env["RD_ENV"] = os.path.expanduser(rd["env_sh"])
        gpu = int(os.environ.get("RD_GPU", rd.get("gpu", 0)))
        if rd.get("confine_gpu", True):  # one GPU only: Vulkan UUID filter (rd_server.sh) + torch/warp pinned (rd_server.py)
            env["RD_GPU_PHYS"] = str(gpu)
        log = open(log_path, "ab") if log_path else subprocess.DEVNULL
        self._proc = subprocess.Popen(["bash", str(HERE / "rd_server.sh"), str(to_child_r), str(from_child_w)],
                                      pass_fds=(to_child_r, from_child_w), stdin=subprocess.DEVNULL,
                                      stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        os.close(to_child_r)
        os.close(from_child_w)
        self._w = os.fdopen(to_child_w, "wb")
        self._r = os.fdopen(from_child_r, "rb")
        return self.rpc("init", task=task["robodojo_task"], layout_id=int(task["layout_id"]),
                        eval_seed=int(task.get("eval_seed", 0)), gpu=gpu,
                        step_limit_factor=float(os.environ.get("RD_STEP_LIMIT_FACTOR", rd.get("step_limit_factor", 20.0))),
                        head_render_scale=float(os.environ.get("RD_HEAD_RENDER_SCALE", rd.get("head_render_scale", 1.0))),
                        layout_variant=variant,
                        out_dir=str(out_dir) if out_dir else None)

    def _kill(self):
        try:
            os.killpg(self._proc.pid, 9)
        except Exception:  # noqa: BLE001
            pass
        try:
            self._proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            pass

    def rpc(self, cmd: str, **kwargs):
        self.rpc_counts[cmd] = self.rpc_counts.get(cmd, 0) + 1
        wire.send(self._w, {"cmd": cmd, **kwargs})
        reply = wire.recv(self._r)
        if not reply["ok"]:
            raise RuntimeError(f"robodojo server {cmd}: {reply['error']}\n{reply.get('traceback', '')}")
        return reply.get("result")

    # ------------------------------------------------------------------ control
    companions: dict = {}   # move_both (robotwin2.environment): arm -> Companion advancing in the same control ticks

    def tick(self, targets: Mapping[str, Mapping], ticks: int = 1) -> dict:
        targets = dict(targets)
        for arm, comp in list(self.companions.items()):
            if arm not in targets and not comp.finished:
                cmd = comp.next_command(self.state)
                if cmd:
                    targets[arm] = cmd
        self.state = self.rpc("step", targets=targets, n=int(ticks))
        self._frames_cache = None
        return self.state

    def set_target(self, arm: str, q: np.ndarray) -> None:
        self.rpc("step", targets={arm: {"q": np.asarray(q, float)[:6]}}, n=0)

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

    def finalize(self, reason: str = "harness") -> dict:
        if self.final_status is None:
            self.final_status = self.rpc("finalize", reason=reason)
        return self.final_status

    def objects(self) -> dict:
        out = {}
        for name, obj in self.rpc("objects").items():
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
            self._proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self._proc.pid, 9)
            except Exception:  # noqa: BLE001
                self._proc.kill()


class ArmDriver:
    """XR1Driver protocol for one X5 arm of a RoboDojoSim (6 measured joints + the twin's locked joint 7)."""

    reread_after_hold = True   # hold() advances simulated time: the facade re-reads the state after it (hardware/facade.py)

    def __init__(self, sim: RoboDojoSim, arm: str, twin_xml: str, *, camera_map: Mapping[str, str],
                 depth_cameras: Sequence[str], status_path: Path | None = None) -> None:
        self.sim = sim
        self.arm = arm
        self.control_frequency = sim.control_frequency
        self._camera_map = dict(camera_map)        # harness camera name -> RoboDojo camera name
        self._depth = set(depth_cameras)
        self._status_path = status_path
        self.model = mujoco.MjModel.from_xml_string(twin_xml)
        self._data = mujoco.MjData(self.model)
        m = self.model
        self._q = np.asarray([m.jnt_qposadr[m.joint(n).id] for n in ARM_JOINTS])
        self._v = np.asarray([m.jnt_dofadr[m.joint(n).id] for n in ARM_JOINTS])
        self._limits = m.jnt_range[[m.joint(n).id for n in ARM_JOINTS]].copy()
        self._site = m.site(GRIP_SITE).id
        self._q_cmd: np.ndarray | None = None   # last joint command sent (IK seed)
        self._loop_ended = False

    # ------------------------------------------------------------------ helpers (also used by robotwin2.sim_driver.Companion)
    _dof = 6
    _stroke = FINGER_STROKE_M

    def fk(self, q) -> np.ndarray:
        m, d = self.model, self._data
        d.qpos[self._q] = self._pad(q)
        mujoco.mj_kinematics(m, d)
        pose = np.eye(4)
        pose[:3, :3] = d.site_xmat[self._site].reshape(3, 3)
        pose[:3, 3] = d.site_xpos[self._site]
        return pose

    @staticmethod
    def _pad(q) -> np.ndarray:
        q = np.asarray(q, float).reshape(-1).copy()
        return np.r_[q, np.zeros(7 - len(q))] if len(q) < 7 else q[:7]

    @staticmethod
    def _q7(q6) -> np.ndarray:
        return np.r_[np.asarray(q6, float).reshape(-1)[:6], 0.0]

    def _ik(self, goal: np.ndarray, q: np.ndarray, iterations: int = 80) -> np.ndarray:
        """Damped least-squares IK on the twin (same solver as the RoboTwin/XR1 sim drivers). 80 iterations (was 20):
        near the X5 reach edge 20 damped steps did not converge where the planner IK (180 + restarts) accepts the
        target, so execution stalled in free space."""
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
            q = np.clip(q + step, self._limits[:, 0] + 1e-4, self._limits[:, 1] - 1e-4)
        return q

    def _state(self) -> DriverState:
        st = self.sim.state[self.arm]
        fingers = np.clip(np.asarray(st["fingers"], float), 0.0, FINGER_STROKE_M)
        return DriverState(joints=self._q7(st["q"]), joint_velocities=self._q7(st["dq"]),
                           gripper_open_fraction=float(np.clip(np.mean(fingers) / FINGER_STROKE_M, 0.0, 1.0)),
                           wrist_force_world=(np.asarray(st["force_world"], float) if st.get("force_world") is not None
                                              else np.zeros(3)),
                           time_s=float(self.sim.state["time_s"]))

    # ------------------------------------------------------------------ driver API
    def reset(self) -> DriverState:
        return self._state()

    def read(self) -> DriverState:
        return self._state()

    def command(self, grip_site_goal: np.ndarray | None, gripper_action: float) -> DriverState:
        cmd = {"open": float(np.clip((1.0 - gripper_action) / 2.0, 0.0, 1.0))}
        if grip_site_goal is not None:
            # Seed the damped IK with the previous joint command, not the measured joints: under load the PD sag makes
            # the measured seed alternate tick to tick and the (not fully converged) IK output follows it, a period-2
            # limit cycle (~0.1 mm/tick EEF) that never lets a loaded move settle. Fall back to measured after large jumps.
            measured = np.asarray(self.sim.state[self.arm]["q"], float)[:6]
            seed = self._q_cmd if self._q_cmd is not None and np.abs(self._q_cmd - measured).max() < 0.05 else measured
            q = self._ik(np.asarray(grip_site_goal, float), self._q7(seed))
            cmd["q"] = q[:6]
            self._q_cmd = q[:6].copy()
        self.sim.tick({self.arm: cmd})
        return self._state()

    def hold(self) -> None:
        """Joint target := measured joints, then two native steps so a released contact load actually relaxes
        (a stiff joint-PD arm stopped against an object would otherwise still read its contact force)."""
        self._q_cmd = np.asarray(self.sim.state[self.arm]["q"], float)[:6].copy()
        self.sim.set_target(self.arm, self._q_cmd)
        self.sim.tick({}, ticks=2)

    def frames(self, camera_names: Sequence[str]) -> Mapping[str, CameraFrame]:
        raw = self.sim.frames([self._camera_map[n] for n in camera_names])
        out = {}
        for name in camera_names:
            f = raw[self._camera_map[name]]
            depth = f.get("depth")
            out[name] = CameraFrame(rgb=np.asarray(f["rgb"]),
                                    depth_m=np.asarray(depth, np.float32) if (name in self._depth and depth is not None) else None,
                                    intrinsics=np.asarray(f["K"], float),
                                    optical_in_robot_base=self.sim.world_to_base @ np.asarray(f["optical_world"], float))
        return out

    def hidden_object_audit(self):
        return RoboDojoObjectAudit(self.sim, self)

    def success(self) -> bool:
        """Per-action hidden audit: latched official success (RoboDojo ends an episode the moment it succeeds)."""
        return self.sim.success()

    def mark_loop_ended(self) -> None:
        self._loop_ended = True

    def host_predicate(self) -> bool:
        """The adapter's host predicate. The harness also samples it after EVERY action (success_ever), so before the
        agent loop has ended it must be side-effect free: latched official success only. Once the loop ended
        (episode.py wraps adapter.finalize) it runs the official final check exactly once."""
        if not self._loop_ended:
            return self.sim.success()
        return self.final_success()

    def final_success(self) -> bool:
        """Final host predicate: the official final check (what RoboDojo runs at its step limit), once."""
        status = dict(self.sim.finalize("agent_loop_ended"))
        status["client"] = {"pid": os.getpid(), "sim_client_pid": self.sim.client_pid, "rpc_counts": dict(self.sim.rpc_counts),
                            "arm": self.arm}
        if self._status_path is not None:
            import json
            self._status_path.write_text(json.dumps(status, indent=2, default=str))
        return bool(status["official_success"])

    def close(self) -> None:
        if self.arm == self.sim.primary:  # the primary facade owns the episode's Isaac process
            if self.sim.final_status is None and self._status_path is not None:
                try:
                    self.final_success()
                except Exception:  # noqa: BLE001
                    pass
            self.sim.close()


class RoboDojoObjectAudit:
    """hidden-object-audit/v2 rows (robot_base): task instance positions, both TCPs, latched official success."""

    def __init__(self, sim: RoboDojoSim, driver: ArmDriver) -> None:
        self._sim = sim
        self._driver = driver
        try:
            self._names = list(sim.objects())
        except Exception:  # noqa: BLE001
            self._names = []

    def header(self) -> dict:
        goal = ["robodojo_official_success", self._sim.task["robodojo_task"]]
        return {"schema_version": "hidden-object-audit/v2", "coordinate_frame": "robot_base", "goal_state": [goal],
                "goals": [{"goal_id": "goal_0", "predicate": goal[0], "arguments": goal[1:], "rule": {"type": "robodojo_reward_manager"}}],
                "entities": self._names, "targets": [], "destinations": [], "movable": self._names, "fixtures": []}

    def sample(self) -> dict:
        try:
            objs = self._sim.objects()
        except Exception:  # noqa: BLE001
            objs = {}
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

"""XR1Driver for the SO-101 twin (harness adapters/hardware/facade.py protocol) + host-only object audit."""
from __future__ import annotations

import collections
import hashlib
import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from captain_harness.adapters.hardware.facade import CameraFrame, DriverState

from .camera import Cameras, Frame
from .perception import depth_noise
from .sim import SoarmSim
from .twin import gripper_action_to_jaw

TWIN_ARM = tuple(f"robot0_joint{k}" for k in range(1, 8))
FORCE_DEADBAND_N = 0.6
FORCE_NOISE_N = 0.15


def image_key(rgb: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(rgb).tobytes()).hexdigest()


class SoarmDriver:
    """One SO-101 in a SoarmSim. IK on the kinematic twin: position exact, orientation as far as 5 DoF allow."""

    reread_after_hold = True

    def __init__(self, sim: SoarmSim, twin_xml: str, *, image_size=(512, 512), force_seed: int = 0,
                 privileged_frames: bool = True, depth_noise_sigma_1m: float = 0.0, sensor_depth: bool = True,
                 timing_path: Path | None = None) -> None:
        self.sim = sim
        self.control_frequency = sim.control_frequency
        self.model = mujoco.MjModel.from_xml_string(twin_xml)
        self._data = mujoco.MjData(self.model)
        m = self.model
        self._q = np.array([m.jnt_qposadr[m.joint(n).id] for n in TWIN_ARM])
        self._v = np.array([m.jnt_dofadr[m.joint(n).id] for n in TWIN_ARM])
        self._limits = m.jnt_range[[m.joint(n).id for n in TWIN_ARM]][:5].copy()
        self._site = m.site("gripper0_grip_site").id
        self._q_cmd: np.ndarray | None = None
        self._jaw_cmd = float(sim.data.ctrl[5])
        # privileged_frames: render MuJoCo segmentation and keep the last 64 frames for perception_sim (privileged
        # perception mode and host tools only); real perception episodes render RGB-D only and keep nothing.
        # sensor_depth=False (SOARM_DEPTH=moge2): RGB only, frames carry depth_m=None and the facade's monocular depth
        # estimator (harness adapters/hardware/mono_depth.py, MoGe-2) fills it from the RGB, as on the RGB-only arm.
        if privileged_frames and not sensor_depth:
            raise ValueError("privileged perception (tests) runs on sensor depth")
        self.privileged_frames = privileged_frames
        self.sensor_depth = sensor_depth
        self.cameras = Cameras(sim.model, sim.geom_object, list(sim.specs), size=image_size,
                               segmentation=privileged_frames, depth=sensor_depth)
        self.frame_registry: collections.OrderedDict[str, Frame] = collections.OrderedDict()
        self._rng = np.random.default_rng(force_seed)
        self.depth_noise_sigma_1m = float(depth_noise_sigma_1m)
        self._depth_rng = np.random.default_rng([force_seed, 7])
        # host-only wall accounting: physics/IK (command, hold) and rendering, next to the simulated (= real-arm) time
        self.timing = {"command_calls": 0, "command_wall_s": 0.0, "render_calls": 0, "render_wall_s": 0.0,
                       # servo-validity check of faster motion profiles: peak |joint velocity| (rad/s) and peak
                       # |joint acceleration| (rad/s^2, between control ticks) of the 5 arm joints over the episode
                       "peak_joint_vel": [0.0] * 5, "peak_joint_acc": [0.0] * 5}
        self._timing_path = timing_path
        self._last_qvel: np.ndarray | None = None

    # ------------------------------------------------------------------ IK
    def _ik(self, goal: np.ndarray, q: np.ndarray, iterations: int = 60, rot_weight: float = 0.35) -> np.ndarray:
        """Damped least squares over the 5 real joints; the rotation residual is down-weighted so the unreachable
        component of an infeasible orientation costs position accuracy as little as possible."""
        m, d = self.model, self._data
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
            q = np.clip(q + step, self._limits[:, 0] + 1e-4, self._limits[:, 1] - 1e-4)
        return q

    # ------------------------------------------------------------------ state
    def _wrist_force(self) -> np.ndarray:
        """External force on the hand (gripper body, jaw, camera mount) in robot_base, as a calibrated servo-load
        estimator would report it: the simulated contact forces on the hand plus Gaussian noise and a deadband.
        (Inverting the arm dynamics from actuator torques in sim gave 10-25 N phantom loads while carrying 20 g, from
        PD tracking/inertia terms; the real-arm estimator must be calibrated against this signal.)"""
        sim = self.sim
        m, d = sim.model, sim.data
        hand = sim.fixed_side | {sim.jaw_body}
        total = np.zeros(3)
        f6 = np.zeros(6)
        n = d.ncon
        g1s, g2s = d.contact.geom1[:n], d.contact.geom2[:n]
        frames = d.contact.frame[:n]
        for i in range(n):
            b1, b2 = int(m.geom_bodyid[g1s[i]]), int(m.geom_bodyid[g2s[i]])
            in1, in2 = b1 in hand, b2 in hand
            if in1 == in2:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            world = frames[i].reshape(3, 3).T @ f6[:3]       # force of geom1 on geom2, world frame
            total += world if in2 else -world
        force = total + self._rng.normal(0.0, FORCE_NOISE_N, 3)
        return np.zeros(3) if float(np.linalg.norm(force)) < FORCE_DEADBAND_N else force

    def _state(self) -> DriverState:
        sim = self.sim
        return DriverState(joints=np.r_[sim.arm_qpos(), 0.0, 0.0], joint_velocities=np.r_[sim.arm_qvel(), 0.0, 0.0],
                           gripper_open_fraction=sim.open_fraction(), wrist_force_world=self._wrist_force(),
                           time_s=sim.time_s)

    # ------------------------------------------------------------------ XR1Driver
    def reset(self) -> DriverState:
        return self._state()

    def read(self) -> DriverState:
        return self._state()

    def command(self, grip_site_goal: np.ndarray | None, gripper_action: float) -> DriverState:
        t0 = time.perf_counter()
        self._jaw_cmd = gripper_action_to_jaw(gripper_action)
        target = None
        if grip_site_goal is not None:
            measured = self.sim.arm_qpos()
            seed = self._q_cmd if self._q_cmd is not None and np.abs(self._q_cmd - measured).max() < 0.05 else measured
            target = self._ik(np.asarray(grip_site_goal, float), seed)
            self._q_cmd = target.copy()
        self.sim.tick(target, self._jaw_cmd)
        qvel = np.asarray(self.sim.arm_qvel(), float)[:5]
        self.timing["peak_joint_vel"] = np.maximum(self.timing["peak_joint_vel"], np.abs(qvel)).tolist()
        if self._last_qvel is not None:
            acc = np.abs(qvel - self._last_qvel) * self.control_frequency
            self.timing["peak_joint_acc"] = np.maximum(self.timing["peak_joint_acc"], acc).tolist()
        self._last_qvel = qvel
        self.timing["command_calls"] += 1
        self.timing["command_wall_s"] += time.perf_counter() - t0
        self._record("command", 1)
        if getattr(self, "_disturb", None) is not None:      # eval-only disturbance demo (soarm/disturb.py)
            self._disturb.after_tick(self._rec_idx - 1)
        return self._state()

    def hold(self) -> None:
        t0 = time.perf_counter()
        self._q_cmd = self.sim.arm_qpos()
        self.sim.tick(self._q_cmd, self._jaw_cmd, ticks=2)
        self.timing["command_wall_s"] += time.perf_counter() - t0
        self._record("hold", 2)
        if getattr(self, "_disturb", None) is not None:      # eval-only disturbance demo (soarm/disturb.py)
            self._disturb.after_tick(self._rec_idx - 1)

    def start_recording(self, record_dir: Path, cameras: Sequence[str], size: int = 512) -> None:
        """Eval-only video recorder (tools/episode_video.py): after every control tick, render the given cameras
        (RGB only, own renderer) side by side to <record_dir>/f<idx>.jpg and log wall time + render cost to
        frames.jsonl, so the video can put motion on the real-arm timeline (one tick = 1/control_frequency s)."""
        import mujoco as _mj
        self._rec_dir = Path(record_dir)
        self._rec_dir.mkdir(parents=True, exist_ok=True)
        self._rec_cams = list(cameras)
        self._rec_renderer = _mj.Renderer(self.sim.model, size, size)
        self._rec_log = (self._rec_dir / "frames.jsonl").open("a")
        self._rec_idx = 0
        self._record("start", 0)
        from .disturb import Disturbance, enabled          # SOARM_DISTURB=<seed>; unset = off
        if enabled() is not None:
            self._disturb = Disturbance(self.sim, enabled(), self._rec_dir)

    def _record(self, kind: str, ticks: int) -> None:
        if getattr(self, "_rec_dir", None) is None:
            return
        from PIL import Image
        t_wall = time.time()
        t0 = time.perf_counter()
        tiles = []
        for cam in self._rec_cams:
            self._rec_renderer.update_scene(self.sim.data, camera=cam)
            tiles.append(self._rec_renderer.render())
        Image.fromarray(np.concatenate(tiles, axis=1)).save(self._rec_dir / f"f{self._rec_idx:06d}.jpg", quality=85)
        self._rec_log.write(json.dumps({"i": self._rec_idx, "t_wall": t_wall, "render_s": time.perf_counter() - t0,
                                        "kind": kind, "ticks": ticks, "sim_time_s": float(self.sim.time_s)}) + "\n")
        self._rec_log.flush()
        self._rec_idx += 1

    def _write_timing(self) -> None:
        if self._timing_path is not None:
            self._timing_path.write_text(json.dumps({**self.timing, "sim_time_s": float(self.sim.time_s)}))

    def frames(self, camera_names: Sequence[str]) -> Mapping[str, CameraFrame]:
        t0 = time.perf_counter()
        out = {}
        for name in camera_names:
            f = self.cameras.render(self.sim.data, name)
            if self.privileged_frames:
                self.frame_registry[image_key(f.rgb)] = f
                while len(self.frame_registry) > 64:
                    self.frame_registry.popitem(last=False)
            depth = (depth_noise(f.depth_m, self.depth_noise_sigma_1m, self._depth_rng)
                     if self.depth_noise_sigma_1m > 0 and f.depth_m is not None else f.depth_m)
            out[name] = CameraFrame(rgb=f.rgb, depth_m=depth, intrinsics=f.intrinsics,
                                    optical_in_robot_base=f.optical_in_base)
        self.timing["render_calls"] += 1
        self.timing["render_wall_s"] += time.perf_counter() - t0
        self._write_timing()
        return out

    def hidden_object_audit(self):
        return SoarmObjectAudit(self)

    def success(self) -> bool:
        return self.sim.success()

    def close(self) -> None:
        self.cameras.close()
        if getattr(self, "_rec_dir", None) is not None:
            self._rec_log.close()
            self._rec_renderer.close()


class SoarmObjectAudit:
    """hidden-object-audit/v2 rows (robot_base frame): object positions/yaws, grip point, jaw, per-goal atoms."""

    def __init__(self, driver: SoarmDriver) -> None:
        self._driver = driver
        self._sim = driver.sim

    def header(self) -> dict:
        task = self._sim.task
        goals = [{"goal_id": f"goal_{i}", "predicate": a[0], "arguments": list(a[1]), "rule": {"type": "soarm_atom",
                  "params": dict(a[2])}} for i, a in enumerate(task.atoms)]
        names = list(self._sim.specs)
        movable = [n for n, s in self._sim.specs.items() if s.kind != "pad"]
        return {"schema_version": "hidden-object-audit/v2", "coordinate_frame": "robot_base",
                "goal_state": [[a[0], *a[1]] for a in task.atoms], "goals": goals, "entities": names,
                "targets": sorted({a[1][0] for a in task.atoms}), "destinations": [], "movable": movable,
                "fixtures": [n for n in names if n not in movable], "task": task.to_json()}

    def sample(self) -> dict:
        sim = self._sim
        poses = sim.poses()
        values = sim.atom_values()
        return {"schema_version": "hidden-object-audit/v2",
                "objects": {n: p.pos.round(6).tolist() for n, p in poses.items()},
                "object_yaw_rad": {n: round(float(np.arctan2(p.rot[1, 0], p.rot[0, 0])), 4) for n, p in poses.items()},
                "eef": sim.grip_pose()[:3, 3].round(6).tolist(),
                "gripper_open_fraction": round(sim.open_fraction(), 4), "held": sorted(sim.held()),
                "goal_states": {f"goal_{i}": {"satisfied": bool(v), "progress_kind": "predicate_only"}
                                for i, v in enumerate(values)}}

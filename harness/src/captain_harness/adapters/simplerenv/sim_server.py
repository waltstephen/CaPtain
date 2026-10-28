"""SimplerEnv (ManiSkill2_real2sim, SAPIEN 2.2) process: runs in the SimplerEnv python env, not the harness one.

    python sim_server.py <read_fd> <write_fd>

Builds the official SimplerEnv environment of one episode (env id, robot, scene, overlay, robot/object init options
exactly as SimplerEnv's evaluation scripts pass them), then serves joint-level control and rendering over a pipe
(wire.py framing), speaking the same protocol as the RoboTwin server so the robotwin2 harness side is reused: the
harness owns IK, the kinematic twin and every public computation. Physics: the robot's own PD joint drives (gains set
by the official controller config) with the official passive-force compensation; arm joint targets come from the
harness, so SimplerEnv's delta-EE controller is bypassed. The task's own evaluate() runs once per OFFICIAL control
period of simulated time (513/3 or 500/5 substeps), so its per-step counters (consecutive grasp, lift) keep their
official timing. Host-only requests (object poses, evaluate results) are only used for scoring and audit.
"""
from __future__ import annotations

import math
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wire  # noqa: E402

import numpy as np  # noqa: E402

_DEBUG = bool(os.environ.get("SE_FORCE_DEBUG"))  # offline diagnosis only: per-contact force lines in the server log
ARM = "right"  # one controlled arm; the robotwin2 client protocol names it

# per robot family: arm joints (= twin robot0_joint1..), gripper joints, wrist-camera link, gripper links (contact
# force), finger joint value range (open, closed) and the closing-speed limit (joint units / s)
ROBOTS = {
    "google_robot": dict(
        arm=["joint_torso", "joint_shoulder", "joint_bicep", "joint_elbow", "joint_forearm", "joint_wrist", "joint_gripper"],
        gripper=["joint_finger_right", "joint_finger_left"], open_q=0.0, closed_q=1.31, rate=1.0,
        gripper_links=["link_gripper", "link_finger_right", "link_finger_tip_right", "link_finger_nail_right",
                       "link_finger_left", "link_finger_tip_left", "link_finger_nail_left"],
        finger_links=["link_finger_tip_right", "link_finger_tip_left"], tcp="link_gripper_tcp", hand="link_gripper",
        jaw_links=["link_finger_right", "link_finger_tip_right", "link_finger_nail_right", "link_finger_left",
                   "link_finger_tip_left", "link_finger_nail_left"], squeeze_internal=True),
    "widowx": dict(
        arm=["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"],
        gripper=["left_finger", "right_finger"], open_q=0.037, closed_q=0.015, rate=0.044,
        gripper_links=["gripper_link", "ee_arm_link", "gripper_prop_link", "gripper_bar_link", "fingers_link",
                       "left_finger_link", "right_finger_link", "ee_gripper_link"],
        finger_links=["left_finger_link", "right_finger_link"], tcp="ee_gripper_link", hand="ee_arm_link",
        jaw_links=["left_finger_link", "right_finger_link"]),
}


def _pose_matrix(pose) -> np.ndarray:
    return np.asarray(pose.to_transformation_matrix(), dtype=np.float64)


def _family(robot: str) -> str:
    return "google_robot" if robot.startswith("google_robot") else "widowx"


class Server:
    def __init__(self) -> None:
        self.env = None
        self.cams = {}
        self.time_s = 0.0

    # ------------------------------------------------------------------ setup
    def init(self, env_name: str, robot: str, scene_name: str, control_freq: int, sim_freq: int,
             max_episode_steps: int, rgb_overlay_path: str | None, robot_init_x: float, robot_init_y: float,
             robot_init_quat, obj_init_x: float | None = None, obj_init_y: float | None = None,
             obj_episode_id: int | None = None, additional_env_build_kwargs: dict | None = None,
             main_camera: str | None = None, cameras: dict | None = None, advance_subtask: bool = False, letterbox: int | None = None, arm_drive=None,
             settle_s: float = 0.0, **_ignored) -> dict:
        import gymnasium as gym  # noqa: F401
        import mani_skill2_real2sim.envs  # noqa: F401  (registers the envs)
        from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
        root = os.environ.get("SIMPLER_ROOT")
        if rgb_overlay_path and root and not os.path.isabs(rgb_overlay_path):
            rgb_overlay_path = os.path.join(root, rgb_overlay_path)
        kwargs = dict(obs_mode="rgbd", robot=robot, sim_freq=int(sim_freq),
                      control_mode=get_robot_control_mode(robot, "rt1" if robot.startswith("google") else "octo-base"),
                      control_freq=int(control_freq), max_episode_steps=int(max_episode_steps), scene_name=scene_name,
                      camera_cfgs={"add_segmentation": True}, rgb_overlay_path=rgb_overlay_path)
        self.env = build_maniskill2_env(env_name, **(additional_env_build_kwargs or {}), **kwargs)
        options = {"robot_init_options": {"init_xy": np.array([robot_init_x, robot_init_y]),
                                          "init_rot_quat": np.asarray(robot_init_quat, float)}}
        if obj_init_x is not None:
            options["obj_init_options"] = {"init_xy": np.array([obj_init_x, obj_init_y])}
        else:
            options["obj_init_options"] = {"episode_id": int(obj_episode_id)}
        obs, self.reset_info = self.env.reset(options=options)
        u = self.u = self.env.unwrapped
        self.instruction = u.get_language_instruction()
        if advance_subtask and hasattr(u, "advance_to_next_subtask"):
            # two-stage task (open drawer, then place): both official instructions at once, scored at the final stage
            u.advance_to_next_subtask()
            self.instruction = f"{self.instruction}, then {u.get_language_instruction()}"
        self.family = _family(robot)
        spec = ROBOTS[self.family]
        self.spec = spec
        self.robot = u.agent.robot
        self.scene = u._scene
        self.dt = float(self.scene.get_timestep())
        self.substeps_per_control = int(u._sim_steps_per_control)
        joints = {j.get_name(): j for j in self.robot.get_active_joints()}
        self.names = [j.get_name() for j in self.robot.get_active_joints()]
        self.arm_joints = [joints[n] for n in spec["arm"]]
        self.grip_joints = [joints[n] for n in spec["gripper"]]
        self.links = {l.get_name(): l for l in self.robot.get_links()}
        self.official_drive = [(j.stiffness, j.damping, j.force_limit) for j in self.arm_joints]
        if arm_drive:  # harness-side joint servo gains (the official delta-EE controller is bypassed anyway)
            for j, (_, _, flim) in zip(self.arm_joints, self.official_drive):
                j.set_drive_property(stiffness=float(arm_drive[0]), damping=float(arm_drive[1]), force_limit=flim)
        q = np.asarray(self.robot.get_qpos(), float)
        self.targets = q[[self.names.index(n) for n in spec["arm"]]].copy()
        # every other active joint (Google Robot head pan/tilt = the overhead camera) holds its reset position; the
        # official controller re-targets them each step, so without this the head sags to drive target 0
        self.other_joints = [(joints[n], float(q[i])) for i, n in enumerate(self.names) if n not in spec["arm"] + spec["gripper"]]
        g = q[[self.names.index(n) for n in spec["gripper"]]]
        self.grip_target = float(np.mean(g))
        self.grip_goal = self._open_fraction(self.grip_target)
        self.main_camera = main_camera
        self.letterbox = letterbox
        self._add_cameras(cameras or {})
        self._substep_count = 0
        self.max_episode_steps = int(max_episode_steps)
        self.horizon_eval = None
        self.eval_info = self._evaluate()
        self.ever_success = bool(self.eval_info.get("success", False))
        if settle_s > 0:
            self._substeps(int(settle_s / self.dt))
        self.time_s = 0.0
        limits = np.asarray(self.robot.get_qlimits(), float)
        return {"bases": {ARM: _pose_matrix(self.robot.get_pose())}, "timestep": self.dt, "state": self.read(),
                "joint_limits": {ARM: limits[[self.names.index(n) for n in spec["arm"]]]},
                "instruction": self.instruction, "reset_info": _plain(self.reset_info),
                "objects": self.objects(), "eval": _plain(self.eval_info), "family": self.family,
                "control_period_substeps": self.substeps_per_control,
                "finger_tip_gap_m": self.finger_gap()}

    def _open_fraction(self, q: float) -> float:
        o, c = self.spec["open_q"], self.spec["closed_q"]
        return float(np.clip((q - c) / (o - c), 0.0, 1.0))

    def finger_gap(self) -> float:
        a, b = (self.links[n].get_pose().p for n in self.spec["finger_links"])
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))

    def _add_cameras(self, cameras: dict) -> None:
        """Extra cameras (harness wrist view): mounted on a robot link at a fixed local pose (forward/left/up axes)."""
        import sapien.core as sapien
        for name, cfg in cameras.items():
            link = self.links[cfg["link"]]
            fwd = np.asarray(cfg["forward"], float); fwd /= np.linalg.norm(fwd)
            left = np.asarray(cfg["left"], float); left -= fwd * (left @ fwd); left /= np.linalg.norm(left)
            mat = np.eye(4); mat[:3, :3] = np.stack([fwd, left, np.cross(fwd, left)], axis=1)
            mat[:3, 3] = np.asarray(cfg.get("position", [0, 0, 0]), float)
            cam = self.scene.add_mounted_camera(name, link, sapien.Pose.from_transformation_matrix(mat),
                                                int(cfg["width"]), int(cfg["height"]),
                                                math.radians(float(cfg["fovy_deg"])), 0.01, 10.0)
            self.cams[name] = (cam, cfg)

    # ------------------------------------------------------------------ physics
    def _evaluate(self) -> dict:
        try:
            return dict(self.u.evaluate())
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": f"{type(exc).__name__}: {exc}"}

    def _substeps(self, n: int) -> np.ndarray:
        force = np.zeros(3)
        gripper_ids = {self.links[k].get_id() for k in self.spec["gripper_links"] if k in self.links}
        own = {l.get_id() for l in self.links.values()}
        finger_ids = {self.links[k].get_id() for k in self.spec["jaw_links"]}
        o, c = self.spec["open_q"], self.spec["closed_q"]
        for _ in range(n):
            self.robot.set_qf(self.robot.compute_passive_force(external=False))
            for j, q in zip(self.arm_joints, self.targets):
                j.set_drive_target(float(q))
            for j, q in self.other_joints:
                j.set_drive_target(q)
            goal = c + self.grip_goal * (o - c)
            step = self.spec["rate"] * self.dt
            self.grip_target += float(np.clip(goal - self.grip_target, -step, step))
            for j in self.grip_joints:
                j.set_drive_target(self.grip_target)
            self.scene.step()
            self.time_s += self.dt
            self._substep_count += 1
            # jaw closing: finger contact along the closing axis is the squeeze on the grasped object, not wrist load
            closing_axis = None
            if self.grip_goal < 0.5 and abs(goal - self.grip_target) > 1e-5:
                a, b = (np.asarray(self.links[k].get_pose().p, float) for k in self.spec["finger_links"])
                if np.linalg.norm(a - b) > 1e-6:
                    closing_axis = (a - b) / np.linalg.norm(a - b)
            for contact in self.scene.get_contacts():
                a, b = contact.actor0.get_id(), contact.actor1.get_id()
                ia, ib = a in gripper_ids, b in gripper_ids
                if ia == ib or (ia and b in own) or (ib and a in own):
                    continue
                impulse = sum((np.asarray(p.impulse, float) for p in contact.points), np.zeros(3))
                jaw = (a if ia else b) in finger_ids
                if jaw and self.spec.get("squeeze_internal") and (self.grip_goal < 0.5 or abs(goal - self.grip_target) > 1e-5):
                    other = contact.actor1 if ia else contact.actor0
                    if other.classname != "ActorStatic":
                        # Google Robot: the curved, revolute fingers squeeze a grasped object with ~250 N and press it
                        # along the approach axis (~80 N net); that is the grasp itself (also while the jaw re-opens),
                        # not load on the wrist
                        continue
                if closing_axis is not None and jaw:
                    impulse = impulse - (impulse @ closing_axis) * closing_axis
                if _DEBUG and np.linalg.norm(impulse) / self.dt > 5.0:
                    print("FORCEDBG", round(self.time_s, 3), contact.actor0.name, contact.actor1.name,
                          np.round(impulse / self.dt, 1).tolist(), "closing" if closing_axis is not None else "", flush=True)
                force += impulse if ia else -impulse  # SAPIEN 2: impulse acts on actor0
            if self._substep_count % self.substeps_per_control == 0:  # official control period elapsed
                self.u._elapsed_steps += 1
                self.eval_info = self._evaluate()
                self.ever_success = self.ever_success or bool(self.eval_info.get("success", False))
                if self.u._elapsed_steps == self.max_episode_steps:  # the official episode horizon ends here
                    self.horizon_eval = _plain(self.eval_info)
        return force / max(n * self.dt, 1e-9)

    def step(self, targets: dict, n: int) -> dict:
        cmd = targets.get(ARM) or {}
        if cmd.get("q") is not None:
            self.targets = np.asarray(cmd["q"], float)
        if cmd.get("open") is not None:
            self.grip_goal = float(np.clip(cmd["open"], 0.0, 1.0))
        if int(n) > 0:
            self.last_force = self._substeps(int(n))
        return self.read(getattr(self, "last_force", None))

    def teleport(self, arm: str, q) -> None:
        qpos = np.asarray(self.robot.get_qpos(), float)
        for n, v in zip(self.spec["arm"], q):
            qpos[self.names.index(n)] = float(v)
        self.robot.set_qpos(qpos)
        self.robot.set_qvel(np.zeros_like(qpos))
        self.targets = np.asarray(q, float).copy()

    def reset_clock(self) -> None:
        self.time_s = 0.0

    def read(self, force=None) -> dict:
        qpos, qvel = np.asarray(self.robot.get_qpos(), float), np.asarray(self.robot.get_qvel(), float)
        idx = [self.names.index(n) for n in self.spec["arm"]]
        g = float(np.mean(qpos[[self.names.index(n) for n in self.spec["gripper"]]]))
        frac = self._open_fraction(g)
        return {"time_s": self.time_s,
                ARM: {"q": qpos[idx], "dq": qvel[idx], "open_fraction": frac, "fingers": np.array([frac, frac]),
                      "force_world": None if force is None else np.asarray(force, float),
                      "hand": _pose_matrix(self.links[self.spec["tcp"]].get_pose())}}

    def exec_debug(self, code: str):
        """Offline probes only (tools); never reachable from the agent API."""
        scope = {"self": self, "np": np}
        exec(code, scope)
        return scope.get("result")

    # ------------------------------------------------------------------ rendering
    def frames(self, names) -> dict:
        out = {}
        want_main = self.main_camera in names
        if want_main:  # the official camera through SimplerEnv's own observation path (incl. the real-image overlay)
            obs = self._wrapped_obs()
            img = obs["image"][self.main_camera]
            par = obs["camera_param"][self.main_camera]
            depth = np.asarray(img["depth"], np.float32)
            depth = depth[..., 0] if depth.ndim == 3 else depth
            rgb = np.asarray(img["rgb"])
            rgb = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8) if rgb.dtype != np.uint8 else rgb
            K = np.asarray(par["intrinsic_cv"], float)
            if self.letterbox:
                rgb, depth, K = _letterbox(rgb, depth, K, int(self.letterbox))
            out[self.main_camera] = {"rgb": rgb, "depth": depth, "K": K,
                                     "optical_world": np.linalg.inv(np.asarray(par["extrinsic_cv"], float))}
        extra = [n for n in names if n in self.cams]
        if extra:
            self.scene.update_render()
            for name in extra:
                cam, cfg = self.cams[name]
                cam.take_picture()
                rgba = np.asarray(cam.get_float_texture("Color"))
                rgb = (np.clip(rgba[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)
                pos = np.asarray(cam.get_float_texture("Position"))
                depth = (-pos[..., 2]).astype(np.float32)
                depth[pos[..., 3] >= 1.0] = 0.0
                gl = np.asarray(cam.get_model_matrix(), float)
                out[name] = {"rgb": rgb, "depth": depth, "K": np.asarray(cam.get_intrinsic_matrix(), float),
                             "optical_world": gl @ np.diag([1.0, -1.0, -1.0, 1.0])}
        return out

    def _wrapped_obs(self):
        """env.get_obs() passed through the env's own gym observation wrappers (rgbd conversion), innermost first."""
        import gymnasium as gym
        chain, e = [], self.env
        while isinstance(e, gym.Wrapper):
            chain.append(e)
            e = e.env
        obs = self.u.get_obs()
        for w in reversed(chain):
            if isinstance(w, gym.ObservationWrapper):
                obs = w.observation(obs)
        return obs

    # ------------------------------------------------------------------ host-only
    def objects(self) -> dict:
        out = {}
        for actor in self.scene.get_all_actors():
            if actor.name in ("arena", "ground", ""):
                continue
            pose = actor.get_pose()
            out[actor.name] = {"p": np.asarray(pose.p, float), "q_wxyz": np.asarray(pose.q, float), "model": actor.name}
        for art in self.scene.get_all_articulations():
            if art is self.robot:
                continue
            pose = art.get_pose()
            out[art.name or "articulation"] = {"p": np.asarray(pose.p, float), "q_wxyz": np.asarray(pose.q, float),
                                               "model": art.name, "qpos": np.asarray(art.get_qpos(), float)}
        return out

    def success(self, **_) -> bool:
        """The task's own verdict at the last official control period (SimplerEnv reports the final step's)."""
        return bool(self.eval_info.get("success", False))

    def evaluation(self) -> dict:
        return {"eval": _plain(self.eval_info), "ever_success": self.ever_success,
                "official_steps": int(self.u._elapsed_steps), "max_episode_steps": self.max_episode_steps,
                "horizon_eval": self.horizon_eval}


def _letterbox(rgb, depth, K, size: int):
    """Scale the official image to `size` px wide (aspect kept) and pad top/bottom to a square (black RGB, depth 0 =
    invalid); intrinsics follow exactly. Nothing of the official view is cropped."""
    import cv2
    h, w = rgb.shape[:2]
    s = size / max(h, w)
    nw, nh = int(round(w * s)), int(round(h * s))
    rgb_s = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    depth_s = cv2.resize(depth, (nw, nh), interpolation=cv2.INTER_NEAREST)
    top, left = (size - nh) // 2, (size - nw) // 2
    rgb_o = np.zeros((size, size, 3), np.uint8)
    depth_o = np.zeros((size, size), np.float32)
    rgb_o[top:top + nh, left:left + nw] = rgb_s
    depth_o[top:top + nh, left:left + nw] = depth_s
    K2 = K.copy()
    K2[0, :] *= nw / w
    K2[1, :] *= nh / h
    K2[0, 2] += left
    K2[1, 2] += top
    return rgb_o, depth_o, K2


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "p") and hasattr(value, "q"):
        return {"p": np.asarray(value.p).tolist(), "q": np.asarray(value.q).tolist()}
    return value if isinstance(value, (int, float, str, type(None))) else str(value)


def main() -> None:
    rfd, wfd = int(sys.argv[1]), int(sys.argv[2])
    reader, writer = os.fdopen(rfd, "rb"), os.fdopen(wfd, "wb")
    server = Server()
    while True:
        try:
            message = wire.recv(reader)
        except EOFError:
            break
        cmd = message.pop("cmd")
        t0 = time.time()
        try:
            if cmd == "close":
                wire.send(writer, {"ok": True})
                break
            result = getattr(server, cmd)(**message)
            wire.send(writer, {"ok": True, "result": result, "wall_s": time.time() - t0})
        except Exception as exc:  # noqa: BLE001
            wire.send(writer, {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})


if __name__ == "__main__":
    main()

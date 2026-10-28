"""robosuite-shaped facade over an XR1 driver and its kinematic twin.

LiberoActionBackend talks to `env.step(action)` with robosuite OSC_POSE delta
actions and reads forward kinematics from `env.sim`. Here `step` turns the delta
into an absolute grip-site goal in robot_base, hands it to a driver (MuJoCo
physics in sim mode, the robot bridge in real mode) for one control tick, then
copies the measured joints into the twin so every pose the harness reads is
the twin's FK of the measured configuration.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .mjsim import LINK_TO_OPTICAL, SimView, site_pose

ARM_JOINTS = tuple(f"robot0_joint{k}" for k in range(1, 8))
FINGER_JOINTS = ("gripper0_finger_joint1", "gripper0_finger_joint2")
GRIP_SITE = "gripper0_grip_site"
# robosuite Panda reports finger qpos in [0, 0.04]; the backend reads qpos[0] / 0.04 as the open fraction.
LIBERO_FINGER_OPEN_QPOS = 0.04


@dataclass
class DriverState:
    joints: np.ndarray                 # 7 measured arm joints (rad)
    joint_velocities: np.ndarray       # 7 (rad/s)
    gripper_open_fraction: float       # 0 closed .. 1 open (measured)
    wrist_force_world: np.ndarray | None  # 3, robot_base frame, None if unavailable
    time_s: float


@dataclass
class CameraFrame:
    rgb: np.ndarray                    # HxWx3 uint8, top-left origin
    depth_m: np.ndarray | None         # HxW float32 metres (optical z), NaN/0 = invalid
    intrinsics: np.ndarray             # 3x3 for this image size
    optical_in_robot_base: np.ndarray | None = None  # 4x4; None -> twin FK of the camera link


class XR1Driver(Protocol):
    control_frequency: int

    def reset(self) -> DriverState: ...

    def read(self) -> DriverState: ...

    def command(self, grip_site_goal: np.ndarray | None, gripper_action: float) -> DriverState:
        """Send one tick: absolute grip-site goal (4x4, robot_base) and gripper action in [-1, 1] (+1 closes)."""

    def hold(self) -> None:
        """Replace the outstanding arm goal by the measured pose (no motion request)."""

    def frames(self, camera_names: Sequence[str]) -> Mapping[str, CameraFrame]: ...

    def close(self) -> None: ...


class _Controller:
    name = "OSC_POSE"
    control_dim = 6

    def __init__(self, env: "XR1FacadeEnv", output_max: Sequence[float]) -> None:
        self._env = env
        self.output_max = np.asarray(output_max, dtype=np.float64)
        self.kp = np.full(6, 150.0)
        self.kd = 2 * np.sqrt(self.kp)
        self._goal_pos = None

    # the residual contact detector reads kp*(goal_pos - ee_pos): the last commanded grip-site position, or the measured
    # one before any command (before this fix it was None -> NaN residuals, and the calibrated detector could not run)
    @property
    def goal_pos(self):
        return self.ee_pos if self._goal_pos is None else self._goal_pos

    @goal_pos.setter
    def goal_pos(self, value):
        self._goal_pos = None if value is None else np.asarray(value, dtype=np.float64).reshape(3)

    @property
    def ee_pos(self):
        return self._env.grip_site_pose()[:3, 3]

    @property
    def ee_pos_vel(self):
        return np.zeros(3)

    def update(self, force=False):
        return None

    def reset_goal(self):
        self.goal_pos = None
        self._env._driver.hold()
        if getattr(self._env._driver, "settle_on_hold", False):  # robotwin2 R5 F2: fresh force reading after a stop
            self._env.state = self._env._driver.settle()
            self._env._sync(self._env.state)
        elif getattr(self._env._driver, "reread_after_hold", False):
            # the driver may advance time while holding (RoboDojo: lets a released contact relax): re-read it
            self._env.state = self._env._driver.read()
            self._env._sync(self._env.state)


class _Gripper:
    def __init__(self) -> None:
        self.joints = list(FINGER_JOINTS)
        self.important_sites = {"grip_site": GRIP_SITE}
        self.important_sensors = {}


class _RobotModel:
    def __init__(self, xml: str) -> None:
        self.joints = list(ARM_JOINTS)
        self._xml = xml

    def get_xml(self) -> str:
        return self._xml


class _Robot:
    def __init__(self, env: "XR1FacadeEnv", xml: str, output_max) -> None:
        self._env = env
        self.controller = _Controller(env, output_max)
        self.gripper = _Gripper()
        self.robot_model = _RobotModel(xml)
        self.action_dim = 7
        model = env.twin_model
        self._ref_joint_vel_indexes = [int(model.jnt_dofadr[model.joint(n).id]) for n in ARM_JOINTS]

    @property
    def ee_force(self):
        force = self._env.state.wrist_force_world
        return None if force is None else np.asarray(force, dtype=np.float64)


class XR1FacadeEnv:
    """The `env` object handed to XR1ActionBackend."""

    def __init__(self, driver: XR1Driver, *, robot_xml_path: Path, camera_names: Sequence[str],
                 camera_links: Mapping[str, str], depth_cameras: Sequence[str] | None = None, image_size: tuple[int, int] = (512, 512),
                 osc_output_max: Sequence[float] = (0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
                 parked_joints: Mapping[str, Sequence[float]] | None = None,
                 max_goal_step_m: float = 0.01, max_goal_step_rad: float = 0.08,
                 workspace_box: Mapping[str, Sequence[float]] | None = None,
                 depth_estimators: Mapping[str, Any] | None = None, sensor_noise: Any = None) -> None:
        self._driver = driver
        # optional sensor-noise model (.next_rng() / .rgb() / .depth()): applied to every frame the agent and the tools see
        self._sensor_noise = sensor_noise
        # cameras without a depth sensor: name -> MonoDepthSource-style callable (RGB + calibration only)
        self._depth_estimators = dict(depth_estimators or {})
        xml = Path(robot_xml_path).read_text()
        self.twin_model = mujoco.MjModel.from_xml_string(xml)
        self.twin_data = mujoco.MjData(self.twin_model)
        self.sim = SimView(self.twin_model, self.twin_data)
        self._camera_names = tuple(camera_names)
        self._depth_cameras = tuple(depth_cameras if depth_cameras is not None else camera_names)
        self._frames: dict[str, CameraFrame] = {}
        self._camera_links = dict(camera_links)
        self._image_size = tuple(image_size)
        self._max_goal_step_m = float(max_goal_step_m)
        self._max_goal_step_rad = float(max_goal_step_rad)
        self._workspace = ({k: np.asarray(v, float) for k, v in workspace_box.items()}
                           if workspace_box else None)
        m = self.twin_model
        self._arm_qadr = np.asarray([m.jnt_qposadr[m.joint(n).id] for n in ARM_JOINTS])
        self._arm_vadr = np.asarray([m.jnt_dofadr[m.joint(n).id] for n in ARM_JOINTS])
        self._finger_qadr = np.asarray([m.jnt_qposadr[m.joint(n).id] for n in FINGER_JOINTS])
        self._finger_open = np.asarray([m.jnt_range[m.joint(n).id][1] for n in FINGER_JOINTS])
        self._grip_site = m.site(GRIP_SITE).id
        self._base_body = m.body("robot0_base").id
        for group, values in (parked_joints or {}).items():
            names = ([f"left_arm_{k}_joint" for k in range(1, 8)] if group == "left_arm"
                     else ["head_yaw_joint", "head_pitch_joint"] if group == "head" else [])
            for name, value in zip(names, values):
                index = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
                if index >= 0:  # twins built by tools/build_twin.py bake these joints
                    self.twin_data.qpos[m.jnt_qposadr[index]] = float(value)
        self.robots = [_Robot(self, xml, osc_output_max)]
        self._gripper_action = -1.0
        self._gripper_action_bounds: tuple[float, float] | None = None
        self.state = driver.reset()
        self._sync(self.state)
        self.goal_clipped_steps = 0

    # ----------------------------------------------------------------- twin
    def _sync(self, state: DriverState) -> None:
        d = self.twin_data
        d.qpos[self._arm_qadr] = state.joints
        d.qvel[:] = 0
        d.qvel[self._arm_vadr] = state.joint_velocities
        d.qpos[self._finger_qadr] = self._finger_open * float(np.clip(state.gripper_open_fraction, 0, 1))
        mujoco.mj_forward(self.twin_model, d)

    def grip_site_pose(self) -> np.ndarray:
        """Grip site pose in robot_base (twin FK of the measured joints)."""
        return site_pose(self.twin_model, self.twin_data, self._grip_site)

    def camera_link_pose(self, camera_name: str) -> np.ndarray:
        body = self.twin_model.body(self._camera_links[camera_name]).id
        pose = np.eye(4)
        pose[:3, :3] = self.twin_data.xmat[body].reshape(3, 3)
        pose[:3, 3] = self.twin_data.xpos[body]
        return pose

    # ------------------------------------------------------------ robosuite API
    def _observation(self) -> "LazyObservation":
        state = self.state
        obs = LazyObservation(self._fetch_frames, self._camera_names, self._depth_cameras)
        obs.update({
            "robot0_joint_pos": np.asarray(state.joints, dtype=np.float64),
            "robot0_joint_vel": np.asarray(state.joint_velocities, dtype=np.float64),
            "robot0_gripper_qpos": np.array([1, -1], dtype=np.float64)
            * LIBERO_FINGER_OPEN_QPOS * float(np.clip(state.gripper_open_fraction, 0, 1)),
        })
        return obs

    def _fetch_frames(self) -> dict[str, Any]:
        self._frames = dict(self._driver.frames(self._camera_names))
        noise_rng = self._sensor_noise.next_rng() if self._sensor_noise is not None else None
        if noise_rng is not None:  # RGB first, so mono depth is estimated from the noisy image like on a real sensor
            for name, frame in list(self._frames.items()):
                self._frames[name] = CameraFrame(rgb=self._sensor_noise.rgb(frame.rgb, noise_rng), depth_m=frame.depth_m,
                                                 intrinsics=frame.intrinsics, optical_in_robot_base=frame.optical_in_robot_base)
        for name, estimate in self._depth_estimators.items():
            frame = self._frames.get(name)
            if frame is None or frame.depth_m is not None:
                continue
            extrinsics = (frame.optical_in_robot_base if frame.optical_in_robot_base is not None
                          else self.camera_link_pose(name) @ LINK_TO_OPTICAL)
            depth = estimate(camera=name, rgb=frame.rgb, raw_sim_depth=None,
                             intrinsics_3x3=frame.intrinsics, camera_to_base_4x4=extrinsics)
            self._frames[name] = CameraFrame(rgb=frame.rgb, depth_m=np.asarray(depth, np.float32),
                                             intrinsics=frame.intrinsics, optical_in_robot_base=frame.optical_in_robot_base)
        if noise_rng is not None:
            for name, frame in list(self._frames.items()):
                if frame.depth_m is not None:
                    self._frames[name] = CameraFrame(rgb=frame.rgb, depth_m=self._sensor_noise.depth(frame.depth_m, noise_rng),
                                                     intrinsics=frame.intrinsics, optical_in_robot_base=frame.optical_in_robot_base)
        images = {}
        for name, frame in self._frames.items():
            # robosuite renders bottom-up; the backend flips rows back.
            images[f"{name}_image"] = np.ascontiguousarray(frame.rgb[::-1])
            if frame.depth_m is not None:
                images[f"{name}_depth"] = np.ascontiguousarray(frame.depth_m[::-1])
        return images

    def current_observation(self) -> dict[str, Any]:
        return self._observation()

    def frame(self, camera_name: str) -> CameraFrame:
        """Frame behind the most recently read image of this camera."""
        return self._frames[camera_name]

    def _goal_from_action(self, values: np.ndarray, output_max: np.ndarray) -> np.ndarray:
        current = self.grip_site_pose()
        delta = np.clip(values[:6], -1, 1) * output_max
        dp, dr = delta[:3], delta[3:6]
        # per-tick goal lead limits: the real controller gets a dense, bounded reference
        norm = np.linalg.norm(dp)
        if norm > self._max_goal_step_m:
            dp = dp * (self._max_goal_step_m / norm)
            self.goal_clipped_steps += 1
        angle = np.linalg.norm(dr)
        if angle > self._max_goal_step_rad:
            dr = dr * (self._max_goal_step_rad / angle)
        goal = current.copy()
        goal[:3, 3] = current[:3, 3] + dp
        goal[:3, :3] = Rotation.from_rotvec(dr).as_matrix() @ current[:3, :3]
        if self._workspace is not None:
            goal[:3, 3] = np.clip(goal[:3, 3], self._workspace["min"], self._workspace["max"])
        return goal

    # robosuite's native per-substep hooks; the force guard patches them to sample
    # the wrist force (here once per control tick) and to zero a triggered action.
    def _pre_action(self, values, policy_step=False):
        return values

    def _update_observables(self, force=False):
        return None

    @property
    def model_timestep(self) -> float:
        return 1.0 / float(self._driver.control_frequency)

    def step(self, action: Sequence[float]):
        values = np.asarray(action, dtype=np.float64).reshape(7)
        values = np.asarray(self._pre_action(values, policy_step=True), dtype=np.float64).reshape(7)
        gripper_action = float(np.clip(values[-1], -1, 1))
        if self._gripper_action_bounds is not None:
            gripper_action = float(np.clip(gripper_action, *self._gripper_action_bounds))
        self._gripper_action = gripper_action
        goal = self._goal_from_action(values, self.robots[0].controller.output_max)
        if goal is not None:
            self.robots[0].controller.goal_pos = np.asarray(goal, dtype=np.float64)[:3, 3]
        self.state = self._driver.command(goal, gripper_action)
        self._sync(self.state)
        self._update_observables()
        return self._observation(), 0.0, False, {}

    @contextlib.contextmanager
    def bounded_gripper_actions(self, low: float, high: float):
        previous = self._gripper_action_bounds
        self._gripper_action_bounds = (float(low), float(high))
        try:
            yield
        finally:
            self._gripper_action_bounds = previous

    @property
    def gripper_action(self) -> float:
        return self._gripper_action

    def hidden_object_audit(self):
        """Host-only object poses (sim drivers only); None -> the env server's LIBERO audit."""
        factory = getattr(self._driver, "hidden_object_audit", None)
        return factory() if callable(factory) else None

    def close(self) -> None:
        self._driver.close()


class LazyObservation(dict):
    """Robot keys eagerly; camera keys fetched once, on first access, for this tick."""

    def __init__(self, fetch, camera_names, depth_cameras) -> None:
        super().__init__()
        self._fetch = fetch
        self._lazy = {f"{n}_image" for n in camera_names} | {f"{n}_depth" for n in depth_cameras}
        self._loaded = False

    def _load(self) -> None:
        if not self._loaded:
            self._loaded = True
            super().update(self._fetch())

    def __getitem__(self, key):
        if key in self._lazy:
            self._load()
        return super().__getitem__(key)

    def __contains__(self, key) -> bool:
        if key in self._lazy:
            self._load()
        return super().__contains__(key)

    def get(self, key, default=None):
        return self[key] if key in self else default

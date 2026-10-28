"""CaP-Gym single-arm Robosuite tasks as a LIBERO-ControlEnv-compatible environment.

Faithful to cap-x 53e9966 (capx/envs/simulators/robosuite_*.py) and its robosuite fork uynitsuj/robosuite@9729273:
  * Lift  (robosuite_cube_lift.py): stock Lift, one red cube, success = cube z > table + 0.04.
  * Stack (robosuite_cubes.py): stock Stack (red 0.02 on green 0.025), sampler x+-0.18 / y+-0.12, success = r_stack > 0.
  * Restack (robosuite_cubes_restack.py): Stack with both cubes 0.02; StackedObjectRandomSampler puts green ON red
    (x -0.12..0.16, y +-0.12); goal red on green; success = r_stack > 0 and not (both cubes > 0.04 m above the robot base).
  * Wipe (robosuite_spill_wipe.py): fork Wipe (WipingGripper sponge, 100 markers, arena shifted x -0.1, marker wiping
    not gated by collisions, early termination on arm contact / joint limit / done); success = all markers wiped.
  * Nut (robosuite_nut_assembly.py, Visual): NutAssemblySquare, arm parked at [0,-1.585,0,-2.645,0,1,0.785], 100 held
    control steps; head camera = birdview; success = square nut on its peg (robosuite on_peg, gripper away).
Common: success is the env's own ``_check_success()`` (CaP-Gym ``task_completed()``) on the final state. CaP-Gym's
robosuite horizon (control steps; Lift/Stack/Restack 1500, Nut 1000 incl. its 100 settle steps, Wipe 4000) is enforced
from the end of the harness's force calibration. Default (record-only): at the horizon step, or at the first Wipe early
termination, the success predicate is snapshotted once ("capped" = what CaP-Gym would score, since robosuite freezes the
episode there) and the episode continues (uncapped = final state). CAPGYM_ENFORCE_HORIZON=1 instead makes the next step
raise "executing action in terminated episode" like robosuite, which ends physical interaction.
Camera: the harness hard-codes ``agentview``; it is moved onto CaP-Gym's camera pose (robotview, or birdview for Nut).
Wrist flip (qpos[6] -= pi): off by default (CAPGYM_WRIST_FLIP=1): the parallel gripper is symmetric under it and it left joint 7 near its limit under OSC.
Remaining differences: robosuite 1.4 (not 1.5), the harness's OSC_POSE motion API instead of CaP-X's joint controller.
(An earlier revision without horizon recording was used only in a pilot with an earlier checkpoint.)
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CapGymTask:
    key: str
    robosuite_env: str
    language: str
    capx_config: str
    capx_low_level: str
    horizon: int  # CaP-Gym robosuite horizon in control steps
    head_camera: str = "robot0_robotview"
    settle: str = "sim"  # "sim": 50 x sim.step (cube tasks / wipe); "held": 100 x env.step holding pose (nut)
    settle_steps: int = 50
    init_qpos: tuple | None = None
    wrist_flip_default: bool = False
    early_termination: bool = False
    fake_gripper: bool = False
    calibration_profile: str | None = None


CAPGYM_TASKS: dict[str, CapGymTask] = {
    "capgym_lift": CapGymTask(
        "capgym_lift", "Lift", "pick up the red cube and lift it",
        "env_configs/cube_lifting/franka_robosuite_cube_lifting.yaml", "franka_robosuite_cube_lift_low_level", 1500),
    "capgym_stack": CapGymTask(
        "capgym_stack", "Stack", "pick up the red cube and stack it on top of the green cube, then release it",
        "env_configs/cube_stack/franka_robosuite_cube_stack.yaml", "franka_robosuite_cubes_low_level", 1500),
    "capgym_restack": CapGymTask(
        "capgym_restack", "Restack", "gently place the red cube on top of the green cube and then open the gripper",
        "env_configs/cube_restack/franka_robosuite_cube_restack.yaml", "franka_robosuite_cubes_restack_low_level", 1500),
    "capgym_wipe": CapGymTask(
        "capgym_wipe", "Wipe",
        "wipe up the brown spill on the table; a sponge is already attached to the end-effector (there is no gripper)",
        "env_configs/spill_wipe/franka_robosuite_spill_wipe.yaml", "franka_robosuite_spill_wipe_low_level", 4000,
        early_termination=True, fake_gripper=True),  # commanded position = gripper0_eef = sponge face centre (measured)
    "capgym_nut": CapGymTask(
        "capgym_nut", "NutAssemblySquare",
        "grasp the brown square nut by its handle and insert it onto the brown square peg",
        "env_configs/nut_assembly/franka_robosuite_nut_assembly.yaml", "franka_robosuite_nut_assembly_low_level_visual", 1000,
        head_camera="birdview", settle="held", settle_steps=100, init_qpos=(0.0, -1.585, 0.0, -2.645, 0.0, 1.0, 0.785)),
}

# robosuite fork (uynitsuj/robosuite@9729273) Wipe defaults that differ from stock 1.4
FORK_WIPE_CONFIG = {
    "arm_limit_collision_penalty": 0.0, "wipe_contact_reward": 0.001, "unit_wiped_reward": 50.0, "ee_accel_penalty": 0,
    "excess_force_penalty_mul": 0.0, "distance_multiplier": 5.0, "distance_th_multiplier": 5.0,
    "table_full_size": [0.5, 0.8, 0.05], "table_offset": [0.15, 0, 0.9], "table_friction": [0.03, 0.005, 0.0001],
    "table_friction_std": 0, "table_height": 0.0, "table_height_std": 0.0, "line_width": 0.04, "two_clusters": False,
    "coverage_factor": 0.6, "num_markers": 100, "contact_threshold": 0.5, "pressure_threshold": 0.3,
    "pressure_threshold_max": 600.0, "print_results": False, "get_info": False, "use_robot_obs": True,
    "use_contact_obs": True, "early_terminations": True, "use_condensed_obj_obs": True,
}


def _wipe_class():
    from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
    from robosuite.environments.manipulation.wipe import Wipe
    from robosuite.models.arenas import WipeArena
    from robosuite.models.tasks import ManipulationTask

    class CapGymWipe(Wipe):
        """Fork semantics: arena shifted x -0.1; markers wiped whenever the sponge plane covers them (no collision gate)."""

        def __init__(self, **kwargs):
            kwargs.setdefault("task_config", dict(FORK_WIPE_CONFIG))
            super().__init__(**kwargs)

        def _load_model(self):
            SingleArmEnv._load_model(self)
            xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
            self.robots[0].robot_model.set_base_xpos(xpos)
            self.robot_contact_geoms = self.robots[0].robot_model.contact_geoms
            wipe_offset = np.array(self.table_offset, dtype=float).copy()
            wipe_offset[0] -= 0.1
            arena = WipeArena(table_full_size=self.table_full_size, table_friction=self.table_friction,
                              table_offset=wipe_offset, table_friction_std=self.table_friction_std,
                              coverage_factor=self.coverage_factor, num_markers=self.num_markers,
                              line_width=self.line_width, two_clusters=self.two_clusters)
            arena.set_origin([0, 0, 0])
            self.model = ManipulationTask(mujoco_arena=arena, mujoco_robots=[r.robot_model for r in self.robots])

        def _active_markers(self):
            c_geoms = self.robots[0].gripper.important_geoms["corners"]
            c1, c2, c3, c4 = [np.array(self.sim.data.geom_xpos[self.sim.model.geom_name2id(g)]) for g in c_geoms[:4]]
            v1 = (c1 - c2) / np.linalg.norm(c1 - c2)
            v2 = (c4 - c2) / np.linalg.norm(c4 - c2)
            t = [np.array([np.dot(c - c2, v1), np.dot(c - c2, v2)]) for c in (c1, c2, c3, c4)]
            pp = [t[0], t[1], t[3], t[2]]
            n = np.cross(v1, v2)
            n /= np.linalg.norm(n)

            def is_left(p0, p1, p2):
                return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])

            def in_rect(x, y, z, w, p):
                return is_left(x, y, p) < 0 and is_left(y, z, p) < 0 and is_left(z, w, p) < 0 and is_left(w, x, p) < 0

            active = []
            if self.sim.data.ncon != 0:
                for marker in self.model.mujoco_arena.markers:
                    marker_pos = np.array(self.sim.data.body_xpos[self.sim.model.body_name2id(marker.root_body)])
                    dist = np.dot(marker_pos - c2, n)
                    projected = marker_pos - dist * n
                    if 0.0 < dist < 0.02:
                        p2 = np.array([np.dot(projected - c2, v1), np.dot(projected - c2, v2)])
                        if in_rect(pp[0], pp[1], pp[2], pp[3], p2):
                            active.append(marker)
            return active

        def reward(self, action=None):
            for marker in self._active_markers():
                if marker in self.wiped_markers:
                    continue
                gid = self.sim.model.geom_name2id(marker.visual_geoms[0])
                self.sim.model.geom_rgba[gid][3] = 0
                self.wiped_markers.append(marker)
            return len(self.wiped_markers) / self.num_markers

    return CapGymWipe


def _restack_sampler_class():
    from robosuite.utils import RandomizationError
    from robosuite.utils.placement_samplers import ObjectPositionSampler
    from robosuite.utils.transform_utils import quat_multiply

    class StackedObjectRandomSampler(ObjectPositionSampler):
        """cap-x StackedObjectRandomSampler (second object directly on top of the first), global numpy RNG (robosuite 1.4)."""

        def __init__(self, name, mujoco_objects=None, x_range=(0, 0), y_range=(0, 0), rotation=None,
                     ensure_object_boundary_in_range=True, ensure_valid_placement=True, reference_pos=(0, 0, 0), z_offset=0.0):
            self.x_range, self.y_range, self.rotation = x_range, y_range, rotation
            super().__init__(name=name, mujoco_objects=mujoco_objects,
                             ensure_object_boundary_in_range=ensure_object_boundary_in_range,
                             ensure_valid_placement=ensure_valid_placement, reference_pos=reference_pos, z_offset=z_offset)

        def _sample_xy(self, rng, r):
            lo, hi = rng
            if self.ensure_object_boundary_in_range:
                lo, hi = lo + r, hi - r
            return np.random.uniform(high=hi, low=lo)

        def _sample_quat(self):
            angle = np.random.uniform(high=2 * np.pi, low=0) if self.rotation is None else self.rotation
            return np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])

        def sample(self, fixtures=None, reference=None, on_top=True):
            placed = {} if fixtures is None else dict(fixtures)
            base = np.array(self.reference_pos)

            def valid(x, y, z, hr, bo):
                if not self.ensure_valid_placement:
                    return True
                for (ox, oy, oz), _, other in placed.values():
                    if np.linalg.norm((x - ox, y - oy)) <= other.horizontal_radius + hr and z - oz <= other.top_offset[-1] - bo[-1]:
                        return False
                return True

            first, second = list(self.mujoco_objects)[:2]
            for _ in range(5000):
                x = self._sample_xy(self.x_range, first.horizontal_radius) + base[0]
                y = self._sample_xy(self.y_range, first.horizontal_radius) + base[1]
                z = self.z_offset + base[2] - (first.bottom_offset[-1] if on_top else 0)
                if not valid(x, y, z, first.horizontal_radius, first.bottom_offset):
                    continue
                q1 = self._sample_quat()
                if hasattr(first, "init_quat"):
                    q1 = quat_multiply(q1, first.init_quat)
                placed[first.name] = ((x, y, z), q1, first)
                gap = max(float(self.z_offset), 0.01)
                x2, y2 = x - 0.01, y
                z2 = z + first.top_offset[-1] - second.bottom_offset[-1] + gap
                if not valid(x2, y2, z2, second.horizontal_radius, second.bottom_offset):
                    placed.pop(first.name, None)
                    continue
                q2 = self._sample_quat()
                if hasattr(second, "init_quat"):
                    q2 = quat_multiply(q2, second.init_quat)
                placed[second.name] = ((x2, y2, z2), q2, second)
                return placed
            raise RandomizationError("Cannot place all objects ):")

    return StackedObjectRandomSampler


def _restack_class():
    import robosuite.environments.manipulation.stack as stack_mod

    class CapGymRestackStack(stack_mod.Stack):
        """robosuite Stack with cubeB 0.02 (the fork's cube_B_length=0.02 that cap-x restack passes)."""

        def _load_model(self):
            original = stack_mod.BoxObject

            def box(**kwargs):
                if kwargs.get("name") == "cubeB":
                    kwargs["size_min"] = [0.02, 0.02, 0.02]
                    kwargs["size_max"] = [0.02, 0.02, 0.02]
                return original(**kwargs)

            stack_mod.BoxObject = box
            try:
                super()._load_model()
            finally:
                stack_mod.BoxObject = original

    return CapGymRestackStack


class _RobotProxy:
    """Presents a 1-DoF gripper action slot for a gripperless robot (WipingGripper): action_dim + 1."""

    def __init__(self, robot):
        object.__setattr__(self, "_robot", robot)

    @property
    def action_dim(self):
        return int(self._robot.action_dim) + 1

    def __getattr__(self, name):
        return getattr(self._robot, name)

    def __setattr__(self, name, value):
        setattr(self._robot, name, value)


class CapGymRobosuiteEnv:
    """The subset of libero.libero.envs.env_wrapper.ControlEnv that the harness adapters use."""

    def __init__(self, task: str, *, controller: str = "OSC_POSE", horizon: int = 45000, camera_size: int = 512,
                 control_freq: int = 20) -> None:
        import robosuite as suite
        from robosuite.utils.placement_samplers import UniformRandomSampler

        if task not in CAPGYM_TASKS:
            raise ValueError(f"unknown CaP-Gym task {task!r}; expected one of {sorted(CAPGYM_TASKS)}")
        self.task = CAPGYM_TASKS[task]
        self.language_instruction = self.task.language
        kwargs = dict(
            robots=["Panda"],
            controller_configs=suite.load_controller_config(default_controller=controller),
            has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
            camera_names=["agentview", "robot0_eye_in_hand"], camera_heights=camera_size, camera_widths=camera_size,
            camera_depths=True, control_freq=control_freq, horizon=horizon, ignore_done=True,
            hard_reset=False,  # one MjModel: the camera re-pose and object ids survive resets
            reward_shaping=True,
        )
        name = self.task.robosuite_env
        if name == "Wipe":
            self.env = _wipe_class()(**kwargs)
        elif name == "Restack":
            self.env = _restack_class()(**kwargs)
        else:
            self.env = suite.make(name, **kwargs)
        env = self.env
        if name == "Stack":
            env.placement_initializer = UniformRandomSampler(
                name="ObjectSampler", mujoco_objects=[env.cubeA, env.cubeB], x_range=[-0.18, 0.18], y_range=[-0.12, 0.12],
                rotation=None, ensure_object_boundary_in_range=False, ensure_valid_placement=True,
                reference_pos=env.table_offset, z_offset=0.01)
        elif name == "Restack":
            env.placement_initializer = _restack_sampler_class()(
                name="ObjectSampler", mujoco_objects=[env.cubeA, env.cubeB], x_range=[-0.12, 0.16], y_range=[-0.12, 0.12],
                rotation=None, ensure_object_boundary_in_range=False, ensure_valid_placement=True,
                reference_pos=env.table_offset, z_offset=0.01)
        self._robots = [_RobotProxy(env.robots[0])] if self.task.fake_gripper else None
        # LIBERO's robots publish robot0_joint_pos (the adapter reads it); plain robosuite 1.4 keeps it inactive.
        env.modify_observable("robot0_joint_pos", "active", True)
        # privileged object poses are never needed by the adapter: switch the object-modality sensors off
        for obs_name, observable in list(env._observables.items()):
            if getattr(observable, "modality", None) == "object":
                env.modify_observable(obs_name, "active", False)
        # parsed_problem-free: the completion gate and hidden goal audit see no BDDL goals
        self.parsed_problem = {"goal_state": []}
        self._clock_started = False
        self._task_steps = 0
        self._terminated_reason: str | None = None
        self._capped: dict | None = None
        self._enforce = os.environ.get("CAPGYM_ENFORCE_HORIZON", "0") == "1"
        self._horizon = int(self.task.horizon) - (self.task.settle_steps if self.task.settle == "held" else 0)

    # ------------------------------------------------------------ ControlEnv surface
    @property
    def robots(self):
        return self._robots if self._robots is not None else self.env.robots

    @property
    def sim(self):
        return self.env.sim

    @property
    def model_timestep(self):
        return self.env.model_timestep

    def seed(self, seed: int) -> None:
        # robosuite 1.4 placement samplers, WipeArena markers and initialization noise draw from the global numpy RNG
        np.random.seed(int(seed) % (2 ** 32))

    def reset(self):
        self._clock_started = False
        self._task_steps = 0
        self._terminated_reason = None
        self._capped = None
        obs = self.env.reset()
        self._pose_head_camera()
        return self._obs(obs)

    def start_task_clock(self) -> None:
        """Called by the episode builder after the adapter's reset + force calibration: CaP-Gym's horizon starts here."""
        self._clock_started = True
        self._task_steps = 0
        self._terminated_reason = None
        self._capped = None

    def step(self, action):
        if self._enforce and self._terminated_reason is not None:
            raise ValueError(f"executing action in terminated episode ({self._terminated_reason})")
        action = np.asarray(action, dtype=np.float64)
        if self.task.fake_gripper:
            action = action[:-1]
        obs, reward, done, info = self.env.step(action)
        if self._clock_started:
            self._task_steps += 1
            if self._terminated_reason is None:
                if self._task_steps >= self._horizon:
                    self._terminated_reason = f"CaP-Gym horizon of {self.task.horizon} control steps reached"
                elif self.task.early_termination and self.env._check_terminated():
                    self._terminated_reason = "Wipe early termination: " + ", ".join(self._wipe_termination_causes())
                if self._terminated_reason is not None:
                    self._capped = {"step": self._task_steps, "reason": self._terminated_reason, "success": self.check_success()}
                    if self.task.robosuite_env == "Wipe":
                        self._capped["wiped_markers"] = len(self.env.wiped_markers)
        return self._obs(obs), reward, done, info

    def _wipe_termination_causes(self):
        env = self.env
        causes = []
        if env.check_contact(env.robots[0].robot_model):
            causes.append("arm contact")
        if env._check_success():
            causes.append("all markers wiped")
        if env.robots[0].check_q_limits():
            causes.append("joint limit")
        return causes or ["terminated"]

    def check_success(self) -> bool:
        ok = bool(self.env._check_success())
        if ok and self.task.robosuite_env == "Restack":
            ok = not self._both_cubes_high()
        return ok

    def _both_cubes_high(self) -> bool:
        sim = self.env.sim
        base_z = float(sim.data.body_xpos[sim.model.body_name2id("robot0_base")][2])
        za = float(sim.data.body_xpos[self.env.cubeA_body_id][2]) - base_z
        zb = float(sim.data.body_xpos[self.env.cubeB_body_id][2]) - base_z
        return za > 0.04 and zb > 0.04

    def task_progress(self) -> dict:
        info = {"task_steps": self._task_steps, "horizon": self._horizon, "terminated": self._terminated_reason,
                "capped": self._capped, "enforced": self._enforce}
        if self.task.robosuite_env == "Wipe":
            info["wiped_markers"] = len(self.env.wiped_markers)
            info["num_markers"] = int(self.env.num_markers)
        return info

    def get_sim_state(self):
        return self.env.sim.get_state().flatten()

    def set_state(self, mujoco_state) -> None:
        self.env.sim.set_state_from_flattened(mujoco_state)

    def set_init_state(self, init_state):
        return self.regenerate_obs_from_state(init_state)

    def regenerate_obs_from_state(self, mujoco_state):
        self.set_state(mujoco_state)
        self.env.sim.forward()
        self.check_success()
        self._post_process()
        self.env._update_observables(force=True)
        return self._obs(self.env._get_observations())

    # No _pre_action here on purpose (as in LIBERO's ControlEnv): substep_force_guard.monitored_step walks to the
    # innermost env that owns _pre_action and instruments that one.

    def _update_observables(self, force: bool = False):
        self.env._update_observables(force=force)

    def _post_process(self):
        post = getattr(self.env, "_post_process", None)  # LIBERO-only hook; plain robosuite has none
        return post() if callable(post) else None

    def close(self) -> None:
        self.env.close()

    # ------------------------------------------------------------ CaP-Gym specifics
    def _obs(self, obs):
        if self.task.fake_gripper and isinstance(obs, dict) and np.asarray(obs.get("robot0_gripper_qpos", [])).size < 2:
            obs = dict(obs)
            obs["robot0_gripper_qpos"] = np.zeros(2)  # sponge: reported as a closed (0 open fraction) gripper
            obs["robot0_gripper_qvel"] = np.zeros(2)
        return obs

    def _pose_head_camera(self) -> None:
        """Put the ``agentview`` camera on CaP-Gym's head camera world pose (agentview is a worldbody camera)."""
        sim = self.env.sim
        model = sim.model
        src = model.camera_name2id(self.task.head_camera)
        dst = model.camera_name2id("agentview")
        sim.forward()
        model.cam_pos[dst] = np.array(sim.data.cam_xpos[src])
        model.cam_quat[dst] = _mat_to_quat_wxyz(np.array(sim.data.cam_xmat[src]).reshape(3, 3))
        model.cam_fovy[dst] = model.cam_fovy[src]
        sim.forward()


def _mat_to_quat_wxyz(m: np.ndarray) -> np.ndarray:
    import robosuite.utils.transform_utils as T
    xyzw = T.mat2quat(m)
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def capgym_init_state(env: CapGymRobosuiteEnv, seed: int, wrist_flip: bool | None = None) -> np.ndarray:
    """The CaP-Gym initial state for ``seed`` as a flattened MuJoCo state."""
    env.seed(seed)
    env.reset()
    task = env.task
    sim = env.env.sim
    if wrist_flip is None:
        wrist_flip = os.environ.get("CAPGYM_WRIST_FLIP", "1" if task.wrist_flip_default else "0") == "1"
    if task.init_qpos is not None:
        sim.data.qpos[:7] = np.asarray(task.init_qpos)
        sim.data.qvel[:7] = 0.0
        sim.forward()
    elif wrist_flip:
        sim.data.qpos[6] -= np.pi
    if task.settle == "held":
        # cap-x holds the arm at its parked joints with a joint-position controller (gripper open) for 100 control steps
        # while the objects settle. OSC zero-delta holding drifts (joint 2 sagged onto its -1.76 rad limit), so the arm
        # and fingers are pinned kinematically through the same simulated time instead.
        robot = env.env.robots[0]
        arm = list(robot._ref_joint_pos_indexes)
        arm_vel = list(robot._ref_joint_vel_indexes)
        fingers = list(getattr(robot, "_ref_gripper_joint_pos_indexes", []) or [])
        finger_vel = list(getattr(robot, "_ref_gripper_joint_vel_indexes", []) or [])
        arm_q = np.asarray(sim.data.qpos[arm]).copy()
        open_q = np.array([0.04, -0.04])[: len(fingers)] if fingers else None
        substeps = int(round(env.env.control_timestep / env.env.model_timestep))
        for _ in range(task.settle_steps * substeps):
            sim.data.qpos[arm] = arm_q
            sim.data.qvel[arm_vel] = 0.0
            if fingers:
                sim.data.qpos[fingers] = open_q
                sim.data.qvel[finger_vel] = 0.0
            sim.step()
        sim.data.qpos[arm] = arm_q
        sim.data.qvel[arm_vel] = 0.0
        if fingers:
            sim.data.qpos[fingers] = open_q
            sim.data.qvel[finger_vel] = 0.0
    else:
        for _ in range(task.settle_steps):
            sim.forward()
            sim.step()
    sim.forward()
    return np.asarray(env.get_sim_state(), dtype=np.float64).copy()


__all__ = ["CAPGYM_TASKS", "CapGymRobosuiteEnv", "CapGymTask", "FORK_WIPE_CONFIG", "capgym_init_state"]

"""SoarmSim: one task's MuJoCo physics (Menagerie SO-101 + objects), reset/settle, ticks, object state, held objects."""
from __future__ import annotations

import mujoco
import numpy as np

from .objects import ObjSpec
from .predicates import Pose, World
from .scene import ARM_JOINTS, Randomization, build_scene
from .tasks import Reject, SceneConfig, TaskInstance, evaluate_goal
from .twin import JAW_OPEN_RAD, grip_in_gripper_body, jaw_to_open_fraction

CONTROL_HZ = 20
# Home: gripper top-down (jaws closing along x) at (0.18, 0, 0.07). The harness checks IK at every sample of a straight
# move_to path with the orientation interpolated about one axis; for a 5-DoF arm that leaves the reachable set unless both
# ends are top-down (rotation about the vertical keeps every sample top-down). A tilted home (pilot v1) made most first moves
# fail with requested_cartesian_path_ik_unavailable. Exact top-down reach ends ~8 cm above the table (wrist_flex limit).
HOME_Q = np.array([-0.003, -0.5422, 0.6269, 1.4861, 0.0456])
GRIPPER_FORCE = 1.5                                   # N*m: LeRobot runs the gripper servo at 50 % torque


class SoarmSim:
    def __init__(self, task: int | str | dict, seed: int, *, config: SceneConfig | None = None, randomize: bool = True,
                 max_resample: int = 20) -> None:
        """`task`: spec dict, tid, or spec id (legacy specs are named after their family)."""
        from . import taskspec
        spec = task if isinstance(task, dict) else taskspec.get(task)
        self.spec = spec
        last = None
        for k in range(max_resample):
            task = taskspec.sample(spec, seed * 1000 + k, config)
            try:
                self._build(task, Randomization.sample(np.random.default_rng([seed, k, 7]), randomize, config))
                self.check_start()
                break
            except Reject as exc:
                last = exc
        else:
            raise RuntimeError(f"{spec['id']} seed {seed}: no valid start ({last})")

    # ------------------------------------------------------------------ build / reset
    def _build(self, task: TaskInstance, rand: Randomization) -> None:
        self.task, self.rand = task, rand
        spec = build_scene(task, rand)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.substeps = max(1, int(round(1.0 / (CONTROL_HZ * m.opt.timestep))))
        self.control_frequency = CONTROL_HZ
        self.arm_q = np.array([m.jnt_qposadr[m.joint(n).id] for n in ARM_JOINTS])
        self.arm_v = np.array([m.jnt_dofadr[m.joint(n).id] for n in ARM_JOINTS])
        self.jaw_q = m.jnt_qposadr[m.joint("gripper").id]
        m.actuator_forcerange[m.actuator("gripper").id] = [-GRIPPER_FORCE, GRIPPER_FORCE]
        self.specs: dict[str, ObjSpec] = {o.name: o for o in task.objects}
        self.body_of = {n: m.body(n).id for n in self.specs}
        self.grip_local = grip_in_gripper_body(m)
        self.gripper_body = m.body("gripper").id
        self.jaw_body = m.body("moving_jaw_so101_v1").id
        # fixed side of the jaw = the gripper body and everything rigidly attached to it (camera mount)
        self.fixed_side = {b for b in range(m.nbody) if b != self.jaw_body
                           and (b == self.gripper_body or int(m.body_parentid[b]) == self.gripper_body)}
        self.geom_object = {}
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if name.startswith("obj:"):
                self.geom_object[g] = name.split(":")[1]
        self.reset()

    def reset(self) -> None:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self.arm_q] = HOME_Q
        d.qpos[self.jaw_q] = JAW_OPEN_RAD
        d.ctrl[:5] = HOME_Q
        d.ctrl[5] = JAW_OPEN_RAD
        mujoco.mj_forward(m, d)
        for _ in range(int(1.0 / m.opt.timestep)):          # settle 1 s
            mujoco.mj_step(m, d)
        self.initial = self.poses()

    def check_start(self) -> None:
        """Reject starts where an object fell or moved > 2 cm while settling, or any goal atom already holds."""
        for n, p in self.initial.items():
            o = self.specs[n]
            if p.pos[2] < -0.01:
                raise Reject(f"{n} fell off the table")
            want = self.task.layout[n].xy
            if np.linalg.norm(p.pos[:2] - np.asarray(want)) > 0.02:
                raise Reject(f"{n} moved while settling")
            if o.category != "region" and not self.task.layout[n].lying and abs(float(p.rot[2, 2])) < 0.9 and o.kind != "ball":
                raise Reject(f"{n} tipped over while settling")
        if any(self.atom_values()):
            raise Reject("a goal atom holds at the start")

    # ------------------------------------------------------------------ state
    def poses(self) -> dict[str, Pose]:
        d = self.data
        return {n: Pose(pos=d.xpos[b].copy(), rot=d.xmat[b].reshape(3, 3).copy()) for n, b in self.body_of.items()}

    def held(self) -> frozenset[str]:
        """Objects touching both the moving jaw and the fixed side of the gripper."""
        m, d = self.model, self.data
        touch: dict[str, set[str]] = {}
        n = d.ncon
        for g1, g2 in zip(d.contact.geom1[:n].tolist(), d.contact.geom2[:n].tolist()):
            for g_obj, g_other in ((g1, g2), (g2, g1)):
                name = self.geom_object.get(int(g_obj))
                body = int(m.geom_bodyid[g_other])
                if name and body == self.jaw_body:
                    touch.setdefault(name, set()).add("jaw")
                elif name and body in self.fixed_side:
                    touch.setdefault(name, set()).add("fixed")
        return frozenset(n for n, sides in touch.items() if len(sides) == 2)

    def world(self) -> World:
        return World(specs=self.specs, poses=self.poses(), initial=self.initial, held=self.held())

    def atom_values(self) -> list[bool]:
        return evaluate_goal(self.world(), self.task.atoms)

    def success(self) -> bool:
        return all(self.atom_values())

    def grip_pose(self) -> np.ndarray:
        d = self.data
        body = np.eye(4)
        body[:3, :3] = d.xmat[self.gripper_body].reshape(3, 3)
        body[:3, 3] = d.xpos[self.gripper_body]
        return body @ self.grip_local

    def arm_qpos(self) -> np.ndarray:
        return self.data.qpos[self.arm_q].copy()

    def arm_qvel(self) -> np.ndarray:
        return self.data.qvel[self.arm_v].copy()

    def open_fraction(self) -> float:
        return jaw_to_open_fraction(float(self.data.qpos[self.jaw_q]))

    # ------------------------------------------------------------------ control
    def tick(self, arm_target: np.ndarray | None, jaw_target: float | None, ticks: int = 1) -> None:
        m, d = self.model, self.data
        if arm_target is not None:
            d.ctrl[:5] = np.clip(arm_target, m.actuator_ctrlrange[:5, 0], m.actuator_ctrlrange[:5, 1])
        if jaw_target is not None:
            d.ctrl[5] = float(np.clip(jaw_target, *m.actuator_ctrlrange[5]))
        for _ in range(ticks * self.substeps):
            mujoco.mj_step(m, d)

    @property
    def time_s(self) -> float:
        return float(self.data.time)

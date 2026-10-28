"""RoboTwin 2.0 SAPIEN process (runs in the RoboTwin python env, not the harness one).

    ROBOTWIN_ROOT=<RoboTwin copy> python sapien_server.py <read_fd> <write_fd>

Builds the official task scene (task class + seed, dual franka-panda, rasterization, no cuRobo), then
serves joint-level control and rendering over a pipe (wire.py framing): the harness side owns IK, the
twin and every public computation. Host-only requests (object poses, the task's check_success) are
only used for hidden scoring/audit.
"""
from __future__ import annotations

import importlib
import io
import math
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wire  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.environ.get("ROBOTWIN_ROOT", os.path.expanduser("~/r3/RoboTwin"))
os.environ.setdefault("ROBOTWIN_SHADER", "default")   # rasterization (ray tracing hangs under load on A100/H100)
os.environ.setdefault("ROBOTWIN_NO_PLANNER", "1")    # cuRobo is not needed: the harness plans
GRIPPER_LINKS = ("panda_hand", "panda_leftfinger", "panda_rightfinger")


def _pose_matrix(pose) -> np.ndarray:
    return np.asarray(pose.to_transformation_matrix(), dtype=np.float64)


class Server:
    def __init__(self) -> None:
        self.task = None
        self.cams = {}
        self.targets = {}
        self.grip_target = {}
        self.grip_goal = {}
        self.time_s = 0.0

    # ------------------------------------------------------------------ setup
    def init(self, task: str, seed: int, embodiment: str = "franka-panda", embodiment_dis: float = 0.6,
             cameras: dict | None = None, home: dict | None = None, settle_s: float = 0.5,
             gripper_rate_m_s: float = 0.05, domain_randomization: dict | None = None,
             arm_drive: tuple = (1e4, 1e3), robot_pose_offset=(0.0, 0.0, 0.0), layout_embodiment: str | None = None,
             task_config: str = "demo_clean", gripper_links: dict | None = None, wrist_camera_links: dict | None = None,
             arm_base_links: dict | None = None) -> dict:
        import yaml
        sys.path.insert(0, ROOT)
        os.chdir(ROOT)
        from envs._GLOBAL_CONFIGS import CONFIGS_PATH
        cls = getattr(importlib.import_module(f"envs.{task}"), task)
        args = yaml.safe_load(open(os.path.join(CONFIGS_PATH, f"{task_config}.yml")))
        args["task_name"] = task
        types = yaml.safe_load(open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml")))
        robot_file = types[embodiment]["file_path"]
        args["left_robot_file"] = args["right_robot_file"] = robot_file
        ecfg = yaml.safe_load(open(os.path.join(robot_file, "config.yml")))
        # one robot file with both arms (aloha-agilex, the official RoboTwin embodiment) vs two single-arm copies (franka)
        args["dual_arm_embodied"] = bool(ecfg.get("dual_arm", False))
        args["embodiment_dis"] = None if args["dual_arm_embodied"] else float(embodiment_dis)
        # per-arm link names: [hand, finger1, finger2] and the wrist camera link (franka defaults)
        self.gripper_links = gripper_links or {a: list(GRIPPER_LINKS) for a in ("left", "right")}
        self.wrist_camera_links = wrist_camera_links or {"left": "camera", "right": "camera"}
        # base placement offset (world), e.g. toward the table so RoboTwin layouts stay inside the top-down reach
        ecfg["robot_pose"] = [[*(np.asarray(p[:3], float) + np.asarray(robot_pose_offset, float)).tolist(), *p[3:]]
                              for p in ecfg["robot_pose"]]
        args["left_embodiment_config"] = args["right_embodiment_config"] = ecfg
        cams = yaml.safe_load(open(os.path.join(CONFIGS_PATH, "_camera_config.yml")))
        hc = cams[args["camera"]["head_camera_type"]]
        args["head_camera_h"], args["head_camera_w"] = hc["h"], hc["w"]
        args["embodiment_name"] = embodiment
        args["render_freq"] = 0
        args["collect_data"] = False
        args["save_path"] = "/tmp/robotwin2_unused"
        if domain_randomization:
            args["domain_randomization"].update(domain_randomization)
        self.gripper_rate = float(gripper_rate_m_s)
        rng_at_actors = None
        if layout_embodiment and layout_embodiment != embodiment:
            rng_at_actors = self._layout_rng_state(cls, task, int(seed), args, layout_embodiment, types)
        quiet = io.StringIO()
        stdout, sys.stdout = sys.stdout, quiet  # RoboTwin prints progress; keep the log readable
        original_load_actors = cls.load_actors
        if rng_at_actors is not None:
            def load_actors(task_self):
                # the instance's layout under layout_embodiment: the embodiments draw different numbers of random
                # values while loading robot and cameras, so continue from the RNG state that embodiment reaches here
                np.random.set_state(rng_at_actors)
                return original_load_actors(task_self)
            cls.load_actors = load_actors
        try:
            env = cls()
            env.setup_demo(now_ep_num=0, seed=int(seed), **args)
        finally:
            sys.stdout = stdout
            cls.load_actors = original_load_actors
        self.layout_embodiment = layout_embodiment or embodiment
        self.task = env
        self.robot = env.robot
        self.scene = env.scene
        self.dt = float(self.scene.get_timestep())
        self.arms = {"left": (env.robot.left_entity, env.robot.left_arm_joints, env.robot.left_gripper),
                     "right": (env.robot.right_entity, env.robot.right_arm_joints, env.robot.right_gripper)}
        self.gripper_scale = {"left": env.robot.left_gripper_scale, "right": env.robot.right_gripper_scale}
        self.links = {arm: {l.get_name(): l for l in ent.get_links()} for arm, (ent, _, _) in self.arms.items()}
        for arm, (_, joints, grip) in self.arms.items():
            # RoboTwin's 1000/200 drives leave ~3 mm of steady-state sag (= the harness pose tolerance) when tracking an
            # IK target instead of a dense planned trajectory; 1e4/1e3 tracks to ~0.2 mm
            for j in joints:
                j.set_drive_property(stiffness=float(arm_drive[0]), damping=float(arm_drive[1]))
            self.targets[arm] = np.array([j.get_drive_target()[0] for j in joints], float)
            self.grip_goal[arm] = 1.0
            self.grip_target[arm] = float(grip[0][0].get_drive_target()[0])
        if home:
            for arm, q in home.items():
                self.teleport(arm, q)
        self._add_cameras(cameras or {})
        self._substeps(int(settle_s / self.dt))
        self.time_s = 0.0
        return {"unstable_objects": list(getattr(env, "unstable_objects", []) or []),
                # per-arm base: the articulation root (one franka per arm) or a named link (aloha: both arms on one root)
                "bases": {arm: (_pose_matrix(self.links[arm][arm_base_links[arm]].get_pose()) if arm_base_links
                                else _pose_matrix(ent.get_root_pose())) for arm, (ent, _, _) in self.arms.items()},
                "timestep": self.dt, "state": self.read(), "objects": self.objects(),
                "joint_limits": {arm: np.asarray([j.get_limits()[0] for j in joints], float)
                                 for arm, (_, joints, _) in self.arms.items()},
                "success_at_start": self.success()}

    @staticmethod
    def _layout_rng_state(cls, task, seed, args, layout_embodiment, types):
        """numpy RNG state at load_actors() entry when the same instance is built with layout_embodiment (official
        robot); the throwaway scene stops there."""
        import copy
        import yaml
        largs = copy.deepcopy(args)
        f = types[layout_embodiment]["file_path"]
        largs["left_robot_file"] = largs["right_robot_file"] = f
        largs["dual_arm_embodied"] = True  # official single-entry embodiment list (e.g. [aloha-agilex]): one dual-arm robot file
        largs["left_embodiment_config"] = largs["right_embodiment_config"] = yaml.safe_load(open(os.path.join(f, "config.yml")))
        largs["embodiment_name"] = layout_embodiment

        class _Captured(Exception):
            pass

        state = {}
        original = cls.load_actors

        def capture(task_self):
            state["rng"] = np.random.get_state()
            raise _Captured()
        cls.load_actors = capture
        stdout, sys.stdout = sys.stdout, io.StringIO()
        try:
            cls().setup_demo(now_ep_num=0, seed=seed, **largs)
        except _Captured:
            pass
        finally:
            sys.stdout = stdout
            cls.load_actors = original
        import gc
        gc.collect()
        return state["rng"]

    def teleport(self, arm: str, q) -> None:
        ent, joints, grip = self.arms[arm]
        names = [j.get_name() for j in ent.get_active_joints()]
        qpos = np.asarray(ent.get_qpos(), float)
        for j, value in zip(joints, q):
            qpos[names.index(j.get_name())] = float(value)
        ent.set_qpos(qpos)
        ent.set_qvel(np.zeros_like(qpos))
        self.targets[arm] = np.asarray(q, float).copy()
        for j, value in zip(joints, q):
            j.set_drive_target(float(value))

    def _add_cameras(self, cameras: dict) -> None:
        import sapien
        for name, cfg in cameras.items():
            cam = self.scene.add_camera(name=name, width=int(cfg["width"]), height=int(cfg["height"]),
                                        fovy=math.radians(float(cfg["fovy_deg"])), near=0.02, far=10.0)
            if cfg["mount"] == "static":
                pos = np.asarray(cfg["position"], float)
                forward = np.asarray(cfg["forward"], float)
                forward /= np.linalg.norm(forward)
                left = np.asarray(cfg["left"], float)
                left -= forward * (left @ forward)
                left /= np.linalg.norm(left)
                up = np.cross(forward, left)
                mat = np.eye(4)
                mat[:3, :3] = np.stack([forward, left, up], axis=1)
                mat[:3, 3] = pos
                cam.entity.set_pose(sapien.Pose(mat))
            self.cams[name] = (cam, cfg)

    def set_camera_pose(self, name: str, position, forward, left) -> None:
        """Re-pose a static camera (world frame); training-aligned variant places agentview LIBERO-like in robot_base."""
        import sapien
        cam, cfg = self.cams[name]
        forward = np.asarray(forward, float); forward /= np.linalg.norm(forward)
        left = np.asarray(left, float); left -= forward * (left @ forward); left /= np.linalg.norm(left)
        mat = np.eye(4); mat[:3, :3] = np.stack([forward, left, np.cross(forward, left)], axis=1); mat[:3, 3] = np.asarray(position, float)
        cam.entity.set_pose(sapien.Pose(mat))

    # ------------------------------------------------------------------ physics
    def _substeps(self, n: int) -> dict:
        force = {arm: np.zeros(3) for arm in self.arms}
        gl = self.gripper_links
        gripper_ids = {arm: {id(self.links[arm][n]) for n in gl[arm] if n in self.links[arm]} for arm in self.arms}
        own = {arm: {id(l) for l in self.links[arm].values()} for arm in self.arms}
        finger_ids = {arm: {id(self.links[arm][n]) for n in gl[arm][1:] if n in self.links[arm]} for arm in self.arms}
        for _ in range(n):
            for arm, (ent, joints, grip) in self.arms.items():
                ent.set_qf(ent.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
                for j, q in zip(joints, self.targets[arm]):
                    j.set_drive_target(float(q))
                # finger targets move at a bounded rate toward the commanded opening (a servo gripper, not a jump)
                lo, hi = self.gripper_scale[arm]
                goal = lo + self.grip_goal[arm] * (hi - lo)
                step = self.gripper_rate * self.dt
                self.grip_target[arm] += float(np.clip(goal - self.grip_target[arm], -step, step))
                for joint, mult, offset in grip:
                    joint.set_drive_target(self.grip_target[arm] * mult + offset)
            self.scene.step()
            self.time_s += self.dt
            # while a jaw is still closing, finger contact along the closing axis is the squeeze on the object being
            # grasped (one finger touches first and pushes it sideways), not load on the wrist: leave it out so the
            # closure guard keeps its meaning (pushing into the support along the approach axis is still measured)
            closing_axis = {}
            for arm in self.arms:
                lo, hi = self.gripper_scale[arm]
                goal = lo + self.grip_goal[arm] * (hi - lo)
                fingers = [self.links[arm][n] for n in gl[arm][1:] if n in self.links[arm]]
                if self.grip_goal[arm] < 0.5 and abs(goal - self.grip_target[arm]) > 1e-5 and len(fingers) == 2:
                    d = np.asarray(fingers[0].get_entity_pose().p, float) - np.asarray(fingers[1].get_entity_pose().p, float)
                    if np.linalg.norm(d) > 1e-6:
                        closing_axis[arm] = d / np.linalg.norm(d)
            for contact in self.scene.get_contacts():
                a, b = contact.bodies
                for arm in self.arms:
                    ia, ib = id(a) in gripper_ids[arm], id(b) in gripper_ids[arm]
                    if ia == ib or (ia and id(b) in own[arm]) or (ib and id(a) in own[arm]):
                        continue
                    impulse = sum((np.asarray(p.impulse, float) for p in contact.points), np.zeros(3))
                    axis = closing_axis.get(arm)
                    if axis is not None and os.environ.get("RTW_FORCE_DEBUG") and np.linalg.norm(impulse) > 1e-4:
                        print("FORCEDBG", arm, a.get_name(), b.get_name(), np.round(impulse / self.dt, 2).tolist(),
                              np.round(axis, 3).tolist(), flush=True)
                    if axis is not None and id(a if ia else b) in finger_ids[arm]:
                        impulse = impulse - (impulse @ axis) * axis
                    # SAPIEN reports the impulse acting on bodies[0]
                    force[arm] += impulse if ia else -impulse
        return {arm: f / max(n * self.dt, 1e-9) for arm, f in force.items()}

    def step(self, targets: dict, n: int) -> dict:
        for arm, cmd in targets.items():
            if cmd.get("q") is not None:
                self.targets[arm] = np.asarray(cmd["q"], float)
            if cmd.get("open") is not None:
                self.grip_goal[arm] = float(np.clip(cmd["open"], 0.0, 1.0))
        if int(n) > 0:
            self.last_forces = self._substeps(int(n))
        return self.read(getattr(self, "last_forces", None))

    def exec_debug(self, code: str):
        """Offline probes only (tools/robotwin); never reachable from the agent API."""
        scope = {"self": self, "np": np}
        exec(code, scope)
        return scope.get("result")

    def reset_clock(self) -> None:
        self.time_s = 0.0

    def read(self, forces: dict | None = None) -> dict:
        out = {"time_s": self.time_s}
        for arm, (ent, joints, grip) in self.arms.items():
            names = [j.get_name() for j in ent.get_active_joints()]
            qpos, qvel = np.asarray(ent.get_qpos(), float), np.asarray(ent.get_qvel(), float)
            idx = [names.index(j.get_name()) for j in joints]
            fingers = [qpos[names.index(g[0].get_name())] for g in grip]
            out[arm] = {"q": qpos[idx], "dq": qvel[idx], "fingers": np.asarray(fingers, float),
                        "force_world": None if forces is None else forces[arm],
                        "camera_link": _pose_matrix(self.links[arm][self.wrist_camera_links[arm]].get_pose())
                        if self.wrist_camera_links[arm] in self.links[arm] else None,
                        "hand": _pose_matrix(self.links[arm][self.gripper_links[arm][0]].get_pose())}
        return out

    # ------------------------------------------------------------------ rendering
    def frames(self, names) -> dict:
        import sapien
        for name in names:
            cam, cfg = self.cams[name]
            if cfg["mount"] in ("left", "right"):
                link = self.links[cfg["mount"]][cfg.get("link") or self.wrist_camera_links[cfg["mount"]]]
                cam.entity.set_pose(link.get_pose())
        self.scene.update_render()
        out = {}
        for name in names:
            cam, cfg = self.cams[name]
            cam.take_picture()
            rgba = np.asarray(cam.get_picture("Color"))
            img = np.clip(rgba[..., :3], 0, 1)
            target = cfg.get("normalize_brightness")
            if target:  # training-aligned variant: global gain so the mean luminance matches a fixed target (no scene change)
                lum = float((img @ np.array([0.299, 0.587, 0.114])).mean())
                if lum > 1e-3:
                    img = np.clip(img * float(np.clip(float(target) / lum, 0.5, 2.5)), 0, 1)
            rgb = (img * 255 + 0.5).astype(np.uint8)
            pos = np.asarray(cam.get_picture("Position"))
            depth = (-pos[..., 2]).astype(np.float32)
            depth[pos[..., 3] >= 1.0] = 0.0
            gl = np.asarray(cam.get_model_matrix(), float)
            out[name] = {"rgb": rgb, "depth": depth, "K": np.asarray(cam.get_intrinsic_matrix(), float),
                         "optical_world": gl @ np.diag([1.0, -1.0, -1.0, 1.0])}
        return out

    # ------------------------------------------------------------------ host-only
    def _actors(self) -> dict:
        from envs.utils.actor_utils import Actor, ArticulationActor
        found = {}
        for key, value in vars(self.task).items():
            items = value if isinstance(value, (list, tuple)) else [value]
            for k, item in enumerate(items):
                if isinstance(item, (Actor, ArticulationActor)):
                    found[key if len(items) == 1 else f"{key}_{k}"] = item
        return found

    def objects(self) -> dict:
        out = {}
        for name, actor in self._actors().items():
            pose = actor.get_pose()
            out[name] = {"p": np.asarray(pose.p, float), "q_wxyz": np.asarray(pose.q, float),
                         "model": getattr(actor, "get_name", lambda: "")()}
        return out

    def segmentation(self, name: str) -> dict:
        """Host-only actor-level label image of one camera (projection checks); ids of the task actors."""
        self.frames([name])
        cam, _ = self.cams[name]
        seg = np.asarray(cam.get_picture("Segmentation"))[..., 1].astype(np.int32)
        ids = {key: int(actor.actor.per_scene_id) for key, actor in self._actors().items()}
        return {"labels": seg, "ids": ids}

    def success(self, open_fraction: dict | None = None) -> bool:
        st = self.read()
        for arm in ("left", "right"):
            frac = float(np.clip(np.mean(st[arm]["fingers"]) / 0.04, 0, 1)) if open_fraction is None else open_fraction[arm]
            setattr(self.robot, f"{arm}_gripper_val", frac)
        try:
            return bool(self.task.check_success())
        except Exception:  # noqa: BLE001
            return False

    def task_attrs(self, names) -> dict:
        """Plain task attributes chosen in load_actors (model names/ids for the instruction templates)."""
        out = {}
        for key in names:
            value = getattr(self.task, key, None)
            out[key] = value.item() if hasattr(value, "item") else value
        return out


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

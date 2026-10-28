"""RoboDojo (Isaac Sim 5.1 / Isaac Lab 2.3) scene server for the harness robodojo_sim suite.

Runs under RoboDojo's own python (cwd = RoboDojo root, PYTHONPATH = RoboDojo root), one process = one episode.
Speaks the same pipe protocol as robotwin2/sapien_server.py (wire.py messages: init / step / read / frames /
success / finalize / objects / close). Control stays native: every control tick is ONE official `take_action`
(joint targets of both X5 arms + gripper openings) followed by the official `is_episode_end`, so the benchmark's
own reward manager decides success, and success latches exactly as in RoboDojo's eval loop. After the official
episode ended (success, failure trigger or extended step limit) the robots keep following commands without any
further scoring, so the harness never sees a frozen robot.

Only host-only calls (success / finalize / objects / status) expose task state; they are never forwarded to the model.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import wire  # noqa: E402

ARMS = ("left", "right")
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def _pose7_to_matrix(v):
    from scipy.spatial.transform import Rotation
    v = np.asarray(v, float).reshape(-1)
    t = np.eye(4)
    w, x, y, z = v[3:7]
    t[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    t[:3, 3] = v[:3]
    return t


def _square(rgb, depth, K, size):
    """Centre-crop to a square and resize to size x size (the harness/LIBERO image contract); intrinsics follow."""
    import cv2
    h, w = rgb.shape[:2]
    s = min(h, w)
    x0, y0 = (w - s) // 2, (h - s) // 2
    rgb = cv2.resize(np.ascontiguousarray(rgb[y0:y0 + s, x0:x0 + s]), (size, size), interpolation=cv2.INTER_AREA)
    if depth is not None:
        depth = cv2.resize(np.ascontiguousarray(depth[y0:y0 + s, x0:x0 + s]), (size, size), interpolation=cv2.INTER_NEAREST)
    k = np.asarray(K, float).copy()
    k[0, 2] -= x0
    k[1, 2] -= y0
    k[:2] *= size / s
    return rgb, depth, k


class Server:
    image_size = 512

    def __init__(self) -> None:
        self.env = None
        self.app = None

    # ------------------------------------------------------------------ init
    def init(self, task: str, layout_id: int = 0, eval_seed: int = 0, gpu: int = 0, step_limit_factor: float = 20.0,
             out_dir: str | None = None, depth: bool = True, head_render_scale: float = 1.0,
             layout_variant: str = "official") -> dict:
        t0 = time.time()
        import argparse
        from isaaclab.app import AppLauncher
        p = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(p)
        args = p.parse_args([])
        args.headless = True
        args.enable_cameras = True
        # under rd_server.sh's CUDA mask our GPU is CUDA ordinal 0; otherwise the physical index
        cuda_gpu = 0 if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() else int(gpu)
        args.device = f"cuda:{cuda_gpu}"
        from env.camera_manager.capture.render_sync import add_zero_delay_kit_args
        add_zero_delay_kit_args(args)
        # Headless GPU nodes: no UI drawing, one GPU chosen through Kit settings, never CUDA_VISIBLE_DEVICES.
        # With H100FIX_ONLY_UUID the renderer's only Vulkan device is our GPU (Vulkan index 0).
        vk_gpu = 0 if os.environ.get("H100FIX_ONLY_UUID") else int(gpu)
        args.kit_args = ((args.kit_args or "") + " --/app/window/disableUi=true --/app/window/hideUi=true"
                         " --/app/window/drawMouse=false --/renderer/multiGpu/enabled=false"
                         f" --/renderer/multiGpu/maxGpuCount=1 --/renderer/activeGpu={vk_gpu}"
                         f" --/physics/cudaDevice={cuda_gpu}").strip()
        self.app = AppLauncher(args).app
        # RoboDojo hardcodes cuda:0 (TiledCaptureManager warp buffers); keep every CUDA allocation on our GPU
        import torch
        import warp as wp
        torch.cuda.set_device(cuda_gpu)
        wp.set_device(f"cuda:{cuda_gpu}")
        _wp_zeros = wp.zeros

        def _zeros(*a, device=None, **k):
            if str(device) == "cuda:0":
                device = f"cuda:{cuda_gpu}"
            return _wp_zeros(*a, device=device, **k)
        wp.zeros = _zeros
        t_app = time.time()

        import importlib
        import types
        stub = types.ModuleType("client_server.ws.model_client")

        class WsModelClient:  # the XPolicyLab websocket client is not used: the harness drives the robots
            def __init__(self, *a, **k):
                pass

            def call(self, *a, **k):
                return None

            def close(self):
                pass

        stub.WsModelClient = WsModelClient
        for name in ("client_server", "client_server.ws"):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["client_server.ws.model_client"] = stub
        from omegaconf import OmegaConf
        from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
        task_registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")
        from src.eval_client.eval_env import create_eval_env
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization

        if float(head_render_scale) != 1.0:
            # render the head camera at a higher resolution (same intrinsics up to scale); the harness still centre-crops
            # and resizes to 512 x 512. RoboDojo's scoring never reads images.
            from env_cfg.camera import template
            w, h = template.GEMINI_345LG["resolution"]
            template.GEMINI_345LG["resolution"] = (int(round(w * head_render_scale)), int(round(h * head_render_scale)))
        self.layout_variant = str(layout_variant)
        if self.layout_variant == "reachable":
            # ROUND 4 reachable-layout variant (rd_layout.py): the official layout file is transformed on load
            import rd_layout
            from env.global_configs import ASSETS_PATH
            from env.seed_manager import seed_manager as _sm
            robots_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, "robot", "dual_x5.yml"))["robots"]
            bases = [(float(r["default_root_pos"][0]), float(r["default_root_pos"][1])) for r in robots_cfg]
            record_path = os.path.join(os.path.dirname(os.path.abspath(out_dir)), "reachable_layout.json") if out_dir else None
            _orig_info = _sm.SeedManager.get_seed_scene_info

            def _reachable_info(sm_self, seed):
                data = _orig_info(sm_self, seed)
                new, rec = rd_layout.make_reachable(data, bases, ASSETS_PATH)
                rec["source_layout"] = sm_self.seed_info.get(seed, {}).get("scene_layout")
                if record_path:
                    with open(record_path, "w") as fh:
                        json.dump(rec, fh, indent=1)
                return new
            _sm.SeedManager.get_seed_scene_info = _reachable_info
        elif self.layout_variant != "official":
            raise ValueError(f"unknown layout_variant {layout_variant!r}")
        eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, "arx_x5.yml"))
        eval_cfg.update(task_name=task, num_envs=1, device_id=0, eval_batch=False, policy_name="captain",
                        additional_info="harness", seed=int(eval_seed), physx_monitor_enabled=False)
        eval_cfg["observation"]["vision"]["depth"] = bool(depth)
        eval_cfg["observation"]["vision"]["approximate_depth"] = False
        eval_cfg["observation"]["vision"]["intrinsic_matrix"] = True
        eval_cfg["observation"]["vision"]["extrinsic_matrix"] = True
        cam = load_yaml(os.path.join(ENV_CONFIG_PATH, "camera", "camera_config.yml"))
        if depth:
            for c in CAMERAS:
                cam["annotator"][c]["distance_to_image_plane_capture"] = {"type": "distance_to_image_plane", "device": "cpu"}
        env_cfg = OmegaConf.create({
            "sim": load_yaml(os.path.join(ENV_CONFIG_PATH, "sim", "sim_config.yml")),
            "scene": load_yaml(os.path.join(ENV_CONFIG_PATH, "scene", "default.yml")),
            "camera": cam,
            "robot": load_yaml(os.path.join(ENV_CONFIG_PATH, "robot", "dual_x5.yml")),
            "task_env": load_yaml(task_registry.task_config_path(os.path.join(ROOT_DIR, "task", BENCHMARK, "config"), task)),
            "eval_cfg": eval_cfg,
            "deploy_cfg": {"policy_name": "captain", "port": 1, "host": "localhost", "protocol": "ws"},
        })
        OmegaConf.update(env_cfg, "sim.scene.num_envs", 1, force_add=True)
        env_cfg = process_randomization(env_cfg)
        env_cfg, _ = process_config(env_cfg, task_name=task)
        OmegaConf.update(env_cfg, "camera.default_frequency", eval_cfg["observation"].get("collect_freq", 0), force_add=True)
        env_cfg.sim.seed = [int(layout_id)]
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            os.chdir(out_dir)  # RoboDojo writes eval_result/ relative to cwd
        env = create_eval_env(env_cfg, self.app)
        self.env = env
        env._stream_vision = lambda *a, **k: None  # no ffmpeg video streaming; the harness records its own frames
        t_env = time.time()
        env.reset(seed=[int(layout_id)])
        # official run_eval preamble: register the task's checks and process score
        env.run_reward()
        if hasattr(env, "get_score"):
            env.get_score()
        rm = env.reward_manager
        counts = {n: len(getattr(rm, n)[0]) for n in ("check_list", "final_check_list", "trigger_check_list")}
        if not sum(counts.values()):
            raise RuntimeError("native completion conditions are empty; refusing vacuous success")
        if getattr(env, "interact", False) and hasattr(env, "query_support_arm_traj"):
            env.query_support_arm_traj(env_idx=0)
        t_reset = time.time()
        self.manager = env.robot_manager
        self.robots = {r.arm_name.split("_")[0]: r for r in self.manager.robot_list if r.type == "target"}
        if set(self.robots) != set(ARMS):
            raise RuntimeError(f"expected left/right target arms, got {list(self.robots)}")
        # wrist force (no F/T sensor in RoboDojo): PhysX incoming joint force of each arm's wrist link (joint6 -> link6),
        # i.e. what a wrist F/T sensor between link5 and link6 measures (hand weight + contacts), rotated to world
        self.wrist = {}
        for arm, r in self.robots.items():
            key = self.manager.robot_key[self.manager.robot_list.index(r)]
            names = list(key.body_names)
            if "link6" in names and "link5" in names:
                self.wrist[arm] = (key, names.index("link6"))
        self.wrench_frame = os.environ.get("RD_WRENCH_FRAME", "child")
        self.native_step_limit = int(env.step_lim)
        env.step_lim = int(round(self.native_step_limit * float(step_limit_factor)))
        self.dt = 1.0 / float(env.obs_manager.collect_freq)
        self.q_cmd = {a: self._q(a) for a in ARMS}
        self.open_cmd = {a: 1.0 for a in ARMS}
        self.steps = 0                      # control ticks executed (official + post-end)
        self.first_success_step = None
        self.end_step = None
        self.end_reason = None
        self.finalized = False
        self.native_verdict = None          # official verdict at the native action budget (headline)
        self.timings = {"take_action_s": 0.0, "n": 0, "frames_s": 0.0, "frames_n": 0}
        obs = env.get_obs()
        bases = {}
        robots_info = {}
        for arm, r in self.robots.items():
            bases[arm] = _pose7_to_matrix(self.manager.get_link_pose(r, r.base_link, is_relative=True)[0])
            key = self.manager.robot_key[self.manager.robot_list.index(r)]
            limits = key.data.soft_joint_pos_limits[0, r.arm_joint_indices].cpu().numpy().astype(float)
            robots_info[arm] = {"urdf": r.urdf_path, "arm_joints": list(r.arm_joints_name), "ee_link": r.ee_link_name,
                                "base_link": r.base_link, "gripper_joints": list(r.gripper_joints_name),
                                "gripper_scale": list(map(float, r.gripper_scale)), "joint_limits": limits,
                                "body_names": list(key.body_names)}
        return {"timestep": self.dt, "bases": bases, "robots": robots_info, "state": self.read(),
                "instruction": obs.get("instruction"), "native_step_limit": self.native_step_limit,
                "extended_step_limit": int(env.step_lim), "condition_counts": counts, "layout_id": int(layout_id),
                "timing_s": {"app": round(t_app - t0, 1), "env": round(t_env - t_app, 1), "reset": round(t_reset - t_env, 1)}}

    # ------------------------------------------------------------------ state
    def _q(self, arm):
        return np.asarray(self.manager.get_joint(self.robots[arm])[0], float)[:6]

    def _dq(self, arm):
        r = self.robots[arm]
        key = self.manager.robot_key[self.manager.robot_list.index(r)]
        return key.data.joint_vel[0, r.arm_joint_indices].cpu().numpy().astype(float)

    def read(self) -> dict:
        out = {"time_s": self.steps * self.dt, "native_step": int(self.env.take_action_cnt[0]),
               "ended": bool(self.env.end_flag[0])}
        for arm, r in self.robots.items():
            fingers = np.asarray(self.manager.get_end_effector_real_val(r)[0], float)
            hand = _pose7_to_matrix(self.manager.get_link_pose(r, r.ee_link_name, is_relative=True)[0])
            out[arm] = {"q": self._q(arm), "dq": self._dq(arm), "fingers": fingers, "hand": hand,
                        **self._wrist_force(arm, r)}
        return out

    def _wrist_force(self, arm, r) -> dict:
        if arm not in self.wrist:
            return {"force_world": None}
        key, idx = self.wrist[arm]
        w = key.data.body_incoming_joint_wrench_b[0, idx].cpu().numpy().astype(float)
        R_child = _pose7_to_matrix(self.manager.get_link_pose(r, "link6", is_relative=True)[0])[:3, :3]
        R_parent = _pose7_to_matrix(self.manager.get_link_pose(r, "link5", is_relative=True)[0])[:3, :3]
        f_child, f_parent = R_child @ w[:3], R_parent @ w[:3]
        return {"force_world": f_child if self.wrench_frame == "child" else f_parent,
                "force_world_child": f_child, "force_world_parent": f_parent}

    # ------------------------------------------------------------------ control
    def _action(self):
        a = {}
        for arm in ARMS:
            a[f"{arm}_arm_joint_state"] = np.asarray(self.q_cmd[arm], np.float32)
            a[f"{arm}_ee_joint_state"] = np.asarray([self.open_cmd[arm]], np.float32)
        return a

    def _raw_step(self, action):
        """take_action without counting/scoring (after the official episode ended)."""
        env = self.env
        control = {}
        for robot in self.manager.robot_list:
            if robot.type != "target":
                continue
            arm = robot.arm_name.split("_")[0]
            control[self.manager.process_name(robot.arm_name)] = {"position": action[f"{arm}_arm_joint_state"]}
            val = float(np.clip(action[f"{arm}_ee_joint_state"][0], 0, 1))
            lo, hi = robot.gripper_scale
            val = val * (hi - lo) + lo if robot.gripper_move["sign"] == 1 else (1 - val) * (hi - lo) + lo
            mimic = robot.gripper_move["mimic"]
            control[self.manager.process_name(robot.gripper_name)] = {"position": [val, val * mimic[1] + mimic[2]]}
        seq = env.process_control_info(control, 0)
        self.manager.control_manager.push([0], [seq])
        while not env.have_empty([0]):
            env.step(env_idx_list=[0])

    def _tick(self):
        env = self.env
        action = self._action()
        t0 = time.time()
        if not env.end_flag[0]:
            env.take_action(action)
            if env.end_flag[0]:
                self.end_step = int(env.take_action_cnt[0])
                self.end_reason = "success" if env.success[0] else (
                    "step_limit" if env.take_action_cnt[0] >= env.step_lim else "failure_check")
                if env.success[0] and self.first_success_step is None:
                    self.first_success_step = int(env.take_action_cnt[0])
            self._native_check()
        else:
            self._raw_step(action)
        self.steps += 1
        self.timings["take_action_s"] += time.time() - t0
        self.timings["n"] += 1

    def step(self, targets: dict, n: int) -> dict:
        for arm, cmd in (targets or {}).items():
            if cmd.get("q") is not None:
                self.q_cmd[arm] = np.asarray(cmd["q"], float).reshape(-1)[:6]
            if cmd.get("open") is not None:
                self.open_cmd[arm] = float(np.clip(cmd["open"], 0.0, 1.0))
        for _ in range(int(n)):
            self._tick()
        return self.read()

    def reset_clock(self) -> None:
        self.steps = 0

    # ------------------------------------------------------------------ rendering
    def frames(self, names) -> dict:
        t0 = time.time()
        env = self.env
        env.obs_manager.render_for_capture()
        data = env.obs_manager.get_obs(env_idx_list=[0])[0]["vision"]
        out = {}
        for name in names:
            d = data[name]
            rgb = np.asarray(d["color"])[..., :3].astype(np.uint8)
            depth = np.asarray(d["depth"], np.float32).reshape(rgb.shape[:2]) if "depth" in d else None
            if depth is not None:
                depth = np.where(np.isfinite(depth) & (depth < 50.0), depth, 0.0).astype(np.float32)
            K = np.asarray(d["intrinsic_matrix"], float).reshape(3, 3)
            cam_world = np.asarray(d["extrinsic_matrix"], float).reshape(4, 4)
            # USD camera (looks along -z, y up) -> optical (z forward, y down); env origin of env 0 subtracted
            origin = np.asarray(self.manager.scene.env_origins[0].cpu(), float)
            cam_world = cam_world.copy()
            cam_world[:3, 3] -= origin
            rgb, depth, K = _square(rgb, depth, K, self.image_size)
            out[name] = {"rgb": rgb, "depth": depth, "K": K, "optical_world": cam_world @ np.diag([1.0, -1.0, -1.0, 1.0])}
        self.timings["frames_s"] += time.time() - t0
        self.timings["frames_n"] += 1
        return out

    # ------------------------------------------------------------------ host-only
    def success(self) -> bool:
        """Latched official success (RoboDojo ends the episode the moment its reward reaches 1)."""
        return bool(self.env.end_flag[0] and self.env.success[0])

    def finalize(self, reason: str = "harness") -> dict:
        """Official final check if the episode has not ended yet (= what RoboDojo does at the step limit)."""
        env = self.env
        if not self.finalized and not env.end_flag[0]:
            steps_at = int(env.take_action_cnt[0])
            env.take_action_cnt[0] = max(steps_at, int(env.step_lim))
            env.is_episode_end()
            env.take_action_cnt[0] = steps_at
            self.end_step = steps_at
            self.end_reason = f"finalize:{reason}"
            if env.success[0] and self.first_success_step is None:
                self.first_success_step = steps_at
        if self.native_verdict is None and int(env.take_action_cnt[0]) <= self.native_step_limit:
            self.native_verdict = self._verdict_now(int(env.take_action_cnt[0]), f"judged at finalize:{reason}")
        self.finalized = True
        return self.status()

    # ------------------------------------------------------------------ native budget (headline)
    def _verdict_now(self, step, basis):
        env = self.env
        success = bool(env.end_flag[0] and env.success[0])
        score = 1.0 if success else None
        if not success and hasattr(env, "get_score"):
            try:
                score = round(float(env.reward_manager.get_score()[0]) / 100.0, 4)
            except Exception:  # noqa: BLE001
                score = None
        return {"success": success, "score": score, "step": int(step), "basis": basis,
                "native_step_limit": self.native_step_limit}

    def _native_check(self):
        """Record the verdict RoboDojo's own client would give at the native action budget. Episodes are run up to
        step_limit_factor x the budget (secondary result); the headline must not depend on the extension."""
        if self.native_verdict is not None:
            return
        env = self.env
        n = int(env.take_action_cnt[0])
        if env.end_flag[0] and n <= self.native_step_limit:
            self.native_verdict = self._verdict_now(n, "episode ended within the native budget")
        elif n >= self.native_step_limit:
            self.native_verdict = self._official_at_limit(n)

    def _official_at_limit(self, n):
        """Exactly what eval_env.is_episode_end does when take_action_cnt reaches step_lim (final checks + gated
        process score), evaluated on a snapshot of the reward/func-parser state that is restored afterwards."""
        import copy
        env = self.env
        rm = env.reward_manager
        holders = [rm] + ([rm.func_parser] if getattr(rm, "func_parser", None) is not None else [])
        saved, skipped = [], []
        for obj in holders:
            snap = {}
            for k, v in vars(obj).items():
                if k in ("env", "layout_manager", "robot_manager", "scene_manager", "func_parser", "sim", "scene"):
                    continue
                try:
                    snap[k] = copy.deepcopy(v)
                except Exception:  # noqa: BLE001
                    skipped.append(f"{type(obj).__name__}.{k}")
            saved.append((obj, snap))
        success_flag = copy.deepcopy(env.success)
        try:
            reward = float(rm.get_reward(final_check=True)[0])
            success = bool(reward > 1 - 1e-3 and env.success[0])
            score = 1.0 if success else (round(float(rm.get_score()[0]) / 100.0, 4) if hasattr(env, "get_score") else 0.0)
            out = {"success": success, "score": score, "step": int(n), "basis": "official final check at the native budget",
                   "native_step_limit": self.native_step_limit, "snapshot_skipped": skipped}
        except Exception as exc:  # noqa: BLE001
            out = {"success": None, "score": None, "step": int(n), "error": f"{type(exc).__name__}: {exc}",
                   "native_step_limit": self.native_step_limit}
        finally:
            for obj, snap in saved:
                for k, v in snap.items():
                    setattr(obj, k, v)
            env.success[:] = success_flag[:] if hasattr(success_flag, "__len__") else success_flag
        return out

    def status(self) -> dict:
        env = self.env
        success = bool(env.end_flag[0] and env.success[0])
        out = {"official_success": success, "ended": bool(env.end_flag[0]), "end_reason": self.end_reason,
               "end_step": self.end_step, "first_success_step": self.first_success_step,
               "native_step_limit": self.native_step_limit, "extended_step_limit": int(env.step_lim),
               "success_within_native_limit": bool(success and self.first_success_step is not None
                                                   and self.first_success_step <= self.native_step_limit),
               "native_verdict": self.native_verdict,
               "control_ticks": self.steps, "pid": os.getpid(), "cmd_counts": dict(getattr(self, "cmd_counts", {})),
               "timings": {**self.timings,
                           "s_per_tick": round(self.timings["take_action_s"] / max(1, self.timings["n"]), 4),
                           "s_per_frames": round(self.timings["frames_s"] / max(1, self.timings["frames_n"]), 4)}}
        try:
            if success:
                out["official_score"] = 1.0
            elif hasattr(env, "get_score"):
                out["official_score"] = round(float(env.reward_manager.get_score()[0]) / 100.0, 4)
        except Exception as exc:  # noqa: BLE001  reporting only
            out["official_score"] = f"error: {type(exc).__name__}: {exc}"
        return out

    def objects(self) -> dict:
        """HOST-ONLY instance poses (world, env-relative) for the hidden audit."""
        lm = self.env.scene_manager.layout_manager
        out = {}
        recs_by_type = lm.object_records_by_type if isinstance(lm.object_records_by_type, dict) else {}
        for t, recs in recs_by_type.items():
            try:
                rows = recs.layout_records_by_env[0]
            except Exception:  # noqa: BLE001
                continue
            for rec in rows:
                name = rec.get("inst_name")
                try:
                    pos, rot = lm.get_instance_pose(0, inst_name=name)
                    pos = np.asarray(pos.cpu() if hasattr(pos, "cpu") else pos, float).reshape(-1)[:3]
                    rot = np.asarray(rot.cpu() if hasattr(rot, "cpu") else rot, float).reshape(-1)[:4]
                    out[str(rec.get("label") or name)] = {"p": pos, "q_wxyz": rot, "model": str(name), "type": str(t)}
                except Exception:  # noqa: BLE001
                    continue
        return out

    def exec_debug(self, code: str):
        """Offline probes only (smoke tests); never reachable from the agent API."""
        scope = {"self": self, "np": np}
        exec(code, scope)
        return scope.get("result")

    def close(self) -> None:
        pass


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
        server.cmd_counts = getattr(server, "cmd_counts", {})
        server.cmd_counts[cmd] = server.cmd_counts.get(cmd, 0) + 1
        t0 = time.time()
        try:
            if cmd == "close":
                wire.send(writer, {"ok": True})
                break
            result = getattr(server, cmd)(**message)
            wire.send(writer, {"ok": True, "result": result, "wall_s": time.time() - t0})
        except Exception as exc:  # noqa: BLE001
            wire.send(writer, {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    try:
        if server.app is not None:
            os._exit(0)  # Kit shutdown can hang; the episode is over
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()

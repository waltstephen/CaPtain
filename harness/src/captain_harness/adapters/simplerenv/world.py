"""Assemble one SimplerEnv episode world: the SimplerEnv process, the arm twin, one XR1Driver-protocol arm driver.

The harness side is the robotwin2 one (RoboTwinSim's control clock + ArmDriver's twin IK) over a server that speaks
the same pipe protocol; only the process start-up, the twin and the host-only audit differ.
"""
from __future__ import annotations

import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from captain_harness.adapters.robotwin2 import wire
from captain_harness.adapters.robotwin2.sim_driver import ArmDriver, RoboTwinSim

from .twin import build_arm_twin

HERE = Path(__file__).resolve().parent
ARM = "right"


def family_of(robot: str) -> str:
    return "google_robot" if robot.startswith("google_robot") else "widowx"


class SimplerSim(RoboTwinSim):
    """The SimplerEnv scene of one official episode, reached over a pipe (RoboTwinSim protocol, one arm)."""

    def __init__(self, config: Mapping, task: Mapping, *, log_path: Path | None = None) -> None:  # noqa: D401
        self.config = config
        self.task = task
        se = config["simplerenv"]
        to_child_r, to_child_w = os.pipe()
        from_child_r, from_child_w = os.pipe()
        env = dict(os.environ, SIMPLER_ROOT=os.path.expanduser(se["root"]),
                   __EGL_VENDOR_LIBRARY_FILENAMES=os.path.expanduser(se["egl_vendor_file"]),
                   VK_ICD_FILENAMES=se.get("vk_icd", "/usr/share/vulkan/icd.d/nvidia_icd.json"))
        env.pop("PYTHONPATH", None)
        log = open(log_path, "ab") if log_path else subprocess.DEVNULL
        self._proc = subprocess.Popen([os.path.expanduser(se["python"]), str(HERE / "sim_server.py"), str(to_child_r),
                                       str(from_child_w)], pass_fds=(to_child_r, from_child_w), stdin=subprocess.DEVNULL,
                                      stdout=log, stderr=subprocess.STDOUT, env=env, cwd=os.path.expanduser(se["root"]))
        os.close(to_child_r)
        os.close(from_child_w)
        self._w = os.fdopen(to_child_w, "wb")
        self._r = os.fdopen(from_child_r, "rb")
        self.control_frequency = int(config["control"]["control_frequency_hz"])
        spec = task["simplerenv"]
        self.family = family_of(spec["robot"])
        cameras = config["families"][self.family]["cameras"]
        cams = {cam["sim_name"]: {k: v for k, v in cam.items() if k not in ("sim_name", "depth", "official")}
                for cam in cameras.values() if not cam.get("official")}
        main = [cam["sim_name"] for cam in cameras.values() if cam.get("official")]
        info = self.rpc("init", **spec, main_camera=main[0] if main else None, cameras=cams,
                        letterbox=config.get("letterbox_px"), advance_subtask=bool(task.get("advance_subtask")),
                        arm_drive=config["families"][self.family].get("arm_drive"))
        self.info = info
        self.timestep = float(info["timestep"])
        self.joint_limits = {ARM: np.asarray(info["joint_limits"][ARM], float)}
        self.unstable_objects = []
        self.primary = ARM
        from .twin import FAMILIES
        mount = np.eye(4)
        mount[:3, 3] = FAMILIES[self.family].get("base_offset", [0.0, 0.0, 0.0])
        self.world_to_base = np.linalg.inv(np.asarray(info["bases"][ARM], float) @ mount)
        self.bases = {ARM: np.eye(4)}
        self.state = info["state"]
        self._substep_debt = 0.0
        self._frames_cache = None

    def objects(self) -> dict:
        out = {}
        for name, obj in self.rpc("objects").items():
            from scipy.spatial.transform import Rotation
            pose = np.eye(4)
            w, x, y, z = obj["q_wxyz"]
            pose[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
            pose[:3, 3] = obj["p"]
            out[name] = {"pose": self.world_to_base @ pose, "model": obj.get("model")}
        return out

    def evaluation(self) -> dict:
        return self.rpc("evaluation")


@dataclass
class SimplerWorld:
    sim: SimplerSim
    drivers: Mapping[str, ArmDriver]
    twin_paths: Mapping[str, Path]


class SimplerObjectAudit:
    """hidden-object-audit/v2 rows (robot_base): scene actor poses, the EEF, SimplerEnv's own evaluate()."""

    def __init__(self, sim: SimplerSim, driver: ArmDriver) -> None:
        self._sim = sim
        self._driver = driver
        self._names = list(sim.objects())

    def header(self) -> dict:
        goal = ["simplerenv_evaluate", self._sim.task["env_name"]]
        return {"schema_version": "hidden-object-audit/v2", "coordinate_frame": "robot_base", "goal_state": [goal],
                "goals": [{"goal_id": "goal_0", "predicate": goal[0], "arguments": goal[1:],
                           "rule": {"type": "simplerenv_evaluate"}}],
                "entities": self._names, "targets": [], "destinations": [], "movable": self._names, "fixtures": []}

    def sample(self) -> dict:
        objs = self._sim.objects()
        objects = {n: o["pose"][:3, 3].round(6).tolist() for n, o in objs.items()}
        yaws = {n: round(float(math.atan2(o["pose"][1, 0], o["pose"][0, 0])), 4) for n, o in objs.items()}
        try:
            ev = self._sim.evaluation()
            satisfied = bool(ev["eval"].get("success"))
        except Exception:  # noqa: BLE001
            ev, satisfied = None, None
        hand = self._sim.world_to_base @ np.asarray(self._sim.state[ARM]["hand"], float)
        return {"schema_version": "hidden-object-audit/v2", "objects": objects, "object_yaw_rad": yaws,
                "eef": hand[:3, 3].round(6).tolist(),
                "gripper_open_fraction": round(float(self._driver.read().gripper_open_fraction), 4),
                "goal_states": {"goal_0": {"satisfied": satisfied, "progress_kind": "predicate_only"}},
                "simplerenv": ev}


class SimplerArmDriver(ArmDriver):
    """ArmDriver with a tighter per-tick IK: the harness issues bounded setpoints (0.5 mm / 0.01 rad per step), and the
    RoboTwin driver's convergence test (0.2 mm / 2 mrad) returned the measured joints unchanged for setpoints inside it,
    so the arm never moved and the move ended as motion_stalled (offline replay of a Google Robot episode)."""

    def _ik(self, goal, q, iterations=20):
        import mujoco
        from scipy.spatial.transform import Rotation
        m, d = self.model, self._data
        q = q.copy()
        for _ in range(max(iterations, 40)):
            d.qpos[self._q] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            dp = goal[:3, 3] - d.site_xpos[self._site]
            dr = Rotation.from_matrix(goal[:3, :3] @ d.site_xmat[self._site].reshape(3, 3).T).as_rotvec()
            if np.linalg.norm(dp) < 2e-5 and np.linalg.norm(dr) < 2e-4:
                break
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, jp, jr, self._site)
            jac = np.vstack([jp[:, self._v], jr[:, self._v]])
            step = jac.T @ np.linalg.solve(jac @ jac.T + 0.002 * np.eye(6), np.r_[dp, dr])
            step *= min(1.0, 0.1 / max(np.abs(step).max(), 1e-9))
            q = np.clip(q + step, self._ik_limits[:, 0], self._ik_limits[:, 1])
        return q

    def hidden_object_audit(self):
        return SimplerObjectAudit(self.sim, self)


def build_world(config: Mapping, task: Mapping, *, output_dir: Path) -> SimplerWorld:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sim = SimplerSim(config, task, log_path=output_dir / "simplerenv_server.log")
    urdf = Path(os.path.expanduser(config["urdf"][sim.family]))
    xml = build_arm_twin(urdf, sim.family, name=f"simplerenv_{sim.family}")
    path = output_dir / f"twin_{sim.family}.xml"
    path.write_text(xml)
    cameras = config["families"][sim.family]["cameras"]
    camera_map = {name: cam["sim_name"] for name, cam in cameras.items()}
    depth = [name for name, cam in cameras.items() if cam.get("depth")]
    q_now = np.asarray(sim.state[ARM]["q"], float)
    # fingers report the open fraction directly ([frac, frac]), so the stroke is 1
    driver = SimplerArmDriver(sim, ARM, xml, home_joints=np.r_[q_now, np.zeros(7 - len(q_now))], camera_map=camera_map,
                              depth_cameras=depth, arm_dof=len(q_now), finger_stroke_m=1.0,
                              ik_limit_margin_rad=float(config.get("driver_ik_limit_margin_rad", 0.02)))
    driver.settle_on_hold = bool(config.get("r5_settle_on_hold"))
    driver.closure_hold_fix = bool(config.get("r5_closure_hold_fix"))
    return SimplerWorld(sim=sim, drivers={ARM: driver}, twin_paths={ARM: path})


__all__ = ["SimplerSim", "SimplerWorld", "build_world", "family_of", "wire"]

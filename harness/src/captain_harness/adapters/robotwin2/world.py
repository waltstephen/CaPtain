"""Assemble one RoboTwin episode world: SAPIEN process, per-arm twins, home posture, per-arm drivers."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from .sim_driver import ArmDriver, RoboTwinSim
from .twin import build_arm_twin

ARMS = ("right", "left")
# RoboTwin franka-panda homestate (embodiment config.yml): IK seed for the home posture
ROBOTWIN_HOMESTATE = [0, 0.19634954084936207, 0.0, -2.617993877991494, 0.0, 2.941592653589793, 0.7853981633974483]


@dataclass
class RoboTwinWorld:
    sim: RoboTwinSim
    drivers: Mapping[str, ArmDriver]
    twin_paths: Mapping[str, Path]
    home_poses: Mapping[str, np.ndarray]


def home_pose(config: Mapping, arm: str, base_in_robot_base: np.ndarray) -> np.ndarray:
    """Configured home grip-site pose of the primary arm, translated by the arm base offset for the others."""
    home = config["home_grip_site_pose_robot_base"]
    pose = np.eye(4)
    w, x, y, z = home["quaternion_wxyz"]
    pose[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    pose[:3, 3] = np.asarray(home["position"], float) + np.asarray(base_in_robot_base[:3, 3], float)
    return pose


def build_world(config: Mapping, task: Mapping, *, seed: int, output_dir: Path) -> RoboTwinWorld:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sim = RoboTwinSim(config, task, seed=seed, log_path=output_dir / "sapien_server.log")
    rt = config["robotwin"]
    urdf, srdf = Path(os.path.expanduser(rt["urdf"])), Path(os.path.expanduser(rt.get("srdf", rt["urdf"])))
    camera_map = {name: cam["sapien_name"] for name, cam in config["cameras"].items()}
    depth = [name for name, cam in config["cameras"].items() if cam.get("depth")]
    drivers, paths, homes, joints = {}, {}, {}, {}
    controlled = tuple(config.get("controlled_arms", ARMS))
    arx = rt.get("twin") == "arx5"  # aloha-agilex (official RoboTwin embodiment): two 6-DoF ARX5 arms on one robot
    for arm in ARMS:
        if arx:
            from .twin_arx import build_arm_twin as build_arx_twin, extract_arm_urdf, FINGER_STROKE_M as ARX_STROKE
            a = rt["arms"][arm]
            sub = extract_arm_urdf(urdf, a["prefix"], a["camera_link"], output_dir / f"arx5_{arm}.urdf")
            # the simulator's own joint limits (aloha URDF: +-10 rad); a tighter twin range made measured joints
            # "exceed" the twin limits after small overshoots and froze the arm (expert-oracle finding)
            clip = rt.get("twin_joint_limit_clip_rad")
            lim = sim.joint_limits.get(arm)
            if lim is not None and clip:
                lim = np.clip(lim, -float(clip), float(clip))
            xml = build_arx_twin(sub, sim.bases[arm], lim, name=f"robotwin_arx5_{arm}")
        else:
            xml = build_arm_twin(urdf, srdf, sim.bases[arm], name=f"robotwin_franka_{arm}")
        path = output_dir / f"twin_{arm}.xml"
        path.write_text(xml)
        arm_map = {**camera_map, **(config.get("per_arm_camera_map") or {}).get(arm, {})}  # e.g. left wrist camera
        if arx:
            q_now = np.asarray(sim.state[arm]["q"], float)
            driver = ArmDriver(sim, arm, xml, home_joints=q_now, camera_map=arm_map, depth_cameras=depth,
                               arm_dof=len(q_now), finger_stroke_m=ARX_STROKE,
                               ik_limit_margin_rad=float(rt.get("driver_ik_limit_margin_rad", 1e-4)))
            if config.get("home_grip_site_pose_robot_base"):
                # RoboTwin's aloha start posture (all joints 0) is folded/singular: straight Cartesian move_to paths out
                # of it fail tracking (expert-oracle finding). Re-home to a bent-elbow ready pose, like Franka.
                homes[arm] = home_pose(config, arm, sim.bases[arm])
                q = driver._ik(homes[arm], driver._pad(q_now) + np.r_[0, 1.0, 1.0, -1.0, 0, 0, 0], iterations=400)[:len(q_now)]
            else:  # RoboTwin's own start posture
                q = q_now
                homes[arm] = None
        else:
            driver = ArmDriver(sim, arm, xml, home_joints=ROBOTWIN_HOMESTATE, camera_map=arm_map, depth_cameras=depth)
            homes[arm] = home_pose(config, arm, sim.bases[arm])
            if arm in controlled:
                q = driver._ik(homes[arm], np.asarray(ROBOTWIN_HOMESTATE, float), iterations=400)
            else:  # single-arm eval: the other arm stays parked (config parked_joints; default RoboTwin's homestate)
                q = np.asarray(config.get("parked_joints", ROBOTWIN_HOMESTATE), float)
        driver.settle_on_hold = bool(config.get("r5_settle_on_hold"))
        driver.closure_hold_fix = bool(config.get("r5_closure_hold_fix"))
        drivers[arm], paths[arm], joints[arm] = driver, path, q
    sim.home(joints, float(config.get("home_settle_s", 0.5)))
    return RoboTwinWorld(sim=sim, drivers=drivers, twin_paths=paths, home_poses=homes)

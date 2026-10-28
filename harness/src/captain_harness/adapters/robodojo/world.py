"""Assemble one RoboDojo episode world: Isaac process (official eval layout), per-arm X5 twins, per-arm drivers."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .sim_driver import ArmDriver, RoboDojoSim
from .twin import build_arm_twin

ARMS = ("right", "left")


@dataclass
class RoboDojoWorld:
    sim: RoboDojoSim
    drivers: Mapping[str, ArmDriver]
    twin_paths: Mapping[str, Path]


def build_world(config: Mapping, task: Mapping, *, output_dir: Path) -> RoboDojoWorld:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sim = RoboDojoSim(config, task, log_path=output_dir / "rd_server.log", out_dir=output_dir / "rd_cwd")
    camera_map = {name: cam["robodojo_name"] for name, cam in config["cameras"].items()}
    depth = [name for name, cam in config["cameras"].items() if cam.get("depth")]
    drivers, paths = {}, {}
    for arm in ARMS:
        info = sim.robots[arm]
        urdf = Path(os.path.expanduser(config["robodojo"]["root"])) / info["urdf"] if not os.path.isabs(info["urdf"]) else Path(info["urdf"])
        xml = build_arm_twin(urdf, sim.bases[arm], info["joint_limits"], name=f"robodojo_x5_{arm}")
        path = output_dir / f"twin_{arm}.xml"
        path.write_text(xml)
        # each arm's eye_in_hand is its own wrist camera (config names the primary's)
        arm_map = {k: v.replace(f"{sim.primary}_wrist", f"{arm}_wrist") for k, v in camera_map.items()}
        drivers[arm] = ArmDriver(sim, arm, xml, camera_map=arm_map, depth_cameras=depth,
                                 status_path=output_dir / "host_status.json" if arm == sim.primary else None)
        paths[arm] = path
    sim.rpc("reset_clock")
    sim.state = sim.rpc("read")
    return RoboDojoWorld(sim=sim, drivers=drivers, twin_paths=paths)

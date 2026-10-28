"""env_server builder for the hardware_real suite: one real robot arm behind the LIBERO/CaP-X tool surface.

The robot is reached through a driver that implements `facade.XR1Driver` (read / command / hold / frames). The driver
is loaded from `--hardware-driver package.module:factory`; the factory is called as `factory(config)` with the parsed
`--hardware-config` JSON and returns the driver. Everything above the driver (tools, receipts, TASK.md, budgets,
completion protocol) is the code used for the LIBERO evaluations; only the robot intro and the relation line of TASK.md
are replaced, because the head camera of a real robot looks forward instead of facing the robot.

There is no success predicate on a real robot: the host records the episode and the outcome is labelled afterwards.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

INTRO_REAL = """You are operating a real robot arm through a small Python API. There is no other way to act."""

# "toward the camera" is LIBERO's agentview geometry; a robot's own head camera looks along +x.
RELATION_LIBERO = "table: front = +x (toward the camera), right = +y, within about 0.10 m of the reference object."
RELATION_FORWARD = "table: front = +x (away from the robot and the head camera), left = +y, within about 0.10 m of the reference object."


def load_driver(spec: str, config: dict):
    """`package.module:factory` -> factory(config)."""
    module_name, _, attr = str(spec).partition(":")
    if not module_name or not attr:
        raise ValueError(f"--hardware-driver must look like package.module:factory, got {spec!r}")
    factory = getattr(importlib.import_module(module_name), attr)
    return factory(config)


def build_hardware_episode(args, output_dir: Path, log):
    from captain_harness.adapters.capx import CapXHttpToolServices, CapXServiceEndpoints, ToolApiLevel
    from captain_harness.core.experience import ExperienceKey, ExperienceSnapshot
    from captain_harness.server.env_server import BuiltEpisode
    from .environment import CapXXR1EnvironmentAdapter
    from .facade import XR1FacadeEnv

    if args.hardware_config is None or args.hardware_task is None or not args.hardware_driver:
        raise ValueError("hardware_real needs --hardware-config, --hardware-task and --hardware-driver")
    root = args.harness_root.resolve()
    config_path = Path(args.hardware_config).resolve()
    config = json.loads(config_path.read_text())
    task = json.loads(Path(args.hardware_task).read_text())
    robot_xml = (config_path.parent / config["robot_xml"]).resolve()   # kinematic twin (MJCF) of the arm
    driver = load_driver(args.hardware_driver, config)
    cameras = config["cameras"]
    estimators = {}
    for name, cam in cameras.items():
        if cam.get("depth") == "mono":   # camera without a depth sensor: monocular depth service + table-plane scale
            from .mono_depth import MonoDepthSource
            mono = config.get("mono_depth", {})
            mask = None
            if mono.get("self_mask_bottom_fraction"):
                import numpy as np
                h, w = config.get("image_size", [512, 512])
                mask = np.zeros((h, w), bool)
                mask[int(h * (1 - mono["self_mask_bottom_fraction"])):] = True
            estimators[name] = MonoDepthSource(args.hardware_depth_url, model=mono.get("model", "moge2"), align="plane",
                                               support_plane_z_m=float(mono.get("support_plane_z_m", 0.0)),
                                               self_masks={name: mask} if mask is not None else None)
    env = XR1FacadeEnv(
        driver, robot_xml_path=robot_xml, camera_names=list(cameras),
        camera_links={name: cam["link"] for name, cam in cameras.items()},
        depth_cameras=[name for name, cam in cameras.items() if cam.get("depth")],
        osc_output_max=config["control"]["osc_output_max"],
        parked_joints=config.get("parked_joints"),
        workspace_box=config.get("workspace_box_robot_base_m"), depth_estimators=estimators)
    services = CapXHttpToolServices(
        CapXServiceEndpoints(sam3=args.sam3_url, molmo=args.molmo_url, openeta_revision="hardware-unused"),
        attempts=1, timeout_s=args.tool_service_timeout)
    key = ExperienceKey(environment_id=f"hardware-{args.suite}", environment_revision=config.get("robot", "hardware"),
                        tool_revisions={"sam3": "server", "molmo": "server"},
                        implementation_revision=args.experience_implementation_revision)
    overrides = {"intro": config.get("intro", INTRO_REAL), "standards_replace": {RELATION_LIBERO: RELATION_FORWARD}}
    adapter = CapXXR1EnvironmentAdapter(
        env, services=services, api_level=ToolApiLevel("S3"),
        tool_backend_options={"motion_position_tolerance_m": args.motion_position_tolerance,
                              "motion_orientation_tolerance_rad": args.motion_orientation_tolerance},
        public_safe_release=False, service_aligned=True, dehardcoded=True, v15_protocol=True,
        motion_tools=True, coding_control=True,
        completion_protocol=root / args.completion_protocol,
        experience_snapshot=ExperienceSnapshot.cold(key),
        completion_gate=False, completion_goals=None,
        observation_dir=output_dir / "observations", horizon=args.horizon,
        final_predicate=lambda: False,   # a real outcome is labelled after the episode
        backend_options={"motion_max_steps": args.motion_max_steps, "cartesian_max_steps": args.motion_max_steps,
                         "contact_detector": args.contact_detector,
                         "contact_calibration": bool(args.contact_calibration),
                         "control_frequency": int(config["control"]["control_frequency_hz"])},
        calibration_profile=root / config.get("calibration_profile", "configs/calibration/hardware_parallel_v0.json"),
        task_markdown_overrides=overrides)
    initial = adapter.reset(args.seed)
    language = task["language"]
    log(f"hardware episode ready suite={args.suite} task={language!r}")
    return BuiltEpisode(env=env, adapter=adapter, task=language, initial=initial)

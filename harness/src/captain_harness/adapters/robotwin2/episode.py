"""env_server builder for the robotwin2_sim suite (RoboTwin 2.0, dual franka-panda, SAPIEN rasterization)."""
from __future__ import annotations

import json
from pathlib import Path

INTRO = """You are operating two simulated Franka Panda arms (RoboTwin 2.0 benchmark, SAPIEN physics) through a
small Python API. There is no other way to act. The right arm is the primary arm: every call without
`arm` acts on it; pass `arm='left'` to the arm tools to use the left arm."""

# single-arm eval (config controlled_arms = [primary]): no `arm` argument, the manifest is the LIBERO one
INTRO_SINGLE = """You are operating a simulated Franka Panda arm (RoboTwin 2.0 benchmark, SAPIEN physics) through a
small Python API. There is no other way to act. A second, parked Franka arm stands at the left side of the
table; it is not controllable."""

# auto-arm routing (config auto_arm): the model may omit `arm`; one-line note on which arm is which
INTRO_AUTO = """You are operating two simulated {arm_model} arms (RoboTwin 2.0 benchmark, SAPIEN physics) through a
small Python API. There is no other way to act. Two arms: the right arm's base is the robot_base origin, the left arm's
base is 0.6 m to its left (+y); a motion without `arm` uses the arm whose base is nearer the target (an arm holding an
object keeps it), other arm tools act on the arm used last, and `arm='left'`/`'right'` overrides."""

# R5 benchmark guide (config r5_task_guide): short generic procedures, no instance information
GUIDE = """## Procedures that work on this benchmark (generic)
- Pick and place: move 0.10-0.15 m above the object, descend to grasp height, close, then lift about 0.10 m and
  check the grasp before transporting: after closing, `gripper_open_fraction` clearly above 0 means something is
  between the fingers; jaws closed fully means the grasp missed (open, re-measure, retry). Carry at a safe height,
  lower until the object is just above the destination, open, and retreat straight up.
- A move that stops with `contact_detected`, `contact_stall` or `emergency_limit` touched something: move up or call
  `recover()`, then change the plan (re-measure the height, shift or rotate the grasp); do not repeat the same descent.
- Articulated objects (lid, door, drawer, laptop screen, microwave): grasp the handle or edge, move along the opening
  direction in a few steps with `contact_mode='manipulation'` (pull, push or swing around the hinge) until it is fully
  open or closed, release, retreat.
- Buttons, bells, switches: close the gripper, move above the button, press straight down with
  `contact_mode='manipulation'` to about 0.02 m below its top surface, then withdraw upward.
- Two arms: each arm works best on its own side of the table. To bring an object to the other side, put it down near
  the middle of the table and pick it up with the other arm.
- As soon as the scene satisfies the task, call `robot.finish()`. Extra verification costs budget and can disturb
  the scene.
"""

RELATION_LIBERO = "table: front = +x (toward the camera), right = +y, within about 0.10 m of the reference object."
RELATION_ROBOTWIN = ("table: front = +x (away from the robot and the head camera), left = +y, within about 0.10 m of the "
                     "reference object.")


def build_robotwin_episode(args, output_dir: Path, log):
    from captain_harness.adapters.capx import CapXHttpToolServices, CapXServiceEndpoints, ToolApiLevel
    from captain_harness.adapters.hardware.facade import XR1FacadeEnv
    from captain_harness.core.experience import ExperienceKey, ExperienceSnapshot
    from captain_harness.server.env_server import BuiltEpisode
    from .environment import CapXRoboTwinEnvironmentAdapter
    from .world import build_world

    root = args.harness_root.resolve()
    config = json.loads(Path(args.robotwin_config).read_text())
    task = json.loads(Path(args.robotwin_task).read_text())
    world = build_world(config, task, seed=args.seed, output_dir=output_dir / "robotwin")
    cameras = config["cameras"]
    envs = {}
    controlled = tuple(config.get("controlled_arms", world.drivers))
    for arm, driver in world.drivers.items():
        if arm not in controlled:
            continue
        envs[arm] = XR1FacadeEnv(
            driver, robot_xml_path=world.twin_paths[arm], camera_names=list(cameras),
            camera_links={name: "panda_hand" for name in cameras},  # unused: frames carry their sim extrinsics
            depth_cameras=[name for name, cam in cameras.items() if cam.get("depth")],
            osc_output_max=config["control"]["osc_output_max"],
            workspace_box=config.get("workspace_box_robot_base_m"))
    primary = config.get("primary_arm", "right")
    services = CapXHttpToolServices(
        CapXServiceEndpoints(sam3=args.sam3_url, molmo=args.molmo_url, openeta_revision="robotwin2-unused"),
        attempts=1, timeout_s=args.tool_service_timeout)
    key = ExperienceKey(environment_id="robotwin2-sim", environment_revision=config.get("robot", "robotwin2"),
                        tool_revisions={"sam3": "server", "molmo": "server"},
                        implementation_revision=args.experience_implementation_revision)
    auto_arm = bool(config.get("auto_arm")) and len(envs) > 1
    intro_auto = INTRO_AUTO.format(arm_model=config.get("arm_model", "Franka Panda"))
    if config.get("embodiment_note"):  # one generic line about this robot's kinematics (e.g. reach)
        intro_auto = intro_auto + "\n" + str(config["embodiment_note"]).strip()
    overrides = {"intro": (intro_auto if auto_arm else INTRO) if len(envs) > 1 else INTRO_SINGLE,
                 # training-aligned variant: the head camera faces the robot as in LIBERO, so LIBERO's own relation line holds
                 "standards_replace": {} if config.get("libero_relation_line") else {RELATION_LIBERO: RELATION_ROBOTWIN}}
    if config.get("r5_task_guide"):
        overrides["guide"] = GUIDE
    adapter = CapXRoboTwinEnvironmentAdapter(
        envs, primary=primary, services=services, api_level=ToolApiLevel("S3"),
        tool_backend_options={"motion_position_tolerance_m": args.motion_position_tolerance,
                              "motion_orientation_tolerance_rad": args.motion_orientation_tolerance},
        public_safe_release=False, service_aligned=True, dehardcoded=True, v15_protocol=True,
        motion_tools=True, coding_control=True,
        completion_protocol=root / args.completion_protocol,
        experience_snapshot=ExperienceSnapshot.cold(key),
        completion_gate=False, completion_goals=None,
        observation_dir=output_dir / "observations", horizon=args.horizon,
        final_predicate=world.drivers[primary].success,
        backend_options={"motion_max_steps": args.motion_max_steps, "cartesian_max_steps": args.motion_max_steps,
                         "contact_detector": args.contact_detector,
                         "contact_calibration": bool(args.contact_calibration),
                         "control_frequency": int(config["control"]["control_frequency_hz"])},
        calibration_profile=root / config.get("calibration_profile", "configs/calibration/robotwin_franka_v0.json"),
        task_markdown_overrides=overrides)
    _adapter_finalize = adapter.finalize

    def _finalize_with_arm_log():
        if getattr(adapter, "_auto_arm_enabled", False):
            (output_dir / "robotwin" / "auto_arm.json").write_text(json.dumps(adapter.auto_arm_log, indent=1))
        if getattr(adapter, "_r4", None):
            (output_dir / "robotwin" / "r4_log.json").write_text(json.dumps(adapter.r4_log, indent=1))
        return _adapter_finalize()
    adapter.finalize = _finalize_with_arm_log
    if config.get("bimanual_tool") and len(envs) > 1:  # move_both (phase 3): added to the manifest at reset
        adapter.enable_bimanual_tool()
    adapter._r5_clear = bool(config.get("r5_clear_idle_arm"))
    r4 = {k[3:]: v for k, v in config.items() if k.startswith("r4_")}
    if r4:
        adapter.enable_r4(r4)
    initial = adapter.reset(args.seed)
    if auto_arm:
        adapter.enable_auto_arm({arm: pose[:3, 3] for arm, pose in world.sim.bases.items()})
        log(f"robotwin2 auto_arm enabled bases={ {a: p[:3, 3].round(3).tolist() for a, p in world.sim.bases.items()} }")
    log(f"robotwin2 episode ready task={task['robotwin_task']} seed={task.get('robotwin_seed', args.seed)} "
        f"language={task['language']!r}")
    return BuiltEpisode(env=envs[primary], adapter=adapter, task=task["language"], initial=initial)

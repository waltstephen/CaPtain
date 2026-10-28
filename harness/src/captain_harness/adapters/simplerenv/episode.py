"""env_server builder for the simplerenv_sim suite (SimplerEnv: Google Robot / WidowX, SAPIEN 2.2, official episodes).

Same model-facing API and TASK.md as the LIBERO eval; only the one-paragraph intro names the robot and the
relation line uses this benchmark's camera geometry (the camera looks forward, along +x, like RoboTwin's head camera).
"""
from __future__ import annotations

import json
from pathlib import Path

INTRO = {
    "google_robot": """You are operating a simulated Google Robot (a 7-joint arm with a two-finger gripper, mounted on a
mobile base that stays still; SimplerEnv benchmark, SAPIEN physics) through a small Python API. There is no other way
to act. The head camera (`agentview`) looks forward over the table; the wrist camera is mounted on the gripper.""",
    "widowx": """You are operating a simulated WidowX 250 arm (6 joints, small two-finger parallel gripper; SimplerEnv
benchmark, SAPIEN physics) through a small Python API. There is no other way to act. The fixed third-person camera
(`agentview`) looks over the robot onto the table; the wrist camera is mounted on the gripper.""",
}

RELATION_LIBERO = "table: front = +x (toward the camera), right = +y, within about 0.10 m of the reference object."
RELATION_FORWARD = ("table: front = +x (away from the robot and the camera), left = +y, within about 0.10 m of the "
                    "reference object.")


def build_simplerenv_episode(args, output_dir: Path, log):
    from captain_harness.adapters.capx import CapXHttpToolServices, CapXServiceEndpoints, ToolApiLevel
    from captain_harness.adapters.robotwin2.environment import CapXRoboTwinEnvironmentAdapter
    from captain_harness.adapters.hardware.facade import XR1FacadeEnv
    from captain_harness.core.experience import ExperienceKey, ExperienceSnapshot
    from captain_harness.server.env_server import BuiltEpisode
    from .world import ARM, build_world

    root = args.harness_root.resolve()
    config = json.loads(Path(args.simplerenv_config).read_text())
    task = json.loads(Path(args.simplerenv_task).read_text())
    world = build_world(config, task, output_dir=output_dir / "simplerenv")
    family = world.sim.family
    fam_cfg = config["families"][family]
    cameras = fam_cfg["cameras"]
    driver = world.drivers[ARM]
    env = XR1FacadeEnv(
        driver, robot_xml_path=world.twin_paths[ARM], camera_names=list(cameras),
        camera_links={name: "gripper0_eef" for name in cameras},  # unused: frames carry their sim extrinsics
        depth_cameras=[name for name, cam in cameras.items() if cam.get("depth")],
        osc_output_max=config["control"]["osc_output_max"], workspace_box=fam_cfg["workspace_box_robot_base_m"])
    services = CapXHttpToolServices(
        CapXServiceEndpoints(sam3=args.sam3_url, molmo=args.molmo_url, openeta_revision="simplerenv-unused"),
        attempts=1, timeout_s=args.tool_service_timeout)
    key = ExperienceKey(environment_id="simplerenv-sim", environment_revision=f"simplerenv-{family}",
                        tool_revisions={"sam3": "server", "molmo": "server"},
                        implementation_revision=args.experience_implementation_revision)
    overrides = {"intro": INTRO[family], "standards_replace": {RELATION_LIBERO: RELATION_FORWARD}}
    adapter = CapXRoboTwinEnvironmentAdapter(
        {ARM: env}, primary=ARM, services=services, api_level=ToolApiLevel("S3"),
        tool_backend_options={"motion_position_tolerance_m": args.motion_position_tolerance,
                              "motion_orientation_tolerance_rad": args.motion_orientation_tolerance},
        public_safe_release=False, service_aligned=True, dehardcoded=True, v15_protocol=True,
        motion_tools=True, coding_control=True,
        completion_protocol=root / args.completion_protocol,
        experience_snapshot=ExperienceSnapshot.cold(key),
        completion_gate=False, completion_goals=None,
        observation_dir=output_dir / "observations", horizon=args.horizon,
        final_predicate=driver.success,
        backend_options={"motion_max_steps": args.motion_max_steps, "cartesian_max_steps": args.motion_max_steps,
                         "contact_detector": args.contact_detector,
                         "contact_calibration": bool(args.contact_calibration),
                         "control_frequency": int(config["control"]["control_frequency_hz"])},
        calibration_profile=root / fam_cfg["calibration_profile"],
        task_markdown_overrides=overrides)
    adapter._r5_clear = bool(config.get("r5_clear_idle_arm"))
    adapter._r6_arrive = bool(config.get("r6_contact_arrival"))
    adapter._r6_clear_path = bool(config.get("r6_clear_path"))
    _adapter_finalize = adapter.finalize

    def _finalize_with_eval():
        try:  # host-only: SimplerEnv's own evaluate() at the end (final verdict + ever-success + partial metrics)
            (output_dir / "simplerenv" / "evaluation.json").write_text(json.dumps(world.sim.evaluation(), indent=1))
        except Exception as exc:  # noqa: BLE001
            log(f"simplerenv evaluation dump failed: {exc}")
        return _adapter_finalize()
    adapter.finalize = _finalize_with_eval
    initial = adapter.reset(args.seed)
    language = task.get("language") or world.sim.info["instruction"]
    log(f"simplerenv episode ready env={task['env_name']} id={task.get('episode_key')} family={family} "
        f"language={language!r}")
    return BuiltEpisode(env=env, adapter=adapter, task=language, initial=initial)

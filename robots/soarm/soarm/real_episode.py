"""env_server builder for suite `soarm_real`: the real SO-101 (armhost) behind the same tool surface, TASK.md (intro v5 /
compact_v1), perception services, MoGe-2 depth and speed profile as the twin episodes of soarm/episode.py.

  SOARM_REAL_CONFIG  hardware config json (configs/hardware/so101_armhost_real.json): calibration files, port, cameras
  SOARM_REAL_TASK    the instruction, e.g. "Put the blue cube on the sponge."
  SOARM_PROFILE      speed profile; the real arm uses fast_v5_real (release motion speeds)
There is no success predicate: a human labels the outcome. The driver parks the arm (rest pose, torque off) when
the episode closes.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .episode import CALIBRATION, CONTROL, WORKSPACE_BOX, prompt_record, task_markdown_overrides


def build_real_episode(args, output_dir: Path, log):
    from captain_harness.adapters.capx import CapXHttpToolServices, CapXServiceEndpoints, ToolApiLevel
    from captain_harness.adapters.hardware.environment import CapXXR1EnvironmentAdapter
    from captain_harness.adapters.hardware.facade import XR1FacadeEnv
    from captain_harness.core.experience import ExperienceKey, ExperienceSnapshot
    from captain_harness.server.env_server import BuiltEpisode

    from . import perception, speed_profile
    from .perception import LoggedMonoDepth
    from .real_depth import AffinePlaneDepth
    from .real_driver import RealSoarmDriver
    from .scene import CAMERAS

    config = json.loads(Path(os.environ["SOARM_REAL_CONFIG"]).read_text())
    language = os.environ["SOARM_REAL_TASK"].strip()
    if INTRO_CHECK not in task_markdown_overrides()["intro"]:
        raise RuntimeError("soarm_real needs the training prompt: SOARM_INTRO=v5")
    real_dir = output_dir / "soarm"
    real_dir.mkdir(parents=True, exist_ok=True)
    perc = perception.config(args, int(args.seed))
    if perc["mode"] != "real" or perc["depth"] != "moge2":
        raise RuntimeError("soarm_real runs real perception with MoGe-2 depth")
    perc["depth_service"] = perception.require_depth_healthy(perc["depth_url"])
    footprint = config["mono_depth"]["table_footprint_m"]
    perc["depth_table_footprint_m"] = footprint
    (real_dir / "prompt.json").write_text(json.dumps(prompt_record(), indent=1))
    (real_dir / "task.json").write_text(json.dumps({"language": language, "config": config}, indent=1))
    speed = speed_profile.load()
    if speed:
        (real_dir / "speed_profile.json").write_text(json.dumps(speed, indent=1))
        log(f"speed profile {speed['name']}: tool_backend_options={speed['tool_backend_options']} "
            f"backend_options={speed['backend_options']}")
    service_opts = speed["services"] if speed else {}
    driver = RealSoarmDriver({**config, "log_path": str(real_dir / "driver.jsonl")})
    twin_path = real_dir / "twin.xml"
    twin_path.write_text(driver.cal.nominal_xml)
    try:
        transport = perception.LoggedHttpTransport(real_dir / "perception.jsonl")
        # MoGe-2's real head-camera table comes out tilted; one scale (the twin's plane alignment) cannot remove it
        source = AffinePlaneDepth(perc["depth_url"], model=perception.MOGE2_MODEL,
                                  support_plane_z_m=float(perc["depth_support_plane_z_m"]), table_footprint_m=footprint)
        perc["depth_align"] = source.align
        (real_dir / "perception.json").write_text(json.dumps(perc, indent=1))
        logged = LoggedMonoDepth(source, transport)
        env = XR1FacadeEnv(driver, robot_xml_path=twin_path, camera_names=list(CAMERAS),
                           camera_links={name: "gripper0_eef" for name in CAMERAS},   # unused: frames carry extrinsics
                           depth_cameras=list(CAMERAS), osc_output_max=CONTROL["osc_output_max"],
                           workspace_box=config.get("workspace_box_robot_base_m", WORKSPACE_BOX),
                           depth_estimators={name: logged for name in CAMERAS})
        services = CapXHttpToolServices(CapXServiceEndpoints(sam3=perc["sam3_url"], molmo=perc["molmo_url"],
                                                             openeta_revision="soarm-unused"),
                                        transport=transport, attempts=1, timeout_s=args.tool_service_timeout,
                                        molmo_stop_after_coords=bool(service_opts.get("molmo_stop_after_coords", False)))
        perception.require_healthy(services)
        key = ExperienceKey(environment_id="soarm-real", environment_revision=config.get("robot", "so101-armhost"),
                            tool_revisions=perc["tool_revisions"],
                            implementation_revision=args.experience_implementation_revision)
        adapter = CapXXR1EnvironmentAdapter(
            env, services=services, api_level=ToolApiLevel("S3"),
            tool_backend_options={"motion_position_tolerance_m": args.motion_position_tolerance,
                                  "motion_orientation_tolerance_rad": args.motion_orientation_tolerance,
                                  **(speed["tool_backend_options"] if speed else {})},
            public_safe_release=False, service_aligned=True, dehardcoded=True, v15_protocol=True,
            motion_tools=True, coding_control=True,
            completion_protocol=args.harness_root.resolve() / args.completion_protocol,
            experience_snapshot=ExperienceSnapshot.cold(key),
            completion_gate=False, completion_goals=None,
            observation_dir=output_dir / "observations", horizon=args.horizon,
            final_predicate=lambda: False,   # a human labels a real outcome
            backend_options={"motion_max_steps": args.motion_max_steps, "cartesian_max_steps": args.motion_max_steps,
                             "contact_detector": args.contact_detector,
                             "contact_calibration": bool(args.contact_calibration),
                             "control_frequency": CONTROL["control_frequency_hz"],
                             **(speed["backend_options"] if speed else {})},
            calibration_profile=CALIBRATION,
            task_markdown_overrides=task_markdown_overrides())
        initial = adapter.reset(args.seed)   # the facade already moved the arm home (driver.reset)
    except BaseException:
        driver.close()
        raise
    log(f"soarm real episode ready sam3={perc['sam3_url']} molmo={perc['molmo_url']} depth={perc['depth_url']} "
        + (f"speed_profile={speed['name']} " if speed else "") + f"language={language!r}")
    return BuiltEpisode(env=env, adapter=adapter, task=language, initial=initial)


INTRO_CHECK = "You are operating an SO-101 arm"

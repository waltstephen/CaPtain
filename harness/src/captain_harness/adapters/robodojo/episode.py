"""env_server builder for the robodojo_sim suite (RoboDojo benchmark, dual ARX X5, Isaac Sim 5.1 RTX)."""
from __future__ import annotations

import json
import os
from pathlib import Path

INTRO = """You are operating two simulated ARX X5 arms (RoboDojo benchmark, Isaac Sim physics) through a
small Python API. There is no other way to act. The right arm is the primary arm: every call without
`arm` acts on it; pass `arm='left'` to the arm tools to use the left arm."""

# embodiment documentation (robot kinematics only, measured on the planner model; no task information)
REACH_NOTE = """Robot reach: each ARX X5 is a small 6-DoF arm. With the gripper pointing straight down, the reachable
EEF height above the table is at most about 0.30 m at 0.25 m in front of that arm's base, 0.24 m at 0.35 m, 0.12 m
at 0.45 m, and there is no straight-down reach beyond about 0.5 m or more than about 0.3 m sideways from its base.
Keep pre-grasp, lift and transport heights low (about 0.08-0.12 m above the table for objects 0.35-0.45 m away).
A move_to rejected with requested_cartesian_path_ik_unavailable is out of reach for that arm: lower or shorten
the motion instead of repeating it."""

RELATION_LIBERO = "table: front = +x (toward the camera), right = +y, within about 0.10 m of the reference object."
RELATION_ROBODOJO = ("table: front = +x (away from the robots and the head camera), left = +y, within about 0.10 m of the "
                     "reference object.")


# ROUND 5 generic benchmark guide (procedures only, no instance information; same style as the training TASK.md)
GUIDE = """Benchmark guide (generic procedures):
- Pick and place: observe; mark the object; measure its geometry; approach 0.06-0.10 m above it; descend to grasp height;
  close; check the close receipt (grasp_check object_between_fingers) before lifting; lift 0.05-0.08 m; transport low;
  lower until the object rests on the destination (for a stack: destination top height + half the object height + 0.01 m);
  open; retreat upward. Do not release an object in the air above its destination.
- Several objects (stack, sort, fill, classify): handle one object at a time, re-observe after each placement and keep
  a short list of what is done and what remains.
- Containers and covers: place the object inside / over the target, centred on the measured destination, then release.
- Buttons and keys: approach above, press down 0.01-0.02 m with the closed gripper, withdraw upward; repeat only as
  required by the instruction.
- Handles, lids, levers, drawers, laptops: grasp the handle or edge, move along the opening/closing direction in small
  steps to the end of travel, release and retreat.
- Pouring: grasp the container, lift, move above the target, tilt about 90 degrees, hold, return upright and set it down.
- Finishing: when the instruction is satisfied, call robot.finish() (the harness returns both arms to their starting
  poses). Do not keep observing or re-measuring once the task is done."""


def build_robodojo_episode(args, output_dir: Path, log):
    from captain_harness.adapters.capx import CapXHttpToolServices, CapXServiceEndpoints, ToolApiLevel
    from .dual_arm import CapXRoboTwinEnvironmentAdapter
    from captain_harness.adapters.hardware.facade import XR1FacadeEnv
    from captain_harness.core.experience import ExperienceKey, ExperienceSnapshot
    from captain_harness.server.env_server import BuiltEpisode
    from .world import build_world

    root = args.harness_root.resolve()
    config = json.loads(Path(args.robodojo_config).read_text())
    task = json.loads(Path(args.robodojo_task).read_text())
    world = build_world(config, task, output_dir=output_dir / "robodojo")
    cameras = config["cameras"]
    envs = {}
    for arm, driver in world.drivers.items():
        envs[arm] = XR1FacadeEnv(
            driver, robot_xml_path=world.twin_paths[arm], camera_names=list(cameras),
            camera_links={name: "link6" for name in cameras},  # unused: frames carry their sim extrinsics
            depth_cameras=[name for name, cam in cameras.items() if cam.get("depth")],
            osc_output_max=config["control"]["osc_output_max"],
            workspace_box=config.get("workspace_box_robot_base_m"))
    primary = config.get("primary_arm", "right")
    services = CapXHttpToolServices(
        CapXServiceEndpoints(sam3=args.sam3_url, molmo=args.molmo_url, openeta_revision="robodojo-unused"),
        attempts=1, timeout_s=args.tool_service_timeout)
    key = ExperienceKey(environment_id="robodojo-sim", environment_revision=config.get("robot", "robodojo"),
                        tool_revisions={"sam3": "server", "molmo": "server"},
                        implementation_revision=args.experience_implementation_revision)
    intro = INTRO
    if config.get("auto_arm") or os.environ.get("RD_AUTO_ARM") == "1":
        intro = intro.replace("The right arm is the primary arm: every call without\n`arm` acts on it; pass `arm='left'` to the arm tools to use the left arm.",
                              "A call without `arm` goes to the arm that can reach the requested pose (the nearer one), and stays\n"
                              "with the arm that is holding something; pass `arm='left'` or `arm='right'` to choose explicitly.")
        assert intro != INTRO, "auto-arm intro replacement failed"
    intro = intro + ("\n\n" + REACH_NOTE if config.get("reach_note") else "")
    if config.get("benchmark_guide"):
        intro += "\n\n" + GUIDE
    if config.get("auto_tilt") or config.get("auto_relay") or config.get("grasp_feedback"):
        intro += ("\n\nHarness assistance (automatic, reported in each receipt): if a requested gripper orientation is out of "
                  "reach, move_to tilts the approach by up to about 54 degrees (receipt field auto_tilt); a close that "
                  "ends with the jaws fully shut on nothing fails with grasp_check closed_empty and reopens the jaws; if "
                  "a held object's target is only reachable by the other arm, move_to hands it over at the table "
                  "midline and continues with the other arm (receipt field auto_relay)."
                  + (" An empty close is retried automatically (1 cm lower, then rotated 90 degrees) before it is "
                     "reported." if config.get("grasp_retry") else ""))
    overrides = {"intro": intro, "standards_replace": {RELATION_LIBERO: RELATION_ROBODOJO}}
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
        final_predicate=world.drivers[primary].host_predicate,
        backend_options={"motion_max_steps": args.motion_max_steps, "cartesian_max_steps": args.motion_max_steps,
                         "contact_detector": args.contact_detector,
                         "contact_calibration": bool(args.contact_calibration),
                         # X5 speed calibration: the harness may plan at a lower tick rate than RoboDojo's 25 Hz, so one
                         # native step executes 25/f ticks' worth of the requested Cartesian path (all durations scale)
                         "control_frequency": int(config["control"].get("harness_frequency_hz", world.sim.control_frequency)),
                         **({"gripper_settle_min_steps": int(config["control"]["gripper_settle_min_steps"])}
                            if config["control"].get("gripper_settle_min_steps") else {})},
        calibration_profile=root / config.get("calibration_profile", "configs/calibration/robodojo_x5_v0.json"),
        task_markdown_overrides=overrides)
    # X5 force semantics (wrist force = PhysX joint6 incoming force, sampled once per control tick; stiff joint PD):
    # calibration triangle tolerates free-motion force; detector timing per tick; after any guarded stop the joint
    # target is released to the measured pose (a stopped stiff PD arm would otherwise keep pressing at 100+ N)
    from captain_harness.adapters.hardware.backend import XR1ActionBackend
    XR1ActionBackend._calibration_rise_limit_n = float(config["control"].get("calibration_rise_limit_n", 30.0))
    XR1ActionBackend._force_samples_per_tick = 1.0
    XR1ActionBackend._closure_rise_limit_n = float(config["control"].get("closure_rise_limit_n", 5.0))
    # r6 step-efficiency profile (RoboDojo counts every native action against the task's budget): faster Cartesian
    # advance per tick, slow only near a manipulation-mode endpoint, looser "settled" motion bound, shorter gripper
    # settle. Defaults keep the R5 behaviour when the config has no "step_profile".
    prof = config["control"].get("step_profile") or {}
    XR1ActionBackend._speed_scale = float(prof.get("speed_scale", 1.0))
    XR1ActionBackend._slow_zone_m = float(prof.get("slow_zone_m", 0.0))
    XR1ActionBackend._settle_motion_m = float(prof.get("settle_motion_m", 0.00005))
    XR1ActionBackend._gripper_still_needed = int(prof.get("gripper_still_needed", 3))
    XR1ActionBackend._velocity_samples = bool(prof.get("velocity_samples", False))
    XR1ActionBackend._settle_samples_needed = int(prof.get("settle_samples_needed", 2))
    XR1ActionBackend._gripper_near_target = float(prof.get("gripper_near_target", 0.0))
    cal_file = config["control"].get("force_calibration_file")
    if cal_file and not getattr(XR1ActionBackend, "_rd_static_cal", False):
        # fixed robot calibration measured offline (non-test layouts, robot-only quantities) instead of the per-episode
        # calibration motion, which would spend native actions before the agent's first call
        from captain_harness.adapters.libero.contact_residual import ForceCalibration
        static = json.loads((root / cal_file).read_text())

        def calibrate_contact_residual(self):
            arm = self._LiberoActionBackend__env._driver.arm
            cal = ForceCalibration.from_json(static["arms"][arm])
            self._force_calibration = cal
            return cal, {"source": "static_file", "file": str(cal_file), "arm": arm, "native_actions": 0,
                         "measured_on": static.get("measured_on")}
        XR1ActionBackend.calibrate_contact_residual = calibrate_contact_residual
        XR1ActionBackend._rd_static_cal = True
    if not getattr(XR1ActionBackend, "_rd_release_wrapped", False):
        _sweep = XR1ActionBackend._execute_contact_sweep
        _stops = {"contact_force", "contact_detected", "emergency_limit", "load_budget_reached", "contact_stall",
                  "motion_stalled", "tracking_error_limit", "lateral_contact"}

        def _execute_contact_sweep(self, *a, **k):
            result = _sweep(self, *a, **k)
            if ((result.value or {}).get("stopped_reason") in _stops) and not k.get("retreat"):
                try:
                    self.cancel_direct_targets(cancel_gripper=False)
                except Exception:  # noqa: BLE001
                    pass
            return result
        XR1ActionBackend._execute_contact_sweep = _execute_contact_sweep
        XR1ActionBackend._rd_release_wrapped = True
    # ROUND 9 guard fixes (r9_guard.py; each behind its own flag in control.guard_fixes; defaults = r6 behaviour)
    from . import r9_guard
    _gf = r9_guard.flags(config)
    if any(_gf[k] for k in ("post_stop_backoff", "retare", "increase_only", "no_initial_limit", "start_blank",
                            "payload_comp", "contact_tolerant")):
        r9_guard.install_backend(XR1ActionBackend, _gf)
    # the official final check (irreversible: it ends the RoboDojo episode) may only run once the agent loop ended
    _adapter_finalize = adapter.finalize

    def _finalize_after_loop():
        if r4 is not None and config.get("auto_home"):
            try:
                r4.auto_home()   # ROUND 5: arms back to their start poses before the official final check
            except Exception as exc:  # noqa: BLE001
                r4.log["auto_home_error"] = f"{type(exc).__name__}: {exc}"
        world.drivers[primary].mark_loop_ended()
        if r4 is not None or config.get("auto_tilt"):
            from . import r4_adapt as _r4
            (output_dir / "robodojo" / "r4_log.json").write_text(json.dumps(
                {"executor": (r4.log if r4 is not None else None), "counts": _r4.COUNTS}, indent=1, default=str))
        if getattr(adapter, "_auto_arm_enabled", False):
            (output_dir / "robodojo" / "auto_arm.json").write_text(json.dumps(adapter.auto_arm_log, indent=1))
        return _adapter_finalize()
    adapter.finalize = _finalize_after_loop
    r4 = None
    if config.get("auto_tilt") or config.get("grasp_feedback") or config.get("auto_relay"):
        from . import r4_adapt
    if config.get("auto_tilt"):
        from captain_harness.adapters.capx.direct_programming import DirectProgramming
        r4_adapt.install_auto_tilt(DirectProgramming)
    if config.get("bimanual_tool"):
        adapter.enable_bimanual_tool()   # same move_both as RoboTwin phase 3 (robotwin2.environment)
    if config.get("reach_hint"):
        from captain_harness.adapters.capx import direct_programming
        direct_programming.REACH_HINT = True
    from captain_harness.adapters.capx import direct_programming as _dp
    _dp.IK_BRANCH_CHECK = True   # always on for RoboDojo (harness r9): an IK branch jump on a straight path = unavailable IK
    initial = adapter.reset(args.seed)
    for _arm, _robot in adapter._robots.items():   # arm base in robot_base (auto-tilt direction)
        _robot._rd_base_xy = world.sim.bases[_arm][:2, 3].copy()
    if config.get("grasp_feedback") or config.get("auto_relay"):
        r4 = r4_adapt.R4Executor(adapter, world, grasp_feedback=bool(config.get("grasp_feedback")),
                                 auto_relay=bool(config.get("auto_relay")), grasp_retry=bool(config.get("grasp_retry")))
        r4.record_starts()
        if any(_gf[k] for k in ("fingertip_floor", "hand_keepout", "approach_from_above", "payload_comp")):
            r9_guard.install_executor(r4, adapter, world, _gf)
    if config.get("auto_arm") or os.environ.get("RD_AUTO_ARM") == "1":
        adapter.enable_auto_arm({arm: pose[:3, 3] for arm, pose in world.sim.bases.items()})
        log(f"robodojo auto_arm enabled bases={ {a: p[:3, 3].round(3).tolist() for a, p in world.sim.bases.items()} }")
    language = str(task.get("language") or world.sim.info.get("instruction") or "").strip()
    suffix = str(config.get("language_suffix", "")).strip()
    if suffix:
        language = f"{language.rstrip('.')}. {suffix}"
    info = {k: world.sim.info.get(k) for k in ("instruction", "native_step_limit", "extended_step_limit", "layout_variant_used",
                                               "condition_counts", "layout_id", "timing_s")}
    (output_dir / "robodojo" / "episode_info.json").write_text(json.dumps({**info, "task": task, "language": language}, indent=2, default=str))
    log(f"robodojo episode ready task={task['robodojo_task']} layout={task['layout_id']} language={language!r} info={info}")
    return BuiltEpisode(env=envs[primary], adapter=adapter, task=language, initial=initial)

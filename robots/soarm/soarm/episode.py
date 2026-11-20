"""env_server builder for suite `soarm_sim`: one SO-101 twin episode behind the unchanged CaPtain tool surface.

--task-id = spec tid (CONTRACT 1: 0-15 = FAMILY_ORDER legacy families, 100+ cube specs), --seed = layout seed.
Everything above the driver (tools, receipts, TASK.md, budgets,
completion protocol) is the harness code used for LIBERO / hardware_real; only the robot intro paragraph is replaced.
The default agentview faces the robot like LIBERO's, so TASK.md's LIBERO relation line is kept; with intro v4 (div2:
camera views vary per episode) that sentence is replaced by a table-fixed one (same predicate semantics).
Perception (soarm/perception.py, SOARM_PERCEPTION): real SAM3 + Molmo2 services as in the LIBERO paper protocol
(default), or the privileged MuJoCo-segmentation stand-in (tests only). Depth (SOARM_DEPTH): MoGe-2 from the RGB frame
through the hardware adapter's MonoDepthSource (default; the real SO-ARM is RGB-only), or sim sensor depth (tests/ablation).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .families import FAMILIES

FAMILY_ORDER = tuple(sorted(FAMILIES))
CALIBRATION = Path(__file__).resolve().parent.parent / "configs" / "calibration" / "soarm101_v0.json"
CONTROL = {"control_frequency_hz": 20, "osc_output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]}
# robot_base box the facade clips per-tick goals to (table top z = 0; the arm cannot reach beyond ~0.44 m)
WORKSPACE_BOX = {"min": [-0.10, -0.40, 0.005], "max": [0.45, 0.40, 0.40]}

_INTRO_BASE = """You are operating a simulated SO-101 arm (a small low-cost 5-joint arm with one fixed and one hinged jaw that
opens to about 7 cm; MuJoCo physics) through a small Python API. There is no other way to act. The arm stands at the
robot_base origin on the table; the head camera (`agentview`) faces the robot from the front and looks down onto the
table; the wrist camera is mounted on the gripper. The arm starts top-down, 7 cm above the table. With 5 joints the
gripper cannot take every orientation: its approach direction must lie in the vertical plane through the base and the
gripper. A straight top-down approach is reachable (with any rotation about the vertical) from the table up to about
8 cm above it, at 0.10-0.32 m from the base. Every point of a move_to path must be reachable, so keep the gripper
top-down and at most ~8 cm above the table for reaching, grasping, carrying and placing; a top-down move only rotates
about the vertical, so it stays reachable. Top-down with the jaws closing along table direction theta (radians,
robot_base frame): quaternion_wxyz = [0, cos(theta/2), sin(theta/2), 0]. There is no force sensor: contact is
estimated from the servo loads"""
# Robot intro versions (TASK.md text, so it is part of the SFT data): v1 = pilot2 / scale-up day 1 / sft_v0 data and the
# un-tuned baselines; v2 (11-09 14:00) adds the push recipe and the finish reminder (deepseek hit the 60-request cap in
# 32/45 pilot episodes; push_to_region was 0/20 because the guarded mode stops at the first contact); v3 (11-10, cube
# domain) adds the fixed-jaw offset: centred top-down grasps of cubes wider than ~3.5 cm land the fixed jaw on the cube
# (scripted pick, 4.5-5 cm cubes: 0/5 centred, 5/5 with the offset; tools/scripted_pick.py).
INTRO_VERSION = os.environ.get("SOARM_INTRO", "v1")      # teacher collection sets v3 (cube batches)
REQUEST_BUDGET = os.environ.get("MAX_REQUESTS", "100")
INTROS = {
    "v1": _INTRO_BASE + ".",
    "v2": _INTRO_BASE + f""", and the default contact_mode='guarded' stops a move at the first contact; to push an
object, put the closed jaws 1-2 cm above the table beside it and move with contact_mode='manipulation' (e.g.
load_budget_n=15). You have at most {REQUEST_BUDGET} model requests: once a fresh measurement shows the goal holds,
call finish() and put your measured numbers in the note.""",
}
INTROS["v3"] = INTROS["v2"].replace(
    "load_budget_n=15). You have", "load_budget_n=15). The fixed jaw's inner face is about 1.8 cm from the grip point, so a "
    "grip point centred on an\nobject wider than about 3.5 cm lands the fixed jaw on top of it: for such an object, move "
    "the grip point (width/2 - 1.4 cm)\nfrom the object's centre in the direction -(cos theta, sin theta). You have")
assert INTROS["v3"] != INTROS["v2"]
# v4 (div2, 11-11): the arm is clamped at the table edge and the head camera's viewpoint varies per episode (front, side,
# oblique, high, low), so directions are tied to the table, not to the camera image (same predicate semantics:
# front = +x, right = +y); TASK.md's LIBERO relation line is replaced to match.
INTROS["v4"] = INTROS["v3"].replace(
    "The arm stands at the\nrobot_base origin on the table; the head camera (`agentview`) faces the robot from the front "
    "and looks down onto the\ntable;",
    "The arm is clamped at the\nback edge of the table at the robot_base origin (x points forward over the table, y to "
    "the arm's left, z up).\nThe head camera (`agentview`) stands somewhere around the table and looks down at the "
    "work area; its\nviewpoint changes from episode to episode (front, side, oblique, high or low), so take its pose "
    "from the\nobservation instead of assuming one. Directions in the task are fixed to the table, not to the camera "
    "image:\nas seen by a person standing in front of the robot and facing it, front = +x (away from the robot), "
    "behind = -x,\nright = +y, left = -y;")
assert INTROS["v4"] != INTROS["v3"]
# TASK.md standards sentence on relations (harness task_markdown.STANDARDS; the harness raises if it is not found)
RELATION_LIBERO = ("are judged in the head camera (`agentview`) view of the\n  table: front = +x (toward the camera), "
                   "right = +y, within about 0.10 m of the reference object.")
RELATION_TABLE = ("are judged on the table, whatever the head camera's viewpoint, as seen by a\n  person in front of the "
                  "robot facing it: front = +x (away from the robot base), right = +y, within about 0.10 m of the "
                  "reference object.")
# v5 (div3c, 11-12): the harness prompt profile compact_v1 (TASK.md ~2.5 k instead of ~13 k o200k tokens: API as
# signatures + one-line summaries with the full docs in workspace API.md / api/<name>.md, a NOTES.md STATUS rule, large
# capability results summarised with the full result in values/). Div3 transcripts spent ~40 % of all requests rebuilding
# state after context compactions that the 36 k-token base prompt forced every 2-3 requests (Div3Diagnose,
# div3 diagnosis). The intro is v4 with sim-neutral wording, so the real arm
# (configs/hardware/soarm101_prompt_depth.json, written by tools/real_arm_config.py) presents the same TASK.md.
INTROS["v5"] = INTROS["v4"].replace("You are operating a simulated SO-101 arm", "You are operating an SO-101 arm").replace(
    "opens to about 7 cm; MuJoCo physics)", "opens to about 7 cm)")
assert "simulated" not in INTROS["v5"] and "MuJoCo" not in INTROS["v5"]
STANDARDS_V5 = """## Standards the final scene is judged by
- Placement ("on"): the object's footprint center must rest within 0.02 m (XY) of the destination's footprint center.
  After the final release, call `measure_object_geometry(target_detection_id, destination_detection_id)` on a fresh
  observation and read `object_to_destination_xy_offset_m`.
- Containment ("in" / "into"): the object's center must lie inside the container's footprint with a consistent resting
  height (`object_center_inside_destination_footprint`, `resting_height_consistent`).
- Relations "left / right / front / back of X" """ + RELATION_TABLE.replace("\n  ", " ") + """
- Several goals: every placed object needs its own fresh `measure_object_geometry(object, its destination)` after your
  last physical action.
- Object identity is your call from the RGB images; text-prompted detections are hypotheses. Pin the instance you mean
  with `mark_point(observation_handle, camera_name, x_norm, y_norm, label=...)`, which returns a `detection_id` to
  measure and act on. When the task names the target through a reference, measure that relation before acting: a
  measurement of the wrong instance stays self-consistent all the way to `finish()`.
- `robot.finish()` checks your recorded measurements against these standards. If it returns
  `{"finished": false, "rejected": true, "message": ...}`, fix the scene or take the missing measurement and call it
  again (after two rejections the next `finish()` is accepted as-is).
"""
PROMPT_PROFILES = {"v5": "compact_v1"}   # intro version -> harness task_markdown profile (absent = full TASK.md)
INTRO = INTROS[INTRO_VERSION]


def task_markdown_overrides(intro_version: str | None = None) -> dict:
    """TASK.md overrides of an intro version; the real-arm config fragment is written from the same function."""
    version = intro_version or INTRO_VERSION
    intro = INTROS[version]
    if version in PROMPT_PROFILES:
        return {"profile": PROMPT_PROFILES[version], "intro": intro, "standards": STANDARDS_V5}
    if version == "v4":
        return {"intro": intro, "standards_replace": {RELATION_LIBERO: RELATION_TABLE}}
    return {"intro": intro}


def prompt_record(intro_version: str | None = None) -> dict:
    """What the agent's prompt was built from (soarm/prompt.json, batch rows): intro version and harness profile."""
    version = intro_version or INTRO_VERSION
    return {"intro": version, "prompt_profile": PROMPT_PROFILES.get(version, "full"),
            "task_md_version": f"soarm-{version}/{PROMPT_PROFILES.get(version, 'full')}"}


def _gate_goals(task) -> list[tuple[str, str, str]] | None:
    """Success atoms -> BDDL-style goal_state for the public-measurement completion gate; None when an atom is not
    gate-able.

    Only containment is gated: ("in", a, b), which the adapter's completion_gate.goals_from_bddl turns into a
    containment goal checked against the agent's own fresh measure_object_geometry(a, b) record; the gate never reads
    simulator state. Every other atom disables the gate and the final predicate decides alone. The gate rejects a
    destination smaller than the object, which fits LIBERO's plates and regions and soarm's mats and containers, but not
    soarm's object references: on_top (stack_two allows a top 5 mm wider than its base) and relation (gatefix smoke
    11-09: a host-successful relational_place was rejected twice).
    """
    goals = []
    for name, args, _params in task.atoms:
        if name not in ("inside", "in_region", "placed_at"):
            return None
        goals.append(("in", args[0], args[1]))
    return goals or None


class _TimedDepth:
    """Depth estimator wrapper: adds the estimator's wall time to the driver's timing.json."""

    def __init__(self, source, timing: dict) -> None:
        self.source = source
        self.timing = timing
        self.timing.setdefault("depth_calls", 0)
        self.timing.setdefault("depth_wall_s", 0.0)

    def __call__(self, **kwargs):
        import time
        t0 = time.perf_counter()
        try:
            return self.source(**kwargs)
        finally:
            self.timing["depth_calls"] += 1
            self.timing["depth_wall_s"] += time.perf_counter() - t0


def build_soarm_episode(args, output_dir: Path, log):
    from captain_harness.adapters.capx import CapXHttpToolServices, CapXServiceEndpoints, ToolApiLevel
    from captain_harness.adapters.hardware.environment import CapXXR1EnvironmentAdapter
    from captain_harness.adapters.hardware.facade import XR1FacadeEnv
    from captain_harness.core.experience import ExperienceKey, ExperienceSnapshot
    from captain_harness.server.env_server import BuiltEpisode

    from . import perception
    from .driver import SoarmDriver
    from .scene import CAMERAS
    from .sim import SoarmSim
    from .twin import build_twin_xml

    root = args.harness_root.resolve()
    from . import taskspec
    spec = taskspec.get(int(args.task_id))
    family = spec["id"]
    from .tasks import SceneConfig
    # eval suites (tools/batch.py --suite) pass their scene distribution here; unset = training scenes
    config = SceneConfig.from_json(json.loads(os.environ.get("SOARM_SCENE_CONFIG") or "{}"))
    import time as _time
    stage_t = {"t0": _time.perf_counter()}
    sim = SoarmSim(spec, int(args.seed), config=config)
    stage_t["sim"] = _time.perf_counter()
    twin_dir = output_dir / "soarm"
    twin_dir.mkdir(parents=True, exist_ok=True)
    twin_xml = build_twin_xml()
    twin_path = twin_dir / "twin.xml"
    twin_path.write_text(twin_xml)
    task_json = {**sim.task.to_json(), "spec": {k: spec[k] for k in ("id", "tid", "domain", "split", "version")
                                                if k in spec}, "scene_config": config.to_json()}
    (twin_dir / "task.json").write_text(json.dumps(task_json, indent=1))   # host only (output dir is hidden)
    perc = perception.config(args, int(args.seed))
    privileged = perc["mode"] == "privileged"
    mono = perc["depth"] == "moge2"
    if mono:   # refuse to start without the MoGe-2 depth service (an infrastructure error, not data)
        perc["depth_service"] = perception.require_depth_healthy(perc["depth_url"])
    (twin_dir / "perception.json").write_text(json.dumps(perc, indent=1))
    (twin_dir / "prompt.json").write_text(json.dumps(prompt_record(), indent=1))   # host only: what TASK.md was built from
    from . import speed_profile
    # SOARM_PROFILE (eval/deploy only; unset = release behaviour): soarm/speed_profile.py, configs/speed_<name>.json.
    # The motion options (speed scaling, step profile) NEED A HUMAN SAFETY REVIEW before the real arm; peak joint
    # speed/acceleration of every episode are in soarm/timing.json for the STS3215 servo-limit check.
    speed = speed_profile.load()
    if speed:
        (twin_dir / "speed_profile.json").write_text(json.dumps(speed, indent=1))   # host only
        log(f"speed profile {speed['name']}: tool_backend_options={speed['tool_backend_options']} "
            f"backend_options={speed['backend_options']}")
    service_opts = speed["services"] if speed else {}
    driver = SoarmDriver(sim, twin_xml, force_seed=int(args.seed), privileged_frames=privileged,
                         depth_noise_sigma_1m=perc["depth_noise_sigma_1m"], sensor_depth=not mono,
                         timing_path=twin_dir / "timing.json")
    if os.environ.get("SOARM_RECORD_FRAMES") == "1":   # eval-only: frames for tools/episode_video.py (host-only dir)
        driver.start_recording(twin_dir / "frames", list(CAMERAS))
    if privileged:   # tests / old data only: answers from MuJoCo segmentation
        from .perception_sim import SimPerceptionTransport
        transport = SimPerceptionTransport(driver, log_path=twin_dir / "perception_sim.jsonl")
    else:            # LIBERO paper protocol: real SAM3 + Molmo2 services
        transport = perception.LoggedHttpTransport(twin_dir / "perception.jsonl")
    # MoGe-2 depth: the real arm's monocular depth path (hardware/episode.py `"depth": "mono"` cameras) for both cameras
    estimators = perception.mono_depth_estimators(perc, CAMERAS, transport) if not privileged else {}
    estimators = {name: _TimedDepth(est, driver.timing) for name, est in estimators.items()}
    env = XR1FacadeEnv(driver, robot_xml_path=twin_path, camera_names=list(CAMERAS),
                       camera_links={name: "gripper0_eef" for name in CAMERAS},   # unused: frames carry extrinsics
                       depth_cameras=list(CAMERAS), osc_output_max=CONTROL["osc_output_max"],
                       workspace_box=WORKSPACE_BOX, depth_estimators=estimators)
    services = CapXHttpToolServices(CapXServiceEndpoints(sam3=perc["sam3_url"], molmo=perc["molmo_url"],
                                                         openeta_revision="soarm-unused"),
                                    transport=transport, attempts=1, timeout_s=args.tool_service_timeout,
                                    molmo_stop_after_coords=bool(service_opts.get("molmo_stop_after_coords", False)))
    if not privileged:   # refuse to start without answering SAM3/Molmo services (an infrastructure error, not data)
        perception.require_healthy(services)
    key = ExperienceKey(environment_id="soarm-sim", environment_revision=f"soarm-{family}",
                        tool_revisions=perc["tool_revisions"],
                        implementation_revision=args.experience_implementation_revision)
    gate_goals = _gate_goals(sim.task)
    stage_t["services"] = _time.perf_counter()
    adapter = CapXXR1EnvironmentAdapter(
        env, services=services, api_level=ToolApiLevel("S3"),
        tool_backend_options={"motion_position_tolerance_m": args.motion_position_tolerance,
                              "motion_orientation_tolerance_rad": args.motion_orientation_tolerance,
                              # speed profile: move_to speed scaling / default limits (harness motion_speed.py)
                              **(speed["tool_backend_options"] if speed else {})},
        public_safe_release=False, service_aligned=True, dehardcoded=True, v15_protocol=True,
        motion_tools=True, coding_control=True,
        completion_protocol=root / args.completion_protocol,
        experience_snapshot=ExperienceSnapshot.cold(key),
        completion_gate=bool(gate_goals), completion_goals=gate_goals,
        observation_dir=output_dir / "observations", horizon=args.horizon,
        final_predicate=driver.success,
        backend_options={"motion_max_steps": args.motion_max_steps, "cartesian_max_steps": args.motion_max_steps,
                         "contact_detector": args.contact_detector,
                         "contact_calibration": bool(args.contact_calibration),
                         "control_frequency": CONTROL["control_frequency_hz"],
                         # speed profile: gripper_settle_min_steps (end an open/close once the MEASURED jaw stopped
                         # moving instead of always 40 ticks = 2 s; a real servo reports the same position signal),
                         # step_profile (harness XR1ActionBackend, per instance)
                         **(speed["backend_options"] if speed else {})},
        calibration_profile=CALIBRATION,
        task_markdown_overrides=task_markdown_overrides())
    stage_t["adapter"] = _time.perf_counter()
    initial = adapter.reset(args.seed)
    stage_t["reset"] = _time.perf_counter()
    stages = dict(zip(("sim_build_s", "twin_driver_services_s", "adapter_s", "reset_s"),
                      (round(stage_t[b] - stage_t[a], 2) for a, b in (("t0", "sim"), ("sim", "services"),
                                                                      ("services", "adapter"), ("adapter", "reset")))))
    log(f"soarm episode ready family={family} seed={args.seed} perception={perc['mode']} "
        f"sam3={perc['sam3_url']} molmo={perc['molmo_url']} depth={perc['depth']} {perc.get('depth_url', '')} "
        + (f"speed_profile={speed['name']} " if speed else "") + f"build_stages={stages} "
        + f"language={sim.task.language!r}")
    return BuiltEpisode(env=env, adapter=adapter, task=sim.task.language, initial=initial)

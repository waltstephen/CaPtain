"""Episode builder for the CaP-Gym Robosuite tasks (``--suite capgym_lift|capgym_stack``).

Same adapter, tools, API level, calibration, completion protocol and motion settings as ``build_libero_episode``; only
the environment differs. The trial seed is CaP-Gym's ``seed=trial`` (seeded robosuite placement); ``--task-id`` is
unused (0). Host success = the robosuite env's own ``_check_success()`` on the final state (CaP-Gym ``task_completed``).
The BDDL-goal completion gate has no goals here, so ``finish()`` is never rejected.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable

CAPGYM_SUITES = ("capgym_lift", "capgym_stack", "capgym_restack", "capgym_wipe", "capgym_nut")


def build_capgym_episode(args: argparse.Namespace, output_dir: Path, log: Callable[[str], None]):
    from captain_harness.server.env_server import BuiltEpisode

    root = args.harness_root.resolve()
    os.environ["MUJOCO_GL"] = args.mujoco_gl
    os.environ["PYOPENGL_PLATFORM"] = args.mujoco_gl
    if args.mujoco_gl == "egl" and args.egl_device is not None:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.egl_device)

    from robosuite.utils.camera_utils import get_real_depth_map

    from captain_harness.adapters.capgym import CapGymRobosuiteEnv, capgym_init_state
    from captain_harness.adapters.capx import (
        CapXHttpToolServices,
        CapXLiberoEnvironmentAdapter,
        CapXServiceEndpoints,
        ToolApiLevel,
    )
    from captain_harness.core.experience import ExperienceKey, ExperienceSnapshot

    dependencies = json.loads((root / "third_party" / "manifest.json").read_text(encoding="utf-8"))["dependencies"]
    env = CapGymRobosuiteEnv(args.suite, controller=args.controller, horizon=args.horizon)
    init_state = capgym_init_state(env, args.seed)
    calibration_profile = root / (env.task.calibration_profile or args.calibration_profile)
    progress_path = output_dir / "capgym_progress.jsonl"
    last_progress: dict = {}

    def final_predicate() -> bool:
        ok = env.check_success()
        row = {**env.task_progress(), "success": ok}
        if row != last_progress:  # host-only bookkeeping, never served
            last_progress.clear()
            last_progress.update(row)
            with progress_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
        return ok
    experience_key = ExperienceKey(
        environment_id="capgym-robosuite-v1",
        environment_revision=str(dependencies["cap_x"]["revision"]),
        tool_revisions={k: str(dependencies[k]["revision"]) for k in ("cap_x", "openeta")},
        implementation_revision=args.experience_implementation_revision,
    )
    services = CapXHttpToolServices(
        CapXServiceEndpoints(sam3=args.sam3_url, molmo=args.molmo_url,
                             openeta_revision=str(dependencies["openeta"]["revision"])),
        attempts=1,
        timeout_s=args.tool_service_timeout,
    )
    adapter = CapXLiberoEnvironmentAdapter(
        env,
        services=services,
        api_level=ToolApiLevel("S3"),
        tool_backend_options={
            "motion_position_tolerance_m": args.motion_position_tolerance,
            "motion_orientation_tolerance_rad": args.motion_orientation_tolerance,
        },
        public_safe_release=False,
        service_aligned=True,
        dehardcoded=True,
        v15_protocol=True,
        motion_tools=True,
        coding_control=True,
        completion_protocol=root / args.completion_protocol,
        experience_snapshot=ExperienceSnapshot.cold(experience_key),
        completion_gate=bool(args.completion_gate),
        completion_goals=None,
        init_states=[init_state],  # index seed % 1 = 0: the state generated for this seed
        fresh_init_states=None,
        observation_dir=output_dir / "observations",
        horizon=args.horizon,
        final_predicate=final_predicate,
        backend_options={
            "depth_converter": lambda value: get_real_depth_map(env.sim, value),
            "motion_max_steps": args.motion_max_steps,
            "cartesian_max_steps": args.motion_max_steps,
            "controller": args.controller,
            "contact_detector": args.contact_detector,
            "contact_calibration": bool(args.contact_calibration),
        },
        calibration_profile=calibration_profile,
    )
    initial = adapter.reset(args.seed)
    env.start_task_clock()  # CaP-Gym's horizon counts from here (after the harness's own force calibration)
    log(f"episode ready suite={args.suite} seed={args.seed} capgym_task={env.task.robosuite_env} "
        f"horizon={env.task.horizon} calibration={calibration_profile.name} success_at_reset={env.check_success()} "
        f"progress={env.task_progress()} task={env.language_instruction!r}")
    return BuiltEpisode(env=env, adapter=adapter, task=env.language_instruction, initial=initial)


__all__ = ["CAPGYM_SUITES", "build_capgym_episode"]

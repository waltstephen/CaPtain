#!/usr/bin/env python
"""One-episode LIBERO environment server (V1.1).

Builds the LIBERO env + CapXLiberoEnvironmentAdapter with the v7 capability backend and the
completion gate, then exposes ONLY the public capability API over local HTTP JSON:

    GET  /health            liveness + public counters
    GET  /manifest          public capability manifest, rendered Python interface, task text
    POST /call              {"name": str, "arguments": {...}}  -> CapabilityResult as JSON
    POST /finish            {"note": str, "source": "agent"|...}  -> gate verdict, then scores once
    POST /shutdown          stop the process (after finish)

Budgets: 150 physical actions (robot_motion/actuator side effects), 600 perception/tool calls,
2400 s wall time; every request counts, rejected ones included. After the wall budget or
finish every /call is rejected. Nothing hidden (object poses, predicate value) is served: the
hidden predicate is sampled host-side after each physical action (kept in memory, written to
`hidden_success_audit.jsonl` only at finish) and once more inside finalize().

`/finish` from the agent first asks the adapter's completion gate, which reads the agent's own
recorded public measurements; a rejection is returned to the agent with its numbers and the
episode continues. Driver-initiated finishes (timeouts) bypass the gate.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import secrets
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
FINISH_SOURCES = ("agent", "driver_timeout", "driver_exit", "driver")


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _json_safe(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:  # pragma: no cover
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover
            pass
    return repr(value)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", required=True,
                   help="libero_spatial/object/goal/10/90; with --libero-variant pro|plus any suite of that fork's registry "
                        "(e.g. libero_goal_swap, libero_object_task); transfer benchmarks: robotwin2_sim, robodojo_sim, "
                        "simplerenv_sim, capgym_{lift,stack,restack,nut,wipe}; real robot: hardware_real")
    p.add_argument("--libero-variant", choices=("libero", "pro", "plus"), default="libero",
                   help="pro = LIBERO-PRO, plus = LIBERO-plus (forks of the libero package; run this server with the fork's venv)")
    p.add_argument("--libero-checkout", type=Path, default=None, help="fork repository root (default ~/benchmarks/LIBERO-PRO|LIBERO-plus)")
    p.add_argument("--init-state-index", type=int, default=None,
                   help="use this init state instead of seed %% n (LIBERO-plus protocol: 0); the seed still seeds the env")
    p.add_argument("--init-state-file", type=Path, default=None,
                   help="frozen init state (.npy, one flattened MuJoCo state, same layout as the task's init file) used instead of "
                        "the init file (eval only; LIBERO-PRO Pos+Task). The seed still seeds env.reset (fixture placement)")
    p.add_argument("--init-state-sha256", default=None, help="required sha256 of --init-state-file (refuse to run on mismatch)")
    p.add_argument("--instruction-source", choices=("auto", "filename", "bddl"), default="auto",
                   help="auto: the task name (original LIBERO, as in training) or the BDDL (:language) for pro/plus")
    p.add_argument("--instruction-text", default=None, help="override the instruction (e.g. a LIBERO-Para paraphrase)")
    p.add_argument("--robot-init-override", choices=("auto", "on", "off"), default="auto",
                   help="LIBERO-plus robot-init perturbation: re-apply the robot class init_qpos after set_init_state (auto = plus only)")
    p.add_argument("--max-case", type=Path, default=None,
                   help="LIBERO-MAX case JSON (one entry of the released manifest): attach its frozen mid-task event (eval only)")
    p.add_argument("--max-arm", choices=("control", "intervention"), default="intervention",
                   help="LIBERO-MAX arm: control = Base (setup only, trigger recorded), intervention = Dynamic (event applied)")
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--harness-root", type=Path, default=REPO_ROOT)
    p.add_argument("--images-dir", type=Path, default=None, help="where observe() PNGs are copied for the agent (default <output-dir>/workspace/images)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--max-actions", type=int, default=150)
    p.add_argument("--max-tool-calls", type=int, default=600)
    p.add_argument("--wall-time", type=float, default=2400.0)
    p.add_argument("--controller", default="OSC_POSE")
    p.add_argument("--horizon", type=int, default=45000)
    p.add_argument("--motion-max-steps", type=int, default=600)
    p.add_argument("--motion-position-tolerance", type=float, default=0.003)
    p.add_argument("--motion-orientation-tolerance", type=float, default=0.03)
    p.add_argument("--tool-service-timeout", type=float, default=600.0)
    p.add_argument("--calibration-profile", default="configs/calibration/libero_franka_v3.json")
    p.add_argument("--completion-protocol", default="configs/protocols/v12_public_completion_v1.json")
    p.add_argument("--experience-implementation-revision", default="v1.1-react")
    p.add_argument("--contact-detector", choices=("legacy", "residual"), default="residual")
    p.add_argument("--contact-calibration", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--completion-gate", action=argparse.BooleanOptionalAction, default=True,
                   help="ask the adapter's completion gate before an agent finish() is accepted")
    p.add_argument("--mujoco-gl", choices=("egl", "osmesa"), default=os.environ.get("MUJOCO_GL", "egl"))
    p.add_argument("--egl-device", type=int, default=None)
    p.add_argument("--libero-config-dir", type=Path, default=None, help="default <harness-root>/.libero-config")
    p.add_argument("--fresh-init-dir", type=Path, default=None,
                   help="fresh (non-official) init states for seeds >= 1000: <dir>/<problem_folder>/<task>.fresh_init "
                        "(default <harness-root>/data/libero_fresh_init); seeds < 1000 keep the official states")
    p.add_argument("--sam3-url", default=os.environ.get("SAM3_URL", "http://127.0.0.1:8114"))
    p.add_argument("--molmo-url", default=os.environ.get("MOLMO_URL", "http://127.0.0.1:8122/v1"))
    p.add_argument("--audit-hidden-success", action=argparse.BooleanOptionalAction, default=True,
                   help="sample the hidden predicate host-side after each physical action (never served)")
    p.add_argument("--idle-exit-s", type=float, default=600.0, help="exit if no request arrives for this long after finish")
    # transfer benchmarks and real robots (suites robotwin2_sim, robodojo_sim, simplerenv_sim, capgym_*, hardware_real)
    p.add_argument("--robotwin-config", type=Path, default=None, help="RoboTwin 2.0 dual-franka config json (robotwin2_sim)")
    p.add_argument("--robotwin-task", type=Path, default=None, help="RoboTwin task json: robotwin_task, robotwin_seed, language")
    p.add_argument("--robodojo-config", type=Path, default=None, help="RoboDojo dual-X5 config json (robodojo_sim)")
    p.add_argument("--robodojo-task", type=Path, default=None, help="RoboDojo task json: robodojo_task, layout_id, language")
    p.add_argument("--simplerenv-config", type=Path, default=None, help="SimplerEnv backend config json (simplerenv_sim)")
    p.add_argument("--simplerenv-task", type=Path, default=None, help="SimplerEnv episode json: env_name + official env/reset kwargs")
    p.add_argument("--hardware-config", type=Path, default=None, help="real-robot config json (hardware_real): twin, cameras, control")
    p.add_argument("--hardware-task", type=Path, default=None, help="real-robot task json: language")
    p.add_argument("--hardware-driver", default=os.environ.get("CAPTAIN_HARDWARE_DRIVER"),
                   help="package.module:factory returning an XR1Driver for the robot (hardware_real)")
    p.add_argument("--hardware-depth-url", default=os.environ.get("CAPTAIN_DEPTH_URL", "http://127.0.0.1:8130"),
                   help="monocular depth service for cameras without a depth sensor (hardware_real)")
    p.add_argument("--receipts", action="store_true",
                   help="add a one-line `receipt` (outcome + compact measured summary) to every /call result (python-runtime/v1)")
    return p


def episode_builder(suite: str):
    """The episode builder of a suite; None = the LIBERO builder (LIBERO, LIBERO-PRO/Plus/MAX)."""
    suite = str(suite)
    if suite == "robotwin2_sim":
        from captain_harness.adapters.robotwin2.episode import build_robotwin_episode
        return build_robotwin_episode
    if suite == "robodojo_sim":
        from captain_harness.adapters.robodojo.episode import build_robodojo_episode
        return build_robodojo_episode
    if suite == "simplerenv_sim":
        from captain_harness.adapters.simplerenv.episode import build_simplerenv_episode
        return build_simplerenv_episode
    if suite.startswith("capgym_"):
        from captain_harness.adapters.capgym.episode import build_capgym_episode
        return build_capgym_episode
    if suite == "hardware_real":
        from captain_harness.adapters.hardware.episode import build_hardware_episode
        return build_hardware_episode
    return None


RECEIPT_LIMIT = 300
MEASURED_KEYS = ("stopped_reason", "stop_reason", "endpoint_reached", "executed_distance_m", "actual_displacement_m",
                 "max_observed_force_n", "object_retention")
PERCEPTION_KEYS = ("detection_id", "label", "grounding_status", "position_m", "point_centroid_m", "center_m", "centroid_m",
                   "object_to_destination_xy_offset_m", "yaw_rad", "segmentation_score", "quality_flags")


def _short(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (int, str)):
        return str(value)[:60]
    if isinstance(value, (list, tuple)) and len(value) <= 4 and all(isinstance(v, (int, float, str)) and not isinstance(v, bool) for v in value):
        return "[" + ", ".join(_short(v) or "" for v in value) + "]"
    return None


def call_receipt(name: str, payload: Mapping[str, Any], consumes_action: bool) -> str:
    """One line per capability call: outcome, message and a compact measured summary (public values only).

    The physical summary keeps the fields of the v4 harness's measured_summary (stop reason, displacement,
    force, jaw opening before/after, tracking)."""
    ok = bool(payload.get("success"))
    head = f"{name}: ok" if ok else f"{name}: FAILED {payload.get('failure_category') or ''}".rstrip()
    if not ok and consumes_action:
        head += " (executed)" if payload.get("executed") else " (not executed)"
    parts = [head]
    message = " ".join(str(payload.get("message") or "").split())
    if message:
        parts.append(message[:160])
    value = payload.get("value") if isinstance(payload.get("value"), Mapping) else {}
    fields = []
    if consumes_action:
        for key in MEASURED_KEYS:
            if key in value and _short(value[key]) is not None:
                fields.append(f"{key}={_short(value[key])}")
        before = (value.get("robot_before") or {}).get("gripper_open_fraction") if isinstance(value.get("robot_before"), Mapping) else None
        after = (value.get("robot_after") or {}).get("gripper_open_fraction") if isinstance(value.get("robot_after"), Mapping) else None
        if before is not None or after is not None:
            fields.append(f"gripper_open_fraction={_short(before)}->{_short(after)}")
        tracking = value.get("tracking_feedback") if isinstance(value.get("tracking_feedback"), Mapping) else {}
        for key in ("stop_reason", "blocked_directions", "remaining_displacement_m"):
            if key in tracking and _short(tracking[key]) is not None:
                fields.append(f"tracking_{key}={_short(tracking[key])}")
    elif name == "observe":
        images = [Path(str(i.get("path"))).name for i in value.get("images") or [] if isinstance(i, Mapping)]
        if images:
            fields.append("images=" + ",".join(images))
    else:
        for key in PERCEPTION_KEYS:
            if key in value and _short(value[key]) is not None:
                fields.append(f"{key}={_short(value[key])}")
            if len(fields) >= 6:
                break
    if fields:
        parts.append(" ".join(fields))
    text = "; ".join(parts)
    return text if len(text) <= RECEIPT_LIMIT else text[: RECEIPT_LIMIT - 3] + "..."


@dataclass
class BuiltEpisode:
    env: Any
    adapter: Any
    task: str
    initial: Any  # EnvironmentObservation


Builder = Callable[[argparse.Namespace, Path, Callable[[str], None]], BuiltEpisode]


def build_libero_episode(args: argparse.Namespace, output_dir: Path, log: Callable[[str], None]) -> BuiltEpisode:
    """Construct env + adapter exactly like the v7 harness did (S3, coding control, gate on)."""
    root = args.harness_root.resolve()
    os.environ["MUJOCO_GL"] = args.mujoco_gl
    os.environ["PYOPENGL_PLATFORM"] = args.mujoco_gl
    if args.mujoco_gl == "egl" and args.egl_device is not None:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.egl_device)
    from captain_harness.server.libero_setup import configure_libero
    from captain_harness.server import libero_variants

    variant = getattr(args, "libero_variant", "libero") or "libero"
    if variant == "libero":
        if args.suite not in ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"):
            raise ValueError(f"unknown LIBERO suite {args.suite!r} (use --libero-variant pro|plus for fork suites)")
        configure_libero(root, args.libero_config_dir)
    else:
        checkout = libero_variants.variant_checkout(variant, args.libero_checkout)
        configure_libero(root, args.libero_config_dir or root / f".libero-config-{variant}", libero_checkout=checkout)

    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.camera_utils import get_real_depth_map

    from captain_harness.adapters.capx import (
        CapXHttpToolServices,
        CapXLiberoEnvironmentAdapter,
        CapXServiceEndpoints,
        ToolApiLevel,
    )
    from captain_harness.core.experience import ExperienceKey, ExperienceSnapshot
    from captain_harness.adapters.libero.fresh_init import (
        DEFAULT_FRESH_DIR, fresh_init_path, is_fresh_seed, load_fresh_init_states, official_dimension,
    )

    dependencies = json.loads((root / "third_party" / "manifest.json").read_text(encoding="utf-8"))["dependencies"]
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    if variant == "libero":
        init_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        try:
            init_states = torch.load(init_path, weights_only=False)
        except TypeError:
            init_states = torch.load(init_path)
    else:
        # the forks map perturbed task names to init files inside the suite (plus: original file, index 0 protocol)
        init_states = libero_variants.load_init_states(suite, args.task_id)
    init_state_override = getattr(args, "init_state_index", None)
    if init_state_override is not None:
        if not 0 <= init_state_override < len(init_states):
            raise ValueError(f"--init-state-index {init_state_override} out of range for {len(init_states)} init states")
        # a one-element list makes the adapter's seed %% n pick exactly this state
        init_states = [init_states[init_state_override]]
    init_state_file = getattr(args, "init_state_file", None)
    init_state_file_sha = None
    if init_state_file is not None:
        import hashlib
        import numpy as np
        if init_state_override is not None:
            raise ValueError("--init-state-file and --init-state-index are exclusive")
        raw = Path(init_state_file).read_bytes()
        init_state_file_sha = hashlib.sha256(raw).hexdigest()
        want = getattr(args, "init_state_sha256", None)
        if want and want != init_state_file_sha:
            raise ValueError(f"{init_state_file}: sha256 {init_state_file_sha} != expected {want}")
        frozen = np.load(init_state_file, allow_pickle=False).astype(np.float64).reshape(-1)
        dim = int(np.asarray(init_states[0]).reshape(-1).shape[0])
        if frozen.shape[0] != dim:
            raise ValueError(f"{init_state_file}: state dimension {frozen.shape[0]} != task init file {dim}")
        init_states = [frozen]  # one-element list: the adapter's seed %% 1 picks exactly this state
        log(f"frozen init state file {init_state_file} sha256 {init_state_file_sha[:12]}")
    fresh_init_states = None
    if is_fresh_seed(args.seed):
        if variant != "libero" or init_state_override is not None:
            raise ValueError("fresh-init seeds (>= 1000) are only defined for the original LIBERO suites")
        # reserved training seeds: fresh states generated by the LIBERO placement samplers, never an official state
        fresh_dir = getattr(args, "fresh_init_dir", None) or (root / DEFAULT_FRESH_DIR)
        fresh_init_states = load_fresh_init_states(fresh_init_path(fresh_dir, task.problem_folder, task.name),
                                                   expected_dim=official_dimension(init_states))
        log(f"fresh init states: {fresh_init_states.path} ({len(fresh_init_states)} states, sha256 {fresh_init_states.sha256[:12]})")
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        controller=args.controller,
        horizon=args.horizon,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=512,
        camera_widths=512,
        camera_depths=True,
        control_freq=20,
    )
    experience_key = ExperienceKey(
        environment_id="libero-v1",
        environment_revision=str(dependencies["libero"]["revision"]),
        tool_revisions={k: str(dependencies[k]["revision"]) for k in ("cap_x", "openeta")},
        implementation_revision=args.experience_implementation_revision,
    )
    services = CapXHttpToolServices(
        CapXServiceEndpoints(
            sam3=args.sam3_url,
            molmo=args.molmo_url,
            openeta_revision=str(dependencies["openeta"]["revision"]),
        ),
        attempts=1,
        timeout_s=args.tool_service_timeout,
    )
    max_info = None
    if getattr(args, "max_case", None) is not None:
        # inner-most patch: plus robot-init / sensor-noise wrap around it (installed below)
        from captain_harness.server.libero_max_events import install_max_event
        max_instruction = " ".join(str(args.instruction_text).split()) if getattr(args, "instruction_text", None) else task.language
        max_info = install_max_event(env, json.loads(Path(args.max_case).read_text(encoding="utf-8")), args.max_arm,
                                     max_instruction, output_dir, log)
    robot_init_mode = getattr(args, "robot_init_override", "auto")
    robot_init_qpos = None
    if robot_init_mode == "on" or (robot_init_mode == "auto" and variant == "plus"):
        robot_init_qpos = libero_variants.install_robot_init_override(env, log)
    # after the robot override, so the frames it returns are corrupted too
    sensor_noise = libero_variants.install_lazy_sensor_noise(env, log) if variant == "plus" else None
    instruction_source = getattr(args, "instruction_source", "auto")
    if instruction_source == "auto":
        instruction_source = "filename" if variant == "libero" else "bddl"
    instruction = task.language
    if instruction_source == "bddl":
        instruction = libero_variants.bddl_language(env) or task.language
    if getattr(args, "instruction_text", None):
        instruction = " ".join(str(args.instruction_text).split())
    goal_state = list((getattr(getattr(env, "env", env), "parsed_problem", None) or {}).get("goal_state") or [])
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
        completion_goals=goal_state if args.completion_gate else None,
        init_states=init_states,
        fresh_init_states=fresh_init_states,
        observation_dir=output_dir / "observations",
        horizon=args.horizon,
        final_predicate=env.check_success,
        backend_options={
            "depth_converter": lambda value: get_real_depth_map(env.sim, value),
            "motion_max_steps": args.motion_max_steps,
            "cartesian_max_steps": args.motion_max_steps,
            "controller": args.controller,
            "contact_detector": args.contact_detector,
            "contact_calibration": bool(args.contact_calibration),
        },
        calibration_profile=root / args.calibration_profile,
    )
    initial = adapter.reset(args.seed)
    log(f"episode ready suite={args.suite} task_id={args.task_id} seed={args.seed} "
        f"init_state_index={initial.public_state.get('init_state_index')} goals={len(goal_state)} task={instruction!r}")
    if variant != "libero" or instruction != task.language or init_state_override is not None or init_state_file is not None:
        variant_info = {"libero_variant": variant, "task_name": task.name, "bddl_file": str(bddl),
                        "filename_language": task.language, "instruction": instruction,
                        "instruction_source": "text" if getattr(args, "instruction_text", None) else instruction_source,
                        "init_state_index_override": init_state_override, "robot_init_qpos_override": robot_init_qpos,
                        "init_state_file": str(init_state_file) if init_state_file is not None else None,
                        "init_state_file_sha256": init_state_file_sha,
                        "sensor_noise": sensor_noise, "robots": [type(r.robot_model).__name__ for r in env.robots],
                        "libero_max": max_info,
                        "goal_state": goal_state}
        (output_dir / "libero_variant.json").write_text(json.dumps(variant_info, indent=1, default=str), encoding="utf-8")
        log("libero variant: " + json.dumps({k: v for k, v in variant_info.items() if k != "goal_state"}, default=str))
    return BuiltEpisode(env=env, adapter=adapter, task=instruction, initial=initial)


class EpisodeServerState:
    """Everything mutable, guarded by one lock (the simulator is single-threaded)."""

    def __init__(self, args: argparse.Namespace, *, builder: Builder | None = None) -> None:
        self.args = args
        self.builder: Builder = builder or build_libero_episode
        self.lock = threading.RLock()
        # MuJoCo/EGL and the adapter's SQLite stores are thread-bound: every simulator
        # touch (build, call, finish, close) runs on this single worker thread.
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sim")
        self.output_dir: Path = args.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir: Path = args.images_dir or (self.output_dir / "workspace" / "images")
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "server.log"
        self.calls_path = self.output_dir / "calls.jsonl"
        self.audit_path = self.output_dir / "hidden_success_audit.jsonl"
        self.finish_attempts_path = self.output_dir / "finish_attempts.jsonl"
        self.started_monotonic = time.monotonic()
        self.started_wall = time.time()
        self.deadline_monotonic = self.started_monotonic + float(args.wall_time)
        self.action_requests = 0
        self.tool_requests = 0
        self.executed_actions = 0
        self.failed_actions = 0
        self.call_index = 0
        self.observation_exports = 0
        self.finished = False
        self.finish_note: str | None = None
        self.finish_source: str | None = None
        self.result: dict[str, Any] | None = None
        self.public_finish_response: dict[str, Any] | None = None
        self.gate_rejections = 0
        self.expired = False
        self.last_request_monotonic = time.monotonic()
        self.hidden_audit_error: str | None = None
        self.hidden_audit_rows: list[dict[str, Any]] = []
        # Private goal telemetry stays in memory until finish, outside capability responses.
        self.object_audit_path = self.output_dir / "hidden_object_audit.jsonl"
        self.hidden_object_rows: list[dict[str, Any]] = []
        self.hidden_object_error: str | None = None
        self.goal_entities: dict[str, Any] | None = None
        self._hidden_object_audit = None
        # Host-only boundary marks (POST /mark from the driver, before each request's tools run): the goal state at the
        # start of every model turn, for per-turn (segment) scoring. Authenticated by a token that only server.json holds.
        self.driver_token = secrets.token_hex(16)
        self.marks_path = self.output_dir / "hidden_request_marks.jsonl"
        self.request_marks: list[dict[str, Any]] = []
        self.live_marks = (self.output_dir / ".live_request_marks").is_file()      # RL early-stop monitor; absent = default behaviour
        # Host-only interface audit (training-side): every /call attempt, including early returns, with the structured
        # conclusions of the validators that decided it (ports/interface_audit.py). Bound to the driver request whose
        # authenticated mark opened the current turn; never part of any capability response.
        self.interface_audit_path = self.output_dir / "interface_audit.jsonl"
        self.attempt_seq = 0
        # Public per-call records served by GET /history (what the agent did and what it got back; no hidden state).
        self.public_history: list[dict[str, Any]] = []
        # Clean handoff (POST /handoff from the driver): calls up to history_floor belong to the earlier actor (a replayed
        # teacher prefix) and are no longer served by /history; the physical world is left exactly as it is.
        self.history_floor = 0
        self.handoffs: list[dict[str, Any]] = []
        self.env = None
        self.adapter = None
        self.manifest = None
        self.task_language: str | None = None
        self.init_state_index: Any = None
        self.initial_images: list[dict[str, str]] = []

    # ------------------------------------------------------------------ setup
    def log(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}\n"
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(line)

    def _build_impl(self) -> None:
        built = self.builder(self.args, self.output_dir, self.log)
        self.env = built.env
        self.adapter = built.adapter
        self.task_language = built.task
        self.manifest = self.adapter.capabilities()
        self.init_state_index = dict(built.initial.public_state or {}).get("init_state_index")
        self.initial_images = self._export_images(built.initial.images)
        if self.args.audit_hidden_success:
            self._record_hidden_objects(call_index=0, capability="reset")

    # ------------------------------------------------------ hidden object audit
    def _goal_entities_impl(self) -> dict[str, Any]:
        return self._object_audit_reader().header()

    def _object_audit_reader(self):
        if self._hidden_object_audit is None:
            from captain_harness.adapters.libero.hidden_object_audit import HiddenObjectAudit
            factory = getattr(self.env, "hidden_object_audit", None)  # non-LIBERO backends: object poses from their simulator
            audit = factory() if callable(factory) else None
            self._hidden_object_audit = audit if audit is not None else HiddenObjectAudit(self.env)
        return self._hidden_object_audit

    def _hidden_object_state(self) -> dict[str, Any]:
        return self._object_audit_reader().sample()

    def _record_hidden_objects(self, *, call_index: int, capability: str) -> None:
        if self.env is None:
            return
        try:
            if self.goal_entities is None:
                self.goal_entities = self._goal_entities_impl()
            row: dict[str, Any] = {"call_index": call_index, "action_index": self.action_requests, "capability": capability,
                                   "environment_step": self._environment_step(), **self._hidden_object_state()}
            if capability == "reset":
                row.update(self.goal_entities)
            self.hidden_object_rows.append(row)
        except Exception as exc:  # hidden bookkeeping must never fail the episode
            self.hidden_object_error = f"{type(exc).__name__}: {exc}"
            self.hidden_object_rows.append({"call_index": call_index, "action_index": self.action_requests, "capability": capability,
                                            "error": self.hidden_object_error})

    def mark(self, request_index: int) -> dict[str, Any]:
        return self.worker.submit(self._mark_impl, request_index).result()

    def _mark_impl(self, request_index: int) -> dict[str, Any]:
        with self.lock:
            if self.env is None or self.finished:
                return {"ok": False, "reason": "finished" if self.finished else "not_ready"}
            row: dict[str, Any] = {"schema_version": "hidden-request-mark/v1", "request_index": int(request_index),
                                   "call_index": self.call_index, "action_index": self.action_requests,
                                   "environment_step": self._environment_step(),
                                   "t_s": round(time.monotonic() - self.started_monotonic, 3)}
            try:
                if self.goal_entities is None:
                    self.goal_entities = self._goal_entities_impl()
                sample = self._hidden_object_state()
                row.update(sample, schema_version="hidden-request-mark/v1", state_schema=sample.get("schema_version"))
            except Exception as exc:  # hidden bookkeeping must never fail the episode
                row["error"] = f"{type(exc).__name__}: {exc}"
            self.request_marks.append(row)
            if self.live_marks:
                # RL early-stop monitor (host side only): append the mark as it is taken. The flag file
                # lives in the episode output dir, which the model's sandbox cannot see; finish rewrites the file with every row.
                try:
                    with self.marks_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")
                except Exception:  # hidden bookkeeping must never fail the episode
                    pass
            return {"ok": True}

    # ---------------------------------------------------------------- helpers
    def _export_images(self, images) -> list[dict[str, str]]:
        index = self.observation_exports
        self.observation_exports += 1
        exported = []
        for image in images:
            role = image.role
            destination = self.images_dir / f"obs_{index:03d}_{role}.png"
            shutil.copyfile(image.rgb_path, destination)
            exported.append({"camera": image.camera, "role": role, "path": str(destination.resolve())})
        return exported

    def _spec(self, name: str):
        return self.manifest.require(name)

    def history(self, start: int | None, end: int | None, limit: int) -> dict[str, Any]:
        """Objective action/result records by call index (the `history` tool)."""
        visible = [r for r in self.public_history if r["call_index"] > self.history_floor]
        rows = [r for r in visible if (start is None or r["call_index"] >= start) and (end is None or r["call_index"] <= end)]
        limit = max(1, min(int(limit or 64), 256))
        rows = rows[-limit:]
        return {"total_calls": len(visible), "returned": len(rows), "records": rows, "budget": self.counters()}

    def handoff(self) -> dict[str, Any]:
        return self.worker.submit(self._handoff_impl).result()

    def _handoff_impl(self) -> dict[str, Any]:
        """Hand the episode to a new actor: export the current camera frames (a render, not a robot call: no call
        index, action or tool budget is used and the simulation does not step) and hide the earlier calls from /history."""
        with self.lock:
            if self.env is None or self.finished:
                return {"ok": False, "reason": "finished" if self.finished else "not_ready"}
            images = self._export_images(self.adapter.observe().images)
            self.history_floor = self.call_index
            row = {"call_index": self.call_index, "action_index": self.action_requests, "environment_step": self._environment_step(),
                   "t_s": round(time.monotonic() - self.started_monotonic, 3), "images": [Path(i["path"]).name for i in images]}
            self.handoffs.append(row)
            return {"ok": True, "images": images, "call_index": self.call_index, "budget": self.counters()}

    def public_manifest(self) -> dict[str, Any]:
        return {
            "task": self.task_language,
            "suite": self.args.suite,
            "task_id": self.args.task_id,
            "seed": self.args.seed,
            "budgets": {
                "max_physical_actions": self.args.max_actions,
                "max_tool_calls": self.args.max_tool_calls,
                "wall_time_s": self.args.wall_time,
            },
            "initial_images": self.initial_images,
            "python_interface": self.manifest.render_python_interface(),
            "manifest": _json_safe(self.manifest),
        }

    def remaining_wall_s(self) -> float:
        return max(0.0, self.deadline_monotonic - time.monotonic())

    def counters(self) -> dict[str, Any]:
        return {
            "finished": self.finished,
            "expired": self.expired,
            "physical_actions_used": self.action_requests,
            "physical_actions_remaining": max(0, self.args.max_actions - self.action_requests),
            "tool_calls_used": self.tool_requests,
            "tool_calls_remaining": max(0, self.args.max_tool_calls - self.tool_requests),
            "wall_time_remaining_s": round(self.remaining_wall_s(), 1),
            "elapsed_s": round(time.monotonic() - self.started_monotonic, 1),
            "finish_rejections": self.gate_rejections,
        }

    def _failed(self, category: str, message: str) -> dict[str, Any]:
        return {
            "success": False,
            "executed": False,
            "failure_category": category,
            "message": message,
            "value": None,
            "diagnostics": None,
            "budget": self.counters(),
        }

    # ------------------------------------------------------------------- call
    def call(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return self.worker.submit(self._call_impl, name, arguments).result()

    def finish(self, note: str = "", source: str = "agent") -> dict[str, Any]:
        return self.worker.submit(self._finish_impl, note, source).result()

    def build(self) -> None:
        self.worker.submit(self._build_impl).result()

    def close_env(self) -> None:
        def _close():
            try:
                if self.env is not None:
                    self.env.close()
            except Exception:
                pass
        self.worker.submit(_close).result()

    def _environment_step(self) -> Any:
        try:
            robot = self.adapter._require_robot()  # type: ignore[union-attr]
            return getattr(robot, "simulation_steps", None)
        except Exception:
            return None

    def _attempt_context(self) -> dict[str, Any]:
        self.attempt_seq += 1
        last = self.request_marks[-1] if self.request_marks else None
        return {"schema_version": "interface-audit/v1", "attempt_seq": self.attempt_seq,
                "driver_request_index": last.get("request_index") if last else None,
                "attribution_basis": "latest_authenticated_mark" if last else "no_mark",
                "t_s": round(time.monotonic() - self.started_monotonic, 3)}

    def _write_attempt(self, row: dict[str, Any]) -> None:
        try:
            with self.interface_audit_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True, default=repr) + "\n")
        except Exception:  # training-side bookkeeping must never fail the episode
            self.log(f"interface audit write failed: {traceback.format_exc()}")

    def record_transport_invalid(self, reason: str) -> None:
        """An HTTP /call rejected before the capability layer (undecodable body, non-string tool name)."""
        with self.lock:
            self._write_attempt({**self._attempt_context(), "tool": None, "stage": "transport", "outcome": "rejected",
                                 "early_return": "transport_invalid", "reason": reason[:200], "events": [],
                                 "public_call_index": None, "executed": False})

    def _call_impl(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        from captain_harness.ports import interface_audit

        with self.lock:
            attempt = {**self._attempt_context(), "tool": name}
            with interface_audit.capture() as events:
                payload = self._call_checked(name, arguments, attempt)
            early = attempt.pop("early_return", None)
            self._write_attempt({**attempt, "stage": "early_return" if early else "capability", "early_return": early,
                                 "outcome": "rejected" if early else ("ok" if payload.get("success") else "failed"),
                                 "events": events, "executed": bool(payload.get("executed")),
                                 "failure_category": payload.get("failure_category"),
                                 "public_call_index": attempt.get("public_call_index")})
            return payload

    def _call_checked(self, name: str, arguments: Mapping[str, Any], attempt: dict[str, Any]) -> dict[str, Any]:
        from captain_harness.ports.capability import SideEffect, UnknownCapabilityError
        from captain_harness.ports.deadline import deadline_failure, deadline_scope, retain_deadline_status

        def early(reason: str, category: str, message: str) -> dict[str, Any]:
            attempt["early_return"] = reason
            return self._failed(category, message)

        with self.lock:
            self.last_request_monotonic = time.monotonic()
            if self.finished:
                return early("finished", "not_executed", "episode finished: finish() was already called; no further calls are accepted")
            if time.monotonic() >= self.deadline_monotonic:
                self.expired = True
                return early("wall_time_exhausted", "not_executed", "episode wall time budget exhausted; call finish()")
            try:
                spec = self._spec(name)
            except UnknownCapabilityError as exc:
                return early("unknown_capability", "schema_error", str(exc))
            if not isinstance(arguments, Mapping):
                return early("arguments_not_object", "schema_error", "arguments must be a JSON object")
            consumes_action = spec.side_effect in {SideEffect.ROBOT_MOTION, SideEffect.ACTUATOR}
            if consumes_action:
                if self.action_requests >= self.args.max_actions:
                    return early("action_budget_exhausted", "not_executed", "episode action budget exhausted")
                self.action_requests += 1
            else:
                if self.tool_requests >= self.args.max_tool_calls:
                    return early("tool_budget_exhausted", "not_executed", "episode perception/tool budget exhausted")
                self.tool_requests += 1
            self.call_index += 1
            index = self.call_index
            attempt["public_call_index"] = index
            started = time.monotonic()
            error_text = None
            try:
                with deadline_scope("episode", expires_s=self.deadline_monotonic):
                    with deadline_scope("tool", timeout_s=spec.timeout_s) as deadline:
                        if deadline.remaining() <= 0:
                            result = deadline_failure(deadline)
                        else:
                            result = retain_deadline_status(self.adapter.execute(name, dict(arguments)), deadline)
                payload = _json_safe(result)
            except Exception as exc:  # adapter bug or service outage: report, never crash the episode
                error_text = traceback.format_exc()
                self.log(f"call {index} {name} raised: {error_text}")
                payload = self._failed("backend_error", f"{type(exc).__name__}: {exc}")
            elapsed = time.monotonic() - started
            payload.pop("attempt_identity", None)
            payload.pop("attempt_evidence", None)
            if consumes_action:
                if payload.get("executed"):
                    self.executed_actions += 1
                if not payload.get("success"):
                    self.failed_actions += 1
            if name == "observe" and payload.get("success") and isinstance(payload.get("value"), dict):
                handle = payload["value"].get("observation_handle")
                try:
                    stored = self.adapter.public_evidence_observations([handle]) if handle else {}
                    if handle in stored:
                        payload["value"]["images"] = self._export_images(stored[handle].images)
                except Exception:
                    self.log(f"image export failed for call {index}: {traceback.format_exc()}")
            payload["budget"] = self.counters()
            if getattr(self.args, "receipts", False):
                payload["receipt"] = call_receipt(name, payload, consumes_action)
            audit = None
            if consumes_action and self.args.audit_hidden_success and payload.get("executed"):
                try:
                    audit = bool(self.adapter.audit_hidden_success())
                except Exception as exc:
                    self.hidden_audit_error = f"{type(exc).__name__}: {exc}"
                self.hidden_audit_rows.append({
                    "action_index": self.action_requests, "call_index": index, "capability": name,
                    "environment_step": self._environment_step(),
                    "success": audit, "error": self.hidden_audit_error,
                })
                self._record_hidden_objects(call_index=index, capability=name)
            self.public_history.append({
                "call_index": index, "t_s": round(started - self.started_monotonic, 1), "name": name, "arguments": _json_safe(arguments),
                "consumes_action": consumes_action, "success": payload.get("success"), "executed": payload.get("executed"),
                "failure_category": payload.get("failure_category"), "message": (payload.get("message") or "")[:300],
                "images": [i.get("path") for i in ((payload.get("value") or {}).get("images") or [])] if isinstance(payload.get("value"), dict) else [],
            })
            with self.calls_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "call_index": index, "t_s": round(started - self.started_monotonic, 3),
                    "elapsed_s": round(elapsed, 3), "name": name, "arguments": _json_safe(arguments),
                    "consumes_action": consumes_action, "success": payload.get("success"),
                    "executed": payload.get("executed"), "failure_category": payload.get("failure_category"),
                    "message": payload.get("message"), "value": payload.get("value"),
                    "diagnostics": payload.get("diagnostics"), "error": error_text, "receipt": payload.get("receipt"),
                }, sort_keys=True, default=repr) + "\n")
            return payload

    # ----------------------------------------------------------------- finish
    def _gate_verdict(self) -> Mapping[str, Any] | None:
        gate = getattr(self.adapter, "completion_gate", None)
        if not callable(gate):
            return None
        try:
            verdict = gate()
        except Exception:
            self.log("completion gate raised: " + traceback.format_exc())
            return None
        if isinstance(verdict, Mapping) and verdict.get("message"):
            return verdict
        return None

    def _finish_impl(self, note: str = "", source: str = "agent") -> dict[str, Any]:
        with self.lock:
            self.last_request_monotonic = time.monotonic()
            if self.finished and self.public_finish_response is not None:
                return {**self.public_finish_response, "already_finished": True}
            if source == "agent" and self.args.completion_gate:
                verdict = self._gate_verdict()
                if verdict is not None:
                    self.gate_rejections += 1
                    response = {
                        "finished": False,
                        "rejected": True,
                        "failure": verdict.get("failure"),
                        "message": str(verdict.get("message")),
                        "rejections": self.gate_rejections,
                        "max_rejections": verdict.get("max_rejections"),
                        "budget": self.counters(),
                    }
                    with self.finish_attempts_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"t_s": round(time.monotonic() - self.started_monotonic, 1), "note": str(note)[:2000],
                                                 "verdict": "reject", **{k: response[k] for k in ("failure", "message", "rejections")}},
                                                sort_keys=True) + "\n")
                    self.log(f"finish rejected by completion gate ({self.gate_rejections}): {response['message'][:200]}")
                    return response
            self.finished = True
            self.finish_note = str(note)[:2000]
            self.finish_source = source
            success_ever = None
            first_success_step = None
            final_success = None
            error = None
            try:
                if self.args.audit_hidden_success:
                    self._record_hidden_objects(call_index=self.call_index, capability="finish")
                completion = self.adapter.finalize()
                final_success = completion.final_success
                success_ever = completion.success_ever
                first_success_step = completion.first_success_step
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                self.log(f"finalize raised: {traceback.format_exc()}")
            wall = time.monotonic() - self.started_monotonic
            self.result = {
                "suite": self.args.suite, "task_id": self.args.task_id, "seed": self.args.seed,
                "task": self.task_language,
                "init_state_index": self.init_state_index,
                "init_state_source": "fresh" if self.args.seed >= 1000 else "official",
                "host_final_success": final_success,
                "host_success_ever": success_ever,
                "first_success_step": first_success_step,
                "action_requests": self.action_requests,
                "executed_actions": self.executed_actions,
                "failed_actions": self.failed_actions,
                "tool_requests": self.tool_requests,
                "observation_exports": self.observation_exports,
                "handoffs": list(self.handoffs),
                "gate_rejections": self.gate_rejections,
                "completion_gate": bool(self.args.completion_gate),
                "wall_time_s": round(wall, 1),
                "wall_time_expired": self.expired or wall >= self.args.wall_time,
                "finish_source": source,
                "finish_note": self.finish_note,
                "finalize_error": error,
                "started_wall_time": self.started_wall,
                "finished_wall_time": time.time(),
            }
            (self.output_dir / "result.json").write_text(json.dumps(self.result, indent=2, sort_keys=True), encoding="utf-8")
            with self.audit_path.open("w", encoding="utf-8") as stream:
                for row in self.hidden_audit_rows:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
            if self.hidden_object_rows:
                with self.object_audit_path.open("w", encoding="utf-8") as stream:
                    for row in self.hidden_object_rows:
                        stream.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")
            if self.request_marks:
                with self.marks_path.open("w", encoding="utf-8") as stream:
                    for row in self.request_marks:
                        stream.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")
            with self.finish_attempts_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"t_s": round(wall, 1), "note": self.finish_note, "verdict": "accepted", "source": source}, sort_keys=True) + "\n")
            # Disk hygiene: the handle object store (~180 MB/episode of RGB-D arrays) and depth .npy
            # files are only needed while the episode is live; PNGs and JSON records stay.
            try:
                shutil.rmtree(self.output_dir / "public_sensor_artifacts", ignore_errors=True)
                for depth in (self.output_dir / "observations").glob("*_depth.npy"):
                    depth.unlink()
            except Exception:
                self.log("cleanup failed: " + traceback.format_exc())
            self.log(f"finished source={source} host_final_success={final_success} actions={self.action_requests} tools={self.tool_requests} wall={wall:.0f}s")
            # The agent never sees the predicate value.
            self.public_finish_response = {
                "finished": True, "message": "episode finished and scored; no further calls are accepted",
                "physical_actions_used": self.action_requests, "tool_calls_used": self.tool_requests,
                "wall_time_s": round(wall, 1),
            }
            return dict(self.public_finish_response)


class EpisodeHandler(BaseHTTPRequestHandler):
    state: EpisodeServerState = None  # type: ignore[assignment]
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):  # quiet
        return

    def _send(self, code: int, payload: Any) -> None:
        body = json.dumps(payload, default=repr).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def do_GET(self):
        state = self.state
        if self.path.startswith("/health"):
            self._send(200, {"ok": True, **state.counters()})
        elif self.path.startswith("/manifest"):
            self._send(200, state.public_manifest())
        elif self.path.startswith("/history"):
            from urllib.parse import parse_qs, urlsplit
            query = parse_qs(urlsplit(self.path).query)
            def _int(key):
                try:
                    return int(query[key][0]) if key in query else None
                except ValueError:
                    return None
            self._send(200, state.history(_int("start"), _int("end"), _int("limit") or 64))
        else:
            self._send(404, {"error": "unknown route"})

    def do_POST(self):
        state = self.state
        try:
            body = self._read_json()
        except Exception as exc:
            if self.path.startswith("/call"):
                state.record_transport_invalid(f"invalid JSON body: {type(exc).__name__}")
            self._send(400, {"error": f"invalid JSON body: {exc}"})
            return
        if self.path.startswith("/call"):
            name = body.get("name") if isinstance(body, dict) else None
            arguments = body.get("arguments", {}) if isinstance(body, dict) else None
            if not isinstance(name, str):
                state.record_transport_invalid("tool name is not a string")
                self._send(400, {"error": "body must be {\"name\": str, \"arguments\": object}"})
                return
            self._send(200, state.call(name, arguments if arguments is not None else {}))
        elif self.path.startswith("/finish"):
            note = body.get("note", "") if isinstance(body, dict) else ""
            source = body.get("source", "agent") if isinstance(body, dict) else "agent"
            if source not in FINISH_SOURCES:
                source = "agent"
            self._send(200, state.finish(note, source=source))
        elif self.path.startswith("/handoff"):
            if not isinstance(body, dict) or not secrets.compare_digest(str(body.get("token") or ""), state.driver_token):
                self._send(403, {"error": "forbidden"})
                return
            self._send(200, state.handoff())
        elif self.path.startswith("/mark"):
            if not isinstance(body, dict) or not secrets.compare_digest(str(body.get("token") or ""), state.driver_token):
                self._send(403, {"error": "forbidden"})
                return
            try:
                index = int(body.get("request_index"))
            except (TypeError, ValueError):
                self._send(400, {"error": "request_index must be an integer"})
                return
            self._send(200, state.mark(index))
        elif self.path.startswith("/shutdown"):
            if not state.finished:
                self._send(409, {"error": "call /finish first"})
                return
            self._send(200, {"ok": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        else:
            self._send(404, {"error": "unknown route"})


def serve(state: EpisodeServerState, host: str, port: int, *, idle_exit_s: float) -> None:
    """Bind, write server.json, serve until /shutdown or idle-after-finish; finish + close on exit."""
    handler = type("BoundEpisodeHandler", (EpisodeHandler,), {"state": state})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    bound_host, bound_port = server.server_address[:2]
    info = {
        "host": bound_host, "port": bound_port, "url": f"http://{bound_host}:{bound_port}", "pid": os.getpid(),
        "task": state.task_language, "suite": state.args.suite, "task_id": state.args.task_id, "seed": state.args.seed,
        "init_state_index": state.init_state_index,
        "init_state_source": "fresh" if state.args.seed >= 1000 else "official",
        "initial_images": state.initial_images, "images_dir": str(state.images_dir.resolve()),
        "budgets": state.public_manifest()["budgets"], "started_wall_time": state.started_wall,
        "driver_token": state.driver_token,  # server.json lives in the hidden episode directory
    }
    overrides = getattr(state.adapter, "task_markdown_overrides", None)
    if overrides:
        info["task_markdown_overrides"] = dict(overrides)
    (state.output_dir / "server.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    state.log(f"listening on {info['url']}")

    def watchdog():
        while True:
            time.sleep(5.0)
            if state.finished and time.monotonic() - state.last_request_monotonic > idle_exit_s:
                state.log("idle after finish; shutting down")
                server.shutdown()
                return

    threading.Thread(target=watchdog, daemon=True).start()
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        try:
            if not state.finished:
                state.finish("server shutdown before finish", source="driver_exit")
        finally:
            state.close_env()
            state.worker.shutdown(wait=False)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state = EpisodeServerState(args, builder=episode_builder(args.suite))
    state.log(f"building episode: {vars(args)}")
    try:
        state.build()
    except Exception:
        state.log("build failed: " + traceback.format_exc())
        (args.output_dir / "server_error.json").write_text(json.dumps({"error": traceback.format_exc()}), encoding="utf-8")
        raise
    try:
        from captain_harness.server.provenance import capture_source_provenance
        capture_source_provenance(args.harness_root, output_dir=args.output_dir)
    except Exception:
        state.log("provenance capture failed: " + traceback.format_exc())
    serve(state, args.host, args.port, idle_exit_s=args.idle_exit_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())

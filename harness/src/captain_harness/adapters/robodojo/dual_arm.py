"""RoboDojo copy of the dual-arm CaP-X tool adapter (the robotwin2.environment revision that RoboDojo harness r9 ran with;
the RoboTwin 2.0 runs use the later revision in adapters/robotwin2/environment.py). CaP-X tool adapter over the RoboTwin dual-arm facades: the XR1/LIBERO manifest plus one optional `arm` argument.

The primary arm is exactly the single-arm XR1 stack (XR1ActionBackend over an XR1FacadeEnv, one
ToolCapabilityBackend, DirectProgramming). Every other arm gets its own XR1ActionBackend over its own
facade. A call with arm=<other> runs the unchanged DirectProgramming code with that arm's backend
swapped in (tools.robot / direct.robot and the per-arm attempt bookkeeping); perception, detections and
measurement records stay shared. All arms share one simulation clock: after any call the backends that
did not act re-read the measured state, so every receipt and freshness check sees the same scene time.
Without `arm` nothing is swapped: the behaviour is the single-arm one.
"""
from __future__ import annotations

import contextlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from captain_harness.adapters.hardware.backend import XR1ActionBackend
from captain_harness.adapters.hardware.environment import CapXXR1EnvironmentAdapter
from captain_harness.ports.capability import CapabilityManifest

# tools that act on or read one arm
ARM_TOOLS = ("move_to", "hold", "recover", "release", "gripper_state", "get_robot_state", "preview_motion",
             "check_motion", "robot_geometry")
# DirectProgramming bookkeeping that belongs to one arm's grasp attempts / failures
DIRECT_ARM_FIELDS = ("attempt_id", "target_id", "_attempt_active", "_unresolved_failure_id", "_recent_motion_timing")
TOOLS_ARM_FIELDS = ("_aligned_pregrasp",)


def _same_scene_any_arm(original):
    """Snapshot equivalence for calls on a non-primary arm: a detection taken in a primary-arm snapshot carries the
    primary arm's robot_state, the other arm's snapshot its own. Same shared-clock time and identical camera data
    (RGB, depth, intrinsics, extrinsics) means nothing moved in between, whichever arm's state the snapshot carries."""
    def same(first, second):
        if original(first, second):
            return True
        if first.time_s is None or second.time_s is None or first.time_s != second.time_s:
            return False
        a = {c.camera_name: c for c in first.cameras}
        b = {c.camera_name: c for c in second.cameras}
        if not a or a.keys() != b.keys():
            return False
        import numpy as np
        return all(np.array_equal(getattr(a[n], f), getattr(b[n], f)) for n in a
                   for f in ("image", "depth", "intrinsics", "extrinsics"))
    return same


def add_arm_argument(manifest: CapabilityManifest, arms: tuple[str, ...], primary: str) -> CapabilityManifest:
    """Same manifest; ARM_TOOLS gain an optional trailing `arm` argument (default = primary arm)."""
    note = (f" Optional arm: one of {', '.join(repr(a) for a in arms)}; default {primary!r} (the primary arm). "
            "Omit it for the primary arm; bimanual work is a sequence of single-arm calls.")
    caps = []
    for spec in manifest.capabilities:
        if spec.name in ARM_TOOLS:
            schema = dict(spec.arguments_schema)
            schema["properties"] = {**schema.get("properties", {}), "arm": {"enum": list(arms), "default": primary}}
            spec = replace(spec, description=spec.description + note, arguments_schema=schema,
                           argument_order=(*spec.argument_order, "arm"))
        caps.append(spec)
    return CapabilityManifest(manifest.schema_version, manifest.provider, tuple(caps), manifest.usage_instructions)


MOVE_BOTH_DESCRIPTION = (
    "Move BOTH arms at the same time (two-handed lifts, handovers, holding while the other arm acts). left and right "
    "each take exactly the arguments of one move_to without `arm`: an absolute EEF pose (position, quaternion, optional "
    "speed/tolerance/contact fields) OR one gripper command ('open'/'close'). The arm with the longer motion executes "
    "with all move_to checks; the other arm follows its own straight Cartesian path (shortest rotation) in the same "
    "control ticks at its own speed limits and stops if its wrist force rises by 5 N or reaches 18 N; if it has not "
    "arrived when the first arm finishes, a normal move_to completes it. A gripper command on one side moves that jaw "
    "while the other arm moves. Result value: {'leader': arm, 'left': ..., 'right': ...} with each arm's receipt "
    "(the following arm: reached, stopped_reason, eef_after, peak_force_n, completion). Single-arm work stays on move_to.")


def move_both_spec(move_to_spec):
    """move_both: two move_to argument objects (left, right); generic, no task logic."""
    side = json.loads(json.dumps(move_to_spec.arguments_schema))
    side["properties"].pop("arm", None)
    schema = {"type": "object", "additionalProperties": False, "required": ["left", "right"],
              "properties": {"left": side, "right": side}}
    return replace(move_to_spec, name="move_both", description=MOVE_BOTH_DESCRIPTION, arguments_schema=schema,
                   argument_order=("left", "right"), result_schema={"type": "object", "additionalProperties": True},
                   examples=())


class CapXRoboTwinEnvironmentAdapter(CapXXR1EnvironmentAdapter):
    def __init__(self, envs: Mapping[str, Any], *, primary: str, **kwargs) -> None:
        super().__init__(envs[primary], **kwargs)
        self._envs = dict(envs)
        self._primary = primary
        self._arms = (primary, *[a for a in envs if a != primary])
        self._robots: dict[str, XR1ActionBackend] = {}
        self._arm_state: dict[str, dict] = {}

    def reset(self, seed: int):
        observation = super().reset(seed)
        self._robots = {self._primary: self._robot}
        options = dict(self._backend_options)
        detector = str(options.pop("contact_detector", "legacy"))
        calibrate = bool(options.pop("contact_calibration", False))
        options.pop("depth_converter", None)
        for arm, env in self._envs.items():
            if arm == self._primary:
                continue
            obs_dir = Path(self._observation_dir) / f"arm_{arm}"
            obs_dir.mkdir(parents=True, exist_ok=True)
            robot = XR1ActionBackend(env, observation_dir=obs_dir, initial_observation=env.current_observation(),
                                     horizon=self._horizon, calibration_profile=self._calibration_profile,
                                     camera_link_to_optical=self._camera_link_to_optical,
                                     guarded_cartesian_motion=self._motion_tools, **options)
            robot._contact_detector_mode = detector
            if detector == "residual" and calibrate:  # same per-episode free-space calibration as the primary arm
                calibration, receipt = robot.calibrate_contact_residual()
                (obs_dir / "force_calibration.json").write_text(json.dumps({"calibration": calibration.to_json(), **receipt}, indent=2))
            self._robots[arm] = robot
        self._arm_state = {arm: {} for arm in self._arms}
        if len(self._arms) > 1:  # one controlled arm: the unchanged single-arm (LIBERO) manifest
            self._manifest = add_arm_argument(self._manifest, self._arms, self._primary)
            if getattr(self, "_bimanual_tool", False):
                caps = list(self._manifest.capabilities)
                i = [c.name for c in caps].index("move_to")
                caps.insert(i + 1, move_both_spec(caps[i]))
                self._manifest = CapabilityManifest(self._manifest.schema_version, self._manifest.provider, tuple(caps),
                                                    self._manifest.usage_instructions)
        return observation

    # ------------------------------------------------------------------ arm routing
    def _resync(self, arm: str, steps: int) -> None:
        """Re-read one arm's measured state after the scene advanced under another arm."""
        env, robot = self._envs[arm], self._robots[arm]
        env.state = env._driver.read()
        env._sync(env.state)
        robot._observation = env.current_observation()
        robot._sim_steps = steps

    @contextlib.contextmanager
    def _arm_selected(self, arm: str):
        direct, tools = self._direct_programming, self._tools
        primary_robot = self._robots[self._primary]
        saved = {f: getattr(direct, f) for f in DIRECT_ARM_FIELDS}
        saved_tools = {f: getattr(tools, f, None) for f in TOOLS_ARM_FIELDS}
        own = self._arm_state.get(arm) or {}
        for f in DIRECT_ARM_FIELDS:
            setattr(direct, f, own.get(f, [] if f == "_recent_motion_timing" else False if f == "_attempt_active" else None))
        for f in TOOLS_ARM_FIELDS:
            setattr(tools, f, own.get(f))
        robot = self._robots[arm]
        self._resync(arm, primary_robot.simulation_steps)
        direct.robot = tools.robot = robot
        from captain_harness.adapters.capx import observation_equivalence
        original_same = observation_equivalence.same_public_snapshot
        observation_equivalence.same_public_snapshot = _same_scene_any_arm(original_same)
        try:
            yield robot
        finally:
            observation_equivalence.same_public_snapshot = original_same
            self._arm_state[arm] = {**{f: getattr(direct, f) for f in DIRECT_ARM_FIELDS},
                                    **{f: getattr(tools, f, None) for f in TOOLS_ARM_FIELDS}}
            for f, v in saved.items():
                setattr(direct, f, v)
            for f, v in saved_tools.items():
                setattr(tools, f, v)
            direct.robot = tools.robot = primary_robot
            steps = robot.simulation_steps
            for other in self._arms:
                if other != arm:
                    self._resync(other, steps)

    _bimanual_tool = False

    def enable_bimanual_tool(self) -> None:
        self._bimanual_tool = True

    def _move_both(self, arguments: Mapping[str, object]):
        from captain_harness.ports.capability import CapabilityResult, FailureCategory
        from scipy.spatial.transform import Rotation
        from captain_harness.adapters.robotwin2.sim_driver import Companion
        args = {arm: dict(arguments[arm]) for arm in ("left", "right")}
        drivers = {arm: self._envs[arm]._driver for arm in ("left", "right")}
        eef = {arm: d.fk(d._pad(d.sim.state[arm]["q"])) for arm, d in drivers.items()}

        def goal(a):
            if "position" not in a:
                return None
            g = np.eye(4)
            w, x, y, z = (float(v) for v in a["quaternion"])
            g[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
            g[:3, 3] = np.asarray(a["position"], float)
            return g
        goals = {arm: goal(a) for arm, a in args.items()}
        dist = {arm: (np.linalg.norm(goals[arm][:3, 3] - eef[arm][:3, 3]) if goals[arm] is not None else -1.0)
                for arm in args}
        lead = max(args, key=lambda a: (dist[a], a == "right"))
        follow = "left" if lead == "right" else "right"
        fa = args[follow]
        open_cmd = None if "gripper" not in fa else (0.0 if fa["gripper"] == "close" else 1.0)
        comp = Companion(drivers[follow], goals[follow], open_cmd=open_cmd,
                         max_speed_m_s=float(fa.get("max_speed_m_s", 0.05)),
                         max_angular_speed_rad_s=float(fa.get("max_angular_speed_rad_s", 0.5)))
        sim = drivers[lead].sim
        sim.companions = {follow: comp}
        try:
            lead_res = self.execute("move_to", {**args[lead], "arm": lead})
        finally:
            sim.companions = {}
        self._resync(follow, self._robots[lead].simulation_steps)
        follow_eef = drivers[follow].fk(drivers[follow]._pad(sim.state[follow]["q"]))
        summary = {"stopped_reason": comp.stopped_reason or "leader_finished_first",
                   "peak_force_n": round(comp.peak_force, 2), "eef_after": follow_eef[:3, 3].round(4).tolist(),
                   "gripper_open_fraction_after": float(np.clip(np.mean(sim.state[follow]["fingers"]) / drivers[follow]._stroke, 0, 1))}
        follow_ok = True
        if goals[follow] is not None:
            err = float(np.linalg.norm(follow_eef[:3, 3] - goals[follow][:3, 3]))
            tol = float(fa.get("position_tolerance_m", 0.003))
            summary["position_error_m"] = round(err, 4)
            if comp.stopped_reason == "contact_force":
                follow_ok = False
            elif err > tol:  # not there yet: finish with the normal guarded move_to
                done = self.execute("move_to", {**fa, "arm": follow})
                summary["completion"] = {"success": done.success, "message": done.message,
                                         "failure_category": done.failure_category.value}
                follow_ok = done.success
        elif "gripper" in fa:  # a jaw command still in flight: let it finish like move_to(gripper=...)
            done = self.execute("move_to", {"gripper": fa["gripper"], "arm": follow})
            summary["completion"] = {"success": done.success, "message": done.message,
                                     "failure_category": done.failure_category.value}
            follow_ok = done.success
        summary["reached"] = bool(follow_ok)
        value = {"leader": lead, lead: {"success": lead_res.success, "message": lead_res.message,
                                         "receipt": lead_res.value}, follow: summary}
        if lead_res.success and follow_ok:
            return CapabilityResult.ok(value, f"move_both: {lead} led; {follow} {summary['stopped_reason']}")
        category = lead_res.failure_category if not lead_res.success else FailureCategory.CONTROLLER_REJECTED
        return CapabilityResult(False, True, category,
                                f"move_both: {lead} {'ok' if lead_res.success else lead_res.message}; "
                                f"{follow} {summary.get('completion', {}).get('message') or summary['stopped_reason']}", value)

    def execute(self, capability: str, arguments: Mapping[str, object]):
        if capability == "move_both" and self._bimanual_tool:
            return self._move_both(arguments)
        direct = getattr(self, "_direct_programming", None)
        if direct is None:
            return super().execute(capability, arguments)
        invoke = direct.invoke
        adapter = self

        def routed(name, args):
            args = dict(args)
            arm = args.pop("arm", None)
            if arm is None:
                arm = adapter._auto_arm(name, args) if adapter._auto_arm_enabled else adapter._primary
            if arm == adapter._primary:
                return invoke(name, args)
            with adapter._arm_selected(arm):
                return invoke(name, args)

        direct.invoke = routed
        try:
            return super().execute(capability, arguments)
        finally:
            direct.invoke = invoke

    # ------------------------------------------------------------------ automatic arm choice (optional)
    def enable_auto_arm(self, bases_xy: Mapping[str, Sequence[float]]) -> None:
        """Calls without `arm` go to an arm chosen from kinematics only (no task knowledge). A motion to a target pose keeps
        the active arm while its gripper is closed (it may hold an object); otherwise the other arm takes over when its
        base is clearly nearer (hysteresis margin) and it can reach the pose, the active arm keeps it when it can reach
        it, else the arm that can reach it (planner IK). Every other arm tool acts on the active arm; explicit `arm` wins."""
        self._auto_arm_enabled = True
        self._bases_xy = {a: np.asarray(v, float)[:2] for a, v in bases_xy.items()}
        self._active_arm = self._primary
        self.auto_arm_log: list[dict] = []

    _auto_arm_enabled = False

    AUTO_ARM_MARGIN_M = 0.10   # hysteresis: the other arm must be this much nearer (base xy) to take over

    def _reachable(self, arm: str, position, quaternion) -> bool | None:
        """Planner-IK reachability of a pose for one arm (hardware model only, no scene); None if no orientation."""
        if quaternion is None:
            return None
        from scipy.spatial.transform import Rotation
        w, x, y, z = (float(v) for v in quaternion)
        target = np.eye(4)
        target[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
        target[:3, 3] = np.asarray(position, float)[:3]
        model = self._robots[arm].planning_model()
        seed = np.asarray(self._envs[arm]._driver.read().joints, float)
        q = model.solve_ik(target, seed, aperture=1., position_tolerance=.003, orientation_tolerance=.03)
        return q is not None and not model.collision(q, 1., np.empty((0, 3)))["collision"]

    def _auto_arm(self, name: str, args: Mapping[str, object]) -> str:
        inner = args.get("arguments") if isinstance(args.get("arguments"), Mapping) else args
        position = inner.get("position") if isinstance(inner, Mapping) else None
        quaternion = inner.get("quaternion") if isinstance(inner, Mapping) else None
        active = self._active_arm
        if position is None or name not in ("move_to", "preview_motion", "check_motion"):
            return active
        try:
            closed = float(self._envs[active].gripper_action) > 0.0   # last jaw command (+1 close)
        except Exception:  # noqa: BLE001
            closed = False
        if closed:  # it may hold an object: never hand the motion to the other arm
            return active
        p = np.asarray(position, float)[:2]
        dist = {a: float(np.linalg.norm(p - b)) for a, b in self._bases_xy.items()}
        other = next(a for a in self._bases_xy if a != active)
        clearly_nearer = dist[other] + self.AUTO_ARM_MARGIN_M < dist[active]
        try:
            reach = {active: self._reachable(active, position, quaternion), other: self._reachable(other, position, quaternion)}
        except Exception:  # noqa: BLE001
            reach = {active: None, other: None}
        if clearly_nearer and reach[other] is not False:
            chosen, why = other, "nearer"
        elif reach[active] is not False:
            chosen, why = active, "active_reaches"
        elif reach[other]:
            chosen, why = other, "only_other_reaches"
        else:
            chosen, why = (other, "neither_reaches_nearer") if clearly_nearer else (active, "neither_reaches_keep")
        if chosen != active:
            self.auto_arm_log.append({"tool": name, "from": active, "to": chosen, "why": why,
                                      "position": [float(v) for v in position]})
        self._active_arm = chosen
        return chosen

    def arm_backends(self) -> Mapping[str, XR1ActionBackend]:
        return dict(self._robots)

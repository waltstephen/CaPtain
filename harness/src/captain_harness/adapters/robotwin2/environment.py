"""CaP-X tool adapter over the RoboTwin dual-arm facades: the XR1/LIBERO manifest plus one optional `arm` argument.

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
    examples = (  # one rendered line each
        "# two hands close on one object at the same time, then lift it together:",
        "move_both(left={'gripper': 'close'}, right={'gripper': 'close'})",
        "move_both(left={'position': [xl, yl, zl + 0.1], 'quaternion': ql, 'contact_mode': 'manipulation'}, "
        "right={'position': [xr, yr, zr + 0.1], 'quaternion': qr, 'contact_mode': 'manipulation'})",
        "# one arm moves while the other holds still: give the other arm its current pose",
        "move_both(left={'position': p_left_new, 'quaternion': q_left}, right={'position': p_right_now, 'quaternion': q_right_now})")
    return replace(move_to_spec, name="move_both", description=MOVE_BOTH_DESCRIPTION, arguments_schema=schema,
                   argument_order=("left", "right"), result_schema={"type": "object", "additionalProperties": True},
                   examples=examples)


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
        from .sim_driver import Companion
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

        def run_on(arm, name, args):
            if arm == adapter._primary:
                return invoke(name, args)
            with adapter._arm_selected(arm):
                return invoke(name, args)

        def routed(name, args):
            args = dict(args)
            arm = args.pop("arm", None)
            if arm is None:
                arm = adapter._auto_arm(name, args) if adapter._auto_arm_enabled else adapter._primary
            elif adapter._auto_arm_enabled:
                adapter._active_arm = arm   # an explicit arm becomes the arm "used last" for later calls without arm
            cleared = adapter._r5_clear_idle(arm, args, run_on) if name == "move_to" and adapter._r5_clear else None
            result = run_on(arm, name, args)
            if cleared is not None:
                from dataclasses import replace as _replace
                result = _replace(result, value={**(result.value or {}), "cleared_idle_arm": cleared})
            if name == "move_to" and adapter._r4:
                result = adapter._r4_after_move(arm, args, result, run_on)
            return result

        direct.invoke = routed
        try:
            return super().execute(capability, arguments)
        finally:
            direct.invoke = invoke

    # ------------------------------------------------------------------ ROUND 5 (config r5_clear_idle_arm)
    _r5_clear = False
    CLEAR_RADIUS_M = 0.22

    def _r5_clear_idle(self, arm, args, run_on):
        """The other arm is idle (last jaw command open) and its hand stands within 0.22 m (xy) of this arm's move_to
        target: lift it 0.10 m and send it to its ready pose first (A2 finding: after an auto-arm switch the new arm ran
        into the hand the other arm left at the work spot -> load_budget / motion_stalled with no wrist force)."""
        if "position" not in args or len(self._arms) < 2:
            return None
        other = next(a for a in self._arms if a != arm)
        env = self._envs[other]
        if float(env.gripper_action) > 0.0:   # may hold an object: never move it implicitly
            return None
        pose = env.grip_site_pose(); eef = pose[:3, 3]
        target = np.asarray(args["position"], float)
        if float(np.linalg.norm(eef[:2] - target[:2])) > self.CLEAR_RADIUS_M:
            return None
        from scipy.spatial.transform import Rotation
        qx, qy, qz, qw = Rotation.from_matrix(pose[:3, :3]).as_quat(); quat = [float(qw), float(qx), float(qy), float(qz)]
        home = np.asarray(env._driver.sim.bases[other][:3, 3], float) + np.array([0.40, 0.0, 0.30])
        if float(np.linalg.norm(eef - home)) < 0.05:   # already at its ready pose
            return None
        log = []
        for a in ({"position": (eef + [0, 0, 0.10]).tolist(), "quaternion": quat, "contact_mode": "manipulation"},
                  {"position": home.tolist(), "quaternion": quat}):
            r = run_on(other, "move_to", dict(a))
            log.append({"position": np.round(a["position"], 3).tolist(), "success": r.success, "message": (r.message or "")[:60]})
        info = {"arm": other, "from": np.round(eef, 3).tolist(), "steps": log}
        self.r4_log = getattr(self, "r4_log", []); self.r4_log.append({"capability": "clear_idle_arm", **info})
        return info

    # ------------------------------------------------------------------ ROUND 4 harness capabilities (config r4_*)
    _r4: dict = {}
    TILT_PITCHES_DEG = (25.0, 40.0, 46.0, 53.5)

    def enable_r4(self, flags: Mapping[str, bool]) -> None:
        self._r4 = {k: bool(v) for k, v in flags.items() if v}
        self.r4_log: list[dict] = []

    def _r4_after_move(self, arm, args, result, run_on):
        from dataclasses import replace as _replace
        from scipy.spatial.transform import Rotation
        if "gripper" in args and args["gripper"] == "close":
            opening = float(self._envs[arm]._driver.read().gripper_open_fraction)
            if (self._r4.get("grip_on_contact") and not result.success and result.executed
                    and (result.message or "") in ("closure_force_increase", "contact_force") and 0.03 < opening < 0.95):
                # jaws stopped on something between the fingers: finish the close without the closure monitor
                # hold the arm where it is and command the jaw fully closed for 0.5 s (squeeze onto the object); the
                # backend's jaw command becomes "closed" so later moves keep the grip
                env, robot = self._envs[arm], self._robots[arm]
                ticks = max(1, int(round(0.5 * env._driver.control_frequency)))
                for _ in range(ticks):
                    env.step(np.r_[np.zeros(6), 1.0])
                robot._gripper_command_fraction = 0.0
                steps = robot.simulation_steps + ticks
                for a in self._arms:
                    self._resync(a, steps)
                after = float(env._driver.read().gripper_open_fraction)
                from captain_harness.ports.capability import CapabilityResult
                closed = CapabilityResult.ok(dict(result.value or {}), "")
                info = {"stopped_by": result.message, "opening_at_stop": round(opening, 3), "opening_after": round(after, 3),
                        "note": "closure-force rule stopped the close with the jaws apart (object between the fingers); "
                                "the harness completed the close onto it"}
                self.r4_log.append({"capability": "grip_on_contact", "arm": arm, **info})
                if closed.success:
                    return _replace(closed, value={**(closed.value or {}), "grip_on_contact": info},
                                    message="grip_on_contact: jaws closed onto the object after a closure-force stop")
                return result
            if result.success or (result.value or {}).get("grip_on_contact"):
                d = self._envs[arm]._driver
                self._r4_grasp = getattr(self, "_r4_grasp", {}); self._r4_grasp[arm] = d.fk(d._pad(d.sim.state[arm]["q"]))
            if self._r4.get("grasp_feedback") and result.success:
                after = float(self._envs[arm]._driver.read().gripper_open_fraction)
                if after < 0.03:
                    return _replace(result, value={**(result.value or {}),
                                                   "grasp_check": "jaws closed fully: nothing between the fingers"})
            return result
        result = self._r4_tilt(arm, args, result, run_on)
        if (self._r4.get("auto_relay") and not result.success and "position" in args and not getattr(self, "_r4_in_relay", False)
                and "ik_unavailable" in (result.message or "") and len(self._arms) > 1):
            self._r4_in_relay = True
            try:
                result = self._r4_relay(arm, args, result, run_on)
            finally:
                self._r4_in_relay = False
        return result

    def _r4_feasible(self, arm, args) -> bool:
        from scipy.spatial.transform import Rotation
        d = self._envs[arm]._driver
        w, x, y, z = (float(v) for v in args["quaternion"])
        g = np.eye(4); g[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix(); g[:3, 3] = np.asarray(args["position"], float)
        cands = [g]
        base = np.asarray(d.sim.bases[arm][:3, 3], float); u = g[:3, 3][:2] - base[:2]
        if np.linalg.norm(u) > 1e-6:
            u = np.r_[u / np.linalg.norm(u), 0.0]; k = np.cross([0.0, 0.0, 1.0], u)
            for p in self.TILT_PITCHES_DEG:
                t = g.copy(); t[:3, :3] = Rotation.from_rotvec(-np.radians(p) * k).as_matrix() @ g[:3, :3]; cands.append(t)
        q0 = d._pad(d.sim.state[arm]["q"])
        return any(np.linalg.norm(d.fk(d._ik(c, q0, iterations=300))[:3, 3] - c[:3, 3]) < 0.002 for c in cands)

    def _r4_relay(self, arm, args, result, run_on):
        """Holding arm cannot reach the requested pose but the other arm can: place at a midline relay point, the other
        arm regrasps at the same pose and performs the requested move (harness capability, disclosed in the receipt)."""
        from dataclasses import replace as _replace
        from scipy.spatial.transform import Rotation
        other = next(a for a in self._arms if a != arm)
        grasp = getattr(self, "_r4_grasp", {}).get(arm)
        if grasp is None or float(self._envs[arm].gripper_action) <= 0.0 or not self._r4_feasible(other, args):
            return result
        b0 = np.asarray(self._envs[arm]._driver.sim.bases[arm][:3, 3]); b1 = np.asarray(self._envs[other]._driver.sim.bases[other][:3, 3])
        mid = np.array([float(np.clip(grasp[0, 3], 0.40, 0.60)), 0.5 * (b0[1] + b1[1]), float(grasp[2, 3])])
        qx, qy, qz, qw = Rotation.from_matrix(grasp[:3, :3]).as_quat(); quat = [float(qw), float(qx), float(qy), float(qz)]

        def home_pos(a):  # this arm's ready position (0.40 m in front of its base, 0.30 m up)
            return np.asarray(self._envs[a]._driver.sim.bases[a][:3, 3], float) + np.array([0.40, 0.0, 0.30])
        steps = [(arm, {"position": (mid + [0, 0, 0.10]).tolist(), "quaternion": quat, "contact_mode": "manipulation"}),
                 (arm, {"position": mid.tolist(), "quaternion": quat, "contact_mode": "manipulation"}),
                 (arm, {"gripper": "open"}),
                 (arm, {"position": (mid + [0, 0, 0.10]).tolist(), "quaternion": quat}),
                 (arm, {"position": home_pos(arm).tolist(), "quaternion": quat}),          # clear the relay point
                 (other, {"position": (mid + [0, 0, 0.10]).tolist(), "quaternion": quat}),
                 (other, {"position": mid.tolist(), "quaternion": quat, "contact_mode": "manipulation"}),
                 (other, {"gripper": "close"}),
                 (other, {"position": (mid + [0, 0, 0.10]).tolist(), "quaternion": quat, "contact_mode": "manipulation"})]
        log = []
        for who, a in steps:
            r = run_on(who, "move_to", dict(a))
            r = self._r4_after_move(who, dict(a), r, run_on) if "gripper" in a or not r.success else r
            log.append({"arm": who, "args": {k: (np.round(v, 3).tolist() if k == "position" else v) for k, v in a.items() if k != "quaternion"},
                        "success": r.success, "message": (r.message or "")[:80]})
            if not r.success:
                info = {"from": arm, "to": other, "relay_point": mid.round(3).tolist(), "steps": log, "completed": False}
                self.r4_log.append({"capability": "auto_relay", **info})
                return _replace(r, value={**(r.value or {}), "auto_relay": info}, message="auto_relay failed: " + (r.message or ""))
        self._active_arm = other
        final = run_on(other, "move_to", dict(args))
        final = self._r4_tilt(other, dict(args), final, run_on)
        info = {"from": arm, "to": other, "relay_point": mid.round(3).tolist(), "steps": log, "completed": True}
        self.r4_log.append({"capability": "auto_relay", "final_success": final.success, **info})
        return _replace(final, value={**(final.value or {}), "auto_relay": info},
                        message=f"auto_relay {arm}->{other} at midline, then: " + (final.message or ""))

    def _r4_tilt(self, arm, args, result, run_on):
        from dataclasses import replace as _replace
        from scipy.spatial.transform import Rotation
        if (self._r4.get("auto_tilt") and not result.success and "position" in args and "quaternion" in args
                and ("ik_unavailable" in (result.message or "") or "ik_failure" in str(result.failure_category).lower())):
            w, x, y, z = (float(v) for v in args["quaternion"])
            R = Rotation.from_quat([x, y, z, w])
            approach = R.as_matrix()[:, 2]
            if approach @ np.array([0.0, 0.0, -1.0]) >= np.cos(np.radians(15.0)):
                base = np.asarray(self._envs[arm]._driver.sim.bases[arm][:3, 3], float)
                u = np.asarray(args["position"], float)[:2] - base[:2]
                if np.linalg.norm(u) > 1e-6:
                    u = np.r_[u / np.linalg.norm(u), 0.0]
                    k = np.cross([0.0, 0.0, 1.0], u)
                    for pitch in self.TILT_PITCHES_DEG:
                        Rt = Rotation.from_rotvec(-np.radians(pitch) * k) * R   # approach leans along base->target
                        qx, qy, qz, qw = Rt.as_quat()
                        tried = run_on(arm, "move_to", {**args, "quaternion": [float(qw), float(qx), float(qy), float(qz)]})
                        if tried.success or "ik_unavailable" not in (tried.message or ""):
                            info = {"pitch_deg": pitch, "requested_quaternion_wxyz": list(args["quaternion"]),
                                    "executed_quaternion_wxyz": [float(qw), float(qx), float(qy), float(qz)],
                                    "reason": "the vertical approach was IK-infeasible for this arm"}
                            self.r4_log.append({"capability": "auto_tilt", "arm": arm, "success": tried.success, **info})
                            return _replace(tried, value={**(tried.value or {}), "auto_tilt": info},
                                            message=f"auto_tilt {pitch:g} deg: " + (tried.message or ""))
        return result

    # ------------------------------------------------------------------ automatic arm choice (optional)
    # (same rule as the RoboDojo copy of this adapter, adapters/robodojo/dual_arm.py)
    def enable_auto_arm(self, bases_xy: Mapping[str, Sequence[float]]) -> None:
        """Calls without `arm` go to an arm chosen from kinematics only: a motion to a target position uses the arm whose
        base is nearest in xy, unless the active arm's gripper is closed (it may hold an object), in which case the
        active arm keeps it; every other arm tool acts on the active arm. Explicit `arm` always wins."""
        self._auto_arm_enabled = True
        self._bases_xy = {a: np.asarray(v, float)[:2] for a, v in bases_xy.items()}
        self._active_arm = self._primary
        self.auto_arm_log: list[dict] = []

    _auto_arm_enabled = False

    def _auto_arm(self, name: str, args: Mapping[str, object]) -> str:
        inner = args.get("arguments") if isinstance(args.get("arguments"), Mapping) else args
        position = inner.get("position") if isinstance(inner, Mapping) else None
        active = self._active_arm
        if position is not None and name in ("move_to", "preview_motion", "check_motion"):
            try:
                # "may hold an object" = the active arm's last jaw COMMAND was close (facade gripper_action > 0).
                # A measured-opening threshold (< 0.9) misfired on aloha, whose open jaw reads ~0.88 of its stroke, and
                # pinned every call to the right arm (an earlier bug)
                closed = float(self._envs[active].gripper_action) > 0.0
            except Exception:  # noqa: BLE001
                closed = False
            if not closed:
                p = np.asarray(position, float)[:2]
                chosen = min(self._bases_xy, key=lambda a: float(np.linalg.norm(p - self._bases_xy[a])))
                if chosen != active:
                    self.auto_arm_log.append({"tool": name, "from": active, "to": chosen, "position": list(map(float, position))})
                self._active_arm = active = chosen
        return active

    def arm_backends(self) -> Mapping[str, XR1ActionBackend]:
        return dict(self._robots)

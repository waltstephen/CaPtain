"""ROUND 4 harness capabilities for RoboDojo (generic, disclosed; model-facing tool API unchanged).

auto_tilt     move_to: when the requested pose is IK-unavailable for the arm, retry the same position with the
              approach tilted 25/40/46/53.5 deg away from the arm base (jaw yaw within +-20 deg), first feasible wins.
grasp_feedback move_to(gripper='close'): if the jaws closed fully on nothing, report it as a failed close and reopen.
auto_relay    move_to with a held object whose target is outside the holding arm's reach but inside the other arm's:
              place it at a midline hand-over point, the other arm regrasps it, then the original move_to continues
              with that arm. Every step is a normal guarded move_to of the respective arm.
All three use robot kinematics and measured robot state only (no object poses, no task information).
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy.spatial.transform import Rotation

TILTS_DEG = (25.0, 40.0, 46.0, 53.5)
YAWS_DEG = (0.0, 10.0, -10.0, 20.0, -20.0)
EMPTY_APERTURE = 0.02          # X5: fully closed jaws read ~0; a held coin (thinnest object) reads ~0.03+
HANDOVER_XY = (0.26, 0.30)     # robot_base: midline between the bases (0, 0) and (0, 0.60), reachable by both arms
HANDOVER_CLEAR_M = 0.10
COUNTS = {"auto_tilt_used": 0, "auto_tilt_failed": 0}


def _quat_wxyz(R):
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return [float(w), float(x), float(y), float(z)]


def _mat(q):
    w, x, y, z = (float(v) for v in q)
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def tilt_candidates(position, quaternion, base_xy):
    """Orientations: requested orientation tilted away from the base about the horizontal axis, then yawed about
    the tilted approach axis. Yields (pitch_deg, yaw_deg, quaternion_wxyz)."""
    R = _mat(quaternion)
    d = np.asarray(position[:2], float) - np.asarray(base_xy, float)
    n = np.linalg.norm(d)
    d = d / n if n > 1e-6 else np.array([1.0, 0.0])
    axis = np.array([-d[1], d[0], 0.0])
    approach = R[:, 2]
    for th in TILTS_DEG:
        Rt = None
        for sgn in (1.0, -1.0):
            cand = Rotation.from_rotvec(sgn * np.radians(th) * axis).as_matrix()
            v = cand @ approach
            if float(v[:2] @ d) > 0:      # approach now points away from the base (wrist on the base side)
                Rt = cand
                break
        if Rt is None:
            continue
        Rn = Rt @ R
        for psi in YAWS_DEG:
            Ry = Rotation.from_rotvec(np.radians(psi) * Rn[:, 2]).as_matrix()
            yield th, psi, _quat_wxyz(Ry @ Rn)


def install_auto_tilt(direct_cls):
    if getattr(direct_cls, "_rd_auto_tilt", False):
        return
    orig = direct_cls._move

    def _move(self, position, quaternion, *a, **k):
        base = getattr(self.robot, "_rd_base_xy", None)
        sticky = getattr(self.robot, "_rd_last_tilt", None)
        if base is not None and sticky is not None and np.allclose(sticky[0], quaternion, atol=0.05):
            res = orig(self, position, sticky[1], *a, **k)
            if res.success or res.executed or "ik_unavailable" not in (res.message or ""):
                return _tag(res, sticky[2], quaternion, sticky[1], "sticky")
        res = orig(self, position, quaternion, *a, **k)
        if base is None or res.success or res.executed or "ik_unavailable" not in (res.message or ""):
            if res.executed:
                self.robot._rd_last_tilt = None
            return res
        tried = 0
        for th, psi, q in tilt_candidates(np.asarray(position, float), quaternion, base):
            tried += 1
            r2 = orig(self, position, q, *a, **k)
            if r2.success or r2.executed or "ik_unavailable" not in (r2.message or ""):
                self.robot._rd_last_tilt = (list(quaternion), q, (th, psi))
                COUNTS["auto_tilt_used"] += 1
                return _tag(r2, (th, psi), quaternion, q, "searched")
        COUNTS["auto_tilt_failed"] += 1
        value = dict(res.value or {})
        value["auto_tilt"] = {"tried_candidates": tried, "result": "no tilted approach reachable either"}
        return dataclasses.replace(res, value=value)

    direct_cls._move = _move
    direct_cls._rd_auto_tilt = True


def _tag(res, tilt, requested, executed, how):
    value = dict(res.value or {})
    value["auto_tilt"] = {"pitch_deg": tilt[0], "yaw_deg": tilt[1], "requested_quaternion_wxyz": list(map(float, requested)),
                          "executed_quaternion_wxyz": list(map(float, executed)), "selection": how,
                          "note": "harness tilted the approach because the requested orientation was out of reach"}
    msg = f"auto_tilt {tilt[0]:.1f} deg pitch / {tilt[1]:+.0f} deg yaw: " + (res.message or "")
    return dataclasses.replace(res, value=value, message=msg)


class R4Executor:
    """Wraps adapter.execute for grasp feedback and auto-relay (RoboDojo dual X5)."""

    def __init__(self, adapter, world, *, grasp_feedback: bool, auto_relay: bool, grasp_retry: bool = False):
        self.adapter, self.world = adapter, world
        self.grasp_retry = grasp_retry
        self.grasp_feedback, self.auto_relay = grasp_feedback, auto_relay
        self.orig = adapter.execute
        self.log = {"empty_closes": 0, "relays": []}
        adapter.execute = self.execute

    def _arm(self, args):
        return args.get("arm") or getattr(self.adapter, "_active_arm", self.adapter._primary)

    def _aperture(self, arm):
        return float(self.world.drivers[arm].read().gripper_open_fraction)

    def _holding(self, arm):
        env = self.adapter._envs[arm]
        return float(env.gripper_action) > 0.0 and self._aperture(arm) > EMPTY_APERTURE

    def execute(self, capability, arguments):
        args = dict(arguments or {})
        if (self.auto_relay and capability == "move_to" and "position" in args and "quaternion" in args):
            relay = self._maybe_relay(args)
            if relay is not None:
                return relay
        res = self.orig(capability, arguments)
        if (self.grasp_feedback and capability == "move_to" and args.get("gripper") == "close" and "position" not in args
                and res.success):
            arm = self._arm(args)
            ap = self._aperture(arm)
            retries = []
            if ap < EMPTY_APERTURE and self.grasp_retry:
                ap, retries = self._retry_grasp(arm)
            if ap < EMPTY_APERTURE:
                self.log["empty_closes"] += 1
                reopen = self.orig("move_to", {"gripper": "open", "arm": arm})
                from captain_harness.ports.capability import FailureCategory
                value = dict(res.value or {})
                value["grasp_check"] = {"result": "closed_empty", "measured_open_fraction": ap,
                                        "reopened": bool(reopen.success), "automatic_retries": retries,
                                        "basis": "jaws reached their fully-closed stroke: nothing is between the fingers"}
                return dataclasses.replace(res, success=False, failure_category=FailureCategory.CONTROLLER_REJECTED,
                                           message="grasp_check: closed_empty - the jaws closed fully on nothing; "
                                                   "they were reopened. Re-observe and re-approach the object.",
                                           value=value)
            drv = self.world.drivers[arm]
            self.adapter._robots[arm]._rd_last_close_z = float(drv.fk(drv.read().joints)[2, 3])
            value = dict(res.value or {})
            value["grasp_check"] = {"result": "object_between_fingers", "measured_open_fraction": ap,
                                    **({"automatic_retries": retries} if retries else {})}
            msg = res.message if not retries else ("grasp_check: object between the fingers after an automatic retry ("
                                                    + retries[-1]["kind"] + "). " + (res.message or ""))
            return dataclasses.replace(res, value=value, message=msg)
        return res

    # ------------------------------------------------------------------ grasp retry (ROUND 5, transparent)
    def _retry_grasp(self, arm):
        """Empty close: reopen, retry 1 cm lower, then rotated 90 deg about the vertical at the original pose."""
        drv = self.world.drivers[arm]
        base = drv.fk(drv.read().joints)
        q0 = _quat_wxyz(base[:3, :3])
        p0 = [float(v) for v in base[:3, 3]]
        tries = [("lower_1cm", [p0[0], p0[1], p0[2] - 0.01], q0),
                 ("yaw_90deg", p0, _quat_wxyz(Rotation.from_euler("z", np.pi / 2).as_matrix() @ base[:3, :3]))]
        log = []
        ap = self._aperture(arm)
        for kind, pos, q in tries:
            self.orig("move_to", {"gripper": "open", "arm": arm})
            mv = self.orig("move_to", {"position": pos, "quaternion": q, "contact_mode": "manipulation", "arm": arm})
            cl = self.orig("move_to", {"gripper": "close", "arm": arm})
            ap = self._aperture(arm)
            log.append({"kind": kind, "move_success": bool(mv.success), "close_success": bool(cl.success),
                        "measured_open_fraction": round(ap, 4)})
            if ap >= EMPTY_APERTURE:
                self.log.setdefault("grasp_retry_success", 0)
                self.log["grasp_retry_success"] += 1
                break
        return ap, log

    # ------------------------------------------------------------------ auto home (ROUND 5)
    def record_starts(self):
        self.starts = {arm: d.fk(d.read().joints) for arm, d in self.world.drivers.items()}

    def auto_home(self):
        """Before the official final check: release, lift and return each arm that left its start pose."""
        out = {}
        for arm, start in getattr(self, "starts", {}).items():
            drv = self.world.drivers[arm]
            cur = drv.fk(drv.read().joints)
            dist = float(np.linalg.norm(cur[:3, 3] - start[:3, 3]))
            ang = float(Rotation.from_matrix(cur[:3, :3] @ start[:3, :3].T).magnitude())
            if dist < 0.02 and ang < 0.15:
                out[arm] = "already_home"
                continue
            steps = []
            if float(self.adapter._envs[arm].gripper_action) > 0.0:
                steps.append(("open", self.orig("move_to", {"gripper": "open", "arm": arm}).success))
            up = [float(cur[0, 3]), float(cur[1, 3]), max(float(cur[2, 3]) + 0.05, float(start[2, 3]))]
            steps.append(("lift", self.orig("move_to", {"position": up, "quaternion": _quat_wxyz(cur[:3, :3]), "arm": arm}).success))
            r = self.orig("move_to", {"position": [float(v) for v in start[:3, 3]], "quaternion": _quat_wxyz(start[:3, :3]),
                                      "arm": arm})
            steps.append(("home", r.success))
            out[arm] = {"steps": steps, "home_reached": bool(r.success), "from_m": round(dist, 3)}
        self.log["auto_home"] = out
        return out

    # ------------------------------------------------------------------ auto relay
    def _reach(self, arm, position, quaternion):
        try:
            return bool(self.adapter._reachable(arm, position, quaternion))
        except Exception:  # noqa: BLE001
            return False

    def _reach_any_height(self, arm, position, quaternion):
        for z in (position[2], 0.12, 0.06):
            p = [position[0], position[1], z]
            if self._reach(arm, p, quaternion):
                return True
            for _, _, q in tilt_candidates(np.asarray(p, float), quaternion, self._base(arm)):
                if self._reach(arm, p, q):
                    return True
        return False

    def _base(self, arm):
        return self.world.sim.bases[arm][:2, 3]

    def _maybe_relay(self, args):
        a = self._arm(args)
        arms = list(self.world.drivers)
        b = next(x for x in arms if x != a)
        pos, quat = [float(v) for v in args["position"]], [float(v) for v in args["quaternion"]]
        if not self._holding(a):
            return None
        if self._reach_any_height(a, pos, quat) or not self._reach_any_height(b, pos, quat):
            return None
        return self._relay(a, b, args)

    def _relay(self, a, b, args):
        log = {"from_arm": a, "to_arm": b, "steps": []}
        drv_a = self.world.drivers[a]
        st = drv_a.read()
        eef_a = drv_a.fk(st.joints)
        grip_z = float(getattr(self.adapter._robots[a], "_rd_last_close_z", None) or max(0.03, eef_a[2, 3] - 0.0))
        q_a = _quat_wxyz(eef_a[:3, :3])
        hx, hy = HANDOVER_XY
        place_z = max(0.02, min(grip_z, 0.08)) + 0.005
        # receiving arm grasps straight down with the same jaw yaw as the giver's current grasp (top-down family)
        yaw = float(np.arctan2(eef_a[1, 0], eef_a[0, 0]))
        q_b = _quat_wxyz(Rotation.from_euler("z", yaw).as_matrix() @ Rotation.from_quat([0.7071068, 0.7071068, 0, 0]).as_matrix())

        def step(name, arm, call):
            r = self.orig("move_to", {**call, "arm": arm})
            log["steps"].append({"step": name, "arm": arm, "success": bool(r.success), "message": (r.message or "")[:120]})
            return r

        plan = [("giver_above_handover", a, {"position": [hx, hy, place_z + HANDOVER_CLEAR_M], "quaternion": q_a}),
                ("giver_place", a, {"position": [hx, hy, place_z], "quaternion": q_a, "contact_mode": "manipulation"}),
                ("giver_release", a, {"gripper": "open"}),
                ("giver_retreat", a, {"position": [hx, hy, place_z + HANDOVER_CLEAR_M], "quaternion": q_a}),
                ("giver_clear", a, {"position": [0.10, 0.0 if a == "right" else 0.60, 0.16], "quaternion": q_a}),
                ("receiver_open", b, {"gripper": "open"}),
                ("receiver_above", b, {"position": [hx, hy, place_z + HANDOVER_CLEAR_M], "quaternion": q_b}),
                ("receiver_descend", b, {"position": [hx, hy, place_z], "quaternion": q_b, "contact_mode": "manipulation"}),
                ("receiver_close", b, {"gripper": "close"}),
                ("receiver_lift", b, {"position": [hx, hy, place_z + HANDOVER_CLEAR_M], "quaternion": q_b,
                                      "contact_mode": "manipulation"})]
        for name, arm, call in plan:
            r = step(name, arm, call)
            if not r.success and name not in ("giver_clear",):
                log["result"] = f"relay stopped at {name}"
                self.log["relays"].append(log)
                value = dict(r.value or {})
                value["auto_relay"] = log
                return dataclasses.replace(r, message=f"auto_relay ({a} -> {b}) stopped at {name}: {r.message}", value=value)
        if self._aperture(b) < EMPTY_APERTURE:
            log["result"] = "receiver closed empty"
            self.log["relays"].append(log)
            from captain_harness.ports.capability import CapabilityResult, FailureCategory
            return CapabilityResult(False, True, FailureCategory.CONTROLLER_REJECTED,
                                    f"auto_relay ({a} -> {b}): the {b} arm closed empty at the hand-over point "
                                    f"({hx:.2f}, {hy:.2f}); the object should be there", {"auto_relay": log})
        if hasattr(self.adapter, "_active_arm"):
            self.adapter._active_arm = b
        final = self.orig("move_to", {**args, "arm": b})
        log["result"] = "relayed; original move executed by the receiving arm"
        log["final_success"] = bool(final.success)
        self.log["relays"].append(log)
        value = dict(final.value or {})
        value["auto_relay"] = log
        return dataclasses.replace(final, value=value,
                                   message=f"auto_relay: {a} handed the object to {b} at ({hx:.2f}, {hy:.2f}); {final.message}")

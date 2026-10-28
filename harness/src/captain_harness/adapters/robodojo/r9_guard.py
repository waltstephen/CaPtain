"""ROUND 9: generic guard fixes for the RoboDojo X5 wrist-force contact guards.

Motion/sensing internals only; model-facing tool API and TASK.md unchanged. Every fix has its own config flag
(config["control"]["guard_fixes"]). Robot-only quantities (wrist force, kinematics, the table plane z = 0 of
robot_base, finger length) — no object poses, no task information.

 1 post_stop_backoff  after a guard stop: back off along -motion, let the force settle, re-tare the residual offset
 2 increase_only      each move is judged by the residual increase over its own settled start (absolute start limit
                      replaced by a hard ceiling)
 3 start_blank        first ~1 cm / few ticks of a move: only the emergency/ceiling stops apply
 4 payload_comp       first move after a successful grasp gets a payload allowance (the held load then becomes part
                      of the next moves' start reference)
 5 fingertip_floor    downward targets are clamped so the fingertips stay >= clearance above the table
 6 contact_tolerant   manipulation mode: a contact stop within contact_tolerant_m of the endpoint = reached with contact
 7 approach_from_above / hand_keepout  low lateral moves: lift -> traverse -> descend; idle other hand homed if the
                      target is within keepout_m of it (no relay in progress)
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULTS = {
    "post_stop_backoff": False, "backoff_m": 0.004, "settle_max_ticks": 4, "settle_delta_n": 1.0, "retare": False,
    "increase_only": False, "no_initial_limit": False, "hard_ceiling_n": 120.0,
    "start_blank": False, "start_blank_m": 0.01, "start_blank_max_ticks": 4,
    "payload_comp": False, "payload_allowance_n": 15.0,
    "fingertip_floor": False, "finger_length_m": 0.0156, "table_clearance_m": 0.002,
    "contact_tolerant": False, "contact_tolerant_m": 0.015,
    "approach_from_above": False, "afa_min_lateral_m": 0.05, "afa_tip_clear_m": 0.08,
    "hand_keepout": False, "keepout_m": 0.10,
}
GUARD_STOPS = {"contact_force", "contact_detected", "emergency_limit", "load_budget_reached", "contact_stall",
               "motion_stalled", "tracking_error_limit", "lateral_contact"}


def flags(config):
    gf = dict(DEFAULTS)
    gf.update((config.get("control") or {}).get("guard_fixes") or {})
    return gf


def _quat_wxyz(R):
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return [float(w), float(x), float(y), float(z)]


def _R(q):
    w, x, y, z = (float(v) for v in q)
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def tip_z(p, R, finger):
    """Fingertip height: EEF z axis is the approach axis; the fingertips are `finger` m past the TCP along it."""
    return float(p[2] + R[2, 2] * finger)


# ---------------------------------------------------------------------------------------------------- backend level
def install_backend(backend_cls, gf):
    if getattr(backend_cls, "_rd_r9", False):
        return
    from captain_harness.adapters.libero.contact_residual import ResidualContactDetector, motion_force
    from captain_harness.adapters.libero.contact_motion import _pose
    from captain_harness.adapters.robotics.cartesian_sweeps import cartesian_sweep
    from captain_harness.adapters.robotics.contact_limits import EMERGENCY_LIMIT_N
    from captain_harness.ports.capability import CapabilityResult

    backend_cls._rd_no_initial_limit = bool(gf["no_initial_limit"])
    backend_cls._rd_hard_ceiling_n = float(gf["hard_ceiling_n"])

    class R9Detector(ResidualContactDetector):
        def _r9_init(self, backend):
            self._r9b = backend
            self._r9_start = None
            self._r9_p0 = None
            self._r9_base_thr = self.threshold_n
            self._r9_base_lb = self.load_budget_n
            if gf["payload_comp"] and getattr(backend, "_rd_holding", False) and getattr(backend, "_rd_payload_pending", False):
                a = float(gf["payload_allowance_n"])
                self.threshold_n = self._r9_base_thr + a
                self.load_budget_n = min(EMERGENCY_LIMIT_N, self._r9_base_lb + a)
                self._r9_payload = a
            else:
                self._r9_payload = 0.0
            try:
                self._r9_p0 = np.asarray(backend._control_site_pose_world(), float)[:3, 3]
            except Exception:  # noqa: BLE001
                self._r9_p0 = None

        def observe_substep(self, sample):
            if sample is None or sample.get("f_world") is None:
                return super().observe_substep(sample)
            f = np.asarray(sample["f_world"], float)
            v = sample.get("v_tick")
            raw = f - np.asarray(self.cal.bias_world_n, float) - motion_force(self.cal, v)
            offs = np.zeros(3)
            tare = getattr(self._r9b, "_rd_tare", None)
            if gf["retare"] and tare is not None:
                offs = offs + np.asarray(tare, float)
            if gf["increase_only"]:
                if self._r9_start is None:
                    self._r9_start = raw - offs
                offs = offs + self._r9_start
            s2 = dict(sample)
            s2["f_world"] = f - offs
            reason = super().observe_substep(s2)
            progress = None
            if self._r9_p0 is not None:
                try:
                    progress = float(np.linalg.norm(np.asarray(self._r9b._control_site_pose_world(), float)[:3, 3] - self._r9_p0))
                except Exception:  # noqa: BLE001
                    progress = None
            if (gf["start_blank"] and reason in ("contact_detected", "load_budget_reached")
                    and self.samples <= int(gf["start_blank_max_ticks"])
                    and (progress is None or progress < float(gf["start_blank_m"]))):
                self.trigger, reason = None, None
                self._persist = self._persist_lat = self._sustain = 0
                self._r9_blanked = getattr(self, "_r9_blanked", 0) + 1
            if reason is None and float(np.linalg.norm(raw)) > float(gf["hard_ceiling_n"]) and self.trigger is None:
                reason = "emergency_limit"
                self.trigger = {"reason": reason, **(self.last or {}), "sample_index": self.samples,
                                "hard_ceiling_n": float(gf["hard_ceiling_n"]), "raw_contact_n": float(np.linalg.norm(raw))}
            return reason

        def receipt(self):
            r = super().receipt()
            r["r9"] = {"increase_only_start_n": None if self._r9_start is None else float(np.linalg.norm(self._r9_start)),
                       "retare_n": None if getattr(self._r9b, "_rd_tare", None) is None else float(np.linalg.norm(self._r9b._rd_tare)),
                       "start_blanked_samples": getattr(self, "_r9_blanked", 0), "payload_allowance_n": self._r9_payload}
            return r

    _make = backend_cls.make_contact_detector

    def make_contact_detector(self, contact_mode="guarded", load_budget_n=None):
        det = _make(self, contact_mode, load_budget_n)
        if det is None:
            return None
        det.__class__ = R9Detector
        det._r9_init(self)
        return det
    backend_cls.make_contact_detector = make_contact_detector

    _sweep = backend_cls._execute_contact_sweep   # r6's wrapper (cancels targets after a guard stop)

    def _execute_contact_sweep(self, *a, **k):
        path = a[0] if a else k.get("path")
        result = _sweep(self, *a, **k)
        value = dict(result.value or {})
        stop = value.get("stopped_reason")
        if k.get("retreat") or stop not in GUARD_STOPS:
            return result
        # 6 contact-tolerant final approach (manipulation mode only)
        if (gf["contact_tolerant"] and k.get("contact_mode") == "manipulation" and k.get("detector") is not None
                and stop in ("load_budget_reached", "contact_stall", "motion_stalled")
                and float(value.get("position_residual_m", 1.0)) <= float(gf["contact_tolerant_m"])):
            short = float(value["position_residual_m"])
            value.update(reached=True, reached_with_contact=True, contact_stop=stop)
            return CapabilityResult.ok(value, f"bounded Cartesian sweep reached its endpoint with contact "
                                              f"(stopped {short * 1000:.0f} mm short on {stop}; contact within "
                                              f"{float(gf['contact_tolerant_m']) * 100:.1f} cm of the endpoint counts as reached)")
        if not gf["post_stop_backoff"]:
            return result
        # 1 back off along -motion, settle, re-tare
        post = {}
        try:
            d = np.asarray(path[-1])[:3, 3] - np.asarray(path[0])[:3, 3]
            n = float(np.linalg.norm(d))
            u = d / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
            cur = _pose(self.get_robot_state().value)
            back = cartesian_sweep(cur, displacement_m=-u * float(gf["backoff_m"]))
            rb = _sweep(self, back, max_speed_m_s=.02, max_force_n=60., max_duration_s=2., retreat=True)
            post["backoff"] = {"m": float(gf["backoff_m"]), "stopped_reason": (rb.value or {}).get("stopped_reason"),
                               "steps": (rb.value or {}).get("control_steps")}
            hold = np.zeros(self._action_dim)
            hold[-1] = 1.0 - 2.0 * self._gripper_command_fraction
            prev = self._eef_force_world()
            ticks = 0
            for _ in range(int(gf["settle_max_ticks"])):
                if not self._step_action(hold):
                    break
                ticks += 1
                now = self._eef_force_world()
                if prev is not None and now is not None and float(np.linalg.norm(np.asarray(now) - np.asarray(prev))) < float(gf["settle_delta_n"]):
                    break
                prev = now
            post["settle_ticks"] = ticks
            cal = getattr(self, "_force_calibration", None)
            fw = self._eef_force_world()
            if gf["retare"] and cal is not None and fw is not None:
                r0 = np.asarray(fw, float) - np.asarray(cal.bias_world_n, float)
                if float(np.linalg.norm(r0)) < EMERGENCY_LIMIT_N:
                    self._rd_tare = r0
                    post["retare_n"] = float(np.linalg.norm(r0))
        except Exception as exc:  # noqa: BLE001
            post["error"] = f"{type(exc).__name__}: {exc}"
        value["post_stop"] = post
        return dataclasses.replace(result, value=value)
    backend_cls._execute_contact_sweep = _execute_contact_sweep
    backend_cls._rd_r9 = True


# ---------------------------------------------------------------------------------------------------- executor level
def install_executor(r4, adapter, world, gf):
    """Wraps adapter.execute (outside the R4 executor): fingertip floor, hand keep-out, approach from above, payload
    bookkeeping."""
    inner = adapter.execute
    finger, clear = float(gf["finger_length_m"]), float(gf["table_clearance_m"])
    log = r4.log.setdefault("r9", {"floor_clamps": 0, "keepout_homes": 0, "approach_from_above": 0})

    def tcp(arm):
        drv = world.drivers[arm]
        return drv.fk(drv.read().joints)

    def execute(capability, arguments):
        args = dict(arguments or {})
        arm = r4._arm(args)
        robot = adapter._robots.get(arm) if hasattr(adapter, "_robots") else None
        notes = []
        if capability == "move_to" and "position" in args and "quaternion" in args:
            p = [float(v) for v in args["position"]]
            R = _R(args["quaternion"])
            # 5 fingertip table floor
            if gf["fingertip_floor"]:
                zmin = clear - R[2, 2] * finger
                if p[2] < zmin:
                    notes.append(f"target z clamped {p[2]:.3f} -> {zmin:.3f} m so the fingertips stay "
                                 f"{clear * 1000:.0f} mm above the table")
                    p[2] = float(zmin)
                    args["position"] = p
                    log["floor_clamps"] += 1
            # 7b hand keep-out (idle other hand only; never during a relay)
            if gf["hand_keepout"] and not getattr(r4, "_relaying", False) and len(world.drivers) == 2:
                other = next(a for a in world.drivers if a != arm)
                po = tcp(other)[:3, 3]
                start_o = getattr(r4, "starts", {}).get(other)
                away = start_o is None or float(np.linalg.norm(po - start_o[:3, 3])) > 0.03
                if (float(np.linalg.norm(np.asarray(p[:2]) - po[:2])) < float(gf["keepout_m"]) and away
                        and not r4._holding(other) and start_o is not None):
                    up = [float(po[0]), float(po[1]), max(float(po[2]) + 0.05, float(start_o[2, 3]))]
                    r4.orig("move_to", {"position": up, "quaternion": _quat_wxyz(tcp(other)[:3, :3]), "arm": other})
                    rh = r4.orig("move_to", {"position": [float(v) for v in start_o[:3, 3]],
                                             "quaternion": _quat_wxyz(start_o[:3, :3]), "arm": other})
                    notes.append(f"the idle {other} hand was within {float(gf['keepout_m']) * 100:.0f} cm of the target "
                                 f"and was moved back to its start pose first ({'ok' if rh.success else 'failed'})")
                    log["keepout_homes"] += 1
            # 7a approach from above for low lateral moves
            if gf["approach_from_above"]:
                T0 = tcp(arm)
                p0, R0 = T0[:3, 3], T0[:3, :3]
                lateral = float(np.linalg.norm(np.asarray(p[:2]) - p0[:2]))
                clear_tip = float(gf["afa_tip_clear_m"])
                if lateral >= float(gf["afa_min_lateral_m"]) and min(tip_z(p0, R0, finger), tip_z(p, R, finger)) < clear_tip:
                    zc = max(float(p0[2]), float(p[2]), clear_tip - R[2, 2] * finger, clear_tip - R0[2, 2] * finger)
                    seg_args = {k: v for k, v in args.items() if k not in ("position", "quaternion")}
                    segs = []
                    if zc - float(p0[2]) > 0.005:
                        segs.append(("lift", {**seg_args, "position": [float(p0[0]), float(p0[1]), zc],
                                              "quaternion": _quat_wxyz(R0)}))
                    segs.append(("traverse", {**seg_args, "position": [p[0], p[1], zc], "quaternion": args["quaternion"]}))
                    for name, sa in segs:
                        rs = inner("move_to", sa)
                        if not rs.success:
                            msg = (f"approach from above: the {name} segment (to z={zc:.3f} m) stopped: "
                                   + (rs.message or ""))
                            return dataclasses.replace(rs, message="; ".join(notes + [msg]))
                    notes.append(f"approached from above (lifted to z={zc:.3f} m, moved across, then descended)")
                    log["approach_from_above"] += 1
        res = inner(capability, args)
        # 4 payload bookkeeping
        if robot is not None and capability == "move_to":
            g = args.get("gripper")
            if g == "close" and "position" not in args and res.success and \
                    ((res.value or {}).get("grasp_check") or {}).get("result") == "object_between_fingers":
                robot._rd_holding, robot._rd_payload_pending = True, True
            elif g == "open" and res.success:
                robot._rd_holding, robot._rd_payload_pending = False, False
            elif "position" in args and res.executed:
                robot._rd_payload_pending = False
        if notes:
            res = dataclasses.replace(res, message="; ".join(notes) + "; " + (res.message or ""))
        return res

    adapter.execute = execute
    # relay flag so the keep-out never interferes with a hand-over
    _relay = r4._relay

    def relay(*a, **k):
        r4._relaying = True
        try:
            return _relay(*a, **k)
        finally:
            r4._relaying = False
    r4._relay = relay

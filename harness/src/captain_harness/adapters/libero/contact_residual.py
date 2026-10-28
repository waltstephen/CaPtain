"""Calibrated residual contact detection for the LIBERO wrist force sensor.

residual = f_world - bias - gain * F_osc, where F_osc = kp*(goal - ee) - kd*ee_vel is the operational-space
controller's own commanded force. Bias, gain and noise scales come from a per-episode free-space calibration,
not from task outcomes. Public sensor and controller state only; no simulator contacts or object state.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np

from captain_harness.adapters.robotics.contact_limits import (
    EMERGENCY_LIMIT_N, K_SIGMA, MIN_THRESHOLD_N, PERSIST_SUBSTEPS, STALL_WINDOW_S, STALL_PROGRESS_M, SUSTAIN_S,
    DEFAULT_LOAD_BUDGET_N, LEGACY_FORCE_LIMIT_N, IMMEDIATE_FACTOR)

MODES = ("guarded", "manipulation")


@dataclass
class ForceCalibration:
    bias_world_n: list
    gain: list
    sigma_axis_n: list
    norm_median_n: float
    norm_mad_n: float
    hold_norm_median_n: float
    hold_norm_mad_n: float
    method: str
    hold_samples: int
    motion_samples: int
    kp: list | None = None
    kd: list | None = None
    notes: str = ""
    # optional (RoboDojo X5, r6): 3x3 map from the grip site's per-tick displacement to the wrist force it causes in free
    # motion (joint-drive damping of a stiff joint-PD arm); subtracted before any threshold so speed is not read as contact
    velocity_matrix: list | None = None

    def to_json(self):
        return asdict(self)

    @classmethod
    def from_json(cls, data):
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})

    @property
    def guarded_threshold_n(self):
        return max(self.norm_median_n + K_SIGMA * self.norm_mad_n, MIN_THRESHOLD_N)

    @property
    def lateral_threshold_n(self):
        return max(K_SIGMA * max(self.sigma_axis_n), MIN_THRESHOLD_N)

    @property
    def stationary_threshold_n(self):
        return max(self.hold_norm_median_n + K_SIGMA * self.hold_norm_mad_n, MIN_THRESHOLD_N)


def motion_force(calibration, v_tick):
    if calibration.velocity_matrix is None or v_tick is None:
        return np.zeros(3)
    return np.asarray(calibration.velocity_matrix, float) @ np.asarray(v_tick, float)


def residual(f_world, f_osc, calibration, v_tick=None):
    return (np.asarray(f_world, float) - np.asarray(calibration.bias_world_n, float)
            - np.asarray(calibration.gain, float) * np.asarray(f_osc, float) - motion_force(calibration, v_tick))


def fit_velocity_calibration(hold_samples, motion_samples, *, kp=None, kd=None):
    """Offline fit (r6): bias from the stationary hold, f - bias = D v_tick by least squares on free-motion samples,
    thresholds from the remaining free-motion residual. Samples need 'f_world' and 'v_tick'."""
    fh = np.asarray([s["f_world"] for s in hold_samples], float).reshape(-1, 3)
    if len(fh) < 20 or len(motion_samples) < 100:
        raise ValueError("velocity calibration needs >= 20 stationary and >= 100 motion samples")
    bias = np.median(fh, axis=0)
    hold_med, hold_mad = _mad(np.linalg.norm(fh - bias, axis=1))
    F = np.asarray([s["f_world"] for s in motion_samples], float) - bias
    V = np.asarray([s["v_tick"] for s in motion_samples], float)
    X, *_ = np.linalg.lstsq(V, F, rcond=None)       # F ~ V @ X  ->  D = X.T
    D = X.T
    res = F - V @ X
    sigma_axis = 1.4826 * np.median(np.abs(res - np.median(res, axis=0)), axis=0)
    norm_med, norm_mad = _mad(np.linalg.norm(res, axis=1))
    r2 = 1.0 - float(np.sum(res ** 2)) / max(float(np.sum((F - F.mean(axis=0)) ** 2)), 1e-12)
    cal = ForceCalibration(bias.tolist(), [0.0, 0.0, 0.0], np.maximum(sigma_axis, 1e-3).tolist(), float(norm_med),
                           float(max(norm_mad, 1e-3)), float(hold_med), float(max(hold_mad, 1e-3)), "free_space_velocity_model",
                           int(len(fh)), int(len(F)), None if kp is None else list(map(float, kp)),
                           None if kd is None else list(map(float, kd)), velocity_matrix=D.tolist())
    cal.notes = f"velocity model R2={r2:.3f}"
    return cal, r2


def _mad(x):
    x = np.asarray(x, float)
    med = float(np.median(x))
    return med, float(1.4826 * np.median(np.abs(x - med)))


def fit_calibration(hold_samples, motion_samples, *, kp=None, kd=None):
    """hold/motion samples are dicts with 'f_world' and 'f_osc' 3-vectors (world frame)."""
    fh = np.asarray([s["f_world"] for s in hold_samples], float).reshape(-1, 3)
    if len(fh) < 20:
        raise ValueError("insufficient stationary samples for force calibration")
    bias = np.median(fh, axis=0)
    hold_res = fh - bias
    hold_med, hold_mad = _mad(np.linalg.norm(hold_res, axis=1))
    fm = np.asarray([s["f_world"] for s in motion_samples], float).reshape(-1, 3) if motion_samples else np.empty((0, 3))
    om = np.asarray([s["f_osc"] for s in motion_samples], float).reshape(-1, 3) if motion_samples else np.empty((0, 3))
    gain = np.zeros(3)
    method = "stationary_fallback"
    res = hold_res
    if len(fm) >= 100 and np.isfinite(om).all():
        for i in range(3):
            x, y = om[:, i], fm[:, i] - bias[i]
            denom = float(np.dot(x, x))
            gain[i] = float(np.clip(np.dot(x, y) / denom, 0.0, 1.0)) if denom > 1e-9 else 0.0
        res = fm - bias - gain * om
        method = "free_space_motion"
    elif len(fm) >= 20:
        # backends sampled once per control tick (not per physics substep) yield < 100 motion samples: no gain fit,
        # but the thresholds come from the measured free-motion force distribution instead of the stationary hold
        res = fm - bias
        method = "free_space_motion_noise"
    sigma_axis = 1.4826 * np.median(np.abs(res - np.median(res, axis=0)), axis=0)
    norm_med, norm_mad = _mad(np.linalg.norm(res, axis=1))
    inflate = 2.0 if method == "stationary_fallback" else 1.0
    return ForceCalibration(bias.tolist(), gain.tolist(), (np.maximum(sigma_axis, 1e-3) * inflate).tolist(),
                            float(norm_med), float(max(norm_mad, 1e-3) * inflate), float(hold_med), float(max(hold_mad, 1e-3)),
                            method, int(len(fh)), int(len(fm)),
                            None if kp is None else list(map(float, kp)), None if kd is None else list(map(float, kd)))


class ResidualContactDetector:
    """Substep-rate contact detector with two modes; thresholds derive from the calibration only."""

    def __init__(self, calibration, mode="guarded", load_budget_n=None, dt_s=0.002):
        if mode not in MODES:
            raise ValueError("contact_mode must be guarded or manipulation")
        self.cal = calibration
        self.mode = mode
        self.dt_s = float(dt_s)
        self.load_budget_n = float(load_budget_n if load_budget_n is not None else DEFAULT_LOAD_BUDGET_N)
        if not 0 < self.load_budget_n <= EMERGENCY_LIMIT_N:
            raise ValueError("load_budget_n must be within the emergency limit")
        self.threshold_n = calibration.guarded_threshold_n
        self.lateral_threshold_n = calibration.lateral_threshold_n
        self._persist = 0
        self._persist_lat = 0
        self._sustain = 0
        self.samples = 0
        self.peak_residual_n = 0.0
        self.peak_par_n = 0.0
        self.peak_lat_n = 0.0
        self.peak_contact_n = 0.0
        self.loaded = False
        self.last = None
        self.trigger = None

    def observe_substep(self, sample):
        if sample is None or sample.get("f_world") is None:
            return None
        self.samples += 1
        f = np.asarray(sample["f_world"], float)
        v = sample.get("v_tick")
        r = residual(f, sample.get("f_osc", np.zeros(3)), self.cal, v)
        contact = float(np.linalg.norm(f - np.asarray(self.cal.bias_world_n) - motion_force(self.cal, v)))
        rn = float(np.linalg.norm(r))
        u = sample.get("u_world")
        if u is not None and np.linalg.norm(u) > 1e-9:
            u = np.asarray(u, float) / np.linalg.norm(u)
            r_par = float(r @ u)
            r_lat = float(np.linalg.norm(r - r_par * u))
        else:
            r_par, r_lat = 0.0, rn
        self.peak_residual_n = max(self.peak_residual_n, rn)
        self.peak_par_n = max(self.peak_par_n, abs(r_par))
        self.peak_lat_n = max(self.peak_lat_n, r_lat)
        self.peak_contact_n = max(self.peak_contact_n, contact)
        self.loaded = abs(r_par) > self.threshold_n
        self.last = {"contact_n": contact, "residual_n": rn, "residual_par_n": r_par, "residual_lat_n": r_lat,
                     "osc_force_norm_n": float(np.linalg.norm(np.asarray(sample.get("f_osc", np.zeros(3)), float)))}
        reason = None
        # Detection first: a steep rise (impact) triggers on the first substep above IMMEDIATE_FACTOR x threshold,
        # a soft contact needs PERSIST_SUBSTEPS consecutive substeps above threshold. Emergency is the last resort.
        if self.mode == "guarded":
            self._persist = self._persist + 1 if rn > self.threshold_n else 0
            if rn > IMMEDIATE_FACTOR * self.threshold_n or self._persist >= PERSIST_SUBSTEPS:
                reason = "contact_detected"
        else:
            # Manipulation: contact along and across the motion is expected (handles, friction, supports).
            # Only the total contact load against the policy's budget (sustained) and the stall rule stop it.
            self._sustain = self._sustain + 1 if contact > self.load_budget_n else 0
            if self._sustain * self.dt_s >= SUSTAIN_S:
                reason = "load_budget_reached"
        if reason is None and contact > EMERGENCY_LIMIT_N:
            reason = "emergency_limit"
        if reason and self.trigger is None:
            self.trigger = {"reason": reason, **self.last, "sample_index": self.samples,
                            "threshold_n": self.threshold_n if self.mode == "guarded" else self.lateral_threshold_n,
                            "load_budget_n": self.load_budget_n, "emergency_exceeded": contact > EMERGENCY_LIMIT_N,
                            "persisted_substeps": self._persist if self.mode == "guarded" else self._sustain}
        return reason

    def observe_control_step(self, progress_m, window_s):
        """Manipulation stall: loaded but not progressing along the commanded direction."""
        if self.mode != "manipulation" or self.trigger is not None:
            return None
        if window_s + 1e-9 >= STALL_WINDOW_S and self.loaded and progress_m < STALL_PROGRESS_M:
            self.trigger = {"reason": "contact_stall", **(self.last or {}), "progress_m": float(progress_m),
                            "window_s": float(window_s), "load_budget_n": self.load_budget_n}
            return "contact_stall"
        return None

    def receipt(self):
        return {"mode": self.mode, "calibration_method": self.cal.method, "threshold_n": self.threshold_n,
                "lateral_threshold_n": self.lateral_threshold_n, "load_budget_n": self.load_budget_n,
                "emergency_limit_n": EMERGENCY_LIMIT_N, "samples": self.samples,
                "peak_residual_n": self.peak_residual_n, "peak_residual_par_n": self.peak_par_n,
                "peak_residual_lat_n": self.peak_lat_n, "peak_contact_n": self.peak_contact_n,
                "trigger": self.trigger,
                "scope": "calibrated residual against the controller model; contact identity and retention unknown"}


def calibrate_free_space(backend, *, hold_s=0.5, leg_m=0.015, speed_m_s=0.05):
    """Per-episode calibration: stationary hold, then a closed 1.5 cm triangle in free space at the default speed
    under the legacy guard. Falls back to stationary-only statistics when the motion cannot complete."""
    from .substep_force_guard import SubstepForceMonitor, active_force_monitor
    from .contact_motion import _pose
    from captain_harness.adapters.robotics.cartesian_sweeps import cartesian_sweep
    controller = backend._controller_gains()
    baseline = backend._eef_force_norm()
    collector = SubstepForceMonitor(initial_force=baseline if baseline is not None else 0.0,
                                    max_force=LEGACY_FORCE_LIMIT_N, retreat=False, collect_samples=True)
    hold = np.zeros(backend._action_dim)
    hold[-1] = 1.0 - 2.0 * backend._gripper_command_fraction
    with active_force_monitor(backend, collector):
        n_hold = int(round(hold_s * backend._control_frequency))
        # backends that sample once per (slow) control tick: keep holding until the fit has its 20 samples
        # (at most 4x the nominal hold); LIBERO already collects >= 20 within hold_s, so it is unaffected
        for i in range(4 * n_hold):
            if i >= n_hold and len(collector.samples) >= 20:
                break
            if not backend._step_action(hold):
                break
    hold_samples = list(collector.samples)
    motion_samples = []
    legs = []
    start = _pose(backend.get_robot_state().value)
    motion_ok = True
    # opt-in (backend._calibration_rise_limit_n, RoboDojo X5): free-motion wrist force of a stiff joint-PD arm exceeds
    # the legacy +5 N rise rule, which would abort the calibration triangle; LIBERO keeps 5 N
    previous_rise = getattr(backend, "_legacy_rise_limit_n", 5.0)
    backend._legacy_rise_limit_n = float(getattr(backend, "_calibration_rise_limit_n", previous_rise))
    for displacement in (np.array([leg_m, 0.0, 0.0]), np.array([0.0, leg_m, 0.0]), np.array([-leg_m, -leg_m, 0.0])):
        current = _pose(backend.get_robot_state().value)
        path = cartesian_sweep(current, displacement_m=displacement)
        sink = []
        result = backend._execute_contact_sweep(path, max_speed_m_s=speed_m_s, max_force_n=LEGACY_FORCE_LIMIT_N,
                                                max_duration_s=5.0, sample_sink=sink)
        value = result.value or {}
        legs.append({"displacement_m": displacement.tolist(), "reached": bool(value.get("reached")),
                     "stopped_reason": value.get("stopped_reason"), "samples": len(sink)})
        motion_samples.extend(sink)
        if not value.get("reached"):
            motion_ok = False
            break
    backend._legacy_rise_limit_n = previous_rise
    calibration = fit_calibration(hold_samples, motion_samples if motion_ok else [],
                                  kp=controller.get("kp"), kd=controller.get("kd"))
    end = _pose(backend.get_robot_state().value)
    calibration.notes = ("triangle returned within %.4f m of start" % float(np.linalg.norm(end[:3, 3] - start[:3, 3]))
                         if motion_ok else "free-space motion incomplete; stationary statistics inflated x2")
    backend._force_calibration = calibration
    return calibration, {"legs": legs, "hold_samples": len(hold_samples), "motion_samples": len(motion_samples),
                         "sim_steps": int(backend.simulation_steps)}

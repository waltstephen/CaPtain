"""Calibrated residual contact detection: synthetic calibration recovery, mode rules, stall, emergency limit."""
import numpy as np
import pytest
from captain_harness.adapters.libero.contact_residual import (ForceCalibration, ResidualContactDetector, fit_calibration, residual)
from captain_harness.adapters.robotics.contact_limits import EMERGENCY_LIMIT_N, PERSIST_SUBSTEPS, MIN_THRESHOLD_N
from captain_harness.adapters.libero.substep_force_guard import SubstepForceMonitor, monitored_step


def synthetic_samples(n_hold=300, n_move=1500, bias=(0.2, -0.1, -5.0), gain=(0.3, 0.3, 0.3), noise=0.03, seed=0):
    rng = np.random.default_rng(seed)
    hold = [{"f_world": np.array(bias) + rng.normal(0, noise, 3), "f_osc": np.zeros(3)} for _ in range(n_hold)]
    move = []
    for _ in range(n_move):
        f_osc = rng.normal(0, 3.0, 3)
        move.append({"f_world": np.array(bias) + np.array(gain) * f_osc + rng.normal(0, noise, 3), "f_osc": f_osc})
    return hold, move


def test_fit_recovers_bias_gain_and_noise_scale():
    hold, move = synthetic_samples()
    cal = fit_calibration(hold, move, kp=[150] * 3, kd=[24.5] * 3)
    assert cal.method == "free_space_motion"
    assert np.allclose(cal.bias_world_n, [0.2, -0.1, -5.0], atol=0.01)
    assert np.allclose(cal.gain, [0.3, 0.3, 0.3], atol=0.02)
    assert 0.02 < max(cal.sigma_axis_n) < 0.05
    assert cal.guarded_threshold_n == MIN_THRESHOLD_N  # noise is far below the 1 N floor
    back = ForceCalibration.from_json(cal.to_json())
    assert back == cal


def test_stationary_fallback_inflates_noise():
    hold, _ = synthetic_samples()
    cal = fit_calibration(hold, [])
    assert cal.method == "stationary_fallback" and cal.gain == [0.0, 0.0, 0.0]
    assert cal.motion_samples == 0


def _cal():
    hold, move = synthetic_samples()
    return fit_calibration(hold, move)


def test_guarded_mode_needs_persistence_and_ignores_controller_force():
    cal = _cal(); det = ResidualContactDetector(cal, mode="guarded", dt_s=0.002)
    bias = np.array(cal.bias_world_n)
    # a large commanded (controller) force explained by the model does not trigger
    f_osc = np.array([8.0, 0.0, 0.0])
    for _ in range(10):
        assert det.observe_substep({"f_world": bias + 0.3 * f_osc, "f_osc": f_osc, "u_world": None}) is None
    # a single-sample spike does not trigger; three persistent samples do
    assert det.observe_substep({"f_world": bias + [3.0, 0, 0], "f_osc": np.zeros(3)}) is None
    assert det.observe_substep({"f_world": bias, "f_osc": np.zeros(3)}) is None
    reasons = [det.observe_substep({"f_world": bias + [2.0, 0, 0], "f_osc": np.zeros(3)}) for _ in range(PERSIST_SUBSTEPS)]
    assert reasons[-1] == "contact_detected" and det.trigger["reason"] == "contact_detected"


def test_manipulation_mode_allows_load_and_stops_on_sustained_budget():
    cal = _cal(); bias = np.array(cal.bias_world_n)
    det = ResidualContactDetector(cal, mode="manipulation", load_budget_n=10.0, dt_s=0.002)
    u = np.array([1.0, 0.0, 0.0])
    for _ in range(50):  # 8 N along the motion: fine
        assert det.observe_substep({"f_world": bias + [8.0, 0, 0], "f_osc": np.zeros(3), "u_world": u}) is None
    assert det.loaded
    for _ in range(50):  # lateral 3 N on top: still fine in manipulation mode (handles and friction load sideways)
        assert det.observe_substep({"f_world": bias + [8.0, 3.0, 0], "f_osc": np.zeros(3), "u_world": u}) is None
    assert det.peak_lat_n > 2.9
    det2 = ResidualContactDetector(cal, mode="manipulation", load_budget_n=10.0, dt_s=0.002)
    r = None
    for _ in range(200):  # 12 N along the motion for > 0.25 s: budget
        r = det2.observe_substep({"f_world": bias + [12.0, 0, 0], "f_osc": np.zeros(3), "u_world": u}) or r
    assert r == "load_budget_reached"


def test_emergency_limit_in_every_mode_and_stall_rule():
    cal = _cal(); bias = np.array(cal.bias_world_n)
    det = ResidualContactDetector(cal, mode="guarded")
    r = det.observe_substep({"f_world": bias + [EMERGENCY_LIMIT_N + 1, 0, 0], "f_osc": np.zeros(3), "u_world": [1, 0, 0]})
    assert r == "contact_detected" and det.trigger["emergency_exceeded"] is True
    det = ResidualContactDetector(cal, mode="manipulation")
    # load purely along the motion above the emergency limit: no lateral/budget rule has fired yet -> emergency
    assert det.observe_substep({"f_world": bias + [EMERGENCY_LIMIT_N + 1, 0, 0], "f_osc": np.zeros(3), "u_world": [1, 0, 0]}) == "emergency_limit"
    # impact: a single substep far above threshold stops guarded mode immediately
    det = ResidualContactDetector(cal, mode="guarded")
    assert det.observe_substep({"f_world": bias + [6.0, 0, 0], "f_osc": np.zeros(3)}) == "contact_detected"
    det = ResidualContactDetector(cal, mode="manipulation")
    det.observe_substep({"f_world": bias + [6.0, 0, 0], "f_osc": np.zeros(3), "u_world": [1, 0, 0]})
    assert det.observe_control_step(progress_m=0.01, window_s=0.5) is None
    assert det.observe_control_step(progress_m=0.0005, window_s=0.5) == "contact_stall"
    with pytest.raises(ValueError):
        ResidualContactDetector(cal, mode="manipulation", load_budget_n=EMERGENCY_LIMIT_N + 1)


class Env:
    model_timestep = .002
    def __init__(self, samples):
        self.samples = samples; self.i = 0; self.actions = []
    def _pre_action(self, action, policy_step=False):
        self.actions.append((list(action), policy_step))
    def _update_observables(self, force=False): pass
    def step(self, action):
        for i, s in enumerate(self.samples):
            self._pre_action(action, policy_step=i == 0); self.cur = s; self.i += 1; self._update_observables()
        return {"state": self.i}, 0, False, {}


def test_monitor_delegates_to_detector_and_cancels_once():
    cal = _cal(); bias = np.array(cal.bias_world_n)
    contact = {"f_world": bias + [2.5, 0, 0], "f_osc": np.zeros(3), "u_world": None}
    quiet = {"f_world": bias, "f_osc": np.zeros(3), "u_world": None}
    env = Env([quiet, contact, contact, contact, quiet, quiet]); env.cur = quiet
    det = ResidualContactDetector(cal, mode="guarded")
    monitor = SubstepForceMonitor(initial_force=5., max_force=60., retreat=False, detector=det)
    cancelled = []
    monitored_step(env, np.array([.2, 0, 0, 0, 0, 0, 1.]), monitor, read_force=lambda: float(np.linalg.norm(env.cur["f_world"])),
                   cancel=lambda: cancelled.append(True) or {"status": "cancelled"}, read_sample=lambda: env.cur)
    assert monitor.trigger["reason"] == "contact_detected" and len(cancelled) == 1
    assert env.actions[-1][0] == [0.] * 7
    # legacy monitor without detector is unchanged: no trigger on a 2.5 N rise
    env2 = Env([quiet, contact, contact, contact]); env2.cur = quiet
    legacy = SubstepForceMonitor(initial_force=float(np.linalg.norm(bias)), max_force=18., retreat=False)
    monitored_step(env2, np.ones(7), legacy, read_force=lambda: float(np.linalg.norm(env2.cur["f_world"])), cancel=lambda: None)
    assert legacy.trigger is None

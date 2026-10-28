"""XR1 action backend: the LIBERO backend's control logic over XR1FacadeEnv.

Only the methods that read simulator internals the twin cannot imitate are
overridden: camera calibration (real intrinsics, twin/bridge extrinsics),
target cancellation and bounded gripper opening (no robosuite actuator codec).
"""
from __future__ import annotations

import contextlib

import numpy as np

from .mjsim import LINK_TO_OPTICAL
from captain_harness.adapters.libero.recorded_backend import ClosingMonitoredPossessionActionBackend



class XR1ActionBackend(ClosingMonitoredPossessionActionBackend):
    """LIBERO-shaped backend whose `env` is an XR1FacadeEnv."""

    def __init__(self, env, *, camera_link_to_optical=None, **kwargs) -> None:
        self._xr1_env = env
        self._camera_link_to_optical = {name: np.asarray(value, float).reshape(4, 4)
                                        for name, value in (camera_link_to_optical or {}).items()}
        super().__init__(env, controller="OSC_POSE", **kwargs)

    def camera_extrinsics(self, name: str) -> np.ndarray:
        frame = self._xr1_env.frame(name)
        if frame.optical_in_robot_base is not None:
            return np.asarray(frame.optical_in_robot_base, float).reshape(4, 4)
        link = self._xr1_env.camera_link_pose(name)
        return link @ self._camera_link_to_optical.get(name, LINK_TO_OPTICAL)

    def _camera_calibration(self, name: str):
        frame = self._xr1_env.frame(name)
        return (tuple(np.asarray(frame.intrinsics, float).ravel()),
                tuple(self.camera_extrinsics(name).ravel()))

    def cancel_direct_targets(self, *, cancel_gripper=False, max_open_fraction_correction=None):
        receipt = {"simulation_steps": self.simulation_steps}
        try:
            self._xr1_env.robots[0].controller.reset_goal()
            receipt["arm_target"] = "reset_to_measured_pose"
            if cancel_gripper:
                measured = self.get_robot_state().value.gripper_open_fraction
                self._gripper_command_fraction = float(measured)
                receipt["gripper_target"] = {"status": "replaced", "measured_open_fraction": measured}
            else:
                receipt["gripper_target"] = "preserved"
            receipt["status"] = "cancelled"
        except Exception as exc:
            receipt.update(status="unconfirmed", error=type(exc).__name__)
            self._physical_execution_unknown = True
        return receipt

    def _replace_gripper_target_with_measured_position(self, *, max_open_fraction_correction=None):
        # LIBERO replaces the robosuite finger target with the measured fingers after a closure stop; the facade's
        # gripper action is an absolute open fraction, so the equivalent is commanding the measured opening. Without
        # this the LIBERO codec check raised, the stop was left "unverified" and every later close-direction step
        # (including the neutral hold) was rejected as motion_guard_rejected until an explicit open (R5 bug F1).
        if not getattr(self._xr1_env._driver, "closure_hold_fix", False):
            return super()._replace_gripper_target_with_measured_position(
                max_open_fraction_correction=max_open_fraction_correction)
        measured = float(self.get_robot_state().value.gripper_open_fraction)
        self._gripper_command_fraction = measured
        return {"status": "replaced", "measured_open_fraction": measured, "codec": "xr1_open_fraction_action"}

    @contextlib.contextmanager
    def _bounded_gripper_opening(self, max_open_fraction_increase):
        if (isinstance(max_open_fraction_increase, bool)
                or not isinstance(max_open_fraction_increase, (int, float))
                or not np.isfinite(max_open_fraction_increase)
                or not 0 < max_open_fraction_increase <= .25):
            raise ValueError("invalid release command budget")
        start = 1. - 2. * float(self.get_robot_state().value.gripper_open_fraction)
        limit = float(np.clip(start - 2. * max_open_fraction_increase, -1., 1.))
        receipt = {"codec": "xr1_open_fraction_action", "start_action": start, "bounded_action": limit,
                   "scope": "commanded aperture only; dynamics can overshoot between samples"}
        with self._xr1_env.bounded_gripper_actions(min(start, limit), max(start, limit)):
            yield receipt

"""Unregistered opening-only unloading with a fixed measured joint posture."""
from dataclasses import replace
import math

import numpy as np

from captain_harness.ports.capability import CapabilityResult, FailureCategory


class NativeGripperReleaseMixin:
    def _set_gripper(self, fraction):
        if fraction != 1.:
            return super()._set_gripper(fraction)
        if self.controller != 'JOINT_POSITION':
            return CapabilityResult.failed(FailureCategory.CONTROLLER_REJECTED,
                'native release requires JOINT_POSITION')
        first = self.simulation_steps
        trace, reason, diagnostic = [], None, None
        drive = self._native_drive()
        initial_force, floor, target = None, None, None

        def sample():
            nonlocal initial_force, floor, reason
            force = self._eef_force_norm()
            force = float(force) if force is not None and math.isfinite(force) else None
            trace.append({'step': self.simulation_steps, 'force_n': force})
            if force is None:
                reason = 'release_force_unavailable'
            elif floor is None:
                initial_force = floor = force
            elif force >= floor+5.:
                reason = 'release_force_increased'
            else:
                floor = min(floor, force)
            return reason is None

        try:
            target = np.asarray(self.get_robot_state().value.joint_positions_rad).copy()
            if sample():
                for _ in range(self._gripper_steps):
                    q = np.asarray(self.get_robot_state().value.joint_positions_rad)
                    if np.max(np.abs(target-q)) > .08:
                        reason = 'release_posture_tracking_error'
                        break
                    if not self._step(target-q, 1.):
                        reason = self._last_step_rejection or 'controller_stopped'
                        break
                    self._gripper_command_fraction = self._gripper_open_fraction = 1.
                    if not sample():
                        break
                else:
                    if trace[-1]['force_n'] >= 18.:
                        reason = 'release_still_loaded'
                    elif self.get_robot_state().value.gripper_open_fraction < .95:
                        reason = 'release_aperture_not_reached'
        except Exception as exc:
            reason = 'release_backend_error'
            diagnostic = f'{type(exc).__name__}: {exc}'
        executed = self.simulation_steps > first
        value = {'stopped_reason': reason or 'release_complete', 'initial_force_n': initial_force,
            'observed_force_n': trace[-1]['force_n'] if trace else None,
            'control_steps': self.simulation_steps-first, 'samples': trace,
            'force_increase_limit_n': 5., 'forward_motion_force_limit_n': 18.,
            'scope': 'Opening only with measured posture hold; force availability and unloading required.',
            'limitation': 'Sampled force checks are not a physical force bound or contact identity.'}
        if diagnostic:
            value['diagnostic'] = diagnostic
        if reason:
            category = FailureCategory.BACKEND_ERROR if reason=='release_backend_error' else FailureCategory.CONTROLLER_REJECTED
            result = CapabilityResult.failed(category, reason, executed=executed, value=value)
        else:
            result = replace(CapabilityResult.ok(value), executed=executed)
        if executed:
            drive._saved_motion = None
            drive._hold_reference = ({'joints': target.copy(), 'end_step': self.simulation_steps}
                                     if result.success else None)
        self._last_native_gripper_execution = result
        return replace(result, value=None)

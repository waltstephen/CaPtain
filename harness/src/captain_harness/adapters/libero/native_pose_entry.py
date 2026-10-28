"""Unregistered native implementations of staged alignment and pose servo."""
from dataclasses import replace
import math

import numpy as np
from scipy.spatial.transform import Rotation

from captain_harness.ports.capability import CapabilityResult, FailureCategory
from .joint_cartesian import _pose


def _target_pose(position, quaternion):
    p = np.asarray(position, dtype=float)
    q = np.asarray(quaternion, dtype=float)
    if (p.shape != (3,) or q.shape != (4,) or not np.isfinite(p).all()
            or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12):
        raise ValueError('pose requires finite position and nonzero quaternion')
    pose = np.eye(4)
    pose[:3, 3] = p
    pose[:3, :3] = Rotation.from_quat(np.roll(q/np.linalg.norm(q), -1)).as_matrix()
    return pose


class NativePoseEntryMixin:
    def align_pregrasp(self, position, quaternion, approach_direction_world, *,
                       approach_distance_m=.06, clearance_m=.09):
        first = self.simulation_steps
        try:
            target = _target_pose(position, quaternion)
            approach = np.asarray(approach_direction_world, dtype=float)
            if approach.shape != (3,) or not np.isfinite(approach).all() or np.linalg.norm(approach) < 1e-9:
                raise ValueError('approach direction must be finite and nonzero')
            for value, lower in ((approach_distance_m, .005), (clearance_m, .02)):
                if isinstance(value, bool) or not math.isfinite(value) or not lower <= value <= .2:
                    raise ValueError('invalid approach or clearance distance')
            pregrasp = target.copy()
            pregrasp[:3, 3] -= approach/np.linalg.norm(approach)*approach_distance_m
        except (ValueError, TypeError) as exc:
            return CapabilityResult.failed(FailureCategory.INVALID_ARGUMENT, str(exc))
        opened = self._set_gripper(1.)
        if not opened.success:
            return opened
        start = _pose(self.get_robot_state().value)
        hover = pregrasp.copy()
        hover[2, 3] = max(pregrasp[2, 3]+clearance_m, start[2, 3])
        clearance = start.copy(); clearance[2, 3] = hover[2, 3]
        translated = clearance.copy(); translated[:3, 3] = hover[:3, 3]
        path = [start, clearance, translated, hover, pregrasp]
        result = self._native_drive().execute_sweep(path, max_speed_m_s=.05,
            max_angular_speed_rad_s=.5,
            max_duration_s=min(30., self._cartesian_max_steps/self._control_frequency))
        value = result.value or {}
        return replace(result, executed=self.simulation_steps > first, value={
            'converged': result.success, 'phase_reached': 'pregrasp' if result.success else 'native_sweep_stopped',
            'failed_phase': None if result.success else 'native_staged_sweep',
            'position_residual_m': value.get('position_residual_m'),
            'orientation_residual_rad': value.get('orientation_residual_rad'),
            'convergence_reason': 'native_staged_pregrasp_reached' if result.success else result.message,
            'pregrasp_position_m': pregrasp[:3, 3].tolist(), 'hover_position_m': hover[:3, 3].tolist(),
            'controller_execution': value,
            'path_scope': 'bounded clearance stages; local IK only, no fresh scene clearance proof'})

    def servo_to(self, position, quaternion, *, max_duration_s, tolerance_m):
        try:
            _target_pose(position, quaternion)
            for value, upper in ((max_duration_s, 30.), (tolerance_m, .1)):
                if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= upper:
                    raise ValueError('invalid servo duration or position tolerance')
        except (ValueError, TypeError) as exc:
            return CapabilityResult.failed(FailureCategory.INVALID_ARGUMENT, str(exc))
        first = self.simulation_steps
        result = self._native_drive().move(position, quaternion, max_speed_m_s=.02,
                                            max_duration_s=max_duration_s, position_tolerance_m=tolerance_m)
        actual = np.asarray(self.get_robot_state().value.end_effector_position_m)
        residual = float(np.linalg.norm(actual-position))
        if result.success and residual > tolerance_m:
            result = CapabilityResult.failed(FailureCategory.CONTROLLER_REJECTED,
                'native_endpoint_outside_public_tolerance', executed=result.executed, value=result.value)
        return replace(result, value={'reached': result.success, 'final_error_m': residual,
            'iterations': self.simulation_steps-first,
            'aborted_reason': None if result.success else result.failure_category.value,
            'controller_execution': result.value})

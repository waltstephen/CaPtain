"""Unregistered native implementation of the sensor robot method contract."""
from dataclasses import replace
import math

import numpy as np

from captain_harness.ports.capability import CapabilityResult, FailureCategory
from .backend import LiberoActionBackend
from .joint_cartesian import _errors, _pose
from .joint_path_guard import GuardedJointPathMixin
from .joint_sweep import NativeJointSweep
from .possession_monitor import PossessionMonitorMixin


class NativeSensorMotionMixin:
    def _native_drive(self):
        drive = getattr(self, '_native_sensor_drive', None)
        if drive is None:
            drive = NativeJointSweep(self)
            self._native_sensor_drive = drive
        return drive

    def _native_hold(self, duration_s, **options):
        drive = self._native_drive()
        first = self.simulation_steps
        try:
            return drive.hold(duration_s, **options)
        except Exception as exc:
            drive._hold_reference = None
            drive._saved_motion = None
            return CapabilityResult.failed(FailureCategory.BACKEND_ERROR,
                f'{type(exc).__name__}: {exc}', executed=self.simulation_steps > first,
                value={'stopped_reason': 'backend_execution_failed',
                       'control_steps': self.simulation_steps-first})

    def move_to_pose(self, position_xyz, quaternion_wxyz, *, position_only=False):
        if position_only:
            quaternion_wxyz = self.get_robot_state().value.end_effector_quaternion_wxyz
        return self._native_drive().move(position_xyz, quaternion_wxyz,
            max_speed_m_s=.05, max_duration_s=min(30., self._cartesian_max_steps/self._control_frequency))

    def move_to_pose_monitored(self, position, quaternion):
        return self._native_drive().move(position, quaternion, max_speed_m_s=.02,
            max_duration_s=min(30., self._cartesian_max_steps/self._control_frequency))

    def _execute_contact_sweep(self, path, *, max_speed_m_s=.02,
            max_angular_speed_rad_s=.2, max_force_n=18., max_duration_s=10.,
            retreat=False, position_tolerance_m=.002, orientation_tolerance_rad=.01,
            position_only=False, command_profile='bounded_setpoints'):
        if retreat or command_profile != 'bounded_setpoints':
            return CapabilityResult.failed(FailureCategory.INVALID_ARGUMENT,
                'native motion requires bounded setpoints; retreat uses the recorded joint route')
        for value, limit in ((position_tolerance_m, .1), (orientation_tolerance_rad, .5)):
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= limit:
                return CapabilityResult.failed(FailureCategory.INVALID_ARGUMENT, 'invalid pose tolerance')
        result = self._native_drive().execute_sweep(path, max_speed_m_s=max_speed_m_s,
            max_angular_speed_rad_s=max_angular_speed_rad_s, max_force_n=max_force_n,
            max_duration_s=max_duration_s, position_tolerance_m=position_tolerance_m,
            orientation_tolerance_rad=orientation_tolerance_rad)
        value = {**(result.value or {}), 'position_tolerance_m': position_tolerance_m,
                 'orientation_tolerance_rad': orientation_tolerance_rad}
        if result.success and (value['position_residual_m'] > position_tolerance_m
                or value['orientation_residual_rad'] > orientation_tolerance_rad):
            return CapabilityResult.failed(FailureCategory.CONTROLLER_REJECTED,
                'native_endpoint_outside_public_tolerance', executed=result.executed,
                value={**value, 'reached': False, 'stopped_reason': 'native_endpoint_outside_public_tolerance'})
        return replace(result, value=value)

    def retreat_last_motion(self, max_distance_m=.05, max_duration_s=10.):
        return self._native_drive().retreat_last_motion(max_distance_m, max_duration_s,
            position_tolerance_m=.002, orientation_tolerance_rad=.01)

    def hold_position(self, duration_s=.5):
        if (isinstance(duration_s, bool) or not isinstance(duration_s, (float, int))
                or not math.isfinite(duration_s) or not 0 < duration_s <= 30.):
            return CapabilityResult.failed(FailureCategory.INVALID_ARGUMENT, 'duration_s must be in (0, 30]')
        first = self.simulation_steps
        start = _pose(self.get_robot_state().value)
        result = self._native_hold(duration_s)
        pe, ae = _errors(_pose(self.get_robot_state().value), start)
        return replace(result, value={
            'requested_duration_s': duration_s, 'elapsed_s': (self.simulation_steps-first)/self._control_frequency,
            'start_time_s': first/self._control_frequency, 'end_time_s': self.simulation_steps/self._control_frequency,
            'steps_executed': self.simulation_steps-first, 'gripper_command_fraction': self._gripper_command_fraction,
            'pose_hold': {'reference': 'end_effector', 'position_residual_m': pe,
                'orientation_residual_rad': ae, 'position_tolerance_m': .004,
                'orientation_tolerance_rad': .05, 'within_tolerance': pe <= .004 and ae <= .05},
            'controller_execution': result.value})

    def _set_gripper(self, fraction):
        result = self._native_hold(self._gripper_steps/self._control_frequency, gripper_fraction=fraction)
        self._last_native_gripper_execution = result
        # Opening/closing shares the existing None-valued robot command API.
        return replace(result, value=None)

    def descend_and_close(self, position, quaternion, *, max_contact_force=None):
        drive = self._native_drive()
        force = 18. if max_contact_force is None else max_contact_force
        if (isinstance(force, bool) or not isinstance(force, (int, float))
                or not math.isfinite(force) or not 0 < force <= 18.):
            return CapabilityResult.failed(FailureCategory.INVALID_ARGUMENT,
                'max_contact_force must be finite and in (0, 18]',
                value={'closure_executed': False, 'grasp_outcome': 'not_attempted'})
        start = self.get_robot_state().value
        baseline = self._eef_force_norm()
        descent = drive.move(position, quaternion, max_speed_m_s=.01,
            max_duration_s=min(30., self._cartesian_max_steps/self._control_frequency), max_force_n=force)
        state = self.get_robot_state().value
        # Invalid arguments must not be converted into an exception while
        # assembling optional geometric diagnostics.
        try:
            target = np.asarray(position, dtype=float).reshape(3)
            valid_target = bool(np.isfinite(target).all())
        except (TypeError, ValueError):
            valid_target = False
        value = {
            'contacted': (descent.value or {}).get('stopped_reason') == 'contact_force',
            'contact_identity': 'unconfirmed', 'closure_executed': False,
            'grasp_outcome': 'not_attempted', 'force_limit_n': force,
            'force_increase_limit_n': 5., 'baseline_force_n': baseline,
            'observed_force_n': self._eef_force_norm(),
            'gripper_open_fraction': float(state.gripper_open_fraction),
            'stopped_reason': (descent.value or {}).get('stopped_reason'),
            'controller_execution': descent.value,
            'contact_travel_m': float(np.linalg.norm(np.asarray(state.end_effector_position_m)
                - start.end_effector_position_m)),
            'observed_eef_position_m': list(state.end_effector_position_m)}
        if valid_target:
            residual = target-state.end_effector_position_m
            value.update(requested_eef_position_m=target.tolist(), remaining_vector_m=residual.tolist(),
                         position_residual_m=float(np.linalg.norm(residual)))
        if not descent.success:
            return replace(descent, value=value)
        closed = self._native_hold(self._gripper_steps/self._control_frequency, gripper_fraction=0., max_force_n=force)
        value.update(closure_executed=closed.executed, closure_execution=closed.value)
        gripper = self.gripper_state()
        if not closed.success or not gripper.success:
            failed = closed if not closed.success else gripper
            return replace(failed, executed=descent.executed or closed.executed, value=value)
        final = self.get_robot_state().value
        pe, ae = self._pose_residuals(position, quaternion)
        value.update(grasp_outcome='object_detected' if gripper.value['object_detected'] else 'empty_close',
            object_detected=gripper.value['object_detected'], gripper_open_fraction=gripper.value['open_fraction'],
            observed_eef_position_m=list(final.end_effector_position_m), position_residual_m=pe,
            orientation_residual_rad=ae, observed_force_n=self._eef_force_norm())
        return replace(CapabilityResult.ok(value), executed=descent.executed or closed.executed)

    def home(self):
        return self._native_drive().move(self._home_eef_position, self._home_eef_quaternion,
            max_speed_m_s=.05, max_duration_s=min(30., self._home_max_steps/self._control_frequency))


class NativeSensorActionBackend(NativeSensorMotionMixin, GuardedJointPathMixin,
                               PossessionMonitorMixin, LiberoActionBackend):
    """Candidate only: no production factory or manifest imports this class."""
    def __init__(self, *args, **kwargs):
        if kwargs.get('controller', 'JOINT_POSITION') != 'JOINT_POSITION':
            raise ValueError('NativeSensorActionBackend requires JOINT_POSITION')
        super().__init__(*args, **{**kwargs, 'controller': 'JOINT_POSITION'})

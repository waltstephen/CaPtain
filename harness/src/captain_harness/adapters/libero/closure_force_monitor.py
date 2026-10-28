"""Sampled closing-force guard and post-closure evidence for sensor motion."""
from dataclasses import replace
import math
import numpy as np

from captain_harness.ports.capability import CapabilityResult, FailureCategory


class ClosureForceMonitorMixin:
    def _closure_force(self):
        try:
            force = self._eef_force_norm()
            return float(force) if force is not None and math.isfinite(force) else None
        except Exception:
            return None

    def _closure_pose_sample(self):
        try:
            result = self.get_robot_state()
            if not result.success or result.value is None:
                return None
            state = result.value
            p = np.asarray(state.end_effector_position_m, dtype=float)
            q = np.asarray(state.end_effector_quaternion_wxyz, dtype=float)
            aperture = float(state.gripper_open_fraction)
            if (p.shape != (3,) or q.shape != (4,) or not np.isfinite(p).all()
                    or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-10
                    or not math.isfinite(aperture)):
                return None
            return {'eef_position_m': p.tolist(),
                'eef_quaternion_wxyz': (q / np.linalg.norm(q)).tolist(),
                'gripper_open_fraction': aperture}
        except Exception:
            return None

    @staticmethod
    def _closure_motion_summary(samples):
        valid = [s['pose'] for s in samples if s.get('pose') is not None]
        initial = samples[0].get('pose') if samples else None
        final = samples[-1].get('pose') if samples else None
        summary = {'status': 'unavailable', 'valid_samples': len(valid),
            'total_samples': len(samples), 'initial_pose': initial, 'final_pose': final,
            'final_translation_m': None, 'max_sampled_translation_m': None,
            'final_rotation_rad': None, 'max_sampled_rotation_rad': None,
            'physical_retention_verified': False,
            'scope': 'Measured EEF drift relative to the pre-closure pose, excluding descent. Sampled maxima do not bound unsampled motion; no target contact or retention proof.'}
        if initial is None or len(valid) < 2:
            return summary
        def drift(pose):
            distance = float(np.linalg.norm(np.asarray(pose['eef_position_m']) - initial['eef_position_m']))
            dot = abs(float(np.dot(pose['eef_quaternion_wxyz'], initial['eef_quaternion_wxyz'])))
            return distance, float(2*np.arccos(np.clip(dot, 0., 1.)))
        deltas = [drift(pose) for pose in valid]
        summary.update(status='measured' if len(valid) == len(samples) else 'partial',
            max_sampled_translation_m=max(d[0] for d in deltas),
            max_sampled_rotation_rad=max(d[1] for d in deltas))
        if final is not None:
            summary['final_translation_m'], summary['final_rotation_rad'] = drift(final)
        return summary

    def _stop_closing_target(self):
        self._gripper_command_fraction = .5
        try:
            replacement = self._replace_gripper_target_with_measured_position()
            if replacement.get('status') != 'replaced':
                raise ValueError('driver did not confirm gripper target replacement')
            self._gripper_target_cancel_error = None
        except Exception as exc:
            self._gripper_target_cancel_error = f'{type(exc).__name__}: {exc}'
            replacement = {'status': 'unavailable', 'error': self._gripper_target_cancel_error}
        return {
            'native_action': 0.,
            'effect': ('hold_measured_joint_target' if replacement['status'] == 'replaced'
                       else 'physics_advance_inhibited_until_explicit_open'),
            'reason': 'monitored_closure_interrupted',
            'target_replacement': replacement,
            'limitation': 'Target replacement cancels the outstanding closing command, not inertia or contact loads; stationary fingers and stable possession require observation.'}

    def _set_gripper(self, fraction):
        self._last_closure_force_monitor = None
        if fraction != 0.:
            # An opening command releases load; it is not forward approach.
            return super()._set_gripper(fraction)
        first = self.simulation_steps
        baseline = self._closure_force()
        limit = getattr(self, '_closure_requested_force_limit', 18.)
        calibration = (getattr(self, '_force_calibration', None)
            if getattr(self, '_contact_detector_mode', 'legacy') == 'residual' else None)
        increase_limit = 5.
        sustain_samples = 1
        if calibration is not None:
            from captain_harness.adapters.robotics.contact_limits import EMERGENCY_LIMIT_N
            # Closing on an object is contact by design: keep the legacy +5 N rise rule for pushing into the
            # support, but the absolute cap becomes the emergency limit instead of 18 N.
            limit = EMERGENCY_LIMIT_N
            # opt-in per robot (RoboDojo X5: stiff joint PD, a rim squeeze reads 11-14 N); LIBERO keeps 5 N
            increase_limit = float(getattr(self, '_closure_rise_limit_n', 5.))
            sustain_samples = 2
        over = {'count': 0}
        outer = getattr(self, '_motion_guard', None)
        monitor = {'samples': [], 'reason': None, 'force_limit_n': limit,
            'force_increase_limit_n': increase_limit, 'initial_force_n': baseline,
            'detector': 'residual_calibrated' if calibration is not None else 'legacy',
            'scope': 'Sampled wrist force during closing; no contact identity or physical force bound.'}

        def guard():
            if monitor['reason'] is not None:
                return False
            if monitor['samples'] and monitor['samples'][-1]['step'] == self.simulation_steps:
                return True
            force = self._closure_force()
            monitor['samples'].append({'step': self.simulation_steps, 'force_n': force,
                                       'pose': self._closure_pose_sample()})
            if force is None:
                monitor['reason'] = 'closure_force_unavailable'
            elif force >= limit:
                monitor['reason'] = 'closure_force_limit'
            elif baseline is not None and force-baseline >= increase_limit and (
                    over.__setitem__('count', over['count'] + 1) or over['count'] >= sustain_samples):
                monitor['reason'] = 'closure_force_increase'
            elif outer is not None:
                try:
                    if not outer():
                        monitor['reason'] = 'outer_motion_guard_rejected'
                except Exception as exc:
                    monitor['reason'] = 'closure_guard_failed'
                    monitor['diagnostic'] = f'{type(exc).__name__}: {exc}'
            return monitor['reason'] is None

        self._motion_guard = guard
        try:
            if not guard():
                result = CapabilityResult.failed(FailureCategory.CONTROLLER_REJECTED, monitor['reason'])
            else:
                try:
                    result = super()._set_gripper(fraction)
                except Exception as exc:
                    result = CapabilityResult.failed(FailureCategory.BACKEND_ERROR,
                        f'{type(exc).__name__}: {exc}', executed=self.simulation_steps > first)
                guard()
            monitor['control_steps'] = self.simulation_steps-first
            monitor['closure_motion'] = self._closure_motion_summary(monitor['samples'])
            self._last_closure_force_monitor = monitor
            if monitor['reason'] is not None:
                category = (FailureCategory.BACKEND_ERROR if monitor['reason']=='closure_guard_failed'
                            else FailureCategory.CONTROLLER_REJECTED)
                if monitor['control_steps'] > 0:
                    # Replace the accumulated target, then carry neutral input.
                    # Neutral alone would retain the rejected closing target.
                    monitor['post_stop_gripper_command'] = self._stop_closing_target()
                return replace(result, success=False, failure_category=category,
                    message=monitor['reason'], executed=self.simulation_steps > first,
                    diagnostics={**(result.diagnostics or {}), 'closure_force_monitor': monitor})
            return replace(result, executed=self.simulation_steps > first,
                diagnostics={**(result.diagnostics or {}), 'closure_force_monitor': monitor})
        finally:
            self._motion_guard = outer

    def descend_and_close(self, *args, max_contact_force=None, **kwargs):
        limit = 18. if max_contact_force is None else max_contact_force
        if (isinstance(limit, bool) or not isinstance(limit, (float, int))
                or not math.isfinite(limit) or not 0 < limit <= 18.):
            return CapabilityResult.failed(FailureCategory.INVALID_ARGUMENT,
                'max_contact_force must be finite and in (0, 18]',
                value={'closure_executed': False})
        previous = getattr(self, '_closure_requested_force_limit', 18.)
        self._closure_requested_force_limit = limit
        self._last_closure_force_monitor = None
        try:
            result = super().descend_and_close(*args, max_contact_force=max_contact_force, **kwargs)
            monitor = self._last_closure_force_monitor
            value = {**(result.value if isinstance(result.value, dict) else {}),
                'observed_force_n': self._closure_force(),
                'force_measurement_phase': 'after_descend_and_close_return'}
            if monitor is not None:
                value.update(closure_force_monitor=monitor,
                             closure_executed=monitor['control_steps'] > 0)
                if (not result.success and (monitor['reason'] is not None
                        or value.get('failed_phase') != 'post_closure_measurement')):
                    deadline_stop = (result.failure_category is FailureCategory.TIMEOUT
                        and (result.diagnostics or {}).get('deadline'))
                    value.update(stopped_reason=monitor['reason'] or
                                 ('deadline_expired' if deadline_stop else 'closure_backend_failed'),
                                 grasp_outcome='closure_interrupted', contact_identity='unknown')
                # The legacy driver drops descent payload on closure failure.
                # Recover its measured route and final pose without inventing
                # an attachment or a permission to reverse after closure.
                if not isinstance(result.value, dict) and hasattr(self, 'get_robot_state'):
                    import numpy as np
                    state = self.get_robot_state().value
                    requested = np.asarray(args[0] if args else kwargs['position'], dtype=float)
                    residual = requested-state.end_effector_position_m
                    value.update(observed_eef_position_m=list(state.end_effector_position_m),
                        requested_eef_position_m=requested.tolist(), remaining_vector_m=residual.tolist(),
                        position_residual_m=float(np.linalg.norm(residual)),
                        gripper_open_fraction=float(state.gripper_open_fraction))
                    recording = getattr(self, '_last_descent_recording', None)
                    if recording:
                        from .contact_motion import _quaternion
                        trajectory = [{'step': step, 'eef_position_m': np.asarray(pose)[:3, 3].tolist(),
                            'eef_quaternion_wxyz': _quaternion(np.asarray(pose)).tolist(), 'force_n': force}
                            for step, pose, force in zip(recording['steps'], recording['poses'], recording['forces_n'])]
                        value['controller_execution'] = {'trajectory': trajectory, 'retreat_available': False,
                            'control_steps': recording['last_step']-recording['start_step'],
                            'scope': 'Measured descent before an interrupted closure; no reverse-route permission.'}
            return replace(result, value=value)
        finally:
            self._closure_requested_force_limit = previous

"""Opt-in possession continuity from the existing public gripper register.

The sensor-motion environment factory composes this monitor. It detects
loss; it cannot prove target identity or prevent a mechanically weak grasp.
"""
from dataclasses import replace
import math

from captain_harness.ports.capability import CapabilityResult, FailureCategory


class PossessionMonitorMixin:
    def _possession_sample(self):
        try:
            result = self.gripper_state()
        except Exception as exc:
            return {'available': False, 'object_detected': None,
                    'message': f'{type(exc).__name__}: {exc}'}
        value = result.value if result.success and isinstance(result.value, dict) else {}
        detected = value.get('object_detected')
        if not isinstance(detected, bool):
            return {'available': False, 'object_detected': None,
                    'message': result.message or 'possession register unavailable'}
        return {'available': True, 'object_detected': detected,
                'open_fraction': value.get('open_fraction')}

    def _record_aperture_sample(self, monitor, sample):
        aperture = sample.get('open_fraction')
        if not isinstance(aperture, (int, float)) or isinstance(aperture, bool) or not math.isfinite(aperture):
            return
        record = {'step': self.simulation_steps, 'open_fraction': float(aperture)}
        samples = monitor['aperture_samples']
        if samples and samples[-1]['step'] == record['step']:
            samples[-1] = record
        else:
            samples.append(record)

    def _check_possession_continuity(self):
        monitor = self._possession_guard
        if monitor['status'] != 'maintained':
            return False
        sample = self._possession_sample()
        monitor['samples'] += 1
        self._record_aperture_sample(monitor, sample)
        if sample['available'] and sample['object_detected']:
            return True
        monitor.update(status='lost' if sample['available'] else 'unavailable',
                       stop_sample=sample, stop_step=self.simulation_steps)
        return False

    def with_possession_guard(self, operation):
        if getattr(self, '_possession_guard', None) is not None:
            raise ValueError('possession monitor is already active')
        sample = self._possession_sample()
        self._possession_guard = None
        if not sample['available'] or not sample['object_detected']:
            status = 'unavailable' if not sample['available'] else 'not_present'
            return CapabilityResult.failed(FailureCategory.PERCEPTION_UNCERTAIN,
                'initial_gripper_possession_' + status,
                value={'possession_monitor': {'status': status, 'initial_sample': sample}})
        monitor = {'status': 'maintained', 'initial_sample': sample, 'samples': 1, 'aperture_samples': [],
                   'scope': 'public_gripper_register_only; target_identity_not_proved'}
        self._record_aperture_sample(monitor, sample)
        self._possession_guard = monitor
        original_guard = getattr(self, '_motion_guard', None)
        def guard():
            if original_guard is not None and not original_guard():
                return False
            return self._check_possession_continuity()
        self._motion_guard = guard
        start_step = self.simulation_steps
        try:
            try:
                result = operation()
            except Exception as exc:
                result = CapabilityResult.failed(FailureCategory.BACKEND_ERROR,
                    f'{type(exc).__name__}: {exc}', executed=self.simulation_steps > start_step)
            # A final-cycle loss must not escape just because no next step was
            # needed. The same guard also samples ordinary hold_position.
            self._check_possession_continuity()
            samples = monitor['aperture_samples']
            if samples:
                fractions = [record['open_fraction'] for record in samples]
                monitor.update(aperture_change_fraction=fractions[-1] - fractions[0],
                    aperture_min_fraction=min(fractions), aperture_max_fraction=max(fractions),
                    sampled_duration_s=(samples[-1]['step'] - samples[0]['step']) / self._control_frequency,
                    aperture_scope='Measured register trend only; maintained possession is not settled-grasp proof.')
            value = {**(result.value if isinstance(result.value, dict) else {}),
                     'possession_monitor': dict(monitor)}
            if monitor['status'] != 'maintained' and (result.success
                    or result.failure_category is FailureCategory.TIMEOUT
                    or result.message == 'motion_guard_rejected'):
                reason = 'gripper_possession_lost' if monitor['status'] == 'lost' else 'possession_sensor_unavailable'
                value.update(reached=False, stopped_reason=reason)
                return replace(result, success=False,
                    failure_category=FailureCategory.CONTROLLER_REJECTED, message=reason, value=value)
            return replace(result, value=value)
        finally:
            self._motion_guard = original_guard
            self._possession_guard = None

    def _move_to_pose_osc(self, *args, **kwargs):
        monitor = getattr(self, '_possession_guard', None)
        if monitor is None:
            return super()._move_to_pose_osc(*args, **kwargs)
        if kwargs.get('max_steps') != 1:
            return super()._move_to_pose_osc(*args, **kwargs)
        result = super()._move_to_pose_osc(*args, **kwargs)
        if not result.executed:
            return result
        if self._check_possession_continuity():
            return result
        if not result.success and result.failure_category is not FailureCategory.TIMEOUT:
            return result
        reason = 'gripper_possession_lost' if monitor['status'] == 'lost' else 'possession_sensor_unavailable'
        return CapabilityResult.failed(FailureCategory.CONTROLLER_REJECTED, reason, executed=True)

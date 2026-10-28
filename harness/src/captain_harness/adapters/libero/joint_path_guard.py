"""Unregistered joint-path guard candidate; no runtime factory imports this.

Compose an outer measurement guard with the path driver's collision monitor,
and sample wrist force before and after every executed control cycle. This
does not turn sampled limits into a physical force guarantee.
"""
from dataclasses import replace
import math

from captain_harness.ports.capability import CapabilityResult, FailureCategory


class GuardedJointPathMixin:
    def _sample_joint_path_guard(self):
        context = self._joint_path_guard_context
        if context['reason'] is not None:
            return False
        # Sampling a second time without physical execution adds no information.
        if context['last_step'] == self.simulation_steps:
            return True
        context['last_step'] = self.simulation_steps
        try:
            force = self._eef_force_norm()
            available = force is not None and math.isfinite(force)
            context['samples'].append({'step': self.simulation_steps,
                                       'force_n': float(force) if available else None})
            if not available:
                context['reason'] = 'path_force_unavailable'
            elif force >= 18. or force - context['initial_force_n'] >= 5.:
                context['reason'] = 'path_contact_force'
            elif context['outer_guard'] is not None and not context['outer_guard']():
                context['reason'] = 'outer_motion_guard_rejected'
        except Exception as exc:
            context['reason'] = 'path_measurement_failed'
            context['diagnostic'] = f'{type(exc).__name__}: {exc}'
        return context['reason'] is None

    def _step_action(self, action):
        context = getattr(self, '_joint_path_guard_context', None)
        if context is None:
            return super()._step_action(action)
        if not self._sample_joint_path_guard():
            self._last_step_rejection = context['reason']
            return False
        executed = super()._step_action(action)
        if executed:
            self._sample_joint_path_guard()
        # Report the actual cycle accurately even if its final sample stopped us.
        # execute_joint_path below gives that stop precedence over convergence.
        return executed

    def execute_joint_path(self, waypoints, collision_check):
        if getattr(self, '_joint_path_guard_context', None) is not None:
            raise RuntimeError('nested joint-path execution is unsupported')
        start = self.simulation_steps
        try:
            force = self._eef_force_norm()
        except Exception:
            force = None
        context = {'initial_force_n': force, 'samples': [], 'reason': None,
                   'last_step': None, 'outer_guard': getattr(self, '_motion_guard', None)}
        self._joint_path_guard_context = context
        try:
            if self._sample_joint_path_guard():
                try:
                    result = super().execute_joint_path(waypoints, collision_check)
                except Exception as exc:
                    result = CapabilityResult.failed(FailureCategory.BACKEND_ERROR,
                        f'{type(exc).__name__}: {exc}', executed=self.simulation_steps > start)
                self._sample_joint_path_guard()
            else:
                result = CapabilityResult.failed(FailureCategory.NOT_EXECUTED, context['reason'])
            value = {**(result.value if isinstance(result.value, dict) else {}),
                     'path_measurement_guard': {k: v for k, v in context.items()
                                                if k not in {'outer_guard', 'last_step'}},
                     'path_force_limit_n': 18., 'path_force_increase_limit_n': 5.}
            if context['reason'] is not None:
                value['reason'] = context['reason']
                return CapabilityResult.failed(FailureCategory.CONTROLLER_REJECTED,
                    context['reason'], executed=self.simulation_steps > start, value=value)
            return replace(result, value=value)
        finally:
            self._joint_path_guard_context = None


"""Bounded Cartesian contact execution using robot pose and measured wrist force.

No environment geometry, contact identities or task state are accessed here.
This mixin is not enabled merely by importing its geometry helpers.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation

from captain_harness.adapters.robotics.cartesian_sweeps import cartesian_sweep
from captain_harness.adapters.robotics.cartesian_path_timing import (
    advance_cartesian_reference, minimum_command_duration,
)
from captain_harness.adapters.robotics.motion_precision import motion_precision
from captain_harness.ports.capability import CapabilityResult, FailureCategory


def _pose(state):
    pose = np.eye(4)
    pose[:3, 3] = state.end_effector_position_m
    pose[:3, :3] = Rotation.from_quat(np.roll(state.end_effector_quaternion_wxyz, -1)).as_matrix()
    return pose


def _quaternion(pose):
    return np.roll(Rotation.from_matrix(pose[:3, :3]).as_quat(), 1)



def _force_guard_receipt(initial, reference, force, limit, retreat):
    available = force is not None and reference is not None
    triggered = []
    if available:
        if not retreat and force >= limit:triggered.append('absolute')
        if (force >= reference + 5. if retreat else force - reference >= 5.):
            triggered.append('increase')
    return {'initial_force_n': initial, 'reference_force_n': reference,
        'evaluated_force_n': force, 'absolute_limit_n': limit,
        'absolute_enabled': not retreat, 'increase_limit_n': 5.,
        'reference_basis': 'lowest_sampled_retreat_force' if retreat else 'operation_start_force',
        'sensor_available': available, 'triggered_comparisons': triggered,
        'scope': 'sampled wrist-force norm; contacting body and inter-sample peaks unknown'}


class ContactMotionMixin:
    def move_to_pose_monitored(self, position, quaternion):
        """Preserve the driver's fixed pose goal with sampled force stops.

        This keeps the existing OSC goal response for grasp probing. It makes
        no command-rate guarantee; commanded and measured motion are logged.
        """
        return self._move_to_pose_guarded(position, quaternion,
            command_profile='fixed_pose_goal')

    def _move_to_pose_guarded(self, position, quaternion, *, position_only=False,
                              max_steps=None, position_tolerance_m=None,
                              orientation_tolerance_rad=None, max_speed_m_s=.1,
                              command_profile='bounded_setpoints', precision_reference=None,
                              orientation_reference_quaternion_wxyz=None):
        """Driver-internal replacement for unbounded free Cartesian setpoints.

        The existing driver selects this path explicitly; contact microsteps
        call its one-cycle OSC implementation directly and do not recurse here.
        """
        try:
            start = _pose(self.get_robot_state().value)
            goal = np.eye(4)
            goal[:3, 3] = np.asarray(position, dtype=float).reshape(3)
            goal[:3, :3] = (start[:3, :3] if position_only else
                Rotation.from_quat(np.roll(np.asarray(quaternion, dtype=float).reshape(4), -1)).as_matrix())
            rotation = Rotation.from_matrix(goal[:3, :3] @ start[:3, :3].T).as_rotvec()
            angle = float(np.linalg.norm(rotation))
            kwargs = {} if angle < 1e-10 else {'axis': rotation / angle,
                'pivot_m': start[:3, 3], 'angle_rad': angle}
            path = (np.asarray([start, goal]) if command_profile == 'fixed_pose_goal' else
                    cartesian_sweep(start, displacement_m=goal[:3, 3] - start[:3, 3], **kwargs))
            return self._execute_contact_sweep(path, max_speed_m_s=max_speed_m_s,
                max_angular_speed_rad_s=.5, max_force_n=18.,
                # Internal home budgets may be twice the normal motion budget.
                # Keep the primitive's bound instead of passing an invalid duration.
                max_duration_s=min(30., (max_steps or self._cartesian_max_steps) / self._control_frequency),
                max_control_steps=max_steps if max_steps is not None else self._cartesian_max_steps,
                position_tolerance_m=position_tolerance_m or self._osc_position_tolerance_m,
                orientation_tolerance_rad=orientation_tolerance_rad or self._osc_orientation_tolerance_rad,
                position_only=position_only, command_profile=command_profile,
                precision_reference=precision_reference,
                orientation_reference_quaternion_wxyz=orientation_reference_quaternion_wxyz)
        except (TypeError, ValueError) as exc:
            return CapabilityResult.failed(FailureCategory.CONTROLLER_REJECTED, str(exc))

    def move_linear(self, direction, distance_m, *, max_speed_m_s=.02,
                    max_force_n=18., max_duration_s=10.):
        try:
            vector = np.asarray(direction, dtype=float)
            if (vector.shape != (3,) or not np.isfinite(vector).all()
                    or np.linalg.norm(vector) < 1e-10
                    or not math.isfinite(distance_m) or not 0 < distance_m <= .25):
                raise ValueError('direction must be nonzero and finite; distance_m must be in (0, .25]')
            start = _pose(self.get_robot_state().value)
            path = cartesian_sweep(start, displacement_m=vector / np.linalg.norm(vector) * distance_m)
            return self._execute_contact_sweep(path, max_speed_m_s=max_speed_m_s,
                max_force_n=max_force_n, max_duration_s=max_duration_s)
        except (ValueError, TypeError) as exc:
            return CapabilityResult.failed(FailureCategory.INVALID_ARGUMENT, str(exc))

    def rotate_about_axis(self, axis, pivot_m, angle_rad, *, max_speed_m_s=.02,
                          max_angular_speed_rad_s=.2, max_force_n=18., max_duration_s=10.):
        try:
            if not math.isfinite(angle_rad) or not 0 < abs(angle_rad) <= math.pi:
                raise ValueError('angle_rad must be nonzero, finite and within [-pi, pi]')
            start = _pose(self.get_robot_state().value)
            if np.linalg.norm(np.asarray(pivot_m, dtype=float) - start[:3, 3]) > .5:
                raise ValueError('pivot must be within .5 m of the current end effector')
            path = cartesian_sweep(start, axis=axis, pivot_m=pivot_m, angle_rad=angle_rad)
            return self._execute_contact_sweep(path, max_speed_m_s=max_speed_m_s,
                max_angular_speed_rad_s=max_angular_speed_rad_s,
                max_force_n=max_force_n, max_duration_s=max_duration_s)
        except (ValueError, TypeError) as exc:
            return CapabilityResult.failed(FailureCategory.INVALID_ARGUMENT, str(exc))

    def _retreat_path(self, max_distance_m=.05, continuation_id=None):
        from captain_harness.adapters.robotics.cartesian_path_timing import truncate_translation
        if isinstance(max_distance_m, bool) or not math.isfinite(max_distance_m) or not 0 < max_distance_m <= .25:
            raise ValueError('max_distance_m must be in (0, .25]')
        continuation = getattr(self, '_retreat_continuation', None)
        if continuation_id is not None:
            if continuation is None or continuation['id'] != continuation_id:
                raise ValueError('retreat_continuation_invalid')
            saved = continuation
            path = np.asarray(saved['remaining_path'])
            last = path[0]
        else:
            if continuation is not None:
                raise ValueError('continuation_id_required')
            saved = getattr(self, '_last_guarded_motion', None)
            if saved is None:
                raise ValueError('recorded_motion_is_stale')
            path = truncate_translation(np.asarray(list(reversed(saved['poses']))), max_distance_m)
            last = np.asarray(saved['poses'][-1])
        state = self.get_robot_state().value
        current = _pose(state)
        if (saved['end_step'] != self.simulation_steps
                or np.linalg.norm(current[:3,3]-last[:3,3]) > .001
                or Rotation.from_matrix(current[:3,:3]@last[:3,:3].T).magnitude() > .01
                or ('gripper_command' in saved and saved['gripper_command'] != self._gripper_command_fraction)
                or ('gripper_measured' in saved and abs(saved['gripper_measured']-state.gripper_open_fraction) > .001)):
            raise ValueError('recorded_motion_is_stale')
        if len(path) < 2:
            raise ValueError('no_recorded_retreat_segment')
        path[0] = current
        return path, saved

    def preview_retreat(self, max_distance_m=.05, max_duration_s=10., mode='full', continuation_id=None):
        from captain_harness.adapters.robotics.cartesian_path_timing import motion_preview
        if mode not in {'full', 'partial'}:
            raise ValueError('mode must be full or partial')
        path, saved = self._retreat_path(max_distance_m, continuation_id)
        return path, motion_preview(path, .01, .2, max_duration_s, self._control_frequency), saved

    def retreat_last_motion(self, max_distance_m=.05, max_duration_s=10., mode='full', continuation_id=None):
        from captain_harness.adapters.robotics.cartesian_path_timing import remaining_exit_path
        try:
            path, preview, saved = self.preview_retreat(max_distance_m, max_duration_s, mode, continuation_id)
        except (ValueError, TypeError) as exc:
            return CapabilityResult.failed(FailureCategory.NOT_EXECUTED, str(exc),
                value={'stopped_reason': str(exc)})
        if mode == 'full' and not preview['command_time_feasible']:
            return CapabilityResult.failed(FailureCategory.NOT_EXECUTED, 'insufficient_command_duration',
                value={**preview, 'stopped_reason': 'insufficient_command_duration', 'endpoint_reached': False,
                    'executed_distance_m': 0., 'continuation_id': continuation_id})
        origin = saved.get('origin_action_id', saved.get('action_id'))
        result = self._execute_contact_sweep(path, max_speed_m_s=.01,
            max_force_n=18., max_duration_s=max_duration_s, retreat=True)
        value = dict(result.value or {})
        value.update(recovery_mode=mode, origin_action_id=origin,
            previous_continuation_id=continuation_id, continuation_id=None, remaining_path=[],
            original_goal_restored='pending_visual_confirmation', object_retention='pending_visual_confirmation')
        if result.executed:
            self._last_guarded_motion = None
            self._retreat_continuation = None
            if not value.get('endpoint_reached'):
                remaining = remaining_exit_path(path, _pose(self.get_robot_state().value),
                    np.asarray(value['last_command_pose']), value['next_command_path_index'])
                if remaining is not None and len(remaining) > 1:
                    state = self.get_robot_state().value
                    token = uuid.uuid4().hex
                    self._retreat_continuation = {'id': token, 'remaining_path': remaining.tolist(),
                        'end_step': self.simulation_steps, 'gripper_command': self._gripper_command_fraction,
                        'gripper_measured': state.gripper_open_fraction, 'origin_action_id': origin}
                    value.update(continuation_id=token, remaining_path=remaining.tolist())
                else:
                    value['continuation_unavailable_reason'] = 'measured_pose_not_on_issued_exit_prefix'
        else:
            value['continuation_id'] = continuation_id
        return replace(result, value=value)

    def _execute_contact_sweep(self, path, *, max_speed_m_s=.02,
                               max_angular_speed_rad_s=.2, max_force_n=18.,
                               max_duration_s=10., retreat=False,
                               position_tolerance_m=.002, orientation_tolerance_rad=.01,
                               position_only=False, command_profile='bounded_setpoints', precision_reference=None,
                               orientation_reference_quaternion_wxyz=None, max_control_steps=None,
                               contact_mode='guarded', detector=None, sample_sink=None):
        from captain_harness.adapters.robotics.contact_limits import EMERGENCY_LIMIT_N, STALL_WINDOW_S
        force_cap = EMERGENCY_LIMIT_N if detector is not None else 18.
        if max_control_steps is not None and (isinstance(max_control_steps, bool)
                or not isinstance(max_control_steps, int) or max_control_steps <= 0):
            return CapabilityResult.failed(FailureCategory.INVALID_ARGUMENT, "invalid control-step limit")
        if command_profile not in {'bounded_setpoints', 'fixed_pose_goal'} or (retreat and command_profile != 'bounded_setpoints'):
            return CapabilityResult.failed(FailureCategory.INVALID_ARGUMENT, 'invalid command profile')
        if precision_reference not in {None, 'planned_path_waypoint', 'planned_path_endpoint', 'pregrasp_phase', 'absolute_pose_goal'}:
            return CapabilityResult.failed(FailureCategory.INVALID_ARGUMENT, 'invalid precision reference')
        bounded = command_profile == 'bounded_setpoints'
        # opt-in step-efficiency profile (RoboDojo X5, native action budget): every per-tick Cartesian advance and the
        # command lead are scaled by _speed_scale (argument bounds stay nominal); manipulation-mode moves fall back to
        # the unscaled speed inside the last _slow_zone_m before the commanded endpoint, where contact is expected.
        speed_scale = float(getattr(self, '_speed_scale', 1.0))
        slow_zone_m = float(getattr(self, '_slow_zone_m', 0.0))
        settle_motion_m = float(getattr(self, '_settle_motion_m', .00005))
        settle_needed = int(getattr(self, '_settle_samples_needed', 2))
        if self.controller != 'OSC_POSE':
            return CapabilityResult.failed(FailureCategory.CONTROLLER_REJECTED, 'contact motion requires OSC_POSE')
        for value, upper, name in [(max_speed_m_s, .1, 'max_speed_m_s'),
                (max_angular_speed_rad_s, 1., 'max_angular_speed_rad_s'),
                (max_force_n, force_cap, 'max_force_n'), (max_duration_s, 30., 'max_duration_s'),
                (position_tolerance_m, .1, 'position_tolerance_m'),
                (orientation_tolerance_rad, .5, 'orientation_tolerance_rad')]:
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= upper:
                return CapabilityResult.failed(FailureCategory.INVALID_ARGUMENT, f'{name} must be in (0, {upper}]')
        baseline = self._eef_force_norm()
        initial_force = baseline
        force_guard = _force_guard_receipt(initial_force, baseline, baseline, max_force_n, retreat)
        if baseline is None:
            return CapabilityResult.failed(FailureCategory.NOT_EXECUTED, 'force_unavailable',
                value={'stopped_reason': 'force_unavailable', 'force_available': False, 'force_guard': force_guard})
        # r9 (opt-in, X5): with the increase-only residual reference the absolute start-force rule is replaced by a hard
        # ceiling (a static post-stop wrist load without contact must not block the next move)
        _r9_skip = (detector is not None and getattr(self, '_rd_no_initial_limit', False)
                    and baseline < float(getattr(self, '_rd_hard_ceiling_n', 120.)))
        if baseline >= max_force_n and not retreat and not _r9_skip:
            return CapabilityResult.failed(FailureCategory.CONTROLLER_REJECTED, 'initial_force_limit',
                value={'stopped_reason': 'initial_force_limit', 'observed_force_n': baseline, 'force_guard': force_guard})
        start_steps = self.simulation_steps
        start = _pose(self.get_robot_state().value)
        reference_rotation = None
        if orientation_reference_quaternion_wxyz is not None:
            reference_rotation = Rotation.from_quat(np.roll(
                np.asarray(orientation_reference_quaternion_wxyz, dtype=float), -1)).as_matrix()
        precision = motion_precision(start, path[-1], position_tolerance_m, orientation_tolerance_rad,
            orientation_reference=("compound_request" if reference_rotation is not None else
                "recorded_pose" if retreat else "commanded_change"),
            orientation_start_rotation=reference_rotation)
        if precision_reference is not None:
            precision.update(position_tolerance_m=float(position_tolerance_m),
                orientation_tolerance_rad=float(orientation_tolerance_rad),
                precision_reference=precision_reference,
                precision_basis=('configured path-following tolerance for an internal waypoint'
                    if precision_reference == 'planned_path_waypoint' else
                    'declared absolute tolerance for an internal pregrasp phase; final descent gate unchanged'
                    if precision_reference == 'pregrasp_phase' else
                    'declared absolute EEF pose tolerance; measured residual is not a relative correction request'
                    if precision_reference == 'absolute_pose_goal' else
                    'endpoint tolerance derived from the whole planned motion'))
        if precision_reference == 'absolute_pose_goal':
            precision['orientation_reference'] = 'absolute_pose_goal'
        position_tolerance_m = precision['position_tolerance_m']
        orientation_tolerance_rad = precision['orientation_tolerance_rad']
        from captain_harness.adapters.robotics.tracking_feedback import DirectionalProgressGuard, tracking_feedback
        # Controller tracking/blockage scale is independent of requested endpoint precision.
        # A tiny final correction must not make noise look like sustained blockage.
        directional_guard = DirectionalProgressGuard(position_tolerance_m=.003,
            orientation_tolerance_rad=.03)
        from .substep_force_guard import SubstepForceMonitor, active_force_monitor
        substep_monitor=SubstepForceMonitor(initial_force=baseline,max_force=max_force_n,retreat=retreat,
            detector=detector,collect_samples=sample_sink is not None,
            rise_limit_n=getattr(self,'_legacy_rise_limit_n',5.))
        directional_block = None
        # Allow at most two native command increments ahead, with a tracking-scale
        # floor so slow impedance motion can still overcome breakaway load.
        # These limits do not shrink with requested final-position tolerance.
        command_lead_m=min(.02*speed_scale,max(.006,2.*speed_scale*max_speed_m_s/self._control_frequency))
        command_lead_rad=min(.1*speed_scale,max(.06,2.*speed_scale*max_angular_speed_rad_s/self._control_frequency))
        if detector is not None and contact_mode == 'manipulation':
            # Let the impedance reference run ahead of a loaded EEF instead of tripping the tracking limit.
            command_lead_m *= 2.
        stall_window = []  # (sim_step, eef position) for the manipulation stall rule
        ramp_ticks = 0
        poses = [start.tolist()]
        trace = []
        target_index, stalled, stop = 1, 0, None
        paced_steps = 0
        settled_samples = 0
        requires_settle = (bounded and precision_reference != 'planned_path_waypoint'
            and not np.allclose(start, path[-1], rtol=0, atol=1e-10))
        desired = start.copy()
        peak = baseline
        unloading_reference = baseline
        unloading_samples = 0
        step_limit = int(math.floor(max_duration_s * self._control_frequency + 1e-9))
        minimum_duration = minimum_command_duration(path, max_speed_m_s, max_angular_speed_rad_s)
        if max_control_steps is not None:
            step_limit = min(step_limit, max_control_steps)
        for _ in range(step_limit):
            current = _pose(self.get_robot_state().value)
            if not bounded:
                desired = path[-1].copy()
                target_index = len(path)
            while target_index < len(path):
                target = path[target_index]
                error = target[:3, 3] - desired[:3, 3]
                angular = Rotation.from_matrix(target[:3, :3] @ desired[:3, :3].T).as_rotvec()
                if np.linalg.norm(error) <= 1e-10 and np.linalg.norm(angular) <= 1e-10:
                    target_index += 1
                else:
                    break
            if target_index >= len(path):
                residual = np.linalg.norm(current[:3, 3] - path[-1, :3, 3])
                turn_error = Rotation.from_matrix(path[-1, :3, :3] @ current[:3, :3].T).magnitude()
                if (residual <= position_tolerance_m and (position_only or turn_error <= orientation_tolerance_rad)
                        and (not requires_settle or settled_samples >= settle_needed)):
                    stop = 'target_reached'
                    break
            else:
                # Advance the commanded path independently of tracking error.
                # Re-basing every tiny goal on measured position can leave an
                # impedance controller forever below breakaway force.
                tick_scale = speed_scale
                if speed_scale != 1.:
                    # ramp: speed grows by at most one nominal step per tick and decelerates so that it could stop
                    # within the remaining path (keeps the arm's inertial/damping force transients at the R5 level)
                    remaining_m = float(np.linalg.norm(desired[:3, 3] - path[-1, :3, 3]))
                    nominal_step = max_speed_m_s / self._control_frequency
                    tick_scale = min(speed_scale, 1. + ramp_ticks,
                                     max(1., math.sqrt(2. * remaining_m / max(nominal_step, 1e-9))))
                    ramp_ticks += 1
                    if (slow_zone_m > 0 and contact_mode == 'manipulation' and detector is not None
                            and remaining_m <= slow_zone_m):
                        tick_scale = 1.
                proposed, proposed_index = advance_cartesian_reference(
                    path, desired, target_index, duration_s=tick_scale / self._control_frequency,
                    max_speed_m_s=max_speed_m_s,
                    max_angular_speed_rad_s=max_angular_speed_rad_s)
                # Keep the previous impedance target while a loaded robot
                # catches up. Never reset it to the measured pose or increase
                # the allowed tracking error to maintain nominal speed.
                # the setpoint may run ahead of the measured pose only as far as the current per-tick speed allows,
                # so a decelerating / slow-zone approach is also slow at the arm (R5 lead at nominal speed)
                lead_m_now, lead_rad_now = command_lead_m, command_lead_rad
                if speed_scale != 1.:
                    lead_m_now = min(command_lead_m, max(.006, 2. * tick_scale * max_speed_m_s / self._control_frequency))
                    lead_rad_now = min(command_lead_rad, max(.06, 2. * tick_scale * max_angular_speed_rad_s / self._control_frequency))
                    if detector is not None and contact_mode == 'manipulation':
                        lead_m_now = min(command_lead_m, 2. * lead_m_now)   # same x2 loaded-EEF allowance as R5
                if (np.linalg.norm(proposed[:3, 3] - current[:3, 3]) <= lead_m_now
                        and (position_only or Rotation.from_matrix(
                            proposed[:3, :3] @ current[:3, :3].T).magnitude() <= lead_rad_now)):
                    desired = proposed
                    target_index = proposed_index
                else:
                    paced_steps += 1
            if bounded and (np.linalg.norm(desired[:3, 3] - current[:3, 3]) > command_lead_m
                    or (not position_only and Rotation.from_matrix(desired[:3, :3] @ current[:3, :3].T).magnitude() > command_lead_rad)):
                stop = 'tracking_error_limit'
                break
            cycle_start = self.simulation_steps
            if detector is not None:
                base_rotation, _ = self._base_transform()
                self._contact_direction_world = base_rotation @ (desired[:3, 3] - current[:3, 3])
            with active_force_monitor(self,substep_monitor):
                step = self._move_to_pose_osc(desired[:3, 3], _quaternion(desired),
                    position_only=position_only, max_steps=1, position_tolerance_m=1e-6, orientation_tolerance_rad=1e-6)
                if (requires_settle and step.success and self.simulation_steps == cycle_start):
                    # At an exact pose, OSC can return without advancing dynamics.
                    # Sample a real zero-delta hold inside this same bounded action;
                    # duplicate reads of one timestamp cannot prove settling.
                    hold = np.zeros(self._action_dim)
                    hold[-1] = 1. - 2. * self._gripper_command_fraction
                    if not self._step_action(hold):
                        step = CapabilityResult.failed(FailureCategory.CONTROLLER_REJECTED,
                            self._last_step_rejection or 'controller_stopped')
            after = _pose(self.get_robot_state().value)
            force = self._eef_force_norm()
            poses.append(after.tolist())
            trace.append({'step': self.simulation_steps, 'eef_position_m': after[:3, 3].tolist(),
                'eef_quaternion_wxyz': _quaternion(after).tolist(), 'force_n': force,
                'gripper_open_fraction': float(self.get_robot_state().value.gripper_open_fraction),
                'joint_positions_rad': list(self.get_robot_state().value.joint_positions_rad),
                'sample_time_s': self.simulation_steps / self._control_frequency,
                'position_error_m': float(np.linalg.norm(after[:3,3]-desired[:3,3])),
                'orientation_error_rad': float(Rotation.from_matrix(after[:3,:3]@desired[:3,:3].T).magnitude()),
                'commanded_eef_position_m': desired[:3, 3].tolist(),
                'commanded_eef_quaternion_wxyz': _quaternion(desired).tolist(),
                **({k: detector.last[k] for k in ('residual_n', 'residual_par_n', 'residual_lat_n', 'osc_force_norm_n', 'contact_n')}
                   if detector is not None and detector.last else {})})
            live_guard = getattr(self, '_motion_guard', None)
            if live_guard is not None and not live_guard():
                stop = 'motion_guard_rejected'
                break
            force_guard = _force_guard_receipt(initial_force, baseline, force, max_force_n, retreat)
            if force is None:
                stop = 'force_unavailable'
                break
            peak = max(peak, force, substep_monitor.peak_force)
            if substep_monitor.trigger is not None:
                stop=substep_monitor.trigger["reason"]
                force_guard=_force_guard_receipt(initial_force,
                    substep_monitor.trigger["reference_force_n"],substep_monitor.trigger["force_n"],
                    max_force_n,retreat)
                force_guard["evaluation_phase"]="native_substep_trigger"
                break
            if detector is None and (force >= max_force_n or force - baseline >= getattr(self, '_legacy_rise_limit_n', 5.)) and not retreat:
                stop = 'contact_force'
                break
            if detector is not None and contact_mode == 'manipulation' and self.simulation_steps > cycle_start:
                stall_window.append((self.simulation_steps, after[:3, 3].copy()))
                window_s = (stall_window[-1][0] - stall_window[0][0]) / self._control_frequency
                while len(stall_window) > 1 and (stall_window[-1][0] - stall_window[1][0]) / self._control_frequency >= STALL_WINDOW_S:
                    stall_window.pop(0)
                direction = desired[:3, 3] - stall_window[0][1]
                norm = np.linalg.norm(direction)
                progress = float((after[:3, 3] - stall_window[0][1]) @ (direction / norm)) if norm > 1e-9 else 0.
                if detector.observe_control_step(progress, window_s):
                    stop = 'contact_stall'
                    break
            # A known reverse path can unload a pre-existing contact. It cannot
            # increase the initial load by 5 N or continue after load rises again.
            if retreat:
                if force >= baseline + 5.:
                    stop = 'retreat_force_increased'
                    break
                baseline = min(baseline, force)
                unloading_samples += int(self.simulation_steps > cycle_start)
                if unloading_samples >= 10:
                    if force >= max_force_n and force >= unloading_reference - 1e-6:
                        stop = 'retreat_load_not_decreasing'
                        break
                    unloading_reference, unloading_samples = force, 0
            if bounded and self.simulation_steps > cycle_start:
                directional_block = directional_guard.observe(current, after, desired, position_only=position_only)
                if directional_block:
                    stop = 'motion_stalled'
                    break
            motion = np.linalg.norm(after[:3, 3] - current[:3, 3])
            turn = Rotation.from_matrix(after[:3, :3] @ current[:3, :3].T).magnitude()
            endpoint_close = (np.linalg.norm(after[:3, 3] - path[-1, :3, 3]) <= position_tolerance_m
                and (position_only or Rotation.from_matrix(path[-1, :3, :3] @ after[:3, :3].T).magnitude()
                    <= orientation_tolerance_rad))
            if (self.simulation_steps > cycle_start and endpoint_close
                    and motion <= max(min(.00005, .1 * position_tolerance_m), settle_motion_m)
                    and (position_only or turn <= min(.0005, .1 * orientation_tolerance_rad))):
                settled_samples += 1
            else:
                settled_samples = 0
            # Resolve slow convergence by measured progress toward the same
            # endpoint. Real impedance motion can be below 50 um/sample while
            # still correcting a fine residual. Mere drift away is not progress.
            position_progress = (np.linalg.norm(current[:3, 3] - path[-1, :3, 3])
                - np.linalg.norm(after[:3, 3] - path[-1, :3, 3]))
            angular_progress = (Rotation.from_matrix(path[-1, :3, :3] @ current[:3, :3].T).magnitude()
                - Rotation.from_matrix(path[-1, :3, :3] @ after[:3, :3].T).magnitude())
            converging = (position_progress > .001 * position_tolerance_m
                or (not position_only and angular_progress > .001 * orientation_tolerance_rad))
            stalled = stalled + 1 if motion < .00005 and turn < .0005 and not converging else 0
            if stalled >= 10:
                stop = 'motion_stalled'
                break
            if not step.executed or (not step.success and step.failure_category is not FailureCategory.TIMEOUT):
                stop = step.message or 'controller_stopped'
                break
        final = _pose(self.get_robot_state().value)
        position_error = float(np.linalg.norm(final[:3, 3] - path[-1, :3, 3]))
        angle_error = float(Rotation.from_matrix(path[-1, :3, :3] @ final[:3, :3].T).magnitude())
        if stop is None:
            stop = ('target_reached' if position_error <= position_tolerance_m
                    and (position_only or angle_error <= orientation_tolerance_rad)
                    and (not requires_settle or settled_samples >= settle_needed) else 'duration_exhausted')
        if retreat and stop == 'target_reached' and self._eef_force_norm() >= max_force_n:
            stop = 'retreat_load_unresolved'
        steps = self.simulation_steps - start_steps
        if steps and not retreat:
            self._last_guarded_motion = {'poses': poses, 'end_step': self.simulation_steps,
                'gripper_command': self._gripper_command_fraction,
                'gripper_measured': self.get_robot_state().value.gripper_open_fraction}
        if sample_sink is not None:
            sample_sink.extend(substep_monitor.samples)
        self._contact_direction_world = None
        value = {**precision,
            'contact_mode': contact_mode if detector is not None else 'legacy',
            'contact_detector': detector.receipt() if detector is not None else {'mode': 'legacy', 'force_limit_n': max_force_n, 'increase_limit_n': 5.},
            'substep_force_monitor':substep_monitor.receipt(),
            'tracking_guard_scale':{'position_m':.003,'orientation_rad':.03,
                'scope':'blockage detection; independent of endpoint tolerance'},
            'tracking_feedback': tracking_feedback(start, final, path[-1], desired,
                stop_reason=stop, blocked=directional_block),
            'endpoint_reached': bool(position_error <= position_tolerance_m and (position_only or angle_error <= orientation_tolerance_rad)),
            'force_decreased': bool(self._eef_force_norm() is not None and self._eef_force_norm() < initial_force),
            'force_still_over_limit': None if self._eef_force_norm() is None else bool(self._eef_force_norm() >= max_force_n),
            'executed_distance_m': float(sum(np.linalg.norm(np.asarray(b)[:3,3]-np.asarray(a)[:3,3]) for a,b in zip(poses[:-1],poses[1:]))),
            'requested_path_distance_m': float(sum(np.linalg.norm(b[:3,3]-a[:3,3]) for a,b in zip(path[:-1],path[1:]))),
            'last_command_pose': desired.tolist(), 'next_command_path_index': target_index,
            'command_endpoint_issued': target_index >= len(path),
            'reached': stop == 'target_reached', 'stopped_reason': stop,
            'control_steps': steps, 'elapsed_s': steps / self._control_frequency,
            'endpoint_settled': settled_samples >= settle_needed if requires_settle else None,
            'endpoint_settle_samples': settled_samples,
            'command_pacing_steps': paced_steps,
            'command_lead_limit_m':command_lead_m if bounded else None,
            'command_lead_limit_rad':command_lead_rad if bounded else None,
            'minimum_command_duration_s': minimum_duration if bounded else None,
            'requested_duration_limit_s': max_duration_s,
            'effective_duration_limit_s': step_limit / self._control_frequency,
            'timing_scope': 'nominal command travel excludes tracking delay, contacts and endpoint settling',
            'actual_displacement_m': (final[:3, 3] - start[:3, 3]).tolist(),
            'actual_rotation_vector_rad': Rotation.from_matrix(final[:3, :3] @ start[:3, :3].T).as_rotvec().tolist(),
            'position_residual_m': position_error, 'orientation_residual_rad': angle_error,
            'position_tolerance_m': position_tolerance_m, 'orientation_tolerance_rad': orientation_tolerance_rad,
            'max_observed_force_n': peak, 'observed_force_n': self._eef_force_norm(),
            'force_available': stop != 'force_unavailable', 'force_limit_n': max_force_n,
            'force_guard': force_guard,
            'command_profile': command_profile,
            'max_command_translation_m_per_step': max_speed_m_s / self._control_frequency if bounded else None,
            'max_command_rotation_rad_per_step': max_angular_speed_rad_s / self._control_frequency if bounded else None,
            'contact_identity': 'unknown', 'retreat': retreat, 'trajectory': trace,
            'limitation': ('Command increments are bounded; measured speed and force can overshoot between samples. '
                if bounded else 'The fixed OSC pose goal has no command-rate guarantee; force is checked at each sampled step. ') +
                'No target identity, collision-free path or task success is inferred.'}
        if value['reached']:
            return CapabilityResult.ok(value, ('bounded Cartesian sweep' if bounded else 'monitored OSC goal') + ' reached its endpoint')
        if stop == 'deadline_expired':
            return self._deadline_rejection_result(steps > 0, value)
        category = (FailureCategory.TIMEOUT if stop in {'motion_stalled', 'duration_exhausted'}
                    else FailureCategory.CONTROLLER_REJECTED)
        return replace(CapabilityResult.failed(category, stop,
            executed=steps > 0, value=value),
            diagnostics={"failure_stage": "physical_execution",
                         "stop_reason": stop, "control_steps": steps})

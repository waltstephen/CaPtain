"""Explicit public-pose clearance routing; no scene or task inference."""
from copy import deepcopy
from dataclasses import replace
from collections.abc import Mapping
import numpy as np
from scipy.spatial.transform import Rotation
from captain_harness.ports.capability import CapabilityResult, FailureCategory
from captain_harness.ports.deadline import check_deadline, current_deadline, retain_deadline_status


def _rotation(q):
    return Rotation.from_quat(np.roll(q, -1)).as_matrix()


def goto_with_clearance(backend, position, quaternion, z_approach):
    robot = backend.robot
    target = position + _rotation(quaternion) @ backend._tcp_offset_local_m
    before = backend._motion_outcome('target', False, '', requested_position=position,
        controller_target=target, requested_quaternion=quaternion)
    if (before.get('pose_measurement') or {}).get('status') == 'unavailable':
        return CapabilityResult.failed(FailureCategory.BACKEND_ERROR,
            'current_pose_unavailable: ' + before['pose_measurement']['message'], value=before)
    limit = getattr(robot, 'cartesian_control_step_limit', None)
    start = getattr(robot, 'simulation_steps', None)
    if (getattr(robot, 'controller', None) != 'OSC_POSE'
            or not callable(getattr(robot, 'move_to_pose_bounded', None))
            or type(limit) is not int or limit <= 0 or type(start) is not int):
        return CapabilityResult.failed(FailureCategory.NOT_EXECUTED,
            'clearance_first requires bounded Cartesian phases and measured control-step counts', value=before)
    orientation_reference = tuple(before['final_eef_quaternion_wxyz'])
    current = np.asarray(before['final_cartesian_position_m'], dtype=float)
    current_quaternion = np.asarray(before['final_cartesian_quaternion_wxyz'], dtype=float)
    altitude = max(float(current[2]), float(position[2] + z_approach))
    lifted = current.copy(); lifted[2] = altitude
    above = position.copy(); above[2] = altitude
    route = []
    previous_position, previous_quaternion = current, current_quaternion
    for phase, point, quat in [('clearance', lifted, current_quaternion),
                              ('approach', above, quaternion), ('target', position, quaternion)]:
        # If the approach already IS the final target, use its final name and
        # never issue another tiny command to correct incidental residual drift.
        if phase == 'approach' and np.allclose(above, position, rtol=0, atol=1e-10):
            continue
        if (np.allclose(point, previous_position, rtol=0, atol=1e-10)
                and np.allclose(_rotation(quat), _rotation(previous_quaternion), rtol=0, atol=1e-10)):
            continue
        route.append((phase, point.copy(), quat.copy()))
        previous_position, previous_quaternion = point, quat
    phases = {}

    def counts():
        now = getattr(robot, 'simulation_steps', None)
        if type(now) is not int or now < start:
            raise RuntimeError('control_step_measurement_unavailable')
        return now - start

    def finish(result, phase, point, quat):
        used = counts()
        # CapabilityResult's existing invariant counts a completed no-op as an
        # executed operation. Physical execution is recorded separately below.
        executed = result.success or used > 0 or any(item['executed'] for item in phases.values())
        value = dict(result.value) if isinstance(result.value, Mapping) else {}
        if 'converged' not in value:
            value = {**backend._motion_outcome(phase, executed, result.message,
                requested_position=point, requested_quaternion=quat,
                controller_target=point + _rotation(quat) @ backend._tcp_offset_local_m), **value}
        value.update(executed=executed, clearance_first=True, clearance_tool_z_m=altitude,
            phase_executions=deepcopy(phases),
            completed_phases=[name for name,item in phases.items() if item['success']],
            failed_phase=None if result.success else phase,
            control_budget={'limit_steps':limit, 'used_steps':used, 'remaining_steps':max(0,limit-used)},
            route_scope='explicit world-Z clearance and exact requested pose; collision freedom and task success unknown')
        category = result.failure_category
        diagnostics = dict(result.diagnostics or {})
        diagnostics['physical_execution'] = {'control_steps':used,
            'status':'executed' if used else 'not_executed'}
        if executed and category is FailureCategory.NOT_EXECUTED:
            diagnostics['original_failure_category'] = category.value
            category = FailureCategory.BACKEND_ERROR
        return replace(result, executed=executed, value=value, failure_category=category,
            diagnostics=diagnostics or None)

    if not route:
        return finish(CapabilityResult.ok(before, 'requested pose already measured; no control step required'),
            'target', position, quaternion)
    for phase, point, quat in route:
        first = counts()
        try:
            check_deadline()
            remaining = limit - first
            if remaining <= 0:
                result = CapabilityResult.failed(FailureCategory.TIMEOUT, 'compound_control_step_budget_exhausted')
            else:
                result = backend._execute_cartesian_phase(phase=phase,
                    requested_position=point, requested_quaternion=quat,
                    controller_target=point + _rotation(quat) @ backend._tcp_offset_local_m,
                    orientation_reference_quaternion_wxyz=orientation_reference,
                    max_control_steps=remaining)
                result = retain_deadline_status(result, current_deadline())
        except Exception as exc:
            from .backend import ToolOperationError
            diagnostics = None
            if isinstance(exc, TimeoutError):
                category = FailureCategory.TIMEOUT
                diagnostics = {'deadline':getattr(exc, 'deadline_receipt', None),
                    'service_attempts':list(getattr(exc, 'attempts', ()))}
            elif isinstance(exc, ToolOperationError):
                category = exc.category
            elif isinstance(exc, (TypeError, ValueError, np.linalg.LinAlgError)):
                category = FailureCategory.INVALID_ARGUMENT
            else:
                category = FailureCategory.TOOL_ERROR
            executed = counts() > first
            if executed and category is FailureCategory.NOT_EXECUTED:
                diagnostics = {**(diagnostics or {}), 'original_failure_category':category.value}
                category = FailureCategory.BACKEND_ERROR
            result = replace(CapabilityResult.failed(category, str(exc), executed=executed),
                diagnostics=diagnostics)
        phases[phase] = deepcopy({'success':result.success, 'executed':result.executed,
            'failure_category':result.failure_category.value, 'message':result.message,
            'value':result.value, 'diagnostics':result.diagnostics, 'attempt_identity':result.attempt_identity})
        if not result.success:
            return finish(result, phase, point, quat)
    return finish(result, phase, point, quat)

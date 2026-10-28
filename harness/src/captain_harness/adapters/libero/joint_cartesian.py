"""Unregistered native-joint Cartesian candidate, using detached robot IK.

This component deliberately does not change LiberoActionBackend or its manifest.
It does not read environment geometry or infer task/contact identity.
"""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
from scipy.spatial.transform import Rotation

from captain_harness.ports.capability import CapabilityResult, FailureCategory


def _pose(state):
    pose = np.eye(4)
    pose[:3, 3] = state.end_effector_position_m
    pose[:3, :3] = Rotation.from_quat(np.roll(state.end_effector_quaternion_wxyz, -1)).as_matrix()
    return pose


def _errors(actual, goal):
    return (float(np.linalg.norm(actual[:3, 3] - goal[:3, 3])),
            float(Rotation.from_matrix(goal[:3, :3] @ actual[:3, :3].T).magnitude()))


def plan_cartesian_joints(model, joints, aperture, start, goal, *, translation_step=.0025,
                          rotation_step=.025):
    """Validate a continuous local IK branch before executing any control step.

    Endpoint IK is not scene collision checking. No restart across IK branches
    is attempted during the contact/servo trajectory.
    """
    q = np.asarray(joints, dtype=float).copy()
    if not np.isfinite(q).all() or q.shape != (len(model.limits),):
        raise ValueError('invalid measured joints')
    position, angle = _errors(model.eef_pose(q, aperture), start)
    if position > .005 or angle > .03:
        raise ValueError('robot FK disagrees with measured start pose')
    dp = goal[:3, 3] - start[:3, 3]
    dr = Rotation.from_matrix(goal[:3, :3] @ start[:3, :3].T).as_rotvec()
    count = max(1, math.ceil(np.linalg.norm(dp) / translation_step),
                math.ceil(np.linalg.norm(dr) / rotation_step))
    if count > 1000:
        raise ValueError('Cartesian plan exceeds 1000 local segments')
    path = [q.copy()]
    for i in range(1, count + 1):
        target = np.eye(4)
        target[:3, 3] = start[:3, 3] + dp * (i / count)
        target[:3, :3] = Rotation.from_rotvec(dr * (i / count)).as_matrix() @ start[:3, :3]
        solved = model._solve_ik_seed(target, q, aperture=aperture, max_iterations=100,
            position_tolerance=.0005, orientation_tolerance=.005)
        if solved is None:
            raise ValueError(f'local Cartesian IK failed at segment {i}/{count}')
        solved = np.asarray(solved, dtype=float)
        if (solved.shape != q.shape or not np.isfinite(solved).all()
                or np.any(solved < model.limits[:, 0]) or np.any(solved > model.limits[:, 1])):
            raise ValueError('local Cartesian IK returned invalid joints')
        if np.max(np.abs(solved - q)) > .15:
            raise ValueError('local Cartesian IK changed joint branch')
        position, angle = _errors(model.eef_pose(solved, aperture), target)
        if position > .0005 or angle > .005:
            raise ValueError('local Cartesian IK failed independent FK validation')
        q = solved.copy()
        path.append(q)
    return np.asarray(path)


class NativeJointCartesian:
    """Development component bound to the driver's public robot measurements."""
    def __init__(self, robot):
        self.robot = robot

    def _force(self):
        value = self.robot._eef_force_norm()
        return float(value) if value is not None and math.isfinite(value) else None

    def _initial_check(self, max_force_n):
        if self.robot.controller != 'JOINT_POSITION':
            return 'native joint Cartesian motion requires JOINT_POSITION'
        force = self._force()
        if force is None:
            return 'force_unavailable'
        if force >= max_force_n:
            return 'initial_force_limit'
        return None

    @staticmethod
    def _result(reason, steps, trace, **payload):
        value = {'stopped_reason': reason, 'control_steps': steps, 'trajectory': trace,
            'contact_identity': 'unknown', 'execution_controller': 'JOINT_POSITION',
            'limitation': 'Sampled force stop is not a physical force bound. No scene or held-object collision proof.',
            **payload}
        if reason in {'target_reached', 'hold_complete'}:
            return replace(CapabilityResult.ok(value), executed=steps > 0)
        category = (FailureCategory.TIMEOUT if reason in {'duration_exhausted', 'motion_stalled', 'horizon_exhausted'}
                    else FailureCategory.CONTROLLER_REJECTED)
        return CapabilityResult.failed(category, reason, executed=steps > 0, value=value)

    def _sample(self, command):
        state = self.robot.get_robot_state().value
        return {'step': self.robot.simulation_steps,
            'joint_positions_rad': list(state.joint_positions_rad),
            'commanded_joint_positions_rad': np.asarray(command).tolist(),
            'eef_position_m': list(state.end_effector_position_m),
            'eef_quaternion_wxyz': list(state.end_effector_quaternion_wxyz),
            'gripper_open_fraction': float(state.gripper_open_fraction), 'force_n': self._force()}

    def move(self, position, quaternion, *, max_speed_m_s=.05, max_duration_s=24., max_force_n=18.):
        robot = self.robot
        try:
            for value, upper in [(max_speed_m_s, .1), (max_duration_s, 30.), (max_force_n, 18.)]:
                if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= upper:
                    raise ValueError('invalid native Cartesian speed, duration or force limit')
            state = robot.get_robot_state().value
            start = _pose(state)
            goal = np.eye(4)
            goal[:3, 3] = np.asarray(position, dtype=float).reshape(3)
            goal[:3, :3] = Rotation.from_quat(np.roll(np.asarray(quaternion, dtype=float).reshape(4), -1)).as_matrix()
            if not np.isfinite(goal).all():
                raise ValueError('nonfinite Cartesian target')
            reason = self._initial_check(max_force_n)
            if reason:
                return self._result(reason, 0, [])
            model = robot.planning_model()
            path = plan_cartesian_joints(model, state.joint_positions_rad, state.gripper_open_fraction,
                start, goal, translation_step=max_speed_m_s / robot._control_frequency)
        except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
            return self._result('cartesian_plan_rejected', 0, [], diagnostic=str(exc))
        self.last_planned_joints = path.copy()
        initial_force = self._force()
        first_step = robot.simulation_steps
        desired = path[0].copy()
        index, stalled, pacing = 1, 0, 0
        trace, reason = [], None
        for _ in range(math.ceil(max_duration_s * robot._control_frequency)):
            state = robot.get_robot_state().value
            actual = _pose(state)
            q = np.asarray(state.joint_positions_rad)
            force = self._force()
            if force is None or force >= max_force_n or force - initial_force >= 5.:
                reason = 'force_unavailable' if force is None else 'contact_force'
                break
            while index < len(path) and np.max(np.abs(desired - path[index])) < 1e-9:
                index += 1
            position_error, angle_error = _errors(actual, goal)
            if index == len(path) and position_error <= .003 and angle_error <= .03:
                reason = 'target_reached'
                break
            if index < len(path):
                delta = path[index] - desired
                proposed = desired + delta * min(1., (1. / robot._control_frequency) /
                    max(float(np.max(np.abs(delta))), 1e-12))
                reference = model.eef_pose(proposed, state.gripper_open_fraction)
                pe, ae = _errors(actual, reference)
                if np.max(np.abs(proposed - q)) <= .08 and pe <= .02 and ae <= .1:
                    desired = proposed
                else:
                    pacing += 1
            pe, ae = _errors(actual, model.eef_pose(desired, state.gripper_open_fraction))
            if np.max(np.abs(desired - q)) > .08 or pe > .02 or ae > .1:
                reason = 'tracking_error_limit'
                break
            if not robot._step(desired - q, robot._gripper_command_fraction):
                reason = robot._last_step_rejection or 'controller_stopped'
                break
            trace.append(self._sample(desired))
            force = trace[-1]['force_n']
            if force is None or force >= max_force_n or force - initial_force >= 5.:
                reason = 'force_unavailable' if force is None else 'contact_force'
                break
            measured = _pose(robot.get_robot_state().value)
            dp, dr = _errors(measured, actual)
            stalled = stalled + 1 if dp < .00005 and dr < .0005 else 0
            if stalled >= 10:
                reason = 'motion_stalled'
                break
        final = _pose(robot.get_robot_state().value)
        pe, ae = _errors(final, goal)
        reason = reason or ('target_reached' if index == len(path) and pe <= .003 and ae <= .03 else 'duration_exhausted')
        self._hold_reference = ({'joints': desired.copy(), 'end_step': robot.simulation_steps}
            if reason == 'target_reached' else None)
        return self._result(reason, robot.simulation_steps - first_step, trace,
            position_residual_m=pe, orientation_residual_rad=ae, command_pacing_steps=pacing,
            requested_segments=len(path)-1, force_limit_n=max_force_n,
            max_observed_force_n=max([initial_force] + [t['force_n'] for t in trace if t['force_n'] is not None]))

    def hold(self, duration_s=.5, *, gripper_fraction=None, max_force_n=18.):
        robot = self.robot
        if (not math.isfinite(duration_s) or not 0 < duration_s <= 30.
                or not math.isfinite(max_force_n) or not 0 < max_force_n <= 18.
                or (gripper_fraction is not None and gripper_fraction not in (0., 1.))):
            return self._result('invalid_hold_argument', 0, [])
        reason = self._initial_check(max_force_n)
        if reason:
            return self._result(reason, 0, [])
        state = robot.get_robot_state().value
        target = np.asarray(state.joint_positions_rad).copy()
        reference = getattr(self, '_hold_reference', None)
        retained = reference is not None and reference['end_step'] == robot.simulation_steps
        if retained:
            target = np.asarray(reference['joints']).copy()
        initial_force = self._force()
        fraction = robot._gripper_command_fraction if gripper_fraction is None else gripper_fraction
        first_step, trace = robot.simulation_steps, []
        reason = 'hold_complete'
        for _ in range(math.ceil(duration_s * robot._control_frequency)):
            q = np.asarray(robot.get_robot_state().value.joint_positions_rad)
            if np.max(np.abs(target - q)) > .08:
                reason = 'tracking_error_limit'
                break
            if not robot._step(target - q, fraction):
                reason = robot._last_step_rejection or 'controller_stopped'
                break
            robot._gripper_command_fraction = fraction
            robot._gripper_open_fraction = fraction
            trace.append(self._sample(target))
            force = trace[-1]['force_n']
            if force is None or force >= max_force_n or force - initial_force >= 5.:
                reason = 'force_unavailable' if force is None else 'contact_force'
                break
        pe, ae = _errors(_pose(robot.get_robot_state().value), _pose(state))
        self._hold_reference = ({'joints': target.copy(), 'end_step': robot.simulation_steps}
            if reason == 'hold_complete' else None)
        return self._result(reason, robot.simulation_steps-first_step, trace,
            position_residual_m=pe, orientation_residual_rad=ae,
            hold_reference='preceding_successful_command' if retained else 'measured_posture')

    def descend_and_close(self, position, quaternion, *, max_force_n=18.):
        descent = self.move(position, quaternion, max_force_n=max_force_n)
        if not descent.success:
            return replace(descent, value={**descent.value, 'closure_executed': False})
        closed = self.hold(self.robot._gripper_steps / self.robot._control_frequency,
            gripper_fraction=0., max_force_n=max_force_n)
        return replace(closed, executed=descent.executed or closed.executed,
            value={**closed.value, 'closure_executed': closed.executed,
                'descent_execution': descent.value, 'gripper_state': self.robot.gripper_state().value,
                'grasp_outcome': 'not_verified'})

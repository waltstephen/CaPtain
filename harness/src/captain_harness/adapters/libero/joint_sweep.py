"""Unregistered continuous native contact sweeps and measured-route retreat."""
import math

import numpy as np
from scipy.spatial.transform import Rotation

from captain_harness.ports.capability import CapabilityResult, FailureCategory
from .joint_cartesian import NativeJointCartesian, _errors, _pose, plan_cartesian_joints


def plan_sweep_joints(model, joints, aperture, poses, *, translation_step, rotation_step):
    """Preserve every public sweep segment instead of taking an endpoint chord."""
    poses = np.asarray(poses, dtype=float)
    if (poses.ndim != 3 or poses.shape[1:] != (4, 4) or not 2 <= len(poses) <= 1001
            or not np.isfinite(poses).all()):
        raise ValueError('sweep requires 2-1001 finite rigid transforms')
    for pose in poses:
        if (not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-8, rtol=0)
                or not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-8, rtol=0)):
            raise ValueError('sweep contains a non-rigid transform')
    if any(not math.isfinite(v) or v <= 0 for v in (translation_step, rotation_step)):
        raise ValueError('sweep resolutions must be finite and positive')
    result = [np.asarray(joints, dtype=float).copy()]
    for start, goal in zip(poses, poses[1:]):
        segment = plan_cartesian_joints(model, result[-1], aperture, start, goal,
            translation_step=translation_step, rotation_step=rotation_step)
        result.extend(segment[1:])
        if len(result) > 1001:
            raise ValueError('native sweep exceeds 1000 joint segments')
    return np.asarray(result)


class NativeJointSweep(NativeJointCartesian):
    def execute_sweep(self, poses, *, max_speed_m_s=.02, max_angular_speed_rad_s=.2,
                      max_force_n=18., max_duration_s=10.,
                      position_tolerance_m=.003, orientation_tolerance_rad=.03):
        try:
            self._validate_limits(max_speed_m_s, max_angular_speed_rad_s,
                                  max_force_n, max_duration_s)
            for tolerance, upper in ((position_tolerance_m, .1), (orientation_tolerance_rad, .5)):
                if isinstance(tolerance, bool) or not math.isfinite(tolerance) or not 0 < tolerance <= upper:
                    raise ValueError('invalid native pose tolerance')
            state = self.robot.get_robot_state().value
            poses = np.asarray(poses, dtype=float)
            # Measurement agreement precedes local IK planning.
            pe, ae = _errors(_pose(state), poses[0])
            if pe > .001 or ae > .01:
                raise ValueError('sweep start differs from measured robot pose')
            reason = self._initial_check(max_force_n)
            if reason:
                return self._result(reason, 0, [], reached=False)
            path = plan_sweep_joints(self.robot.planning_model(), state.joint_positions_rad,
                state.gripper_open_fraction, poses,
                translation_step=max_speed_m_s/self.robot._control_frequency,
                rotation_step=max_angular_speed_rad_s/self.robot._control_frequency)
        except (ValueError, TypeError, IndexError, np.linalg.LinAlgError) as exc:
            return self._result('cartesian_plan_rejected', 0, [], reached=False, diagnostic=str(exc))
        self.last_planned_joints = path.copy()
        return self._execute_reference(path, poses[-1], max_speed_m_s=max_speed_m_s,
            max_angular_speed_rad_s=max_angular_speed_rad_s, max_force_n=max_force_n,
            max_duration_s=max_duration_s, position_tolerance_m=position_tolerance_m,
            orientation_tolerance_rad=orientation_tolerance_rad)

    @staticmethod
    def _validate_limits(speed, angular_speed, force, duration):
        for value, upper in ((speed, .1), (angular_speed, 1.), (force, 18.), (duration, 30.)):
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= upper:
                raise ValueError('invalid native sweep speed, duration or force limit')

    def _execute_reference(self, path, goal, *, max_speed_m_s, max_angular_speed_rad_s,
                           max_force_n, max_duration_s, retreat=False,
                           position_tolerance_m=.003, orientation_tolerance_rad=.03):
        first = self.robot.simulation_steps
        try:
            return self._execute_reference_checked(path, goal,
                max_speed_m_s=max_speed_m_s, max_angular_speed_rad_s=max_angular_speed_rad_s,
                max_force_n=max_force_n, max_duration_s=max_duration_s, retreat=retreat,
                position_tolerance_m=position_tolerance_m, orientation_tolerance_rad=orientation_tolerance_rad)
        except Exception as exc:
            self._hold_reference = None
            self._saved_motion = None
            return CapabilityResult.failed(FailureCategory.BACKEND_ERROR,
                f'{type(exc).__name__}: {exc}', executed=self.robot.simulation_steps > first,
                value={'reached': False, 'stopped_reason': 'backend_execution_failed',
                       'control_steps': self.robot.simulation_steps-first, 'contact_identity': 'unknown'})

    def _execute_reference_checked(self, path, goal, *, max_speed_m_s, max_angular_speed_rad_s,
                                   max_force_n, max_duration_s, retreat=False,
                                   position_tolerance_m=.003, orientation_tolerance_rad=.03):
        robot = self.robot
        first = robot.simulation_steps
        initial = robot.get_robot_state().value
        start = _pose(initial)
        q0 = np.asarray(initial.joint_positions_rad)
        if np.max(np.abs(q0-path[0])) > 1e-6:
            return self._result('recorded_motion_is_stale', 0, [], reached=False)
        baseline = self._force()
        if baseline is None:
            return self._result('force_unavailable', 0, [], reached=False)
        if baseline >= max_force_n and not retreat:
            return self._result('initial_force_limit', 0, [], reached=False)
        model = robot.planning_model()
        trace, measured_joints, measured_poses = [], [q0.tolist()], [start.tolist()]
        desired = path[0].copy()
        index, stalled, pacing, reason, peak = 1, 0, 0, None, baseline
        force_floor = baseline

        def force_stop():
            nonlocal force_floor, peak
            force = self._force()
            if force is None:
                return 'force_unavailable'
            peak = max(peak, force)
            if retreat:
                if force >= force_floor + 5.:
                    return 'retreat_force_increased'
                force_floor = min(force_floor, force)
            elif force >= max_force_n or force - baseline >= 5.:
                return 'contact_force'
            return None

        for _ in range(math.ceil(max_duration_s*robot._control_frequency)):
            state = robot.get_robot_state().value
            actual, q = _pose(state), np.asarray(state.joint_positions_rad)
            reason = force_stop()
            if reason:
                break
            while index < len(path) and np.max(np.abs(desired-path[index])) < 1e-9:
                index += 1
            pe, ae = _errors(actual, goal)
            if index == len(path) and pe <= position_tolerance_m and ae <= orientation_tolerance_rad:
                reason = 'target_reached'
                break
            if index < len(path):
                reference_progress = {}
                proposed_index = index
                if retreat:
                    from .native_path_timing import advance_reference
                    proposed, proposed_index, reference_progress = advance_reference(model, path, desired,
                        index, state.gripper_open_fraction,
                        translation_budget=max_speed_m_s/robot._control_frequency,
                        rotation_budget=max_angular_speed_rad_s/robot._control_frequency,
                        joint_budget=1./robot._control_frequency)
                else:
                    delta = path[index]-desired
                    dp, da = _errors(model.eef_pose(desired, state.gripper_open_fraction),
                                      model.eef_pose(path[index], state.gripper_open_fraction))
                    fraction = min(1., 1./robot._control_frequency/max(float(np.max(np.abs(delta))), 1e-12),
                        max_speed_m_s/robot._control_frequency/max(dp, 1e-12),
                        max_angular_speed_rad_s/robot._control_frequency/max(da, 1e-12))
                    proposed = desired + delta*fraction
                pe, ae = _errors(actual, model.eef_pose(proposed, state.gripper_open_fraction))
                if np.max(np.abs(proposed-q)) <= .08 and pe <= .02 and ae <= .1:
                    desired = proposed
                    index = proposed_index
                else:
                    pacing += 1
                    reference_progress = {}
            else:
                reference_progress = {}
            pe, ae = _errors(actual, model.eef_pose(desired, state.gripper_open_fraction))
            if np.max(np.abs(desired-q)) > .08 or pe > .02 or ae > .1:
                reason = 'tracking_error_limit'
                break
            if not robot._step(desired-q, robot._gripper_command_fraction):
                reason = robot._last_step_rejection or 'controller_stopped'
                break
            sample = self._sample(desired)
            sample.update(reference_progress)
            trace.append(sample)
            after = _pose(robot.get_robot_state().value)
            measured_poses.append(after.tolist())
            measured_joints.append(sample['joint_positions_rad'])
            reason = force_stop()
            if reason:
                break
            dp, da = _errors(after, actual)
            command_pe, command_ae = _errors(after, model.eef_pose(desired, state.gripper_open_fraction))
            # A recorded stopped route includes near-stationary samples. A
            # reached microscopic command does not require measurable motion;
            # retain those waypoints without mistaking them for a blocked arm.
            unmet_command = command_pe > position_tolerance_m or command_ae > orientation_tolerance_rad
            stalled = stalled+1 if dp < .00005 and da < .0005 and unmet_command else 0
            if stalled >= 10:
                reason = 'motion_stalled'
                break
        final = _pose(robot.get_robot_state().value)
        pe, ae = _errors(final, goal)
        while index < len(path) and np.max(np.abs(desired-path[index])) < 1e-9:
            index += 1
        reason = reason or ('target_reached' if index == len(path) and pe <= position_tolerance_m and ae <= orientation_tolerance_rad
                            else 'duration_exhausted')
        steps = robot.simulation_steps-first
        if steps and not retreat:
            self._saved_motion = {'joints': measured_joints, 'poses': measured_poses,
                                   'end_step': robot.simulation_steps}
        self._hold_reference = ({'joints': desired.copy(), 'end_step': robot.simulation_steps}
                                if reason == 'target_reached' else None)
        return self._result(reason, steps, trace, reached=reason == 'target_reached',
            position_residual_m=pe, orientation_residual_rad=ae,
            position_tolerance_m=position_tolerance_m, orientation_tolerance_rad=orientation_tolerance_rad,
            actual_displacement_m=(final[:3, 3]-start[:3, 3]).tolist(),
            requested_segments=len(path)-1, command_pacing_steps=pacing,
            force_available=reason != 'force_unavailable', observed_force_n=self._force(),
            max_observed_force_n=peak, force_limit_n=max_force_n, retreat=retreat,
            reference_path_scope='continuous_local_joint_branch; scene_and_carried_object_collision_not_checked')

    def move(self, position, quaternion, *, max_speed_m_s=.05, max_duration_s=24., max_force_n=18.,
             position_tolerance_m=.003, orientation_tolerance_rad=.03):
        try:
            start = _pose(self.robot.get_robot_state().value)
            goal = np.eye(4)
            goal[:3, 3] = np.asarray(position, dtype=float).reshape(3)
            goal[:3, :3] = Rotation.from_quat(np.roll(np.asarray(quaternion, dtype=float).reshape(4), -1)).as_matrix()
            return self.execute_sweep([start, goal], max_speed_m_s=max_speed_m_s,
                max_angular_speed_rad_s=.5, max_duration_s=max_duration_s, max_force_n=max_force_n,
                position_tolerance_m=position_tolerance_m, orientation_tolerance_rad=orientation_tolerance_rad)
        except (ValueError, TypeError) as exc:
            return self._result('cartesian_plan_rejected', 0, [], reached=False, diagnostic=str(exc))

    def retreat_last_motion(self, max_distance_m=.05, max_duration_s=10., *,
                            position_tolerance_m=.003, orientation_tolerance_rad=.03):
        try:
            self._validate_limits(.01, .2, 18., max_duration_s)
            if not math.isfinite(max_distance_m) or not 0 < max_distance_m <= .25:
                raise ValueError('max_distance_m must be in (0, .25]')
            for tolerance, upper in ((position_tolerance_m, .1), (orientation_tolerance_rad, .5)):
                if isinstance(tolerance, bool) or not math.isfinite(tolerance) or not 0 < tolerance <= upper:
                    raise ValueError('invalid native retreat pose tolerance')
        except (ValueError, TypeError) as exc:
            return self._result('invalid_retreat_argument', 0, [], reached=False, diagnostic=str(exc))
        saved = getattr(self, '_saved_motion', None)
        if saved is None or saved['end_step'] != self.robot.simulation_steps:
            return self._result('recorded_motion_is_stale', 0, [], reached=False)
        poses = np.asarray(saved['poses'])
        joints = np.asarray(saved['joints'])
        distance, indices = 0., [len(poses)-1]
        for index in range(len(poses)-2, -1, -1):
            segment = np.linalg.norm(poses[index, :3, 3]-poses[index+1, :3, 3])
            if distance+segment > max_distance_m:
                break
            indices.append(index)
            distance += segment
        if len(indices) < 2:
            return self._result('no_recorded_retreat_segment', 0, [], reached=False)
        result = self._execute_reference(joints[indices], poses[indices[-1]],
            max_speed_m_s=.01, max_angular_speed_rad_s=.2, max_force_n=18.,
            max_duration_s=max_duration_s, retreat=True, position_tolerance_m=position_tolerance_m,
            orientation_tolerance_rad=orientation_tolerance_rad)
        if result.executed:
            self._saved_motion = None
        return result

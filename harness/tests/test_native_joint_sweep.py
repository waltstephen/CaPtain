"""Public robot measurements only; candidate remains absent from the factory."""
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from captain_harness.adapters.libero.joint_sweep import NativeJointSweep
from captain_harness.adapters.robotics.cartesian_sweeps import cartesian_sweep


class Model:
    limits = np.tile([-3., 3.], (7, 1))

    def eef_pose(self, q, aperture):
        pose = np.eye(4)
        pose[:3, 3] = q[:3]
        pose[:3, :3] = Rotation.from_rotvec(q[3:6]).as_matrix()
        return pose

    def _solve_ik_seed(self, pose, seed, **kwargs):
        q = np.array(seed, copy=True)
        q[:3] = pose[:3, 3]
        q[3:6] = Rotation.from_matrix(pose[:3, :3]).as_rotvec()
        return q


class Robot:
    controller = 'JOINT_POSITION'
    _control_frequency = 20.
    _gripper_steps = 2
    _gripper_command_fraction = 1.
    _gripper_open_fraction = 1.
    simulation_steps = 0
    _last_step_rejection = None
    _motion_guard = None

    def __init__(self):
        self.q = np.zeros(7)
        self.model = Model()
        self.force = lambda: 5.
        self.actions = []

    def planning_model(self):
        return self.model

    def get_robot_state(self):
        return SimpleNamespace(value=SimpleNamespace(joint_positions_rad=self.q.copy(),
            end_effector_position_m=self.q[:3].copy(),
            end_effector_quaternion_wxyz=np.roll(Rotation.from_rotvec(self.q[3:6]).as_quat(), 1),
            gripper_open_fraction=self._gripper_open_fraction))

    def _eef_force_norm(self):
        return self.force()

    def _step(self, dq, fraction):
        if self._motion_guard is not None and not self._motion_guard():
            self._last_step_rejection = 'motion_guard_rejected'
            return False
        self.q += dq
        self.simulation_steps += 1
        self.actions.append((self.q.copy(), fraction))
        return True

    def gripper_state(self):
        return SimpleNamespace(value={'object_detected': True, 'open_fraction': .3})


def test_rotating_about_pivot_preserves_arc_in_actual_joint_commands():
    robot = Robot()
    robot.q[0] = .3
    pivot = [.2, 0, 0]
    poses = cartesian_sweep(robot.model.eef_pose(robot.q, 1.),
                           axis=[0, 0, 1], pivot_m=pivot, angle_rad=np.pi/2)
    result = NativeJointSweep(robot).execute_sweep(poses, max_speed_m_s=.1,
                                                   max_angular_speed_rad_s=1., max_duration_s=10.)
    assert result.success
    positions = np.array([q[:3] for q, _ in robot.actions])
    np.testing.assert_allclose(np.linalg.norm(positions-pivot, axis=1), .1, atol=.0001)
    np.testing.assert_allclose(robot.q[:3], [.2, .1, 0], atol=.0001)


def test_later_segment_ik_failure_rejects_entire_sweep_before_motion():
    robot = Robot()
    solve = robot.model._solve_ik_seed
    robot.model._solve_ik_seed = lambda pose, seed, **kw: (
        None if pose[0, 3] > .012 else solve(pose, seed, **kw))
    result = NativeJointSweep(robot).move([.03, 0, 0], [1, 0, 0, 0])
    assert not result.success and not result.executed and not robot.actions


def test_force_stop_on_last_cycle_precedes_endpoint_success():
    robot = Robot()
    robot.force = lambda: 18. if robot.q[0] >= .005-1e-12 else 5.
    result = NativeJointSweep(robot).move([.005, 0, 0], [1, 0, 0, 0])
    assert not result.success and result.message == 'contact_force' and result.executed
    assert robot.simulation_steps == 2


def test_exact_duration_endpoint_is_not_reported_as_timeout():
    robot = Robot()
    result = NativeJointSweep(robot).move([.005, 0, 0], [1, 0, 0, 0], max_duration_s=.1)
    assert result.success and robot.simulation_steps == 2


def test_outer_guard_still_stops_native_sweep():
    robot = Robot()
    guard = lambda: robot.simulation_steps < 2
    robot._motion_guard = guard
    result = NativeJointSweep(robot).move([.03, 0, 0], [1, 0, 0, 0])
    assert not result.success and result.message == 'motion_guard_rejected'
    assert robot.simulation_steps == 2 and robot._motion_guard is guard


def test_contact_stopped_descent_does_not_close_and_can_reverse_measured_joints():
    robot = Robot()
    robot.force = lambda: 20. if robot.q[0] >= .005-1e-12 else 5.
    drive = NativeJointSweep(robot)
    result = drive.descend_and_close([.03, 0, 0], [1, 0, 0, 0])
    assert not result.success and result.value['closure_executed'] is False
    assert all(fraction == 1. for _, fraction in robot.actions)
    retreat = drive.retreat_last_motion(max_distance_m=.05)
    assert retreat.success and retreat.executed
    np.testing.assert_allclose(robot.q, 0., atol=1e-9)
    assert drive.retreat_last_motion().message == 'recorded_motion_is_stale'


def test_time_advance_makes_recorded_joint_route_stale():
    robot = Robot()
    drive = NativeJointSweep(robot)
    assert drive.move([.02, 0, 0], [1, 0, 0, 0]).success
    assert drive.hold(.1).success
    before = robot.simulation_steps
    result = drive.retreat_last_motion()
    assert not result.executed and result.message == 'recorded_motion_is_stale'
    assert robot.simulation_steps == before


def test_unavailable_force_before_motion_is_not_an_executed_failure():
    robot = Robot()
    robot.force = lambda: None
    result = NativeJointSweep(robot).move([.02, 0, 0], [1, 0, 0, 0])
    assert not result.executed and result.message == 'force_unavailable'


def test_driver_exception_after_motion_is_explicit_and_expires_recovery():
    robot = Robot()
    step = robot._step
    def broken(dq, fraction):
        step(dq, fraction)
        raise RuntimeError('driver disconnected')
    robot._step = broken
    drive = NativeJointSweep(robot)
    result = drive.move([.02, 0, 0], [1, 0, 0, 0])
    assert not result.success and result.executed
    assert result.value['stopped_reason'] == 'backend_execution_failed'
    assert drive.retreat_last_motion().message == 'recorded_motion_is_stale'


def test_retreat_stops_on_force_rise_after_a_previous_decrease():
    robot = Robot()
    drive = NativeJointSweep(robot)
    assert drive.move([.02, 0, 0], [1, 0, 0, 0]).success
    first = robot.simulation_steps
    robot.force = lambda: [20., 10., 16.][min(robot.simulation_steps-first, 2)]
    result = drive.retreat_last_motion()
    assert not result.success and result.message == 'retreat_force_increased'
    assert result.value['control_steps'] == 2
    assert drive.retreat_last_motion().message == 'recorded_motion_is_stale'


def test_nearly_stationary_recorded_tail_is_not_a_failed_retreat():
    robot = Robot()
    drive = NativeJointSweep(robot)
    path = np.zeros((14, 7))
    path[:13, 0] = np.arange(13)*1e-6
    path[-1, 0] = .01
    result = drive._execute_reference(path, robot.model.eef_pose(path[-1], 1.),
        max_speed_m_s=.01, max_angular_speed_rad_s=.2, max_force_n=18.,
        max_duration_s=3., retreat=True)
    assert result.success
    np.testing.assert_allclose(robot.q, path[-1], atol=1e-9)


def test_unmoving_robot_with_unmet_pose_command_still_stops_as_stalled():
    robot = Robot()
    def blocked(dq, fraction):
        robot.simulation_steps += 1
        return True
    robot._step = blocked
    result = NativeJointSweep(robot).move([.03, 0, 0], [1, 0, 0, 0])
    assert not result.success and result.message == 'motion_stalled'


def test_driver_converges_to_callers_stricter_endpoint_tolerance():
    robot = Robot()
    def lagging(dq, fraction):
        robot.q += .2*dq
        robot.simulation_steps += 1
        return True
    robot._step = lagging
    result = NativeJointSweep(robot).move([.02, 0, 0], [1, 0, 0, 0],
        position_tolerance_m=.001, orientation_tolerance_rad=.01)
    assert result.success
    assert result.value['position_residual_m'] <= .001


def test_dense_measured_return_route_fits_physical_distance_budget():
    robot = Robot()
    path = np.zeros((251, 7)); path[:, 0] = np.linspace(0, .005, 251)
    result = NativeJointSweep(robot)._execute_reference(path, robot.model.eef_pose(path[-1], 1.),
        max_speed_m_s=.01, max_angular_speed_rad_s=.2, max_force_n=18.,
        max_duration_s=1., retreat=True)
    assert result.success
    assert result.value['control_steps'] <= 20
    assert sum(p.get('traversed_waypoint_count', 0) for p in result.value['trajectory']) == 250

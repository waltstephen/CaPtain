"""A singular starting posture is not evidence that the endpoint is unreachable."""
import numpy as np

from captain_harness.adapters.robotics.model import RobotCollisionModel


XML = '''<mujoco><compiler angle="radian"/><worldbody><body name="base">
<body><joint name="a" axis="0 0 1" range="-3 3" limited="true"/>
<geom type="sphere" size=".01" mass="1"/>
<body pos="1 0 0"><joint name="b" axis="0 0 1" range="-3 3" limited="true"/>
<geom type="sphere" size=".01" mass="1"/>
<body pos="1 0 0"><joint name="c" axis="0 0 1" range="-3 3" limited="true"/>
<geom type="sphere" size=".01" mass="1"/><body name="eef" pos="1 0 0"/>
</body></body></body></body></worldbody></mujoco>'''


def test_ik_escapes_a_singular_seed_with_bounded_hardware_only_restarts():
    robot = RobotCollisionModel(XML, joints=['a', 'b', 'c'], gripper_joints=[],
        base='base', end_effector='eef', gripper_prefix='unused')
    target = np.eye(4)
    target[:3, 3] = [2., 0., 0.]
    solved = robot.solve_ik(target, [0., 0., 0.])
    assert solved is not None, robot.last_ik
    np.testing.assert_allclose(robot.eef_pose(solved)[:3, 3], target[:3, 3], atol=.002)
    assert np.all(solved >= robot.limits[:, 0]) and np.all(solved <= robot.limits[:, 1])
    assert 1 < robot.last_ik['seed_attempts'] <= 5
    assert robot.last_ik['total_iterations'] <= 5 * 180


def test_ik_failure_keeps_best_residual_and_never_returns_an_approximate_solution():
    robot = RobotCollisionModel(XML, joints=['a', 'b', 'c'], gripper_joints=[],
        base='base', end_effector='eef', gripper_prefix='unused')
    target = np.eye(4)
    target[:3, 3] = [10., 0., 0.]
    assert robot.solve_ik(target, [0., 0., 0.]) is None
    assert robot.last_ik['position_residual_m'] >= 7. - 1e-6
    assert robot.last_ik['seed_attempts'] <= 5


def test_ik_retries_an_endpoint_solution_rejected_by_scene_validity():
    robot = RobotCollisionModel(XML, joints=['a', 'b', 'c'], gripper_joints=[],
        base='base', end_effector='eef', gripper_prefix='unused')
    target = np.eye(4)
    target[:3, 3] = [2., 0., 0.]
    # The positive elbow solution already reaches the endpoint but is invalid
    # in the caller's scene. A mirrored elbow solution reaches the same pose.
    seed = [np.pi / 3, -2 * np.pi / 3, np.pi / 3]
    solved = robot.solve_ik(target, seed, valid=lambda q: q[0] < 0.)
    assert solved is not None and solved[0] < 0.
    np.testing.assert_allclose(robot.eef_pose(solved)[:3, 3], target[:3, 3], atol=.002)
    assert robot.last_ik['pose_solution_count'] >= 2


def test_validity_rejection_never_returns_a_colliding_ik_solution():
    robot = RobotCollisionModel(XML, joints=['a', 'b', 'c'], gripper_joints=[],
        base='base', end_effector='eef', gripper_prefix='unused')
    target = np.eye(4)
    target[:3, 3] = [2., 0., 0.]
    assert robot.solve_ik(target, [0., 0., 0.], valid=lambda q: False) is None
    assert robot.last_ik['pose_solution_count'] > 0
    assert robot.last_ik['valid_solution'] is False

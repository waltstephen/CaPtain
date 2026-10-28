"""Round hardware shapes must retain their geometry in a detached robot model."""
import numpy as np
import pytest

from test_sensor_motion_tools import XML
from captain_harness.adapters.robotics.model import RobotCollisionModel


@pytest.mark.parametrize('shape,size,z_inside,z_outside,corner,support', [
    ('sphere', '.1', .09, .11, [.08,.08,0], -.1),
    ('cylinder', '.1 .2', .19, .21, [.08,.08,0], -.3/np.sqrt(2)),
    ('capsule', '.1 .2', .29, .31, [.08,0,.28], -.1-.2/np.sqrt(2)),
])
def test_round_shapes_have_correct_membership_margin_and_support(shape, size, z_inside, z_outside, corner, support):
    xml = XML.replace('type="box" size=".05 .05 .05"', f'type="{shape}" size="{size}"')
    robot = RobotCollisionModel(xml, joints=['x','y'], gripper_joints=[],
        base='base', end_effector='eef', gripper_prefix='arm_')
    robot.set_state([0,0], 1.)
    points = np.array([[0,0,-z_inside], [0,0,-z_outside], [.102,0,0], [.12,0,0], corner])
    mask,_ = robot.points_in_robot(points, gripper_only=True, margin=.003)
    np.testing.assert_array_equal(mask, [True,False,True,False,False])
    normal = np.array([1.,0,1.])/np.sqrt(2)
    assert robot.gripper_support_distance(robot.eef_pose(), normal, 0) == pytest.approx(support)
    # Coordinate translation must not change the shape membership.
    robot.set_state([.4,0], 1.)
    moved,_ = robot.points_in_robot(points+[.4,0,0], gripper_only=True, margin=.003)
    np.testing.assert_array_equal(moved,mask)

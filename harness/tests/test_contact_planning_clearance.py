"""Contact clearance applies only to the selected target and gripper."""
import numpy as np
from test_sensor_motion_tools import model


def test_target_near_finger_uses_contact_clearance_without_ignoring_collision():
    robot = model()
    near_finger = np.array([[.026, 0., .15]])
    assert robot.collision([0., 0.], 1., near_finger, .01)['collision']
    assert not robot.collision([0., 0.], 1., np.empty((0, 3)), .01,
        contact_points=near_finger, contact_margin=.003)['collision']
    # Points inside the physical finger remain collisions even for the target.
    assert robot.collision([0., 0.], 1., np.empty((0, 3)), .01,
        contact_points=np.array([[.018, 0., .15]]), contact_margin=.003)['collision']


def test_other_objects_and_the_arm_keep_transport_clearance():
    robot = model()
    near_finger = np.array([[.026, 0., .15]])
    assert robot.collision([0., 0.], 1., near_finger, .01,
        contact_points=np.empty((0, 3)), contact_margin=.003)['collision']
    near_arm = np.array([[.056, 0., 0.]])
    result = robot.collision([0., 0.], 1., np.empty((0, 3)), .01,
        contact_points=near_arm, contact_margin=.003)
    assert result['collision'] and 'arm_box' in result['observed_collisions']

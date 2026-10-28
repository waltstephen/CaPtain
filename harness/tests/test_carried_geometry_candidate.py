from itertools import product
import numpy as np
import pytest
from captain_harness.adapters.robotics.carried_geometry import (
    ObservedCarriedGeometry, partition_measured_target, check_carried_configuration)


def cube(center=(0,0,0), half=.02):
    return np.array(list(product((-half,half), repeat=3))) + center


def pose(x=0, y=0, z=0):
    result = np.eye(4); result[:3,3] = [x,y,z]
    return result


def test_carried_geometry_catches_payload_collision_with_clear_hand():
    geom = ObservedCarriedGeometry.from_observation(cube((0,0,-.12)), pose())
    obstacle = np.array([[.2,0,-.12]])
    assert not geom.scene_collision(pose(), obstacle, .003)['collision']
    result = geom.scene_collision(pose(.2), obstacle, .003)
    assert result['collision'] and result['observed_scene_points_in_carried_hull'] == 1
    assert result['geometry_scope'] == 'partial_visible_convex_hull'


def test_payload_rotates_with_eef_and_does_not_alias_input():
    points = cube((.12,0,0)); start = pose(.1)
    geom = ObservedCarriedGeometry.from_observation(points, start)
    points[:] = 100
    end = pose(.1);end[:3,:3] = [[0,-1,0],[1,0,0],[0,0,1]]
    np.testing.assert_allclose(geom.world_points(end).mean(axis=0), [.1,.02,0], atol=1e-12)
    assert geom.scene_collision(end, [[.1,.02,0]], 0)['collision']
    assert not geom.scene_collision(end, [[.14,0,0]], 0)['collision']


def test_scene_exclusion_only_removes_measured_correspondences_not_a_box():
    target = cube()
    background = np.array([[0,0,0],[.2,0,0],[.02001,.02,.02]])
    scene = np.concatenate([target, background])
    other, selected = partition_measured_target(scene, target)
    np.testing.assert_array_equal(selected, target)
    np.testing.assert_array_equal(other, background)


@pytest.mark.parametrize('bad_points', [[], [[0,0,0]], [[0,0,0],[1,0,0],[0,1,0],[1,1,0]], [[np.nan,0,0]]])
def test_insufficient_geometry_is_rejected(bad_points):
    with pytest.raises(ValueError): ObservedCarriedGeometry.from_observation(bad_points, pose())


@pytest.mark.parametrize('bad_pose', [np.zeros((4,4)), np.diag([2,1,1,1]), np.full((4,4),np.nan)])
def test_nonrigid_or_invalid_pose_is_rejected(bad_pose):
    with pytest.raises(ValueError): ObservedCarriedGeometry.from_observation(cube(), bad_pose)


def test_negative_margin_and_nonfinite_scene_cannot_report_clear():
    geom=ObservedCarriedGeometry.from_observation(cube(), pose())
    with pytest.raises(ValueError): geom.scene_collision(pose(), [], -.01)
    with pytest.raises(ValueError): geom.scene_collision(pose(), [[np.nan,0,0]], .003)


def test_payload_check_composes_with_independent_robot_kinematics():
    from test_sensor_motion_tools import model
    robot = model()
    geometry = ObservedCarriedGeometry.from_observation(cube((.15,0,.15)), robot.eef_pose([0,0]))
    obstacle = np.array([[.45,0,.15]])
    assert not robot.collision([.3,0], 1., obstacle)['collision']
    checked = check_carried_configuration(robot, [.3,0], 1., obstacle, geometry, .003)
    assert checked['collision'] and checked['carried']['collision']
    assert not checked['robot']['collision']


def test_payload_cannot_pass_through_non_gripper_robot_even_with_empty_scene():
    from test_sensor_motion_tools import model
    robot = model()
    geometry = ObservedCarriedGeometry.from_observation(cube(), robot.eef_pose([0,0]))
    checked = check_carried_configuration(robot, [0,0], 1., np.empty((0,3)), geometry, .003)
    assert checked['collision'] and checked['carried_points_in_non_gripper_robot']

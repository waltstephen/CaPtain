"""A detached observed payload must participate in original planning geometry."""
import numpy as np
import pytest
from captain_harness.adapters.robotics.model import RobotCollisionModel
from captain_harness.adapters.robotics.swept_collision import SweptSceneValidator


def robot_model():
    xml="""<mujoco><worldbody><body name="base">
      <body name="arm"><joint name="x" type="slide" axis="1 0 0" range="-1 1" limited="true"/>
      <geom name="arm_geometry" type="box" size=".03 .03 .03" pos="0 0 .18"/>
      <body name="eef"><joint name="y" type="slide" axis="0 1 0" range="-1 1" limited="true"/>
      <geom name="hand_geometry" type="box" size=".02 .02 .02"/>
      </body></body></body></worldbody></mujoco>"""
    return RobotCollisionModel(xml,joints=['x','y'],gripper_joints=[],base='base',
        end_effector='eef',gripper_prefix='hand_',angle_unit='radian')


def box(center,half=.015):
    return np.array([[x,y,z] for x in [-half,half] for y in [-half,half] for z in [-half,half]])+center


def test_payload_only_scene_collision_and_unchanged_robot_kinematics():
    original=robot_model();q=np.array([.1,.05]);vertices=box([.2,0,0])
    augmented=original.with_observed_carried_geometry(vertices)
    np.testing.assert_array_equal(augmented.limits,original.limits)
    np.testing.assert_allclose(augmented.eef_pose(q),original.eef_pose(q),rtol=0,atol=1e-12)
    obstacle=np.array([[.3,.05,0.]])
    assert not original.collision(q,1.,obstacle)['collision']
    result=augmented.collision(q,1.,obstacle)
    assert result['collision']
    assert augmented.carried_geometry_name in result['observed_collisions']
    assert original.carried_geometry_name is None


def test_payload_cannot_hide_arm_collision_behind_parent_body_filter():
    original=robot_model();q=np.zeros(2)
    augmented=original.with_observed_carried_geometry(box([0,0,.18],.02))
    assert not original.collision(q,1.,np.empty((0,3)))['collision']
    hit=augmented.collision(q,1.,np.empty((0,3)))
    assert hit['collision']
    assert any('arm_geometry' in pair and augmented.carried_geometry_name in pair for pair in hit['self_collisions'])


def test_payload_does_not_create_forbidden_gripper_self_contacts():
    augmented=robot_model().with_observed_carried_geometry(box([0,0,0],.01))
    assert not augmented.collision([0,0],1.,np.empty((0,3)))['collision']


def test_swept_check_detects_carried_collision_between_clear_endpoints():
    model=robot_model().with_observed_carried_geometry(box([.2,0,0],.01))
    obstacle=np.array([[.25,0,0]])
    def collision(q,extra_margin=0.):
        return model.collision(q,1.,obstacle,margin=.003,
            **({'geometry_padding_m':extra_margin} if isinstance(extra_margin,dict) else {}))
    assert not collision([0,0])['collision'] and not collision([.1,0])['collision']
    swept=SweptSceneValidator(collision,lambda d:model.geometry_point_motion(d,point_padding_m=.006))
    assert not swept([0,0],[.1,0])
    assert swept.receipt()['colliding_midpoints']>=1


def test_pose_tracking_reserve_includes_carried_lever_arm():
    model=robot_model().with_observed_carried_geometry(box([.4,0,0],.02))
    padding=model.gripper_pose_error_padding([0,0],1.,position_tolerance_m=.003,orientation_tolerance_rad=.03)
    assert padding[model.carried_geometry_name]>padding['hand_geometry']
    assert padding['arm_geometry']==0.


@pytest.mark.parametrize('points',[np.zeros((4,3)),[[0,0,0],[1,0,0],[0,1,0]],
    [[0,0,0],[1,0,0],[0,1,0],[1,1,0]],[[0,0,0],[1,0,0],[0,1,0],[0,0,float('nan')]]])
def test_inadequate_geometry_is_rejected_without_mutating_robot(points):
    model=robot_model();before=model.eef_pose([.1,.2]).copy()
    with pytest.raises(ValueError):model.with_observed_carried_geometry(points)
    assert model.carried_geometry_name is None
    np.testing.assert_array_equal(before,model.eef_pose())


def test_negative_robot_contact_mask_cannot_hide_a_payload_pair():
    model=robot_model();model.model.geom_contype[0]=-1
    with pytest.raises(ValueError,match='negative robot contact masks'):
        model.with_observed_carried_geometry(box([.2,0,0]))


def test_real_panda_rotated_offset_mesh_preserves_fk_and_measured_vertices():
    from robosuite.models.robots import Panda
    from robosuite.models.grippers import gripper_factory
    from scipy.spatial.transform import Rotation
    robot=Panda(idn=0);gripper=gripper_factory('PandaGripper',idn=0);robot.add_gripper(gripper)
    original=RobotCollisionModel(robot.get_xml(),joints=list(robot.joints),
        gripper_joints=list(gripper.joints),base='robot0_base',end_effector='gripper0_eef',
        gripper_prefix='gripper0_',angle_unit='radian')
    rotation=Rotation.from_euler('xyz',[.3,.7,.4]).as_matrix()
    points=(box([0,0,0]) * [3.,1.,2.])@rotation.T + [.19,.11,.2]
    augmented=original.with_observed_carried_geometry(points)
    for q in ([0,-.7,0,-2.3,0,1.6,.7],[.3,-.5,.2,-1.8,-.3,1.4,.2]):
        pose=original.eef_pose(q,.4)
        np.testing.assert_allclose(augmented.eef_pose(q,.4),pose,rtol=0,atol=1e-12)
        expected=points@pose[:3,:3].T+pose[:3,3]
        geom=next(g for g in augmented.geometries if g.name==augmented.carried_geometry_name)
        compiled=geom.vertices@augmented.data.geom_xmat[geom.geom_id].reshape(3,3).T+augmented.data.geom_xpos[geom.geom_id]
        # Compilers may center/rotate mesh coordinates, but world geometry must agree.
        from scipy.spatial import cKDTree
        np.testing.assert_allclose(cKDTree(compiled).query(expected)[0],0.,atol=1e-7)
        center=expected.mean(axis=0)
        assert not original.collision(q,.4,[center])['collision']
        hit=augmented.collision(q,.4,[center])
        assert augmented.carried_geometry_name in hit['observed_collisions']

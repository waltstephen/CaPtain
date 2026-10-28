import numpy as np
import pytest
from test_sensor_motion_tools import model, XML
from captain_harness.adapters.robotics.model import RobotCollisionModel
from captain_harness.adapters.robotics.planning import SensorScene
from captain_harness.adapters.robotics.swept_collision import SweptSceneValidator
from captain_harness.adapters.capx.motion_tools import MotionToolMixin


@pytest.mark.parametrize('hinge',[False,True])
def test_each_hardware_geometry_bound_covers_its_own_vertices(hinge):
    xml=XML.replace('type="slide" axis="1 0 0" range="-1 1"',
        'type="hinge" pos=".4 .1 0" axis="0 0 1" range="-2 2"') if hinge else XML
    robot=RobotCollisionModel(xml,joints=['x','y'],gripper_joints=[],base='base',
        end_effector='eef',gripper_prefix='hand_',angle_unit='radian')
    a=np.array([-.7,.8]);b=np.array([.6,-.4]);padding=.02
    bounds=robot.geometry_point_motion(b-a,point_padding_m=padding)
    robot.set_state(a,1.)
    local={g.name:np.concatenate([g.vertices,(g.radius+padding)*np.r_[np.eye(3),-np.eye(3)]])
        for g in robot.geometries if g.moving and g.collidable}
    first={g.name:local[g.name]@robot.geom_pose(g)[1].T+robot.geom_pose(g)[0]
        for g in robot.geometries if g.name in local}
    assert bounds.keys()==first.keys()
    assert robot.max_point_motion(b-a,point_padding_m=padding)==max(bounds.values())
    for fraction in np.linspace(0,1,101):
        robot.set_state(a+(b-a)*fraction,1.)
        for g in robot.geometries:
            if g.name in first:
                actual=local[g.name]@robot.geom_pose(g)[1].T+robot.geom_pose(g)[0]
                assert np.max(np.linalg.norm(actual-first[g.name],axis=1))<=bounds[g.name]+1e-12


def test_distal_slide_does_not_inflate_unchanged_upstream_link():
    robot=model();points=np.array([[-.06,0.,0.]])
    scene=SensorScene(points,'public',(0.,-.02),1.,0.,.003,0)
    check,_=MotionToolMixin._scene_collision_checker(None,robot,scene)
    collision=lambda q,extra_margin=0.:check(q,extra_margin=extra_margin,first_hit_only=True)
    scalar=SweptSceneValidator(collision,lambda d:robot.max_point_motion(d,point_padding_m=.006))
    precise=SweptSceneValidator(collision,lambda d:robot.geometry_point_motion(d,point_padding_m=.006))
    start,end=np.array([0.,-.02]),np.array([0.,.02])
    assert scalar(start,end) and precise(start,end)
    assert precise.stats['collision_checks']==1
    assert scalar.stats['collision_checks']>precise.stats['collision_checks']
    for q in np.linspace(start,end,101):assert not check(q)['collision']
    bounds=robot.geometry_point_motion([0.,.02])
    assert bounds['arm_box']==0 and bounds['hand_box']==pytest.approx(.02)


@pytest.mark.parametrize('bad',[{'hand_box':.01},{'arm_box':0.,'hand_box':float('nan')},
    {'arm_box':0.,'hand_box':-.01},{'arm_box':0.,'hand_box':.01,'unknown':0.},
    {'arm_box':0.,'hand_box':1j},{'arm_box':0.,'hand_box':True}])
def test_incomplete_or_invalid_geometry_mapping_cannot_certify_clear(bad):
    with pytest.raises(ValueError):
        model().collision([0.,0.],1.,np.empty((0,3)),geometry_padding_m=bad)


def test_per_geometry_inflation_keeps_selected_target_contact_policy():
    robot=model();points=np.array([[.026,0.,.15]])
    scene=SensorScene(points,'public',(0.,0.),1.,0.,.01,0)
    check,policy=MotionToolMixin._scene_collision_checker(None,robot,scene,points)
    assert not check([0.,0.])['collision']
    assert check([0.,0.],extra_margin={'arm_box':0.,'hand_box':.004})['collision']
    assert policy['gripper_target_margin_m']==.003
    assert policy['other_surface_and_arm_margin_m']==.01


def test_per_geometry_edge_still_rejects_nominal_midpoint_obstacle():
    robot=model();points=np.array([[0.,0.,.15]])
    scene=SensorScene(points,'public',(-.05,0.),1.,0.,.003,0)
    check,_=MotionToolMixin._scene_collision_checker(None,robot,scene)
    collision=lambda q,extra_margin=0.:check(q,extra_margin=extra_margin,first_hit_only=True)
    edge=SweptSceneValidator(collision,lambda d:robot.geometry_point_motion(d,point_padding_m=.006))
    assert not check([-.05,0.])['collision'] and not check([.05,0.])['collision']
    assert not edge([-.05,0.],[.05,0.])
    assert edge.receipt()['first_rejection']['kind']=='nominal_collision'

import numpy as np
import pytest
from captain_harness.adapters.robotics import planning
from captain_harness.ports.deadline import deadline_scope, DeadlineExceeded


def validator():
    from captain_harness.adapters.robotics.swept_collision import SweptSceneValidator
    def collision(q, *, extra_margin=0.):
        inside=abs(float(q[0])-.005) <= .001 + extra_margin and abs(float(q[1])) <= .001 + extra_margin
        return {'collision':inside,'observed_collisions':{'obstacle':1} if inside else {},'self_collisions':[]}
    return SweptSceneValidator(collision,lambda delta:float(np.abs(delta).sum())),collision


def test_narrow_obstacle_between_clear_endpoints_and_midpoint_is_rejected():
    edge,check=validator()
    assert all(not check(q)['collision'] for q in ([0,0],[.01,0],[.02,0]))
    assert not edge([0,0],[.02,0])
    assert edge([0,.02],[.02,.02])


def test_validated_detour_and_shortcuts_never_reintroduce_between_sample_collision(monkeypatch):
    edge,check=validator()
    calls=[]
    def candidate(start,goal,*args,**kwargs):
        calls.append(1)
        return np.array([start,[0,.03],[.02,.03],goal])
    monkeypatch.setattr(planning,'_rrt_candidate',candidate)
    path,info=planning.plan_joint_path([0,0],[.02,0],[[-1,1],[-1,1]],
        lambda q:not check(q)['collision'],edge_valid=edge,timeout_s=1.)
    assert path is not None and calls
    assert all(edge(a,b) for a,b in zip(path[:-1],path[1:]))
    assert info['edge_checks']>0
    assert np.max(np.abs(np.diff(path,axis=0))) <= .025+1e-12


def test_repeated_rrt_proposals_with_bad_edges_return_no_plan(monkeypatch):
    edge,check=validator()
    monkeypatch.setattr(planning,'_rrt_candidate',lambda start,goal,*a,**k:np.array([start,goal]))
    path,info=planning.plan_joint_path([0,0],[.02,0],[[-1,1],[-1,1]],
        lambda q:not check(q)['collision'],edge_valid=edge,timeout_s=1.)
    assert path is None and info['rejected_candidate_paths']==3


def test_swept_checks_obey_earlier_parent_deadline(monkeypatch):
    from captain_harness.adapters.robotics.swept_collision import SweptSceneValidator
    from captain_harness.ports import deadline
    now=[0.]
    monkeypatch.setattr(deadline.time,'monotonic',lambda:now[0])
    def collision(q,*,extra_margin=0.):
        now[0]+=.04
        return {'collision':extra_margin>0}
    edge=SweptSceneValidator(collision,lambda delta:float(np.abs(delta).sum()))
    with deadline_scope('policy',timeout_s=.1):
        with pytest.raises(DeadlineExceeded) as exc:edge([0],[1])
    assert exc.value.scope=='policy'


def test_unresolved_clearance_is_rejected_instead_of_assumed_clear():
    from captain_harness.adapters.robotics.swept_collision import SweptSceneValidator
    edge=SweptSceneValidator(lambda q,extra_margin=0.:{'collision':extra_margin>0},
        lambda delta:float(np.abs(delta).sum()))
    assert not edge([0.],[1e-8])
    assert edge.stats['unresolved_intervals']==1


@pytest.mark.parametrize('rotation',[False,True])
def test_hardware_motion_bound_covers_vertices_with_offset_pivots(rotation):
    from test_sensor_motion_tools import XML
    from captain_harness.adapters.robotics.model import RobotCollisionModel
    xml=XML
    if rotation:
        xml=xml.replace('type="slide" axis="1 0 0" range="-1 1"',
            'type="hinge" pos=".4 .1 0" axis="0 0 1" range="-2 2"')
    robot=RobotCollisionModel(xml,joints=['x','y'],gripper_joints=[],base='base',
        end_effector='eef',gripper_prefix='hand_',angle_unit='radian')
    a=np.array([-.7,.8]);b=np.array([.6,-.4])
    bound=robot.max_point_motion(b-a)
    assert np.isfinite(bound) and bound>0
    robot.set_state(a,1.)
    first={g.name:g.vertices@robot.geom_pose(g)[1].T+robot.geom_pose(g)[0]
        for g in robot.geometries if g.collidable and g.moving}
    for f in np.linspace(0,1,101):
        robot.set_state(a+f*(b-a),1.)
        for g in robot.geometries:
            if g.name in first:
                actual=g.vertices@robot.geom_pose(g)[1].T+robot.geom_pose(g)[0]
                assert np.max(np.linalg.norm(actual-first[g.name],axis=1)) <= bound+1e-12
    assert robot.max_point_motion(np.zeros(2))==0


def test_shortcut_edge_check_local_expiry_preserves_validated_remainder(monkeypatch):
    from captain_harness.ports import deadline
    now=[0.]
    monkeypatch.setattr(deadline.time,'monotonic',lambda:now[0])
    path=np.array([[0.,0.],[0.,.02],[.02,.02],[.02,0.]])
    def edge(a,b):
        now[0]=.6
        deadline.check_deadline()
        return True
    with deadline_scope('planning',expires_s=1.):
        result,info=planning._shorten_validated_path(path,lambda q:True,
            expires_s=.5,max_joint_step=.025,edge_valid=edge)
    np.testing.assert_array_equal(result,path)
    assert info['budget_exhausted'] and now[0]<1.


def test_actual_selected_contact_checker_inflates_both_contact_and_other_surfaces():
    from test_sensor_motion_tools import model
    from captain_harness.adapters.capx.motion_tools import MotionToolMixin
    from captain_harness.adapters.robotics.planning import SensorScene
    robot=model();points=np.array([[.026,0.,.15]])
    scene=SensorScene(points,'public',(0.,0.),1.,0.,.01,0)
    contact,_=MotionToolMixin._scene_collision_checker(None,robot,scene,points)
    assert not contact([0.,0.])['collision']
    assert contact([0.,0.],extra_margin=.004)['collision']
    free,_=MotionToolMixin._scene_collision_checker(None,robot,scene)
    assert not free([.1,0.])['collision']
    assert free([.1,0.],extra_margin=.1)['collision']


def test_ompl_search_checks_edges_during_tree_growth():
    edge,check=validator()
    used=[]
    def checked_edge(a,b):used.append((a.copy(),b.copy()));return edge(a,b)
    path=planning._rrt_candidate(np.array([0.,0.]),np.array([.02,0.]),
        np.array([[-.03,.05],[-.04,.04]]),lambda q:not check(q)['collision'],
        timeout_s=1.,max_joint_step=.025,edge_valid=checked_edge)
    assert path is not None and used
    assert all(edge(a,b) for a,b in zip(path[:-1],path[1:]))


def test_large_edges_subdivide_before_large_scene_inflation():
    from captain_harness.adapters.robotics.swept_collision import SweptSceneValidator
    margins=[]
    def collision(q,extra_margin=0.):margins.append(extra_margin);return {'collision':False}
    edge=SweptSceneValidator(collision,lambda d:float(np.abs(d).sum()),maximum_inflation_m=.003)
    assert edge([0.],[.2])
    assert max(margins)<=.003 and len(margins)>2


def test_rotation_bound_includes_inflated_collision_envelope():
    from captain_harness.adapters.robotics.model import RobotCollisionModel
    xml='<mujoco><worldbody><body name="base"><body name="eef"><joint name="yaw" type="hinge" axis="0 0 1" range="-1 1" limited="true"/><geom name="hand" type="box" size=".02 .02 .02"/></body></body></worldbody></mujoco>'
    robot=RobotCollisionModel(xml,joints=['yaw'],gripper_joints=[],base='base',end_effector='eef',gripper_prefix='hand',angle_unit='radian')
    local=np.array([.3,0.,0.]);a=robot.eef_pose([0.]);b=robot.eef_pose([.5])
    distance=np.linalg.norm((a[:3,:3]@local+a[:3,3])-(b[:3,:3]@local+b[:3,3]))
    assert robot.max_point_motion([.5],point_padding_m=.3)>=distance
    assert robot.max_point_motion([.5])<distance

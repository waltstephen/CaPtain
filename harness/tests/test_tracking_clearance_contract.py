import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from types import SimpleNamespace
from test_sensor_motion_tools import model
from test_whole_planning_deadline import make_tool,clock_at
from captain_harness.ports.capability import CapabilityResult,FailureCategory
from captain_harness.ports.deadline import deadline_scope


def test_gripper_pose_allowance_covers_original_inflated_envelope():
    robot=model();q=[.2,-.3];aperture=1.;pos=.003;angle=.03;inflation=.02
    pads=robot.gripper_pose_error_padding(q,aperture,position_tolerance_m=pos,
        orientation_tolerance_rad=angle,point_padding_m=inflation)
    eef=robot.eef_pose(q,aperture);origin=eef[:3,3]
    assert pads['arm_box']==0.
    assert pads['hand_box']>pos
    for geom in robot.geometries:
        if geom.name!='hand_box':continue
        c,r=robot.geom_pose(geom)
        local=(geom.radius+inflation)*np.r_[np.eye(3),-np.eye(3)]
        world=local@r.T+c
        for axis in [[1,0,0],[0,1,0],[0,0,1],[1,2,3]]:
            axis=np.asarray(axis)/np.linalg.norm(axis)
            rotate=Rotation.from_rotvec(axis*angle).as_matrix()
            moved=(world-origin)@rotate.T+origin+pos*axis
            assert np.linalg.norm(moved-world,axis=1).max()<=pads[geom.name]+1e-12


@pytest.mark.parametrize('changes',[{'position_tolerance_m':-1.},{'position_tolerance_m':True},
    {'orientation_tolerance_rad':float('nan')},{'orientation_tolerance_rad':4.},
    {'point_padding_m':-1.}])
def test_invalid_tracking_allowance_cannot_produce_padding(changes):
    kw={'position_tolerance_m':.003,'orientation_tolerance_rad':.03,'point_padding_m':.02,**changes}
    with pytest.raises(ValueError):model().gripper_pose_error_padding([0.,0.],1.,**kw)


def tool_with_tracking():
    tool,robot_model=make_tool();queries=[]
    tool.robot.planned_path_tracking_policy=lambda:{'position_tolerance_m':.003,
        'orientation_tolerance_rad':.03,'scope':'rigid_gripper_only'}
    robot_model.gripper_pose_error_padding=lambda *a,**kw:{'arm':0.,'hand':.006}
    def nominal(q,extra_margin=0.,first_hit_only=False):
        queries.append(extra_margin)
        pad=extra_margin.get('hand',0.) if isinstance(extra_margin,dict) else extra_margin
        blocked=q[0]>.1 and pad>=.006
        return {'collision':blocked,'observed_collisions':{'hand':1} if blocked else {},'self_collisions':[]}
    tool._scene_collision_checker=lambda *a,**kw:(nominal,{'margin_m':.01})
    return tool,robot_model,queries


def test_nominally_clear_accurate_endpoint_reports_tracking_clearance_not_ik_error():
    tool,m,queries=tool_with_tracking()
    def solve(*a,valid,**kw):
        q=np.array([.2]);m.last_ik={'joints':q.tolist(),'pose_solution_count':1,
            'position_residual_m':0.,'orientation_residual_rad':0.}
        return q if valid(q) else None
    m.solve_ik=solve
    result=tool.plan_motion({'position_m':[.2,0,0],'quaternion_wxyz':[1,0,0,0]},'end_effector','scene')
    assert not result.success and not result.executed
    assert result.value['reason']=='tracking_clearance_insufficient'
    assert result.value['best_endpoint_collision']['collision']
    assert not result.value['best_endpoint_nominal_collision']['collision']
    assert 'different' in result.value['recovery_guidance']
    assert 'plan_handle' not in result.value
    assert result.value['planning_tracking_reserve']['geometry_padding_m']['hand']==.006


def test_execution_uses_original_collision_margin_not_the_planning_reserve():
    tool,m,queries=tool_with_tracking()
    m.solve_ik=lambda *a,**kw:np.array([.05])
    tool.robot.execute_joint_path=lambda path,check,**kw:CapabilityResult.ok({'control_steps':1,
        'test_original_collision':check(np.array([.2]))})
    planned=tool.plan_motion({'position_m':[.05,0,0],'quaternion_wxyz':[1,0,0,0]},'end_effector','scene')
    assert planned.success,planned.value
    assert any(isinstance(q,dict) and q.get('hand',0)>=.006 for q in queries)
    moved=tool.execute_motion(planned.value['plan_handle'])
    assert moved.success
    assert moved.value['test_original_collision']['collision'] is False
    assert queries[-1]==0.
    assert moved.value['planning_tracking_reserve']==planned.value['planning_tracking_reserve']


def test_tracking_setup_expiry_does_not_start_ik(monkeypatch):
    now=clock_at(monkeypatch);tool,m,queries=tool_with_tracking()
    def slow(*a,**kw):now[0]=2.;return {'arm':0.,'hand':.006}
    m.gripper_pose_error_padding=slow
    m.solve_ik=lambda *a,**kw:pytest.fail('IK after tracking setup deadline')
    with deadline_scope('episode',expires_s=1.):
        result=tool.plan_motion({'position_m':[.2,0,0],'quaternion_wxyz':[1,0,0,0]},'end_effector','scene')
    assert not result.success and result.failure_category is FailureCategory.TIMEOUT
    assert result.diagnostics['deadline']['scope']=='episode'
    assert not queries


def test_nominally_clear_start_is_not_misreported_as_an_observed_collision():
    tool,m,queries=tool_with_tracking()
    def collision(q,extra_margin=0.,first_hit_only=False):
        reserved=isinstance(extra_margin,dict) and extra_margin['hand']>0.
        return {'collision':q[0]<.1 and reserved}
    tool._scene_collision_checker=lambda *a,**kw:(collision,{})
    m.solve_ik=lambda *a,**kw:np.array([.2])
    result=tool.plan_motion({'position_m':[.2,0,0],'quaternion_wxyz':[1,0,0,0]},'end_effector','scene')
    assert not result.success and not result.executed
    assert result.value['reason']=='tracking_start_clearance_insufficient'
    assert result.value['start_nominal_collision']['collision'] is False
    assert result.value['start_collision']['collision'] is True
    assert 'plan_handle' not in result.value


def test_existing_nominal_collision_is_not_relabelled_as_tracking_uncertainty():
    tool,m,queries=tool_with_tracking()
    # Clear start isolates the endpoint classification contract. Invalid starts
    # now reject before IK, tested independently in test_start_collision_feedback.
    tool._scene_collision_checker=lambda *a,**kw:(lambda q,**kw:{'collision':q[0]>.1},{})
    def solve(*a,valid,**kw):
        m.last_ik={'joints':[.2],'pose_solution_count':1,'position_residual_m':0.,'orientation_residual_rad':0.}
        assert valid(np.array([.2])) is False
        return None
    m.solve_ik=solve
    result=tool.plan_motion({'position_m':[.2,0,0],'quaternion_wxyz':[1,0,0,0]},'end_effector','scene')
    assert not result.success
    assert result.value['reason']=='no_collision_free_ik_returned'
    assert result.value['best_endpoint_nominal_collision']['collision'] is True



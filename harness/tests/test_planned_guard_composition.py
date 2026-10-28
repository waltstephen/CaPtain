"""A nested planned executor must retain its enclosing evidence/safety guard."""
from types import SimpleNamespace
import numpy as np
import pytest
from captain_harness.ports.capability import CapabilityResult, FailureCategory
from test_possession_monitor import robot_fixture


def path_fixture(tmp_path,**options):
    robot,env=robot_fixture(tmp_path,**options)
    start=np.asarray(robot.get_robot_state().value.end_effector_position_m)
    def fk(q,aperture):
        pose=np.eye(4);pose[:3,3]=start+q[:3];return pose
    robot.planning_model=lambda:SimpleNamespace(eef_pose=fk)
    path=np.zeros((4,7));path[:,0]=[0,.01,.02,.03]
    return robot,env,path


def run_path(robot,path,collision=None):
    return robot.execute_joint_path(path,collision or (lambda q:{'collision':False}))


@pytest.mark.parametrize('mode',['lost','unavailable','transient'])
@pytest.mark.parametrize('guarded',[False,True])
def test_planned_multistep_execution_stops_at_first_possession_failure(tmp_path,mode,guarded):
    robot,env,path=path_fixture(tmp_path,**({'lost_after':2} if mode=='lost' else
        {'unavailable_after':2} if mode=='unavailable' else {}))
    robot._guarded_cartesian_motion=guarded
    if mode=='transient':
        robot.gripper_state=lambda:CapabilityResult.ok({'object_detected':not 2<=len(env.actions)<3,
            'open_fraction':.001 if 2<=len(env.actions)<3 else .4})
    result=robot.with_possession_guard(lambda:run_path(robot,path))
    assert not result.success and result.executed
    assert len(env.actions)==2
    assert result.value['control_steps']==2
    assert result.value['possession_monitor']['status']==('unavailable' if mode=='unavailable' else 'lost')
    assert result.value['reached'] is False
    assert robot._motion_guard is None and robot._possession_guard is None


def test_initial_outer_guard_rejection_executes_zero_steps_and_restores_it(tmp_path):
    robot,env,path=path_fixture(tmp_path)
    guard=lambda:False;robot._motion_guard=guard
    result=run_path(robot,path)
    assert not result.success and not result.executed and not env.actions
    assert result.value['reason']=='motion_guard_rejected'
    assert robot._motion_guard is guard


def test_outer_guard_exception_is_a_backend_failure_with_partial_receipt(tmp_path):
    robot,env,path=path_fixture(tmp_path)
    def guard():
        if len(env.actions)>=2:raise OSError('guard sensor disconnected')
        return True
    robot._motion_guard=guard
    result=run_path(robot,path)
    assert not result.success and result.executed and len(env.actions)==2
    assert result.failure_category is FailureCategory.BACKEND_ERROR
    assert result.value['stop_evidence']['outer_guard_error']=='OSError: guard sensor disconnected'
    assert result.value['control_steps']==2 and robot._motion_guard is guard


def test_collision_and_outer_rejection_retain_primary_collision(tmp_path):
    robot,env,path=path_fixture(tmp_path)
    robot._motion_guard=lambda:len(env.actions)<2
    result=run_path(robot,path,lambda q:{'collision':len(env.actions)>=2})
    assert not result.success and len(env.actions)==2
    assert result.value['reason']=='observed_collision_during_execution'
    assert result.value['stop_evidence']['collision']['collision'] is True


def test_possession_failure_keeps_original_driver_diagnostics(tmp_path):
    from dataclasses import replace
    robot,env,path=path_fixture(tmp_path,lost_after=2)
    robot._guarded_cartesian_motion=False
    def operation():
        outcome=run_path(robot,path)
        return replace(outcome,diagnostics={'attempt_id':'planned-attempt','driver_receipt':{'control_steps':len(env.actions)}})
    result=robot.with_possession_guard(operation)
    assert not result.success and len(env.actions)==2
    assert result.diagnostics['attempt_id']=='planned-attempt'
    assert result.diagnostics['driver_receipt']['control_steps']==2


def test_maintained_outer_guard_keeps_normal_planned_execution(tmp_path):
    robot,env,path=path_fixture(tmp_path)
    result=robot.with_possession_guard(lambda:run_path(robot,path))
    assert result.success and result.executed
    assert result.value['completed_segments']==3
    assert result.value['possession_monitor']['status']=='maintained'
    assert result.value['possession_monitor']['samples']>2


def test_final_boundary_guard_defeats_completed_path_success(tmp_path):
    robot,env,path=path_fixture(tmp_path)
    robot._guarded_cartesian_motion=False
    guard=lambda:len(env.actions)<3
    robot._motion_guard=guard
    result=run_path(robot,path)
    assert len(env.actions)==3 and result.value['completed_segments']==3
    assert not result.success and result.executed
    assert result.value['converged'] is True and result.value['reached'] is False
    assert result.value['reason']=='motion_guard_rejected'
    assert robot._motion_guard is guard

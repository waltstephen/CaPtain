from dataclasses import replace
import numpy as np
import pytest
from captain_harness.ports.capability import CapabilityResult, FailureCategory
from test_tool_backend import _fixture


@pytest.mark.parametrize('category',[FailureCategory.IK_FAILURE,FailureCategory.TIMEOUT])
def test_second_phase_preexecution_failure_does_not_erase_executed_approach(tmp_path,category):
    f=_fixture(tmp_path,v15_protocol=True);f.robot.controller='OSC_POSE'
    original=f.backend._solve_ik_target;calls=[]
    def solve(*a,**kw):
        calls.append(a)
        if len(calls)==1:return original(*a,**kw)
        return replace(CapabilityResult.failed(category,'target_phase_rejected'),
            diagnostics={'deadline':{'scope':'policy','expired':True}} if category is FailureCategory.TIMEOUT else {'ik':'rejected'},
            attempt_identity={'request':'second-phase'})
    f.backend._solve_ik_target=solve
    result=f.backend.goto_pose([.4,.2,.3],[1,0,0,0],z_approach=.1,approach_frame='world',position_reference='tool')
    assert not result.success and result.executed
    assert result.failure_category is category and result.message=='target_phase_rejected'
    assert result.attempt_identity=={'request':'second-phase'}
    assert len([c for c in f.robot.calls if c[0]=='move_to_pose'])==1
    assert result.value['failed_phase']=='target'
    assert result.value['phase_executions']['approach']['executed'] is True
    assert result.value['phase_executions']['target']['executed'] is False
    assert result.value['completed_phases']==['approach']
    if category is FailureCategory.TIMEOUT:assert result.diagnostics['deadline']['scope']=='policy'


def test_missing_postmotion_pose_never_certifies_invented_origin(tmp_path):
    f=_fixture(tmp_path,v15_protocol=True);f.robot.controller='OSC_POSE'
    moved=[];original=f.robot.move_to_pose;state=f.robot.get_robot_state
    def move(*a,**kw):
        result=original(*a,**kw);moved.append(True);return result
    f.robot.move_to_pose=move
    f.robot.get_robot_state=lambda:CapabilityResult.failed(FailureCategory.BACKEND_ERROR,'encoder_offline') if moved else state()
    # The old fallback invents EEF zero: choose its corresponding public tool
    # target so that unobserved motion is incorrectly reported converged.
    pos=-np.asarray(f.backend._tcp_offset_local_m)
    result=f.backend.goto_pose(pos,[1,0,0,0],position_reference='tool')
    assert moved and result.executed and not result.success
    assert result.failure_category is FailureCategory.BACKEND_ERROR
    assert result.value['pose_measurement']['status']=='unavailable'
    assert result.value['final_eef_position_m'] is None
    assert result.value['position_residual_m'] is None
    assert result.value['converged'] is False
    assert 'encoder_offline' in result.value['pose_measurement']['message']


def test_public_invocation_retains_approach_when_target_service_raises(tmp_path):
    from captain_harness.adapters.capx.backend import ToolOperationError
    f=_fixture(tmp_path,v15_protocol=True);f.robot.controller='OSC_POSE';f.robot.simulation_steps=0
    move=f.robot.move_to_pose;solve=f.backend._solve_ik_target;calls=[]
    def counted_move(*a,**kw):
        result=move(*a,**kw);f.robot.simulation_steps+=3;return result
    def rejecting_solve(*a,**kw):
        calls.append(a)
        if len(calls)==1:return solve(*a,**kw)
        raise ToolOperationError(FailureCategory.IK_FAILURE,'no second endpoint')
    f.robot.move_to_pose=counted_move;f.backend._solve_ik_target=rejecting_solve
    result=f.backend.invoke('goto_pose',dict(position=[.4,.2,.3],quaternion=[1,0,0,0],z_approach=.1,approach_frame='world',position_reference='tool'))
    assert not result.success and result.executed
    assert result.failure_category is FailureCategory.IK_FAILURE
    assert result.diagnostics['physical_execution']['control_steps']==3
    assert result.value['completed_phases']==['approach']
    assert result.value['phase_executions']['target']['message']=='no second endpoint'


@pytest.mark.parametrize('scene',[None,'unused-scene'])
def test_planning_does_not_start_without_current_pose_measurement(tmp_path,scene):
    f=_fixture(tmp_path,v15_protocol=True)
    f.robot.get_robot_state=lambda:CapabilityResult.failed(FailureCategory.BACKEND_ERROR,'encoder_offline')
    calls=[];f.backend._plan_motion=lambda *a,**kw:calls.append(a)
    result=f.backend.invoke('goto_pose',dict(position=[.4,.2,.3],quaternion=[1,0,0,0],scene_handle=scene,position_reference='tool'))
    assert not result.executed and not result.success and not calls
    assert not [c for c in f.robot.calls if c[0] in {'move_to_pose','move_to_joints'}]
    assert result.failure_category is FailureCategory.BACKEND_ERROR
    assert result.value['pose_measurement']['status']=='unavailable'


def test_two_phase_success_survives_public_schema_and_preserves_both_phases(tmp_path):
    from test_tool_backend import _invoke
    from captain_harness.adapters.capx.manifest import ToolApiLevel
    f=_fixture(tmp_path,v15_protocol=True);f.robot.controller='OSC_POSE'
    result=_invoke(f,ToolApiLevel.S3_LOW_LEVEL,'goto_pose',dict(position=[.4,.2,.3],quaternion=[1,0,0,0],z_approach=.1,approach_frame='world',position_reference='tool'),service_aligned=True,dehardcoded=True,v15_protocol=True,motion_tools=True)
    assert result.success,result.message
    assert result.value['completed_phases']==['approach','target']
    assert result.value['failed_phase'] is None
    assert len([c for c in f.robot.calls if c[0]=='move_to_pose'])==2


def test_expired_approach_preserves_shared_deadline_and_never_starts_target(tmp_path,monkeypatch):
    import captain_harness.ports.deadline as timing
    f=_fixture(tmp_path,v15_protocol=True);f.robot.controller='OSC_POSE'
    now=[10.];monkeypatch.setattr(timing.time,'monotonic',lambda:now[0])
    move=f.robot.move_to_pose
    def late_move(*a,**kw):
        result=move(*a,**kw);now[0]+=.3;return result
    f.robot.move_to_pose=late_move
    with timing.deadline_scope('policy',timeout_s=.2):
        result=f.backend.invoke('goto_pose',dict(position=[.4,.2,.3],quaternion=[1,0,0,0],z_approach=.1,approach_frame='world',position_reference='tool'))
    assert not result.success and result.executed and result.failure_category is FailureCategory.TIMEOUT
    assert len([c for c in f.robot.calls if c[0]=='move_to_pose'])==1
    assert result.value['failed_phase']=='approach'
    assert 'target' not in result.value['phase_executions']
    assert result.diagnostics['deadline']['scope']=='policy'


@pytest.mark.parametrize('bad',[[float('nan'),0,0],[0,0]])
def test_malformed_postmotion_pose_is_measurement_failure_not_argument_error(tmp_path,bad):
    f=_fixture(tmp_path,v15_protocol=True);f.robot.controller='OSC_POSE'
    moved=[];move=f.robot.move_to_pose;state=f.robot.get_robot_state
    def run(*a,**kw):
        result=move(*a,**kw);moved.append(True);return result
    f.robot.move_to_pose=run
    f.robot.get_robot_state=lambda:CapabilityResult.ok(replace(state().value,end_effector_position_m=bad)) if moved else state()
    result=f.backend.invoke('goto_pose',dict(position=[.4,.2,.3],quaternion=[1,0,0,0],position_reference='tool'))
    assert result.executed and not result.success
    assert result.failure_category is FailureCategory.BACKEND_ERROR
    assert result.value['final_eef_position_m'] is None
    assert result.value['pose_measurement']['status']=='unavailable'

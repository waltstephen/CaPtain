"""Carried planning consumes fresh accepted public evidence, never a naked handle."""
from types import SimpleNamespace
import numpy as np
import pytest
from captain_harness.adapters.capx.artifacts import EpisodeArtifactRegistry
from captain_harness.adapters.capx.motion_tools import MotionToolMixin
from captain_harness.adapters.robotics.planning import SensorScene
from captain_harness.ports.capability import CapabilityResult
from test_carried_robot_model import robot_model,box


def fixture():
    tool=MotionToolMixin();tool.registry=EpisodeArtifactRegistry('carried',ttl_s=None)
    model=robot_model();points=box([.2,0,0]);obstacles=np.array([[-.5,0,0]])
    state=SimpleNamespace(joint_positions_rad=(0.,0.),gripper_open_fraction=.4,
        end_effector_position_m=(0.,0.,0.),end_effector_quaternion_wxyz=(1.,0.,0.,0.))
    obs=SimpleNamespace(robot_state=state,time_s=0.)
    oh=tool.registry.put('observation',obs)
    mark=SimpleNamespace(observation_handle=oh,views_used=2,cross_view_consistent=True)
    ph=tool.registry.put('marked_point',mark)
    tracked=SimpleNamespace(attachment_verified=True,eef_to_object_transform=np.eye(4),
        carried_geometry_epoch=1,attachment_current_point_handle=ph)
    th=tool.registry.put('entity_track',tracked);tool._active_attachment_handle=th
    calls=[]
    def execute(path,check,**kw):
        calls.append(path.copy())
        assert all(not check(q)['collision'] for q in path)
        return CapabilityResult.ok({'reached':True,'control_steps':len(path)-1,'trajectory':[]})
    tool.robot=SimpleNamespace(time_s=0.,controller='OSC_POSE',get_robot_state=lambda:CapabilityResult.ok(state),
        planning_model=lambda:model,execute_joint_path=execute)
    tool._points_from_mark=lambda handle,**kw:(mark,np.vstack([points,obstacles]),points,.9)
    tool._pose_mapping=lambda pose:(pose['position_m'],pose['quaternion_wxyz'])
    sh=tool.registry.put('sensor_scene',SensorScene(np.vstack([points,obstacles]),oh,(0.,0.),.4,0.,.003,0))
    return SimpleNamespace(tool=tool,model=model,state=state,obs=obs,obs_handle=oh,point=ph,track=th,
        tracked=tracked,scene=sh,points=points,obstacles=obstacles,executions=calls)


def capture(f):
    return f.tool.capture_carried_geometry(f.track,f.point)


def plan(f,handle):
    return f.tool.plan_motion({'position_m':[.03,0,0],'quaternion_wxyz':[1,0,0,0]},
        'end_effector',f.scene,carried_geometry_handle=handle)


def test_capture_plan_execute_retains_target_binding_and_exact_scene_partition():
    f=fixture();captured=capture(f)
    assert captured.success and not f.executions
    handle=captured.value['carried_geometry_handle']
    assert captured.value['source_observation_handle']==f.obs_handle
    assert captured.value['motion_authorized'] is False
    planned=plan(f,handle);assert planned.success,planned.message
    policy=planned.value['collision_policy']
    assert policy['kind']=='robot_and_observed_carried_geometry'
    assert policy['selected_target_sample_count']==len(f.points)
    assert policy['other_scene_point_count']==len(f.obstacles)
    executed=f.tool.execute_motion(planned.value['plan_handle'])
    assert executed.success and len(f.executions)==1
    assert executed.value['carried_geometry_handle']==handle


@pytest.mark.parametrize('change',['released','lost','new_assertion','wrong_mark','clock','joints','aperture'])
def test_capture_rejects_stale_or_unaccepted_attachment(change):
    f=fixture()
    if change=='released':f.tool._active_attachment_handle=None
    if change=='lost':f.tracked.attachment_verified=False
    if change=='new_assertion':f.tracked.attachment_current_point_handle=None
    if change=='wrong_mark':f.point=f.tool.registry.put('marked_point',SimpleNamespace(observation_handle='wrong'))
    if change=='clock':f.tool.robot.time_s=.05
    if change=='joints':f.obs.robot_state=SimpleNamespace(**{**vars(f.state),'joint_positions_rad':(.01,0.)})
    if change=='aperture':f.obs.robot_state=SimpleNamespace(**{**vars(f.state),'gripper_open_fraction':.1})
    result=capture(f)
    assert not result.success and not result.executed and not f.executions


@pytest.mark.parametrize('change',['released','lost','new_assertion','clock','joints','aperture'])
def test_plan_and_execution_reject_invalidated_carried_geometry(change):
    for stage in ['plan','execute']:
        f=fixture();captured=capture(f);assert captured.success
        handle=captured.value['carried_geometry_handle']
        planned=plan(f,handle) if stage=='execute' else None
        if planned is not None:assert planned.success,planned.message
        if change=='released':f.tool._active_attachment_handle=None
        if change=='lost':f.tracked.attachment_verified=False
        if change=='new_assertion':f.tracked.carried_geometry_epoch+=1
        if change=='clock':f.tool.robot.time_s=.05
        if change=='joints':f.state.joint_positions_rad=(.01,0.)
        if change=='aperture':f.state.gripper_open_fraction=.1
        try:
            result=plan(f,handle) if stage=='plan' else f.tool.execute_motion(planned.value['plan_handle'])
        except ValueError as exc:
            assert 'stale' in str(exc)
        else:assert not result.success and not result.executed
        assert not f.executions


def test_active_attachment_without_geometry_still_cannot_plan():
    f=fixture()
    result=f.tool.plan_motion({'position_m':[.03,0,0],'quaternion_wxyz':[1,0,0,0]},'end_effector',f.scene)
    assert not result.success and not f.executions
    assert 'held_object_planning_unavailable' in result.message


def test_an_extra_nonselected_point_inside_carried_hull_is_never_removed():
    f=fixture();captured=capture(f)
    scene=f.tool.registry.get(f.scene,expected_kind='sensor_scene')
    from dataclasses import replace
    f.scene=f.tool.registry.put('sensor_scene',replace(scene,points=np.vstack([scene.points,[.2,0,0]])))
    planned=plan(f,captured.value['carried_geometry_handle'])
    assert not planned.success and not f.executions


def test_capture_and_plan_contracts_are_declared_without_automatic_motion():
    from captain_harness.adapters.capx.manifest import build_capx_tool_manifest
    from captain_harness.adapters.capx.runtime_manifest import runtime_manifest
    from captain_harness.ports.capability import SideEffect
    manifest=runtime_manifest(build_capx_tool_manifest('S3',joint_count=7,service_aligned=True,
        dehardcoded=True,v15_protocol=True),controller='OSC_POSE',motion_tools=True)
    spec=manifest.require('capture_carried_geometry')
    assert spec.side_effect is SideEffect.COMPUTE
    assert set(spec.arguments_schema['required'])=={'tracking_handle','current_point_handle'}
    assert 'carried_geometry_handle' in manifest.require('plan_motion').arguments_schema['properties']


def test_capture_requires_the_accepted_primary_cloud_during_reconstruction():
    f=fixture()
    def changed_cloud(handle,**kwargs):
        assert kwargs.get('require_primary') is True
        raise ValueError('carried_geometry_primary_view_not_retained')
    f.tool._points_from_mark=changed_cloud
    result=capture(f)
    assert not result.success and not result.executed and not f.executions
    assert 'primary_view_not_retained' in result.message


def test_real_attachment_assertion_generation_invalidates_on_failed_reassertion(tmp_path):
    from test_lift_possession_contract import case,assert_held
    f,track,handle,mark,point,_=case(tmp_path)
    assert not hasattr(track,'carried_geometry_epoch')
    first=assert_held(f,handle,point)
    assert first.success and first.value['pose_observation_accepted']
    assert track.attachment_current_point_handle==point and track.carried_geometry_epoch==1
    f.robot.time_s+=.05
    second=assert_held(f,handle,point)
    assert not second.success and not second.executed
    assert track.attachment_current_point_handle is None and track.carried_geometry_epoch==2


def test_reassertion_during_geometry_build_cannot_issue_a_stale_bundle():
    f=fixture();original=f.model.with_observed_carried_geometry
    def reassert(points):
        model=original(points);f.tracked.carried_geometry_epoch+=1;return model
    f.model.with_observed_carried_geometry=reassert
    result=capture(f)
    assert not result.success and not result.executed
    assert 'evidence_changed' in result.message
    assert 'carried_geometry_handle' not in result.value and not f.executions


def test_late_geometry_compute_keeps_parent_deadline_and_issues_no_motion(monkeypatch):
    from captain_harness.ports import deadline as deadlines
    now=[100.];monkeypatch.setattr(deadlines.time,'monotonic',lambda:now[0])
    f=fixture();original=f.model.with_observed_carried_geometry
    def late(points):
        model=original(points);now[0]+=2.;return model
    f.model.with_observed_carried_geometry=late
    with deadlines.deadline_scope('episode',timeout_s=1.):
        with pytest.raises(deadlines.DeadlineExceeded) as failure:capture(f)
    assert failure.value.scope=='episode' and not f.executions


def test_carried_capture_filters_projected_robot_points_and_publishes_view_receipt():
    f=fixture();old=f.tool._points_from_mark
    def reconstruct(handle,**kwargs):
        # The visible seed filter uses calibrated robot geometry only.
        sample=np.vstack([f.points,[0,0,0]])
        filtered=kwargs['projection_filter'](sample)
        np.testing.assert_array_equal(filtered,f.points)
        mark,full,points,score=old(handle,**kwargs)
        mark.carried_view_reconstruction=[{'camera_name':'wrist','status':'supported','seed_pixel_xy':[12,12]}]
        return mark,full,points,score
    f.tool._points_from_mark=reconstruct
    result=capture(f)
    assert result.success and not f.executions
    assert result.value['view_reconstruction'][0]['camera_name']=='wrist'

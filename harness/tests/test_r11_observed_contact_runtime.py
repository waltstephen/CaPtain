from dataclasses import replace
from types import SimpleNamespace
from copy import deepcopy

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from test_tool_backend import _fixture
from test_selection_evidence import selection_fixture
from captain_harness.adapters.capx.contact_tools import ContactToolMixin, contact_motion_capabilities
from captain_harness.adapters.capx.artifacts import EpisodeArtifactRegistry
from captain_harness.ports.capability import CapabilityResult


def test_observed_contact_uses_declared_target_and_reports_partial_execution(monkeypatch):
    from captain_harness.adapters.capx import visual_tracking
    from test_sensor_motion_tools import camera
    camera0=camera(np.zeros((128,128,3),np.uint8)); camera0.camera_name='cam'
    capture=SimpleNamespace(cameras=(camera0,),time_s=0.,robot_state=SimpleNamespace(
        joint_positions_rad=[0]*7,gripper_open_fraction=.5,end_effector_position_m=[0,0,0],
        end_effector_quaternion_wxyz=[1,0,0,0]))
    backend=ContactToolMixin();backend.registry=EpisodeArtifactRegistry('contact-runtime',ttl_s=60.)
    handle=backend.registry.put('observation',capture)
    detection=SimpleNamespace(observation_handle=handle,camera_name='cam',mask=np.ones((128,128),bool))
    backend._require_selected_detection=lambda target:detection
    backend._camera_by_name=lambda obs,name:obs.cameras[0]
    backend._capture_observation=lambda:deepcopy(capture)
    backend._aligned_pregrasp=object()
    calls=[]
    def move(direction,distance,**kwargs):
        calls.append(distance)
        return CapabilityResult.ok({'reached':True,'stopped_reason':'target_reached','control_steps':10,
            'actual_displacement_m':[distance,0,0],'contact_identity':'unknown',
            'observed_force_n':5.2,'trajectory':[{'step':len(calls)}]})
    backend.robot=SimpleNamespace(move_linear=move,time_s=0.,_eef_force_norm=lambda:5.,
        get_robot_state=lambda:CapabilityResult.ok(capture.robot_state))
    monkeypatch.setattr(visual_tracking,'track_rgbd',lambda *a,**kw:{'status':'supported',
        'translation_m':[.006,0,0],'motion_spread_m':.0001})
    result=backend.move_linear([1,0,0],.03,target_detection_id='selected')
    assert result.executed and not result.success
    assert len(calls)==2
    assert backend._aligned_pregrasp is None
    assert result.value['stopped_reason']=='no_observed_target_progress'
    assert len(backend.registry.get(result.value['trajectory_handle'],expected_kind='motion_trajectory'))==2
    spec=next(s for s in contact_motion_capabilities() if s.name=='move_linear')
    Draft202012Validator(spec.result_schema).validate(result.value)


@pytest.mark.parametrize('visual_support',[True,False])
def test_linear_segments_keep_original_goal_despite_each_segment_tolerance(monkeypatch,visual_support):
    from captain_harness.adapters.capx import visual_tracking
    from test_sensor_motion_tools import camera
    state=SimpleNamespace(joint_positions_rad=[0]*7,gripper_open_fraction=.5,
        end_effector_position_m=np.zeros(3),end_effector_quaternion_wxyz=[1,0,0,0])
    cam=camera(np.zeros((128,128,3),np.uint8));cam.camera_name='cam'
    source=SimpleNamespace(cameras=(cam,),time_s=0.,robot_state=deepcopy(state))
    backend=ContactToolMixin();backend.registry=EpisodeArtifactRegistry('absolute-goal',ttl_s=60.)
    handle=backend.registry.put('observation',source)
    detection=SimpleNamespace(observation_handle=handle,camera_name='cam',mask=np.ones((128,128),bool))
    backend._require_selected_detection=lambda _:detection
    backend._camera_by_name=lambda obs,name:obs.cameras[0]
    backend._capture_observation=lambda:SimpleNamespace(cameras=(cam,),time_s=0.,robot_state=deepcopy(state))
    backend._aligned_pregrasp=None
    commanded=[]
    def move(direction,distance,**kwargs):
        direction=np.asarray(direction,float);direction/=np.linalg.norm(direction)
        commanded.append(state.end_effector_position_m+direction*distance)
        actual=direction*(distance-.0015)
        state.end_effector_position_m+=actual
        return CapabilityResult.ok({'reached':True,'stopped_reason':'target_reached','control_steps':10,
            'position_residual_m':.0015,'orientation_residual_rad':0.,
            'actual_displacement_m':actual.tolist(),'contact_identity':'unknown','observed_force_n':5.})
    backend.robot=SimpleNamespace(move_linear=move,time_s=0.,_eef_force_norm=lambda:5.,
        get_robot_state=lambda:CapabilityResult.ok(state))
    monkeypatch.setattr(visual_tracking,'track_rgbd',lambda *a,**kw:{'status':'supported',
        'translation_m':state.end_effector_position_m.tolist(),'motion_spread_m':.0001}
        if visual_support else {'status':'image_only','translation_m':None})
    distance=.02 if visual_support else .01
    result=backend.move_linear([1,0,0],distance,target_detection_id='selected')
    np.testing.assert_allclose(commanded[-1],[distance,0,0],atol=1e-12)
    np.testing.assert_allclose(result.value['actual_displacement_m'],[distance-.0015,0,0],atol=1e-12)
    assert result.success is visual_support
    # The original .01 m request has a .001 m endpoint tolerance.
    # A .0015 m residual is no longer a reached robot goal for that case.
    assert result.value['reached'] is (distance >= .015)
    assert result.value['robot_goal_reached'] is (distance >= .015)
    if not visual_support:
        assert result.value['target_progress']['status']=='target_motion_unresolved'


def test_target_selection_extension_passes_actual_strict_manifest(tmp_path):
    from captain_harness.adapters.capx.runtime_manifest import runtime_manifest
    from captain_harness.adapters.capx.manifest import build_capx_tool_manifest
    backend,target,_,_=selection_fixture(tmp_path,multiple=False)
    backend._sensor_motion=True
    manifest=runtime_manifest(build_capx_tool_manifest('S3',joint_count=7,service_aligned=True,
        dehardcoded=True,v15_protocol=True),controller='OSC_POSE',motion_tools=True)
    spec=manifest.require('select_detection')
    Draft202012Validator(spec.arguments_schema).validate({'detection_id':target,'previous_detection_id':None})
    result=backend.select_detection(target)
    Draft202012Validator(spec.result_schema).validate(result.value)


def test_segmented_rotation_receipt_describes_the_whole_requested_motion(monkeypatch):
    from scipy.spatial.transform import Rotation
    from captain_harness.adapters.capx import visual_tracking
    from test_sensor_motion_tools import camera
    state=SimpleNamespace(joint_positions_rad=[0]*7,gripper_open_fraction=.5,
        end_effector_position_m=np.zeros(3),end_effector_quaternion_wxyz=[1,0,0,0])
    cam=camera(np.zeros((128,128,3),np.uint8));cam.camera_name='cam'
    capture=lambda:SimpleNamespace(cameras=(cam,),time_s=0.,robot_state=deepcopy(state))
    backend=ContactToolMixin();backend.registry=EpisodeArtifactRegistry('rotation-total',ttl_s=60.)
    handle=backend.registry.put('observation',capture())
    detection=SimpleNamespace(observation_handle=handle,camera_name='cam',mask=np.ones((128,128),bool))
    backend._require_selected_detection=lambda _:detection
    backend._camera_by_name=lambda obs,name:obs.cameras[0]
    backend._capture_observation=capture
    total=[0.]
    def rotate(axis,pivot,angle,**kwargs):
        total[0]+=angle
        state.end_effector_quaternion_wxyz=np.roll(Rotation.from_rotvec([0,0,total[0]]).as_quat(),1)
        return CapabilityResult.ok({'reached':True,'stopped_reason':'target_reached','control_steps':10,
            'actual_displacement_m':[0,0,0],'actual_rotation_vector_rad':[0,0,angle],
            'contact_identity':'unknown','observed_force_n':5.})
    backend.robot=SimpleNamespace(rotate_about_axis=rotate,time_s=0.,_eef_force_norm=lambda:5.,
        get_robot_state=lambda:CapabilityResult.ok(state))
    monkeypatch.setattr(visual_tracking,'track_rgbd',lambda *a,**kw:{'status':'supported',
        'rotation_vector_rad':[0,0,total[0]],'rotation_resolution_rad':.001})
    result=backend.rotate_about_axis([0,0,1],[0,0,0],.2,target_detection_id='selected')
    assert result.success and result.value['executed_segments']==2
    np.testing.assert_allclose(result.value['requested_rotation_vector_rad'],[0,0,.2],atol=1e-12)
    np.testing.assert_allclose(result.value['actual_rotation_vector_rad'],[0,0,.2],atol=1e-12)

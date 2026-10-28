import numpy as np
import pytest
from test_grasp_final_measurement import robot_fixture


def test_explicit_precision_controls_preclosure_endpoint(tmp_path):
    env,robot=robot_fixture(tmp_path)
    state=robot.get_robot_state().value
    target=np.asarray(state.end_effector_position_m)+[.01,0,0]
    result=robot.descend_and_close(target,state.end_effector_quaternion_wxyz,position_tolerance_m=.001)
    assert result.success, result.message
    assert result.value['descent_execution']['position_residual_m'] <= .001
    precision=result.value['contact_precision']
    assert precision['requested_position_tolerance_m']==.001
    assert precision['preclosure_endpoint_within_tolerance'] is True
    assert precision['postclosure_endpoint_within_tolerance'] is False
    assert result.value['closure_executed'] is True


@pytest.mark.parametrize('tolerance',[0.,-.001,.004,float('nan'),True])
def test_invalid_precision_does_not_move_or_close(tmp_path,tolerance):
    env,robot=robot_fixture(tmp_path);state=robot.get_robot_state().value
    result=robot.descend_and_close(state.end_effector_position_m,state.end_effector_quaternion_wxyz,
        position_tolerance_m=tolerance)
    assert not result.success and not result.executed
    assert not env.actions


def test_requested_precision_never_waives_force_stop(tmp_path):
    env,robot=robot_fixture(tmp_path);state=robot.get_robot_state().value
    robot._eef_force_norm=lambda:5. if robot.simulation_steps==0 else 19.
    result=robot.descend_and_close(np.asarray(state.end_effector_position_m)+[.04,0,0],
        state.end_effector_quaternion_wxyz,position_tolerance_m=.001)
    assert not result.success and result.value['closure_executed'] is False
    assert all(a[-1] <= 0 for a in env.actions)


def test_precision_request_is_forwarded_through_validated_public_broker(tmp_path):
    from test_v18_grasp_references import issued
    f,broker,candidate,obs=issued(tmp_path)
    assert broker.invoke('align_pregrasp',{'candidate':candidate['grasp_handle']}).success
    old=f.robot.descend_and_close; captured=[]
    def measured(position,quaternion,**kwargs):
        captured.append(kwargs.pop('position_tolerance_m'))
        return old(position,quaternion,**kwargs)
    f.robot.descend_and_close=measured
    result=broker.invoke('descend_and_close',{'position_tolerance_m':.001})
    assert result.success, result.message
    assert captured==[.001]


def test_contact_precision_receipt_has_declared_result_schema():
    from captain_harness.adapters.capx.closure_force_feedback import with_closure_force_contract
    from captain_harness.ports.capability import CapabilitySpec,SideEffect
    from jsonschema import validate
    spec=CapabilitySpec(name='descend_and_close',description='test',arguments_schema={'type':'object','properties':{}},
        result_schema={'type':'object','properties':{'closure_executed':{'type':'boolean'},
        'gripper_open_fraction':{'type':'number'}},'additionalProperties':False},
        units={},coordinate_frames={},timeout_s=1,side_effect=SideEffect.ROBOT_MOTION)
    contract=with_closure_force_contract(spec)
    assert 'contact_precision' in contract.result_schema['properties']
    validate({'requested_position_tolerance_m':.001,'contact_precision':{
        'requested_position_tolerance_m':.001,'preclosure_endpoint_within_tolerance':True,
        'postclosure_endpoint_within_tolerance':False,'scope':'measured endpoint'}},contract.result_schema)

from captain_harness.adapters.capx.closure_force_feedback import with_closure_force_contract
from captain_harness.ports.capability import CapabilitySpec, SideEffect


def test_only_closure_payloads_gain_optional_typed_force_feedback():
    schema = {'type': 'object', 'properties': {
        'closure_executed': {'type': 'boolean'}, 'contacted': {'type': 'boolean'},
        'gripper_open_fraction': {'type': 'number'}, 'stopped_reason': {'type': 'string'}},
        'additionalProperties': False}
    spec = CapabilitySpec(name='descend_and_close', description='Close at a measured target.',
        arguments_schema={'type': 'object'}, result_schema=schema, return_type='Closure',
        units={}, coordinate_frames={}, timeout_s=10., side_effect=SideEffect.ROBOT_MOTION)
    updated = with_closure_force_contract(spec)
    assert 'force_measurement_phase' not in schema['properties']
    assert updated.result_schema['additionalProperties'] is False
    assert updated.result_schema['properties']['closure_force_monitor']['properties']['samples']['x-decision-feedback']=='on_demand'
    assert 'after' in updated.description


def test_measured_closure_pose_and_summary_cross_strict_result_schema():
    from test_closure_force_monitor import Robot
    from jsonschema import validate
    robot=Robot()
    result=robot.descend_and_close([0,0,0],[1,0,0,0])
    spec=CapabilitySpec(name='closure_test',description='test',arguments_schema={'type':'object'},
        result_schema={'type':'object','properties':{'closure_executed':{'type':'boolean'},
        'gripper_open_fraction':{'type':'number'}}},return_type='Closure',units={},coordinate_frames={},
        timeout_s=10.,side_effect=SideEffect.ROBOT_MOTION)
    contract=with_closure_force_contract(spec).result_schema
    validate({'closure_force_monitor':result.value['closure_force_monitor']},contract)

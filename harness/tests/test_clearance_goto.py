from dataclasses import replace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from captain_harness.ports.capability import CapabilityResult, FailureCategory
from test_compound_orientation_reference import fixture


def request(robot, tool, delta=(.03, .01, .02), **kwargs):
    state = tool._public_robot_state(robot.get_robot_state().value)
    return dict(position=(np.array(state['tool_position_m']) + delta).tolist(),
        quaternion=state['tool_quaternion_wxyz'], position_reference='tool',
        approach_frame='world', clearance_first=True, **kwargs)


def test_public_clearance_preserves_reference_and_skips_duplicate_endpoint(tmp_path):
    robot, tool = fixture(tmp_path, .003)
    args = request(robot, tool)
    result = tool.invoke('goto_pose', args)
    assert result.success, result.message
    assert result.value['completed_phases'] == ['clearance', 'target']
    assert result.value['failed_phase'] is None
    phases = result.value['phase_executions']
    first = phases['clearance']['value']['requested_cartesian_position_m']
    assert first[:2] == pytest.approx(np.array(args['position'])[:2] - [.03,.01])
    assert first[2] == pytest.approx(args['position'][2])
    for phase in phases.values():
        execution = phase['value']['controller_execution']
        assert execution['orientation_tolerance_rad'] == pytest.approx(.05)
        assert execution['orientation_reference'] == 'compound_request'
    budget = result.value['control_budget']
    assert budget['used_steps'] == robot.simulation_steps <= budget['limit_steps'] == 80


def test_clearance_transit_then_descends_in_explicit_eef_reference(tmp_path):
    robot, tool = fixture(tmp_path, 0.)
    state = robot.get_robot_state().value
    pos = np.array(state.end_effector_position_m) + [.02, 0, -.01]
    result = tool.invoke('goto_pose', dict(position=pos.tolist(), quaternion=list(state.end_effector_quaternion_wxyz),
        position_reference='end_effector', approach_frame='world', clearance_first=True, z_approach=.03))
    assert result.success, result.message
    assert result.value['completed_phases'] == ['clearance', 'approach', 'target']
    assert result.value['final_eef_position_m'] == pytest.approx(pos, abs=.003)
    assert result.value['input_pose_reference'] == 'end_effector'


@pytest.mark.parametrize('angle', [-.004,.004])
def test_clearance_cannot_hide_explicit_fine_rotation_failure(tmp_path, angle):
    robot, tool = fixture(tmp_path, 0.)
    args = request(robot, tool)
    q = Rotation.from_rotvec([0,0,angle]) * Rotation.from_quat(np.roll(args['quaternion'],-1))
    args['quaternion'] = np.roll(q.as_quat(),1).tolist()
    result = tool.invoke('goto_pose', args)
    assert result.executed and not result.success
    actual = result.value['controller_execution']
    assert actual['reached'] is False
    assert actual['orientation_tolerance_rad'] == pytest.approx(.0004)


def test_remaining_one_step_keeps_force_guard_and_bounded_command(tmp_path):
    robot, _ = fixture(tmp_path, 0.)
    state = robot.get_robot_state().value
    start = np.asarray(state.end_effector_position_m)
    result = robot.move_to_pose_bounded(start + [.2,0,0], state.end_effector_quaternion_wxyz,
        max_steps=1, orientation_reference_quaternion_wxyz=state.end_effector_quaternion_wxyz)
    assert not result.success and result.executed
    assert result.value['control_steps'] == 1
    commanded = result.value['trajectory'][0]['commanded_eef_position_m']
    assert np.linalg.norm(np.asarray(commanded)-start) <= .1/robot._control_frequency + 1e-10
    assert result.value['force_guard']['absolute_limit_n'] == 18.
    robot._eef_force_norm = lambda:19.
    before = robot.simulation_steps
    stopped = robot.move_to_pose_bounded(start + [.2,0,0], state.end_effector_quaternion_wxyz,
        max_steps=1, orientation_reference_quaternion_wxyz=state.end_effector_quaternion_wxyz)
    assert not stopped.executed and not stopped.success
    assert robot.simulation_steps == before


@pytest.mark.parametrize('limit',[0, -1, True, 1.2, 81])
def test_invalid_budget_never_moves(tmp_path, limit):
    robot, _ = fixture(tmp_path, 0.)
    state = robot.get_robot_state().value
    result = robot.move_to_pose_bounded([.3,0,.3], state.end_effector_quaternion_wxyz,
        max_steps=limit, orientation_reference_quaternion_wxyz=state.end_effector_quaternion_wxyz)
    assert not result.success and not result.executed
    assert robot.simulation_steps == 0


def bounded_fixture(tmp_path, limit=10):
    from test_tool_backend import _fixture
    f = _fixture(tmp_path, v15_protocol=True)
    f.robot.controller = 'OSC_POSE'
    f.robot.cartesian_control_step_limit = limit
    f.robot.simulation_steps = 0
    f.limits = []
    def move(pos, quat, *, max_steps, orientation_reference_quaternion_wxyz):
        f.limits.append(max_steps)
        result = f.robot.move_to_pose(pos, quat)
        f.robot.simulation_steps += 3
        return result
    f.robot.move_to_pose_bounded = move
    return f


def test_budget_exhaustion_blocks_later_ik_and_retains_clearance(tmp_path):
    f = bounded_fixture(tmp_path, 3)
    args = request(f.robot, f.backend, z_approach=.02)
    solved = []; solve = f.backend._solve_ik_target
    def ik(*a):
        solved.append(a)
        return solve(*a)
    f.backend._solve_ik_target = ik
    result = f.backend.invoke('goto_pose', args)
    assert result.executed and not result.success
    assert result.failure_category is FailureCategory.TIMEOUT
    assert f.limits == [3] and len(solved) == 1
    assert result.value['completed_phases'] == ['clearance']
    assert result.value['failed_phase'] == 'approach'
    assert result.value['control_budget']['remaining_steps'] == 0
    assert result.value['phase_executions']['approach']['executed'] is False


def test_partial_ik_failure_preserves_attempt_and_original_steps(tmp_path):
    f = bounded_fixture(tmp_path)
    solve = f.backend._solve_ik_target; calls=[]
    def ik(*a):
        calls.append(a)
        if len(calls)==1: return solve(*a)
        return replace(CapabilityResult.failed(FailureCategory.IK_FAILURE,'second endpoint rejected'),
            attempt_identity={'attempt':'second'}, diagnostics={'service_attempts':[{'status':'failed'}]})
    f.backend._solve_ik_target = ik
    result = f.backend.invoke('goto_pose', request(f.robot,f.backend))
    assert result.executed and not result.success
    assert result.value['completed_phases'] == ['clearance']
    assert result.value['failed_phase'] == 'target'
    assert result.value['control_budget']['used_steps'] == 3
    assert result.attempt_identity == {'attempt':'second'}
    assert result.diagnostics['service_attempts'] == [{'status':'failed'}]
    assert f.limits == [10]


def test_clearance_expiry_never_starts_next_phase(tmp_path,monkeypatch):
    import captain_harness.ports.deadline as timing
    f = bounded_fixture(tmp_path)
    args = request(f.robot,f.backend)
    now = [10.]; monkeypatch.setattr(timing.time,'monotonic',lambda:now[0])
    move = f.robot.move_to_pose_bounded
    def late(*a,**kw):
        result=move(*a,**kw);now[0]+=.3;return result
    f.robot.move_to_pose_bounded = late
    with timing.deadline_scope('policy',timeout_s=.2):
        result = f.backend.invoke('goto_pose',args)
    assert result.executed and not result.success
    assert result.failure_category is FailureCategory.TIMEOUT
    assert f.limits == [10]
    assert result.value['failed_phase'] == 'clearance'
    assert result.diagnostics['deadline']['scope'] == 'policy'


def test_missing_entry_pose_blocks_clearance_and_ik(tmp_path):
    f = bounded_fixture(tmp_path)
    args = request(f.robot,f.backend)
    f.robot.get_robot_state = lambda:CapabilityResult.failed(FailureCategory.BACKEND_ERROR,'encoder_offline')
    calls=[];f.backend._solve_ik_target=lambda *a:calls.append(a)
    result=f.backend.invoke('goto_pose',args)
    assert not result.executed and not result.success and not calls and not f.limits
    assert result.value['pose_measurement']['status']=='unavailable'


def test_missing_post_clearance_sensor_retains_completed_motion(tmp_path):
    f = bounded_fixture(tmp_path)
    args = request(f.robot,f.backend)
    move=f.robot.move_to_pose_bounded
    def lost(*a,**kw):
        result=move(*a,**kw)
        f.robot.get_robot_state=lambda:CapabilityResult.failed(FailureCategory.BACKEND_ERROR,'encoder_offline')
        return result
    f.robot.move_to_pose_bounded=lost
    result=f.backend.invoke('goto_pose',args)
    assert result.executed and not result.success and f.limits==[10]
    assert result.value['failed_phase']=='clearance'
    assert result.value['final_eef_position_m'] is None
    assert result.value['control_budget']['used_steps']==3


@pytest.mark.parametrize('extra',[{'approach_frame':'tool'}, {'scene_handle':'scene'}, {'clearance_first':1}])
def test_invalid_clearance_combinations_do_not_move(tmp_path,extra):
    f=bounded_fixture(tmp_path)
    args=request(f.robot,f.backend);args.update(extra)
    result=f.backend.invoke('goto_pose',args)
    assert not result.success and not result.executed and not f.limits


def test_unsupported_driver_cannot_silently_fall_back(tmp_path):
    from test_tool_backend import _fixture
    f=_fixture(tmp_path,v15_protocol=True)
    result=f.backend.invoke('goto_pose',request(f.robot,f.backend))
    assert not result.success and not result.executed
    assert not [c for c in f.robot.calls if c[0] in {'move_to_pose','move_to_joints'}]


def test_clearance_schema_and_all_phases_survive_public_broker(tmp_path):
    from test_tool_backend import _invoke
    from captain_harness.adapters.capx.manifest import ToolApiLevel
    f=bounded_fixture(tmp_path)
    result=_invoke(f,ToolApiLevel.S3_LOW_LEVEL,'goto_pose',request(f.robot,f.backend,z_approach=.02),
        service_aligned=True,dehardcoded=True,v15_protocol=True,motion_tools=True)
    assert result.success,result.message
    assert result.value['completed_phases']==['clearance','approach','target']
    assert f.limits==[10,7,4]
    assert result.value['control_budget']['used_steps']==9


def test_already_requested_pose_is_a_measured_noop(tmp_path):
    f=bounded_fixture(tmp_path)
    args=request(f.robot,f.backend,delta=(0,0,0))
    result=f.backend.invoke('goto_pose',args)
    assert result.success and not f.limits
    assert result.diagnostics['physical_execution']=={'control_steps':0,'status':'not_executed'}
    assert result.value['control_budget']['used_steps']==0
    assert result.value['completed_phases']==[]


def test_start_above_destination_does_not_descend_before_traversal(tmp_path):
    f=bounded_fixture(tmp_path)
    args=request(f.robot,f.backend,delta=(.03,0,-.1))
    current=f.backend._public_robot_state(f.robot.get_robot_state().value)['tool_position_m']
    result=f.backend.invoke('goto_pose',args)
    assert result.success
    assert result.value['completed_phases']==['approach','target']
    phases=result.value['phase_executions']
    assert phases['approach']['value']['requested_cartesian_position_m'][2]==pytest.approx(current[2])
    assert phases['target']['value']['requested_cartesian_position_m']==pytest.approx(args['position'])


def test_new_option_is_not_exposed_for_joint_controller():
    from captain_harness.adapters.capx.manifest import ToolApiLevel,build_capx_tool_manifest
    from captain_harness.adapters.capx.runtime_manifest import runtime_manifest
    manifest=build_capx_tool_manifest(ToolApiLevel.S3_LOW_LEVEL,joint_count=7,service_aligned=True)
    for controller in ['OSC_POSE','JOINT_POSITION']:
        runtime=runtime_manifest(manifest,controller=controller,motion_tools=True)
        spec=next(s for s in runtime.capabilities if s.name=='goto_pose')
        assert ('clearance_first' in spec.arguments_schema['properties'])==(controller=='OSC_POSE')


def test_exception_after_clearance_keeps_actual_phase_execution(tmp_path):
    from captain_harness.adapters.capx.backend import ToolOperationError
    f=bounded_fixture(tmp_path)
    args=request(f.robot,f.backend)
    move=f.robot.move_to_pose_bounded
    def fail(*a,**kw):
        move(*a,**kw)
        raise ToolOperationError(FailureCategory.NOT_EXECUTED,'sensor failed after motion')
    f.robot.move_to_pose_bounded=fail
    result=f.backend.invoke('goto_pose',args)
    assert result.executed and not result.success
    assert result.failure_category is FailureCategory.BACKEND_ERROR
    assert result.value['failed_phase']=='clearance'
    assert result.value['phase_executions']['clearance']['executed'] is True
    assert result.value['control_budget']['used_steps']==3
    assert result.diagnostics['original_failure_category']=='not_executed'

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_guarded_cartesian_motion import make_robot
from test_libero_adapter import FakeOscEnv
from captain_harness.adapters.libero.contact_motion import _pose
from captain_harness.adapters.robotics.cartesian_path_timing import motion_preview, truncate_translation
from captain_harness.adapters.capx.direct_programming import DirectProgramming
from types import SimpleNamespace


def saved_robot(tmp_path, distance):
    env=FakeOscEnv(); robot=make_robot(tmp_path,env)
    end=_pose(robot.get_robot_state().value); start=end.copy(); start[0,3]-=distance
    robot._last_guarded_motion={'poses':[start.tolist(),end.tolist()], 'end_step':robot.simulation_steps,
        'action_id':'original_failure'}
    return robot,env


@pytest.mark.parametrize('distance,duration,minimum',[(.0796,6.,7.96),(.1197,10.,11.97)])
def test_known_impossible_full_recover_zero_execution(tmp_path,distance,duration,minimum):
    robot,env=saved_robot(tmp_path,distance)
    direct=DirectProgramming(SimpleNamespace(robot=robot))
    args=dict(max_distance_m=.2,max_duration_s=duration)
    preview=direct.invoke('preview_motion',dict(action='recover',arguments=args))
    assert preview.success and not preview.value["physical_action_executed"] and not env.actions
    assert preview.value['minimum_command_duration_s']==pytest.approx(minimum)
    result=robot.retreat_last_motion(**args)
    assert not result.executed and not env.actions
    assert result.value['stopped_reason']=='insufficient_command_duration'
    assert robot._last_guarded_motion['action_id']=='original_failure'


def test_partial_unloads_keeps_remaining_and_only_exits(tmp_path):
    robot,env=saved_robot(tmp_path,.0796)
    env.robots[0].ee_force=[177.88,0,0]
    original=env.step
    def unload(action):
        result=original(action);env.robots[0].ee_force=[5.,0,0];return result
    env.step=unload
    result=robot.retreat_last_motion(max_distance_m=.08,max_duration_s=6.,mode='partial')
    assert result.executed and not result.success
    assert result.value['force_decreased'] and not result.value['force_still_over_limit']
    assert not result.value['endpoint_reached'] and result.value['executed_distance_m']>0
    assert result.value['origin_action_id']=='original_failure'
    token=result.value['continuation_id'];assert token
    remaining=np.array(result.value['remaining_path'])
    assert np.all(np.diff(remaining[:,0,3])<=1e-10)
    before=robot.simulation_steps
    robot.get_observation_data()
    assert robot.simulation_steps==before
    continued=robot.retreat_last_motion(max_duration_s=10.,continuation_id=token)
    assert continued.success,continued.message
    assert continued.value['endpoint_reached'] and continued.value['continuation_id'] is None
    assert continued.value['original_goal_restored']=='pending_visual_confirmation'
    assert not robot.retreat_last_motion(continuation_id=token).executed
    assert not robot.retreat_last_motion().executed


@pytest.mark.parametrize('invalidate',['step','pose','gripper_command','gripper_measured'])
def test_continuation_binds_all_measured_state(tmp_path,invalidate):
    robot,env=saved_robot(tmp_path,.08)
    result=robot.retreat_last_motion(max_distance_m=.08,max_duration_s=2.,mode='partial')
    token=result.value['continuation_id']; assert token
    if invalidate=='step':robot.hold_position(.1)
    elif invalidate=='pose':env.joints[0]+=.01
    elif invalidate=='gripper_command':robot._gripper_command_fraction=.5
    else:robot._retreat_continuation['gripper_measured']-=.1
    n=len(env.actions)
    assert not robot.retreat_last_motion(continuation_id=token).executed
    assert len(env.actions)==n


def test_rotation_segments_and_exact_distance_boundary():
    path=np.repeat(np.eye(4)[None],3,axis=0)
    path[1,:3,:3]=Rotation.from_rotvec([0,0,1.]).as_matrix()
    path[2]=path[1];path[2,0,3]=.1
    truncated=truncate_translation(path,.0796)
    assert truncated[-1,0,3]==pytest.approx(.0796)
    value=motion_preview(truncated,.01,.2,10.,20)
    assert value['minimum_command_duration_s']==pytest.approx(12.96)
    assert value['budget_command_pose'][0][3]==pytest.approx(.05)


def test_move_preview_and_execution_share_time_refusal(tmp_path):
    robot,env=saved_robot(tmp_path,.08)
    direct=DirectProgramming(SimpleNamespace(robot=robot))
    state=robot.get_robot_state().value
    position=list(state.end_effector_position_m);position[0]+=.0796
    args=dict(position=position,quaternion=list(state.end_effector_quaternion_wxyz),max_speed_m_s=.01,max_duration_s=6.)
    preview=direct.invoke('preview_motion',dict(action='move_to',arguments=args))
    assert preview.success and not preview.value["physical_action_executed"]
    result=direct.invoke('move_to',args)
    assert not result.executed and result.message=='insufficient_command_duration'
    assert not env.actions
    assert result.value['minimum_command_duration_s']==preview.value['minimum_command_duration_s']
    jaw=direct.invoke('preview_motion',dict(action='move_to',arguments={'gripper':'close'}))
    assert jaw.success and not jaw.value['physical_action_executed']
    assert jaw.value['minimum_command_duration_s']==robot._gripper_steps/robot._control_frequency
    assert not env.actions


@pytest.mark.parametrize('action,args', [('hold',{'duration_s':.1}),('move_to',{'gripper':'close'}),('move_to',{'gripper':'open'})])
def test_stationary_final_sample_force_stop_and_target_cancellation(tmp_path,action,args):
    robot,env=saved_robot(tmp_path,.08)
    model=SimpleNamespace(limits=np.array([[-10.,10.]]*7),collision=lambda *a:{'collision':False})
    direct=DirectProgramming(SimpleNamespace(robot=robot,_motion_model=lambda:model))
    calls=[]
    robot.cancel_direct_targets=lambda **kw: calls.append(kw) or {'status':'cancelled'}
    robot._gripper_steps=2
    step=env.step
    def force_last(command):
        result=step(command)
        if len(env.actions)==2:env.robots[0].ee_force=[25.,0,0]
        return result
    env.step=force_last
    result=direct.invoke(action,args)
    assert result.executed and not result.success
    assert result.value['stopped_reason']=='contact_force'
    assert result.value['trajectory'][-1]['force_n']==25.
    assert result.value['trajectory'][-1]['step']==robot.simulation_steps
    assert calls==[{'cancel_gripper':True}]
    assert robot._last_guarded_motion is None


def test_selected_pixel_measurement_uses_only_declared_depth(tmp_path):
    robot,env=saved_robot(tmp_path,.08)
    direct=DirectProgramming(SimpleNamespace(robot=robot))
    cameras=robot.get_observation_data().value.cameras
    camera=cameras[0]
    if camera.depth_path is None:
        # Supply a declared camera measurement, without any object metadata.
        from dataclasses import replace
        from captain_harness.ports.capability import CapabilityResult
        depth=tmp_path/'selected_depth.npy';np.save(depth,np.full((64,64),.8))
        observation=robot.get_observation_data().value
        robot.get_observation_data=lambda:CapabilityResult.ok(replace(observation,
            cameras=(replace(camera,depth_path=str(depth)),)))
    before=robot.simulation_steps
    result=direct.invoke('measure_image_points',dict(camera=camera.camera_name,pixels=[[.5,.5],[.2,.7]]))
    assert result.success and robot.simulation_steps==before
    assert not result.value['physical_action_executed']
    assert len(result.value['points'])==2 and result.value['evidence_content_sha256']
    assert not direct.invoke('measure_image_points',dict(camera='hidden_camera',pixels=[[.5,.5]])).success


def test_continuation_never_commands_backwards_to_passed_exit_waypoint():
    from captain_harness.adapters.robotics.cartesian_path_timing import remaining_exit_path
    path=np.repeat(np.eye(4)[None],2,axis=0);path[1,0,3]=.02
    current=path[-1].copy();current[0,3]+=.001
    assert remaining_exit_path(path,current,path[-1],2) is None

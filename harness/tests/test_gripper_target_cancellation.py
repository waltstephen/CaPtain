"""Native position target cancellation: robot command state, never dynamics state."""
from types import SimpleNamespace
import numpy as np
import pytest

from captain_harness.adapters.libero.gripper_target import replace_gripper_target


@pytest.fixture
def hardware():
    from robosuite.models.grippers.panda_gripper import PandaGripper
    gripper=PandaGripper(idn=0)
    model=SimpleNamespace(
        actuator_name2id=lambda name: gripper.actuators.index(name)+1,
        joint_name2id=lambda name: gripper.joints.index(name)+7,
        actuator_ctrlrange=np.array([[-9.,9.],[0.,.04],[-.04,0.]]),
        actuator_trntype=np.array([0,0,0]),
        actuator_trnid=np.array([[0,-1],[7,-1],[8,-1]]),
        actuator_gear=np.array([[1.,0,0,0,0,0]]*3),
        actuator_dyntype=np.zeros(3,int),actuator_gaintype=np.zeros(3,int),
        actuator_biastype=np.ones(3,int),
        actuator_gainprm=np.array([[1000.,0,0,0,0,0,0,0,0,0]]*3),
        actuator_biasprm=np.array([[0.,-1000.,0,0,0,0,0,0,0,0]]*3))
    sim=SimpleNamespace(model=model,data=SimpleNamespace(ctrl=np.array([3.,0.,0.]),
        qpos=np.arange(10,dtype=float),qvel=np.arange(10,dtype=float)))
    return SimpleNamespace(robots=[SimpleNamespace(gripper=gripper)],sim=sim)


def native_tick(env,action):
    g=env.robots[0].gripper
    normalized=g.format_action(np.array([action]))
    bounds=env.sim.model.actuator_ctrlrange[1:]
    env.sim.data.ctrl[1:]=bounds.mean(axis=1)+np.diff(bounds,axis=1).ravel()*.5*normalized


def test_replaces_saturated_target_using_both_asymmetric_public_joints(hardware):
    for _ in range(200):native_tick(hardware,1.)
    np.testing.assert_array_equal(hardware.sim.data.ctrl[1:],[0.,0.])
    native_tick(hardware,0.)  # Counterexample: zero alone retains closed target.
    np.testing.assert_array_equal(hardware.sim.data.ctrl[1:],[0.,0.])
    before_q=hardware.sim.data.qpos.copy();before_v=hardware.sim.data.qvel.copy()
    receipt=replace_gripper_target(hardware,[.024,-.022])
    assert receipt['status']=='replaced'
    assert receipt['previous_target_m']==[0.,0.]
    np.testing.assert_allclose(receipt['target_m'],[.024,-.022],atol=1e-15)
    for _ in range(50):native_tick(hardware,0.)
    np.testing.assert_allclose(hardware.sim.data.ctrl,[3.,.024,-.022],atol=1e-15)
    np.testing.assert_array_equal(hardware.sim.data.qpos,before_q)
    np.testing.assert_array_equal(hardware.sim.data.qvel,before_v)
    native_tick(hardware,-1.)
    np.testing.assert_allclose(hardware.sim.data.ctrl[1:],[.0242,-.0222],atol=1e-15)


@pytest.mark.parametrize('reading',[[float('nan'),-.02],[.02],[.02,-.05],[True,False]])
def test_invalid_public_reading_does_not_mutate_command(hardware,reading):
    old=hardware.robots[0].gripper.current_action.copy();ctrl=hardware.sim.data.ctrl.copy()
    with pytest.raises(ValueError):replace_gripper_target(hardware,reading)
    np.testing.assert_array_equal(old,hardware.robots[0].gripper.current_action)
    np.testing.assert_array_equal(ctrl,hardware.sim.data.ctrl)


@pytest.mark.parametrize('field,index,value',[
    ('actuator_trntype',1,1),('actuator_trnid',(1,0),8),
    ('actuator_gear',(1,0),2.),('actuator_ctrlrange',(1,1),.08),
    ('actuator_dyntype',1,1),('actuator_gaintype',1,1),
    ('actuator_biastype',1,0),('actuator_biasprm',(1,1),-900.),
    ('actuator_gainprm',(1,0),900.)])
def test_mismatched_compiled_mapping_is_rejected_before_writes(hardware,field,index,value):
    getattr(hardware.sim.model,field)[index]=value
    old=hardware.robots[0].gripper.current_action.copy();ctrl=hardware.sim.data.ctrl.copy()
    with pytest.raises(ValueError):replace_gripper_target(hardware,[.02,-.02])
    np.testing.assert_array_equal(old,hardware.robots[0].gripper.current_action)
    np.testing.assert_array_equal(ctrl,hardware.sim.data.ctrl)


def test_unknown_codec_is_not_guessed(hardware):
    hardware.robots[0].gripper=SimpleNamespace(current_action=np.zeros(2))
    with pytest.raises(ValueError,match='codec'):replace_gripper_target(hardware,[.02,-.02])


def test_force_stop_publishes_replacement_without_changing_failure():
    from test_closure_force_monitor import Robot
    robot=Robot();calls=[]
    robot.force=lambda: 5. if robot.simulation_steps<2 else 12.
    def cancel():
        calls.append(robot.simulation_steps)
        return {'status':'replaced','target_m':[.024,-.022]}
    robot._replace_gripper_target_with_measured_position=cancel
    result=robot._set_gripper(0.)
    assert calls==[2]
    assert not result.success and result.executed and result.message=='closure_force_increase'
    command=result.diagnostics['closure_force_monitor']['post_stop_gripper_command']
    assert command['effect']=='hold_measured_joint_target'
    assert command['target_replacement']['target_m']==[.024,-.022]
    assert robot._gripper_command_fraction==.5


def test_unavailable_cancellation_blocks_advance_until_explicit_open(tmp_path):
    from test_closure_force_monitor import make_actual_monitored_robot
    env,robot=make_actual_monitored_robot(tmp_path)
    def broken():raise ValueError('unsupported position servo')
    robot._replace_gripper_target_with_measured_position=broken
    robot._eef_force_norm=lambda:5. if robot.simulation_steps<2 else 12.
    failed=robot.close_gripper();count=len(env.actions)
    assert not failed.success and failed.executed
    command=failed.diagnostics['closure_force_monitor']['post_stop_gripper_command']
    assert command['target_replacement']['status']=='unavailable'
    assert not robot.hold_position(.1).executed
    assert len(env.actions)==count
    assert robot.open_gripper().success
    assert robot.hold_position(.1).success



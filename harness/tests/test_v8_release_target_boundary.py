import numpy as np
import pytest
from test_gripper_target_cancellation import hardware, native_tick
from captain_harness.adapters.libero.gripper_target import replace_gripper_target

@pytest.fixture(autouse=True)
def initialized_command(hardware):
    native_tick(hardware,0.)

def test_loaded_closed_boundary_projects_only_toward_opening(hardware):
    measured=np.array([-0.00006139888929207253,-0.0010611291624763534])
    before_q=hardware.sim.data.qpos.copy();before_v=hardware.sim.data.qvel.copy()
    r=replace_gripper_target(hardware,measured,max_open_fraction_correction=.2)
    np.testing.assert_allclose(r['target_m'],[0.,measured[1]],atol=1e-15)
    assert r['opening_projection_fraction']==pytest.approx(-measured[0]/.04)
    assert r['target_basis']=='opening_only_command_range_projection'
    assert r['measured_joint_positions_m']==measured.tolist()
    native_tick(hardware,0.)
    np.testing.assert_allclose(hardware.sim.data.ctrl[1:],[0.,measured[1]],atol=1e-15)
    np.testing.assert_array_equal(hardware.sim.data.qpos,before_q)
    np.testing.assert_array_equal(hardware.sim.data.qvel,before_v)
    native_tick(hardware,-1.)
    assert hardware.sim.data.ctrl[1]>0 and hardware.sim.data.ctrl[2]<measured[1]

@pytest.mark.parametrize('measured,budget',[
    ([.041,-.02],.2),([.02,-.041],.2),([-.01,-.02],.2),
    ([-.004,-.02],.1),([float('nan'),-.02],.2)])
def test_correction_never_closes_or_exceeds_budget(hardware,measured,budget):
    old=hardware.robots[0].gripper.current_action.copy();ctrl=hardware.sim.data.ctrl.copy()
    with pytest.raises(ValueError):
        replace_gripper_target(hardware,measured,max_open_fraction_correction=budget)
    np.testing.assert_array_equal(hardware.robots[0].gripper.current_action,old)
    np.testing.assert_array_equal(hardware.sim.data.ctrl,ctrl)

def test_default_cancel_keeps_strict_range_contract(hardware):
    with pytest.raises(ValueError,match='outside command range'):
        replace_gripper_target(hardware,[-.00006,-.001])

def test_projection_still_validates_compiled_servo_before_writes(hardware):
    hardware.sim.model.actuator_gainprm[1,0]=900.
    old=hardware.sim.data.ctrl.copy()
    with pytest.raises(ValueError):
        replace_gripper_target(hardware,[-.00006,-.001],max_open_fraction_correction=.2)
    np.testing.assert_array_equal(hardware.sim.data.ctrl,old)

def test_release_debits_projection_from_remaining_opening_budget(tmp_path):
    from test_r30_v8_release import setup
    direct,robot,env,cancel=setup(tmp_path,[40.,10.,5.])
    requests=[];resets=[]
    def reset(**kwargs):
        resets.append(kwargs)
        return {'status':'cancelled','gripper_target':{'opening_projection_fraction':.01}}
    robot.cancel_direct_targets=reset
    original=robot.release_opening
    def opening(duration,budget):
        requests.append((duration,budget));return original(duration,budget)
    robot.release_opening=opening
    result=direct.invoke('release',{'max_duration_s':.2,'max_open_fraction_increase':.2})
    assert result.executed and result.success
    assert requests==[(.2,pytest.approx(.19))]
    assert resets[0]['max_open_fraction_correction']==.2

def test_native_substeps_cannot_accumulate_beyond_release_command_budget(hardware):
    from captain_harness.adapters.libero.gripper_target import bounded_opening_commands
    replace_gripper_target(hardware,[.001,-.002])
    original=hardware.robots[0].gripper.format_action
    with bounded_opening_commands(hardware.robots[0].gripper,.2) as receipt:
        for _ in range(500):native_tick(hardware,-1.)
        np.testing.assert_allclose(hardware.sim.data.ctrl[1:],[.009,-.010],atol=1e-15)
        native_tick(hardware,1.)  # Even an erroneous close cannot exceed the starting bound.
        assert hardware.sim.data.ctrl[1]>=.001 and hardware.sim.data.ctrl[2]<=-.002
        assert receipt['maximum_open_fraction_increase']==.2
    assert hardware.robots[0].gripper.format_action==original

def test_release_command_limiter_is_removed_on_failure(hardware):
    from captain_harness.adapters.libero.gripper_target import bounded_opening_commands
    original=hardware.robots[0].gripper.format_action
    with pytest.raises(RuntimeError):
        with bounded_opening_commands(hardware.robots[0].gripper,.2):
            raise RuntimeError('controller stop')
    assert hardware.robots[0].gripper.format_action==original

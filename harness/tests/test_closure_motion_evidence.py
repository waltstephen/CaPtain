from types import SimpleNamespace
import pytest
from test_closure_force_monitor import Robot
from captain_harness.ports.capability import CapabilityResult


def test_peak_closure_motion_is_not_final_position_error():
    robot = Robot()
    positions = [0., .004, .007, .003, .001]
    robot.get_robot_state = lambda: CapabilityResult.ok(SimpleNamespace(
        end_effector_position_m=(positions[robot.simulation_steps],0.,0.),
        end_effector_quaternion_wxyz=(1.,0.,0.,0.), gripper_open_fraction=1.-robot.simulation_steps/4))
    result = robot.descend_and_close([0,0,0], [1,0,0,0])
    assert result.success and robot.simulation_steps == 4
    motion = result.value['closure_force_monitor']['closure_motion']
    assert motion['status'] == 'measured'
    assert motion['max_sampled_translation_m'] == pytest.approx(.007)
    assert motion['final_translation_m'] == pytest.approx(.001)
    assert motion['final_rotation_rad'] == pytest.approx(0.)
    assert motion['valid_samples'] == motion['total_samples'] == 5
    assert motion['physical_retention_verified'] is False


def test_missing_pose_is_unknown_without_changing_closure_result():
    robot = Robot()
    result = robot.descend_and_close([0,0,0], [1,0,0,0])
    assert result.success and robot.simulation_steps == 4
    motion = result.value['closure_force_monitor']['closure_motion']
    assert motion['status'] == 'unavailable'
    assert motion['max_sampled_translation_m'] is None


def test_missing_initial_pose_cannot_rebase_drift_to_later_sample():
    robot = Robot()
    def state():
        if robot.simulation_steps == 0:
            raise RuntimeError('unavailable')
        return CapabilityResult.ok(SimpleNamespace(end_effector_position_m=(1.,0.,0.),
            end_effector_quaternion_wxyz=(1.,0.,0.,0.), gripper_open_fraction=.5))
    robot.get_robot_state = state
    result = robot.descend_and_close([0,0,0], [1,0,0,0])
    motion = result.value['closure_force_monitor']['closure_motion']
    assert result.success and motion['status'] == 'unavailable'
    assert motion['valid_samples'] == 4
    assert motion['final_translation_m'] is None


def test_successful_direct_close_keeps_monitor_for_public_adapter(tmp_path):
    from test_closure_force_monitor import make_actual_monitored_robot
    env, robot=make_actual_monitored_robot(tmp_path)
    robot._eef_force_norm=lambda:5.
    result=robot.close_gripper()
    assert result.success
    assert result.diagnostics['closure_force_monitor']['closure_motion']['status']=='measured'
    assert result.diagnostics['closure_force_monitor']['control_steps']==4

from dataclasses import replace
import numpy as np
import pytest
from captain_harness.adapters.libero.recorded_backend import ClosingMonitoredPossessionActionBackend
from captain_harness.ports.capability import CapabilityResult, FailureCategory
from test_libero_adapter import FakeOscEnv


def robot_fixture(tmp_path):
    class DisturbedClosingEnv(FakeOscEnv):
        def step(self, action):
            result=super().step(action)
            if action[-1]>0:
                self.joints[0]+=.003
            return (self._get_observations(),*result[1:])
    env=DisturbedClosingEnv()
    robot=ClosingMonitoredPossessionActionBackend(env,observation_dir=tmp_path,
        initial_observation=env.reset(),horizon=1000,controller='OSC_POSE',gripper_steps=4)
    return env,robot


def test_successful_closure_reports_final_pose_not_preclosure_pose(tmp_path):
    env,robot=robot_fixture(tmp_path);initial=robot.get_robot_state().value
    target=np.asarray(initial.end_effector_position_m)
    result=robot.descend_and_close(target,initial.end_effector_quaternion_wxyz)
    post=robot.get_robot_state().value;remaining=target-np.asarray(post.end_effector_position_m)
    assert result.success and result.executed and len(env.actions)==4
    assert np.linalg.norm(remaining)>.001
    np.testing.assert_allclose(result.value['observed_eef_position_m'],post.end_effector_position_m,atol=1e-12,rtol=0)
    np.testing.assert_allclose(result.value['remaining_vector_m'],remaining,atol=1e-12,rtol=0)
    assert result.value['position_residual_m']==pytest.approx(np.linalg.norm(remaining))
    assert result.value['measurement_phase']=='after_closure'
    np.testing.assert_allclose(result.value['descent_execution']['observed_eef_position_m'],target)
    assert result.value['grasp_outcome']=='empty_close'


def test_postclosure_gripper_sensor_failure_keeps_physical_and_descent_receipts(tmp_path):
    env,robot=robot_fixture(tmp_path);initial=robot.get_robot_state().value
    robot.gripper_state=lambda:replace(CapabilityResult.failed(FailureCategory.BACKEND_ERROR,
        'gripper_sensor_disconnected'),diagnostics={'sensor_receipt':{'available':False}},
        attempt_identity={'sensor_attempt':'original'})
    result=robot.descend_and_close(initial.end_effector_position_m,initial.end_effector_quaternion_wxyz)
    assert not result.success and result.executed and len(env.actions)==4
    assert result.failure_category is FailureCategory.BACKEND_ERROR
    assert result.diagnostics['sensor_receipt']=={'available':False}
    assert result.attempt_identity=={'sensor_attempt':'original'}
    assert result.value['failed_phase']=='post_closure_measurement'
    assert result.value['closure_executed'] is True
    assert result.value['grasp_outcome']=='unverified'
    np.testing.assert_allclose(result.value['observed_eef_position_m'],robot.get_robot_state().value.end_effector_position_m)
    assert result.value['descent_execution']['stopped_reason']=='target_reached'

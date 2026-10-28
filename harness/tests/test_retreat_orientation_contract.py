"""Measured path recovery has an absolute orientation contract, with strict stops."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_guarded_cartesian_motion import make_robot
from test_libero_adapter import FakeOscEnv, RotationLockedAxisOscEnv


class ContactYaw(RotationLockedAxisOscEnv):
    def __init__(self, recorded_yaw, residual_yaw, rising_force=False):
        super().__init__()
        self.recorded_yaw = recorded_yaw
        self.residual_yaw = residual_yaw
        self.rising_force = rising_force
        self.contacted = False

    def step(self, action):
        response = FakeOscEnv.step(self, action)
        if self.joints[0] >= .005:
            self.contacted = True
            self.yaw = self.recorded_yaw
            self.robots[0].ee_force = [23., 0., 0.]
        elif self.contacted:
            self.yaw = self.residual_yaw
            self.robots[0].ee_force = [29. if self.rising_force else 5., 0., 0.]
        return response


@pytest.mark.parametrize("recorded,residual",[(.003,.001),(-.008,-.003)])
def test_recorded_retreat_accepts_absolute_orientation_but_preserves_residual(tmp_path,recorded,residual):
    env=ContactYaw(recorded,residual)
    robot=make_robot(tmp_path,env)
    start=np.array(robot.get_robot_state().value.end_effector_position_m)
    forward=robot.move_linear([1,0,0],.04)
    assert not forward.success and forward.value["stopped_reason"]=="contact_force"
    result=robot.retreat_last_motion(max_distance_m=.04,max_duration_s=6.)
    assert result.success,result.message
    assert result.value["orientation_reference"]=="recorded_pose"
    assert result.value["orientation_tolerance_rad"]==pytest.approx(.01)
    assert result.value["orientation_residual_rad"]==pytest.approx(abs(residual))
    assert result.value["requested_rotation_vector_rad"][2]==pytest.approx(-recorded)
    assert result.value["actual_rotation_vector_rad"][2]==pytest.approx(residual-recorded)
    assert result.value["position_residual_m"]<=result.value["position_tolerance_m"]
    assert np.linalg.norm(np.array(robot.get_robot_state().value.end_effector_position_m)-start)<=.0005
    assert result.value["observed_force_n"]==5.
    assert result.value["endpoint_settled"] is True
    assert result.value["control_steps"]<120
    assert robot._last_guarded_motion is None


@pytest.mark.parametrize("case",["orientation","force","duration"])
def test_retreat_still_rejects_bad_orientation_force_or_duration(tmp_path,case):
    env=ContactYaw(.024 if case=="orientation" else .003,.02 if case=="orientation" else .001,case=="force")
    robot=make_robot(tmp_path,env)
    assert not robot.move_linear([1,0,0],.04).success
    result=robot.retreat_last_motion(max_distance_m=.04,max_duration_s=.05 if case=="duration" else 6., mode="partial" if case=="duration" else "full")
    assert result.executed and not result.success
    if case=="orientation":
        assert result.value["orientation_residual_rad"]>.01
    elif case=="force":
        assert result.value["stopped_reason"]=="retreat_force_increased"
        assert result.value["control_steps"]==1
    else:
        assert result.value["stopped_reason"]=="duration_exhausted"
        assert result.value["control_steps"]==1


@pytest.mark.parametrize("angle",[-.004,.004])
@pytest.mark.parametrize("kind",["relative","absolute"])
def test_explicit_small_rotation_still_requires_requested_precision(tmp_path,angle,kind):
    robot=make_robot(tmp_path,FakeOscEnv()) # This controller cannot actuate yaw.
    state=robot.get_robot_state().value
    if kind=="relative":
        result=robot.rotate_about_axis([0,0,1],state.end_effector_position_m,angle)
    else:
        q=np.roll(Rotation.from_rotvec([0,0,angle]).as_quat(),1)
        result=robot.move_to_pose_monitored(state.end_effector_position_m,q)
    assert result.executed and not result.success
    assert result.value["orientation_tolerance_rad"]==pytest.approx(.0004)
    assert result.value["orientation_residual_rad"]==pytest.approx(abs(angle))

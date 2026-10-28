import numpy as np
import pytest
from captain_harness.adapters.libero.backend import LiberoActionBackend
from test_libero_adapter import FakeOscEnv

class SlowPoseEnv(FakeOscEnv):
    direction=1.
    def step(self,action):
        bounded=np.asarray(action,dtype=float).copy()
        bounded[:3]=np.clip(bounded[:3],-.002,.002)*self.direction
        return super().step(bounded)

def fixture(tmp_path,direction=1.):
    env=SlowPoseEnv();env.direction=direction
    robot=LiberoActionBackend(env,observation_dir=tmp_path,initial_observation=env.reset(),
        horizon=1000,controller="OSC_POSE",cartesian_max_steps=100,gripper_steps=1)
    state=robot.get_robot_state().value
    return env,robot,np.asarray(state.end_effector_position_m)+[.003,0,0],state.end_effector_quaternion_wxyz

def test_small_but_consistent_progress_reaches_declared_descent_tolerance(tmp_path):
    env,robot,target,quat=fixture(tmp_path)
    r=robot.descend_and_close(target,quat,position_tolerance_m=.0005)
    assert r.value["closure_executed"],r.message
    assert r.value["descent_execution"]["position_residual_m"]<=.0005
    assert len(env.actions)>3

@pytest.mark.parametrize("direction",[0.,-1.])
def test_stationary_or_slowly_diverging_descent_stops_without_closing(tmp_path,direction):
    env,robot,target,quat=fixture(tmp_path,direction)
    r=robot.descend_and_close(target,quat,position_tolerance_m=.0005)
    assert not r.success and not r.value["closure_executed"]
    assert r.value["stopped_reason"]=="position_stall"
    assert len(env.actions)==3

def test_force_stop_still_prevents_closure_during_small_progress(tmp_path):
    env,robot,target,quat=fixture(tmp_path)
    calls=[0]
    def force():
        calls[0]+=1
        return 20. if calls[0]>2 else 0.
    robot._eef_force_norm=force
    r=robot.descend_and_close(target,quat,position_tolerance_m=.0005)
    assert not r.success and not r.value["closure_executed"]
    assert r.value["stopped_reason"]=="contact_force"

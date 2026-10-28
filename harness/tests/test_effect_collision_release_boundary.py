from dataclasses import dataclass,replace
from types import SimpleNamespace
import numpy as np
import pytest
from captain_harness.ports.capability import CapabilityResult
from captain_harness.adapters.capx.direct_programming import DirectProgramming
from test_effect_collision_geometry import hit

@dataclass
class State:
    joint_positions_rad:tuple=(0.,)*7
    end_effector_position_m:tuple=(0.,0.,0.)
    end_effector_quaternion_wxyz:tuple=(1.,0.,0.,0.)
    gripper_open_fraction:float=.01

class Robot:
    def __init__(self):
        self.state=State();self.simulation_steps=0;self._control_frequency=20
        self._gripper_command_fraction=.01;self.actions=0
    def get_robot_state(self):return CapabilityResult.ok(self.state)
    def _eef_force_norm(self):return 5.-.01*self.simulation_steps
    def cancel_direct_targets(self,**kw):return {'status':'cancelled'}
    def release_opening(self,*args):
        for i in range(4):
            if not self._motion_guard():break
            self.actions+=1;self.simulation_steps+=1
            self.state=replace(self.state,gripper_open_fraction=self.state.gripper_open_fraction+.02)
        return CapabilityResult.ok({'control_steps':self.actions})

@pytest.mark.parametrize('bad_preview',[False,True])
def test_explicit_release_predicted_then_monitored_and_ordinary_hold_still_refused(bad_preview):
    robot=Robot()
    def collision(q,a,p):
        return hit(.001+a*.01 if bad_preview else max(.002-a*.05,0.))
    model=SimpleNamespace(limits=np.array([[-2,2]]*7),collision=collision)
    direct=DirectProgramming(SimpleNamespace(robot=robot,_motion_model=lambda:model))
    held,_=direct._guarded_execution(lambda:CapabilityResult.ok({}),model=model,stationary=True)
    assert not held.executed and robot.actions==0
    released=direct._release(max_duration_s=.5,max_open_fraction_increase=.25)
    assert released.executed is (not bad_preview)
    assert released.success is (not bad_preview)
    if bad_preview:
        assert robot.actions==0 and released.message=='bounded_release_geometry_unavailable'
    else:
        assert robot.actions==4
        assert released.value['execution_stop']['collision_release']['remaining_pairs']==[]

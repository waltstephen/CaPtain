import numpy as np
from captain_harness.adapters.libero.substep_force_guard import SubstepForceMonitor, monitored_step

class Env:
    model_timestep=.002
    def __init__(self, forces):
        self.forces=forces;self.force=5.;self.actions=[];self.i=0
    def _pre_action(self,action,policy_step=False):
        self.actions.append((list(action),policy_step))
    def _update_observables(self,force=False): pass
    def step(self,action):
        for i,f in enumerate(self.forces):
            self._pre_action(action,policy_step=i==0)
            self.force=f;self.i+=1
            self._update_observables()
        return {"state":self.i},0,False,{}

def test_impulse_triggers_within_native_period_and_preserves_physics_steps():
    env=Env([6,20,12,7,5])
    monitor=SubstepForceMonitor(initial_force=5,max_force=18,retreat=False)
    cancelled=[]
    pre,update=env._pre_action,env._update_observables
    result=monitored_step(env,np.array([.2,0,0,0,0,0,1.]),monitor,
        read_force=lambda:env.force,cancel=lambda:cancelled.append(True) or {"status":"cancelled"})
    assert result[0]["state"]==5 and len(cancelled)==1
    assert env.actions[0][0][0]==.2 and env.actions[1][0][0]==.2
    assert all(a==[0.]*7 for a,p in env.actions[2:])
    assert monitor.peak_force==20 and monitor.trigger["sample_index"]==2
    assert monitor.receipt()["sampling_interval_s"]==.002
    assert env._pre_action==pre and env._update_observables==update

def test_no_contact_preserves_original_commands_and_native_updates():
    env=Env([5,5.1,5.2,5.1])
    m=SubstepForceMonitor(initial_force=5,max_force=18,retreat=False)
    action=np.array([.1,0,0,0,0,0,-1.])
    monitored_step(env,action,m,read_force=lambda:env.force,cancel=lambda:None)
    assert all(np.array_equal(a,action) for a,p in env.actions)
    assert [p for a,p in env.actions]==[True,False,False,False]
    assert m.trigger is None

def test_loaded_retreat_allowed_but_new_force_rise_stops():
    env=Env([100,80,81,86,70])
    env.force=134
    m=SubstepForceMonitor(initial_force=134,max_force=18,retreat=True)
    monitored_step(env,np.ones(7),m,read_force=lambda:env.force,cancel=lambda:{"status":"cancelled"})
    assert m.trigger["reason"]=="retreat_force_increased"
    assert m.trigger["force_n"]==86
    assert env.actions[0][0]==[1.]*7
    assert env.actions[-1][0]==[0.]*7

def test_exception_restores_hooks_without_claiming_success():
    import pytest
    env=Env([5])
    pre=env._pre_action
    def broken(force=False):raise RuntimeError("sensor failed")
    env._update_observables=broken
    with pytest.raises(RuntimeError):
        monitored_step(env,np.ones(7),SubstepForceMonitor(initial_force=5,max_force=18,retreat=False),
            read_force=lambda:env.force,cancel=lambda:None)
    assert env._pre_action==pre and env._update_observables==broken


def test_libero_style_outer_wrapper_is_instrumented_at_native_method_owner():
    class Wrapper:
        def __init__(self):self.env=Env([5,20,7,5])
        def step(self,action):return self.env.step(action)
        def _update_observables(self,force=False):return self.env._update_observables(force=force)
    outer=Wrapper()
    m=SubstepForceMonitor(initial_force=5,max_force=18,retreat=False)
    monitored_step(outer,np.ones(7),m,read_force=lambda:outer.env.force,cancel=lambda:{"status":"cancelled"})
    assert m.trigger and outer.env.actions[-1][0]==[0.]*7
    assert "_pre_action" not in vars(outer) and "_pre_action" not in vars(outer.env)

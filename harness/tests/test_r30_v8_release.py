import numpy as np
import pytest
from types import SimpleNamespace
from test_r30_v8_motion import saved_robot
from captain_harness.adapters.capx.direct_programming import DirectProgramming, direct_manifest

def setup(tmp_path, forces, aperture_step=.01, drift=0.):
    robot, env = saved_robot(tmp_path,.08)
    env.gripper=.01
    robot._observation=env._get_observations()
    env.robots[0].ee_force=[forces[0],0,0]
    model=SimpleNamespace(limits=np.array([[-10.,10.]]*7), collision=lambda *a:{"collision":False})
    direct=DirectProgramming(SimpleNamespace(robot=robot,_motion_model=lambda:model))
    from contextlib import nullcontext
    robot._bounded_gripper_opening=lambda budget:nullcontext({"test_scope":"force guard fixture"})
    cancellations=[]
    robot.cancel_direct_targets=lambda **kw: cancellations.append(kw) or {"status":"cancelled"}
    old=env.step
    def step(action):
        before=env.gripper
        old(action)
        env.gripper=before+aperture_step*.04
        env.joints[0]+=drift
        env.robots[0].ee_force=[forces[min(len(env.actions),len(forces)-1)],0,0]
        return env._get_observations(),0,False,{}
    env.step=step
    return direct, robot, env, cancellations

@pytest.mark.parametrize("initial",[18.,40.,177.88])
def test_explicit_release_unloads_when_ordinary_open_refused(tmp_path,initial):
    direct,robot,env,cancel=setup(tmp_path,[initial,10.,5.])
    rejected=direct.invoke("move_to",{"gripper":"open"})
    assert not rejected.executed and not env.actions
    result=direct.invoke("release",{"max_duration_s":.2})
    assert result.executed and result.success, result
    assert result.value["force_decreased"] and not result.value["force_still_over_limit"]
    assert result.value["object_released"]=="pending_visual_confirmation"
    assert result.value["stopped_reason"]=="release_duration_limit"
    assert result.value["related_failure_action_id"]==rejected.value["action_id"]
    assert all(a[-1]<0 for a in env.actions)
    assert cancel==[{"cancel_gripper":True,"max_open_fraction_correction":.25},{"cancel_gripper":True}]
    assert robot._last_guarded_motion is None

@pytest.mark.parametrize("forces,reason,steps",[
    ([40.,43.],"release_force_rise",1),
    ([40.,30.,33.],"release_force_rise",2),
    ([40.],"release_not_unloading",10),
    ([40.,float("nan")],"force_unavailable",1),
])
def test_release_stops_at_force_boundary_including_final_sample(tmp_path,forces,reason,steps):
    direct,robot,env,cancel=setup(tmp_path,forces)
    result=direct.invoke("release",{"max_duration_s":1.})
    assert not result.success and result.executed
    assert result.value["stopped_reason"]==reason
    assert len(env.actions)==steps
    assert result.value["trajectory"][-1]["step"]==robot.simulation_steps
    assert cancel==[{"cancel_gripper":True,"max_open_fraction_correction":.25},{"cancel_gripper":True}]

@pytest.mark.parametrize("kwargs,reason",[
    ({"drift":.004},"tracking_error_limit"),
    ({"aperture_step":-.02},"release_aperture_reversed"),
])
def test_release_rejects_drift_or_closing(tmp_path,kwargs,reason):
    direct,robot,env,cancel=setup(tmp_path,[40.,30.],**kwargs)
    result=direct.invoke("release",{})
    assert result.value["stopped_reason"]==reason and len(env.actions)==1

@pytest.mark.parametrize("args",[
    {"max_duration_s":.01}, {"max_duration_s":3.}, {"max_duration_s":True},
    {"max_open_fraction_increase":-.1}, {"max_open_fraction_increase":.3},
])
def test_release_invalid_budget_zero_execution(tmp_path,args):
    direct,robot,env,cancel=setup(tmp_path,[40.])
    assert not direct.invoke("release",args).executed
    assert not env.actions

def test_release_guard_rejection_preserves_trace_and_command(tmp_path):
    direct,robot,env,cancel=setup(tmp_path,[40.])
    direct.tools._motion_model().collision=lambda *a:{"collision":True}
    previous=robot._gripper_command_fraction
    result=direct.invoke("release",{})
    assert not result.executed and not env.actions
    assert robot._gripper_command_fraction==previous
    assert robot._last_guarded_motion is not None
    assert result.value["trajectory"]

def test_release_manifest_is_physical_and_no_arm_or_close_arguments():
    from jsonschema import validate, ValidationError
    from captain_harness.ports.capability import SideEffect
    spec=next(s for s in direct_manifest(SimpleNamespace(capabilities=(),provider="test")).capabilities if s.name=="release")
    assert spec.side_effect==SideEffect.ROBOT_MOTION
    with pytest.raises(ValidationError):validate({"position":[0,0,0]},spec.arguments_schema)
    with pytest.raises(ValidationError):validate({"gripper":"close"},spec.arguments_schema)


def test_release_target_reset_failure_never_steps(tmp_path):
    direct,robot,env,cancel=setup(tmp_path,[40.])
    robot.cancel_direct_targets=lambda **kw: {"status":"unconfirmed"}
    result=direct.invoke("release",{})
    assert not result.executed and not env.actions
    assert result.message=="release_target_reset_unconfirmed"


def test_release_invalidates_exit_continuation_only_after_execution(tmp_path):
    direct,robot,env,cancel=setup(tmp_path,[40.,10.,5.])
    saved={"continuation_id":"prior-exit","remaining_path":["outward-only"]}
    robot._retreat_continuation=saved
    rejected=direct.invoke("release",{"max_duration_s":.01})
    assert not rejected.executed and robot._retreat_continuation is saved
    result=direct.invoke("release",{"max_duration_s":.2})
    assert result.executed and robot._retreat_continuation is None

def test_release_opening_limit_is_distinct_from_unloading(tmp_path):
    direct,robot,env,cancel=setup(tmp_path,[40.,39.],aperture_step=.1)
    result=direct.invoke("release",{"max_open_fraction_increase":.05})
    assert result.executed and not result.success
    assert result.value["opening_limit_reached"] and result.value["endpoint_reached"]
    assert result.value["force_decreased"] and result.value["force_still_over_limit"]
    assert result.value["stopped_reason"]=="release_opening_limit"
    assert len(env.actions)==1


def test_already_open_release_refusal_does_not_reset_targets(tmp_path):
    direct,robot,env,cancel=setup(tmp_path,[40.])
    env.gripper=.04
    robot._observation=env._get_observations()
    saved=robot._last_guarded_motion
    result=direct.invoke("release",{})
    assert not result.executed and not env.actions
    assert result.message=="release_already_open"
    assert not cancel
    assert robot._last_guarded_motion is saved


def test_loaded_release_does_not_require_any_retreat_history(tmp_path):
    direct,robot,env,cancel=setup(tmp_path,[40.,10.,5.])
    robot._last_guarded_motion=None
    robot._retreat_continuation=None
    assert not direct.invoke("recover",{}).executed
    assert not direct.invoke("move_to",{"gripper":"open"}).executed
    assert not env.actions
    result=direct.invoke("release",{"max_duration_s":.2})
    assert result.success and result.executed
    assert result.value["force_decreased"] and not result.value["force_still_over_limit"]
    assert robot._last_guarded_motion is None and robot._retreat_continuation is None

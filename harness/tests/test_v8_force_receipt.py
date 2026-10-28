"""Public force receipts must include return-time samples without extra motion."""
import numpy as np
import pytest
from types import SimpleNamespace
from test_r30_v8_motion import saved_robot
from captain_harness.adapters.capx.direct_programming import DirectProgramming


@pytest.mark.parametrize("action,args", [
    ("move_to", {"gripper": "close"}),
    ("move_to", {"gripper": "open"}),
    ("hold", {"duration_s": .1}),
])
def test_post_stop_force_is_in_peak_and_final_boundary(tmp_path, action, args):
    robot, env = saved_robot(tmp_path, .08)
    model = SimpleNamespace(limits=np.array([[-10.,10.]]*7),
        collision=lambda *a: {"collision":False})
    direct = DirectProgramming(SimpleNamespace(robot=robot, _motion_model=lambda:model))
    robot._gripper_steps = 2
    original = env.step
    def step(command):
        result = original(command)
        if len(env.actions) == 2:
            env.robots[0].ee_force = [18.125760955932805, 0., 0.]
        return result
    env.step = step
    cancellations = []
    def cancel(**kwargs):
        cancellations.append(kwargs)
        env.robots[0].ee_force = [21.965626406813, 0., 0.]
        return {"status":"cancelled"}
    robot.cancel_direct_targets = cancel
    result = direct.invoke(action, args)
    assert result.executed and not result.success
    assert result.value["stopped_reason"] == "contact_force"
    assert len(env.actions) == 2 and len(cancellations) == 1
    assert result.value["max_observed_force_n"] == pytest.approx(21.965626406813)
    assert result.value["observed_force_n"] == pytest.approx(21.965626406813)
    final = result.value["final_force_sample"]
    assert final["step"] == robot.simulation_steps == result.value["end_step"]
    assert final["force_n"] == result.value["observed_force_n"]
    assert final["phase"] == "after_driver_and_target_stop"
    assert result.value["force_still_over_limit"] is True


def test_same_step_refreshed_guard_force_is_not_discarded(tmp_path):
    robot, env = saved_robot(tmp_path, .08)
    model = SimpleNamespace(limits=np.array([[-10.,10.]]*7),
        collision=lambda *a: {"collision":False})
    direct = DirectProgramming(SimpleNamespace(robot=robot, _motion_model=lambda:model))
    from captain_harness.ports.capability import CapabilityResult
    def execute():
        env.robots[0].ee_force = [19., 0., 0.]
        return CapabilityResult.ok({})
    result, _ = direct._guarded_execution(execute, model=model, stationary=True)
    assert not result.success and not env.actions
    assert result.value["trajectory"][-1]["force_n"] == 19.
    assert result.value["max_observed_force_n"] == 19.
    assert len({x["step"] for x in result.value["trajectory"]}) == 1


@pytest.mark.parametrize("final_force", [4., None, float("nan")])
def test_final_boundary_replaces_stale_driver_force_without_losing_peak(tmp_path, final_force):
    from captain_harness.ports.capability import CapabilityResult
    robot, env = saved_robot(tmp_path, .08)
    direct = DirectProgramming(SimpleNamespace(robot=robot))
    def execute(**kwargs):
        robot._sim_steps += 1
        return CapabilityResult.ok({"observed_force_n":25., "max_observed_force_n":30.,
            "force_decreased":False,"force_still_over_limit":True})
    direct._recover = execute
    robot._eef_force_norm = lambda: 10. if robot.simulation_steps == 0 else final_force
    result = direct.invoke("recover", {})
    v = result.value
    assert v["max_observed_force_n"] == 30.
    assert v["driver_force_summary"]["observed_force_n"] == 25.
    expected = final_force if final_force == 4. else None
    assert v["observed_force_n"] == expected
    assert v["force_decreased"] is (True if expected is not None else None)
    assert v["force_still_over_limit"] is (False if expected is not None else None)
    assert v["final_force_sample"]["force_n"] == expected

"""The compact feedback must retain measured aperture changes during Cartesian motion."""
from dataclasses import replace
import pytest
from test_direct_programming import Robot, State, fake_tools
from captain_harness.adapters.capx.direct_programming import DirectProgramming
from captain_harness.ports.capability import CapabilityResult

@pytest.mark.parametrize("action", ["move_to", "recover"])
def test_driver_pose_feedback_does_not_suppress_measured_gripper_change(action):
    class MeasuredRobot(Robot):
        reads = 0
        _gripper_command_fraction = 0.
        def get_robot_state(self):
            self.reads += 1
            return CapabilityResult.ok(replace(State(),
                gripper_open_fraction=.16082417796079196 if self.simulation_steps==0 else .010923063550265214))
    robot=MeasuredRobot()
    direct=DirectProgramming(fake_tools(robot))
    feedback={"commanded_displacement_m":[.01,0,0],
              "measured_displacement_m":[.009,0,0],
              "blocked_directions":None}
    def execute(**kwargs):
        robot.simulation_steps+=1
        return CapabilityResult.ok({"tracking_feedback":dict(feedback)})
    if action=="move_to":
        direct._move=execute
        args={"position":[.01,0,0],"quaternion":[1,0,0,0]}
    else:
        direct._recover=execute
        args={}
    result=direct.invoke(action,args)
    value=result.value["tracking_feedback"]
    assert value["gripper_before"]==pytest.approx(.16082417796079196)
    assert value["gripper_after"]==pytest.approx(.010923063550265214)
    assert value["measured_gripper_change"]==pytest.approx(-.14990111441052675)
    assert {k:value[k] for k in feedback}==feedback
    assert robot.reads==2 and robot.simulation_steps==1
    assert result.value["object_retention"]=="not_verified"

@pytest.mark.parametrize("command", ["open", "close"])
def test_explicit_gripper_commands_report_change_without_object_inference(command):
    class MeasuredRobot(Robot):
        def get_robot_state(self):
            return CapabilityResult.ok(replace(State(),
                gripper_open_fraction=.5 if self.simulation_steps==0 else (1. if command=="open" else 0.)))
    result=DirectProgramming(fake_tools(MeasuredRobot())).invoke("move_to",{"gripper":command})
    f=result.value["tracking_feedback"]
    assert f["requested_gripper"]==command
    assert f["measured_gripper_change"]==(.5 if command=="open" else -.5)
    assert result.value["object_retention"]=="not_verified"

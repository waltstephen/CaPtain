import numpy as np
import pytest
from captain_harness.adapters.robotics.model import RobotCollisionModel
from captain_harness.adapters.robotics.contact_separation import ContactSeparationChecker


def model():
    xml="""<mujoco><worldbody><body name="base"><body name="arm">
      <joint name="x" type="slide" axis="1 0 0" range="-1 1" limited="true"/>
      <geom name="arm_geometry" type="box" size=".03 .03 .03" pos="0 0 .18"/>
      <body name="eef"><joint name="y" type="slide" axis="0 1 0" range="-1 1" limited="true"/>
      <geom name="hand_geometry" type="box" size=".02 .02 .02"/>
      </body></body></body></worldbody></mujoco>"""
    return RobotCollisionModel(xml,joints=["x","y"],gripper_joints=[],base="base",
        end_effector="eef",gripper_prefix="hand_",angle_unit="radian")


def checker(points,**kwargs):
    return ContactSeparationChecker(model(),[0.,0.],1.,points,.01,**kwargs)


def test_existing_clearance_can_separate_without_becoming_physical_penetration():
    check=checker([[-.021,0,0]])
    assert not check.check([0,0],stage="initial")["collision"]
    assert not check.check([.003,0])["collision"]
    assert not check.check([.02,0])["collision"]
    # Execution has independent progress state: planning's clearance cannot
    # incorrectly reject the still-contacting actual starting configuration.
    assert not check.check([0,0],stage="execution")["collision"]
    assert not check.check([.02,0],stage="execution")["collision"]


def test_inward_motion_new_obstacle_and_reentry_are_rejected():
    inward=checker([[-.025,0,0]])
    result=inward.check([-.001,0])
    assert result["collision"]
    assert result["separation"]["reasons"]["hand_geometry"]["increased_clearance_overlap"]
    new=checker([[-.021,0,0],[.061,0,0]])
    assert not new.check([.015,0])["collision"]
    assert new.check([.042,0])["collision"]
    reentry=checker([[-.021,0,0]])
    assert not reentry.check([.02,0])["collision"]
    assert reentry.check([0,0])["collision"]


def test_existing_penetration_and_arm_contacts_cannot_be_exempted():
    penetration=checker([[-.019,0,0]])
    assert penetration.check([0,0],stage="initial")["collision"]
    assert penetration.check([.2,0])["collision"]
    arm=checker([[0,0,.18]])
    assert arm.check([0,0],stage="initial")["collision"]


def test_execution_slack_does_not_allow_physical_penetration():
    check=checker([[-.021,0,0]])
    assert not check.check([-.0005,0],stage="execution")["collision"]
    assert check.check([-.002,0],stage="execution")["collision"]


def test_selected_contact_margin_does_not_apply_to_other_scene_samples():
    # The first sample is in the3mm selected-target margin; the second is
    # outside that margin but within the10mm ordinary obstacle margin.
    check=checker([[-.021,0,0],[.025,0,0]],target_mask=[True,False])
    assert check.initial_sample_pairs==2
    selected_only=checker([[.025,0,0]],target_mask=[True])
    assert selected_only.initial_sample_pairs==0
    # It cannot become a new target contact under a separation permission.
    assert selected_only.check([.003,0])["collision"]


def test_separation_free_scene_matches_existing_nominal_collision():
    hardware=model();points=np.array([[.3,0,0],[0,.3,.18]])
    check=ContactSeparationChecker(hardware,[0,0],1,points,.01)
    for q in ([0,0],[.1,0],[.275,0]):
        expected=hardware.collision(q,1.,points,margin=.01)
        actual=check.check(q)
        assert actual["collision"]==expected["collision"]
        assert actual["observed_collisions"]==expected["observed_collisions"]


def test_geometry_padding_cannot_omit_a_robot_link():
    with pytest.raises(ValueError,match="every moving"):
        checker([[-.021,0,0]]).check([0,0],extra_margin={"hand_geometry":.003})

def test_payload_proxy_uncertainty_is_bounded_by_public_scene_resolution():
    hardware=model()
    vertices=np.array([[x,y,z] for x in [-.015,.015] for y in [-.015,.015] for z in [-.015,.015]])+[.2,0,0]
    payload=hardware.with_observed_carried_geometry(vertices)
    check=ContactSeparationChecker(payload,[0,0],1.,[[.186,0,0]],.01)
    assert not check.check([0,0],stage="initial")["collision"]
    assert not check.check([.005,0])["collision"]
    assert not check.check([.03,0])["collision"]
    assert check.check([0,0])["collision"]  # Cannot re-enter after clearing.
    deep=ContactSeparationChecker(payload,[0,0],1.,[[.2,0,0]],.01)
    assert deep.check([0,0],stage="initial")["collision"]
    # Hardware has no corresponding uncertainty exception.
    assert checker([[-.019,0,0]]).check([0,0],stage="initial")["collision"]

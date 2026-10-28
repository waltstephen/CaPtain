"""Native staged entry and servo must never dispatch to an OSC controller."""
import numpy as np

from captain_harness.adapters.libero.native_pose_entry import NativePoseEntryMixin
from test_native_backend_contract import ContractRobot


class Robot(NativePoseEntryMixin, ContractRobot):
    pass


def test_staged_pregrasp_keeps_clearance_and_requested_orientation():
    robot = Robot()
    result = robot.align_pregrasp([.08, .03, .02], [1, 0, 0, 0], [0, 0, -1],
                                   approach_distance_m=.03, clearance_m=.04)
    assert result.success and result.value['phase_reached'] == 'pregrasp'
    np.testing.assert_allclose(robot.q[:3], [.08, .03, .05], atol=1e-9)
    # Translation starts only after the hand has reached clearance.
    lateral = next(q for q, _ in robot.actions if abs(q[0]) > 1e-6)
    assert lateral[2] >= .089
    assert all(fraction == 1. for _, fraction in robot.actions)


def test_invalid_alignment_has_no_gripper_or_arm_side_effect():
    robot = Robot()
    result = robot.align_pregrasp([.1, 0, 0], [0, 0, 0, 0], [0, 0, -1])
    assert not result.success and not result.executed and not robot.actions


def test_native_servo_obeys_requested_duration_and_reports_real_residual():
    robot = Robot()
    result = robot.servo_to([.1, 0, 0], [1, 0, 0, 0], max_duration_s=.1, tolerance_m=.001)
    assert not result.success and result.executed
    assert result.value['iterations'] == 2 and result.value['reached'] is False
    assert result.value['final_error_m'] > .09


def test_servo_cannot_report_success_outside_callers_tolerance():
    robot = Robot()
    from captain_harness.ports.capability import CapabilityResult
    robot._native_drive().move = lambda *a, **kw: CapabilityResult.ok({})
    result = robot.servo_to([.0025, 0, 0], [1, 0, 0, 0], max_duration_s=1., tolerance_m=.002)
    assert not result.success and result.value['reached'] is False


def test_force_stop_during_alignment_is_not_reclassified_as_convergence():
    robot = Robot()
    robot.force = lambda: 20. if robot.q[2] > .01 else 5.
    result = robot.align_pregrasp([.05, 0, .01], [1, 0, 0, 0], [0, 0, -1])
    assert not result.success and result.executed
    assert result.value['controller_execution']['stopped_reason'] == 'contact_force'

"""Time advance exposes alignment invalidation without weakening descent gates."""
import pytest
from jsonschema import Draft202012Validator

from captain_harness.adapters.capx.manifest import ToolApiLevel, build_capx_tool_manifest
from captain_harness.ports.capability import CapabilityResult, FailureCategory
from test_tool_backend import _fixture, _invoke


CANDIDATE = {
    "position_m": [.1, .2, .3],
    "quaternion_wxyz": [1., 0., 0., 0.],
    "approach_direction_world": [0., 0., -1.],
}


def _elapsed(duration=.2):
    return {"requested_duration_s": duration, "elapsed_s": duration,
            "start_time_s": 1., "end_time_s": 1. + duration,
            "steps_executed": 4, "gripper_command_fraction": 1.}


def _hold(f):
    return _invoke(f, ToolApiLevel.S3_LOW_LEVEL, "hold_position", {"duration_s": .2},
                   service_aligned=True, dehardcoded=True, v15_protocol=True)


@pytest.mark.parametrize("initially_aligned", [False, True])
def test_wait_reports_alignment_state_and_preserves_time_evidence(tmp_path, initially_aligned):
    f = _fixture(tmp_path, dehardcoded=True, v15_protocol=True)
    if initially_aligned:
        assert f.backend.align_pregrasp(CANDIDATE).success
    raw = _elapsed()
    f.robot.hold_position = lambda duration: CapabilityResult.ok(raw, "time advanced")
    result = _hold(f)
    assert result.success and result.executed and result.message == "time advanced"
    assert {k: result.value[k] for k in raw} == raw
    assert "pregrasp_alignment" not in raw
    state = result.value["pregrasp_alignment"]
    assert state["available"] is False
    assert state["invalidated"] is initially_aligned
    assert state["required_before_descend_and_close"] == "align_pregrasp"
    assert state["reason"] == "hold_position_requested"
    spec = build_capx_tool_manifest(ToolApiLevel.S3_LOW_LEVEL, joint_count=7,
        service_aligned=True, dehardcoded=True, v15_protocol=True).require("hold_position")
    Draft202012Validator(spec.result_schema).validate(result.value)


@pytest.mark.parametrize("executed", [False, True])
def test_failed_wait_reports_invalidation_and_keeps_failure_status(tmp_path, executed):
    f = _fixture(tmp_path, dehardcoded=True, v15_protocol=True)
    assert f.backend.align_pregrasp(CANDIDATE).success
    raw = _elapsed(.2)
    raw.update(elapsed_s=.05 if executed else 0.,
               end_time_s=1.05 if executed else 1.,
               steps_executed=1 if executed else 0)
    f.robot.hold_position = lambda duration: CapabilityResult.failed(
        FailureCategory.TIMEOUT, "time advance incomplete", executed=executed, value=raw)
    result = _hold(f)
    assert not result.success and result.executed is executed
    assert result.failure_category is FailureCategory.TIMEOUT
    assert result.message == "time advance incomplete"
    assert {k: result.value[k] for k in raw} == raw
    assert result.value["pregrasp_alignment"]["invalidated"] is executed
    assert f.backend.descend_and_close().executed is not executed


def test_unavailable_wait_reports_state_without_claiming_execution(tmp_path):
    f = _fixture(tmp_path, dehardcoded=True, v15_protocol=True)
    assert f.backend.align_pregrasp(CANDIDATE).success
    f.robot.hold_position = None
    result = _hold(f)
    assert not result.success and not result.executed
    assert result.failure_category is FailureCategory.NOT_EXECUTED
    assert result.value["pregrasp_alignment"]["invalidated"] is False
    assert result.value["pregrasp_alignment"]["available"] is True
    assert f.backend.descend_and_close().executed


def test_wait_requires_realignment_before_another_physical_descent(tmp_path):
    f = _fixture(tmp_path, dehardcoded=True, v15_protocol=True)
    f.robot.hold_position = lambda duration: CapabilityResult.ok(_elapsed(duration))
    assert f.backend.align_pregrasp(CANDIDATE).success
    result = _hold(f)
    assert result.value["pregrasp_alignment"]["available"] is False
    before = list(f.robot.calls)
    blocked = f.backend.descend_and_close()
    assert not blocked.executed and blocked.value["reason"] == "pregrasp_alignment_required"
    assert f.robot.calls == before
    assert f.backend.align_pregrasp(CANDIDATE).success
    assert f.backend.descend_and_close().executed


@pytest.mark.parametrize("within_tolerance", [True, False])
def test_measured_pose_hold_preserves_alignment_only_when_revalidated(tmp_path, within_tolerance):
    f = _fixture(tmp_path, dehardcoded=True, v15_protocol=True)
    assert f.backend.align_pregrasp(CANDIDATE).success
    raw = {**_elapsed(), "pose_hold": {"reference": "end_effector",
        "position_residual_m": .001 if within_tolerance else .02,
        "orientation_residual_rad": .002, "position_tolerance_m": .004,
        "orientation_tolerance_rad": .05, "within_tolerance": within_tolerance}}
    f.robot.hold_position = lambda duration: CapabilityResult.ok(raw)
    f.robot.pregrasp_pose_status = lambda position, quaternion: CapabilityResult.ok({
        "valid": within_tolerance, "position_residual_m": .001 if within_tolerance else .02,
        "orientation_residual_rad": .002, "position_tolerance_m": .015,
        "orientation_tolerance_rad": .18})
    result = _hold(f)
    state = result.value["pregrasp_alignment"]
    assert state["available"] is within_tolerance
    assert state["invalidated"] is not within_tolerance
    assert f.backend.descend_and_close().executed is within_tolerance


def test_failed_hold_cannot_revalidate_alignment_even_with_small_residual(tmp_path):
    f = _fixture(tmp_path, dehardcoded=True, v15_protocol=True)
    assert f.backend.align_pregrasp(CANDIDATE).success
    f.robot.hold_position = lambda duration: CapabilityResult.failed(
        FailureCategory.TIMEOUT, "incomplete", executed=True,
        value={**_elapsed(), "pose_hold": {"within_tolerance": True}})
    f.robot.pregrasp_pose_status = lambda position, quaternion: CapabilityResult.ok({"valid": True})
    result = _hold(f)
    assert not result.value["pregrasp_alignment"]["available"]
    assert not f.backend.descend_and_close().executed


def test_small_hold_drift_cannot_override_invalid_pregrasp_geometry(tmp_path):
    f = _fixture(tmp_path, dehardcoded=True, v15_protocol=True)
    assert f.backend.align_pregrasp(CANDIDATE).success
    f.robot.hold_position = lambda duration: CapabilityResult.ok({**_elapsed(),
        "pose_hold": {"reference": "end_effector", "within_tolerance": True}})
    f.robot.pregrasp_pose_status = lambda position, quaternion: CapabilityResult.ok({
        "valid": False, "position_residual_m": .02, "orientation_residual_rad": .002,
        "position_tolerance_m": .015, "orientation_tolerance_rad": .18})
    result = _hold(f)
    assert not result.value["pregrasp_alignment"]["available"]
    assert not f.backend.descend_and_close().executed

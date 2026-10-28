from __future__ import annotations

from captain_harness.adapters.capx.manifest import (
    ToolApiLevel,
    build_capx_tool_manifest,
)
from captain_harness.adapters.capx.reference_skills import (
    S3_PICK_REFERENCE,
    S3_PLACE_REFERENCE,
    run_pick_reference,
    run_place_reference,
)


class ScriptedCapabilities:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def __call__(self, name, arguments):
        self.calls.append((name, arguments))
        expected_name, result = self.results.pop(0)
        assert name == expected_name
        return result


def test_s2_pick_reference_executes_only_s3_primitives() -> None:
    call = ScriptedCapabilities(
        [
            ("open_gripper", {}),
            ("approach_and_close", {"object_detected": True}),
            ("lift_probe", {"actual_vertical_displacement_m": 0.1}),
            ("gripper_state", {"object_detected": True}),
        ]
    )
    result = run_pick_reference(
        call,
        {
            "tracking_handle": "artifact://v1/track",
            "position_m": [0.4, 0.0, 0.3],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
    )

    assert result["picked"] is True
    assert tuple(name for name, _ in call.calls) == S3_PICK_REFERENCE


def test_s2_place_reference_executes_only_s3_primitives() -> None:
    call = ScriptedCapabilities(
        [
            (
                "placement_estimate",
                {
                    "relation_estimate": True,
                    "placement_estimate_handle": "artifact://v1/estimate",
                },
            ),
            ("authorize_safe_release", {"safe_to_release": True}),
            ("open_gripper", {}),
        ]
    )
    result = run_place_reference(
        call,
        tracking_handle="artifact://v1/track",
        target_detection_id="artifact://v1/target",
        destination_detection_id="artifact://v1/destination",
        relation_type="on",
    )

    assert result == {"placed": True, "released": True}
    assert tuple(name for name, _ in call.calls) == (
        "placement_estimate",
        "authorize_safe_release",
        "open_gripper",
    )


def test_every_s2_skill_reference_is_expressible_in_s3_manifest() -> None:
    s3 = build_capx_tool_manifest(
        ToolApiLevel.S3_LOW_LEVEL,
        joint_count=7,
        service_aligned=True,
    )

    assert set(S3_PICK_REFERENCE).issubset(s3.names)
    assert set(S3_PLACE_REFERENCE).issubset(s3.names)

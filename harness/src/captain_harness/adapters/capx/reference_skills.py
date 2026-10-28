from __future__ import annotations

from typing import Callable, Mapping


CapabilityCall = Callable[[str, Mapping[str, object]], Mapping[str, object]]

S3_PICK_REFERENCE = (
    "open_gripper",
    "approach_and_close",
    "lift_probe",
    "gripper_state",
)

S3_PLACE_REFERENCE = (
    "placement_estimate",
    "observe",
    "goto_pose",
    "authorize_safe_release",
    "open_gripper",
)


def run_pick_reference(
    call: CapabilityCall,
    candidate: Mapping[str, object],
    *,
    lift_height_m: float = 0.10,
) -> dict[str, object]:
    call("open_gripper", {})
    approach = call(
        "approach_and_close",
        {"candidate": dict(candidate)},
    )
    if approach.get("object_detected") is not True:
        return {
            "picked": False,
            "object_detected": False,
            "tracking_handle": candidate.get("tracking_handle"),
        }
    call("lift_probe", {"height": lift_height_m})
    state = call("gripper_state", {})
    return {
        "picked": state.get("object_detected") is True,
        "object_detected": state.get("object_detected") is True,
        "tracking_handle": candidate.get("tracking_handle"),
    }


def run_place_reference(
    call: CapabilityCall,
    *,
    tracking_handle: str,
    target_detection_id: str,
    destination_detection_id: str,
    relation_type: str,
) -> dict[str, object]:
    estimate = call(
        "placement_estimate",
        {
            "tracking_handle": tracking_handle,
            "target_detection_id": target_detection_id,
            "destination_detection_id": destination_detection_id,
            "relation_type": relation_type,
        },
    )
    if estimate.get("relation_estimate") is not True:
        observation = call("observe", {})
        state = observation["robot_state"]
        current = state["end_effector_position_m"]
        step = estimate["suggested_step_m"]
        target = {
            "position_m": [
                float(current[index]) + float(step[index])
                for index in range(3)
            ],
            "quaternion_wxyz": state["end_effector_quaternion_wxyz"],
        }
        call(
            "goto_pose",
            {
                "position": target["position_m"],
                "quaternion": target["quaternion_wxyz"],
                "z_approach": 0.0,
                "approach_frame": "world",
            },
        )
        return {"placed": False, "released": False}
    authorization = call(
        "authorize_safe_release",
        {
            "tracking_handle": tracking_handle,
            "placement_estimate_handle": estimate[
                "placement_estimate_handle"
            ],
        },
    )
    if authorization.get("safe_to_release") is not True:
        return {"placed": False, "released": False}
    call("open_gripper", {})
    return {"placed": True, "released": True}

from __future__ import annotations

from captain_harness.ports.capability import (
    CapabilityManifest,
    CapabilitySpec,
    SideEffect,
)


def _arguments(properties, required):
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def build_libero_manifest(*, joint_count: int) -> CapabilityManifest:
    number = {"type": "number"}
    vector3 = {"type": "array", "items": number, "minItems": 3, "maxItems": 3}
    quaternion = {"type": "array", "items": number, "minItems": 4, "maxItems": 4}
    joints = {
        "type": "array",
        "items": number,
        "minItems": joint_count,
        "maxItems": joint_count,
    }
    robot_state = {
        "type": "object",
        "required": [
            "joint_positions_rad",
            "end_effector_position_m",
            "end_effector_quaternion_wxyz",
            "gripper_open_fraction",
            "joint_velocity_norm_rad_s",
        ],
        "properties": {
            "joint_positions_rad": joints,
            "end_effector_position_m": vector3,
            "end_effector_quaternion_wxyz": quaternion,
            "gripper_open_fraction": number,
            "joint_velocity_norm_rad_s": {
                "type": "number",
                "minimum": 0,
            },
        },
        "additionalProperties": False,
    }
    definitions = (
        (
            "get_robot_state",
            "Read public proprioception in the robot base frame.",
            {},
            (),
            robot_state,
            {},
            {},
            2.0,
            SideEffect.READ_ONLY,
        ),
        (
            "get_observation_data",
            "Capture current public RGB-D paths, calibration, and proprioception.",
            {},
            (),
            {"type": "object", "required": ["cameras", "robot_state"]},
            {},
            {},
            10.0,
            SideEffect.SENSOR_CAPTURE,
        ),
        (
            "pixel_to_world",
            "Back-project a model-selected normalized image point using current measured depth.",
            {
                "camera": {"type": "string", "enum": ["agentview", "robot0_eye_in_hand"]},
                "x_norm": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "y_norm": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            ("camera", "x_norm", "y_norm"),
            vector3,
            {"x_norm": "normalized_image_coordinate", "y_norm": "normalized_image_coordinate", "result": "metre"},
            {"x_norm": "selected_camera_image", "y_norm": "selected_camera_image", "result": "robot_base"},
            2.0,
            SideEffect.READ_ONLY,
        ),
        (
            "move_to_pose",
            "Move the end effector to a base-frame pose using closed-loop DLS IK.",
            {"position_xyz": vector3, "quaternion_wxyz": quaternion},
            ("position_xyz", "quaternion_wxyz"),
            {"type": "null"},
            {"position_xyz": "metre", "quaternion_wxyz": "unit_quaternion"},
            {"position_xyz": "robot_base", "quaternion_wxyz": "robot_base"},
            120.0,
            SideEffect.ROBOT_MOTION,
        ),
        (
            "move_by",
            "Move the end effector by a base-frame Cartesian delta.",
            {"delta_xyz": vector3},
            ("delta_xyz",),
            {"type": "null"},
            {"delta_xyz": "metre"},
            {"delta_xyz": "robot_base"},
            120.0,
            SideEffect.ROBOT_MOTION,
        ),
        (
            "move_to_joints",
            "Move all arm joints to absolute target positions.",
            {"joints": joints},
            ("joints",),
            {"type": "null"},
            {"joints": "radian"},
            {"joints": "robot_joint_space"},
            120.0,
            SideEffect.ROBOT_MOTION,
        ),
        ("open_gripper", "Open the gripper.", {}, (), {"type": "null"}, {}, {}, 30.0, SideEffect.ACTUATOR),
        ("close_gripper", "Close the gripper.", {}, (), {"type": "null"}, {}, {}, 30.0, SideEffect.ACTUATOR),
        ("home", "Return to the reset arm configuration.", {}, (), {"type": "null"}, {}, {}, 120.0, SideEffect.ROBOT_MOTION),
    )
    specs = tuple(
        CapabilitySpec(
            name=name,
            description=description,
            arguments_schema=_arguments(properties, order),
            result_schema=result_schema,
            units=units,
            coordinate_frames=frames,
            timeout_s=timeout,
            side_effect=side_effect,
            argument_order=order,
        )
        for name, description, properties, order, result_schema, units, frames, timeout, side_effect
        in definitions
    )
    return CapabilityManifest("captain-capabilities/v1", "libero-stage1/v1", specs)

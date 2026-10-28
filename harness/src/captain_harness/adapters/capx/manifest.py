from __future__ import annotations

from enum import Enum
from typing import Mapping

from captain_harness.ports.capability import (
    CapabilityManifest,
    CapabilitySpec,
    ControlSemantic,
    SideEffect,
)


class ToolApiLevel(str, Enum):
    S2_HIGH_LEVEL = "S2"
    S3_LOW_LEVEL = "S3"


def _arguments(
    properties: Mapping[str, object], required: tuple[str, ...]
) -> dict[str, object]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
    }


def build_capx_tool_manifest(
    level: ToolApiLevel | str,
    *,
    joint_count: int,
    public_safe_release: bool = False,
    service_aligned: bool = False,
    dehardcoded: bool = False,
    completion_evidence: bool = False,
    completion_confidence: bool = False,
    v15_protocol: bool = False,
) -> CapabilityManifest:
    level = ToolApiLevel(level)
    if joint_count <= 0:
        raise ValueError("joint_count must be positive")
    if dehardcoded and not service_aligned:
        raise ValueError("dehardcoded manifests require service_aligned=True")
    if completion_evidence and not dehardcoded:
        raise ValueError(
            "completion evidence manifests require dehardcoded=True"
        )
    if completion_confidence and not completion_evidence:
        raise ValueError(
            "completion confidence manifests require completion_evidence=True"
        )
    if service_aligned:
        from .service_aligned_manifest import (
            build_service_aligned_manifest,
        )

        return build_service_aligned_manifest(
            level,
            joint_count=joint_count,
            dehardcoded=dehardcoded,
            completion_evidence=completion_evidence,
            completion_confidence=completion_confidence,
            v15_protocol=v15_protocol,
        )
    number = {"type": "number"}
    vector3 = {"type": "array", "items": number, "minItems": 3, "maxItems": 3}
    quaternion = {"type": "array", "items": number, "minItems": 4, "maxItems": 4}
    joints = {
        "type": "array",
        "items": number,
        "minItems": joint_count,
        "maxItems": joint_count,
    }
    handle = {"type": "string", "pattern": r"^artifact://v1/"}
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
            "gripper_open_fraction": {"type": "number", "minimum": 0, "maximum": 1},
            "joint_velocity_norm_rad_s": {"type": "number", "minimum": 0},
        },
        "additionalProperties": False,
    }
    camera = {
        "type": "object",
        "required": [
            "camera_name",
            "rgb_handle",
            "depth_handle",
            "intrinsics_3x3",
            "extrinsics_4x4",
        ],
        "properties": {
            "camera_name": {"type": "string", "minLength": 1},
            "rgb_handle": handle,
            "depth_handle": {"anyOf": [handle, {"type": "null"}]},
            "intrinsics_3x3": {
                "type": "array",
                "items": number,
                "minItems": 9,
                "maxItems": 9,
            },
            "extrinsics_4x4": {
                "type": "array",
                "items": number,
                "minItems": 16,
                "maxItems": 16,
            },
        },
        "additionalProperties": False,
    }
    observation = {
        "type": "object",
        "required": ["observation_handle", "cameras", "robot_state"],
        "properties": {
            "observation_handle": handle,
            "cameras": {"type": "array", "items": camera, "minItems": 1},
            "robot_state": robot_state,
            "time_s": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        },
        "additionalProperties": False,
    }
    pose = {
        "type": "object",
        "required": ["position", "quaternion"],
        "properties": {"position": vector3, "quaternion": quaternion},
        "additionalProperties": False,
    }
    grasp_pose = {
        **pose,
        "required": [
            "position",
            "quaternion",
            "score",
            "predicted_joint_travel_rad",
            "approach_direction_base",
            "top_down_alignment",
            "grasp_style",
        ],
        "properties": {
            **pose["properties"],
            "score": number,
            "predicted_joint_travel_rad": {
                "type": "number",
                "minimum": 0,
            },
            "approach_direction_base": vector3,
            "top_down_alignment": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "grasp_style": {
                "type": "string",
                "enum": ["auto", "top_down", "side"],
            },
        },
    }
    sampled_grasp_pose = {
        **grasp_pose,
        "required": [
            *grasp_pose["required"],
            "rejected_unreachable_count",
            "object_name",
            "tracking_handle",
            "identity_preserved",
            "candidate_index",
            "candidate_count",
            "candidates",
            "candidate_source",
        ],
        "properties": {
            **grasp_pose["properties"],
            "rejected_unreachable_count": {
                "type": "integer",
                "minimum": 0,
            },
            "object_name": {"type": "string", "minLength": 1},
            "tracking_handle": {
                "anyOf": [handle, {"type": "null"}]
            },
            "identity_preserved": {"type": "boolean"},
            "candidate_index": {"type": "integer", "minimum": 0},
            "candidate_count": {"type": "integer", "minimum": 1},
            "candidates": {
                "type": "array",
                "minItems": 1,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "required": [
                        "candidate_index",
                        "position",
                        "quaternion",
                        "score",
                        "predicted_joint_travel_rad",
                        "approach_direction_base",
                        "top_down_alignment",
                    ],
                    "properties": {
                        "candidate_index": {
                            "type": "integer",
                            "minimum": 0,
                        },
                        "position": vector3,
                        "quaternion": quaternion,
                        "score": number,
                        "predicted_joint_travel_rad": {
                            "type": "number",
                            "minimum": 0,
                        },
                        "approach_direction_base": vector3,
                        "top_down_alignment": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                    "additionalProperties": False,
                },
            },
            "candidate_source": {
                "type": "string",
                "const": "contact_graspnet",
            },
        },
    }
    motion_outcome = {
        "type": "object",
        "required": [
            "phase",
            "executed",
            "converged",
            "convergence_reason",
            "max_joint_residual_rad",
            "l2_joint_residual_rad",
            "position_tolerance_m",
            "orientation_tolerance_rad",
            "requested_cartesian_position_m",
            "requested_cartesian_quaternion_wxyz",
            "controller_target_position_m",
            "final_cartesian_position_m",
            "final_cartesian_quaternion_wxyz",
            "position_residual_m",
            "orientation_residual_rad",
            "final_eef_position_m",
            "final_eef_quaternion_wxyz",
        ],
        "properties": {
            "phase": {"type": "string", "enum": ["clearance", "approach", "target"]},
            "executed": {"type": "boolean"},
            "converged": {"type": "boolean"},
            "convergence_reason": {
                "type": "string",
                "enum": [
                    "within_tolerance",
                    "position_residual_high",
                    "orientation_residual_high",
                    "position_and_orientation_residual_high",
                ],
            },
            "max_joint_residual_rad": {"type": "number", "minimum": 0},
            "l2_joint_residual_rad": {"type": "number", "minimum": 0},
            "position_tolerance_m": {
                "type": "number",
                "exclusiveMinimum": 0,
            },
            "orientation_tolerance_rad": {
                "type": "number",
                "exclusiveMinimum": 0,
            },
            "requested_cartesian_position_m": vector3,
            "requested_cartesian_quaternion_wxyz": quaternion,
            "controller_target_position_m": vector3,
            "final_cartesian_position_m": vector3,
            "final_cartesian_quaternion_wxyz": quaternion,
            "position_residual_m": {"type": "number", "minimum": 0},
            "orientation_residual_rad": {"type": "number", "minimum": 0},
            "final_eef_position_m": vector3,
            "final_eef_quaternion_wxyz": quaternion,
        },
        "additionalProperties": False,
    }
    # Missing proprioception is represented explicitly, never as a zero pose.
    measured_fields = {
        "final_cartesian_position_m": vector3,
        "final_cartesian_quaternion_wxyz": quaternion,
        "final_eef_position_m": vector3,
        "final_eef_quaternion_wxyz": quaternion,
        "position_residual_m": {"type": "number", "minimum": 0},
        "orientation_residual_rad": {"type": "number", "minimum": 0},
    }
    motion_outcome["properties"].update({
        name: {"anyOf": [schema, {"type": "null"}]}
        for name, schema in measured_fields.items()
    })
    motion_outcome["properties"]["convergence_reason"]["enum"].append("pose_measurement_unavailable")
    motion_outcome["properties"].update({
        "pose_measurement": {"type": "object", "x-attempt-feedback": "priority",
            "properties": {"status": {"const": "unavailable"},
                "message": {"type": "string"}, "scope": {"type": "string"}},
            "required": ["status", "message", "scope"], "additionalProperties": False},
        "completed_phases": {"type": "array", "x-attempt-feedback": "priority",
            "items": {"enum": ["clearance", "approach", "target"]}, "uniqueItems": True},
        "failed_phase": {"enum": ["clearance", "approach", "target", None], "x-attempt-feedback": "priority"},
        "phase_executions": {"type": "object", "x-decision-feedback": "on_demand",
            "description": "Immutable per-phase outcomes; the outer executed flag covers the whole request.",
            "properties": {phase: {"type": "object", "properties": {
                "success": {"type": "boolean"}, "executed": {"type": "boolean"},
                "failure_category": {"type": "string"}, "message": {"type": "string"},
                "value": {}, "diagnostics": {"type": ["object", "null"]},
                "attempt_identity": {"type": ["object", "null"]}},
                "required": ["success", "executed", "failure_category", "message", "value", "diagnostics", "attempt_identity"],
                "additionalProperties": False} for phase in ("clearance", "approach", "target")},
            "additionalProperties": False},
    })
    motion_outcome["allOf"] = [{
        "if": {"required": ["pose_measurement"]},
        "then": {"properties": {"converged": {"const": False},
            "convergence_reason": {"const": "pose_measurement_unavailable"},
            **{name: {"type": "null"} for name in measured_fields}}},
        "else": {"properties": measured_fields},
    }]
    common_motion = (
        {
            "position": vector3,
            "quaternion": quaternion,
            "z_approach": {"type": "number", "minimum": 0.0, "maximum": 0.5, "default": 0.0},
        },
        ("position", "quaternion"),
        ("position", "quaternion", "z_approach"),
    )
    entity_track_result = {
        "type": "object",
        "required": [
            "tracking_handle",
            "object_name",
            "pre_centroid_m",
            "pre_bounds_min_m",
            "pre_bounds_max_m",
            "pre_point_count",
        ],
        "properties": {
            "tracking_handle": handle,
            "object_name": {"type": "string", "minLength": 1},
            "pre_centroid_m": vector3,
            "pre_bounds_min_m": vector3,
            "pre_bounds_max_m": vector3,
            "pre_point_count": {"type": "integer", "minimum": 3},
        },
        "additionalProperties": False,
    }
    marked_point_result = {
        "type": "object",
        "required": [
            "point_handle",
            "observation_handle",
            "camera_name",
            "rgb_handle",
            "depth_handle",
            "x_norm",
            "y_norm",
            "pixel_xy",
            "position_m",
            "label",
            "grounding_status",
            "cross_view_consistent",
            "views_used",
            "quality_flags",
            "segmentation_score",
            "mask_area_fraction",
            "depth_continuity",
            "geometry_diagonal_m",
            "semantic_point_distance_px",
            "point_centroid_m",
            "point_bounds_min_m",
            "point_bounds_max_m",
            "point_count",
        ],
        "properties": {
            "point_handle": handle,
            "observation_handle": handle,
            "camera_name": {"type": "string", "minLength": 1},
            "rgb_handle": handle,
            "depth_handle": handle,
            "x_norm": {"type": "number", "minimum": 0, "maximum": 1},
            "y_norm": {"type": "number", "minimum": 0, "maximum": 1},
            "pixel_xy": {
                "type": "array",
                "items": number,
                "minItems": 2,
                "maxItems": 2,
            },
            "position_m": vector3,
            "label": {
                "anyOf": [
                    {"type": "string", "minLength": 1},
                    {"type": "null"},
                ]
            },
            "grounding_status": {
                "type": "string",
                "enum": ["grounded", "grounded_with_warning"],
            },
            "cross_view_consistent": {"type": "boolean"},
            "views_used": {"type": "integer", "minimum": 1},
            "quality_flags": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 8,
            },
            "segmentation_score": number,
            "mask_area_fraction": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "depth_continuity": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "geometry_diagonal_m": {
                "type": "number",
                "minimum": 0,
            },
            "semantic_point_distance_px": {
                "anyOf": [number, {"type": "null"}]
            },
            "point_centroid_m": vector3,
            "point_bounds_min_m": vector3,
            "point_bounds_max_m": vector3,
            "point_count": {"type": "integer", "minimum": 3},
            "detection_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "detection_selected": {"type": "boolean"},
        },
        "additionalProperties": False,
    }
    point_track_result = {
        **entity_track_result,
        "required": [
            *entity_track_result["required"],
            "source_point_handle",
            "source_observation_handle",
            "source_camera_name",
            "source_pixel_xy",
            "segmentation_score",
            "grounding_status",
            "cross_view_consistent",
            "views_used",
        ],
        "properties": {
            **entity_track_result["properties"],
            "source_point_handle": handle,
            "source_observation_handle": handle,
            "source_camera_name": {"type": "string", "minLength": 1},
            "source_pixel_xy": {
                "type": "array",
                "items": number,
                "minItems": 2,
                "maxItems": 2,
            },
            "segmentation_score": number,
            "grounding_status": {
                "type": "string",
                "enum": ["grounded", "grounded_with_warning"],
            },
            "cross_view_consistent": {"type": "boolean"},
            "views_used": {"type": "integer", "minimum": 1},
        },
    }
    attached_mark_result = {
        "type": "object",
        "required": [
            "point_handle",
            "tracking_handle",
            "observation_handle",
            "camera_name",
            "pixel_xy",
            "position_m",
            "predicted_target_position_m",
            "prediction_residual_m",
            "max_prediction_residual_m",
            "depth_continuity",
            "appearance_distance_rgb",
            "grounding_status",
            "quality_flags",
        ],
        "properties": {
            "point_handle": handle,
            "tracking_handle": handle,
            "observation_handle": handle,
            "camera_name": {"type": "string", "minLength": 1},
            "pixel_xy": {
                "type": "array",
                "items": number,
                "minItems": 2,
                "maxItems": 2,
            },
            "position_m": vector3,
            "predicted_target_position_m": vector3,
            "prediction_residual_m": {"type": "number", "minimum": 0},
            "max_prediction_residual_m": {"type": "number", "minimum": 0},
            "depth_continuity": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "appearance_distance_rgb": {"type": "number", "minimum": 0},
            "grounding_status": {
                "type": "string",
                "const": "grounded_attachment",
            },
            "quality_flags": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 8,
            },
        },
        "additionalProperties": False,
    }
    attachment_assertion_result = {
        "type": "object",
        "required": [
            "tracking_handle",
            "verified",
            "reason",
            "object_name",
            "post_object_name",
            "current_point_handle",
            "identity_preserved",
            "pre_centroid_m",
            "post_centroid_m",
            "post_bounds_min_m",
            "post_bounds_max_m",
            "end_effector_position_m",
            "displacement_m",
            "vertical_displacement_m",
            "target_to_eef_distance_m",
            "min_displacement_m",
            "min_vertical_displacement_m",
            "max_target_to_eef_distance_m",
            "gripper_open_fraction",
            "post_point_count",
            "point_count_ratio",
            "extent_ratio",
            "point_scale_consistent",
            "extent_scale_consistent",
            "appearance_distance_rgb",
            "attachment_transform_eef_object",
            "attachment_evidence_basis",
            "lift_evidence_available",
            "lift_requested_height_m",
            "lift_actual_vertical_displacement_m",
            "lift_position_residual_m",
            "lift_object_detected",
            "lift_supports_attachment",
            "lift_contradicts_attachment",
        ],
        "properties": {
            "tracking_handle": handle,
            "verified": {"type": "boolean"},
            "reason": {
                "type": "string",
                "enum": [
                    "verified",
                    "target_did_not_move",
                    "target_was_not_lifted",
                    "target_not_near_end_effector",
                    "tracked_geometry_inconsistent",
                    "tracked_appearance_inconsistent",
                    "closed_on_nothing",
                    "lift_probe_not_confirmed",
                    "lift_probe_contradicted_attachment",
                ],
            },
            "object_name": {"type": "string", "minLength": 1},
            "post_object_name": {"type": "string", "minLength": 1},
            "current_point_handle": {
                "anyOf": [handle, {"type": "null"}]
            },
            "identity_preserved": {"type": "boolean"},
            "pose_valid": {"type": "boolean"},
            "pose_observation_accepted": {"type": "boolean"},
            "pose_guidance": {"type": "string"},
            "pre_centroid_m": vector3,
            "post_centroid_m": vector3,
            "post_bounds_min_m": vector3,
            "post_bounds_max_m": vector3,
            "end_effector_position_m": vector3,
            "displacement_m": {"type": "number", "minimum": 0},
            "vertical_displacement_m": number,
            "target_to_eef_distance_m": {
                "type": "number",
                "minimum": 0,
            },
            "min_displacement_m": {"type": "number", "minimum": 0},
            "min_vertical_displacement_m": number,
            "max_target_to_eef_distance_m": {
                "type": "number",
                "minimum": 0,
            },
            "gripper_open_fraction": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "post_point_count": {"type": "integer", "minimum": 3},
            "point_count_ratio": {"type": "number", "minimum": 0},
            "extent_ratio": {"type": "number", "minimum": 0},
            "point_scale_consistent": {"type": "boolean"},
            "extent_scale_consistent": {"type": "boolean"},
            "appearance_distance_rgb": {
                "anyOf": [number, {"type": "null"}]
            },
            "attachment_transform_eef_object": {
                "anyOf": [
                    {
                        "type": "array",
                        "items": number,
                        "minItems": 16,
                        "maxItems": 16,
                    },
                    {"type": "null"},
                ]
            },
            "attachment_evidence_basis": {
                "type": "string",
                "enum": [
                    "gripper_register_only",
                    "lift_probe_and_gripper_register",
                ],
            },
            "lift_evidence_available": {"type": "boolean"},
            "lift_requested_height_m": {
                "anyOf": [number, {"type": "null"}]
            },
            "lift_actual_vertical_displacement_m": {
                "anyOf": [number, {"type": "null"}]
            },
            "lift_position_residual_m": {
                "anyOf": [
                    {"type": "number", "minimum": 0},
                    {"type": "null"},
                ]
            },
            "lift_object_detected": {
                "anyOf": [
                    {"type": "boolean"},
                    {"type": "null"},
                ]
            },
            "lift_supports_attachment": {
                "anyOf": [
                    {"type": "boolean"},
                    {"type": "null"},
                ]
            },
            "lift_contradicts_attachment": {
                "anyOf": [
                    {"type": "boolean"},
                    {"type": "null"},
                ]
            },
        },
        "additionalProperties": False,
    }
    placement_estimate_result = {
        "type": "object",
        "required": [
            "placement_estimate_handle",
            "tracking_handle",
            "source_observation_handle",
            "observation_handle",
            "target_point_handle",
            "destination_point_handle",
            "relation_type",
            "relation_estimate",
            "identity_consistent",
            "residual_ok",
            "appearance_ok",
            "scale_ok",
            "scale_check_basis",
            "correction_stable",
            "motion_allowed",
            "safe_release_candidate",
            "correction_stage",
            "estimated_target_centroid_m",
            "destination_mark_position_m",
            "destination_centroid_m",
            "destination_bounds_min_m",
            "destination_bounds_max_m",
            "target_visual_residual_m",
            "max_target_visual_residual_m",
            "target_extent_ratio",
            "target_point_count_ratio",
            "appearance_distance_rgb",
            "horizontal_distance_m",
            "release_height_m",
            "footprint_overlap_ratio",
            "recommended_eef_delta_m",
            "recommended_eef_position_m",
        ],
        "properties": {
            "placement_estimate_handle": handle,
            "tracking_handle": handle,
            "source_observation_handle": handle,
            "observation_handle": handle,
            "target_point_handle": handle,
            "destination_point_handle": handle,
            "relation_type": {"type": "string", "enum": ["on", "in"]},
            "relation_estimate": {"type": "boolean"},
            "identity_consistent": {"type": "boolean"},
            "residual_ok": {"type": "boolean"},
            "appearance_ok": {"type": "boolean"},
            "scale_ok": {"type": "boolean"},
            "scale_check_basis": {
                "type": "string",
                "enum": ["metric_extent", "attachment_prediction"],
            },
            "correction_stable": {"type": "boolean"},
            "motion_allowed": {"type": "boolean"},
            "safe_release_candidate": {"type": "boolean"},
            "correction_stage": {
                "type": "string",
                "enum": [
                    "align_xy",
                    "approach_support",
                    "approach_container",
                    "hold_for_release",
                    "relocalize_required",
                ],
            },
            "estimated_target_centroid_m": vector3,
            "destination_mark_position_m": vector3,
            "destination_centroid_m": vector3,
            "destination_bounds_min_m": vector3,
            "destination_bounds_max_m": vector3,
            "target_visual_residual_m": {"type": "number", "minimum": 0},
            "max_target_visual_residual_m": {"type": "number", "minimum": 0},
            "target_extent_ratio": {"type": "number", "minimum": 0},
            "target_point_count_ratio": {"type": "number", "minimum": 0},
            "appearance_distance_rgb": {"type": "number", "minimum": 0},
            "horizontal_distance_m": {"type": "number", "minimum": 0},
            "release_height_m": number,
            "footprint_overlap_ratio": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "recommended_eef_delta_m": vector3,
            "recommended_eef_position_m": vector3,
        },
        "additionalProperties": False,
    }
    safe_release_result = {
        "type": "object",
        "required": [
            "safe_to_release",
            "reason",
            "release_authorization",
            "tracking_handle",
            "placement_estimate_handle",
            "source_observation_handle",
            "observation_handle",
            "attachment_confirmed",
            "physically_safe",
            "semantic_ready",
            "target_visible",
            "destination_visible",
            "horizontal_distance_m",
            "release_height_m",
            "joint_velocity_norm_rad_s",
            "max_safe_joint_velocity_norm_rad_s",
        ],
        "properties": {
            "safe_to_release": {"type": "boolean"},
            "reason": {
                "type": "string",
                "enum": [
                    "safe_release_authorized",
                    "attachment_unverified",
                    "target_not_visually_consistent",
                    "destination_not_visible",
                    "robot_not_stable",
                    "semantic_relation_not_ready",
                ],
            },
            "release_authorization": {
                "anyOf": [handle, {"type": "null"}]
            },
            "tracking_handle": handle,
            "placement_estimate_handle": handle,
            "source_observation_handle": handle,
            "observation_handle": handle,
            "attachment_confirmed": {"type": "boolean"},
            "physically_safe": {"type": "boolean"},
            "semantic_ready": {"type": "boolean"},
            "target_visible": {"type": "boolean"},
            "destination_visible": {"type": "boolean"},
            "horizontal_distance_m": {"type": "number", "minimum": 0},
            "release_height_m": number,
            "joint_velocity_norm_rad_s": {
                "type": "number",
                "minimum": 0,
            },
            "max_safe_joint_velocity_norm_rad_s": {
                "type": "number",
                "minimum": 0,
            },
        },
        "additionalProperties": False,
    }
    placement_assertion_result = {
        "type": "object",
        "required": [
            "placement_ready",
            "reason",
            "placement_authorization",
            "tracking_handle",
            "tracked_object_name",
            "current_target_name",
            "destination_name",
            "relation_type",
            "point_grounded",
            "target_point_handle",
            "destination_point_handle",
            "source_observation_handle",
            "target_centroid_m",
            "target_bounds_min_m",
            "target_bounds_max_m",
            "destination_centroid_m",
            "destination_bounds_min_m",
            "destination_bounds_max_m",
            "end_effector_position_m",
            "target_to_eef_distance_m",
            "max_target_to_eef_distance_m",
            "horizontal_distance_m",
            "max_horizontal_distance_m",
            "footprint_overlap_ratio",
            "support_vertical_gap_m",
            "recommended_eef_delta_m",
            "recommended_eef_position_m",
            "distinct_entities",
            "in_destination_range",
            "xy_ready",
            "height_ready",
            "correction_stage",
            "target_point_count",
            "destination_point_count",
        ],
        "properties": {
            "placement_ready": {"type": "boolean"},
            "reason": {
                "type": "string",
                "enum": [
                    "placement_ready",
                    "target_not_near_end_effector",
                    "target_destination_not_distinct",
                    "target_outside_destination_range",
                    "point_grounding_required",
                ],
            },
            "placement_authorization": {
                "anyOf": [handle, {"type": "null"}]
            },
            "tracking_handle": handle,
            "tracked_object_name": {"type": "string", "minLength": 1},
            "current_target_name": {"type": "string", "minLength": 1},
            "destination_name": {"type": "string", "minLength": 1},
            "relation_type": {"type": "string", "enum": ["on", "in"]},
            "point_grounded": {"type": "boolean"},
            "target_point_handle": {
                "anyOf": [handle, {"type": "null"}]
            },
            "destination_point_handle": {
                "anyOf": [handle, {"type": "null"}]
            },
            "source_observation_handle": {
                "anyOf": [handle, {"type": "null"}]
            },
            "target_centroid_m": vector3,
            "target_bounds_min_m": vector3,
            "target_bounds_max_m": vector3,
            "destination_centroid_m": vector3,
            "destination_bounds_min_m": vector3,
            "destination_bounds_max_m": vector3,
            "end_effector_position_m": vector3,
            "target_to_eef_distance_m": {
                "type": "number",
                "minimum": 0,
            },
            "max_target_to_eef_distance_m": {
                "type": "number",
                "minimum": 0,
            },
            "horizontal_distance_m": {
                "type": "number",
                "minimum": 0,
            },
            "max_horizontal_distance_m": {
                "type": "number",
                "minimum": 0,
            },
            "footprint_overlap_ratio": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "support_vertical_gap_m": {"type": "number"},
            "recommended_eef_delta_m": vector3,
            "recommended_eef_position_m": vector3,
            "distinct_entities": {"type": "boolean"},
            "in_destination_range": {"type": "boolean"},
            "xy_ready": {"type": "boolean"},
            "height_ready": {"type": "boolean"},
            "correction_stage": {
                "type": "string",
                "enum": [
                    "surface_alignment",
                    "lower_to_surface",
                    "transport_xy",
                    "lower_into_container",
                    "ready",
                ],
            },
            "target_point_count": {"type": "integer", "minimum": 3},
            "destination_point_count": {"type": "integer", "minimum": 3},
        },
        "additionalProperties": False,
    }
    spatial_relation_result = {
        "type": "object",
        "required": [
            "relation_ready",
            "reason",
            "target_name",
            "destination_name",
            "relation_type",
            "target_centroid_m",
            "target_bounds_min_m",
            "target_bounds_max_m",
            "destination_centroid_m",
            "destination_bounds_min_m",
            "destination_bounds_max_m",
            "centroid_inside_destination_xy",
            "footprint_inside_destination_xy",
            "footprint_overlap_ratio",
            "support_vertical_gap_m",
            "robust_support_vertical_gap_m",
            "support_height_basis",
            "distinct_entities",
            "target_point_count",
            "destination_point_count",
        ],
        "properties": {
            "relation_ready": {"type": "boolean"},
            "reason": {
                "type": "string",
                "enum": [
                    "relation_ready",
                    "target_destination_not_distinct",
                    "target_centroid_outside_destination_xy",
                    "target_footprint_outside_destination_xy",
                    "target_not_on_destination_support_height",
                    "target_not_inside_destination_height",
                    "point_grounding_required",
                ],
            },
            "target_name": {"type": "string", "minLength": 1},
            "destination_name": {"type": "string", "minLength": 1},
            "relation_type": {"type": "string", "enum": ["on", "in"]},
            "target_centroid_m": vector3,
            "target_bounds_min_m": vector3,
            "target_bounds_max_m": vector3,
            "destination_centroid_m": vector3,
            "destination_bounds_min_m": vector3,
            "destination_bounds_max_m": vector3,
            "centroid_inside_destination_xy": {"type": "boolean"},
            "footprint_inside_destination_xy": {"type": "boolean"},
            "footprint_overlap_ratio": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "support_vertical_gap_m": {"type": "number"},
            "robust_support_vertical_gap_m": {"type": "number"},
            "support_height_basis": {
                "type": "string",
                "enum": [
                    "bounds",
                    "robust_percentile",
                    "bounds_rejected",
                ],
            },
            "distinct_entities": {"type": "boolean"},
            "target_point_count": {"type": "integer", "minimum": 3},
            "destination_point_count": {"type": "integer", "minimum": 3},
        },
        "additionalProperties": False,
    }
    marked_spatial_relation_result = {
        **spatial_relation_result,
        "required": [
            *spatial_relation_result["required"],
            "target_point_handle",
            "destination_point_handle",
            "source_observation_handle",
            "target_mark_position_m",
            "destination_mark_position_m",
            "mark_inside_destination_xy",
            "mark_horizontal_distance_m",
            "mark_vertical_delta_m",
            "strict_geometry_satisfied",
            "strict_geometry_reason",
            "relation_basis",
        ],
        "properties": {
            **spatial_relation_result["properties"],
            "target_point_handle": handle,
            "destination_point_handle": handle,
            "source_observation_handle": handle,
            "target_mark_position_m": vector3,
            "destination_mark_position_m": vector3,
            "mark_inside_destination_xy": {"type": "boolean"},
            "mark_horizontal_distance_m": {
                "type": "number",
                "minimum": 0,
            },
            "mark_vertical_delta_m": {"type": "number"},
            "strict_geometry_satisfied": {"type": "boolean"},
            "strict_geometry_reason": {
                "type": "string",
                "enum": [
                    "relation_ready",
                    "target_destination_not_distinct",
                    "target_centroid_outside_destination_xy",
                    "target_footprint_outside_destination_xy",
                    "target_not_on_destination_support_height",
                    "target_not_inside_destination_height",
                ],
            },
            "relation_basis": {
                "type": "string",
                "const": "point_grounded_marks",
            },
        },
    }
    public_completion_result = {
        **marked_spatial_relation_result,
        "required": [
            *(
                name
                for name in marked_spatial_relation_result["required"]
                if name != "relation_ready"
            ),
            "public_relation_satisfied",
            "mark_centering_required",
            "mark_centered_for_completion",
            "destination_xy_diagonal_m",
            "max_mark_horizontal_distance_m",
            "tracking_handle",
            "released_target_identity_consistent",
            "completion_target_extent_ratio",
            "completion_target_appearance_distance_rgb",
            "target_to_eef_distance_m",
            "min_target_to_eef_distance_m",
            "robot_clear_for_completion",
            "assessment_basis",
        ],
        "properties": {
            **{
                name: value
                for name, value in marked_spatial_relation_result[
                    "properties"
                ].items()
                if name != "relation_ready"
            },
            "reason": {
                "type": "string",
                "enum": [
                    "relation_ready",
                    "target_destination_not_distinct",
                    "target_centroid_outside_destination_xy",
                    "target_footprint_outside_destination_xy",
                    "target_not_on_destination_support_height",
                    "target_not_inside_destination_height",
                    "target_mark_too_far_from_destination_center",
                    "released_target_identity_inconsistent",
                    "robot_not_clear_of_released_target",
                ],
            },
            "public_relation_satisfied": {"type": "boolean"},
            "mark_centering_required": {"type": "boolean"},
            "mark_centered_for_completion": {"type": "boolean"},
            "destination_xy_diagonal_m": {
                "type": "number",
                "minimum": 0,
            },
            "max_mark_horizontal_distance_m": {
                "type": "number",
                "minimum": 0,
                "maximum": 0.10,
            },
            "tracking_handle": handle,
            "released_target_identity_consistent": {
                "type": "boolean",
            },
            "completion_target_extent_ratio": {
                "type": "number",
                "minimum": 0,
            },
            "completion_target_appearance_distance_rgb": {
                "type": "number",
                "minimum": 0,
            },
            "target_to_eef_distance_m": {
                "type": "number",
                "minimum": 0,
            },
            "min_target_to_eef_distance_m": {
                "type": "number",
                "minimum": 0.12,
                "maximum": 0.30,
            },
            "robot_clear_for_completion": {
                "type": "boolean",
            },
            "assessment_basis": {
                "type": "string",
                "const": "public_rgbd_only",
            },
        },
    }
    verification_definitions = (
        ("mark_point", "Validate and back-project one model-selected normalized pixel from immutable public RGB-D. Both x_norm and y_norm are required on every call. The tool checks point-guided mask area, local depth continuity, 3D extent, optional label-point agreement, and cross-view geometry before returning a usable handle. Uncertain clicks fail as perception_uncertain and must be re-observed; no simulator object pose, contact, reward, done flag, or task checker is consulted.", {"observation_handle": handle, "camera_name": {"type": "string", "enum": ["agentview", "robot0_eye_in_hand"]}, "x_norm": {"type": "number", "minimum": 0, "maximum": 1}, "y_norm": {"type": "number", "minimum": 0, "maximum": 1}, "label": {"anyOf": [{"type": "string", "minLength": 1, "maxLength": 200}, {"type": "null"}], "default": None}}, ("observation_handle", "camera_name", "x_norm", "y_norm"), ("observation_handle", "camera_name", "x_norm", "y_norm", "label"), marked_point_result, "ValidatedMarkedPoint", {"x_norm": "normalized_image_coordinate", "y_norm": "normalized_image_coordinate", "pixel_xy": "pixel", "position_m": "metre", "geometry_diagonal_m": "metre"}, {"x_norm": "selected_camera_image", "y_norm": "selected_camera_image", "pixel_xy": "selected_camera_image", "position_m": "robot_base"}, 120.0, SideEffect.PERCEPTION, ('obs = get_observation(); point = mark_point(obs.value["observation_handle"], "agentview", 0.42, 0.61, label="ketchup bottle") if obs.success else None',)),
        ("track_entity_from_mark", "Create an identity-preserving tracking handle by running point-guided segmentation at an immutable marked pixel from the same handled observation. This avoids a second natural-language localization. Pass the returned tracking_handle to sample_grasp_pose and later attachment assertions.", {"point_handle": handle, "label": {"type": "string", "minLength": 1, "maxLength": 200}, "use_multiview": {"type": "boolean", "default": True}}, ("point_handle", "label"), ("point_handle", "label", "use_multiview"), point_track_result, "PointGroundedEntityTrackingHandle", {"pre_centroid_m": "metre", "pre_bounds_min_m": "metre", "pre_bounds_max_m": "metre", "source_pixel_xy": "pixel", "segmentation_score": "dimensionless"}, {"result": "robot_base", "source_pixel_xy": "selected_camera_image"}, 120.0, SideEffect.PERCEPTION, ('track = track_entity_from_mark(point.value["point_handle"], "the clicked ketchup bottle", use_multiview=True) if point.success else None',)),
        ("mark_attached_target", "Validate one model-selected pixel on a verified held target using the frozen public T_eef_object prediction, local RGB appearance, and local depth continuity. This attachment-aware tool does not require a complete SAM mask around the occluded object and cannot be called before public attachment verification.", {"tracking_handle": handle, "observation_handle": handle, "camera_name": {"type": "string", "enum": ["agentview", "robot0_eye_in_hand"]}, "x_norm": {"type": "number", "minimum": 0, "maximum": 1}, "y_norm": {"type": "number", "minimum": 0, "maximum": 1}}, ("tracking_handle", "observation_handle", "camera_name", "x_norm", "y_norm"), ("tracking_handle", "observation_handle", "camera_name", "x_norm", "y_norm"), attached_mark_result, "AttachedTargetMark", {"x_norm": "normalized_image_coordinate", "y_norm": "normalized_image_coordinate", "pixel_xy": "pixel", "position_m": "metre", "predicted_target_position_m": "metre", "prediction_residual_m": "metre", "max_prediction_residual_m": "metre"}, {"x_norm": "selected_camera_image", "y_norm": "selected_camera_image", "pixel_xy": "selected_camera_image", "position_m": "robot_base", "predicted_target_position_m": "robot_base"}, 30.0, SideEffect.PERCEPTION, ('obs = get_observation(); held = mark_attached_target(track.value["tracking_handle"], obs.value["observation_handle"], "robot0_eye_in_hand", 0.50, 0.62) if obs.success else None',)),
        ("track_entity", "Create an episode-local pre-grasp tracking handle from current public RGB-D evidence. Call this before the attachment attempt for the exact relational target. The handle contains only sensor-derived geometry and never hidden environment state.", {"object_name": {"type": "string", "minLength": 1, "maxLength": 200}, "use_multiview": {"type": "boolean", "default": True}}, ("object_name",), ("object_name", "use_multiview"), entity_track_result, "EntityTrackingHandle", {"pre_centroid_m": "metre", "pre_bounds_min_m": "metre", "pre_bounds_max_m": "metre"}, {"result": "robot_base"}, 120.0, SideEffect.PERCEPTION, ('track = track_entity("black bowl next to the plate", use_multiview=True)',)),
        ("assert_entity_attached", "Validate a close-and-lift attempt from the original track and a fresh point on the same visible target. A matching lift_probe from the current attachment attempt is primary public evidence together with the gripper object-detection register; lift evidence is never shared across tracking handles. Appearance, point-cloud scale, displacement, and end-effector proximity remain identity diagnostics. A verified result reports possession evidence. Only pose_valid permits pose-dependent transport; contradictory visual geometry never replaces a valid transform. If pose_valid is false, observe and mark the held target and assert again.", {"tracking_handle": handle, "post_object_name": {"anyOf": [{"type": "string", "minLength": 1, "maxLength": 200}, {"type": "null"}], "default": None}, "use_multiview": {"type": "boolean", "default": True}, "current_point_handle": {"anyOf": [handle, {"type": "null"}], "default": None}}, ("tracking_handle",), ("tracking_handle", "post_object_name", "use_multiview", "current_point_handle"), attachment_assertion_result, "AttachmentAssertion", {"pre_centroid_m": "metre", "post_centroid_m": "metre", "end_effector_position_m": "metre", "displacement_m": "metre", "vertical_displacement_m": "metre", "target_to_eef_distance_m": "metre", "lift_requested_height_m": "metre", "lift_actual_vertical_displacement_m": "metre", "lift_position_residual_m": "metre"}, {"result": "robot_base"}, 120.0, SideEffect.PERCEPTION, ('obs = get_observation(); held = mark_point(obs.value["observation_handle"], "agentview", 0.42, 0.40, label="same held target"); attached = assert_entity_attached(track.value["tracking_handle"], current_point_handle=held.value["point_handle"], use_multiview=True) if held.success else None',)),
        ("estimate_placement_relation", "Measure the held target relative to a candidate destination using only public RGB-D and robot state. Identity, correction direction, and flip count are diagnostics. The full measured correction is returned without a host-authored gain or distance cap. This tool never authorizes release and never judges benchmark completion.", {"tracking_handle": handle, "target_point_handle": handle, "destination_point_handle": handle, "relation_type": {"type": "string", "enum": ["on", "in"], "default": "on"}}, ("tracking_handle", "target_point_handle", "destination_point_handle"), ("tracking_handle", "target_point_handle", "destination_point_handle", "relation_type"), placement_estimate_result, "PlacementRelationEstimate", {"estimated_target_centroid_m": "metre", "destination_mark_position_m": "metre", "target_visual_residual_m": "metre", "horizontal_distance_m": "metre", "release_height_m": "metre", "recommended_eef_delta_m": "metre", "recommended_eef_position_m": "metre"}, {"result": "robot_base", "recommended_eef_delta_m": "robot_base", "recommended_eef_position_m": "robot_base"}, 120.0, SideEffect.PERCEPTION, ('obs = get_observation(); held = mark_point(obs.value["observation_handle"], "agentview", 0.42, 0.40, label="held target"); destination = mark_point(obs.value["observation_handle"], "agentview", 0.68, 0.55, label="destination"); estimate = estimate_placement_relation(track.value["tracking_handle"], held.value["point_handle"], destination.value["point_handle"], relation_type="on") if held.success and destination.success else None',)),
        ("authorize_safe_release", "Report verified attachment identity, target/destination visibility, current release geometry, semantic readiness, and low robot velocity. Horizontal distance and release height are diagnostics rather than host-authored rejection thresholds. It does not decide whether the task is complete and never reads reward, done, ready, or a task checker.", {"tracking_handle": handle, "placement_estimate_handle": handle}, ("tracking_handle", "placement_estimate_handle"), ("tracking_handle", "placement_estimate_handle"), safe_release_result, "SafeReleaseAuthorization", {"horizontal_distance_m": "metre", "release_height_m": "metre", "joint_velocity_norm_rad_s": "radian_per_second", "max_safe_joint_velocity_norm_rad_s": "radian_per_second"}, {"result": "robot_base"}, 30.0, SideEffect.COMPUTE, ('release_gate = authorize_safe_release(track.value["tracking_handle"], estimate.value["placement_estimate_handle"]) if estimate.success else None',)),
        ("assert_placement_ready", "Authorize release using only fresh public RGB-D, calibration, and robot state. Corrections are receding-horizon and must be executed one reflected action at a time. V7 callers must pass target_point_handle and destination_point_handle from the same fresh observation; only point-grounded results can authorize release. For relation_type='on', correction_stage='surface_alignment' aligns XY, then 'lower_to_surface' lowers the target bottom near the support before ready. For relation_type='in', 'transport_xy' centers the target footprint inside the container, then 'lower_into_container' lowers its centroid below the rim. A ready result grants an episode-local placement authorization consumed by open_gripper.", {"tracking_handle": handle, "current_target_name": {"type": "string", "minLength": 1, "maxLength": 200}, "destination_name": {"type": "string", "minLength": 1, "maxLength": 200}, "use_multiview": {"type": "boolean", "default": True}, "relation_type": {"type": "string", "enum": ["on", "in"], "default": "on"}, "target_point_handle": {"anyOf": [handle, {"type": "null"}], "default": None}, "destination_point_handle": {"anyOf": [handle, {"type": "null"}], "default": None}}, ("tracking_handle", "current_target_name", "destination_name"), ("tracking_handle", "current_target_name", "destination_name", "use_multiview", "relation_type", "target_point_handle", "destination_point_handle"), placement_assertion_result, "PlacementAssertion", {"target_centroid_m": "metre", "destination_centroid_m": "metre", "end_effector_position_m": "metre", "target_to_eef_distance_m": "metre", "horizontal_distance_m": "metre", "max_horizontal_distance_m": "metre", "recommended_eef_delta_m": "metre", "recommended_eef_position_m": "metre"}, {"result": "robot_base", "recommended_eef_delta_m": "robot_base", "recommended_eef_position_m": "robot_base"}, 120.0, SideEffect.PERCEPTION, ('obs = get_observation(); target = mark_point(obs.value["observation_handle"], "agentview", 0.42, 0.61, label="held target"); destination = mark_point(obs.value["observation_handle"], "agentview", 0.68, 0.55, label="destination"); ready = assert_placement_ready(track.value["tracking_handle"], "held target", "destination", relation_type="on", target_point_handle=target.value["point_handle"], destination_point_handle=destination.value["point_handle"])',)),
        ("assert_spatial_relation", "Verify a released or pushed target relation from one fresh public RGB-D capture. Set relation_type='on' for support placement: target centroid must be inside and at least 83 percent of target XY footprint must overlap the destination. Set relation_type='in' for containers: target centroid must be inside destination XY and lowered below the container rim.", {"target_name": {"type": "string", "minLength": 1, "maxLength": 200}, "destination_name": {"type": "string", "minLength": 1, "maxLength": 200}, "use_multiview": {"type": "boolean", "default": True}, "relation_type": {"type": "string", "enum": ["on", "in"], "default": "on"}}, ("target_name", "destination_name"), ("target_name", "destination_name", "use_multiview", "relation_type"), spatial_relation_result, "SpatialRelationAssertion", {"target_centroid_m": "metre", "target_bounds_min_m": "metre", "target_bounds_max_m": "metre", "destination_centroid_m": "metre", "destination_bounds_min_m": "metre", "destination_bounds_max_m": "metre", "support_vertical_gap_m": "metre"}, {"result": "robot_base"}, 120.0, SideEffect.PERCEPTION, ('relation = assert_spatial_relation("the released ketchup", "the basket", use_multiview=True, relation_type="in")',)),
        ("assert_spatial_relation_from_marks", "Verify a target/destination relation using two model-selected pixels from the same immutable fresh RGB-D observation. Each mark is independently point-segmented; no natural-language re-localization is performed. This is the required point-grounded completion assertion. For relation_type='on', the raw target mark and segmented target centroid must lie inside destination XY, at least 83 percent of the segmented target XY footprint must overlap the destination, and the raw target/destination mark height difference must match support placement. Centroid-only overlap is not sufficient. For 'in', the marked target and segmented centroid must be inside container XY and below its rim.", {"target_point_handle": handle, "destination_point_handle": handle, "target_label": {"type": "string", "minLength": 1, "maxLength": 200}, "destination_label": {"type": "string", "minLength": 1, "maxLength": 200}, "relation_type": {"type": "string", "enum": ["on", "in"], "default": "on"}}, ("target_point_handle", "destination_point_handle", "target_label", "destination_label"), ("target_point_handle", "destination_point_handle", "target_label", "destination_label", "relation_type"), marked_spatial_relation_result, "PointGroundedSpatialRelationAssertion", {"target_centroid_m": "metre", "target_bounds_min_m": "metre", "target_bounds_max_m": "metre", "destination_centroid_m": "metre", "destination_bounds_min_m": "metre", "destination_bounds_max_m": "metre", "target_mark_position_m": "metre", "destination_mark_position_m": "metre", "mark_horizontal_distance_m": "metre", "mark_vertical_delta_m": "metre"}, {"result": "robot_base"}, 120.0, SideEffect.PERCEPTION, ('obs = get_observation(); target = mark_point(obs.value["observation_handle"], "agentview", 0.42, 0.61, label="released bowl"); destination = mark_point(obs.value["observation_handle"], "agentview", 0.68, 0.55, label="plate"); relation = assert_spatial_relation_from_marks(target.value["point_handle"], destination.value["point_handle"], "released bowl", "plate", relation_type="on")',)),
        ("assert_task_completion_public", "Assess the released target/destination relation from two validated marks in the same fresh public RGB-D observation. Pass action_truth_state.released_tracking_handle so the target mark is checked against immutable pre-grasp metric extent and appearance. If robot_clear_for_completion is false, move the arm away and repeat from a fresh observation before submitting completion. For relation_type='on', click near the geometric center of both the target and destination: their public 3D mark separation must fit a destination-size-aware limit capped at 10 cm, in addition to footprint and robust support-height checks. This rejects identity-switched, occluded, edge, or wrong-burner placements. The centering check is not applied to relation_type='in', where a destination click may legitimately lie on a container rim. This is an online public estimate only; it never reads or predicts the environment's reward, done flag, ready state, or task checker. A satisfied result may trigger no-action visual verification and trajectory freeze.", {"target_point_handle": handle, "destination_point_handle": handle, "target_label": {"type": "string", "minLength": 1, "maxLength": 200}, "destination_label": {"type": "string", "minLength": 1, "maxLength": 200}, "tracking_handle": handle, "relation_type": {"type": "string", "enum": ["on", "in"], "default": "on"}}, ("target_point_handle", "destination_point_handle", "target_label", "destination_label", "tracking_handle"), ("target_point_handle", "destination_point_handle", "target_label", "destination_label", "tracking_handle", "relation_type"), public_completion_result, "PublicTaskCompletionAssessment", {"target_centroid_m": "metre", "destination_centroid_m": "metre", "target_mark_position_m": "metre", "destination_mark_position_m": "metre", "mark_horizontal_distance_m": "metre", "mark_vertical_delta_m": "metre", "destination_xy_diagonal_m": "metre", "max_mark_horizontal_distance_m": "metre", "target_to_eef_distance_m": "metre"}, {"result": "robot_base"}, 120.0, SideEffect.PERCEPTION, ('complete = assert_task_completion_public(target.value["point_handle"], destination.value["point_handle"], "released target", "destination", release_identity.payload["tracking_handle"], relation_type="on") if target.success and destination.success else None',)),
    )
    if level is ToolApiLevel.S2_HIGH_LEVEL:
        definitions = (
            ("get_observation", "Capture public RGB-D, calibration, and robot state; large arrays remain behind handles.", {}, (), (), observation, "ObservationHandle", {}, {}, 10.0, SideEffect.SENSOR_CAPTURE, ('obs = get_observation(); cameras = obs.value["cameras"] if obs.success else []',)),
            *verification_definitions,
            ("get_object_pose", "Estimate an object pose from current camera evidence with SAM3 and Molmo fallback.", {"object_name": {"type": "string", "minLength": 1, "maxLength": 200}, "use_multiview": {"type": "boolean", "default": True}}, ("object_name",), ("object_name", "use_multiview"), pose, "ObjectPose", {"position": "metre", "quaternion": "unit_quaternion"}, {"result": "robot_base"}, 120.0, SideEffect.PERCEPTION, ('pose = get_object_pose("black bowl", use_multiview=True); xyz = pose.value["position"] if pose.success else None',)),
            ("sample_grasp_pose", "Estimate grasp poses from sensor-derived masks and point clouds, discard candidates without feasible target IK, then rank them joint-travel-first using the explicit high-level grasp_style. The model supplies any approach distance when it executes the returned pose. Pass the tracking_handle returned by track_entity_from_mark to bind planning to exactly the same sensor-derived target and prevent a second language query from switching to a distractor. Start with candidate_index=0, then inspect candidate_count and candidates in the successful result. Never request candidate_index >= candidate_count; when only index 0 exists, re-ground or change grasp_style instead of guessing index 1. top_down requires a genuinely downward approach; side requires a genuinely lateral approach and never falls back to a near-vertical candidate. Use auto when either geometry is acceptable. Keep object_name purely semantic; do not insert grasp instructions into the segmentation description.", {"object_name": {"type": "string", "minLength": 1, "maxLength": 200}, "use_multiview": {"type": "boolean", "default": True}, "grasp_style": {"type": "string", "enum": ["auto", "top_down", "side"], "default": "auto"}, "tracking_handle": {"anyOf": [handle, {"type": "null"}], "default": None}, "candidate_index": {"type": "integer", "minimum": 0, "maximum": 14, "default": 0}}, ("object_name",), ("object_name", "use_multiview", "grasp_style", "tracking_handle", "candidate_index"), sampled_grasp_pose, "ReachableGraspPose", {"position": "metre", "quaternion": "unit_quaternion", "score": "dimensionless", "predicted_joint_travel_rad": "radian", "approach_direction_base": "unit_vector", "top_down_alignment": "dimensionless"}, {"result": "robot_base", "approach_direction_base": "robot_base"}, 180.0, SideEffect.PERCEPTION, ('point = mark_point(obs.value["observation_handle"], "agentview", 0.42, 0.61, label="black bowl next to the ramekin"); track = track_entity_from_mark(point.value["point_handle"], "black bowl next to the ramekin", use_multiview=True) if point.success else None; grasp = sample_grasp_pose(track.value["object_name"], use_multiview=True, grasp_style="auto", tracking_handle=track.value["tracking_handle"], candidate_index=0) if track is not None and track.success else None',)),
            ("goto_pose", "Use PyRoKi reachability and Cartesian control to reach a base-frame CaP-X tool pose. The pinned TCP offset maps the public pose to the PyRoKi panda_hand target; execution truth maps the observed LIBERO gripper0_eef pose back into the same public tool frame. With positive z_approach, one call performs the approach along local negative tool Z and then the final target motion. Position and orientation residuals are checked against the returned calibrated tolerances; either residual exceeding its tolerance makes the action fail with converged=false.", common_motion[0], common_motion[1], common_motion[2], motion_outcome, "MotionOutcome", {"position": "metre", "quaternion": "unit_quaternion", "z_approach": "metre", "max_joint_residual_rad": "radian", "l2_joint_residual_rad": "radian", "position_tolerance_m": "metre", "orientation_tolerance_rad": "radian", "position_residual_m": "metre", "orientation_residual_rad": "radian"}, {"position": "robot_base", "quaternion": "robot_base", "requested_cartesian_position_m": "robot_base", "requested_cartesian_quaternion_wxyz": "robot_base", "controller_target_position_m": "robot_base", "final_cartesian_position_m": "robot_base", "final_cartesian_quaternion_wxyz": "robot_base", "final_eef_position_m": "robot_base", "final_eef_quaternion_wxyz": "robot_base"}, 180.0, SideEffect.ROBOT_MOTION, ("goto_pose(grasp.value['position'], grasp.value['quaternion'], z_approach=0.08)",)),
            ("open_gripper", "Open the gripper.", {}, (), (), {"type": "null"}, "None", {}, {}, 30.0, SideEffect.ACTUATOR, ("open_gripper()",)),
            ("close_gripper", "Close the gripper.", {}, (), (), {"type": "null"}, "None", {}, {}, 30.0, SideEffect.ACTUATOR, ("close_gripper()",)),
            ("home", "Return to the reset arm configuration.", {}, (), (), {"type": "null"}, "None", {}, {}, 120.0, SideEffect.ROBOT_MOTION, ("home()",)),
        )
        provider = "capx-s2-nonprivileged/v6"
    else:
        mask_result = {
            "type": "object",
            "required": ["masks"],
            "properties": {
                "masks": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "required": ["mask_handle", "score", "box", "label"],
                        "properties": {
                            "mask_handle": handle,
                            "score": number,
                            "box": {"anyOf": [{"type": "array", "items": number, "minItems": 4, "maxItems": 4}, {"type": "null"}]},
                            "label": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        },
                        "additionalProperties": False,
                    },
                }
            },
            "additionalProperties": False,
        }
        cloud_result = {
            "type": "object",
            "required": ["full_point_cloud_handle", "object_point_cloud_handle", "full_point_count", "object_point_count", "object_centroid_m", "object_bounds_min_m", "object_bounds_max_m"],
            "properties": {
                "full_point_cloud_handle": handle,
                "object_point_cloud_handle": handle,
                "full_point_count": {"type": "integer", "minimum": 1},
                "object_point_count": {"type": "integer", "minimum": 1},
                "object_centroid_m": vector3,
                "object_bounds_min_m": vector3,
                "object_bounds_max_m": vector3,
            },
            "additionalProperties": False,
        }
        definitions = (
            ("get_observation", "Capture public RGB-D, calibration, and robot state into episode-local handles.", {}, (), (), observation, "ObservationHandle", {}, {}, 10.0, SideEffect.SENSOR_CAPTURE, ('obs = get_observation(); cameras = obs.value["cameras"] if obs.success else []',)),
            *verification_definitions,
            ("segment_text", "Segment a handled RGB image with a SAM3 text prompt and return at most eight highest-score mask handles. Use this directly for a unique visual class; when multiple instances satisfy the class or the task specifies a relation, prefer point_molmo with the full relational description followed by segment_point instead of constructing a point cloud for every text mask.", {"rgb_handle": handle, "text_prompt": {"type": "string", "minLength": 1, "maxLength": 200}}, ("rgb_handle", "text_prompt"), ("rgb_handle", "text_prompt"), mask_result, "MaskHandles", {}, {"rgb_handle": "camera_image"}, 120.0, SideEffect.PERCEPTION, ("masks = segment_text(rgb_handle, 'black bowl')",)),
            ("segment_point", "Segment a handled RGB image around a sensor-grounded pixel point and return at most eight highest-score mask handles. For relational or repeated-instance targets, pass point_molmo's point from the same RGB handle so target identity is preserved.", {"rgb_handle": handle, "point_xy": {"type": "array", "items": {"type": "number", "minimum": 0}, "minItems": 2, "maxItems": 2}}, ("rgb_handle", "point_xy"), ("rgb_handle", "point_xy"), mask_result, "MaskHandles", {"point_xy": "pixel"}, {"point_xy": "camera_image"}, 120.0, SideEffect.PERCEPTION, ('point = point_molmo(rgb_handle, "black bowl next to the plate"); masks = segment_point(rgb_handle, point.value["point_xy"])',)),
            ("point_molmo", "Ask Molmo for one pixel point in a handled RGB image. This is the preferred first localization step when the target is relationally described or one of several same-class instances; include the complete relation in text_prompt and pass the result to segment_point on the same image.", {"rgb_handle": handle, "text_prompt": {"type": "string", "minLength": 1, "maxLength": 200}}, ("rgb_handle", "text_prompt"), ("rgb_handle", "text_prompt"), {"type": "object", "required": ["point_xy"], "properties": {"point_xy": {"type": "array", "items": number, "minItems": 2, "maxItems": 2}}, "additionalProperties": False}, "ImagePoint", {"point_xy": "pixel"}, {"point_xy": "camera_image"}, 120.0, SideEffect.PERCEPTION, ('point = point_molmo(rgb_handle, "black bowl next to the plate")',)),
            ("construct_point_cloud", "Construct full and segmented base-frame point clouds from handled RGB-D evidence and return compact object geometry. For placement or recovery, re-localize both target and destination from the same fresh observation and compute correction from their returned robot-base centroids; never translate image-direction guesses into base-frame motion. After close and lift, verify that the target moved from its pre-grasp centroid and remains spatially consistent with the current end effector before transfer. Treat near-identical target/destination centroids and bounds as an invalid same-entity localization.", {"observation_handle": handle, "mask_handles": {"type": "array", "items": handle, "minItems": 1, "maxItems": 8}, "use_multiview": {"type": "boolean", "default": True}}, ("observation_handle", "mask_handles"), ("observation_handle", "mask_handles", "use_multiview"), cloud_result, "PointCloudHandles", {"object_centroid_m": "metre", "object_bounds_min_m": "metre", "object_bounds_max_m": "metre"}, {"result": "robot_base"}, 30.0, SideEffect.COMPUTE, ('clouds = construct_point_cloud(obs.value["observation_handle"], [masks.value["masks"][0]["mask_handle"]], use_multiview=True); center = clouds.value["object_centroid_m"]',)),
            ("plan_grasp_candidates", "Plan Contact-GraspNet candidates, discard candidates without feasible target IK, and rank them with the explicit grasp_style. The model supplies any approach distance when it executes a returned pose. Pass the point-grounded tracking_handle so planning reuses one cached sensor-derived target. If the service returns no usable candidate, the capability fails and returns feedback to the model; it never fabricates a geometric pose.", {"full_point_cloud_handle": handle, "object_point_cloud_handle": handle, "top_k": {"type": "integer", "minimum": 1, "maximum": 64, "default": 15}, "grasp_style": {"type": "string", "enum": ["auto", "top_down", "side"], "default": "auto"}, "tracking_handle": {"anyOf": [handle, {"type": "null"}], "default": None}}, ("full_point_cloud_handle", "object_point_cloud_handle"), ("full_point_cloud_handle", "object_point_cloud_handle", "top_k", "grasp_style", "tracking_handle"), {"type": "object", "required": ["grasp_set_handle", "candidate_count", "best_score", "best_top_down_alignment", "grasp_style", "rejected_unreachable_count", "tracking_handle", "identity_preserved", "candidate_source"], "properties": {"grasp_set_handle": handle, "candidate_count": {"type": "integer", "minimum": 1}, "best_score": number, "best_top_down_alignment": {"type": "number", "minimum": 0, "maximum": 1}, "grasp_style": {"type": "string", "enum": ["auto", "top_down", "side"]}, "rejected_unreachable_count": {"type": "integer", "minimum": 0}, "tracking_handle": {"anyOf": [handle, {"type": "null"}]}, "identity_preserved": {"type": "boolean"}, "candidate_source": {"type": "string", "const": "contact_graspnet"}}, "additionalProperties": False}, "GraspCandidateHandle", {"best_score": "dimensionless", "best_top_down_alignment": "dimensionless"}, {"grasp_set_handle": "robot_base"}, 180.0, SideEffect.PERCEPTION, ('grasps = plan_grasp_candidates(clouds.value["full_point_cloud_handle"], clouds.value["object_point_cloud_handle"], top_k=15, grasp_style="top_down", tracking_handle=track.value["tracking_handle"])',)),
            ("get_grasp_candidate", "Read one ranked pose from an episode-local set. Start with index 0 and inspect more only when execution evidence justifies recovery. The result includes the base-frame approach direction and top-down alignment used by the ranking policy.", {"grasp_set_handle": handle, "index": {"type": "integer", "minimum": 0, "maximum": 63, "default": 0}}, ("grasp_set_handle",), ("grasp_set_handle", "index"), grasp_pose, "GraspPose", {"position": "metre", "quaternion": "unit_quaternion", "score": "dimensionless", "approach_direction_base": "unit_vector", "top_down_alignment": "dimensionless"}, {"grasp_set_handle": "robot_base", "result": "robot_base", "approach_direction_base": "robot_base"}, 10.0, SideEffect.READ_ONLY, ('candidate = get_grasp_candidate(grasps.value["grasp_set_handle"], index=0); joints = solve_ik(candidate.value["position"], candidate.value["quaternion"])',)),
            ("solve_ik", "Solve a base-frame tool pose with the external PyRoKi service using the same local TCP convention as goto_pose.", {"position": vector3, "quaternion": quaternion}, ("position", "quaternion"), ("position", "quaternion"), joints, "JointVector", {"position": "metre", "quaternion": "unit_quaternion", "result": "radian"}, {"position": "robot_base", "quaternion": "robot_base", "result": "robot_joint_space"}, 120.0, SideEffect.COMPUTE, ("joints = solve_ik([0.4, 0.0, 0.3], [0, 1, 0, 0])",)),
            ("move_to_joints", "Execute a JOINT_POSITION target.", {"joints": joints}, ("joints",), ("joints",), {"type": "null"}, "None", {"joints": "radian"}, {"joints": "robot_joint_space"}, 120.0, SideEffect.ROBOT_MOTION, ("move_to_joints(joints.value)",)),
            ("goto_pose", "Solve PyRoKi reachability and execute a base-frame CaP-X tool pose. The pinned TCP offset maps the public pose to the PyRoKi panda_hand target; execution truth maps the observed LIBERO gripper0_eef pose back into the same public tool frame. With positive z_approach, one call performs the approach along local negative tool Z and then the final target motion. Position and orientation residuals are checked against the returned calibrated tolerances; either residual exceeding its tolerance makes the action fail with converged=false.", common_motion[0], common_motion[1], common_motion[2], motion_outcome, "MotionOutcome", {"position": "metre", "quaternion": "unit_quaternion", "z_approach": "metre", "max_joint_residual_rad": "radian", "l2_joint_residual_rad": "radian", "position_tolerance_m": "metre", "orientation_tolerance_rad": "radian", "position_residual_m": "metre", "orientation_residual_rad": "radian"}, {"position": "robot_base", "quaternion": "robot_base", "requested_cartesian_position_m": "robot_base", "requested_cartesian_quaternion_wxyz": "robot_base", "controller_target_position_m": "robot_base", "final_cartesian_position_m": "robot_base", "final_cartesian_quaternion_wxyz": "robot_base", "final_eef_position_m": "robot_base", "final_eef_quaternion_wxyz": "robot_base"}, 180.0, SideEffect.ROBOT_MOTION, ("goto_pose([0.4, 0, 0.3], [0, 1, 0, 0], z_approach=0.08)",)),
            ("open_gripper", "Open the gripper.", {}, (), (), {"type": "null"}, "None", {}, {}, 30.0, SideEffect.ACTUATOR, ("open_gripper()",)),
            ("close_gripper", "Close the gripper.", {}, (), (), {"type": "null"}, "None", {}, {}, 30.0, SideEffect.ACTUATOR, ("close_gripper()",)),
            ("home", "Return to the reset arm configuration.", {}, (), (), {"type": "null"}, "None", {}, {}, 120.0, SideEffect.ROBOT_MOTION, ("home()",)),
        )
        provider = "capx-s3-composition/v5"

    if public_safe_release:
        deprecated_for_v8 = {
            "track_entity",
            "assert_placement_ready",
            "assert_spatial_relation",
            "assert_spatial_relation_from_marks",
            "get_object_pose",
        }
        definitions = tuple(
            definition
            for definition in definitions
            if definition[0] not in deprecated_for_v8
        )
        provider = (
            "capx-s2-public-safe/v8"
            if level is ToolApiLevel.S2_HIGH_LEVEL
            else "capx-s3-public-safe/v8"
        )
    else:
        v8_only = {
            "estimate_placement_relation",
            "authorize_safe_release",
            "assert_task_completion_public",
            "mark_attached_target",
        }
        definitions = tuple(
            definition
            for definition in definitions
            if definition[0] not in v8_only
        )

    control_semantics = {
        "goto_pose": ControlSemantic.CARTESIAN_MOTION,
        "move_to_joints": ControlSemantic.MOTION,
        "home": ControlSemantic.MOTION,
        "open_gripper": ControlSemantic.ATTACHMENT_RELEASE,
        "close_gripper": ControlSemantic.ATTACHMENT_ATTEMPT,
        "assert_entity_attached": ControlSemantic.ATTACHMENT_ASSERTION,
        "assert_placement_ready": ControlSemantic.PLACEMENT_ASSERTION,
        "estimate_placement_relation": ControlSemantic.PLACEMENT_ESTIMATE,
        "authorize_safe_release": ControlSemantic.SAFE_RELEASE_AUTHORIZATION,
        "assert_spatial_relation": ControlSemantic.COMPLETION_ASSERTION,
        "assert_spatial_relation_from_marks": ControlSemantic.COMPLETION_ASSERTION,
        "assert_task_completion_public": ControlSemantic.COMPLETION_ASSERTION,
        "sample_grasp_pose": ControlSemantic.GRASP_PLANNING,
        "plan_grasp_candidates": ControlSemantic.GRASP_PLANNING,
    }
    specs = tuple(
        CapabilitySpec(
            name=name,
            description=description,
            arguments_schema=_arguments(properties, required),
            result_schema=result_schema,
            units=units,
            coordinate_frames=frames,
            timeout_s=timeout_s,
            side_effect=side_effect,
            argument_order=argument_order,
            return_type=return_type,
            examples=examples,
            control_semantic=control_semantics.get(name),
        )
        for (
            name,
            description,
            properties,
            required,
            argument_order,
            result_schema,
            return_type,
            units,
            frames,
            timeout_s,
            side_effect,
            examples,
        ) in definitions
    )
    return CapabilityManifest(
        (
            "captain-capabilities/v8"
            if public_safe_release
            else "captain-capabilities/v6"
        ),
        provider,
        specs,
    )

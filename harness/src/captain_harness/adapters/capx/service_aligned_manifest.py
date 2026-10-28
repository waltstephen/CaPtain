from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from captain_harness.ports.capability import (
    CapabilityManifest,
    CapabilitySpec,
    ControlSemantic,
    SideEffect,
)

from .manifest import ToolApiLevel, build_capx_tool_manifest


def _arguments(
    properties: Mapping[str, object],
    required: tuple[str, ...],
) -> dict[str, object]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
    }


def _spec(
    name: str,
    description: str,
    properties: Mapping[str, object],
    required: tuple[str, ...],
    argument_order: tuple[str, ...],
    result_schema: Mapping[str, object],
    return_type: str,
    units: Mapping[str, str],
    frames: Mapping[str, str],
    timeout_s: float,
    side_effect: SideEffect,
    example: str,
    real_robot_equivalent: str,
    control_semantic: ControlSemantic | None = None,
) -> CapabilitySpec:
    return CapabilitySpec(
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
        examples=(example,),
        control_semantic=control_semantic,
        real_robot_equivalent=real_robot_equivalent,
    )


def build_service_aligned_manifest(
    level: ToolApiLevel,
    *,
    joint_count: int,
    dehardcoded: bool = False,
    completion_evidence: bool = False,
    completion_confidence: bool = False,
    v15_protocol: bool = False,
) -> CapabilityManifest:
    if completion_evidence and not dehardcoded:
        raise ValueError(
            "completion evidence requires a de-hardcoded manifest"
        )
    if completion_confidence and not completion_evidence:
        raise ValueError(
            "completion confidence requires completion evidence"
        )
    number = {"type": "number"}
    nullable_number = {"anyOf": [number, {"type": "null"}]}
    vector2 = {
        "type": "array",
        "items": number,
        "minItems": 2,
        "maxItems": 2,
    }
    vector3 = {
        "type": "array",
        "items": number,
        "minItems": 3,
        "maxItems": 3,
    }
    quaternion = {
        "type": "array",
        "items": number,
        "minItems": 4,
        "maxItems": 4,
    }
    joints = {
        "type": "array",
        "items": number,
        "minItems": joint_count,
        "maxItems": joint_count,
    }
    handle = {"type": "string", "pattern": r"^artifact://v1/"}
    pose = {
        "type": "object",
        "required": ["position_m", "quaternion_wxyz"],
        "properties": {
            "position_m": vector3,
            "quaternion_wxyz": quaternion,
        },
        "additionalProperties": True,
    }
    gripper_profile = {
        "type": "object",
        "required": [
            "profile_id",
            "max_width_m",
            "fingertip_depth_m",
        ],
        "properties": {
            "profile_id": {"type": "string", "minLength": 1},
            "max_width_m": {"type": "number", "exclusiveMinimum": 0},
            "fingertip_depth_m": {
                "type": "number",
                "exclusiveMinimum": 0,
            },
        },
        "additionalProperties": False,
    }
    candidate = {
        "type": "object",
        "required": [
            "candidate_index",
            "position_m",
            "quaternion_wxyz",
            "model_score",
            "predicted_grasp_width_m",
            "width_feasible",
            "approach_direction_world",
            "reachable",
            "predicted_joint_travel_rad",
            "backend",
            "tracking_handle",
            "detection_id",
        ],
        "properties": {
            "candidate_index": {"type": "integer", "minimum": 0},
            "position_m": vector3,
            "quaternion_wxyz": quaternion,
            "model_score": number,
            "predicted_grasp_width_m": {
                "type": "number",
                "minimum": 0,
            },
            "width_feasible": {"const": True},
            "approach_direction_world": vector3,
            "reachable": {"const": True},
            "predicted_joint_travel_rad": {
                "type": "number",
                "minimum": 0,
            },
            "backend": {
                "type": "string",
                "enum": ["contact_graspnet", "graspgenx"],
            },
            "tracking_handle": handle,
            "detection_id": handle,
        },
        "additionalProperties": False,
    }
    if dehardcoded:
        candidate["required"] = [
            "candidate_index",
            "position_m",
            "quaternion_wxyz",
            "model_score",
            "predicted_grasp_width_m",
            "width_feasible",
            "width_known",
            "approach_direction_world",
            "reachable",
            "ik_feasible",
            "predicted_joint_travel_rad",
            "top_down_alignment",
            "style_preference_score",
            "quality_flags",
            "backend_candidate",
            "backend",
            "tracking_handle",
            "detection_id",
        ]
        candidate["properties"] = {
            **candidate["properties"],
            "predicted_grasp_width_m": nullable_number,
            "width_feasible": {"type": "boolean"},
            "width_known": {"type": "boolean"},
            "reachable": {"type": "boolean"},
            "ik_feasible": {"type": "boolean"},
            "predicted_joint_travel_rad": nullable_number,
            "top_down_alignment": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "style_preference_score": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "quality_flags": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            },
            "backend_candidate": {
                "anyOf": [
                    {
                        "type": "object",
                        "additionalProperties": True,
                    },
                    {"type": "null"},
                ]
            },
            "backend": {
                "type": "string",
                "enum": [
                    "contact_graspnet",
                    "anygrasp",
                    "graspgenx",
                    *(
                        ["mask_topdown_fallback"]
                        if v15_protocol
                        else []
                    ),
                ],
            },
        }
        if v15_protocol:
            candidate["required"] = [
                *candidate["required"],
                "convergence_risk",
            ]
            candidate["properties"] = {
                **candidate["properties"],
                "convergence_risk": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                },
                "target_contact_evidence": {"type": "object", "additionalProperties": True},
                "grasp_handle": handle,
                "source_observation_handle": handle,
                "request_style_match": {
                    "type": "string", "enum": ["matched", "mismatched", "not_requested"],
                },
                "feasibility_evidence": {
                    "type": "object",
                    "required": ["endpoint_ik", "width", "motion_path"],
                    "properties": {
                        "endpoint_ik": {"enum": ["solution_returned", "unavailable"], "type": "string"},
                        "width": {"enum": ["unknown", "within_reported_limits", "outside_reported_limits"], "type": "string"},
                        "motion_path": {"const": "not_checked", "type": "string"},
                    },
                    "additionalProperties": False,
                },
            }
    detection = {
        "type": "object",
        "required": [
            "detection_id",
            "camera_name",
            "label",
            "score",
            "box",
            "centroid_m",
            "extent_m",
            "backend",
            "selected",
        ],
        "properties": {
            "detection_id": handle,
            "camera_name": {"type": "string", "minLength": 1},
            "label": {"type": "string", "minLength": 1},
            "score": number,
            "box": {
                "anyOf": [
                    {
                        "type": "array",
                        "items": number,
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    {"type": "null"},
                ]
            },
            "centroid_m": vector3,
            "extent_m": vector3,
            "backend": {
                "type": "string",
                "enum": ["sam3_text", "molmo_point_sam3"],
            },
            "selected": {"type": "boolean"},
        },
        "additionalProperties": False,
    }
    legacy = build_capx_tool_manifest(
        ToolApiLevel.S3_LOW_LEVEL,
        joint_count=joint_count,
        public_safe_release=True,
        service_aligned=False,
    )
    legacy_by_name = {
        capability.name: capability
        for capability in legacy.capabilities
    }
    equivalents = {
        "mark_point": "Intel RealSense deprojection plus point-guided segmentation",
        "mark_attached_target": "FoundationPose or SAM2 tracked target update",
        "assert_entity_attached": "Robotiq OBJ register plus wrist-camera identity check",
        "authorize_safe_release": "collision and release safety monitor",
        "assert_task_completion_public": "task-facing public perception verifier",
        "open_gripper": "Franka gripper open command",
        "close_gripper": "Franka gripper close command",
        "home": "robot joint or Cartesian home controller",
    }
    retained = tuple(
        replace(
            legacy_by_name[name],
            real_robot_equivalent=equivalents[name],
            control_semantic=(
                ControlSemantic.ATTACHMENT_CONTINUITY
                if name == "mark_attached_target"
                else legacy_by_name[name].control_semantic
            ),
        )
        for name in (
            "mark_point",
            "mark_attached_target",
            "assert_entity_attached",
            "authorize_safe_release",
            "assert_task_completion_public",
        )
    )
    if dehardcoded:
        adjusted_retained = []
        for capability in retained:
            if capability.name == "authorize_safe_release":
                result_schema = dict(capability.result_schema)
                properties = {
                    key: value
                    for key, value in result_schema["properties"].items()
                    if key
                    not in {
                        "max_safe_horizontal_distance_m",
                        "max_safe_release_height_m",
                    }
                }
                reason = dict(properties["reason"])
                reason["enum"] = [
                    "release_recommended",
                    "robot_not_stable",
                    "semantic_relation_not_ready",
                ]
                properties["reason"] = reason
                properties["release_recommended"] = {
                    "type": "boolean"
                }
                properties["quality_flags"] = {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                }
                result_schema["properties"] = properties
                result_schema["required"] = [
                    name
                    for name in result_schema["required"]
                    if name
                    not in {
                        "max_safe_horizontal_distance_m",
                        "max_safe_release_height_m",
                    }
                ] + [
                    "release_recommended",
                    "quality_flags",
                ]
                adjusted_retained.append(
                    replace(
                        capability,
                        description=(
                            "Report release stability, semantic readiness, "
                            "geometry, and skipped-evidence quality flags. "
                            "V10 treats the result as advisory provenance."
                        ),
                        result_schema=result_schema,
                    )
                )
                continue
            if (
                capability.name
                == "assert_task_completion_public"
                and completion_evidence
            ):
                completion_evidence_schema = {
                    "type": "object",
                    "required": [
                        "evidence",
                        "quality_flags",
                        "confidence",
                        "recommendation",
                        "assessment_basis",
                        "relation_type",
                        "source_observation_handle",
                        "tracking_handle",
                        "target_point_handle",
                        "destination_point_handle",
                    ],
                    "properties": {
                        "evidence": {
                            "type": "object",
                            "required": [
                                "xy_overlap_ratio",
                                "centroid_inside_destination_xy",
                                "mark_inside_destination_xy",
                                "mark_horizontal_distance_m",
                                "max_mark_horizontal_distance_m",
                                "mark_vertical_delta_m",
                                "robust_vertical_gap_m",
                                "support_gap_bounds_m",
                                "gripper_released",
                                "gripper_object_detected",
                                "gripper_open_fraction",
                                "target_static_after_release_m",
                                "max_static_displacement_m",
                                "target_destination_distinct",
                                "eef_clearance_m",
                                "min_eef_clearance_m",
                                "appearance_distance_rgb",
                                "extent_ratio",
                                "identity_confidence",
                            ],
                            "properties": {
                                "xy_overlap_ratio": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "centroid_inside_destination_xy": {
                                    "type": "boolean"
                                },
                                "mark_inside_destination_xy": {
                                    "type": "boolean"
                                },
                                "mark_horizontal_distance_m": {
                                    "type": "number",
                                    "minimum": 0,
                                },
                                "max_mark_horizontal_distance_m": {
                                    "type": "number",
                                    "minimum": 0,
                                },
                                "mark_vertical_delta_m": number,
                                "robust_vertical_gap_m": number,
                                "support_gap_bounds_m": number,
                                "gripper_released": {
                                    "type": "boolean"
                                },
                                "gripper_object_detected": {
                                    "type": "boolean"
                                },
                                "gripper_open_fraction": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "target_static_after_release_m": {
                                    "anyOf": [
                                        {
                                            "type": "number",
                                            "minimum": 0,
                                        },
                                        {"type": "null"},
                                    ]
                                },
                                "max_static_displacement_m": {
                                    "type": "number",
                                    "minimum": 0,
                                },
                                "target_destination_distinct": {
                                    "type": "boolean"
                                },
                                "eef_clearance_m": {
                                    "type": "number",
                                    "minimum": 0,
                                },
                                "min_eef_clearance_m": {
                                    "type": "number",
                                    "minimum": 0,
                                },
                                "appearance_distance_rgb": {
                                    "type": "number",
                                    "minimum": 0,
                                },
                                "extent_ratio": {
                                    "type": "number",
                                    "minimum": 0,
                                },
                                "identity_confidence": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                            },
                            "additionalProperties": False,
                        },
                        "quality_flags": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "minLength": 1,
                            },
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "recommendation": {
                            "type": "string",
                            "enum": [
                                "satisfied",
                                "insufficient",
                            ],
                        },
                        "assessment_basis": {
                            "const": "public_rgbd_only"
                        },
                        "relation_type": {
                            "type": "string",
                            "enum": ["on", "in"],
                        },
                        "source_observation_handle": handle,
                        "tracking_handle": handle,
                        "target_point_handle": handle,
                        "destination_point_handle": handle,
                    },
                    "additionalProperties": False,
                }
                if v15_protocol:
                    completion_evidence_schema["properties"]["verification_requirements"] = {
                        "type": "array", "items": {"type": "object",
                            "required": ["name", "satisfied", "status", "next_observation"],
                            "properties": {"name": {"type": "string"},
                                "satisfied": {"type": "boolean"},
                                "status": {"enum": ["supported", "contradicted", "insufficient"]},
                                "next_observation": {"type": "string"}},
                            "additionalProperties": False}}
                    completion_evidence_schema["properties"]["relation_evidence"] = {
                        "type": "object",
                        "required": ["status", "scope", "proxy_reason", "limitation"],
                        "properties": {
                            "status": {"enum": ["supported", "contradicted", "insufficient"]},
                            "scope": {"type": "string"}, "proxy_reason": {"type": "string"},
                            "limitation": {"type": "string"},
                            **{key: {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
                               for key in ("target_centroid_m", "target_bounds_min_m", "target_bounds_max_m",
                                           "destination_bounds_min_m", "destination_bounds_max_m")},
                        }, "additionalProperties": False,
                    }
                    completion_evidence_schema["properties"]["verification_questions"] = {
                        "type": "array", "minItems": 1, "items": {
                            "type": "object", "required": ["name", "prompt"],
                            "properties": {"name": {"type": "string"}, "prompt": {"type": "string"}},
                            "additionalProperties": False,
                        },
                    }
                    evidence_schema = completion_evidence_schema[
                        "properties"
                    ]["evidence"]
                    evidence_schema["required"] = [
                        *evidence_schema["required"],
                        "identity_provenance_required",
                        "identity_provenance_confirmed",
                        "identity_chain_confirmed",
                        "identity_anchor_observation_handle",
                        "identity_source_detection_id",
                        "identity_point_handle",
                        "distractor_detection_id",
                        "distractor_stationary",
                        "distractor_position_residual_m",
                        "instance_reference_distances_m",
                        "instance_selection_margin_ratio",
                    ]
                    evidence_schema["properties"].update({
                        "observation_time_s": nullable_number,
                        "comparison_time_delta_s": nullable_number,
                        "comparison_observation_time_s": nullable_number,
                        "comparison_source_observation_handle": {"anyOf": [handle, {"type": "null"}]},
                        "temporal_comparison_available": {"type": "boolean"},
                        "stability_measurement_basis": {"type": "string"},
                    })
                    evidence_schema["properties"].update(
                        {
                            "identity_provenance_required": {
                                "type": "boolean"
                            },
                            "identity_provenance_confirmed": {
                                "type": "boolean"
                            },
                            "identity_chain_confirmed": {
                                "type": "boolean"
                            },
                            "identity_anchor_observation_handle": {
                                "anyOf": [handle, {"type": "null"}]
                            },
                            "identity_source_detection_id": {
                                "anyOf": [handle, {"type": "null"}]
                            },
                            "identity_point_handle": {
                                "anyOf": [handle, {"type": "null"}]
                            },
                            "distractor_detection_id": {
                                "anyOf": [handle, {"type": "null"}]
                            },
                            "distractor_stationary": {
                                "anyOf": [
                                    {"type": "boolean"},
                                    {"type": "null"},
                                ]
                            },
                            "distractor_position_residual_m": {
                                "anyOf": [
                                    {
                                        "type": "number",
                                        "minimum": 0,
                                    },
                                    {"type": "null"},
                                ]
                            },
                            "instance_reference_distances_m": {
                                "anyOf": [
                                    {
                                        "type": "object",
                                        "additionalProperties": {
                                            "type": "number",
                                            "minimum": 0,
                                        },
                                    },
                                    {"type": "null"},
                                ]
                            },
                            "instance_selection_margin_ratio": {
                                "anyOf": [
                                    {
                                        "type": "number",
                                        "minimum": 0,
                                    },
                                    {"type": "null"},
                                ]
                            },
                        }
                    )
                if completion_confidence:
                    completion_evidence_schema["required"] = [
                        *completion_evidence_schema["required"],
                        "confidence_components",
                        "confidence_penalty",
                    ]
                    completion_evidence_schema["properties"].update(
                        {
                            "confidence_components": {
                                "type": "object",
                                "additionalProperties": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                            },
                            "confidence_penalty": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                        }
                    )
                adjusted_retained.append(
                    replace(
                        capability,
                        description=(
                            "Measure post-release public RGB-D completion "
                            "evidence without returning a binary task verdict. "
                            + (
                                "V12 reports repaired confidence components "
                                "and multiplicative quality penalties, led by "
                                "robust vertical gap. "
                                if completion_confidence
                                else (
                                    "The model weighs overlap, mark height, "
                                    "gripper release, target stability, "
                                    "identity diagnostics, and EEF clearance "
                                    "before proposing COMPLETE_CANDIDATE; "
                                    "bounds support gap is the "
                                    "lowest-reliability signal and never a "
                                    "hard gate."
                                )
                            )
                        ),
                        result_schema=completion_evidence_schema,
                        return_type="PublicTaskCompletionEvidence",
                        units={
                            "xy_overlap_ratio": "dimensionless",
                            "mark_horizontal_distance_m": "metre",
                            "mark_vertical_delta_m": "metre",
                            "robust_vertical_gap_m": "metre",
                            "support_gap_bounds_m": "metre",
                            "target_static_after_release_m": "metre",
                            "eef_clearance_m": "metre",
                        },
                    )
                )
                continue
            if capability.name != "assert_entity_attached":
                adjusted_retained.append(capability)
                continue
            result_schema = dict(capability.result_schema)
            properties = dict(result_schema["properties"])
            reason = dict(properties["reason"])
            if not completion_evidence:
                reason["enum"] = [
                    *reason["enum"],
                    "verified_with_identity_warning",
                ]
            properties["reason"] = reason
            properties["quality_flags"] = {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            }
            if completion_evidence:
                properties["gripper_object_detected"] = {
                    "type": "boolean"
                }
                properties["identity_confidence"] = {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                }
            result_schema["properties"] = properties
            result_schema["required"] = [
                *result_schema["required"],
                "quality_flags",
                *(
                    (
                        "gripper_object_detected",
                        "identity_confidence",
                    )
                    if completion_evidence
                    else ()
                ),
            ]
            adjusted_retained.append(
                replace(
                    capability,
                    description=(
                        "Use the matching lift_probe from the current tracking "
                        "handle and attachment attempt together with the "
                        "gripper object-detection register. A confirmed lift "
                        "is primary public attachment evidence; public "
                        "appearance and point-cloud geometry remain identity "
                        "diagnostics and never cross tracking handles."
                        if completion_evidence
                        else capability.description
                    ),
                    result_schema=result_schema,
                )
            )
        retained = tuple(adjusted_retained)
    observation_schema = legacy_by_name["get_observation"].result_schema
    if v15_protocol:
        import copy
        observation_schema = copy.deepcopy(observation_schema)
        state_properties = observation_schema["properties"]["robot_state"]["properties"]
        state_properties.update({
            "end_effector_pose_reference": {"type": "string", "const": "end_effector"},
            "tool_position_m": vector3, "tool_quaternion_wxyz": quaternion,
            "tool_pose_reference": {"type": "string", "const": "tool"},
        })
    point_schema = {
        **legacy_by_name["mark_point"].result_schema,
        "required": [
            *legacy_by_name["mark_point"].result_schema["required"],
            "prompt",
        ],
        "properties": {
            **legacy_by_name["mark_point"].result_schema["properties"],
            "prompt": {"type": "string", "minLength": 1},
        },
    }
    grasp_result = {
        "type": "object",
        "required": [
            "candidates",
            "candidate_count",
            "rejected",
            "object_extent_m",
            "object_centroid_m",
            "gripper_profile",
            "grasp_style",
            "tracking_handle",
            "detection_id",
            "identity_preserved",
        ],
        "properties": {
            "candidates": {
                "type": "array",
                "items": candidate,
                "minItems": 1,
                "maxItems": 32,
            },
            "candidate_count": {"type": "integer", "minimum": 1},
            "rejected": {
                "type": "object",
                "required": [
                    "width_infeasible",
                    "unreachable",
                    "alignment",
                ],
                "properties": {
                    "width_infeasible": {"type": "integer", "minimum": 0},
                    "unreachable": {"type": "integer", "minimum": 0},
                    "alignment": {"type": "integer", "minimum": 0},
                },
                "additionalProperties": False,
            },
            "object_extent_m": vector3,
            "object_centroid_m": vector3,
            "gripper_profile": gripper_profile,
            "grasp_style": {
                "type": "string",
                "enum": ["auto", "top_down", "side"],
            },
            "tracking_handle": handle,
            "detection_id": handle,
            "identity_preserved": {"const": True},
        },
        "additionalProperties": False,
    }
    if dehardcoded:
        # top_k has no fixed upper bound in this profile. Keep the output
        # contract consistent when the caller requests more than 32 options.
        grasp_result["properties"]["candidates"].pop("maxItems", None)
        grasp_result["required"] = [
            *grasp_result["required"],
            "returned",
            "filtered",
            "backend",
            "depth_source",
            "depth_quality",
            "quality_summary",
            "episode_ledger",
        ]
        grasp_result["properties"] = {
            **grasp_result["properties"],
            "candidate_count": {"type": "integer", "minimum": 1},
            "returned": {"type": "integer", "minimum": 1},
            "filtered": {"const": 0},
            "backend": {
                "type": "string",
                "enum": [
                    "contact_graspnet",
                    "anygrasp",
                    "graspgenx",
                    *(
                        ["mask_topdown_fallback"]
                        if v15_protocol
                        else []
                    ),
                ],
            },
            "depth_source": {
                "type": "string",
                "enum": ["sensor", "enhanced"],
            },
            "depth_quality": {
                "type": "object",
                "additionalProperties": True,
            },
            "quality_summary": {
                "type": "object",
                "additionalProperties": {
                    "type": "integer",
                    "minimum": 0,
                },
            },
            "episode_ledger": {
                "type": "object",
                "additionalProperties": True,
            },
        }
        if v15_protocol:
            grasp_result["required"] += ["selection_feedback", "requires_decision", "reason"]
            grasp_result["properties"].update({
                "requires_decision": {"type": "boolean"},
                "reason": {"type": "string", "enum": ["no_explicit_style_requested",
                    "requested_style_available", "matching_candidates_outside_top_k", "requested_style_unavailable"]},
                "selection_feedback": {
                    "type": "object",
                    "required": ["requested_style", "considered_count", "matching_count",
                        "returned_matching_count", "matching_candidate_indices", "alternative_candidate_indices",
                        "source_observation_handle", "style_reference", "guidance"],
                    "properties": {
                        "requested_style": {"type": "string", "enum": ["auto", "top_down", "side"]},
                        **{name: {"type": "integer", "minimum": 0} for name in
                           ("considered_count", "matching_count", "returned_matching_count")},
                        **{name: {"type": "array", "items": {"type": "integer", "minimum": 0}} for name in
                           ("matching_candidate_indices", "alternative_candidate_indices")},
                        "source_observation_handle": handle,
                        "style_reference": {"type": "object", "required": ["frame", "up_axis",
                            "min_top_down_alignment", "max_side_vertical_alignment"],
                            "properties": {"frame": {"type": "string"}, "up_axis": vector3,
                                "min_top_down_alignment": number, "max_side_vertical_alignment": number},
                            "additionalProperties": False},
                        "guidance": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            })
    cloud_result = {
        "type": "object",
        "required": [
            "full_point_cloud_handle",
            "object_point_cloud_handle",
            "full_point_count",
            "object_point_count",
            "object_centroid_m",
            "object_bounds_min_m",
            "object_bounds_max_m",
            "detection_id",
            "views_used",
            "cross_view_consistent",
        ],
        "properties": {
            "full_point_cloud_handle": handle,
            "object_point_cloud_handle": handle,
            "full_point_count": {"type": "integer", "minimum": 1},
            "object_point_count": {"type": "integer", "minimum": 3},
            "object_centroid_m": vector3,
            "object_bounds_min_m": vector3,
            "object_bounds_max_m": vector3,
            "detection_id": handle,
            "views_used": {"type": "integer", "minimum": 1},
            "cross_view_consistent": {"type": "boolean"},
        },
        "additionalProperties": False,
    }
    placement_result = {
        "type": "object",
        "required": [
            "placement_estimate_handle",
            "tracking_handle",
            "source_observation_handle",
            "observation_handle",
            "target_detection_id",
            "destination_detection_id",
            "relation_type",
            "relation_estimate",
            "correction_stage",
            "identity_consistent",
            "error_vector_m",
            "distance_m",
            "max_safe_step_m",
            "suggested_step_m",
            "estimated_steps_remaining",
            "measurement_confidence",
            "correction_flip_count",
            "estimated_target_centroid_m",
            "destination_centroid_m",
            "horizontal_distance_m",
            "release_height_m",
            "footprint_overlap_ratio",
        ],
        "properties": {
            "placement_estimate_handle": handle,
            "tracking_handle": handle,
            "source_observation_handle": handle,
            "observation_handle": handle,
            "target_detection_id": handle,
            "destination_detection_id": handle,
            "relation_type": {"type": "string", "enum": ["on", "in"]},
            "relation_estimate": {"type": "boolean"},
            "correction_stage": {
                "type": "string",
                "enum": [
                    "align_xy",
                    "approach_support",
                    "approach_container",
                    "hold_for_release",
                ],
            },
            "identity_consistent": {"type": "boolean"},
            "error_vector_m": vector3,
            "distance_m": {"type": "number", "minimum": 0},
            "max_safe_step_m": {"type": "number", "exclusiveMinimum": 0},
            "suggested_step_m": vector3,
            "estimated_steps_remaining": {"type": "integer", "minimum": 0},
            "measurement_confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "correction_flip_count": {"type": "integer", "minimum": 0},
            "estimated_target_centroid_m": vector3,
            "destination_centroid_m": vector3,
            "horizontal_distance_m": {"type": "number", "minimum": 0},
            "release_height_m": number,
            "footprint_overlap_ratio": {
                "type": "number",
                "minimum": 0,
            },
        },
        "additionalProperties": False,
    }
    if dehardcoded:
        placement_result["required"] = [
            name
            for name in placement_result["required"]
            if name != "max_safe_step_m"
        ] + [
            "attachment_verified",
            "quality_flags",
            "requested_destination_detection_id",
            "alignment_recovered",
            "alignment_reason",
            "alignment_residual_m",
            "distance_to_nearest_obstacle_m",
            "placement_backend",
            "selected_grasp_backend",
            "selected_placement_candidate_index",
            "reference_tolerance_m",
            "selected_placement_pose",
            "raw_selected_placement_pose",
            "placement_pose_basis",
            "geometry_refinement_m",
            "raw_to_selected_orientation_delta_rad",
            "predicted_target_centroid_m",
            "predicted_support_gap_m",
            "predicted_footprint_overlap_ratio",
            "current_tool_position_m",
            "placement_candidates",
            "depth_quality",
            *(("anyplace_failure",) if v15_protocol else ()),
        ]
        placement_result["properties"] = {
            key: value
            for key, value in placement_result["properties"].items()
            if key != "max_safe_step_m"
        }
        placement_result["properties"].update(
            {
                "attachment_verified": {"type": "boolean"},
                "quality_flags": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "requested_destination_detection_id": handle,
                "alignment_recovered": {"type": "boolean"},
                "alignment_reason": {
                    "anyOf": [
                        {"type": "string", "minLength": 1},
                        {"type": "null"},
                    ]
                },
                "alignment_residual_m": nullable_number,
                "distance_to_nearest_obstacle_m": nullable_number,
                "placement_backend": (
                    {
                        "type": "string",
                        "enum": [
                            "anyplace",
                            "container_mouth_fallback",
                            "public_support_geometry",
                        ],
                    }
                    if v15_protocol
                    else {"const": "anyplace"}
                ),
                **(
                    {
                        "anyplace_failure": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "null"},
                            ]
                        }
                    }
                    if v15_protocol
                    else {}
                ),
                "selected_grasp_backend": {
                    "type": "string",
                    "enum": [
                        "contact_graspnet",
                        "anygrasp",
                        "graspgenx",
                        *(
                            ["mask_topdown_fallback"]
                            if v15_protocol
                            else []
                        ),
                    ],
                },
                "selected_placement_candidate_index": {
                    "type": "integer",
                    "minimum": 0,
                },
                "reference_tolerance_m": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                },
                "selected_placement_pose": pose,
                "raw_selected_placement_pose": pose,
                "placement_pose_basis": {
                    "type": "string",
                    "enum": [
                        "anyplace_raw",
                        "attachment_translation_current_orientation",
                        *(
                            ["container_mouth_fallback", "public_support_geometry"]
                            if v15_protocol
                            else []
                        ),
                    ],
                },
                "geometry_refinement_m": vector3,
                "raw_to_selected_orientation_delta_rad": {
                    "type": "number",
                    "minimum": 0,
                },
                "predicted_target_centroid_m": vector3,
                "predicted_support_gap_m": number,
                "predicted_footprint_overlap_ratio": {
                    "type": "number",
                    "minimum": 0,
                },
                "current_tool_position_m": vector3,
                "placement_candidates": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": [
                            "candidate_index",
                            "position_m",
                            "quaternion_wxyz",
                            "model_score",
                            "backend",
                            "selected",
                            "backend_candidate",
                        ],
                        "properties": {
                            "candidate_index": {
                                "type": "integer",
                                "minimum": 0,
                            },
                            "position_m": vector3,
                            "quaternion_wxyz": quaternion,
                            "model_score": nullable_number if v15_protocol else number,
                            "backend": (
                                {
                                    "type": "string",
                                    "enum": [
                                        "anyplace",
                                        "container_mouth_fallback",
                                        "public_support_geometry",
                                    ],
                                }
                                if v15_protocol
                                else {"const": "anyplace"}
                            ),
                            "selected": {"type": "boolean"},
                            "backend_candidate": {
                                "anyOf": [
                                    {
                                        "type": "object",
                                        "additionalProperties": True,
                                    },
                                    {"type": "null"},
                                ]
                            },
                        },
                        "additionalProperties": False,
                    },
                },
                "depth_quality": {
                    "type": "object",
                    "additionalProperties": True,
                },
            }
        )
    definitions = [
        *([_spec(
            "hold_position", "Advance environment time while holding the current arm and commanded "
            "gripper state. This is a physical action charged to action/horizon budgets, not a "
            "stability assertion. Use between public observations when a temporal comparison is "
            "needed; observe() alone may return the same paused state. Subsequent fresh perception "
            "is required. After a successful wait the backend rechecks the original pregrasp "
            "using its existing pose tolerances; failed or unverified pregrasps are invalidated. "
            "Inspect pregrasp_alignment.available: "
            "if false, call align_pregrasp again before descend_and_close. pose_hold reports robot "
            "pose residuals, not target stability; descent still refreshes target geometry. "
            "It neither chooses a target nor verifies attachment/completion.",
            {"duration_s": {"type": "number", "exclusiveMinimum": 0, "maximum": 30, "default": 0.5}},
            (), ("duration_s",),
            {"type": "object", "required": ["requested_duration_s", "elapsed_s", "start_time_s", "end_time_s", "steps_executed", "gripper_command_fraction", "pregrasp_alignment"],
             "properties": {"requested_duration_s": number, "elapsed_s": number, "start_time_s": number,
                            "end_time_s": number, "steps_executed": {"type": "integer", "minimum": 0},
                            "gripper_command_fraction": number,
                            "pose_hold": {
                                "type": "object", "additionalProperties": False,
                                "required": ["reference", "within_tolerance"],
                                "properties": {
                                    "reference": {"const": "end_effector"},
                                    "within_tolerance": {"type": "boolean"},
                                    "position_residual_m": {"type": "number", "minimum": 0},
                                    "orientation_residual_rad": {"type": "number", "minimum": 0},
                                    "position_tolerance_m": {"type": "number", "exclusiveMinimum": 0},
                                    "orientation_tolerance_rad": {"type": "number", "exclusiveMinimum": 0},
                                },
                            },
                            "pregrasp_alignment": {
                                "type": "object", "additionalProperties": False,
                                "required": ["available", "invalidated", "reason", "required_before_descend_and_close"],
                                "properties": {
                                    "available": {"type": "boolean"},
                                    "invalidated": {"type": "boolean"},
                                    "reason": {"type": "string", "enum": ["hold_position_requested", "pregrasp_pose_revalidated"]},
                                    "required_before_descend_and_close": {"enum": ["align_pregrasp", None]},
                                    "pose_check": {
                                        "type": "object", "additionalProperties": False,
                                        "required": ["valid", "position_residual_m", "orientation_residual_rad",
                                                     "position_tolerance_m", "orientation_tolerance_rad"],
                                        "properties": {
                                            "valid": {"type": "boolean"},
                                            "position_residual_m": {"type": "number", "minimum": 0},
                                            "orientation_residual_rad": {"type": "number", "minimum": 0},
                                            "position_tolerance_m": {"type": "number", "exclusiveMinimum": 0},
                                            "orientation_tolerance_rad": {"type": "number", "exclusiveMinimum": 0},
                                        },
                                    },
                                },
                            }}, "additionalProperties": False},
            "TimeAdvanceOutcome", {"duration_s": "second", "elapsed_s": "second"}, {}, 35.0,
            SideEffect.ACTUATOR, "hold_position(duration_s=0.5)",
            "Hold the arm and gripper controller commands while waiting for the scene to evolve",
            ControlSemantic.TIME_ADVANCE,
        )] if v15_protocol else []),
        _spec(
            "observe",
            "Capture current public RGB-D, calibration, robot state and environment time_s. Observation handles can change without time advancing; repeated reads of a paused simulation do not establish stability.",
            {},
            (),
            (),
            observation_schema,
            "ObservationHandle",
            {},
            {"result": "robot_base"},
            10.0,
            SideEffect.SENSOR_CAPTURE,
            "obs = observe()",
            "ROS camera, depth, calibration, and robot-state topics",
        ),
        _spec(
            "segment",
            "Run SAM3 text segmentation with Molmo point fallback internally and return depth-backed detections. Detection rank is not identity confirmation; call select_detection even when one result is returned.",
            {
                "prompt": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 200,
                },
                "observation_handle": handle,
                "camera_name": {
                    "anyOf": [
                        {"type": "string", "minLength": 1},
                        {"type": "null"},
                    ],
                    "default": None,
                },
            },
            ("prompt", "observation_handle"),
            ("prompt", "observation_handle", "camera_name"),
            {
                "type": "object",
                "required": [
                    "observation_handle",
                    "prompt",
                    "detections",
                    "cache_hit",
                ],
                "properties": {
                    "observation_handle": handle,
                    "prompt": {"type": "string", "minLength": 1},
                    "detections": {
                        "type": "array",
                        "items": detection,
                        "minItems": 1,
                        "maxItems": 8,
                    },
                    "cache_hit": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
            "Detections",
            {"centroid_m": "metre", "extent_m": "metre"},
            {"result": "robot_base"},
            180.0,
            SideEffect.PERCEPTION,
            "detections = segment('black bowl beside the plate', obs.value['observation_handle'])",
            "SAM3 and Molmo perception service",
        ),
        _spec(
            "select_detection",
            (
                "Bind the chosen detection to public identity evidence. Distinct same-camera "
                "candidates require identity_point_handle regardless of object wording. Every "
                "explicit identity/reference point must use the same camera and public snapshot. "
                "selection_relation='identity' is the default and does not infer a distance "
                "rule from words such as beside or closest. Only selection_relation='nearest_xy' "
                "requires a reference_point_handle and checks the minimum XY centroid distance. "
                "That measured relation does not establish general language or task truth."
                if v15_protocol
                else
                "Confirm the exact task-specified detection identity before geometry, grasp, or placement use."
            ),
            {
                "detection_id": handle,
                **(
                    {
                        "selection_relation": {"type": "string", "enum": ["identity", "nearest_xy"], "default": "identity"},
                        "identity_point_handle": {
                            "anyOf": [handle, {"type": "null"}],
                            "default": None,
                        },
                        "reference_point_handle": {
                            "anyOf": [handle, {"type": "null"}],
                            "default": None,
                        },
                    }
                    if v15_protocol
                    else {}
                ),
            },
            ("detection_id",),
            (
                "detection_id",
                *(
                    (
                        "identity_point_handle",
                        "reference_point_handle",
                        "selection_relation",
                    )
                    if v15_protocol
                    else ()
                ),
            ),
            {
                "type": "object",
                "required": [
                    "detection_id",
                    "observation_handle",
                    "camera_name",
                    "prompt",
                    "score",
                    "centroid_m",
                    "extent_m",
                    "selected",
                    "cache_hit",
                    *(
                        (
                            "identity_provenance_required",
                            "identity_provenance_confirmed",
                            "identity_point_handle",
                            "distractor_detection_id",
                            "reference_point_handle",
                            "instance_reference_distances_m",
                            "instance_selection_margin_ratio",
                            "instance_margin_low",
                        )
                        if v15_protocol
                        else ()
                    ),
                ],
                "properties": {
                    "detection_id": handle,
                    "observation_handle": handle,
                    "camera_name": {"type": "string", "minLength": 1},
                    "prompt": {"type": "string", "minLength": 1},
                    "score": number,
                    "centroid_m": vector3,
                    "extent_m": vector3,
                    "selected": {"const": True},
                    "cache_hit": {"type": "boolean"},
                    **(
                        {
                            "selection_relation": {"type": "string", "enum": ["identity", "nearest_xy"]},
                            "identity_provenance_required": {
                                "type": "boolean"
                            },
                            "identity_provenance_confirmed": {
                                "type": "boolean"
                            },
                            "identity_point_handle": {
                                "anyOf": [handle, {"type": "null"}]
                            },
                            "distractor_detection_id": {
                                "anyOf": [handle, {"type": "null"}]
                            },
                            "reference_point_handle": {
                                "anyOf": [handle, {"type": "null"}]
                            },
                            "instance_reference_distances_m": {
                                "anyOf": [
                                    {
                                        "type": "object",
                                        "additionalProperties": number,
                                    },
                                    {"type": "null"},
                                ]
                            },
                            "instance_selection_margin_ratio": {
                                "anyOf": [
                                    number,
                                    {"type": "null"},
                                ]
                            },
                            "instance_margin_low": {
                                "anyOf": [
                                    {"type": "boolean"},
                                    {"type": "null"},
                                ]
                            },
                        }
                        if v15_protocol
                        else {}
                    ),
                },
                "additionalProperties": False,
            },
            "SelectedDetection",
            {"centroid_m": "metre", "extent_m": "metre"},
            {"result": "robot_base"},
            10.0,
            SideEffect.READ_ONLY,
            (
                "identity = point('black bowl next to the ramekin', "
                "obs.value['observation_handle'], camera_name='agentview'); "
                "target = select_detection("
                "detections.value['detections'][0]['detection_id'], "
                "identity_point_handle=identity.value['point_handle'])"
                if v15_protocol
                else
                "target = select_detection(detections.value['detections'][0]['detection_id'])"
            ),
            "perception track-ID selection",
        ),
        _spec(
            "point",
            "Ask the point service for one pixel and internally deproject and point-segment it into public metric evidence.",
            {
                "prompt": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 200,
                },
                "observation_handle": handle,
                "camera_name": {
                    "type": "string",
                    "default": "agentview",
                },
            },
            ("prompt", "observation_handle"),
            ("prompt", "observation_handle", "camera_name"),
            point_schema,
            "GroundedPoint",
            {"position_m": "metre", "pixel_xy": "pixel"},
            {"result": "robot_base", "pixel_xy": "camera_image"},
            180.0,
            SideEffect.PERCEPTION,
            "target_point = point('released bowl', obs.value['observation_handle'])",
            "Molmo pointing plus RealSense deprojection",
        ),
        _spec(
            "gripper_state",
            "Read gripper opening, hardware-style object detection, force, and the immutable public gripper profile.",
            {},
            (),
            (),
            {
                "type": "object",
                "required": [
                    "open_fraction",
                    "object_detected",
                    "force_n",
                    "profile",
                ],
                "properties": {
                    "open_fraction": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "object_detected": {"type": "boolean"},
                    "force_n": {"type": ["number", "null"], "minimum": 0},
                    "force_available": {"type": "boolean"},
                    "profile": gripper_profile,
                },
                "additionalProperties": False,
            },
            "GripperState",
            {
                "open_fraction": "dimensionless",
                "force_n": "newton",
            },
            {},
            10.0,
            SideEffect.READ_ONLY,
            "hand = gripper_state()",
            "Robotiq OBJ register and force sensor",
            ControlSemantic.ATTACHMENT_CONTINUITY,
        ),
        _spec(
            "construct_point_cloud",
            "Project the selected detection into every available camera, point-segment each projected view, and fuse scored base-frame object clouds.",
            {
                "observation_handle": handle,
                "detection_id": handle,
                "use_multiview": {"type": "boolean", "default": True},
            },
            ("observation_handle", "detection_id"),
            ("observation_handle", "detection_id", "use_multiview"),
            cloud_result,
            "PointCloudHandles",
            {
                "object_centroid_m": "metre",
                "object_bounds_min_m": "metre",
                "object_bounds_max_m": "metre",
            },
            {"result": "robot_base"},
            60.0,
            SideEffect.COMPUTE,
            "clouds = construct_point_cloud(obs.value['observation_handle'], target.value['detection_id'])",
            "ROS depth_image_proc and multi-view point-cloud fusion",
        ),
        _spec(
            "grasp_pose_estimate",
            (
                "Estimate grasp candidates in one service call. V10 returns "
                "every backend candidate up to model-supplied top_k; width, "
                "style, IK, and joint travel are diagnostics and never filters."
                if dehardcoded
                else
                "Estimate grasp candidates in one service call. Only selected detections enter the service; every returned candidate is width-feasible and reachable under the active gripper, TCP, approach, IK, and controller contract."
            ),
            {
                "observation_handle": handle,
                "detection_id": handle,
                "gripper_profile": gripper_profile,
                **(
                    {
                        "backend": {
                            "type": "string",
                            "enum": [
                                "auto",
                                "contact_graspnet",
                                "anygrasp",
                                "graspgenx",
                                *(["mask_topdown_fallback"] if v15_protocol else []),
                            ],
                            "default": "auto",
                        },
                        "depth_source": {
                            "type": "string",
                            "enum": ["sensor", "enhanced"],
                            "default": "sensor",
                        },
                    }
                    if dehardcoded
                    else {}
                ),
                "grasp_style": {
                    "type": "string",
                    "enum": ["auto", "top_down", "side"],
                    "default": "auto",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    **({} if dehardcoded else {"maximum": 32}),
                    "default": 15 if dehardcoded else 5,
                },
            },
            ("observation_handle", "detection_id", "gripper_profile"),
            (
                "observation_handle",
                "detection_id",
                "gripper_profile",
                *(("backend", "depth_source") if dehardcoded else ()),
                "grasp_style",
                "top_k",
            ),
            grasp_result,
            "GraspEstimate",
            {
                "position_m": "metre",
                "predicted_grasp_width_m": "metre",
            },
            {"result": "robot_base"},
            180.0,
            SideEffect.PERCEPTION,
            (
                "grasps = grasp_pose_estimate(obs.value['observation_handle'], "
                "target.value['detection_id'], hand.value['profile'], "
                "backend='auto', grasp_style='auto', top_k=15)"
                if dehardcoded
                else
                "grasps = grasp_pose_estimate(obs.value['observation_handle'], target.value['detection_id'], hand.value['profile'], grasp_style='auto')"
            ),
            "Contact-GraspNet or GraspGenX service",
            ControlSemantic.GRASP_PLANNING,
        ),
        _spec(
            "plan_grasp_from_point_clouds",
            (
                "Compose grasp estimation explicitly from selected-detection "
                "point-cloud handles. V10 reports all candidate diagnostics "
                "without feasibility filtering."
                if dehardcoded
                else
                "Compose grasp estimation explicitly from selected-detection point-cloud handles. It has the same feasibility filters and result contract as grasp_pose_estimate."
            ),
            {
                "full_point_cloud_handle": handle,
                "object_point_cloud_handle": handle,
                "detection_id": handle,
                **(
                    {
                        "backend": {
                            "type": "string",
                            "enum": [
                                "auto",
                                "contact_graspnet",
                                "anygrasp",
                                "graspgenx",
                                *(["mask_topdown_fallback"] if v15_protocol else []),
                            ],
                            "default": "auto",
                        },
                        "depth_source": {
                            "type": "string",
                            "enum": ["sensor", "enhanced"],
                            "default": "sensor",
                        },
                    }
                    if dehardcoded
                    else {}
                ),
                "grasp_style": {
                    "type": "string",
                    "enum": ["auto", "top_down", "side"],
                    "default": "auto",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    **({} if dehardcoded else {"maximum": 32}),
                    "default": 15 if dehardcoded else 5,
                },
            },
            (
                "full_point_cloud_handle",
                "object_point_cloud_handle",
                "detection_id",
            ),
            (
                "full_point_cloud_handle",
                "object_point_cloud_handle",
                "detection_id",
                *(("backend", "depth_source") if dehardcoded else ()),
                "grasp_style",
                "top_k",
            ),
            grasp_result,
            "GraspEstimate",
            {
                "position_m": "metre",
                "predicted_grasp_width_m": "metre",
            },
            {"result": "robot_base"},
            180.0,
            SideEffect.PERCEPTION,
            "grasps = plan_grasp_from_point_clouds(clouds.value['full_point_cloud_handle'], clouds.value['object_point_cloud_handle'], target.value['detection_id'])",
            "Contact-GraspNet point-cloud service",
            ControlSemantic.GRASP_PLANNING,
        ),
        _spec(
            "placement_estimate",
            (
                "Run AnyPlace from a target, placement-region, and explicit "
                "model-selected grasp captured in one aligned pre-grasp "
                "observation. placement_candidate_index and "
                "reference_tolerance_m are model decisions. When public "
                "attachment geometry is verified, selected_placement_pose "
                "preserves the current grasp orientation and solves only the "
                "translation needed to center the object footprint and target "
                "a 1 cm support gap for 'on', or a below-rim centroid for "
                "'in'. This avoids amplifying orientation error through the "
                "TCP lever arm. "
                "For an issued public-geometry grasp, relation=on can return public_support_geometry "
                "with model_score=null and no native payload. It preserves the measured held orientation "
                "and proposes a translation from partial visible support geometry; it is not contact, path "
                "or completion proof. Check the same release and completion evidence before finishing. "
                "A container_mouth_fallback pose is only an above-rim hover, not "
                "a release or containment certificate. Use a fresh destination "
                "detection and inspect the opening instead of substituting a "
                "surface centroid. Execute selected_placement_pose with "
                "position_reference='tool', or its predicted target center with "
                "position_reference='held_target' and tracking_handle. The "
                "raw AnyPlace pose, orientation delta, applied refinement, predicted object centroid, support gap, "
                "and footprint overlap remain visible for audit. The Harness "
                "does not add a motion gain or distance cap."
                if dehardcoded
                else
                "Measure held-target placement error from two explicitly selected detections. The error and suggested step are advisory; max_safe_step_m is the only motion envelope enforced by the Harness."
            ),
            {
                "tracking_handle": handle,
                "target_detection_id": handle,
                "destination_detection_id": handle,
                "relation_type": {
                    "type": "string",
                    "enum": ["on", "in"],
                    "default": "on",
                },
                **(
                    {
                        "selected_grasp": candidate,
                        "placement_candidate_index": {
                            "type": "integer",
                            "minimum": 0,
                            "default": 0,
                        },
                        "reference_tolerance_m": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                        },
                    }
                    if dehardcoded
                    else {}
                ),
            },
            (
                "tracking_handle",
                "target_detection_id",
                "destination_detection_id",
                *(
                    ("selected_grasp", "reference_tolerance_m")
                    if dehardcoded
                    else ()
                ),
            ),
            (
                "tracking_handle",
                "target_detection_id",
                "destination_detection_id",
                "relation_type",
                *(
                    (
                        "selected_grasp",
                        "placement_candidate_index",
                        "reference_tolerance_m",
                    )
                    if dehardcoded
                    else ()
                ),
            ),
            placement_result,
            "PlacementEstimate",
            {
                "error_vector_m": "metre",
                "distance_m": "metre",
                "max_safe_step_m": "metre",
                "suggested_step_m": "metre",
            },
            {"result": "robot_base"},
            120.0,
            SideEffect.PERCEPTION,
            (
                "estimate = placement_estimate("
                "grasps.value['tracking_handle'], "
                "target.value['detection_id'], "
                "destination.value['detection_id'], "
                "relation_type='on', "
                "selected_grasp=grasps.value['candidates'][0], "
                "placement_candidate_index=0, "
                "reference_tolerance_m=0.02)"
                if dehardcoded
                else
                "estimate = placement_estimate(track, held.value['detection_id'], destination.value['detection_id'], relation_type='on')"
            ),
            "AnyPlace placement service",
            ControlSemantic.PLACEMENT_ESTIMATE,
        ),
        _spec(
            "subsample_point_cloud",
            "Deterministically reduce a handled point cloud to at most max_points.",
            {
                "point_cloud_handle": handle,
                "max_points": {
                    "type": "integer",
                    "minimum": 3,
                    "maximum": 100000,
                    "default": 10000,
                },
            },
            ("point_cloud_handle",),
            ("point_cloud_handle", "max_points"),
            {
                "type": "object",
                "required": ["point_cloud_handle", "point_count"],
                "properties": {
                    "point_cloud_handle": handle,
                    "point_count": {"type": "integer", "minimum": 3},
                },
                "additionalProperties": False,
            },
            "PointCloudHandle",
            {},
            {"point_cloud_handle": "robot_base"},
            10.0,
            SideEffect.COMPUTE,
            "small = subsample_point_cloud(clouds.value['object_point_cloud_handle'])",
            "PCL voxel or uniform sampling filter",
        ),
        _spec(
            "filter_noise",
            "Apply the frozen point-cloud radius filter and return a new handled cloud.",
            {"point_cloud_handle": handle},
            ("point_cloud_handle",),
            ("point_cloud_handle",),
            {
                "type": "object",
                "required": [
                    "point_cloud_handle",
                    "point_count",
                    "removed_count",
                ],
                "properties": {
                    "point_cloud_handle": handle,
                    "point_count": {"type": "integer", "minimum": 3},
                    "removed_count": {"type": "integer", "minimum": 0},
                },
                "additionalProperties": False,
            },
            "FilteredPointCloud",
            {},
            {"point_cloud_handle": "robot_base"},
            30.0,
            SideEffect.COMPUTE,
            "clean = filter_noise(clouds.value['object_point_cloud_handle'])",
            "PCL radius or statistical outlier filter",
        ),
        _spec(
            "check_reachable",
            "Check a base-frame pose with the same TCP and PyRoKi contract used by goto_pose without moving the robot.",
            {"pose": pose},
            ("pose",),
            ("pose",),
            {
                "type": "object",
                "required": ["reachable", "reason", "joints"],
                "properties": {
                    "reachable": {"type": "boolean"},
                    "reason": {"type": "string", "minLength": 1},
                    "joints": {
                        "anyOf": [joints, {"type": "null"}]
                    },
                },
                "additionalProperties": False,
            },
            "Reachability",
            {"joints": "radian"},
            {"pose": "robot_base", "joints": "robot_joint_space"},
            120.0,
            SideEffect.COMPUTE,
            "reachable = check_reachable(grasps.value['candidates'][0])",
            "MoveIt /compute_ik",
        ),
        replace(
            legacy_by_name["solve_ik"],
            real_robot_equivalent="MoveIt /compute_ik",
        ),
        replace(
            legacy_by_name["move_to_joints"],
            description="Execute a joint target when the active robot controller supports joint-space commands. Under the V9 OSC_POSE LIBERO profile this returns controller_rejected instead of silently switching controllers.",
            real_robot_equivalent="joint_trajectory_controller",
        ),
        _spec(
            "goto_pose",
            "Move toward a base-frame pose. While holding, explicitly declare position_reference: held_target means position is the OBJECT center and requires tracking_handle; the adapter applies its verified offset. tool means an already computed public tool waypoint, such as selected_placement_pose. A container surface centroid is not necessarily its opening center; inspect the opening and object footprint, first transport above obstacles, then lower in separately observed actions. Keep current orientation when possible. approach_frame defaults to world. Uses calibrated public position and orientation tolerances; residual above either tolerance is a hard failure. Inspect convergence and residuals.",
            {
                "position": vector3,
                "quaternion": quaternion,
                "z_approach": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 0.5,
                    "default": 0.0,
                },
                "approach_frame": {
                    "type": "string",
                    "enum": ["world", "tool"],
                    "default": "world",
                },
                "position_reference": {"type": "string", "enum": ["tool", "held_target"],
                    "description": "Required while holding a verified object. held_target interprets position as object center and applies the verified attachment offset. tool requires an already computed tool waypoint."},
                "tracking_handle": handle,
            },
            ("position", "quaternion"),
            (
                "position",
                "quaternion",
                "z_approach",
                "approach_frame",
                "position_reference",
                "tracking_handle",
            ),
            legacy_by_name["goto_pose"].result_schema,
            "MotionOutcome",
            legacy_by_name["goto_pose"].units,
            legacy_by_name["goto_pose"].coordinate_frames,
            180.0,
            SideEffect.ROBOT_MOTION,
            "motion = goto_pose(candidate['position_m'], candidate['quaternion_wxyz'], z_approach=0.08, approach_frame='world')",
            "Franka Cartesian impedance controller",
            ControlSemantic.CARTESIAN_MOTION,
        ),
        *(
            (
                _spec(
                    "align_pregrasp",
                    "Open the gripper and align one candidate through staged controller phases: vertical clearance at the current XY/orientation before horizontal hover translation, in-place orientation convergence, a position-only settle back to hover, then position-only descent to pregrasp. A failed result names the failed phase and residuals. convergence_risk is an uncalibrated candidate-time heuristic, not a path check or a probability. After a stall, inspect the failed phase, supplied pose and contact evidence before choosing a new candidate, observation, or recovery.",
                    {"candidate": candidate},
                    ("candidate",),
                    ("candidate",),
                    {
                        "type": "object",
                        "required": [
                            "converged",
                            "phase_reached",
                            "failed_phase",
                            "position_residual_m",
                            "orientation_residual_rad",
                            "convergence_reason",
                            "pregrasp_position_m",
                            "hover_position_m",
                            "tracking_handle",
                            "detection_id",
                            "candidate_index",
                            "convergence_risk",
                        ],
                        "properties": {
                            "converged": {"const": True},
                            "phase_reached": {"const": "pregrasp"},
                            "failed_phase": {"type": "null"},
                            "position_residual_m": {
                                "type": "number",
                                "minimum": 0,
                            },
                            "orientation_residual_rad": {
                                "type": "number",
                                "minimum": 0,
                            },
                            "convergence_reason": {
                                "type": "string",
                                "minLength": 1,
                            },
                            "pregrasp_position_m": vector3,
                            "hover_position_m": vector3,
                            "tracking_handle": handle,
                            "detection_id": handle,
                            "candidate_index": {
                                "type": "integer",
                                "minimum": 0,
                            },
                            "convergence_risk": {
                                "type": "string",
                                "enum": ["low", "medium", "high"],
                            },
                        },
                        "additionalProperties": False,
                    },
                    "PregraspAlignment",
                    {
                        "position_residual_m": "metre",
                        "orientation_residual_rad": "radian",
                        "pregrasp_position_m": "metre",
                        "hover_position_m": "metre",
                    },
                    {"candidate": "robot_base", "result": "robot_base"},
                    180.0,
                    SideEffect.ROBOT_MOTION,
                    "aligned = align_pregrasp(grasps.value['candidates'][0])",
                    "Staged Cartesian impedance pregrasp controller",
                    ControlSemantic.CARTESIAN_MOTION,
                ),
                _spec(
                    "descend_and_close",
                    "After successful align_pregrasp, descend toward its contact pose. Early force or position-stall stops outside the 3 mm target tolerance do NOT close the fingers; inspect remaining_vector_m and pre/post wrist evidence, retreat and re-localize. At target, close and report object detection separately from execution success. Default force limit is 18 N; an independent force increase of 5 N also stops descent. Never increase force to push through an unidentified obstacle.",
                    {
                        "max_contact_force": {
                            "anyOf": [
                                {
                                    "type": "number",
                                    "exclusiveMinimum": 0,
                                },
                                {"type": "null"},
                            ],
                            "default": None,
                        },
                    },
                    (),
                    ("max_contact_force",),
                    {
                        "type": "object",
                        "required": [
                            "contacted",
                            "contact_travel_m",
                            "gripper_open_fraction",
                            "object_detected",
                            "stopped_reason",
                            "position_residual_m",
                            "orientation_residual_rad",
                            "tracking_handle",
                            "detection_id",
                            "candidate_quality_flags",
                            "candidate_width_known",
                            "candidate_width_feasible",
                            "candidate_ik_feasible",
                            "candidate_convergence_risk",
                            "pre_descent_centering",
                        ],
                        "properties": {
                            "closure_executed": {"type": "boolean"},
                            "measurement_phase": {
                                "enum": ["after_closure", "after_interrupted_closure"],
                                "x-attempt-feedback": "priority",
                            },
                            "contact_geometry_audit": {
                                "type": "object", "required": ["status"],
                                "x-attempt-feedback": "priority",
                                "description": "Requested and measured robot poses on the same source target cloud; includes positional contact sensitivity. Not a fresh target measurement or attachment proof.",
                                "properties": {"status": {"enum": [
                                    "unknown", "measured_robot_pose_on_source_cloud"]}},
                                "additionalProperties": True,
                            },
                            "descent_execution": {
                                "type": "object", "x-decision-feedback": "on_demand",
                                "description": "Measurements before closure; top-level fields refer to the final post-closure state.",
                                "properties": {
                                    "requested_position_tolerance_m": {
                                        "type": "number", "minimum": 0.0005, "maximum": 0.003,
                                    },
                                    "contacted": {"type": "boolean"},
                                    "stopped_reason": {"type": "string"},
                                    "contact_travel_m": number,
                                    "position_residual_m": number,
                                    "remaining_vector_m": vector3,
                                    "observed_eef_position_m": vector3,
                                    "requested_eef_position_m": vector3,
                                    "force_limit_n": number,
                                    "force_increase_limit_n": number,
                                    "baseline_force_n": {"type": ["number", "null"]},
                                    "observed_force_n": {"type": ["number", "null"]},
                                    "contact_identity": {"type": "string"},
                                },
                                "additionalProperties": False,
                            },
                            "grasp_outcome": {"type": "string"},
                            "remaining_vector_m": vector3,
                            "observed_eef_position_m": vector3,
                            "requested_eef_position_m": vector3,
                            "force_limit_n": number,
                            "force_increase_limit_n": number,
                            "baseline_force_n": {"type": ["number", "null"]},
                            "observed_force_n": {"type": ["number", "null"]},
                            "contact_identity": {"type": "string"},
                            "contacted": {"type": "boolean"},
                            "contact_travel_m": {
                                "type": "number",
                                "minimum": 0,
                            },
                            "pre_descent_centering": {
                                "type": "object",
                                "required": ["reason", "applied_m"],
                                "properties": {
                                    "reason": {
                                        "type": "string",
                                        "minLength": 1,
                                    },
                                    "applied_m": {
                                        "anyOf": [
                                            vector3,
                                            {"type": "null"},
                                        ]
                                    },
                                    "measured_m": {
                                        "type": "number",
                                        "minimum": 0,
                                    },
                                    "clamped": {"type": "boolean"},
                                    "decision": {"type":"string", "enum":["retain","translate","reacquire"]},
                                    "basis": {"type":"string"},
                                    "motion_evidence": {"type":"object", "additionalProperties":True},
                                    "measured_translation_m": vector3,
                                    "source_detection_id": {"type":"string"},
                                    "observation_handle": {"type":"string"},
                                },
                                "additionalProperties": False,
                            },
                            "gripper_open_fraction": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "object_detected": {"type": "boolean"},
                            "stopped_reason": {
                                "type": "string",
                                "enum": [
                                    "contact_force",
                                    "position_stall",
                                    "target_reached",
                                ],
                            },
                            "position_residual_m": {
                                "type": "number",
                                "minimum": 0,
                            },
                            "orientation_residual_rad": {
                                "type": "number",
                                "minimum": 0,
                            },
                            "tracking_handle": handle,
                            "detection_id": handle,
                            "candidate_quality_flags": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "minLength": 1,
                                },
                            },
                            "candidate_width_known": {
                                "type": "boolean"
                            },
                            "candidate_width_feasible": {
                                "type": "boolean"
                            },
                            "candidate_ik_feasible": {
                                "type": "boolean"
                            },
                            "candidate_convergence_risk": {
                                "type": "string",
                                "enum": ["low", "medium", "high"],
                            },
                        },
                        "additionalProperties": False,
                    },
                    "DescendCloseOutcome",
                    {
                        "contact_travel_m": "metre",
                        "max_contact_force": "newton",
                        "position_residual_m": "metre",
                        "orientation_residual_rad": "radian",
                    },
                    {"result": "robot_base"},
                    180.0,
                    SideEffect.ROBOT_MOTION,
                    "closed = descend_and_close()",
                    "Guarded Cartesian descent plus gripper register",
                    ControlSemantic.ATTACHMENT_ATTEMPT,
                ),
            )
            if v15_protocol
            else (
                _spec(
                    "approach_and_close",
                    "Run the latency-critical compliant approach, contact or stall stop, gripper close, and hardware-style object-detection read in one primitive.",
                    {
                        "candidate": candidate,
                        "max_contact_force": {
                            "anyOf": [
                                {
                                    "type": "number",
                                    "exclusiveMinimum": 0,
                                },
                                {"type": "null"},
                            ],
                            "default": None,
                        },
                    },
                    ("candidate",),
                    ("candidate", "max_contact_force"),
                    {
                        "type": "object",
                        "required": [
                            "contacted",
                            "contact_travel_m",
                            "gripper_open_fraction",
                            "object_detected",
                            "stopped_reason",
                            "position_residual_m",
                            "orientation_residual_rad",
                            "tracking_handle",
                            "detection_id",
                            *(
                                (
                                    "candidate_quality_flags",
                                    "candidate_width_known",
                                    "candidate_width_feasible",
                                    "candidate_ik_feasible",
                                )
                                if dehardcoded
                                else ()
                            ),
                        ],
                        "properties": {
                            "contacted": {"type": "boolean"},
                            "contact_travel_m": {
                                "type": "number",
                                "minimum": 0,
                            },
                            "gripper_open_fraction": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "object_detected": {"type": "boolean"},
                            "stopped_reason": {
                                "type": "string",
                                "enum": [
                                    "contact_force",
                                    "position_stall",
                                    "target_reached",
                                ],
                            },
                            "position_residual_m": {
                                "type": "number",
                                "minimum": 0,
                            },
                            "orientation_residual_rad": {
                                "type": "number",
                                "minimum": 0,
                            },
                            "tracking_handle": handle,
                            "detection_id": handle,
                            **(
                                {
                                    "candidate_quality_flags": {
                                        "type": "array",
                                        "items": {
                                            "type": "string",
                                            "minLength": 1,
                                        },
                                    },
                                    "candidate_width_known": {
                                        "type": "boolean"
                                    },
                                    "candidate_width_feasible": {
                                        "type": "boolean"
                                    },
                                    "candidate_ik_feasible": {
                                        "type": "boolean"
                                    },
                                }
                                if dehardcoded
                                else {}
                            ),
                        },
                        "additionalProperties": False,
                    },
                    "ApproachCloseOutcome",
                    {
                        "contact_travel_m": "metre",
                        "max_contact_force": "newton",
                        "position_residual_m": "metre",
                        "orientation_residual_rad": "radian",
                    },
                    {"candidate": "robot_base"},
                    180.0,
                    SideEffect.ROBOT_MOTION,
                    "closed = approach_and_close(grasps.value['candidates'][0])",
                    "Cartesian impedance approach loop plus gripper register",
                    ControlSemantic.ATTACHMENT_ATTEMPT,
                ),
            )
        ),
        _spec(
            "servo_to",
            "Run the internal Cartesian servo loop with public robot pose feedback. "
            "target_pose defaults to the same tool reference as grasp candidates and goto_pose. "
            "For observe().robot_state.end_effector_* fields, pass position_reference='end_effector'; "
            "for tool_* fields or grasp candidates use 'tool'. The adapter converts the full pose, "
            "including orientation. camera labels the view; this loop does not track image features. "
            "The model chooses pose, duration and tolerance; inspect reached and residuals.",
            {
                "target_pose": pose,
                "position_reference": {"type": "string", "enum": ["tool", "end_effector"], "default": "tool"},
                "camera": {
                    "type": "string",
                    "enum": ["agentview", "robot0_eye_in_hand"],
                },
                "max_duration_s": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "maximum": 30,
                },
                "tolerance_m": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "maximum": 0.1,
                },
            },
            (
                "target_pose",
                "camera",
                "max_duration_s",
                "tolerance_m",
            ),
            (
                "target_pose",
                "camera",
                "max_duration_s",
                "tolerance_m",
                "position_reference",
            ),
            {
                "type": "object",
                "required": [
                    "reached",
                    "final_error_m",
                    "iterations",
                    "aborted_reason",
                ],
                "properties": {
                    "reached": {"type": "boolean"},
                    "input_pose_reference": {"type": "string", "enum": ["tool", "end_effector"]},
                    "input_position_m": vector3,
                    "input_quaternion_wxyz": quaternion,
                    "requested_eef_position_m": vector3,
                    "requested_eef_quaternion_wxyz": quaternion,
                    "final_error_m": {
                        "type": "number",
                        "minimum": 0,
                    },
                    "iterations": {"type": "integer", "minimum": 0},
                    "aborted_reason": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "null"},
                        ]
                    },
                },
                "additionalProperties": False,
            },
            "ServoOutcome",
            {
                "max_duration_s": "second",
                "tolerance_m": "metre",
                "final_error_m": "metre",
            },
            {"target_pose": "robot_base"},
            60.0,
            SideEffect.ROBOT_MOTION,
            "servo = servo_to(candidate, 'robot0_eye_in_hand', 3.0, 0.005)",
            "visual servo or Cartesian impedance loop",
            ControlSemantic.CARTESIAN_MOTION,
        ),
        _spec(
            "lift_probe",
            "Lift vertically in robot-base Z, then report actual displacement, residual, and gripper object-detection state.",
            {
                "height": {
                    "type": "number",
                    "minimum": 0.01,
                    "maximum": 0.35,
                }
            },
            ("height",),
            ("height",),
            {
                "type": "object",
                "required": [
                    "requested_height_m",
                    "actual_vertical_displacement_m",
                    "position_residual_m",
                    "gripper_open_fraction",
                    "object_detected",
                    "associated_tracking_handle",
                    "attachment_attempt_id",
                    "supports_attachment",
                    "contradicts_attachment",
                ],
                "properties": {
                    "requested_height_m": number,
                    "actual_vertical_displacement_m": number,
                    "position_residual_m": {
                        "type": "number",
                        "minimum": 0,
                    },
                    "gripper_open_fraction": nullable_number,
                    "object_detected": {
                        "anyOf": [
                            {"type": "boolean"},
                            {"type": "null"},
                        ]
                    },
                    "associated_tracking_handle": {
                        "anyOf": [handle, {"type": "null"}]
                    },
                    "attachment_attempt_id": {
                        "anyOf": [
                            {"type": "integer", "minimum": 1},
                            {"type": "null"},
                        ]
                    },
                    "supports_attachment": {"type": "boolean"},
                    "contradicts_attachment": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
            "LiftProbe",
            {
                "height": "metre",
                "actual_vertical_displacement_m": "metre",
                "position_residual_m": "metre",
            },
            {"result": "robot_base"},
            180.0,
            SideEffect.ROBOT_MOTION,
            "lifted = lift_probe(0.10)",
            "Cartesian impedance lift with gripper register read",
            ControlSemantic.ATTACHMENT_PROBE,
        ),
        _spec(
            "wrist_alignment",
            "Project a candidate into the wrist camera and return the image error and advisory base-frame alignment delta.",
            {
                "observation_handle": handle,
                "candidate": candidate,
            },
            ("observation_handle", "candidate"),
            ("observation_handle", "candidate"),
            {
                "type": "object",
                "required": [
                    "pixel_error",
                    "normalized_error",
                    "delta_m",
                    "centered",
                ],
                "properties": {
                    "pixel_error": vector2,
                    "normalized_error": vector2,
                    "delta_m": vector3,
                    "centered": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
            "WristAlignment",
            {"pixel_error": "pixel", "delta_m": "metre"},
            {
                "pixel_error": "wrist_camera",
                "delta_m": "robot_base",
            },
            30.0,
            SideEffect.COMPUTE,
            "alignment = wrist_alignment(obs.value['observation_handle'], candidate)",
            "wrist-camera visual alignment service",
        ),
        replace(
            legacy_by_name["open_gripper"],
            real_robot_equivalent=equivalents["open_gripper"],
        ),
        replace(
            legacy_by_name["close_gripper"],
            real_robot_equivalent=equivalents["close_gripper"],
        ),
        replace(
            legacy_by_name["home"],
            real_robot_equivalent=equivalents["home"],
        ),
        *retained,
    ]
    if level is ToolApiLevel.S2_HIGH_LEVEL:
        definitions.extend(
            (
                _spec(
                    "pick",
                    "Execute the supplied candidate open-loop: open, compliant approach-and-close, lift, and gripper-first verification. The skill never selects or changes the candidate.",
                    {"candidate": candidate},
                    ("candidate",),
                    ("candidate",),
                    {
                        "type": "object",
                        "required": [
                            "picked",
                            "object_detected",
                            "tracking_handle",
                        ],
                        "properties": {
                            "picked": {"type": "boolean"},
                            "object_detected": {"type": "boolean"},
                            "tracking_handle": handle,
                        },
                        "additionalProperties": True,
                    },
                    "PickOutcome",
                    {},
                    {"candidate": "robot_base"},
                    300.0,
                    SideEffect.ROBOT_MOTION,
                    "picked = pick(grasps.value['candidates'][0])",
                    "ROS2 pick skill composed from V9 primitives",
                    ControlSemantic.ATTACHMENT_ATTEMPT,
                ),
                _spec(
                    "place",
                    "Execute one open-loop placement toward the model-selected destination detection. It does not choose the destination and does not run hidden recovery.",
                    {"destination_detection_id": handle},
                    ("destination_detection_id",),
                    ("destination_detection_id",),
                    {
                        "type": "object",
                        "required": ["placed", "released"],
                        "properties": {
                            "placed": {"type": "boolean"},
                            "released": {"type": "boolean"},
                        },
                        "additionalProperties": True,
                    },
                    "PlaceOutcome",
                    {},
                    {"destination_detection_id": "robot_base"},
                    300.0,
                    SideEffect.ROBOT_MOTION,
                    "placed = place(destination.value['detection_id'])",
                    "ROS2 place skill composed from V9 primitives",
                    ControlSemantic.ATTACHMENT_RELEASE,
                ),
            )
        )
    if v15_protocol:
        status_schema = {"type": "string", "enum": ["supported", "contradicted", "insufficient"]}
        revised = []
        for spec in definitions:
            schema = dict(spec.result_schema)
            description = spec.description
            additions = {}
            if spec.name == "gripper_state":
                additions = {"object_detection_basis": {"type": "string"},
                    "target_attachment_status": status_schema, "evidence_limitation": {"type": "string"}}
                description = "Read aperture, force and the public gripper profile. object_detected is a backend gripper signal (LIBERO uses command plus aperture), not target identity or secure grasp."
            elif spec.name == "align_pregrasp":
                additions = {
                    "target_contact_evidence": {"type": "object", "additionalProperties": True},
                    "candidate_evidence": {
                        "type": "object",
                        "required": ["position_m", "quaternion_wxyz", "approach_direction_world",
                            "execution_eef_position_m", "execution_eef_quaternion_wxyz",
                            "source_observation_handle", "motion_path", "basis"],
                        "properties": {"position_m": vector3, "quaternion_wxyz": quaternion,
                            "approach_direction_world": vector3, "execution_eef_position_m": vector3,
                            "execution_eef_quaternion_wxyz": quaternion,
                            "source_observation_handle": {"type": ["string", "null"]},
                            "motion_path": {"type": "string", "const": "not_checked"},
                            "basis": {"type": "string"}},
                        "additionalProperties": False,
                    },
                }
                description += " candidate_evidence and target_contact_evidence are computed before motion for the supplied pose and selected target cloud, and retained on failure. source_observation_handle names that cloud's observation; no fresh image is implied. Each new attempt invalidates prior alignment, and only confirmed pregrasp convergence enables descent. Inspect local target bounds and wrist images; movement convergence alone does not establish contact."
            elif spec.name in {"grasp_pose_estimate", "plan_grasp_from_point_clouds"}:
                description += " Inspect selection_feedback and each request_style_match: API success means candidates were returned, not that the requested style or a feasible path was found. Matching/alternative indices refer to this returned list; matching_count includes candidates omitted by top_k. If requires_decision=true, end this code cell to inspect the feedback before selecting an alternative or replanning in a later cell. Contact-support ordering and native candidate binding are preserved; no new feasibility filter is applied. feasibility_evidence separates endpoint IK availability and estimated width from the unchecked motion path. Partial geometry does not prove finger contact or force closure."
            elif spec.name == "authorize_safe_release":
                arguments_schema = dict(spec.arguments_schema)
                arguments_schema["properties"] = {**arguments_schema["properties"],
                    "target_point_handle": {"type": ["string", "null"], "default": None},
                    "destination_point_handle": {"type": ["string", "null"], "default": None}}
                spec = replace(spec, arguments_schema=arguments_schema,
                    argument_order=(*spec.argument_order, "target_point_handle", "destination_point_handle"))
                additions = {
                    "relation_type": {"type": "string"},
                    "relation_estimate": {"type": "boolean"},
                    "correction_stage": {"type": "string"},
                    "recommended_eef_delta_m": {"type": "array", "items": number, "minItems": 3, "maxItems": 3},
                    "correction_stable": {"type": "boolean"},
                    "measurement_confidence": number,
                    "measurement_confidence_basis": {"type": "string"},
                    "confidence_calibrated": {"type": "boolean"},
                    "placement_geometry_confidence": {"type": "null"},
                    "correction_stable_basis": {"type": "string"},
                    "correction_resolution": {
                        "type": "object",
                        "required": ["status", "nominal_correction_m", "destination_change_along_correction_m",
                                     "accuracy_guaranteed", "basis"],
                        "properties": {
                            "status": {"enum": ["not_assessed", "no_nominal_correction", "view_changed",
                                                 "below_observed_change", "exceeds_observed_change"]},
                            "nominal_correction_m": {"type": ["number", "null"], "minimum": 0},
                            "destination_change_along_correction_m": {"type": ["number", "null"], "minimum": 0},
                            "accuracy_guaranteed": {"const": False},
                            "basis": {"type": "string"},
                            "guidance": {"type": "string"},
                        },
                        "additionalProperties": False,
                    },
                    "target_camera_name": {"type": "string"},
                    "destination_camera_name": {"type": "string"},
                    "correction_geometry_feasible": {"type": "boolean"},
                    "correction_basis": {"type": "string"},
                    "visible_center_interval_min_xy_m": {"type": "array", "items": number, "minItems": 2, "maxItems": 2},
                    "visible_center_interval_max_xy_m": {"type": "array", "items": number, "minItems": 2, "maxItems": 2},
                    "refresh_comparison": {"type": ["object", "null"], "additionalProperties": True},
                    "target_visual_residual_m": number,
                    "geometry_basis": {"type": "string"},
                    "estimate_fresh": {"type": "boolean"},
                    "requires_decision": {"type": "boolean"},
                    "refresh_applied": {"type": "boolean"},
                    "robot_translation_since_observation_m": number,
                    "robot_rotation_since_observation_rad": number,
                    "freshness_basis": {"type": "string"},
                    "refresh_guidance": {"type": "string"},
                }
                schema["properties"] = dict(schema["properties"])
                schema["properties"]["reason"] = {"type": "string", "enum": [
                    *schema["properties"]["reason"]["enum"], "stale_placement_estimate"]}
                description = "Inspect release evidence. After transport, observe and point to the held target and destination in the SAME fresh observation, then supply both optional point handles to refresh the existing estimate. Old pose-dependent estimates return estimate_fresh=false. Inspect safe_to_release and reason, not just API success. When requires_decision=true, end this code cell: further physical actions wait for a new model decision. In the next turn, settle/relocalize or explain an override from current images; no hidden success information is used."
                description += " For containment, recommended_eef_delta_m is the minimum translation satisfying the visible footprint bounds, not a move to an arbitrary surface point. If correction_geometry_feasible=false, re-localize instead of treating a zero vector as ready. measurement_confidence covers target visual consistency only; placement_geometry_confidence is unknown. correction_stable only checks historical reversal. correction_resolution compares nominal correction with observed destination-bound changes in that direction; below_observed_change advises against chasing the exact delta, and no status guarantees accuracy or authorizes motion. Inspect refresh_comparison for changed views/bounds and compare current images before applying any correction; visible bounds do not prove container interior clearance."
            elif spec.name == "descend_and_close":
                additions = {"target_attachment_status": status_schema, "evidence_limitation": {"type": "string"}}
            elif spec.name == "lift_probe":
                additions = {"target_attachment_status": status_schema, "evidence_limitation": {"type": "string"}}
                description = "Lift vertically and report actual EEF displacement and aperture. supports_attachment is indirect gripper/motion evidence only; inspect the same target in pre/post images, then assert_entity_attached. A small successful probe may be insufficient for the current attachment verifier: it requires 0.04 to 0.08 m of measured target translation, depending on the initial observed target extent, including at least 0.025 m upward. These are evidence thresholds, not a requested lift height or motion clearance; original force, possession and motion guards still apply."
            elif spec.name == "assert_entity_attached":
                additions = {"target_attachment_status": status_schema, "target_attachment_basis": {"type": "string"}}
                description = "Compare public target displacement, height, appearance and proximity with gripper/lift evidence. The existing translation threshold is max(0.04, min(0.08, 0.25 * initial observed target diagonal)) metres, with at least 0.025 m upward target displacement. Displacement is relative to the initial tracked target, not the latest EEF command; successive small lifts can accumulate evidence. These necessary thresholds alone do not verify identity, attachment or safe transport. verified requires target motion support; aperture plus EEF lift alone is insufficient. If target_attachment_status is insufficient, inspect or re-mark the same target; do not infer physical failure from an uncertain partial cloud. pose_valid separately describes the usable attachment transform."
            elif spec.name == "close_gripper":
                schema = {"type": "object", "properties": {
                    "closure_executed": {"type": "boolean"}, "closure_converged": {"type": "boolean"},
                    "gripper_open_fraction": nullable_number, "object_detected": {"type": ["boolean", "null"]},
                    "force_n": nullable_number, "attachment_attempt_id": {"type": "integer", "minimum": 0},
                    "associated_tracking_handle": {"type": "null"},
                    "target_attachment_status": status_schema,
                    "previous_descent": {"type": ["object", "null"]}, "evidence_limitation": {"type": "string"}},
                    "required": ["closure_executed", "closure_converged", "object_detected", "target_attachment_status"],
                    "additionalProperties": False}
                description = "Close jaws at the current pose and return measured gripper feedback. This does not finish an obstructed descent. Old lift evidence is invalidated; target identity remains unconfirmed. Inspect previous_descent and current wrist views before another attempt."
            elif spec.name == "goto_pose":
                arguments_schema = dict(spec.arguments_schema)
                arguments_schema["properties"] = dict(arguments_schema["properties"])
                arguments_schema["properties"]["position_reference"] = {
                    "type": "string", "enum": ["tool", "held_target", "end_effector"],
                    "description": "Use end_effector for BOTH end_effector_position_m and end_effector_quaternion_wxyz from observe. Use tool for BOTH tool_position_m/tool_quaternion_wxyz or planned tool poses. held_target uses object center and TOOL quaternion plus tracking_handle. All positions are in robot-base coordinates, but local origins and axes differ."}
                spec = replace(spec, arguments_schema=arguments_schema)
                additions = {"input_pose_reference": {"type": "string", "enum": ["tool", "held_target", "end_effector"]},
                    "input_position_m": vector3, "input_quaternion_wxyz": quaternion}
                description = "Move a pose with an explicit reference. For a vertical retreat from observe().value['robot_state'], copy end_effector_position_m, add world Z clearance, retain end_effector_quaternion_wxyz, and pass position_reference='end_effector'. Those fields are NOT a tool-frame pose. Alternatively use tool_position_m/tool_quaternion_wxyz with position_reference='tool'. For held_target, supply object center, TOOL quaternion and verified tracking_handle; the attachment offset is applied. Keep approach_frame='world' for a waypoint above the target. By default z_approach moves directly to the approach waypoint, then to the target; it does not first lift at current XY or certify the travel corridor. When the runtime exposes clearance_first, that explicit option performs the initial vertical clearance within a shared budget. Inspect convergence/residuals, completed_phases, failed_phase and input_pose_reference. Unavailable robot pose measurements cannot establish convergence."
            elif spec.name == "observe":
                description += " robot_state supplies both end_effector and tool poses with matching *_pose_reference tags. Copy the matching position, quaternion AND reference into goto_pose; their local origins/axes differ despite a common robot-base world frame."
            if additions:
                schema["properties"] = {**schema.get("properties", {}), **additions}
            revised.append(replace(spec, result_schema=schema, description=description,
                return_type="GripperClosureEvidence" if spec.name == "close_gripper" else spec.return_type))
        definitions = revised
        def feedback_schema(schema):
            if isinstance(schema, dict):
                result = {key: feedback_schema(value) for key, value in schema.items()}
                properties = result.get("properties", {})
                if "backend_candidate" in properties:
                    properties["backend_candidate"]["x-decision-feedback"] = "on_demand"
                return result
            if isinstance(schema, list):
                return [feedback_schema(value) for value in schema]
            return schema
        definitions = [replace(spec, arguments_schema=feedback_schema(dict(spec.arguments_schema)),
                               result_schema=feedback_schema(dict(spec.result_schema)))
                       for spec in definitions]
    return CapabilityManifest(
        (
            "captain-capabilities/v13"
            if v15_protocol
            else (
                "captain-capabilities/v12"
                if completion_confidence
                else (
                    "captain-capabilities/v11"
                    if completion_evidence
                    else (
                        "captain-capabilities/v10"
                        if dehardcoded
                        else "captain-capabilities/v9"
                    )
                )
            )
        ),
        (
            (
                "capx-s2-pregrasp-identity/v15"
                if level is ToolApiLevel.S2_HIGH_LEVEL
                else "capx-s3-pregrasp-identity/v15"
            )
            if v15_protocol
            else (
                (
                    "capx-s2-confidence-review/v12"
                    if level is ToolApiLevel.S2_HIGH_LEVEL
                    else "capx-s3-confidence-review/v12"
                )
                if completion_confidence
                else (
                    (
                        "capx-s2-completion-mechanism/v11"
                        if level is ToolApiLevel.S2_HIGH_LEVEL
                        else "capx-s3-completion-mechanism/v11"
                    )
                    if completion_evidence
                    else (
                        (
                            "capx-s2-dehardcoded/v10"
                            if level is ToolApiLevel.S2_HIGH_LEVEL
                            else "capx-s3-dehardcoded/v10"
                        )
                        if dehardcoded
                        else (
                            "capx-s2-service-aligned/v9"
                            if level is ToolApiLevel.S2_HIGH_LEVEL
                            else "capx-s3-service-aligned/v9"
                        )
                    )
                )
            )
        ),
        tuple(definitions),
    )

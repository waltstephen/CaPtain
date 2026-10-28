from __future__ import annotations

from jsonschema import Draft202012Validator

from captain_harness.adapters.capx.manifest import ToolApiLevel, build_capx_tool_manifest


def test_high_level_manifest_has_exact_non_privileged_surface_and_docs() -> None:
    manifest = build_capx_tool_manifest(ToolApiLevel.S2_HIGH_LEVEL, joint_count=7)
    assert manifest.schema_version == "captain-capabilities/v6"
    assert manifest.provider == "capx-s2-nonprivileged/v6"
    assert manifest.names == (
        "get_observation",
        "mark_point",
        "track_entity_from_mark",
        "track_entity",
        "assert_entity_attached",
        "assert_placement_ready",
        "assert_spatial_relation",
        "assert_spatial_relation_from_marks",
        "get_object_pose",
        "sample_grasp_pose",
        "goto_pose",
        "open_gripper",
        "close_gripper",
        "home",
    )
    rendered = manifest.render_python_interface()
    assert "def get_object_pose(object_name: str, use_multiview: bool = True)" in rendered
    assert "-> CapabilityResult[ObjectPose]" in rendered
    assert 'get_object_pose("black bowl", use_multiview=True)' in rendered
    assert 'pose.value["position"]' in rendered
    assert "local negative tool Z" in rendered
    assert "then the final target motion" in rendered
    assert "converged=false" in rendered
    assert "grasp_style: str = 'auto'" in rendered
    assert "tracking_handle: str | None = None" in rendered
    assert "candidate_index: int = 0" in rendered
    assert "prevent a second language query from switching to a distractor" in rendered
    assert "either residual exceeding its tolerance makes the action fail" in rendered
    assert "def mark_point(" in rendered
    assert "def track_entity_from_mark(" in rendered
    assert "def assert_spatial_relation_from_marks(" in rendered
    assert "no simulator object pose, contact, reward, done flag, or task checker" in rendered
    assert "top_down_alignment" in rendered
    assert "check_task" not in rendered


def test_low_level_manifest_has_handle_based_composition_surface_and_docs() -> None:
    manifest = build_capx_tool_manifest(ToolApiLevel.S3_LOW_LEVEL, joint_count=7)
    assert manifest.schema_version == "captain-capabilities/v6"
    assert manifest.provider == "capx-s3-composition/v5"
    assert manifest.names == (
        "get_observation",
        "mark_point",
        "track_entity_from_mark",
        "track_entity",
        "assert_entity_attached",
        "assert_placement_ready",
        "assert_spatial_relation",
        "assert_spatial_relation_from_marks",
        "segment_text",
        "segment_point",
        "point_molmo",
        "construct_point_cloud",
        "plan_grasp_candidates",
        "get_grasp_candidate",
        "solve_ik",
        "move_to_joints",
        "goto_pose",
        "open_gripper",
        "close_gripper",
        "home",
    )
    rendered = manifest.render_python_interface()
    assert "rgb_handle: str" in rendered
    assert "mask_handles: list[str]" in rendered
    assert "point_cloud_handle" in rendered
    assert "Returns .value:" in rendered
    assert "masks: list[{" in rendered
    assert "mask_handle: str" in rendered
    assert "preferred first localization step" in rendered
    assert "same fresh observation" in rendered
    assert "never translate image-direction guesses" in rendered
    assert "moved from its pre-grasp centroid" in rendered
    assert "invalid same-entity localization" in rendered
    assert "Control semantic: attachment_assertion" in rendered
    assert "Control semantic: placement_assertion" in rendered
    assert "Control semantic: completion_assertion" in rendered
    assert "Only pose_valid permits pose-dependent transport" in rendered
    assert "grants an episode-local placement authorization" in rendered
    assert "post_object_name: str | None = None" in rendered
    assert 'candidate.value["position"]' in rendered
    assert "Read one ranked pose" in rendered
    assert "index 0" in rendered
    assert "grasp_style: str = 'auto'" in rendered
    assert "top_down_alignment" in rendered
    assert "local negative tool Z" in rendered
    assert "np.ndarray" not in rendered


def test_every_tool_capability_has_examples_and_valid_result_schema() -> None:
    for level in ToolApiLevel:
        manifest = build_capx_tool_manifest(level, joint_count=7)
        for spec in manifest.capabilities:
            assert spec.examples, spec.name
            assert spec.return_type != "object", spec.name
            Draft202012Validator.check_schema(dict(spec.arguments_schema))
            Draft202012Validator.check_schema(dict(spec.result_schema))


def test_v8_manifest_exposes_only_public_safe_release_surface() -> None:
    manifest = build_capx_tool_manifest(
        ToolApiLevel.S2_HIGH_LEVEL,
        joint_count=7,
        public_safe_release=True,
    )

    assert manifest.schema_version == "captain-capabilities/v8"
    assert manifest.provider == "capx-s2-public-safe/v8"
    assert "estimate_placement_relation" in manifest.names
    assert "authorize_safe_release" in manifest.names
    assert "assert_task_completion_public" in manifest.names
    assert "mark_attached_target" in manifest.names
    assert "assert_placement_ready" not in manifest.names
    assert "assert_spatial_relation" not in manifest.names
    assert "assert_spatial_relation_from_marks" not in manifest.names
    assert "track_entity" not in manifest.names
    rendered = manifest.render_python_interface()
    assert "never authorizes release" in rendered
    assert "does not decide whether the task is complete" in rendered


def test_v9_service_aligned_manifests_compare_action_abstraction_only() -> None:
    s3 = build_capx_tool_manifest(
        ToolApiLevel.S3_LOW_LEVEL,
        joint_count=7,
        service_aligned=True,
    )
    s2 = build_capx_tool_manifest(
        ToolApiLevel.S2_HIGH_LEVEL,
        joint_count=7,
        service_aligned=True,
    )

    assert s3.schema_version == "captain-capabilities/v9"
    assert s3.provider == "capx-s3-service-aligned/v9"
    assert s2.provider == "capx-s2-service-aligned/v9"
    assert set(s2.names) == set(s3.names) | {"pick", "place"}
    assert "pick" not in s3.names
    assert "place" not in s3.names
    for required in (
        "observe",
        "segment",
        "select_detection",
        "point",
        "grasp_pose_estimate",
        "plan_grasp_from_point_clouds",
        "placement_estimate",
        "construct_point_cloud",
        "subsample_point_cloud",
        "filter_noise",
        "gripper_state",
        "check_reachable",
        "solve_ik",
        "goto_pose",
        "approach_and_close",
        "servo_to",
        "lift_probe",
        "wrist_alignment",
        "move_to_joints",
        "open_gripper",
        "close_gripper",
        "home",
    ):
        assert required in s3.names
    attachment = s3.require("assert_entity_attached").result_schema
    assert "tracking_handle" in attachment["required"]
    assert "tracking_handle" in attachment["properties"]
    placement = s3.require("placement_estimate").result_schema
    assert "observation_handle" in placement["required"]
    assert "observation_handle" in placement["properties"]
    release = s3.require("authorize_safe_release").result_schema
    assert "observation_handle" in release["required"]
    assert "observation_handle" in release["properties"]


def test_v9_manifest_exposes_real_robot_equivalents_and_world_approach() -> None:
    manifest = build_capx_tool_manifest(
        ToolApiLevel.S3_LOW_LEVEL,
        joint_count=7,
        service_aligned=True,
    )

    assert all(
        capability.real_robot_equivalent.strip()
        and capability.real_robot_equivalent
        != "generic runtime capability"
        for capability in manifest.capabilities
    )
    rendered = manifest.render_python_interface()
    assert "approach_frame: str = 'world'" in rendered
    assert "calibrated public position and orientation tolerances" in rendered
    assert "residual above either tolerance is a hard failure" in rendered
    assert "Real-robot equivalent:" in rendered
    grasp = manifest.require("grasp_pose_estimate").result_schema
    candidate = grasp["properties"]["candidates"]["items"]
    assert candidate["properties"]["width_feasible"] == {"const": True}
    assert candidate["properties"]["reachable"] == {"const": True}


def test_v10_manifest_exposes_unfiltered_candidate_diagnostics() -> None:
    manifest = build_capx_tool_manifest(
        ToolApiLevel.S3_LOW_LEVEL,
        joint_count=7,
        service_aligned=True,
        dehardcoded=True,
    )

    assert manifest.schema_version == "captain-capabilities/v10"
    assert manifest.provider == "capx-s3-dehardcoded/v10"
    grasp_spec = manifest.require("grasp_pose_estimate")
    assert grasp_spec.arguments_schema["properties"]["backend"]["enum"] == [
        "auto",
        "contact_graspnet",
        "anygrasp",
        "graspgenx",
    ]
    assert (
        grasp_spec.arguments_schema["properties"]["top_k"].get("maximum")
        is None
    )
    result = grasp_spec.result_schema
    assert result["properties"]["filtered"] == {"const": 0}
    candidate = result["properties"]["candidates"]["items"]
    assert candidate["properties"]["predicted_grasp_width_m"] == {
        "anyOf": [
            {"type": "number"},
            {"type": "null"},
        ]
    }
    assert candidate["properties"]["width_feasible"] == {
        "type": "boolean"
    }
    assert candidate["properties"]["ik_feasible"] == {"type": "boolean"}
    assert candidate["properties"]["quality_flags"]["type"] == "array"


def test_v11_manifest_exposes_evidence_only_completion_and_gripper_identity() -> None:
    manifest = build_capx_tool_manifest(
        ToolApiLevel.S3_LOW_LEVEL,
        joint_count=7,
        service_aligned=True,
        dehardcoded=True,
        completion_evidence=True,
    )

    assert manifest.schema_version == "captain-capabilities/v11"
    assert manifest.provider == "capx-s3-completion-mechanism/v11"
    completion = manifest.require(
        "assert_task_completion_public"
    ).result_schema
    assert "evidence" in completion["required"]
    assert "public_relation_satisfied" not in completion["properties"]
    assert "recommendation" in completion["required"]
    assert completion["properties"]["recommendation"]["enum"] == [
        "satisfied",
        "insufficient",
    ]
    evidence = completion["properties"]["evidence"]
    assert {
        "xy_overlap_ratio",
        "gripper_released",
        "target_static_after_release_m",
        "support_gap_bounds_m",
    }.issubset(evidence["properties"])
    attachment = manifest.require("assert_entity_attached").result_schema
    assert "identity_confidence" in attachment["required"]
    assert "verified_with_identity_warning" not in (
        attachment["properties"]["reason"]["enum"]
    )


def test_v12_manifest_exposes_confidence_breakdown() -> None:
    manifest = build_capx_tool_manifest(
        ToolApiLevel.S3_LOW_LEVEL,
        joint_count=7,
        service_aligned=True,
        dehardcoded=True,
        completion_evidence=True,
        completion_confidence=True,
    )

    assert manifest.schema_version == "captain-capabilities/v12"
    assert manifest.provider == "capx-s3-confidence-review/v12"
    completion = manifest.require(
        "assert_task_completion_public"
    ).result_schema
    assert "confidence_components" in completion["required"]
    assert "confidence_penalty" in completion["required"]
    assert "public_relation_satisfied" not in completion["properties"]

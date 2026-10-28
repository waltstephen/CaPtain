from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pytest
from jsonschema import Draft202012Validator
from PIL import Image

from captain_harness.adapters.capx.artifacts import EpisodeArtifactRegistry
from captain_harness.adapters.capx.backend import (
    GraspServiceBatch,
    MaskCandidate,
    ToolCapabilityBackend,
    _Detection,
    _MarkedPoint,
)
from captain_harness.adapters.capx.manifest import ToolApiLevel, build_capx_tool_manifest
from captain_harness.adapters.capx.protocol import (
    load_public_completion_protocol,
)
from captain_harness.adapters.libero.types import CameraData, ObservationData, RobotState
from captain_harness.ports.broker import ValidatedCapabilityBroker
from captain_harness.ports.capability import CapabilityResult, FailureCategory


class FakeRobot:
    def __init__(self, root: Path) -> None:
        self.controller = "JOINT_POSITION"
        root.mkdir(parents=True, exist_ok=True)
        self.calls: list[tuple[str, object]] = []
        self.joints = [0.0] * 7
        self.pending_eef_position = (0.0, 0.0, 0.5)
        self.pending_eef_quaternion = (1.0, 0.0, 0.0, 0.0)
        cameras = []
        for index, name in enumerate(("agentview", "robot0_eye_in_hand")):
            rgb_path = root / f"{name}.png"
            depth_path = root / f"{name}.npy"
            Image.new("RGB", (4, 4), (20 + index, 30, 40)).save(rgb_path)
            np.save(depth_path, np.full((4, 4), 0.5 + index * 0.1, dtype=np.float32))
            intrinsics = (400.0, 0.0, 1.5, 0.0, 400.0, 1.5, 0.0, 0.0, 1.0)
            extrinsics = tuple(np.eye(4, dtype=np.float64).ravel())
            cameras.append(CameraData(name, str(rgb_path), str(depth_path), intrinsics, extrinsics))
        self.observation = ObservationData(
            tuple(cameras),
            RobotState(tuple(self.joints), (0.0, 0.0, 0.5), (1.0, 0.0, 0.0, 0.0), 1.0),
        )

    def get_observation_data(self) -> CapabilityResult:
        self.calls.append(("get_observation_data", None))
        return CapabilityResult.ok(self.observation)

    def get_robot_state(self) -> CapabilityResult:
        return CapabilityResult.ok(self.observation.robot_state)

    def move_to_joints(self, joints) -> CapabilityResult:
        self.calls.append(("move_to_joints", tuple(joints)))
        self.joints = list(joints)
        self.observation = ObservationData(
            self.observation.cameras,
            RobotState(
                tuple(self.joints),
                tuple(self.pending_eef_position),
                tuple(self.pending_eef_quaternion),
                self.observation.robot_state.gripper_open_fraction,
            ),
        )
        return CapabilityResult.ok(None)

    def move_to_pose(
        self,
        position,
        quaternion,
        *,
        position_only=False,
    ) -> CapabilityResult:
        self.calls.append(("move_to_pose", (tuple(position), tuple(quaternion))))
        self.pending_eef_position = tuple(float(value) for value in position)
        self.pending_eef_quaternion = tuple(float(value) for value in quaternion)
        self.observation = ObservationData(
            self.observation.cameras,
            RobotState(
                tuple(self.joints),
                self.pending_eef_position,
                self.pending_eef_quaternion,
                self.observation.robot_state.gripper_open_fraction,
            ),
        )
        return CapabilityResult.ok(None)

    @staticmethod
    def external_ik_target_from_eef_pose(position, quaternion):
        return tuple(position), tuple(quaternion)

    @staticmethod
    def eef_pose_from_external_ik_target(position, quaternion):
        return tuple(position), tuple(quaternion)

    def open_gripper(self) -> CapabilityResult:
        self.calls.append(("open_gripper", None))
        return CapabilityResult.ok(None)

    def close_gripper(self) -> CapabilityResult:
        self.calls.append(("close_gripper", None))
        return CapabilityResult.ok(None)

    def gripper_state(self) -> CapabilityResult:
        open_fraction = self.observation.robot_state.gripper_open_fraction
        return CapabilityResult.ok(
            {
                "open_fraction": open_fraction,
                "object_detected": 0.02 <= open_fraction < 0.90,
                "force_n": 0.0,
            }
        )

    def approach_and_close(
        self,
        position,
        quaternion,
        approach_direction_world,
        *,
        max_contact_force=None,
        approach_distance_m=0.06,
    ) -> CapabilityResult:
        self.calls.append(
            (
                "approach_and_close",
                (
                    tuple(position),
                    tuple(quaternion),
                    tuple(approach_direction_world),
                ),
            )
        )
        self.pending_eef_position = tuple(float(value) for value in position)
        self.pending_eef_quaternion = tuple(
            float(value) for value in quaternion
        )
        self.observation = ObservationData(
            self.observation.cameras,
            RobotState(
                tuple(self.joints),
                self.pending_eef_position,
                self.pending_eef_quaternion,
                0.05,
            ),
        )
        return CapabilityResult.ok(
            {
                "contacted": True,
                "contact_travel_m": 0.04,
                "gripper_open_fraction": 0.05,
                "object_detected": True,
                "stopped_reason": "contact_force",
                "position_residual_m": 0.002,
                "orientation_residual_rad": 0.03,
            }
        )

    def align_pregrasp(
        self,
        position,
        quaternion,
        approach_direction_world,
        *,
        approach_distance_m=0.06,
        clearance_m=0.09,
    ) -> CapabilityResult:
        self.calls.append(
            (
                "align_pregrasp",
                (
                    tuple(position),
                    tuple(quaternion),
                    tuple(approach_direction_world),
                ),
            )
        )
        self.pending_eef_position = tuple(float(value) for value in position)
        self.pending_eef_quaternion = tuple(
            float(value) for value in quaternion
        )
        return CapabilityResult.ok(
            {
                "converged": True,
                "phase_reached": "pregrasp",
                "failed_phase": None,
                "position_residual_m": 0.002,
                "orientation_residual_rad": 0.03,
                "convergence_reason": "three_phase_pregrasp_reached",
                "pregrasp_position_m": list(position),
                "hover_position_m": [
                    float(position[0]),
                    float(position[1]),
                    float(position[2]) + clearance_m,
                ],
            }
        )

    def descend_and_close(
        self,
        position,
        quaternion,
        *,
        max_contact_force=None,
    ) -> CapabilityResult:
        self.calls.append(
            (
                "descend_and_close",
                (tuple(position), tuple(quaternion)),
            )
        )
        self.observation = ObservationData(
            self.observation.cameras,
            RobotState(
                tuple(self.joints),
                tuple(float(value) for value in position),
                tuple(float(value) for value in quaternion),
                0.05,
            ),
        )
        return CapabilityResult.ok(
            {
                "contacted": True,
                "contact_travel_m": 0.04,
                "gripper_open_fraction": 0.05,
                "object_detected": True,
                "stopped_reason": "contact_force",
                "position_residual_m": 0.002,
                "orientation_residual_rad": 0.03,
            }
        )

    def servo_to(
        self,
        position,
        quaternion,
        *,
        max_duration_s,
        tolerance_m,
    ) -> CapabilityResult:
        self.calls.append(("servo_to", (tuple(position), tuple(quaternion))))
        self.pending_eef_position = tuple(float(value) for value in position)
        self.pending_eef_quaternion = tuple(
            float(value) for value in quaternion
        )
        return CapabilityResult.ok(
            {
                "reached": True,
                "final_error_m": 0.0,
                "iterations": 2,
                "aborted_reason": None,
            }
        )

    def home(self) -> CapabilityResult:
        self.calls.append(("home", None))
        return CapabilityResult.ok(None)


class FakeServices:
    def __init__(self, robot: FakeRobot | None = None) -> None:
        self.robot = robot
        self.calls: list[str] = []
        self.ik_previous = []
        self.ik_positions = []
        self.ik_quaternions = []

    def segment_text(self, image: np.ndarray, text: str) -> list[MaskCandidate]:
        self.calls.append("sam3_text")
        return [MaskCandidate(np.ones(image.shape[:2], dtype=bool), 0.9, (0, 0, 3, 3), text)]

    def segment_point(self, image: np.ndarray, point_xy) -> list[MaskCandidate]:
        self.calls.append("sam3_point")
        return [MaskCandidate(np.ones(image.shape[:2], dtype=bool), 0.8)]

    def point(self, image: np.ndarray, text: str):
        self.calls.append("molmo")
        return (1.0, 1.0)

    def plan_grasps(self, full_points: np.ndarray, object_points: np.ndarray):
        self.calls.append("contact_graspnet")
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = object_points.mean(axis=0)
        return np.stack([pose]), np.asarray([0.75])

    def predict_grasps(
        self,
        *,
        backend,
        image,
        depth,
        mask,
        intrinsics,
        extrinsics,
        gripper_profile,
        depth_source,
        collision_relaxed=False,
    ):
        self.calls.append(str(backend))
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = [0.0, 0.0, 0.5]
        native = {
            "id": "grasp_000",
            "frame": "camera",
            "camera_frame": "opencv",
            "grasp_frame": "graspnet",
            "score": 0.75,
            "translation_xyz": [0.0, 0.0, 0.5],
            "rotation_matrix": np.eye(3).tolist(),
            "gripper_tip_position_xyz": [0.05, 0.0, 0.5],
            "depth": 0.05,
            "width": 0.04,
            "height": 0.03,
        }
        return GraspServiceBatch(
            poses=np.stack([pose]),
            scores=np.asarray([0.75]),
            widths_m=(None,),
            backend=(
                "contact_graspnet"
                if backend == "auto"
                else str(backend)
            ),
            public_tool_convention=False,
            grasp_frame="capx_native",
            native_candidates=(native,),
            depth_quality={"source": depth_source},
        )

    def predict_placements(
        self,
        *,
        image,
        depth,
        object_mask,
        placement_region_mask,
        intrinsics,
        extrinsics,
        selected_grasp,
        depth_source,
    ):
        self.calls.append("anyplace")
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = [0.2, 0.0, 0.5]
        return GraspServiceBatch(
            poses=np.stack([pose]),
            scores=np.asarray([0.65]),
            widths_m=(0.04,),
            backend="anyplace",
            public_tool_convention=True,
            grasp_frame="capx_native",
            native_candidates=(
                {
                    "id": "placement_000",
                    "place_grasp_pose": dict(selected_grasp),
                },
            ),
            depth_quality={"source": depth_source},
        )

    def solve_ik(self, position, quaternion, previous_joints):
        self.calls.append("pyroki")
        self.ik_previous.append(previous_joints)
        self.ik_positions.append(list(position))
        self.ik_quaternions.append(list(quaternion))
        if self.robot is not None:
            self.robot.pending_eef_position = tuple(float(value) for value in position)
            self.robot.pending_eef_quaternion = tuple(
                float(value) for value in quaternion
            )
        return np.asarray([0.1] * 7 + [0.04])


@dataclass
class Fixture:
    backend: ToolCapabilityBackend
    robot: FakeRobot
    services: FakeServices


def _fixture(
    tmp_path: Path,
    *,
    dehardcoded: bool = False,
    completion_evidence: bool = False,
    v15_protocol: bool = False,
) -> Fixture:
    robot = FakeRobot(tmp_path)
    services = FakeServices(robot)
    backend = ToolCapabilityBackend(
        robot=robot,
        services=services,
        registry=EpisodeArtifactRegistry("episode-test", ttl_s=60),
        dehardcoded=dehardcoded,
        v15_protocol=v15_protocol,
        completion_protocol=(
            load_public_completion_protocol(
                (
                    "configs/protocols/v12_public_completion_v1.json"
                    if v15_protocol
                    else "configs/protocols/v11_public_completion_v1.json"
                )
            )
            if completion_evidence or v15_protocol
            else None
        ),
    )
    return Fixture(backend, robot, services)


def _invoke(
    fixture: Fixture,
    level: ToolApiLevel,
    name: str,
    arguments: Mapping[str, object],
    *,
    public_safe_release: bool = False,
    service_aligned: bool = False,
    dehardcoded: bool = False,
    completion_evidence: bool = False,
    v15_protocol: bool = False,
    motion_tools: bool = False,
):
    manifest = build_capx_tool_manifest(
        level,
        joint_count=7,
        public_safe_release=public_safe_release,
        service_aligned=service_aligned,
        dehardcoded=dehardcoded,
        completion_evidence=completion_evidence or v15_protocol,
        completion_confidence=v15_protocol,
        v15_protocol=v15_protocol,
    )
    if motion_tools:
        from captain_harness.adapters.capx.runtime_manifest import runtime_manifest
        manifest = runtime_manifest(manifest, controller='OSC_POSE', motion_tools=True)
    broker = ValidatedCapabilityBroker(manifest, fixture.backend.invoke)
    result = broker.invoke(name, arguments)
    if result.success:
        Draft202012Validator(manifest.require(name).result_schema).validate(result.value)
    return result


def _v9_selected_detection(
    fixture: Fixture,
    *,
    prompt: str = "bowl",
    v15_protocol: bool = False,
) -> tuple[dict[str, object], dict[str, object]]:
    observation = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "observe",
        {},
        service_aligned=True,
        dehardcoded=v15_protocol,
        v15_protocol=v15_protocol,
    ).value
    segmented = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "segment",
        {
            "prompt": prompt,
            "observation_handle": observation["observation_handle"],
            "camera_name": "agentview",
        },
        service_aligned=True,
        dehardcoded=v15_protocol,
        v15_protocol=v15_protocol,
    ).value
    selected = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "select_detection",
        {"detection_id": segmented["detections"][0]["detection_id"]},
        service_aligned=True,
        dehardcoded=v15_protocol,
        v15_protocol=v15_protocol,
    ).value
    return observation, selected


def test_high_level_capabilities_are_sensor_derived_and_schema_valid(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    level = ToolApiLevel.S2_HIGH_LEVEL
    calls = (
        ("get_observation", {}),
        ("get_object_pose", {"object_name": "bowl", "use_multiview": True}),
        ("sample_grasp_pose", {"object_name": "bowl", "use_multiview": True}),
        ("goto_pose", {"position": [0.1, 0.0, 0.4], "quaternion": [1, 0, 0, 0], "z_approach": 0.05}),
        ("open_gripper", {}),
        ("close_gripper", {}),
        ("home", {}),
    )
    for name, arguments in calls:
        result = _invoke(fixture, level, name, arguments)
        assert result.success, (name, result)
    assert {"molmo", "sam3_point", "contact_graspnet", "pyroki"}.issubset(
        fixture.services.calls
    )


def test_low_level_capabilities_use_json_safe_handles_and_validate_outputs(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    level = ToolApiLevel.S3_LOW_LEVEL
    observation = _invoke(fixture, level, "get_observation", {}).value
    rgb_handle = observation["cameras"][0]["rgb_handle"]
    segmented = _invoke(
        fixture, level, "segment_text", {"rgb_handle": rgb_handle, "text_prompt": "bowl"}
    ).value
    point_segmented = _invoke(
        fixture, level, "segment_point", {"rgb_handle": rgb_handle, "point_xy": [1, 1]}
    ).value
    point = _invoke(
        fixture, level, "point_molmo", {"rgb_handle": rgb_handle, "text_prompt": "bowl"}
    )
    mask_handles = [segmented["masks"][0]["mask_handle"], point_segmented["masks"][0]["mask_handle"]]
    clouds = _invoke(
        fixture,
        level,
        "construct_point_cloud",
        {
            "observation_handle": observation["observation_handle"],
            "mask_handles": mask_handles,
            "use_multiview": True,
        },
    ).value
    grasps = _invoke(
        fixture,
        level,
        "plan_grasp_candidates",
        {
            "full_point_cloud_handle": clouds["full_point_cloud_handle"],
            "object_point_cloud_handle": clouds["object_point_cloud_handle"],
            "top_k": 3,
        },
    )
    calls = (
        ("solve_ik", {"position": [0.1, 0.0, 0.4], "quaternion": [1, 0, 0, 0]}),
        ("move_to_joints", {"joints": [0.1] * 7}),
        ("goto_pose", {"position": [0.1, 0.0, 0.4], "quaternion": [1, 0, 0, 0], "z_approach": 0}),
        ("open_gripper", {}),
        ("close_gripper", {}),
        ("home", {}),
    )
    assert point.success and grasps.success
    for name, arguments in calls:
        assert _invoke(fixture, level, name, arguments).success, name
    assert isinstance(grasps.value["grasp_set_handle"], str)
    assert len(clouds["object_centroid_m"]) == 3
    assert len(clouds["object_bounds_min_m"]) == 3
    assert len(clouds["object_bounds_max_m"]) == 3
    candidate = _invoke(
        fixture,
        level,
        "get_grasp_candidate",
        {"grasp_set_handle": grasps.value["grasp_set_handle"], "index": 0},
    )
    assert candidate.success
    assert candidate.value["score"] == 0.75
    assert len(candidate.value["position"]) == 3
    assert len(candidate.value["quaternion"]) == 4
    assert _invoke(
        fixture,
        level,
        "solve_ik",
        {
            "position": candidate.value["position"],
            "quaternion": candidate.value["quaternion"],
        },
    ).success
    stored = fixture.backend.registry.get(
        grasps.value["grasp_set_handle"], expected_kind="grasp_set"
    )
    assert abs(stored["poses"][0, 2, 3] - 0.62) < 1e-12

    missing = _invoke(
        fixture,
        level,
        "get_grasp_candidate",
        {"grasp_set_handle": grasps.value["grasp_set_handle"], "index": 1},
    )
    assert missing.failure_category is FailureCategory.INVALID_ARGUMENT


def test_v9_requires_explicit_detection_selection_before_grasp_estimation(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    observation = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "observe",
        {},
        service_aligned=True,
    ).value
    segmented = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "segment",
        {
            "prompt": "bowl",
            "observation_handle": observation["observation_handle"],
            "camera_name": "agentview",
        },
        service_aligned=True,
    ).value
    hand = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "gripper_state",
        {},
        service_aligned=True,
    ).value

    result = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "grasp_pose_estimate",
        {
            "observation_handle": observation["observation_handle"],
            "detection_id": segmented["detections"][0]["detection_id"],
            "gripper_profile": hand["profile"],
        },
        service_aligned=True,
    )

    assert result.failure_category is FailureCategory.PERCEPTION_UNCERTAIN
    assert "select_detection" in result.message


def test_v9_grasp_estimate_returns_only_width_feasible_service_candidates(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    observation, selected = _v9_selected_detection(fixture)
    hand = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "gripper_state",
        {},
        service_aligned=True,
    ).value

    result = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "grasp_pose_estimate",
        {
            "observation_handle": observation["observation_handle"],
            "detection_id": selected["detection_id"],
            "gripper_profile": hand["profile"],
            "grasp_style": "auto",
            "top_k": 5,
        },
        service_aligned=True,
    )

    assert result.success, result
    assert result.value["candidate_count"] >= 1
    assert {
        candidate["backend"]
        for candidate in result.value["candidates"]
    } == {"contact_graspnet"}
    assert all(
        candidate["width_feasible"] is True
        and candidate["predicted_grasp_width_m"]
        <= hand["profile"]["max_width_m"]
        and candidate["reachable"] is True
        for candidate in result.value["candidates"]
    )


def test_v9_width_infeasible_candidates_fail_without_fabricated_fallback(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    observation, selected = _v9_selected_detection(fixture)
    fixture.backend._predicted_grasp_width = (  # type: ignore[method-assign]
        lambda pose, object_points: 0.12
    )
    hand = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "gripper_state",
        {},
        service_aligned=True,
    ).value

    result = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "grasp_pose_estimate",
        {
            "observation_handle": observation["observation_handle"],
            "detection_id": selected["detection_id"],
            "gripper_profile": hand["profile"],
        },
        service_aligned=True,
    )

    assert not result.success
    assert result.value["reason"] == "no_candidate"
    assert result.value["candidates_returned"] == 1
    assert result.value["rejected"]["width_infeasible"] == 1
    assert "fallback" not in result.message


def test_v10_width_unknown_candidate_is_returned_with_quality_flag(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, dehardcoded=True)
    observation, selected = _v9_selected_detection(fixture)
    fixture.backend._predicted_grasp_width = (  # type: ignore[method-assign]
        lambda pose, object_points: None
    )
    hand = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "gripper_state",
        {},
        service_aligned=True,
        dehardcoded=True,
    ).value

    result = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "grasp_pose_estimate",
        {
            "observation_handle": observation["observation_handle"],
            "detection_id": selected["detection_id"],
            "gripper_profile": hand["profile"],
            "backend": "contact_graspnet",
            "depth_source": "sensor",
            "grasp_style": "top_down",
            "top_k": 15,
        },
        service_aligned=True,
        dehardcoded=True,
    )

    assert result.success, result
    assert result.value["filtered"] == 0
    candidate = result.value["candidates"][0]
    assert candidate["predicted_grasp_width_m"] is None
    assert candidate["width_known"] is False
    assert candidate["width_feasible"] is True
    assert "width_unknown" in candidate["quality_flags"]
    assert "style_preference_low" not in candidate["quality_flags"]


def test_v10_grasp_ranking_preserves_native_candidate_binding(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, dehardcoded=True)
    observation, selected = _v9_selected_detection(fixture)

    def ranked_batch(**_arguments):
        first = np.eye(4, dtype=np.float64)
        first[:3, 3] = [0.1, 0.0, 0.5]
        second = np.eye(4, dtype=np.float64)
        second[:3, 3] = [0.2, 0.0, 0.5]
        return GraspServiceBatch(
            poses=np.stack([first, second]),
            scores=np.asarray([0.9, 0.2]),
            widths_m=(0.04, 0.04),
            backend="anygrasp",
            public_tool_convention=True,
            approach_directions_base=(
                (1.0, 0.0, 0.1),
                (0.0, 0.0, -1.0),
            ),
            native_candidates=(
                {"id": "low-style"},
                {"id": "high-style"},
            ),
        )

    fixture.services.predict_grasps = ranked_batch  # type: ignore[method-assign]
    hand = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "gripper_state",
        {},
        service_aligned=True,
        dehardcoded=True,
    ).value

    result = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "grasp_pose_estimate",
        {
            "observation_handle": observation["observation_handle"],
            "detection_id": selected["detection_id"],
            "gripper_profile": hand["profile"],
            "backend": "anygrasp",
            "depth_source": "sensor",
            "grasp_style": "top_down",
            "top_k": 2,
        },
        service_aligned=True,
        dehardcoded=True,
    )

    assert result.success, result
    candidates = result.value["candidates"]
    assert candidates[0]["position_m"] == [0.2, 0.0, 0.5]
    assert candidates[0]["backend_candidate"]["id"] == "high-style"
    assert candidates[1]["backend_candidate"]["id"] == "low-style"


def test_v10_backend_input_shortage_is_reported_as_no_candidate(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, dehardcoded=True)
    observation, selected = _v9_selected_detection(fixture)

    class InputShortage(OSError):
        reason = "insufficient_object_points"
        details = {
            "metadata": {
                "object_point_count": 0,
                "min_object_points": 100,
            }
        }

    def fail_prediction(**_arguments):
        raise InputShortage("OpenETA service failed")

    fixture.services.predict_grasps = fail_prediction  # type: ignore[method-assign]
    hand = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "gripper_state",
        {},
        service_aligned=True,
        dehardcoded=True,
    ).value

    result = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "grasp_pose_estimate",
        {
            "observation_handle": observation["observation_handle"],
            "detection_id": selected["detection_id"],
            "gripper_profile": hand["profile"],
            "backend": "graspgenx",
            "depth_source": "sensor",
            "grasp_style": "auto",
            "top_k": 5,
        },
        service_aligned=True,
        dehardcoded=True,
    )

    assert not result.success
    assert result.value["reason"] == "no_candidate"
    assert result.value["candidate_diagnostics"][0][
        "backend_reason"
    ] == "insufficient_object_points"


def test_v10_out_of_range_width_and_missing_ik_are_diagnostics(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, dehardcoded=True)
    observation, selected = _v9_selected_detection(fixture)
    fixture.backend._predicted_grasp_width = (  # type: ignore[method-assign]
        lambda pose, object_points: 0.12
    )
    fixture.services.solve_ik = (  # type: ignore[method-assign]
        lambda position, quaternion, previous_joints: None
    )
    hand = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "gripper_state",
        {},
        service_aligned=True,
        dehardcoded=True,
    ).value

    result = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "grasp_pose_estimate",
        {
            "observation_handle": observation["observation_handle"],
            "detection_id": selected["detection_id"],
            "gripper_profile": hand["profile"],
            "backend": "auto",
            "depth_source": "sensor",
            "grasp_style": "auto",
            "top_k": 15,
        },
        service_aligned=True,
        dehardcoded=True,
    )

    assert result.success, result
    candidate = result.value["candidates"][0]
    assert candidate["width_known"] is True
    assert candidate["width_feasible"] is False
    assert candidate["ik_feasible"] is False
    assert candidate["reachable"] is False
    assert candidate["predicted_joint_travel_rad"] is None
    assert set(candidate["quality_flags"]) >= {
        "width_out_of_range",
        "ik_unavailable",
    }

    executed = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "approach_and_close",
        {"candidate": candidate},
        service_aligned=True,
        dehardcoded=True,
    )
    assert executed.success
    assert executed.value["candidate_width_feasible"] is False
    assert executed.value["candidate_ik_feasible"] is False


def test_v9_approach_and_close_uses_same_tcp_calibration_as_goto_pose(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture.robot.eef_pose_from_external_ik_target = (  # type: ignore[method-assign]
        lambda position, quaternion: (
            (
                float(position[0]),
                float(position[1]),
                float(position[2]) + 0.2,
            ),
            tuple(quaternion),
        )
    )
    candidate = {
        "candidate_index": 0,
        "position_m": [0.4, 0.0, 0.4],
        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "model_score": 0.8,
        "predicted_grasp_width_m": 0.04,
        "width_feasible": True,
        "approach_direction_world": [0.0, 0.0, 1.0],
        "reachable": True,
        "predicted_joint_travel_rad": 0.5,
        "backend": "contact_graspnet",
        "tracking_handle": "artifact://v1/test/track",
        "detection_id": "artifact://v1/test/detection",
    }

    result = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "approach_and_close",
        {"candidate": candidate},
        service_aligned=True,
    )

    assert result.success
    call = next(
        value
        for name, value in fixture.robot.calls
        if name == "approach_and_close"
    )
    assert call[0] == (0.4, 0.0, 0.5)


def test_v15_known_oversize_grasp_is_rejected_before_motion(tmp_path):
    fixture = _fixture(tmp_path, dehardcoded=True, v15_protocol=True)
    result = fixture.backend.align_pregrasp({"width_feasible": False,
        "predicted_grasp_width_m": 0.10, "convergence_risk": "low"})
    assert not result.success and not result.executed
    assert result.value["reason"] == "grasp_width_infeasible"
    assert not fixture.robot.calls


@pytest.mark.parametrize("yaw", [0.0, np.pi / 2])
def test_v15_object_center_waypoint_preserves_nonzero_attachment_offset(tmp_path, yaw):
    from captain_harness.adapters.capx.backend import _TrackedEntity
    fixture = _fixture(tmp_path, dehardcoded=True, v15_protocol=True)
    fixture.robot.controller = "OSC_POSE"
    transform = np.eye(4)
    transform[:3, 3] = [0.03, -0.02, -0.04]
    tracked = _TrackedEntity('public target', np.zeros(3), np.zeros(3), np.ones(3)*.04,
        3, np.zeros((3,3)), np.zeros((3,3)), eef_to_object_transform=transform,
        attachment_verified=True)
    handle = fixture.backend.registry.put('entity_track', tracked)
    fixture.backend._active_attachment_handle = handle
    rejected = fixture.backend.goto_pose([0.4, .2, .3], [1, 0, 0, 0])
    assert not rejected.success and not rejected.executed
    result = _invoke(fixture, ToolApiLevel.S3_LOW_LEVEL, 'goto_pose', {
        'position': [.4, .2, .3], 'quaternion': [np.cos(yaw/2), 0, 0, np.sin(yaw/2)],
        'position_reference': 'held_target', 'tracking_handle': handle},
        service_aligned=True, dehardcoded=True, v15_protocol=True)
    assert result.success, result
    actual_eef = np.asarray(fixture.robot.observation.robot_state.end_effector_position_m)
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    assert np.allclose(actual_eef + rotation @ transform[:3, 3], [.4, .2, .3])
    tracked.eef_to_object_transform = None
    refused = fixture.backend.goto_pose([.5, .2, .3], [1,0,0,0],
        position_reference='held_target', tracking_handle=handle)
    assert not refused.success and not refused.executed


def test_v15_split_pregrasp_preserves_candidate_and_attachment_attempt(
    tmp_path: Path,
) -> None:
    fixture = _fixture(
        tmp_path,
        dehardcoded=True,
        v15_protocol=True,
    )
    observation, selected = _v9_selected_detection(
        fixture,
        v15_protocol=True,
    )
    hand = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "gripper_state",
        {},
        service_aligned=True,
        dehardcoded=True,
        v15_protocol=True,
    )
    grasps = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "grasp_pose_estimate",
        {
            "observation_handle": observation["observation_handle"],
            "detection_id": selected["detection_id"],
            "gripper_profile": hand.value["profile"],
            "backend": "auto",
            "depth_source": "sensor",
            "grasp_style": "auto",
            "top_k": 5,
        },
        service_aligned=True,
        dehardcoded=True,
        v15_protocol=True,
    )
    candidate = grasps.value["candidates"][0]

    missing = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "descend_and_close",
        {},
        service_aligned=True,
        dehardcoded=True,
        v15_protocol=True,
    )
    assert missing.failure_category is FailureCategory.NOT_EXECUTED
    assert missing.value["reason"] == "pregrasp_alignment_required"

    aligned = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "align_pregrasp",
        {"candidate": candidate},
        service_aligned=True,
        dehardcoded=True,
        v15_protocol=True,
    )
    assert aligned.success
    assert aligned.value["converged"] is True
    assert aligned.value["convergence_risk"] in {
        "low",
        "medium",
        "high",
    }

    closed = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "descend_and_close",
        {},
        service_aligned=True,
        dehardcoded=True,
        v15_protocol=True,
    )
    assert closed.success
    assert closed.value["tracking_handle"] == candidate["tracking_handle"]
    assert closed.value["object_detected"] is True
    assert fixture.backend._attachment_attempt_id == 1
    # The pre-descent centering probe always reports its outcome; when the
    # fresh wrist view matches the tracked centroid, no correction applies.
    centering = closed.value["pre_descent_centering"]
    assert isinstance(centering, dict)
    assert "reason" in centering
    applied = centering.get("applied_m")
    if applied is not None:
        assert float(np.linalg.norm(np.asarray(applied))) <= 0.03 + 1e-9


def test_v15_top_down_alignment_sign_and_convergence_risk(
    tmp_path: Path,
) -> None:
    fixture = _fixture(
        tmp_path,
        dehardcoded=True,
        v15_protocol=True,
    )
    observation, selected = _v9_selected_detection(
        fixture,
        v15_protocol=True,
    )

    def top_down_grasp(**_kwargs):
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = [0.0, 0.0, 0.5]
        return GraspServiceBatch(
            poses=np.stack([pose]),
            scores=np.asarray([0.9]),
            widths_m=(0.04,),
            backend="contact_graspnet",
            public_tool_convention=True,
            grasp_frame="capx_native",
            approach_directions_base=((0.0, 0.0, -1.0),),
            native_candidates=({"id": "top"},),
            depth_quality={"source": "sensor"},
        )

    fixture.services.predict_grasps = top_down_grasp  # type: ignore[method-assign]
    hand = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "gripper_state",
        {},
        service_aligned=True,
        dehardcoded=True,
        v15_protocol=True,
    )
    result = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "grasp_pose_estimate",
        {
            "observation_handle": observation["observation_handle"],
            "detection_id": selected["detection_id"],
            "gripper_profile": hand.value["profile"],
            "backend": "contact_graspnet",
            "depth_source": "sensor",
            "grasp_style": "top_down",
            "top_k": 1,
        },
        service_aligned=True,
        dehardcoded=True,
        v15_protocol=True,
    )

    assert result.success
    candidate = result.value["candidates"][0]
    assert candidate["top_down_alignment"] == 1.0
    # A well-aligned top-down grasp lowers the diagnostic risk level; the
    # V15 mainline shipped the opposite direction and mis-ranked candidates.
    assert candidate["convergence_risk"] in {"low", "medium"}


def test_v15_synthesizes_mask_topdown_when_services_have_no_candidate(
    tmp_path: Path,
) -> None:
    fixture = _fixture(
        tmp_path,
        dehardcoded=True,
        v15_protocol=True,
    )
    observation, selected = _v9_selected_detection(
        fixture,
        v15_protocol=True,
    )

    class NoCandidate(OSError):
        reason = "no_grasp_candidates"

    def no_candidate(**_kwargs):
        raise NoCandidate("no candidate")

    fixture.services.predict_grasps = no_candidate  # type: ignore[method-assign]
    hand = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "gripper_state",
        {},
        service_aligned=True,
        dehardcoded=True,
        v15_protocol=True,
    )
    result = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "grasp_pose_estimate",
        {
            "observation_handle": observation["observation_handle"],
            "detection_id": selected["detection_id"],
            "gripper_profile": hand.value["profile"],
            "backend": "contact_graspnet",
            "depth_source": "sensor",
            "grasp_style": "auto",
            "top_k": 5,
        },
        service_aligned=True,
        dehardcoded=True,
        v15_protocol=True,
    )

    assert result.success
    candidate = result.value["candidates"][0]
    assert candidate["backend"] == "mask_topdown_fallback"
    assert candidate["approach_direction_world"] == [0.0, 0.0, -1.0]
    assert candidate["top_down_alignment"] == 1.0
    assert candidate["convergence_risk"] in {"low", "medium", "high"}


def test_v15_multi_instance_selection_requires_public_point_provenance(
    tmp_path: Path,
) -> None:
    fixture = _fixture(
        tmp_path,
        dehardcoded=True,
        v15_protocol=True,
    )
    registry = fixture.backend.registry
    observation_handle = registry.put("observation", object())
    prompt = "black bowl next to the ramekin"
    selected_mask = np.zeros((10, 10), dtype=bool)
    selected_mask[1:4, 1:4] = True
    distractor_mask = np.zeros((10, 10), dtype=bool)
    distractor_mask[6:9, 6:9] = True

    def detection(mask, centroid):
        return _Detection(
            observation_handle=observation_handle,
            camera_name="agentview",
            prompt=prompt,
            label="black bowl",
            mask=mask,
            score=0.9,
            box=None,
            backend="sam3_text",
            centroid=np.asarray(centroid, dtype=np.float64),
            bounds_min=np.asarray(centroid, dtype=np.float64) - 0.05,
            bounds_max=np.asarray(centroid, dtype=np.float64) + 0.05,
            point_count=100,
            appearance_rgb_mean=np.asarray(
                [40.0, 40.0, 40.0],
                dtype=np.float64,
            ),
        )

    selected_id = registry.put(
        "detection",
        detection(selected_mask, [0.4, 0.2, 0.05]),
    )
    distractor_id = registry.put(
        "detection",
        detection(distractor_mask, [0.6, -0.1, 0.05]),
    )
    fixture.backend._detection_groups[
        (observation_handle, "agentview", prompt)
    ] = (selected_id, distractor_id)

    bare = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "select_detection",
        {"detection_id": selected_id},
        service_aligned=True,
        dehardcoded=True,
        v15_protocol=True,
    )
    assert bare.failure_category is FailureCategory.PERCEPTION_UNCERTAIN
    assert (
        bare.value["reason"]
        == "ambiguous_detection_requires_identity_provenance"
    )

    point_handle = registry.put(
        "marked_point",
        _MarkedPoint(
            observation_handle=observation_handle,
            camera_name="agentview",
            rgb_handle=registry.put("rgb", object()),
            depth_handle=registry.put("depth", np.ones((10, 10))),
            x_norm=0.2,
            y_norm=0.2,
            pixel_xy=(2.0, 2.0),
            position=np.asarray([0.4, 0.2, 0.05]),
            label=prompt,
            mask=selected_mask,
            segmentation_score=0.9,
            mask_area_fraction=0.09,
            depth_continuity=1.0,
            geometry_diagonal_m=0.1,
            appearance_rgb_mean=np.asarray([40.0, 40.0, 40.0]),
            centroid=np.asarray([0.4, 0.2, 0.05]),
            bounds_min=np.asarray([0.35, 0.15, 0.0]),
            bounds_max=np.asarray([0.45, 0.25, 0.1]),
            point_count=100,
            grounding_status="grounded",
        ),
    )
    # A proximity relation additionally requires a measured reference point.
    missing_reference = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "select_detection",
        {
            "detection_id": selected_id,
            "identity_point_handle": point_handle,
            "selection_relation": "nearest_xy",
        },
        service_aligned=True,
        dehardcoded=True,
        v15_protocol=True,
    )
    assert (
        missing_reference.failure_category
        is FailureCategory.PERCEPTION_UNCERTAIN
    )
    assert (
        missing_reference.value["reason"]
        == "proximity_relation_requires_reference_point"
    )

    def reference_point(position, pixel_xy):
        reference_mask = np.zeros((10, 10), dtype=bool)
        reference_mask[
            int(pixel_xy[1]) : int(pixel_xy[1]) + 1,
            int(pixel_xy[0]) : int(pixel_xy[0]) + 1,
        ] = True
        return registry.put(
            "marked_point",
            _MarkedPoint(
                observation_handle=observation_handle,
                camera_name="agentview",
                rgb_handle=registry.put("rgb", object()),
                depth_handle=registry.put("depth", np.ones((10, 10))),
                x_norm=pixel_xy[0] / 10.0,
                y_norm=pixel_xy[1] / 10.0,
                pixel_xy=(float(pixel_xy[0]), float(pixel_xy[1])),
                position=np.asarray(position, dtype=np.float64),
                label="ramekin",
                mask=reference_mask,
                segmentation_score=0.9,
                mask_area_fraction=0.01,
                depth_continuity=1.0,
                geometry_diagonal_m=0.05,
                appearance_rgb_mean=np.asarray([200.0, 200.0, 200.0]),
                centroid=np.asarray(position, dtype=np.float64),
                bounds_min=np.asarray(position, dtype=np.float64) - 0.02,
                bounds_max=np.asarray(position, dtype=np.float64) + 0.02,
                point_count=20,
                grounding_status="grounded",
            ),
        )

    # Reference near the distractor: the selected instance is not closest.
    wrong_reference = reference_point([0.62, -0.12, 0.05], (5, 5))
    not_closest = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "select_detection",
        {
            "detection_id": selected_id,
            "identity_point_handle": point_handle,
            "selection_relation": "nearest_xy",
            "reference_point_handle": wrong_reference,
        },
        service_aligned=True,
        dehardcoded=True,
        v15_protocol=True,
    )
    assert (
        not_closest.failure_category
        is FailureCategory.PERCEPTION_UNCERTAIN
    )
    assert not_closest.value["reason"] == "selected_instance_not_closest"
    assert not_closest.value["closest_detection_id"] == distractor_id

    good_reference = reference_point([0.45, 0.25, 0.05], (5, 5))
    selected = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "select_detection",
        {
            "detection_id": selected_id,
            "identity_point_handle": point_handle,
            "selection_relation": "nearest_xy",
            "reference_point_handle": good_reference,
        },
        service_aligned=True,
        dehardcoded=True,
        v15_protocol=True,
    )

    assert selected.success
    assert selected.value["identity_provenance_confirmed"] is True
    assert selected.value["distractor_detection_id"] == distractor_id
    distances = selected.value["instance_reference_distances_m"]
    assert distances[selected_id] < distances[distractor_id]
    assert selected.value["instance_selection_margin_ratio"] > 1.25
    assert selected.value["instance_margin_low"] is False


def test_empty_close_routes_next_grasp_estimate_to_mask_candidate(
    tmp_path: Path,
) -> None:
    fixture = _fixture(
        tmp_path,
        dehardcoded=True,
        v15_protocol=True,
    )
    observation, selected = _v9_selected_detection(
        fixture,
        v15_protocol=True,
    )
    hand = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "gripper_state",
        {},
        service_aligned=True,
        dehardcoded=True,
        v15_protocol=True,
    )

    def estimate():
        return _invoke(
            fixture,
            ToolApiLevel.S3_LOW_LEVEL,
            "grasp_pose_estimate",
            {
                "observation_handle": observation["observation_handle"],
                "detection_id": selected["detection_id"],
                "gripper_profile": hand.value["profile"],
                "backend": "auto",
                "depth_source": "sensor",
                "grasp_style": "auto",
                "top_k": 5,
            },
            service_aligned=True,
            dehardcoded=True,
            v15_protocol=True,
        )

    first = estimate()
    assert first.success
    assert (
        first.value["candidates"][0]["backend"]
        != "mask_topdown_fallback"
    )

    # A recorded close-on-air routes the retry to the mask-geometry
    # candidate, whose grasp point is the segmentation centroid. Matching
    # is by object identity (prompt + nearby centroid) because every grasp
    # estimate re-tracks the target under a fresh handle.
    fixture.backend._last_empty_close = {
        "backend": "contact_graspnet",
        "object_name": "a differently worded description",
        "source_detection_id": selected["detection_id"],
        "centroid": np.asarray(selected["centroid_m"]),
    }
    retry = estimate()
    assert retry.success
    candidate = retry.value["candidates"][0]
    assert candidate["backend"] == "mask_topdown_fallback"
    assert candidate["approach_direction_world"] == [0.0, 0.0, -1.0]
    # The synthesized yaw uses the parallel-gripper pi symmetry: the
    # commanded rotation from the current wrist never exceeds a quarter
    # turn.
    w, x, y, z = candidate["quaternion_wxyz"]
    hand_yaw = 2.0 * np.arctan2(z, w)
    assert abs(np.remainder(hand_yaw + np.pi / 2, np.pi) - np.pi / 2) <= (
        np.pi / 2 + 1e-6
    )

    # A mask attempt that itself closed on air routes back to the model
    # backends.
    fixture.backend._last_empty_close = {
        "backend": "mask_topdown_fallback",
        "object_name": selected["prompt"],
        "centroid": np.asarray(selected["centroid_m"]),
    }
    back = estimate()
    assert back.success
    assert (
        back.value["candidates"][0]["backend"] != "mask_topdown_fallback"
    )


def test_reference_phrase_extraction_from_relational_prompts() -> None:
    from captain_harness.adapters.capx.backend import (
        _reference_phrase_from_prompt,
    )

    assert (
        _reference_phrase_from_prompt(
            "black bowl next to the ramekin and place it on the plate"
        )
        == "ramekin"
    )
    assert (
        _reference_phrase_from_prompt("the black bowl next to the ramekin")
        == "ramekin"
    )
    assert (
        _reference_phrase_from_prompt(
            "bowl closest to the wooden cabinet, on the table"
        )
        == "wooden cabinet"
    )
    assert _reference_phrase_from_prompt("the black bowl") is None


@pytest.mark.parametrize('refined_grasp,fresh_destination', [(False, False), (True, False), (True, True)])
def test_v15_container_placement_falls_back_when_anyplace_fails(
    tmp_path: Path,
    refined_grasp: bool,
    fresh_destination: bool,
) -> None:
    fixture = _fixture(
        tmp_path,
        dehardcoded=True,
        v15_protocol=True,
    )
    observation, target = _v9_selected_detection(
        fixture,
        v15_protocol=True,
    )
    destination_result = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "segment",
        {
            "prompt": "basket",
            "observation_handle": observation["observation_handle"],
            "camera_name": "agentview",
        },
        service_aligned=True,
        dehardcoded=True,
        v15_protocol=True,
    )
    destination = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "select_detection",
        {
            "detection_id": destination_result.value["detections"][0][
                "detection_id"
            ]
        },
        service_aligned=True,
        dehardcoded=True,
        v15_protocol=True,
    )
    hand = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "gripper_state",
        {},
        service_aligned=True,
        dehardcoded=True,
        v15_protocol=True,
    )
    grasps = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "grasp_pose_estimate",
        {
            "observation_handle": observation["observation_handle"],
            "detection_id": target["detection_id"],
            "gripper_profile": hand.value["profile"],
            "backend": "auto",
            "depth_source": "sensor",
            "grasp_style": "auto",
            "top_k": 1,
        },
        service_aligned=True,
        dehardcoded=True,
        v15_protocol=True,
    )
    candidate = dict(grasps.value["candidates"][0])
    if refined_grasp:
        candidate.update(backend='public_geometry_refinement', backend_candidate=None,
                         model_score=None, convergence_risk='unassessed')
        candidate.pop('grasp_handle', None)
        candidate = fixture.backend._issue_grasp_candidate(candidate)
    assert _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "align_pregrasp",
        {"candidate": candidate},
        motion_tools=True,
        service_aligned=True,
        dehardcoded=True,
        v15_protocol=True,
    ).success
    assert _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "descend_and_close",
        {},
        motion_tools=True,
        service_aligned=True,
        dehardcoded=True,
        v15_protocol=True,
    ).success
    # Supply public visual motion before accepting an attachment pose. A
    # stationary table segmentation must not authorize placement geometry.
    lifted_track = fixture.backend.registry.get(grasps.value["tracking_handle"], expected_kind="entity_track")
    delta = np.array([0.0, 0.0, 0.10])
    fixture.backend._object_clouds = lambda *_args: (lifted_track.full_points, lifted_track.object_points + delta)
    prior_state = fixture.robot.observation.robot_state
    fixture.robot.observation = ObservationData(fixture.robot.observation.cameras,
        RobotState(prior_state.joint_positions_rad, tuple(lifted_track.centroid + delta),
                   prior_state.end_effector_quaternion_wxyz, 0.05))
    attached = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "assert_entity_attached",
        {
            "tracking_handle": grasps.value["tracking_handle"],
            "post_object_name": "bowl",
            "use_multiview": True,
        },
        service_aligned=True,
        dehardcoded=True,
        v15_protocol=True,
    )
    assert attached.success and attached.value["verified"] is True

    def fail_anyplace(**_kwargs):
        raise OSError("object_pointcloud_too_small")

    fixture.services.predict_placements = None if refined_grasp else fail_anyplace
    if fresh_destination:
        from dataclasses import replace
        fresh = fixture.backend.observe().value['observation_handle']
        old = fixture.backend.registry.get(destination.value['detection_id'], expected_kind='detection')
        destination.value['detection_id'] = fixture.backend.registry.put('detection', replace(old, observation_handle=fresh))
        def must_not_resegment(*args, **kwargs):
            raise AssertionError('public geometry placement must not resegment into the old grasp camera')
        fixture.backend._align_detection_to_observation_camera = must_not_resegment
    estimate = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "placement_estimate",
        {
            "tracking_handle": grasps.value["tracking_handle"],
            "target_detection_id": target["detection_id"],
            "destination_detection_id": destination.value[
                "detection_id"
            ],
            "relation_type": "in",
            "selected_grasp": candidate,
            "placement_candidate_index": 0,
            "reference_tolerance_m": 0.02,
        },
        motion_tools=True,
        service_aligned=True,
        dehardcoded=True,
        v15_protocol=True,
    )

    assert estimate.success, estimate
    assert estimate.value["placement_backend"] == (
        "container_mouth_fallback"
    )
    assert estimate.value["placement_pose_basis"] == (
        "container_mouth_fallback"
    )
    assert ("public geometry" if refined_grasp else "object_pointcloud_too_small") in estimate.value["anyplace_failure"]
    assert estimate.value['selected_grasp_backend'] == candidate['backend']
    if refined_grasp:
        assert candidate['backend_candidate'] is None
        assert 'container_mouth_pose_is_hover_only' in estimate.value['quality_flags']


def test_attachment_assertion_skips_visual_pipeline_when_gripper_closed_on_air(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    tracked = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "track_entity",
        {"object_name": "bowl", "use_multiview": True},
    )
    state = fixture.robot.observation.robot_state
    fixture.robot.observation = ObservationData(
        fixture.robot.observation.cameras,
        RobotState(
            state.joint_positions_rad,
            state.end_effector_position_m,
            state.end_effector_quaternion_wxyz,
            0.0,
        ),
    )
    service_calls_before = list(fixture.services.calls)

    result = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "assert_entity_attached",
        {
            "tracking_handle": tracked.value["tracking_handle"],
            "post_object_name": "bowl",
            "use_multiview": True,
        },
    )

    assert result.success
    assert result.value["verified"] is False
    assert result.value["reason"] == "closed_on_nothing"
    assert result.value["tracking_handle"] == tracked.value["tracking_handle"]
    assert fixture.services.calls == service_calls_before


def test_mark_point_backprojects_immutable_handled_rgbd(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    observation = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "get_observation",
        {},
    ).value
    marked = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "mark_point",
        {
            "observation_handle": observation["observation_handle"],
            "camera_name": "agentview",
            "x_norm": 0.5,
            "y_norm": 0.5,
            "label": "clicked target center",
        },
    )

    assert marked.success
    assert marked.value["point_handle"].startswith("artifact://v1/")
    assert marked.value["observation_handle"] == observation["observation_handle"]
    assert marked.value["camera_name"] == "agentview"
    assert marked.value["pixel_xy"] == [2.0, 2.0]
    assert np.allclose(
        marked.value["position_m"],
        [0.000625, 0.000625, 0.5],
    )


def test_segment_and_selection_cache_reuses_immutable_observation(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, dehardcoded=True)
    observation = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "observe",
        {},
        service_aligned=True,
        dehardcoded=True,
    ).value
    arguments = {
        "prompt": "black bowl",
        "observation_handle": observation["observation_handle"],
        "camera_name": "agentview",
    }

    first = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "segment",
        arguments,
        service_aligned=True,
        dehardcoded=True,
    )
    second = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "segment",
        arguments,
        service_aligned=True,
        dehardcoded=True,
    )
    detection_id = first.value["detections"][0]["detection_id"]
    selected = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "select_detection",
        {"detection_id": detection_id},
        service_aligned=True,
        dehardcoded=True,
    )
    selected_again = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "select_detection",
        {"detection_id": detection_id},
        service_aligned=True,
        dehardcoded=True,
    )

    assert first.value["cache_hit"] is False
    assert second.value["cache_hit"] is True
    assert second.value["detections"] == first.value["detections"]
    assert fixture.services.calls.count("sam3_text") == 1
    assert selected.value["cache_hit"] is False
    assert selected_again.value["cache_hit"] is True


def test_segment_cache_reuses_equivalent_rgbd_across_observation_handles(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, dehardcoded=True)
    first_observation = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "observe",
        {},
        service_aligned=True,
        dehardcoded=True,
    ).value
    second_observation = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "observe",
        {},
        service_aligned=True,
        dehardcoded=True,
    ).value
    assert (
        first_observation["observation_handle"]
        != second_observation["observation_handle"]
    )

    first = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "segment",
        {
            "prompt": "black bowl",
            "observation_handle": first_observation["observation_handle"],
            "camera_name": "agentview",
        },
        service_aligned=True,
        dehardcoded=True,
    )
    second = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "segment",
        {
            "prompt": "BLACK BOWL",
            "observation_handle": second_observation["observation_handle"],
            "camera_name": "agentview",
        },
        service_aligned=True,
        dehardcoded=True,
    )

    assert first.value["cache_hit"] is False
    assert second.value["cache_hit"] is True
    assert (
        second.value["detections"][0]["detection_id"]
        != first.value["detections"][0]["detection_id"]
    )
    selected = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "select_detection",
        {
            "detection_id": second.value["detections"][0]["detection_id"]
        },
        service_aligned=True,
        dehardcoded=True,
    )
    assert (
        selected.value["observation_handle"]
        == second_observation["observation_handle"]
    )
    assert fixture.services.calls.count("sam3_text") == 1


def test_track_entity_from_mark_reuses_clicked_observation_without_text_pointing(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    observation = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "get_observation",
        {},
    ).value
    marked = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "mark_point",
        {
            "observation_handle": observation["observation_handle"],
            "camera_name": "agentview",
            "x_norm": 0.5,
            "y_norm": 0.5,
        },
    )
    fixture.services.calls.clear()
    tracked = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "track_entity_from_mark",
        {
            "point_handle": marked.value["point_handle"],
            "label": "the clicked ketchup bottle",
            "use_multiview": True,
        },
    )

    assert tracked.success
    assert tracked.value["source_point_handle"] == marked.value["point_handle"]
    assert tracked.value["source_observation_handle"] == observation["observation_handle"]
    assert tracked.value["source_camera_name"] == "agentview"
    assert tracked.value["source_pixel_xy"] == [2.0, 2.0]
    assert tracked.value["segmentation_score"] == 0.8
    assert "sam3_point" in fixture.services.calls
    assert "molmo" not in fixture.services.calls
    assert "sam3_text" not in fixture.services.calls

    grasp = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "sample_grasp_pose",
        {
            "object_name": tracked.value["object_name"],
            "tracking_handle": tracked.value["tracking_handle"],
            "grasp_style": "auto",
            "candidate_index": 0,
        },
    )
    assert grasp.success
    assert grasp.value["identity_preserved"] is True


def test_mark_point_rejects_missing_depth_and_cross_episode_handles(
    tmp_path: Path,
) -> None:
    first = _fixture(tmp_path / "first")
    observation = _invoke(
        first,
        ToolApiLevel.S2_HIGH_LEVEL,
        "get_observation",
        {},
    ).value
    captured = first.backend.registry.get(
        observation["observation_handle"],
        expected_kind="observation",
    )
    captured.cameras[0].depth[:] = np.nan
    missing_depth = _invoke(
        first,
        ToolApiLevel.S2_HIGH_LEVEL,
        "mark_point",
        {
            "observation_handle": observation["observation_handle"],
            "camera_name": "agentview",
            "x_norm": 0.5,
            "y_norm": 0.5,
        },
    )
    assert missing_depth.failure_category is FailureCategory.TOOL_ERROR
    assert "no finite positive depth" in missing_depth.message

    second = _fixture(tmp_path / "second")
    cross_episode = _invoke(
        second,
        ToolApiLevel.S2_HIGH_LEVEL,
        "mark_point",
        {
            "observation_handle": observation["observation_handle"],
            "camera_name": "agentview",
            "x_norm": 0.5,
            "y_norm": 0.5,
        },
    )
    assert cross_episode.failure_category is FailureCategory.INVALID_ARGUMENT
    assert cross_episode.message in {
        "unknown artifact handle",
        "artifact handle belongs to another episode",
    }


def test_mark_point_flags_semantic_disagreement_without_discarding_geometry(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    observation = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "get_observation",
        {},
        public_safe_release=True,
    ).value
    fixture.services.point = lambda image, text: (100.0, 100.0)  # type: ignore[method-assign]

    marked = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "mark_point",
        {
            "observation_handle": observation["observation_handle"],
            "camera_name": "agentview",
            "x_norm": 0.5,
            "y_norm": 0.5,
            "label": "bowl",
        },
        public_safe_release=True,
    )

    assert marked.success
    assert marked.value["grounding_status"] == "grounded_with_warning"
    assert "label_point_disagreement" in marked.value["quality_flags"]


def test_v10_mark_point_keeps_uncertain_geometry_as_information(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, dehardcoded=True)
    observation = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "observe",
        {},
        service_aligned=True,
        dehardcoded=True,
    ).value
    fixture.services.segment_point = (  # type: ignore[method-assign]
        lambda image, point_xy: []
    )

    marked = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "mark_point",
        {
            "observation_handle": observation["observation_handle"],
            "camera_name": "agentview",
            "x_norm": 0.5,
            "y_norm": 0.5,
            "label": "uncertain target",
        },
        service_aligned=True,
        dehardcoded=True,
    )

    assert marked.success, marked
    assert marked.value["point_handle"].startswith("artifact://v1/")
    assert marked.value["grounding_status"] == "grounded_with_warning"
    assert "no_mask_at_selected_pixel" in marked.value["quality_flags"]


def test_v10_mark_attached_target_reports_unverified_transform(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, dehardcoded=True)
    observation, selected = _v9_selected_detection(fixture)
    hand = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "gripper_state",
        {},
        service_aligned=True,
        dehardcoded=True,
    ).value
    grasps = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "grasp_pose_estimate",
        {
            "observation_handle": observation["observation_handle"],
            "detection_id": selected["detection_id"],
            "gripper_profile": hand["profile"],
            "backend": "auto",
            "depth_source": "sensor",
            "top_k": 1,
        },
        service_aligned=True,
        dehardcoded=True,
    ).value
    approached = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "approach_and_close",
        {"candidate": grasps["candidates"][0]},
        service_aligned=True,
        dehardcoded=True,
    )
    assert approached.success
    fresh = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "observe",
        {},
        service_aligned=True,
        dehardcoded=True,
    ).value

    marked = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "mark_attached_target",
        {
            "tracking_handle": grasps["tracking_handle"],
            "observation_handle": fresh["observation_handle"],
            "camera_name": "agentview",
            "x_norm": 0.5,
            "y_norm": 0.5,
        },
        service_aligned=True,
        dehardcoded=True,
    )

    assert marked.success, marked
    assert marked.value["point_handle"].startswith("artifact://v1/")
    assert (
        "attachment_transform_unavailable"
        in marked.value["quality_flags"]
    )


def _prepare_ordinary_placement_identity(
    tmp_path: Path,
) -> tuple[Fixture, str, str, str, object, object]:
    fixture = _fixture(tmp_path)
    level = ToolApiLevel.S2_HIGH_LEVEL
    observation = _invoke(
        fixture,
        level,
        "get_observation",
        {},
        public_safe_release=True,
    ).value
    initial_mark = _invoke(
        fixture,
        level,
        "mark_point",
        {
            "observation_handle": observation["observation_handle"],
            "camera_name": "agentview",
            "x_norm": 0.5,
            "y_norm": 0.5,
        },
        public_safe_release=True,
    )
    tracked = _invoke(
        fixture,
        level,
        "track_entity_from_mark",
        {
            "point_handle": initial_mark.value["point_handle"],
            "label": "target bowl",
            "use_multiview": False,
        },
        public_safe_release=True,
    )
    target_mark = _invoke(
        fixture,
        level,
        "mark_point",
        {
            "observation_handle": observation["observation_handle"],
            "camera_name": "agentview",
            "x_norm": 0.5,
            "y_norm": 0.5,
        },
        public_safe_release=True,
    )
    destination_mark = _invoke(
        fixture,
        level,
        "mark_point",
        {
            "observation_handle": observation["observation_handle"],
            "camera_name": "agentview",
            "x_norm": 0.75,
            "y_norm": 0.5,
        },
        public_safe_release=True,
    )
    tracking_handle = tracked.value["tracking_handle"]
    tracked_record = fixture.backend.registry.get(
        tracking_handle,
        expected_kind="entity_track",
    )
    target_record = fixture.backend.registry.get(
        target_mark.value["point_handle"],
        expected_kind="marked_point",
    )
    destination_record = fixture.backend.registry.get(
        destination_mark.value["point_handle"],
        expected_kind="marked_point",
    )
    eef = np.asarray(
        fixture.robot.observation.robot_state.end_effector_position_m,
        dtype=np.float64,
    )
    tracked_record.eef_to_object_transform = np.eye(4, dtype=np.float64)
    tracked_record.eef_to_object_transform[:3, 3] = (
        target_record.position - eef
    )
    tracked_record.attachment_verified = True
    destination_record.position = (
        target_record.position + np.array([0.04, 0.0, -0.04])
    )
    return (
        fixture,
        tracking_handle,
        target_mark.value["point_handle"],
        destination_mark.value["point_handle"],
        tracked_record,
        target_record,
    )


def _estimate_prepared_placement(
    fixture: Fixture,
    tracking_handle: str,
    target_point_handle: str,
    destination_point_handle: str,
):
    return _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "estimate_placement_relation",
        {
            "tracking_handle": tracking_handle,
            "target_point_handle": target_point_handle,
            "destination_point_handle": destination_point_handle,
            "relation_type": "on",
        },
        public_safe_release=True,
    )


def test_placement_identity_ignores_raw_point_count_changes(
    tmp_path: Path,
) -> None:
    (
        fixture,
        tracking_handle,
        target_point_handle,
        destination_point_handle,
        _tracked_record,
        target_record,
    ) = _prepare_ordinary_placement_identity(tmp_path)

    target_record.point_count = 1
    sparse = _estimate_prepared_placement(
        fixture,
        tracking_handle,
        target_point_handle,
        destination_point_handle,
    )
    target_record.point_count = 1_000_000
    dense = _estimate_prepared_placement(
        fixture,
        tracking_handle,
        target_point_handle,
        destination_point_handle,
    )

    assert sparse.success and dense.success
    assert sparse.value["target_point_count_ratio"] < 0.20
    assert dense.value["target_point_count_ratio"] > 4.0
    assert sparse.value["scale_check_basis"] == "metric_extent"
    assert sparse.value["residual_ok"] is True
    assert sparse.value["appearance_ok"] is True
    assert sparse.value["scale_ok"] is True
    assert sparse.value["identity_consistent"] is True
    assert dense.value["identity_consistent"] is True


def test_placement_identity_reports_metric_extent_mismatch(
    tmp_path: Path,
) -> None:
    (
        fixture,
        tracking_handle,
        target_point_handle,
        destination_point_handle,
        tracked_record,
        target_record,
    ) = _prepare_ordinary_placement_identity(tmp_path)
    reference_extent = tracked_record.bounds_max - tracked_record.bounds_min
    target_record.bounds_min = (
        target_record.position - reference_extent * 2.0
    )
    target_record.bounds_max = (
        target_record.position + reference_extent * 2.0
    )
    target_record.point_count = tracked_record.point_count

    estimate = _estimate_prepared_placement(
        fixture,
        tracking_handle,
        target_point_handle,
        destination_point_handle,
    )

    assert estimate.success
    assert estimate.value["target_point_count_ratio"] == 1.0
    assert estimate.value["target_extent_ratio"] > 2.75
    assert estimate.value["residual_ok"] is True
    assert estimate.value["appearance_ok"] is True
    assert estimate.value["scale_ok"] is False
    assert estimate.value["identity_consistent"] is False
    assert estimate.value["motion_allowed"] is True


def test_placement_identity_reports_the_blocking_predicate(
    tmp_path: Path,
) -> None:
    (
        fixture,
        tracking_handle,
        target_point_handle,
        destination_point_handle,
        _tracked_record,
        target_record,
    ) = _prepare_ordinary_placement_identity(tmp_path)
    original_position = target_record.position.copy()
    target_record.position = original_position + np.array([0.20, 0.0, 0.0])

    residual_failure = _estimate_prepared_placement(
        fixture,
        tracking_handle,
        target_point_handle,
        destination_point_handle,
    )
    target_record.position = original_position
    target_record.appearance_rgb_mean = (
        target_record.appearance_rgb_mean
        + np.array([100.0, 100.0, 100.0])
    )
    appearance_failure = _estimate_prepared_placement(
        fixture,
        tracking_handle,
        target_point_handle,
        destination_point_handle,
    )

    assert residual_failure.success and appearance_failure.success
    assert residual_failure.value["residual_ok"] is False
    assert residual_failure.value["appearance_ok"] is True
    assert residual_failure.value["scale_ok"] is True
    assert residual_failure.value["identity_consistent"] is False
    assert appearance_failure.value["residual_ok"] is True
    assert appearance_failure.value["appearance_ok"] is False
    assert appearance_failure.value["scale_ok"] is True
    assert appearance_failure.value["identity_consistent"] is False


def test_attachment_relative_placement_uses_support_bounds_and_preserves_identity(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    level = ToolApiLevel.S2_HIGH_LEVEL
    observation = _invoke(
        fixture,
        level,
        "get_observation",
        {},
        public_safe_release=True,
    ).value
    initial_mark = _invoke(
        fixture,
        level,
        "mark_point",
        {
            "observation_handle": observation["observation_handle"],
            "camera_name": "agentview",
            "x_norm": 0.5,
            "y_norm": 0.5,
        },
        public_safe_release=True,
    )
    tracked = _invoke(
        fixture,
        level,
        "track_entity_from_mark",
        {
            "point_handle": initial_mark.value["point_handle"],
            "label": "target bowl",
            "use_multiview": False,
        },
        public_safe_release=True,
    )
    held_mark = _invoke(
        fixture,
        level,
        "mark_point",
        {
            "observation_handle": observation["observation_handle"],
            "camera_name": "agentview",
            "x_norm": 0.5,
            "y_norm": 0.5,
        },
        public_safe_release=True,
    )
    tracked_record = fixture.backend.registry.get(
        tracked.value["tracking_handle"],
        expected_kind="entity_track",
    )
    post_center = tracked_record.centroid + np.array([0.0, 0.0, 0.10])
    fixture.robot.observation = ObservationData(
        fixture.robot.observation.cameras,
        RobotState(
            tuple(fixture.robot.joints),
            tuple(post_center + np.array([0.0, 0.0, 0.02])),
            (1.0, 0.0, 0.0, 0.0),
            0.2,
            0.0,
        ),
    )
    held_record = fixture.backend.registry.get(
        held_mark.value["point_handle"],
        expected_kind="marked_point",
    )
    held_record.centroid = held_record.centroid + np.array([0.0, 0.0, 0.10])
    held_record.bounds_min = held_record.bounds_min + np.array([0.0, 0.0, 0.10])
    held_record.bounds_max = held_record.bounds_max + np.array([0.0, 0.0, 0.10])
    held_record.position = held_record.position + np.array([0.0, 0.0, 0.10])
    attached = _invoke(
        fixture,
        level,
        "assert_entity_attached",
        {
            "tracking_handle": tracked.value["tracking_handle"],
            "current_point_handle": held_mark.value["point_handle"],
            "use_multiview": False,
        },
        public_safe_release=True,
    )
    assert attached.success and attached.value["verified"] is True
    assert attached.value["identity_preserved"] is True
    assert len(attached.value["attachment_transform_eef_object"]) == 16

    state = fixture.robot.observation.robot_state
    fixture.robot.observation = ObservationData(
        fixture.robot.observation.cameras,
        RobotState(
            state.joint_positions_rad,
            (
                state.end_effector_position_m[0] + 0.08,
                state.end_effector_position_m[1],
                state.end_effector_position_m[2],
            ),
            state.end_effector_quaternion_wxyz,
            state.gripper_open_fraction,
            0.0,
        ),
    )
    placement_observation = _invoke(
        fixture,
        level,
        "get_observation",
        {},
        public_safe_release=True,
    ).value
    tracked_record.eef_to_object_transform = np.eye(4, dtype=np.float64)
    tracked_record.eef_to_object_transform[:3, 3] = (
        np.array([0.000625, 0.000625, 0.5])
        - np.asarray(
            fixture.robot.observation.robot_state.end_effector_position_m
        )
    )
    target_mark = _invoke(
        fixture,
        level,
        "mark_attached_target",
        {
            "tracking_handle": tracked.value["tracking_handle"],
            "observation_handle": placement_observation[
                "observation_handle"
            ],
            "camera_name": "agentview",
            "x_norm": 0.5,
            "y_norm": 0.5,
        },
        public_safe_release=True,
    )
    assert target_mark.success
    assert target_mark.value["grounding_status"] == "grounded_attachment"
    destination_mark = _invoke(
        fixture,
        level,
        "mark_point",
        {
            "observation_handle": placement_observation[
                "observation_handle"
            ],
            "camera_name": "agentview",
            "x_norm": 0.75,
            "y_norm": 0.5,
        },
        public_safe_release=True,
    )
    target_record = fixture.backend.registry.get(
        target_mark.value["point_handle"],
        expected_kind="marked_point",
    )
    stable_extent = (
        tracked_record.bounds_max - tracked_record.bounds_min
    )
    target_record.bounds_min = (
        target_record.position - stable_extent * 0.005
    )
    target_record.bounds_max = (
        target_record.position + stable_extent * 0.005
    )
    target_record.point_count = 1
    destination_record = fixture.backend.registry.get(
        destination_mark.value["point_handle"],
        expected_kind="marked_point",
    )
    predicted_target = np.asarray(
        fixture.robot.observation.robot_state.end_effector_position_m
    ) + np.asarray(
        tracked_record.eef_to_object_transform,
        dtype=np.float64,
    )[:3, 3]
    object.__setattr__(target_record, "position", predicted_target)
    object.__setattr__(
        destination_record,
        "position",
        predicted_target + np.array([0.04, 0.0, -0.04]),
    )
    # The valid mark is offset on a support that already covers the target.
    # It must not induce a spurious 4 cm translation toward the marked point.
    destination_record.bounds_min = predicted_target + [-0.05, -0.05, -0.04]
    destination_record.bounds_max = predicted_target + [0.05, 0.05, 0.0]

    estimate = _invoke(
        fixture,
        level,
        "estimate_placement_relation",
        {
            "tracking_handle": tracked.value["tracking_handle"],
            "target_point_handle": target_mark.value["point_handle"],
            "destination_point_handle": destination_mark.value[
                "point_handle"
            ],
            "relation_type": "on",
        },
        public_safe_release=True,
    )

    assert estimate.success
    assert np.allclose(
        estimate.value["recommended_eef_delta_m"],
        [0.0, 0.0, 0.0],
    )
    assert "correction_limit_m" not in estimate.value
    assert estimate.value["identity_consistent"] is True
    assert estimate.value["residual_ok"] is True
    assert estimate.value["appearance_ok"] is True
    assert estimate.value["scale_ok"] is True
    assert estimate.value["scale_check_basis"] == "attachment_prediction"
    assert estimate.value["target_extent_ratio"] < 0.35
    assert estimate.value["target_point_count_ratio"] < 0.20
    safe = _invoke(
        fixture,
        level,
        "authorize_safe_release",
        {
            "tracking_handle": tracked.value["tracking_handle"],
            "placement_estimate_handle": estimate.value[
                "placement_estimate_handle"
            ],
        },
        public_safe_release=True,
    )
    assert safe.success
    assert safe.value["safe_to_release"] is True
    assert isinstance(safe.value["release_authorization"], str)


def test_s2_and_s3_share_the_same_grasp_pose_convention(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    tracked = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "track_entity",
        {"object_name": "bowl", "use_multiview": True},
    )
    tracking_handle = tracked.value["tracking_handle"]
    record = fixture.backend.registry.get(
        tracking_handle,
        expected_kind="entity_track",
    )
    s2 = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "sample_grasp_pose",
        {
            "object_name": "bowl",
            "tracking_handle": tracking_handle,
            "use_multiview": True,
            "grasp_style": "auto",
            "candidate_index": 0,
        },
    )
    full_handle = fixture.backend.registry.put(
        "point_cloud",
        record.full_points,
    )
    object_handle = fixture.backend.registry.put(
        "point_cloud",
        record.object_points,
    )
    planned = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "plan_grasp_candidates",
        {
            "full_point_cloud_handle": full_handle,
            "object_point_cloud_handle": object_handle,
            "top_k": 1,
            "grasp_style": "auto",
        },
    )
    s3 = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "get_grasp_candidate",
        {
            "grasp_set_handle": planned.value["grasp_set_handle"],
            "index": 0,
        },
    )

    assert s2.success and s3.success
    assert np.allclose(s2.value["position"], s3.value["position"])
    assert np.allclose(s2.value["quaternion"], s3.value["quaternion"])


def test_tracked_grasp_candidates_are_cached_and_stable(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    tracked = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "track_entity",
        {"object_name": "bowl", "use_multiview": True},
    )
    arguments = {
        "object_name": "bowl",
        "tracking_handle": tracked.value["tracking_handle"],
        "use_multiview": True,
        "grasp_style": "auto",
        "candidate_index": 0,
    }

    first = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "sample_grasp_pose",
        arguments,
    )
    second = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "sample_grasp_pose",
        arguments,
    )

    assert first.success and second.success
    assert first.value["position"] == second.value["position"]
    assert first.value["quaternion"] == second.value["quaternion"]
    assert fixture.services.calls.count("contact_graspnet") == 1


def test_empty_grasp_service_result_is_returned_as_failure_without_fallback(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    tracked = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "track_entity",
        {"object_name": "bowl", "use_multiview": True},
    )
    fixture.services.plan_grasps = (  # type: ignore[method-assign]
        lambda full_points, object_points: (
            np.empty((0, 4, 4), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )
    )

    grasp = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "sample_grasp_pose",
        {
            "object_name": "bowl",
            "tracking_handle": tracked.value["tracking_handle"],
            "use_multiview": True,
            "grasp_style": "top_down",
            "candidate_index": 0,
        },
    )

    assert not grasp.success
    assert grasp.failure_category is FailureCategory.TOOL_ERROR
    assert "grasp service returned no candidates" in grasp.message
    assert "pyroki" not in fixture.services.calls


def test_public_attachment_assertion_uses_sensor_geometry_and_robot_state(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    full = np.asarray(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
        dtype=np.float64,
    )
    pre = np.asarray(
        [
            [0.49, 0.29, 0.08],
            [0.51, 0.29, 0.08],
            [0.50, 0.31, 0.10],
            [0.50, 0.30, 0.09],
        ],
        dtype=np.float64,
    )
    post = pre + np.asarray([0.02, -0.01, 0.10], dtype=np.float64)
    clouds = [(full, pre), (full, post)]
    fixture.backend._object_clouds = (  # type: ignore[method-assign]
        lambda object_name, use_multiview: clouds.pop(0)
    )

    tracked = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "track_entity",
        {"object_name": "black bowl next to plate", "use_multiview": True},
    )
    assert tracked.success
    cameras = fixture.robot.observation.cameras
    fixture.robot.observation = ObservationData(
        cameras,
        RobotState(
            tuple(fixture.robot.joints),
            (0.52, 0.28, 0.21),
            (1.0, 0.0, 0.0, 0.0),
            0.2,
        ),
    )
    asserted = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "assert_entity_attached",
        {
            "tracking_handle": tracked.value["tracking_handle"],
            "post_object_name": "same black bowl held by gripper",
            "use_multiview": True,
        },
    )

    assert asserted.success
    assert asserted.value["verified"] is True
    assert asserted.value["reason"] == "verified"
    assert asserted.value["vertical_displacement_m"] > 0.09
    assert asserted.value["target_to_eef_distance_m"] < 0.05


def test_public_attachment_assertion_rejects_stationary_target(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    tracked = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "track_entity",
        {"object_name": "bowl", "use_multiview": True},
    )
    assert tracked.success
    asserted = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "assert_entity_attached",
        {
            "tracking_handle": tracked.value["tracking_handle"],
            "post_object_name": "bowl held by gripper",
            "use_multiview": True,
        },
    )
    assert asserted.success
    assert asserted.value["verified"] is False
    assert asserted.value["reason"] == "target_did_not_move"


def test_v10_attachment_identity_thresholds_are_diagnostics(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, dehardcoded=True)
    observation, selected = _v9_selected_detection(fixture)
    hand = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "gripper_state",
        {},
        service_aligned=True,
        dehardcoded=True,
    ).value
    grasps = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "grasp_pose_estimate",
        {
            "observation_handle": observation["observation_handle"],
            "detection_id": selected["detection_id"],
            "gripper_profile": hand["profile"],
            "backend": "auto",
            "depth_source": "sensor",
            "top_k": 1,
        },
        service_aligned=True,
        dehardcoded=True,
    ).value
    tracking_handle = grasps["tracking_handle"]
    tracked = fixture.backend.registry.get(
        tracking_handle,
        expected_kind="entity_track",
    )
    fresh = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "observe",
        {},
        service_aligned=True,
        dehardcoded=True,
    ).value
    marked = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "mark_point",
        {
            "observation_handle": fresh["observation_handle"],
            "camera_name": "agentview",
            "x_norm": 0.5,
            "y_norm": 0.5,
        },
        service_aligned=True,
        dehardcoded=True,
    ).value
    current = fixture.backend.registry.get(
        marked["point_handle"],
        expected_kind="marked_point",
    )
    lifted = tracked.centroid + np.array([0.0, 0.0, 0.10])
    current.centroid = lifted
    current.bounds_min = lifted - 0.0001
    current.bounds_max = lifted + 0.0001
    current.point_count = 3
    current.appearance_rgb_mean = (
        tracked.appearance_rgb_mean + 200.0
    )
    state = fixture.robot.observation.robot_state
    fixture.robot.observation = ObservationData(
        fixture.robot.observation.cameras,
        RobotState(
            state.joint_positions_rad,
            tuple(lifted),
            state.end_effector_quaternion_wxyz,
            0.05,
        ),
    )

    asserted = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "assert_entity_attached",
        {
            "tracking_handle": tracking_handle,
            "current_point_handle": marked["point_handle"],
            "use_multiview": False,
        },
        service_aligned=True,
        dehardcoded=True,
    )

    assert asserted.success, asserted
    assert asserted.value["verified"] is True
    assert asserted.value["identity_preserved"] is False
    assert asserted.value["reason"] == "verified_with_identity_warning"
    assert set(asserted.value["quality_flags"]) >= {
        "point_count_ratio_outlier",
        "extent_ratio_outlier",
        "appearance_mismatch",
    }


def test_v11_attachment_uses_gripper_register_and_reports_identity_confidence(
    tmp_path: Path,
) -> None:
    fixture = _fixture(
        tmp_path,
        dehardcoded=True,
        completion_evidence=True,
    )
    observation, selected = _v9_selected_detection(fixture)
    hand = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "gripper_state",
        {},
        service_aligned=True,
        dehardcoded=True,
        completion_evidence=True,
    ).value
    grasps = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "grasp_pose_estimate",
        {
            "observation_handle": observation["observation_handle"],
            "detection_id": selected["detection_id"],
            "gripper_profile": hand["profile"],
            "backend": "auto",
            "depth_source": "sensor",
            "top_k": 1,
        },
        service_aligned=True,
        dehardcoded=True,
        completion_evidence=True,
    ).value
    tracking_handle = grasps["tracking_handle"]
    tracked = fixture.backend.registry.get(
        tracking_handle,
        expected_kind="entity_track",
    )
    fresh = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "observe",
        {},
        service_aligned=True,
        dehardcoded=True,
        completion_evidence=True,
    ).value
    marked = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "mark_point",
        {
            "observation_handle": fresh["observation_handle"],
            "camera_name": "agentview",
            "x_norm": 0.5,
            "y_norm": 0.5,
        },
        service_aligned=True,
        dehardcoded=True,
        completion_evidence=True,
    ).value
    current = fixture.backend.registry.get(
        marked["point_handle"],
        expected_kind="marked_point",
    )
    current.centroid = tracked.centroid.copy()
    current.bounds_min = tracked.centroid - 0.0001
    current.bounds_max = tracked.centroid + 0.0001
    current.point_count = 3
    current.appearance_rgb_mean = (
        tracked.appearance_rgb_mean + 200.0
    )
    state = fixture.robot.observation.robot_state
    fixture.robot.observation = ObservationData(
        fixture.robot.observation.cameras,
        RobotState(
            state.joint_positions_rad,
            tuple(tracked.centroid),
            state.end_effector_quaternion_wxyz,
            0.05,
        ),
    )

    asserted = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "assert_entity_attached",
        {
            "tracking_handle": tracking_handle,
            "current_point_handle": marked["point_handle"],
            "use_multiview": False,
        },
        service_aligned=True,
        dehardcoded=True,
        completion_evidence=True,
    )

    assert asserted.success
    assert asserted.value["verified"] is True
    assert asserted.value["reason"] == "verified"
    assert asserted.value["gripper_object_detected"] is True
    assert asserted.value["identity_confidence"] < 1.0
    assert asserted.value["pose_valid"] is False
    assert tracked.eef_to_object_transform is None
    assert set(asserted.value["quality_flags"]) >= {
        "point_count_ratio_outlier",
        "extent_ratio_outlier",
        "appearance_mismatch",
        "target_displacement_low",
        "target_lift_low",
        "lift_evidence_unavailable",
    }

    # A later contradictory observation cannot replace an earlier accepted pose.
    previous_transform = np.eye(4)
    previous_transform[:3, 3] = [.01, .02, .03]
    tracked.eef_to_object_transform = previous_transform.copy()
    tracked.attachment_verified = True
    rechecked = fixture.backend.assert_entity_attached(
        tracking_handle, current_point_handle=marked["point_handle"], use_multiview=False)
    assert rechecked.value["verified"] is True
    assert rechecked.value["pose_observation_accepted"] is False
    assert rechecked.value["pose_valid"] is True
    np.testing.assert_array_equal(tracked.eef_to_object_transform, previous_transform)


def test_lift_probe_evidence_is_bound_to_attachment_assertion(
    tmp_path: Path,
) -> None:
    fixture = _fixture(
        tmp_path,
        dehardcoded=True,
        completion_evidence=True,
    )
    observation, selected = _v9_selected_detection(fixture)
    hand = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "gripper_state",
        {},
        service_aligned=True,
        dehardcoded=True,
        completion_evidence=True,
    ).value
    grasps = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "grasp_pose_estimate",
        {
            "observation_handle": observation["observation_handle"],
            "detection_id": selected["detection_id"],
            "gripper_profile": hand["profile"],
            "backend": "auto",
            "depth_source": "sensor",
            "top_k": 1,
        },
        service_aligned=True,
        dehardcoded=True,
        completion_evidence=True,
    ).value
    candidate = grasps["candidates"][0]
    assert _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "approach_and_close",
        {"candidate": candidate},
        service_aligned=True,
        dehardcoded=True,
        completion_evidence=True,
    ).success

    lifted = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "lift_probe",
        {"height": 0.10},
        service_aligned=True,
        dehardcoded=True,
        completion_evidence=True,
    )
    asserted = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "assert_entity_attached",
        {
            "tracking_handle": grasps["tracking_handle"],
            "post_object_name": "bowl",
            "use_multiview": True,
        },
        service_aligned=True,
        dehardcoded=True,
        completion_evidence=True,
    )

    assert lifted.success
    assert lifted.value["associated_tracking_handle"] == (
        grasps["tracking_handle"]
    )
    assert lifted.value["attachment_attempt_id"] == 1
    assert lifted.value["supports_attachment"] is True
    assert lifted.value["contradicts_attachment"] is False
    assert asserted.success
    assert asserted.value["verified"] is True
    assert asserted.value["attachment_evidence_basis"] == (
        "lift_probe_and_gripper_register"
    )
    assert asserted.value["lift_evidence_available"] is True
    assert asserted.value["lift_actual_vertical_displacement_m"] == (
        pytest.approx(0.10)
    )
    assert asserted.value["lift_object_detected"] is True
    assert asserted.value["lift_supports_attachment"] is True
    assert asserted.value["lift_contradicts_attachment"] is False


def test_lift_probe_without_vertical_progress_contradicts_attachment(
    tmp_path: Path,
) -> None:
    fixture = _fixture(
        tmp_path,
        dehardcoded=True,
        completion_evidence=True,
    )
    observation, selected = _v9_selected_detection(fixture)
    hand = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "gripper_state",
        {},
        service_aligned=True,
        dehardcoded=True,
        completion_evidence=True,
    ).value
    grasps = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "grasp_pose_estimate",
        {
            "observation_handle": observation["observation_handle"],
            "detection_id": selected["detection_id"],
            "gripper_profile": hand["profile"],
            "backend": "auto",
            "depth_source": "sensor",
            "top_k": 1,
        },
        service_aligned=True,
        dehardcoded=True,
        completion_evidence=True,
    ).value
    assert _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "approach_and_close",
        {"candidate": grasps["candidates"][0]},
        service_aligned=True,
        dehardcoded=True,
        completion_evidence=True,
    ).success

    def no_vertical_progress(position, quaternion, *, position_only=False):
        return CapabilityResult.ok(None)

    fixture.robot.move_to_pose = no_vertical_progress
    lifted = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "lift_probe",
        {"height": 0.10},
        service_aligned=True,
        dehardcoded=True,
        completion_evidence=True,
    )
    asserted = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "assert_entity_attached",
        {
            "tracking_handle": grasps["tracking_handle"],
            "post_object_name": "bowl",
            "use_multiview": True,
        },
        service_aligned=True,
        dehardcoded=True,
        completion_evidence=True,
    )

    assert lifted.success
    assert lifted.value["supports_attachment"] is False
    assert lifted.value["contradicts_attachment"] is True
    assert asserted.success
    assert asserted.value["verified"] is False
    assert asserted.value["reason"] == (
        "lift_probe_contradicted_attachment"
    )
    assert asserted.value["lift_supports_attachment"] is False
    assert asserted.value["lift_contradicts_attachment"] is True
    assert "lift_probe_contradicted_attachment" in (
        asserted.value["quality_flags"]
    )


def test_v10_placement_returns_full_error_without_attachment_gate(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, dehardcoded=True)
    observation, target = _v9_selected_detection(fixture)
    destination_result = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "segment",
        {
            "prompt": "destination",
            "observation_handle": observation["observation_handle"],
            "camera_name": "agentview",
        },
        service_aligned=True,
        dehardcoded=True,
    ).value
    destination = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "select_detection",
        {
            "detection_id": destination_result["detections"][0][
                "detection_id"
            ]
        },
        service_aligned=True,
        dehardcoded=True,
    ).value
    destination_record = fixture.backend.registry.get(
        destination["detection_id"],
        expected_kind="detection",
    )
    destination_record.centroid = destination_record.centroid + np.array(
        [0.20, 0.0, 0.0]
    )
    destination_record.bounds_min = (
        destination_record.bounds_min + np.array([0.20, 0.0, 0.0])
    )
    destination_record.bounds_max = (
        destination_record.bounds_max + np.array([0.20, 0.0, 0.0])
    )
    hand = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "gripper_state",
        {},
        service_aligned=True,
        dehardcoded=True,
    ).value
    grasps = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "grasp_pose_estimate",
        {
            "observation_handle": observation["observation_handle"],
            "detection_id": target["detection_id"],
            "gripper_profile": hand["profile"],
            "backend": "auto",
            "depth_source": "sensor",
            "top_k": 1,
        },
        service_aligned=True,
        dehardcoded=True,
    ).value
    approached = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "approach_and_close",
        {"candidate": grasps["candidates"][0]},
        service_aligned=True,
        dehardcoded=True,
    )
    assert approached.success

    estimate = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "placement_estimate",
        {
            "tracking_handle": grasps["tracking_handle"],
            "target_detection_id": target["detection_id"],
            "destination_detection_id": destination["detection_id"],
            "relation_type": "on",
            "selected_grasp": grasps["candidates"][0],
            "placement_candidate_index": 0,
            "reference_tolerance_m": 0.02,
        },
        service_aligned=True,
        dehardcoded=True,
    )

    assert estimate.success, estimate
    assert estimate.value["attachment_verified"] is False
    assert {
        "attachment_unverified",
        "attachment_transform_unavailable",
    }.issubset(estimate.value["quality_flags"])
    assert np.linalg.norm(estimate.value["suggested_step_m"]) > 0.025
    assert "max_safe_step_m" not in estimate.value
    assert "distance_to_nearest_obstacle_m" in estimate.value
    assert estimate.value["selected_placement_candidate_index"] == 0
    assert estimate.value["reference_tolerance_m"] == 0.02
    assert np.allclose(
        np.asarray(estimate.value["current_tool_position_m"])
        + np.asarray(estimate.value["suggested_step_m"]),
        estimate.value["selected_placement_pose"]["position_m"],
    )
    assert estimate.value["observation_handle"] == observation["observation_handle"]
    assert (
        estimate.value["raw_selected_placement_pose"]
        == estimate.value["selected_placement_pose"]
    )
    assert estimate.value["placement_pose_basis"] == "anyplace_raw"


def test_v12_2_placement_refines_anyplace_pose_from_verified_attachment(
    tmp_path: Path,
) -> None:
    fixture = _fixture(
        tmp_path,
        dehardcoded=True,
        completion_evidence=True,
    )
    observation, target = _v9_selected_detection(fixture)
    destination_result = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "segment",
        {
            "prompt": "destination",
            "observation_handle": observation["observation_handle"],
            "camera_name": "agentview",
        },
        service_aligned=True,
        dehardcoded=True,
        completion_evidence=True,
    ).value
    destination = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "select_detection",
        {
            "detection_id": destination_result["detections"][0][
                "detection_id"
            ]
        },
        service_aligned=True,
        dehardcoded=True,
        completion_evidence=True,
    ).value
    destination_record = fixture.backend.registry.get(
        destination["detection_id"],
        expected_kind="detection",
    )
    destination_record.centroid = destination_record.centroid + np.array(
        [0.20, 0.10, 0.0]
    )
    destination_record.bounds_min = (
        destination_record.bounds_min + np.array([0.20, 0.10, 0.0])
    )
    destination_record.bounds_max = (
        destination_record.bounds_max + np.array([0.20, 0.10, 0.0])
    )
    hand = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "gripper_state",
        {},
        service_aligned=True,
        dehardcoded=True,
        completion_evidence=True,
    ).value
    grasps = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "grasp_pose_estimate",
        {
            "observation_handle": observation["observation_handle"],
            "detection_id": target["detection_id"],
            "gripper_profile": hand["profile"],
            "backend": "auto",
            "depth_source": "sensor",
            "top_k": 1,
        },
        service_aligned=True,
        dehardcoded=True,
        completion_evidence=True,
    ).value
    assert _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "approach_and_close",
        {"candidate": grasps["candidates"][0]},
        service_aligned=True,
        dehardcoded=True,
        completion_evidence=True,
    ).success
    # Supply public visual motion before accepting an attachment pose. A
    # stationary table segmentation must not authorize placement geometry.
    lifted_track = fixture.backend.registry.get(grasps["tracking_handle"], expected_kind="entity_track")
    delta = np.array([0.0, 0.0, 0.10])
    fixture.backend._object_clouds = lambda *_args: (lifted_track.full_points, lifted_track.object_points + delta)
    prior_state = fixture.robot.observation.robot_state
    fixture.robot.observation = ObservationData(fixture.robot.observation.cameras,
        RobotState(prior_state.joint_positions_rad, tuple(lifted_track.centroid + delta),
                   prior_state.end_effector_quaternion_wxyz, 0.05))
    attached = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "assert_entity_attached",
        {
            "tracking_handle": grasps["tracking_handle"],
            "post_object_name": "bowl",
            "use_multiview": True,
        },
        service_aligned=True,
        dehardcoded=True,
        completion_evidence=True,
    )
    assert attached.success
    assert attached.value["verified"] is True
    original_predict_placements = fixture.services.predict_placements

    def rotated_predict_placements(**kwargs):
        batch = original_predict_placements(**kwargs)
        poses = np.asarray(batch.poses, dtype=np.float64).copy()
        poses[0, :3, :3] = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
        )
        return GraspServiceBatch(
            poses=poses,
            scores=batch.scores,
            widths_m=batch.widths_m,
            backend=batch.backend,
            public_tool_convention=batch.public_tool_convention,
            grasp_frame=batch.grasp_frame,
            approach_directions_base=batch.approach_directions_base,
            native_candidates=batch.native_candidates,
            depth_quality=batch.depth_quality,
        )

    fixture.services.predict_placements = rotated_predict_placements
    estimate = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "placement_estimate",
        {
            "tracking_handle": grasps["tracking_handle"],
            "target_detection_id": target["detection_id"],
            "destination_detection_id": destination["detection_id"],
            "relation_type": "on",
            "selected_grasp": grasps["candidates"][0],
            "placement_candidate_index": 0,
            "reference_tolerance_m": 0.02,
        },
        service_aligned=True,
        dehardcoded=True,
        completion_evidence=True,
    )

    assert estimate.success, estimate
    value = estimate.value
    assert value["placement_pose_basis"] == (
        "attachment_translation_current_orientation"
    )
    assert value["raw_selected_placement_pose"] != value["selected_placement_pose"]
    assert np.linalg.norm(value["geometry_refinement_m"]) > 0.01
    assert value["raw_to_selected_orientation_delta_rad"] > 0.5
    assert np.allclose(
        value["selected_placement_pose"]["quaternion_wxyz"],
        fixture.robot.observation.robot_state.end_effector_quaternion_wxyz,
    )
    assert np.allclose(
        value["predicted_target_centroid_m"][:2],
        destination_record.centroid[:2],
    )
    assert np.isclose(value["predicted_support_gap_m"], 0.01)
    assert value["predicted_footprint_overlap_ratio"] > 0.95
    assert np.allclose(
        np.asarray(value["current_tool_position_m"])
        + np.asarray(value["suggested_step_m"]),
        value["selected_placement_pose"]["position_m"],
    )


def test_v13_anyplace_aligns_destination_to_target_observation_and_camera(
    tmp_path: Path,
) -> None:
    fixture = _fixture(
        tmp_path,
        dehardcoded=True,
        completion_evidence=True,
    )
    observation, target = _v9_selected_detection(fixture)
    destination_result = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "segment",
        {
            "prompt": "destination",
            "observation_handle": observation["observation_handle"],
            "camera_name": "robot0_eye_in_hand",
        },
        service_aligned=True,
        dehardcoded=True,
        completion_evidence=True,
    ).value
    destination = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "select_detection",
        {
            "detection_id": destination_result["detections"][0][
                "detection_id"
            ]
        },
        service_aligned=True,
        dehardcoded=True,
        completion_evidence=True,
    ).value
    hand = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "gripper_state",
        {},
        service_aligned=True,
        dehardcoded=True,
        completion_evidence=True,
    ).value
    grasps = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "grasp_pose_estimate",
        {
            "observation_handle": observation["observation_handle"],
            "detection_id": target["detection_id"],
            "gripper_profile": hand["profile"],
            "backend": "auto",
            "depth_source": "sensor",
            "top_k": 1,
        },
        service_aligned=True,
        dehardcoded=True,
        completion_evidence=True,
    ).value

    estimate = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "placement_estimate",
        {
            "tracking_handle": grasps["tracking_handle"],
            "target_detection_id": target["detection_id"],
            "destination_detection_id": destination["detection_id"],
            "relation_type": "on",
            "selected_grasp": grasps["candidates"][0],
            "placement_candidate_index": 0,
            "reference_tolerance_m": 0.02,
        },
        service_aligned=True,
        dehardcoded=True,
        completion_evidence=True,
    )

    assert estimate.success, estimate
    assert estimate.value["alignment_recovered"] is True
    assert (
        estimate.value["requested_destination_detection_id"]
        == destination["detection_id"]
    )
    assert (
        estimate.value["destination_detection_id"]
        != destination["detection_id"]
    )
    assert estimate.value["source_observation_handle"] == (
        observation["observation_handle"]
    )
    assert fixture.services.calls.count("anyplace") == 1


def test_v13_auto_grasp_falls_back_to_graspgenx_on_empty_contact_candidates(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, dehardcoded=True)
    original_predict = fixture.services.predict_grasps

    def predict_with_fallback(**kwargs):
        if kwargs["backend"] == "contact_graspnet":
            fixture.services.calls.append("contact_graspnet")
            raise OSError("contact_graspnet returned no normalized grasp candidates")
        return original_predict(**kwargs)

    fixture.services.predict_grasps = predict_with_fallback  # type: ignore[method-assign]
    fixture.services.available_grasp_backends = (  # type: ignore[attr-defined]
        lambda: ("contact_graspnet", "graspgenx")
    )
    observation, target = _v9_selected_detection(fixture)
    hand = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "gripper_state",
        {},
        service_aligned=True,
        dehardcoded=True,
    ).value

    grasps = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "grasp_pose_estimate",
        {
            "observation_handle": observation["observation_handle"],
            "detection_id": target["detection_id"],
            "gripper_profile": hand["profile"],
            "backend": "auto",
            "depth_source": "sensor",
            "top_k": 1,
        },
        service_aligned=True,
        dehardcoded=True,
    )

    assert grasps.success, grasps
    assert grasps.value["backend"] == "graspgenx"
    assert fixture.services.calls.count("contact_graspnet") == 1
    assert fixture.services.calls.count("graspgenx") == 1


def test_v10_release_distance_and_height_are_advisory(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, dehardcoded=True)
    observation, target = _v9_selected_detection(fixture)
    destination_result = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "segment",
        {
            "prompt": "destination",
            "observation_handle": observation["observation_handle"],
            "camera_name": "agentview",
        },
        service_aligned=True,
        dehardcoded=True,
    ).value
    destination = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "select_detection",
        {
            "detection_id": destination_result["detections"][0][
                "detection_id"
            ]
        },
        service_aligned=True,
        dehardcoded=True,
    ).value
    destination_record = fixture.backend.registry.get(
        destination["detection_id"],
        expected_kind="detection",
    )
    destination_record.centroid = destination_record.centroid + np.array(
        [0.20, 0.0, 0.0]
    )
    destination_record.bounds_min = (
        destination_record.bounds_min + np.array([0.20, 0.0, 0.0])
    )
    destination_record.bounds_max = (
        destination_record.bounds_max + np.array([0.20, 0.0, 0.0])
    )
    hand = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "gripper_state",
        {},
        service_aligned=True,
        dehardcoded=True,
    ).value
    grasps = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "grasp_pose_estimate",
        {
            "observation_handle": observation["observation_handle"],
            "detection_id": target["detection_id"],
            "gripper_profile": hand["profile"],
            "backend": "auto",
            "depth_source": "sensor",
            "top_k": 1,
        },
        service_aligned=True,
        dehardcoded=True,
    ).value
    approached = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "approach_and_close",
        {"candidate": grasps["candidates"][0]},
        service_aligned=True,
        dehardcoded=True,
    )
    assert approached.success
    estimate = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "placement_estimate",
        {
            "tracking_handle": grasps["tracking_handle"],
            "target_detection_id": target["detection_id"],
            "destination_detection_id": destination["detection_id"],
            "relation_type": "on",
            "selected_grasp": grasps["candidates"][0],
            "placement_candidate_index": 0,
            "reference_tolerance_m": 0.02,
        },
        service_aligned=True,
        dehardcoded=True,
    ).value
    estimate_record = fixture.backend.registry.get(
        estimate["placement_estimate_handle"],
        expected_kind="placement_estimate",
    )
    object.__setattr__(estimate_record, "relation_estimate", True)
    object.__setattr__(
        estimate_record,
        "correction_stage",
        "hold_for_release",
    )
    object.__setattr__(
        estimate_record,
        "horizontal_distance_m",
        10.0,
    )
    object.__setattr__(estimate_record, "release_height_m", 10.0)

    release = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "authorize_safe_release",
        {
            "tracking_handle": grasps["tracking_handle"],
            "placement_estimate_handle": estimate[
                "placement_estimate_handle"
            ],
        },
        service_aligned=True,
        dehardcoded=True,
    )

    assert release.success, release
    assert release.value["physically_safe"] is True
    assert release.value["semantic_ready"] is True
    assert release.value["release_recommended"] is True
    assert release.value["safe_to_release"] is True
    assert "attachment_unverified" in release.value["quality_flags"]
    assert release.value["horizontal_distance_m"] == 10.0
    assert release.value["release_height_m"] == 10.0
    assert "max_safe_horizontal_distance_m" not in release.value
    assert "max_safe_release_height_m" not in release.value


def test_placement_assertion_returns_fresh_xy_eef_correction(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    tracked = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "track_entity",
        {"object_name": "bowl", "use_multiview": True},
    )
    state = fixture.robot.observation.robot_state
    fixture.robot.observation = ObservationData(
        fixture.robot.observation.cameras,
        RobotState(
            state.joint_positions_rad,
            (0.5, 0.0, 0.2),
            state.end_effector_quaternion_wxyz,
            state.gripper_open_fraction,
        ),
    )

    def object_points(observation, object_name, *, use_multiview):
        center = (
            np.array([0.55, 0.05, 0.1])
            if "held" in object_name
            else np.array([0.45, -0.05, 0.0])
        )
        offsets = np.array(
            [
                [-0.01, -0.01, -0.01],
                [-0.01, -0.01, 0.01],
                [-0.01, 0.01, -0.01],
                [-0.01, 0.01, 0.01],
                [0.01, -0.01, -0.01],
                [0.01, -0.01, 0.01],
                [0.01, 0.01, -0.01],
                [0.01, 0.01, 0.01],
            ]
        )
        return center + offsets

    fixture.backend._object_points_from_observation = object_points  # type: ignore[method-assign]
    ready = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "assert_placement_ready",
        {
            "tracking_handle": tracked.value["tracking_handle"],
            "current_target_name": "the held bowl",
            "destination_name": "the plate",
            "use_multiview": True,
        },
    )

    assert ready.success
    assert ready.value["placement_ready"] is False
    assert np.allclose(
        ready.value["recommended_eef_delta_m"],
        [-0.1, -0.1, 0.0],
    )
    assert np.allclose(
        ready.value["recommended_eef_position_m"],
        [0.4, -0.1, 0.2],
    )


def test_support_placement_assertion_lowers_before_authorizing_release(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    tracked = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "track_entity",
        {"object_name": "bowl", "use_multiview": True},
    )
    target_center = np.array([0.45, -0.05, 0.08], dtype=np.float64)

    def box_points(center, half_extent):
        return np.array(
            [
                center
                + [
                    sx * half_extent[0],
                    sy * half_extent[1],
                    sz * half_extent[2],
                ]
                for sx in (-1, 1)
                for sy in (-1, 1)
                for sz in (-1, 1)
            ]
        )

    def object_points(observation, object_name, *, use_multiview):
        if "held" in object_name:
            return box_points(target_center, np.array([0.04, 0.04, 0.03]))
        return box_points(
            np.array([0.45, -0.05, 0.0]),
            np.array([0.10, 0.10, 0.01]),
        )

    fixture.backend._object_points_from_observation = object_points  # type: ignore[method-assign]
    state = fixture.robot.observation.robot_state
    fixture.robot.observation = ObservationData(
        fixture.robot.observation.cameras,
        RobotState(
            state.joint_positions_rad,
            (0.45, -0.05, 0.12),
            state.end_effector_quaternion_wxyz,
            state.gripper_open_fraction,
        ),
    )
    lowering = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "assert_placement_ready",
        {
            "tracking_handle": tracked.value["tracking_handle"],
            "current_target_name": "the held bowl",
            "destination_name": "the plate",
            "use_multiview": True,
            "relation_type": "on",
        },
    )

    assert lowering.success
    assert lowering.value["xy_ready"] is True
    assert lowering.value["height_ready"] is False
    assert lowering.value["correction_stage"] == "lower_to_surface"
    assert np.allclose(
        lowering.value["recommended_eef_delta_m"],
        [0.0, 0.0, -0.03],
    )
    assert lowering.value["placement_ready"] is False

    target_center[2] = 0.05
    ready = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "assert_placement_ready",
        {
            "tracking_handle": tracked.value["tracking_handle"],
            "current_target_name": "the held bowl",
            "destination_name": "the plate",
            "use_multiview": True,
            "relation_type": "on",
        },
    )
    assert ready.success
    assert ready.value["placement_ready"] is True
    assert ready.value["correction_stage"] == "ready"


def test_container_placement_assertion_stages_xy_then_lowering(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    tracked = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "track_entity",
        {"object_name": "ketchup bottle", "use_multiview": True},
    )
    target_center = np.array([0.70, -0.17, 0.23], dtype=np.float64)
    target_half_extent = np.array([0.035, 0.05, 0.05], dtype=np.float64)
    destination_center = np.array([0.63, 0.32, 0.08], dtype=np.float64)
    destination_half_extent = np.array([0.08, 0.11, 0.08], dtype=np.float64)

    def box_points(center, half_extent):
        return np.array(
            [
                center
                + [
                    sx * half_extent[0],
                    sy * half_extent[1],
                    sz * half_extent[2],
                ]
                for sx in (-1, 1)
                for sy in (-1, 1)
                for sz in (-1, 1)
            ]
        )

    def object_points(observation, object_name, *, use_multiview):
        if "held" in object_name:
            return box_points(target_center, target_half_extent)
        return box_points(destination_center, destination_half_extent)

    fixture.backend._object_points_from_observation = object_points  # type: ignore[method-assign]

    def set_eef(position):
        state = fixture.robot.observation.robot_state
        fixture.robot.observation = ObservationData(
            fixture.robot.observation.cameras,
            RobotState(
                state.joint_positions_rad,
                tuple(position),
                state.end_effector_quaternion_wxyz,
                state.gripper_open_fraction,
            ),
        )

    set_eef((0.70, -0.17, 0.19))
    horizontal = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "assert_placement_ready",
        {
            "tracking_handle": tracked.value["tracking_handle"],
            "current_target_name": "the held ketchup bottle",
            "destination_name": "the basket",
            "use_multiview": True,
            "relation_type": "in",
        },
    )

    assert horizontal.success
    assert horizontal.value["placement_ready"] is False
    assert horizontal.value["xy_ready"] is False
    assert horizontal.value["correction_stage"] == "transport_xy"
    assert np.allclose(
        horizontal.value["recommended_eef_delta_m"],
        [-0.07, 0.49, 0.0],
    )

    target_center[:] = [0.63, 0.32, 0.20]
    set_eef((0.63, 0.32, 0.17))
    lowering = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "assert_placement_ready",
        {
            "tracking_handle": tracked.value["tracking_handle"],
            "current_target_name": "the held ketchup bottle",
            "destination_name": "the basket",
            "use_multiview": True,
            "relation_type": "in",
        },
    )

    assert lowering.success
    assert lowering.value["placement_ready"] is False
    assert lowering.value["xy_ready"] is True
    assert lowering.value["height_ready"] is False
    assert lowering.value["correction_stage"] == "lower_into_container"
    assert np.allclose(
        lowering.value["recommended_eef_delta_m"],
        [0.0, 0.0, -0.045],
    )

    target_center[:] = [0.63, 0.32, 0.15]
    set_eef((0.63, 0.32, 0.12))
    ready = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "assert_placement_ready",
        {
            "tracking_handle": tracked.value["tracking_handle"],
            "current_target_name": "the held ketchup bottle",
            "destination_name": "the basket",
            "use_multiview": True,
            "relation_type": "in",
        },
    )

    assert ready.success
    assert ready.value["placement_ready"] is True
    assert ready.value["xy_ready"] is True
    assert ready.value["height_ready"] is True
    assert ready.value["correction_stage"] == "ready"
    assert np.allclose(ready.value["recommended_eef_delta_m"], [0.0, 0.0, 0.0])


def test_container_relation_requires_centroid_below_rim(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    target_center = np.array([0.63, 0.32, 0.17], dtype=np.float64)

    def box_points(center, half_extent):
        return np.array(
            [
                center
                + [
                    sx * half_extent[0],
                    sy * half_extent[1],
                    sz * half_extent[2],
                ]
                for sx in (-1, 1)
                for sy in (-1, 1)
                for sz in (-1, 1)
            ]
        )

    def object_points(observation, object_name, *, use_multiview):
        if "ketchup" in object_name:
            return box_points(target_center, np.array([0.03, 0.04, 0.05]))
        return box_points(
            np.array([0.63, 0.32, 0.08]),
            np.array([0.08, 0.11, 0.08]),
        )

    fixture.backend._object_points_from_observation = object_points  # type: ignore[method-assign]
    high = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "assert_spatial_relation",
        {
            "target_name": "the ketchup bottle",
            "destination_name": "the basket",
            "use_multiview": True,
            "relation_type": "in",
        },
    )

    assert high.success
    assert high.value["relation_ready"] is False
    assert high.value["reason"] == "target_not_inside_destination_height"

    target_center[2] = 0.15
    lowered = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "assert_spatial_relation",
        {
            "target_name": "the ketchup bottle",
            "destination_name": "the basket",
            "use_multiview": True,
            "relation_type": "in",
        },
    )

    assert lowered.success
    assert lowered.value["relation_ready"] is True
    assert lowered.value["reason"] == "relation_ready"


def test_point_grounded_relation_uses_same_observation_and_strict_support_height(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    observation = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "get_observation",
        {},
    ).value
    target_mark = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "mark_point",
        {
            "observation_handle": observation["observation_handle"],
            "camera_name": "agentview",
            "x_norm": 0.4,
            "y_norm": 0.5,
            "label": "bowl",
        },
    ).value["point_handle"]
    tracked = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "track_entity_from_mark",
        {
            "point_handle": target_mark,
            "label": "released bowl",
            "use_multiview": True,
        },
        public_safe_release=True,
    )
    assert tracked.success
    tracking_handle = tracked.value["tracking_handle"]
    tracked_record = fixture.backend.registry.get(
        tracking_handle,
        expected_kind="entity_track",
    )
    destination_mark = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "mark_point",
        {
            "observation_handle": observation["observation_handle"],
            "camera_name": "agentview",
            "x_norm": 0.7,
            "y_norm": 0.5,
            "label": "plate",
        },
    ).value["point_handle"]
    target_mark_record = fixture.backend.registry.get(
        target_mark,
        expected_kind="marked_point",
    )
    destination_mark_record = fixture.backend.registry.get(
        destination_mark,
        expected_kind="marked_point",
    )
    target_mark_record.position[:] = [0.5, 0.3, 0.03]
    destination_mark_record.position[:] = [0.5, 0.3, 0.0]
    target_center = np.array([0.5, 0.3, 0.03], dtype=np.float64)
    target_half_extent = np.array([0.03, 0.03, 0.02])
    destination_half_extent = np.array(
        [0.10, 0.10, 0.01],
        dtype=np.float64,
    )
    destination_contamination_z = [None]
    tracked_record.bounds_min = target_center - target_half_extent
    tracked_record.bounds_max = target_center + target_half_extent

    def box_points(center, half_extent):
        return np.array(
            [
                center
                + [
                    sx * half_extent[0],
                    sy * half_extent[1],
                    sz * half_extent[2],
                ]
                for sx in (-1, 1)
                for sy in (-1, 1)
                for sz in (-1, 1)
            ]
        )

    def points_from_mark(handle, *, use_multiview):
        marked = fixture.backend.registry.get(
            handle,
            expected_kind="marked_point",
        )
        if handle == target_mark:
            points = box_points(
                target_center,
                target_half_extent,
            )
        else:
            points = box_points(
                np.array([0.5, 0.3, 0.0]),
                destination_half_extent,
            )
            if destination_contamination_z[0] is not None:
                support = np.repeat(points, 3, axis=0)
                contamination = box_points(
                    np.array(
                        [
                            0.5,
                            0.3,
                            destination_contamination_z[0],
                        ]
                    ),
                    np.array([0.02, 0.02, 0.005]),
                )
                points = np.vstack((support, contamination))
        return marked, points, points, 0.9

    fixture.backend._points_from_mark = points_from_mark  # type: ignore[method-assign]
    ready = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "assert_spatial_relation_from_marks",
        {
            "target_point_handle": target_mark,
            "destination_point_handle": destination_mark,
            "target_label": "released bowl",
            "destination_label": "plate",
            "relation_type": "on",
        },
    )
    assert ready.success
    assert ready.value["relation_ready"] is True
    assert ready.value["strict_geometry_satisfied"] is True
    assert ready.value["strict_geometry_reason"] == "relation_ready"
    assert ready.value["support_height_basis"] == "bounds"
    assert ready.value["source_observation_handle"] == observation["observation_handle"]
    assert ready.value["relation_basis"] == "point_grounded_marks"

    target_center[0] = 0.59
    target_mark_record.position[0] = 0.59
    edge_overlap = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "assert_spatial_relation_from_marks",
        {
            "target_point_handle": target_mark,
            "destination_point_handle": destination_mark,
            "target_label": "released bowl",
            "destination_label": "plate",
            "relation_type": "on",
        },
    )
    assert edge_overlap.success
    assert edge_overlap.value["centroid_inside_destination_xy"] is True
    assert edge_overlap.value["mark_inside_destination_xy"] is True
    assert edge_overlap.value["footprint_overlap_ratio"] < 0.83
    assert edge_overlap.value["relation_ready"] is False
    assert edge_overlap.value["strict_geometry_satisfied"] is False
    assert (
        edge_overlap.value["reason"]
        == "target_footprint_outside_destination_xy"
    )

    target_center[0] = 0.5
    target_mark_record.position[0] = 0.5
    target_center[2] = -0.05
    target_mark_record.position[2] = 0.03
    embedded_support = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "assert_spatial_relation_from_marks",
        {
            "target_point_handle": target_mark,
            "destination_point_handle": destination_mark,
            "target_label": "released bowl",
            "destination_label": "plate",
            "relation_type": "on",
        },
    )
    assert embedded_support.success
    assert -0.02 <= embedded_support.value["mark_vertical_delta_m"] <= 0.12
    assert embedded_support.value["relation_ready"] is False
    assert embedded_support.value["strict_geometry_satisfied"] is False
    assert (
        embedded_support.value["strict_geometry_reason"]
        == "target_not_on_destination_support_height"
    )
    assert (
        embedded_support.value["reason"]
        == "target_not_on_destination_support_height"
    )

    target_center[2] = 0.15
    target_mark_record.position[2] = 0.15
    high = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "assert_spatial_relation_from_marks",
        {
            "target_point_handle": target_mark,
            "destination_point_handle": destination_mark,
            "target_label": "released bowl",
            "destination_label": "plate",
            "relation_type": "on",
        },
    )
    assert high.success
    assert high.value["relation_ready"] is False
    assert high.value["reason"] == "target_not_on_destination_support_height"

    target_center[:] = [0.5, 0.3, 0.03]
    target_mark_record.position[:] = target_center
    destination_contamination_z[0] = 0.08
    contaminated_support = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "assert_spatial_relation_from_marks",
        {
            "target_point_handle": target_mark,
            "destination_point_handle": destination_mark,
            "target_label": "released bowl",
            "destination_label": "plate",
            "relation_type": "on",
        },
    )
    assert contaminated_support.success
    assert contaminated_support.value["support_vertical_gap_m"] < -0.04
    assert -0.04 <= (
        contaminated_support.value["robust_support_vertical_gap_m"]
    ) <= 0.03
    assert contaminated_support.value["support_height_basis"] == (
        "bounds_rejected"
    )
    assert contaminated_support.value["relation_ready"] is False
    destination_contamination_z[0] = None

    destination_half_extent[:] = [0.20, 0.20, 0.01]
    for offset_m, expected_centered in (
        (0.06, True),
        (0.08, True),
        (0.095, True),
        (0.11, True),
    ):
        target_center[:] = [0.5 + offset_m, 0.3, 0.03]
        target_mark_record.position[:] = target_center
        completion = _invoke(
            fixture,
            ToolApiLevel.S2_HIGH_LEVEL,
            "assert_task_completion_public",
            {
                "target_point_handle": target_mark,
                "destination_point_handle": destination_mark,
                "target_label": "released bowl",
                "destination_label": "stove",
                "tracking_handle": tracking_handle,
                "relation_type": "on",
            },
            public_safe_release=True,
        )
        assert completion.success
        assert completion.value["strict_geometry_satisfied"] is True
        assert completion.value["mark_centering_required"] is False
        assert completion.value["max_mark_horizontal_distance_m"] == 0.10
        assert (
            completion.value["mark_centered_for_completion"]
            is expected_centered
        )
        assert (
            completion.value["public_relation_satisfied"]
            is expected_centered
        )
        if not expected_centered:
            assert (
                completion.value["reason"]
                == "target_mark_too_far_from_destination_center"
            )

    target_center[:] = [0.5, 0.3, 0.03]
    target_mark_record.position[:] = target_center
    target_half_extent[:] = [0.08, 0.08, 0.04]
    identity_switched = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "assert_task_completion_public",
        {
            "target_point_handle": target_mark,
            "destination_point_handle": destination_mark,
            "target_label": "released bowl",
            "destination_label": "stove",
            "tracking_handle": tracking_handle,
            "relation_type": "on",
        },
        public_safe_release=True,
    )
    assert identity_switched.success
    assert (
        identity_switched.value["released_target_identity_consistent"]
        is False
    )
    assert identity_switched.value["public_relation_satisfied"] is False
    assert (
        identity_switched.value["reason"]
        == "released_target_identity_inconsistent"
    )

    target_half_extent[:] = [0.03, 0.03, 0.02]
    captured = fixture.backend.registry.get(
        observation["observation_handle"],
        expected_kind="observation",
    )
    prior_state = captured.robot_state
    object.__setattr__(
        captured,
        "robot_state",
        RobotState(
            prior_state.joint_positions_rad,
            (0.5, 0.3, 0.10),
            prior_state.end_effector_quaternion_wxyz,
            prior_state.gripper_open_fraction,
            prior_state.joint_velocity_norm_rad_s,
        ),
    )
    arm_too_close = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "assert_task_completion_public",
        {
            "target_point_handle": target_mark,
            "destination_point_handle": destination_mark,
            "target_label": "released bowl",
            "destination_label": "stove",
            "tracking_handle": tracking_handle,
            "relation_type": "on",
        },
        public_safe_release=True,
    )
    assert arm_too_close.success
    assert arm_too_close.value["min_target_to_eef_distance_m"] > 0.12
    assert arm_too_close.value["robot_clear_for_completion"] is False
    assert arm_too_close.value["public_relation_satisfied"] is False
    assert (
        arm_too_close.value["reason"]
        == "robot_not_clear_of_released_target"
    )


@pytest.mark.parametrize("v15_protocol,relation,time_advances", [(False,"on",True),(True,"on",True),(True,"in",True),(True,"in",False)])
def test_v11_completion_reports_evidence_without_bounds_gate(
    tmp_path: Path, v15_protocol: bool, relation: str, time_advances: bool,
) -> None:
    fixture = _fixture(
        tmp_path,
        dehardcoded=True,
        completion_evidence=True,
        v15_protocol=v15_protocol,
    )
    observation, selected = _v9_selected_detection(fixture, v15_protocol=v15_protocol)
    hand = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "gripper_state",
        {},
        service_aligned=True,
        dehardcoded=True,
        completion_evidence=True,
        v15_protocol=v15_protocol,
    ).value
    grasps = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "grasp_pose_estimate",
        {
            "observation_handle": observation["observation_handle"],
            "detection_id": selected["detection_id"],
            "gripper_profile": hand["profile"],
            "backend": "auto",
            "depth_source": "sensor",
            "top_k": 1,
        },
        service_aligned=True,
        dehardcoded=True,
        completion_evidence=True,
        v15_protocol=v15_protocol,
    ).value
    tracking_handle = grasps["tracking_handle"]
    target_center = np.array([0.5, 0.3, 0.13 if relation == "in" else 0.03], dtype=np.float64)
    target_half_extent = np.array([0.03, 0.03, 0.02])
    destination_center = np.array([0.5, 0.3, 0.0])
    destination_half_extent = np.array([0.10, 0.10, 0.01])

    def box_points(center, half_extent):
        return np.array(
            [
                center
                + [
                    sx * half_extent[0],
                    sy * half_extent[1],
                    sz * half_extent[2],
                ]
                for sx in (-1, 1)
                for sy in (-1, 1)
                for sz in (-1, 1)
            ]
        )

    target_handles: set[str] = set()

    def points_from_mark(handle, *, use_multiview):
        marked = fixture.backend.registry.get(
            handle,
            expected_kind="marked_point",
        )
        if handle in target_handles:
            points = box_points(target_center, target_half_extent)
        else:
            support = np.repeat(
                box_points(
                    destination_center,
                    destination_half_extent,
                ),
                3,
                axis=0,
            )
            contamination = box_points(
                np.array([0.5, 0.3, 0.08]),
                np.array([0.02, 0.02, 0.005]),
            )
            points = np.vstack((support, contamination))
        return marked, points, points, 0.9

    fixture.backend._points_from_mark = points_from_mark  # type: ignore[method-assign]

    sample_index = 0
    def completion_evidence() -> CapabilityResult:
        nonlocal sample_index
        from dataclasses import replace
        fixture.robot.observation = replace(fixture.robot.observation,
            time_s=sample_index * .5 if time_advances else 0.0)
        sample_index += 1
        fresh = _invoke(
            fixture,
            ToolApiLevel.S3_LOW_LEVEL,
            "observe",
            {},
            service_aligned=True,
            dehardcoded=True,
            completion_evidence=True,
            v15_protocol=v15_protocol,
        ).value
        target = _invoke(
            fixture,
            ToolApiLevel.S3_LOW_LEVEL,
            "mark_point",
            {
                "observation_handle": fresh["observation_handle"],
                "camera_name": "agentview",
                "x_norm": 0.4,
                "y_norm": 0.5,
                "label": "released bowl",
            },
            service_aligned=True,
            dehardcoded=True,
            completion_evidence=True,
            v15_protocol=v15_protocol,
        ).value["point_handle"]
        target_handles.add(target)
        destination = _invoke(
            fixture,
            ToolApiLevel.S3_LOW_LEVEL,
            "mark_point",
            {
                "observation_handle": fresh["observation_handle"],
                "camera_name": "agentview",
                "x_norm": 0.7,
                "y_norm": 0.5,
                "label": "plate",
            },
            service_aligned=True,
            dehardcoded=True,
            completion_evidence=True,
            v15_protocol=v15_protocol,
        ).value["point_handle"]
        fixture.backend.registry.get(
            target,
            expected_kind="marked_point",
        ).position[:] = target_center
        fixture.backend.registry.get(
            destination,
            expected_kind="marked_point",
        ).position[:] = destination_center
        return _invoke(
            fixture,
            ToolApiLevel.S3_LOW_LEVEL,
            "assert_task_completion_public",
            {
                "target_point_handle": target,
                "destination_point_handle": destination,
                "target_label": "released bowl",
                "destination_label": "plate",
                "tracking_handle": tracking_handle,
                "relation_type": relation,
            },
            service_aligned=True,
            dehardcoded=True,
            completion_evidence=True,
            v15_protocol=v15_protocol,
        )

    first = completion_evidence()
    assert first.success
    assert "public_relation_satisfied" not in first.value
    assert first.value["recommendation"] == "insufficient"
    evidence = first.value["evidence"]
    assert evidence["gripper_released"] is True
    assert evidence["target_static_after_release_m"] is None
    if relation == "on":
        assert evidence["support_gap_bounds_m"] < -0.04
        assert -0.08 <= evidence["robust_vertical_gap_m"] <= 0.08
        assert set(first.value["quality_flags"]) >= {
            "target_static_not_yet_measurable", "target_underside_occluded", "destination_partially_covered"}
    if v15_protocol:
        assert first.value["relation_evidence"]["status"] == "insufficient"
        assert {x["name"] for x in first.value["verification_questions"]} == {"task_relation", "released_and_stable", "distinct_task_entities"}
        assert "container_height_supported" not in {r["name"] for r in first.value["verification_requirements"]}
        assert next(r for r in first.value["verification_requirements"]
                    if r["name"] == "post_release_temporal_comparison_available")["status"] == "insufficient"

    target_center[0] += 0.002
    second = completion_evidence()
    assert second.success
    if not v15_protocol:
        assert second.value["confidence"] > first.value["confidence"]
        assert second.value["recommendation"] == "satisfied"
    else:
        assert next(r for r in second.value["verification_requirements"]
                    if r["name"] == "post_release_temporal_comparison_available")["status"] == ("supported" if time_advances else "insufficient")
        assert second.value["evidence"]["comparison_time_delta_s"] == (.5 if time_advances else 0.0)
        assert second.value["evidence"]["temporal_comparison_available"] is time_advances
    assert 0 < second.value["evidence"][
        "target_static_after_release_m"
    ] < 0.01
    if v15_protocol:
        sample_index -= 1  # A new observation handle at the same paused time.
        repeated = completion_evidence()
        assert repeated.success
        assert repeated.value['evidence']['temporal_comparison_available'] is time_advances
        assert repeated.value['evidence']['comparison_time_delta_s'] == (.5 if time_advances else 0.)
        if time_advances:
            assert repeated.value['evidence']['comparison_source_observation_handle'] == first.value['source_observation_handle']
            assert repeated.value['evidence']['target_static_after_release_m'] == pytest.approx(.002)
        # Release invalidates the track participating in the new attachment
        # attempt, just as it does after a physical close in the adapter.
        fixture.backend._pending_attachment_handle = tracking_handle
        assert fixture.backend.open_gripper().executed
        after_release = completion_evidence()
        assert not after_release.value['evidence']['temporal_comparison_available']


def test_point_grounded_placement_uses_same_observation_marks(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    tracked = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "track_entity",
        {"object_name": "bowl", "use_multiview": True},
    )
    state = fixture.robot.observation.robot_state
    fixture.robot.observation = ObservationData(
        fixture.robot.observation.cameras,
        RobotState(
            state.joint_positions_rad,
            (0.5, 0.3, 0.08),
            state.end_effector_quaternion_wxyz,
            state.gripper_open_fraction,
        ),
    )
    observation = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "get_observation",
        {},
    ).value
    target_mark = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "mark_point",
        {
            "observation_handle": observation["observation_handle"],
            "camera_name": "agentview",
            "x_norm": 0.4,
            "y_norm": 0.5,
        },
    ).value["point_handle"]
    destination_mark = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "mark_point",
        {
            "observation_handle": observation["observation_handle"],
            "camera_name": "agentview",
            "x_norm": 0.7,
            "y_norm": 0.5,
        },
    ).value["point_handle"]

    def box_points(center, half_extent):
        return np.array(
            [
                center
                + [
                    sx * half_extent[0],
                    sy * half_extent[1],
                    sz * half_extent[2],
                ]
                for sx in (-1, 1)
                for sy in (-1, 1)
                for sz in (-1, 1)
            ]
        )

    def points_from_mark(handle, *, use_multiview):
        marked = fixture.backend.registry.get(
            handle,
            expected_kind="marked_point",
        )
        if handle == target_mark:
            points = box_points(
                np.array([0.5, 0.3, 0.04]),
                np.array([0.03, 0.03, 0.02]),
            )
        else:
            points = box_points(
                np.array([0.5, 0.3, 0.0]),
                np.array([0.10, 0.10, 0.01]),
            )
        return marked, points, points, 0.9

    fixture.backend._points_from_mark = points_from_mark  # type: ignore[method-assign]
    ready = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "assert_placement_ready",
        {
            "tracking_handle": tracked.value["tracking_handle"],
            "current_target_name": "the held bowl",
            "destination_name": "the plate",
            "relation_type": "on",
            "target_point_handle": target_mark,
            "destination_point_handle": destination_mark,
        },
    )

    assert ready.success
    assert ready.value["point_grounded"] is True
    assert ready.value["source_observation_handle"] == observation["observation_handle"]
    assert ready.value["target_point_handle"] == target_mark
    assert ready.value["destination_point_handle"] == destination_mark
    assert ready.value["placement_ready"] is True


def test_spatial_relation_assertion_rejects_centroid_outside_without_margin(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def object_points(observation, object_name, *, use_multiview):
        if "bowl" in object_name:
            center = np.array([0.4821, 0.08, 0.0206])
            half_extent = np.array([0.02, 0.03, 0.01])
        else:
            center = np.array([0.42, 0.18, 0.0])
            half_extent = np.array([0.11, 0.1152, 0.02])
        return np.array(
            [
                center + [sx * half_extent[0], sy * half_extent[1], sz * half_extent[2]]
                for sx in (-1, 1)
                for sy in (-1, 1)
                for sz in (-1, 1)
            ]
        )

    fixture.backend._object_points_from_observation = object_points  # type: ignore[method-assign]
    relation = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "assert_spatial_relation",
        {
            "target_name": "the bowl",
            "destination_name": "the stove",
            "use_multiview": True,
        },
    )

    assert relation.success
    assert relation.value["relation_ready"] is False
    assert relation.value["centroid_inside_destination_xy"] is True
    assert relation.value["footprint_inside_destination_xy"] is False
    assert relation.value["reason"] == "target_footprint_outside_destination_xy"


def test_goto_pose_applies_capx_tcp_and_local_approach_offsets(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    half_sqrt_two = 2 ** -0.5
    result = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "goto_pose",
        {
            "position": [0.4, 0.0, 0.4],
            "quaternion": [half_sqrt_two, 0.0, half_sqrt_two, 0.0],
            "z_approach": 0.05,
        },
    )
    assert result.success
    assert np.allclose(fixture.services.ik_positions[-2], [0.25, 0.0, 0.4])
    assert np.allclose(fixture.services.ik_positions[-1], [0.30, 0.0, 0.4])


def test_public_solve_ik_uses_the_same_capx_tcp_convention(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "solve_ik",
        {"position": [0.1, 0.0, 0.4], "quaternion": [1.0, 0.0, 0.0, 0.0]},
    )
    assert result.success
    assert np.allclose(fixture.services.ik_positions[-1], [0.1, 0.0, 0.3])


def test_high_level_multiview_uses_higher_score_when_clouds_disagree(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "get_object_pose",
        {"object_name": "bowl", "use_multiview": True},
    )
    assert result.success
    assert abs(result.value["position"][2] - 0.5) < 1e-9


def test_low_level_segmentation_exposes_only_top_eight_mask_handles(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    observation = _invoke(
        fixture, ToolApiLevel.S3_LOW_LEVEL, "get_observation", {}
    ).value
    rgb_handle = observation["cameras"][0]["rgb_handle"]
    fixture.services.segment_text = lambda image, text: [  # type: ignore[method-assign]
        MaskCandidate(
            np.ones(image.shape[:2], dtype=bool),
            float(index),
            (0, 0, 3, 3),
            f"candidate-{index}",
        )
        for index in range(12)
    ]

    segmented = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "segment_text",
        {"rgb_handle": rgb_handle, "text_prompt": "bowl"},
    )

    assert segmented.success
    assert [item["score"] for item in segmented.value["masks"]] == [
        11.0,
        10.0,
        9.0,
        8.0,
        7.0,
        6.0,
        5.0,
        4.0,
    ]
    schema = build_capx_tool_manifest(
        ToolApiLevel.S3_LOW_LEVEL, joint_count=7
    ).require("segment_text").result_schema
    assert schema["properties"]["masks"]["maxItems"] == 8


def test_high_level_grasp_applies_capx_tcp_offset_and_local_z_rotation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "sample_grasp_pose",
        {"object_name": "bowl", "use_multiview": False},
    )
    assert result.success
    assert result.value["position"][2] > 0.5
    assert abs(result.value["quaternion"][0] - 2 ** -0.5) < 1e-6
    assert abs(result.value["quaternion"][3] - 2 ** -0.5) < 1e-6
    assert len(result.value["approach_direction_base"]) == 3
    assert 0 <= result.value["top_down_alignment"] <= 1
    assert result.value["grasp_style"] == "auto"
    assert result.value["candidate_index"] == 0
    assert result.value["candidate_count"] >= 1


def test_high_level_grasp_reuses_tracked_identity_point_cloud(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    tracked = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "track_entity",
        {"object_name": "the exact ketchup bottle", "use_multiview": True},
    )
    assert tracked.success

    def unexpected_relocalization(*args, **kwargs):
        raise AssertionError("tracked grasp must not re-localize by text")

    fixture.backend._object_clouds = unexpected_relocalization  # type: ignore[method-assign]
    grasp = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "sample_grasp_pose",
        {
            "object_name": "a conflicting distractor description",
            "use_multiview": True,
            "grasp_style": "auto",
            "tracking_handle": tracked.value["tracking_handle"],
        },
    )

    assert grasp.success
    assert grasp.value["identity_preserved"] is True
    assert grasp.value["tracking_handle"] == tracked.value["tracking_handle"]
    assert grasp.value["object_name"] == "the exact ketchup bottle"


def test_high_level_grasp_exposes_ranked_candidate_index(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def two_grasps(*args):
        first = np.eye(4, dtype=np.float64)
        first[:3, 3] = [0.2, 0.0, 0.5]
        second = np.eye(4, dtype=np.float64)
        second[:3, 3] = [0.3, 0.0, 0.5]
        return np.stack([first, second]), np.asarray([0.9, 0.7])

    fixture.services.plan_grasps = two_grasps  # type: ignore[method-assign]
    second = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "sample_grasp_pose",
        {
            "object_name": "bowl",
            "use_multiview": False,
            "grasp_style": "auto",
            "candidate_index": 1,
        },
    )

    assert second.success
    assert second.value["candidate_index"] == 1
    assert second.value["candidate_count"] == 2
    assert second.value["score"] == 0.7

    missing = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "sample_grasp_pose",
        {
            "object_name": "bowl",
            "use_multiview": False,
            "grasp_style": "auto",
            "candidate_index": 2,
        },
    )
    assert missing.failure_category is FailureCategory.INVALID_ARGUMENT
    assert missing.value == {
        "requested_candidate_index": 2,
        "candidate_count": 2,
        "available_candidate_indices": [0, 1],
        "tracking_handle": None,
        "grasp_style": "auto",
    }


def test_grasp_style_ranks_top_down_and_side_candidates_differently(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def styled_grasps(*args):
        side = np.eye(4, dtype=np.float64)
        side[:3, :3] = np.array(
            [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        )
        side[:3, 3] = [0.2, 0.0, 0.5]
        top = np.eye(4, dtype=np.float64)
        top[:3, :3] = np.diag([1.0, -1.0, -1.0])
        top[:3, 3] = [0.4, 0.0, 0.5]
        return np.stack([side, top]), np.asarray([0.9, 0.5])

    fixture.services.plan_grasps = styled_grasps  # type: ignore[method-assign]
    top = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "sample_grasp_pose",
        {
            "object_name": "bowl",
            "use_multiview": False,
            "grasp_style": "top_down",
        },
    )
    side = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "sample_grasp_pose",
        {
            "object_name": "bowl",
            "use_multiview": False,
            "grasp_style": "side",
        },
    )

    assert top.success and side.success
    assert top.value["position"][0] == 0.4
    assert top.value["top_down_alignment"] == 1.0
    assert top.value["grasp_style"] == "top_down"
    assert abs(side.value["position"][0] - 0.08) < 1e-12
    assert side.value["top_down_alignment"] == 0.0
    assert side.value["grasp_style"] == "side"


def test_top_down_grasp_style_rejects_nonvertical_candidates(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    result = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "sample_grasp_pose",
        {
            "object_name": "bowl",
            "use_multiview": False,
            "grasp_style": "top_down",
        },
    )

    assert not result.success
    assert result.failure_category is FailureCategory.TOOL_ERROR
    assert "top_down alignment" in result.message


def test_side_grasp_style_rejects_near_vertical_candidates(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def vertical_grasps(*args):
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = np.diag([1.0, -1.0, -1.0])
        pose[:3, 3] = [0.4, 0.0, 0.5]
        return np.stack([pose]), np.asarray([0.9])

    fixture.services.plan_grasps = vertical_grasps  # type: ignore[method-assign]
    result = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "sample_grasp_pose",
        {
            "object_name": "bowl",
            "use_multiview": False,
            "grasp_style": "side",
        },
    )

    assert not result.success
    assert result.failure_category is FailureCategory.TOOL_ERROR
    assert "side alignment" in result.message


def test_side_grasp_style_scans_past_higher_scored_vertical_candidates(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def mixed_grasps(*args):
        poses = []
        scores = []
        for index in range(20):
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = np.diag([1.0, -1.0, -1.0])
            pose[:3, 3] = [0.2 + index * 0.001, 0.0, 0.5]
            poses.append(pose)
            scores.append(1.0 - index * 0.01)
        side = np.eye(4, dtype=np.float64)
        side[:3, :3] = np.array(
            [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        )
        side[:3, 3] = [0.5, 0.0, 0.5]
        poses.append(side)
        scores.append(0.1)
        return np.stack(poses), np.asarray(scores)

    fixture.services.plan_grasps = mixed_grasps  # type: ignore[method-assign]
    result = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "sample_grasp_pose",
        {
            "object_name": "bowl",
            "use_multiview": False,
            "grasp_style": "side",
        },
    )

    assert result.success
    assert abs(result.value["position"][0] - 0.38) < 1e-12
    assert result.value["top_down_alignment"] == 0.0


def test_every_capability_rejects_missing_arguments_before_backend(tmp_path: Path) -> None:
    for level in ToolApiLevel:
        fixture = _fixture(tmp_path / level.value)
        manifest = build_capx_tool_manifest(level, joint_count=7)
        broker = ValidatedCapabilityBroker(manifest, fixture.backend.invoke)
        for spec in manifest.capabilities:
            if not spec.arguments_schema.get("required"):
                continue
            before = len(fixture.robot.calls) + len(fixture.services.calls)
            result = broker.invoke(spec.name, {})
            after = len(fixture.robot.calls) + len(fixture.services.calls)
            assert result.failure_category is FailureCategory.SCHEMA_ERROR, spec.name
            assert before == after, spec.name


def test_unknown_and_cross_episode_handles_are_invalid_arguments(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    for handle in (
        "artifact://v1/another-episode/rgb/abcdef",
        "artifact://v1/episode-test/rgb/not-present",
        "not-a-handle",
    ):
        result = _invoke(
            fixture,
            ToolApiLevel.S3_LOW_LEVEL,
            "segment_text",
            {"rgb_handle": handle, "text_prompt": "bowl"},
        )
        expected = (
            FailureCategory.SCHEMA_ERROR
            if handle == "not-a-handle"
            else FailureCategory.INVALID_ARGUMENT
        )
        assert result.failure_category is expected


def test_empty_segmentation_ik_failure_and_timeout_are_explicit(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    observation = _invoke(fixture, ToolApiLevel.S3_LOW_LEVEL, "get_observation", {}).value
    rgb_handle = observation["cameras"][0]["rgb_handle"]
    fixture.services.segment_text = lambda image, text: []  # type: ignore[method-assign]
    empty = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "segment_text",
        {"rgb_handle": rgb_handle, "text_prompt": "missing"},
    )
    assert empty.failure_category is FailureCategory.TOOL_ERROR

    fixture.services.solve_ik = lambda *args: None  # type: ignore[method-assign]
    ik = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "solve_ik",
        {"position": [0, 0, 0.4], "quaternion": [1, 0, 0, 0]},
    )
    assert ik.failure_category is FailureCategory.IK_FAILURE

    def timeout(*args):
        raise TimeoutError("service deadline")

    fixture.services.point = timeout  # type: ignore[method-assign]
    timed_out = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "point_molmo",
        {"rgb_handle": rgb_handle, "text_prompt": "bowl"},
    )
    assert timed_out.failure_category is FailureCategory.TIMEOUT


def test_ik_keeps_private_eight_dof_warm_start_and_returns_seven_arm_joints(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    arguments = {"position": [0.1, 0.0, 0.4], "quaternion": [1, 0, 0, 0]}
    first = _invoke(fixture, ToolApiLevel.S3_LOW_LEVEL, "solve_ik", arguments)
    second = _invoke(fixture, ToolApiLevel.S3_LOW_LEVEL, "solve_ik", arguments)
    assert first.success and len(first.value) == 7
    assert second.success and len(second.value) == 7
    assert len(fixture.services.ik_previous[0]) == 8
    assert fixture.services.ik_previous[0][-1] == 0.04
    assert len(fixture.services.ik_previous[1]) == 8


def test_no_grasp_candidates_is_an_explicit_tool_failure(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    observation = _invoke(fixture, ToolApiLevel.S3_LOW_LEVEL, "get_observation", {}).value
    rgb_handle = observation["cameras"][0]["rgb_handle"]
    segmented = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "segment_text",
        {"rgb_handle": rgb_handle, "text_prompt": "bowl"},
    ).value
    clouds = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "construct_point_cloud",
        {
            "observation_handle": observation["observation_handle"],
            "mask_handles": [segmented["masks"][0]["mask_handle"]],
            "use_multiview": False,
        },
    ).value
    fixture.services.plan_grasps = lambda *args: (  # type: ignore[method-assign]
        np.empty((0, 4, 4)),
        np.empty((0,)),
    )
    result = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "plan_grasp_candidates",
        {
            "full_point_cloud_handle": clouds["full_point_cloud_handle"],
            "object_point_cloud_handle": clouds["object_point_cloud_handle"],
            "top_k": 3,
        },
    )
    assert result.failure_category is FailureCategory.TOOL_ERROR
    assert "no candidates" in result.message


def test_grasp_candidates_filter_unreachable_poses_before_exposure(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def two_grasps(*args):
        first = np.eye(4, dtype=np.float64)
        first[:3, 3] = [0.1, 0.0, 0.5]
        second = np.eye(4, dtype=np.float64)
        second[:3, 3] = [0.3, 0.0, 0.5]
        return np.stack([first, second]), np.asarray([0.9, 0.7])

    fixture.services.plan_grasps = two_grasps  # type: ignore[method-assign]
    original_solve = fixture.services.solve_ik

    def selective_ik(position, quaternion, previous_joints):
        if float(position[0]) < 0.2:
            return None
        return original_solve(position, quaternion, previous_joints)

    fixture.services.solve_ik = selective_ik  # type: ignore[method-assign]

    high = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "sample_grasp_pose",
        {"object_name": "bowl", "use_multiview": False},
    )
    assert high.success
    assert high.value["score"] == 0.7
    assert high.value["rejected_unreachable_count"] == 1
    assert high.value["predicted_joint_travel_rad"] > 0
    assert len(high.value["approach_direction_base"]) == 3

    observation = _invoke(
        fixture, ToolApiLevel.S3_LOW_LEVEL, "get_observation", {}
    ).value
    segmented = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "segment_text",
        {
            "rgb_handle": observation["cameras"][0]["rgb_handle"],
            "text_prompt": "bowl",
        },
    ).value
    clouds = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "construct_point_cloud",
        {
            "observation_handle": observation["observation_handle"],
            "mask_handles": [segmented["masks"][0]["mask_handle"]],
            "use_multiview": False,
        },
    ).value
    planned = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "plan_grasp_candidates",
        {
            "full_point_cloud_handle": clouds["full_point_cloud_handle"],
            "object_point_cloud_handle": clouds["object_point_cloud_handle"],
            "top_k": 2,
        },
    )
    assert planned.success
    assert planned.value["candidate_count"] == 1
    assert planned.value["rejected_unreachable_count"] == 1
    candidate = _invoke(
        fixture,
        ToolApiLevel.S3_LOW_LEVEL,
        "get_grasp_candidate",
        {
            "grasp_set_handle": planned.value["grasp_set_handle"],
            "index": 0,
        },
    )
    assert candidate.value["score"] == 0.7
    assert candidate.value["predicted_joint_travel_rad"] > 0
    assert len(candidate.value["approach_direction_base"]) == 3


def test_goto_pose_returns_structured_motion_outcome(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "goto_pose",
        {
            "position": [0.3, 0.0, 0.5],
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "z_approach": 0.08,
        },
    )

    assert result.success
    assert result.value["phase"] == "target"
    assert result.value["executed"] is True
    assert result.value["converged"] is True
    assert result.value["convergence_reason"] == "within_tolerance"
    assert result.value["max_joint_residual_rad"] == 0.0
    assert result.value["l2_joint_residual_rad"] == 0.0
    assert result.value["position_tolerance_m"] == 0.03
    assert result.value["orientation_tolerance_rad"] == pytest.approx(
        np.deg2rad(5.0)
    )
    assert result.value["position_residual_m"] <= 0.03
    assert result.value["orientation_residual_rad"] == 0.0
    assert result.value["requested_cartesian_position_m"] == [0.3, 0.0, 0.5]
    assert result.value["requested_cartesian_quaternion_wxyz"] == [
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    assert result.value["controller_target_position_m"] == [0.3, 0.0, 0.4]
    assert result.value["final_cartesian_position_m"] == [0.3, 0.0, 0.5]
    assert result.value["final_cartesian_quaternion_wxyz"] == [
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    assert len(result.value["final_eef_position_m"]) == 3
    assert len(result.value["final_eef_quaternion_wxyz"]) == 4


def test_goto_pose_reports_large_cartesian_position_residual(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    original_move = fixture.robot.move_to_joints

    def drifted_move(joints):
        result = original_move(joints)
        state = fixture.robot.observation.robot_state
        fixture.robot.observation = ObservationData(
            fixture.robot.observation.cameras,
            RobotState(
                state.joint_positions_rad,
                (
                    state.end_effector_position_m[0] + 0.05,
                    state.end_effector_position_m[1],
                    state.end_effector_position_m[2],
                ),
                state.end_effector_quaternion_wxyz,
                state.gripper_open_fraction,
            ),
        )
        return result

    fixture.robot.move_to_joints = drifted_move
    result = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "goto_pose",
        {
            "position": [0.3, 0.0, 0.5],
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "z_approach": 0.0,
        },
    )

    assert not result.success
    assert result.executed
    assert result.failure_category is FailureCategory.TIMEOUT
    assert result.value["converged"] is False
    assert result.value["convergence_reason"] == "position_residual_high"
    assert result.value["position_residual_m"] > 0.03
    assert not any(name == "move_to_pose" for name, _ in fixture.robot.calls)


def test_goto_pose_does_not_synthesize_cartesian_correction(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    original_move = fixture.robot.move_to_joints

    def drifted_move(joints):
        result = original_move(joints)
        state = fixture.robot.observation.robot_state
        fixture.robot.observation = ObservationData(
            fixture.robot.observation.cameras,
            RobotState(
                state.joint_positions_rad,
                (
                    state.end_effector_position_m[0] + 0.08,
                    state.end_effector_position_m[1],
                    state.end_effector_position_m[2],
                ),
                state.end_effector_quaternion_wxyz,
                state.gripper_open_fraction,
            ),
        )
        return result

    fixture.robot.move_to_joints = drifted_move
    result = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "goto_pose",
        {
            "position": [0.3, 0.0, 0.5],
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "z_approach": 0.0,
        },
    )

    assert not result.success
    assert result.failure_category is FailureCategory.TIMEOUT
    assert result.value["converged"] is False
    assert result.value["position_residual_m"] > 0.03
    assert not any(name == "move_to_pose" for name, _ in fixture.robot.calls)


def test_goto_pose_propagates_joint_controller_timeout(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    original_move = fixture.robot.move_to_joints

    def cartesian_met_with_joint_timeout(joints):
        original_move(joints)
        return CapabilityResult.failed(
            FailureCategory.TIMEOUT,
            "joint target did not converge in 480 steps "
            "(joint 7 max residual 0.008000 rad, L2 residual 0.013000 rad)",
            executed=True,
        )

    fixture.robot.move_to_joints = cartesian_met_with_joint_timeout
    result = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "goto_pose",
        {
            "position": [0.3, 0.0, 0.5],
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "z_approach": 0.0,
        },
    )

    assert not result.success
    assert result.executed
    assert result.failure_category is FailureCategory.TIMEOUT
    assert result.value["converged"] is True
    assert result.value["position_residual_m"] <= 0.03
    assert result.value["orientation_residual_rad"] <= 0.35
    assert result.value["max_joint_residual_rad"] == 0.008


def test_goto_pose_fails_large_orientation_residual(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    wrong_quaternion = (0.0, 1.0, 0.0, 0.0)
    original_move = fixture.robot.move_to_joints
    original_pose = fixture.robot.move_to_pose

    def wrong_orientation_move(joints):
        result = original_move(joints)
        state = fixture.robot.observation.robot_state
        fixture.robot.observation = ObservationData(
            fixture.robot.observation.cameras,
            RobotState(
                state.joint_positions_rad,
                (
                    state.end_effector_position_m[0],
                    state.end_effector_position_m[1],
                    state.end_effector_position_m[2] + 0.2,
                ),
                wrong_quaternion,
                state.gripper_open_fraction,
            ),
        )
        return result

    def wrong_orientation_pose(position, quaternion, *, position_only=False):
        result = original_pose(
            position,
            quaternion,
            position_only=position_only,
        )
        state = fixture.robot.observation.robot_state
        fixture.robot.observation = ObservationData(
            fixture.robot.observation.cameras,
            RobotState(
                state.joint_positions_rad,
                (
                    state.end_effector_position_m[0],
                    state.end_effector_position_m[1],
                    state.end_effector_position_m[2] + 0.2,
                ),
                wrong_quaternion,
                state.gripper_open_fraction,
            ),
        )
        return result

    fixture.robot.move_to_joints = wrong_orientation_move
    fixture.robot.move_to_pose = wrong_orientation_pose
    result = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "goto_pose",
        {
            "position": [0.3, 0.0, 0.5],
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "z_approach": 0.0,
        },
    )

    assert not result.success
    assert result.executed
    assert result.failure_category is FailureCategory.TIMEOUT
    assert result.value["converged"] is False
    assert result.value["convergence_reason"] == "orientation_residual_high"
    assert result.value["position_residual_m"] <= 0.03
    assert result.value["orientation_residual_rad"] > np.deg2rad(5.0)


def test_goto_pose_accepts_osc_timeout_within_public_pose_tolerance(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture.robot.controller = "OSC_POSE"
    original_pose = fixture.robot.move_to_pose

    def strict_controller_timeout(position, quaternion, *, position_only=False):
        original_pose(
            position,
            quaternion,
            position_only=position_only,
        )
        return CapabilityResult.failed(
            FailureCategory.TIMEOUT,
            "OSC_POSE command exceeded strict controller tolerance",
            executed=True,
        )

    fixture.robot.move_to_pose = strict_controller_timeout
    result = _invoke(
        fixture,
        ToolApiLevel.S2_HIGH_LEVEL,
        "goto_pose",
        {
            "position": [0.3, 0.0, 0.5],
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "z_approach": 0.0,
        },
    )

    assert result.success
    assert result.value["converged"] is True
    assert result.value["position_residual_m"] == 0.0
    assert result.value["orientation_residual_rad"] == 0.0
    assert result.message == (
        "OSC_POSE reached the calibrated public pose tolerance"
    )


@pytest.mark.parametrize('tilt', [0.0, 0.4, 1.1, 1.57])
def test_geometry_grasp_constructs_full_downward_frame(tmp_path, tilt):
    from captain_harness.adapters.capx.backend import _matrix_to_quaternion_wxyz
    fixture = _fixture(tmp_path, dehardcoded=True, v15_protocol=True)
    state = fixture.robot.observation.robot_state
    fixture.robot.observation = ObservationData(
        fixture.robot.observation.cameras,
        RobotState(state.joint_positions_rad, state.end_effector_position_m,
                   (np.cos(tilt / 2), np.sin(tilt / 2), 0.0, 0.0), 1.0),
    )
    points = np.array([[x, y, z] for x in (-.02, .02)
                       for y in (-.04, .04) for z in (.02, .04)])
    batch = fixture.backend._mask_topdown_fallback_batch(points, backend_failure='test')
    rotation = batch.poses[0, :3, :3]
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
    assert np.linalg.det(rotation) == pytest.approx(1)
    np.testing.assert_allclose(rotation[:, 2], [0, 0, -1], atol=1e-10)
    np.testing.assert_allclose(rotation[:, 2], batch.approach_directions_base[0])
    # Panda finger joints translate on hand-local Y (the public IK frame),
    # so their WORLD closing direction must span the object's short side.
    world_jaw_axis = rotation @ np.array([0.0, 1.0, 0.0])
    assert abs(world_jaw_axis[0]) == pytest.approx(1.0)
    assert np.ptp(points @ world_jaw_axis) == pytest.approx(batch.widths_m[0])
    candidate = {'position_m': batch.poses[0, :3, 3].tolist(),
                 'approach_direction_world': [0, 0, -1],
                 'quaternion_wxyz': list(_matrix_to_quaternion_wxyz(rotation))}
    # The public pose round-trips to the observed contact center through calibration.
    _, _, _, position, _ = fixture.backend._candidate_execution_pose(candidate)
    np.testing.assert_allclose(position, [0, 0, .03], atol=1e-10)
    assert batch.widths_m[0] == pytest.approx(.04)


def test_explicit_geometry_backend_passes_public_rpc_schema(tmp_path):
    fixture = _fixture(tmp_path, dehardcoded=True, v15_protocol=True)
    observation, selected = _v9_selected_detection(fixture, v15_protocol=True)
    hand = fixture.backend.gripper_state().value
    result = _invoke(fixture, ToolApiLevel.S3_LOW_LEVEL, 'grasp_pose_estimate', {
        'observation_handle': observation['observation_handle'],
        'detection_id': selected['detection_id'], 'gripper_profile': hand['profile'],
        'backend': 'mask_topdown_fallback', 'depth_source': 'sensor',
        'grasp_style': 'top_down', 'top_k': 1,
    }, service_aligned=True, dehardcoded=True, v15_protocol=True)
    assert result.success, result.message
    assert result.value['backend'] == 'mask_topdown_fallback'


def test_geometry_contact_center_roundtrips_real_adapter_calibration(tmp_path):
    import json
    from captain_harness.adapters.libero.backend import LiberoActionBackend
    from captain_harness.adapters.capx.backend import _matrix_to_quaternion_wxyz
    fixture = _fixture(tmp_path, dehardcoded=True, v15_protocol=True)
    # Only public frame conversion is exercised; no simulator or object state.
    calibration = json.loads(Path('configs/calibration/libero_franka_v3.json').read_text())
    transforms = object.__new__(LiberoActionBackend)
    transforms._pyroki_hand_to_libero_eef = np.asarray(calibration['pyroki_hand_to_libero_eef_4x4']).reshape(4, 4)
    fixture.robot.external_ik_target_from_eef_pose = transforms.external_ik_target_from_eef_pose
    fixture.robot.eef_pose_from_external_ik_target = transforms.eef_pose_from_external_ik_target
    points = np.array([[x, y, z] for x in (.1, .14) for y in (.2, .26) for z in (.01, .03)])
    batch = fixture.backend._mask_topdown_fallback_batch(points, backend_failure='calibration test')
    pose = batch.poses[0]
    candidate = {'position_m': pose[:3, 3].tolist(),
                 'quaternion_wxyz': list(_matrix_to_quaternion_wxyz(pose[:3, :3])),
                 'approach_direction_world': [0, 0, -1]}
    _, _, _, eef_position, _ = fixture.backend._candidate_execution_pose(candidate)
    np.testing.assert_allclose(eef_position, [.12, .23, .02], atol=1e-10)
    jaw_direction = pose[:3, :3] @ np.array([0., 1., 0.])
    assert np.ptp(points @ jaw_direction) == pytest.approx(.04)


@pytest.mark.parametrize('executed,success', [(True, True), (True, False), (False, False)])
def test_gripper_open_invalidates_only_executed_attachment(tmp_path, executed, success):
    from captain_harness.adapters.capx.backend import _TrackedEntity
    fixture = _fixture(tmp_path, dehardcoded=True, v15_protocol=True)
    tracked = _TrackedEntity('public target', np.zeros(3), np.zeros(3), np.ones(3)*.04,
        3, np.zeros((3,3)), np.zeros((3,3)), eef_to_object_transform=np.eye(4),
        attachment_verified=True)
    handle = fixture.backend.registry.put('entity_track', tracked)
    fixture.backend._active_attachment_handle = handle
    fixture.backend._pending_attachment_handle = handle
    fixture.robot.open_gripper = lambda: CapabilityResult.ok(None) if success else CapabilityResult.failed(
        FailureCategory.TIMEOUT, 'opening incomplete', executed=executed)
    result = fixture.backend.open_gripper()
    assert result.success is success
    assert tracked.attachment_verified is (not executed)
    assert (tracked.eef_to_object_transform is None) is executed
    assert fixture.backend._active_attachment_handle == (None if executed else handle)
    # Stale object coordinates cannot drive the hand after contact may have changed.
    if executed:
        move = fixture.backend.goto_pose([.4, .2, .3], [1,0,0,0],
            position_reference='held_target', tracking_handle=handle)
        assert not move.success and not move.executed


def test_public_wait_rpc_preserves_elapsed_time_and_action_semantics(tmp_path):
    fixture=_fixture(tmp_path,dehardcoded=True,v15_protocol=True)
    from captain_harness.ports.capability import ControlSemantic,SideEffect
    requested=[]
    def wait(duration_s):
        requested.append(duration_s)
        return CapabilityResult.ok({'requested_duration_s':duration_s,'elapsed_s':duration_s,
            'start_time_s':1.0,'end_time_s':1.0+duration_s,'steps_executed':4,'gripper_command_fraction':0.0})
    fixture.robot.hold_position=wait
    result=_invoke(fixture,ToolApiLevel.S3_LOW_LEVEL,'hold_position',{'duration_s':.2},
        service_aligned=True,dehardcoded=True,v15_protocol=True)
    assert result.success and requested==[.2]
    manifest=build_capx_tool_manifest(ToolApiLevel.S3_LOW_LEVEL,joint_count=7,
        service_aligned=True,dehardcoded=True,v15_protocol=True)
    spec=manifest.require('hold_position')
    assert spec.side_effect is SideEffect.ACTUATOR
    assert spec.control_semantic is ControlSemantic.TIME_ADVANCE

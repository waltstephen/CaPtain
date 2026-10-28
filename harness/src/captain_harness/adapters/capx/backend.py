from __future__ import annotations

from captain_harness.adapters.capx.placement_resolution import correction_resolution

import hashlib
import math
import re
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Mapping, Protocol, Sequence, cast

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from captain_harness.adapters.libero.geometry import (
    DepthUnavailableError,
    pixel_to_world,
)
from captain_harness.adapters.libero.calibration import GripperProfile
from captain_harness.adapters.libero.types import ObservationData, RobotState
from captain_harness.core.experience import ExperienceSnapshot
from captain_harness.ports.capability import CapabilityResult, FailureCategory

from .artifacts import ArtifactHandleError, EpisodeArtifactRegistry
from .grasp_feedback import grasp_style_evidence, summarize_grasp_choices
from .motion_tools import MotionToolMixin
from .pose_tools import PoseCandidateToolMixin
from .selection_evidence import bind_selection_evidence, point_source_error
from .support_geometry import minimum_support_translation
from .protocol import (
    PublicCompletionProtocol,
    load_public_completion_protocol,
)

_MAX_PUBLIC_MASKS = 8
_MULTIVIEW_MERGE_DISTANCE_M = 0.01
_NOISE_RADIUS_M = 0.005
_NOISE_MIN_SAMPLES = 10
_MAX_PREDICTED_JOINT_TRAVEL_RAD = 2.5
_MAX_PREDICTED_SINGLE_JOINT_DELTA_RAD = 1.25
_MIN_TOP_DOWN_ALIGNMENT = 0.9
_MAX_SIDE_VERTICAL_ALIGNMENT = 0.45
_SMALL_TARGET_EXTENT_M = 0.06
_PRE_DESCENT_CENTERING_MAX_M = 0.03
_PRE_DESCENT_CENTERING_MIN_M = 0.005
_PRE_DESCENT_CENTERING_MATCH_RADIUS_M = 0.15
_CONTAINER_RIM_CLEARANCE_M = 0.005
_MAX_MARK_MASK_AREA_FRACTION = 0.45
_MIN_MARK_MASK_AREA_FRACTION = 0.0001
_MIN_MARK_DEPTH_CONTINUITY = 0.55
_MAX_MARK_GEOMETRY_DIAGONAL_M = 0.60
_MIN_MARK_GEOMETRY_DIAGONAL_M = 0.003
_MAX_TRACK_POINT_COUNT_RATIO = 4.0
_MIN_TRACK_POINT_COUNT_RATIO = 0.20
_MAX_TRACK_EXTENT_RATIO = 2.75
_MIN_TRACK_EXTENT_RATIO = 0.35
_MAX_ATTACHMENT_VISUAL_RESIDUAL_M = 0.10
_MAX_SAFE_RELEASE_JOINT_SPEED_RAD_S = 0.10
_RESIDUAL_PATTERN = re.compile(
    r"max residual ([0-9.]+) rad, L2 residual ([0-9.]+) rad"
)


@dataclass(frozen=True, slots=True)
class MaskCandidate:
    mask: np.ndarray
    score: float
    box: tuple[float, float, float, float] | None = None
    label: str | None = None


@dataclass(frozen=True, slots=True)
class GraspServiceBatch:
    poses: np.ndarray
    scores: np.ndarray
    widths_m: tuple[float | None, ...]
    backend: str
    public_tool_convention: bool
    grasp_frame: str = "capx_native"
    approach_directions_base: (
        tuple[tuple[float, float, float], ...] | None
    ) = None
    native_candidates: tuple[Mapping[str, object], ...] = ()
    depth_quality: Mapping[str, object] = field(default_factory=dict)


class RobotToolBackend(Protocol):
    @property
    def controller(self) -> str: ...

    def get_observation_data(self) -> CapabilityResult[ObservationData]: ...

    def get_robot_state(self) -> CapabilityResult[RobotState]: ...

    def move_to_joints(self, joints: Sequence[float]) -> CapabilityResult[None]: ...

    def move_to_pose(
        self,
        position: Sequence[float],
        quaternion: Sequence[float],
        *,
        position_only: bool = False,
    ) -> CapabilityResult[dict[str, object] | None]: ...

    def external_ik_target_from_eef_pose(
        self,
        position: Sequence[float],
        quaternion: Sequence[float],
    ) -> tuple[Sequence[float], Sequence[float]]: ...

    def move_to_pose_monitored(self, position, quaternion) -> CapabilityResult[dict[str, object]]: ...

    def eef_pose_from_external_ik_target(
        self,
        position: Sequence[float],
        quaternion: Sequence[float],
    ) -> tuple[Sequence[float], Sequence[float]]: ...

    def open_gripper(self) -> CapabilityResult[None]: ...

    def close_gripper(self) -> CapabilityResult[None]: ...

    def home(self) -> CapabilityResult[None]: ...

    def gripper_state(self) -> CapabilityResult[dict[str, object]]: ...

    def approach_and_close(
        self,
        position: Sequence[float],
        quaternion: Sequence[float],
        approach_direction_world: Sequence[float],
        *,
        max_contact_force: float | None = None,
        approach_distance_m: float = 0.06,
    ) -> CapabilityResult[dict[str, object]]: ...

    def align_pregrasp(
        self,
        position: Sequence[float],
        quaternion: Sequence[float],
        approach_direction_world: Sequence[float],
        *,
        approach_distance_m: float = 0.06,
        clearance_m: float = 0.09,
    ) -> CapabilityResult[dict[str, object]]: ...

    def descend_and_close(
        self,
        position: Sequence[float],
        quaternion: Sequence[float],
        *,
        max_contact_force: float | None = None,
        position_tolerance_m: float | None = None,
    ) -> CapabilityResult[dict[str, object]]: ...

    def servo_to(
        self,
        position: Sequence[float],
        quaternion: Sequence[float],
        *,
        max_duration_s: float,
        tolerance_m: float,
    ) -> CapabilityResult[dict[str, object]]: ...


class ExternalToolServices(Protocol):
    def segment_text(self, image: np.ndarray, text: str) -> list[MaskCandidate]: ...

    def segment_point(
        self, image: np.ndarray, point_xy: Sequence[float]
    ) -> list[MaskCandidate]: ...

    def point(self, image: np.ndarray, text: str) -> tuple[float, float] | None: ...

    def plan_grasps(
        self, full_points: np.ndarray, object_points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def predict_grasps(
        self,
        *,
        backend: str,
        image: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        gripper_profile: GripperProfile,
        depth_source: str,
        collision_relaxed: bool = False,
    ) -> GraspServiceBatch: ...

    def predict_placements(
        self,
        *,
        image: np.ndarray,
        depth: np.ndarray,
        object_mask: np.ndarray,
        placement_region_mask: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        selected_grasp: Mapping[str, object],
        depth_source: str,
    ) -> GraspServiceBatch: ...

    def solve_ik(
        self,
        position: Sequence[float],
        quaternion: Sequence[float],
        previous_joints: Sequence[float] | None,
    ) -> Sequence[float] | None: ...


@dataclass(frozen=True, slots=True)
class _ImageArtifact:
    camera_name: str
    image: np.ndarray


@dataclass(frozen=True, slots=True)
class _MaskArtifact:
    camera_name: str
    mask: np.ndarray
    score: float


@dataclass(slots=True)
class _Detection:
    observation_handle: str
    camera_name: str
    prompt: str
    label: str
    mask: np.ndarray
    score: float
    box: tuple[float, float, float, float] | None
    backend: str
    centroid: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    point_count: int
    appearance_rgb_mean: np.ndarray
    selected: bool = False
    cross_view_consistent: bool = False
    views_used: int = 1
    identity_point_handle: str | None = None
    distractor_detection_id: str | None = None
    reference_point_handle: str | None = None
    instance_reference_distances_m: Mapping[str, float] | None = None
    instance_selection_margin_ratio: float | None = None


@dataclass(frozen=True, slots=True)
class _CapturedCamera:
    camera_name: str
    image: np.ndarray
    depth: np.ndarray | None
    intrinsics: np.ndarray
    extrinsics: np.ndarray
    rgb_handle: str
    depth_handle: str | None


@dataclass(frozen=True, slots=True)
class _CapturedObservation:
    cameras: tuple[_CapturedCamera, ...]
    robot_state: RobotState
    time_s: float | None = None


def _camera_content_fingerprint(camera: _CapturedCamera) -> str:
    digest = hashlib.sha256()
    digest.update(camera.camera_name.encode("utf-8"))
    for label, value in (
        ("image", camera.image),
        ("depth", camera.depth),
        ("intrinsics", camera.intrinsics),
        ("extrinsics", camera.extrinsics),
    ):
        digest.update(label.encode("ascii"))
        if value is None:
            digest.update(b"<none>")
            continue
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(
            ",".join(str(item) for item in array.shape).encode("ascii")
        )
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _ReachableGrasp:
    pose: np.ndarray
    score: float
    predicted_joint_travel_rad: float | None
    approach_direction_base: tuple[float, float, float]
    top_down_alignment: float
    convergence_risk: str = "high"
    predicted_grasp_width_m: float | None = None
    width_feasible: bool = True
    width_known: bool = True
    ik_feasible: bool = True
    style_preference_score: float = 1.0
    quality_flags: tuple[str, ...] = ()
    backend: str = "contact_graspnet"
    backend_candidate: Mapping[str, object] | None = None
    request_style_match: str = "not_requested"


@dataclass(slots=True)
class _TrackedEntity:
    object_name: str
    centroid: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    point_count: int
    full_points: np.ndarray
    object_points: np.ndarray
    source_point_handle: str | None = None
    source_camera_name: str | None = None
    appearance_rgb_mean: np.ndarray | None = None
    current_centroid: np.ndarray | None = None
    current_bounds_min: np.ndarray | None = None
    current_bounds_max: np.ndarray | None = None
    current_point_count: int | None = None
    eef_to_object_transform: np.ndarray | None = None
    attachment_verified: bool = False
    carried_geometry_epoch: int = 0
    attachment_current_point_handle: str | None = None
    last_correction_m: np.ndarray | None = None
    correction_flip_count: int = 0
    grasp_cache: dict[str, tuple[list[_ReachableGrasp], int, str]] = field(
        default_factory=dict
    )
    selected_grasp_candidate: Mapping[str, object] | None = None
    selected_grasp_descriptor: Mapping[str, object] | None = None
    issued_grasp_candidates: dict[int, Mapping[str, object]] = field(default_factory=dict)
    last_completion_observation_handle: str | None = None
    last_completion_centroid: np.ndarray | None = None
    last_completion_time_s: float | None = None
    last_completion_released: bool = False
    completion_samples: list[dict[str, object]] = field(default_factory=list)
    source_detection_id: str | None = None
    source_observation_handle: str | None = None
    identity_point_handle: str | None = None
    distractor_detection_id: str | None = None
    distractor_initial_centroid: np.ndarray | None = None
    distractor_initial_extent: np.ndarray | None = None
    instance_reference_distances_m: Mapping[str, float] | None = None
    instance_selection_margin_ratio: float | None = None
    distractor_initial_appearance_rgb_mean: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class _AlignedPregrasp:
    candidate: Mapping[str, object]
    eef_position: tuple[float, float, float]
    eef_quaternion: tuple[float, float, float, float]
    tracking_handle: str | None
    detection_id: str | None
    pregrasp_position: tuple[float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class _LiftProbeEvidence:
    tracking_handle: str | None
    attachment_attempt_id: int | None
    requested_height_m: float
    actual_vertical_displacement_m: float
    position_residual_m: float
    object_detected: bool | None
    supports_attachment: bool
    contradicts_attachment: bool
    execution: Mapping[str, object] | None = None


@dataclass(slots=True)
class _MarkedPoint:
    observation_handle: str
    camera_name: str
    rgb_handle: str
    depth_handle: str
    x_norm: float
    y_norm: float
    pixel_xy: tuple[float, float]
    position: np.ndarray
    label: str | None
    mask: np.ndarray
    segmentation_score: float
    mask_area_fraction: float
    depth_continuity: float
    geometry_diagonal_m: float
    appearance_rgb_mean: np.ndarray
    centroid: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    point_count: int
    grounding_status: str
    cross_view_consistent: bool = False
    views_used: int = 1
    carried_view_reconstruction: list[dict[str, object]] = field(default_factory=list)
    carried_selected_masks: tuple[tuple[str, np.ndarray], ...] = ()


@dataclass(frozen=True, slots=True)
class _PlacementEstimate:
    tracking_handle: str
    source_observation_handle: str
    target_point_handle: str
    destination_point_handle: str
    relation_type: str
    estimated_target_centroid: np.ndarray
    destination_position: np.ndarray
    destination_bounds_min: np.ndarray
    destination_bounds_max: np.ndarray
    target_visual_residual_m: float
    horizontal_distance_m: float
    release_height_m: float
    correction_m: np.ndarray
    correction_stable: bool
    relation_estimate: bool = False
    correction_stage: str = "relocalize_required"
    max_safe_step_m: float | None = None
    distance_to_nearest_obstacle_m: float | None = None
    measurement_confidence: float = 0.0
    quality_flags: tuple[str, ...] = ()
    geometry_diagnostics: Mapping[str, object] = field(default_factory=dict)


class ToolOperationError(RuntimeError):
    def __init__(self, category: FailureCategory, message: str) -> None:
        super().__init__(message)
        self.category = category


class ToolCapabilityBackend(PoseCandidateToolMixin, MotionToolMixin):
    """Compose sensor-only CaP-X tools while keeping arrays behind handles."""

    def __init__(
        self,
        *,
        robot: RobotToolBackend,
        services: ExternalToolServices,
        registry: EpisodeArtifactRegistry,
        tcp_offset_local_m: Sequence[float] = (0.0, 0.0, -0.1),
        gripper_profile: GripperProfile | None = None,
        default_approach_frame: str = "tool",
        completion_protocol: PublicCompletionProtocol | None = None,
        dehardcoded: bool = False,
        experience_snapshot: ExperienceSnapshot | None = None,
        graspnet_to_eef_transform: np.ndarray | None = None,
        motion_position_tolerance_m: float = 0.03,
        motion_orientation_tolerance_rad: float = math.radians(5.0),
        v15_protocol: bool = False,
        motion_tools: bool = False,
    ) -> None:
        self.robot = robot
        self.services = services
        self.registry = registry
        self._tcp_offset_local_m = _finite_vector(
            tcp_offset_local_m,
            3,
            "tcp_offset_local_m",
        )
        self.gripper_profile = gripper_profile or GripperProfile(
            profile_id="franka-panda-hand-v1",
            max_width_m=0.08,
            fingertip_depth_m=0.0584,
        )
        self.completion_protocol = (
            completion_protocol or load_public_completion_protocol()
        )
        self._dehardcoded = bool(dehardcoded)
        self._sensor_motion = bool(motion_tools)
        self._last_configuration_motion_time_s = getattr(robot, 'time_s', 0.)
        self._public_observation_history = {}
        self._experience_snapshot = experience_snapshot
        self._graspnet_to_eef_transform = (
            np.asarray(
                graspnet_to_eef_transform,
                dtype=np.float64,
            ).reshape(4, 4)
            if graspnet_to_eef_transform is not None
            else None
        )
        normalized_approach_frame = str(default_approach_frame).strip().lower()
        if normalized_approach_frame not in {"world", "tool"}:
            raise ValueError("default_approach_frame must be 'world' or 'tool'")
        self._default_approach_frame = normalized_approach_frame
        if (
            not math.isfinite(float(motion_position_tolerance_m))
            or float(motion_position_tolerance_m) <= 0.0
        ):
            raise ValueError(
                "motion_position_tolerance_m must be finite and positive"
            )
        if (
            not math.isfinite(float(motion_orientation_tolerance_rad))
            or float(motion_orientation_tolerance_rad) <= 0.0
        ):
            raise ValueError(
                "motion_orientation_tolerance_rad must be finite and positive"
            )
        self._motion_position_tolerance_m = float(
            motion_position_tolerance_m
        )
        self._motion_orientation_tolerance_rad = float(
            motion_orientation_tolerance_rad
        )
        self._v15_protocol = bool(v15_protocol)
        self._ik_cfg: np.ndarray | None = None
        self._active_attachment_handle: str | None = None
        self._pending_attachment_handle: str | None = None
        self._attachment_attempt_id = 0
        self._latest_lift_evidence: _LiftProbeEvidence | None = None
        self._aligned_pregrasp: _AlignedPregrasp | None = None
        self._last_descent_evidence: dict[str, object] | None = None
        # Most recent close-on-air (backend, object name, centroid). Grasp
        # estimates re-track the target under a fresh handle on every call,
        # so the retry match is by object identity: same prompt and a nearby
        # centroid. Used to route the retry to the mask-geometry candidate,
        # whose grasp point is the segmentation centroid.
        self._last_empty_close: dict[str, object] | None = None
        self._grasp_attempt_ledger = 0
        self._grasp_styles_tried: set[str] = set()
        self._grasp_backends_tried: set[str] = set()
        self._grasp_quality_ledger: Counter[str] = Counter()
        self._segment_cache: dict[
            tuple[str, str, str],
            dict[str, tuple[str, ...]],
        ] = {}
        self._observation_camera_fingerprints: dict[
            str,
            dict[str, str],
        ] = {}
        self._detection_groups: dict[
            tuple[str, str, str],
            tuple[str, ...],
        ] = {}
        self._v11_completion_evidence = (
            self.completion_protocol.evidence_only
        )

    @property
    def completion_protocol_metadata(self) -> Mapping[str, object]:
        return self.completion_protocol.public_metadata()

    def invoke(self, name: str, arguments: Mapping[str, object]) -> CapabilityResult:
        from dataclasses import replace
        from captain_harness.ports.deadline import check_deadline, current_deadline, retain_deadline_status
        method = getattr(self, name, None)
        if method is None or name.startswith("_"):
            return CapabilityResult.failed(
                FailureCategory.NOT_EXECUTED, f"unknown tool capability: {name!r}")
        steps_before = getattr(self.robot, 'simulation_steps', None)
        receipts = getattr(self.services, 'request_receipts', ())
        receipt_start = len(receipts)
        error_diagnostics = {}
        try:
            check_deadline()
            resolved = dict(arguments)
            for key in ("candidate", "selected_grasp"):
                if isinstance(resolved.get(key), (Mapping, str)):
                    resolved[key] = self._resolve_projected_grasp(resolved[key])
            result = method(**resolved)
        except ArtifactHandleError as exc:
            result = CapabilityResult.invalid_argument(str(exc))
        except TimeoutError as exc:
            result = CapabilityResult.failed(FailureCategory.TIMEOUT, str(exc))
            error_diagnostics = {"deadline": getattr(exc, 'deadline_receipt', None),
                "service_attempts": list(getattr(exc, 'attempts', ()))}
        except ToolOperationError as exc:
            result = CapabilityResult.failed(exc.category, str(exc))
        except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
            result = CapabilityResult.invalid_argument(f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            result = CapabilityResult.failed(FailureCategory.TOOL_ERROR, f"{type(exc).__name__}: {exc}")
        if not isinstance(result, CapabilityResult):
            return result  # The generic validated broker owns malformed returns.
        steps_after = getattr(self.robot, 'simulation_steps', None)
        measured_steps = (steps_after - steps_before
            if type(steps_before) is int and type(steps_after) is int
                and 0 <= steps_before <= steps_after else None)
        diagnostics = {**(result.diagnostics or {}), **error_diagnostics}
        # Operation completion and physical movement are different observations.
        # Successful reads/no-ops and missing counters must remain distinguishable.
        diagnostics['physical_execution'] = {
            'control_steps': measured_steps,
            'status': ('executed' if measured_steps else 'not_executed')
                if measured_steps is not None else 'unknown'}
        if not result.success:
            if measured_steps is not None and measured_steps > 0 and not result.executed:
                diagnostics['original_failure_category'] = result.failure_category.value
                result = replace(result, executed=True,
                    failure_category=(FailureCategory.BACKEND_ERROR
                        if result.failure_category is FailureCategory.NOT_EXECUTED else result.failure_category))
        if len(receipts) > receipt_start:
            diagnostics['service_requests'] = deepcopy(receipts[receipt_start:])
        if diagnostics:
            result = replace(result, diagnostics=diagnostics)
        return retain_deadline_status(result, current_deadline())

    def get_observation(self) -> CapabilityResult[dict[str, object]]:
        result = self.robot.get_observation_data()
        if not result.success or result.value is None:
            return CapabilityResult.failed(
                result.failure_category, result.message, executed=result.executed
            )
        if result.value.robot_state is None:
            raise ToolOperationError(FailureCategory.TOOL_ERROR, "robot state is unavailable")
        cameras: list[_CapturedCamera] = []
        public_cameras: list[dict[str, object]] = []
        for camera in result.value.cameras:
            image = np.asarray(Image.open(Path(camera.rgb_path)).convert("RGB"), dtype=np.uint8)
            depth = None
            depth_handle = None
            if camera.depth_path is not None:
                depth = np.asarray(np.load(camera.depth_path), dtype=np.float64).squeeze()
                depth_handle = self.registry.put("depth", depth)
            rgb_handle = self.registry.put(
                "rgb", _ImageArtifact(camera.camera_name, image)
            )
            captured = _CapturedCamera(
                camera.camera_name,
                image,
                depth,
                np.asarray(camera.intrinsics_3x3, dtype=np.float64).reshape(3, 3),
                np.asarray(camera.extrinsics_4x4, dtype=np.float64).reshape(4, 4),
                rgb_handle,
                depth_handle,
            )
            cameras.append(captured)
            public_cameras.append(
                {
                    "camera_name": camera.camera_name,
                    "rgb_handle": rgb_handle,
                    "depth_handle": depth_handle,
                    "intrinsics_3x3": [float(value) for value in captured.intrinsics.ravel()],
                    "extrinsics_4x4": [float(value) for value in captured.extrinsics.ravel()],
                }
            )
        captured_observation = _CapturedObservation(
            tuple(cameras), result.value.robot_state, result.value.time_s
        )
        observation_handle = self.registry.put("observation", captured_observation)
        from captain_harness.ports.environment import EnvironmentObservation, ImageObservation
        self._public_observation_history[observation_handle] = EnvironmentObservation(
            step=int(getattr(self.robot, 'simulation_steps', 0)),
            images=tuple(ImageObservation(c.camera_name, c.rgb_path, c.depth_path,
                         'head' if index == 0 else 'wrist',
                         {"intrinsics_3x3":list(c.intrinsics_3x3),
                          "camera_to_robot_base_4x4":list(c.extrinsics_4x4),
                          "depth_units":"meter","depth_convention":"camera optical z",
                          "pixel_origin":"top_left","sample_time_s":result.value.time_s,
                          "source":"declared_rgbd_sensor_and_camera_calibration"})
                         for index,c in enumerate(result.value.cameras)),
            public_state={'time_s':result.value.time_s,
                          'robot_state':self._public_robot_state(result.value.robot_state)})
        self._observation_camera_fingerprints[observation_handle] = {
            camera.camera_name: _camera_content_fingerprint(camera)
            for camera in cameras
        }
        return CapabilityResult.ok(
            {
                "observation_handle": observation_handle,
                "time_s": result.value.time_s,
                "cameras": public_cameras,
                "robot_state": self._public_robot_state(result.value.robot_state),
            }
        )

    def observe(self) -> CapabilityResult[dict[str, object]]:
        return self.get_observation()

    def _public_robot_state(self, state: RobotState) -> dict[str, object]:
        public = _robot_state_json(state)
        if self._v15_protocol:
            hand_position, hand_quaternion = self.robot.external_ik_target_from_eef_pose(
                state.end_effector_position_m, state.end_effector_quaternion_wxyz)
            tool_position = np.asarray(hand_position) - _quaternion_wxyz_to_matrix(hand_quaternion) @ self._tcp_offset_local_m
            public.update({
                "end_effector_pose_reference": "end_effector",
                "tool_position_m": tool_position.tolist(),
                "tool_quaternion_wxyz": list(hand_quaternion),
                "tool_pose_reference": "tool",
            })
        return public

    @staticmethod
    def _public_detection(
        detection_id: str,
        detection: _Detection,
    ) -> dict[str, object]:
        return {
            "detection_id": detection_id,
            "camera_name": detection.camera_name,
            "label": detection.label,
            "score": float(detection.score),
            "box": (
                list(detection.box)
                if detection.box is not None
                else None
            ),
            "centroid_m": [
                float(value) for value in detection.centroid
            ],
            "extent_m": [
                float(value)
                for value in detection.bounds_max - detection.bounds_min
            ],
            "backend": detection.backend,
            "selected": detection.selected,
        }

    def segment(
        self,
        prompt: str,
        observation_handle: str,
        camera_name: str | None = None,
    ) -> CapabilityResult[dict[str, object]]:
        normalized_prompt = " ".join(str(prompt).split())
        if not normalized_prompt:
            raise ValueError("prompt must not be empty")
        observation = cast(
            _CapturedObservation,
            self.registry.get(
                observation_handle,
                expected_kind="observation",
            ),
        )
        fingerprints = self._observation_camera_fingerprints.get(
            observation_handle
        )
        if fingerprints is None:
            fingerprints = {
                camera.camera_name: _camera_content_fingerprint(camera)
                for camera in observation.cameras
            }
            self._observation_camera_fingerprints[
                observation_handle
            ] = fingerprints
        if camera_name is not None and camera_name not in fingerprints:
            self._camera_by_name(observation, camera_name)
        selected_fingerprints = (
            (fingerprints[camera_name],)
            if camera_name is not None
            else tuple(
                fingerprints[name]
                for name in sorted(fingerprints)
            )
        )
        content_fingerprint = hashlib.sha256(
            "\0".join(selected_fingerprints).encode("ascii")
        ).hexdigest()
        cache_key = (
            content_fingerprint,
            camera_name or "*",
            normalized_prompt.casefold(),
        )
        cache_variants = self._segment_cache.get(cache_key)
        cached_detection_ids = (
            cache_variants.get(observation_handle)
            if cache_variants is not None
            else None
        )
        if cache_variants is not None and cached_detection_ids is None:
            source_ids = next(iter(cache_variants.values()))
            cloned_ids: list[str] = []
            for source_id in source_ids:
                source = cast(
                    _Detection,
                    self.registry.get(
                        source_id,
                        expected_kind="detection",
                    ),
                )
                cloned_ids.append(
                    self.registry.put(
                        "detection",
                        replace(
                            source,
                            observation_handle=observation_handle,
                            selected=False,
                        ),
                    )
                )
            cached_detection_ids = tuple(cloned_ids)
            cache_variants[observation_handle] = cached_detection_ids
        if cached_detection_ids is not None:
            self._register_detection_groups(cached_detection_ids)
            cached = [
                self._public_detection(
                    detection_id,
                    cast(
                        _Detection,
                        self.registry.get(
                            detection_id,
                            expected_kind="detection",
                        ),
                    ),
                )
                for detection_id in cached_detection_ids
            ]
            if not cached:
                return CapabilityResult.failed(
                    FailureCategory.PERCEPTION_UNCERTAIN,
                    "no_detection: cached immutable observation contains no "
                    "valid depth-backed detection",
                    executed=True,
                    value={
                        "reason": "no_detection",
                        "prompt": normalized_prompt,
                        "detections": [],
                        "cache_hit": True,
                    },
                )
            return CapabilityResult.ok(
                {
                    "observation_handle": observation_handle,
                    "prompt": normalized_prompt,
                    "detections": cached,
                    "cache_hit": True,
                }
            )
        cameras = (
            (self._camera_by_name(observation, camera_name),)
            if camera_name is not None
            else observation.cameras
        )
        detected: list[tuple[_CapturedCamera, MaskCandidate, np.ndarray, str]] = []
        for camera in cameras:
            candidates = self.services.segment_text(
                camera.image,
                normalized_prompt,
            )
            backend = "sam3_text"
            if not candidates:
                point = self.services.point(
                    camera.image,
                    normalized_prompt,
                )
                candidates = (
                    self.services.segment_point(camera.image, point)
                    if point is not None
                    else []
                )
                backend = "molmo_point_sam3"
            for candidate in candidates:
                mask = np.asarray(candidate.mask, dtype=bool).squeeze()
                if (
                    mask.ndim == 2
                    and mask.shape == camera.image.shape[:2]
                    and mask.any()
                    and math.isfinite(float(candidate.score))
                ):
                    detected.append((camera, candidate, mask, backend))
        detected.sort(key=lambda item: float(item[1].score), reverse=True)
        public: list[dict[str, object]] = []
        detection_ids: list[str] = []
        for camera, candidate, mask, backend in detected[:_MAX_PUBLIC_MASKS]:
            if camera.depth is None:
                continue
            points = _filter_point_cloud_noise(
                _depth_to_base_points(
                    camera.depth,
                    camera.intrinsics,
                    camera.extrinsics,
                    mask=mask,
                )
            )
            if len(points) < 3:
                continue
            centroid, bounds_min, bounds_max = _point_geometry(points)
            appearance = np.mean(
                np.asarray(camera.image, dtype=np.float64)[mask],
                axis=0,
            )
            detection = _Detection(
                observation_handle=observation_handle,
                camera_name=camera.camera_name,
                prompt=normalized_prompt,
                label=candidate.label or normalized_prompt,
                mask=mask.copy(),
                score=float(candidate.score),
                box=(
                    tuple(float(value) for value in candidate.box)
                    if candidate.box is not None
                    else None
                ),
                backend=backend,
                centroid=centroid,
                bounds_min=bounds_min,
                bounds_max=bounds_max,
                point_count=int(len(points)),
                appearance_rgb_mean=appearance,
            )
            detection_id = self.registry.put("detection", detection)
            detection_ids.append(detection_id)
            public.append(self._public_detection(detection_id, detection))
        self._segment_cache[cache_key] = {
            observation_handle: tuple(detection_ids)
        }
        self._register_detection_groups(detection_ids)
        if not public:
            return CapabilityResult.failed(
                FailureCategory.PERCEPTION_UNCERTAIN,
                "no_detection: perception services returned no valid "
                "depth-backed detection",
                executed=True,
                value={
                    "reason": "no_detection",
                    "prompt": normalized_prompt,
                    "detections": [],
                    "cache_hit": False,
                },
            )
        return CapabilityResult.ok(
            {
                "observation_handle": observation_handle,
                "prompt": normalized_prompt,
                "detections": public,
                "cache_hit": False,
            }
        )

    def select_detection(
        self,
        detection_id: str,
        identity_point_handle: str | None = None,
        reference_point_handle: str | None = None,
        selection_relation: str = "identity",
        previous_detection_id: str | None = None,
        identity_update: str = "same_target",
        identity_update_reason: str = "",
    ) -> CapabilityResult[dict[str, object]]:
        detection = cast(
            _Detection,
            self.registry.get(
                detection_id,
                expected_kind="detection",
            ),
        )
        continuity = None
        if self._sensor_motion:
            from .target_continuity import prepare_selection
            continuity = prepare_selection(self, detection_id, detection, previous_detection_id,
                                           identity_update, identity_update_reason)
        distractors, failure = bind_selection_evidence(
            self, detection_id, detection, identity_point_handle, reference_point_handle,
            selection_relation=selection_relation,
        )
        if failure is not None:
            return failure
        cache_hit = detection.selected
        detection.selected = True
        selection_history = {}
        if continuity is not None:
            from .target_continuity import record_selection
            selection_history = record_selection(self, continuity)
        return CapabilityResult.ok(
            {
                "detection_id": detection_id,
                "observation_handle": detection.observation_handle,
                "camera_name": detection.camera_name,
                "prompt": detection.prompt,
                "score": detection.score,
                "centroid_m": [
                    float(value) for value in detection.centroid
                ],
                "extent_m": [
                    float(value)
                    for value in detection.bounds_max - detection.bounds_min
                ],
                "selected": True,
                "cache_hit": cache_hit,
                **selection_history,
                **(
                    {
                        "selection_relation": selection_relation,
                        "identity_provenance_required": bool(distractors),
                        "identity_provenance_confirmed": bool(
                            not distractors
                            or detection.identity_point_handle
                        ),
                        "identity_point_handle": (
                            detection.identity_point_handle
                        ),
                        "distractor_detection_id": (
                            detection.distractor_detection_id
                        ),
                        "reference_point_handle": (
                            detection.reference_point_handle
                        ),
                        "instance_reference_distances_m": (
                            {
                                key: round(float(value), 4)
                                for key, value in (
                                    detection.instance_reference_distances_m
                                ).items()
                            }
                            if detection.instance_reference_distances_m
                            is not None
                            else None
                        ),
                        "instance_selection_margin_ratio": (
                            detection.instance_selection_margin_ratio
                        ),
                        "instance_margin_low": (
                            detection.instance_selection_margin_ratio
                            < 1.25
                            if detection.instance_selection_margin_ratio
                            is not None
                            else None
                        ),
                    }
                    if self._v15_protocol
                    else {}
                ),
            }
        )

    def _proximity_reference_check(
        self,
        detection_id: str,
        detection: _Detection,
        distractors: Sequence[str],
        reference_point_handle: str | None,
    ) -> CapabilityResult[dict[str, object]] | None:
        """Measured disambiguation for proximity relations.

        The selected instance must be the closest candidate to a public
        point on the reference object; the recorded distances become
        completion-verifier evidence. Returns None when the selection is
        measurably correct.
        """
        if reference_point_handle is None:
            return CapabilityResult.failed(
                FailureCategory.PERCEPTION_UNCERTAIN,
                "proximity_relation_requires_reference_point: the prompt "
                "explicitly requests nearest_xy, so also point or mark_point "
                "the reference object (not the target) in the same "
                "observation, then retry with reference_point_handle",
                executed=True,
                value={
                    "reason": (
                        "proximity_relation_requires_reference_point"
                    ),
                    "detection_id": detection_id,
                    "candidate_detection_ids": [
                        detection_id,
                        *distractors,
                    ],
                    "required_call": (
                        "select_detection(detection_id, "
                        "identity_point_handle=..., "
                        "reference_point_handle="
                        "point.value['point_handle'])"
                    ),
                },
            )
        reference = cast(
            _MarkedPoint,
            self.registry.get(
                reference_point_handle,
                expected_kind="marked_point",
            ),
        )
        if point_source_error(self.registry, detection, reference) is not None:
            return CapabilityResult.failed(
                FailureCategory.PERCEPTION_UNCERTAIN,
                "reference_point_stale: the reference point must come "
                "from the same observation as the selected detection",
                executed=True,
                value={
                    "reason": "reference_point_stale",
                    "detection_id": detection_id,
                    "reference_point_handle": reference_point_handle,
                },
            )
        candidate_ids = (detection_id, *distractors)
        if reference.camera_name == detection.camera_name:
            pixel_x = int(round(reference.pixel_xy[0]))
            pixel_y = int(round(reference.pixel_xy[1]))
            for candidate_id in candidate_ids:
                candidate = cast(
                    _Detection,
                    self.registry.get(
                        candidate_id,
                        expected_kind="detection",
                    ),
                )
                mask = candidate.mask
                if (
                    0 <= pixel_y < mask.shape[0]
                    and 0 <= pixel_x < mask.shape[1]
                    and mask[pixel_y, pixel_x]
                ):
                    return CapabilityResult.failed(
                        FailureCategory.PERCEPTION_UNCERTAIN,
                        "reference_point_on_candidate_instance: the "
                        "reference point must mark the reference object "
                        "named by the relation, not one of the candidate "
                        "target instances",
                        executed=True,
                        value={
                            "reason": (
                                "reference_point_on_candidate_instance"
                            ),
                            "detection_id": detection_id,
                            "reference_point_handle": (
                                reference_point_handle
                            ),
                            "conflicting_detection_id": candidate_id,
                        },
                    )
        reference_failure = self._reference_semantics_check(
            detection,
            reference,
            reference_point_handle,
        )
        if reference_failure is not None:
            return reference_failure
        reference_xy = np.asarray(
            reference.position,
            dtype=np.float64,
        ).reshape(3)[:2]
        distances: dict[str, float] = {}
        for candidate_id in candidate_ids:
            candidate = cast(
                _Detection,
                self.registry.get(
                    candidate_id,
                    expected_kind="detection",
                ),
            )
            distances[candidate_id] = float(
                np.linalg.norm(candidate.centroid[:2] - reference_xy)
            )
        closest_id = min(distances, key=distances.get)
        if closest_id != detection_id:
            return CapabilityResult.failed(
                FailureCategory.PERCEPTION_UNCERTAIN,
                "selected_instance_not_closest: the proximity relation "
                "selects the candidate closest to the reference point; "
                f"measured distances put {closest_id} closer than the "
                "selected detection",
                executed=True,
                value={
                    "reason": "selected_instance_not_closest",
                    "detection_id": detection_id,
                    "closest_detection_id": closest_id,
                    "instance_reference_distances_m": {
                        key: round(value, 4)
                        for key, value in distances.items()
                    },
                },
            )
        selected_distance = distances[detection_id]
        runner_up = min(
            (
                value
                for key, value in distances.items()
                if key != detection_id
            ),
            default=None,
        )
        margin_ratio = (
            float(runner_up / max(selected_distance, 1e-6))
            if runner_up is not None
            else None
        )
        detection.reference_point_handle = reference_point_handle
        detection.instance_reference_distances_m = {
            key: float(value) for key, value in distances.items()
        }
        detection.instance_selection_margin_ratio = margin_ratio
        return None

    def _reference_semantics_check(
        self,
        detection: _Detection,
        reference: _MarkedPoint,
        reference_point_handle: str,
    ) -> CapabilityResult[dict[str, object]] | None:
        """Verify the reference point lands on the named reference object.

        The harness extracts the reference noun from the relational prompt
        itself and segments it in the same observation; the model's point
        must fall on one of those masks (or within 5 cm of a segmented
        centroid). In the V15.1 mainline the model, once its bowl-placed
        reference was rejected, satisfied the gate by pointing at an
        arbitrary passable object. Segmentation-service failures skip this
        check rather than brick the gate.
        """
        reference_phrase = _reference_phrase_from_prompt(detection.prompt)
        if not reference_phrase:
            return None
        try:
            segmented = self.segment(
                reference_phrase,
                detection.observation_handle,
                camera_name=reference.camera_name,
            )
        except Exception:
            return None
        if not segmented.success or not isinstance(
            segmented.value, Mapping
        ):
            return None
        items = segmented.value.get("detections")
        if not isinstance(items, Sequence) or not items:
            return None
        pixel_x = int(round(reference.pixel_xy[0]))
        pixel_y = int(round(reference.pixel_xy[1]))
        reference_xy = np.asarray(
            reference.position,
            dtype=np.float64,
        ).reshape(3)[:2]
        for item in items:
            if not isinstance(item, Mapping):
                continue
            item_id = item.get("detection_id")
            if not isinstance(item_id, str):
                continue
            try:
                segmented_detection = cast(
                    _Detection,
                    self.registry.get(
                        item_id,
                        expected_kind="detection",
                    ),
                )
            except ArtifactHandleError:
                continue
            if (
                segmented_detection.camera_name == reference.camera_name
            ):
                mask = segmented_detection.mask
                if (
                    0 <= pixel_y < mask.shape[0]
                    and 0 <= pixel_x < mask.shape[1]
                    and mask[pixel_y, pixel_x]
                ):
                    return None
            if (
                float(
                    np.linalg.norm(
                        segmented_detection.centroid[:2] - reference_xy
                    )
                )
                <= 0.05
            ):
                return None
        return CapabilityResult.failed(
            FailureCategory.PERCEPTION_UNCERTAIN,
            "reference_point_not_on_reference_object: the reference point "
            f"must mark the '{reference_phrase}' named by the relation; "
            "the segmentation of that phrase in the same observation does "
            "not contain the point",
            executed=True,
            value={
                "reason": "reference_point_not_on_reference_object",
                "reference_phrase": reference_phrase,
                "reference_point_handle": reference_point_handle,
                "segmented_reference_count": len(items),
            },
        )

    def _register_detection_groups(
        self,
        detection_ids: Sequence[str],
    ) -> None:
        grouped: dict[tuple[str, str, str], list[str]] = {}
        for detection_id in detection_ids:
            detection = cast(
                _Detection,
                self.registry.get(
                    detection_id,
                    expected_kind="detection",
                ),
            )
            key = (
                detection.observation_handle,
                detection.camera_name,
                detection.prompt.casefold(),
            )
            grouped.setdefault(key, []).append(detection_id)
        for key, values in grouped.items():
            self._detection_groups[key] = tuple(values)

    def _identity_distractors(
        self,
        detection_id: str,
        detection: _Detection,
    ) -> tuple[str, ...]:
        key = (
            detection.observation_handle,
            detection.camera_name,
            detection.prompt.casefold(),
        )
        selected_extent = np.maximum(
            detection.bounds_max - detection.bounds_min,
            1e-6,
        )
        selected_size = float(np.linalg.norm(selected_extent))
        candidates: list[tuple[float, str]] = []
        for peer_id in self._detection_groups.get(key, ()):
            if peer_id == detection_id:
                continue
            peer = cast(
                _Detection,
                self.registry.get(
                    peer_id,
                    expected_kind="detection",
                ),
            )
            peer_extent = np.maximum(
                peer.bounds_max - peer.bounds_min,
                1e-6,
            )
            size_ratio = float(
                np.linalg.norm(peer_extent) / max(selected_size, 1e-6)
            )
            mask_union = np.count_nonzero(detection.mask | peer.mask)
            mask_iou = (
                float(
                    np.count_nonzero(detection.mask & peer.mask)
                    / mask_union
                )
                if mask_union
                else 0.0
            )
            centroid_distance = float(
                np.linalg.norm(peer.centroid - detection.centroid)
            )
            appearance_distance = float(
                np.linalg.norm(
                    peer.appearance_rgb_mean
                    - detection.appearance_rgb_mean
                )
            )
            if (
                0.60 <= size_ratio <= 1.65
                and mask_iou < 0.35
                and centroid_distance > 0.04
                and appearance_distance <= 140.0
            ):
                candidates.append((-peer.score, peer_id))
        candidates.sort()
        return tuple(peer_id for _, peer_id in candidates)

    def point(
        self,
        prompt: str,
        observation_handle: str,
        camera_name: str = "agentview",
    ) -> CapabilityResult[dict[str, object]]:
        normalized_prompt = " ".join(str(prompt).split())
        if not normalized_prompt:
            raise ValueError("prompt must not be empty")
        observation = cast(
            _CapturedObservation,
            self.registry.get(
                observation_handle,
                expected_kind="observation",
            ),
        )
        camera = self._camera_by_name(observation, camera_name)
        point_xy = self.services.point(camera.image, normalized_prompt)
        if point_xy is None:
            return CapabilityResult.failed(
                FailureCategory.PERCEPTION_UNCERTAIN,
                f"no_point: point service returned no location for "
                f"{normalized_prompt!r}",
                executed=True,
                value={"reason": "no_point", "prompt": normalized_prompt,
                    "observation_handle": observation_handle, "camera_name": camera_name,
                    "grounding_recovery": {"absence_established": False,
                        "guidance": "The point service could not localize this request. Inspect the supplied "
                        "current image. If the intended feature is visible, mark_point on this observation "
                        "and camera using its normalized image coordinates; you need not keep rewording "
                        "the same point query. If occluded, inspect the last reliable before/after evidence "
                        "and acquire a view that resolves that occlusion. Do not invent coordinates."}},
            )
        height, width = camera.image.shape[:2]
        x_norm = float(point_xy[0]) / max(1, width - 1)
        y_norm = float(point_xy[1]) / max(1, height - 1)
        marked = self.mark_point(
            observation_handle,
            camera_name,
            x_norm,
            y_norm,
            label=None,
        )
        if not marked.success or not isinstance(marked.value, Mapping):
            return marked
        return CapabilityResult.ok(
            {
                **dict(marked.value),
                "prompt": normalized_prompt,
            }
        )

    def mark_point(
        self,
        observation_handle: str,
        camera_name: str,
        x_norm: float,
        y_norm: float,
        label: str | None = None,
        granularity: str = "object",
    ) -> CapabilityResult[dict[str, object]]:
        if granularity not in ("object", "part", "whole"):
            raise ValueError("granularity must be object, part or whole")
        observation = cast(
            _CapturedObservation,
            self.registry.get(
                observation_handle,
                expected_kind="observation",
            ),
        )
        camera = self._camera_by_name(observation, camera_name)
        if camera.depth is None or camera.depth_handle is None:
            raise ToolOperationError(
                FailureCategory.TOOL_ERROR,
                f"camera {camera_name!r} has no public depth",
            )
        try:
            position = np.asarray(
                pixel_to_world(
                    camera.depth,
                    camera.intrinsics,
                    camera.extrinsics,
                    x_norm=float(x_norm),
                    y_norm=float(y_norm),
                ),
                dtype=np.float64,
            )
        except DepthUnavailableError as exc:
            raise ToolOperationError(
                FailureCategory.TOOL_ERROR,
                str(exc),
            ) from exc
        height, width = camera.image.shape[:2]
        pixel_xy = (
            float(round(float(x_norm) * (width - 1))),
            float(round(float(y_norm) * (height - 1))),
        )
        normalized_label = (
            " ".join(label.split()) if isinstance(label, str) and label.strip() else None
        )
        candidates = self.services.segment_point(camera.image, pixel_xy)
        pixel_x = int(pixel_xy[0])
        pixel_y = int(pixel_xy[1])
        valid_candidates: list[tuple[MaskCandidate, np.ndarray]] = []
        for candidate in candidates:
            mask = np.asarray(candidate.mask, dtype=bool).squeeze()
            if (
                mask.ndim == 2
                and mask.shape == camera.image.shape[:2]
                and mask.any()
                and mask[pixel_y, pixel_x]
                and math.isfinite(float(candidate.score))
            ):
                valid_candidates.append((candidate, mask))
        quality_flags: list[str] = []
        if not valid_candidates:
            if not self._dehardcoded:
                return CapabilityResult.failed(
                    FailureCategory.PERCEPTION_UNCERTAIN,
                    "grounding_uncertain: point-guided segmentation did not "
                    "return a valid mask containing the selected pixel",
                    executed=True,
                    value={
                        "point_handle": None,
                        "grounding_status": "grounding_uncertain",
                        "quality_flags": ["no_mask_at_selected_pixel"],
                    },
                )
            mask = np.zeros(camera.image.shape[:2], dtype=bool)
            radius = 4
            mask[
                max(0, pixel_y - radius) : min(height, pixel_y + radius + 1),
                max(0, pixel_x - radius) : min(width, pixel_x + radius + 1),
            ] = True
            candidate = MaskCandidate(
                mask=mask,
                score=0.0,
                label=normalized_label,
            )
            quality_flags.append("no_mask_at_selected_pixel")
        else:
            # v7: point-guided SAM3 returns several granularities containing the pixel. 'object' keeps the historical
            # best-score choice; 'part' takes the smallest mask (drawer front, handle, knob); 'whole' the largest.
            if granularity == "part":
                candidate, mask = min(valid_candidates, key=lambda item: int(np.count_nonzero(item[1])))
            elif granularity == "whole":
                candidate, mask = max(valid_candidates, key=lambda item: int(np.count_nonzero(item[1])))
            else:
                candidate, mask = max(
                    valid_candidates,
                    key=lambda item: float(item[0].score),
                )
        mask_area_fraction = float(np.count_nonzero(mask) / mask.size)
        ys = slice(max(0, pixel_y - 3), min(height, pixel_y + 4))
        xs = slice(max(0, pixel_x - 3), min(width, pixel_x + 4))
        depth_patch = np.asarray(camera.depth[ys, xs], dtype=np.float64)
        finite_depth = depth_patch[np.isfinite(depth_patch) & (depth_patch > 0)]
        selected_depth = float(
            np.median(
                np.asarray(camera.depth, dtype=np.float64)[
                    max(0, pixel_y - 2) : min(height, pixel_y + 3),
                    max(0, pixel_x - 2) : min(width, pixel_x + 3),
                ][
                    np.isfinite(
                        np.asarray(camera.depth, dtype=np.float64)[
                            max(0, pixel_y - 2) : min(height, pixel_y + 3),
                            max(0, pixel_x - 2) : min(width, pixel_x + 3),
                        ]
                    )
                    & (
                        np.asarray(camera.depth, dtype=np.float64)[
                            max(0, pixel_y - 2) : min(height, pixel_y + 3),
                            max(0, pixel_x - 2) : min(width, pixel_x + 3),
                        ]
                        > 0
                    )
                ]
            )
        )
        depth_tolerance = max(0.015, abs(selected_depth) * 0.05)
        depth_continuity = (
            float(
                np.count_nonzero(
                    np.abs(finite_depth - selected_depth) <= depth_tolerance
                )
                / len(finite_depth)
            )
            if len(finite_depth)
            else 0.0
        )
        raw_object_points = _depth_to_base_points(
            camera.depth,
            camera.intrinsics,
            camera.extrinsics,
            mask=mask,
        )
        object_points = _filter_point_cloud_noise(raw_object_points)
        if self._dehardcoded and len(object_points) < 3:
            object_points = raw_object_points
            quality_flags.append("sparse_geometry")
        geometry_diagonal = (
            float(np.linalg.norm(np.max(object_points, axis=0) - np.min(object_points, axis=0)))
            if len(object_points) >= 3
            else math.inf
        )
        semantic_distance: float | None = None
        semantic_consistent = True
        if normalized_label is not None:
            semantic_point = self.services.point(
                camera.image,
                normalized_label,
            )
            if semantic_point is None:
                semantic_consistent = False
            else:
                semantic_distance = float(
                    np.linalg.norm(
                        np.asarray(semantic_point, dtype=np.float64)
                        - np.asarray(pixel_xy, dtype=np.float64)
                    )
                )
                semantic_consistent = semantic_distance <= max(
                    3.0,
                    0.18 * max(width, height),
                )
        if mask.size > 64 and not (
            _MIN_MARK_MASK_AREA_FRACTION
            <= mask_area_fraction
            <= _MAX_MARK_MASK_AREA_FRACTION
        ):
            quality_flags.append("implausible_mask_area")
        if depth_continuity < _MIN_MARK_DEPTH_CONTINUITY:
            quality_flags.append("depth_discontinuity")
        if not (
            _MIN_MARK_GEOMETRY_DIAGONAL_M
            <= geometry_diagonal
            <= _MAX_MARK_GEOMETRY_DIAGONAL_M
        ):
            quality_flags.append("implausible_geometry_extent")
        if not semantic_consistent:
            quality_flags.append("label_point_disagreement")
        critical_quality_flags = [
            flag
            for flag in quality_flags
            if flag != "label_point_disagreement"
        ]
        if critical_quality_flags and not self._dehardcoded:
            return CapabilityResult.failed(
                FailureCategory.PERCEPTION_UNCERTAIN,
                "grounding_uncertain: "
                + ", ".join(critical_quality_flags),
                executed=True,
                value={
                    "point_handle": None,
                    "grounding_status": "grounding_uncertain",
                    "quality_flags": quality_flags,
                    "mask_area_fraction": mask_area_fraction,
                    "depth_continuity": depth_continuity,
                    "geometry_diagonal_m": geometry_diagonal,
                    "semantic_point_distance_px": semantic_distance,
                },
            )
        centroid, bounds_min, bounds_max = _point_geometry(object_points)
        grounding_status = (
            "grounded_with_warning"
            if quality_flags
            else "grounded"
        )
        appearance_rgb_mean = np.mean(
            np.asarray(camera.image, dtype=np.float64)[mask],
            axis=0,
        )
        handle = self.registry.put(
            "marked_point",
            _MarkedPoint(
                observation_handle=observation_handle,
                camera_name=camera.camera_name,
                rgb_handle=camera.rgb_handle,
                depth_handle=camera.depth_handle,
                x_norm=float(x_norm),
                y_norm=float(y_norm),
                pixel_xy=pixel_xy,
                position=position.copy(),
                label=normalized_label,
                mask=mask.copy(),
                segmentation_score=float(candidate.score),
                mask_area_fraction=mask_area_fraction,
                depth_continuity=depth_continuity,
                geometry_diagonal_m=geometry_diagonal,
                appearance_rgb_mean=appearance_rgb_mean,
                centroid=centroid,
                bounds_min=bounds_min,
                bounds_max=bounds_max,
                point_count=int(len(object_points)),
                grounding_status=grounding_status,
            ),
        )
        # v7: a model-chosen pixel with a real SAM3 mask is also a selectable detection, so identity decided from
        # the image can flow straight into measure_object_geometry / compare_object_observations.
        detection_id: str | None = None
        detection_selected = False
        if float(candidate.score) > 0.0 and "no_mask_at_selected_pixel" not in quality_flags:
            ys_idx, xs_idx = np.nonzero(mask)
            detection = _Detection(
                observation_handle=observation_handle,
                camera_name=camera.camera_name,
                prompt=f"mark_point:{normalized_label or 'unlabeled'}",
                label=normalized_label or "model_point",
                mask=mask.copy(),
                score=float(candidate.score),
                box=(float(xs_idx.min()), float(ys_idx.min()), float(xs_idx.max()), float(ys_idx.max())),
                backend="model_point_sam3",
                centroid=centroid,
                bounds_min=bounds_min,
                bounds_max=bounds_max,
                point_count=int(len(object_points)),
                appearance_rgb_mean=appearance_rgb_mean,
            )
            detection_id = self.registry.put("detection", detection)
            try:
                detection_selected = bool(self.select_detection(detection_id, identity_point_handle=handle).success)
            except Exception:  # selection bookkeeping must never fail the point itself
                detection_selected = False
        return CapabilityResult.ok(
            {
                "point_handle": handle,
                "detection_id": detection_id,
                "detection_selected": detection_selected,
                "observation_handle": observation_handle,
                "camera_name": camera.camera_name,
                "rgb_handle": camera.rgb_handle,
                "depth_handle": camera.depth_handle,
                "x_norm": float(x_norm),
                "y_norm": float(y_norm),
                "pixel_xy": [float(value) for value in pixel_xy],
                "position_m": [float(value) for value in position],
                "label": normalized_label,
                "grounding_status": grounding_status,
                "cross_view_consistent": False,
                "views_used": 1,
                "quality_flags": quality_flags,
                "segmentation_score": float(candidate.score),
                "mask_area_fraction": mask_area_fraction,
                "depth_continuity": depth_continuity,
                "geometry_diagonal_m": geometry_diagonal,
                "semantic_point_distance_px": semantic_distance,
                "point_centroid_m": [
                    float(value) for value in centroid
                ],
                "point_bounds_min_m": [
                    float(value) for value in bounds_min
                ],
                "point_bounds_max_m": [
                    float(value) for value in bounds_max
                ],
                "point_count": int(len(object_points)),
            }
        )

    def track_entity_from_mark(
        self,
        point_handle: str,
        label: str,
        use_multiview: bool = True,
    ) -> CapabilityResult[dict[str, object]]:
        marked, full_points, object_points, score = self._points_from_mark(
            point_handle,
            use_multiview=use_multiview,
        )
        centroid, bounds_min, bounds_max = _point_geometry(object_points)
        object_name = " ".join(label.split())
        if not object_name:
            raise ValueError("label must not be empty")
        tracking_handle = self.registry.put(
            "entity_track",
            _TrackedEntity(
                object_name=object_name,
                centroid=centroid,
                bounds_min=bounds_min,
                bounds_max=bounds_max,
                point_count=int(len(object_points)),
                full_points=np.asarray(full_points, dtype=np.float64).copy(),
                object_points=np.asarray(
                    object_points,
                    dtype=np.float64,
                ).copy(),
                source_point_handle=point_handle,
                source_camera_name=marked.camera_name,
                appearance_rgb_mean=marked.appearance_rgb_mean.copy(),
                current_centroid=centroid.copy(),
                current_bounds_min=bounds_min.copy(),
                current_bounds_max=bounds_max.copy(),
                current_point_count=int(len(object_points)),
            ),
        )
        return CapabilityResult.ok(
            {
                "tracking_handle": tracking_handle,
                "object_name": object_name,
                "pre_centroid_m": [float(value) for value in centroid],
                "pre_bounds_min_m": [float(value) for value in bounds_min],
                "pre_bounds_max_m": [float(value) for value in bounds_max],
                "pre_point_count": int(len(object_points)),
                "source_point_handle": point_handle,
                "source_observation_handle": marked.observation_handle,
                "source_camera_name": marked.camera_name,
                "source_pixel_xy": [
                    float(value) for value in marked.pixel_xy
                ],
                "segmentation_score": score,
                "grounding_status": marked.grounding_status,
                "cross_view_consistent": marked.cross_view_consistent,
                "views_used": marked.views_used,
            }
        )

    def mark_attached_target(
        self,
        tracking_handle: str,
        observation_handle: str,
        camera_name: str,
        x_norm: float,
        y_norm: float,
    ) -> CapabilityResult[dict[str, object]]:
        tracked = cast(
            _TrackedEntity,
            self.registry.get(
                tracking_handle,
                expected_kind="entity_track",
            ),
        )
        attachment_transform_available = bool(
            tracked.attachment_verified
            and tracked.eef_to_object_transform is not None
        )
        if not attachment_transform_available and not self._dehardcoded:
            return CapabilityResult.failed(
                FailureCategory.SAFETY_REJECTED,
                "attachment_unverified: attached-target grounding requires "
                "a verified public attachment transform",
            )
        observation = cast(
            _CapturedObservation,
            self.registry.get(
                observation_handle,
                expected_kind="observation",
            ),
        )
        camera = self._camera_by_name(observation, camera_name)
        if camera.depth is None or camera.depth_handle is None:
            raise ToolOperationError(
                FailureCategory.PERCEPTION_UNCERTAIN,
                f"camera {camera_name!r} has no public depth",
            )
        height, width = camera.image.shape[:2]
        pixel_xy = (
            float(round(float(x_norm) * (width - 1))),
            float(round(float(y_norm) * (height - 1))),
        )
        pixel_x = int(pixel_xy[0])
        pixel_y = int(pixel_xy[1])
        position = np.asarray(
            pixel_to_world(
                camera.depth,
                camera.intrinsics,
                camera.extrinsics,
                x_norm=float(x_norm),
                y_norm=float(y_norm),
            ),
            dtype=np.float64,
        )
        world_from_eef = _transform_from_pose(
            observation.robot_state.end_effector_position_m,
            observation.robot_state.end_effector_quaternion_wxyz,
        )
        predicted = (
            (
                world_from_eef
                @ cast(np.ndarray, tracked.eef_to_object_transform)
            )[:3, 3]
            if attachment_transform_available
            else np.asarray(
                tracked.current_centroid
                if tracked.current_centroid is not None
                else tracked.centroid,
                dtype=np.float64,
            )
        )
        target_extent = np.maximum(
            tracked.bounds_max - tracked.bounds_min,
            1e-6,
        )
        max_prediction_residual = max(
            0.08,
            min(0.18, float(np.linalg.norm(target_extent))),
        )
        prediction_residual = float(np.linalg.norm(position - predicted))
        radius = 4
        ys = slice(max(0, pixel_y - radius), min(height, pixel_y + radius + 1))
        xs = slice(max(0, pixel_x - radius), min(width, pixel_x + radius + 1))
        patch_depth = np.asarray(camera.depth[ys, xs], dtype=np.float64)
        finite = patch_depth[np.isfinite(patch_depth) & (patch_depth > 0)]
        selected_depth = float(np.linalg.norm(position - camera.extrinsics[:3, 3]))
        depth_tolerance = max(0.015, selected_depth * 0.06)
        depth_continuity = (
            float(
                np.count_nonzero(
                    np.abs(finite - np.median(finite)) <= depth_tolerance
                )
                / len(finite)
            )
            if len(finite)
            else 0.0
        )
        patch_rgb = np.asarray(camera.image[ys, xs], dtype=np.float64)
        appearance_mean = np.mean(
            patch_rgb.reshape(-1, 3),
            axis=0,
        )
        appearance_distance = (
            float(
                np.linalg.norm(
                    appearance_mean - tracked.appearance_rgb_mean
                )
            )
            if tracked.appearance_rgb_mean is not None
            else 0.0
        )
        quality_flags = []
        if not attachment_transform_available:
            quality_flags.append("attachment_transform_unavailable")
        if prediction_residual > max_prediction_residual:
            quality_flags.append("outside_attachment_prediction")
        if depth_continuity < 0.45:
            quality_flags.append("depth_discontinuity")
        if appearance_distance > 110.0:
            quality_flags.append("appearance_mismatch")
        if quality_flags and not self._dehardcoded:
            return CapabilityResult.failed(
                FailureCategory.PERCEPTION_UNCERTAIN,
                "grounding_uncertain: " + ", ".join(quality_flags),
                executed=True,
                value={
                    "point_handle": None,
                    "grounding_status": "grounding_uncertain",
                    "quality_flags": quality_flags,
                    "prediction_residual_m": prediction_residual,
                    "max_prediction_residual_m": max_prediction_residual,
                    "depth_continuity": depth_continuity,
                    "appearance_distance_rgb": appearance_distance,
                },
            )
        mask = np.zeros((height, width), dtype=bool)
        mask[ys, xs] = True
        local_points = _filter_point_cloud_noise(
            _depth_to_base_points(
                camera.depth,
                camera.intrinsics,
                camera.extrinsics,
                mask=mask,
            )
        )
        if len(local_points) < 3:
            local_points = np.vstack(
                (
                    position,
                    position + np.array([1e-4, 0.0, 0.0]),
                    position + np.array([0.0, 1e-4, 0.0]),
                )
            )
        centroid, bounds_min, bounds_max = _point_geometry(local_points)
        handle = self.registry.put(
            "marked_point",
            _MarkedPoint(
                observation_handle=observation_handle,
                camera_name=camera.camera_name,
                rgb_handle=camera.rgb_handle,
                depth_handle=camera.depth_handle,
                x_norm=float(x_norm),
                y_norm=float(y_norm),
                pixel_xy=pixel_xy,
                position=position.copy(),
                label=tracked.object_name,
                mask=mask,
                segmentation_score=1.0,
                mask_area_fraction=float(np.count_nonzero(mask) / mask.size),
                depth_continuity=depth_continuity,
                geometry_diagonal_m=float(
                    np.linalg.norm(bounds_max - bounds_min)
                ),
                appearance_rgb_mean=appearance_mean,
                centroid=centroid,
                bounds_min=bounds_min,
                bounds_max=bounds_max,
                point_count=int(len(local_points)),
                grounding_status="grounded_attachment",
            ),
        )
        return CapabilityResult.ok(
            {
                "point_handle": handle,
                "tracking_handle": tracking_handle,
                "observation_handle": observation_handle,
                "camera_name": camera.camera_name,
                "pixel_xy": [float(value) for value in pixel_xy],
                "position_m": [float(value) for value in position],
                "predicted_target_position_m": [
                    float(value) for value in predicted
                ],
                "prediction_residual_m": prediction_residual,
                "max_prediction_residual_m": max_prediction_residual,
                "depth_continuity": depth_continuity,
                "appearance_distance_rgb": appearance_distance,
                "grounding_status": "grounded_attachment",
                "quality_flags": quality_flags,
            }
        )

    def segment_text(
        self, rgb_handle: str, text_prompt: str
    ) -> CapabilityResult[dict[str, object]]:
        image = cast(_ImageArtifact, self.registry.get(rgb_handle, expected_kind="rgb"))
        candidates = self.services.segment_text(image.image, text_prompt)
        return CapabilityResult.ok(self._store_masks(image.camera_name, candidates))

    def segment_point(
        self, rgb_handle: str, point_xy: Sequence[float]
    ) -> CapabilityResult[dict[str, object]]:
        image = cast(_ImageArtifact, self.registry.get(rgb_handle, expected_kind="rgb"))
        x, y = _finite_vector(point_xy, 2, "point_xy")
        height, width = image.image.shape[:2]
        if x >= width or y >= height:
            raise ValueError(f"point_xy lies outside image bounds {width}x{height}")
        candidates = self.services.segment_point(image.image, (x, y))
        return CapabilityResult.ok(self._store_masks(image.camera_name, candidates))

    def point_molmo(
        self, rgb_handle: str, text_prompt: str
    ) -> CapabilityResult[dict[str, object]]:
        image = cast(_ImageArtifact, self.registry.get(rgb_handle, expected_kind="rgb"))
        point = self.services.point(image.image, text_prompt)
        if point is None or len(point) != 2 or not np.isfinite(point).all():
            raise ToolOperationError(
                FailureCategory.TOOL_ERROR, f"Molmo returned no point for {text_prompt!r}"
            )
        return CapabilityResult.ok({"point_xy": [float(point[0]), float(point[1])]})

    def construct_point_cloud(
        self,
        observation_handle: str,
        mask_handles: Sequence[str] | None = None,
        use_multiview: bool = True,
        detection_id: str | None = None,
    ) -> CapabilityResult[dict[str, object]]:
        observation = cast(
            _CapturedObservation,
            self.registry.get(observation_handle, expected_kind="observation"),
        )
        if detection_id is not None:
            detection = cast(
                _Detection,
                self.registry.get(
                    detection_id,
                    expected_kind="detection",
                ),
            )
            if detection.observation_handle != observation_handle:
                raise ValueError(
                    "detection does not belong to the supplied observation"
                )
            if not detection.selected:
                return CapabilityResult.failed(
                    FailureCategory.PERCEPTION_UNCERTAIN,
                    "detection_selection_required: call select_detection "
                    "before constructing target geometry",
                )
            full_points, object_points = self._clouds_from_detection(
                observation,
                detection,
                use_multiview=use_multiview,
            )
            return CapabilityResult.ok(
                {
                    **self._store_clouds(full_points, object_points),
                    "detection_id": detection_id,
                    "views_used": detection.views_used,
                    "cross_view_consistent": (
                        detection.cross_view_consistent
                    ),
                }
            )
        if not mask_handles:
            raise ValueError(
                "construct_point_cloud requires mask_handles or detection_id"
            )
        masks = [
            cast(_MaskArtifact, self.registry.get(handle, expected_kind="mask"))
            for handle in mask_handles
        ]
        full_points, object_points = self._clouds_from_masks(
            observation, masks, use_multiview=use_multiview
        )
        return CapabilityResult.ok(self._store_clouds(full_points, object_points))

    def track_entity(
        self,
        object_name: str,
        use_multiview: bool = True,
    ) -> CapabilityResult[dict[str, object]]:
        full_points, object_points = self._object_clouds(
            object_name,
            use_multiview,
        )
        centroid, bounds_min, bounds_max = _point_geometry(object_points)
        handle = self.registry.put(
            "entity_track",
            _TrackedEntity(
                object_name=object_name,
                centroid=centroid,
                bounds_min=bounds_min,
                bounds_max=bounds_max,
                point_count=int(len(object_points)),
                full_points=np.asarray(full_points, dtype=np.float64).copy(),
                object_points=np.asarray(
                    object_points,
                    dtype=np.float64,
                ).copy(),
            ),
        )
        return CapabilityResult.ok(
            {
                "tracking_handle": handle,
                "object_name": object_name,
                "pre_centroid_m": [float(value) for value in centroid],
                "pre_bounds_min_m": [float(value) for value in bounds_min],
                "pre_bounds_max_m": [float(value) for value in bounds_max],
                "pre_point_count": int(len(object_points)),
            }
        )

    def _lift_evidence_for(
        self,
        tracking_handle: str,
    ) -> _LiftProbeEvidence | None:
        evidence = self._latest_lift_evidence
        if (
            evidence is None
            or evidence.tracking_handle != tracking_handle
            or evidence.attachment_attempt_id != self._attachment_attempt_id
        ):
            return None
        return evidence

    def _lift_evidence_payload(
        self,
        tracking_handle: str,
    ) -> dict[str, object]:
        evidence = self._lift_evidence_for(tracking_handle)
        if evidence is None:
            return {
                "attachment_evidence_basis": "gripper_register_only",
                "lift_evidence_available": False,
                "lift_requested_height_m": None,
                "lift_actual_vertical_displacement_m": None,
                "lift_position_residual_m": None,
                "lift_object_detected": None,
                "lift_supports_attachment": None,
                "lift_contradicts_attachment": None,
            }
        return {
            "attachment_evidence_basis": (
                "lift_probe_and_gripper_register"
            ),
            "lift_evidence_available": True,
            "lift_requested_height_m": evidence.requested_height_m,
            "lift_actual_vertical_displacement_m": (
                evidence.actual_vertical_displacement_m
            ),
            "lift_position_residual_m": evidence.position_residual_m,
            "lift_object_detected": evidence.object_detected,
            "lift_supports_attachment": evidence.supports_attachment,
            "lift_contradicts_attachment": (
                evidence.contradicts_attachment
            ),
            **({"lift_execution": deepcopy(evidence.execution)}
               if self._sensor_motion else {}),
        }

    def assert_entity_attached(
        self,
        tracking_handle: str,
        post_object_name: str | None = None,
        use_multiview: bool = True,
        current_point_handle: str | None = None,
    ) -> CapabilityResult[dict[str, object]]:
        tracked = cast(
            _TrackedEntity,
            self.registry.get(
                tracking_handle,
                expected_kind="entity_track",
            ),
        )
        # Any new assertion attempt invalidates previously captured planning
        # geometry, while the established legacy pose-retention rules stay intact.
        tracked.carried_geometry_epoch = getattr(tracked, 'carried_geometry_epoch', 0) + 1
        tracked.attachment_current_point_handle = None
        prompt = post_object_name or tracked.object_name
        state = self.robot.get_robot_state()
        if not state.success or state.value is None:
            raise ToolOperationError(state.failure_category, state.message)
        gripper = self._robot_gripper_state()
        if not gripper.success or not isinstance(
            gripper.value, Mapping
        ):
            raise ToolOperationError(
                gripper.failure_category,
                gripper.message
                or "gripper object-detection register is unavailable",
            )
        gripper_object_detected = gripper.value.get(
            "object_detected"
        )
        if not isinstance(gripper_object_detected, bool):
            raise ToolOperationError(
                FailureCategory.TOOL_ERROR,
                "gripper object-detection register is missing",
            )
        lift_evidence = self._lift_evidence_for(tracking_handle)
        lift_payload = self._lift_evidence_payload(tracking_handle)
        if self._sensor_motion:
            lift_payload.update(
                motion_authorized=False,
                motion_guidance=(
                    "Possession and object pose do not certify safe transport or resolve a lift contact stop. "
                    "Inspect lift_execution and current force; recover unresolved contact using the recorded "
                    "retreat where available, then reassess before moving. All motion guards still apply."
                ),
            )
        if (
            (
                self._v11_completion_evidence
                and not gripper_object_detected
            )
            or (
                not self._v11_completion_evidence
                and float(
                    gripper.value.get("open_fraction", 0.0)
                )
                < 0.02
            )
        ):
            tracked.attachment_verified = False
            tracked.eef_to_object_transform = None
            if self._active_attachment_handle == tracking_handle:
                self._active_attachment_handle = None
            eef = np.asarray(
                state.value.end_effector_position_m,
                dtype=np.float64,
            )
            return CapabilityResult.ok(
                {
                    "tracking_handle": tracking_handle,
                    "verified": False,
                    **({"target_attachment_status": "contradicted",
                        "target_attachment_basis": "closed_gripper_aperture_empty"} if self._v15_protocol else {}),
                    "reason": "closed_on_nothing",
                    "object_name": tracked.object_name,
                    "post_object_name": prompt,
                    "current_point_handle": current_point_handle,
                    "identity_preserved": False,
                    "pose_valid": False,
                    "pose_observation_accepted": False,
                    "pose_guidance": "No possession evidence; attachment pose invalidated.",
                    "pre_centroid_m": [
                        float(value) for value in tracked.centroid
                    ],
                    "post_centroid_m": [
                        float(value) for value in tracked.centroid
                    ],
                    "post_bounds_min_m": [
                        float(value) for value in tracked.bounds_min
                    ],
                    "post_bounds_max_m": [
                        float(value) for value in tracked.bounds_max
                    ],
                    "end_effector_position_m": [
                        float(value) for value in eef
                    ],
                    "displacement_m": 0.0,
                    "vertical_displacement_m": 0.0,
                    "target_to_eef_distance_m": float(
                        np.linalg.norm(tracked.centroid - eef)
                    ),
                    "min_displacement_m": max(
                        0.04,
                        min(
                            0.08,
                            float(
                                np.linalg.norm(
                                    tracked.bounds_max
                                    - tracked.bounds_min
                                )
                            )
                            * 0.25,
                        ),
                    ),
                    "min_vertical_displacement_m": 0.025,
                    "max_target_to_eef_distance_m": min(
                        0.25,
                        max(
                            0.12,
                            float(
                                np.linalg.norm(
                                    tracked.bounds_max
                                    - tracked.bounds_min
                                )
                            )
                            + 0.08,
                        ),
                    ),
                    "gripper_open_fraction": float(
                        gripper.value.get("open_fraction", 0.0)
                    ),
                    **(
                        {
                            "gripper_object_detected": False,
                            "identity_confidence": 0.0,
                        }
                        if self._v11_completion_evidence
                        else {}
                    ),
                    "post_point_count": tracked.point_count,
                    "point_count_ratio": 1.0,
                    "extent_ratio": 1.0,
                    "point_scale_consistent": True,
                    "extent_scale_consistent": True,
                    "appearance_distance_rgb": None,
                    "attachment_transform_eef_object": None,
                    **lift_payload,
                    **(
                        {"quality_flags": ["closed_on_nothing"]}
                        if self._dehardcoded
                        else {}
                    ),
                }
            )
        current_mark: _MarkedPoint | None = None
        if current_point_handle is not None:
            current_mark = cast(
                _MarkedPoint,
                self.registry.get(
                    current_point_handle,
                    expected_kind="marked_point",
                ),
            )
            if self._sensor_motion:
                captured = self.registry.get(current_mark.observation_handle, expected_kind='observation')
                now = getattr(self.robot, 'time_s', None)
                then = captured.time_s
                fresh = (now is not None and then is not None
                    and math.isfinite(now) and math.isfinite(then) and abs(now-then) <= 1e-9
                    and np.allclose(captured.robot_state.joint_positions_rad,
                                    state.value.joint_positions_rad, rtol=0, atol=1e-6)
                    and abs(captured.robot_state.gripper_open_fraction-state.value.gripper_open_fraction) <= 1e-6)
                if not fresh:
                    return CapabilityResult.failed(FailureCategory.NOT_EXECUTED,
                        'stale_attachment_observation', value={'reason': 'stale_attachment_observation',
                        'source_time_s': then, 'current_time_s': now,
                        'guidance': 'Observe and mark the target at the current robot state before rechecking attachment.'})
            post_centroid = current_mark.centroid.copy()
            post_bounds_min = current_mark.bounds_min.copy()
            post_bounds_max = current_mark.bounds_max.copy()
            post_point_count = current_mark.point_count
        else:
            _, object_points = self._object_clouds(prompt, use_multiview)
            post_centroid, post_bounds_min, post_bounds_max = (
                _point_geometry(object_points)
            )
            post_point_count = int(len(object_points))
        eef = np.asarray(
            state.value.end_effector_position_m,
            dtype=np.float64,
        )
        displacement = float(np.linalg.norm(post_centroid - tracked.centroid))
        vertical_displacement = float(post_centroid[2] - tracked.centroid[2])
        target_to_eef = float(np.linalg.norm(post_centroid - eef))
        extent = float(np.linalg.norm(tracked.bounds_max - tracked.bounds_min))
        min_displacement = max(0.04, min(0.08, extent * 0.25))
        min_vertical_displacement = 0.025
        max_target_to_eef = min(0.25, max(0.12, extent + 0.08))
        point_count_ratio = float(
            post_point_count / max(1, tracked.point_count)
        )
        pre_extent = np.maximum(
            tracked.bounds_max - tracked.bounds_min,
            1e-6,
        )
        post_extent = np.maximum(
            post_bounds_max - post_bounds_min,
            1e-6,
        )
        extent_ratio = float(
            np.linalg.norm(post_extent) / np.linalg.norm(pre_extent)
        )
        appearance_distance = None
        appearance_consistent = True
        if (
            current_mark is not None
            and tracked.appearance_rgb_mean is not None
        ):
            appearance_distance = float(
                np.linalg.norm(
                    current_mark.appearance_rgb_mean
                    - tracked.appearance_rgb_mean
                )
            )
            appearance_consistent = appearance_distance <= 90.0
        point_scale_consistent = (
            _MIN_TRACK_POINT_COUNT_RATIO
            <= point_count_ratio
            <= _MAX_TRACK_POINT_COUNT_RATIO
        )
        extent_scale_consistent = (
            _MIN_TRACK_EXTENT_RATIO
            <= extent_ratio
            <= _MAX_TRACK_EXTENT_RATIO
        )
        geometry_consistent = (
            point_scale_consistent or extent_scale_consistent
        )
        identity_preserved = (
            current_mark is not None
            and geometry_consistent
            and appearance_consistent
        )
        physically_verified = (
            displacement >= min_displacement
            and vertical_displacement >= min_vertical_displacement
            and target_to_eef <= max_target_to_eef
        )
        verified = (
            (
                gripper_object_detected
                and (
                    lift_evidence.supports_attachment
                    if lift_evidence is not None
                    else True
                )
            )
            if self._v11_completion_evidence
            else (
                physically_verified
                if self._dehardcoded
                else (
                    physically_verified
                    and (
                        current_point_handle is None
                        or identity_preserved
                    )
                )
            )
        )
        # Requested lift completion is indirect evidence. A fresh same-view
        # target measurement can independently support possession after a
        # correctly bound, inconclusive probe, without erasing its force stop.
        fresh_target_support = bool(
            self._sensor_motion and self._v15_protocol
            and lift_evidence is not None
            and not lift_evidence.supports_attachment
            and not lift_evidence.contradicts_attachment
            and gripper_object_detected and physically_verified
            and current_mark is not None and identity_preserved
            and extent_scale_consistent
            and (tracked.source_camera_name is None
                 or current_mark.camera_name == tracked.source_camera_name)
        )
        if self._v15_protocol:
            # EEF motion and aperture alone do not show that the target moved.
            verified = bool((verified or fresh_target_support)
                            and physically_verified and appearance_consistent)
        if fresh_target_support:
            lift_payload["attachment_evidence_basis"] = "fresh_target_motion_and_gripper_register"
        quality_flags: list[str] = []
        if not point_scale_consistent:
            quality_flags.append("point_count_ratio_outlier")
        if not extent_scale_consistent:
            quality_flags.append("extent_ratio_outlier")
        if not appearance_consistent:
            quality_flags.append("appearance_mismatch")
        if current_point_handle is None:
            quality_flags.append("identity_check_skipped")
        if self._v11_completion_evidence:
            if lift_evidence is None:
                quality_flags.append("lift_evidence_unavailable")
            elif lift_evidence.contradicts_attachment:
                quality_flags.append(
                    "lift_probe_contradicted_attachment"
                )
            elif not lift_evidence.supports_attachment:
                quality_flags.append("lift_probe_incomplete_target_motion_supported"
                                     if fresh_target_support else "lift_probe_not_confirmed")
        if (
            self._v11_completion_evidence
            and displacement < min_displacement
        ):
            quality_flags.append("target_displacement_low")
        if (
            self._v11_completion_evidence
            and vertical_displacement < min_vertical_displacement
        ):
            quality_flags.append("target_lift_low")
        if (
            self._v11_completion_evidence
            and target_to_eef > max_target_to_eef
        ):
            quality_flags.append("target_eef_residual_high")
        identity_components = (
            float(point_scale_consistent),
            float(extent_scale_consistent),
            float(appearance_consistent),
            float(displacement >= min_displacement),
            float(vertical_displacement >= min_vertical_displacement),
            float(target_to_eef <= max_target_to_eef),
        )
        identity_confidence = float(
            sum(identity_components) / len(identity_components)
        )
        if self._v11_completion_evidence:
            if not gripper_object_detected:
                reason = "closed_on_nothing"
            elif lift_evidence is not None and (
                lift_evidence.contradicts_attachment
            ):
                reason = "lift_probe_contradicted_attachment"
            elif lift_evidence is not None and (
                not lift_evidence.supports_attachment and not fresh_target_support
            ):
                reason = "lift_probe_not_confirmed"
            else:
                reason = "verified"
        elif displacement < min_displacement:
            reason = "target_did_not_move"
        elif vertical_displacement < min_vertical_displacement:
            reason = "target_was_not_lifted"
        elif target_to_eef > max_target_to_eef:
            reason = "target_not_near_end_effector"
        elif self._dehardcoded and quality_flags:
            reason = "verified_with_identity_warning"
        elif current_point_handle is not None and not geometry_consistent:
            reason = "tracked_geometry_inconsistent"
        elif current_point_handle is not None and not appearance_consistent:
            reason = "tracked_appearance_inconsistent"
        else:
            reason = "verified"
        if self._v15_protocol and not verified and gripper_object_detected:
            if not appearance_consistent:
                reason = "tracked_appearance_inconsistent"
            elif displacement < min_displacement:
                reason = "target_did_not_move"
            elif vertical_displacement < min_vertical_displacement:
                reason = "target_was_not_lifted"
            elif target_to_eef > max_target_to_eef:
                reason = "target_not_near_end_effector"
        partial_motion = bool(self._sensor_motion and gripper_object_detected
            and current_mark is not None and (not extent_scale_consistent or
                (tracked.source_camera_name is not None
                 and current_mark.camera_name != tracked.source_camera_name)))
        target_motion_evidence = None
        if partial_motion:
            verified = False
            reason = 'partial_geometry_motion_unresolved'
            quality_flags.append('centroids_cover_different_visible_surfaces')
            if tracked.source_detection_id is not None:
                try:
                    measured = self.track_target(tracked.source_detection_id, current_mark.observation_handle)
                    target_motion_evidence = measured.value if measured.success else {
                        'status': 'insufficient', 'reason': measured.message}
                except (ValueError, RuntimeError, ToolOperationError) as exc:
                    target_motion_evidence = {'status': 'insufficient',
                        'reason': 'source_tracking_unavailable', 'diagnostic': str(exc)}
        motion_diagnosis = None
        if self._sensor_motion:
            if reason == "target_did_not_move" and displacement > 1e-9:
                reason = "measured_motion_below_attachment_threshold"
            elif reason == "target_was_not_lifted" and vertical_displacement > 1e-9:
                reason = "measured_lift_below_attachment_threshold"
            motion_diagnosis = {
                "observation_comparability": "unresolved" if partial_motion or not appearance_consistent else "geometry_checks_passed",
                "measured_translation_m": displacement,
                "measured_vertical_translation_m": vertical_displacement,
                "translation_threshold_met": displacement >= min_displacement,
                "vertical_threshold_met": vertical_displacement >= min_vertical_displacement,
                "stationary_proven": False,
                "retention_status": "unverified",
                "scope": "Visible-surface measurements and verifier thresholds; no calibrated stationarity test or sustained retention proof."
            }
        attachment_gaps = None
        if self._sensor_motion:
            from .attachment_diagnostics import attachment_evidence_gaps
            attachment_gaps = attachment_evidence_gaps(current_mark=current_mark,
                source_camera=tracked.source_camera_name, extent_comparable=extent_scale_consistent,
                appearance_consistent=appearance_consistent, appearance_available=appearance_distance is not None,
                identity_checks=identity_preserved,
                gripper_detected=gripper_object_detected, translation_m=displacement,
                vertical_m=vertical_displacement, distance_m=target_to_eef,
                min_translation_m=min_displacement, min_vertical_m=min_vertical_displacement,
                max_distance_m=max_target_to_eef)
            if 'target_translation' in attachment_gaps['unknown_checks']:
                motion_diagnosis['observation_comparability'] = 'unresolved'
        # Possession is evidence about the gripper, not proof that a fresh
        # segmentation belongs to the held target. Never poison the transform.
        pose_observation_valid = bool(
            verified and physically_verified and appearance_consistent
            and (geometry_consistent or current_point_handle is not None)
        )
        if not pose_observation_valid:
            quality_flags.append("attachment_pose_observation_rejected")
        if not verified and not (self._v15_protocol and gripper_object_detected):
            tracked.attachment_verified = False
            tracked.eef_to_object_transform = None
        if pose_observation_valid:
            world_from_eef = _transform_from_pose(
                eef,
                state.value.end_effector_quaternion_wxyz,
            )
            world_from_object = np.eye(4, dtype=np.float64)
            world_from_object[:3, 3] = post_centroid
            tracked.eef_to_object_transform = (
                np.linalg.inv(world_from_eef) @ world_from_object
            )
            tracked.current_centroid = post_centroid.copy()
            tracked.current_bounds_min = post_bounds_min.copy()
            tracked.current_bounds_max = post_bounds_max.copy()
            tracked.current_point_count = post_point_count
            tracked.attachment_verified = True
            tracked.attachment_current_point_handle = current_point_handle
            self._active_attachment_handle = tracking_handle
        return CapabilityResult.ok(
            {
                "tracking_handle": tracking_handle,
                "verified": verified,
                **({"target_attachment_status": "supported" if verified else "insufficient",
                    "target_attachment_basis": "public_target_motion_plus_aperture; partial_geometry_can_be_uncertain"}
                   if self._v15_protocol else {}),
                "reason": reason,
                **({"motion_diagnosis": motion_diagnosis} if motion_diagnosis is not None else {}),
                **({"attachment_evidence_gaps": attachment_gaps} if attachment_gaps is not None else {}),
                **({'target_motion_evidence': target_motion_evidence,
                    'next_evidence_camera': tracked.source_camera_name,
                    'recovery_guidance': 'Changed visible coverage makes centroid differences unreliable as target motion. '
                        'Inspect the supplied before/after RGB and corresponding-point evidence. Obtain a clearer '
                        'current target mark in the original camera and recheck before deciding on release or transport. '
                        'Supplementary feature motion does not by itself authorize attachment or define an object pose.'}
                   if partial_motion else {}),
                "object_name": tracked.object_name,
                "post_object_name": prompt,
                "current_point_handle": current_point_handle,
                "identity_preserved": identity_preserved,
                "pose_valid": bool(tracked.attachment_verified and tracked.eef_to_object_transform is not None),
                "pose_observation_accepted": pose_observation_valid,
                "pose_guidance": (
                    "Use the retained attachment transform; current segmentation did not replace it."
                    if tracked.eef_to_object_transform is not None and not pose_observation_valid
                    else "" if pose_observation_valid else
                    "Pose unknown. Observe and mark the same held target, then assert again before pose-dependent placement."
                ),
                "pre_centroid_m": [
                    float(value) for value in tracked.centroid
                ],
                "post_centroid_m": [
                    float(value) for value in post_centroid
                ],
                "post_bounds_min_m": [
                    float(value) for value in post_bounds_min
                ],
                "post_bounds_max_m": [
                    float(value) for value in post_bounds_max
                ],
                "end_effector_position_m": [float(value) for value in eef],
                "displacement_m": displacement,
                "vertical_displacement_m": vertical_displacement,
                "target_to_eef_distance_m": target_to_eef,
                "min_displacement_m": min_displacement,
                "min_vertical_displacement_m": min_vertical_displacement,
                "max_target_to_eef_distance_m": max_target_to_eef,
                "gripper_open_fraction": float(
                    state.value.gripper_open_fraction
                ),
                **(
                    {
                        "gripper_object_detected": (
                            gripper_object_detected
                        ),
                        "identity_confidence": identity_confidence,
                    }
                    if self._v11_completion_evidence
                    else {}
                ),
                "post_point_count": post_point_count,
                "point_count_ratio": point_count_ratio,
                "extent_ratio": extent_ratio,
                "point_scale_consistent": point_scale_consistent,
                "extent_scale_consistent": extent_scale_consistent,
                "appearance_distance_rgb": appearance_distance,
                **lift_payload,
                **(
                    {"quality_flags": quality_flags}
                    if self._dehardcoded
                    else {}
                ),
                "attachment_transform_eef_object": (
                    [
                        float(value)
                        for value in tracked.eef_to_object_transform.ravel()
                    ]
                    if tracked.eef_to_object_transform is not None
                    else None
                ),
            }
        )

    def estimate_placement_relation(
        self,
        tracking_handle: str,
        target_point_handle: str,
        destination_point_handle: str,
        relation_type: str = "on",
    ) -> CapabilityResult[dict[str, object]]:
        relation = _validate_relation_type(relation_type)
        tracked = cast(
            _TrackedEntity,
            self.registry.get(
                tracking_handle,
                expected_kind="entity_track",
            ),
        )
        if (
            not tracked.attachment_verified
            or tracked.eef_to_object_transform is None
        ):
            return CapabilityResult.failed(
                FailureCategory.SAFETY_REJECTED,
                "attachment_unverified: placement relation requires an "
                "identity-preserving public attachment verification",
            )
        target_mark = cast(
            _MarkedPoint,
            self.registry.get(
                target_point_handle,
                expected_kind="marked_point",
            ),
        )
        destination_mark = cast(
            _MarkedPoint,
            self.registry.get(
                destination_point_handle,
                expected_kind="marked_point",
            ),
        )
        if (
            target_mark.observation_handle
            != destination_mark.observation_handle
        ):
            raise ValueError(
                "target and destination marks must come from the same fresh observation"
            )
        captured = cast(
            _CapturedObservation,
            self.registry.get(
                target_mark.observation_handle,
                expected_kind="observation",
            ),
        )
        state = captured.robot_state
        world_from_eef = _transform_from_pose(
            state.end_effector_position_m,
            state.end_effector_quaternion_wxyz,
        )
        estimated_target = (
            world_from_eef @ tracked.eef_to_object_transform
        )[:3, 3]
        target_extent = np.maximum(
            tracked.bounds_max - tracked.bounds_min,
            1e-6,
        )
        marked_target_extent = np.maximum(
            target_mark.bounds_max - target_mark.bounds_min,
            1e-6,
        )
        destination_centroid = destination_mark.centroid.copy()
        destination_min = destination_mark.bounds_min.copy()
        destination_max = destination_mark.bounds_max.copy()
        target_visual_residual = float(
            np.linalg.norm(target_mark.position - estimated_target)
        )
        max_visual_residual = max(
            _MAX_ATTACHMENT_VISUAL_RESIDUAL_M,
            float(np.linalg.norm(target_extent)) * 0.75,
        )
        target_extent_ratio = float(
            np.linalg.norm(marked_target_extent)
            / np.linalg.norm(target_extent)
        )
        point_count_ratio = float(
            target_mark.point_count / max(1, tracked.point_count)
        )
        appearance_distance = (
            float(
                np.linalg.norm(
                    target_mark.appearance_rgb_mean
                    - tracked.appearance_rgb_mean
                )
            )
            if tracked.appearance_rgb_mean is not None
            else 0.0
        )
        attachment_aware_mark = (
            target_mark.grounding_status == "grounded_attachment"
        )
        residual_ok = (
            target_visual_residual <= max_visual_residual
        )
        appearance_ok = appearance_distance <= 110.0
        scale_ok = (
            attachment_aware_mark
            or (
                _MIN_TRACK_EXTENT_RATIO
                <= target_extent_ratio
                <= _MAX_TRACK_EXTENT_RATIO
            )
        )
        identity_consistent = residual_ok and appearance_ok and scale_ok
        destination_position = destination_mark.position.copy()
        target_min = estimated_target - target_extent * 0.5
        target_max = estimated_target + target_extent * 0.5
        target_area = max(
            float(target_extent[0] * target_extent[1]),
            1e-9,
        )
        overlap_x = max(
            0.0,
            float(
                min(target_max[0], destination_max[0])
                - max(target_min[0], destination_min[0])
            ),
        )
        overlap_y = max(
            0.0,
            float(
                min(target_max[1], destination_max[1])
                - max(target_min[1], destination_min[1])
            ),
        )
        footprint_overlap_ratio = overlap_x * overlap_y / target_area
        horizontal_delta = destination_position[:2] - estimated_target[:2]
        horizontal_distance = float(np.linalg.norm(horizontal_delta))
        correction = np.zeros(3, dtype=np.float64)
        geometry_diagnostics: dict[str, object] = {}
        placement_quality_flags: list[str] = []
        if self._v15_protocol:
            geometry_diagnostics = {
                "measurement_confidence_basis": "target identity/visual consistency heuristic only; not a placement probability",
                "confidence_calibrated": False,
                "placement_geometry_confidence": None,
                "target_camera_name": target_mark.camera_name,
                "destination_camera_name": destination_mark.camera_name,
                "correction_stable_basis": "absence of a large reversed correction; not evidence of stable destination geometry",
                "correction_basis": "visible support mark and bounds; advisory partial geometry",
                "correction_geometry_feasible": True,
            }
        if relation == "on":
            release_height = float(target_min[2] - destination_max[2])
            support_delta = minimum_support_translation(
                estimated_target, target_extent, destination_min, destination_max,
            )
            if self._v15_protocol:
                geometry_diagnostics.update({
                    "correction_basis": "minimum translation for 50% visible footprint overlap; partial bounds do not prove physical support",
                    "correction_geometry_feasible": support_delta is not None,
                })
            if support_delta is None:
                correction_stage = "relocalize_required"
                placement_quality_flags.append("visible_destination_extent_insufficient")
            elif footprint_overlap_ratio < 0.50:
                correction[:2] = support_delta
                correction_stage = "align_xy"
            elif abs(release_height - 0.01) > 0.015:
                correction[2] = 0.01 - release_height
                correction_stage = "approach_support"
            else:
                correction_stage = "hold_for_release"
            relation_estimate = bool(
                footprint_overlap_ratio >= 0.50
                and -0.03 <= release_height <= 0.04
            )
        else:
            release_height = float(
                estimated_target[2] - destination_max[2]
            )
            inside_xy = bool(
                np.all(target_min[:2] >= destination_min[:2])
                and np.all(target_max[:2] <= destination_max[:2])
            )
            center_min = destination_min[:2] + target_extent[:2] * 0.5
            center_max = destination_max[:2] - target_extent[:2] * 0.5
            visible_footprint_fits = bool(np.all(center_min <= center_max))
            if self._v15_protocol:
                geometry_diagnostics.update({
                    "correction_basis": "minimum translation into visible bounds eroded by target half extent; bounds are not verified interior walls",
                    "correction_geometry_feasible": visible_footprint_fits,
                    "visible_center_interval_min_xy_m": center_min.tolist(),
                    "visible_center_interval_max_xy_m": center_max.tolist(),
                })
            if self._v15_protocol and not visible_footprint_fits:
                correction_stage = "relocalize_required"
                placement_quality_flags.append("visible_destination_extent_insufficient")
            elif not inside_xy:
                correction[:2] = (
                    np.clip(estimated_target[:2], center_min, center_max)
                    - estimated_target[:2]
                    if self._v15_protocol else horizontal_delta
                )
                correction_stage = "align_xy"
            elif release_height > -_CONTAINER_RIM_CLEARANCE_M:
                correction[2] = (
                    destination_max[2]
                    - _CONTAINER_RIM_CLEARANCE_M
                    - estimated_target[2]
                )
                correction_stage = "approach_container"
            else:
                correction_stage = "hold_for_release"
            relation_estimate = bool(
                inside_xy
                and release_height <= -_CONTAINER_RIM_CLEARANCE_M
            )
        if self._v15_protocol:
            geometry_diagnostics["correction_resolution"] = correction_resolution(correction)
        correction_stable = True
        if (
            tracked.last_correction_m is not None
            and float(np.linalg.norm(tracked.last_correction_m)) > 0.01
            and float(np.linalg.norm(correction)) > 0.01
            and float(np.dot(tracked.last_correction_m, correction)) < 0.0
        ):
            tracked.correction_flip_count += 1
            correction_stable = False
        else:
            tracked.correction_flip_count = 0
        tracked.last_correction_m = correction.copy()
        motion_allowed = True
        safe_release_candidate = bool(
            math.isfinite(target_visual_residual)
            and math.isfinite(horizontal_distance)
            and math.isfinite(release_height)
        )
        estimate = _PlacementEstimate(
            tracking_handle=tracking_handle,
            source_observation_handle=target_mark.observation_handle,
            target_point_handle=target_point_handle,
            destination_point_handle=destination_point_handle,
            relation_type=relation,
            estimated_target_centroid=estimated_target.copy(),
            destination_position=destination_position.copy(),
            destination_bounds_min=destination_min.copy(),
            destination_bounds_max=destination_max.copy(),
            target_visual_residual_m=target_visual_residual,
            horizontal_distance_m=horizontal_distance,
            release_height_m=release_height,
            correction_m=correction.copy(),
            correction_stable=correction_stable,
            relation_estimate=relation_estimate,
            correction_stage=correction_stage,
            max_safe_step_m=None,
            measurement_confidence=(
                1.0 if identity_consistent else 0.0
            ),
            quality_flags=tuple(placement_quality_flags),
            geometry_diagnostics=geometry_diagnostics,
        )
        estimate_handle = self.registry.put(
            "placement_estimate",
            estimate,
        )
        eef = np.asarray(
            state.end_effector_position_m,
            dtype=np.float64,
        )
        return CapabilityResult.ok(
            {
                "placement_estimate_handle": estimate_handle,
                "tracking_handle": tracking_handle,
                "source_observation_handle": target_mark.observation_handle,
                "observation_handle": target_mark.observation_handle,
                "target_point_handle": target_point_handle,
                "destination_point_handle": destination_point_handle,
                "relation_type": relation,
                "relation_estimate": relation_estimate,
                "identity_consistent": identity_consistent,
                "residual_ok": residual_ok,
                "appearance_ok": appearance_ok,
                "scale_ok": scale_ok,
                "scale_check_basis": (
                    "attachment_prediction"
                    if attachment_aware_mark
                    else "metric_extent"
                ),
                "correction_stable": correction_stable,
                "motion_allowed": motion_allowed,
                "safe_release_candidate": safe_release_candidate,
                "correction_stage": correction_stage,
                "estimated_target_centroid_m": [
                    float(value) for value in estimated_target
                ],
                "destination_mark_position_m": [
                    float(value) for value in destination_position
                ],
                "destination_centroid_m": [
                    float(value) for value in destination_centroid
                ],
                "destination_bounds_min_m": [
                    float(value) for value in destination_min
                ],
                "destination_bounds_max_m": [
                    float(value) for value in destination_max
                ],
                "target_visual_residual_m": target_visual_residual,
                "max_target_visual_residual_m": max_visual_residual,
                "target_extent_ratio": target_extent_ratio,
                "target_point_count_ratio": point_count_ratio,
                "appearance_distance_rgb": appearance_distance,
                "horizontal_distance_m": horizontal_distance,
                "release_height_m": release_height,
                "footprint_overlap_ratio": footprint_overlap_ratio,
                "recommended_eef_delta_m": [
                    float(value) for value in correction
                ],
                "recommended_eef_position_m": [
                    float(value) for value in eef + correction
                ],
                **geometry_diagnostics,
                **({"quality_flags": placement_quality_flags} if self._v15_protocol else {}),
            }
        )

    def placement_estimate(
        self,
        tracking_handle: str,
        target_detection_id: str | None = None,
        destination_detection_id: str | None = None,
        relation_type: str = "on",
        selected_grasp: Mapping[str, object] | None = None,
        placement_candidate_index: int = 0,
        reference_tolerance_m: float | None = None,
    ) -> CapabilityResult[dict[str, object]]:
        if not isinstance(destination_detection_id, str):
            raise ValueError('destination_detection_id is required')
        if self._dehardcoded:
            return self._dehardcoded_placement_estimate(
                tracking_handle=tracking_handle,
                target_detection_id=target_detection_id,
                destination_detection_id=destination_detection_id,
                relation_type=relation_type,
                selected_grasp=selected_grasp,
                placement_candidate_index=placement_candidate_index,
                reference_tolerance_m=reference_tolerance_m,
            )
        relation = _validate_relation_type(relation_type)
        tracked = cast(
            _TrackedEntity,
            self.registry.get(
                tracking_handle,
                expected_kind="entity_track",
            ),
        )
        if (
            not tracked.attachment_verified
            or tracked.eef_to_object_transform is None
        ):
            return CapabilityResult.failed(
                FailureCategory.SAFETY_REJECTED,
                "attachment_unverified: placement estimation requires "
                "a verified held target",
            )
        target_detection = self._require_selected_detection(
            target_detection_id
        )
        destination_detection = self._require_selected_detection(
            destination_detection_id
        )
        if (
            target_detection.observation_handle
            != destination_detection.observation_handle
        ):
            raise ValueError(
                "target and destination detections must come from the same "
                "fresh observation"
            )
        distinct = (
            target_detection_id != destination_detection_id
            and float(
                np.linalg.norm(
                    target_detection.centroid
                    - destination_detection.centroid
                )
            )
            > 0.015
        )
        state_result = self.robot.get_robot_state()
        if not state_result.success or state_result.value is None:
            return CapabilityResult.failed(
                state_result.failure_category,
                state_result.message,
                executed=state_result.executed,
            )
        state = state_result.value
        world_from_eef = _transform_from_pose(
            state.end_effector_position_m,
            state.end_effector_quaternion_wxyz,
        )
        estimated_target = (
            world_from_eef @ tracked.eef_to_object_transform
        )[:3, 3]
        target_extent = np.maximum(
            tracked.bounds_max - tracked.bounds_min,
            1e-6,
        )
        visual_residual = float(
            np.linalg.norm(
                target_detection.centroid - estimated_target
            )
        )
        appearance_distance = (
            float(
                np.linalg.norm(
                    target_detection.appearance_rgb_mean
                    - tracked.appearance_rgb_mean
                )
            )
            if tracked.appearance_rgb_mean is not None
            else 0.0
        )
        measured_extent = np.maximum(
            target_detection.bounds_max
            - target_detection.bounds_min,
            1e-6,
        )
        extent_ratio = float(
            np.linalg.norm(measured_extent)
            / np.linalg.norm(target_extent)
        )
        identity_consistent = bool(
            distinct
            and visual_residual
            <= max(
                _MAX_ATTACHMENT_VISUAL_RESIDUAL_M,
                float(np.linalg.norm(target_extent)) * 0.75,
            )
            and appearance_distance <= 110.0
            and _MIN_TRACK_EXTENT_RATIO
            <= extent_ratio
            <= _MAX_TRACK_EXTENT_RATIO
        )
        destination = destination_detection.centroid.copy()
        desired_target = destination.copy()
        if relation == "on":
            desired_target[2] = (
                destination_detection.bounds_max[2]
                + target_extent[2] * 0.5
                + 0.01
            )
        else:
            desired_target[2] = (
                destination_detection.bounds_max[2]
                - max(
                    _CONTAINER_RIM_CLEARANCE_M,
                    target_extent[2] * 0.25,
                )
            )
        error = desired_target - estimated_target
        distance = float(np.linalg.norm(error))
        target_min = estimated_target - target_extent * 0.5
        target_max = estimated_target + target_extent * 0.5
        destination_min = destination_detection.bounds_min
        destination_max = destination_detection.bounds_max
        horizontal_distance = float(
            np.linalg.norm(
                estimated_target[:2] - destination[:2]
            )
        )
        release_height = (
            float(target_min[2] - destination_max[2])
            if relation == "on"
            else float(estimated_target[2] - destination_max[2])
        )
        if relation == "on":
            target_area = max(
                float(target_extent[0] * target_extent[1]),
                1e-9,
            )
            overlap_x = max(
                0.0,
                float(
                    min(target_max[0], destination_max[0])
                    - max(target_min[0], destination_min[0])
                ),
            )
            overlap_y = max(
                0.0,
                float(
                    min(target_max[1], destination_max[1])
                    - max(target_min[1], destination_min[1])
                ),
            )
            footprint_overlap = overlap_x * overlap_y / target_area
            relation_estimate = bool(
                identity_consistent
                and horizontal_distance <= 0.04
                and footprint_overlap >= 0.50
                and -0.03 <= release_height <= 0.04
            )
        else:
            footprint_overlap = 1.0
            inside_xy = bool(
                np.all(target_min[:2] >= destination_min[:2])
                and np.all(target_max[:2] <= destination_max[:2])
            )
            relation_estimate = bool(
                identity_consistent
                and inside_xy
                and release_height <= -_CONTAINER_RIM_CLEARANCE_M
            )
        correction_stage = (
            "hold_for_release"
            if relation_estimate
            else (
                "align_xy"
                if horizontal_distance > 0.025
                else (
                    "approach_support"
                    if relation == "on"
                    else "approach_container"
                )
            )
        )
        flip_count = tracked.correction_flip_count
        if (
            tracked.last_correction_m is not None
            and float(np.linalg.norm(tracked.last_correction_m)) > 0.01
            and distance > 0.01
            and float(np.dot(tracked.last_correction_m, error)) < 0.0
        ):
            flip_count += 1
        else:
            flip_count = 0
        tracked.correction_flip_count = flip_count
        tracked.last_correction_m = error.copy()
        max_safe_step = self._placement_safety_envelope(
            tracked.full_points,
            estimated_target,
            error,
            destination_min,
            destination_max,
        )
        suggested = np.zeros(3, dtype=np.float64)
        if distance > 1e-9:
            suggested = error / distance * min(distance, max_safe_step)
        confidence = float(
            np.clip(
                1.0
                - visual_residual
                / max(
                    0.10,
                    float(np.linalg.norm(target_extent)),
                )
                - appearance_distance / 510.0,
                0.0,
                1.0,
            )
        )
        estimate = _PlacementEstimate(
            tracking_handle=tracking_handle,
            source_observation_handle=(
                target_detection.observation_handle
            ),
            target_point_handle=target_detection_id,
            destination_point_handle=destination_detection_id,
            relation_type=relation,
            estimated_target_centroid=estimated_target.copy(),
            destination_position=destination.copy(),
            destination_bounds_min=destination_min.copy(),
            destination_bounds_max=destination_max.copy(),
            target_visual_residual_m=visual_residual,
            horizontal_distance_m=horizontal_distance,
            release_height_m=release_height,
            correction_m=error.copy(),
            correction_stable=True,
            relation_estimate=relation_estimate,
            correction_stage=correction_stage,
            max_safe_step_m=max_safe_step,
            measurement_confidence=confidence,
        )
        estimate_handle = self.registry.put(
            "placement_estimate",
            estimate,
        )
        return CapabilityResult.ok(
            {
                "placement_estimate_handle": estimate_handle,
                "tracking_handle": tracking_handle,
                "source_observation_handle": (
                    target_detection.observation_handle
                ),
                "observation_handle": target_detection.observation_handle,
                "target_detection_id": target_detection_id,
                "destination_detection_id": destination_detection_id,
                "relation_type": relation,
                "relation_estimate": relation_estimate,
                "correction_stage": correction_stage,
                "identity_consistent": identity_consistent,
                "error_vector_m": [
                    float(value) for value in error
                ],
                "distance_m": distance,
                "max_safe_step_m": max_safe_step,
                "suggested_step_m": [
                    float(value) for value in suggested
                ],
                "estimated_steps_remaining": (
                    int(math.ceil(distance / max_safe_step))
                    if distance > 0
                    else 0
                ),
                "measurement_confidence": confidence,
                "correction_flip_count": flip_count,
                "estimated_target_centroid_m": [
                    float(value) for value in estimated_target
                ],
                "destination_centroid_m": [
                    float(value) for value in destination
                ],
                "horizontal_distance_m": horizontal_distance,
                "release_height_m": release_height,
                "footprint_overlap_ratio": footprint_overlap,
            }
        )

    def _dehardcoded_placement_estimate(
        self,
        *,
        tracking_handle: str,
        target_detection_id: str | None,
        destination_detection_id: str,
        relation_type: str,
        selected_grasp: Mapping[str, object] | None,
        placement_candidate_index: int,
        reference_tolerance_m: float | None,
    ) -> CapabilityResult[dict[str, object]]:
        relation = _validate_relation_type(relation_type)
        selected_track = cast(_TrackedEntity, self.registry.get(
            tracking_handle, expected_kind="entity_track"))
        if selected_grasp is None:
            selected_grasp = selected_track.selected_grasp_descriptor
        if not isinstance(selected_grasp, Mapping):
            raise ValueError(
                "selected_grasp must be one candidate returned by "
                "grasp_pose_estimate"
            )
        selected_grasp = self._resolve_projected_grasp(selected_grasp)
        executed = selected_track.selected_grasp_descriptor
        if executed is not None and dict(selected_grasp) != dict(executed):
            raise ValueError('executed_grasp_mismatch: placement must use the descriptor actually used for closure')
        if target_detection_id is None:
            if executed is None:
                raise ValueError('executed_grasp_unavailable: target_detection_id cannot be inferred before closure')
            target_detection_id = executed.get('detection_id')
        if (
            not isinstance(placement_candidate_index, int)
            or isinstance(placement_candidate_index, bool)
            or placement_candidate_index < 0
        ):
            raise ValueError(
                "placement_candidate_index must be a non-negative integer"
            )
        if (
            not isinstance(reference_tolerance_m, (int, float))
            or isinstance(reference_tolerance_m, bool)
            or not math.isfinite(float(reference_tolerance_m))
            or float(reference_tolerance_m) <= 0
        ):
            raise ValueError(
                "reference_tolerance_m must be a finite positive number"
            )
        tolerance = float(reference_tolerance_m)
        tracked = cast(
            _TrackedEntity,
            self.registry.get(
                tracking_handle,
                expected_kind="entity_track",
            ),
        )
        target_detection = self._require_selected_detection(
            target_detection_id
        )
        requested_destination_detection_id = destination_detection_id
        destination_detection = self._require_selected_detection(
            destination_detection_id
        )
        if selected_grasp.get("tracking_handle") != tracking_handle:
            raise ValueError(
                "selected_grasp does not belong to tracking_handle"
            )
        if selected_grasp.get("detection_id") != target_detection_id:
            raise ValueError(
                "selected_grasp does not belong to target_detection_id"
            )
        attachment_transform_available = bool(
            tracked.attachment_verified
            and tracked.eef_to_object_transform is not None
        )
        if self._v15_protocol and not attachment_transform_available:
            return CapabilityResult.failed(
                FailureCategory.PERCEPTION_UNCERTAIN,
                "attachment_pose_unknown: observe and mark the held target, then assert_entity_attached before placement",
                executed=False,
                value={"reason": "attachment_pose_unknown", "backend": "anyplace"},
            )
        backend_candidate = selected_grasp.get("backend_candidate")
        public_geometry_fallback = bool(
            self._v15_protocol
            and selected_grasp.get("backend")
            in {"mask_topdown_fallback", "public_geometry_refinement"}
        )
        geometry_backend = ("container_mouth_fallback" if relation == "in"
                            else "public_support_geometry")
        anyplace_failure: str | None = (
            "selected grasp uses public geometry; no native placement binding"
            if public_geometry_fallback
            else None
        )
        alignment_recovered = False
        alignment_reason: str | None = None
        alignment_residual_m: float | None = None
        if not public_geometry_fallback and (
            target_detection.observation_handle
            != destination_detection.observation_handle
            or target_detection.camera_name
            != destination_detection.camera_name
        ):
            try:
                (
                    destination_detection_id,
                    destination_detection,
                    alignment_residual_m,
                ) = self._align_detection_to_observation_camera(
                    destination_detection,
                    observation_handle=(
                        target_detection.observation_handle
                    ),
                    camera_name=target_detection.camera_name,
                )
            except ToolOperationError as exc:
                return CapabilityResult.failed(
                    exc.category,
                    f"placement_alignment_failed: {exc}",
                    executed=True,
                    value={
                        "reason": "placement_alignment_failed",
                        "backend": "anyplace",
                        "requested_destination_detection_id": (
                            requested_destination_detection_id
                        ),
                    },
                )
            alignment_recovered = True
            alignment_reason = (
                "destination_resegmented_to_target_observation_camera"
            )
        if (
            not isinstance(backend_candidate, Mapping)
            and not public_geometry_fallback
        ):
            return CapabilityResult.failed(
                FailureCategory.PERCEPTION_UNCERTAIN,
                "selected_grasp_unavailable: AnyPlace requires the "
                "model-native OpenETA candidate",
                executed=True,
                value={
                    "reason": "selected_grasp_unavailable",
                    "backend": "anyplace",
                },
            )
        state_result = self.robot.get_robot_state()
        if not state_result.success or state_result.value is None:
            return CapabilityResult.failed(
                state_result.failure_category,
                state_result.message,
                executed=state_result.executed,
            )
        state = state_result.value
        placement_method = getattr(
            self.services,
            "predict_placements",
            None,
        )
        if not callable(placement_method) and not public_geometry_fallback:
            return CapabilityResult.failed(
                FailureCategory.TOOL_ERROR,
                "backend_unavailable: anyplace",
                executed=True,
                value={
                    "reason": "backend_unavailable",
                    "backend": "anyplace",
                },
            )
        observation = cast(
            _CapturedObservation,
            self.registry.get(
                target_detection.observation_handle,
                expected_kind="observation",
            ),
        )
        camera = self._camera_by_name(
            observation,
            target_detection.camera_name,
        )
        if camera.depth is None:
            return CapabilityResult.failed(
                FailureCategory.PERCEPTION_UNCERTAIN,
                "selected placement camera has no public depth",
                executed=True,
            )
        if not public_geometry_fallback:
            try:
                placement_batch = placement_method(
                    image=camera.image,
                    depth=camera.depth,
                    object_mask=target_detection.mask,
                    placement_region_mask=destination_detection.mask,
                    intrinsics=camera.intrinsics,
                    extrinsics=camera.extrinsics,
                    selected_grasp=backend_candidate,
                    depth_source="sensor",
                )
                placement_poses, placement_scores = _validate_grasps(
                    placement_batch.poses,
                    placement_batch.scores,
                )
                if placement_batch.grasp_frame == "graspnet":
                    placement_poses, _ = self._compile_graspnet_poses(
                        placement_poses
                    )
            except Exception as exc:
                if not (self._v15_protocol and relation == "in"):
                    return CapabilityResult.failed(
                        FailureCategory.TOOL_ERROR,
                        f"anyplace_failed: {type(exc).__name__}: {exc}",
                        executed=True,
                        value={
                            "reason": "anyplace_failed",
                            "backend": "anyplace",
                        },
                    )
                public_geometry_fallback = True
                anyplace_failure = f"{type(exc).__name__}: {exc}"
        if public_geometry_fallback:
            placement_batch = GraspServiceBatch(
                poses=np.asarray([np.eye(4)], dtype=np.float64),
                scores=np.asarray([0.0], dtype=np.float64),
                widths_m=(None,),
                backend=geometry_backend,
                public_tool_convention=True,
                grasp_frame="capx_native",
                depth_quality={
                    "source": "public_destination_geometry",
                    "anyplace_failure": anyplace_failure,
                },
            )
            placement_poses = placement_batch.poses
            placement_scores = placement_batch.scores
        if placement_candidate_index >= len(placement_poses):
            return CapabilityResult.failed(
                FailureCategory.INVALID_ARGUMENT,
                f"placement_candidate_index {placement_candidate_index} is "
                f"outside [0, {len(placement_poses) - 1}]",
                executed=True,
                value={
                    "reason": "placement_candidate_index_out_of_range",
                    "backend": "anyplace",
                    "candidate_count": len(placement_poses),
                },
            )
        placement_candidates = [
            {
                "candidate_index": index,
                "position_m": [
                    float(value) for value in pose[:3, 3]
                ],
                "quaternion_wxyz": list(
                    _matrix_to_quaternion_wxyz(pose[:3, :3])
                ),
                "model_score": float(score),
                "backend": "anyplace",
                "selected": index == placement_candidate_index,
                "backend_candidate": (
                    dict(placement_batch.native_candidates[index])
                    if index < len(
                        placement_batch.native_candidates
                    )
                    else None
                ),
            }
            for index, (pose, score) in enumerate(
                zip(placement_poses, placement_scores)
            )
        ]
        raw_selected_placement_pose = np.asarray(
            placement_poses[placement_candidate_index],
            dtype=np.float64,
        )
        current_hand_position, current_hand_quaternion = (
            self.robot.external_ik_target_from_eef_pose(
                state.end_effector_position_m,
                state.end_effector_quaternion_wxyz,
            )
        )
        current_hand_rotation = _quaternion_wxyz_to_matrix(
            current_hand_quaternion
        )
        current_public_position = (
            np.asarray(current_hand_position, dtype=np.float64)
            - current_hand_rotation @ self._tcp_offset_local_m
        )
        if public_geometry_fallback:
            raw_selected_placement_pose[:3, :3] = current_hand_rotation
            raw_selected_placement_pose[:3, 3] = current_public_position
        if attachment_transform_available:
            world_from_eef = _transform_from_pose(
                state.end_effector_position_m,
                state.end_effector_quaternion_wxyz,
            )
            estimated_target = (
                world_from_eef
                @ cast(np.ndarray, tracked.eef_to_object_transform)
            )[:3, 3]
        else:
            estimated_target = target_detection.centroid.copy()
        if self._v15_protocol and attachment_transform_available:
            # Native candidates may need their original aligned masks. The
            # executed translation uses the currently requested destination.
            destination_detection_id = requested_destination_detection_id
            destination_detection = self._require_selected_detection(destination_detection_id)
        target_extent = np.maximum(
            tracked.bounds_max - tracked.bounds_min,
            1e-6,
        )
        measured_extent = np.maximum(
            target_detection.bounds_max
            - target_detection.bounds_min,
            1e-6,
        )
        distinct = bool(
            target_detection_id != destination_detection_id
            and float(
                np.linalg.norm(
                    target_detection.centroid
                    - destination_detection.centroid
                )
            )
            > 0.015
        )
        visual_residual = float(
            np.linalg.norm(
                target_detection.centroid - estimated_target
            )
        )
        appearance_distance = (
            float(
                np.linalg.norm(
                    target_detection.appearance_rgb_mean
                    - tracked.appearance_rgb_mean
                )
            )
            if tracked.appearance_rgb_mean is not None
            else 0.0
        )
        extent_ratio = float(
            np.linalg.norm(measured_extent)
            / np.linalg.norm(target_extent)
        )
        residual_limit = max(
            _MAX_ATTACHMENT_VISUAL_RESIDUAL_M,
            float(np.linalg.norm(target_extent)) * 0.75,
        )
        residual_ok = visual_residual <= residual_limit
        appearance_ok = appearance_distance <= 110.0
        extent_ok = (
            _MIN_TRACK_EXTENT_RATIO
            <= extent_ratio
            <= _MAX_TRACK_EXTENT_RATIO
        )
        identity_consistent = bool(
            distinct and residual_ok and appearance_ok and extent_ok
        )
        quality_flags: list[str] = []
        if not attachment_transform_available:
            quality_flags.append("attachment_transform_unavailable")
        if not tracked.attachment_verified:
            quality_flags.append("attachment_unverified")
        if not distinct:
            quality_flags.append("target_destination_not_distinct")
        if not residual_ok:
            quality_flags.append("prediction_residual_large")
        if not appearance_ok:
            quality_flags.append("appearance_mismatch")
        if not extent_ok:
            quality_flags.append("extent_ratio_outlier")
        if (
            tracked.selected_grasp_candidate is not None
            and dict(tracked.selected_grasp_candidate)
            != dict(backend_candidate)
        ):
            quality_flags.append("selected_grasp_differs_from_executed")
        destination = destination_detection.centroid.copy()
        target_min = estimated_target - target_extent * 0.5
        target_max = estimated_target + target_extent * 0.5
        destination_min = destination_detection.bounds_min
        destination_max = destination_detection.bounds_max
        selected_placement_pose = raw_selected_placement_pose.copy()
        raw_error = (
            raw_selected_placement_pose[:3, 3]
            - current_public_position
        )
        placement_pose_basis = "anyplace_raw"
        if self._v15_protocol and relation == "in":
            quality_flags.append("visible_outer_bounds_not_verified_container_opening")
            if public_geometry_fallback:
                quality_flags.append("container_mouth_pose_is_hover_only")
        if public_geometry_fallback and relation == "on":
            quality_flags.append("visible_support_geometry_is_advisory")
        geometry_refinement = np.zeros(3, dtype=np.float64)
        raw_to_selected_orientation_delta = 0.0
        if attachment_transform_available:
            raw_rotation = raw_selected_placement_pose[:3, :3]
            raw_quaternion = _matrix_to_quaternion_wxyz(raw_rotation)
            desired_target = estimated_target.copy()
            if relation == "on":
                desired_target[:2] = destination[:2]
                desired_target[2] = (
                    destination_max[2]
                    + target_extent[2] * 0.5
                    + 0.01
                )
            else:
                desired_target[:2] = (
                    destination_min[:2] + destination_max[:2]
                ) * 0.5
                desired_target[2] = (
                    destination_max[2]
                    + max(0.04, target_extent[2] * 0.5 + 0.02)
                    if public_geometry_fallback
                    else max(
                        destination_min[2] + target_extent[2] * 0.5,
                        destination_max[2] - _CONTAINER_RIM_CLEARANCE_M,
                    )
                )
            selected_placement_pose[:3, :3] = current_hand_rotation
            selected_placement_pose[:3, 3] = (
                current_public_position
                + desired_target
                - estimated_target
            )
            geometry_refinement = (
                selected_placement_pose[:3, 3]
                - raw_selected_placement_pose[:3, 3]
            )
            raw_to_selected_orientation_delta = (
                _quaternion_angular_distance(
                    raw_quaternion,
                    _normalized_quaternion(current_hand_quaternion),
                )
            )
            selected_controller_target = (
                selected_placement_pose[:3, 3]
                + current_hand_rotation @ self._tcp_offset_local_m
            )
            selected_eef_position, selected_eef_quaternion = (
                self.robot.eef_pose_from_external_ik_target(
                    selected_controller_target,
                    current_hand_quaternion,
                )
            )
            predicted_target = (
                _transform_from_pose(
                    selected_eef_position,
                    selected_eef_quaternion,
                )
                @ cast(np.ndarray, tracked.eef_to_object_transform)
            )[:3, 3]
            placement_pose_basis = (
                geometry_backend
                if public_geometry_fallback
                else "attachment_translation_current_orientation"
            )
            if public_geometry_fallback:
                raw_selected_placement_pose = (
                    selected_placement_pose.copy()
                )
                if relation == "on":
                    # There is no learned raw pose behind this geometry proposal.
                    geometry_refinement = np.zeros(3, dtype=np.float64)
                placement_candidates = [
                    {
                        "candidate_index": 0,
                        "position_m": [
                            float(value)
                            for value in selected_placement_pose[:3, 3]
                        ],
                        "quaternion_wxyz": list(
                            _matrix_to_quaternion_wxyz(
                                selected_placement_pose[:3, :3]
                            )
                        ),
                        "model_score": None if relation == "on" else 0.0,
                        "backend": geometry_backend,
                        "selected": True,
                        "backend_candidate": None,
                    }
                ]
        else:
            predicted_target = estimated_target + raw_error
        error = (
            selected_placement_pose[:3, 3] - current_public_position
        )
        distance = float(np.linalg.norm(error))
        horizontal_distance = float(np.linalg.norm(error[:2]))
        release_height = float(error[2])
        target_area = max(
            float(target_extent[0] * target_extent[1]),
            1e-9,
        )
        overlap_x = max(
            0.0,
            float(
                min(target_max[0], destination_max[0])
                - max(target_min[0], destination_min[0])
            ),
        )
        overlap_y = max(
            0.0,
            float(
                min(target_max[1], destination_max[1])
                - max(target_min[1], destination_min[1])
            ),
        )
        footprint_overlap = overlap_x * overlap_y / target_area
        predicted_target_min = predicted_target - target_extent * 0.5
        predicted_target_max = predicted_target + target_extent * 0.5
        predicted_overlap_x = max(
            0.0,
            float(
                min(predicted_target_max[0], destination_max[0])
                - max(predicted_target_min[0], destination_min[0])
            ),
        )
        predicted_overlap_y = max(
            0.0,
            float(
                min(predicted_target_max[1], destination_max[1])
                - max(predicted_target_min[1], destination_min[1])
            ),
        )
        predicted_footprint_overlap = (
            predicted_overlap_x * predicted_overlap_y / target_area
        )
        predicted_support_gap = (
            float(predicted_target_min[2] - destination_max[2])
            if relation == "on"
            else float(predicted_target[2] - destination_max[2])
        )
        relation_estimate = distance <= tolerance
        correction_stage = (
            "hold_for_release"
            if relation_estimate
            else (
                "align_xy"
                if horizontal_distance > tolerance
                else (
                    "approach_support"
                    if relation == "on"
                    else "approach_container"
                )
            )
        )
        flip_count = tracked.correction_flip_count
        if (
            tracked.last_correction_m is not None
            and float(np.linalg.norm(tracked.last_correction_m)) > 0.01
            and distance > 0.01
            and float(np.dot(tracked.last_correction_m, error)) < 0.0
        ):
            flip_count += 1
        else:
            flip_count = 0
        tracked.correction_flip_count = flip_count
        tracked.last_correction_m = error.copy()
        obstacle_distance = self._distance_to_nearest_obstacle(
            tracked.full_points,
            current_public_position,
            error,
            destination_min,
            destination_max,
        )
        confidence = float(
            np.clip(
                1.0
                - visual_residual
                / max(
                    0.10,
                    float(np.linalg.norm(target_extent)),
                )
                - appearance_distance / 510.0,
                0.0,
                1.0,
            )
        )
        estimate = _PlacementEstimate(
            tracking_handle=tracking_handle,
            source_observation_handle=(
                target_detection.observation_handle
            ),
            target_point_handle=target_detection_id,
            destination_point_handle=destination_detection_id,
            relation_type=relation,
            estimated_target_centroid=estimated_target.copy(),
            destination_position=destination.copy(),
            destination_bounds_min=destination_min.copy(),
            destination_bounds_max=destination_max.copy(),
            target_visual_residual_m=visual_residual,
            horizontal_distance_m=horizontal_distance,
            release_height_m=release_height,
            correction_m=error.copy(),
            correction_stable=True,
            relation_estimate=relation_estimate,
            correction_stage=correction_stage,
            distance_to_nearest_obstacle_m=obstacle_distance,
            measurement_confidence=confidence,
            quality_flags=tuple(quality_flags),
        )
        estimate_handle = self.registry.put(
            "placement_estimate",
            estimate,
        )
        return CapabilityResult.ok(
            {
                "placement_estimate_handle": estimate_handle,
                "tracking_handle": tracking_handle,
                "source_observation_handle": (
                    target_detection.observation_handle
                ),
                "observation_handle": target_detection.observation_handle,
                "target_detection_id": target_detection_id,
                "destination_detection_id": destination_detection_id,
                "requested_destination_detection_id": (
                    requested_destination_detection_id
                ),
                "alignment_recovered": alignment_recovered,
                "alignment_reason": alignment_reason,
                "alignment_residual_m": alignment_residual_m,
                "relation_type": relation,
                "relation_estimate": relation_estimate,
                "correction_stage": correction_stage,
                "attachment_verified": tracked.attachment_verified,
                "identity_consistent": identity_consistent,
                "quality_flags": quality_flags,
                "placement_backend": (
                    geometry_backend
                    if public_geometry_fallback
                    else "anyplace"
                ),
                **(
                    {"anyplace_failure": anyplace_failure}
                    if self._v15_protocol
                    else {}
                ),
                "selected_grasp_backend": selected_grasp.get("backend"),
                "selected_placement_candidate_index": (
                    placement_candidate_index
                ),
                "reference_tolerance_m": tolerance,
                "selected_placement_pose": {
                    "position_m": [
                        float(value)
                        for value in selected_placement_pose[:3, 3]
                    ],
                    "quaternion_wxyz": list(
                        _matrix_to_quaternion_wxyz(
                            selected_placement_pose[:3, :3]
                        )
                    ),
                },
                "raw_selected_placement_pose": {
                    "position_m": [
                        float(value)
                        for value in raw_selected_placement_pose[:3, 3]
                    ],
                    "quaternion_wxyz": list(
                        _matrix_to_quaternion_wxyz(
                            raw_selected_placement_pose[:3, :3]
                        )
                    ),
                },
                "placement_pose_basis": placement_pose_basis,
                "geometry_refinement_m": [
                    float(value) for value in geometry_refinement
                ],
                "raw_to_selected_orientation_delta_rad": (
                    raw_to_selected_orientation_delta
                ),
                "predicted_target_centroid_m": [
                    float(value) for value in predicted_target
                ],
                "predicted_support_gap_m": predicted_support_gap,
                "predicted_footprint_overlap_ratio": (
                    predicted_footprint_overlap
                ),
                "current_tool_position_m": [
                    float(value) for value in current_public_position
                ],
                "placement_candidates": placement_candidates,
                "depth_quality": dict(
                    placement_batch.depth_quality
                ),
                "error_vector_m": [
                    float(value) for value in error
                ],
                "distance_m": distance,
                "distance_to_nearest_obstacle_m": obstacle_distance,
                "suggested_step_m": [
                    float(value) for value in error
                ],
                "estimated_steps_remaining": 0 if distance == 0 else 1,
                "measurement_confidence": confidence,
                "correction_flip_count": flip_count,
                "estimated_target_centroid_m": [
                    float(value) for value in estimated_target
                ],
                "destination_centroid_m": [
                    float(value) for value in destination
                ],
                "horizontal_distance_m": horizontal_distance,
                "release_height_m": release_height,
                "footprint_overlap_ratio": footprint_overlap,
            }
        )

    @staticmethod
    def _distance_to_nearest_obstacle(
        full_points: np.ndarray,
        target: np.ndarray,
        error: np.ndarray,
        destination_min: np.ndarray,
        destination_max: np.ndarray,
    ) -> float | None:
        distance = float(np.linalg.norm(error))
        points = np.asarray(full_points, dtype=np.float64).reshape(-1, 3)
        if not len(points):
            return None
        outside_destination = np.any(
            (points < destination_min - 0.02)
            | (points > destination_max + 0.02),
            axis=1,
        )
        obstacles = points[outside_destination]
        if not len(obstacles):
            return None
        if distance <= 1e-9:
            samples = np.asarray(target, dtype=np.float64).reshape(1, 3)
        else:
            direction = error / distance
            sample_count = max(
                2,
                min(12, int(math.ceil(distance / 0.01))),
            )
            samples = target + np.linspace(
                0.0,
                min(distance, 0.10),
                sample_count,
            )[:, None] * direction
        return float(
            np.min(cKDTree(obstacles).query(samples, k=1)[0])
        )

    @staticmethod
    def _placement_safety_envelope(
        full_points: np.ndarray,
        target: np.ndarray,
        error: np.ndarray,
        destination_min: np.ndarray,
        destination_max: np.ndarray,
    ) -> float:
        distance = float(np.linalg.norm(error))
        if distance <= 1e-9:
            return 0.01
        points = np.asarray(full_points, dtype=np.float64).reshape(-1, 3)
        outside_destination = np.any(
            (points < destination_min - 0.02)
            | (points > destination_max + 0.02),
            axis=1,
        )
        obstacles = points[outside_destination]
        if not len(obstacles):
            return min(0.05, max(0.01, distance))
        direction = error / distance
        sample_count = max(2, min(12, int(math.ceil(distance / 0.01))))
        samples = target + np.linspace(
            0.0,
            min(distance, 0.10),
            sample_count,
        )[:, None] * direction
        tree = cKDTree(obstacles)
        clearance = float(np.min(tree.query(samples, k=1)[0]))
        return float(
            np.clip(
                clearance - 0.025,
                0.01,
                min(0.08, max(0.01, distance)),
            )
        )

    def authorize_safe_release(
        self,
        tracking_handle: str,
        placement_estimate_handle: str,
        target_point_handle: str | None = None,
        destination_point_handle: str | None = None,
    ) -> CapabilityResult[dict[str, object]]:
        tracked = cast(
            _TrackedEntity,
            self.registry.get(
                tracking_handle,
                expected_kind="entity_track",
            ),
        )
        estimate = cast(
            _PlacementEstimate,
            self.registry.get(
                placement_estimate_handle,
                expected_kind="placement_estimate",
            ),
        )
        if estimate.tracking_handle != tracking_handle:
            raise ValueError(
                "placement estimate does not belong to the tracked attachment"
            )
        if (target_point_handle is None) != (destination_point_handle is None):
            raise ValueError("release refresh requires both target and destination point handles")
        refresh_comparison: dict[str, object] | None = None
        if target_point_handle is not None and destination_point_handle is not None:
            previous_estimate = estimate
            refreshed = self.estimate_placement_relation(
                tracking_handle, target_point_handle, destination_point_handle,
                relation_type=estimate.relation_type,
            )
            if not refreshed.success or refreshed.value is None:
                return refreshed
            placement_estimate_handle = str(refreshed.value["placement_estimate_handle"])
            estimate = cast(_PlacementEstimate, self.registry.get(
                placement_estimate_handle, expected_kind="placement_estimate"))
            previous_camera = previous_estimate.geometry_diagnostics.get("destination_camera_name")
            current_camera = estimate.geometry_diagnostics.get("destination_camera_name")
            refresh_comparison = {
                "previous_observation_handle": previous_estimate.source_observation_handle,
                "current_observation_handle": estimate.source_observation_handle,
                "previous_destination_camera_name": previous_camera,
                "current_destination_camera_name": current_camera,
                "destination_camera_changed": (
                    previous_camera != current_camera
                    if previous_camera is not None and current_camera is not None else None
                ),
                "destination_bounds_min_change_m": float(np.linalg.norm(estimate.destination_bounds_min - previous_estimate.destination_bounds_min)),
                "destination_bounds_max_change_m": float(np.linalg.norm(estimate.destination_bounds_max - previous_estimate.destination_bounds_max)),
                "destination_bounds_min_delta_m": (estimate.destination_bounds_min - previous_estimate.destination_bounds_min).tolist(),
                "destination_bounds_max_delta_m": (estimate.destination_bounds_max - previous_estimate.destination_bounds_max).tolist(),
                "estimated_target_displacement_m": float(np.linalg.norm(estimate.estimated_target_centroid - previous_estimate.estimated_target_centroid)),
                "correction_delta_m": (estimate.correction_m - previous_estimate.correction_m).tolist(),
                "previous_correction_stage": previous_estimate.correction_stage,
                "current_correction_stage": estimate.correction_stage,
                "interpretation": "Changes in public estimates can reflect view, segmentation, or scene motion; inspect images before moving. This is not a measurement of robot drift.",
            }
        state_result = self.robot.get_robot_state()
        if not state_result.success or state_result.value is None:
            raise ToolOperationError(
                state_result.failure_category,
                state_result.message or "robot state is unavailable",
            )
        state = state_result.value
        if self._dehardcoded:
            freshness: dict[str, object] = {}
            estimate_fresh = True
            if self._v15_protocol:
                captured = cast(_CapturedObservation, self.registry.get(
                    estimate.source_observation_handle, expected_kind="observation"))
                translation = float(np.linalg.norm(
                    np.asarray(state.end_effector_position_m)
                    - np.asarray(captured.robot_state.end_effector_position_m)))
                before_rotation = _quaternion_wxyz_to_matrix(captured.robot_state.end_effector_quaternion_wxyz)
                now_rotation = _quaternion_wxyz_to_matrix(state.end_effector_quaternion_wxyz)
                rotation_delta = float(np.arccos(np.clip((np.trace(before_rotation.T @ now_rotation) - 1) / 2, -1, 1)))
                aperture_delta = abs(state.gripper_open_fraction - captured.robot_state.gripper_open_fraction)
                estimate_fresh = translation <= 0.001 and rotation_delta <= 0.01 and aperture_delta <= 0.01
                freshness = {
                    "relation_type": estimate.relation_type,
                    "relation_estimate": estimate.relation_estimate,
                    "correction_stage": estimate.correction_stage,
                    "recommended_eef_delta_m": estimate.correction_m.tolist(),
                    "correction_stable": estimate.correction_stable,
                    "measurement_confidence": estimate.measurement_confidence,
                    "measurement_confidence_basis": "target visual consistency heuristic only; not a placement probability",
                    "confidence_calibrated": False,
                    "placement_geometry_confidence": None,
                    "correction_stable_basis": "absence of a large reversed correction; not evidence of stable destination geometry",
                    **estimate.geometry_diagnostics,
                    "refresh_comparison": refresh_comparison,
                    "correction_resolution": correction_resolution(estimate.correction_m, refresh_comparison),
                    "target_visual_residual_m": estimate.target_visual_residual_m,
                    "geometry_basis": "advisory public RGB-D bounds and attachment prediction; visible destination bounds are not measured interior walls, and corrections require current image inspection",
                    "estimate_fresh": estimate_fresh,
                    "robot_translation_since_observation_m": translation,
                    "robot_rotation_since_observation_rad": rotation_delta,
                    "freshness_basis": "public robot pose and aperture versus source observation; scene motion still requires current images",
                    "refresh_applied": target_point_handle is not None,
                    "refresh_guidance": "After transport, observe and mark the held target and destination in the SAME new observation, then pass both point handles here. End the check cell before release when requires_decision=true; hold_position can settle motion in a later turn. Geometry is advisory: explain current visual evidence if overriding it.",
                }
            stable = (
                float(state.joint_velocity_norm_rad_s)
                <= _MAX_SAFE_RELEASE_JOINT_SPEED_RAD_S
            )
            attachment_confirmed = bool(
                tracked.attachment_verified
                and tracked.eef_to_object_transform is not None
            )
            target_visible = math.isfinite(
                estimate.target_visual_residual_m
            )
            destination_visible = bool(
                np.isfinite(estimate.destination_position).all()
            )
            semantic_ready = bool(
                estimate.relation_estimate
                or estimate.correction_stage == "hold_for_release"
            )
            release_recommended = stable and semantic_ready and estimate_fresh
            quality_flags = list(estimate.quality_flags) if self._v15_protocol else []
            if (self._v15_protocol
                    and freshness.get("correction_resolution", {}).get("status") == "below_observed_change"):
                quality_flags.append("correction_below_observed_geometry_change")
            if self._v15_protocol and refresh_comparison and refresh_comparison["destination_camera_changed"]:
                quality_flags.append("destination_view_changed")
            if not estimate_fresh:
                quality_flags.append("stale_placement_estimate")
            if not attachment_confirmed:
                quality_flags.append("attachment_unverified")
            if not target_visible:
                quality_flags.append("target_visibility_unknown")
            if not destination_visible:
                quality_flags.append("destination_visibility_unknown")
            if not stable:
                quality_flags.append("robot_not_stable")
            if not semantic_ready:
                quality_flags.append("semantic_relation_not_ready")
            if not estimate_fresh:
                reason = "stale_placement_estimate"
            elif not stable:
                reason = "robot_not_stable"
            elif not semantic_ready:
                reason = "semantic_relation_not_ready"
            else:
                reason = "release_recommended"
            authorization = (
                self.registry.put(
                    "safe_release_authorization",
                    {
                        "tracking_handle": tracking_handle,
                        "placement_estimate_handle": (
                            placement_estimate_handle
                        ),
                        "source_observation_handle": (
                            estimate.source_observation_handle
                        ),
                        "advisory_only": True,
                    },
                )
                if release_recommended
                else None
            )
            return CapabilityResult.ok(
                {
                    "safe_to_release": release_recommended,
                    **freshness,
                    **({"requires_decision": not release_recommended} if self._v15_protocol else {}),
                    "release_recommended": release_recommended,
                    "reason": reason,
                    "release_authorization": authorization,
                    "tracking_handle": tracking_handle,
                    "placement_estimate_handle": (
                        placement_estimate_handle
                    ),
                    "source_observation_handle": (
                        estimate.source_observation_handle
                    ),
                    "observation_handle": estimate.source_observation_handle,
                    "attachment_confirmed": attachment_confirmed,
                    "physically_safe": stable,
                    "semantic_ready": semantic_ready,
                    "target_visible": target_visible,
                    "destination_visible": destination_visible,
                    "horizontal_distance_m": (
                        estimate.horizontal_distance_m
                    ),
                    "release_height_m": estimate.release_height_m,
                    "joint_velocity_norm_rad_s": float(
                        state.joint_velocity_norm_rad_s
                    ),
                    "max_safe_joint_velocity_norm_rad_s": (
                        _MAX_SAFE_RELEASE_JOINT_SPEED_RAD_S
                    ),
                    "quality_flags": quality_flags,
                }
            )
        stable = (
            float(state.joint_velocity_norm_rad_s)
            <= _MAX_SAFE_RELEASE_JOINT_SPEED_RAD_S
        )
        attachment_confirmed = (
            tracked.attachment_verified
            and tracked.eef_to_object_transform is not None
        )
        target_visible = (
            estimate.target_visual_residual_m
            <= max(
                _MAX_ATTACHMENT_VISUAL_RESIDUAL_M,
                float(
                    np.linalg.norm(
                        tracked.bounds_max - tracked.bounds_min
                    )
                )
                * 0.75,
            )
        )
        destination_visible = bool(
            np.isfinite(estimate.destination_position).all()
        )
        physically_safe = (
            attachment_confirmed
            and target_visible
            and destination_visible
            and stable
        )
        semantic_ready = bool(
            estimate.relation_estimate
            or estimate.correction_stage == "hold_for_release"
        )
        safe_to_release = physically_safe and semantic_ready
        if not attachment_confirmed:
            reason = "attachment_unverified"
        elif not target_visible:
            reason = "target_not_visually_consistent"
        elif not destination_visible:
            reason = "destination_not_visible"
        elif not stable:
            reason = "robot_not_stable"
        elif not semantic_ready:
            reason = "semantic_relation_not_ready"
        else:
            reason = "safe_release_authorized"
        authorization = (
            self.registry.put(
                "safe_release_authorization",
                {
                    "tracking_handle": tracking_handle,
                    "placement_estimate_handle": placement_estimate_handle,
                    "source_observation_handle": (
                        estimate.source_observation_handle
                    ),
                },
            )
            if safe_to_release
            else None
        )
        return CapabilityResult.ok(
            {
                "safe_to_release": safe_to_release,
                "reason": reason,
                "release_authorization": authorization,
                "tracking_handle": tracking_handle,
                "placement_estimate_handle": placement_estimate_handle,
                "source_observation_handle": (
                    estimate.source_observation_handle
                ),
                "observation_handle": estimate.source_observation_handle,
                "attachment_confirmed": attachment_confirmed,
                "physically_safe": physically_safe,
                "semantic_ready": semantic_ready,
                "target_visible": target_visible,
                "destination_visible": destination_visible,
                "horizontal_distance_m": estimate.horizontal_distance_m,
                "release_height_m": estimate.release_height_m,
                "joint_velocity_norm_rad_s": float(
                    state.joint_velocity_norm_rad_s
                ),
                "max_safe_joint_velocity_norm_rad_s": (
                    _MAX_SAFE_RELEASE_JOINT_SPEED_RAD_S
                ),
            }
        )

    def assert_task_completion_public(
        self,
        target_point_handle: str,
        destination_point_handle: str,
        target_label: str,
        destination_label: str,
        tracking_handle: str,
        relation_type: str = "on",
    ) -> CapabilityResult[dict[str, object]]:
        result = self.assert_spatial_relation_from_marks(
            target_point_handle,
            destination_point_handle,
            target_label,
            destination_label,
            relation_type,
        )
        if not result.success or not isinstance(result.value, Mapping):
            return result
        payload = dict(result.value)
        tracked = cast(
            _TrackedEntity,
            self.registry.get(
                tracking_handle,
                expected_kind="entity_track",
            ),
        )
        target_mark = cast(
            _MarkedPoint,
            self.registry.get(
                target_point_handle,
                expected_kind="marked_point",
            ),
        )
        captured = cast(
            _CapturedObservation,
            self.registry.get(
                target_mark.observation_handle,
                expected_kind="observation",
            ),
        )
        identity_provenance = (
            self._completion_identity_provenance(
                tracked,
                target_mark=target_mark,
            )
            if self._v15_protocol
            else {
                "identity_provenance_required": False,
                "identity_provenance_confirmed": True,
            }
        )
        relation_satisfied = bool(
            payload.pop("relation_ready", False)
        )
        destination_min = np.asarray(
            payload["destination_bounds_min_m"],
            dtype=np.float64,
        )
        destination_max = np.asarray(
            payload["destination_bounds_max_m"],
            dtype=np.float64,
        )
        destination_xy_diagonal = float(
            np.linalg.norm(destination_max[:2] - destination_min[:2])
        )
        max_mark_horizontal_distance = min(
            self.completion_protocol.max_mark_distance_m,
            max(
                self.completion_protocol.min_mark_distance_m,
                destination_xy_diagonal
                * self.completion_protocol.mark_distance_destination_ratio,
            ),
        )
        # A mark identifies a visible point, not a measured surface center.
        # Actual footprint, support-height and identity checks remain below.
        mark_centering_required = False
        mark_centered_for_completion = (
            not mark_centering_required
            or float(payload["mark_horizontal_distance_m"])
            <= max_mark_horizontal_distance
        )
        tracked_extent = max(
            float(np.linalg.norm(tracked.bounds_max - tracked.bounds_min)),
            1e-6,
        )
        completion_extent = float(
            np.linalg.norm(
                np.asarray(payload["target_bounds_max_m"], dtype=np.float64)
                - np.asarray(payload["target_bounds_min_m"], dtype=np.float64)
            )
        )
        completion_extent_ratio = completion_extent / tracked_extent
        extent_identity_consistent = (
            self.completion_protocol.min_extent_ratio
            <= completion_extent_ratio
            <= self.completion_protocol.max_extent_ratio
        )
        appearance_distance = (
            float(
                np.linalg.norm(
                    target_mark.appearance_rgb_mean
                    - tracked.appearance_rgb_mean
                )
            )
            if tracked.appearance_rgb_mean is not None
            else 0.0
        )
        appearance_identity_consistent = appearance_distance <= 110.0
        released_target_identity_consistent = (
            extent_identity_consistent
            and appearance_identity_consistent
        )
        target_to_eef_distance = float(
            np.linalg.norm(
                np.asarray(payload["target_centroid_m"], dtype=np.float64)
                - np.asarray(
                    captured.robot_state.end_effector_position_m,
                    dtype=np.float64,
                )
            )
        )
        min_target_to_eef_distance = min(
            self.completion_protocol.max_eef_clearance_m,
            max(
                self.completion_protocol.min_eef_clearance_m,
                tracked_extent
                * self.completion_protocol.eef_clearance_extent_ratio,
            ),
        )
        robot_clear_for_completion = (
            target_to_eef_distance >= min_target_to_eef_distance
        )
        if self._v11_completion_evidence:
            gripper = self._robot_gripper_state()
            if not gripper.success or not isinstance(
                gripper.value, Mapping
            ):
                raise ToolOperationError(
                    gripper.failure_category,
                    gripper.message
                    or "gripper object-detection register is unavailable",
                )
            object_detected = gripper.value.get("object_detected")
            if not isinstance(object_detected, bool):
                raise ToolOperationError(
                    FailureCategory.TOOL_ERROR,
                    "gripper object-detection register is missing",
                )
            gripper_released = not object_detected
            temporal_evidence = None
            if self._v15_protocol:
                from .completion_sampling import compare_completion_samples
                # Existing release/attachment invalidation clears this marker.
                # A fresh sample must not borrow an older attachment epoch.
                if tracked.last_completion_observation_handle is None:
                    tracked.completion_samples.clear()
                tracked.completion_samples, temporal_evidence = compare_completion_samples(
                    tracked.completion_samples, observation_handle=target_mark.observation_handle,
                    centroid=payload['target_centroid_m'], time_s=captured.time_s,
                    released=gripper_released)
            static_displacement: float | None = None
            comparison_time_delta_s = (
                captured.time_s - tracked.last_completion_time_s
                if captured.time_s is not None and tracked.last_completion_time_s is not None else None
            )
            temporal_comparison_available = bool(
                comparison_time_delta_s is not None and comparison_time_delta_s > 0
                and gripper_released and tracked.last_completion_released
            )
            if (
                tracked.last_completion_observation_handle
                is not None
                and tracked.last_completion_centroid is not None
                and tracked.last_completion_observation_handle
                != target_mark.observation_handle
            ):
                static_displacement = float(
                    np.linalg.norm(
                        np.asarray(
                            payload["target_centroid_m"],
                            dtype=np.float64,
                        )
                        - tracked.last_completion_centroid
                    )
                )
            if (
                tracked.last_completion_observation_handle
                != target_mark.observation_handle
            ):
                tracked.last_completion_observation_handle = (
                    target_mark.observation_handle
                )
                tracked.last_completion_centroid = np.asarray(
                    payload["target_centroid_m"],
                    dtype=np.float64,
                ).copy()
                tracked.last_completion_time_s = captured.time_s
                tracked.last_completion_released = gripper_released
            if temporal_evidence is not None:
                comparison_time_delta_s = temporal_evidence['comparison_time_delta_s']
                temporal_comparison_available = temporal_evidence['temporal_comparison_available']
                static_displacement = temporal_evidence['target_static_after_release_m']
            quality_flags: list[str] = []
            if static_displacement is None:
                quality_flags.append(
                    "target_static_not_yet_measurable"
                )
            if not released_target_identity_consistent:
                quality_flags.append("identity_low_confidence")
            if (
                identity_provenance.get(
                    "identity_provenance_required"
                )
                is True
                and identity_provenance.get(
                    "identity_provenance_confirmed"
                )
                is not True
            ):
                quality_flags.append(
                    "instance_identity_provenance_unconfirmed"
                )
            if not robot_clear_for_completion:
                quality_flags.append("eef_close_to_target")
            if not gripper_released:
                quality_flags.append(
                    "gripper_object_detected_after_release"
                )
            support_gap = float(payload["support_vertical_gap_m"])
            robust_gap = float(
                payload["robust_support_vertical_gap_m"]
            )
            target_intersects_destination = (
                relation_type == "on" and robust_gap < 0.0
            )
            if (
                self.completion_protocol.confidence_repair
                and target_intersects_destination
            ):
                quality_flags.append("target_intersects_destination")
            if (
                not -0.04 <= support_gap <= 0.03
                and -0.08 <= robust_gap <= 0.08
            ):
                quality_flags.extend(
                    (
                        "target_underside_occluded",
                        "destination_partially_covered",
                    )
                )
            min_mark_vertical = (
                self.completion_protocol.candidate_profile_number(
                    "min_mark_vertical_delta_m",
                    default=-0.02,
                )
            )
            max_mark_vertical = (
                self.completion_protocol.candidate_profile_number(
                    "max_mark_vertical_delta_m",
                    default=0.12,
                )
            )
            mark_vertical_plausible = (
                relation_type == "in"
                or min_mark_vertical
                <= float(payload["mark_vertical_delta_m"])
                <= max_mark_vertical
            )
            max_static_displacement = (
                self.completion_protocol.candidate_profile_number(
                    "max_static_displacement_m",
                    default=0.01,
                )
            )
            static_stable = (
                static_displacement is not None
                and static_displacement <= max_static_displacement
                and (not self._v15_protocol or temporal_comparison_available)
            )
            min_robust_vertical_gap = (
                self.completion_protocol.candidate_profile_number(
                    "min_robust_vertical_gap_m",
                    default=0.002,
                )
            )
            robust_vertical_gap_supported = (
                relation_satisfied
                if relation_type == "in"
                else robust_gap >= min_robust_vertical_gap
            )
            confidence_signals = {
                "gripper_released": float(gripper_released),
                "robust_vertical_gap_m": float(
                    robust_vertical_gap_supported
                ),
                "xy_overlap_ratio": float(
                    np.clip(
                        payload["footprint_overlap_ratio"],
                        0.0,
                        1.0,
                    )
                ),
                "centroid_inside_destination_xy": float(
                    bool(
                        payload[
                            "centroid_inside_destination_xy"
                        ]
                    )
                ),
                "mark_vertical_delta_m": float(
                    mark_vertical_plausible
                ),
                "target_destination_distinct": float(
                    bool(payload["distinct_entities"])
                ),
                "eef_clearance_m": float(
                    robot_clear_for_completion
                ),
                "target_static_after_release_m": float(static_stable),
            }
            (
                confidence,
                confidence_components,
                confidence_penalty,
            ) = self.completion_protocol.score_confidence(
                confidence_signals,
                quality_flags,
            )
            required_boolean_evidence = (
                self.completion_protocol.completion_candidate_profile.get(
                    "required_boolean_evidence",
                    (),
                )
            )
            boolean_evidence = {
                "gripper_released": gripper_released,
                "target_destination_distinct": bool(
                    payload["distinct_entities"]
                ),
                "centroid_inside_destination_xy": bool(
                    payload["centroid_inside_destination_xy"]
                ),
                "mark_inside_destination_xy": bool(
                    payload["mark_inside_destination_xy"]
                ),
            }
            min_xy_overlap = (
                self.completion_protocol.candidate_profile_number(
                    "min_xy_overlap_ratio",
                    default=0.70,
                )
            )
            recommendation_satisfied = bool(
                isinstance(required_boolean_evidence, list)
                and all(
                    boolean_evidence.get(str(name)) is True
                    for name in required_boolean_evidence
                )
                and float(payload["footprint_overlap_ratio"])
                >= min_xy_overlap
                and mark_vertical_plausible
                and static_stable
                and robot_clear_for_completion
            )
            if self.completion_protocol.confidence_repair:
                recommendation_satisfied = bool(
                    confidence
                    >= self.completion_protocol.recommendation_threshold
                    and gripper_released
                )
            if (
                identity_provenance.get(
                    "identity_provenance_required"
                )
                is True
            ):
                recommendation_satisfied = bool(
                    recommendation_satisfied
                    and identity_provenance.get(
                        "identity_provenance_confirmed"
                    )
                    is True
                )
            completion_value = {
                    "evidence": {
                        "xy_overlap_ratio": float(
                            payload["footprint_overlap_ratio"]
                        ),
                        "centroid_inside_destination_xy": bool(
                            payload[
                                "centroid_inside_destination_xy"
                            ]
                        ),
                        "mark_inside_destination_xy": bool(
                            payload["mark_inside_destination_xy"]
                        ),
                        "mark_horizontal_distance_m": float(
                            payload["mark_horizontal_distance_m"]
                        ),
                        "max_mark_horizontal_distance_m": (
                            max_mark_horizontal_distance
                        ),
                        "mark_vertical_delta_m": float(
                            payload["mark_vertical_delta_m"]
                        ),
                        "robust_vertical_gap_m": robust_gap,
                        "support_gap_bounds_m": support_gap,
                        "gripper_released": gripper_released,
                        "gripper_object_detected": object_detected,
                        "gripper_open_fraction": float(
                            gripper.value.get(
                                "open_fraction", 0.0
                            )
                        ),
                        "target_static_after_release_m": (
                            static_displacement
                        ),
                        "max_static_displacement_m": (
                            max_static_displacement
                        ),
                        "target_destination_distinct": bool(
                            payload["distinct_entities"]
                        ),
                        "eef_clearance_m": (
                            target_to_eef_distance
                        ),
                        "min_eef_clearance_m": (
                            min_target_to_eef_distance
                        ),
                        "appearance_distance_rgb": (
                            appearance_distance
                        ),
                        "extent_ratio": completion_extent_ratio,
                        "identity_confidence": float(
                            (
                                float(extent_identity_consistent)
                                + float(
                                    appearance_identity_consistent
                                )
                            )
                            / 2.0
                        ),
                        **(
                            identity_provenance
                            if self._v15_protocol
                            else {}
                        ),
                    },
                    "quality_flags": list(
                        dict.fromkeys(quality_flags)
                    ),
                    "confidence": float(
                        np.clip(confidence, 0.0, 1.0)
                    ),
                    "recommendation": (
                        "satisfied"
                        if recommendation_satisfied
                        else "insufficient"
                    ),
                    "assessment_basis": "public_rgbd_only",
                    "relation_type": relation_type,
                    "source_observation_handle": (
                        target_mark.observation_handle
                    ),
                    "tracking_handle": tracking_handle,
                    "target_point_handle": target_point_handle,
                    "destination_point_handle": (
                        destination_point_handle
                    ),
                }
            if self.completion_protocol.confidence_repair:
                completion_value.update(
                    {
                        "confidence_components": {
                            name: float(value)
                            for name, value in (
                                confidence_components.items()
                            )
                        },
                        "confidence_penalty": float(
                            np.clip(confidence_penalty, 0.0, 1.0)
                        ),
                    }
                )
            if self._v15_protocol:
                completion_value["verification_requirements"] = [{
                    "name": "post_release_temporal_comparison_available",
                    "satisfied": temporal_comparison_available and static_displacement is not None,
                    "status": "supported" if temporal_comparison_available and static_displacement is not None else "insufficient",
                    "next_observation": "Acquire released-target measurements before and after actual "
                        "environment time advances. Repeated observe() calls can read the same paused state. "
                        "Use hold_position(duration_s=0.5) between observations when needed; it consumes "
                        "one action. Keep semantic marks consistent. A large cloud shift is diagnostic, "
                        "not a demand to move the object.",
                }]
                completion_value["evidence"].update({
                    "observation_time_s": captured.time_s,
                    "comparison_time_delta_s": comparison_time_delta_s,
                    "temporal_comparison_available": temporal_comparison_available,
                    "stability_measurement_basis": "apparent_localized_centroid_shift; "
                        "different marks/masks can change it without physical motion",
                    'comparison_source_observation_handle': temporal_evidence['comparison_source_observation_handle'],
                    'comparison_observation_time_s': temporal_evidence['comparison_observation_time_s'],
                })
                completion_value["relation_evidence"] = {
                    "status": "supported" if relation_satisfied else "insufficient",
                    "scope": "partial_visible_surface_geometry_only",
                    "proxy_reason": str(payload["reason"]),
                    "target_centroid_m": payload["target_centroid_m"],
                    "target_bounds_min_m": payload["target_bounds_min_m"],
                    "target_bounds_max_m": payload["target_bounds_max_m"],
                    "destination_bounds_min_m": payload["destination_bounds_min_m"],
                    "destination_bounds_max_m": payload["destination_bounds_max_m"],
                    "limitation": "Visible surface centroids/bounds are not whole-object centers, "
                        "rim height or support contact. A failed proxy is not proof of a bad placement; "
                        "a passed proxy is not proof of completion. Keep the same semantic marks. "
                        "Inspect current raw views before deciding whether correction is needed.",
                }
                relation_question = (
                    "Does the task-specified object rest inside the container opening, "
                    "rather than balance on the rim or remain outside? Its top may protrude."
                    if relation_type == "in" else
                    "Does the task-specified object rest on and receive support from the "
                    "intended destination, rather than sit beside it or merely overlap in the image?"
                )
                completion_value["verification_questions"] = [{
                    "name": "released_and_stable",
                    "prompt": "Is the target released, independently supported, and stable across "
                        "the post-release views with advancing environment time? Inspect hand contact "
                        "and actual visible movement. Gripper object detection is adverse evidence. "
                        "A localization-centroid shift may be caused by changed marks or occlusion; "
                        "do not equate it with physical motion without inspecting images. Hand distance "
                        "is an occlusion proxy, not proof of visibility or support.",
                }, {
                    "name": "distinct_task_entities",
                    "prompt": "Identify the task-specified target and destination as distinct entities "
                        "from initial/current public views. Partial masks can merge when one covers "
                        "the other; neither merged nor separated clouds alone settle identity. "
                        "Explain how the visible target matches the task, including any distractor.",
                }, {
                    "name": "task_relation",
                    "prompt": relation_question + " Explain direct current visual evidence, reconcile "
                        "the reported geometric caveats, and retain uncertainty if contact/containment "
                        "cannot be determined. Release, stability and visibility are prerequisites "
                        "only and do not establish this relation.",
                }]
            return CapabilityResult.ok(
                completion_value,
                result.message,
            )
        if relation_satisfied and not released_target_identity_consistent:
            payload["reason"] = (
                "released_target_identity_inconsistent"
            )
        elif relation_satisfied and not robot_clear_for_completion:
            payload["reason"] = "robot_not_clear_of_released_target"
        elif relation_satisfied and not mark_centered_for_completion:
            payload["reason"] = (
                "target_mark_too_far_from_destination_center"
            )
        payload["public_relation_satisfied"] = (
            relation_satisfied
            and released_target_identity_consistent
            and robot_clear_for_completion
            and mark_centered_for_completion
        )
        payload["tracking_handle"] = tracking_handle
        payload["released_target_identity_consistent"] = (
            released_target_identity_consistent
        )
        payload["completion_target_extent_ratio"] = (
            completion_extent_ratio
        )
        payload["completion_target_appearance_distance_rgb"] = (
            appearance_distance
        )
        payload["target_to_eef_distance_m"] = target_to_eef_distance
        payload["min_target_to_eef_distance_m"] = (
            min_target_to_eef_distance
        )
        payload["robot_clear_for_completion"] = (
            robot_clear_for_completion
        )
        payload["mark_centering_required"] = mark_centering_required
        payload["mark_centered_for_completion"] = (
            mark_centered_for_completion
        )
        payload["destination_xy_diagonal_m"] = destination_xy_diagonal
        payload["max_mark_horizontal_distance_m"] = (
            max_mark_horizontal_distance
        )
        payload["assessment_basis"] = "public_rgbd_only"
        return CapabilityResult.ok(payload, result.message)

    def _completion_identity_provenance(
        self,
        tracked: _TrackedEntity,
        *,
        target_mark: _MarkedPoint,
    ) -> dict[str, object]:
        required = bool(
            tracked.source_detection_id
            and tracked.source_observation_handle
            and tracked.identity_point_handle
            and tracked.distractor_detection_id
            and tracked.distractor_initial_centroid is not None
            and tracked.distractor_initial_extent is not None
        )
        if not required:
            return {
                "identity_provenance_required": False,
                "identity_provenance_confirmed": True,
                "identity_chain_confirmed": True,
                "identity_anchor_observation_handle": (
                    tracked.source_observation_handle
                ),
                "identity_source_detection_id": (
                    tracked.source_detection_id
                ),
                "identity_point_handle": tracked.identity_point_handle,
                "distractor_detection_id": (
                    tracked.distractor_detection_id
                ),
                "distractor_stationary": None,
                "distractor_position_residual_m": None,
                "instance_reference_distances_m": None,
                "instance_selection_margin_ratio": None,
            }
        chain_confirmed = bool(
            tracked.source_detection_id
            and tracked.identity_point_handle
            and tracked.distractor_detection_id
        )
        segmented = self.segment(
            tracked.object_name,
            target_mark.observation_handle,
            target_mark.camera_name,
        )
        best_residual: float | None = None
        distractor_stationary: bool | None = None
        if segmented.success and isinstance(segmented.value, Mapping):
            candidates = segmented.value.get("detections")
            if isinstance(candidates, list):
                initial_centroid = cast(
                    np.ndarray,
                    tracked.distractor_initial_centroid,
                )
                initial_extent = np.maximum(
                    cast(np.ndarray, tracked.distractor_initial_extent),
                    1e-6,
                )
                for public in candidates:
                    detection_id = (
                        public.get("detection_id")
                        if isinstance(public, Mapping)
                        else None
                    )
                    if not isinstance(detection_id, str):
                        continue
                    current = cast(
                        _Detection,
                        self.registry.get(
                            detection_id,
                            expected_kind="detection",
                        ),
                    )
                    extent_ratio = float(
                        np.linalg.norm(
                            np.maximum(
                                current.bounds_max - current.bounds_min,
                                1e-6,
                            )
                        )
                        / np.linalg.norm(initial_extent)
                    )
                    appearance_ok = True
                    if (
                        tracked.distractor_initial_appearance_rgb_mean
                        is not None
                    ):
                        appearance_ok = bool(
                            np.linalg.norm(
                                current.appearance_rgb_mean
                                - tracked.distractor_initial_appearance_rgb_mean
                            )
                            <= 110.0
                        )
                    residual = float(
                        np.linalg.norm(
                            current.centroid - initial_centroid
                        )
                    )
                    if (
                        0.60 <= extent_ratio <= 1.65
                        and appearance_ok
                        and (
                            best_residual is None
                            or residual < best_residual
                        )
                    ):
                        best_residual = residual
                if best_residual is not None:
                    limit = max(
                        0.05,
                        float(np.linalg.norm(initial_extent)) * 0.75,
                    )
                    distractor_stationary = best_residual <= limit
        confirmed = bool(
            chain_confirmed and distractor_stationary is True
        )
        return {
            "identity_provenance_required": True,
            "identity_provenance_confirmed": confirmed,
            "identity_chain_confirmed": chain_confirmed,
            "identity_anchor_observation_handle": (
                tracked.source_observation_handle
            ),
            "identity_source_detection_id": (
                tracked.source_detection_id
            ),
            "identity_point_handle": tracked.identity_point_handle,
            "distractor_detection_id": tracked.distractor_detection_id,
            "distractor_stationary": distractor_stationary,
            "distractor_position_residual_m": best_residual,
            "instance_reference_distances_m": (
                {
                    key: round(float(value), 4)
                    for key, value in (
                        tracked.instance_reference_distances_m
                    ).items()
                }
                if tracked.instance_reference_distances_m is not None
                else None
            ),
            "instance_selection_margin_ratio": (
                tracked.instance_selection_margin_ratio
            ),
        }

    def assert_placement_ready(
        self,
        tracking_handle: str,
        current_target_name: str,
        destination_name: str,
        use_multiview: bool = True,
        relation_type: str = "on",
        target_point_handle: str | None = None,
        destination_point_handle: str | None = None,
    ) -> CapabilityResult[dict[str, object]]:
        relation = _validate_relation_type(relation_type)
        tracked = cast(
            _TrackedEntity,
            self.registry.get(
                tracking_handle,
                expected_kind="entity_track",
            ),
        )
        normalized_target = " ".join(current_target_name.split()).casefold()
        normalized_destination = " ".join(destination_name.split()).casefold()
        if not normalized_target or not normalized_destination:
            raise ValueError(
                "current_target_name and destination_name must not be empty"
            )
        point_handles = (
            target_point_handle,
            destination_point_handle,
        )
        if any(handle is not None for handle in point_handles) and not all(
            isinstance(handle, str) for handle in point_handles
        ):
            raise ValueError(
                "target_point_handle and destination_point_handle must be "
                "provided together"
            )
        if isinstance(target_point_handle, str) and isinstance(
            destination_point_handle,
            str,
        ):
            target_mark, _, target_points, _ = self._points_from_mark(
                target_point_handle,
                use_multiview=use_multiview,
            )
            destination_mark, _, destination_points, _ = (
                self._points_from_mark(
                    destination_point_handle,
                    use_multiview=use_multiview,
                )
            )
            if (
                target_mark.observation_handle
                != destination_mark.observation_handle
            ):
                raise ValueError(
                    "placement target and destination marks must come from "
                    "the same observation"
                )
            captured = cast(
                _CapturedObservation,
                self.registry.get(
                    target_mark.observation_handle,
                    expected_kind="observation",
                ),
            )
            source_observation_handle: str | None = (
                target_mark.observation_handle
            )
            point_grounded = True
        else:
            captured = self._capture_observation()
            target_points = self._object_points_from_observation(
                captured,
                current_target_name,
                use_multiview=use_multiview,
            )
            destination_points = self._object_points_from_observation(
                captured,
                destination_name,
                use_multiview=use_multiview,
            )
            source_observation_handle = None
            point_grounded = False
        target_centroid, target_min, target_max = _point_geometry(target_points)
        destination_centroid, destination_min, destination_max = _point_geometry(
            destination_points
        )
        state = captured.robot_state
        eef = np.asarray(state.end_effector_position_m, dtype=np.float64)
        target_extent = np.maximum(target_max - target_min, 0.0)
        destination_extent = np.maximum(destination_max - destination_min, 0.0)
        target_to_eef = float(np.linalg.norm(target_centroid - eef))
        max_target_to_eef = min(
            0.25,
            max(0.12, float(np.linalg.norm(target_extent)) + 0.08),
        )
        same_geometry = (
            float(np.linalg.norm(target_centroid - destination_centroid)) <= 0.015
            and float(np.linalg.norm(target_min - destination_min)) <= 0.02
            and float(np.linalg.norm(target_max - destination_max)) <= 0.02
        )
        distinct_entities = (
            normalized_target != normalized_destination and not same_geometry
        )
        horizontal_distance = float(
            np.linalg.norm(
                target_centroid[:2]
                - (
                    (destination_min[:2] + destination_max[:2]) * 0.5
                    if relation == "in"
                    else destination_centroid[:2]
                )
            )
        )
        max_horizontal_distance = max(
            0.04,
            min(0.07, float(np.min(destination_extent[:2])) * 0.4),
        )
        centered_on_xy = bool(
            np.all(target_centroid[:2] >= destination_min[:2])
            and np.all(target_centroid[:2] <= destination_max[:2])
            and horizontal_distance <= max_horizontal_distance
        )
        target_area = max(
            float(target_extent[0] * target_extent[1]),
            1e-9,
        )
        overlap_x = max(
            0.0,
            float(
                min(target_max[0], destination_max[0])
                - max(target_min[0], destination_min[0])
            ),
        )
        overlap_y = max(
            0.0,
            float(
                min(target_max[1], destination_max[1])
                - max(target_min[1], destination_min[1])
            ),
        )
        footprint_overlap_ratio = overlap_x * overlap_y / target_area
        support_gap = float(target_min[2] - destination_max[2])
        on_height_ready = bool(
            -0.03 <= support_gap <= 0.02
        )
        in_container_xy = bool(
            np.all(target_min[:2] >= destination_min[:2])
            and np.all(target_max[:2] <= destination_max[:2])
        )
        in_container_height = bool(
            target_centroid[2] >= destination_min[2] - 0.03
            and target_centroid[2]
            <= destination_max[2] - _CONTAINER_RIM_CLEARANCE_M
        )
        xy_ready = (
            bool(
                np.all(target_centroid[:2] >= destination_min[:2])
                and np.all(target_centroid[:2] <= destination_max[:2])
                and footprint_overlap_ratio
                >= self.completion_protocol.min_on_footprint_overlap
            )
            if relation == "on"
            else in_container_xy
        )
        height_ready = (
            on_height_ready if relation == "on" else in_container_height
        )
        in_destination_range = xy_ready and height_ready
        if relation == "in":
            destination_bounds_center = (
                destination_min + destination_max
            ) * 0.5
            recommended_eef_delta = np.zeros(3, dtype=np.float64)
            if not in_container_xy:
                recommended_eef_delta[:2] = (
                    destination_bounds_center[:2] - target_centroid[:2]
                )
                correction_stage = "transport_xy"
            elif not in_container_height:
                desired_target_centroid_z = max(
                    destination_min[2],
                    destination_max[2] - _CONTAINER_RIM_CLEARANCE_M,
                )
                recommended_eef_delta[2] = (
                    desired_target_centroid_z - target_centroid[2]
                )
                correction_stage = "lower_into_container"
            else:
                correction_stage = "ready"
        else:
            recommended_eef_delta = np.zeros(3, dtype=np.float64)
            if not xy_ready:
                recommended_eef_delta[:2] = (
                    destination_centroid[:2] - target_centroid[:2]
                )
                correction_stage = "surface_alignment"
            elif not on_height_ready:
                recommended_eef_delta[2] = 0.01 - support_gap
                correction_stage = "lower_to_surface"
            else:
                correction_stage = "ready"
        recommended_eef_position = eef + recommended_eef_delta
        placement_ready = (
            target_to_eef <= max_target_to_eef
            and distinct_entities
            and in_destination_range
        )
        if target_to_eef > max_target_to_eef:
            reason = "target_not_near_end_effector"
        elif not distinct_entities:
            reason = "target_destination_not_distinct"
        elif not in_destination_range:
            reason = "target_outside_destination_range"
        else:
            reason = "placement_ready"
        authorization = (
            self.registry.put(
                "placement_authorization",
                {
                    "tracking_handle": tracking_handle,
                    "current_target_name": current_target_name,
                    "destination_name": destination_name,
                    "relation_type": relation,
                },
            )
            if placement_ready
            else None
        )
        return CapabilityResult.ok(
            {
                "placement_ready": placement_ready,
                "reason": reason,
                "placement_authorization": authorization,
                "tracking_handle": tracking_handle,
                "tracked_object_name": tracked.object_name,
                "current_target_name": current_target_name,
                "destination_name": destination_name,
                "relation_type": relation,
                "point_grounded": point_grounded,
                "target_point_handle": target_point_handle,
                "destination_point_handle": destination_point_handle,
                "source_observation_handle": source_observation_handle,
                "target_centroid_m": [
                    float(value) for value in target_centroid
                ],
                "target_bounds_min_m": [float(value) for value in target_min],
                "target_bounds_max_m": [float(value) for value in target_max],
                "destination_centroid_m": [
                    float(value) for value in destination_centroid
                ],
                "destination_bounds_min_m": [
                    float(value) for value in destination_min
                ],
                "destination_bounds_max_m": [
                    float(value) for value in destination_max
                ],
                "end_effector_position_m": [float(value) for value in eef],
                "target_to_eef_distance_m": target_to_eef,
                "max_target_to_eef_distance_m": max_target_to_eef,
                "horizontal_distance_m": horizontal_distance,
                "max_horizontal_distance_m": max_horizontal_distance,
                "footprint_overlap_ratio": footprint_overlap_ratio,
                "support_vertical_gap_m": support_gap,
                "recommended_eef_delta_m": [
                    float(value) for value in recommended_eef_delta
                ],
                "recommended_eef_position_m": [
                    float(value) for value in recommended_eef_position
                ],
                "distinct_entities": distinct_entities,
                "in_destination_range": in_destination_range,
                "xy_ready": xy_ready,
                "height_ready": height_ready,
                "correction_stage": correction_stage,
                "target_point_count": int(len(target_points)),
                "destination_point_count": int(len(destination_points)),
            }
        )

    def assert_spatial_relation(
        self,
        target_name: str,
        destination_name: str,
        use_multiview: bool = True,
        relation_type: str = "on",
    ) -> CapabilityResult[dict[str, object]]:
        relation = _validate_relation_type(relation_type)
        normalized_target = " ".join(target_name.split()).casefold()
        normalized_destination = " ".join(destination_name.split()).casefold()
        if not normalized_target or not normalized_destination:
            raise ValueError("target_name and destination_name must not be empty")
        captured = self._capture_observation()
        target_points = self._object_points_from_observation(
            captured,
            target_name,
            use_multiview=use_multiview,
        )
        destination_points = self._object_points_from_observation(
            captured,
            destination_name,
            use_multiview=use_multiview,
        )
        return CapabilityResult.ok(
            self._spatial_relation_payload(
                target_name=target_name,
                destination_name=destination_name,
                relation=relation,
                target_points=target_points,
                destination_points=destination_points,
                strict_support_height=False,
            )
        )

    def assert_spatial_relation_from_marks(
        self,
        target_point_handle: str,
        destination_point_handle: str,
        target_label: str,
        destination_label: str,
        relation_type: str = "on",
    ) -> CapabilityResult[dict[str, object]]:
        relation = _validate_relation_type(relation_type)
        target_mark, _, target_points, _ = self._points_from_mark(
            target_point_handle,
            use_multiview=True,
        )
        destination_mark, _, destination_points, _ = self._points_from_mark(
            destination_point_handle,
            use_multiview=True,
        )
        if (
            target_mark.observation_handle
            != destination_mark.observation_handle
        ):
            raise ValueError(
                "target and destination marks must come from the same observation"
            )
        target_name = " ".join(target_label.split())
        destination_name = " ".join(destination_label.split())
        if not target_name or not destination_name:
            raise ValueError(
                "target_label and destination_label must not be empty"
            )
        payload = self._spatial_relation_payload(
            target_name=target_name,
            destination_name=destination_name,
            relation=relation,
            target_points=target_points,
            destination_points=destination_points,
            strict_support_height=True,
        )
        strict_geometry_satisfied = payload["relation_ready"] is True
        strict_geometry_reason = str(payload["reason"])
        target_mark_position = np.asarray(
            target_mark.position,
            dtype=np.float64,
        )
        destination_mark_position = np.asarray(
            destination_mark.position,
            dtype=np.float64,
        )
        _, destination_min, destination_max = _point_geometry(
            destination_points
        )
        mark_inside_destination_xy = bool(
            np.all(target_mark_position[:2] >= destination_min[:2])
            and np.all(target_mark_position[:2] <= destination_max[:2])
        )
        mark_horizontal_distance = float(
            np.linalg.norm(
                target_mark_position[:2]
                - destination_mark_position[:2]
            )
        )
        mark_vertical_delta = float(
            target_mark_position[2] - destination_mark_position[2]
        )
        if relation == "on":
            marked_relation_ready = (
                payload["relation_ready"] is True
                and mark_inside_destination_xy
                and -0.02 <= mark_vertical_delta <= 0.12
            )
            if not mark_inside_destination_xy:
                marked_reason = "target_centroid_outside_destination_xy"
            elif payload["relation_ready"] is not True:
                marked_reason = str(payload["reason"])
            elif not -0.02 <= mark_vertical_delta <= 0.12:
                marked_reason = "target_not_on_destination_support_height"
            else:
                marked_reason = "relation_ready"
        else:
            marked_relation_ready = (
                payload["distinct_entities"] is True
                and payload["centroid_inside_destination_xy"] is True
                and mark_inside_destination_xy
                and payload["reason"] != "target_not_inside_destination_height"
            )
            marked_reason = (
                "relation_ready"
                if marked_relation_ready
                else str(payload["reason"])
            )
        payload["relation_ready"] = marked_relation_ready
        payload["reason"] = (
            marked_reason
            if payload["distinct_entities"] is True
            else "target_destination_not_distinct"
        )
        payload.update(
            {
                "target_point_handle": target_point_handle,
                "destination_point_handle": destination_point_handle,
                "source_observation_handle": target_mark.observation_handle,
                "target_mark_position_m": [
                    float(value) for value in target_mark_position
                ],
                "destination_mark_position_m": [
                    float(value) for value in destination_mark_position
                ],
                "mark_inside_destination_xy": mark_inside_destination_xy,
                "mark_horizontal_distance_m": mark_horizontal_distance,
                "mark_vertical_delta_m": mark_vertical_delta,
                "strict_geometry_satisfied": strict_geometry_satisfied,
                "strict_geometry_reason": strict_geometry_reason,
                "relation_basis": "point_grounded_marks",
            }
        )
        return CapabilityResult.ok(payload)

    def gripper_state(self) -> CapabilityResult[dict[str, object]]:
        state = self._robot_gripper_state()
        if not state.success or not isinstance(state.value, Mapping):
            return state
        return CapabilityResult.ok(
            {
                **dict(state.value),
                "profile": self.gripper_profile.public(),
                **({"object_detection_basis": "backend_gripper_signal_not_target_identity",
                    "target_attachment_status": "insufficient" if state.value.get("object_detected") else "contradicted",
                    "evidence_limitation": "Aperture/force does not identify the target or prove stable possession. Inspect same-target motion in pre/post images."}
                   if self._v15_protocol else {}),
            },
            state.message,
        )

    def _robot_gripper_state(
        self,
    ) -> CapabilityResult[dict[str, object]]:
        method = getattr(self.robot, "gripper_state", None)
        if callable(method):
            return method()
        state = self.robot.get_robot_state()
        if not state.success or state.value is None:
            return CapabilityResult.failed(
                state.failure_category,
                state.message,
                executed=state.executed,
            )
        open_fraction = float(state.value.gripper_open_fraction)
        return CapabilityResult.ok(
            {
                "open_fraction": open_fraction,
                "object_detected": bool(0.02 <= open_fraction < 0.90),
                "force_n": None,
                "force_available": False,
            }
        )

    def grasp_pose_estimate(
        self,
        observation_handle: str,
        detection_id: str,
        gripper_profile: Mapping[str, object],
        backend: str = "auto",
        depth_source: str = "sensor",
        grasp_style: str = "auto",
        top_k: int = 15,
    ) -> CapabilityResult[dict[str, object]]:
        observation = cast(
            _CapturedObservation,
            self.registry.get(
                observation_handle,
                expected_kind="observation",
            ),
        )
        detection = self._require_selected_detection(
            detection_id,
            observation_handle=observation_handle,
        )
        self._validate_public_gripper_profile(gripper_profile)
        full_points, object_points = self._clouds_from_detection(
            observation,
            detection,
            use_multiview=True,
        )
        tracking_handle = self._track_selected_detection(
            detection_id,
            detection,
            full_points,
            object_points,
        )
        return self._service_aligned_grasp_result(
            full_points,
            object_points,
            observation=observation,
            detection=detection,
            tracking_handle=tracking_handle,
            detection_id=detection_id,
            backend=backend,
            depth_source=depth_source,
            grasp_style=grasp_style,
            top_k=top_k,
        )

    def plan_grasp_from_point_clouds(
        self,
        full_point_cloud_handle: str,
        object_point_cloud_handle: str,
        detection_id: str,
        backend: str = "auto",
        depth_source: str = "sensor",
        grasp_style: str = "auto",
        top_k: int = 15,
    ) -> CapabilityResult[dict[str, object]]:
        detection = self._require_selected_detection(detection_id)
        full_points = np.asarray(
            self.registry.get(
                full_point_cloud_handle,
                expected_kind="point_cloud",
            ),
            dtype=np.float64,
        )
        object_points = np.asarray(
            self.registry.get(
                object_point_cloud_handle,
                expected_kind="point_cloud",
            ),
            dtype=np.float64,
        )
        tracking_handle = self._track_selected_detection(
            detection_id,
            detection,
            full_points,
            object_points,
        )
        observation = cast(
            _CapturedObservation,
            self.registry.get(
                detection.observation_handle,
                expected_kind="observation",
            ),
        )
        return self._service_aligned_grasp_result(
            full_points,
            object_points,
            observation=observation,
            detection=detection,
            tracking_handle=tracking_handle,
            detection_id=detection_id,
            backend=backend,
            depth_source=depth_source,
            grasp_style=grasp_style,
            top_k=top_k,
        )

    def _service_aligned_grasp_result(
        self,
        full_points: np.ndarray,
        object_points: np.ndarray,
        *,
        observation: _CapturedObservation,
        detection: _Detection,
        tracking_handle: str,
        detection_id: str,
        backend: str,
        depth_source: str,
        grasp_style: str,
        top_k: int,
    ) -> CapabilityResult[dict[str, object]]:
        if self._dehardcoded:
            return self._dehardcoded_grasp_result(
                full_points,
                object_points,
                observation=observation,
                detection=detection,
                tracking_handle=tracking_handle,
                detection_id=detection_id,
                backend=backend,
                depth_source=depth_source,
                grasp_style=grasp_style,
                top_k=top_k,
            )
        style = _validate_grasp_style(grasp_style)
        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or not 1 <= top_k <= 32
        ):
            raise ValueError("top_k must be an integer in [1, 32]")
        object_centroid, object_min, object_max = _point_geometry(
            object_points
        )
        rejected = {
            "width_infeasible": 0,
            "unreachable": 0,
            "alignment": 0,
        }
        candidate_diagnostics: list[dict[str, object]] = []
        try:
            poses, scores = _validate_grasps(
                *self.services.plan_grasps(
                    full_points,
                    object_points,
                )
            )
        except ToolOperationError as exc:
            reason = (
                "no_candidate"
                if "no candidate" in str(exc).casefold()
                else "service_error"
            )
            return self._grasp_estimate_failure(
                reason,
                rejected=rejected,
                object_extent=object_max - object_min,
                message=str(exc),
                tracking_handle=tracking_handle,
                detection_id=detection_id,
            )
        except Exception as exc:
            return self._grasp_estimate_failure(
                "service_error",
                rejected=rejected,
                object_extent=object_max - object_min,
                message=f"{type(exc).__name__}: {exc}",
                tracking_handle=tracking_handle,
                detection_id=detection_id,
            )
        poses = _apply_capx_grasp_convention(
            poses,
            rotate_tool=True,
        )
        state_result = self.robot.get_robot_state()
        if not state_result.success or state_result.value is None:
            return self._grasp_estimate_failure(
                "service_error",
                rejected=rejected,
                object_extent=object_max - object_min,
                message=state_result.message or "robot state unavailable",
                tracking_handle=tracking_handle,
                detection_id=detection_id,
            )
        state = state_result.value
        seed = np.asarray(
            [
                *state.joint_positions_rad,
                state.gripper_open_fraction * 0.04,
            ],
            dtype=np.float64,
        )
        candidates: list[_ReachableGrasp] = []
        for pose, score in zip(poses, scores):
            rotation = np.asarray(pose[:3, :3], dtype=np.float64)
            outward = rotation @ np.array(
                [0.0, 0.0, -1.0],
                dtype=np.float64,
            )
            top_down_alignment = float(np.clip(-outward[2], 0.0, 1.0))
            if (
                style == "top_down"
                and top_down_alignment < _MIN_TOP_DOWN_ALIGNMENT
            ) or (
                style == "side"
                and abs(float(outward[2]))
                > _MAX_SIDE_VERTICAL_ALIGNMENT
            ):
                rejected["alignment"] += 1
                candidate_diagnostics.append(
                    {
                        "model_score": float(score),
                        "reject_reason": "alignment",
                        "top_down_alignment": top_down_alignment,
                    }
                )
                continue
            predicted_width = self._predicted_grasp_width(
                np.asarray(pose, dtype=np.float64),
                object_points,
            )
            width_feasible = (
                predicted_width is not None
                and 0.003 <= predicted_width
                <= self.gripper_profile.max_width_m
            )
            if not width_feasible:
                rejected["width_infeasible"] += 1
                candidate_diagnostics.append(
                    {
                        "model_score": float(score),
                        "reject_reason": "width_infeasible",
                        "predicted_grasp_width_m": predicted_width,
                    }
                )
                continue
            quaternion = _matrix_to_quaternion_wxyz(rotation)
            target = (
                pose[:3, 3]
                + rotation @ self._tcp_offset_local_m
            )
            target_joints = self.services.solve_ik(
                target,
                quaternion,
                seed,
            )
            target_cfg = _valid_ik_configuration(target_joints)
            if target_cfg is None:
                rejected["unreachable"] += 1
                candidate_diagnostics.append(
                    {
                        "model_score": float(score),
                        "reject_reason": "ik",
                        "target_ik": target_cfg is not None,
                        "predicted_grasp_width_m": predicted_width,
                    }
                )
                continue
            target_delta = target_cfg[:7] - seed[:7]
            total_travel = float(
                np.linalg.norm(target_cfg[:7] - seed[:7])
            )
            if (
                total_travel > _MAX_PREDICTED_JOINT_TRAVEL_RAD
                or float(np.max(np.abs(target_delta)))
                > _MAX_PREDICTED_SINGLE_JOINT_DELTA_RAD
            ):
                rejected["unreachable"] += 1
                candidate_diagnostics.append(
                    {
                        "model_score": float(score),
                        "reject_reason": "controller_path_risk",
                        "predicted_joint_travel_rad": total_travel,
                        "max_target_joint_delta_rad": float(
                            np.max(np.abs(target_delta))
                        ),
                        "predicted_grasp_width_m": predicted_width,
                    }
                )
                continue
            candidates.append(
                _ReachableGrasp(
                    pose=np.asarray(pose, dtype=np.float64),
                    score=float(score),
                    predicted_joint_travel_rad=total_travel,
                    approach_direction_base=tuple(
                        float(value) for value in outward
                    ),
                    top_down_alignment=top_down_alignment,
                    convergence_risk=_grasp_convergence_risk(
                        total_travel,
                        top_down_alignment=top_down_alignment,
                    ),
                    predicted_grasp_width_m=float(predicted_width),
                    width_feasible=True,
                    backend="contact_graspnet",
                )
            )
        if not candidates:
            if rejected["width_infeasible"] and not (
                rejected["unreachable"] or rejected["alignment"]
            ):
                reason = "no_candidate"
            elif rejected["unreachable"] and not rejected["alignment"]:
                reason = "no_reachable_candidate"
            elif rejected["alignment"]:
                reason = "no_candidate_matching_style"
            else:
                reason = "no_candidate"
            return self._grasp_estimate_failure(
                reason,
                rejected=rejected,
                object_extent=object_max - object_min,
                message="no service candidate passed V9 feasibility checks",
                diagnostics=candidate_diagnostics,
                tracking_handle=tracking_handle,
                detection_id=detection_id,
                candidates_returned=len(scores),
            )
        candidates.sort(
            key=lambda item: (
                item.predicted_joint_travel_rad,
                -item.score,
            )
        )
        public_candidates = [
            {
                "candidate_index": index,
                "position_m": [
                    float(value) for value in candidate.pose[:3, 3]
                ],
                "quaternion_wxyz": list(
                    _matrix_to_quaternion_wxyz(
                        candidate.pose[:3, :3]
                    )
                ),
                "model_score": candidate.score,
                "predicted_grasp_width_m": (
                    candidate.predicted_grasp_width_m
                ),
                "width_feasible": candidate.width_feasible,
                "approach_direction_world": list(
                    candidate.approach_direction_base
                ),
                "reachable": True,
                "predicted_joint_travel_rad": (
                    candidate.predicted_joint_travel_rad
                ),
                "backend": candidate.backend,
                "tracking_handle": tracking_handle,
                "detection_id": detection_id,
            }
            for index, candidate in enumerate(candidates[:top_k])
        ]
        return CapabilityResult.ok(
            {
                "candidates": public_candidates,
                "candidate_count": len(public_candidates),
                "rejected": rejected,
                "object_extent_m": [
                    float(value) for value in object_max - object_min
                ],
                "object_centroid_m": [
                    float(value) for value in object_centroid
                ],
                "gripper_profile": self.gripper_profile.public(),
                "grasp_style": style,
                "tracking_handle": tracking_handle,
                "detection_id": detection_id,
                "identity_preserved": True,
            }
        )

    def _dehardcoded_grasp_result(
        self,
        full_points: np.ndarray,
        object_points: np.ndarray,
        *,
        observation: _CapturedObservation,
        detection: _Detection,
        tracking_handle: str,
        detection_id: str,
        backend: str,
        depth_source: str,
        grasp_style: str,
        top_k: int,
    ) -> CapabilityResult[dict[str, object]]:
        style = _validate_grasp_style(grasp_style)
        selected_backend = _validate_grasp_backend(backend)
        selected_depth_source = _validate_depth_source(depth_source)
        self._grasp_attempt_ledger += 1
        self._grasp_styles_tried.add(style)
        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or top_k <= 0
        ):
            raise ValueError("top_k must be a positive integer")
        object_centroid, object_min, object_max = _point_geometry(
            object_points
        )
        # After a close-on-air, retry with the mask-geometry candidate: its
        # grasp point is the fresh segmentation centroid, which does not
        # carry the model backends' small-object pose error. A mask attempt
        # that itself closed on air routes back to the model backends.
        empty_close_backend = None
        if self._v15_protocol and self._last_empty_close is not None:
            last = self._last_empty_close
            last_centroid = np.asarray(
                last.get("centroid"),
                dtype=np.float64,
            ).reshape(3)
            same_detection = (
                last.get("source_detection_id") is not None
                and last.get("source_detection_id") == detection_id
            )
            appearance = last.get("appearance_rgb_mean")
            old_extent = last.get("extent")
            # Cross-observation association needs geometry AND appearance;
            # nearby distractors and changed wording alone are insufficient.
            associated = False
            if appearance is not None and old_extent is not None:
                prior_extent = np.asarray(old_extent, dtype=np.float64)
                new_extent = object_max - object_min
                scale = float(np.linalg.norm(new_extent) / max(np.linalg.norm(prior_extent), 1e-6))
                radius = min(0.05, max(0.015, float(np.linalg.norm(prior_extent)) * 0.25))
                associated = (
                    float(np.linalg.norm(object_centroid - last_centroid)) <= radius
                    and 0.5 <= scale <= 2.0
                    and float(np.linalg.norm(detection.appearance_rgb_mean - np.asarray(appearance))) <= 45.0
                )
            if same_detection or associated:
                empty_close_backend = str(last.get("backend"))
        mask_retry = False
        if (
            selected_backend == "auto"
            and empty_close_backend is not None
            and empty_close_backend != "mask_topdown_fallback"
        ):
            try:
                batch = self._mask_topdown_fallback_batch(
                    object_points,
                    backend_failure=(
                        "empty_close_retry: previous "
                        f"{empty_close_backend} grasp closed on air"
                    ),
                )
                poses, scores = _validate_grasps(
                    batch.poses,
                    batch.scores,
                )
                mask_retry = True
            except Exception:
                mask_retry = False
        if not mask_retry:
            try:
                batch = self._plan_grasps_for_backend(
                    selected_backend,
                    full_points,
                    object_points,
                    observation=observation,
                    detection=detection,
                    depth_source=selected_depth_source,
                )
                poses, scores = _validate_grasps(
                    batch.poses,
                    batch.scores,
                )
            except ToolOperationError as exc:
                reason = (
                    "no_candidate"
                    if "no candidate" in str(exc).casefold()
                    else "service_error"
                )
                if self._v15_protocol and reason == "no_candidate":
                    batch = self._mask_topdown_fallback_batch(
                        object_points,
                        backend_failure=str(exc),
                    )
                    poses, scores = _validate_grasps(
                        batch.poses,
                        batch.scores,
                    )
                else:
                    return self._grasp_estimate_failure(
                        reason,
                        rejected={
                            "width_infeasible": 0,
                            "unreachable": 0,
                            "alignment": 0,
                        },
                        object_extent=object_max - object_min,
                        message=str(exc),
                        tracking_handle=tracking_handle,
                        detection_id=detection_id,
                    )
            except Exception as exc:
                backend_reason = getattr(exc, "reason", None)
                no_candidate_reasons = {
                    "no_grasp_candidates",
                    "no_candidate",
                    "insufficient_object_points",
                    "empty_object_mask",
                    "empty_target_mask",
                    "empty_object_pointcloud",
                    "object_pointcloud_too_small",
                    "all_grasps_colliding",
                }
                reason = (
                    "no_candidate"
                    if backend_reason in no_candidate_reasons
                    else "service_error"
                )
                details = getattr(exc, "details", None)
                if self._v15_protocol and reason == "no_candidate":
                    batch = self._mask_topdown_fallback_batch(
                        object_points,
                        backend_failure=f"{type(exc).__name__}: {exc}",
                    )
                    poses, scores = _validate_grasps(
                        batch.poses,
                        batch.scores,
                    )
                else:
                    return self._grasp_estimate_failure(
                        reason,
                        rejected={
                            "width_infeasible": 0,
                            "unreachable": 0,
                            "alignment": 0,
                        },
                        object_extent=object_max - object_min,
                        message=f"{type(exc).__name__}: {exc}",
                        diagnostics=(
                            (
                                {
                                    "backend_reason": backend_reason,
                                    "backend_details": dict(details),
                                },
                            )
                            if isinstance(details, Mapping)
                            else ()
                        ),
                        tracking_handle=tracking_handle,
                        detection_id=detection_id,
                    )
        batch_approaches = batch.approach_directions_base
        if (
            batch_approaches is not None
            and len(batch_approaches) != len(poses)
        ):
            batch_approaches = None
        if batch.grasp_frame == "graspnet":
            poses, batch_approaches = self._compile_graspnet_poses(
                poses
            )
        elif not batch.public_tool_convention:
            poses = _apply_capx_grasp_convention(
                poses,
                rotate_tool=True,
            )
        widths = (
            batch.widths_m
            if len(batch.widths_m) == len(poses)
            else tuple(None for _ in poses)
        )
        native_candidates = (
            batch.native_candidates
            if len(batch.native_candidates) == len(poses)
            else tuple({} for _ in poses)
        )
        object_extent_vector = object_max - object_min
        small_target_topdown_priority = bool(
            self._v15_protocol
            and style == "auto"
            and float(
                max(
                    object_extent_vector[0],
                    object_extent_vector[1],
                )
            )
            < _SMALL_TARGET_EXTENT_M
        )
        preference_style = (
            "top_down" if small_target_topdown_priority else style
        )
        state_result = self.robot.get_robot_state()
        if not state_result.success or state_result.value is None:
            return self._grasp_estimate_failure(
                "service_error",
                rejected={
                    "width_infeasible": 0,
                    "unreachable": 0,
                    "alignment": 0,
                },
                object_extent=object_max - object_min,
                message=state_result.message or "robot state unavailable",
                tracking_handle=tracking_handle,
                detection_id=detection_id,
            )
        state = state_result.value
        seed = np.asarray(
            [
                *state.joint_positions_rad,
                state.gripper_open_fraction * 0.04,
            ],
            dtype=np.float64,
        )
        candidates: list[_ReachableGrasp] = []
        quality_summary: dict[str, int] = {}
        for index, (
            pose,
            score,
            service_width,
            native_candidate,
        ) in enumerate(
            zip(poses, scores, widths, native_candidates)
        ):
            rotation = np.asarray(pose[:3, :3], dtype=np.float64)
            outward = (
                np.asarray(batch_approaches[index], dtype=np.float64)
                if batch_approaches is not None
                else rotation
                @ np.array(
                    [0.0, 0.0, -1.0],
                    dtype=np.float64,
                )
            )
            outward_norm = float(np.linalg.norm(outward))
            if math.isfinite(outward_norm) and outward_norm > 1e-12:
                outward = outward / outward_norm
            else:
                outward = rotation @ np.array(
                    [0.0, 0.0, -1.0],
                    dtype=np.float64,
                )
            top_down_alignment = float(np.clip(-outward[2], 0.0, 1.0))
            style_preference_score = _grasp_style_preference(
                preference_style,
                top_down_alignment=top_down_alignment,
            )
            request_style_match = "not_requested"
            if self._v15_protocol:
                style_evidence = grasp_style_evidence(
                    outward, style, reference_up=(0., 0., 1.),
                    min_top_down_alignment=_MIN_TOP_DOWN_ALIGNMENT,
                    max_side_vertical_alignment=_MAX_SIDE_VERTICAL_ALIGNMENT,
                )
                request_style_match = str(style_evidence["match"])
                # Retain the existing auto preference, but use the signed
                # vertical component: an upward grasp is not a side grasp.
                preference_evidence = grasp_style_evidence(
                    outward, preference_style, reference_up=(0., 0., 1.),
                    min_top_down_alignment=_MIN_TOP_DOWN_ALIGNMENT,
                    max_side_vertical_alignment=_MAX_SIDE_VERTICAL_ALIGNMENT,
                )
                style_preference_score = float(preference_evidence["score"])
            quality_flags: list[str] = []
            if batch.backend == "mask_topdown_fallback":
                quality_flags.append("mask_topdown_fallback")
            if small_target_topdown_priority:
                quality_flags.append("small_target_topdown_priority")
            if (
                style == "top_down"
                and top_down_alignment < _MIN_TOP_DOWN_ALIGNMENT
            ) or (
                style == "side"
                and abs(float(outward[2]))
                > _MAX_SIDE_VERTICAL_ALIGNMENT
            ):
                quality_flags.append("style_preference_low")
            predicted_width = (
                service_width
                if service_width is not None
                else self._predicted_grasp_width(
                    np.asarray(pose, dtype=np.float64),
                    object_points,
                )
            )
            if (
                predicted_width is not None
                and not math.isfinite(float(predicted_width))
            ):
                predicted_width = None
            width_known = predicted_width is not None
            width_feasible = (
                True
                if predicted_width is None
                else 0.003
                <= predicted_width
                <= self.gripper_profile.max_width_m
            )
            if not width_known:
                quality_flags.append("width_unknown")
            elif not width_feasible:
                quality_flags.append("width_out_of_range")
            quaternion = _matrix_to_quaternion_wxyz(rotation)
            target = pose[:3, 3] + rotation @ self._tcp_offset_local_m
            target_joints = self.services.solve_ik(
                target,
                quaternion,
                seed,
            )
            target_cfg = _valid_ik_configuration(target_joints)
            ik_feasible = target_cfg is not None
            predicted_joint_travel: float | None = None
            if target_cfg is None:
                quality_flags.append("ik_unavailable")
            else:
                delta = target_cfg[:7] - seed[:7]
                predicted_joint_travel = float(np.linalg.norm(delta))
                if (
                    predicted_joint_travel
                    > _MAX_PREDICTED_JOINT_TRAVEL_RAD
                ):
                    quality_flags.append("joint_travel_large")
                if (
                    float(np.max(np.abs(delta)))
                    > _MAX_PREDICTED_SINGLE_JOINT_DELTA_RAD
                ):
                    quality_flags.append("single_joint_delta_large")
            for flag in quality_flags:
                quality_summary[flag] = quality_summary.get(flag, 0) + 1
                self._grasp_quality_ledger[flag] += 1
            candidates.append(
                _ReachableGrasp(
                    pose=np.asarray(pose, dtype=np.float64),
                    score=float(score),
                    predicted_joint_travel_rad=predicted_joint_travel,
                    approach_direction_base=tuple(
                        float(value) for value in outward
                    ),
                    top_down_alignment=top_down_alignment,
                    convergence_risk=_grasp_convergence_risk(
                        predicted_joint_travel,
                        top_down_alignment=top_down_alignment,
                    ),
                    predicted_grasp_width_m=predicted_width,
                    width_feasible=width_feasible,
                    width_known=width_known,
                    ik_feasible=ik_feasible,
                    style_preference_score=style_preference_score,
                    request_style_match=request_style_match,
                    quality_flags=tuple(quality_flags),
                    backend=batch.backend,
                    backend_candidate=(
                        dict(native_candidate)
                        if isinstance(native_candidate, Mapping)
                        else None
                    ),
                )
            )
        if not candidates:
            return self._grasp_estimate_failure(
                "no_candidate",
                rejected={
                    "width_infeasible": 0,
                    "unreachable": 0,
                    "alignment": 0,
                },
                object_extent=object_max - object_min,
                message="grasp backend returned no candidates",
                tracking_handle=tracking_handle,
                detection_id=detection_id,
            )
        # convergence_risk is a diagnostic only: the V15 mainline showed the
        # joint-travel thresholds are non-monotone in observed convergence
        # under the staged pregrasp path, so it must not steer ordering.
        candidates.sort(
            key=lambda item: (
                (not self._grasp_contact_evidence(item.pose, object_points).get("visible_target_in_pad_sweep", True)
                 if self._v15_protocol else False),
                (not self._grasp_contact_evidence(item.pose, object_points)["visible_target_within_reach"]
                 if self._v15_protocol else False),
                -item.style_preference_score,
                -item.score,
            )
        )
        public_candidates = [
            {
                "candidate_index": index,
                "position_m": [
                    float(value) for value in candidate.pose[:3, 3]
                ],
                "quaternion_wxyz": list(
                    _matrix_to_quaternion_wxyz(
                        candidate.pose[:3, :3]
                    )
                ),
                "model_score": candidate.score,
                "predicted_grasp_width_m": (
                    candidate.predicted_grasp_width_m
                ),
                "width_feasible": candidate.width_feasible,
                "width_known": candidate.width_known,
                "approach_direction_world": list(
                    candidate.approach_direction_base
                ),
                "reachable": candidate.ik_feasible,
                "ik_feasible": candidate.ik_feasible,
                "predicted_joint_travel_rad": (
                    candidate.predicted_joint_travel_rad
                ),
                "top_down_alignment": candidate.top_down_alignment,
                **(
                    {
                        "convergence_risk": (
                            candidate.convergence_risk
                        ),
                        "target_contact_evidence": self._grasp_contact_evidence(candidate.pose, object_points),
                        "request_style_match": candidate.request_style_match,
                        "feasibility_evidence": {
                            "endpoint_ik": "solution_returned" if candidate.ik_feasible else "unavailable",
                            "width": (
                                "unknown" if not candidate.width_known else
                                "within_reported_limits" if candidate.width_feasible else "outside_reported_limits"
                            ),
                            "motion_path": "not_checked",
                        },
                    }
                    if self._v15_protocol
                    else {}
                ),
                "style_preference_score": (
                    candidate.style_preference_score
                ),
                "quality_flags": list(candidate.quality_flags),
                "backend_candidate": (
                    dict(candidate.backend_candidate)
                    if candidate.backend_candidate is not None
                    else None
                ),
                "backend": candidate.backend,
                "tracking_handle": tracking_handle,
                "detection_id": detection_id,
            }
            for index, candidate in enumerate(candidates[:top_k])
        ]
        choice_feedback = {}
        if self._v15_protocol:
            public_candidates = [self._issue_grasp_candidate({
                **item, "source_observation_handle": detection.observation_handle,
            }) for item in public_candidates]
        tracked = cast(_TrackedEntity, self.registry.get(tracking_handle, expected_kind="entity_track"))
        tracked.issued_grasp_candidates = {
            item["candidate_index"]: deepcopy(item) for item in public_candidates
        }
        if self._v15_protocol:
            choice_feedback = summarize_grasp_choices(
                [item.request_style_match for item in candidates],
                returned_count=len(public_candidates), requested_style=style,
                source_observation_handle=detection.observation_handle,
            )
            choice_feedback["selection_feedback"]["style_reference"] = {
                "frame": "robot_base", "up_axis": [0., 0., 1.],
                "min_top_down_alignment": _MIN_TOP_DOWN_ALIGNMENT,
                "max_side_vertical_alignment": _MAX_SIDE_VERTICAL_ALIGNMENT,
            }
        result = CapabilityResult.ok(
            {
                "candidates": public_candidates,
                **choice_feedback,
                "candidate_count": len(public_candidates),
                "returned": len(public_candidates),
                "filtered": 0,
                "rejected": {
                    "width_infeasible": 0,
                    "unreachable": 0,
                    "alignment": 0,
                },
                "quality_summary": quality_summary,
                "object_extent_m": [
                    float(value) for value in object_max - object_min
                ],
                "object_centroid_m": [
                    float(value) for value in object_centroid
                ],
                "gripper_profile": self.gripper_profile.public(),
                "grasp_style": style,
                "backend": batch.backend,
                "depth_source": selected_depth_source,
                "depth_quality": dict(batch.depth_quality),
                "tracking_handle": tracking_handle,
                "detection_id": detection_id,
                "identity_preserved": True,
                "episode_ledger": self._public_grasp_ledger(
                    detection_id
                ),
            }
        )
        if self._sensor_motion:
            from .grasp_screening import screen_grasp_result
            result = screen_grasp_result(self, result, local_search=True, orientation_search=True)
            tracked.issued_grasp_candidates = {
                item['candidate_index']: deepcopy(item) for item in result.value['candidates']}
        return result

    def _plan_grasps_for_backend(
        self,
        backend: str,
        full_points: np.ndarray,
        object_points: np.ndarray,
        *,
        observation: _CapturedObservation,
        detection: _Detection,
        depth_source: str,
    ) -> GraspServiceBatch:
        if backend == "mask_topdown_fallback":
            self._grasp_backends_tried.add(backend)
            return self._mask_topdown_fallback_batch(
                object_points, backend_failure="explicit public geometry request"
            )
        method = getattr(self.services, "predict_grasps", None)
        if callable(method):
            camera = self._camera_by_name(
                observation,
                detection.camera_name,
            )
            if camera.depth is None:
                raise ToolOperationError(
                    FailureCategory.PERCEPTION_UNCERTAIN,
                    "selected detection camera has no public depth",
                )
            if backend == "auto":
                available_method = getattr(
                    self.services,
                    "available_grasp_backends",
                    None,
                )
                available = (
                    tuple(str(value) for value in available_method())
                    if callable(available_method)
                    else ("contact_graspnet",)
                )
                backends = tuple(
                    name
                    for name in (
                        "contact_graspnet",
                        "graspgenx",
                        "anygrasp",
                    )
                    if name in available
                )
            else:
                backends = (backend,)
            empty_errors = []
            for selected in backends:
                self._grasp_backends_tried.add(selected)
                try:
                    request = {
                        "backend": selected,
                        "image": camera.image,
                        "depth": camera.depth,
                        "mask": detection.mask,
                        "intrinsics": camera.intrinsics,
                        "extrinsics": camera.extrinsics,
                        "gripper_profile": self.gripper_profile,
                        "depth_source": depth_source,
                    }
                    if self._v15_protocol:
                        request["collision_relaxed"] = False
                    return method(
                        **request,
                    )
                except OSError as exc:
                    lowered = str(exc).casefold()
                    backend_reason = str(
                        getattr(exc, "reason", "")
                    ).casefold()
                    if (
                        self._v15_protocol
                        and selected == "graspgenx"
                        and backend_reason == "all_grasps_colliding"
                    ):
                        try:
                            return method(
                                backend=selected,
                                image=camera.image,
                                depth=camera.depth,
                                mask=detection.mask,
                                intrinsics=camera.intrinsics,
                                extrinsics=camera.extrinsics,
                                gripper_profile=self.gripper_profile,
                                depth_source=depth_source,
                                collision_relaxed=True,
                            )
                        except OSError as relaxed_exc:
                            exc = relaxed_exc
                            lowered = str(exc).casefold()
                            backend_reason = str(
                                getattr(exc, "reason", "")
                            ).casefold()
                    no_candidate = backend_reason in {
                        "no_grasp_candidates",
                        "no_candidate",
                        "all_grasps_colliding",
                        "insufficient_object_points",
                        "empty_object_mask",
                        "empty_target_mask",
                        "empty_object_pointcloud",
                        "object_pointcloud_too_small",
                    } or any(
                        marker in lowered
                        for marker in (
                            "no normalized grasp candidates",
                            "no grasp candidates",
                            "no_grasp_candidates",
                            "no candidate",
                            "empty object",
                            "empty target",
                        )
                    )
                    if backend != "auto" or not no_candidate:
                        raise
                    empty_errors.append(f"{selected}: {exc}")
            raise ToolOperationError(
                FailureCategory.PERCEPTION_UNCERTAIN,
                "no candidate from available auto grasp backends: "
                + "; ".join(empty_errors),
            )
        selected = (
            "contact_graspnet"
            if backend == "auto"
            else backend
        )
        self._grasp_backends_tried.add(selected)
        if selected != "contact_graspnet":
            raise ToolOperationError(
                FailureCategory.TOOL_ERROR,
                f"backend_unavailable: {selected}",
            )
        poses, scores = self.services.plan_grasps(
            full_points,
            object_points,
        )
        return GraspServiceBatch(
            poses=np.asarray(poses, dtype=np.float64),
            scores=np.asarray(scores, dtype=np.float64),
            widths_m=tuple(None for _ in scores),
            backend=selected,
            public_tool_convention=False,
            grasp_frame="capx_native",
            native_candidates=(),
            depth_quality={"source": "sensor_point_cloud"},
        )

    def _mask_topdown_fallback_batch(
        self,
        object_points: np.ndarray,
        *,
        backend_failure: str,
    ) -> GraspServiceBatch:
        points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
        centroid, bounds_min, bounds_max = _point_geometry(points)
        state = self.robot.get_robot_state()
        if not state.success or state.value is None:
            raise ToolOperationError(
                state.failure_category,
                state.message or "robot state unavailable for fallback grasp",
            )
        _hand_position, hand_quaternion = (
            self.robot.external_ik_target_from_eef_pose(
                state.value.end_effector_position_m,
                state.value.end_effector_quaternion_wxyz,
            )
        )
        hand_rotation = _quaternion_wxyz_to_matrix(hand_quaternion)
        centered_xy = points[:, :2] - np.median(points[:, :2], axis=0)
        covariance = centered_xy.T @ centered_xy
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        minor_axis = eigenvectors[:, int(np.argmin(eigenvalues))]
        closing_axis = np.array([*minor_axis, 0.0], dtype=np.float64)
        # Parallel jaws have equivalent signs. Choose the nearest full frame;
        # never retain the current roll/pitch in a top-down candidate.
        if float(np.dot(closing_axis, hand_rotation[:, 1])) < 0:
            closing_axis = -closing_axis
        approach = np.array([0.0, 0.0, -1.0])
        hand_rotation = np.column_stack(
            # Public IK orientation is Panda hand: finger joints slide on Y.
            (np.cross(closing_axis, approach), closing_axis, approach)
        )
        # Observed surface center, not a privileged or inferred volume center.
        # Round-trip the intended EEF center through public frame calibration.
        contact_center = (bounds_min + bounds_max) * 0.5
        hand_position, _ = self.robot.external_ik_target_from_eef_pose(
            contact_center,
            self.robot.eef_pose_from_external_ik_target(
                (0.0, 0.0, 0.0), _matrix_to_quaternion_wxyz(hand_rotation)
            )[1],
        )
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = hand_rotation
        pose[:3, 3] = np.asarray(hand_position) - hand_rotation @ self._tcp_offset_local_m
        # Do not clamp oversized geometry into a falsely feasible jaw width.
        projection = points @ closing_axis
        width = float(np.max(projection) - np.min(projection))
        native_candidate = {
            "id": "mask_topdown_fallback_000",
            "source_model": "public_mask_geometry",
            "frame": "robot_base",
            "score": 0.0,
            "width": width,
            "collision_relaxed": True,
        }
        return GraspServiceBatch(
            poses=np.asarray([pose], dtype=np.float64),
            scores=np.asarray([0.0], dtype=np.float64),
            widths_m=(width,),
            backend="mask_topdown_fallback",
            public_tool_convention=True,
            grasp_frame="capx_native",
            approach_directions_base=((0.0, 0.0, -1.0),),
            native_candidates=(native_candidate,),
            depth_quality={
                "source": "sensor_mask_geometry",
                "fallback_reason": backend_failure,
                "collision_relaxed": True,
            },
        )

    def _compile_graspnet_poses(
        self,
        poses: np.ndarray,
    ) -> tuple[
        np.ndarray,
        tuple[tuple[float, float, float], ...],
    ]:
        if self._graspnet_to_eef_transform is None:
            raise ToolOperationError(
                FailureCategory.TOOL_ERROR,
                "graspnet calibration is unavailable",
            )
        compiled = []
        approaches = []
        for grasp_pose in np.asarray(poses, dtype=np.float64):
            world_from_eef = (
                grasp_pose @ self._graspnet_to_eef_transform
            )
            eef_position = world_from_eef[:3, 3]
            eef_quaternion = _matrix_to_quaternion_wxyz(
                world_from_eef[:3, :3]
            )
            hand_position, hand_quaternion = (
                self.robot.external_ik_target_from_eef_pose(
                    eef_position,
                    eef_quaternion,
                )
            )
            hand_rotation = _quaternion_wxyz_to_matrix(
                hand_quaternion
            )
            public_pose = np.eye(4, dtype=np.float64)
            public_pose[:3, :3] = hand_rotation
            public_pose[:3, 3] = (
                np.asarray(hand_position, dtype=np.float64)
                - hand_rotation @ self._tcp_offset_local_m
            )
            compiled.append(public_pose)
            approach = np.asarray(
                grasp_pose[:3, 0],
                dtype=np.float64,
            )
            norm = float(np.linalg.norm(approach))
            approaches.append(
                tuple(
                    float(value)
                    for value in (
                        approach / norm
                        if norm > 1e-12
                        else np.array([0.0, 0.0, -1.0])
                    )
                )
            )
        return np.stack(compiled), tuple(approaches)

    def _public_grasp_ledger(
        self,
        detection_id: str | None,
    ) -> dict[str, object]:
        available_method = getattr(
            self.services,
            "available_grasp_backends",
            None,
        )
        available = (
            tuple(str(value) for value in available_method())
            if callable(available_method)
            else ("contact_graspnet",)
        )
        object_classes: tuple[str, ...] = ()
        if isinstance(detection_id, str):
            try:
                detection = cast(
                    _Detection,
                    self.registry.get(
                        detection_id,
                        expected_kind="detection",
                    ),
                )
            except ArtifactHandleError:
                pass
            else:
                object_classes = tuple(
                    token
                    for token in re.findall(
                        r"[A-Za-z0-9_-]+",
                        detection.prompt.casefold(),
                    )
                    if len(token) > 2
                )
        experience_entries: tuple[Mapping[str, object], ...] = ()
        if self._experience_snapshot is not None:
            experience_entries = (
                self._experience_snapshot.entries_for_object_classes(
                    object_classes
                )
            )
        return {
            "attempts_this_episode": self._grasp_attempt_ledger,
            "styles_tried": sorted(self._grasp_styles_tried),
            "backends_tried": sorted(self._grasp_backends_tried),
            "rejection_breakdown": dict(
                sorted(self._grasp_quality_ledger.items())
            ),
            "untried": {
                "backends": [
                    value
                    for value in available
                    if value not in self._grasp_backends_tried
                ],
                "views": ["wrist"],
            },
            "experience_entries": [
                dict(value) for value in experience_entries
            ],
        }

    def _grasp_estimate_failure(
        self,
        reason: str,
        *,
        rejected: Mapping[str, int],
        object_extent: Sequence[float],
        message: str,
        diagnostics: Sequence[Mapping[str, object]] = (),
        tracking_handle: str | None = None,
        detection_id: str | None = None,
        candidates_returned: int = 0,
    ) -> CapabilityResult[dict[str, object]]:
        return CapabilityResult.failed(
            (
                FailureCategory.PERCEPTION_UNCERTAIN
                if reason != "service_error"
                else FailureCategory.TOOL_ERROR
            ),
            f"{reason}: {message}",
            executed=True,
            value={
                "reason": reason,
                "candidates_returned": int(candidates_returned),
                "rejected": dict(rejected),
                "object_extent_m": [
                    float(value) for value in object_extent
                ],
                "gripper_max_width_m": (
                    self.gripper_profile.max_width_m
                ),
                "hint_available": [
                    "side",
                    "auto",
                    "other_camera",
                    "closer_view",
                ],
                "candidate_diagnostics": [
                    dict(value) for value in diagnostics[:8]
                ],
                "tracking_handle": tracking_handle,
                "detection_id": detection_id,
                **(
                    {
                        "episode_ledger": self._public_grasp_ledger(
                            detection_id
                        )
                    }
                    if self._dehardcoded
                    else {}
                ),
            },
        )

    def _contact_local_points(self, pose: np.ndarray, object_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rotation = np.asarray(pose[:3, :3], dtype=np.float64)
        controller_target = pose[:3, 3] + rotation @ self._tcp_offset_local_m
        contact_center, _ = self.robot.eef_pose_from_external_ik_target(
            controller_target, _matrix_to_quaternion_wxyz(rotation))
        center = np.asarray(contact_center, dtype=np.float64)
        points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        return center, (points - center) @ rotation

    def _grasp_contact_evidence(self, pose: np.ndarray, object_points: np.ndarray) -> dict[str, object]:
        center, local = self._contact_local_points(pose, object_points)
        # Panda hand Y is the jaw closing axis. Use only the public hardware
        # envelope: lack of visible support is a warning, not a hidden collision test.
        reach = np.array([self.gripper_profile.fingertip_depth_m,
                          self.gripper_profile.max_width_m / 2,
                          self.gripper_profile.fingertip_depth_m])
        nearby = np.all(np.abs(local) <= reach, axis=1)
        pad_evidence: dict[str, object] = {}
        hardware_geometry = getattr(self.robot, "gripper_contact_envelope", None)
        envelope = hardware_geometry() if callable(hardware_geometry) else None
        if isinstance(envelope, Mapping):
            lower = np.asarray(envelope["pad_min_local_m"], dtype=np.float64).copy()
            upper = np.asarray(envelope["pad_max_local_m"], dtype=np.float64).copy()
            lower[1], upper[1] = -reach[1], reach[1]
            # Same 3 mm contact-position tolerance as the guarded descent.
            margin = 0.003
            inside = np.all((local >= lower - margin) & (local <= upper + margin), axis=1)
            from .contact_geometry import pad_support
            pad_evidence = {
                "contact_region_support": pad_support(local, lower, upper, margin),
                "visible_target_in_pad_sweep": bool(inside.any()),
                "pad_sweep_min_local_m": lower.tolist(),
                "pad_sweep_max_local_m": upper.tolist(),
                "contact_position_tolerance_m": margin,
                "pad_geometry_basis": envelope["basis"],
            }
        return {
            **pad_evidence,
            "contact_center_m": center.tolist(),
            "closing_axis_hand": "y",
            "target_local_bounds_min_m": np.min(local, axis=0).tolist() if len(local) else None,
            "target_local_bounds_max_m": np.max(local, axis=0).tolist() if len(local) else None,
            "nearest_visible_target_m": float(np.min(np.linalg.norm(local, axis=1))) if len(local) else None,
            "visible_target_within_reach": bool(nearby.any()),
            "visible_point_count_in_envelope": int(nearby.sum()),
            "basis": "calibrated EEF contact center and selected public target cloud",
            "limitation": "Neither hardware reach nor pad-sweep overlap proves a grasp. Compare signed target bounds with pad bounds and wrist images before descending. A target below the pad sweep can produce an empty close even after perfect motion convergence. Missing visible support suggests re-localizing or changing the candidate, but occlusion can hide contact surfaces.",
        }

    def _predicted_grasp_width(
        self,
        pose: np.ndarray,
        object_points: np.ndarray,
    ) -> float | None:
        if self._v15_protocol:
            _center, local = self._contact_local_points(pose, object_points)
            depth = self.gripper_profile.fingertip_depth_m
            slab = local[(np.abs(local[:, 2]) <= depth) & (np.abs(local[:, 0]) <= depth)]
            if len(slab) < 3:
                return None
            lower, upper = np.percentile(slab[:, 1], (5., 95.))
            return float(upper - lower)
        rotation = np.asarray(pose[:3, :3], dtype=np.float64)
        contact_center = (
            np.asarray(pose[:3, 3], dtype=np.float64)
            + rotation @ self._tcp_offset_local_m
        )
        local = (
            rotation.T
            @ (
                np.asarray(object_points, dtype=np.float64)
                - contact_center
            ).T
        ).T
        depth = self.gripper_profile.fingertip_depth_m
        slab = local[
            (np.abs(local[:, 2]) <= depth)
            & (np.abs(local[:, 1]) <= max(0.03, depth))
        ]
        if len(slab) < 3:
            return None
        lower, upper = np.percentile(slab[:, 0], (5.0, 95.0))
        width = float(upper - lower)
        return width if math.isfinite(width) else None

    def _require_selected_detection(
        self,
        detection_id: str,
        *,
        observation_handle: str | None = None,
    ) -> _Detection:
        detection = cast(
            _Detection,
            self.registry.get(
                detection_id,
                expected_kind="detection",
            ),
        )
        if (
            observation_handle is not None
            and detection.observation_handle != observation_handle
        ):
            raise ValueError(
                "detection does not belong to the supplied observation"
            )
        if not detection.selected:
            raise ToolOperationError(
                FailureCategory.PERCEPTION_UNCERTAIN,
                "detection_selection_required: call select_detection first",
            )
        return detection

    def _align_detection_to_observation_camera(
        self,
        source: _Detection,
        *,
        observation_handle: str,
        camera_name: str,
    ) -> tuple[str, _Detection, float]:
        aligned = self.segment(
            source.prompt,
            observation_handle,
            camera_name,
        )
        if not aligned.success or not isinstance(aligned.value, Mapping):
            raise ToolOperationError(
                FailureCategory.PERCEPTION_UNCERTAIN,
                aligned.message
                or "destination segmentation produced no aligned detection",
            )
        public_candidates = aligned.value.get("detections")
        if not isinstance(public_candidates, list) or not public_candidates:
            raise ToolOperationError(
                FailureCategory.PERCEPTION_UNCERTAIN,
                "destination segmentation produced no aligned detection",
            )
        candidates: list[tuple[float, float, str, _Detection]] = []
        source_extent = source.bounds_max - source.bounds_min
        for public in public_candidates:
            detection_id = (
                public.get("detection_id")
                if isinstance(public, Mapping)
                else None
            )
            if not isinstance(detection_id, str):
                continue
            detection = cast(
                _Detection,
                self.registry.get(
                    detection_id,
                    expected_kind="detection",
                ),
            )
            centroid_residual = float(
                np.linalg.norm(detection.centroid - source.centroid)
            )
            extent_residual = float(
                np.linalg.norm(
                    (detection.bounds_max - detection.bounds_min)
                    - source_extent
                )
            )
            candidates.append(
                (
                    centroid_residual,
                    extent_residual,
                    detection_id,
                    detection,
                )
            )
        if not candidates:
            raise ToolOperationError(
                FailureCategory.PERCEPTION_UNCERTAIN,
                "aligned destination detections contain no valid handles",
            )
        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
                -item[3].score,
            )
        )
        residual, _, detection_id, detection = candidates[0]
        if residual > 0.20:
            raise ToolOperationError(
                FailureCategory.PERCEPTION_UNCERTAIN,
                "aligned destination differs from the requested destination "
                f"by {residual:.3f} m",
            )
        detection.selected = True
        return detection_id, detection, residual

    def _validate_public_gripper_profile(
        self,
        profile: Mapping[str, object],
    ) -> None:
        expected = self.gripper_profile.public()
        if (
            profile.get("profile_id") != expected["profile_id"]
            or not math.isclose(
                float(profile.get("max_width_m", math.nan)),
                self.gripper_profile.max_width_m,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                float(profile.get("fingertip_depth_m", math.nan)),
                self.gripper_profile.fingertip_depth_m,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ValueError(
                "gripper_profile must exactly match the public hardware profile"
            )

    def _track_selected_detection(
        self,
        detection_id: str,
        detection: _Detection,
        full_points: np.ndarray,
        object_points: np.ndarray,
    ) -> str:
        centroid, bounds_min, bounds_max = _point_geometry(object_points)
        distractor: _Detection | None = None
        if detection.distractor_detection_id is not None:
            distractor = cast(
                _Detection,
                self.registry.get(
                    detection.distractor_detection_id,
                    expected_kind="detection",
                ),
            )
        return self.registry.put(
            "entity_track",
            _TrackedEntity(
                object_name=detection.prompt,
                centroid=centroid,
                bounds_min=bounds_min,
                bounds_max=bounds_max,
                point_count=int(len(object_points)),
                full_points=np.asarray(
                    full_points,
                    dtype=np.float64,
                ).copy(),
                object_points=np.asarray(
                    object_points,
                    dtype=np.float64,
                ).copy(),
                source_camera_name=detection.camera_name,
                appearance_rgb_mean=(
                    detection.appearance_rgb_mean.copy()
                ),
                current_centroid=centroid.copy(),
                current_bounds_min=bounds_min.copy(),
                current_bounds_max=bounds_max.copy(),
                current_point_count=int(len(object_points)),
                source_detection_id=detection_id,
                source_observation_handle=detection.observation_handle,
                identity_point_handle=detection.identity_point_handle,
                distractor_detection_id=(
                    detection.distractor_detection_id
                ),
                distractor_initial_centroid=(
                    distractor.centroid.copy()
                    if distractor is not None
                    else None
                ),
                distractor_initial_extent=(
                    (
                        distractor.bounds_max
                        - distractor.bounds_min
                    ).copy()
                    if distractor is not None
                    else None
                ),
                distractor_initial_appearance_rgb_mean=(
                    distractor.appearance_rgb_mean.copy()
                    if distractor is not None
                    else None
                ),
                instance_reference_distances_m=(
                    dict(detection.instance_reference_distances_m)
                    if detection.instance_reference_distances_m
                    is not None
                    else None
                ),
                instance_selection_margin_ratio=(
                    detection.instance_selection_margin_ratio
                ),
            ),
        )

    def plan_grasp_candidates(
        self,
        full_point_cloud_handle: str,
        object_point_cloud_handle: str,
        top_k: int = 15,
        grasp_style: str = "auto",
        tracking_handle: str | None = None,
    ) -> CapabilityResult[dict[str, object]]:
        style = _validate_grasp_style(grasp_style)
        if tracking_handle is not None:
            tracked = cast(
                _TrackedEntity,
                self.registry.get(
                    tracking_handle,
                    expected_kind="entity_track",
                ),
            )
            full_points = tracked.full_points
            object_points = tracked.object_points
            identity_preserved = True
        else:
            full_points = np.asarray(
                self.registry.get(
                    full_point_cloud_handle,
                    expected_kind="point_cloud",
                ),
                dtype=np.float64,
            )
            object_points = np.asarray(
                self.registry.get(
                    object_point_cloud_handle,
                    expected_kind="point_cloud",
                ),
                dtype=np.float64,
            )
            identity_preserved = False
        if tracking_handle is not None and style in tracked.grasp_cache:
            reachable, rejected, candidate_source = tracked.grasp_cache[style]
        else:
            reachable, rejected, candidate_source = (
                self._reachable_grasp_candidates(
                    full_points,
                    object_points,
                    grasp_style=style,
                    limit=int(top_k),
                )
            )
            if tracking_handle is not None:
                tracked.grasp_cache[style] = (
                    reachable,
                    rejected,
                    candidate_source,
                )
        selected = reachable[: int(top_k)]
        payload = {
            "poses": np.stack([item.pose for item in selected]),
            "scores": np.asarray([item.score for item in selected]),
            "joint_travel": np.asarray(
                [item.predicted_joint_travel_rad for item in selected]
            ),
            "approach_directions": np.asarray(
                [item.approach_direction_base for item in selected]
            ),
            "top_down_alignment": np.asarray(
                [item.top_down_alignment for item in selected]
            ),
            "grasp_style": style,
            "tracking_handle": tracking_handle,
            "identity_preserved": identity_preserved,
            "candidate_source": candidate_source,
        }
        handle = self.registry.put("grasp_set", payload)
        return CapabilityResult.ok(
            {
                "grasp_set_handle": handle,
                "candidate_count": len(selected),
                "best_score": selected[0].score,
                "best_top_down_alignment": selected[0].top_down_alignment,
                "grasp_style": style,
                "rejected_unreachable_count": rejected,
                "tracking_handle": tracking_handle,
                "identity_preserved": identity_preserved,
                "candidate_source": candidate_source,
            }
        )

    def get_grasp_candidate(
        self, grasp_set_handle: str, index: int = 0
    ) -> CapabilityResult[dict[str, object]]:
        payload = cast(
            Mapping[str, np.ndarray],
            self.registry.get(grasp_set_handle, expected_kind="grasp_set"),
        )
        poses = np.asarray(payload["poses"], dtype=np.float64)
        scores = np.asarray(payload["scores"], dtype=np.float64).reshape(-1)
        joint_travel = np.asarray(
            payload["joint_travel"], dtype=np.float64
        ).reshape(-1)
        approach_directions = np.asarray(
            payload["approach_directions"], dtype=np.float64
        ).reshape(-1, 3)
        top_down_alignment = np.asarray(
            payload["top_down_alignment"], dtype=np.float64
        ).reshape(-1)
        if index < 0 or index >= len(scores):
            raise ValueError(
                f"grasp candidate index {index} is outside [0, {len(scores) - 1}]"
            )
        pose = poses[index]
        return CapabilityResult.ok(
            {
                "position": [float(value) for value in pose[:3, 3]],
                "quaternion": list(_matrix_to_quaternion_wxyz(pose[:3, :3])),
                "score": float(scores[index]),
                "predicted_joint_travel_rad": float(joint_travel[index]),
                "approach_direction_base": [
                    float(value) for value in approach_directions[index]
                ],
                "top_down_alignment": float(top_down_alignment[index]),
                "grasp_style": str(payload["grasp_style"]),
            }
        )

    def solve_ik(
        self, position: Sequence[float], quaternion: Sequence[float]
    ) -> CapabilityResult[list[float]]:
        pos = _finite_vector(position, 3, "position")
        quat = _normalized_quaternion(quaternion)
        target = (
            pos
            + _quaternion_wxyz_to_matrix(quat)
            @ self._tcp_offset_local_m
        )
        return self._solve_ik_target(target, quat)

    def _solve_ik_target(
        self, position: Sequence[float], quaternion: Sequence[float]
    ) -> CapabilityResult[list[float]]:
        pos = _finite_vector(position, 3, "position")
        quat = _normalized_quaternion(quaternion)
        state = self.robot.get_robot_state()
        if state.success and state.value is not None:
            self._ik_cfg = np.asarray(
                [
                    *state.value.joint_positions_rad,
                    state.value.gripper_open_fraction * 0.04,
                ],
                dtype=np.float64,
            )
        joints = self.services.solve_ik(pos, quat, self._ik_cfg)
        if self._sensor_motion:
            from .ik_resolution import resolve_external_ik
            from .motion_tools import pose_matrix
            eef_position, eef_quaternion = self.robot.eef_pose_from_external_ik_target(pos, quat)
            current = state.value if state.success and state.value is not None else None
            resolved = resolve_external_ik(self.robot.planning_model(), joints,
                pose_matrix(eef_position, eef_quaternion),
                seed=current.joint_positions_rad if current else None,
                aperture=current.gripper_open_fraction if current else 1.,
                position_tolerance_m=self._motion_position_tolerance_m,
                orientation_tolerance_rad=self._motion_orientation_tolerance_rad)
            if resolved['configuration'] is None:
                evidence = resolved['evidence']
                return CapabilityResult.failed(FailureCategory.IK_FAILURE,
                    evidence['reason'], value=evidence)
            configuration = np.asarray(resolved['configuration'], dtype=np.float64)
            if resolved['used_fallback']:
                self._ik_cfg = configuration.copy()
                return CapabilityResult.ok(configuration[:7].tolist(),
                    'External IK reply unusable; bounded robot-model IK recovered a validated endpoint. Motion path not checked.')
        else:
            if joints is None:
                raise ToolOperationError(FailureCategory.IK_FAILURE, "PyRoKi found no IK solution")
            configuration = _finite_vector(joints, 8, "IK configuration")
        self._ik_cfg = configuration.copy()
        return CapabilityResult.ok([float(value) for value in configuration[:7]])

    def move_to_joints(self, joints: Sequence[float]) -> CapabilityResult[None]:
        values = _finite_vector(joints, 7, "joints")
        result = self.robot.move_to_joints(values)
        if result.executed:
            self._aligned_pregrasp = None
        return result

    def goto_pose(
        self,
        position: Sequence[float],
        quaternion: Sequence[float],
        z_approach: float = 0.0,
        approach_frame: str | None = None,
        position_reference: str | None = None,
        tracking_handle: str | None = None,
        scene_handle: str | None = None,
        clearance_first: bool = False,
    ) -> CapabilityResult[dict[str, object]]:
        pos = _finite_vector(position, 3, "position")
        quat = _normalized_quaternion(quaternion)
        input_position, input_quaternion = pos.copy(), quat.copy()
        if position_reference not in {None, "tool", "held_target", "end_effector"}:
            raise ValueError("position_reference must be tool, held_target or end_effector")
        if self._v15_protocol and self._active_attachment_handle and position_reference is None:
            return CapabilityResult.failed(FailureCategory.NOT_EXECUTED,
                "held_target_reference_required: declare position_reference='held_target' "
                "and tracking_handle when position denotes the held object center; use 'tool' "
                "only for an explicitly computed tool waypoint. Container centroids are not tool poses.",
                value={"reason": "held_target_reference_required"})
        if position_reference == "end_effector":
            hand_position, hand_quaternion = self.robot.external_ik_target_from_eef_pose(pos, quat)
            quat = np.asarray(hand_quaternion)
            pos = np.asarray(hand_position) - _quaternion_wxyz_to_matrix(quat) @ self._tcp_offset_local_m
        elif position_reference == "held_target":
            if tracking_handle is None or tracking_handle != self._active_attachment_handle:
                return CapabilityResult.failed(FailureCategory.PERCEPTION_UNCERTAIN,
                    "held_target_pose_unknown: use the currently verified attachment handle",
                    value={"reason": "held_target_pose_unknown"})
            tracked = cast(_TrackedEntity, self.registry.get(tracking_handle, expected_kind="entity_track"))
            if not tracked.attachment_verified or tracked.eef_to_object_transform is None:
                return CapabilityResult.failed(FailureCategory.PERCEPTION_UNCERTAIN,
                    "held_target_pose_unknown: reobserve the held target and verify its pose",
                    value={"reason": "held_target_pose_unknown"})
            # Input quaternion retains the public hand convention. Convert the
            # observed object offset through the requested EEF rotation, then
            # round-trip back to the public tool position before execution.
            _, eef_quaternion = self.robot.eef_pose_from_external_ik_target((0, 0, 0), quat)
            eef_rotation = _quaternion_wxyz_to_matrix(eef_quaternion)
            desired_eef = pos - eef_rotation @ tracked.eef_to_object_transform[:3, 3]
            hand_position, _ = self.robot.external_ik_target_from_eef_pose(desired_eef, eef_quaternion)
            pos = np.asarray(hand_position) - _quaternion_wxyz_to_matrix(quat) @ self._tcp_offset_local_m
        if not math.isfinite(float(z_approach)) or not 0 <= float(z_approach) <= 0.5:
            raise ValueError("z_approach must be finite and in [0, 0.5]")
        normalized_approach_frame = (
            self._default_approach_frame
            if approach_frame is None
            else str(approach_frame).strip().lower()
        )
        if normalized_approach_frame not in {"world", "tool"}:
            raise ValueError("approach_frame must be 'world' or 'tool'")
        rotation = _quaternion_wxyz_to_matrix(quat)
        target = pos + rotation @ self._tcp_offset_local_m
        def with_input_reference(result):
            if not self._v15_protocol or not isinstance(result.value, Mapping):
                return result
            return replace(result, value={**result.value,
                "input_pose_reference": position_reference or "tool",
                "input_position_m": input_position.tolist(),
                "input_quaternion_wxyz": input_quaternion.tolist()})
        if not isinstance(clearance_first, bool):
            raise ValueError('clearance_first must be boolean')
        if clearance_first:
            if scene_handle is not None or normalized_approach_frame != 'world':
                raise ValueError('clearance_first requires approach_frame=world and no scene_handle')
            from .clearance_motion import goto_with_clearance
            return with_input_reference(goto_with_clearance(self, pos, quat, float(z_approach)))
        if scene_handle is not None:
            if z_approach:
                raise ValueError('planned goto_pose requires z_approach=0; request each free waypoint explicitly')
            before = self._motion_outcome('target', False, '', requested_position=pos,
                controller_target=target, requested_quaternion=quat)
            if (before.get('pose_measurement') or {}).get('status') == 'unavailable':
                return with_input_reference(CapabilityResult.failed(FailureCategory.BACKEND_ERROR,
                    'current_pose_unavailable: ' + before['pose_measurement']['message'], value=before))
            distance, angle = before['position_residual_m'], before['orientation_residual_rad']
            position_accuracy = (min(self._motion_position_tolerance_m, .1 * distance)
                                 if distance > 1e-10 else self._motion_position_tolerance_m)
            orientation_accuracy = (min(self._motion_orientation_tolerance_rad, .1 * angle)
                                    if angle > 1e-10 else self._motion_orientation_tolerance_rad)
            planned = self._plan_motion({'position_m': pos.tolist(), 'quaternion_wxyz': quat.tolist()},
                                        'tool', scene_handle)
            if not planned.success:
                return with_input_reference(planned)
            moved = self.execute_motion(planned.value['plan_handle'])
            outcome = {**dict(moved.value or {}), 'planning': planned.value,
                **self._motion_outcome('target', moved.executed, moved.message,
                    requested_position=pos, controller_target=target, requested_quaternion=quat,
                    position_tolerance_m=position_accuracy, orientation_tolerance_rad=orientation_accuracy)}
            outcome['precision_reference'] = 'whole_public_planned_command'
            return with_input_reference(self._finalize_motion_outcome(moved, outcome,
                                                                      accept_timeout_if_converged=False))
        if z_approach:
            phase_executions = {}
            def run_phase(phase, requested, controller):
                from captain_harness.ports.deadline import check_deadline, current_deadline, retain_deadline_status
                first = getattr(self.robot, 'simulation_steps', None)
                try:
                    check_deadline()
                    result = self._execute_cartesian_phase(phase=phase,
                        requested_position=requested, controller_target=controller,
                        requested_quaternion=quat,
                        orientation_reference_quaternion_wxyz=orientation_reference)
                    result = retain_deadline_status(result, current_deadline())
                except Exception as exc:
                    last = getattr(self.robot, 'simulation_steps', None)
                    executed = isinstance(first, int) and isinstance(last, int) and last > first
                    if isinstance(exc, TimeoutError):
                        category = FailureCategory.TIMEOUT
                        diagnostics = {'deadline': getattr(exc, 'deadline_receipt', None),
                            'service_attempts': list(getattr(exc, 'attempts', ()))}
                    elif isinstance(exc, ToolOperationError):
                        category, diagnostics = exc.category, None
                    elif isinstance(exc, (TypeError, ValueError, np.linalg.LinAlgError)):
                        category, diagnostics = FailureCategory.INVALID_ARGUMENT, None
                    else:
                        category, diagnostics = FailureCategory.TOOL_ERROR, None
                    if executed and category is FailureCategory.NOT_EXECUTED:
                        category = FailureCategory.BACKEND_ERROR
                    result = replace(CapabilityResult.failed(category, str(exc), executed=executed),
                        diagnostics=diagnostics)
                phase_executions[phase] = deepcopy({
                    'success': result.success, 'executed': result.executed,
                    'failure_category': result.failure_category.value, 'message': result.message,
                    'value': result.value, 'diagnostics': result.diagnostics,
                    'attempt_identity': result.attempt_identity})
                return result

            def finish_phases(result, phase, requested, controller):
                executed = any(item['executed'] for item in phase_executions.values())
                value = dict(result.value) if isinstance(result.value, Mapping) else {}
                if 'converged' not in value:
                    value = {**self._motion_outcome(phase, executed, result.message,
                        requested_position=requested, controller_target=controller,
                        requested_quaternion=quat), **value}
                value.update(executed=executed, phase_executions=phase_executions,
                    completed_phases=[name for name, item in phase_executions.items() if item['success']],
                    failed_phase=phase if not result.success else None)
                category = result.failure_category
                diagnostics = dict(result.diagnostics or {})
                if executed and category is FailureCategory.NOT_EXECUTED:
                    diagnostics['original_failure_category'] = category.value
                    category = FailureCategory.BACKEND_ERROR
                return with_input_reference(replace(result, executed=executed, value=value,
                    failure_category=category, diagnostics=diagnostics or None))

            approach_delta = (
                np.array(
                    [0.0, 0.0, float(z_approach)],
                    dtype=np.float64,
                )
                if normalized_approach_frame == "world"
                else rotation
                @ np.array(
                    [0.0, 0.0, -float(z_approach)],
                    dtype=np.float64,
                )
            )
            approach_request = pos + approach_delta
            approach = approach_request + rotation @ self._tcp_offset_local_m
            reference_outcome = self._motion_outcome("approach", False, "",
                requested_position=approach_request, controller_target=approach,
                requested_quaternion=quat)
            orientation_reference = reference_outcome.get("final_eef_quaternion_wxyz")
            approached = run_phase("approach", approach_request, approach)
            if not approached.success:
                return finish_phases(approached, "approach", approach_request, approach)
            moved = run_phase("target", pos, target)
            return finish_phases(moved, "target", pos, target)
        return with_input_reference(self._execute_cartesian_phase(
            phase="target",
            requested_position=pos,
            controller_target=target,
            requested_quaternion=quat,
        ))

    def _execute_cartesian_phase(
        self,
        *,
        phase: str,
        requested_position: np.ndarray,
        controller_target: np.ndarray,
        requested_quaternion: np.ndarray,
        orientation_reference_quaternion_wxyz=None,
        max_control_steps: int | None = None,
    ) -> CapabilityResult[dict[str, object]]:
        before = self._motion_outcome(phase, False, "", requested_position=requested_position,
            controller_target=controller_target, requested_quaternion=requested_quaternion)
        if (before.get('pose_measurement') or {}).get('status') == 'unavailable':
            return CapabilityResult.failed(FailureCategory.BACKEND_ERROR,
                'current_pose_unavailable: ' + before['pose_measurement']['message'], value=before)
        solved = self._solve_ik_target(
            controller_target,
            requested_quaternion,
        )
        if not solved.success or solved.value is None:
            return cast(CapabilityResult[dict[str, object]], solved)
        if getattr(self.robot, "controller", "JOINT_POSITION") == "OSC_POSE":
            correction_position, correction_quaternion = (
                self.robot.eef_pose_from_external_ik_target(
                    controller_target,
                    requested_quaternion,
                )
            )
            reference_move = getattr(self.robot, "move_to_pose_with_orientation_reference", None)
            if max_control_steps is not None:
                cartesian_motion = self.robot.move_to_pose_bounded(correction_position, correction_quaternion,
                    max_steps=max_control_steps,
                    orientation_reference_quaternion_wxyz=orientation_reference_quaternion_wxyz)
            elif orientation_reference_quaternion_wxyz is not None and callable(reference_move):
                cartesian_motion = reference_move(correction_position, correction_quaternion,
                    orientation_reference_quaternion_wxyz=orientation_reference_quaternion_wxyz)
            else:
                cartesian_motion = self.robot.move_to_pose(
                    correction_position, correction_quaternion, position_only=False)
            if cartesian_motion.executed:
                self._aligned_pregrasp = None
            outcome = self._motion_outcome(
                phase,
                cartesian_motion.executed,
                cartesian_motion.message,
                requested_position=requested_position,
                controller_target=controller_target,
                requested_quaternion=requested_quaternion,
            )
            if isinstance(cartesian_motion.value, Mapping):
                outcome['controller_execution'] = dict(cartesian_motion.value)
            return self._finalize_motion_outcome(
                cartesian_motion,
                outcome,
                accept_timeout_if_converged=True,
            )
        joint_motion = self.robot.move_to_joints(solved.value)
        if joint_motion.executed:
            self._aligned_pregrasp = None
        outcome = self._motion_outcome(
            phase,
            joint_motion.executed,
            joint_motion.message,
            requested_position=requested_position,
            controller_target=controller_target,
            requested_quaternion=requested_quaternion,
        )
        return self._finalize_motion_outcome(
            joint_motion,
            outcome,
            accept_timeout_if_converged=False,
        )

    def _finalize_motion_outcome(
        self,
        motion: CapabilityResult[None],
        outcome: dict[str, object],
        *,
        accept_timeout_if_converged: bool,
    ) -> CapabilityResult[dict[str, object]]:
        if (outcome.get("pose_measurement") or {}).get("status") == "unavailable":
            detail = "final_pose_unavailable: " + outcome["pose_measurement"]["message"]
            return replace(motion, success=False,
                failure_category=(FailureCategory.BACKEND_ERROR if motion.success else motion.failure_category),
                message=(motion.message + "; " + detail if motion.message else detail),
                value=outcome)
        controller_execution = outcome.get("controller_execution")
        if isinstance(controller_execution, Mapping) and controller_execution.get("reached") is False:
            # A structured physical failure cannot be reclassified using the
            # coarser public pose tolerance (e.g. an unexecuted fine rotation).
            return replace(motion, success=False,
                failure_category=(FailureCategory.BACKEND_ERROR if motion.success else motion.failure_category),
                value={**outcome, "converged": False, "convergence_reason": "controller_endpoint_not_reached"})
        converged = outcome.get("converged") is True
        timeout_accepted = bool(
            accept_timeout_if_converged
            and converged
            and not motion.success
            and motion.executed
            and motion.failure_category is FailureCategory.TIMEOUT
        )
        if converged and (motion.success or timeout_accepted):
            message = (
                motion.message
                if motion.success
                else "OSC_POSE reached the calibrated public pose tolerance"
            )
            return replace(motion, success=True, failure_category=FailureCategory.NONE,
                value=outcome, message=message)
        if not converged:
            detail = (
                "Cartesian target did not converge: "
                f"position residual {outcome['position_residual_m']:.6f} m "
                f"(limit {outcome['position_tolerance_m']:.6f} m), "
                f"orientation residual "
                f"{outcome['orientation_residual_rad']:.6f} rad "
                f"(limit {outcome['orientation_tolerance_rad']:.6f} rad)"
            )
            message = (
                f"{motion.message}; {detail}"
                if motion.message
                else detail
            )
            return replace(motion, success=False,
                failure_category=motion.failure_category if not motion.success else FailureCategory.TIMEOUT,
                message=message, value=outcome)
        return replace(motion, value=outcome)

    def check_reachable(
        self,
        pose: Mapping[str, object],
    ) -> CapabilityResult[dict[str, object]]:
        if isinstance(pose, str) or "grasp_handle" in pose or "approach_direction_world" in pose:
            candidate = self._resolve_projected_grasp(pose)
            state = self.robot.get_robot_state()
            if not state.success or state.value is None:
                return CapabilityResult.failed(state.failure_category, state.message)
            if not callable(getattr(self.robot, "planning_model", None)):
                return CapabilityResult.failed(FailureCategory.NOT_EXECUTED,
                    "grasp reachability requires the robot planning model")
            return CapabilityResult.ok(self._grasp_endpoint_reachability(
                candidate, state.value.joint_positions_rad))
        position, quaternion = self._pose_mapping(pose)
        solved = self.solve_ik(position, quaternion)
        return CapabilityResult.ok(
            {
                "reachable": bool(solved.success),
                "reason": (
                    "reachable"
                    if solved.success
                    else solved.failure_category.value
                ),
                "joints": solved.value if solved.success else None,
            }
        )

    def subsample_point_cloud(
        self,
        point_cloud_handle: str,
        max_points: int = 10000,
    ) -> CapabilityResult[dict[str, object]]:
        if (
            not isinstance(max_points, int)
            or isinstance(max_points, bool)
            or not 3 <= max_points <= 100000
        ):
            raise ValueError("max_points must be an integer in [3, 100000]")
        points = np.asarray(
            self.registry.get(
                point_cloud_handle,
                expected_kind="point_cloud",
            ),
            dtype=np.float64,
        ).reshape(-1, 3)
        if len(points) > max_points:
            indices = np.linspace(
                0,
                len(points) - 1,
                num=max_points,
                dtype=np.int64,
            )
            points = points[indices]
        handle = self.registry.put("point_cloud", points.copy())
        return CapabilityResult.ok(
            {
                "point_cloud_handle": handle,
                "point_count": int(len(points)),
            }
        )

    def filter_noise(
        self,
        point_cloud_handle: str,
    ) -> CapabilityResult[dict[str, object]]:
        points = np.asarray(
            self.registry.get(
                point_cloud_handle,
                expected_kind="point_cloud",
            ),
            dtype=np.float64,
        ).reshape(-1, 3)
        filtered = _filter_point_cloud_noise(points)
        if len(filtered) < 3:
            return CapabilityResult.failed(
                FailureCategory.PERCEPTION_UNCERTAIN,
                "noise filtering left fewer than three points",
                executed=True,
            )
        handle = self.registry.put("point_cloud", filtered.copy())
        return CapabilityResult.ok(
            {
                "point_cloud_handle": handle,
                "point_count": int(len(filtered)),
                "removed_count": int(len(points) - len(filtered)),
            }
        )

    def _issue_grasp_candidate(self, candidate: Mapping[str, object]) -> dict[str, object]:
        descriptor = deepcopy(dict(candidate))
        descriptor.pop("grasp_handle", None)
        handle = self.registry.put("grasp_candidate", descriptor)
        return {**deepcopy(descriptor), "grasp_handle": handle}

    def _resolve_projected_grasp(self, candidate: Mapping[str, object] | str) -> Mapping[str, object]:
        """Resolve a native payload omitted from feedback using its issued identity.

        The policy may carry the small public descriptor across turns. A modified
        pose cannot borrow a native service payload from its original candidate.
        The episode registry, not a path into mutable latest feedback, owns it.
        """
        if isinstance(candidate, str):
            candidate = {'grasp_handle':candidate}
        if "grasp_handle" in candidate:
            handle = candidate["grasp_handle"]
            issued = self.registry.get(handle, expected_kind="grasp_candidate")
            for key, value in candidate.items():
                if key == "grasp_handle":
                    continue
                # Decision feedback may omit large payloads. The immutable
                # registry supplies them; an explicit replacement must match.
                if isinstance(value, Mapping) and value.get("omitted_from_feedback") is True:
                    if key not in issued:
                        raise ValueError(f"grasp_reference_mismatch: unknown field {key}")
                    continue
                if key not in issued or value != issued[key]:
                    raise ValueError(f"grasp_reference_mismatch: {key} differs; use refine_grasp for a new pose")
            return {**deepcopy(issued), "grasp_handle": handle}
        payload = candidate.get("backend_candidate")
        if not isinstance(payload, Mapping) or payload.get("omitted_from_feedback") is not True:
            return candidate
        tracked = cast(_TrackedEntity, self.registry.get(
            candidate.get("tracking_handle"), expected_kind="entity_track"))
        issued = tracked.issued_grasp_candidates.get(candidate.get("candidate_index"))
        if issued is None:
            raise ValueError("native_grasp_payload_unavailable: candidate index was not issued for this tracking handle")
        for key in ("backend", "tracking_handle", "detection_id", "predicted_grasp_width_m"):
            if candidate.get(key) != issued.get(key):
                raise ValueError(f"native_grasp_payload_mismatch: {key} differs from the issued candidate")
        for key in ("position_m", "quaternion_wxyz", "approach_direction_world"):
            supplied = np.asarray(candidate.get(key), dtype=float)
            original = np.asarray(issued.get(key), dtype=float)
            if supplied.shape != original.shape or not np.allclose(supplied, original, rtol=0, atol=1e-9):
                raise ValueError(f"native_grasp_payload_mismatch: {key} differs from the issued candidate")
        if not isinstance(issued.get("backend_candidate"), Mapping):
            raise ValueError("native_grasp_payload_unavailable: issued candidate has no native backend payload")
        return {**candidate, "backend_candidate": deepcopy(issued["backend_candidate"])}

    def align_pregrasp(
        self,
        candidate: Mapping[str, object] | str | None = None,
        scene_handle: str | None = None,
        *,
        grasp_handle: str | None = None,
    ) -> CapabilityResult[dict[str, object]]:
        if (candidate is None) == (grasp_handle is None):
            raise ValueError('provide exactly one of candidate or grasp_handle')
        if grasp_handle is not None:
            candidate = grasp_handle
        candidate = self._resolve_projected_grasp(candidate)
        # A new attempt supersedes the previous target, including early failures.
        previous_alignment = self._aligned_pregrasp
        if scene_handle is None:
            self._aligned_pregrasp = None
        if self._v15_protocol and candidate.get("width_feasible") is False:
            return CapabilityResult.failed(
                FailureCategory.CONTROLLER_REJECTED,
                "grasp_width_infeasible: re-localize the object from a fresh view or choose "
                "a different closing axis. Low convergence risk does not imply the jaws fit.",
                value={"reason": "grasp_width_infeasible",
                       "predicted_grasp_width_m": candidate.get("predicted_grasp_width_m"),
                       "gripper_max_width_m": self.gripper_profile.max_width_m},
            )
        (
            _position,
            _quaternion,
            approach,
            eef_position,
            eef_quaternion,
        ) = self._candidate_execution_pose(candidate)
        tracking_handle = candidate.get("tracking_handle")
        detection_id = candidate.get("detection_id")
        candidate_context: dict[str, object] = {
            "tracking_handle": tracking_handle,
            "detection_id": detection_id,
            "candidate_index": candidate.get("candidate_index"),
            "convergence_risk": candidate.get("convergence_risk", "high"),
        }
        if self._v15_protocol:
            source_observation = None
            if isinstance(tracking_handle, str):
                tracked = cast(_TrackedEntity, self.registry.get(tracking_handle, expected_kind="entity_track"))
                source_observation = tracked.source_observation_handle
                candidate_context["target_contact_evidence"] = self._grasp_contact_evidence(
                    _transform_from_pose(_position, _quaternion), tracked.object_points)
            candidate_context["candidate_evidence"] = {
                "position_m": _position.tolist(),
                "quaternion_wxyz": _quaternion.tolist(),
                "approach_direction_world": approach.tolist(),
                "execution_eef_position_m": list(eef_position),
                "execution_eef_quaternion_wxyz": list(eef_quaternion),
                "source_observation_handle": source_observation,
                "motion_path": "not_checked",
                "basis": "supplied pose and selected target cloud before this alignment; not a fresh observation",
            }
        method = getattr(self.robot, "align_pregrasp", None)
        if not callable(method):
            return CapabilityResult.failed(
                FailureCategory.CONTROLLER_REJECTED,
                "align_pregrasp is unavailable on the active robot backend",
                value=candidate_context,
            )
        if scene_handle is not None:
            try:
                result = self._planned_alignment(candidate, scene_handle)
            except Exception:
                self._aligned_pregrasp = previous_alignment
                raise
            if not result.executed:
                self._aligned_pregrasp = previous_alignment
            candidate_context['candidate_evidence']['motion_path'] = (
                'planned_observed_surfaces' if result.success else 'planning_or_execution_failed')
        else:
            result = method(eef_position, eef_quaternion, approach,
                approach_distance_m=self._grasp_approach_distance(candidate))
        payload = {**(dict(result.value) if isinstance(result.value, Mapping) else {}),
                   **candidate_context}
        if not result.success:
            return replace(result, value=payload)
        if (not isinstance(result.value, Mapping)
                or result.value.get("converged") is not True
                or result.value.get("phase_reached") != "pregrasp"
                or result.value.get("failed_phase") is not None):
            return CapabilityResult.failed(
                (FailureCategory.CONTROLLER_REJECTED
                 if isinstance(result.value, Mapping) and result.value.get("converged") is False
                 else FailureCategory.BACKEND_ERROR),
                "alignment_not_confirmed: backend did not confirm reaching pregrasp; inspect phase feedback before retrying",
                executed=result.executed,
                value={**payload, "reason": "alignment_not_confirmed"},
            )
        self._aligned_pregrasp = _AlignedPregrasp(
            candidate=dict(candidate),
            eef_position=tuple(float(value) for value in eef_position),
            eef_quaternion=tuple(
                float(value) for value in eef_quaternion
            ),
            tracking_handle=(
                tracking_handle
                if isinstance(tracking_handle, str)
                else None
            ),
            detection_id=(
                detection_id if isinstance(detection_id, str) else None
            ),
            pregrasp_position=(
                tuple(float(x) for x in _finite_vector(result.value["pregrasp_position_m"], 3, "pregrasp_position_m"))
                if result.value.get("pregrasp_position_m") is not None else None
            ),
        )
        return replace(result, value=payload)

    def _pre_descent_centering(
        self,
        aligned: _AlignedPregrasp,
    ) -> tuple[np.ndarray, dict[str, object]]:
        from .grasp_motion_correction import measure_pre_descent_correction
        return measure_pre_descent_correction(self, aligned)

    def descend_and_close(
        self,
        max_contact_force: float | None = None,
        position_tolerance_m: float | None = None,
    ) -> CapabilityResult[dict[str, object]]:
        from captain_harness.adapters.robotics.contact_precision import contact_position_tolerance
        tolerance = contact_position_tolerance(position_tolerance_m)
        aligned = self._aligned_pregrasp
        if aligned is None:
            return CapabilityResult.failed(
                FailureCategory.NOT_EXECUTED,
                "pregrasp_alignment_required: call "
                "align_pregrasp(candidate) successfully before "
                "descend_and_close",
                value={
                    "reason": "pregrasp_alignment_required",
                    "required_call": "align_pregrasp(candidate)",
                },
            )
        method = getattr(self.robot, "descend_and_close", None)
        if not callable(method):
            self._aligned_pregrasp = None
            return CapabilityResult.failed(
                FailureCategory.CONTROLLER_REJECTED,
                "descend_and_close is unavailable on the active robot backend",
            )
        centering_delta, centering_diagnostics = (
            self._pre_descent_centering(aligned)
        )
        if centering_diagnostics.get('decision') == 'reacquire':
            self._aligned_pregrasp = None
            self._pending_attachment_handle = None
            return CapabilityResult.failed(FailureCategory.PERCEPTION_UNCERTAIN,
                'fresh_grasp_required: ' + str(centering_diagnostics['reason']),
                value={'closure_executed':False, 'pre_descent_centering':centering_diagnostics,
                       'tracking_handle':aligned.tracking_handle, 'detection_id':aligned.detection_id})
        descent_position = tuple(
            float(value)
            for value in (
                np.asarray(aligned.eef_position, dtype=np.float64)
                + centering_delta
            )
        )
        result = method(
            descent_position,
            aligned.eef_quaternion,
            max_contact_force=max_contact_force,
            **({"position_tolerance_m": tolerance} if position_tolerance_m is not None else {}),
        )
        if result.executed and isinstance(result.value, Mapping):
            from .contact_geometry import measured_grasp_geometry
            result = replace(result, value={**result.value,
                "contact_geometry_audit": measured_grasp_geometry(
                    self, aligned, descent_position, centering_delta)})
        self._aligned_pregrasp = None
        closure_skipped = isinstance(result.value, Mapping) and result.value.get("closure_executed") is False
        self._last_descent_evidence = dict(result.value) if isinstance(result.value, Mapping) else None
        if closure_skipped:
            self._pending_attachment_handle = None
            self._latest_lift_evidence = None
        if result.executed and not closure_skipped:
            self._attachment_attempt_id += 1
            self._pending_attachment_handle = aligned.tracking_handle
            self._latest_lift_evidence = None
        if not result.success or not isinstance(result.value, Mapping):
            if isinstance(result.value, Mapping):
                return replace(result, value={**result.value,
                    "tracking_handle": aligned.tracking_handle,
                    "detection_id": aligned.detection_id,
                    "pre_descent_centering": centering_diagnostics})
            return result
        if isinstance(aligned.tracking_handle, str):
            detected = result.value.get("object_detected")
            if detected is False:
                try:
                    tracked = cast(
                        _TrackedEntity,
                        self.registry.get(
                            aligned.tracking_handle,
                            expected_kind="entity_track",
                        ),
                    )
                except ArtifactHandleError:
                    tracked = None
                if tracked is not None:
                    reference = (
                        tracked.current_centroid
                        if tracked.current_centroid is not None
                        else tracked.centroid
                    )
                    self._last_empty_close = {
                        "backend": str(
                            aligned.candidate.get("backend")
                        ),
                        "object_name": tracked.object_name,
                        "source_detection_id": tracked.source_detection_id,
                        "appearance_rgb_mean": tracked.appearance_rgb_mean,
                        "extent": tracked.bounds_max - tracked.bounds_min,
                        "centroid": np.asarray(
                            reference,
                            dtype=np.float64,
                        ).reshape(3),
                    }
            elif detected is True:
                self._last_empty_close = None
        self._record_selected_grasp_candidate(
            aligned.tracking_handle,
            aligned.candidate,
        )
        return replace(result, value={
                **dict(result.value),
                "tracking_handle": aligned.tracking_handle,
                "detection_id": aligned.detection_id,
                "pre_descent_centering": centering_diagnostics,
                **({"target_attachment_status": "contradicted" if result.value.get("object_detected") is False else "insufficient",
                    "evidence_limitation": "Closure and aperture do not establish target identity or secure attachment; compare target motion before transport."}
                   if self._v15_protocol else {}),
                "candidate_quality_flags": list(
                    aligned.candidate.get("quality_flags", [])
                ),
                "candidate_width_known": aligned.candidate.get(
                    "width_known"
                ),
                "candidate_width_feasible": aligned.candidate.get(
                    "width_feasible"
                ),
                "candidate_ik_feasible": aligned.candidate.get(
                    "ik_feasible"
                ),
                "candidate_convergence_risk": aligned.candidate.get(
                    "convergence_risk", "high"
                ),
            })

    def approach_and_close(
        self,
        candidate: Mapping[str, object],
        max_contact_force: float | None = None,
    ) -> CapabilityResult[dict[str, object]]:
        self._aligned_pregrasp = None
        if (
            not self._dehardcoded
            and candidate.get("width_feasible") is not True
        ):
            return CapabilityResult.failed(
                FailureCategory.SAFETY_REJECTED,
                "grasp candidate is not width feasible",
            )
        if (
            not self._dehardcoded
            and candidate.get("reachable") is not True
        ):
            return CapabilityResult.failed(
                FailureCategory.IK_FAILURE,
                "grasp candidate is not reachable",
            )
        (
            _position,
            _quaternion,
            approach,
            eef_position,
            eef_quaternion,
        ) = self._candidate_execution_pose(candidate)
        result = self.robot.approach_and_close(
            eef_position,
            eef_quaternion,
            approach,
            max_contact_force=max_contact_force,
        )
        tracking_handle = candidate.get("tracking_handle")
        if result.executed:
            self._attachment_attempt_id += 1
            self._pending_attachment_handle = (
                tracking_handle
                if isinstance(tracking_handle, str)
                else None
            )
            self._latest_lift_evidence = None
        if not result.success or not isinstance(result.value, Mapping):
            return result
        self._record_selected_grasp_candidate(
            tracking_handle
            if isinstance(tracking_handle, str)
            else None,
            candidate,
        )
        return CapabilityResult.ok(
            {
                **dict(result.value),
                "tracking_handle": tracking_handle,
                "detection_id": candidate.get("detection_id"),
                **(
                    {
                        "candidate_quality_flags": list(
                            candidate.get("quality_flags", [])
                        ),
                        "candidate_width_known": candidate.get(
                            "width_known"
                        ),
                        "candidate_width_feasible": candidate.get(
                            "width_feasible"
                        ),
                        "candidate_ik_feasible": candidate.get(
                            "ik_feasible"
                        ),
                    }
                    if self._dehardcoded
                    else {}
                ),
            },
            result.message,
        )

    def _candidate_execution_pose(
        self,
        candidate: Mapping[str, object],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Sequence[float],
        Sequence[float],
    ]:
        candidate = self._resolve_projected_grasp(candidate)
        position, quaternion = self._pose_mapping(candidate)
        approach = _finite_vector(
            cast(
                Sequence[float],
                candidate.get("approach_direction_world"),
            ),
            3,
            "approach_direction_world",
        )
        rotation = _quaternion_wxyz_to_matrix(quaternion)
        controller_target = position + rotation @ self._tcp_offset_local_m
        eef_position, eef_quaternion = (
            self.robot.eef_pose_from_external_ik_target(
                controller_target,
                quaternion,
            )
        )
        return (
            position,
            quaternion,
            approach,
            eef_position,
            eef_quaternion,
        )

    def _record_selected_grasp_candidate(
        self,
        tracking_handle: str | None,
        candidate: Mapping[str, object],
    ) -> None:
        if tracking_handle is None:
            return
        try:
            tracked = cast(
                _TrackedEntity,
                self.registry.get(
                    tracking_handle,
                    expected_kind="entity_track",
                ),
            )
        except ArtifactHandleError:
            return
        backend_candidate = candidate.get("backend_candidate")
        tracked.selected_grasp_descriptor = deepcopy(dict(candidate))
        tracked.selected_grasp_candidate = (
            dict(backend_candidate)
            if isinstance(backend_candidate, Mapping)
            else None
        )

    def servo_to(
        self,
        target_pose: Mapping[str, object],
        camera: str,
        max_duration_s: float,
        tolerance_m: float,
        position_reference: str = "tool",
    ) -> CapabilityResult[dict[str, object]]:
        if self._v15_protocol and self._active_attachment_handle:
            return CapabilityResult.failed(FailureCategory.NOT_EXECUTED,
                "held_target_reference_required: use goto_pose with an explicit "
                "position_reference and the verified tracking_handle for held-object transport",
                value={"reason": "held_target_reference_required"})
        if camera not in {"agentview", "robot0_eye_in_hand"}:
            raise ValueError(f"unknown camera: {camera!r}")
        if position_reference not in {"tool", "end_effector"}:
            raise ValueError("position_reference must be tool or end_effector")
        position, quaternion = self._pose_mapping(target_pose)
        eef_position, eef_quaternion = position, quaternion
        if position_reference == "tool":
            hand_position = position + _quaternion_wxyz_to_matrix(quaternion) @ self._tcp_offset_local_m
            eef_position, eef_quaternion = self.robot.eef_pose_from_external_ik_target(
                hand_position, quaternion)
        result = self.robot.servo_to(
            eef_position,
            eef_quaternion,
            max_duration_s=float(max_duration_s),
            tolerance_m=float(tolerance_m),
        )
        if result.executed:
            self._aligned_pregrasp = None
        if isinstance(result.value, Mapping):
            return replace(result, value={**result.value,
                "input_pose_reference": position_reference,
                "input_position_m": position.tolist(),
                "input_quaternion_wxyz": quaternion.tolist(),
                "requested_eef_position_m": list(eef_position),
                "requested_eef_quaternion_wxyz": list(eef_quaternion)})
        return result

    def lift_probe(
        self,
        height: float,
    ) -> CapabilityResult[dict[str, object]]:
        if not math.isfinite(float(height)) or not 0.01 <= float(height) <= 0.35:
            raise ValueError("height must be finite and in [0.01, 0.35]")
        state = self.robot.get_robot_state()
        if not state.success or state.value is None:
            return replace(state, value=None)
        start = np.asarray(
            state.value.end_effector_position_m,
            dtype=np.float64,
        )
        target = start + np.array([0.0, 0.0, float(height)])
        if self._sensor_motion and self.robot.controller == 'OSC_POSE':
            moved = self.robot.move_to_pose_monitored(target, state.value.end_effector_quaternion_wxyz)
        else:
            moved = self.robot.move_to_pose(target, state.value.end_effector_quaternion_wxyz,
                                            position_only=False)
        if moved.executed:
            self._aligned_pregrasp = None
        final = self.robot.get_robot_state()
        final_position = (
            np.asarray(
                final.value.end_effector_position_m,
                dtype=np.float64,
            )
            if final.success and final.value is not None
            else start
        )
        gripper = self.gripper_state()
        actual_vertical_displacement = float(
            final_position[2] - start[2]
        )
        position_residual = float(
            np.linalg.norm(final_position - target)
        )
        object_detected = (
            gripper.value.get("object_detected")
            if gripper.success and isinstance(gripper.value, Mapping)
            else None
        )
        minimum_confirmed_displacement = 0.8 * float(height)
        maximum_contradicted_displacement = 0.25 * float(height)
        supports_attachment = bool(
            object_detected is True
            and actual_vertical_displacement
            >= minimum_confirmed_displacement
        )
        contradicts_attachment = bool(
            object_detected is False
            or (not self._v15_protocol and actual_vertical_displacement
                <= maximum_contradicted_displacement)
        )
        associated_tracking_handle = self._pending_attachment_handle
        attachment_attempt_id = (
            self._attachment_attempt_id
            if associated_tracking_handle is not None
            else None
        )
        self._latest_lift_evidence = _LiftProbeEvidence(
            tracking_handle=associated_tracking_handle,
            attachment_attempt_id=attachment_attempt_id,
            requested_height_m=float(height),
            actual_vertical_displacement_m=(
                actual_vertical_displacement
            ),
            position_residual_m=position_residual,
            object_detected=(
                object_detected
                if isinstance(object_detected, bool)
                else None
            ),
            supports_attachment=supports_attachment,
            contradicts_attachment=contradicts_attachment,
            execution={
                "success": moved.success, "executed": moved.executed,
                "failure_category": moved.failure_category.value,
                "message": moved.message,
                **({key: deepcopy(moved.value[key]) for key in (
                    "stopped_reason", "reached", "control_steps", "elapsed_s",
                    "observed_force_n", "max_observed_force_n", "force_limit_n",
                    "force_available", "force_guard", "command_profile",
                ) if key in moved.value} if isinstance(moved.value, Mapping) else {}),
                "diagnostics": deepcopy(moved.diagnostics),
            },
        )
        payload = {
            "requested_height_m": float(height),
            "actual_vertical_displacement_m": actual_vertical_displacement,
            "position_residual_m": position_residual,
            **({'controller_execution': dict(moved.value)} if isinstance(moved.value, Mapping) else {}),
            "gripper_open_fraction": (
                gripper.value.get("open_fraction")
                if gripper.success and isinstance(gripper.value, Mapping)
                else None
            ),
            "object_detected": object_detected,
            "associated_tracking_handle": associated_tracking_handle,
            "attachment_attempt_id": attachment_attempt_id,
            "supports_attachment": supports_attachment,
            "contradicts_attachment": contradicts_attachment,
            **({"target_attachment_status": "contradicted" if object_detected is False else "insufficient",
                "evidence_limitation": "supports_attachment describes EEF lift plus aperture only; it does not measure target lift. Confirm the same target moved with the hand using fresh visual evidence."}
               if self._v15_protocol else {}),
        }
        return replace(moved, value=payload)

    def wrist_alignment(
        self,
        observation_handle: str,
        candidate: Mapping[str, object],
    ) -> CapabilityResult[dict[str, object]]:
        observation = cast(
            _CapturedObservation,
            self.registry.get(
                observation_handle,
                expected_kind="observation",
            ),
        )
        camera = self._camera_by_name(
            observation,
            "robot0_eye_in_hand",
        )
        position, _ = self._pose_mapping(candidate)
        pixel = _project_base_point_to_pixel(
            position,
            camera.intrinsics,
            camera.extrinsics,
            camera.image.shape[:2],
        )
        if pixel is None:
            return CapabilityResult.failed(
                FailureCategory.PERCEPTION_UNCERTAIN,
                "candidate is outside the wrist camera view",
                executed=True,
            )
        height, width = camera.image.shape[:2]
        error_px = np.asarray(
            [pixel[0] - (width - 1) * 0.5, pixel[1] - (height - 1) * 0.5],
            dtype=np.float64,
        )
        normalized = np.asarray(
            [
                error_px[0] / max(1, width - 1),
                error_px[1] / max(1, height - 1),
            ],
            dtype=np.float64,
        )
        camera_delta = np.array(
            [normalized[0] * 0.04, normalized[1] * 0.04, 0.0],
            dtype=np.float64,
        )
        base_delta = camera.extrinsics[:3, :3] @ camera_delta
        return CapabilityResult.ok(
            {
                "pixel_error": [
                    float(value) for value in error_px
                ],
                "normalized_error": [
                    float(value) for value in normalized
                ],
                "delta_m": [
                    float(value) for value in base_delta
                ],
                "centered": bool(np.linalg.norm(error_px) <= 12.0),
            }
        )

    def pick(
        self,
        candidate: Mapping[str, object],
    ) -> CapabilityResult[dict[str, object]]:
        opened = self.open_gripper()
        if not opened.success:
            return CapabilityResult.failed(
                opened.failure_category,
                opened.message,
                executed=opened.executed,
            )
        closed = self.approach_and_close(candidate)
        if not closed.success or not isinstance(closed.value, Mapping):
            return closed
        if closed.value.get("object_detected") is not True:
            return CapabilityResult.ok(
                {
                    "picked": False,
                    "object_detected": False,
                    "tracking_handle": candidate.get("tracking_handle"),
                    "reason": "closed_on_nothing",
                }
            )
        lifted = self.lift_probe(0.10)
        state = self.gripper_state()
        object_detected = bool(
            state.success
            and isinstance(state.value, Mapping)
            and state.value.get("object_detected") is True
        )
        tracking_handle = candidate.get("tracking_handle")
        if object_detected and isinstance(tracking_handle, str):
            tracked = cast(
                _TrackedEntity,
                self.registry.get(
                    tracking_handle,
                    expected_kind="entity_track",
                ),
            )
            robot_state = self.robot.get_robot_state()
            if robot_state.success and robot_state.value is not None:
                world_from_eef = _transform_from_pose(
                    robot_state.value.end_effector_position_m,
                    robot_state.value.end_effector_quaternion_wxyz,
                )
                world_from_object = np.eye(4, dtype=np.float64)
                world_from_object[:3, 3] = (
                    tracked.centroid
                    + np.array([0.0, 0.0, 0.10], dtype=np.float64)
                )
                tracked.eef_to_object_transform = (
                    np.linalg.inv(world_from_eef) @ world_from_object
                )
                tracked.attachment_verified = True
                self._active_attachment_handle = tracking_handle
        return CapabilityResult.ok(
            {
                "picked": bool(lifted.success and object_detected),
                "object_detected": object_detected,
                "tracking_handle": tracking_handle,
                "lift": (
                    dict(lifted.value)
                    if isinstance(lifted.value, Mapping)
                    else None
                ),
            }
        )

    def place(
        self,
        destination_detection_id: str,
    ) -> CapabilityResult[dict[str, object]]:
        if self._active_attachment_handle is None:
            return CapabilityResult.failed(
                FailureCategory.SAFETY_REJECTED,
                "place requires a verified active attachment",
            )
        destination = self._require_selected_detection(
            destination_detection_id
        )
        tracked = cast(
            _TrackedEntity,
            self.registry.get(
                self._active_attachment_handle,
                expected_kind="entity_track",
            ),
        )
        if (
            not tracked.attachment_verified
            or tracked.eef_to_object_transform is None
        ):
            return CapabilityResult.failed(
                FailureCategory.SAFETY_REJECTED,
                "place requires a verified attachment transform",
            )
        state = self.robot.get_robot_state()
        if not state.success or state.value is None:
            return CapabilityResult.failed(
                state.failure_category,
                state.message,
                executed=state.executed,
            )
        world_from_eef = _transform_from_pose(
            state.value.end_effector_position_m,
            state.value.end_effector_quaternion_wxyz,
        )
        estimated_target = (
            world_from_eef @ tracked.eef_to_object_transform
        )[:3, 3]
        target_extent = tracked.bounds_max - tracked.bounds_min
        desired_target = destination.centroid.copy()
        desired_target[2] = (
            destination.bounds_max[2]
            + max(0.005, float(target_extent[2]) * 0.5)
        )
        delta = desired_target - estimated_target
        current_eef = np.asarray(
            state.value.end_effector_position_m,
            dtype=np.float64,
        )
        transport = current_eef + delta + np.array([0.0, 0.0, 0.05])
        final = current_eef + delta
        first = self.robot.move_to_pose(
            transport,
            state.value.end_effector_quaternion_wxyz,
            position_only=False,
        )
        if not first.success:
            return CapabilityResult.failed(
                first.failure_category,
                first.message,
                executed=first.executed,
            )
        second = self.robot.move_to_pose(
            final,
            state.value.end_effector_quaternion_wxyz,
            position_only=False,
        )
        if not second.success:
            return CapabilityResult.failed(
                second.failure_category,
                second.message,
                executed=True,
            )
        released = self.open_gripper()
        if released.success:
            tracked.attachment_verified = False
            self._active_attachment_handle = None
        return CapabilityResult.ok(
            {
                "placed": released.success,
                "released": released.success,
                "destination_detection_id": destination_detection_id,
            }
        )

    @staticmethod
    def _pose_mapping(
        pose: Mapping[str, object],
    ) -> tuple[np.ndarray, np.ndarray]:
        position_value = (
            pose.get("position_m")
            if "position_m" in pose
            else pose.get("position")
        )
        quaternion_value = (
            pose.get("quaternion_wxyz")
            if "quaternion_wxyz" in pose
            else pose.get("quaternion")
        )
        return (
            _finite_vector(
                cast(Sequence[float], position_value),
                3,
                "position",
            ),
            _normalized_quaternion(
                cast(Sequence[float], quaternion_value)
            ),
        )

    def hold_position(self, duration_s: float = 0.5) -> CapabilityResult[dict[str, object]]:
        aligned = self._aligned_pregrasp
        alignment_feedback = {
            "available": False,
            "invalidated": self._aligned_pregrasp is not None,
            "reason": "hold_position_requested",
            "required_before_descend_and_close": "align_pregrasp",
        }
        method = getattr(self.robot, "hold_position", None)
        if method is None:
            alignment_feedback.update(available=aligned is not None, invalidated=False,
                reason='wait_not_executed', required_before_descend_and_close=None if aligned else 'align_pregrasp')
            return CapabilityResult.failed(FailureCategory.NOT_EXECUTED,
                "time advance is unavailable on the active backend",
                value={"pregrasp_alignment": alignment_feedback})
        result = method(duration_s)
        value = dict(result.value) if isinstance(result.value, Mapping) else {}
        if result.executed:
            self._aligned_pregrasp = None
        else:
            alignment_feedback.update(available=aligned is not None, invalidated=False,
                reason='wait_not_executed', required_before_descend_and_close=None if aligned else 'align_pregrasp')
        check_pose = getattr(self.robot, "pregrasp_pose_status", None)
        if (aligned is not None and aligned.pregrasp_position is not None
                and result.success and result.executed and callable(check_pose)):
            try:
                check = check_pose(aligned.pregrasp_position, aligned.eef_quaternion)
                if isinstance(check.value, Mapping):
                    alignment_feedback["pose_check"] = dict(check.value)
                if check.success and isinstance(check.value, Mapping) and check.value.get("valid") is True:
                    # Validate against the planned pregrasp, not just hold-start
                    # drift: settling can move closer to the original goal.
                    self._aligned_pregrasp = aligned
                    alignment_feedback.update(available=True, invalidated=False,
                        reason="pregrasp_pose_revalidated", required_before_descend_and_close=None)
            except Exception:
                # Preserve the physical hold result; absent pose evidence never
                # grants permission to use the old approach target.
                pass
        value["pregrasp_alignment"] = alignment_feedback
        return replace(result, value=value)

    def open_gripper(self) -> CapabilityResult[None]:
        result = self.robot.open_gripper()
        if result.executed:
            # Even an opening that times out can change contact. The previous
            # rigid attachment transform is no longer verified.
            for handle in {self._active_attachment_handle, self._pending_attachment_handle} - {None}:
                try:
                    tracked = self.registry.get(handle, expected_kind="entity_track")
                except ArtifactHandleError:
                    continue
                tracked.attachment_verified = False
                tracked.eef_to_object_transform = None
                tracked.last_completion_observation_handle = None
                tracked.last_completion_centroid = None
                tracked.last_completion_time_s = None
                tracked.last_completion_released = False
            self._active_attachment_handle = None
            self._pending_attachment_handle = None
            self._latest_lift_evidence = None
        return result

    def close_gripper(self, tracking_handle: str | None = None) -> CapabilityResult:
        if tracking_handle is not None:
            # Explicit issued identity only, never inferred from jaw aperture.
            from .attempt_identity import _track_reference
            _track_reference(self, tracking_handle)
        result = self.robot.close_gripper()
        if result.executed:
            self._aligned_pregrasp = None
        if not self._v15_protocol:
            return result
        if result.executed:
            self._attachment_attempt_id += 1
            self._latest_lift_evidence = None
            self._pending_attachment_handle = tracking_handle
            self._active_attachment_handle = None
        hand = self._robot_gripper_state()
        measured = hand.value if hand.success and isinstance(hand.value, Mapping) else {}
        prior = self._last_descent_evidence
        payload = {
            "closure_executed": result.executed,
            "closure_converged": result.success,
            "gripper_open_fraction": measured.get("open_fraction"),
            "object_detected": measured.get("object_detected"),
            "force_n": measured.get("force_n"),
            "attachment_attempt_id": self._attachment_attempt_id,
            "associated_tracking_handle": tracking_handle if result.executed else None,
            "target_attachment_status": "contradicted" if measured.get("object_detected") is False else "insufficient",
            "previous_descent": prior,
            **({"closure_force_monitor": deepcopy(result.diagnostics["closure_force_monitor"])}
               if isinstance(result.diagnostics, Mapping) and "closure_force_monitor" in result.diagnostics else {}),
            "evidence_limitation": "Direct close changes the jaws only; it does not finish an obstructed descent. "
                "No target identity is established by aperture. Compare current wrist/target views with "
                "the failed descent before deciding whether to retreat, re-localize or test target motion.",
        }
        return replace(result, value=payload)

    def home(self) -> CapabilityResult[None]:
        result = self.robot.home()
        if result.executed:
            self._aligned_pregrasp = None
        return result

    def get_object_pose(
        self, object_name: str, use_multiview: bool = True
    ) -> CapabilityResult[dict[str, object]]:
        _, object_points = self._object_clouds(object_name, use_multiview)
        center, rotation = _estimate_pose(object_points)
        return CapabilityResult.ok(
            {
                "position": [float(value) for value in center],
                "quaternion": list(_matrix_to_quaternion_wxyz(rotation)),
            }
        )

    def sample_grasp_pose(
        self,
        object_name: str,
        use_multiview: bool = True,
        grasp_style: str = "auto",
        tracking_handle: str | None = None,
        candidate_index: int = 0,
    ) -> CapabilityResult[dict[str, object]]:
        style = _validate_grasp_style(grasp_style)
        if (
            not isinstance(candidate_index, int)
            or isinstance(candidate_index, bool)
            or candidate_index < 0
        ):
            raise ValueError("candidate_index must be a non-negative integer")
        if tracking_handle is None:
            full_points, object_points = self._object_clouds(
                object_name,
                use_multiview,
            )
            identity_preserved = False
        else:
            tracked = cast(
                _TrackedEntity,
                self.registry.get(
                    tracking_handle,
                    expected_kind="entity_track",
                ),
            )
            full_points = tracked.full_points
            object_points = tracked.object_points
            object_name = tracked.object_name
            identity_preserved = True
        if tracking_handle is not None and style in tracked.grasp_cache:
            reachable, rejected, candidate_source = tracked.grasp_cache[style]
        else:
            reachable, rejected, candidate_source = (
                self._reachable_grasp_candidates(
                    full_points,
                    object_points,
                    grasp_style=style,
                    limit=candidate_index + 1,
                )
            )
            if tracking_handle is not None:
                tracked.grasp_cache[style] = (
                    reachable,
                    rejected,
                    candidate_source,
                )
        if candidate_index >= len(reachable):
            return CapabilityResult.failed(
                FailureCategory.INVALID_ARGUMENT,
                f"candidate_index {candidate_index} is outside "
                f"[0, {len(reachable) - 1}]; inspect candidate_count and "
                "candidates from the prior successful result instead of "
                "guessing another index",
                value={
                    "requested_candidate_index": candidate_index,
                    "candidate_count": len(reachable),
                    "available_candidate_indices": list(
                        range(min(len(reachable), 5))
                    ),
                    "tracking_handle": tracking_handle,
                    "grasp_style": style,
                },
            )
        selected = reachable[candidate_index]
        pose = selected.pose
        public_candidates = [
            {
                "candidate_index": index,
                "position": [
                    float(value) for value in item.pose[:3, 3]
                ],
                "quaternion": list(
                    _matrix_to_quaternion_wxyz(item.pose[:3, :3])
                ),
                "score": item.score,
                "predicted_joint_travel_rad": (
                    item.predicted_joint_travel_rad
                ),
                "approach_direction_base": list(
                    item.approach_direction_base
                ),
                "top_down_alignment": item.top_down_alignment,
            }
            for index, item in enumerate(reachable[:5])
        ]
        return CapabilityResult.ok(
            {
                "position": [float(value) for value in pose[:3, 3]],
                "quaternion": list(_matrix_to_quaternion_wxyz(pose[:3, :3])),
                "score": selected.score,
                "predicted_joint_travel_rad": (
                    selected.predicted_joint_travel_rad
                ),
                "approach_direction_base": list(
                    selected.approach_direction_base
                ),
                "top_down_alignment": selected.top_down_alignment,
                "grasp_style": style,
                "rejected_unreachable_count": rejected,
                "object_name": object_name,
                "tracking_handle": tracking_handle,
                "identity_preserved": identity_preserved,
                "candidate_index": candidate_index,
                "candidate_count": len(reachable),
                "candidates": public_candidates,
                "candidate_source": candidate_source,
            }
        )

    def _reachable_grasp_candidates(
        self,
        full_points: np.ndarray,
        object_points: np.ndarray,
        *,
        grasp_style: str,
        limit: int,
    ) -> tuple[list[_ReachableGrasp], int, str]:
        poses, scores = _validate_grasps(
            *self.services.plan_grasps(
                full_points,
                object_points,
            )
        )
        poses = _apply_capx_grasp_convention(
            poses,
            rotate_tool=True,
        )
        reachable, rejected = self._filter_reachable_grasps(
            poses,
            scores,
            limit=limit,
            grasp_style=grasp_style,
        )
        return reachable, rejected, "contact_graspnet"

    def _filter_reachable_grasps(
        self,
        poses: np.ndarray,
        scores: np.ndarray,
        *,
        limit: int,
        grasp_style: str = "auto",
    ) -> tuple[list[_ReachableGrasp], int]:
        style = _validate_grasp_style(grasp_style)
        directions = np.asarray(poses[:, :3, :3], dtype=np.float64) @ np.array(
            [0.0, 0.0, -1.0],
            dtype=np.float64,
        )
        if style == "top_down":
            eligible = np.flatnonzero(
                directions[:, 2] >= _MIN_TOP_DOWN_ALIGNMENT
            )
            eligible = np.asarray(
                sorted(
                    eligible,
                    key=lambda index: (
                        -float(directions[index, 2]),
                        -float(scores[index]),
                    ),
                ),
                dtype=np.int64,
            )
            if not len(eligible):
                raise ToolOperationError(
                    FailureCategory.TOOL_ERROR,
                    "no grasp candidate satisfies top_down alignment",
                )
        elif style == "side":
            eligible = np.flatnonzero(
                np.abs(directions[:, 2])
                <= _MAX_SIDE_VERTICAL_ALIGNMENT
            )
            eligible = np.asarray(
                sorted(
                    eligible,
                    key=lambda index: (
                        abs(float(directions[index, 2])),
                        -float(scores[index]),
                    ),
                ),
                dtype=np.int64,
            )
            if not len(eligible):
                raise ToolOperationError(
                    FailureCategory.TOOL_ERROR,
                    "no grasp candidate satisfies side alignment",
                )
        else:
            eligible = np.argsort(-scores)
        state_result = self.robot.get_robot_state()
        if not state_result.success or state_result.value is None:
            raise ToolOperationError(
                state_result.failure_category,
                state_result.message or "robot state is unavailable",
            )
        state = state_result.value
        seed = np.asarray(
            [
                *state.joint_positions_rad,
                state.gripper_open_fraction * 0.04,
            ],
            dtype=np.float64,
        )
        reachable: list[_ReachableGrasp] = []
        rejected = len(scores) - len(eligible)
        for index in eligible:
            pose = np.asarray(poses[index], dtype=np.float64)
            rotation = pose[:3, :3]
            quaternion = _matrix_to_quaternion_wxyz(rotation)
            target = pose[:3, 3] + rotation @ self._tcp_offset_local_m
            target_joints = self.services.solve_ik(
                target,
                quaternion,
                seed,
            )
            target_cfg = _valid_ik_configuration(target_joints)
            if target_cfg is None:
                rejected += 1
                continue
            target_delta = target_cfg[:7] - seed[:7]
            total_travel = float(
                np.linalg.norm(target_cfg[:7] - seed[:7])
            )
            if (
                total_travel > _MAX_PREDICTED_JOINT_TRAVEL_RAD
                or float(np.max(np.abs(target_delta)))
                > _MAX_PREDICTED_SINGLE_JOINT_DELTA_RAD
            ):
                rejected += 1
                continue
            reachable.append(
                _ReachableGrasp(
                    pose=pose,
                    score=float(scores[index]),
                    predicted_joint_travel_rad=total_travel,
                    approach_direction_base=tuple(
                        float(value)
                        for value in (
                            rotation @ np.array(
                                [0.0, 0.0, -1.0],
                                dtype=np.float64,
                            )
                        )
                    ),
                    top_down_alignment=float(np.clip(directions[index, 2], 0.0, 1.0)),
                )
            )
        if not reachable:
            raise ToolOperationError(
                FailureCategory.IK_FAILURE,
                "no grasp candidate has feasible target IK",
            )
        if style == "top_down":
            reachable.sort(
                key=lambda item: (
                    item.predicted_joint_travel_rad,
                    -item.top_down_alignment,
                    -item.score,
                )
            )
        elif style == "side":
            reachable.sort(
                key=lambda item: (
                    item.predicted_joint_travel_rad,
                    abs(item.approach_direction_base[2]),
                    -item.score,
                )
            )
        else:
            reachable.sort(
                key=lambda item: (
                    item.predicted_joint_travel_rad,
                    -item.score,
                )
            )
        return reachable[:limit], rejected

    def _motion_outcome(
        self,
        phase: str,
        executed: bool,
        message: str,
        *,
        requested_position: Sequence[float],
        controller_target: Sequence[float],
        requested_quaternion: Sequence[float],
        position_tolerance_m: float | None = None,
        orientation_tolerance_rad: float | None = None,
    ) -> dict[str, object]:
        position_tolerance_m = self._motion_position_tolerance_m if position_tolerance_m is None else position_tolerance_m
        orientation_tolerance_rad = self._motion_orientation_tolerance_rad if orientation_tolerance_rad is None else orientation_tolerance_rad
        max_residual = 0.0
        l2_residual = 0.0
        matched = _RESIDUAL_PATTERN.search(message)
        if matched is not None:
            max_residual = float(matched.group(1))
            l2_residual = float(matched.group(2))
        measurement_error = None
        try:
            state = self.robot.get_robot_state()
            if state.success and state.value is not None:
                eef_position = _finite_vector(state.value.end_effector_position_m,
                    3, "measured_eef_position").tolist()
                eef_quaternion = _normalized_quaternion(
                    state.value.end_effector_quaternion_wxyz).tolist()
            else:
                measurement_error = state.message or "robot pose measurement unavailable"
        except Exception as exc:
            measurement_error = f"{type(exc).__name__}: {exc}"
        if measurement_error is not None:
            # A failed observation is not the robot origin or a cached pose.
            # Preserve the requested frame and report the missing measurement.
            return {
                "phase": phase, "executed": bool(executed), "converged": False,
                "convergence_reason": "pose_measurement_unavailable",
                "max_joint_residual_rad": max_residual, "l2_joint_residual_rad": l2_residual,
                "position_tolerance_m": position_tolerance_m,
                "orientation_tolerance_rad": orientation_tolerance_rad,
                "requested_cartesian_position_m": _finite_vector(requested_position, 3, "requested_position").tolist(),
                "requested_cartesian_quaternion_wxyz": _normalized_quaternion(requested_quaternion).tolist(),
                "controller_target_position_m": _finite_vector(controller_target, 3, "controller_target").tolist(),
                "final_cartesian_position_m": None, "final_cartesian_quaternion_wxyz": None,
                "position_residual_m": None, "orientation_residual_rad": None,
                "final_eef_position_m": None, "final_eef_quaternion_wxyz": None,
                "pose_measurement": {"status": "unavailable", "message": measurement_error,
                    "scope": "current robot proprioception; no replacement pose inferred"},
            }
        requested = _finite_vector(
            requested_position, 3, "requested_position"
        )
        controller = _finite_vector(
            controller_target, 3, "controller_target"
        )
        final_hand_position, final_hand_quaternion = (
            self.robot.external_ik_target_from_eef_pose(
                eef_position,
                eef_quaternion,
            )
        )
        final_hand_position_array = _finite_vector(
            final_hand_position,
            3,
            "final_hand_position",
        )
        final_quaternion = _normalized_quaternion(final_hand_quaternion)
        final_position = (
            final_hand_position_array
            - _quaternion_wxyz_to_matrix(final_quaternion)
            @ self._tcp_offset_local_m
        )
        requested_quat = _normalized_quaternion(requested_quaternion)
        position_residual = float(
            np.linalg.norm(final_position - requested)
        )
        orientation_residual = _quaternion_angular_distance(
            requested_quat,
            final_quaternion,
        )
        position_converged = (
            position_residual <= position_tolerance_m
        )
        orientation_converged = (
            orientation_residual
            <= orientation_tolerance_rad
        )
        if position_converged and orientation_converged:
            convergence_reason = "within_tolerance"
        elif not position_converged and not orientation_converged:
            convergence_reason = (
                "position_and_orientation_residual_high"
            )
        elif not position_converged:
            convergence_reason = "position_residual_high"
        else:
            convergence_reason = "orientation_residual_high"
        return {
            "phase": phase,
            "executed": bool(executed),
            "converged": bool(
                position_converged and orientation_converged
            ),
            "convergence_reason": convergence_reason,
            "max_joint_residual_rad": max_residual,
            "l2_joint_residual_rad": l2_residual,
            "position_tolerance_m": position_tolerance_m,
            "orientation_tolerance_rad": (
                orientation_tolerance_rad
            ),
            "requested_cartesian_position_m": [
                float(value) for value in requested
            ],
            "requested_cartesian_quaternion_wxyz": [
                float(value) for value in requested_quat
            ],
            "controller_target_position_m": [
                float(value) for value in controller
            ],
            "final_cartesian_position_m": [
                float(value) for value in final_position
            ],
            "final_cartesian_quaternion_wxyz": [
                float(value) for value in final_quaternion
            ],
            "position_residual_m": position_residual,
            "orientation_residual_rad": orientation_residual,
            "final_eef_position_m": eef_position,
            "final_eef_quaternion_wxyz": eef_quaternion,
        }

    def _store_masks(
        self, camera_name: str, candidates: Sequence[MaskCandidate]
    ) -> dict[str, object]:
        if not candidates:
            raise ToolOperationError(FailureCategory.TOOL_ERROR, "segmentation returned no masks")
        valid_candidates: list[tuple[MaskCandidate, np.ndarray]] = []
        for candidate in candidates:
            mask = np.asarray(candidate.mask, dtype=bool).squeeze()
            if mask.ndim == 2 and mask.any():
                valid_candidates.append((candidate, mask))
        valid_candidates.sort(key=lambda item: item[0].score, reverse=True)

        public = []
        for candidate, mask in valid_candidates[:_MAX_PUBLIC_MASKS]:
            handle = self.registry.put(
                "mask", _MaskArtifact(camera_name, mask, float(candidate.score))
            )
            public.append(
                {
                    "mask_handle": handle,
                    "score": float(candidate.score),
                    "box": list(candidate.box) if candidate.box is not None else None,
                    "label": candidate.label,
                }
            )
        if not public:
            raise ToolOperationError(FailureCategory.TOOL_ERROR, "segmentation masks were empty")
        return {"masks": public}

    def _object_clouds(
        self, object_name: str, use_multiview: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        observation = self._capture_observation()
        object_points = self._object_points_from_observation(
            observation,
            object_name,
            use_multiview=use_multiview,
        )
        selected = observation.cameras if use_multiview else observation.cameras[:1]
        return self._full_cloud(observation, selected), object_points

    def _capture_observation(self) -> _CapturedObservation:
        captured_result = self.get_observation()
        if not captured_result.success or captured_result.value is None:
            raise ToolOperationError(
                captured_result.failure_category,
                captured_result.message,
            )
        return cast(
            _CapturedObservation,
            self.registry.get(
                cast(str, captured_result.value["observation_handle"]),
                expected_kind="observation",
            ),
        )

    @staticmethod
    def _camera_by_name(
        observation: _CapturedObservation,
        camera_name: str,
    ) -> _CapturedCamera:
        normalized = str(camera_name).strip()
        for camera in observation.cameras:
            if camera.camera_name == normalized:
                return camera
        available = ", ".join(
            camera.camera_name for camera in observation.cameras
        )
        raise ValueError(
            f"unknown camera {camera_name!r}; available cameras: {available}"
        )

    def _points_from_mark(
        self,
        point_handle: str,
        *,
        use_multiview: bool,
        require_primary: bool = False,
        projection_filter=None,
    ) -> tuple[_MarkedPoint, np.ndarray, np.ndarray, float]:
        marked = cast(
            _MarkedPoint,
            self.registry.get(
                point_handle,
                expected_kind="marked_point",
            ),
        )
        observation = cast(
            _CapturedObservation,
            self.registry.get(
                marked.observation_handle,
                expected_kind="observation",
            ),
        )
        camera = self._camera_by_name(observation, marked.camera_name)
        masks = [
            _MaskArtifact(
                camera.camera_name,
                marked.mask,
                marked.segmentation_score,
            )
        ]
        primary_points = _depth_to_base_points(
            camera.depth,
            camera.intrinsics,
            camera.extrinsics,
            mask=marked.mask,
        )
        primary_points = _filter_point_cloud_noise(primary_points)
        if len(primary_points) < 3:
            raise ToolOperationError(
                FailureCategory.PERCEPTION_UNCERTAIN,
                "grounding_uncertain: stored point mask has too few valid 3D points",
            )
        primary_centroid, _, _ = _point_geometry(primary_points)
        seed_points = (projection_filter(primary_points) if projection_filter is not None else primary_points)
        view_reconstruction = []
        if use_multiview:
            for other in observation.cameras:
                if (
                    other.camera_name == camera.camera_name
                    or other.depth is None
                ):
                    continue
                if require_primary:
                    from .visible_crossview import visible_seed
                    projected, view_receipt = visible_seed(seed_points, other)
                    view_receipt['camera_name'] = other.camera_name
                    view_reconstruction.append(view_receipt)
                else:
                    projected = _project_base_point_to_pixel(
                        primary_centroid, other.intrinsics, other.extrinsics, other.image.shape[:2])
                if projected is None:
                    continue
                candidates = self.services.segment_point(
                    other.image,
                    projected,
                )
                selected: tuple[MaskCandidate, np.ndarray] | None = None
                pixel_x = int(round(projected[0]))
                pixel_y = int(round(projected[1]))
                for candidate in sorted(
                    candidates,
                    key=lambda item: float(item.score),
                    reverse=True,
                ):
                    candidate_mask = np.asarray(
                        candidate.mask,
                        dtype=bool,
                    ).squeeze()
                    if (
                        candidate_mask.ndim == 2
                        and candidate_mask.shape == other.image.shape[:2]
                        and candidate_mask.any()
                        and candidate_mask[pixel_y, pixel_x]
                    ):
                        selected = (candidate, candidate_mask)
                        break
                if selected is None:
                    if require_primary:view_receipt['segmentation_status'] = 'no_mask_at_supported_seed'
                    continue
                candidate, candidate_mask = selected
                candidate_points = _filter_point_cloud_noise(
                    _depth_to_base_points(
                        other.depth,
                        other.intrinsics,
                        other.extrinsics,
                        mask=candidate_mask,
                    )
                )
                if len(candidate_points) < 3:
                    continue
                if require_primary:
                    from .visible_crossview import overlap_receipt
                    overlap = overlap_receipt(seed_points, candidate_points)
                    view_receipt['candidate_overlap'] = overlap
                    if not overlap['supported']:
                        view_receipt['segmentation_status'] = 'rejected_without_primary_surface_overlap'
                        continue
                    view_receipt['segmentation_status'] = 'accepted_with_primary_surface_overlap'
                else:
                    candidate_centroid, _, _ = _point_geometry(candidate_points)
                    if float(np.linalg.norm(candidate_centroid - primary_centroid)) > 0.08:
                        continue
                masks.append(
                    _MaskArtifact(
                        other.camera_name,
                        candidate_mask,
                        float(candidate.score),
                    )
                )
        full_points, object_points = self._clouds_from_masks(
            observation,
            tuple(masks),
            # In bound mode include the primary mask even when it belongs to
            # a later camera. This does not request another segmentation view.
            use_multiview=use_multiview or require_primary,
        )
        if require_primary:
            # Scored fusion may replace a disjoint low-score primary cloud.
            # Never bind that replacement to an already accepted target mark.
            from captain_harness.ports.deadline import check_deadline
            check_deadline()
            distances, _ = cKDTree(object_points).query(primary_points, k=1)
            if not np.all(np.isfinite(distances) & (distances <= 1e-7)):
                raise ValueError('carried_geometry_primary_view_not_retained')
            check_deadline()
        if require_primary:
            marked.carried_view_reconstruction = view_reconstruction
            # Pixel membership is separate provenance from the denoised mesh.
            # These masks never independently authorize scene removal.
            selected_masks = []
            for selected_mask in masks:
                frozen_mask = np.asarray(selected_mask.mask, dtype=bool).copy()
                frozen_mask.setflags(write=False)
                selected_masks.append((selected_mask.camera_name, frozen_mask))
            marked.carried_selected_masks = tuple(selected_masks)
        marked.views_used = len(masks)
        marked.cross_view_consistent = len(masks) > 1
        return (
            marked,
            full_points,
            object_points,
            min(mask.score for mask in masks),
        )

    def _clouds_from_detection(
        self,
        observation: _CapturedObservation,
        detection: _Detection,
        *,
        use_multiview: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        primary = self._camera_by_name(
            observation,
            detection.camera_name,
        )
        masks = [
            _MaskArtifact(
                detection.camera_name,
                detection.mask,
                detection.score,
            )
        ]
        if use_multiview:
            for camera in observation.cameras:
                if (
                    camera.camera_name == detection.camera_name
                    or camera.depth is None
                ):
                    continue
                projected = _project_base_point_to_pixel(
                    detection.centroid,
                    camera.intrinsics,
                    camera.extrinsics,
                    camera.image.shape[:2],
                )
                if projected is None:
                    continue
                candidates = self.services.segment_point(
                    camera.image,
                    projected,
                )
                pixel_x = int(round(projected[0]))
                pixel_y = int(round(projected[1]))
                for candidate in sorted(
                    candidates,
                    key=lambda item: float(item.score),
                    reverse=True,
                ):
                    mask = np.asarray(
                        candidate.mask,
                        dtype=bool,
                    ).squeeze()
                    if (
                        mask.ndim != 2
                        or mask.shape != camera.image.shape[:2]
                        or not mask.any()
                        or not mask[pixel_y, pixel_x]
                    ):
                        continue
                    points = _filter_point_cloud_noise(
                        _depth_to_base_points(
                            camera.depth,
                            camera.intrinsics,
                            camera.extrinsics,
                            mask=mask,
                        )
                    )
                    if len(points) < 3:
                        continue
                    centroid, _, _ = _point_geometry(points)
                    if (
                        float(
                            np.linalg.norm(
                                centroid - detection.centroid
                            )
                        )
                        > 0.08
                    ):
                        continue
                    masks.append(
                        _MaskArtifact(
                            camera.camera_name,
                            mask,
                            float(candidate.score),
                        )
                    )
                    break
        full_points, object_points = self._clouds_from_masks(
            observation,
            tuple(masks),
            use_multiview=use_multiview,
        )
        detection.views_used = len(masks)
        detection.cross_view_consistent = len(masks) > 1
        if primary.depth is None:
            raise ToolOperationError(
                FailureCategory.PERCEPTION_UNCERTAIN,
                "selected detection camera has no public depth",
            )
        return full_points, object_points

    def _spatial_relation_payload(
        self,
        *,
        target_name: str,
        destination_name: str,
        relation: str,
        target_points: np.ndarray,
        destination_points: np.ndarray,
        strict_support_height: bool,
    ) -> dict[str, object]:
        target_centroid, target_min, target_max = _point_geometry(
            target_points
        )
        destination_centroid, destination_min, destination_max = (
            _point_geometry(destination_points)
        )
        normalized_target = " ".join(target_name.split()).casefold()
        normalized_destination = " ".join(
            destination_name.split()
        ).casefold()
        same_geometry = (
            float(
                np.linalg.norm(
                    target_centroid - destination_centroid
                )
            )
            <= 0.015
            and float(np.linalg.norm(target_min - destination_min)) <= 0.02
            and float(np.linalg.norm(target_max - destination_max)) <= 0.02
        )
        distinct_entities = (
            normalized_target != normalized_destination and not same_geometry
        )
        centroid_inside_xy = bool(
            np.all(target_centroid[:2] >= destination_min[:2])
            and np.all(target_centroid[:2] <= destination_max[:2])
        )
        footprint_inside_xy = bool(
            np.all(target_min[:2] >= destination_min[:2])
            and np.all(target_max[:2] <= destination_max[:2])
        )
        target_area = max(
            float(
                (target_max[0] - target_min[0])
                * (target_max[1] - target_min[1])
            ),
            1e-9,
        )
        overlap_x = max(
            0.0,
            float(
                min(target_max[0], destination_max[0])
                - max(target_min[0], destination_min[0])
            ),
        )
        overlap_y = max(
            0.0,
            float(
                min(target_max[1], destination_max[1])
                - max(target_min[1], destination_min[1])
            ),
        )
        footprint_overlap_ratio = overlap_x * overlap_y / target_area
        vertical_gap = float(target_min[2] - destination_max[2])
        robust_vertical_gap = float(
            np.percentile(target_points[:, 2], 10.0)
            - np.percentile(destination_points[:, 2], 50.0)
        )
        strict_bounds_support_valid = -0.04 <= vertical_gap <= 0.03
        if strict_support_height:
            support_height_valid = strict_bounds_support_valid
        else:
            support_height_valid = -0.12 <= vertical_gap <= 0.08
        support_height_basis = (
            "bounds"
            if (
                not strict_support_height
                or strict_bounds_support_valid
            )
            else "bounds_rejected"
        )
        in_container_height = bool(
            target_centroid[2] >= destination_min[2] - 0.03
            and target_centroid[2]
            <= destination_max[2] - _CONTAINER_RIM_CLEARANCE_M
        )
        relation_ready = (
            distinct_entities
            and centroid_inside_xy
            and (
                (
                    footprint_overlap_ratio
                    >= self.completion_protocol.min_on_footprint_overlap
                    and support_height_valid
                )
                if relation == "on"
                else in_container_height
            )
        )
        if not distinct_entities:
            reason = "target_destination_not_distinct"
        elif not centroid_inside_xy:
            reason = "target_centroid_outside_destination_xy"
        elif (
            relation == "on"
            and footprint_overlap_ratio
            < self.completion_protocol.min_on_footprint_overlap
        ):
            reason = "target_footprint_outside_destination_xy"
        elif relation == "on" and not support_height_valid:
            reason = "target_not_on_destination_support_height"
        elif relation == "in" and not in_container_height:
            reason = "target_not_inside_destination_height"
        else:
            reason = "relation_ready"
        return {
            "relation_ready": relation_ready,
            "reason": reason,
            "target_name": target_name,
            "destination_name": destination_name,
            "relation_type": relation,
            "target_centroid_m": [
                float(value) for value in target_centroid
            ],
            "target_bounds_min_m": [
                float(value) for value in target_min
            ],
            "target_bounds_max_m": [
                float(value) for value in target_max
            ],
            "destination_centroid_m": [
                float(value) for value in destination_centroid
            ],
            "destination_bounds_min_m": [
                float(value) for value in destination_min
            ],
            "destination_bounds_max_m": [
                float(value) for value in destination_max
            ],
            "centroid_inside_destination_xy": centroid_inside_xy,
            "footprint_inside_destination_xy": footprint_inside_xy,
            "footprint_overlap_ratio": footprint_overlap_ratio,
            "support_vertical_gap_m": vertical_gap,
            "robust_support_vertical_gap_m": robust_vertical_gap,
            "support_height_basis": support_height_basis,
            "distinct_entities": distinct_entities,
            "target_point_count": int(len(target_points)),
            "destination_point_count": int(len(destination_points)),
        }

    def _object_points_from_observation(
        self,
        observation: _CapturedObservation,
        object_name: str,
        *,
        use_multiview: bool,
    ) -> np.ndarray:
        selected = observation.cameras if use_multiview else observation.cameras[:1]
        masks: list[_MaskArtifact] = []
        for camera in selected:
            point = self.services.point(camera.image, object_name)
            candidates = (
                self.services.segment_point(camera.image, point)
                if point is not None
                else []
            )
            if not candidates:
                candidates = self.services.segment_text(camera.image, object_name)
            if not candidates:
                continue
            best = max(candidates, key=lambda item: item.score)
            mask = np.asarray(best.mask, dtype=bool).squeeze()
            if mask.ndim == 2 and mask.any():
                masks.append(_MaskArtifact(camera.camera_name, mask, float(best.score)))
        if not masks:
            raise ToolOperationError(
                FailureCategory.TOOL_ERROR,
                f"could not segment {object_name!r} in the selected views",
            )
        per_view = []
        for camera in selected:
            camera_masks = [item for item in masks if item.camera_name == camera.camera_name]
            if not camera_masks or camera.depth is None:
                continue
            best = max(camera_masks, key=lambda item: item.score)
            points = _depth_to_base_points(
                camera.depth, camera.intrinsics, camera.extrinsics, mask=best.mask
            )
            if len(points):
                per_view.append((points, best.score))
        object_points = _fuse_scored_point_clouds(per_view)
        object_points = _filter_point_cloud_noise(object_points)
        if len(object_points) < 3:
            raise ToolOperationError(
                FailureCategory.TOOL_ERROR, "segmentation produced too few valid 3D points"
            )
        return object_points

    def _clouds_from_masks(
        self,
        observation: _CapturedObservation,
        masks: Sequence[_MaskArtifact],
        *,
        use_multiview: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        selected = observation.cameras if use_multiview else observation.cameras[:1]
        full_points = self._full_cloud(observation, selected)
        object_parts: list[tuple[np.ndarray, float]] = []
        for camera in selected:
            if camera.depth is None:
                continue
            camera_masks = [
                item
                for item in masks
                if item.camera_name == camera.camera_name
            ]
            if camera_masks:
                combined = np.logical_or.reduce(
                    [item.mask for item in camera_masks]
                )
                if combined.shape != camera.depth.shape:
                    raise ValueError(
                        f"mask shape {combined.shape} does not match depth {camera.depth.shape}"
                    )
                points = _depth_to_base_points(
                        camera.depth, camera.intrinsics, camera.extrinsics, mask=combined
                    )
                if len(points):
                    object_parts.append(
                        (
                            points,
                            max(item.score for item in camera_masks),
                        )
                    )
        object_points = _fuse_scored_point_clouds(object_parts)
        object_points = _filter_point_cloud_noise(object_points)
        if len(object_points) < 3:
            raise ToolOperationError(
                FailureCategory.TOOL_ERROR, "segmentation produced too few valid 3D points"
            )
        return full_points, object_points

    @staticmethod
    def _full_cloud(
        observation: _CapturedObservation, selected: Sequence[_CapturedCamera]
    ) -> np.ndarray:
        full_parts = [
            _depth_to_base_points(camera.depth, camera.intrinsics, camera.extrinsics)
            for camera in selected
            if camera.depth is not None
        ]
        if not full_parts:
            raise ToolOperationError(FailureCategory.TOOL_ERROR, "no depth was available")
        return np.concatenate(full_parts)

    def _store_clouds(
        self, full_points: np.ndarray, object_points: np.ndarray
    ) -> dict[str, object]:
        centroid, bounds_min, bounds_max = _point_geometry(object_points)
        return {
            "full_point_cloud_handle": self.registry.put("point_cloud", full_points),
            "object_point_cloud_handle": self.registry.put("point_cloud", object_points),
            "full_point_count": int(len(full_points)),
            "object_point_count": int(len(object_points)),
            "object_centroid_m": [float(value) for value in centroid],
            "object_bounds_min_m": [float(value) for value in bounds_min],
            "object_bounds_max_m": [float(value) for value in bounds_max],
        }


def _point_geometry(
    object_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("object point cloud must contain finite 3D points")
    return (
        np.median(points, axis=0),
        np.min(points, axis=0),
        np.max(points, axis=0),
    )


def _robot_state_json(state: RobotState) -> dict[str, object]:
    return {
        "joint_positions_rad": list(state.joint_positions_rad),
        "end_effector_position_m": list(state.end_effector_position_m),
        "end_effector_quaternion_wxyz": list(state.end_effector_quaternion_wxyz),
        "gripper_open_fraction": float(state.gripper_open_fraction),
        "joint_velocity_norm_rad_s": float(
            state.joint_velocity_norm_rad_s
        ),
    }


def _finite_vector(value: Sequence[float], size: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain {size} finite numbers")
    return array


def _normalized_quaternion(value: Sequence[float]) -> np.ndarray:
    quaternion = _finite_vector(value, 4, "quaternion")
    norm = float(np.linalg.norm(quaternion))
    if abs(norm - 1.0) > 1e-3:
        raise ValueError("quaternion must be normalized within 1e-3")
    return quaternion / norm


def _validate_grasp_style(value: str) -> str:
    style = str(value).strip().casefold()
    if style not in {"auto", "top_down", "side"}:
        raise ValueError("grasp_style must be auto, top_down, or side")
    return style


def _validate_grasp_backend(value: str) -> str:
    backend = str(value).strip().casefold()
    if backend not in {
        "auto",
        "contact_graspnet",
        "anygrasp",
        "graspgenx",
        "mask_topdown_fallback",
    }:
        raise ValueError(
            "backend must be auto, contact_graspnet, anygrasp, graspgenx, or mask_topdown_fallback"
        )
    return backend


def _validate_depth_source(value: str) -> str:
    source = str(value).strip().casefold()
    if source not in {"sensor", "enhanced"}:
        raise ValueError("depth_source must be sensor or enhanced")
    return source


def _grasp_style_preference(
    style: str,
    *,
    top_down_alignment: float,
) -> float:
    if style == "top_down":
        return float(np.clip(top_down_alignment, 0.0, 1.0))
    if style == "side":
        return float(np.clip(1.0 - abs(top_down_alignment), 0.0, 1.0))
    return 1.0


def _grasp_convergence_risk(
    predicted_joint_travel_rad: float | None,
    *,
    top_down_alignment: float,
) -> str:
    if predicted_joint_travel_rad is None:
        return "high"
    if predicted_joint_travel_rad < 1.2:
        level = 0
    elif predicted_joint_travel_rad < 1.6:
        level = 1
    else:
        level = 2
    if top_down_alignment >= _MIN_TOP_DOWN_ALIGNMENT:
        level = max(0, level - 1)
    return ("low", "medium", "high")[level]


def _reference_phrase_from_prompt(value: str) -> str | None:
    """Extract the reference-object noun phrase from a proximity prompt.

    "black bowl next to the ramekin and place it on the plate" -> "ramekin".
    The phrase anchors a harness-side segmentation check so the model cannot
    satisfy the reference gate by pointing at an arbitrary passable object.
    """
    text = " ".join(str(value).casefold().split())
    for marker in ("next to", "closest to", "nearest to", "beside", "near"):
        index = text.find(marker)
        if index < 0:
            continue
        tail = text[index + len(marker):].strip()
        for stop in (" and ", " then ", ",", ";", "."):
            cut = tail.find(stop)
            if cut >= 0:
                tail = tail[:cut]
        tail = tail.strip()
        if tail.startswith("the "):
            tail = tail[4:]
        return tail or None
    return None


def _validate_relation_type(value: str) -> str:
    relation = str(value).strip().casefold()
    if relation not in {"on", "in"}:
        raise ValueError("relation_type must be one of on or in")
    return relation


def _quaternion_angular_distance(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    q1 = _normalized_quaternion(first)
    q2 = _normalized_quaternion(second)
    dot = min(1.0, max(-1.0, abs(float(np.dot(q1, q2)))))
    return float(2.0 * math.acos(dot))


def _quaternion_wxyz_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    w, x, y, z = _normalized_quaternion(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _transform_from_pose(
    position: Sequence[float],
    quaternion: Sequence[float],
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _quaternion_wxyz_to_matrix(quaternion)
    transform[:3, 3] = _finite_vector(position, 3, "position")
    return transform


def _filter_point_cloud_noise(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) < _NOISE_MIN_SAMPLES:
        return np.empty((0, 3), dtype=np.float64)
    tree = cKDTree(points)
    neighbor_counts = tree.query_ball_point(
        points, r=_NOISE_RADIUS_M, return_length=True
    )
    core = np.asarray(neighbor_counts) >= _NOISE_MIN_SAMPLES
    if not core.any():
        return np.empty((0, 3), dtype=np.float64)
    distances, _ = cKDTree(points[core]).query(points, k=1)
    return points[np.asarray(distances) <= _NOISE_RADIUS_M]


def _fuse_scored_point_clouds(
    clouds: Sequence[tuple[np.ndarray, float]],
) -> np.ndarray:
    nonempty = [
        (np.asarray(points, dtype=np.float64).reshape(-1, 3), float(score))
        for points, score in clouds
        if len(points)
    ]
    if not nonempty:
        return np.empty((0, 3), dtype=np.float64)
    fused, fused_score = nonempty[0]
    for points, score in nonempty[1:]:
        nearest, _ = cKDTree(points).query(fused, k=1)
        if float(np.min(nearest)) < _MULTIVIEW_MERGE_DISTANCE_M:
            fused = np.concatenate((fused, points))
            fused_score = max(fused_score, score)
        elif score > fused_score:
            fused, fused_score = points, score
    return fused


def _depth_to_base_points(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float64).squeeze()
    if depth.ndim != 2:
        raise ValueError("depth must be 2D")
    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(valid)
    if not len(xs):
        return np.empty((0, 3), dtype=np.float64)
    pixels = np.vstack((xs, ys, np.ones(len(xs), dtype=np.float64)))
    camera = np.linalg.solve(intrinsics, pixels) * depth[ys, xs]
    homogeneous = np.vstack((camera, np.ones(len(xs), dtype=np.float64)))
    points = (extrinsics @ homogeneous)[:3].T
    return points[np.isfinite(points).all(axis=1)]


def _project_base_point_to_pixel(
    point: Sequence[float],
    intrinsics: np.ndarray,
    camera_to_base: np.ndarray,
    image_shape: Sequence[int],
) -> tuple[float, float] | None:
    base_point = np.append(
        _finite_vector(point, 3, "base point"),
        1.0,
    )
    camera_point = np.linalg.solve(
        np.asarray(camera_to_base, dtype=np.float64).reshape(4, 4),
        base_point,
    )[:3]
    if not np.isfinite(camera_point).all() or camera_point[2] <= 1e-6:
        return None
    pixel = (
        np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
        @ camera_point
    )
    x = float(pixel[0] / pixel[2])
    y = float(pixel[1] / pixel[2])
    height, width = (int(image_shape[0]), int(image_shape[1]))
    if not (0 <= x < width and 0 <= y < height):
        return None
    return x, y


def _estimate_pose(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    center = np.median(points, axis=0)
    covariance = np.cov(points - center, rowvar=False)
    _, vectors = np.linalg.eigh(covariance)
    rotation = vectors[:, ::-1]
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1
    if rotation[2, 2] > 0:
        rotation[:, (0, 2)] *= -1
    return center, rotation


def _validate_grasps(
    poses: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    poses = np.asarray(poses, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if poses.size == 0 or scores.size == 0:
        raise ToolOperationError(
            FailureCategory.TOOL_ERROR, "grasp service returned no candidates"
        )
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) != len(scores):
        raise ToolOperationError(
            FailureCategory.TOOL_ERROR, "grasp service returned invalid shapes"
        )
    valid = np.isfinite(poses).all(axis=(1, 2)) & np.isfinite(scores)
    poses, scores = poses[valid], scores[valid]
    if not len(scores):
        raise ToolOperationError(
            FailureCategory.TOOL_ERROR, "grasp service returned no candidates"
        )
    return poses, scores


def _valid_ik_configuration(
    value: Sequence[float] | None,
) -> np.ndarray | None:
    if value is None:
        return None
    configuration = np.asarray(value, dtype=np.float64).reshape(-1)
    if configuration.shape != (8,) or not np.isfinite(configuration).all():
        return None
    return configuration


def _apply_capx_grasp_convention(
    poses: np.ndarray, *, rotate_tool: bool = False
) -> np.ndarray:
    local_transform = np.eye(4, dtype=np.float64)
    local_transform[2, 3] = 0.12
    if rotate_tool:
        local_transform[:3, :3] = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    return np.asarray(poses, dtype=np.float64) @ local_transform


def _matrix_to_quaternion_wxyz(matrix: np.ndarray) -> tuple[float, float, float, float]:
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        values = (
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = math.sqrt(1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            values = ((matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale)
        elif axis == 1:
            scale = math.sqrt(1 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            values = ((matrix[0, 2] - matrix[2, 0]) / scale, (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale, (matrix[1, 2] + matrix[2, 1]) / scale)
        else:
            scale = math.sqrt(1 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            values = ((matrix[1, 0] - matrix[0, 1]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale, (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale)
    norm = math.sqrt(sum(value * value for value in values))
    return cast(tuple[float, float, float, float], tuple(float(value / norm) for value in values))

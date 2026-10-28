from __future__ import annotations

import base64
import io
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import numpy as np
from PIL import Image

from captain_harness.adapters.libero.calibration import GripperProfile
from captain_harness.ports.deadline import DeadlineExceeded, deadline_scope

from .backend import GraspServiceBatch, MaskCandidate


class JsonTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        payload: Mapping[str, object] | None = None,
        timeout_s: float,
    ) -> Mapping[str, object]: ...


class UrllibJsonTransport:
    def request(
        self,
        method: str,
        url: str,
        *,
        payload: Mapping[str, object] | None = None,
        timeout_s: float,
    ) -> Mapping[str, object]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                value = json.loads(response.read().decode("utf-8"))
        except TimeoutError:
            raise
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OSError(f"HTTP {exc.code} from {url}: {detail[:300]}") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise exc.reason
            raise OSError(f"request to {url} failed: {exc.reason}") from exc
        if not isinstance(value, Mapping):
            raise OSError(f"service {url} returned non-object JSON")
        return value


class OpenEtaServiceError(OSError):
    def __init__(
        self,
        reason: str,
        details: Mapping[str, object],
    ) -> None:
        super().__init__(f"OpenETA service failed: {reason}")
        self.reason = str(reason)
        self.details = dict(details)


@dataclass(frozen=True, slots=True)
class CapXServiceEndpoints:
    sam3: str = "http://127.0.0.1:8114"
    molmo: str = "http://127.0.0.1:8122/v1"
    contact_graspnet: str = "http://127.0.0.1:8115"
    pyroki: str = "http://127.0.0.1:8116"
    openeta_contact_graspnet: str | None = None
    anygrasp: str | None = None
    graspgenx: str | None = None
    anyplace: str | None = None
    unidepth_v2: str | None = None
    openeta_revision: str | None = None
    graspgenx_gripper_name: str = "franka_panda"


class CapXHttpToolServices:
    """HTTP clients matching the pinned CaP-X SAM3/Molmo/CGN/PyRoKi APIs."""

    def __init__(
        self,
        endpoints: CapXServiceEndpoints = CapXServiceEndpoints(),
        *,
        transport: JsonTransport | None = None,
        attempts: int = 2,
        timeout_s: float = 180.0,
        retry_delay_s: float = 0.0,
        molmo_model: str = "allenai/Molmo2-8B",
    ) -> None:
        if attempts <= 0 or timeout_s <= 0 or retry_delay_s < 0:
            raise ValueError("service attempts/timeouts must be positive")
        self.endpoints = endpoints
        self.transport = transport or UrllibJsonTransport()
        self.attempts = attempts
        self.timeout_s = timeout_s
        self.retry_delay_s = retry_delay_s
        self.molmo_model = molmo_model
        self.request_receipts = []

    def _request(
        self,
        method: str,
        url: str,
        payload: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        attempts = []
        started = time.monotonic()
        with deadline_scope("service_request", timeout_s=self.timeout_s) as deadline:
            try:
                for attempt in range(self.attempts):
                    remaining = deadline.remaining()
                    if remaining <= 0:
                        raise DeadlineExceeded(deadline, attempts=attempts)
                    attempt_started = time.monotonic()
                    try:
                        result = self.transport.request(method, url, payload=payload, timeout_s=remaining)
                    except OSError as exc:
                        attempts.append({"attempt": attempt + 1, "status": "failed",
                            "error_type": type(exc).__name__, "timeout_s": remaining,
                            "elapsed_s": time.monotonic() - attempt_started})
                        if isinstance(exc, TimeoutError):
                            raise DeadlineExceeded(deadline, str(exc), attempts=attempts) from exc
                        if attempt + 1 >= self.attempts:
                            raise
                        if self.retry_delay_s:
                            time.sleep(min(self.retry_delay_s, deadline.remaining()))
                    else:
                        attempts.append({"attempt": attempt + 1, "status": "returned",
                            "timeout_s": remaining, "elapsed_s": time.monotonic() - attempt_started})
                        if deadline.remaining() <= 0:
                            raise DeadlineExceeded(deadline, "service response arrived after deadline", attempts=attempts)
                        return result
            finally:
                self.request_receipts.append({"request_index": len(self.request_receipts),
                    "elapsed_s": time.monotonic() - started, "deadline": deadline.receipt(),
                    "attempts": attempts})

    def health(self, names=None) -> dict[str, dict[str, object]]:
        checks = [
            ("sam3", f"{self.endpoints.sam3.rstrip('/')}/openapi.json"),
            ("molmo", f"{self.endpoints.molmo.rstrip('/')}/models"),
            (
                "contact_graspnet",
                f"{self.endpoints.contact_graspnet.rstrip('/')}/openapi.json",
            ),
            ("pyroki", f"{self.endpoints.pyroki.rstrip('/')}/openapi.json"),
        ]
        for name, endpoint in (
            (
                "openeta_contact_graspnet",
                self.endpoints.openeta_contact_graspnet,
            ),
            ("anygrasp", self.endpoints.anygrasp),
            ("graspgenx", self.endpoints.graspgenx),
            ("anyplace", self.endpoints.anyplace),
            ("unidepth_v2", self.endpoints.unidepth_v2),
        ):
            if endpoint:
                checks.append((name, f"{endpoint.rstrip('/')}/health"))
        report: dict[str, dict[str, object]] = {}
        for name, url in checks:
            if names is not None and name not in names:
                continue
            try:
                payload = self._request("GET", url)
            except Exception as exc:
                report[name] = {
                    "healthy": False,
                    "url": url,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            else:
                is_openeta = name in {
                    "openeta_contact_graspnet",
                    "anygrasp",
                    "graspgenx",
                    "anyplace",
                    "unidepth_v2",
                }
                expected_backend = (
                    "contact_graspnet"
                    if name == "openeta_contact_graspnet"
                    else name
                )
                healthy = (
                    not is_openeta
                    or (
                        payload.get("ok") is True
                        and payload.get("backend") == expected_backend
                        and (
                            self.endpoints.openeta_revision is None
                            or payload.get("openeta_revision")
                            == self.endpoints.openeta_revision
                        )
                    )
                )
                report[name] = {
                    "healthy": healthy,
                    "url": url,
                    "error": (
                        None
                        if healthy
                        else "OpenETA health identity or revision mismatch"
                    ),
                    **(
                        {"metadata": dict(payload)}
                        if is_openeta
                        else {}
                    ),
                }
        return report

    def available_grasp_backends(self) -> tuple[str, ...]:
        available = ["contact_graspnet"]
        if self.endpoints.anygrasp:
            available.append("anygrasp")
        if self.endpoints.graspgenx:
            available.append("graspgenx")
        return tuple(available)

    def segment_text(self, image: np.ndarray, text: str) -> list[MaskCandidate]:
        response = self._request(
            "POST",
            f"{self.endpoints.sam3.rstrip('/')}/segment",
            {"image_base64": _encode_png(image), "text_prompt": text},
        )
        results = response.get("results", [])
        if not isinstance(results, list):
            raise OSError("SAM3 results must be a list")
        candidates = []
        for item in results:
            if not isinstance(item, Mapping):
                raise OSError("SAM3 result entry must be an object")
            shape = tuple(int(value) for value in item["shape"])
            mask = np.frombuffer(
                base64.b64decode(str(item["mask_base64"])), dtype=np.uint8
            ).reshape(shape)
            box_value = item.get("box")
            box = tuple(float(value) for value in box_value) if box_value is not None else None
            candidates.append(
                MaskCandidate(
                    mask.astype(bool),
                    float(item["score"]),
                    box,  # type: ignore[arg-type]
                    str(item["label"]) if item.get("label") is not None else None,
                )
            )
        return candidates

    def segment_point(
        self, image: np.ndarray, point_xy: Sequence[float]
    ) -> list[MaskCandidate]:
        response = self._request(
            "POST",
            f"{self.endpoints.sam3.rstrip('/')}/segment_point",
            {
                "image_base64": _encode_png(image),
                "point_coords": [float(point_xy[0]), float(point_xy[1])],
            },
        )
        scores = np.asarray(response.get("scores", []), dtype=np.float64)
        shape = tuple(int(value) for value in response.get("masks_shape", (0, 0, 0)))
        if not scores.size:
            return []
        dtype = np.dtype(str(response.get("masks_dtype", "float32")))
        masks = np.frombuffer(
            base64.b64decode(str(response.get("masks_base64", ""))), dtype=dtype
        ).reshape(shape)
        return [
            MaskCandidate(np.asarray(mask, dtype=bool), float(score))
            for mask, score in zip(masks, scores)
        ]

    def point(self, image: np.ndarray, text: str) -> tuple[float, float] | None:
        response = self._request(
            "POST",
            f"{self.endpoints.molmo.rstrip('/')}/chat/completions",
            {
                "model": self.molmo_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Point at {text}"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{_encode_png(image)}"
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 1024,
                "temperature": 0.0,
            },
        )
        try:
            generated = str(response["choices"][0]["message"]["content"])  # type: ignore[index]
        except (KeyError, IndexError, TypeError):
            return None
        parsed = _parse_molmo_point(generated)
        if parsed is None:
            return None
        x, y, scale = parsed
        height, width = image.shape[:2]
        return (x / scale * width, y / scale * height)

    def plan_grasps(
        self, full_points: np.ndarray, object_points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        response = self._request(
            "POST",
            f"{self.endpoints.contact_graspnet.rstrip('/')}/plan_point_clouds",
            {
                "pc_full_base64": _encode_npy(full_points),
                "pc_segment_base64": _encode_npy(object_points),
                "segmap_id": 1,
                "local_regions": True,
                "filter_grasps": True,
                "forward_passes": 2,
                "max_retries": 10,
            },
        )
        return (
            _decode_npy(str(response["grasps_base64"])),
            _decode_npy(str(response["scores_base64"])),
        )

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
    ) -> GraspServiceBatch:
        selected = str(backend).strip().casefold()
        if selected not in {
            "contact_graspnet",
            "anygrasp",
            "graspgenx",
        }:
            raise OSError(f"unsupported grasp backend: {backend}")
        metric_depth, depth_quality = self._depth_for_prediction(
            image=image,
            sensor_depth=depth,
            intrinsics=intrinsics,
            depth_source=depth_source,
            fill_mask=mask,
        )
        collision_scope: dict[str, object] = {}
        service_depth = metric_depth
        if selected == "graspgenx":
            service_depth, collision_scope = _scoped_grasp_collision_depth(
                metric_depth,
                mask=mask,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                relaxed=bool(collision_relaxed),
            )
        if (
            selected == "contact_graspnet"
            and not self.endpoints.openeta_contact_graspnet
        ):
            full_points = _depth_to_points(
                metric_depth,
                intrinsics,
                extrinsics,
            )
            object_points = _depth_to_points(
                metric_depth,
                intrinsics,
                extrinsics,
                mask=mask,
            )
            poses, scores = self.plan_grasps(
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
                depth_quality=depth_quality,
            )
        endpoint, path = self._openeta_grasp_endpoint(selected)
        intrinsics_payload = _intrinsics_payload(intrinsics)
        depth_payload = _depth_png_payload(service_depth)
        mask_payload = _mask_png_payload(mask)
        if selected == "contact_graspnet":
            payload = {
                "depth": depth_payload,
                "object_mask": mask_payload,
                "intrinsics": intrinsics_payload,
            }
        elif selected == "anygrasp":
            payload = {
                "rgb": _rgb_png_payload(image),
                "depth": depth_payload,
                "intrinsics": intrinsics_payload,
                "mode": "targeted",
                "target_mask": mask_payload,
                "collision_detection": depth_source == "sensor",
                "dense_grasp": False,
            }
        else:
            camera_rotation = np.asarray(
                extrinsics,
                dtype=np.float64,
            ).reshape(4, 4)[:3, :3]
            up_camera = camera_rotation.T @ np.array(
                [0.0, 0.0, 1.0],
                dtype=np.float64,
            )
            payload = {
                "depth": depth_payload,
                "object_mask": mask_payload,
                "intrinsics": intrinsics_payload,
                "gripper_name": (
                    self.endpoints.graspgenx_gripper_name
                ),
                "up_direction_camera": [
                    float(value) for value in up_camera
                ],
            }
        response = self._request(
            "POST",
            f"{endpoint.rstrip('/')}/{path}",
            payload,
        )
        details = _openeta_details(response)
        candidates = details.get("grasp_candidates")
        if not isinstance(candidates, list) or not candidates:
            raise OSError(
                f"{selected} returned no normalized grasp candidates"
            )
        camera_to_base = np.asarray(
            extrinsics,
            dtype=np.float64,
        ).reshape(4, 4)
        poses = []
        scores = []
        widths: list[float | None] = []
        approaches = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise OSError("OpenETA grasp candidate must be an object")
            rotation = np.asarray(
                candidate.get("rotation_matrix"),
                dtype=np.float64,
            ).reshape(3, 3)
            translation = np.asarray(
                candidate.get("translation_xyz"),
                dtype=np.float64,
            ).reshape(3)
            camera_pose = np.eye(4, dtype=np.float64)
            camera_pose[:3, :3] = rotation
            camera_pose[:3, 3] = translation
            base_pose = camera_to_base @ camera_pose
            poses.append(base_pose)
            scores.append(float(candidate["score"]))
            width = candidate.get("width")
            widths.append(
                float(width)
                if isinstance(width, (int, float))
                else None
            )
            approach = base_pose[:3, 0]
            norm = float(np.linalg.norm(approach))
            approaches.append(
                tuple(
                    float(value)
                    for value in approach / max(norm, 1e-12)
                )
            )
        return GraspServiceBatch(
            poses=np.stack(poses),
            scores=np.asarray(scores, dtype=np.float64),
            widths_m=tuple(widths),
            backend=selected,
            public_tool_convention=True,
            grasp_frame="graspnet",
            approach_directions_base=tuple(approaches),
            native_candidates=tuple(
                _normalize_openeta_grasp_candidate(
                    candidate,
                    gripper_profile=gripper_profile,
                )
                for candidate in candidates
                if isinstance(candidate, Mapping)
            ),
            depth_quality={
                **depth_quality,
                **collision_scope,
                "backend_metadata": dict(
                    details.get("metadata", {})
                )
                if isinstance(details.get("metadata"), Mapping)
                else {},
                "gripper_max_width_m": gripper_profile.max_width_m,
            },
        )

    def _openeta_grasp_endpoint(
        self,
        backend: str,
    ) -> tuple[str, str]:
        if backend == "contact_graspnet":
            endpoint = self.endpoints.openeta_contact_graspnet
            path = "predict_grasps"
        elif backend == "anygrasp":
            endpoint = self.endpoints.anygrasp
            path = "detect_grasps"
        else:
            endpoint = self.endpoints.graspgenx
            path = "predict_grasps"
        if not endpoint:
            raise OSError(f"backend_unavailable: {backend}")
        return endpoint, path

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
    ) -> GraspServiceBatch:
        endpoint = self.endpoints.anyplace
        if not endpoint:
            raise OSError("backend_unavailable: anyplace")
        metric_depth, depth_quality = self._depth_for_prediction(
            image=image,
            sensor_depth=depth,
            intrinsics=intrinsics,
            depth_source=depth_source,
        )
        response = self._request(
            "POST",
            f"{endpoint.rstrip('/')}/predict_placement",
            {
                "rgb": _rgb_png_payload(image),
                "depth": _depth_png_payload(metric_depth),
                "object_mask": _mask_png_payload(object_mask),
                "placement_region_mask": _mask_png_payload(
                    placement_region_mask
                ),
                "intrinsics": _intrinsics_payload(intrinsics),
                "selected_grasp": dict(selected_grasp),
            },
        )
        details = _openeta_details(response)
        candidates = details.get("placement_candidates")
        if not isinstance(candidates, list) or not candidates:
            raise OSError("AnyPlace returned no placement candidates")
        camera_to_base = np.asarray(
            extrinsics,
            dtype=np.float64,
        ).reshape(4, 4)
        poses = []
        scores = []
        widths: list[float | None] = []
        approaches = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise OSError(
                    "AnyPlace placement candidate must be an object"
                )
            place_grasp = candidate.get("place_grasp_pose")
            if not isinstance(place_grasp, Mapping):
                raise OSError(
                    "AnyPlace candidate is missing place_grasp_pose"
                )
            rotation = np.asarray(
                place_grasp.get("rotation_matrix"),
                dtype=np.float64,
            ).reshape(3, 3)
            translation = np.asarray(
                place_grasp.get("translation_xyz"),
                dtype=np.float64,
            ).reshape(3)
            camera_pose = np.eye(4, dtype=np.float64)
            camera_pose[:3, :3] = rotation
            camera_pose[:3, 3] = translation
            base_pose = camera_to_base @ camera_pose
            poses.append(base_pose)
            scores.append(float(place_grasp.get("score", 0.0)))
            width = place_grasp.get("width")
            widths.append(
                float(width)
                if isinstance(width, (int, float))
                else None
            )
            approach = base_pose[:3, 0]
            approaches.append(
                tuple(
                    float(value)
                    for value in approach
                    / max(float(np.linalg.norm(approach)), 1e-12)
                )
            )
        return GraspServiceBatch(
            poses=np.stack(poses),
            scores=np.asarray(scores, dtype=np.float64),
            widths_m=tuple(widths),
            backend="anyplace",
            public_tool_convention=True,
            grasp_frame="graspnet",
            approach_directions_base=tuple(approaches),
            native_candidates=tuple(
                dict(value)
                for value in candidates
                if isinstance(value, Mapping)
            ),
            depth_quality={
                **depth_quality,
                "backend_metadata": dict(
                    details.get("metadata", {})
                )
                if isinstance(details.get("metadata"), Mapping)
                else {},
            },
        )

    def _depth_for_prediction(
        self,
        *,
        image: np.ndarray,
        sensor_depth: np.ndarray,
        intrinsics: np.ndarray,
        depth_source: str,
        fill_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        sensor = np.asarray(sensor_depth, dtype=np.float64).squeeze()
        valid_sensor = np.isfinite(sensor) & (sensor > 0)
        if depth_source == "sensor":
            return sensor, {
                "source": "sensor",
                "sensor_valid_ratio": float(valid_sensor.mean()),
                "enhanced_pixel_count": 0,
                "collision_source": "sensor",
            }
        endpoint = self.endpoints.unidepth_v2
        if not endpoint:
            raise OSError("backend_unavailable: unidepth_v2")
        response = self._request(
            "POST",
            f"{endpoint.rstrip('/')}/estimate_depth",
            {
                "rgb": _rgb_png_payload(image),
                "intrinsics": {
                    **_intrinsics_payload(intrinsics),
                    "scale": 1.0,
                },
                "camera_id": "selected_detection_camera",
                "camera_model": "pinhole",
                "calibration_profile_id": "",
                "bundle_id": "",
            },
        )
        details = _openeta_details(response)
        prior = _decode_npy(str(details["depth_npy_base64"]))
        confidence = _decode_npy(
            str(details["confidence_npy_base64"])
        )
        if prior.shape != sensor.shape or confidence.shape != sensor.shape:
            raise OSError("UniDepth output shape mismatch")
        valid_prior = np.isfinite(prior) & (prior > 0)
        fill = ~valid_sensor & valid_prior
        if fill_mask is not None:
            mask = np.asarray(fill_mask, dtype=bool).squeeze()
            if mask.shape != sensor.shape:
                raise OSError("depth enhancement mask shape mismatch")
            fill &= mask
        enhanced = sensor.copy()
        enhanced[fill] = prior[fill]
        return enhanced, {
            "source": "sensor_plus_unidepth_v2",
            "sensor_valid_ratio": float(valid_sensor.mean()),
            "prior_valid_ratio": float(valid_prior.mean()),
            "enhanced_pixel_count": int(np.count_nonzero(fill)),
            "enhancement_scope": (
                "target_mask_only"
                if fill_mask is not None
                else "all_missing_sensor_pixels"
            ),
            "confidence_semantics": details.get(
                "confidence_semantics"
            ),
            "collision_source": "sensor",
        }

    def solve_ik(
        self,
        position: Sequence[float],
        quaternion: Sequence[float],
        previous_joints: Sequence[float] | None,
    ) -> Sequence[float] | None:
        response = self._request(
            "POST",
            f"{self.endpoints.pyroki.rstrip('/')}/ik",
            {
                "target_pose_wxyz_xyz": [
                    *[float(value) for value in quaternion],
                    *[float(value) for value in position],
                ],
                "prev_cfg": (
                    [float(value) for value in previous_joints]
                    if previous_joints is not None
                    else None
                ),
            },
        )
        joints = response.get("joint_positions")
        return [float(value) for value in joints] if isinstance(joints, list) else None


def _encode_png(image: np.ndarray) -> str:
    stream = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _rgb_png_payload(image: np.ndarray) -> dict[str, object]:
    return {"format": "png", "base64": _encode_png(image)}


def _depth_png_payload(depth_m: np.ndarray) -> dict[str, object]:
    depth = np.asarray(depth_m, dtype=np.float64).squeeze()
    millimetres = np.zeros(depth.shape, dtype=np.uint16)
    valid = np.isfinite(depth) & (depth > 0)
    millimetres[valid] = np.clip(
        np.rint(depth[valid] * 1000.0),
        1,
        np.iinfo(np.uint16).max,
    ).astype(np.uint16)
    stream = io.BytesIO()
    Image.fromarray(millimetres).save(
        stream,
        format="PNG",
    )
    return {
        "format": "png",
        "base64": base64.b64encode(stream.getvalue()).decode("ascii"),
    }


def _mask_png_payload(mask: np.ndarray) -> dict[str, object]:
    encoded = np.asarray(mask, dtype=bool).astype(np.uint8) * 255
    stream = io.BytesIO()
    Image.fromarray(encoded).save(stream, format="PNG")
    return {
        "format": "png",
        "base64": base64.b64encode(stream.getvalue()).decode("ascii"),
    }


def _intrinsics_payload(
    intrinsics: np.ndarray,
) -> dict[str, float]:
    matrix = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    return {
        "fx": float(matrix[0, 0]),
        "fy": float(matrix[1, 1]),
        "cx": float(matrix[0, 2]),
        "cy": float(matrix[1, 2]),
        "scale": 1000.0,
    }


def _openeta_details(
    response: Mapping[str, object],
) -> Mapping[str, object]:
    if response.get("success") is not True:
        details = response.get("details")
        reason = (
            details.get("reason")
            if isinstance(details, Mapping)
            else "service_error"
        )
        raise OpenEtaServiceError(
            str(reason),
            details if isinstance(details, Mapping) else {},
        )
    details = response.get("details")
    if not isinstance(details, Mapping):
        raise OSError("OpenETA response is missing details")
    return details


def _normalize_openeta_grasp_candidate(
    candidate: Mapping[str, object],
    *,
    gripper_profile: GripperProfile,
) -> dict[str, object]:
    normalized = dict(candidate)
    if "depth" not in normalized:
        gripper_depth = normalized.get("gripper_depth")
        if isinstance(gripper_depth, (int, float)):
            normalized["depth"] = float(gripper_depth)
    finger_height = getattr(gripper_profile, "finger_height_m", None)
    if "height" not in normalized and finger_height is not None:
        normalized["height"] = float(finger_height)
    return normalized


def _depth_to_points(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(depth, dtype=np.float64).squeeze()
    valid = np.isfinite(values) & (values > 0)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(valid)
    if not len(xs):
        return np.empty((0, 3), dtype=np.float64)
    matrix = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    pixels = np.vstack((xs, ys, np.ones(len(xs))))
    camera = np.linalg.solve(matrix, pixels) * values[ys, xs]
    homogeneous = np.vstack((camera, np.ones(len(xs))))
    base = (
        np.asarray(extrinsics, dtype=np.float64).reshape(4, 4)
        @ homogeneous
    )[:3].T
    return base[np.isfinite(base).all(axis=1)]


def _scoped_grasp_collision_depth(
    depth: np.ndarray,
    *,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    relaxed: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    values = np.asarray(depth, dtype=np.float64).squeeze()
    object_mask = np.asarray(mask, dtype=bool).squeeze()
    if values.shape != object_mask.shape or not object_mask.any():
        return values, {
            "collision_scope": "full_depth_fallback",
            "collision_relaxed": bool(relaxed),
        }
    valid = np.isfinite(values) & (values > 0)
    object_valid = valid & object_mask
    if not object_valid.any():
        return values, {
            "collision_scope": "full_depth_fallback",
            "collision_relaxed": bool(relaxed),
        }
    ys, xs = np.nonzero(object_mask)
    height, width = values.shape
    span = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    margin = max(4, int(round(span * (0.25 if relaxed else 0.50))))
    left = max(0, int(xs.min()) - margin)
    right = min(width, int(xs.max()) + margin + 1)
    top = max(0, int(ys.min()) - margin)
    bottom = min(height, int(ys.max()) + margin + 1)
    bbox = np.zeros_like(object_mask)
    bbox[top:bottom, left:right] = True

    matrix = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    rows, cols = np.indices(values.shape)
    camera_x = (cols - matrix[0, 2]) / matrix[0, 0] * values
    camera_y = (rows - matrix[1, 2]) / matrix[1, 1] * values
    camera_points = np.stack(
        (camera_x, camera_y, values, np.ones_like(values)),
        axis=-1,
    )
    base_z = (
        camera_points
        @ np.asarray(extrinsics, dtype=np.float64).reshape(4, 4).T
    )[..., 2]
    object_heights = base_z[object_valid]
    lower = float(np.percentile(object_heights, 5.0)) - (
        0.005 if relaxed else 0.03
    )
    upper = float(np.percentile(object_heights, 95.0)) + 0.10
    keep = valid & bbox & (base_z >= lower) & (base_z <= upper)
    keep |= object_valid
    scoped = np.zeros_like(values)
    scoped[keep] = values[keep]
    return scoped, {
        "collision_scope": (
            "target_bbox_without_below_support"
            if relaxed
            else "target_local_bbox"
        ),
        "collision_relaxed": bool(relaxed),
        "collision_scope_input_points": int(np.count_nonzero(valid)),
        "collision_scope_kept_points": int(np.count_nonzero(keep)),
        "collision_scope_bbox_xyxy": [left, top, right, bottom],
        "collision_scope_min_base_z_m": lower,
        "collision_scope_max_base_z_m": upper,
    }


def _encode_npy(array: np.ndarray) -> str:
    stream = io.BytesIO()
    np.save(stream, np.asarray(array))
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _decode_npy(value: str) -> np.ndarray:
    return np.load(io.BytesIO(base64.b64decode(value)), allow_pickle=False)


def _parse_molmo_point(text: str) -> tuple[float, float, float] | None:
    match = re.search(
        r'<point\b[^>]*\bx=["\']([0-9.]+)["\'][^>]*\by=["\']([0-9.]+)["\']',
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return float(match.group(1)), float(match.group(2)), 100.0
    match = re.search(r'<points\s+coords=["\']([^"\']+)["\']', text, re.IGNORECASE)
    if match:
        values = [float(value) for value in match.group(1).split()]
        if len(values) >= 4:
            return values[2], values[3], 1000.0
    return None

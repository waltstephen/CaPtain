from __future__ import annotations

import base64
import io
from collections import deque

import numpy as np
import pytest
from PIL import Image

from captain_harness.adapters.capx.services import (
    CapXHttpToolServices,
    CapXServiceEndpoints,
    OpenEtaServiceError,
    _normalize_openeta_grasp_candidate,
    _scoped_grasp_collision_depth,
)
from captain_harness.adapters.libero.calibration import GripperProfile


def _encoded_array(value: np.ndarray) -> str:
    stream = io.BytesIO()
    np.save(stream, value)
    return base64.b64encode(stream.getvalue()).decode("ascii")


class ScriptedTransport:
    def __init__(self, responses) -> None:
        self.responses = deque(responses)
        self.calls = []

    def request(self, method, url, *, payload=None, timeout_s=0):
        self.calls.append((method, url, payload, timeout_s))
        value = self.responses.popleft()
        if isinstance(value, BaseException):
            raise value
        return value


def _endpoints() -> CapXServiceEndpoints:
    return CapXServiceEndpoints(
        sam3="http://sam3",
        molmo="http://molmo/v1",
        contact_graspnet="http://grasp",
        pyroki="http://ik",
    )


def test_http_clients_decode_all_four_pinned_service_contracts() -> None:
    mask = np.asarray([[1, 0], [0, 1]], dtype=np.uint8)
    grasp = np.eye(4)[None]
    transport = ScriptedTransport(
        [
            {
                "results": [
                    {
                        "mask_base64": base64.b64encode(mask.tobytes()).decode("ascii"),
                        "shape": [2, 2],
                        "box": [0, 0, 1, 1],
                        "score": 0.8,
                        "label": "bowl",
                    }
                ]
            },
            {"choices": [{"message": {"content": '<point x="50" y="25">'}}]},
            {
                "grasps_base64": _encoded_array(grasp),
                "scores_base64": _encoded_array(np.asarray([0.7])),
                "contact_pts_base64": _encoded_array(np.zeros((1, 3))),
            },
            {"joint_positions": [0.1] * 7 + [0.04]},
        ]
    )
    services = CapXHttpToolServices(_endpoints(), transport=transport, attempts=1)
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    masks = services.segment_text(image, "bowl")
    point = services.point(image, "bowl")
    poses, scores = services.plan_grasps(np.ones((4, 3)), np.ones((3, 3)))
    joints = services.solve_ik([0, 0, 0.4], [1, 0, 0, 0], None)
    assert masks[0].mask.tolist() == [[True, False], [False, True]]
    assert point == (100.0, 25.0)
    assert poses.shape == (1, 4, 4) and scores.tolist() == [0.7]
    assert list(joints) == [0.1] * 7 + [0.04]
    grasp_payload = transport.calls[2][2]
    assert grasp_payload["local_regions"] is True
    assert grasp_payload["filter_grasps"] is True
    assert grasp_payload["forward_passes"] == 2
    assert grasp_payload["max_retries"] == 10
    assert transport.calls[-1][2]["prev_cfg"] is None


def test_service_client_retries_restart_but_preserves_timeout() -> None:
    restarted = ScriptedTransport(
        [
            {"openapi": "3.1"},
            OSError("connection reset"),
            {"models": []},
            {"openapi": "3.1"},
            {"openapi": "3.1"},
        ]
    )
    services = CapXHttpToolServices(_endpoints(), transport=restarted, attempts=2)
    assert services.health()["molmo"]["healthy"] is True
    assert len(restarted.calls) == 5

    timed_out = ScriptedTransport([TimeoutError("deadline")])
    services = CapXHttpToolServices(_endpoints(), transport=timed_out, attempts=3)
    with pytest.raises(TimeoutError, match="deadline"):
        services.point(np.zeros((2, 2, 3), dtype=np.uint8), "bowl")
    assert len(timed_out.calls) == 1


def test_health_checks_all_services_without_substituting_failures() -> None:
    transport = ScriptedTransport(
        [
            {"openapi": "3.1"},
            {"models": []},
            OSError("offline"),
            {"openapi": "3.1"},
        ]
    )
    report = CapXHttpToolServices(_endpoints(), transport=transport, attempts=1).health()
    assert tuple(report) == ("sam3", "molmo", "contact_graspnet", "pyroki")
    assert report["sam3"]["healthy"] is True
    assert report["contact_graspnet"]["healthy"] is False
    assert "offline" in report["contact_graspnet"]["error"]


def test_health_checks_every_configured_openeta_service() -> None:
    revision = "openeta-revision"
    transport = ScriptedTransport(
        [
            {"openapi": "3.1"},
            {"models": []},
            {"openapi": "3.1"},
            {"openapi": "3.1"},
            *[
                {
                    "ok": True,
                    "backend": backend,
                    "openeta_revision": revision,
                    "settings": {},
                }
                for backend in (
                    "contact_graspnet",
                    "anygrasp",
                    "graspgenx",
                    "anyplace",
                    "unidepth_v2",
                )
            ],
        ]
    )
    endpoints = CapXServiceEndpoints(
        sam3="http://sam3",
        molmo="http://molmo/v1",
        contact_graspnet="http://grasp",
        pyroki="http://ik",
        openeta_contact_graspnet="http://openeta-cgn",
        anygrasp="http://anygrasp",
        graspgenx="http://graspgenx",
        anyplace="http://anyplace",
        unidepth_v2="http://unidepth",
        openeta_revision=revision,
    )

    report = CapXHttpToolServices(
        endpoints,
        transport=transport,
        attempts=1,
    ).health()

    assert tuple(report) == (
        "sam3",
        "molmo",
        "contact_graspnet",
        "pyroki",
        "openeta_contact_graspnet",
        "anygrasp",
        "graspgenx",
        "anyplace",
        "unidepth_v2",
    )
    assert all(value["healthy"] is True for value in report.values())


def test_health_rejects_stale_openeta_service_revision() -> None:
    transport = ScriptedTransport(
        [
            {"openapi": "3.1"},
            {"models": []},
            {"openapi": "3.1"},
            {"openapi": "3.1"},
            {
                "ok": True,
                "backend": "anygrasp",
                "openeta_revision": "stale",
                "settings": {},
            },
        ]
    )
    endpoints = CapXServiceEndpoints(
        sam3="http://sam3",
        molmo="http://molmo/v1",
        contact_graspnet="http://grasp",
        pyroki="http://ik",
        anygrasp="http://anygrasp",
        openeta_revision="expected",
    )

    report = CapXHttpToolServices(
        endpoints,
        transport=transport,
        attempts=1,
    ).health()

    assert report["anygrasp"]["healthy"] is False
    assert "revision mismatch" in report["anygrasp"]["error"]


def test_openeta_anygrasp_is_transformed_to_base_without_fallback() -> None:
    transport = ScriptedTransport(
        [
            {
                "success": True,
                "details": {
                    "candidate_count": 1,
                    "grasp_candidates": [
                        {
                            "score": 0.8,
                            "translation_xyz": [0.0, 0.0, 0.5],
                            "rotation_matrix": np.eye(3).tolist(),
                            "width": 0.04,
                            "depth": 0.05,
                            "height": 0.03,
                        }
                    ],
                    "metadata": {"model": "anygrasp"},
                },
            }
        ]
    )
    endpoints = CapXServiceEndpoints(
        sam3="http://sam3",
        molmo="http://molmo/v1",
        contact_graspnet="http://grasp",
        pyroki="http://ik",
        anygrasp="http://anygrasp",
    )
    services = CapXHttpToolServices(
        endpoints,
        transport=transport,
        attempts=1,
    )
    extrinsics = np.eye(4)
    extrinsics[0, 3] = 1.0

    batch = services.predict_grasps(
        backend="anygrasp",
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        depth=np.full((4, 4), 0.5),
        mask=np.ones((4, 4), dtype=bool),
        intrinsics=np.array(
            [[400.0, 0.0, 1.5], [0.0, 400.0, 1.5], [0, 0, 1]]
        ),
        extrinsics=extrinsics,
        gripper_profile=type(
            "Profile",
            (),
            {"max_width_m": 0.08},
        )(),
        depth_source="sensor",
    )

    assert batch.backend == "anygrasp"
    assert batch.grasp_frame == "graspnet"
    assert batch.widths_m == (0.04,)
    assert batch.native_candidates[0]["depth"] == 0.05
    assert batch.native_candidates[0]["height"] == 0.03
    assert np.allclose(batch.poses[0, :3, 3], [1.0, 0.0, 0.5])
    assert services.available_grasp_backends() == (
        "contact_graspnet",
        "anygrasp",
    )
    method, url, payload, _ = transport.calls[0]
    assert method == "POST"
    assert url == "http://anygrasp/detect_grasps"
    assert payload["mode"] == "targeted"
    assert payload["intrinsics"]["scale"] == 1000.0


def test_contact_grasp_candidate_is_normalized_for_anyplace() -> None:
    normalized = _normalize_openeta_grasp_candidate(
        {
            "id": "grasp_000",
            "gripper_depth": 0.1034,
            "width": 0.04,
        },
        gripper_profile=GripperProfile(
            "franka",
            0.08,
            0.0584,
            0.018,
        ),
    )

    assert normalized["depth"] == 0.1034
    assert normalized["height"] == 0.018


def test_graspgenx_relaxed_collision_scope_is_tighter() -> None:
    depth = np.full((60, 60), 0.5, dtype=np.float64)
    mask = np.zeros((60, 60), dtype=bool)
    mask[20:40, 20:40] = True
    intrinsics = np.array(
        [[100.0, 0.0, 30.0], [0.0, 100.0, 30.0], [0.0, 0.0, 1.0]]
    )

    normal, normal_meta = _scoped_grasp_collision_depth(
        depth,
        mask=mask,
        intrinsics=intrinsics,
        extrinsics=np.eye(4),
        relaxed=False,
    )
    relaxed, relaxed_meta = _scoped_grasp_collision_depth(
        depth,
        mask=mask,
        intrinsics=intrinsics,
        extrinsics=np.eye(4),
        relaxed=True,
    )

    assert np.count_nonzero(relaxed) < np.count_nonzero(normal)
    assert relaxed_meta["collision_relaxed"] is True
    assert normal_meta["collision_relaxed"] is False


def test_openeta_structured_failure_preserves_reason_and_metadata() -> None:
    transport = ScriptedTransport(
        [
            {
                "success": False,
                "details": {
                    "reason": "insufficient_object_points",
                    "metadata": {"object_point_count": 0},
                },
            }
        ]
    )
    services = CapXHttpToolServices(
        CapXServiceEndpoints(anygrasp="http://anygrasp"),
        transport=transport,
        attempts=1,
    )

    with pytest.raises(OpenEtaServiceError) as raised:
        services.predict_grasps(
            backend="anygrasp",
            image=np.zeros((2, 2, 3), dtype=np.uint8),
            depth=np.full((2, 2), 0.5),
            mask=np.ones((2, 2), dtype=bool),
            intrinsics=np.eye(3),
            extrinsics=np.eye(4),
            gripper_profile=type(
                "Profile",
                (),
                {"max_width_m": 0.08},
            )(),
            depth_source="sensor",
        )

    assert raised.value.reason == "insufficient_object_points"
    assert raised.value.details["metadata"]["object_point_count"] == 0


def test_unidepth_enhancement_keeps_sensor_as_collision_source() -> None:
    prior = np.full((2, 2), 0.7, dtype=np.float32)
    confidence = np.ones((2, 2), dtype=np.float32)
    transport = ScriptedTransport(
        [
            {
                "success": True,
                "details": {
                    "depth_npy_base64": _encoded_array(prior),
                    "confidence_npy_base64": _encoded_array(confidence),
                    "confidence_semantics": "higher_is_better",
                },
            },
            {
                "success": True,
                "details": {
                    "candidate_count": 1,
                    "grasp_candidates": [
                        {
                            "score": 0.9,
                            "translation_xyz": [0.0, 0.0, 0.7],
                            "rotation_matrix": np.eye(3).tolist(),
                            "width": 0.05,
                        }
                    ],
                    "metadata": {},
                },
            },
        ]
    )
    endpoints = CapXServiceEndpoints(
        sam3="http://sam3",
        molmo="http://molmo/v1",
        contact_graspnet="http://grasp",
        pyroki="http://ik",
        graspgenx="http://graspgenx",
        unidepth_v2="http://unidepth",
    )
    services = CapXHttpToolServices(
        endpoints,
        transport=transport,
        attempts=1,
    )

    batch = services.predict_grasps(
        backend="graspgenx",
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.array([[0.5, np.nan], [0.5, 0.5]]),
        mask=np.ones((2, 2), dtype=bool),
        intrinsics=np.array(
            [[400.0, 0.0, 0.5], [0.0, 400.0, 0.5], [0, 0, 1]]
        ),
        extrinsics=np.eye(4),
        gripper_profile=type(
            "Profile",
            (),
            {"max_width_m": 0.08},
        )(),
        depth_source="enhanced",
    )

    assert batch.depth_quality["enhanced_pixel_count"] == 1
    assert batch.depth_quality["enhancement_scope"] == "target_mask_only"
    assert batch.depth_quality["collision_source"] == "sensor"
    assert transport.calls[0][1] == "http://unidepth/estimate_depth"
    assert transport.calls[1][1] == "http://graspgenx/predict_grasps"


def test_enhanced_anygrasp_fills_only_target_and_disables_collision_filter() -> None:
    prior = np.full((2, 2), 0.7, dtype=np.float32)
    confidence = np.ones((2, 2), dtype=np.float32)
    transport = ScriptedTransport(
        [
            {
                "success": True,
                "details": {
                    "depth_npy_base64": _encoded_array(prior),
                    "confidence_npy_base64": _encoded_array(confidence),
                    "confidence_semantics": "higher_is_better",
                },
            },
            {
                "success": True,
                "details": {
                    "grasp_candidates": [
                        {
                            "score": 0.9,
                            "translation_xyz": [0.0, 0.0, 0.7],
                            "rotation_matrix": np.eye(3).tolist(),
                            "width": 0.05,
                        }
                    ],
                    "metadata": {},
                },
            },
        ]
    )
    endpoints = CapXServiceEndpoints(
        sam3="http://sam3",
        molmo="http://molmo/v1",
        contact_graspnet="http://grasp",
        pyroki="http://ik",
        anygrasp="http://anygrasp",
        unidepth_v2="http://unidepth",
    )
    services = CapXHttpToolServices(
        endpoints,
        transport=transport,
        attempts=1,
    )

    services.predict_grasps(
        backend="anygrasp",
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.array([[np.nan, np.nan], [0.5, 0.5]]),
        mask=np.array([[True, False], [False, False]]),
        intrinsics=np.array(
            [[400.0, 0.0, 0.5], [0.0, 400.0, 0.5], [0, 0, 1]]
        ),
        extrinsics=np.eye(4),
        gripper_profile=type(
            "Profile",
            (),
            {"max_width_m": 0.08},
        )(),
        depth_source="enhanced",
    )

    payload = transport.calls[1][2]
    encoded_depth = base64.b64decode(payload["depth"]["base64"])
    depth_mm = np.asarray(Image.open(io.BytesIO(encoded_depth)))
    assert depth_mm.tolist() == [[700, 0], [500, 500]]
    assert payload["collision_detection"] is False


def test_openeta_anyplace_returns_base_frame_place_grasps() -> None:
    transport = ScriptedTransport(
        [
            {
                "success": True,
                "details": {
                    "placement_candidates": [
                        {
                            "id": "placement_000",
                            "place_grasp_pose": {
                                "score": 0.6,
                                "translation_xyz": [0.0, 0.0, 0.4],
                                "rotation_matrix": np.eye(3).tolist(),
                                "width": 0.04,
                            },
                        }
                    ],
                    "metadata": {"model": "anyplace"},
                },
            }
        ]
    )
    endpoints = CapXServiceEndpoints(
        sam3="http://sam3",
        molmo="http://molmo/v1",
        contact_graspnet="http://grasp",
        pyroki="http://ik",
        anyplace="http://anyplace",
    )
    services = CapXHttpToolServices(
        endpoints,
        transport=transport,
        attempts=1,
    )
    extrinsics = np.eye(4)
    extrinsics[1, 3] = 0.5
    selected_grasp = {
        "id": "grasp_000",
        "frame": "camera",
        "camera_frame": "opencv",
        "score": 0.8,
        "translation_xyz": [0.0, 0.0, 0.3],
        "rotation_matrix": np.eye(3).tolist(),
        "gripper_tip_position_xyz": [0.05, 0.0, 0.3],
        "depth": 0.05,
        "width": 0.04,
        "height": 0.03,
    }

    batch = services.predict_placements(
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.full((2, 2), 0.5),
        object_mask=np.ones((2, 2), dtype=bool),
        placement_region_mask=np.ones((2, 2), dtype=bool),
        intrinsics=np.array(
            [[400.0, 0.0, 0.5], [0.0, 400.0, 0.5], [0, 0, 1]]
        ),
        extrinsics=extrinsics,
        selected_grasp=selected_grasp,
        depth_source="sensor",
    )

    assert batch.backend == "anyplace"
    assert batch.grasp_frame == "graspnet"
    assert np.allclose(batch.poses[0, :3, 3], [0.0, 0.5, 0.4])
    assert batch.widths_m == (0.04,)
    assert transport.calls[0][1] == "http://anyplace/predict_placement"

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class GripperProfile:
    profile_id: str
    max_width_m: float
    fingertip_depth_m: float
    finger_height_m: float | None = None

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise ValueError("gripper profile_id must not be empty")
        if (
            not math.isfinite(self.max_width_m)
            or not math.isfinite(self.fingertip_depth_m)
            or self.max_width_m <= 0
            or self.fingertip_depth_m <= 0
            or (
                self.finger_height_m is not None
                and (
                    not math.isfinite(self.finger_height_m)
                    or self.finger_height_m <= 0
                )
            )
        ):
            raise ValueError("gripper dimensions must be finite and positive")

    def public(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "max_width_m": self.max_width_m,
            "fingertip_depth_m": self.fingertip_depth_m,
        }


@dataclass(frozen=True, slots=True)
class LiberoCalibrationProfile:
    schema_version: str
    profile_id: str
    frame: str
    calibration_residual_m: float
    residual_basis: str
    tcp_offset_local_m: tuple[float, float, float]
    pyroki_hand_to_libero_eef_4x4: tuple[float, ...]
    graspnet_to_libero_eef_4x4: tuple[float, ...] | None
    graspnet_calibration: Mapping[str, object] | None
    gripper: GripperProfile
    source_path: str
    sha256: str

    def __post_init__(self) -> None:
        if self.schema_version not in {
            "captain-calibration/v1",
            "captain-calibration/v2",
        }:
            raise ValueError(
                "unsupported calibration schema: "
                f"{self.schema_version!r}"
            )
        if not self.profile_id.strip() or not self.frame.strip():
            raise ValueError("calibration profile_id and frame must not be empty")
        if (
            not math.isfinite(self.calibration_residual_m)
            or self.calibration_residual_m < 0
        ):
            raise ValueError(
                "calibration_residual_m must be finite and non-negative"
            )
        if not self.residual_basis.strip():
            raise ValueError("calibration residual_basis must not be empty")
        tcp = np.asarray(self.tcp_offset_local_m, dtype=np.float64)
        transform = np.asarray(
            self.pyroki_hand_to_libero_eef_4x4,
            dtype=np.float64,
        )
        if tcp.shape != (3,) or not np.isfinite(tcp).all():
            raise ValueError("tcp_offset_local_m must contain three finite values")
        if transform.shape != (16,) or not np.isfinite(transform).all():
            raise ValueError(
                "pyroki_hand_to_libero_eef_4x4 must contain 16 finite values"
            )
        matrix = transform.reshape(4, 4)
        if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0)):
            raise ValueError("calibration transform must be homogeneous")
        if self.schema_version == "captain-calibration/v2":
            if self.graspnet_to_libero_eef_4x4 is None:
                raise ValueError(
                    "V2 calibration requires graspnet_to_libero_eef_4x4"
                )
            grasp_transform = np.asarray(
                self.graspnet_to_libero_eef_4x4,
                dtype=np.float64,
            )
            if (
                grasp_transform.shape != (16,)
                or not np.isfinite(grasp_transform).all()
                or not np.allclose(
                    grasp_transform.reshape(4, 4)[3],
                    (0.0, 0.0, 0.0, 1.0),
                )
            ):
                raise ValueError(
                    "graspnet_to_libero_eef_4x4 must be homogeneous"
                )
            if not isinstance(self.graspnet_calibration, Mapping):
                raise ValueError(
                    "V2 calibration requires graspnet_calibration metadata"
                )
        if not self.sha256.strip():
            raise ValueError("calibration sha256 must not be empty")

    @property
    def hand_to_eef_transform(self) -> np.ndarray:
        return np.asarray(
            self.pyroki_hand_to_libero_eef_4x4,
            dtype=np.float64,
        ).reshape(4, 4)

    @property
    def graspnet_to_eef_transform(self) -> np.ndarray | None:
        return (
            np.asarray(
                self.graspnet_to_libero_eef_4x4,
                dtype=np.float64,
            ).reshape(4, 4)
            if self.graspnet_to_libero_eef_4x4 is not None
            else None
        )

    def public_metadata(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "frame": self.frame,
            "calibration_residual_m": self.calibration_residual_m,
            "residual_basis": self.residual_basis,
            "source_path": self.source_path,
            "sha256": self.sha256,
            "gripper": self.gripper.public(),
            "graspnet_calibration": (
                dict(self.graspnet_calibration)
                if self.graspnet_calibration is not None
                else None
            ),
        }


def default_calibration_profile_path() -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "configs"
        / "calibration"
        / "libero_franka_v1.json"
    )


def load_calibration_profile(
    path: Path | str | None = None,
) -> LiberoCalibrationProfile:
    source = (
        default_calibration_profile_path()
        if path is None
        else Path(path).expanduser().resolve()
    )
    raw = source.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("calibration profile must be a JSON object")
    gripper_payload = payload.get("gripper")
    if not isinstance(gripper_payload, Mapping):
        raise ValueError("calibration profile is missing gripper metadata")
    return LiberoCalibrationProfile(
        schema_version=str(payload.get("schema_version", "")),
        profile_id=str(payload.get("profile_id", "")),
        frame=str(payload.get("frame", "")),
        calibration_residual_m=float(
            payload.get("calibration_residual_m", math.nan)
        ),
        residual_basis=str(payload.get("residual_basis", "")),
        tcp_offset_local_m=tuple(
            float(value) for value in payload.get("tcp_offset_local_m", ())
        ),  # type: ignore[arg-type]
        pyroki_hand_to_libero_eef_4x4=tuple(
            float(value)
            for value in payload.get(
                "pyroki_hand_to_libero_eef_4x4",
                (),
            )
        ),
        graspnet_to_libero_eef_4x4=(
            tuple(
                float(value)
                for value in payload.get(
                    "graspnet_to_libero_eef_4x4",
                    (),
                )
            )
            if payload.get("graspnet_to_libero_eef_4x4")
            is not None
            else None
        ),
        graspnet_calibration=(
            dict(payload["graspnet_calibration"])
            if isinstance(
                payload.get("graspnet_calibration"),
                Mapping,
            )
            else None
        ),
        gripper=GripperProfile(
            profile_id=str(gripper_payload.get("profile_id", "")),
            max_width_m=float(gripper_payload.get("max_width_m", math.nan)),
            fingertip_depth_m=float(
                gripper_payload.get("fingertip_depth_m", math.nan)
            ),
            finger_height_m=(
                float(gripper_payload["finger_height_m"])
                if gripper_payload.get("finger_height_m") is not None
                else None
            ),
        ),
        source_path=str(source),
        sha256=hashlib.sha256(raw).hexdigest(),
    )

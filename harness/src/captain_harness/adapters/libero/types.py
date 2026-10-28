from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RobotState:
    """Public LIBERO robot state in radians and the robot base frame."""

    joint_positions_rad: tuple[float, ...]
    end_effector_position_m: tuple[float, float, float]
    end_effector_quaternion_wxyz: tuple[float, float, float, float]
    gripper_open_fraction: float
    joint_velocity_norm_rad_s: float = 0.0


@dataclass(frozen=True, slots=True)
class CameraData:
    camera_name: str
    rgb_path: str
    depth_path: str | None
    intrinsics_3x3: tuple[float, ...]
    extrinsics_4x4: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ObservationData:
    cameras: tuple[CameraData, ...]
    robot_state: RobotState | None
    time_s: float | None = None

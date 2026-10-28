from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from captain_harness.adapters.libero.calibration import (
    default_calibration_profile_path,
    load_calibration_profile,
)


def test_default_libero_calibration_profile_is_externalized_and_hashed() -> None:
    path = default_calibration_profile_path()
    profile = load_calibration_profile(path)

    assert path == Path("configs/calibration/libero_franka_v1.json").resolve()
    assert profile.profile_id == "libero-franka-capx-v1"
    assert profile.frame == "robot_base"
    assert profile.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert profile.tcp_offset_local_m == (0.0, 0.0, -0.1)
    assert profile.gripper.max_width_m == 0.08


def test_v10_calibration_includes_pinned_graspnet_to_eef_transform() -> None:
    profile = load_calibration_profile(
        "configs/calibration/libero_franka_v2.json"
    )

    assert profile.schema_version == "captain-calibration/v2"
    assert profile.profile_id == "libero-franka-openeta-v2"
    assert profile.graspnet_to_eef_transform is not None
    assert profile.graspnet_to_eef_transform.shape == (4, 4)
    assert (
        profile.graspnet_calibration["openeta_revision"]
        == "8989fbe936109364e71b20032b8dc870df87a855"
    )
    assert profile.gripper.fingertip_depth_m == 0.0584
    assert profile.gripper.finger_height_m == 0.018
    assert profile.hand_to_eef_transform.shape == (4, 4)
    assert np.allclose(
        profile.hand_to_eef_transform[3],
        (0.0, 0.0, 0.0, 1.0),
    )


def test_v10_frame_corrected_calibration_targets_libero_grip_site() -> None:
    source = load_calibration_profile(
        "configs/calibration/libero_franka_v2.json"
    )
    corrected = load_calibration_profile(
        "configs/calibration/libero_franka_v3.json"
    )

    assert (
        corrected.profile_id
        == "libero-franka-openeta-frame-corrected-v3"
    )
    assert corrected.graspnet_to_eef_transform is not None
    assert source.graspnet_to_eef_transform is not None
    assert np.allclose(
        corrected.graspnet_to_eef_transform,
        source.graspnet_to_eef_transform
        @ source.hand_to_eef_transform,
    )
    assert np.allclose(
        corrected.graspnet_to_eef_transform[:3, 3],
        (0.1001, 0.0, 0.0),
    )

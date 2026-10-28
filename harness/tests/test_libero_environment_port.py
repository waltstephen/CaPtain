from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from captain_harness.adapters.libero import LiberoEnvironmentAdapter
from captain_harness.adapters.libero.geometry import DepthUnavailableError, pixel_to_world
from captain_harness.ports.environment import EnvironmentPort, HostCompletion

from test_libero_adapter import FakeEnv


def test_pixel_to_world_uses_finite_5x5_median_and_base_extrinsics() -> None:
    depth = np.full((7, 7), np.nan, dtype=np.float64)
    depth[1:6, 1:6] = np.arange(1.0, 26.0).reshape(5, 5)
    intrinsics = np.array([[2.0, 0.0, 3.0], [0.0, 2.0, 3.0], [0.0, 0.0, 1.0]])
    camera_to_base = np.eye(4)
    camera_to_base[:3, 3] = (0.1, -0.2, 0.3)
    point = pixel_to_world(depth, intrinsics, camera_to_base, x_norm=0.5, y_norm=0.5)
    assert np.allclose(point, (0.1, -0.2, 13.3))


def test_pixel_to_world_fails_when_patch_has_no_valid_depth() -> None:
    with pytest.raises(DepthUnavailableError, match="no finite positive depth"):
        pixel_to_world(
            np.full((8, 8), np.nan),
            np.eye(3),
            np.eye(4),
            x_norm=0.5,
            y_norm=0.5,
        )


def test_pixel_to_world_keeps_full_5x5_window_at_image_edge() -> None:
    depth = np.full((6, 6), np.nan)
    depth[4, 4] = 7.0
    point = pixel_to_world(
        depth,
        np.eye(3),
        np.eye(4),
        x_norm=0.0,
        y_norm=0.0,
    )
    assert point == (0.0, 0.0, 7.0)


def _adapter(tmp_path: Path):
    env = FakeEnv()
    calls = []
    adapter = LiberoEnvironmentAdapter(
        env,
        init_states=np.stack([np.zeros(7), np.full(7, 0.05)]),
        observation_dir=tmp_path,
        horizon=40,
        final_predicate=lambda: calls.append("score") or True,
        backend_options={"motion_max_steps": 5, "gripper_steps": 1},
    )
    return adapter, calls


def test_libero_port_publishes_versioned_manifest_only_after_reset(tmp_path: Path) -> None:
    adapter, _ = _adapter(tmp_path)
    with pytest.raises(RuntimeError, match="reset"):
        adapter.capabilities()
    observation = adapter.reset(seed=1)
    assert isinstance(adapter, EnvironmentPort)
    assert observation.step == 0
    assert [image.camera for image in observation.images] == [
        "agentview",
        "robot0_eye_in_hand",
    ]
    manifest = adapter.capabilities()
    assert manifest.schema_version == "captain-capabilities/v1"
    assert manifest.names == (
        "get_robot_state",
        "get_observation_data",
        "pixel_to_world",
        "move_to_pose",
        "move_by",
        "move_to_joints",
        "open_gripper",
        "close_gripper",
        "home",
    )
    assert manifest.require("move_by").units["delta_xyz"] == "metre"
    assert manifest.require("move_by").coordinate_frames["delta_xyz"] == "robot_base"
    assert "check_success" not in manifest.names
    assert "finalize" not in manifest.names


def test_libero_port_routes_public_capabilities_and_returns_json_values(tmp_path: Path) -> None:
    adapter, _ = _adapter(tmp_path)
    adapter.reset(seed=0)
    state = adapter.execute("get_robot_state", {})
    assert state.success and isinstance(state.value, dict)
    assert len(state.value["joint_positions_rad"]) == 7
    projected = adapter.execute(
        "pixel_to_world", {"camera": "agentview", "x_norm": 0.5, "y_norm": 0.5}
    )
    assert projected.success and len(projected.value) == 3
    motion = adapter.execute("move_by", {"delta_xyz": [0.01, 0.0, 0.0]})
    assert motion.success
    unknown = adapter.execute("read_object_truth", {})
    assert not unknown.success
    assert unknown.failure_category.value == "not_executed"


def test_libero_hidden_predicate_is_host_only_and_single_use(tmp_path: Path) -> None:
    adapter, calls = _adapter(tmp_path)
    adapter.reset(seed=0)
    assert calls == []
    assert not hasattr(adapter, "host_completion")
    assert adapter.finalize() == HostCompletion(None, True, None)
    assert calls == ["score"]
    with pytest.raises(RuntimeError, match="only once"):
        adapter.finalize()


def test_libero_hidden_audit_samples_are_separate_from_final_score(
    tmp_path: Path,
) -> None:
    adapter, calls = _adapter(tmp_path)
    observation = adapter.reset(seed=0)

    assert adapter.audit_hidden_success() is True
    assert calls == ["score"]
    assert adapter.finalize() == HostCompletion(
        True,
        True,
        observation.step,
    )
    assert calls == ["score", "score"]


def test_libero_adapter_ignores_transient_done_and_scores_only_final_state(
    tmp_path: Path,
) -> None:
    class DoneSequenceEnv(FakeEnv):
        def __init__(self):
            super().__init__()
            self.done_values = iter((False,) * 5 + (False, True, False))

        def step(self, action):
            observation, reward, _, info = super().step(action)
            return observation, reward, next(self.done_values), info

    env = DoneSequenceEnv()
    adapter = LiberoEnvironmentAdapter(
        env,
        init_states=[env.joints.copy()],
        observation_dir=tmp_path,
        horizon=10,
        final_predicate=lambda: False,
        backend_options={"gripper_steps": 3},
    )
    adapter.reset(seed=0)

    assert adapter.execute("open_gripper", {}).success
    assert adapter.finalize() == HostCompletion(None, False, None)

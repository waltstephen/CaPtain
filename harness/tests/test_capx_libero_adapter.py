from __future__ import annotations

from pathlib import Path

from captain_harness.adapters.capx.environment import CapXLiberoEnvironmentAdapter
from captain_harness.adapters.capx.manifest import ToolApiLevel
from captain_harness.ports.capability import FailureCategory
from captain_harness.ports.environment import HostCompletion

from test_libero_adapter import FakeEnv
from test_tool_backend import FakeServices


def test_capx_libero_adapter_selects_versioned_level_and_rotates_episode_handles(
    tmp_path: Path,
) -> None:
    native = FakeEnv()
    adapter = CapXLiberoEnvironmentAdapter(
        native,
        init_states=[native.joints.copy()],
        observation_dir=tmp_path / "observations",
        horizon=100,
        final_predicate=lambda: True,
        services=FakeServices(),
        api_level=ToolApiLevel.S3_LOW_LEVEL,
        backend_options={"motion_max_steps": 5},
    )
    first = adapter.reset(0)
    assert first.public_state["seed"] == 0
    assert adapter.capabilities().provider == "capx-s3-composition/v5"
    observation = adapter.execute("get_observation", {})
    assert observation.success
    old_handle = observation.value["cameras"][0]["rgb_handle"]

    adapter.reset(1)
    invalid = adapter.execute(
        "segment_text", {"rgb_handle": old_handle, "text_prompt": "bowl"}
    )
    assert invalid.failure_category is FailureCategory.INVALID_ARGUMENT
    assert "belongs to episode" in invalid.message or "registry is closed" in invalid.message


def test_capx_libero_adapter_keeps_host_predicate_outside_visible_tools(tmp_path: Path) -> None:
    native = FakeEnv()
    adapter = CapXLiberoEnvironmentAdapter(
        native,
        init_states=[native.joints.copy()],
        observation_dir=tmp_path / "observations",
        horizon=100,
        final_predicate=lambda: True,
        services=FakeServices(),
        api_level=ToolApiLevel.S2_HIGH_LEVEL,
        backend_options={"motion_max_steps": 5},
    )
    adapter.reset(0)
    assert "finalize" not in adapter.capabilities().names
    assert "check_success" not in adapter.capabilities().names
    assert not hasattr(adapter, "host_completion")
    assert adapter.finalize() == HostCompletion(None, True, None)


def test_capx_libero_adapter_keeps_action_audit_host_only(
    tmp_path: Path,
) -> None:
    native = FakeEnv()
    calls = []
    adapter = CapXLiberoEnvironmentAdapter(
        native,
        init_states=[native.joints.copy()],
        observation_dir=tmp_path / "observations",
        horizon=100,
        final_predicate=lambda: calls.append("score") or True,
        services=FakeServices(),
        api_level=ToolApiLevel.S2_HIGH_LEVEL,
        backend_options={"motion_max_steps": 5},
    )
    observation = adapter.reset(0)

    assert adapter.audit_hidden_success() is True
    assert "audit_hidden_success" not in adapter.capabilities().names
    assert adapter.finalize() == HostCompletion(
        True,
        True,
        observation.step,
    )
    assert calls == ["score", "score"]

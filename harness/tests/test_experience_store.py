from __future__ import annotations

import json
from pathlib import Path

import pytest

from captain_harness.core.experience import (
    EnvironmentExperienceStore,
    ExperienceKey,
    ExperienceMode,
    ExperienceSnapshot,
)


def _key(**overrides: object) -> ExperienceKey:
    values = {
        "environment_id": "libero-v1",
        "environment_revision": "libero-commit",
        "tool_revisions": {
            "openeta": "openeta-commit",
            "cap_x": "capx-commit",
        },
        "source_tree_sha256": "source-tree",
    }
    values.update(overrides)
    return ExperienceKey(**values)  # type: ignore[arg-type]


def _entry() -> dict[str, object]:
    return {
        "object_class": "bowl",
        "observed_extent_m": [0.109, 0.110, 0.046],
        "gripper_max_width_m": 0.08,
        "attempts": 34,
        "by_backend": {
            "contact_graspnet": {
                "candidates": 0.43,
                "attached": 0.12,
            }
        },
        "by_style": {
            "top_down": 0.08,
            "side": 0.31,
            "auto": 0.22,
        },
        "width_measurable_rate": 0.19,
        "notes": "rim grasp was more effective",
    }


def test_experience_store_build_freeze_and_read_only_warm(
    tmp_path: Path,
) -> None:
    build = EnvironmentExperienceStore.create_build(
        tmp_path,
        _key(),
        built_from=(
            {"suite": "libero_goal", "task_id": 4, "seed": 0},
        ),
    )
    build.append(_entry())
    frozen = build.freeze()

    assert frozen.mode is ExperienceMode.WARM
    assert frozen.public_metadata()["read_only"] is True
    assert frozen.entries_for_object_classes(("bowl",)) == (
        _entry(),
    )
    reopened = EnvironmentExperienceStore.open_frozen(
        tmp_path,
        _key(),
    )
    assert reopened.snapshot().store_sha256 == frozen.store_sha256
    with pytest.raises(PermissionError, match="read-only"):
        reopened.append(_entry())


def test_experience_store_rejects_environment_or_source_mismatch(
    tmp_path: Path,
) -> None:
    build = EnvironmentExperienceStore.create_build(tmp_path, _key())
    build.append(_entry())
    build.freeze()

    with pytest.raises(ValueError, match="key does not match"):
        EnvironmentExperienceStore.open_frozen(
            tmp_path,
            _key(source_tree_sha256="different-source"),
        )
    with pytest.raises(ValueError, match="path-safe"):
        _key(environment_id="../robocasa")


def test_experience_store_rejects_task_solution_content(
    tmp_path: Path,
) -> None:
    build = EnvironmentExperienceStore.create_build(tmp_path, _key())
    polluted = _entry()
    polluted["policy"] = "pick the known task target"

    with pytest.raises(ValueError, match="forbidden experience field"):
        build.append(polluted)


def test_experience_store_rejects_unapproved_or_invalid_statistics(
    tmp_path: Path,
) -> None:
    build = EnvironmentExperienceStore.create_build(tmp_path, _key())
    extra = _entry()
    extra["task_name"] = "put the bowl on the stove"
    with pytest.raises(ValueError, match="approved schema"):
        build.append(extra)

    invalid_rate = _entry()
    invalid_rate["by_backend"]["contact_graspnet"]["attached"] = 1.5
    with pytest.raises(ValueError, match="valid range"):
        build.append(invalid_rate)

    policy_note = _entry()
    policy_note["notes"] = "policy: always use the known task solution"
    with pytest.raises(ValueError, match="tool-efficacy text"):
        build.append(policy_note)


def test_cold_experience_snapshot_is_empty_and_deterministic() -> None:
    first = ExperienceSnapshot.cold(_key())
    second = ExperienceSnapshot.cold(_key())

    assert first.entries == ()
    assert first.store_sha256 == second.store_sha256
    payload = json.loads(first.context_text())
    assert payload["experience_mode"] == "cold"
    assert payload["entries"] == []

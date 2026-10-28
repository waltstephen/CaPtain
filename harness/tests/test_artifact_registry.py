from __future__ import annotations

import pytest

from captain_harness.adapters.capx.artifacts import ArtifactHandleError, EpisodeArtifactRegistry


class Clock:
    def __init__(self) -> None:
        self.value = 10.0

    def __call__(self) -> float:
        return self.value


def test_artifact_registry_keeps_values_opaque_and_episode_local() -> None:
    clock = Clock()
    first = EpisodeArtifactRegistry("episode-a", ttl_s=30.0, clock=clock)
    second = EpisodeArtifactRegistry("episode-b", ttl_s=30.0, clock=clock)
    handle = first.put("rgb", {"large": [1, 2, 3]})
    assert handle.startswith("artifact://v1/episode-a/rgb/")
    assert first.get(handle, expected_kind="rgb") == {"large": [1, 2, 3]}
    with pytest.raises(ArtifactHandleError, match="belongs to episode 'episode-a'"):
        second.get(handle, expected_kind="rgb")


def test_artifact_registry_expiry_unknown_kind_and_close_are_explicit() -> None:
    clock = Clock()
    registry = EpisodeArtifactRegistry("episode-a", ttl_s=2.0, clock=clock)
    handle = registry.put("mask", object())
    with pytest.raises(ArtifactHandleError, match="kind 'mask'"):
        registry.get(handle, expected_kind="rgb")
    with pytest.raises(ArtifactHandleError, match="unknown artifact handle"):
        registry.get("artifact://v1/episode-a/mask/not-present")
    clock.value = 13.0
    with pytest.raises(ArtifactHandleError, match="expired"):
        registry.get(handle)
    fresh = registry.put("mask", object())
    registry.close()
    with pytest.raises(ArtifactHandleError, match="registry is closed"):
        registry.get(fresh)

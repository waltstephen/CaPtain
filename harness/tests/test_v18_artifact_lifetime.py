"""Slow providers must not erase episode-owned identity and source evidence."""
import pytest

from captain_harness.adapters.capx import environment as environment_module
from captain_harness.adapters.capx.artifacts import ArtifactHandleError, EpisodeArtifactRegistry
from captain_harness.adapters.capx.environment import CapXLiberoEnvironmentAdapter
from captain_harness.adapters.capx.manifest import ToolApiLevel
from test_artifact_registry import Clock
from test_libero_adapter import FakeEnv
from test_tool_backend import FakeServices


def test_episode_lifetime_keeps_identity_dependencies_until_close():
    clock = Clock()
    registry = EpisodeArtifactRegistry("long-episode", ttl_s=None, clock=clock)
    source = registry.put("observation", {"step": 5, "time_s": 0.25})
    mask = registry.put("mask", {"source": source})
    track = registry.put("entity_track", {"mask": mask, "source": source})
    clock.value += 7200
    identity = registry.get(track, expected_kind="entity_track")
    assert registry.get(identity["mask"], expected_kind="mask")["source"] == source
    # Keeping the historical handle does not make the measurement current.
    assert registry.get(source, expected_kind="observation") == {"step": 5, "time_s": 0.25}
    with pytest.raises(ArtifactHandleError, match="expected kind"):
        registry.get(track, expected_kind="observation")
    other = EpisodeArtifactRegistry("next-episode", ttl_s=None, clock=clock)
    with pytest.raises(ArtifactHandleError, match="belongs to episode"):
        other.get(track)
    registry.close()
    for handle in (source, mask, track):
        with pytest.raises(ArtifactHandleError, match="closed"):
            registry.get(handle)


def make_adapter(tmp_path, monkeypatch, *, ttl=None, explicit=False, fail_final=False):
    clock = Clock()
    original = environment_module.EpisodeArtifactRegistry
    monkeypatch.setattr(environment_module, "EpisodeArtifactRegistry",
                        lambda episode_id, **kw: original(episode_id, clock=clock, **kw))
    native = FakeEnv()
    services = FakeServices()
    scores = []

    def final_predicate():
        scores.append("final")
        if fail_final:
            raise RuntimeError("synthetic finalization failure")
        return True

    adapter = CapXLiberoEnvironmentAdapter(
        native, init_states=[native.joints.copy()], observation_dir=tmp_path,
        horizon=100, final_predicate=final_predicate, services=services,
        api_level=ToolApiLevel.S3_LOW_LEVEL, backend_options={"motion_max_steps": 5},
        **({"artifact_ttl_s": ttl} if explicit else {}),
    )
    adapter.reset(0)
    observed = adapter.execute("get_observation", {})
    assert observed.success
    rgb = observed.value["cameras"][0]["rgb_handle"]
    return adapter, clock, rgb, scores, services


@pytest.mark.parametrize("fail_final", [False, True])
def test_adapter_default_survives_long_episode_but_finalization_revokes(tmp_path, monkeypatch, fail_final):
    adapter, clock, rgb, scores, services = make_adapter(
        tmp_path, monkeypatch, fail_final=fail_final)
    clock.value += 3601
    old_evidence = adapter.execute("segment_text", {"rgb_handle": rgb, "text_prompt": "object"})
    assert old_evidence.success, old_evidence.message
    assert services.calls == ["sam3_text"]
    assert scores == []
    if fail_final:
        with pytest.raises(RuntimeError, match="synthetic finalization"):
            adapter.finalize()
    else:
        adapter.finalize()
    revoked = adapter.execute("segment_text", {"rgb_handle": rgb, "text_prompt": "object"})
    assert not revoked.success and not revoked.executed
    assert "registry is closed" in revoked.message
    assert scores == ["final"]
    assert services.calls == ["sam3_text"]


def test_adapter_explicit_expiry_remains_available(tmp_path, monkeypatch):
    adapter, clock, rgb, scores, services = make_adapter(
        tmp_path, monkeypatch, ttl=30, explicit=True)
    clock.value += 31
    expired = adapter.execute("segment_text", {"rgb_handle": rgb, "text_prompt": "object"})
    assert not expired.success and not expired.executed
    assert "expired" in expired.message
    assert not scores and not services.calls

"""r5b step-overhead fixes: critic rows featurized as episodes land (cache), checkpoint weights staged off the run-dir mount."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import physics_critic as pc  # noqa: E402


def test_prefetch_fills_the_cache_and_live_rows_uses_it(monkeypatch):
    calls = []

    def fake_states(d):
        calls.append(d)
        return [{"request_index": 2, "replayed": False, "x": 1}, {"request_index": 1, "replayed": False, "x": 0},
                {"request_index": 0, "replayed": True, "x": 9}], None

    monkeypatch.setattr(pc.cf, "episode_states", fake_states)
    pc.ROWS_CACHE.clear()
    pc.prefetch_rows("/ep/a")
    assert calls == ["/ep/a"] and [r["request_index"] for r in pc.ROWS_CACHE["/ep/a"]] == [1, 2]
    critic = pc.FrozenCritic.__new__(pc.FrozenCritic)
    rows = critic.live_rows("/ep/a")
    assert calls == ["/ep/a"]                       # served from the cache, no second read
    rows[0]["x"] = 99                               # callers may mutate their copy
    assert pc.ROWS_CACHE["/ep/a"][0]["x"] == 0
    assert [r["request_index"] for r in critic.live_rows("/ep/b")] == [1, 2] and calls[-1] == "/ep/b"   # miss: direct read


def test_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(pc.cf, "episode_states", lambda d: ([{"request_index": 0, "replayed": False}], None))
    monkeypatch.setattr(pc, "ROWS_CACHE_MAX", 3)
    pc.ROWS_CACHE.clear()
    for i in range(5):
        pc.prefetch_rows(f"/ep/{i}")
    assert list(pc.ROWS_CACHE) == ["/ep/2", "/ep/3", "/ep/4"]


def test_staging_links_locally_and_finishes_in_the_run_dir(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import train as T

    live = tmp_path / "live" / "run"
    live.mkdir(parents=True)
    (live / "model.safetensors").write_bytes(b"w" * 10)
    out = tmp_path / "runs" / "run"
    out.mkdir(parents=True)
    monkeypatch.setattr(T, "_same_fs", lambda a, b: False)   # the run dir is another filesystem (an sshfs mount)
    staged = T.stage_checkpoint_weights(live, out, 4)
    assert staged.parent == live.parent                 # linked next to the live dir, not copied over the mount
    (live / "model.safetensors").unlink()               # the live dir moves on (save_full swaps in new files)
    final = T.finish_staged_weights(staged, out, 4)
    assert final.parent == out and (final / "model.safetensors").read_bytes() == b"w" * 10 and not staged.exists()


def test_prune_keeps_previous_state_with_flag(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import train as T

    for k in (16, 18, 20, 22):
        (tmp_path / f"checkpoint-{k}" / "fsdp").mkdir(parents=True)
    (tmp_path / "checkpoint-20.corrupt" / "fsdp").mkdir(parents=True)     # not a checkpoint: left alone
    monkeypatch.setenv("RL_KEEP_PREV_FULL_STATE", "1")
    T.prune_full_states(tmp_path, tmp_path / "checkpoint-22", lambda m: None)
    left = sorted(p.parent.name for p in tmp_path.glob("*/fsdp"))
    assert left == ["checkpoint-20", "checkpoint-20.corrupt", "checkpoint-22"]
    monkeypatch.delenv("RL_KEEP_PREV_FULL_STATE")
    T.prune_full_states(tmp_path, tmp_path / "checkpoint-22", lambda m: None)
    assert sorted(p.parent.name for p in tmp_path.glob("*/fsdp")) == ["checkpoint-20.corrupt", "checkpoint-22"]

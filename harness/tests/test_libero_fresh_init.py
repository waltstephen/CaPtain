from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from captain_harness.adapters.libero.backend import reset_to_seeded_init_state
from captain_harness.adapters.libero.fresh_init import (
    FRESH_SCHEMA,
    FRESH_SEED_BASE,
    fresh_init_path,
    load_fresh_init_states,
)

from test_libero_adapter import FakeEnv


def write_fresh(path: Path, states: np.ndarray, seeds: list[int]) -> Path:
    torch = pytest.importorskip("torch")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(states, path)
    meta = {"schema": FRESH_SCHEMA, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "reset_seeds": seeds}
    path.with_name(path.name + ".json").write_text(json.dumps(meta))
    return path


def official_states() -> np.ndarray:
    return np.stack([np.full(7, value) for value in (0.1, 0.2, 0.3)])


def test_official_seeds_keep_the_modulo_mapping_even_with_fresh_states_loaded(tmp_path: Path):
    fresh = load_fresh_init_states(write_fresh(tmp_path / "t.fresh_init", np.full((2, 7), 0.9), [5001, 5002]))
    for seed in (0, 4, 11, 12, 13, 14, 15, 16, 49, 999):
        env = FakeEnv()
        result = reset_to_seeded_init_state(env, official_states(), seed=seed, fresh_init_states=fresh)
        assert result.init_state_source == "official"
        assert result.init_state_index == seed % 3
        assert env.seed_calls == [seed]
        assert np.array_equal(result.observation["robot0_joint_pos"], official_states()[seed % 3])


def test_fresh_seed_selects_fresh_state_and_resets_with_its_generation_seed(tmp_path: Path):
    states = np.stack([np.full(7, 0.7), np.full(7, 0.8)])
    fresh = load_fresh_init_states(write_fresh(tmp_path / "t.fresh_init", states, [1017, 1042]), expected_dim=7)
    env = FakeEnv()
    result = reset_to_seeded_init_state(env, official_states(), seed=FRESH_SEED_BASE + 1, fresh_init_states=fresh)
    assert result.init_state_source == "fresh"
    assert result.init_state_index == 1
    assert env.seed_calls == [1042]
    assert np.array_equal(result.observation["robot0_joint_pos"], states[1])


def test_fresh_seed_never_falls_back_to_an_official_state(tmp_path: Path):
    with pytest.raises(ValueError, match="no fresh init file"):
        reset_to_seeded_init_state(FakeEnv(), official_states(), seed=FRESH_SEED_BASE)
    fresh = load_fresh_init_states(write_fresh(tmp_path / "t.fresh_init", np.full((2, 7), 0.9), [1000, 1001]))
    with pytest.raises(IndexError):
        reset_to_seeded_init_state(FakeEnv(), official_states(), seed=FRESH_SEED_BASE + 2, fresh_init_states=fresh)


def test_fresh_file_is_checked_against_its_sidecar(tmp_path: Path):
    path = write_fresh(tmp_path / "t.fresh_init", np.full((2, 7), 0.9), [1000, 1001])
    with pytest.raises(ValueError, match="dimension"):
        load_fresh_init_states(path, expected_dim=8)
    meta = json.loads(path.with_name(path.name + ".json").read_text())
    path.with_name(path.name + ".json").write_text(json.dumps({**meta, "reset_seeds": [1000]}))
    with pytest.raises(ValueError):
        load_fresh_init_states(path)
    path.with_name(path.name + ".json").write_text(json.dumps({**meta, "reset_seeds": [11, 12]}))
    with pytest.raises(ValueError, match=">= 1000"):
        load_fresh_init_states(path)
    path.with_name(path.name + ".json").write_text(json.dumps({**meta, "sha256": "0" * 64}))
    with pytest.raises(ValueError, match="sha256"):
        load_fresh_init_states(path)


def test_fresh_init_path_is_outside_the_official_init_dir():
    path = fresh_init_path(Path("/h/data/libero_fresh_init"), "libero_10", "TASK")
    assert path == Path("/h/data/libero_fresh_init/libero_10/TASK.fresh_init")

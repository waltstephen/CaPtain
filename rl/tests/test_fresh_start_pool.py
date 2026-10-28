"""Fresh start pool: fresh initial states only (seeds >= 1000), drawn without replacement per epoch; the
official evaluation seeds 11-16 and every official state (seed < 1000) can never be sampled or posted."""
import hashlib
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fresh_start_bank import build  # noqa: E402
from segment_bank import load_bank, prompt_starts, SCHEMA_VERSION, bank_sha256  # noqa: E402
from segment_curriculum import EVAL_SEEDS, FRESH_SEED_BASE, SegmentCurriculum, check_fresh_population  # noqa: E402

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TASKS = [f"{s}:{t}" for s in SUITES for t in range(10)]


def write_fresh(tmp_path: Path, n_per_task: dict[str, int]) -> tuple[Path, dict[str, str]]:
    fresh = tmp_path / "libero_fresh_init"
    task_map = {}
    for key, n in n_per_task.items():
        rel = f"folder_{key.split(':')[0]}/task_{key.split(':')[1]}"
        task_map[key] = rel
        path = fresh / f"{rel}.fresh_init"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps([[0.0] * 4] * n))      # content only matters through its sha256 here
        sidecar = {"schema": "libero-fresh-init/v1", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "reset_seeds": [100000 + i for i in range(n)]}
        path.with_name(path.name + ".json").write_text(json.dumps(sidecar))
    return fresh, task_map


def curriculum(bank, seed=17):
    return SegmentCurriculum(bank, horizon=4, episode_fraction=1.0, mode="uniform", unique_states=True, budget=40,
                             fresh_horizon=40, handoff="clean", reward_mode="final_success_binary_v1",
                             train_all_on_success=True, seed=seed)


@pytest.fixture()
def pool(tmp_path):
    fresh, task_map = write_fresh(tmp_path, {k: 50 + (i % 3) * 25 for i, k in enumerate(TASKS)})
    bank, manifest = build(task_map, fresh, min_per_task=50)
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(bank))
    return load_bank(path), manifest


def test_bank_holds_only_fresh_seeds_for_all_40_tasks(pool):
    bank, manifest = pool
    seeds = [t["seed"] for t in bank["trajectories"].values()]
    assert min(seeds) >= FRESH_SEED_BASE and not set(seeds) & set(EVAL_SEEDS)
    assert manifest["tasks"] == 40 and manifest["per_task_min"] == 50 and manifest["per_task_max"] == 100
    assert manifest["bank_sha256"] == bank["sha256"]
    assert all(t["requests"] == 0 and t["source_dir"] is None for t in bank["trajectories"].values())


def test_without_replacement_per_epoch_and_never_official(pool):
    bank, _ = pool
    cur = curriculum(bank)
    n = len(cur.prompts)
    seen_epochs = []
    for _ in range(3):
        batch = []
        while len(batch) < n:                 # streaming draws in chunks of the batch size
            batch += cur.next_batch(min(256, n - len(batch)))
        keys = [(e["suite"], e["task_id"], e["seed"]) for e in batch]
        assert len(set(keys)) == n            # every fresh state exactly once per epoch
        assert all(e["K"] == 0 and e["seed"] >= FRESH_SEED_BASE and e["seed"] not in EVAL_SEEDS for e in batch)
        assert all(e["horizon"] == 40 and "handoff" not in e for e in batch)   # whole fresh episodes, no teacher replay
        seen_epochs.append(keys)
    assert seen_epochs[0] != seen_epochs[1]   # reshuffled between epochs
    # consecutive steps (256 each) never repeat a state inside an epoch
    first = seen_epochs[0]
    assert not set(first[:256]) & set(first[256:512])


def test_curriculum_resume_keeps_the_epoch(pool):
    bank, _ = pool
    a = curriculum(bank)
    a.next_batch(300)
    b = curriculum(bank)
    b.load_state_dict(json.loads(json.dumps(a.state_dict(), default=list)))
    assert [e["seed"] for e in a.next_batch(50)] == [e["seed"] for e in b.next_batch(50)]


@pytest.mark.parametrize("seed", [*EVAL_SEEDS, 0, 1, 6, 49, 999])
def test_population_with_an_official_or_eval_seed_is_refused(seed):
    trajectories, _ = prompt_starts([("libero_goal", 3, 1000), ("libero_goal", 3, seed)])
    with pytest.raises(ValueError, match="official/evaluation"):
        check_fresh_population({t["trajectory_id"]: t for t in trajectories})


def test_teacher_sourced_trajectories_are_refused():
    t = {"trajectory_id": "x", "suite": "libero_goal", "task_id": 1, "seed": 1003, "requests": 7, "source_dir": "/ep"}
    with pytest.raises(ValueError, match="teacher source"):
        check_fresh_population({"x": t})


def test_missing_tasks_or_short_tasks_are_refused(tmp_path):
    fresh, task_map = write_fresh(tmp_path, {k: 50 for k in TASKS[:39]})
    bank, _ = build(task_map, fresh, min_per_task=50)
    with pytest.raises(ValueError, match="39 tasks"):
        check_fresh_population(bank["trajectories"], min_tasks=40)
    fresh2, task_map2 = write_fresh(tmp_path / "b", {"libero_10:5": 10})
    with pytest.raises(ValueError, match="fewer than 50"):
        build(task_map2, fresh2, min_per_task=50)


def test_tampered_fresh_file_is_refused(tmp_path):
    fresh, task_map = write_fresh(tmp_path, {"libero_goal:0": 50})
    f = fresh / f"{task_map['libero_goal:0']}.fresh_init"
    f.write_bytes(f.read_bytes() + b"x")
    with pytest.raises(ValueError, match="sidecar"):
        build(task_map, fresh, min_per_task=50)


def test_trainer_post_guard_refuses_official_seeds(monkeypatch):
    torch = pytest.importorskip("torch")  # noqa: F841
    import train as T

    posted = []
    monkeypatch.setattr(T.requests, "post", lambda *a, **k: posted.append(k) or (_ for _ in ()).throw(RuntimeError("net")))
    monkeypatch.setitem(T.FRESH_ONLY, "on", True)
    for seed in [*EVAL_SEEDS, 1, 999]:
        with pytest.raises(RuntimeError, match="fresh-seeds-only"):
            T._post_rollout({"entry": {"suite": "libero_goal", "task_id": 0, "seed": seed, "K": 0}, "_route": "/segment"})
        with pytest.raises(RuntimeError, match="fresh-seeds-only"):
            T._post_rollout({"suite": "libero_goal", "task_id": 0, "seed": seed})
    assert posted == []                       # nothing reached the env server
    T._assert_fresh({"entry": {"seed": 1000}})  # a fresh seed passes

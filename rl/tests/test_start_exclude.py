"""Thin-basket fix: start-exclude/v1 training-start exclusion, default off.

Contracts: off = byte-identical curriculum behaviour and state; on = excluded starts are never drawn, everything else of the
population/order/rng is kept so a run can switch it on at a K = 4 resume boundary; fresh_run validates it, passes it to the
trainer from the config, hashes the file, and relocate accepts it only for a pinned sha256 (or switching it off)."""
import copy
import hashlib
import json
import socket
import sys
import tempfile
import time
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fresh_run as run  # noqa: E402
from segment_bank import bank_sha256, prompt_starts  # noqa: E402
from segment_curriculum import SegmentCurriculum, load_start_exclude, load_start_weights  # noqa: E402

PROMPTS = [(suite, task, 1000 + k) for suite in ("libero_object", "libero_goal") for task in range(3) for k in range(6)]


def make_bank():
    trajectories, starts = prompt_starts(PROMPTS)
    bank = {"schema_version": 1, "trajectories": {t["trajectory_id"]: t for t in trajectories},
            "starts": {s["start_id"]: s for s in starts}, "skipped_sources": 0}
    bank["sha256"] = bank_sha256(bank)
    return bank


def write_exclude(path, keys, **extra):
    path.write_text(json.dumps({"schema": "start-exclude/v1", "exclude": list(keys), **extra}))
    return path


THIN = [f"libero_object:{t}:{1000 + k}" for t in range(3) for k in (0, 2, 5)]


def key(entry):
    return f'{entry["suite"]}:{entry["task_id"]}:{entry["seed"]}'


def curriculum(**kw):
    return SegmentCurriculum(make_bank(), episode_fraction=1.0, budget=40, seed=17, **kw)


def test_load_start_exclude_validates(tmp_path):
    assert load_start_exclude(None) is None
    spec = load_start_exclude(write_exclude(tmp_path / "x.json", THIN))
    assert spec["exclude"] == frozenset(THIN) and spec["sha256"] == hashlib.sha256((tmp_path / "x.json").read_bytes()).hexdigest()
    for bad in ({"schema": "start-weights/v1", "exclude": THIN}, {"schema": "start-exclude/v1", "exclude": []},
                {"schema": "start-exclude/v1", "exclude": ["libero_object:0"]}, {"schema": "start-exclude/v1", "exclude": [3]},
                {"schema": "start-exclude/v1", "exclude": THIN + THIN[:1]}):
        (tmp_path / "bad.json").write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            load_start_exclude(tmp_path / "bad.json")


def test_default_off_is_unchanged():
    a, b = curriculum(), curriculum(start_exclude=None)
    assert "start_exclude_sha" not in a.state_dict() and a.state_dict() == b.state_dict()
    assert [key(e) for e in a.next_batch(100)] == [key(e) for e in b.next_batch(100)]
    assert a.excluded == frozenset()


def test_excluded_starts_are_never_drawn_and_the_rest_cover_every_epoch(tmp_path):
    c = curriculum(start_exclude=load_start_exclude(write_exclude(tmp_path / "x.json", THIN)))
    assert len(c.excluded) == len(THIN)
    drawn = Counter(key(e) for e in c.next_batch(27 * 10))
    assert not set(drawn) & set(THIN)
    kept = {f"{s}:{t}:{seed}" for s, t, seed in PROMPTS} - set(THIN)
    assert set(drawn) == kept and set(drawn.values()) == {10}          # without replacement per epoch over the remaining starts
    assert c.prompts == curriculum().prompts                            # population (and so the resume check) unchanged


def test_exclusion_with_start_weights(tmp_path):
    w = tmp_path / "w.json"
    w.write_text(json.dumps({"schema": "start-weights/v1", "weights": {"libero_goal:0:1000": 0.02}, "default": 1.0}))
    c = curriculum(start_weights=load_start_weights(w), start_exclude=load_start_exclude(write_exclude(tmp_path / "x.json", THIN)))
    drawn = Counter(key(e) for e in c.next_batch(2000))
    assert not set(drawn) & set(THIN) and drawn["libero_goal:0:1000"] < 0.1 * drawn["libero_goal:0:1001"]


def test_unknown_or_task_emptying_exclusions_are_refused(tmp_path):
    with pytest.raises(ValueError, match="not in the training population"):
        curriculum(start_exclude=load_start_exclude(write_exclude(tmp_path / "x.json", ["libero_object:0:1999"])))
    whole = [f"libero_object:1:{1000 + k}" for k in range(6)]
    with pytest.raises(ValueError, match="removes every start"):
        curriculum(start_exclude=load_start_exclude(write_exclude(tmp_path / "y.json", whole)))


def test_resume_boundary_switches_the_exclusion_on_and_keeps_order_and_rng(tmp_path):
    live = curriculum()
    live.next_batch(40)                                     # some history: a K = 4 boundary checkpoint mid-epoch
    state = json.loads(json.dumps(live.state_dict(), default=list))   # as saved (tuples become lists)
    spec = load_start_exclude(write_exclude(tmp_path / "x.json", THIN))
    resumed = curriculum(start_exclude=spec)
    resumed.load_state_dict(state)
    assert resumed.start_exclude_previous == "none" and resumed.cursor == live.cursor and resumed.order == live.order
    expected = [key(e) for e in live.next_batch(200) if key(e) not in THIN]
    got = [key(e) for e in resumed.next_batch(len(expected))]
    assert got == expected                                  # same order, thin turns skipped
    # its own state round-trips and records the exclusion; switching back off (rollback) is reported, not refused
    again = curriculum(start_exclude=spec)
    again.load_state_dict(resumed.state_dict())
    assert again.start_exclude_previous is None and resumed.state_dict()["start_exclude_sha"] == spec["sha256"]
    off = curriculum()
    off.load_state_dict(resumed.state_dict())
    assert off.start_exclude_previous == spec["sha256"]


class StartExcludeLauncherTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        nodes = copy.deepcopy(run.DEFAULTS["nodes"])
        nodes["a"].update(hostname=socket.gethostname().split(".")[0], host="127.0.0.1")
        for name in run.SOURCE_FILES:
            path = self.root / "repo" / "rl" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# fixture\n")
        for name in ("server/env_server.py", "adapters/libero/hidden_object_audit.py", "adapters/libero/backend.py"):
            path = self.root / "harness" / "src/captain_harness" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# fixture\n")
        (self.root / "harness/scripts").mkdir(parents=True)
        (self.root / "harness/scripts/react_episode.py").write_text("# fixture\n")
        for name in ("configs/calibration/libero_franka_v3.json", "third_party/manifest.json"):    # hashed under python-runtime/v1
            (self.root / "harness" / name).parent.mkdir(parents=True, exist_ok=True)
            (self.root / "harness" / name).write_text("{}")
        (self.root / "sft").mkdir()
        (self.root / "sft/config.json").write_text("{}")
        self.bank = self.root / "bank.json"
        self.bank.write_text(json.dumps(make_bank()))
        self.overrides = {"run_dir": str(self.root / "run"), "repo": str(self.root / "repo"), "harness": str(self.root / "harness"),
                          "base_model": str(self.root / "sft"), "nodes": nodes, "segment_bank": str(self.bank),
                          "runtime_tool_contract": "python-runtime/v1"}
        self.manifest_path = self.root / "run" / "manifest.json"
        self.manifest = run.initialize(self.manifest_path, self.overrides, "a")
        self.cfg = run.normalize_config(self.overrides)
        self.exclude = write_exclude(self.root / "thin.json", THIN)

    def stop_roles(self):
        (self.root / "run" / "metrics.jsonl").write_text('{"step": 116}\n')
        for node, role in [("a", "train"), ("a", "env")]:
            run.atomic_json(run.record_path(self.cfg, node, role), {"stopped_at": time.time(), "node": node, "role": role})

    def test_off_by_default_and_passed_hashed_when_set(self):
        self.assertIsNone(self.cfg["start_exclude"])
        self.assertNotIn("--start-exclude", run.role_commands(self.cfg, "a")["train"])
        cfg = run.normalize_config(dict(self.overrides, start_exclude=str(self.exclude)))
        train = run.role_commands(cfg, "a")["train"]
        self.assertEqual(train[train.index("--start-exclude") + 1], str(self.exclude.resolve()))
        self.assertIn(str(self.exclude.resolve()), run.source_hashes(cfg))
        with self.assertRaisesRegex(run.RunError, "not in the segment bank"):
            run.normalize_config(dict(self.overrides, start_exclude=str(write_exclude(self.root / "u.json", ["libero_10:0:1000"]))))
        with self.assertRaisesRegex(run.RunError, "segment curriculum"):
            run.normalize_config(dict(self.overrides, segment_bank=None, start_exclude=str(self.exclude)))

    def test_relocate_accepts_only_a_pinned_exclusion_and_logs_it(self):
        self.stop_roles()
        planned = dict(self.overrides, start_exclude=str(self.exclude))
        with self.assertRaisesRegex(run.RunError, "START_EXCLUDE_ALLOWED"):
            run.relocate(self.manifest_path, planned, "a")
        sha = hashlib.sha256(self.exclude.read_bytes()).hexdigest()
        with mock.patch.dict(run.START_EXCLUDE_ALLOWED, {sha: "fixture"}):
            with self.assertRaisesRegex(run.RunError, "these would change: lr"):
                run.relocate(self.manifest_path, dict(planned, lr=2e-5), "a")
            moved = run.relocate(self.manifest_path, planned, "a")
        rec = moved["relocations"][-1]
        self.assertEqual(rec["changed"], ["start_exclude"])
        self.assertEqual(rec["start_exclude_allowance"], {"sha256": sha, "what": "fixture"})
        self.assertEqual(rec["changed_sources"], [str(self.exclude.resolve())])
        # switching it off again (rollback) needs no allowance
        back = run.relocate(self.manifest_path, self.overrides, "a")
        self.assertEqual(back["relocations"][-1]["changed"], ["start_exclude"])
        self.assertNotIn("start_exclude_allowance", back["relocations"][-1])


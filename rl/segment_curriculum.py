"""Reverse curriculum over segment starts (per-call RL).

Each source trajectory (a successful teacher episode in the segment bank) has a frontier f: rollouts start at
K = N - f (N = the trajectory's python requests), i.e. f = 1 lets the policy make only the teacher's last call.
When at least ``advance_fraction`` of a group reaches the criterion of reward_segment.reached(), the frontier moves
``frontier_step`` requests earlier, until K = 0 (the fresh episode). A fixed share of the slots always starts fresh
(K = 0) so the policy keeps seeing its own complete episodes; prompt-only trajectories (no source) are always K = 0.

mode="uniform" draws K uniformly over the trajectory's requests instead (every teacher call is a start from the first step;
the per-call reward is a dense potential difference, so the manipulation calls need not wait for the frontier to reach them).
active_fraction (uniform mode) draws that share of starts among the teacher calls that changed the goal potential
(teacher_delta != 0; most calls only perceive). sourced_only drops the prompt-only trajectories. A start whose replay diverged from its source (every rollout of the group
was replay_infidelity) is remembered and not drawn again in uniform mode.

The window a rollout is scored over is ``horizon`` live requests; ``fresh_horizon`` (default: horizon) is the window of
the K = 0 starts, so a large value makes them complete episodes (to the finish or the request cap) instead of one request
from the initial state. ``active_min_delta`` > 0 makes an active start one whose teacher call made at least that much
progress (instead of any nonzero, possibly negative, change). ``suites`` restricts the population to those LIBERO suites;
``exclude`` holds trajectories out (the held-out half of a probe set, see load_probe_set). ``focus_starts`` (start ids)
makes every non-fresh slot cycle through that fixed list instead: the same states are practised repeatedly, so a probe of
them shows whether a fixed capability is learnable (the fresh K = 0 slots still come from the whole population). A start
listed several times is drawn that many times per cycle (a weighted practice pool).

``budget`` B (the RFC "remaining task" protocol) makes every teacher start a whole-remaining-task attempt: the live window is
B - K requests (the episode's total request budget stays B, as for a fresh episode), starts with K > B - ``min_live`` are not
used, and neither are starts where the teacher had already satisfied every goal (the expected mark at K + 1): the student
must still have work to do. In reverse mode the initial frontier of each trajectory is then its last such unsolved start
(not K = N - 1, which the teacher has always already solved), and the frontier only moves over eligible starts. Focus starts
that are solved or do not fit the budget are rejected. ``handoff`` ("continuation" or "clean") is passed with each teacher
start to the rollout service: clean keeps only the physical state at K (fresh Python session, workspace, conversation).

The state (order, cursor, frontier) is part of the training checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from reward_segment import reached
from segment_bank import start_for


def solved_at(start: dict[str, Any]) -> bool:
    """Whether every goal was already satisfied when the teacher handed over at this start (expected mark K + 1)."""
    states = (((start.get("expected") or {}).get("mark") or {}).get("goal_states")) or {}
    return bool(states) and all(bool((s or {}).get("satisfied")) for s in states.values())


def population(bank: dict[str, Any], sourced_only: bool = False, suites: Iterable[str] | None = None,
               exclude: Iterable[str] = (), unique_states: bool = False) -> dict[str, dict[str, Any]]:
    """The trajectories a curriculum samples from (also the prompt population a resume must match). unique_states keeps
    one trajectory per official initial state (suite, task, seed): fresh episodes of two teacher runs from the same
    state are the same training prompt and would otherwise be drawn twice as often."""
    suites = set(suites) if suites else None
    exclude = set(exclude)
    pool = {tid: t for tid, t in bank["trajectories"].items()
            if (t.get("requests") or not sourced_only) and (suites is None or t.get("suite") in suites) and tid not in exclude}
    if unique_states:
        seen: dict[tuple, str] = {}
        for tid in sorted(pool):
            t = pool[tid]
            seen.setdefault((t.get("suite"), int(t.get("task_id")), int(t.get("seed"))), tid)
        pool = {tid: pool[tid] for tid in seen.values()}
    return pool


# Training starts only from FRESH LIBERO initial states. The harness maps
# seed = FRESH_SEED_BASE + k to fresh state k of the task's <task>.fresh_init file and never falls back to an official state;
# every seed below FRESH_SEED_BASE is an official state (init_states[seed % 50]), among them the official evaluation
# seeds 11-16.
FRESH_SEED_BASE = 1000
EVAL_SEEDS = (11, 12, 13, 14, 15, 16)


def is_fresh_seed(seed: int) -> bool:
    return int(seed) >= FRESH_SEED_BASE


def check_fresh_population(trajectories: dict[str, dict[str, Any]], *, eval_seeds: Iterable[int] = EVAL_SEEDS,
                           min_tasks: int | None = None) -> dict[str, Any]:
    """--fresh-seeds-only: every trajectory a curriculum may draw is a prompt-only K = 0 start (no teacher source, so no
    official-state replay) on a fresh seed >= FRESH_SEED_BASE; an official or evaluation seed anywhere is an error.
    Returns a summary (states, tasks, per-task min/max)."""
    bad = sorted({(t.get("suite"), int(t.get("task_id")), int(t.get("seed"))) for t in trajectories.values()
                  if not is_fresh_seed(int(t.get("seed"))) or int(t.get("seed")) in set(eval_seeds)})
    if bad:
        raise ValueError(f"fresh-seeds-only: {len(bad)} official/evaluation states in the start population, e.g. {bad[:3]}")
    sourced = sorted(tid for tid, t in trajectories.items() if t.get("requests") or t.get("source_dir"))
    if sourced:
        raise ValueError(f"fresh-seeds-only: {len(sourced)} trajectories carry a teacher source (official-state replays), e.g. {sourced[:3]}")
    per_task: dict[tuple, int] = defaultdict(int)
    for t in trajectories.values():
        per_task[(t.get("suite"), int(t.get("task_id")))] += 1
    if min_tasks is not None and len(per_task) < int(min_tasks):
        raise ValueError(f"fresh-seeds-only: {len(per_task)} tasks in the start population, need {min_tasks}")
    return {"states": len(trajectories), "tasks": len(per_task), "per_task_min": min(per_task.values(), default=0),
            "per_task_max": max(per_task.values(), default=0)}


PROBE_SCHEMA = "segment-probe-set/v1"


START_WEIGHTS_SCHEMA = "start-weights/v1"
MIN_START_WEIGHT = 0.02


def load_start_weights(path: str | Path | None) -> dict[str, Any] | None:
    """start-weights/v1 {"weights": {"suite:task:seed": w}, "default": w} with every w in [MIN_START_WEIGHT, 1] (bounds the
    rejection loop); the file's sha256 is kept so a resume refuses changed weights."""
    if not path:
        return None
    raw = Path(path).read_bytes()
    spec = json.loads(raw)
    if spec.get("schema") != START_WEIGHTS_SCHEMA or not isinstance(spec.get("weights"), dict):
        raise ValueError(f"{path}: not a {START_WEIGHTS_SCHEMA} file")
    values = list(spec["weights"].values()) + [spec.get("default")]
    if any(type(w) not in (int, float) or not MIN_START_WEIGHT <= w <= 1.0 for w in values):
        raise ValueError(f"{path}: start weights must be numbers in [{MIN_START_WEIGHT}, 1]")
    return {"weights": spec["weights"], "default": float(spec["default"]), "sha256": hashlib.sha256(raw).hexdigest()}


START_EXCLUDE_SCHEMA = "start-exclude/v1"


def load_start_exclude(path: str | Path | None) -> dict[str, Any] | None:
    """start-exclude/v1 {"exclude": ["suite:task:seed", ...], "rule": ...}: training starts the segment curriculum never
    draws (thin-basket fix: starts whose host success predicate scores a correct placement
    as failure). The population, order and rng are unchanged (an excluded start is skipped when its turn comes), so the
    file can be introduced or changed at a resume boundary; its sha256 is kept in the curriculum state and a change is
    reported by load_state_dict."""
    if not path:
        return None
    raw = Path(path).read_bytes()
    spec = json.loads(raw)
    keys = spec.get("exclude") if spec.get("schema") == START_EXCLUDE_SCHEMA else None
    if not isinstance(keys, list) or not keys:
        raise ValueError(f"{path}: not a {START_EXCLUDE_SCHEMA} file with a non-empty exclude list")
    for key in keys:
        parts = key.split(":") if isinstance(key, str) else []
        if len(parts) != 3 or not parts[1].isdigit() or not parts[2].isdigit():
            raise ValueError(f"{path}: exclude entries are 'suite:task:seed', got {key!r}")
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path}: duplicate exclude entries")
    return {"exclude": frozenset(keys), "sha256": hashlib.sha256(raw).hexdigest()}


def load_probe_set(path: str | Path | None, bank: dict[str, Any] | None = None) -> dict[str, Any]:
    """{"sets": {name: [start_id, ...]}, "heldout_trajectories": [...]}; empty when no path. An optional "practice"
    {name: [start_id, ...]} holds focus lists that are not probed; a start may repeat there to weight it."""
    if not path:
        return {"sets": {}, "heldout_trajectories": []}
    probe = json.loads(Path(path).read_text(encoding="utf-8"))
    if probe.get("schema_version") != PROBE_SCHEMA or not isinstance(probe.get("sets"), dict):
        raise ValueError(f"not a {PROBE_SCHEMA} file: {path}")
    if bank is not None:
        if probe.get("bank_sha256") != bank["sha256"]:
            raise ValueError("probe set belongs to a different segment bank")
        missing = [s for ids in [*probe["sets"].values(), *(probe.get("practice") or {}).values()] for s in ids if s not in bank["starts"]]
        if missing:
            raise ValueError(f"probe starts missing from the bank: {missing[:3]}")
    return probe


class SegmentCurriculum:
    def __init__(self, bank: dict[str, Any], *, horizon: int = 1, episode_fraction: float = 0.25, frontier_step: int = 1,
                 advance_fraction: float = 0.5, initial_frontier: int = 1, seed: int = 17, mode: str = "reverse",
                 sourced_only: bool = False, active_fraction: float = 0.0, fresh_horizon: int | None = None,
                 active_min_delta: float = 0.0, suites: Iterable[str] | None = None, exclude: Iterable[str] = (),
                 focus_starts: Iterable[str] = (), budget: int | None = None, min_live: int = 4,
                 handoff: str = "continuation", unique_states: bool = False, reward_mode: str | None = None,
                 train_all_on_success: bool = False, start_weights: dict[str, Any] | None = None,
                 start_exclude: dict[str, Any] | None = None) -> None:
        if mode not in ("reverse", "uniform"):
            raise ValueError(f"unknown curriculum mode: {mode}")
        if handoff not in ("continuation", "clean"):
            raise ValueError(f"unknown handoff: {handoff}")
        self.budget = int(budget) if budget else None
        self.min_live = int(min_live)
        self.handoff = handoff
        self.reward_mode = reward_mode                       # carried by every entry (the env server scores with it)
        self.train_all_on_success = bool(train_all_on_success)
        self.bank = bank
        self._starts = {(st["trajectory_id"], int(st["K"])): st for st in bank["starts"].values()}
        self.mode = mode
        self.sourced_only = bool(sourced_only)
        self.active_fraction = float(active_fraction)
        self.fresh_horizon = int(fresh_horizon) if fresh_horizon else int(horizon)
        self.active_min_delta = float(active_min_delta)
        self.suites = sorted(set(suites)) if suites else None
        self.exclude = sorted(set(exclude))
        self.focus = list(focus_starts)     # a repeated start is drawn proportionally more often
        missing = [s for s in self.focus if s not in bank["starts"]]
        if missing:
            raise ValueError(f"focus starts missing from the bank: {missing[:3]}")
        unfit = [s for s in self.focus if not self.eligible(bank["starts"][s])]
        if unfit:
            raise ValueError(f"focus starts already solved or beyond the budget: {unfit[:3]}")
        self.horizon = int(horizon)
        self.episode_fraction = float(episode_fraction)
        self.frontier_step = int(frontier_step)
        self.advance_fraction = float(advance_fraction)
        self.unique_states = bool(unique_states)
        self.trajectories = population(bank, self.sourced_only, self.suites, self.exclude, self.unique_states)
        if not self.trajectories:
            raise ValueError("segment bank has no trajectories with a source")
        self.prompts = sorted(self.trajectories)
        # prioritized starts (start-weights/v1): a drawn trajectory is kept with probability w(suite:task:seed), else the
        # cycle moves on; None = uniform
        self.start_weights_sha = start_weights.get("sha256") if start_weights else None
        self.weights = None
        if start_weights:
            table, default = start_weights["weights"], float(start_weights["default"])
            self.weights = {tid: float(table.get(f'{t.get("suite")}:{int(t.get("task_id"))}:{int(t.get("seed"))}', default))
                            for tid, t in self.trajectories.items()}
        # excluded starts (start-exclude/v1; None = off): skipped by _next_trajectory, everything else unchanged
        self.start_exclude_sha = start_exclude.get("sha256") if start_exclude else None
        self.start_exclude_previous: str | None = None      # set by load_state_dict when the exclusion changed on resume
        self.start_weights_previous: str | None = None      # set by load_state_dict on a pinned start-weights switch
        self.excluded: frozenset[str] = frozenset()
        if start_exclude:
            key_of = {f'{t.get("suite")}:{int(t.get("task_id"))}:{int(t.get("seed"))}': tid for tid, t in self.trajectories.items()}
            unknown = sorted(set(start_exclude["exclude"]) - set(key_of))
            if unknown:
                raise ValueError(f"start_exclude: {len(unknown)} starts are not in the training population, e.g. {unknown[:3]}")
            self.excluded = frozenset(key_of[k] for k in start_exclude["exclude"])
            per_task = defaultdict(lambda: [0, 0])
            for tid, t in self.trajectories.items():
                c = per_task[(t.get("suite"), int(t.get("task_id")))]
                c[0] += 1
                c[1] += tid in self.excluded
            emptied = sorted(f"{s}:{k}" for (s, k), (n, x) in per_task.items() if x >= n)
            if emptied:
                raise ValueError(f"start_exclude removes every start of {emptied[:3]}")
            if self.weights is not None and not any(w > 0 for tid, w in self.weights.items() if tid not in self.excluded):
                raise ValueError("start_exclude leaves no start with a positive weight")
        if self.budget:
            # frontier f starts at the last eligible (unsolved, within budget) start: K = N - f
            self.frontier = {}
            for tid, t in self.trajectories.items():
                ks = self._eligible_ks(tid, int(t.get("requests") or 0))
                if ks:
                    self.frontier[tid] = int(t["requests"]) - max(ks)
        else:
            self.frontier = {tid: min(int(initial_frontier), t["requests"]) for tid, t in self.trajectories.items() if t.get("requests")}
        self.rng = random.Random(seed)
        self.order = list(range(len(self.prompts)))
        self.rng.shuffle(self.order)
        self.cursor = 0
        self.slots = 0
        self.bad: set[str] = set()      # "<trajectory_id>:<K>" starts whose replay diverged
        self.focus_order = list(range(len(self.focus)))
        self.rng.shuffle(self.focus_order)
        self.focus_cursor = 0
        self.active: dict[str, set[int]] = defaultdict(set)
        for start in bank["starts"].values():
            delta = float(start.get("teacher_delta") or 0.0)
            if int(start.get("K") or 0) > 0 and (delta >= self.active_min_delta if self.active_min_delta > 0 else delta != 0):
                self.active[start["trajectory_id"]].add(int(start["K"]))

    def eligible(self, start: dict[str, Any]) -> bool:
        """A teacher start the budget protocol may use (always true without a budget and for K = 0)."""
        k = int(start.get("K") or 0)
        return not self.budget or k == 0 or (k <= self.budget - self.min_live and not solved_at(start))

    def _start(self, tid: str, k: int) -> dict[str, Any]:
        start = self._starts.get((tid, int(k)))
        return start if start is not None else start_for(self.bank, tid, k)

    def _eligible_ks(self, tid: str, requests: int) -> list[int]:
        return [k for k in range(1, requests) if self.eligible(self._start(tid, k))]

    def dress(self, start: dict[str, Any], *, curriculum: bool | None = None) -> dict[str, Any]:
        """The rollout entry for a bank start: its live window and handoff under this curriculum's protocol."""
        entry = dict(start)
        k = int(entry.get("K") or 0)
        if k == 0:
            entry["horizon"] = self.budget or self.fresh_horizon
        else:
            entry["horizon"] = self.budget - k if self.budget else self.horizon
            entry["handoff"] = self.handoff
        if curriculum is not None:
            entry["curriculum"] = curriculum
        if self.reward_mode:
            entry["reward_mode"] = self.reward_mode
        if self.train_all_on_success:
            entry["train_all_on_success"] = True
        return entry

    # ------------------------------------------------------------ checkpoint
    def state_dict(self) -> dict[str, Any]:
        return {"bank_sha256": self.bank["sha256"], "prompts": list(self.prompts), "order": list(self.order), "cursor": self.cursor,
                "slots": self.slots, "frontier": dict(self.frontier), "rng": self.rng.getstate(), "horizon": self.horizon,
                "mode": self.mode, "sourced_only": self.sourced_only, "active_fraction": self.active_fraction, "bad": sorted(self.bad),
                "fresh_horizon": self.fresh_horizon, "active_min_delta": self.active_min_delta, "suites": self.suites,
                "exclude": self.exclude, "focus": self.focus, "focus_order": list(self.focus_order), "focus_cursor": self.focus_cursor,
                "budget": self.budget, "min_live": self.min_live, "handoff": self.handoff,
                "reward_mode": self.reward_mode, "train_all_on_success": self.train_all_on_success,
                "start_weights_sha": self.start_weights_sha,
                **({"start_exclude_sha": self.start_exclude_sha} if self.start_exclude_sha else {})}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (state.get("reward_mode"), bool(state.get("train_all_on_success", False))) != (self.reward_mode, self.train_all_on_success):
            raise ValueError("segment curriculum reward mode changed on resume")
        if state.get("bank_sha256") != self.bank["sha256"] or state.get("prompts") != self.prompts:
            raise ValueError("segment curriculum state belongs to a different bank")
        if state.get("horizon") != self.horizon:
            raise ValueError("segment horizon changed on resume")
        if ((state.get("mode", "reverse"), bool(state.get("sourced_only", False)), float(state.get("active_fraction", 0.0)))
                != (self.mode, self.sourced_only, self.active_fraction)):
            raise ValueError("segment curriculum mode changed on resume")
        saved = (int(state.get("fresh_horizon", state.get("horizon"))), float(state.get("active_min_delta", 0.0)),
                 state.get("suites"), list(state.get("exclude", [])))
        if saved != (self.fresh_horizon, self.active_min_delta, self.suites, self.exclude):
            raise ValueError("segment curriculum windows, suites or held-out trajectories changed on resume")
        if (state.get("budget"), int(state.get("min_live", self.min_live if state.get("budget") else 4)), state.get("handoff", "continuation")) \
                != (self.budget, self.min_live, self.handoff):
            raise ValueError("segment curriculum budget or handoff changed on resume")
        if state.get("start_weights_sha") != self.start_weights_sha:
            # only a pinned switch (training_resume.RESUME_SWITCHES; the weights only decide which drawn turns are kept, so the
            # population, order and rng state stay valid); the caller logs it
            from training_resume import resume_switch
            if not resume_switch("start_weights_sha256", state.get("start_weights_sha"), self.start_weights_sha):
                raise ValueError("segment curriculum start weights changed on resume")
            self.start_weights_previous = state.get("start_weights_sha")
        if state.get("start_exclude_sha") != self.start_exclude_sha:
            # allowed (a start exclusion only skips turns: population, order and rng state stay valid); the caller logs it
            self.start_exclude_previous = state.get("start_exclude_sha") or "none"
        if list(state.get("focus", [])) != self.focus:
            raise ValueError("segment curriculum focus starts changed on resume")
        if self.focus:
            self.focus_order, self.focus_cursor = [int(i) for i in state["focus_order"]], int(state["focus_cursor"])
        if sorted(state["order"]) != list(range(len(self.prompts))):
            raise ValueError("segment curriculum order is not a permutation")
        self.order, self.cursor, self.slots = list(state["order"]), int(state["cursor"]), int(state["slots"])
        self.frontier = {k: int(v) for k, v in state["frontier"].items()}
        self.bad = set(state.get("bad", []))
        version, internal, gauss = state["rng"]
        self.rng.setstate((version, tuple(internal), gauss))

    # ------------------------------------------------------------ sampling
    def _next_trajectory(self) -> str:
        while True:
            tid = self.prompts[self.order[self.cursor % len(self.order)]]
            self.cursor += 1
            if self.cursor % len(self.order) == 0:
                self.rng.shuffle(self.order)
            if tid in self.excluded:
                continue
            if self.weights is None or self.rng.random() < self.weights[tid]:
                return tid

    def next_batch(self, n: int) -> list[dict[str, Any]]:
        fresh_every = round(1 / self.episode_fraction) if self.episode_fraction > 0 else 0
        entries = []
        for _ in range(n):
            tid = self._next_trajectory()
            requests = int(self.trajectories[tid].get("requests") or 0)
            self.slots += 1
            fresh = requests == 0 or (fresh_every and self.slots % fresh_every == 0)
            if not fresh and self.focus:
                entries.append(self.dress(self._next_focus(), curriculum=True))
                continue
            if not fresh and self.budget and tid not in self.frontier:
                fresh = True        # no unsolved teacher start fits the budget
            if fresh:
                k = 0
            elif self.mode == "uniform":
                k = self._uniform_k(tid, requests)
            else:
                k = max(0, requests - self.frontier[tid])
            entries.append(self.dress(self._start(tid, k), curriculum=not fresh))
        return entries

    def _next_focus(self) -> dict[str, Any]:
        """The next focus start (a fixed shuffled cycle; starts whose replay diverged are skipped while others remain)."""
        for _ in range(len(self.focus)):
            start = self.bank["starts"][self.focus[self.focus_order[self.focus_cursor % len(self.focus)]]]
            self.focus_cursor += 1
            if self.focus_cursor % len(self.focus) == 0:
                self.rng.shuffle(self.focus_order)
            if f'{start["trajectory_id"]}:{int(start["K"])}' not in self.bad:
                return start
        return start

    def _uniform_k(self, tid: str, requests: int) -> int:
        good = [k for k in range(requests) if f"{tid}:{k}" not in self.bad and self.eligible(self._start(tid, k))]
        active = [k for k in good if k in self.active.get(tid, ())]
        if active and self.rng.random() < self.active_fraction:
            return self.rng.choice(active)
        return self.rng.choice(good) if good else 0

    def update(self, entries: list[dict[str, Any]], results: Iterable[dict[str, Any]]) -> dict[str, Any]:
        """Advance frontiers from one step's groups; returns metrics."""
        by_group: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for result in results:
            by_group[int(result["group"])].append(result)
        advanced = scored = reached_total = 0
        for group, entry in enumerate(entries):
            group_rows = by_group.get(group, [])
            if group_rows and all(r.get("verdict") == "replay_infidelity" for r in group_rows) and entry.get("K"):
                self.bad.add(f'{entry["trajectory_id"]}:{int(entry["K"])}')
            rows = [r for r in group_rows if r.get("verdict") not in ("infrastructure_error", "replay_infidelity")]
            if not rows:
                continue
            hits = sum(1 for r in rows if reached(r, entry.get("teacher_delta")))
            scored += len(rows)
            reached_total += hits
            tid = entry["trajectory_id"]
            if entry.get("curriculum") and tid in self.frontier and hits >= self.advance_fraction * len(rows):
                requests = int(self.trajectories[tid]["requests"])
                if self.frontier[tid] < requests:
                    self.frontier[tid] = min(requests, self.frontier[tid] + self.frontier_step)
                    if self.budget:     # skip earlier starts the protocol cannot use (e.g. beyond the budget)
                        while self.frontier[tid] < requests and not self.eligible(self._start(tid, requests - self.frontier[tid])):
                            self.frontier[tid] += 1
                    advanced += 1
        values = list(self.frontier.values())
        return {"segment/reached_rate": reached_total / max(1, scored), "segment/advanced_groups": advanced,
                "segment/frontier_mean": sum(values) / max(1, len(values)),
                "segment/frontier_done": sum(1 for tid, f in self.frontier.items() if f >= self.trajectories[tid]["requests"]) / max(1, len(values)),
                "segment/mean_K": sum(e["K"] for e in entries) / max(1, len(entries)), "segment/bad_starts": len(self.bad)}


__all__ = ["BoundaryCurriculum", "SegmentCurriculum", "load_boundary_manifest", "solved_at", "FRESH_SEED_BASE", "EVAL_SEEDS",
           "is_fresh_seed", "check_fresh_population"]


MANIFEST_SCHEMA = "curriculum-boundaries/v1"


def load_boundary_manifest(path: str | Path, bank: dict[str, Any]) -> dict[str, Any]:
    """{"tracks": [{"track", "boundaries": [start_id latest..earliest], "fresh": start_id (K = 0), "frontier": index}]}."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA or manifest.get("bank_sha256") != bank["sha256"]:
        raise ValueError(f"not a {MANIFEST_SCHEMA} manifest of this bank: {path}")
    for tr in manifest["tracks"]:
        ids = list(tr["boundaries"]) + [tr["fresh"]]
        missing = [s for s in ids if s not in bank["starts"]]
        if missing:
            raise ValueError(f"track {tr['track']}: starts missing from the bank: {missing[:2]}")
        if int(bank["starts"][tr["fresh"]]["K"]) != 0 or any(solved_at(bank["starts"][s]) for s in tr["boundaries"]):
            raise ValueError(f"track {tr['track']}: fresh start must be K = 0 and boundaries unsolved")
        ks = [int(bank["starts"][s]["K"]) for s in tr["boundaries"]]
        if ks != sorted(ks, reverse=True) or (ks and ks[-1] <= 0):
            raise ValueError(f"track {tr['track']}: boundaries must run from the latest K down to K > 0")
    return manifest


class BoundaryCurriculum:
    """Remaining-task reverse curriculum over calibrated tracks .

    A track is (task, official training initial state, teacher source) with boundaries from the latest unsolved clean
    start to the official reset. Group slots follow ``slot_cycle`` (default three frontier groups, then one K = 0
    group); tracks rotate fairly. Every group of a round is one start with G candidates. Promotion: the last two complete
    groups at the current boundary each have >= promote_each successes and >= promote_total together -> one boundary
    earlier (the evidence window is cleared). Rollback: after a promotion, two complete groups with <= rollback_each
    successes each -> back one boundary. A track whose frontier reaches K = 0 samples fresh starts only. fresh_only
    turns every slot into a K = 0 group (the F-bin arm). Entries carry the remaining budget (budget - K), the clean
    handoff, the reward mode and train_all_on_success."""

    def __init__(self, bank: dict[str, Any], manifest: dict[str, Any], *, budget: int = 40, handoff: str = "clean",
                 reward_mode: str | None = None, slot_cycle: Iterable[str] = ("frontier", "frontier", "frontier", "fresh"),
                 group_size: int = 16, promote_each: int = 8, promote_total: int = 20, rollback_each: int = 2,
                 fresh_only: bool = False, train_all_on_success: bool = True, seed: int = 17) -> None:
        self.bank, self.budget, self.handoff, self.reward_mode = bank, int(budget), handoff, reward_mode
        self.slot_cycle, self.group_size = list(slot_cycle), int(group_size)
        self.promote_each, self.promote_total, self.rollback_each = int(promote_each), int(promote_total), int(rollback_each)
        self.fresh_only, self.train_all_on_success = bool(fresh_only), bool(train_all_on_success)
        self.tracks = {tr["track"]: tr for tr in manifest["tracks"]}
        self.prompts = sorted(self.tracks)
        self.frontier = {t: min(int(tr.get("frontier", 0)), len(tr["boundaries"])) for t, tr in self.tracks.items()}
        self.evidence: dict[str, list[int]] = {t: [] for t in self.tracks}
        self.promoted: dict[str, bool] = {t: False for t in self.tracks}
        self.rng = random.Random(seed)
        self.order = list(self.prompts)
        self.rng.shuffle(self.order)
        self.cursor = self.slot = 0
        self.promotions = self.rollbacks = 0

    def _start(self, track: str, kind: str) -> dict[str, Any]:
        tr, f = self.tracks[track], self.frontier[track]
        sid = tr["fresh"] if kind == "fresh" or f >= len(tr["boundaries"]) else tr["boundaries"][f]
        return self.bank["starts"][sid]

    def dress(self, start: dict[str, Any], **extra: Any) -> dict[str, Any]:
        entry = dict(start, **extra)
        k = int(entry.get("K") or 0)
        entry["horizon"] = self.budget - k
        if k:
            entry["handoff"] = self.handoff
        if self.reward_mode:
            entry["reward_mode"] = self.reward_mode
        entry["train_all_on_success"] = self.train_all_on_success
        return entry

    def next_batch(self, n: int) -> list[dict[str, Any]]:
        entries = []
        for _ in range(n):
            track = self.order[self.cursor % len(self.order)]
            self.cursor += 1
            if self.cursor % len(self.order) == 0:
                self.rng.shuffle(self.order)
            kind = "fresh" if self.fresh_only else self.slot_cycle[self.slot % len(self.slot_cycle)]
            self.slot += 1
            start = self._start(track, kind)
            entries.append(self.dress(start, track=track, slot=kind, curriculum=kind == "frontier" and int(start["K"]) > 0))
        return entries

    def update(self, entries: list[dict[str, Any]], results: Iterable[dict[str, Any]]) -> dict[str, Any]:
        by_group: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for r in results:
            by_group[int(r["group"])].append(r)
        mixed = all_fail = all_success = incomplete = 0
        for gi, entry in enumerate(entries):
            rows = [r for r in by_group.get(gi, []) if r.get("verdict") not in ("infrastructure_error", "replay_infidelity")]
            wins = sum(bool((r.get("diagnostics") or {}).get("final_success")) for r in rows)
            if len(rows) < self.group_size:
                incomplete += 1
                continue
            mixed += 0 < wins < len(rows)
            all_fail += wins == 0
            all_success += wins == len(rows)
            track = entry.get("track")
            if not entry.get("curriculum") or track not in self.tracks:
                continue
            if entry["start_id"] != self._start(track, "frontier")["start_id"]:
                continue                         # the frontier moved since this group was scheduled
            ev = self.evidence[track]
            ev.append(wins)
            last = ev[-2:]
            if len(last) == 2 and min(last) >= self.promote_each and sum(last) >= self.promote_total:
                self.frontier[track] += 1
                self.evidence[track], self.promoted[track] = [], True
                self.promotions += 1
            elif len(last) == 2 and self.promoted[track] and max(last) <= self.rollback_each and self.frontier[track] > 0:
                self.frontier[track] -= 1
                self.evidence[track], self.promoted[track] = [], False
                self.rollbacks += 1
        ks = [int(self._start(t, "frontier")["K"]) for t in self.prompts]
        return {"curriculum/mixed_groups": mixed, "curriculum/all_fail_groups": all_fail, "curriculum/all_success_groups": all_success,
                "curriculum/incomplete_groups": incomplete, "curriculum/promotions": self.promotions, "curriculum/rollbacks": self.rollbacks,
                "curriculum/frontier_mean_K": sum(ks) / max(1, len(ks)), "curriculum/tracks_at_fresh": sum(k == 0 for k in ks),
                "segment/mean_K": sum(int(e["K"]) for e in entries) / max(1, len(entries))}

    def state_dict(self) -> dict[str, Any]:
        return {"kind": "boundary", "bank_sha256": self.bank["sha256"], "prompts": self.prompts, "frontier": dict(self.frontier),
                "evidence": {t: list(v) for t, v in self.evidence.items()}, "promoted": dict(self.promoted), "order": list(self.order),
                "cursor": self.cursor, "slot": self.slot, "rng": self.rng.getstate(), "promotions": self.promotions,
                "rollbacks": self.rollbacks, "fresh_only": self.fresh_only, "slot_cycle": self.slot_cycle, "budget": self.budget,
                "reward_mode": self.reward_mode, "handoff": self.handoff}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        same = (state.get("kind"), state.get("bank_sha256"), state.get("prompts"), bool(state.get("fresh_only")), state.get("slot_cycle"),
                state.get("budget"), state.get("reward_mode"), state.get("handoff"))
        if same != ("boundary", self.bank["sha256"], self.prompts, self.fresh_only, self.slot_cycle, self.budget, self.reward_mode, self.handoff):
            raise ValueError("boundary curriculum state belongs to a different manifest or protocol")
        self.frontier = {k: int(v) for k, v in state["frontier"].items()}
        self.evidence = {k: [int(x) for x in v] for k, v in state["evidence"].items()}
        self.promoted = {k: bool(v) for k, v in state["promoted"].items()}
        self.order, self.cursor, self.slot = list(state["order"]), int(state["cursor"]), int(state["slot"])
        self.promotions, self.rollbacks = int(state["promotions"]), int(state["rollbacks"])
        version, internal, gauss = state["rng"]
        self.rng.setstate((version, tuple(internal), gauss))

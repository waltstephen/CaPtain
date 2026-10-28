#!/usr/bin/env python3
"""Segment-start bank for per-call RL (python-runtime/v1).

A start is (episode, K): "replay the first K recorded model responses of this episode, then let the policy make
request K+1". Sources are complete python-mode episodes with host marks (react_episode.py --mark-requests), usually
successful teacher episodes; K = 0 entries (a fresh episode) are added for every (suite, task, seed) prompt.

    python rl/segment_bank.py build --episodes <dir> [<dir> ...] --out bank.json [--prompts spatial:0-9:1-3 ...]
    python rl/segment_bank.py summary bank.json

The bank records, per start, what the replayed state must look like (the source's mark before request K+1 and the
(name, success) sequence of the calls made so far): reward_segment.fidelity() rejects a rollout whose replay diverged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from reward_segment import read_marks
from reward import _read_json, _read_jsonl, goal_potentials, read_goal_audit

SCHEMA_VERSION = "segment-bank/v1"


def _python_responses(episode_dir: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(episode_dir / "driver" / "responses.jsonl")
    return [r for r in sorted(rows, key=lambda r: int(r.get("request") or 0))]


def _slim_mark(row: dict[str, Any]) -> dict[str, Any]:
    return {k: row.get(k) for k in ("request_index", "call_index", "action_index", "eef", "objects", "goal_states", "t_s")}


def episode_starts(episode_dir: Path, *, require_success: bool = True) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """(trajectory record, start entries) for one source episode, or (None, []) when it is not usable."""
    episode_dir = Path(episode_dir)
    er = _read_json(episode_dir / "episode_result.json")
    server = _read_json(episode_dir / "result.json")
    if not er or not server or er.get("runtime_tool_contract") != "python-runtime/v1" or er.get("replay_prefix"):
        return None, []
    if require_success and server.get("host_final_success") is not True:
        return None, []
    calls = _read_jsonl(episode_dir / "calls.jsonl")
    try:
        audit = read_goal_audit(episode_dir, calls)
        marks = read_marks(episode_dir)
    except ValueError:
        return None, []
    responses = _python_responses(episode_dir)
    tool_requests = [r["request"] for r in responses
                     if any(it.get("type") == "custom_tool_call" and it.get("name") == "python" for it in r.get("items") or [])]
    if not tool_requests or tool_requests != list(range(1, len(tool_requests) + 1)) or any(i not in marks for i in tool_requests):
        return None, []
    goals = audit["goals"]
    trajectory_id = hashlib.sha256(str(episode_dir.resolve()).encode()).hexdigest()[:16]
    starts = []
    n = len(tool_requests)
    for k in range(n):
        before = marks[k + 1]
        after = marks.get(k + 2) or audit["final"]
        delta = goal_potentials({"goals": goals, "initial": before, "final": after})
        expected_calls = [[c.get("name"), bool(c.get("success"))] for c in calls
                          if type(c.get("call_index")) is int and c["call_index"] <= (before.get("call_index") or 0)]
        starts.append({"start_id": hashlib.sha256(f"{trajectory_id}:{k}".encode()).hexdigest()[:16], "trajectory_id": trajectory_id,
                       "suite": er["suite"], "task_id": int(er["task_id"]), "seed": int(er["seed"]), "K": k, "remaining": n - k,
                       "source_dir": str(episode_dir.resolve()), "source_responses": str((episode_dir / "driver" / "responses.jsonl").resolve()),
                       "expected": {"mark": _slim_mark(before), "calls": expected_calls},
                       "teacher_delta": round(delta["goal_progress"] + delta["goal_satisfied"], 6)})
    trajectory = {"trajectory_id": trajectory_id, "suite": er["suite"], "task_id": int(er["task_id"]), "seed": int(er["seed"]),
                  "requests": n, "source_dir": str(episode_dir.resolve()), "model": er.get("model"),
                  "host_final_success": server.get("host_final_success")}
    return trajectory, starts


def prompt_starts(prompts: Iterable[tuple[str, int, int]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trajectories, starts = [], []
    for suite, task, seed in prompts:
        tid = hashlib.sha256(f"prompt:{suite}:{task}:{seed}".encode()).hexdigest()[:16]
        trajectories.append({"trajectory_id": tid, "suite": suite, "task_id": task, "seed": seed, "requests": 0, "source_dir": None})
        starts.append({"start_id": hashlib.sha256(f"{tid}:0".encode()).hexdigest()[:16], "trajectory_id": tid, "suite": suite,
                       "task_id": task, "seed": seed, "K": 0, "remaining": None, "source_dir": None, "source_responses": None,
                       "expected": None, "teacher_delta": None})
    return trajectories, starts


def parse_prompts(specs: Iterable[str]) -> list[tuple[str, int, int]]:
    """suite:tasks:seeds with ranges, e.g. libero_spatial:0-9:1-3 or libero_goal:3,5:11."""
    def numbers(text: str) -> list[int]:
        out: list[int] = []
        for part in text.split(","):
            lo, _, hi = part.partition("-")
            out.extend(range(int(lo), int(hi or lo) + 1))
        return out

    prompts = []
    for spec in specs:
        suite, tasks, seeds = spec.split(":")
        prompts.extend((suite, t, s) for t in numbers(tasks) for s in numbers(seeds))
    return prompts


def build(episode_dirs: Iterable[Path], *, prompts: Iterable[tuple[str, int, int]] = (), require_success: bool = True) -> dict[str, Any]:
    trajectories: dict[str, dict[str, Any]] = {}
    starts: dict[str, dict[str, Any]] = {}
    skipped = []
    for directory in episode_dirs:
        trajectory, entries = episode_starts(Path(directory), require_success=require_success)
        if trajectory is None:
            skipped.append(str(directory))
            continue
        trajectories[trajectory["trajectory_id"]] = trajectory
        starts.update({e["start_id"]: e for e in entries})
    extra_trajectories, extra_starts = prompt_starts(prompts)
    for trajectory, entry in zip(extra_trajectories, extra_starts):
        trajectories.setdefault(trajectory["trajectory_id"], trajectory)
        starts.setdefault(entry["start_id"], entry)
    bank = {"schema_version": SCHEMA_VERSION, "trajectories": dict(sorted(trajectories.items())), "starts": dict(sorted(starts.items())),
            "skipped_sources": len(skipped)}
    bank["sha256"] = bank_sha256(bank)
    return bank


def bank_sha256(bank: dict[str, Any]) -> str:
    identity = {k: bank[k] for k in ("schema_version", "trajectories", "starts")}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_bank(path: Path) -> dict[str, Any]:
    bank = json.loads(Path(path).read_text(encoding="utf-8"))
    if bank.get("schema_version") != SCHEMA_VERSION or bank.get("sha256") != bank_sha256(bank):
        raise ValueError(f"segment bank {path} is invalid or was edited")
    return bank


def start_for(bank: dict[str, Any], trajectory_id: str, k: int) -> dict[str, Any]:
    starts = [s for s in bank["starts"].values() if s["trajectory_id"] == trajectory_id and s["K"] == k]
    if len(starts) != 1:
        raise KeyError(f"no unique start for {trajectory_id} K={k}")
    return starts[0]


def iter_episode_dirs(roots: Iterable[Path]) -> list[Path]:
    found = []
    for root in roots:
        root = Path(root)
        if (root / "episode_result.json").is_file():
            found.append(root)
        else:
            found.extend(p.parent for p in sorted(root.glob("**/episode_result.json")) if ".partial-" not in str(p) and ".infra-" not in str(p))
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--episodes", nargs="*", type=Path, default=[])
    b.add_argument("--prompts", nargs="*", default=[])
    b.add_argument("--include-failures", action="store_true", help="also use unsuccessful source episodes (their segments still carry marks)")
    b.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("summary")
    s.add_argument("bank", type=Path)
    args = ap.parse_args()
    if args.cmd == "build":
        bank = build(iter_episode_dirs(args.episodes), prompts=parse_prompts(args.prompts), require_success=not args.include_failures)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(bank, indent=1), encoding="utf-8")
        print(json.dumps({"trajectories": len(bank["trajectories"]), "starts": len(bank["starts"]), "skipped": bank["skipped_sources"],
                          "sha256": bank["sha256"]}))
    else:
        bank = load_bank(args.bank)
        with_source = [t for t in bank["trajectories"].values() if t.get("source_dir")]
        print(json.dumps({"trajectories": len(bank["trajectories"]), "source_trajectories": len(with_source),
                          "starts": len(bank["starts"]), "sha256": bank["sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Torch-free scheduling helpers of the RL trainer (learner pipeline): slot occupancy of
a collection, pack-level rank assignment, and the lookahead draw that keeps checkpoints exact while the pipeline stays full."""
from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Iterable


def slot_stats(results: Iterable[dict], concurrency: int) -> dict:
    """Occupancy of the env server's episode slots during one collection, from the per-episode wall clocks the env server
    returns (t_arrival_wall: POST received, t_start_wall: a worker slot taken, t_end_wall: the slot released). util = busy slot-seconds
    over concurrency x span; tail = last end minus the 90th-percentile end (the stragglers every collection waits for)."""
    rows = [r for r in results if all(isinstance(r.get(k), (int, float)) for k in ("t_arrival_wall", "t_start_wall", "t_end_wall"))]
    if not rows or concurrency < 1:
        return {}
    t0 = min(r["t_arrival_wall"] for r in rows)
    t1 = max(r["t_end_wall"] for r in rows)
    span = max(1e-9, t1 - t0)
    busy = sum(max(0.0, r["t_end_wall"] - r["t_start_wall"]) for r in rows)
    ends = sorted(r["t_end_wall"] - t0 for r in rows)
    queue = sorted(max(0.0, r["t_start_wall"] - r["t_arrival_wall"]) for r in rows)
    return {"slots/n": len(rows), "slots/concurrency": concurrency, "slots/span_s": round(span, 1),
            "slots/util": round(busy / (concurrency * span), 4), "slots/busy_slot_s": round(busy, 1),
            "slots/tail_s": round(ends[-1] - ends[int(0.9 * (len(ends) - 1))], 1),
            "slots/queue_p50_s": round(queue[len(queue) // 2], 1), "slots/queue_max_s": round(queue[-1], 1)}


def pack_cost(n_tokens: int, quadratic_tokens: float = 140_000.0) -> float:
    """Learner cost proxy of one packed sequence (prefix_chains.packed_cost): projections/MLP ~ n, full attention ~ n^2."""
    return n_tokens + n_tokens * n_tokens / quadratic_tokens


def assign_packs(packs: list[tuple[int, int, float]], world: int) -> list[list[tuple[int, int]]]:
    """Longest-processing-time assignment of packed sequences (trajectory index, chain index, cost) to ranks. FSDP runs the
    passes in lockstep (pass i costs the slowest rank's i-th sequence), so balancing individual sequences instead of whole
    trajectories removes most of the idle time (B1: lockstep efficiency 0.90 -> 0.97). Deterministic; every pack lands on
    exactly one rank, so the objective (a sum over packs) is unchanged."""
    loads = [[0.0, 0, r] for r in range(world)]
    out: list[list[tuple[int, int]]] = [[] for _ in range(world)]
    for traj, chain, cost in sorted(packs, key=lambda p: (-p[2], p[0], p[1])):
        slot = min(loads)
        out[slot[2]].append((traj, chain))
        slot[0] += cost
        slot[1] += 1
    return [sorted(x) for x in out]


def trajectory_owner(assignment: list[list[tuple[int, int]]]) -> dict[int, int]:
    """The rank that accounts a trajectory's once-per-trajectory quantities (its completion-token denominator): the holder of
    its lowest chain index."""
    best: dict[int, tuple[int, int]] = {}
    for rank, items in enumerate(assignment):
        for traj, chain in items:
            if traj not in best or chain < best[traj][0]:
                best[traj] = (chain, rank)
    return {traj: rank for traj, (chain, rank) in best.items()}


def pack_plan(active: list[int], packed_lengths: dict[int, list[int]], world: int) -> tuple[list[dict[int, set[int]]], dict[int, int]]:
    """Per rank: {trajectory index: the chain indices it trains}; plus each trajectory's owner rank (trajectory_owner), which
    alone counts the trajectory in n_trajectories and its completion tokens in the objective denominator."""
    packs = [(i, c, pack_cost(int(n))) for i in active for c, n in enumerate(packed_lengths[i])]
    assignment = assign_packs(packs, world)
    per_rank: list[dict[int, set[int]]] = []
    for items in assignment:
        chains: dict[int, set[int]] = defaultdict(set)
        for traj, chain in items:
            chains[traj].add(chain)
        per_rank.append(dict(chains))
    return per_rank, trajectory_owner(assignment)


def lookahead_draw(curriculum: Any, n: int, *, snapshot: bool) -> tuple[list, dict | None]:
    """Draw the next batch for the one-batch lookahead. snapshot=True (a checkpoint round) also returns the curriculum state
    from BEFORE the draw: a resume from that checkpoint redraws exactly this batch and collects it synchronously, so the
    lookahead no longer has to drain on checkpoint rounds."""
    before = copy.deepcopy(curriculum.state_dict()) if snapshot else None
    return curriculum.next_batch(n), before

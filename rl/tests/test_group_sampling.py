"""Group sampling: StreamPool group_size (complete groups, adjacent members, one draw per
group, liveness) and the group-relative advantages of critic_cotrain.physics_only_advantages (group_keys None = unchanged).
Runs under pytest or directly (python test_group_sampling.py) where pytest is not installed."""
import json
import random
import sys
import tempfile
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import critic_cotrain as CC  # noqa: E402
import physics_critic as PC  # noqa: E402
from stream_pool import StreamPool  # noqa: E402


def _pool(G, batch=8, inflight=4, slow=0.03, seed=0):
    rnd = random.Random(seed)
    drawn, calls = [], []
    lock = threading.Lock()

    def draw(n):
        with lock:
            calls.append(n)
            out = [{"id": len(drawn) + i} for i in range(n)]
            drawn.extend(out)
            return out

    def run_one(entry, step, gi):
        time.sleep(rnd.uniform(0.001, slow))
        return {"verdict": "ok", "entry_id": entry["id"], "gi": gi}

    kw = {"group_size": G} if G != 1 else {}
    return StreamPool(draw=draw, run_one=run_one, batch=batch, inflight=inflight, draw_chunk=batch, version=lambda: 0,
                      step=lambda: 1, **kw), drawn, calls


def test_group_size_one_is_the_old_pool():
    pool, drawn, calls = _pool(1)
    try:
        entries, results, _ = pool.take(8)
        assert len(results) == 8 and all(r["entry_id"] == e["id"] for e, r in zip(entries, results))
        assert all("start_group" not in r and "group_key" not in r["stream"] for r in results)
        assert set(calls) == {8}
        assert len({r["entry_id"] for r in results}) == 8           # every episode its own start
    finally:
        pool.close()


def test_groups_are_complete_adjacent_and_one_draw_per_group():
    for G in (2, 4):
        pool, drawn, calls = _pool(G, batch=8, inflight=6, seed=G)
        try:
            seen_keys = set()
            for _ in range(4):
                entries, results, _ = pool.take(8)
                assert len(entries) == len(results) == 8
                assert all(r["entry_id"] == e["id"] for e, r in zip(entries, results))
                for j in range(0, 8, G):
                    members = results[j:j + G]
                    keys = {r["start_group"] for r in members}
                    assert len(keys) == 1 and {r["stream"]["group_key"] for r in members} == keys
                    assert len({r["entry_id"] for r in members}) == 1          # one start per group ...
                    assert len({r["gi"] for r in members}) == G                # ... run G separate times
                    (k,) = keys
                    assert k not in seen_keys
                    seen_keys.add(k)
                assert len({r["entry_id"] for r in results}) == 8 // G         # n / G distinct starts per batch
            assert set(calls) == {8 // G}                                      # the curriculum draws n / G starts per chunk
            # no start is drawn twice
            assert len({d["id"] for d in drawn}) == len(drawn)
        finally:
            pool.close()


def test_take_is_live_with_the_minimum_inflight():
    pool, _, _ = _pool(4, batch=8, inflight=4, slow=0.01)
    done = []

    def taker():
        for _ in range(3):
            pool.take(8)
            done.append(1)

    try:
        t = threading.Thread(target=taker, daemon=True)
        t.start()
        t.join(timeout=30)
        assert len(done) == 3, "take() blocked with inflight == group_size"
    finally:
        pool.close()


def test_bad_group_sizes_are_rejected():
    for G, batch, inflight in ((3, 8, 8), (4, 8, 2), (0, 8, 8)):
        try:
            StreamPool(draw=lambda n: [], run_one=lambda e, s, g: {}, batch=batch, inflight=inflight, draw_chunk=batch,
                       version=lambda: 0, step=lambda: 0, group_size=G)
        except ValueError:
            continue
        raise AssertionError(f"group_size {G} batch {batch} inflight {inflight} accepted")


# ---- group-relative advantages (fixtures as in test_physics_critic.py, copied so this file needs no pytest)
def write_episode(d, eef_zs):
    d.mkdir(parents=True)
    marks, calls, reqs = [], [], []
    for t, z in enumerate(eef_zs, start=1):
        goal = {"satisfied": False, "target_position_m": [0.5, 0.0, 0.02], "destination_position_m": [0.6, 0.1, 0.02],
                "distance_to_goal": 0.14, "progress_kind": "object"}
        marks.append({"request_index": t, "call_index": t - 1, "eef": [0.5, 0.0, z], "goal_states": {"g0": goal},
                      "objects": {"bowl": [0.5, 0.0, 0.02]}, "environment_step": 10 * t})
        calls.append({"call_index": t, "name": "move_to", "arguments": {"gripper": "open"}, "success": True, "consumes_action": True})
        reqs.append({"index": t, "response_id": f"{d.name}-r{t}"})
    for name, rows in (("hidden_request_marks.jsonl", marks), ("calls.jsonl", calls)):
        (d / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "driver").mkdir()
    (d / "driver" / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reqs))


def write_critic(d, features, tasks=("libero_object:4",), seed=0):
    torch.manual_seed(seed)
    d.mkdir(parents=True)
    spec = {"feature_version": PC.cf.FEATURE_VERSION, "features": list(features), "mean": [0.0] * len(features),
            "std": [1.0] * len(features), "clip": 10, "task_vocab": list(tasks)}
    (d / "feature_spec.json").write_text(json.dumps(spec))
    net, _ = PC.mlp(len(features) + len(tasks))
    torch.save({"arch": {"in_dim": len(features) + len(tasks)}, "state_dict": net.state_dict()}, d / "critic_mlp.pt")


def _episodes(tmp):
    write_critic(tmp / "critic", ["eef_z", "eef_tgt_dist"])
    frozen = PC.FrozenCritic(tmp / "critic")
    co = CC.CoCritic(tmp / "critic", 8, use_hidden=False)
    spec = [("a", [0.3, 0.2, 0.1], True, "agent_finished"), ("b", [0.3, 0.25, 0.2, 0.15], False, "request_cap_exhausted"),
            ("c", [0.3, 0.1], True, "agent_finished"), ("d", [0.3, 0.3, 0.3], False, "agent_finished"),
            ("e", [0.2, 0.1], True, "agent_finished")]
    results = []
    for name, zs, ok, term in spec:
        write_episode(tmp / name, zs)
        results.append({"sample_path": "x", "verdict": "segment_success" if ok else ("false_finish" if term == "agent_finished" else "no_progress"),
                        "episode_dir": str(tmp / name), "suite": "libero_object", "task_id": 4,
                        "diagnostics": {"host_final_success": ok, "termination_reason": term}})
    results.append({"sample_path": None, "verdict": "infrastructure_error", "diagnostics": {}})      # unscored
    return co, frozen, results


COSTS = {"c_act": 0.01, "c_fail": 0.02, "missed_finish_return": 0.5, "exhaust_reward": -0.1, "ref": {"libero_object:*": 1}}


def test_group_keys_none_is_bit_identical(tmp_path=None):
    tmp = Path(tmp_path or tempfile.mkdtemp())
    co, frozen, results = _episodes(tmp)
    for costs in (None, COSTS):
        a, ba = CC.physics_only_advantages(co, frozen, results, lam=0.9, sigma=0.7, device="cpu", false_finish_reward=-0.3, costs=costs)
        b, bb = CC.physics_only_advantages(co, frozen, results, lam=0.9, sigma=0.7, device="cpu", false_finish_reward=-0.3, costs=costs,
                                           group_keys=None)
        assert a == b and ba["adv"] == bb["adv"] and "group_stats" not in bb


def test_group_relative_outcome_plus_critic_shaping(tmp_path=None):
    tmp = Path(tmp_path or tempfile.mkdtemp())
    co, frozen, results = _episodes(tmp)
    sigma = 0.7
    keys = [0, 0, 1, 1, 2, 1]        # groups {a, b}, {c, d, (infra)}, {e} alone
    plain, pb = CC.physics_only_advantages(co, frozen, results, lam=0.9, sigma=sigma, device="cpu", false_finish_reward=-0.3, costs=COSTS)
    got, gb = CC.physics_only_advantages(co, frozen, results, lam=0.9, sigma=sigma, device="cpu", false_finish_reward=-0.3, costs=COSTS,
                                         group_keys=keys)
    # total return G_i = terminal return + immediate costs (the same functions the trainer uses)
    def total(i):
        r = results[i]
        rows = frozen.live_rows(r["episode_dir"])
        imm = PC.request_costs(rows, COSTS, "libero_object:4") or {}
        return PC.advantage_return(r, -0.3, COSTS) + sum(imm.values())
    G = {i: total(i) for i in range(5)}
    assert abs(G[0] - (1.0 + sum((PC.request_costs(frozen.live_rows(results[0]["episode_dir"]), COSTS, "libero_object:4") or {}).values()))) < 1e-12
    for group in ([0, 1], [2, 3]):
        base = sum(G[i] for i in group) / len(group)
        for i in group:
            p = [v * sigma for v in plain[i].values()]
            g = [v * sigma for v in got[i].values()]
            assert list(plain[i]) == list(got[i])                                         # same requests trained
            assert abs(sum(g) / len(g) - (G[i] - base)) < 1e-9                             # outcome part: group-relative
            pm = sum(p) / len(p)
            assert all(abs((gv - sum(g) / len(g)) - (pv - pm)) < 1e-9 for gv, pv in zip(g, p))   # shaping: the critic's
    # the group outcome parts sum to zero within each group
    for group in ([0, 1], [2, 3]):
        assert abs(sum(sum(got[i].values()) / len(got[i]) for i in group)) < 1e-9
    assert got[4] == plain[4] and got[5] is None and plain[5] is None                      # singleton keeps the critic; unscored
    assert len(gb["adv"]) == len(pb["adv"])
    flat = [v * sigma for i in range(5) for v in got[i].values()]
    assert all(abs(x - y) < 1e-9 for x, y in zip(flat, gb["adv"]))                         # buf["adv"] = advantages used
    assert gb["group_stats"]["groups"] == 2 and gb["group_stats"]["fallback_episodes"] == 1 and gb["group_stats"]["episodes"] == 4


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"GROUP_SAMPLING_TESTS OK ({len(tests)})")

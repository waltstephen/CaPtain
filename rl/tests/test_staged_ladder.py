"""Staged ONE-SHOT subgoal reward on the 8 hard tasks' ladders (costs["staged"] / trainer --staged-reward / fresh_run
staged_reward; physics_critic.stage_ladder), with the ladders of configs/staged-reward-spec.json.
Runs under pytest or directly (python test_staged_ladder.py)."""
import hashlib
import importlib.util
import itertools
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import physics_critic as PC  # noqa: E402

HARD = ["libero_10:3", "libero_10:6", "libero_10:8", "libero_10:9", "libero_goal:0", "libero_goal:3", "libero_goal:5", "libero_goal:7"]
REF = {"libero_10:*": 12, "libero_object:*": 6}
LIVE_COSTS = {"c_act": 0.008, "c_fail": 0.02, "missed_finish_return": 0.25, "exhaust_reward": -0.1, "ref": REF,
              "post_success_act": True, "post_success_req": 0.005, "post_success_cap": 0.45}
SPEC_TEXT = (HERE.parent / "configs" / "staged-reward-spec.json").read_text()
SPEC = json.loads(SPEC_TEXT)
STAGED = SPEC["staged_reward"]


# synthetic marks. libero_10:3 = goal_0 Close drawer (joint, closure), goal_1 In bowl (object target); goal_7 = goal_0 knob (joint).
# A row carries what request_costs, the ladder bookkeeping and the stub critic read.
ROWLIB = r'''
BOWL0 = (0.33, 0.42, -0.014)
FAR = [0.45, 0.0, 0.26]
def r3(i, sat=(0, 0), cl0=0.0, z=0.0, near=False, p=1, b=0, xy=(0.0, 0.0)):
    """libero_10_3 mark: bowl lifted by z above its start (xy shifted), gripper at the bowl when near"""
    tp = [BOWL0[0] + xy[0], BOWL0[1] + xy[1], BOWL0[2] + z]
    eef = [tp[0], tp[1], tp[2] + 0.04] if near else list(FAR)
    return {"request_index": i, "req_phys": p, "req_phys_bad": b, "req_perc": 1 - min(p, 1), "replayed": False,
            "f": {"all_sat": float(all(sat)), "req_idx": float(i)}, "goal_ids": ["goal_0", "goal_1"], "goal_sat": [bool(s) for s in sat],
            "goal_closure": [cl0, None], "goal_dist": [1.1 * (1 - cl0), 0.3], "goal_tp": [None, tp], "goal_dest": [[0.65, 0.15, 0.04]] * 2,
            "eef": eef}
def rg7(i, sat=0, cl=0.0, p=1, b=0):
    return {"request_index": i, "req_phys": p, "req_phys_bad": b, "req_perc": 1 - min(p, 1), "replayed": False,
            "f": {"all_sat": float(sat), "req_idx": float(i)}, "goal_ids": ["goal_0"], "goal_sat": [bool(sat)], "goal_closure": [cl],
            "goal_dist": [1.0 - cl], "goal_tp": [None], "goal_dest": [None], "eef": list(FAR)}
'''
exec(ROWLIB)
EPS_SRC = r'''
G, P, H = dict(z=0.10, near=True), dict(sat=(0, 1), xy=(0.3, -0.27)), dict(sat=(0, 1), xy=(0.3, -0.27), cl0=0.3)
EP = {
 "fin":   ([r3(1), r3(2, **G), r3(3, **P), r3(4, sat=(1, 1), cl0=1.0, xy=(0.3, -0.27))], True, "agent_finished", "segment_success"),
 "miss":  ([r3(1), r3(2, **G), r3(3, **P), r3(4, sat=(1, 1), cl0=1.0, xy=(0.3, -0.27)), r3(5, sat=(1, 1), cl0=1.0, p=0)], True, "request_cap_exhausted", "segment_success"),
 "ff":    ([r3(1), r3(2, **G), r3(3, **P), r3(4, **P)], False, "agent_finished", "false_finish"),
 "fail":  ([r3(1), r3(2, b=1), r3(3, p=0), r3(4)], False, "request_cap_exhausted", "no_progress"),
 "d1":    ([r3(1), r3(2, **G), r3(3, z=0.0), r3(4, p=0)], False, "request_cap_exhausted", "no_progress"),
 "d2":    ([r3(1), r3(2, **G), r3(3, **P), r3(4, p=0, **P)], False, "request_cap_exhausted", "no_progress"),
 "d3":    ([r3(1), r3(2, **G), r3(3, **P), r3(4, **H)], False, "request_cap_exhausted", "no_progress"),
 "undone":([r3(1), r3(2, **G), r3(3, **P), r3(4, z=0.0, xy=(0.1, 0.0))], False, "request_cap_exhausted", "no_progress"),
 "farm":  ([r3(1)] + [r3(i, **G) if i % 2 == 0 else r3(i) for i in range(2, 12)], False, "request_cap_exhausted", "no_progress"),
 "close1":([r3(1), r3(2, cl0=0.6), r3(3, sat=(1, 0), cl0=1.0)], False, "request_cap_exhausted", "no_progress"),
 "g1prog":([rg7(1), rg7(2, cl=0.1), rg7(3, cl=0.3)], False, "request_cap_exhausted", "no_progress"),
 "g1fail":([rg7(1), rg7(2, cl=0.1), rg7(3, cl=0.05)], False, "wall_time_exhausted", "no_progress"),
 "g1fin": ([rg7(1), rg7(2, cl=0.5), rg7(3, sat=1, cl=1.0)], True, "agent_finished", "segment_success"),
}
GROUPS = ((("libero_10", 3), ["fin", "miss", "d2", "fail"]), (("libero_10", 3), ["d1", "undone", "ff", "fail"]),
          (("libero_goal", 7), ["g1prog", "g1fail", "g1fin", "g1fail"]), (("libero_object", 2), ["d3", "d1", "fail", "fin"]),
          (("libero_10", 3), ["d3", "farm", "close1", "fail"]))
'''
exec(EPS_SRC)
NAMES = [n for _, ns in GROUPS for n in ns]
TASKS = [f"{s}:{t}" for (s, t), ns in GROUPS for _ in ns]

DRIVER = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
import torch
import physics_critic as PC
import critic_cotrain as CC
costs = json.loads(sys.argv[2])
''' + ROWLIB + EPS_SRC + r'''
class Frozen:
    same_t = None
    def live_rows(self, d): return [dict(r) for r in EP[d.split("/")[0]][0]]
    def featurize(self, rows, tk): return torch.tensor([[r["f"]["all_sat"], r["f"]["req_idx"] / 60.0] for r in rows], dtype=torch.float32)
class Critic:
    def values(self, x): return torch.sigmoid(2.5 * x[:, 0] - 0.7 * x[:, 1] - 0.4)
PC.driver_response_ids = lambda d: {k + 1: f"{d}#{k + 1}" for k in range(len(EP[d.split("/")[0]][0]))}
res = []
for g, ((suite, tid), names) in enumerate(GROUPS):
    for j, n in enumerate(names):
        rows_, host, term, verdict = EP[n]
        res.append({"suite": suite, "task_id": tid, "episode_dir": f"{n}/{g}{j}", "sample_path": "x", "verdict": verdict,
                    "start_group": g, "diagnostics": {"host_final_success": host, "termination_reason": term}})
out = {}
for gk in (None, [r["start_group"] for r in res]):
    kw = {"group_keys": gk} if gk is not None else {}
    keyed, buf = CC.physics_only_advantages(Critic(), Frozen(), [dict(r) for r in res], lam=0.9, sigma=0.7, device="cpu",
                                            false_finish_reward=-0.3, costs=costs, **kw)
    out["group" if gk else "plain"] = {"keyed": keyed, "y": buf["y"], "adv": buf["adv"],
                                       "cost_stats": [{k: c[k] for k in ("phys", "bad", "perc", "requests", "cost", "R", "missed_finish", "exhausted")}
                                                      for c in buf["cost_stats"]],
                                       "post": [c.get("post_req") for c in buf["cost_stats"]],
                                       "group_stats": {k: v for k, v in (buf.get("group_stats") or {}).items() if k != "group_sizes"},
                                       "keys": sorted(buf)}
    out["staged_stats"] = buf.get("staged_stats")
print("REPR " + repr(out))
'''


def _run(code_dir, costs):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    p = subprocess.run([sys.executable, "-c", DRIVER, str(code_dir), json.dumps(costs)], capture_output=True, text=True, timeout=300, env=env)
    assert p.returncode == 0, p.stderr[-2000:]
    return [line for line in p.stdout.splitlines() if line.startswith("REPR ")][0]


def _eval(line):
    nan = float("nan")  # noqa: F841
    return eval(line[5:])  # noqa: S307


def _strip(o):
    return {m: o[m] for m in ("plain", "group")}


def _ret(name, task="libero_10:3", costs=None):
    rows, host, term, verdict = EP[name]
    r = {"verdict": verdict, "diagnostics": {"host_final_success": host, "termination_reason": term}}
    return PC.staged_return(r, [dict(q) for q in rows], task, -0.3, costs or dict(LIVE_COSTS, staged=STAGED))


def test_staged_changes_only_laddered_failures_on_production_path():
    off = _eval(_run(HERE, LIVE_COSTS))
    on = _eval(_run(HERE, dict(LIVE_COSTS, staged=STAGED)))
    R_off = [c["R"] for c in off["plain"]["cost_stats"]]
    R_on = [c["R"] for c in on["plain"]["cost_stats"]]
    exp = {"d1": 0.15 / 4, "d2": 0.15 * 2 / 4, "d3": 0.15 * 3 / 4, "undone": 0.15 / 4, "farm": 0.15 / 4, "g1prog": 0.15 / 2}
    for name, task, a, b in zip(NAMES, TASKS, R_off, R_on):
        if task in HARD and name in exp:
            assert abs(b - exp[name]) < 1e-12 and a == -0.1, (name, a, b)
        else:   # successes, missed finish, false finish, zero-depth failures, the dead-end close-first, the non-laddered task
            assert a == b, (name, task, a, b)
    for m in ("plain", "group"):          # critic target (host success) and the immediate costs unchanged
        assert on[m]["y"] == off[m]["y"]
        assert [c["cost"] for c in on[m]["cost_stats"]] == [c["cost"] for c in off[m]["cost_stats"]]
    assert on["group"]["group_stats"]["outcome_mean_absmax"] < 1e-12
    k_on, k_off = on["group"]["keyed"], off["group"]["keyed"]
    assert k_on[12:16] == k_off[12:16]        # the non-laddered group: bit-identical advantages
    mean = lambda d: sum(d.values()) / len(d)  # noqa: E731
    assert mean(k_on[2]) > mean(k_on[3])                                          # d2 above fail
    assert mean(k_on[4]) > mean(k_on[7]) and abs(mean(k_on[4]) - mean(k_on[5])) < 0.05   # d1 ~ undone (both depth 1) > fail
    assert mean(k_on[6]) < mean(k_on[7])                                          # the false finish stays the worst
    assert mean(k_on[8]) > mean(k_on[9]) and abs(mean(k_off[8]) - mean(k_off[9])) < abs(mean(k_on[8]) - mean(k_on[9]))
    assert mean(k_on[16]) > mean(k_on[17]) > mean(k_on[18])                       # d3 > farm (depth 1) > close-first (0)
    st = on["staged_stats"]
    assert len(st) == 16 and all(s["task"] in HARD for s in st)
    assert sum(s["credited"] for s in st) == 6 and sum(s["false_finish"] for s in st) == 1
    assert [s["depth"] for s in st if s["task"] == "libero_10:3"] == [3, 3, 2, 0, 1, 1, 2, 0, 3, 1, 0, 0]


# ------------------------------------------------------------------------------------------------ ladder semantics
def test_one_shot_monotone_and_held_rungs():
    S = dict(STAGED)
    lad = lambda rows, task="libero_10:3": PC.stage_ladder(rows, S, task)  # noqa: E731
    G, P = dict(z=0.10, near=True), dict(sat=(0, 1), xy=(0.3, -0.27))
    # one-shot: grasping and releasing the bowl 10 times is depth 1, the same as grasping it once
    once = lad([r3(1), r3(2, **G), r3(3)])
    many = lad([r3(1)] + [r3(i, **G) if i % 2 else r3(i) for i in range(2, 22)])
    assert once["depth"] == many["depth"] == 1 and once["marks"] == [2] and many["marks"] == [3]
    # grasp needs both the lift and the gripper at the object (a knocked-up object far from the gripper is no grasp)
    assert lad([r3(1), r3(2, z=0.10)])["depth"] == 0 and lad([r3(1), r3(2, z=0.03, near=True)])["depth"] == 0
    # order: the bowl in the drawer without a recorded grasp mark does not count (rung 2 needs rung 1 first)
    assert lad([r3(1), r3(2, **P), r3(3, **P)])["depth"] == 0
    # place must still hold at the last live mark
    assert lad([r3(1), r3(2, **G), r3(3, **P)])["depth"] == 2
    assert lad([r3(1), r3(2, **G), r3(3, **P), r3(4)])["depth"] == 1
    # closing the drawer counts only at a mark where the bowl is in (the empty-drawer dead end earns nothing)
    assert lad([r3(1), r3(2, **G), r3(3, cl0=0.6), r3(4, **P)])["depth"] == 2
    assert lad([r3(1), r3(2, **G), r3(3, **P), r3(4, cl0=0.25, **P)])["depth"] == 3
    assert lad([r3(1), r3(2, cl0=0.6), r3(3, sat=(1, 0), cl0=1.0)])["depth"] == 0
    # a later rung does not survive an undone placement (prefix): closing reached, then the bowl taken out again
    assert lad([r3(1), r3(2, **G), r3(3, **P), r3(4, cl0=0.3, **P), r3(5, cl0=0.3)])["depth"] == 1
    # rows without the per-goal bookkeeping / mismatched ids / no ladder for the task -> None (no credit)
    assert PC.stage_ladder([{"request_index": 1, "f": {}}], S, "libero_10:3") is None
    assert PC.stage_ladder([r3(1), r3(2, **G)], S, "libero_object:2") is None
    assert PC.stage_ladder([dict(r3(1), goal_ids=["goal_0", "goal_9"])], S, "libero_10:3") is None
    # a rung condition already true at the first live mark (e.g. a mid-task start) blocks all credit
    assert lad([r3(1, **P), r3(2, **P)])["blocked"] == "rung_true_at_first_mark"


def _mk(goals, i, sat, tp=None, dest=None, dist=None, cl=None, eef=None):
    n = len(goals)
    return {"request_index": i, "f": {}, "goal_ids": goals, "goal_sat": [bool(s) for s in sat], "goal_closure": cl or [None] * n,
            "goal_dist": dist or [None] * n, "goal_tp": tp or [None] * n, "goal_dest": dest or [None] * n, "eef": eef or [0.45, 0.0, 0.26]}


def test_two_object_ladder_and_initial_goal_guard():
    """libero_10_6 / _8: grasp obj1 / place obj1 (held) / grasp obj2 while obj1 stays placed; libero_10_8's goal_2 (stove on) holds
    initially: switching it off blocks the credit"""
    g = ["goal_0", "goal_1", "goal_2"]
    A0, B0 = [0.37, 0.25, 0.05], [0.58, 0.14, 0.05]
    up = lambda p, dz=0.1: [p[0], p[1], p[2] + dz]  # noqa: E731
    near = lambda p: [p[0], p[1], p[2] + 0.05]  # noqa: E731
    on = [0.6, -0.2, 0.0]
    m = lambda i, sat, a, b, eef=None: _mk(g, i, sat, tp=[a, b, None], eef=eef)  # noqa: E731
    lad = lambda rows: PC.stage_ladder(rows, STAGED, "libero_10:8")["depth"]  # noqa: E731
    first = m(1, (0, 0, 1), A0, B0)
    assert lad([first, m(2, (0, 0, 1), up(A0), B0, near(up(A0)))]) == 1
    assert lad([first, m(2, (0, 0, 1), up(A0), B0, near(up(A0))), m(3, (1, 0, 1), on, B0)]) == 2
    full = [first, m(2, (0, 0, 1), up(A0), B0, near(up(A0))), m(3, (1, 0, 1), on, B0), m(4, (1, 0, 1), on, up(B0), near(up(B0)))]
    assert lad(full) == 3
    # regrasping the already placed pot is not "grasp obj2"; grasping obj2 after obj1 fell off is not either
    assert lad([first, m(2, (0, 0, 1), up(A0), B0, near(up(A0))), m(3, (1, 0, 1), on, B0), m(4, (0, 0, 1), up(on), B0, near(up(on)))]) == 1
    assert lad([first, m(2, (0, 0, 1), up(A0), B0, near(up(A0))), m(3, (1, 0, 1), on, B0), m(4, (0, 0, 1), A0, up(B0), near(up(B0)))]) == 1
    # stove switched off at the end: no credit at all
    off = PC.stage_ladder(full + [m(5, (1, 0, 0), on, up(B0), near(up(B0)))], STAGED, "libero_10:8")
    assert off["depth"] == 0 and off["blocked"] == "initial_goal_undone"


def test_dest_and_toward_rungs():
    # goal_3: the top drawer region moved >= 5 cm (drawer opened), then grasp the bowl; carrying the bowl toward a closed drawer is nothing
    g = ["goal_0"]
    D0, B0 = [0.70, -0.24, 0.18], [0.55, 0.0, -0.014]
    open_ = [0.70, -0.16, 0.18]
    mk = lambda i, dest, bowl, eef=None, sat=0: _mk(g, i, (sat,), tp=[bowl], dest=[dest], eef=eef)  # noqa: E731
    lad = lambda rows: PC.stage_ladder(rows, STAGED, "libero_goal:3")["depth"]  # noqa: E731
    lifted = [B0[0], B0[1], B0[2] + 0.1]
    eef = [lifted[0], lifted[1], lifted[2] + 0.04]
    assert lad([mk(1, D0, B0), mk(2, D0, lifted, eef)]) == 0           # grasp before the drawer is open: nothing
    assert lad([mk(1, D0, B0), mk(2, open_, B0)]) == 1
    assert lad([mk(1, D0, B0), mk(2, open_, B0), mk(3, open_, lifted, eef)]) == 2
    # goal_5: the plate >= 3 cm closer to its region, still at the end; pushed elsewhere / back again: nothing
    t = lambda i, d, sat=0: _mk(g, i, (sat,), dist=[d])  # noqa: E731
    lad5 = lambda rows: PC.stage_ladder(rows, STAGED, "libero_goal:5")["depth"]  # noqa: E731
    assert lad5([t(1, 0.19), t(2, 0.15)]) == 1 and lad5([t(1, 0.19), t(2, 0.17)]) == 0
    assert lad5([t(1, 0.19), t(2, 0.25)]) == 0 and lad5([t(1, 0.19), t(2, 0.12), t(3, 0.19)]) == 0
    # goal_0 / goal_7: joint closure >= 20 % (first reached)
    j = lambda i, c: _mk(g, i, (0,), cl=[c])  # noqa: E731
    assert PC.stage_ladder([j(1, 0.0), j(2, 0.25), j(3, 0.1)], STAGED, "libero_goal:0")["depth"] == 1
    assert PC.stage_ladder([j(1, 0.0), j(2, 0.15)], STAGED, "libero_goal:7")["depth"] == 0


def test_returns_bounded_for_random_episodes():
    """10k random libero_10_3 mark sequences x all terminations: success / missed / false finish unchanged; a credited failure is
    in (0, credit * n / (n + 1)] < missed-finish return; never below the plain return; a model claim (no bookkeeping) never pays"""
    rng = random.Random(7)
    costs = dict(LIVE_COSTS, staged=STAGED)
    n_checked = 0
    for _ in range(2000):
        rows = [r3(1)]
        for i in range(2, rng.randint(2, 12)):
            rows.append(r3(i, sat=(rng.random() < 0.2, rng.random() < 0.4), cl0=rng.choice((0.0, 0.1, 0.25, 0.6, 1.0)),
                           z=rng.choice((0.0, 0.03, 0.08)), near=rng.random() < 0.5, xy=rng.choice(((0, 0), (0.3, -0.27)))))
        for host, term, verdict in ((True, "agent_finished", "segment_success"), (True, "request_cap_exhausted", "segment_success"),
                                    (False, "agent_finished", "false_finish"), (False, "request_cap_exhausted", "no_progress"),
                                    (False, "wall_time_exhausted", "no_progress")):
            r = {"verdict": verdict, "diagnostics": {"host_final_success": host, "termination_reason": term}}
            R, base = PC.staged_return(r, rows, "libero_10:3", -0.3, costs), PC.advantage_return(r, -0.3, costs)
            n_checked += 1
            if host or verdict == "false_finish":
                assert R == base
            else:
                assert R >= base and (R == base or 0 < R <= 0.15 * 3 / 4 + 1e-12) and R < 0.25
            assert PC.staged_return(r, rows, "libero_10:4", -0.3, costs) == base
            assert PC.staged_return(r, [{"request_index": q["request_index"], "f": q["f"]} for q in rows], "libero_10:3", -0.3, costs) == base
    assert n_checked == 10000
    # the pinned spec: every ladder's best partial stays below the missed-finish return 0.25 (and below the base credit)
    assert all(STAGED["credit"] * len(l) / (len(l) + 1) < 0.25 for l in STAGED["ladders"].values())


# ------------------------------------------------------------------------------------------------ critic_features bookkeeping


# ------------------------------------------------------------------------------------------------ C3: early-stop exemption
class _Resp:
    status_code = 200


def _env_case(E, spec, task, values_seq, sat_seq, tmp):
    """drive _early_evaluate mark by mark; returns (es, posted reasons)"""
    import types
    posted = []
    E.EARLY["spec"] = dict(spec, snapshot_root=str(tmp / "snap"))
    E.STATE["proxy"] = "http://proxy"
    E.requests = types.SimpleNamespace(post=lambda url, json=None, timeout=None: (posted.append(json["reason"]), _Resp())[1])
    out = tmp / f"ep-{task.replace(':', '-')}"
    out.mkdir(parents=True, exist_ok=True)
    suite, tid = task.split(":")
    es = E.early_stop_begin(out, "sid-" + task, suite, int(tid), None)
    es["control"] = False
    state = {"n": 0}

    class Crit:
        featurizer = types.SimpleNamespace(featurize=lambda rows, t: rows)
        def values(self, rows):
            import types as _t
            return _t.SimpleNamespace(tolist=lambda: values_seq[:len(rows)])
    E._early_critic = lambda: (Crit(), 7)
    E._early_rows = lambda o, s: [{"request_index": i + 1, "f": {"all_sat": float(sat_seq[i])}} for i in range(state["n"])]
    for n in range(1, len(values_seq) + 1):
        state["n"] = n
        (out / "hidden_request_marks.jsonl").write_text("x" * n)
        if es["decision"] is None:
            E._early_evaluate(es, out)
    return es, posted


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"STAGED_LADDER_TESTS OK ({len(tests)})")

"""Per-request advantages from the simulator critic: a physics-privileged value function over request boundaries.

The critic V(s) reads the simulator state recorded before every student request (hidden_request_marks.jsonl, featurized
by critic_features.episode_states) and predicts the probability that the episode ends in host_final_success. It is
warm-started offline (critic_dir) and, with critic_cotrain, co-trained online (critic_cotrain.py). For one trajectory with live requests 1..T,
states s_1..s_T and final reward R in {0, 1}:

  delta_t = V(s_{t+1}) - V(s_t)   (t < T),    delta_T = R - V(s_T)
  A_t     = sum_k lambda^k delta_{t+k}        (GAE over requests, no discount)

lambda = 0 gives the pure value differences (sum_t A_t = R - V(s_1)); lambda = 1 gives R - V(s_t). The request
advantages of a batch are normalized globally (one mean and std over every scored request; no per-group std).
States are indexed by the driver's request index (marks); the trainer's samples by the proxy's request index. The two
are joined through the response id (driver/requests.jsonl <-> rollout.json "requests"), as the solved-suffix and code
labels are, so batch_advantages keys every advantage by response id and proxy_values maps it onto a rollout.

Critic directory: feature_spec.json {feature_version, features, mean, std, clip?, task_vocab} + either critic_mlp.pt
{"arch": {"in_dim", "hidden"?}, "state_dict"} (one MLP, layout net.*) or critic_default.pt {"arch", "state_dicts"}
(an ensemble in the layout body.* / head.*, V = mean of the members' sigmoids; the warm-start critic released as
critic/critic_default.pt in the CaPtain-artifacts repository). ``features`` selects the critic's inputs among critic_features'
rows, so a physics-only critic is a spec without the request-index / policy-counter features.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import critic_features as cf

TIME_FEATURES = ("req_idx", "env_step", "n_perc", "n_phys", "n_fail_phys", "n_fail_calls", "n_close", "n_recover",
                 "n_nocall_req", "req_since_phys")
SKIP_VERDICTS = ("infrastructure_error", "replay_infidelity")


CRITIC_FILES = ("critic_default.pt", "critic_mlp.pt")

# Streaming: the episode worker threads featurize each episode as it lands (prefetch_rows), so the learner step does not
# read ~256 episodes' mark files over the run-dir mount on its critical path. Bounded (oldest dropped first).
from collections import OrderedDict as _OrderedDict
import threading as _threading
ROWS_CACHE: "_OrderedDict[str, list[dict]]" = _OrderedDict()
_ROWS_LOCK = _threading.Lock()
ROWS_CACHE_MAX = 4096


def episode_live_rows(episode_dir) -> list[dict]:
    rows, _ = cf.episode_states(str(episode_dir))
    return sorted((r for r in rows if not r["replayed"]), key=lambda r: r["request_index"])


def prefetch_rows(episode_dir) -> None:
    """Featurize one finished episode into ROWS_CACHE (errors are left to the learner's own read)."""
    if not episode_dir:
        return
    try:
        rows = episode_live_rows(episode_dir)
    except Exception:  # noqa: BLE001
        return
    with _ROWS_LOCK:
        ROWS_CACHE[str(episode_dir)] = rows
        while len(ROWS_CACHE) > ROWS_CACHE_MAX:
            ROWS_CACHE.popitem(last=False)


def mlp(d: int, h: int = 256, p: float = 0.1, layout: str = "net"):
    """The critic MLP (3 x 256 ReLU, dropout; one logit). layout "net": keys net.*; "body": keys body.* + head.*.
    Returns (module, logits function). torch is imported here so that the launcher (config validation) does not need it."""
    from torch import nn
    layers = [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, h), nn.ReLU(), nn.Dropout(p)]
    m = nn.Module()
    if layout == "net":
        m.net = nn.Sequential(*layers, nn.Linear(h, 1))
        return m, lambda x: m.net(x).squeeze(-1)
    m.body, m.head = nn.Sequential(*layers), nn.Linear(h, 1)
    return m, lambda x: m.head(m.body(x))[:, 0]


def validate_critic_dir(path) -> dict:
    """The parsed feature spec of a critic directory; raises ValueError when it cannot be scored with critic_features."""
    path = Path(path)
    try:
        spec = json.loads((path / "feature_spec.json").read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"critic {path}: unreadable feature_spec.json ({exc})") from exc
    if not any((path / name).is_file() for name in CRITIC_FILES):
        raise ValueError(f"critic {path}: none of {CRITIC_FILES}")
    if spec.get("feature_version") != cf.FEATURE_VERSION:
        raise ValueError(f"critic {path}: feature version {spec.get('feature_version')} != {cf.FEATURE_VERSION}")
    feats = spec.get("features")
    if not isinstance(feats, list) or not feats or len(spec.get("mean") or []) != len(feats) or len(spec.get("std") or []) != len(feats):
        raise ValueError(f"critic {path}: features/mean/std must be lists of equal nonzero length")
    if not isinstance(spec.get("task_vocab"), list):
        raise ValueError(f"critic {path}: task_vocab missing")
    return spec


class FrozenCritic:
    """V(s) of every live request of an episode directory, on CPU, in eval mode (no dropout)."""

    def __init__(self, path):
        import torch
        self.path = Path(path)
        self.spec = validate_critic_dir(self.path)
        self.features = list(self.spec["features"])
        self.tasks = {t: i for i, t in enumerate(self.spec["task_vocab"])}
        ensemble = (self.path / CRITIC_FILES[0]).is_file()
        ck = torch.load(self.path / CRITIC_FILES[0 if ensemble else 1], map_location="cpu", weights_only=False)
        if ensemble and (list(ck.get("features", self.features)) != self.features or list(ck.get("task_vocab", self.spec["task_vocab"]))
                         != list(self.spec["task_vocab"])):
            raise ValueError(f"critic {self.path}: checkpoint features / task vocabulary differ from feature_spec.json")
        in_dim = int(ck["arch"]["in_dim"])
        if in_dim != len(self.features) + len(self.tasks):
            raise ValueError(f"critic {self.path}: in_dim {in_dim} != {len(self.features)} features + {len(self.tasks)} tasks")
        self.members = []
        for sd in (ck["state_dicts"] if ensemble else [ck["state_dict"]]):
            module, logits = mlp(in_dim, int(ck["arch"].get("hidden", 256)), layout="body" if ensemble else "net")
            module.load_state_dict(sd)
            module.eval()
            self.members.append((module, logits))
        tau = (self.spec.get("ood_std_tau_train_percentiles") or {}).get("99")
        self.std_tau = float(tau) if tau is not None else None
        self.mean = torch.tensor(self.spec["mean"], dtype=torch.float64)
        self.std = torch.tensor(self.spec["std"], dtype=torch.float64)
        self.clip = float(self.spec.get("clip", 10))
        self.uses_time = sorted(set(self.features) & set(TIME_FEATURES))
        # "same_t": a time feature (req_idx) held fixed inside the value difference (line C round 2 "phys+t_sameT"):
        # delta_t = V(s_{t+1}, t) - V(s_t, t), so the passage of time itself creates no credit; delta_T = R - V(s_T, T)
        self.same_t = self.spec.get("same_t")
        if self.same_t is not None and self.same_t not in self.features:
            raise ValueError(f"critic {self.path}: same_t feature {self.same_t!r} is not an input feature")

    def featurize(self, rows: list[dict], task_key: str):
        import torch
        x = torch.tensor([[float(r["f"][k]) for k in self.features] for r in rows], dtype=torch.float64)
        x = torch.nan_to_num(x, nan=-1.0, posinf=10.0, neginf=-10.0)
        x = ((x - self.mean) / self.std).clamp(-self.clip, self.clip)
        onehot = torch.zeros(len(rows), len(self.tasks), dtype=torch.float64)
        if task_key in self.tasks:              # an unseen task scores with an all-zero one-hot
            onehot[:, self.tasks[task_key]] = 1.0
        return torch.cat([x, onehot], 1).float()

    def score_rows(self, rows: list[dict], task_key: str, with_std: bool = False):
        """V per row (ensemble: mean of the members' sigmoids); with_std: also the members' std (0 for one MLP)."""
        import torch
        if not rows:
            return ([], []) if with_std else []
        with torch.no_grad():
            x = self.featurize(rows, task_key)
            p = torch.stack([torch.sigmoid(logits(x)) for _, logits in self.members]).double()
        v = p.mean(0).tolist()
        return (v, p.std(0, unbiased=False).tolist()) if with_std else v

    def live_rows(self, episode_dir) -> list[dict]:
        cached = ROWS_CACHE.get(str(episode_dir))
        if cached is not None:
            return [dict(r) for r in cached]
        return episode_live_rows(episode_dir)

    def next_rows_same_t(self, rows: list[dict]) -> list[dict | None]:
        """For each live row t < T: the next row's features with the time feature set to row t's (None for the last)."""
        if self.same_t is None:
            return [None] * len(rows)
        out = []
        for i, r in enumerate(rows):
            if i + 1 >= len(rows):
                out.append(None)
                continue
            nxt = rows[i + 1]
            out.append(dict(nxt, f=dict(nxt["f"], **{self.same_t: r["f"][self.same_t]})))
        return out

    def next_values_same_t(self, rows: list[dict], task_key: str) -> list[float | None]:
        """V(s_{t+1}, t) per live row (None for the last row, and for every row without same_t)."""
        nr = self.next_rows_same_t(rows)
        have = [i for i, r in enumerate(nr) if r is not None]
        v = self.score_rows([nr[i] for i in have], task_key) if have else []
        out: list[float | None] = [None] * len(rows)
        for i, x in zip(have, v):
            out[i] = x
        return out

    def values(self, episode_dir, task_key: str, with_std: bool = False):
        """[(request_index, V(s_t))] over the live (not replayed) requests, in request order; [] without marks.
        with_std: ([(request_index, V)], [ensemble std])."""
        rows, _ = cf.episode_states(str(episode_dir))
        rows = sorted((r for r in rows if not r["replayed"]), key=lambda r: r["request_index"])
        v, sd = self.score_rows(rows, task_key, with_std=True)
        pairs = list(zip((int(r["request_index"]) for r in rows), v))
        return (pairs, sd) if with_std else pairs


def driver_response_ids(episode_dir) -> dict[int, str]:
    """{driver request index: response id} of the live (not replayed) requests."""
    return {int(r["index"]): str(r["response_id"]) for r in cf.jl(str(Path(episode_dir) / "driver" / "requests.jsonl"))
            if r.get("response_id") and not r.get("replayed") and type(r.get("index")) is int}


def proxy_values(by_response: dict[str, float] | None, raw: dict) -> tuple[dict[int, float] | None, int]:
    """Advantages keyed by response id -> {proxy request index: A} for the rollout's selected requests (a selected request
    without a matching response id gets 0) and the number of such unmapped selected requests."""
    if by_response is None:
        return None, 0
    rid = {int(q["request_index"]): str(q.get("response_id") or "") for q in raw.get("requests") or []}
    out, unmapped = {}, 0
    for q in raw["training_requests"]:
        index = int(q["request_index"])
        a = by_response.get(rid.get(index, ""))
        if a is None:
            unmapped += 1
        out[index] = 0.0 if a is None else float(a)
    return out, unmapped


def request_advantages(values: list[tuple[int, float]], reward: float, lam: float,
                       next_values: list[float | None] | None = None, rewards: dict[int, float] | None = None) -> dict[int, float]:
    """GAE over one trajectory's requests: {request_index: A_t} (see the module docstring). next_values (same-t critics):
    V(s_{t+1}, t) per request, used instead of the next request's own value V(s_{t+1}, t+1); the last request uses R."""
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lambda must be in [0, 1]")
    out: dict[int, float] = {}
    acc = 0.0
    for t in range(len(values) - 1, -1, -1):
        if t + 1 >= len(values):
            nxt = float(reward)
        elif next_values is not None and next_values[t] is not None:
            nxt = next_values[t]
        else:
            nxt = values[t + 1][1]
        imm = rewards.get(values[t][0], 0.0) if rewards else 0.0      # per-request immediate reward (costs)
        acc = (imm + nxt - values[t][1]) + lam * acc
        out[values[t][0]] = acc
    return out


def normalize_global(per_traj: list[dict[int, float] | None], eps: float = 1e-6) -> list[dict[int, float] | None]:
    """(A - mean) / (std + eps) with one mean/std over every request of the batch (None entries are left out)."""
    flat = [a for d in per_traj if d for a in d.values()]
    if not flat:
        return [None if d is None else dict(d) for d in per_traj]
    mean = sum(flat) / len(flat)
    std = math.sqrt(sum((a - mean) ** 2 for a in flat) / len(flat))
    if std < 1e-8:
        return [None if d is None else {k: 0.0 for k in d} for d in per_traj]
    return [None if d is None else {k: (a - mean) / (std + eps) for k, a in d.items()} for d in per_traj]


def auc(scores: list[float], labels: list[int]) -> float | None:
    """ROC AUC (Mann-Whitney, ties count half); None with a single class."""
    pos = sum(1 for y in labels if y)
    neg = len(labels) - pos
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_pos = sum(r for r, y in zip(ranks, labels) if y)
    return (rank_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def calibration(per_traj_values: list[list[tuple[int, float]]], rewards: list[float]) -> dict:
    """Critic monitoring on one batch of fresh rollouts: AUC / Brier at the first state and over every state (each
    state labelled with its trajectory's final reward)."""
    first = [(v[0][1], int(r > 0.5)) for v, r in zip(per_traj_values, rewards) if v]
    states = [(p, int(r > 0.5)) for v, r in zip(per_traj_values, rewards) for _, p in v]
    out = {"critic/trajectories": len(first), "critic/states": len(states),
           "critic/success_rate": sum(y for _, y in first) / max(1, len(first))}
    for name, rows in (("s0", first), ("states", states)):
        if rows:
            out[f"critic/auc_{name}"] = auc([p for p, _ in rows], [y for _, y in rows])
            out[f"critic/brier_{name}"] = sum((p - y) ** 2 for p, y in rows) / len(rows)
            out[f"critic/mean_v_{name}"] = sum(p for p, _ in rows) / len(rows)
            out[f"critic/cal_{name}"] = out[f"critic/mean_v_{name}"] - sum(y for _, y in rows) / len(rows)   # calibration-in-the-large
    return out


def termination(r: dict) -> str:
    """the episode's termination reason: a rollouts.jsonl row carries it as `termination`, the trainer's reward result
    (reward.score_episode) only as diagnostics.termination_reason"""
    return str(r.get("termination") or (r.get("diagnostics") or {}).get("termination_reason") or "")


def budget_exhausted(r: dict) -> bool:
    """the episode ended because a budget ran out (request cap, physical-action budget, wall time), not by finish()"""
    return termination(r).endswith("_exhausted")


def advantage_return(r: dict, false_finish_reward: float = 0.0, costs: dict | None = None) -> float:
    """The return R of the last-step advantage A_T = R - V(s_T): 1 on host success,
    false_finish_reward (e.g. -0.3) when the agent called finish() and the host predicate does not hold (env verdict
    false_finish; the completion gate is unchanged), else 0. The critic stays a success-probability model: its target is
    the 0/1 host success, never this R. false_finish_reward 0 = no false-finish penalty.
    costs (None = the behaviour above, bit for bit): missed_finish_return = return of a host
    success the agent did not finish (termination is not agent_finished; e.g. +0.5), exhaust_reward = return of a failure that
    ended on an exhausted budget (termination *_exhausted: request cap, action budget, wall; e.g. -0.1)."""
    if bool((r.get("diagnostics") or {}).get("host_final_success")):
        if costs and costs.get("missed_finish_return") is not None and termination(r) != "agent_finished":
            return float(costs["missed_finish_return"])
        return 1.0
    if r.get("verdict") == "false_finish":
        return float(false_finish_reward)
    if costs and costs.get("exhaust_reward") is not None and budget_exhausted(r):
        return float(costs["exhaust_reward"])
    return 0.0


# ---------------------------------------------------------------- staged ONE-SHOT subgoal reward (hard tasks)
# costs["staged"] = {"credit": c, "ladders": {suite:task_id: [rung, ...]}, "grasp": {"lift_m": 0.05, "eef_m": 0.13}} (absent = off,
# advantage_return bit for bit). The ladder of a task is its subgoal sequence (configs/staged-reward-spec.json);
# every rung is judged ONLY from the host's request marks (hidden_request_marks: privileged simulator state before each request, never
# shown to the model; never from the model's text, tool claims or finish calls). A rung is credited at most once per episode (depth of
# the ladder, not a count of events), and only in order: rung k counts at the first mark t_k >= t_{k-1} where its condition holds while
# every earlier "held" rung (place / toward) also holds at t_k. Rung kinds:
#   grasp      any listed goal's target object is >= lift_m above its first-mark height with the gripper (eef) within eef_m of it
#              (first reached; the stepwise analysis' grasp milestone) - transient by nature, so it never needs to persist;
#   place      a listed goal's LIBERO predicate holds (first reached in order) AND one of them still holds at the last live mark (held:
#              a placement that was undone is no stage; dragging an object through a region and out again earns nothing);
#   grasp_next with one listed goal satisfied, another listed goal's target is grasped while its own predicate is false (second object);
#   joint      the goal's normalised joint distance has closed by >= closure of its first-mark value (first reached; articulated parts
#              rarely move back, and a later rung or success needs the joint anyway);
#   dest       the goal's destination site moved >= move_m from its first-mark position (a drawer opened; first reached);
#   toward     the goal's object distance to its region fell by >= closer_m metres (or the predicate holds) at the mark AND at the
#              last live mark (held; pushing the object anywhere else earns nothing).
# Depth d = the number of consecutive rungs reached (one-shot: repeating a grasp / release / push cannot add credit). For a listed task,
# an episode that is neither a host success nor a false finish and reaches d >= 1 of its n rungs gets the return c * d / (n + 1) instead
# of 0 / exhaust_reward (so <= c * n / (n + 1) < c < missed_finish_return). Host successes (finished 1.0 / missed) and false finishes
# (-0.3) are never changed; the critic target (host success) is unchanged. No credit when a goal satisfied at the first live mark is
# undone at the last one (e.g. libero_10_8's stove switched off), when a rung condition already holds at the first live mark, or when
# the rows lack the per-goal bookkeeping.
STAGED_RUNGS = {"grasp": (), "place": (), "grasp_next": (), "joint": ("closure",), "dest": ("move_m",), "toward": ("closer_m",)}
STAGED_HELD = ("place", "toward")
STAGED_GRASP = {"lift_m": 0.05, "eef_m": 0.13}


def _dist3(a, b) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _rung_holds(rung: dict, row: dict, first: dict, grasp: dict) -> bool:
    ids = row["goal_ids"]
    gi = [ids.index(g) for g in rung["goals"]]
    sat = row["goal_sat"]
    kind = rung["rung"]

    def held(i):
        tp, tp0, eef = (row.get("goal_tp") or [None] * len(ids))[i], (first.get("goal_tp") or [None] * len(ids))[i], row.get("eef")
        return bool(tp and tp0 and eef) and float(tp[2]) - float(tp0[2]) >= grasp["lift_m"] and _dist3(eef, tp) <= grasp["eef_m"]
    if kind == "grasp":
        return any(held(i) for i in gi)
    if kind == "place":
        return any(sat[i] for i in gi)
    if kind == "grasp_next":
        return any(held(i) and not sat[i] and any(sat[j] for j in gi if j != i) for i in gi)
    i = gi[0]
    if kind == "joint":
        c = (row.get("goal_closure") or [None] * len(ids))[i]
        return bool(sat[i]) or (c is not None and c >= rung["closure"])
    if kind == "dest":
        d, d0 = (row.get("goal_dest") or [None] * len(ids))[i], (first.get("goal_dest") or [None] * len(ids))[i]
        return bool(d and d0) and _dist3(d, d0) >= rung["move_m"]
    if kind == "toward":
        x, x0 = (row.get("goal_dist") or [None] * len(ids))[i], (first.get("goal_dist") or [None] * len(ids))[i]
        return bool(sat[i]) or (isinstance(x, (int, float)) and isinstance(x0, (int, float)) and x0 - x >= rung["closer_m"])
    raise ValueError(f"unknown rung {kind}")


def stage_ladder(rows: list[dict], spec: dict, task_key: str) -> dict | None:
    """The staged-reward ladder of one episode from its live rows (rows[0] = the first live mark = the reference state): {"depth": d,
    "n": number of rungs, "marks": [request_index where rung k was first reached, ...], "blocked": reason | None}. None = the task
    has no ladder or the rows lack the per-goal bookkeeping (no credit)."""
    ladder = (spec.get("ladders") or {}).get(task_key)
    if not ladder or not rows:
        return None
    first, last = rows[0], rows[-1]
    ids = first.get("goal_ids")
    need = {g for rung in ladder for g in rung["goals"]}
    if not ids or any(q.get("goal_ids") != ids or len(q.get("goal_sat") or []) != len(ids) for q in rows) or need - set(ids):
        return None
    grasp = dict(STAGED_GRASP, **(spec.get("grasp") or {}))
    out = {"depth": 0, "n": len(ladder), "marks": [], "blocked": None}
    if any(s0 and not s1 for s0, s1 in zip(first["goal_sat"], last["goal_sat"])):
        out["blocked"] = "initial_goal_undone"
        return out
    if any(_rung_holds(rung, first, first, grasp) for rung in ladder):
        out["blocked"] = "rung_true_at_first_mark"
        return out
    t = 0
    for k, rung in enumerate(ladder):
        held_before = [rg for rg in ladder[:k] if rg["rung"] in STAGED_HELD]
        hit = next((j for j in range(t, len(rows)) if _rung_holds(rung, rows[j], first, grasp)
                    and all(_rung_holds(rg, rows[j], first, grasp) for rg in held_before)), None)
        if hit is None or (rung["rung"] in STAGED_HELD and not _rung_holds(rung, last, first, grasp)):
            break
        out["depth"] = k + 1
        out["marks"].append(int(rows[hit]["request_index"]))
        t = hit
    return out


def staged_return(r: dict, rows: list[dict], task_key: str, false_finish_reward: float = 0.0, costs: dict | None = None) -> float:
    """advantage_return with the staged-reward staged one-shot subgoal reward (block comment above). Off (no costs["staged"], or a task
    without a ladder) = advantage_return bit for bit."""
    base = advantage_return(r, false_finish_reward, costs)
    st = costs.get("staged") if costs else None
    if not st or task_key not in (st.get("ladders") or {}):
        return base
    if bool((r.get("diagnostics") or {}).get("host_final_success")) or r.get("verdict") == "false_finish":
        return base
    lad = stage_ladder(rows, st, task_key)
    if not lad or lad["depth"] < 1:
        return base
    return float(st["credit"]) * lad["depth"] / (lad["n"] + 1)


def staged_spec_error(spec, missed_finish_return) -> str | None:
    """validation of a staged-reward spec (trainer and launcher); missed_finish_return = the effective return of a missed finish
    (None = 1.0, a missed finish is paid like a finished success)"""
    if not isinstance(spec, dict) or set(spec) - {"credit", "ladders", "grasp"} or not {"credit", "ladders"} <= set(spec):
        return "staged_reward must be {credit, ladders, grasp?}"
    mfr = 1.0 if missed_finish_return is None else float(missed_finish_return)
    if type(spec["credit"]) not in (int, float) or not 0 < spec["credit"] < mfr:
        return f"staged_reward credit must be in (0, missed_finish_return = {mfr}) (a partial episode stays below a missed finish)"
    gr = spec.get("grasp")
    if gr is not None and (not isinstance(gr, dict) or set(gr) != {"lift_m", "eef_m"} or any(
            type(v) not in (int, float) or not 0 < v < 0.5 for v in gr.values())):
        return "staged_reward grasp must be {lift_m, eef_m} in (0, 0.5) m"
    lads = spec["ladders"]
    gid = lambda x: isinstance(x, str) and x.startswith("goal_") and x[5:].isdigit()  # noqa: E731
    if not isinstance(lads, dict) or not lads or not all(isinstance(k, str) and k.count(":") == 1 and k.split(":")[1].isdigit() for k in lads):
        return "staged_reward ladders must be a non-empty {'suite:task_id': [rung, ...]}"
    for task, ladder in lads.items():
        if not isinstance(ladder, list) or not 1 <= len(ladder) <= 6:
            return f"staged_reward ladder {task}: 1-6 rungs"
        for rung in ladder:
            kind = rung.get("rung") if isinstance(rung, dict) else None
            if kind not in STAGED_RUNGS or set(rung) != {"rung", "goals", *STAGED_RUNGS[kind]}:
                return f"staged_reward ladder {task}: rung must be {{rung, goals, <params>}} with rung in {sorted(STAGED_RUNGS)}"
            g = rung["goals"]
            if not isinstance(g, list) or not g or len(set(g)) != len(g) or not all(gid(x) for x in g):
                return f"staged_reward ladder {task}: goals must be distinct goal_i ids"
            if (kind == "grasp_next" and len(g) < 2) or (kind in ("joint", "dest", "toward") and len(g) != 1):
                return f"staged_reward ladder {task}: grasp_next needs >= 2 goals; joint / dest / toward exactly 1"
            for p in STAGED_RUNGS[kind]:
                v = rung[p]
                if type(v) not in (int, float) or not 0 < v < (1 if p == "closure" else 0.5):
                    return f"staged_reward ladder {task}: {p} out of range"
    return None


def request_costs(rows: list[dict], costs: dict | None, task_key: str) -> dict[int, float] | None:
    """Immediate per-request rewards (<= 0) of the action costs, keyed by request_index: -(c_act x physical actions
    of the request beyond the task's reference length (counted cumulatively over the episode; ref 0 = every action) + c_fail x bad
    physical actions of the request). Perception / observation calls are never charged. None when both costs are off.
    costs["post_success_act"] (missed-finish fix C): from the first request whose mark shows all goals satisfied (the host already judges
    success: the state left by the previous request) on, every physical action of the request is charged c_act.
    costs["post_success_req"] (missed-finish fix, part b): from that same first request on, EVERY request
    (perception / measurement / finish included) is charged this flat cost on top; before the host success perception stays free.
    costs["post_success_cap"] (optional, with b): the episode's whole post-success charge (b + the action / bad-action costs of the
    requests from the first success on) is capped at this total (charges beyond it are dropped), so a missed finish
    (missed_finish_return - post-success charge) always stays above a false finish declared at the success time (false_finish_reward)
    by >= missed_finish_return - false_finish_reward - cap."""
    c_req = float(costs.get("post_success_req") or 0.0) if costs else 0.0
    cap = costs.get("post_success_cap") if costs else None
    if not costs or not (costs.get("c_act") or costs.get("c_fail") or c_req):
        return None
    refs = costs.get("ref") or {}
    ref = refs.get(task_key)
    if ref is None and refs:
        ref = refs.get(task_key.split(":")[0] + ":*", 0.0)
    ref = float(ref or 0.0)
    out, done, post_charge = {}, 0, 0.0
    post = bool(costs.get("post_success_act"))      # missed-finish fix part C: charge every action once all goals held
    seen_success = False
    for q in rows:
        n = int(q.get("req_phys") or 0)
        charged = max(0.0, done + n - ref) - max(0.0, done - ref)
        done += n
        if post or c_req:
            seen_success = seen_success or bool((q.get("f") or {}).get("all_sat"))
        if post and seen_success:
            charged = float(n)              # all of this request's physical actions (the ref-based part is included, not added)
        out[int(q["request_index"])] = -(float(costs.get("c_act") or 0.0) * charged + float(costs.get("c_fail") or 0.0) * int(q.get("req_phys_bad") or 0))
        if c_req and seen_success:          # part b: every request after the host's first success judgement
            if cap is None:
                out[int(q["request_index"])] -= c_req
            else:                           # this request's whole post-success charge (actions + bad actions + b) within the cap
                charge = min(c_req - out[int(q["request_index"])], max(0.0, float(cap) - post_charge))
                post_charge += charge
                out[int(q["request_index"])] = -charge
    return out


def first_success_row(rows: list[dict]) -> int | None:
    """position of the first live row whose mark shows all goals satisfied (the first request issued after the host judges success)"""
    for k, q in enumerate(rows):
        if bool((q.get("f") or {}).get("all_sat")):
            return k
    return None


def critic_target(r: dict, costs: dict | None = None) -> float:
    """The co-critic's 0/1 training target: host success; with costs["critic_target"] == "success_finish" (missed-finish fix round 2,
    part c) host success AND the agent ended the episode with finish() (a missed finish and a false finish are both 0)."""
    host = float(bool((r.get("diagnostics") or {}).get("host_final_success")))
    if costs and costs.get("critic_target") == "success_finish":
        return float(bool(host) and termination(r) == "agent_finished")
    return host


def batch_advantages(results: list[dict], critic: FrozenCritic, *, lam: float, normalize: bool = True,
                     false_finish_reward: float = 0.0) -> tuple[list, dict]:
    """Per-request advantages of one batch of rollout results keyed by response id (None for a trajectory that is not
    trained or has no marks) and the critic metrics. R = advantage_return (default: diagnostics.host_final_success); the
    calibration metrics use the 0/1 success. Normalization statistics use every scored request (including any without a
    response id)."""
    values, rewards, raw, stds = [], [], [], []
    missing = 0
    for r in results:
        if not r.get("sample_path") or r.get("verdict") in SKIP_VERDICTS:
            raw.append(None)
            continue
        R = float(bool((r.get("diagnostics") or {}).get("host_final_success")))
        task_key = f'{r["suite"]}:{int(r["task_id"])}'
        rows = critic.live_rows(r["episode_dir"]) if r.get("episode_dir") else []
        vv, sd = critic.score_rows(rows, task_key, with_std=True)
        v = list(zip((int(x["request_index"]) for x in rows), vv))
        stds += sd
        if not v:
            missing += 1
            raw.append(None)
            continue
        values.append(v)
        rewards.append(R)
        nv = critic.next_values_same_t(rows, task_key) if critic.same_t is not None else None
        raw.append(request_advantages(v, advantage_return(r, false_finish_reward), lam, nv))
    metrics = calibration(values, rewards)
    if normalize:
        raw = normalize_global(raw)
    keyed, no_rid = [], 0
    for r, d in zip(results, raw):
        if d is None:
            keyed.append(None)
            continue
        rids = driver_response_ids(r["episode_dir"])
        no_rid += sum(1 for i in d if i not in rids)
        keyed.append({rids[i]: a for i, a in d.items() if i in rids})
    flat = [a for d in raw if d for a in d.values()]
    if len(critic.members) > 1 and stds:
        metrics["critic/ens_std_mean"] = sum(stds) / len(stds)
        if critic.std_tau is not None:
            metrics["critic/frac_std_over_tau99"] = sum(1 for x in stds if x > critic.std_tau) / len(stds)
    metrics.update({"critic/missing_trajectories": missing, "critic/lambda": lam, "critic/same_t": critic.same_t or "", "critic/requests_without_response_id": no_rid,
                    "critic/raw_adv_mean": sum(flat) / max(1, len(flat)),
                    "critic/raw_adv_abs_mean": sum(abs(a) for a in flat) / max(1, len(flat))})
    return keyed, metrics


__all__ = ["FrozenCritic", "TIME_FEATURES", "advantage_return", "stage_ladder", "staged_return", "staged_spec_error", "critic_target", "first_success_row", "budget_exhausted", "termination", "auc", "batch_advantages", "calibration", "driver_response_ids", "normalize_global",
           "proxy_values", "request_advantages", "validate_critic_dir"]

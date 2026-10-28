"""Simulator (physics) critic co-trained with the policy, SAO-style.

V(s_t) = sigmoid( phys(x_t) + hid(h_t) ):
  x_t  the physics features of the state before request t (critic_features, the critic directory's feature spec and
       standardization, task one-hot) -- the frozen-critic pipeline of physics_critic.py;
  h_t  the CURRENT policy's last hidden state at the request boundary (the last prompt token of request t), taken from
       the trainer's own forward of that request and detached: no value gradient reaches the LM, no extra forward.
phys is initialised from the offline critic (one MLP or the mean logit of an ensemble), hid is an MLP on h whose output
layer starts at zero, so at the first step V equals the offline physics critic (the warm start); a warm-start checkpoint
with a trained hid branch (offline value pretraining on student rollouts with their hidden states) can be loaded too.
After every policy step the critic takes K updates (BCE to the episode's final 0/1 reward, Monte-Carlo target as SAO's
lambda_critic = 1) at its own learning rate on that step's states; all ranks keep identical copies (gradients averaged).

Advantages are computed inside the training forward. A trajectory's packed chains are processed newest first, so when a
chain's members are scored the value of the state after its last member (the next chain's first state, or R at the end)
is already known: A_t = sum_k lambda^k delta_{t+k}, delta_t = V(s_{t+1}) - V(s_t), delta_T = R - V(s_T), exactly the
physics_critic GAE but with V from this step's weights. Every eligible request must be trained (no subsampling: an
untrained request has no hidden state); a request without physics features gets A = 0 and does not break the chain.
Advantages are scaled by 1 / sigma with sigma an EMA of the previous steps' advantage std (the current batch's std is
not known before its first backward); no centering (SAO uses none).
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from physics_critic import FrozenCritic


class CoCritic(nn.Module):
    def __init__(self, critic_dir, hidden_dim: int, *, use_hidden: bool = True, width: int = 256):
        super().__init__()
        frozen = FrozenCritic(critic_dir)
        self.featurizer = frozen                    # featurize(rows, task_key) -> standardized physics + task one-hot
        self.phys = nn.ModuleList(m for m, _ in frozen.members)
        self.phys_logits = [f for _, f in frozen.members]
        for p in self.phys.parameters():
            p.requires_grad_(True)
        self.use_hidden = use_hidden
        self.hidden_dim = hidden_dim
        if use_hidden:
            self.hid = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, width), nn.ReLU(), nn.Linear(width, 1))
            nn.init.zeros_(self.hid[-1].weight)
            nn.init.zeros_(self.hid[-1].bias)
        warm = Path(critic_dir) / "cocritic_warm.pt"     # optional offline value pretraining (physics + hidden)
        if warm.is_file():
            self.load_state_dict(torch.load(warm, map_location="cpu", weights_only=False)["state_dict"])

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
        """Logits for standardized physics inputs x [n, F] and boundary hidden states h [n, H] (detached by the caller)."""
        z = torch.stack([f(x) for f in self.phys_logits]).mean(0)
        if self.use_hidden:
            if h is None:
                raise ValueError("this critic reads the policy hidden state")
            z = z + self.hid(h.float())[:, 0]
        return z

    def values(self, x, h=None) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self(x, h))


def chain_advantages(values: list[float | None], next_value: float, acc: float, lam: float,
                     next_values: list[float | None] | None = None) -> tuple[list[float], float, float]:
    """GAE over one chain's members (in chain order) given the value after its last member and the running GAE sum of
    the later chains; returns (A per member, the value before the first member, the running sum). A member without a
    value (None) gets 0 and passes the chain state through unchanged. next_values (same-t critic): V(s_{t+1}, t) per
    member, used instead of the following state's own value (None for the trajectory's last request -> R)."""
    out = [0.0] * len(values)
    for i in range(len(values) - 1, -1, -1):
        if values[i] is None:
            continue
        nxt = next_values[i] if next_values is not None and next_values[i] is not None else next_value
        acc = (nxt - values[i]) + lam * acc
        out[i] = acc
        next_value = values[i]
    return out, next_value, acc


class CreditState:
    """Per-trajectory running state of the newest-first chain pass: {trajectory index: (next value, GAE sum)}."""

    def __init__(self, rewards: dict[int, float]):
        self.state = {i: (float(r), 0.0) for i, r in rewards.items()}

    def step(self, traj: int, values: list[float | None], lam: float, next_values: list[float | None] | None = None) -> list[float]:
        nxt, acc = self.state[traj]
        adv, nxt, acc = chain_advantages(values, nxt, acc, lam, next_values)
        self.state[traj] = (nxt, acc)
        return adv


def sample_order_key(s: dict) -> tuple[int, int]:
    """Co-training order on a rank: each trajectory's chains newest first (the GAE needs the later states' values)."""
    return int(s["index"]), -int(s.get("chain", 0))


def update(critic: CoCritic, opt, xs: torch.Tensor, hs: torch.Tensor | None, ys: torch.Tensor, *, steps: int,
           reduce_sum=None) -> list[float]:
    """K critic updates on this step's states: mean BCE to the final reward over the states of ALL ranks (``reduce_sum(t)``
    sums a tensor over ranks in place; every rank calls update, also without local states, so the copies stay identical).
    Returns the global mean loss before each update."""
    device = next(critic.parameters()).device
    n = torch.tensor([float(len(ys))], dtype=torch.float64, device=device)
    if reduce_sum is not None:
        reduce_sum(n)
    total = max(1.0, float(n))
    losses = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=False)
        loss_sum = torch.zeros((), dtype=torch.float32, device=device)
        if len(ys):
            loss_sum = nn.functional.binary_cross_entropy_with_logits(critic(xs, hs), ys.float(), reduction="sum")
            (loss_sum / total).backward()
        for p in critic.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            if reduce_sum is not None:
                reduce_sum(p.grad)
        ls = loss_sum.detach().reshape(1).double()
        if reduce_sum is not None:
            reduce_sum(ls)
        losses.append(float(ls) / total)
        opt.step()
    return losses


def ema_std(prev: float | None, observed: float, decay: float = 0.9) -> float:
    if prev is None or not math.isfinite(prev):
        return observed
    return decay * prev + (1 - decay) * observed


__all__ = ["CoCritic", "CreditState", "chain_advantages", "ema_std", "sample_order_key", "update"]


class CachedCore:
    """The model core computed once per sample: the trainer reads the boundary hidden states from the same forward that
    sample_loss then reuses (no second forward)."""

    def __init__(self, core):
        self.core, self.out = core, None

    def __call__(self, **inputs):
        if self.out is None:
            self.out = self.core(**inputs)
        return self.out


def physics_inputs(frozen: FrozenCritic, episode_dir, task_key: str) -> dict[str, list[float]]:
    """{response id: standardized physics features + task one-hot} of an episode's live requests (rank 0, CPU)."""
    import critic_features as cf
    from physics_critic import driver_response_ids
    rows, _ = cf.episode_states(str(episode_dir))
    rows = sorted((r for r in rows if not r["replayed"]), key=lambda r: r["request_index"])
    if not rows:
        return {}
    rids = driver_response_ids(episode_dir)
    x = frozen.featurize(rows, task_key)
    out = {rids[int(r["request_index"])]: x[i].tolist() for i, r in enumerate(rows) if int(r["request_index"]) in rids}
    if frozen.same_t is not None:
        # same-t critic: the next state's features with this request's time, under the key NEXT + response id
        nr = frozen.next_rows_same_t(rows)
        have = [i for i, r in enumerate(nr) if r is not None and int(rows[i]["request_index"]) in rids]
        if have:
            xn = frozen.featurize([nr[i] for i in have], task_key)
            out.update({NEXT + rids[int(rows[i]["request_index"])]: xn[k].tolist() for k, i in enumerate(have)})
    return out


NEXT = "next_same_t:"


def chain_members(sample: dict, raw: dict, phys: dict[str, list[float]] | None) -> list[tuple]:
    """(boundary position = last prompt token, completion tokens, physics input or None, same-t next-state input or None)
    of a built sample's requests in chain order."""
    by_index = {int(q["request_index"]): q for q in raw["training_requests"]}
    rid = {int(q["request_index"]): str(q.get("response_id") or "") for q in raw.get("requests") or []}
    out = []
    for index in sample.get("packed_requests") or [sample["request_index"]]:
        q = by_index[int(index)]
        key = rid.get(int(index), "")
        out.append((len(q["prompt_token_ids"]) - 1, len(q["token_ids"]), (phys or {}).get(key), (phys or {}).get(NEXT + key) if key else None))
    if sum(m[1] for m in out) != int(sample["n_tokens"]):
        raise ValueError("chain members do not cover the sample's target tokens")
    return out


def new_buffers() -> dict:
    return {"x": [], "h": [], "y": [], "rows": [], "adv": []}


def score_chain(critic: CoCritic, hidden_seq: torch.Tensor, members: list, traj: int, chain: int, reward: float,
                credit: CreditState, lam: float, sigma: float, buf: dict) -> torch.Tensor:
    """One chain in the training forward: the boundary hidden states (rows of ``hidden_seq`` = the forward's
    last_hidden_state[0]) at the members' positions, detached; the critic's values; the chain's GAE advantages given the
    later chains (``credit``); the per-target-token advantage tensor (A / sigma per member's completion tokens). The states
    (physics, hidden, reward) and values go to ``buf`` for the critic update and the calibration log."""
    device = hidden_seq.device
    pos = torch.tensor([m[0] for m in members], device=device)
    h = hidden_seq.index_select(0, pos).detach().float()
    have = [k for k, m in enumerate(members) if m[2] is not None]
    vals: list = [None] * len(members)
    if have:
        xk = torch.tensor([members[k][2] for k in have], dtype=torch.float32, device=device)
        hk = h[have] if critic.use_hidden else None
        for k, v in zip(have, critic.values(xk, hk).tolist()):
            vals[k] = v
        buf["x"].append(xk)
        if hk is not None:
            buf["h"].append(hk)
        buf["y"] += [float(reward)] * len(have)
        buf["rows"] += [(traj, -int(chain), k, vals[k]) for k in have]
    nvals: list = [None] * len(members)
    nhave = [k for k, m in enumerate(members) if len(m) > 3 and m[3] is not None and vals[k] is not None]
    if nhave:
        if critic.use_hidden:
            raise ValueError("a same-t critic with the policy hidden input is not supported (no next-boundary hidden here)")
        xn = torch.tensor([members[k][3] for k in nhave], dtype=torch.float32, device=device)
        for k, v in zip(nhave, critic.values(xn, None).tolist()):
            nvals[k] = v
    adv = credit.step(traj, vals, lam, nvals if nhave else None)
    buf["adv"] += [a for a, v in zip(adv, vals) if v is not None]
    return torch.cat([torch.full((m[1],), a / sigma, dtype=torch.float32) for m, a in zip(members, adv)])


def physics_only_advantages(critic: CoCritic, frozen: FrozenCritic, results: list[dict], *, lam: float, sigma: float,
                            device, false_finish_reward: float = 0.0, costs: dict | None = None,
                            group_keys: list | None = None) -> tuple[list[dict | None], dict]:
    """Physics-only co-training (critic_hidden none; the CaPtain pipeline): the critic reads no policy hidden state,
    so its values need no training forward. Every live request state of every scored trajectory is scored here with the
    co-trained critic (before this step's update), same-t next values when the critic has a time feature, GAE lambda over
    ALL requests (identical to the newest-first in-forward chain GAE when every request is trained), A / sigma.
    Returns (per result: {response id: A / sigma} or None, buffers for the critic update/calibration over all states).
    group_keys (group sampling; None = the above bit for bit): per result the key of its start
    group (G rollouts of one start). Each scored episode's advantages become  A_t = (G_i - mean_group G) + (A_t^GAE - mean_t A^GAE):
    the episode-level (outcome) part is the group-relative total return G_i = terminal return + sum of the immediate costs,
    the per-request shaping stays the critic's GAE minus its episode mean. Episodes whose group has fewer than 2 scored members
    keep the plain GAE advantages. buf["adv"] holds the advantages actually used (sigma is their EMA std)."""
    from physics_critic import (SKIP_VERDICTS, budget_exhausted, critic_target, driver_response_ids, first_success_row,
                                request_advantages, request_costs, stage_ladder, staged_return, termination)
    round2 = bool(costs and (costs.get("post_success_req") or costs.get("critic_target")))   # missed-finish round 2 (b / c) on
    buf = new_buffers()
    keyed: list[dict | None] = []
    pend: dict[int, tuple] = {}     # group sampling: result index -> (adv, total return, rids, start in buf["adv"])
    for i, r in enumerate(results):
        if not r.get("sample_path") or r.get("verdict") in SKIP_VERDICTS or not r.get("episode_dir"):
            keyed.append(None)
            continue
        task_key = f'{r["suite"]}:{int(r["task_id"])}'
        rows = frozen.live_rows(r["episode_dir"])
        if not rows:
            keyed.append(None)
            continue
        host = float(bool((r.get("diagnostics") or {}).get("host_final_success")))
        R = critic_target(r, costs)         # = host unless costs["critic_target"] == "success_finish" (round 2, part c)
        r["critic_R"] = R
        if round2:
            r["host_R"] = host
        x = frozen.featurize(rows, task_key).to(device)
        v = critic.values(x).tolist()
        nv = None
        if frozen.same_t is not None:
            nr = frozen.next_rows_same_t(rows)
            have = [k for k, q in enumerate(nr) if q is not None]
            nv = [None] * len(rows)
            if have:
                for k, val in zip(have, critic.values(frozen.featurize([nr[k] for k in have], task_key).to(device)).tolist()):
                    nv[k] = val
        values = [(int(q["request_index"]), val) for q, val in zip(rows, v)]
        # the GAE's last step uses A_T = R_adv - V(s_T) (false finish -> false_finish_reward); the critic target stays R
        imm = request_costs(rows, costs, task_key)
        # staged one-shot subgoal reward (costs["staged"]; absent = advantage_return): a laddered task's partial episode
        ret = staged_return(r, rows, task_key, false_finish_reward, costs)
        adv = request_advantages(values, ret, lam, nv, imm)
        if costs is not None:       # per-episode cost statistics for the metrics
            buf.setdefault("cost_stats", []).append({"phys": sum(int(q.get("req_phys") or 0) for q in rows),
                                                     "bad": sum(int(q.get("req_phys_bad") or 0) for q in rows),
                                                     "perc": sum(int(q.get("req_perc") or 0) for q in rows),
                                                     "requests": len(rows), "cost": -sum((imm or {}).values()),
                                                     "R": ret,
                                                     "missed_finish": bool(host) and termination(r) != "agent_finished",
                                                     "exhausted": budget_exhausted(r)})
            st = costs.get("staged")
            if st and task_key in (st.get("ladders") or {}):      # staged reward monitoring (laddered tasks only)
                lad = stage_ladder(rows, st, task_key) or {}
                buf.setdefault("staged_stats", []).append({"task": task_key, "depth": lad.get("depth", 0), "n": lad.get("n"),
                                                           "R": ret, "host_success": bool(host), "false_finish": r.get("verdict") == "false_finish",
                                                           "credited": not host and r.get("verdict") != "false_finish" and lad.get("depth", 0) > 0})
            if round2:          # requests issued after the host's first success judgement (incl. the finish request)
                fs = first_success_row(rows)
                buf["cost_stats"][-1]["post_req"] = 0 if fs is None else len(rows) - fs
                buf["cost_stats"][-1]["host_success"] = bool(host)
        rids = driver_response_ids(r["episode_dir"])
        if group_keys is not None:
            pend[i] = (adv, ret + sum((imm or {}).values()), rids, len(buf["adv"]))
        keyed.append({rids[k]: a / sigma for k, a in adv.items() if k in rids})
        buf["x"].append(x)
        buf["y"] += [R] * len(rows)
        buf["rows"] += [(i, 0, k, val) for k, val in enumerate(v)]
        buf["adv"] += list(adv.values())
    if group_keys is not None:
        _group_relative(keyed, buf, pend, group_keys, sigma)
    return keyed, buf


def _group_relative(keyed: list, buf: dict, pend: dict, group_keys: list, sigma: float) -> None:
    """group sampling: replace each scored episode's mean GAE advantage by its group-relative total return (in place)"""
    groups: dict = {}
    for i in pend:
        if group_keys[i] is not None:
            groups.setdefault(group_keys[i], []).append(i)
    stats = {"groups": 0, "fallback_episodes": 0, "flat_groups": 0, "outcome_abs": 0.0, "episodes": 0, "outcome_mean_absmax": 0.0,
             "group_sizes": {}}
    for i in pend:
        members = groups.get(group_keys[i], [])
        if len(members) < 2:
            stats["fallback_episodes"] += 1
    for members in groups.values():
        if len(members) < 2:
            continue
        stats["groups"] += 1
        totals = [pend[i][1] for i in members]
        base = sum(totals) / len(totals)
        stats["flat_groups"] += max(totals) - min(totals) < 1e-9
        # smoke check: the outcome part (G_i - group mean) is zero-mean in every group
        stats["outcome_mean_absmax"] = max(stats["outcome_mean_absmax"], abs(sum(t - base for t in totals) / len(totals)))
        stats["group_sizes"][len(members)] = stats["group_sizes"].get(len(members), 0) + 1
        for i, total in zip(members, totals):
            adv, _, rids, start = pend[i]
            mean = sum(adv.values()) / len(adv) if adv else 0.0
            new = {k: (total - base) + (a - mean) for k, a in adv.items()}
            keyed[i] = {rids[k]: a / sigma for k, a in new.items() if k in rids}
            buf["adv"][start:start + len(new)] = list(new.values())
            stats["outcome_abs"] += abs(total - base)
            stats["episodes"] += 1
    buf["group_stats"] = stats


__all__ += ["CachedCore", "NEXT", "chain_members", "new_buffers", "physics_inputs", "physics_only_advantages", "score_chain"]

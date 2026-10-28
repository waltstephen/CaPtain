"""SAO-style critic co-training: newest-first chain GAE, warm start equal to the offline physics critic, critic updates,
chain members, response-id physics inputs, the in-forward chain scoring, and the configuration."""
import json
import math
from types import SimpleNamespace

import pytest
import torch

import critic_cotrain as CC
import fresh_run as FR
import physics_critic as PC
import train as T
from test_physics_critic import write_critic, write_episode


def gae_reference(values, R, lam):
    return PC.request_advantages(list(enumerate(values, start=1)), R, lam)


@pytest.mark.parametrize("lam", [0.0, 0.9, 1.0])
def test_newest_first_chain_processing_equals_the_whole_trajectory_gae(lam):
    values = [0.1, 0.12, 0.4, 0.35, 0.5, 0.2]
    R = 1.0
    chains = [[0, 1], [2], [3, 4, 5]]                      # chain order = request order
    credit = CC.CreditState({7: R})
    got = {}
    for chain in reversed(chains):
        adv = credit.step(7, [values[i] for i in chain], lam)
        got.update({i + 1: a for i, a in zip(chain, adv)})
    want = gae_reference(values, R, lam)
    assert got == pytest.approx(want)
    if lam == 0.0:
        assert sum(got.values()) == pytest.approx(R - values[0])


def test_a_member_without_physics_gets_zero_and_passes_the_chain_through():
    adv, first, acc = CC.chain_advantages([0.2, None, 0.5], next_value=1.0, acc=0.0, lam=0.0)
    assert adv == pytest.approx([0.3, 0.0, 0.5]) and first == 0.2


def test_warm_start_equals_the_offline_physics_critic_and_updates_learn(tmp_path):
    write_critic(tmp_path / "critic", ["eef_z", "eef_tgt_dist"])
    frozen = PC.FrozenCritic(tmp_path / "critic")
    co = CC.CoCritic(tmp_path / "critic", hidden_dim=8).eval()
    x = torch.randn(5, 3)
    h = torch.randn(5, 8)
    want = torch.sigmoid(torch.stack([f(x) for _, f in frozen.members]).mean(0))
    assert torch.allclose(co.values(x, h), want, atol=1e-6)                     # zero-initialised hidden branch
    assert torch.allclose(CC.CoCritic(tmp_path / "critic", 8, use_hidden=False).eval().values(x), want, atol=1e-6)
    with pytest.raises(ValueError):
        co(x)
    # the hidden branch can learn what physics cannot: a label that depends on h only
    torch.manual_seed(0)
    xs, hs = torch.randn(256, 3), torch.randn(256, 8)
    ys = (hs[:, 0] > 0).float()
    opt = torch.optim.AdamW(co.parameters(), lr=3e-3)
    losses = CC.update(co, opt, xs, hs, ys, steps=60)
    assert losses[-1] < 0.5 * losses[0]
    # rank-weighted mean: a rank without states still steps (identical copies), reduce_sum sees every tensor
    seen = []
    CC.update(co, opt, xs[:0], hs[:0], ys[:0], steps=2, reduce_sum=lambda t: seen.append(t.shape))
    assert len(seen) > 2


def test_chain_members_and_physics_inputs_join_by_response_id(tmp_path):
    write_critic(tmp_path / "critic", ["eef_z", "eef_tgt_dist"])
    frozen = PC.FrozenCritic(tmp_path / "critic")
    write_episode(tmp_path / "ep", [0.3, 0.2, 0.1])
    phys = CC.physics_inputs(frozen, tmp_path / "ep", "libero_object:4")
    assert sorted(phys) == ["ep-r1", "ep-r2", "ep-r3"] and len(phys["ep-r1"]) == 3
    raw = {"training_requests": [{"request_index": 1, "prompt_token_ids": [0] * 10, "token_ids": [1, 2]},
                                 {"request_index": 2, "prompt_token_ids": [0] * 15, "token_ids": [3]}],
           "requests": [{"request_index": 1, "response_id": "ep-r1"}, {"request_index": 2, "response_id": "zzz"}]}
    members = CC.chain_members({"packed_requests": [1, 2], "n_tokens": 3}, raw, phys)
    assert members[0][:2] == (9, 2) and members[0][2] == phys["ep-r1"] and members[1] == (14, 1, None, None)
    with pytest.raises(ValueError):
        CC.chain_members({"packed_requests": [1, 2], "n_tokens": 4}, raw, phys)


def test_score_chain_reads_the_boundary_hidden_states_and_spreads_scaled_advantages(tmp_path):
    write_critic(tmp_path / "critic", ["eef_z"])
    co = CC.CoCritic(tmp_path / "critic", hidden_dim=4).eval()
    torch.nn.init.normal_(co.hid[-1].weight)                 # make the hidden branch matter
    hidden = torch.randn(20, 4, requires_grad=True)
    members = [(4, 2, [0.1, 1.0]), (9, 3, [0.3, 1.0])]
    buf = CC.new_buffers()
    credit = CC.CreditState({3: 1.0})
    tok = CC.score_chain(co, hidden, members, 3, 0, 1.0, credit, 0.0, 2.0, buf)
    x = torch.tensor([m[2] for m in members])
    v = co.values(x, hidden.detach()[[4, 9]]).tolist()
    want = [(v[1] - v[0]) / 2.0] * 2 + [(1.0 - v[1]) / 2.0] * 3
    assert tok.tolist() == pytest.approx(want, abs=1e-6)
    assert buf["y"] == [1.0, 1.0] and len(buf["rows"]) == 2 and not buf["h"][0].requires_grad   # detached: no value gradient into the LM
    assert credit.state[3][0] == pytest.approx(v[0])


def test_cotrain_configuration_is_validated(tmp_path):
    write_critic(tmp_path / "critic", ["eef_z"])
    base = dict(advantage_mode="physcritic", critic_dir=tmp_path / "critic", gae_lambda=0.0, ratio_mask_low=None, ratio_mask_high=None,
                score_centering=True, is_cap=None, advantage_shuffle=False, reward_mode="final_success_binary_v1",
                critic_cotrain=True, full_finetune=True, pack_assignment="trajectory", critic_lr=1e-4, critic_updates=2)
    assert T.advantage_arg_error(SimpleNamespace(**base)) is None
    for bad in (dict(full_finetune=False), dict(pack_assignment="pack"), dict(critic_updates=0), dict(critic_lr=0.0),
                dict(advantage_mode="grpo", critic_dir=None)):
        assert T.advantage_arg_error(SimpleNamespace(**dict(base, **bad))), bad
    cfg = dict(FR.DEFAULTS, score_centering=True, sampler_topk=20, advantage_mode="physcritic", critic_dir=str(tmp_path / "critic"),
               reward_mode="final_success_binary_v1", critic_cotrain=True, full_finetune=True, max_training_requests=40, max_requests=40)
    FR.validate_advantage_config(dict(cfg))
    for bad in (dict(max_training_requests=8), dict(full_finetune=False), dict(pack_assignment="pack"), dict(critic_hidden="sft"),
                dict(critic_updates=0)):
        with pytest.raises(FR.RunError):
            FR.validate_advantage_config(dict(cfg, **bad))
    assert "critic_cotrain.py" in FR.SOURCE_FILES
    assert math.isclose(CC.ema_std(None, 0.3), 0.3) and math.isclose(CC.ema_std(0.2, 0.3), 0.21)
    assert json.dumps(CC.new_buffers())


def test_same_t_chain_gae_matches_request_gae():
    """same-t critic: newest-first chains with next values V(s_{t+1}, t) give the same GAE as the per-trajectory
    request_advantages with the same next values (lambda 0 / 0.9 / 1)."""
    vals = [(1, 0.1), (2, 0.2), (3, 0.25), (4, 0.3)]
    nv = [0.15, 0.22, 0.27, None]
    for lam in (0.0, 0.9, 1.0):
        a = PC.request_advantages(vals, 1.0, lam, nv)
        cs = CC.CreditState({0: 1.0})
        late = cs.step(0, [0.25, 0.3], lam, [0.27, None])
        early = cs.step(0, [0.1, 0.2], lam, [0.15, 0.22])
        assert [round(x, 9) for x in early + late] == [round(a[k], 9) for k in (1, 2, 3, 4)]
    # without next values the same-t path reduces to the plain value differences
    assert PC.request_advantages(vals, 1.0, 0.0, None) == PC.request_advantages(vals, 1.0, 0.0, [None] * 4)


def test_physics_only_precompute_equals_in_forward_chain_gae(tmp_path):
    """The precomputed physics-only advantages (all request states, same-t off) equal the newest-first
    chain GAE of the in-forward path when every request is trained (lambda 0.9)."""
    write_critic(tmp_path / "critic", ["eef_z", "eef_tgt_dist"])
    frozen = PC.FrozenCritic(tmp_path / "critic")
    write_episode(tmp_path / "ep", [0.3, 0.2, 0.1])
    co = CC.CoCritic(tmp_path / "critic", 8, use_hidden=False)
    r = {"sample_path": "x", "verdict": "progress", "episode_dir": str(tmp_path / "ep"), "suite": "libero_object", "task_id": 4,
         "diagnostics": {"host_final_success": True}}
    keyed, buf = CC.physics_only_advantages(co, frozen, [r], lam=0.9, sigma=2.0, device="cpu")
    phys = CC.physics_inputs(frozen, tmp_path / "ep", "libero_object:4")
    rids = sorted(phys)
    vals = co.values(torch.tensor([phys[k] for k in rids], dtype=torch.float32)).tolist()
    cs = CC.CreditState({0: 1.0})
    late = cs.step(0, vals[2:], 0.9)
    early = cs.step(0, vals[:2], 0.9)
    ref = {k: a / 2.0 for k, a in zip(rids, early + late)}
    assert set(keyed[0]) == set(ref) and all(abs(keyed[0][k] - ref[k]) < 1e-6 for k in ref)
    assert len(buf["y"]) == 3 and buf["y"] == [1.0] * 3 and len(buf["rows"]) == 3

"""Simulator-critic advantages: request-level GAE from a frozen physics critic, global normalization, token spreading, the
per-token advantage / SAO ratio mask in sample_loss, and the advantage-mode configuration (trainer + launcher)."""
import json
import math
from types import SimpleNamespace

import pytest
import torch

import fresh_run as FR
import physics_critic as PC
import raw_request_training as RRT
import train as T
from test_score_centering import Core, FakePeft


# ---------------------------------------------------------------- GAE over requests
VALUES = [(1, 0.10), (2, 0.12), (3, 0.40), (4, 0.35)]


@pytest.mark.parametrize("R", [0.0, 1.0])
def test_lambda_zero_is_the_value_difference_and_telescopes_to_R_minus_V_s0(R):
    adv = PC.request_advantages(VALUES, R, 0.0)
    assert sorted(adv) == [1, 2, 3, 4]
    assert adv[1] == pytest.approx(0.02) and adv[2] == pytest.approx(0.28) and adv[3] == pytest.approx(-0.05)
    assert adv[4] == pytest.approx(R - 0.35)                       # last request: R - V(s_T)
    assert sum(adv.values()) == pytest.approx(R - VALUES[0][1])    # sum_t A_t = R - V(s_1)


def test_lambda_one_is_R_minus_V_and_intermediate_lambda_is_the_discounted_sum():
    R = 1.0
    one = PC.request_advantages(VALUES, R, 1.0)
    assert all(one[i] == pytest.approx(R - v) for i, v in VALUES)
    lam = 0.9
    deltas = [VALUES[t + 1][1] - VALUES[t][1] for t in range(3)] + [R - VALUES[3][1]]
    got = PC.request_advantages(VALUES, R, lam)
    for t, (i, _) in enumerate(VALUES):
        assert got[i] == pytest.approx(sum(lam ** k * deltas[t + k] for k in range(4 - t)))
    assert PC.request_advantages([], 1.0, 0.0) == {}
    with pytest.raises(ValueError):
        PC.request_advantages(VALUES, 1.0, 1.5)


def test_global_normalization_uses_one_mean_and_std_over_the_batch_not_per_group():
    a = {1: 0.0, 2: 0.5}
    b = {1: -0.1, 2: 0.1, 3: 2.0}
    out = PC.normalize_global([a, None, b])
    assert out[1] is None
    flat = list(out[0].values()) + list(out[2].values())
    mean = sum(flat) / len(flat)
    std = math.sqrt(sum((x - mean) ** 2 for x in flat) / len(flat))
    assert mean == pytest.approx(0.0, abs=1e-9) and std == pytest.approx(1.0, abs=1e-5)
    # one affine map for the whole batch: within-trajectory ratios of centered values are kept across trajectories
    raw = [0.0, 0.5, -0.1, 0.1, 2.0]
    m = sum(raw) / 5
    s = math.sqrt(sum((x - m) ** 2 for x in raw) / 5)
    assert out[2][3] == pytest.approx((2.0 - m) / (s + 1e-6))
    assert PC.normalize_global([{1: 0.3, 2: 0.3}, None]) == [{1: 0.0, 2: 0.0}, None]   # no spread: no signal
    assert PC.normalize_global([None]) == [None]


def test_auc_and_calibration():
    assert PC.auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)
    assert PC.auc([0.5, 0.5], [0, 1]) == pytest.approx(0.5)
    assert PC.auc([0.2, 0.3], [1, 1]) is None
    m = PC.calibration([[(1, 0.2), (2, 0.9)], [(1, 0.1)]], [1.0, 0.0])
    assert m["critic/auc_s0"] == 1.0 and m["critic/brier_s0"] == pytest.approx((0.8 ** 2 + 0.1 ** 2) / 2)
    assert m["critic/states"] == 3 and m["critic/brier_states"] == pytest.approx((0.64 + 0.01 + 0.01) / 3)


# ---------------------------------------------------------------- frozen critic on an episode directory
def write_episode(d, eef_zs, replayed=()):
    """A minimal one-goal episode: marks with the eef descending towards an object, one move_to call per request."""
    d.mkdir(parents=True)
    marks, calls, reqs = [], [], []
    for t, z in enumerate(eef_zs, start=1):
        goal = {"satisfied": False, "target_position_m": [0.5, 0.0, 0.02], "destination_position_m": [0.6, 0.1, 0.02],
                "distance_to_goal": 0.14, "progress_kind": "object"}
        marks.append({"request_index": t, "call_index": t - 1, "eef": [0.5, 0.0, z], "goal_states": {"g0": goal},
                      "objects": {"bowl": [0.5, 0.0, 0.02]}, "environment_step": 10 * t})
        calls.append({"call_index": t, "name": "move_to", "arguments": {"gripper": "open"}, "success": True, "consumes_action": True})
        reqs.append({"index": t, "replayed": t in replayed, **({} if t in replayed else {"response_id": f"{d.name}-r{t}"})})
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
    return net


def test_frozen_critic_scores_live_requests_in_order_with_the_spec_features(tmp_path):
    feats = ["eef_z", "eef_tgt_dist", "grip_open", "perc_called"]
    net = write_critic(tmp_path / "critic", feats)
    write_episode(tmp_path / "ep", [0.30, 0.20, 0.10], replayed=(1,))
    critic = PC.FrozenCritic(tmp_path / "critic")
    assert critic.uses_time == []
    got = critic.values(tmp_path / "ep", "libero_object:4")
    assert [i for i, _ in got] == [2, 3]                           # the replayed request 1 is not a student state
    net.eval()
    x = torch.tensor([[0.20, 0.18, 1.0, 0.0, 1.0], [0.10, 0.08, 1.0, 0.0, 1.0]])
    with torch.no_grad():
        want = torch.sigmoid(net.net(x).squeeze(-1)).tolist()
    assert [v for _, v in got] == pytest.approx(want, abs=1e-6)
    unseen = critic.values(tmp_path / "ep", "libero_10:9")         # unseen task: zero one-hot, still scored
    assert len(unseen) == 2
    assert critic.values(tmp_path / "nothing", "libero_object:4") == []
    timed = write_critic(tmp_path / "critic_t", feats + ["req_idx"])
    assert timed is not None and PC.FrozenCritic(tmp_path / "critic_t").uses_time == ["req_idx"]


def test_ensemble_critic_in_the_line_a_layout_averages_its_members_and_reports_their_spread(tmp_path):
    feats, tasks = ["eef_z", "eef_tgt_dist"], ["libero_object:4"]
    d = tmp_path / "default"
    d.mkdir()
    spec = {"feature_version": PC.cf.FEATURE_VERSION, "features": feats, "mean": [0.1, 0.0], "std": [2.0, 1.0],
            "task_vocab": tasks, "ood_std_tau_train_percentiles": {"99": 0.05}}
    (d / "feature_spec.json").write_text(json.dumps(spec))
    members = [PC.mlp(3, layout="body") for _ in range(3)]
    torch.save({"arch": {"in_dim": 3, "hidden": 256}, "state_dicts": [m.state_dict() for m, _ in members], "features": feats,
                "task_vocab": tasks, "mean": spec["mean"], "std": spec["std"]}, d / "critic_default.pt")
    write_episode(tmp_path / "ep", [0.3, 0.2])
    critic = PC.FrozenCritic(d)
    assert len(critic.members) == 3 and critic.std_tau == 0.05
    got, sd = critic.values(tmp_path / "ep", "libero_object:4", with_std=True)
    x = torch.tensor([[(0.3 - 0.1) / 2, 0.28, 1.0], [(0.2 - 0.1) / 2, 0.18, 1.0]])
    with torch.no_grad():
        p = torch.stack([torch.sigmoid(f(x)) for m, f in members if not m.eval().training])
    assert [v for _, v in got] == pytest.approx(p.mean(0).tolist(), abs=1e-6)
    assert sd == pytest.approx(p.std(0, unbiased=False).tolist(), abs=1e-6)
    results = [{"suite": "libero_object", "task_id": 4, "sample_path": "x", "verdict": "ok", "episode_dir": str(tmp_path / "ep"),
                "diagnostics": {"host_final_success": True}}]
    _, metrics = PC.batch_advantages(results, critic, lam=0.0)
    assert "critic/ens_std_mean" in metrics and "critic/frac_std_over_tau99" in metrics and "critic/cal_states" in metrics
    torch.save({"arch": {"in_dim": 3}, "state_dicts": [], "features": ["eef_x", "eef_z"], "task_vocab": tasks}, d / "critic_default.pt")
    with pytest.raises(ValueError, match="differ"):
        PC.FrozenCritic(d)


def test_validate_critic_dir_rejects_incomplete_or_mismatched_critics(tmp_path):
    write_critic(tmp_path / "ok", ["eef_z"])
    assert PC.validate_critic_dir(tmp_path / "ok")["features"] == ["eef_z"]
    with pytest.raises(ValueError):
        PC.validate_critic_dir(tmp_path / "missing")
    bad = tmp_path / "bad"
    write_critic(bad, ["eef_z"])
    spec = json.loads((bad / "feature_spec.json").read_text())
    (bad / "feature_spec.json").write_text(json.dumps(dict(spec, feature_version="other")))
    with pytest.raises(ValueError, match="feature version"):
        PC.validate_critic_dir(bad)
    (bad / "feature_spec.json").write_text(json.dumps(dict(spec, mean=[])))
    with pytest.raises(ValueError):
        PC.validate_critic_dir(bad)
    (tmp_path / "ok" / "critic_mlp.pt").unlink()
    with pytest.raises(ValueError, match="none of"):
        PC.validate_critic_dir(tmp_path / "ok")


def test_batch_advantages_skip_untrained_results_count_missing_marks_and_normalize(tmp_path):
    write_critic(tmp_path / "critic", ["eef_z", "eef_tgt_dist"])
    critic = PC.FrozenCritic(tmp_path / "critic")
    write_episode(tmp_path / "a", [0.3, 0.2, 0.1])
    write_episode(tmp_path / "b", [0.3, 0.3])
    (tmp_path / "c").mkdir()
    base = {"suite": "libero_object", "task_id": 4, "sample_path": "x", "verdict": "ok"}
    results = [dict(base, episode_dir=str(tmp_path / "a"), diagnostics={"host_final_success": True}),
               dict(base, episode_dir=str(tmp_path / "b"), diagnostics={"host_final_success": False}),
               dict(base, episode_dir=str(tmp_path / "c"), diagnostics={"host_final_success": False}),
               dict(base, episode_dir=str(tmp_path / "a"), verdict="infrastructure_error"),
               dict(base, episode_dir=str(tmp_path / "a"), sample_path=None)]
    raw, metrics = PC.batch_advantages(results, critic, lam=0.0, normalize=False)
    assert raw[2] is None and raw[3] is None and raw[4] is None and metrics["critic/missing_trajectories"] == 1
    va = critic.values(tmp_path / "a", "libero_object:4")
    assert sorted(raw[0]) == ["a-r1", "a-r2", "a-r3"]               # keyed by response id
    assert sum(raw[0].values()) == pytest.approx(1.0 - va[0][1])   # telescoping on a real critic
    assert metrics["critic/trajectories"] == 2 and metrics["critic/auc_s0"] is not None
    norm, _ = PC.batch_advantages(results, critic, lam=0.0)
    flat = [x for d in norm if d for x in d.values()]
    assert sum(flat) / len(flat) == pytest.approx(0.0, abs=1e-9) and len(flat) == 5


def test_proxy_values_join_driver_states_and_proxy_requests_by_response_id(tmp_path):
    # a segment rollout: driver requests 1-2 replayed, live 3-5; the proxy numbers the live requests 1-3
    write_episode(tmp_path / "seg", [0.3, 0.25, 0.2, 0.1, 0.05], replayed=(1, 2))
    assert PC.driver_response_ids(tmp_path / "seg") == {3: "seg-r3", 4: "seg-r4", 5: "seg-r5"}
    raw = {"requests": [{"request_index": 1, "response_id": "seg-r3"}, {"request_index": 2, "response_id": "seg-r4"},
                        {"request_index": 3, "response_id": "other"}],
           "training_requests": [{"request_index": 1}, {"request_index": 3}]}
    values, unmapped = PC.proxy_values({"seg-r3": 0.4, "seg-r4": -0.1, "seg-r5": 0.2}, raw)
    assert values == {1: 0.4, 3: 0.0} and unmapped == 1
    assert PC.proxy_values(None, raw) == (None, 0)


# ---------------------------------------------------------------- token spreading
RAW = {"training_requests": [{"request_index": 3, "token_ids": [1, 2]}, {"request_index": 7, "token_ids": [4, 5, 6]}]}


def test_request_values_spread_over_single_and_packed_samples():
    single = {"request_index": 7, "n_tokens": 3}
    assert RRT.request_token_values(single, RAW, {3: 0.5, 7: -1.0}).tolist() == [-1.0, -1.0, -1.0]
    packed = {"packed_requests": [3, 7], "n_tokens": 5}
    assert RRT.request_token_values(packed, RAW, {3: 0.5, 7: -1.0}).tolist() == [0.5, 0.5, -1.0, -1.0, -1.0]
    assert RRT.request_token_values(packed, RAW, None).tolist() == [0.0] * 5
    with pytest.raises(RRT.InvalidRollout):
        RRT.request_token_values(packed, RAW, {3: 0.5})
    with pytest.raises(RRT.InvalidRollout):
        RRT.request_token_values(dict(packed, n_tokens=4), RAW, {3: 0.5, 7: 1.0})


# ---------------------------------------------------------------- sample_loss: token advantages and ratio masking
def setup_loss(seed=3):
    torch.manual_seed(seed)
    core, head = Core(torch.randn(4, 5)), torch.nn.Linear(5, 9, bias=False)
    inputs, mask = {"input_ids": torch.tensor([[0, 2, 5, 7]])}, torch.tensor([[False, True, True, True]])
    with torch.no_grad():
        logp = torch.log_softmax(head(core.hidden), -1)
    lp_tok = torch.stack([logp[0, 2], logp[1, 5], logp[2, 7]])
    return core, head, inputs, mask, logp, lp_tok


def grad_of(loss, head):
    return torch.autograd.grad(loss, head.weight)[0]


def test_token_advantages_weight_each_token_and_reduce_to_the_scalar_advantage():
    core, head, inputs, mask, logp, lp_tok = setup_loss()
    kw = dict(kl_beta=0.0, full_completion_tokens=3, world_size=1, score_centering=False)
    base = {"weight": 1.0, "advantage": 1.0}
    adv = torch.tensor([0.5, -2.0, 1.0])
    loss, _, _ = T.sample_loss(FakePeft(), core, head, inputs, mask, dict(base, token_advantage=adv), **kw)
    want = -(adv * torch.log_softmax(head(core.hidden), -1)[[0, 1, 2], [2, 5, 7]]).sum() / 3
    assert torch.allclose(grad_of(loss, head), grad_of(want, head), atol=1e-6)
    same = T.sample_loss(FakePeft(), core, head, inputs, mask, dict(base, token_advantage=torch.full((3,), 0.7)), **kw)[0]
    scalar = T.sample_loss(FakePeft(), core, head, inputs, mask, dict(base, advantage=0.7), **kw)[0]
    assert torch.allclose(same, scalar, atol=1e-6)


def test_pure_error_credit_is_withheld_per_request_only_where_the_advantage_is_positive():
    core, head, inputs, mask, logp, lp_tok = setup_loss()
    kw = dict(kl_beta=0.0, full_completion_tokens=3, world_size=1, score_centering=False, code_weight=1e-9)
    s = {"weight": 1.0, "advantage": 1.0, "token_advantage": torch.tensor([1.0, -1.0, 1.0]),
         "token_pure_error": torch.tensor([1.0, 1.0, 0.0]), "token_code": torch.zeros(3)}
    loss = T.sample_loss(FakePeft(), core, head, inputs, mask, s, **kw)[0]
    # token 0 (A > 0, pure error) is withheld; token 1 (A < 0, pure error) and token 2 (clean) keep their advantage
    want = T.sample_loss(FakePeft(), core, head, inputs, mask, dict(s, token_advantage=torch.tensor([0.0, -1.0, 1.0]),
                                                                    token_pure_error=torch.zeros(3)), **kw)[0]
    assert torch.allclose(grad_of(loss, head), grad_of(want, head), atol=1e-6)


def test_double_sided_ratio_mask_weights_in_band_tokens_by_r_and_drops_the_rest_from_the_policy_gradient_only():
    core, head, inputs, mask, logp, lp_tok = setup_loss()
    kw = dict(kl_beta=0.0, full_completion_tokens=3, world_size=1, score_centering=False)
    on = {"weight": 1.0, "advantage": 1.0, "token_advantage": torch.tensor([1.0, 1.0, 1.0]), "sampled_logprobs": lp_tok.clone()}
    rows = []
    a = T.sample_loss(FakePeft(), core, head, inputs, mask, on, ratio_band=(0.2, 0.28), mask_stats=rows, **kw)[0]
    b = T.sample_loss(FakePeft(), core, head, inputs, mask, on, **kw)[0]
    assert torch.allclose(a, b) and rows[0].tolist() == [3.0, 0.0, 0.0, 3.0]          # on-policy: nothing is masked
    # token 0: ratio e^0.5 = 1.65 > 1.28 (high); token 1: e^-0.5 = 0.61 < 0.8 (low); token 2: e^0.1 = 1.105 kept
    off = dict(on, sampled_logprobs=lp_tok - torch.tensor([0.5, -0.5, 0.1]))
    rows = []
    loss = T.sample_loss(FakePeft(), core, head, inputs, mask, off, ratio_band=(0.2, 0.28), mask_stats=rows, **kw)[0]
    assert rows[0].tolist() == [1.0, 1.0, 1.0, 3.0]
    # SAO f(r) = r inside the band: token 2 keeps weight e^0.1
    only2 = T.sample_loss(FakePeft(), core, head, inputs, mask, dict(on, token_advantage=torch.tensor([0.0, 0.0, math.exp(0.1)])), **kw)[0]
    assert torch.allclose(grad_of(loss, head), grad_of(only2, head), atol=1e-6)
    # the KL anchor keeps every token
    kw_kl = dict(kw, kl_beta=0.1)
    ref = lambda inputs, mask: lp_tok.detach() - 0.3        # noqa: E731
    _, kl_masked, _ = T.sample_loss(FakePeft(), core, head, inputs, mask, off, ratio_band=(0.2, 0.28), reference=ref, **kw_kl)
    _, kl_plain, _ = T.sample_loss(FakePeft(), core, head, inputs, mask, off, reference=ref, **kw_kl)
    assert torch.allclose(kl_masked, kl_plain)
    with pytest.raises(RRT.InvalidRollout):
        T.sample_loss(FakePeft(), core, head, inputs, mask, {k: v for k, v in on.items() if k != "sampled_logprobs"},
                      ratio_band=(0.2, 0.2), **kw)


def expected_gradient_band(p, q, band, head=2):
    """E_{y ~ q}[grad loss(y)] at one prefix with score centering and SAO masking (test_sc_is_credit.expected_gradient)."""
    from test_sc_is_credit import Core as Core1
    logits = torch.tensor(p, dtype=torch.float32).log().requires_grad_()
    q = torch.tensor(q, dtype=torch.float32)
    heads = q.argsort(descending=True)[:head]
    total = 0.0
    for y in range(len(q)):
        s = {"advantage": 1.0, "weight": 1.0, "topk_ids": heads[None], "topk_logprobs": q[heads].log()[None],
             "sampled_logprobs": q[y].log()[None]}
        inputs = {"input_ids": torch.tensor([[0, y]]), "attention_mask": torch.ones(1, 2, dtype=torch.long)}
        loss, _, _ = T.sample_loss(None, Core1(logits), torch.nn.Identity(), inputs, torch.tensor([[False, True]]), s,
                                   score_centering=True, kl_beta=0.0, full_completion_tokens=1, world_size=1, ratio_band=band)
        total = total + q[y] * loss
    return torch.autograd.grad(total, logits)[0]


@pytest.mark.parametrize("p", [[.405, .298, .1782, .1188], [.55, .25, .12, .08], [.85, .075, .045, .03], [.4, .3, .18, .12]])
@pytest.mark.parametrize("band", [(0.2, 0.28), (0.05, 0.05), (0.3, 5.0)])
def test_band_weighted_centering_has_zero_sampler_mean(p, band):
    # proportional tail as in the authors' derivation; heads inside or outside the band, tail inside or outside
    assert expected_gradient_band(p, [.4, .3, .18, .12], band).norm() < 1e-5


# ---------------------------------------------------------------- configuration
def args_of(**kw):
    base = dict(advantage_mode="grpo", critic_dir=None, gae_lambda=0.0, ratio_mask_low=None, ratio_mask_high=None,
                score_centering=True, is_cap=None, advantage_shuffle=False, reward_mode="final_success_binary_v1")
    return SimpleNamespace(**dict(base, **kw))


def test_trainer_advantage_arguments_are_validated(tmp_path):
    write_critic(tmp_path / "critic", ["eef_z"])
    good = dict(advantage_mode="physcritic", critic_dir=tmp_path / "critic", gae_lambda=0.9, ratio_mask_low=0.2, ratio_mask_high=0.28)
    assert T.advantage_arg_error(args_of()) is None and T.advantage_arg_error(args_of(**good)) is None
    for bad in (dict(critic_dir=tmp_path / "critic"), dict(gae_lambda=0.5), dict(ratio_mask_low=0.2),
                dict(ratio_mask_low=0.2, ratio_mask_high=0.2, score_centering=False),
                dict(ratio_mask_low=0.2, ratio_mask_high=0.2, is_cap=2.0), dict(ratio_mask_low=1.2, ratio_mask_high=0.2),
                dict(good, critic_dir=None), dict(good, critic_dir=tmp_path / "none"), dict(good, gae_lambda=1.1),
                dict(good, advantage_shuffle=True), dict(good, reward_mode="staged_progress_v2")):
        assert T.advantage_arg_error(args_of(**bad)), bad


def test_launcher_validates_and_passes_the_advantage_mode(tmp_path):
    write_critic(tmp_path / "critic", ["eef_z"])
    cfg = dict(FR.DEFAULTS, score_centering=True, sampler_topk=20, advantage_mode="physcritic", critic_dir=str(tmp_path / "critic"),
               gae_lambda=0.9, ratio_mask_low=0.2, ratio_mask_high=0.28, reward_mode="final_success_binary_v1")
    FR.validate_advantage_config(cfg)
    assert cfg["critic_dir"] == str((tmp_path / "critic").resolve())
    FR.validate_advantage_config(dict(FR.DEFAULTS))
    for bad in (dict(cfg, advantage_mode="ppo"), dict(cfg, critic_dir=None), dict(cfg, gae_lambda=2.0), dict(cfg, reward_mode=None),
                dict(cfg, ratio_mask_high=None), dict(cfg, is_cap=2.0), dict(cfg, score_centering=False),
                dict(FR.DEFAULTS, critic_dir=str(tmp_path / "critic")), dict(cfg, advantage_shuffle=True),
                dict(cfg, critic_dir=str(tmp_path / "nothing"))):
        with pytest.raises(FR.RunError):
            FR.validate_advantage_config(dict(bad))
    assert "physics_critic.py" in FR.SOURCE_FILES and "critic_features.py" in FR.SOURCE_FILES


# ---------------------------------------------------------------- prioritized start sampling
def test_start_weights_bias_the_fresh_draws_and_must_match_on_resume(tmp_path):
    from collections import Counter

    import make_start_weights as MSW
    from segment_curriculum import SegmentCurriculum, load_start_weights
    from test_staged_reward import bank

    rows = ([{"suite": "libero_object", "task_id": 0, "seed": 1, "success": False, "verdict": "x"}] * 30
            + [{"suite": "libero_object", "task_id": 1, "seed": 1, "success": s, "verdict": "x"} for s in [True, False] * 15]
            + [{"suite": "libero_object", "task_id": 2, "seed": 1, "success": True, "verdict": "infrastructure_error"}])
    spec = MSW.start_weights(rows, floor=0.1)
    assert spec["weights"]["libero_object:0:1"] == 0.1 and spec["weights"]["libero_object:1:1"] > 0.99
    assert "libero_object:2:1" not in spec["weights"] and spec["default"] == 1.0          # infra errors are not outcomes
    path = tmp_path / "w.json"
    path.write_text(json.dumps(spec))
    weights = load_start_weights(path)
    c = SegmentCurriculum(bank(), episode_fraction=1.0, budget=40, start_weights=weights)
    drawn = Counter(e["task_id"] for e in c.next_batch(3000))
    assert drawn[0] < 0.2 * drawn[1] and drawn[2] > 0.8 * drawn[1]                        # 0.1 vs ~1 vs default 1
    uniform = SegmentCurriculum(bank(), episode_fraction=1.0, budget=40)
    with pytest.raises(ValueError, match="start weights"):
        uniform.load_state_dict(c.state_dict())
    again = SegmentCurriculum(bank(), episode_fraction=1.0, budget=40, start_weights=load_start_weights(path))
    again.load_state_dict(c.state_dict())
    assert [e["task_id"] for e in again.next_batch(20)] == [e["task_id"] for e in c.next_batch(20)]
    path.write_text(json.dumps(dict(spec, default=0.0)))
    with pytest.raises(ValueError):
        load_start_weights(path)
    assert load_start_weights(None) is None


# ---------------------------------------------------------------- false-finish return
def _res(success, verdict):
    return {"verdict": verdict, "diagnostics": {"host_final_success": success}}


def test_advantage_return_mapping_v11():
    ff = -0.2
    assert PC.advantage_return(_res(True, "segment_success"), ff) == 1.0
    # success wins over any verdict label
    assert PC.advantage_return(_res(True, "false_finish"), ff) == 1.0
    assert PC.advantage_return(_res(False, "false_finish"), ff) == pytest.approx(-0.2)
    for verdict in ("no_progress", "progress", "regress", "stopped", "format_error", "segment_finish"):
        assert PC.advantage_return(_res(False, verdict), ff) == 0.0
    assert PC.advantage_return({"verdict": "false_finish"}, ff) == pytest.approx(-0.2)   # no diagnostics = not a success
    # default: plain 0/1 success
    assert PC.advantage_return(_res(False, "false_finish")) == 0.0
    assert PC.advantage_return(_res(True, "segment_success")) == 1.0


def test_false_finish_return_only_moves_last_step_target():
    values = [(0, 0.3), (1, 0.2), (2, 0.1)]
    a0 = PC.request_advantages(values, PC.advantage_return(_res(False, "false_finish")), 0.9)
    a1 = PC.request_advantages(values, PC.advantage_return(_res(False, "false_finish"), -0.2), 0.9)
    assert a1[2] == pytest.approx(-0.2 - 0.1) and a0[2] == pytest.approx(-0.1)
    # the shift propagates through GAE as lambda^k * (-0.2)
    for t, k in ((2, 0), (1, 1), (0, 2)):
        assert a1[t] - a0[t] == pytest.approx(-0.2 * 0.9 ** k)


def test_false_finish_reward_flag_validation(tmp_path):
    write_critic(tmp_path / "critic", ["eef_z"])
    good = dict(advantage_mode="physcritic", critic_dir=tmp_path / "critic", gae_lambda=0.9, ratio_mask_low=0.2, ratio_mask_high=0.28)
    assert T.advantage_arg_error(args_of(**good, false_finish_reward=-0.2)) is None
    assert T.advantage_arg_error(args_of(**good, false_finish_reward=0.3))
    assert T.advantage_arg_error(args_of(false_finish_reward=-0.2))          # needs physcritic
    assert FR.DEFAULTS["false_finish_reward"] == 0.0
    cfg = dict(FR.DEFAULTS, score_centering=True, sampler_topk=20, advantage_mode="physcritic", critic_dir=str(tmp_path / "critic"),
               gae_lambda=0.9, ratio_mask_low=0.2, ratio_mask_high=0.28, reward_mode="final_success_binary_v1")
    FR.validate_advantage_config(dict(cfg, false_finish_reward=-0.2))
    for bad in (1.0, -2.0, "x"):
        with pytest.raises(FR.RunError):
            FR.validate_advantage_config(dict(cfg, false_finish_reward=bad))

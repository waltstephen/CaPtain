"""Score centering: sampler top-k recording (proxy), alignment (sample builder) and the loss correction (trainer)."""
from types import SimpleNamespace

import pytest
import torch

import raw_request_training as RRT
import train as T
import policy_proxy as P


def choice(token_ids, k=2):
    return {"logprobs": {"content": [{"token": f"token_id:{t}", "logprob": -0.1 * (i + 1),
                                      "top_logprobs": [{"token": f"token_id:{t + j}", "logprob": -0.5 * (j + 1)} for j in range(k)]}
                                     for i, t in enumerate(token_ids)]}}


def test_proxy_parses_sampler_topk_aligned_with_the_sampled_tokens():
    got = P.sampler_topk(choice([5, 9]), [5, 9], 2)
    assert got == {"k": 2, "ids": [[5, 6], [9, 10]], "logprobs": [[-0.5, -1.0], [-0.5, -1.0]], "sampled_logprob": [-0.1, -0.2]}
    for bad, ids, k in ((choice([5, 9]), [5, 8], 2), (choice([5]), [5, 9], 2), (choice([5, 9], k=1), [5, 9], 2)):
        with pytest.raises(ValueError):
            P.sampler_topk(bad, ids, k)
    with pytest.raises(ValueError):
        P.sampler_topk({"logprobs": {"content": [{"token": "hello", "logprob": 0, "top_logprobs": []}]}}, [1], 0)


def test_sample_builder_attaches_sampler_tensors_only_when_aligned():
    topk = {"k": 2, "ids": [[1, 2], [3, 4]], "logprobs": [[-1.0, -2.0], [-1.0, -2.0]], "sampled_logprob": [-1.0, -3.0]}
    got = RRT.sampler_tensors(topk, 2)
    assert got["topk_ids"].shape == (2, 2) and got["sampled_logprobs"].tolist() == [-1.0, -3.0]
    assert RRT.sampler_tensors(topk, 3) == {} and RRT.sampler_tensors(None, 2) == {} and RRT.sampler_tensors({**topk, "k": 3}, 2) == {}


class Core(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.hidden = torch.nn.Parameter(hidden)

    def forward(self, input_ids, **_):
        return SimpleNamespace(last_hidden_state=self.hidden.unsqueeze(0))


def test_correction_cancels_the_expected_score_under_the_sampler_and_vanishes_on_policy():
    torch.manual_seed(0)
    vocab, width, k = 6, 4, 3
    core, head = Core(torch.randn(2, width)), torch.nn.Linear(width, vocab, bias=False)
    ids = torch.tensor([[0, 2]])
    mask = torch.tensor([[False, True]])
    with torch.no_grad():
        p = torch.softmax(head(core.hidden[0]), -1)
    # a sampler that differs on the head and whose tail is proportional to the trainer's (the modelled tail)
    top = torch.tensor([[1, 3, 4]])
    q_head = torch.tensor([0.5, 0.2, 0.1])
    tail = [v for v in range(vocab) if v not in top[0].tolist()]
    rho = (1 - q_head.sum()) / p[tail].sum()
    q = torch.zeros(vocab)
    q[top[0]] = q_head
    q[tail] = rho * p[tail]
    lp, corr = T.masked_logprobs_centered(core, head, {"input_ids": ids}, mask, top, torch.log(q_head).unsqueeze(0))
    logp_all = torch.log_softmax(head(core.hidden[0]), -1)
    expected_score = (q.detach() * logp_all).sum()               # E_q[log p_y] as a surrogate: its gradient is E_q[score]
    params = [core.hidden, head.weight]
    g_expected = torch.autograd.grad(expected_score, params, retain_graph=True)
    g_corr = torch.autograd.grad(corr.sum(), params)
    for a, b in zip(g_expected, g_corr):
        assert torch.allclose(a, b, atol=1e-5)                    # centered expected score = 0
    on_policy = torch.log(p[top[0]]).unsqueeze(0)
    _, corr_on = T.masked_logprobs_centered(core, head, {"input_ids": ids}, mask, top, on_policy)
    assert abs(float(corr_on.sum())) < 1e-6
    assert torch.allclose(lp, logp_all[2].unsqueeze(0))
    with pytest.raises(RRT.InvalidRollout):
        T.masked_logprobs_centered(core, head, {"input_ids": ids}, mask, torch.tensor([[1, 3, 4], [0, 1, 2]]), torch.zeros(2, 3))


class FakePeft:
    """disable_adapter() as a context manager, like peft.PeftModel."""
    def __init__(self):
        self.disabled = 0

    def disable_adapter(self):
        import contextlib

        @contextlib.contextmanager
        def ctx():
            self.disabled += 1
            yield
        return ctx()


@pytest.mark.parametrize("kl_beta", [0.0, 0.05])
@pytest.mark.parametrize("centering", [False, True])
def test_sample_loss_runs_every_branch_and_centering_is_a_no_op_on_policy(kl_beta, centering):
    torch.manual_seed(1)
    core, head = Core(torch.randn(3, 4)), torch.nn.Linear(4, 7, bias=False)
    inputs, mask = {"input_ids": torch.tensor([[0, 2, 5]])}, torch.tensor([[False, True, True]])
    with torch.no_grad():
        logp = torch.log_softmax(head(core.hidden), -1)
    top = torch.stack([torch.topk(logp[i], 3).indices for i in (0, 1)])
    s = {"advantage": 0.7, "weight": 2.0, "topk_ids": top, "topk_logprobs": torch.stack([logp[i][top[i]] for i in (0, 1)]),
         "sampled_logprobs": torch.stack([logp[0, 2], logp[1, 5]])}
    model = FakePeft()
    kw = dict(kl_beta=kl_beta, full_completion_tokens=2, world_size=1)
    loss, kl, row = T.sample_loss(model, core, head, inputs, mask, s, score_centering=centering, **kw)
    loss.backward()
    assert (row is not None) == centering and model.disabled == (1 if kl_beta else 0)
    plain, _, _ = T.sample_loss(model, core, head, inputs, mask, s, score_centering=False, **kw)
    assert torch.allclose(loss, plain, atol=1e-6)          # sampler == trainer: the correction is zero
    if centering:
        assert float(row[0]) < 1e-5 and int(row[2]) == 2
        missing = {k: v for k, v in s.items() if not k.startswith(("topk", "sampled"))}
        _, _, none_row = T.sample_loss(model, core, head, inputs, mask, missing, score_centering=True, **kw)
        assert none_row is None


def test_save_full_writes_sharded_safetensors_with_index_and_base_files(tmp_path, monkeypatch):
    import json as _json
    load_file = pytest.importorskip("safetensors.torch").load_file

    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    (base / "tokenizer.json").write_text("{}")
    (base / "model-00001-of-00001.safetensors").write_bytes(b"old")
    sd = {"model.a.weight": torch.ones(3, 2, dtype=torch.bfloat16), "lm_head.weight": torch.zeros(4, dtype=torch.bfloat16)}
    monkeypatch.setattr(T, "full_state_dict_bf16", lambda model: sd)
    out = tmp_path / "live"
    out.mkdir()
    (out / "stale.bin").write_text("x")
    T.save_full(None, str(base), out, True)
    names = sorted(p.name for p in out.iterdir())
    assert names == ["config.json", "model-00001-of-00001.safetensors", "model.safetensors.index.json", "tokenizer.json"]
    index = _json.loads((out / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(sd) and index["metadata"]["total_size"] == 3 * 2 * 2 + 4 * 2
    loaded = load_file(str(out / "model-00001-of-00001.safetensors"))
    assert torch.equal(loaded["model.a.weight"], sd["model.a.weight"])


def test_full_finetune_checkpoints_are_complete_with_a_linked_sharded_state_and_prune_keeps_only_the_newest(tmp_path):
    import json as _json

    import training_resume as TR

    run, local = tmp_path / "run", tmp_path / "local"
    for step in (0, 10):
        ck = run / f"checkpoint-{step}"
        ck.mkdir(parents=True)
        (ck / "training_state.pt").write_bytes(b"x")
        (ck / "trainer_state.json").write_text(_json.dumps({"global_step": step, "full_finetune": True}))
        (ck / "model.safetensors.index.json").write_text("{}")
        state = local / f"checkpoint-{step}"
        state.mkdir(parents=True)
        (state / ".metadata").write_bytes(b"m")
        (ck / "fsdp").symlink_to(state)
    assert TR.complete_checkpoint_step(run / "checkpoint-10") == 10 and TR.latest_complete_checkpoint(run).name == "checkpoint-10"
    T.prune_full_states(run, run / "checkpoint-10", lambda m: None)
    assert not (run / "checkpoint-0" / "fsdp").exists() and not (local / "checkpoint-0").exists()
    assert (local / "checkpoint-10" / ".metadata").is_file() and TR.complete_checkpoint_step(run / "checkpoint-0") is None
    assert (run / "checkpoint-0" / "model.safetensors.index.json").is_file()     # weights and metadata stay


def test_truncated_importance_weight_is_one_on_policy_and_caps_stale_tokens():
    torch.manual_seed(2)
    core, head = Core(torch.randn(3, 4)), torch.nn.Linear(4, 7, bias=False)
    inputs, mask = {"input_ids": torch.tensor([[0, 2, 5]])}, torch.tensor([[False, True, True]])
    with torch.no_grad():
        logp = torch.log_softmax(head(core.hidden), -1)
    top = torch.stack([torch.topk(logp[i], 3).indices for i in (0, 1)])
    s = {"advantage": 0.7, "weight": 1.0, "topk_ids": top, "topk_logprobs": torch.stack([logp[i][top[i]] for i in (0, 1)]),
         "sampled_logprobs": torch.stack([logp[0, 2], logp[1, 5]])}
    kw = dict(kl_beta=0.0, full_completion_tokens=2, world_size=1, score_centering=True)
    plain, _, _ = T.sample_loss(FakePeft(), core, head, inputs, mask, s, **kw)
    rows = []
    same, _, _ = T.sample_loss(FakePeft(), core, head, inputs, mask, s, is_cap=2.0, is_stats=rows, **kw)
    assert torch.allclose(plain, same, atol=1e-6)                  # behavior == learner: rho = 1, the surrogate is unchanged
    assert torch.allclose(rows[0], torch.tensor([2.0, 0.0, 2.0, 2.0], dtype=torch.float64), atol=1e-5)
    stale = dict(s, sampled_logprobs=s["sampled_logprobs"] + torch.tensor([0.0, -5.0]))   # token 2: pi/pi_b = e^5 -> capped at 2
    rows = []
    T.sample_loss(FakePeft(), core, head, inputs, mask, stale, is_cap=2.0, is_stats=rows, **kw)
    assert abs(float(rows[0][0]) - 3.0) < 1e-5 and int(rows[0][1]) == 1
    g_plain = torch.autograd.grad(T.sample_loss(FakePeft(), core, head, inputs, mask, s, **kw)[0], head.weight)[0]
    g_is = torch.autograd.grad(T.sample_loss(FakePeft(), core, head, inputs, mask, s, is_cap=2.0, **kw)[0], head.weight)[0]
    assert torch.allclose(g_plain, g_is, atol=1e-6)                # the weight is detached: the on-policy gradient is unchanged
    missing = {k: v for k, v in s.items() if k != "sampled_logprobs"}
    with pytest.raises(RRT.InvalidRollout):
        T.sample_loss(FakePeft(), core, head, inputs, mask, missing, is_cap=2.0, **dict(kw, score_centering=False))


def test_balanced_assignment_equalizes_trained_requests_deterministically():
    active = list(range(10))
    costs = [40, 8, 8, 35, 8, 8, 30, 8, 8, 8]
    out = T.balanced_assignment(active, costs, 4)
    assert sorted(i for part in out for i in part) == active      # every trajectory on exactly one rank
    loads = [sum(costs[i] for i in part) for part in out]
    assert sorted(loads) == [38, 40, 40, 43]                       # round-robin active[r::4] gives 56 / 24 / 38 / 43
    assert T.balanced_assignment(active, costs, 4) == out
    assert T.balanced_assignment([], [], 3) == [[], [], []]

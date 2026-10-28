"""Off-policy score centering with truncated IS (authors' weighted composition) and solved-suffix task credit."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

import raw_request_training as RRT
import reward_segment as RS
import train as T
import env_server as env


class Core:
    def __init__(self, logits):
        self.logits = logits

    def __call__(self, **inputs):
        length = inputs["input_ids"].shape[1]
        return SimpleNamespace(last_hidden_state=self.logits.view(1, 1, -1).expand(1, length, -1))


def expected_gradient(p, q, cap, head=2):
    """E_{y ~ q}[grad loss(y)] at one prefix with a fixed advantage: zero iff the centered score has zero sampler mean."""
    logits = torch.tensor(p, dtype=torch.float32).log().requires_grad_()
    q = torch.tensor(q, dtype=torch.float32)
    heads = q.argsort(descending=True)[:head]
    total = 0.0
    for y in range(len(q)):
        s = {"advantage": 1.0, "weight": 1.0, "topk_ids": heads[None], "topk_logprobs": q[heads].log()[None],
             "sampled_logprobs": q[y].log()[None]}
        inputs = {"input_ids": torch.tensor([[0, y]]), "attention_mask": torch.ones(1, 2, dtype=torch.long)}
        loss, _, _ = T.sample_loss(None, Core(logits), torch.nn.Identity(), inputs, torch.tensor([[False, True]]), s,
                                   score_centering=True, kl_beta=0.0, full_completion_tokens=1, world_size=1, is_cap=cap)
        total = total + q[y] * loss
    return torch.autograd.grad(total, logits)[0]


@pytest.mark.parametrize("p", [[.405, .298, .1782, .1188], [.55, .25, .12, .08], [.85, .075, .045, .03]])
@pytest.mark.parametrize("cap", [None, 2.0, 1.2])
def test_centered_score_has_zero_sampler_mean_with_and_without_truncated_is(p, cap):
    # exact proportional tail (p and q agree on the tail's shape), as in the authors' derivation
    assert expected_gradient(p, [.4, .3, .18, .12], cap).norm() < 1e-5


def test_truncated_is_reduces_to_plain_centering_on_policy():
    q = [.4, .3, .18, .12]
    for cap in (None, 2.0):
        assert expected_gradient(q, q, cap).norm() < 1e-6


def test_zero_task_credit_leaves_only_the_kl_term():
    torch.manual_seed(0)
    logits = torch.randn(5, requires_grad=True)
    ref = lambda inputs, mask: torch.log_softmax(torch.zeros(5), -1)[inputs["input_ids"][0, 1:]][mask[0, 1:]]
    inputs, mask = {"input_ids": torch.tensor([[0, 2, 3]])}, torch.tensor([[False, True, True]])
    base = {"advantage": 0.9, "weight": 1.5}
    kw = dict(score_centering=False, kl_beta=0.05, full_completion_tokens=2, world_size=1, reference=ref)
    full, kl, _ = T.sample_loss(None, Core(logits), torch.nn.Identity(), inputs, mask, base, **kw)
    zero, kl0, _ = T.sample_loss(None, Core(logits), torch.nn.Identity(), inputs, mask, dict(base, token_credit=torch.zeros(2)), **kw)
    one, _, _ = T.sample_loss(None, Core(logits), torch.nn.Identity(), inputs, mask, dict(base, token_credit=torch.ones(2)), **kw)
    assert torch.allclose(full, one) and torch.allclose(kl, kl0)
    assert torch.allclose(zero, 0.05 * kl0 / 2)            # returned kl already carries the weight: beta * weight * KL / tokens
    half, _, _ = T.sample_loss(None, Core(logits), torch.nn.Identity(), inputs, mask, dict(base, token_credit=torch.tensor([1.0, 0.0])), **kw)
    assert not torch.allclose(half, full) and not torch.allclose(half, zero)


def write_episode(tmp_path, solved_by_mark, live_indices, response_ids):
    (tmp_path / "driver").mkdir(parents=True)
    rows = [{"index": i, "replayed": i not in live_indices, "response_id": response_ids.get(i)} for i in range(1, max(live_indices) + 1)]
    (tmp_path / "driver" / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    marks = [{"schema_version": RS.MARK_SCHEMA, "state_schema": RS.GOAL_AUDIT_SCHEMA, "request_index": i,
              "goal_states": {"g": {"satisfied": solved}}} for i, solved in solved_by_mark.items()]
    (tmp_path / "hidden_request_marks.jsonl").write_text("".join(json.dumps(m) + "\n" for m in marks))


def test_solved_suffix_is_the_final_stably_solved_run_of_a_successful_trajectory(tmp_path):
    # K = 2 replayed; live 3..9; solved at mark 4 (transient), broken at 5, solved again from mark 7 to the end;
    # request 9 is a text-only finish (no mark)
    marks = {3: False, 4: True, 5: False, 6: False, 7: True, 8: True}
    ids = {i: f"r{i}" for i in range(3, 10)}
    write_episode(tmp_path, marks, list(range(3, 10)), ids)
    proxy = [{"request_index": j + 1, "response_id": f"r{i}"} for j, i in enumerate(range(3, 10))] + [{"request_index": 8}]  # a retry
    got = RS.solved_suffix_requests(tmp_path, proxy, final_success=True)
    assert got == {"proxy_request_indices": {5, 6, 7}, "unmapped": 0}           # driver requests 7, 8, 9
    assert RS.solved_suffix_requests(tmp_path, proxy, final_success=False)["proxy_request_indices"] == set()
    got = RS.solved_suffix_requests(tmp_path, proxy[:5], final_success=True)     # r8, r9 missing in the proxy record
    assert got == {"proxy_request_indices": {5}, "unmapped": 2}


def test_task_credit_reaches_the_samples_and_packed_chains():
    base = {1: ([1, 2, 3], [4, 5]), 2: ([1, 2, 3, 4, 5, 6], [7])}
    from test_prefix_packing import raw
    r = raw(select=(1, 2), prompts=base)
    r["training_requests"][1]["task_credit"] = 0
    plain, _ = RRT.build_request_samples(r, None, image_pad_id=-1, hard_max_tokens=100)
    assert "token_credit" not in plain[0] and plain[1]["token_credit"].tolist() == [0.0]
    r["training_requests"][0]["task_credit"] = 1
    packed, _ = RRT.build_request_samples(r, None, image_pad_id=-1, hard_max_tokens=100, pack_prefixes=True)
    assert len(packed) == 1 and packed[0]["token_credit"].tolist() == [1.0, 1.0, 0.0]
    r["training_requests"][0]["task_credit"] = 0.5
    with pytest.raises(RRT.InvalidRollout):
        RRT.build_request_samples(r, None, image_pad_id=-1, hard_max_tokens=100)


def test_build_training_sample_marks_the_solved_suffix():
    import test_raw_request_samples as TR
    reqs = [TR.request(1, [1, 2], [3]), TR.request(2, [1, 2, 3, 4], [5])]
    sample = env.build_training_sample(TR.record(reqs), sid="fixed-session", max_training_requests=-1, solved_suffix={2})
    assert [q["task_credit"] for q in sample["training_requests"]] == [1, 0]
    sample = env.build_training_sample(TR.record(reqs), sid="fixed-session", max_training_requests=-1)
    assert all("task_credit" not in q for q in sample["training_requests"])

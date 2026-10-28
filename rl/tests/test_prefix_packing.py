"""Exact-prefix packing of a trajectory's selected requests: chain rules, sample layout and loss/gradient parity."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import raw_request_training as RRT
import train as T
import policy_proxy as P

V = 16


def topk(n, seed):
    g = torch.Generator().manual_seed(seed)
    ids = [[int(x) for x in torch.randperm(V, generator=g)[:3]] for _ in range(n)]
    return {"k": 3, "ids": ids, "logprobs": [[-0.5, -1.5, -2.5]] * n, "sampled_logprob": [float(-0.3 - 0.1 * i) for i in range(n)]}


def raw(select=(1, 2, 4), prompts=None):
    prompts = prompts or {1: ([1, 2, 3], [4, 5]), 2: ([1, 2, 3, 4, 5, 6, 7], [8]), 3: ([1, 9, 9], [10, 11]),
                          4: ([1, 9, 9, 10, 11, 12], [13, 14])}
    eligible = [{"request_index": i, "prompt_length": len(p), "completion_length": len(c)} for i, (p, c) in prompts.items()]
    prob = len(select) / len(eligible)
    chosen = [{"request_index": i, "prompt_token_ids": prompts[i][0], "token_ids": prompts[i][1], "image_hashes": [], "images": [],
               "prompt_length": len(prompts[i][0]), "completion_length": len(prompts[i][1]),
               "sampling": dict(P.DEFAULT_SAMPLING), "inclusion_probability": prob, "loss_weight": 1 / prob,
               "sampler_topk": topk(len(prompts[i][1]), i)} for i in select]
    return {"sample_format": "per_request_v1", "eligible_requests": eligible, "training_requests": chosen,
            "full_completion_tokens": sum(e["completion_length"] for e in eligible), "eligible_request_count": len(eligible),
            "selected_request_count": len(chosen)}


def test_chains_join_only_exact_token_and_image_prefixes():
    selected, _ = RRT.validate_rollout(raw(select=(1, 2, 3, 4)))
    assert [[r["request_index"] for r in c] for c in RRT.prefix_chains(selected)] == [[1, 2], [3, 4]]   # 3: compacted history
    selected, _ = RRT.validate_rollout(raw(select=(1, 2, 4)))
    assert [[r["request_index"] for r in c] for c in RRT.prefix_chains(selected)] == [[1, 2], [4]]   # unselected 3 stays context
    changed = raw(select=(1, 2))
    changed["training_requests"][1]["image_hashes"] = ["x"]      # an image appears after the prefix: still a prefix
    assert len(RRT.prefix_chains(changed["training_requests"])) == 1
    changed["training_requests"][0]["image_hashes"] = ["y"]      # a different earlier image: not a prefix
    assert len(RRT.prefix_chains(changed["training_requests"])) == 2
    changed = raw(select=(1, 2))
    changed["training_requests"][1]["sampler_topk"] = None
    assert len(RRT.prefix_chains(changed["training_requests"])) == 2


def test_packed_sample_keeps_every_target_weight_and_sampler_record():
    plain, total = RRT.build_request_samples(raw(), None, image_pad_id=-1, hard_max_tokens=100)
    packed, total_p = RRT.build_request_samples(raw(), None, image_pad_id=-1, hard_max_tokens=100, pack_prefixes=True)
    assert total == total_p and len(plain) == 3 and len(packed) == 2
    first = packed[0]
    assert first["packed_requests"] == [1, 2] and first["input_ids"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert first["mask"].tolist() == [False, False, False, True, True, False, False, True]
    assert first["n_tokens"] == plain[0]["n_tokens"] + plain[1]["n_tokens"]
    assert torch.equal(first["token_weights"], torch.full((3,), plain[0]["weight"]))
    for key in ("topk_ids", "topk_logprobs", "sampled_logprobs"):
        assert torch.equal(first[key], torch.cat([plain[0][key], plain[1][key]]))
    assert "packed_requests" not in packed[1] and torch.equal(packed[1]["input_ids"], plain[2]["input_ids"])
    with pytest.raises(RRT.InvalidRollout):
        RRT.build_request_samples(raw(), None, image_pad_id=-1, hard_max_tokens=7, pack_prefixes=True)


class CausalCore(torch.nn.Module):
    """hidden_t depends on tokens <= t only (running mean of embeddings), like a causal LM trunk."""
    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.emb = torch.nn.Parameter(torch.randn(V, 6, generator=g))
        self.mix = torch.nn.Parameter(torch.randn(6, 6, generator=g) * 0.3)

    def forward(self, input_ids, **_):
        e = self.emb[input_ids[0]]
        running = e.cumsum(0) / torch.arange(1, len(e) + 1).unsqueeze(-1)
        return SimpleNamespace(last_hidden_state=torch.tanh(running @ self.mix).unsqueeze(0))


@pytest.mark.parametrize("is_cap", [None, 1.5])
def test_packed_loss_and_gradient_equal_the_sum_of_per_request_objectives(is_cap):
    core, ref_core, head = CausalCore(0), CausalCore(1), torch.nn.Linear(6, V, bias=False)
    reference = lambda inputs, mask: T.masked_logprobs(ref_core, head, inputs, mask, per_token=True)

    def total(samples):
        loss = torch.zeros(())
        for s in samples:
            s = dict(s, advantage=0.8, trajectory_tokens=11)
            inputs, mask = {"input_ids": s["input_ids"].unsqueeze(0)}, s["mask"].unsqueeze(0)
            l, _, _ = T.sample_loss(None, core, head, inputs, mask, s, score_centering=True, kl_beta=0.05,
                                    full_completion_tokens=11, world_size=2, reference=reference, is_cap=is_cap)
            loss = loss + l
        return loss

    plain, _ = RRT.build_request_samples(raw(), None, image_pad_id=-1, hard_max_tokens=100)
    packed, _ = RRT.build_request_samples(raw(), None, image_pad_id=-1, hard_max_tokens=100, pack_prefixes=True)
    lp, lq = total(plain), total(packed)
    params = [core.emb, core.mix, head.weight]
    gp, gq = torch.autograd.grad(lp, params), torch.autograd.grad(lq, params)
    assert torch.allclose(lp, lq, atol=1e-6)
    for a, b in zip(gp, gq):
        assert torch.allclose(a, b, atol=1e-6)


def test_filler_is_a_short_text_pass_or_none():
    s = {"input_ids": torch.arange(40), "mask": torch.ones(40, dtype=torch.bool), "length": 40, "index": 3, "trajectory_tokens": 9}
    f = T.filler_sample(s, image_pad_id=99)
    assert f["length"] == 32 and int(f["mask"].sum()) == 1 and "pixel_values" not in f and f["trajectory_tokens"] == 9
    assert T.filler_sample(s, image_pad_id=5) is None           # an image placeholder in the first tokens


def test_packed_cost_counts_one_pass_per_chain():
    import prefix_chains as PC
    selected, _ = RRT.validate_rollout(raw(select=(1, 2, 3, 4)))
    got = PC.packed_cost(selected, quadratic_tokens=10.0)
    assert got["packed_sequences"] == 2 and got["packed_context_tokens"] == 8 + 8     # [1,2] -> 8 tokens, [3,4] -> 8 tokens
    assert got["packed_cost"] == pytest.approx(2 * (8 + 6.4))
    assert RRT.prefix_chains is PC.prefix_chains


def test_chains_never_exceed_one_lm_head_chunk_of_targets():
    import prefix_chains as PC
    selected, _ = RRT.validate_rollout(raw(select=(1, 2, 3, 4)))
    assert [[r["request_index"] for r in c] for c in PC.prefix_chains(selected, max_targets=3)] == [[1, 2], [3], [4]]
    assert [[r["request_index"] for r in c] for c in PC.prefix_chains(selected, max_targets=2)] == [[1], [2], [3], [4]]
    long = raw(select=(1, 2), prompts={1: ([1, 2], list(range(3, 3 + 3000))), 2: ([1, 2] + list(range(3, 3 + 3000)) + [7], [9] * 2000)})
    packed, _ = RRT.build_request_samples(long, None, image_pad_id=-1, hard_max_tokens=10_000, pack_prefixes=True)
    assert len(packed) == 2 and all(int(p["mask"].sum()) <= RRT.LM_HEAD_CHUNK for p in packed)   # 5000 targets: not one chain
    assert PC.packed_cost(long["training_requests"])["packed_sequences"] == 2

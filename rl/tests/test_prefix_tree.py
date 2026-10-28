"""Unit checks of rl/prefix_tree.py on a tiny random Qwen3.5 (2 GatedDeltaNet + 1 full-attention layer per block, small vision
tower), fp32, GPU (fla kernels). Run: python test_prefix_tree.py  (prints PREFIX_TREE_CHECK OK)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prefix_tree as PT  # noqa: E402

PAD, VS, VE = 900, 901, 902
try:                       # fla's Triton kernels compute fp32 dots in tf32: ~1e-4 relative; the torch reference path is exact fp32
    import fla  # noqa: F401
    TOL = 1e-3
except ImportError:
    TOL = 1e-5


def tiny_model(seed=0):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    torch.manual_seed(seed)
    cfg = Qwen3_5Config(
        text_config=dict(hidden_size=128, intermediate_size=256, num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=64, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32,
                         linear_value_head_dim=32, linear_conv_kernel_dim=4, vocab_size=1000, full_attention_interval=4,
                         layer_types=["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"],
                         rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_interleaved": True,
                                          "mrope_section": [3, 3, 2], "partial_rotary_factor": 0.25}),
        vision_config=dict(depth=1, hidden_size=64, num_heads=2, intermediate_size=128, out_hidden_size=128, patch_size=16,
                           spatial_merge_size=2, temporal_patch_size=2, in_channels=3, num_position_embeddings=64),
        image_token_id=PAD, vision_start_token_id=VS, vision_end_token_id=VE)
    m = Qwen3_5ForConditionalGeneration(cfg).float().cuda()
    with torch.no_grad():                       # non-trivial norms / decay
        for n, p in m.named_parameters():
            if p.ndim == 1:
                p.add_(0.1 * torch.randn_like(p))
    m.config.use_cache = False
    for p in m.model.visual.parameters():
        p.requires_grad_(False)
    return m


def image(h=4, w=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    grid = torch.tensor([[1, h, w]])
    pix = torch.randn(h * w, 3 * 2 * 16 * 16, generator=g)
    return pix, grid, [VS] + [PAD] * (h * w // 4) + [VE]


def episode(n_branch=3, prefix_text=150, seed=0, bad_prefix_image=False):
    g = torch.Generator().manual_seed(seed)
    txt = lambda n: torch.randint(10, 800, (n,), generator=g).tolist()  # noqa: E731
    p0, g0, t0 = image(seed=1)
    pre = txt(prefix_text) + t0 + txt(37)
    out = []
    for b in range(n_branch):
        ids, pix, grids = list(pre), [p0 if not (bad_prefix_image and b == 1) else p0 + 1], [g0]
        for k in range(b % 2 + 1):
            pk, gk, tk = image(seed=10 * b + k + 2)
            ids += txt(23 + 7 * b) + tk
            pix.append(pk); grids.append(gk)
        prompt = len(ids)
        ids += txt(30 + 5 * b)
        mask = torch.zeros(len(ids), dtype=torch.bool)
        mask[prompt:] = True
        out.append({"input_ids": torch.tensor(ids), "mask": mask, "length": len(ids), "pixel_values": torch.cat(pix),
                    "image_grid_thw": torch.cat(grids), "w": torch.randn(int(mask.sum()), generator=g)})
    return out


def logp(core, head, inputs, mask):
    h = core(**inputs).last_hidden_state[0]
    pos = torch.nonzero(mask[0, 1:]).flatten()
    return torch.log_softmax(head(h[pos]).float(), -1).gather(-1, inputs["input_ids"][0][pos + 1, None]).squeeze(-1)


def full_inputs(s):
    ids = s["input_ids"].unsqueeze(0).cuda()
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "pixel_values": s["pixel_values"].cuda(),
            "image_grid_thw": s["image_grid_thw"].cuda(), "mm_token_type_ids": (ids == PAD).long()}


def grads(m):
    return [None if p.grad is None else p.grad.detach().clone() for p in m.parameters() if p.requires_grad]


def rel(a, b):
    num = sum(float((x - y).double().norm() ** 2) for x, y in zip(a, b) if x is not None)
    den = sum(float(x.double().norm() ** 2) for x in a if x is not None)
    return (num / den) ** 0.5


def per_request(m, samples):
    m.zero_grad(set_to_none=True)
    lps = []
    for s in samples:
        lp = logp(m.model, m.lm_head, full_inputs(s), s["mask"].unsqueeze(0).cuda())
        (lp * s["w"].cuda()).sum().backward()
        lps.append(lp.detach())
    return grads(m), lps


def tree(m, samples, align=PT.ALIGN):
    m.zero_grad(set_to_none=True)
    plan = PT.plan_episode(samples, m.model, PAD, align=align, device="cuda")
    lps = []

    def loss_fn(core_b, inputs, mask, s, ref_lp, scale):
        lp = logp(core_b, m.lm_head, inputs, mask)
        (lp * s["w"].cuda() * scale).sum().backward()
        lps.append(lp.detach())
    PT.run_episode(plan, core=m.model, lm_head=m.lm_head, device="cuda", loss_fn=loss_fn, masked_logprobs=None)
    return grads(m), lps, plan


def main():
    m = tiny_model()
    samples = episode()
    # 1. default path unchanged by installing the wrappers/hooks (they only act on PrefixBranchMask / capture)
    m.train()
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    g0, l0 = per_request(m, samples)
    g0b, _ = per_request(m, samples)         # backward kernels may be nondeterministic (atomics): measure the floor
    floor = rel(g0, g0b)
    PT.install_attention()
    PT.allow_cache_under_checkpointing(m.model)
    PT.install_capture_hooks(m.model)
    g1, l1 = per_request(m, samples)
    assert all(torch.equal(a, b) for a, b in zip(l0, l1)), "installed wrappers changed the per-request log-probs"
    if floor == 0.0:
        assert all((a is None and b is None) or torch.equal(a, b) for a, b in zip(g0, g1)), "installed wrappers changed the gradients"
    else:                                    # nondeterministic backward: same distance as two plain runs
        assert rel(g0, g1) <= 3 * floor + 1e-12, f"installed wrappers changed the gradients ({rel(g0, g1):.2e} vs floor {floor:.2e})"
    print(f"default path with wrappers installed: log-probs bit-identical, gradients {'bit-identical' if floor == 0 else f'within the backward nondeterminism floor {floor:.1e}'}: OK")
    # 2. tree == per-request (fp32), with activation checkpointing and without
    for ckpt in (True, False):
        if ckpt:
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            m.gradient_checkpointing_disable()
        gb, lb = per_request(m, samples)
        for align in (64, 16, 1):
            gt, lt, plan = tree(m, samples, align=align)
            dl = max(float((a - b).abs().max()) for a, b in zip(lb, lt))
            r = rel(gb, gt)
            print(f"ckpt={ckpt} align={align} P={plan.P} lcp={plan.lcp}: max |dlogp| {dl:.2e}, grad rel L2 {r:.2e}")
            assert dl < TOL and r < TOL, "tree differs from per-request beyond fp32 rounding"
            assert plan.P % align == 0
    # 3. planning rules
    plan = PT.plan_episode(samples, m.model, PAD, align=1, device="cuda")
    pre_runs = PT._pad_runs(plan.ids, PAD)
    assert all(b <= plan.P for a, b in pre_runs)
    for s in samples:
        assert int(torch.nonzero(s["mask"]).flatten()[0]) - 1 >= plan.P
    try:
        PT.plan_episode(episode(bad_prefix_image=True), m.model, PAD, align=1, device="cuda")
        raise AssertionError("a differing prefix image must be refused")
    except ValueError:
        pass
    # an image straddling the LCP-derived cut: P moves to the image start
    s2 = episode(n_branch=2, prefix_text=60, seed=3)
    runs = PT._pad_runs(s2[0]["input_ids"], PAD)
    a, b = runs[0]
    plan2 = PT.plan_episode(s2, m.model, PAD, align=1, device="cuda")
    assert not (a < plan2.P < b)
    # 4. zero-weight dummies: no gradient, same structure
    m.zero_grad(set_to_none=True)
    dp = PT.dummy_plan(3)

    def zero_loss(core_b, inputs, mask, s, ref_lp, scale):
        (logp(core_b, m.lm_head, inputs, mask).sum() * 0.0).backward()
    PT.run_episode(dp, core=m.model, lm_head=m.lm_head, device="cuda", loss_fn=zero_loss, masked_logprobs=None)
    assert all(p.grad is None or float(p.grad.abs().max()) == 0.0 for p in m.parameters() if p.requires_grad)
    print("PREFIX_TREE_CHECK OK")


if __name__ == "__main__":
    main()

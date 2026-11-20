"""CPU unit tests for the MTP drafter pipeline (tiny random Qwen3.5 config; no GPU, no checkpoints).

    cd tools/drafter && python -m pytest -q test_drafter.py
"""
from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F

from bodies import Sample, chain_starts, select_bodies
from core import chain_pass, simulate_acceptance
from mtp_head import MTPHead
from target import use_torch_gdn

use_torch_gdn()


def tiny_text_config():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    return Qwen3_5TextConfig(
        vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
        linear_value_head_dim=16, full_attention_interval=2, mtp_num_hidden_layers=1,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [1, 1, 2], "mrope_interleaved": True},
        partial_rotary_factor=0.25,
    )


class TinyTarget:
    """Stand-in for target.Target: random text model, final-norm hidden states, M-RoPE positions = arange."""

    def __init__(self, cfg):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

        torch.manual_seed(0)
        self.lm = Qwen3_5TextModel(cfg).eval().requires_grad_(False)
        for p in self.lm.parameters():  # zero-init norms otherwise make everything degenerate
            p.data.normal_(0, 0.2)
        self.embed_tokens = self.lm.embed_tokens
        self.rotary = self.lm.rotary_emb
        self.lm_head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False).requires_grad_(False)
        self.text_config = cfg

    def features(self, ids, chunk=None):
        from transformers.cache_utils import DynamicCache

        emb = self.embed_tokens(ids[None])
        pos = torch.arange(ids.numel())[None, None].expand(3, 1, -1)
        if chunk is None:
            h = self.lm(inputs_embeds=emb, position_ids=pos).last_hidden_state[0]
        else:
            cache, hs = DynamicCache(config=self.text_config), []
            for a in range(0, ids.numel(), chunk):
                out = self.lm(inputs_embeds=emb[:, a:a + chunk], position_ids=pos[:, :, a:a + chunk],
                              past_key_values=cache, use_cache=True)
                hs.append(out.last_hidden_state[0])
            h = torch.cat(hs)
        return {"h": h, "emb": emb[0], "pos": pos[:, 0]}

    def rope(self, pos3, dtype):
        cos, sin = self.rotary(torch.empty(0, dtype=dtype), pos3[:, None, :])
        return cos[0], sin[0]


@pytest.fixture(scope="module")
def setup():
    cfg = tiny_text_config()
    tgt = TinyTarget(cfg)
    torch.manual_seed(1)
    head = MTPHead(cfg)
    for p in head.parameters():
        p.data.normal_(0, 0.1)
    return cfg, tgt, head


def test_chunked_prefill_matches_full(setup):
    _, tgt, _ = setup
    ids = torch.randint(0, 97, (37,))
    full = tgt.features(ids)["h"]
    chunked = tgt.features(ids, chunk=8)["h"]
    torch.testing.assert_close(chunked, full, atol=2e-4, rtol=1e-3)


def vllm_reference(head, tgt, feats, ids, s, steps):
    """Literal vLLM decode for chain s: step-0 KV over 0..s from target states, then draft steps appending KV."""
    h, emb, pos = feats["h"], feats["emb"], feats["pos"]
    lay = head.layers[0]
    ks, vs, o_prev, outs = [], [], None, []
    for t in range(s + 1):  # step-0 KV cache (prefill, causal over the whole context)
        z = head._project(emb[t + 1][None], h[t][None])
        c, si = tgt.rope(pos[:, t:t + 1], h.dtype)
        q, gate, k, v = head._qkv(lay.input_layernorm(z), c, si)
        ks.append(k), vs.append(v)
        if t == s:
            K, V = torch.cat(ks, 1), torch.cat(vs, 1)
            a = F.scaled_dot_product_attention(q[None], K[None], V[None], scale=head.scaling, enable_gqa=True)[0]
            o_prev = head._finish(z, a, gate)
            outs.append(o_prev)
    for j in range(1, steps):
        z = head._project(tgt.embed_tokens(ids[s + 1 + j])[None], o_prev)
        c, si = tgt.rope((pos[0, s:s + 1] + j)[None].expand(3, -1), h.dtype)
        q, gate, k, v = head._qkv(lay.input_layernorm(z), c, si)
        ks.append(k), vs.append(v)
        K, V = torch.cat(ks, 1), torch.cat(vs, 1)
        a = F.scaled_dot_product_attention(q[None], K[None], V[None], scale=head.scaling, enable_gqa=True)[0]
        o_prev = head._finish(z, a, gate)
        outs.append(o_prev)
    return [tgt.lm_head(o)[0] for o in outs]


def test_unrolled_chains_match_vllm_decode(setup):
    _, tgt, head = setup
    torch.manual_seed(2)
    ids = torch.randint(0, 97, (40,))
    spans = [(10, 22), (30, 39)]
    feats = tgt.features(ids)
    with torch.no_grad():
        _, stats = chain_pass(head, tgt, feats, ids, spans, steps=3, T=0.7)
        # rebuild the per-chain logits through the same path for comparison
        for s in (9, 15, 29, 35):
            ref = vllm_reference(head, tgt, feats, ids, s, 3)
            for j, r in enumerate(ref):
                st = stats[j]
                idx = (st["S"] == s).nonzero()
                if idx.numel() == 0:
                    continue
                hit = st["hit"][idx[0, 0]].item()
                assert hit == (r.argmax().item() == ids[s + 2 + j].item())
    # logits equality on a full step through the module API
    with torch.no_grad():
        o0, k0, v0 = head.step0(feats["emb"][1:], feats["h"][:-1], *tgt.rope(feats["pos"][:, :39], torch.float32),
                                torch.arange(39))
        S = torch.tensor(chain_starts(spans, 1))
        c, si = tgt.rope((feats["pos"][0, S] + 1)[None].expand(3, -1), torch.float32)
        o1, _, _ = head.step_j(tgt.embed_tokens(ids[S + 2]), o0[S], c, si, S, k0, v0, [], [], q_chunk=4)
        for i, s in enumerate(S.tolist()):
            ref = vllm_reference(head, tgt, feats, ids, s, 2)
            torch.testing.assert_close(tgt.lm_head(o0[s]), ref[0], atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(tgt.lm_head(o1[i]), ref[1], atol=1e-4, rtol=1e-4)


def test_training_step_reduces_loss_and_freezes_target(setup):
    cfg, tgt, _ = setup
    torch.manual_seed(3)
    head = MTPHead(cfg)
    for p in head.parameters():
        p.data.normal_(0, 0.05)
    ids = torch.randint(0, 97, (48,))
    spans = [(8, 30), (35, 47)]
    feats = tgt.features(ids)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3)
    losses = []
    for _ in range(25):
        loss, _ = chain_pass(head, tgt, feats, ids, spans, steps=3, T=0.7, row_chunk=7)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.7 * losses[0], losses
    assert all(p.grad is None for p in tgt.lm.parameters()) and tgt.lm_head.weight.grad is None


def test_save_load_roundtrip(setup, tmp_path):
    cfg, _, head = setup
    path = tmp_path / "mtp.safetensors"
    head.save_mtp(str(path), dtype=torch.float32)
    other = MTPHead(cfg)
    other.load_mtp(str(path))
    for (k, a), (_, b) in zip(head.state_dict().items(), other.state_dict().items()):
        torch.testing.assert_close(a, b, msg=k)
    from safetensors import safe_open

    keys = set(safe_open(str(path), "pt").keys())
    assert "mtp.fc.weight" in keys and "mtp.layers.0.self_attn.q_proj.weight" in keys and "mtp.norm.weight" in keys


def test_chain_starts_and_acceptance_replay():
    assert chain_starts([(5, 9)], 0) == [4, 5, 6, 7]  # drafts x6..x9
    assert chain_starts([(5, 9)], 2) == [4, 5]  # third drafts x8, x9
    stats = {j: {"S": torch.tensor(chain_starts([(5, 9)], j)), "hit": torch.ones(len(chain_starts([(5, 9)], j)),
                                                                                      dtype=torch.bool)}
             for j in range(3)}
    # all drafts hit: s=4 emits x6,x7,x8 + bonus x9=<|im_end|> -> one decode step, 4 tokens
    assert simulate_acceptance([(5, 9)], stats, 3) == (1, 4)
    for st in stats.values():
        st["hit"][:] = False
    assert simulate_acceptance([(5, 9)], stats, 3) == (4, 4)


def _body(cwd, n_turns, image_cap_after=None):
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": f"<cwd>{cwd}</cwd>"},
            {"role": "user", "content": [{"type": "text", "text": "go"},
                                         {"type": "image_url", "image_url": {"url": "data:x"}}]}]
    for i in range(n_turns):
        msgs.append({"role": "assistant", "content": f"a{i}", "reasoning_content": f"r{i}"})
        msgs.append({"role": "tool", "content": f"t{i}"})
    if image_cap_after is not None and n_turns > image_cap_after:
        msgs[2] = {"role": "user", "content": [{"type": "text", "text": "go"}]}
    return {"messages": msgs}


def test_select_bodies_modes(tmp_path):
    p = tmp_path / "rec.jsonl"
    rows = [_body("A", n, image_cap_after=2) for n in range(5)] + [_body("B", n) for n in range(3)]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    exact = select_bodies(str(p), "exact")
    cover = select_bodies(str(p), "cover")
    # episode A: body n=2 is not a prefix of n=3 (image capped) -> maximal; B: only the last body
    assert [len(s.train_turns) for s in exact if s.episode == "A"] == [2, 2]
    assert [len(s.train_turns) for s in cover if s.episode == "A"] == [4]
    assert [len(s.train_turns) for s in exact if s.episode == "B"] == [2]
    assert all(isinstance(s, Sample) for s in exact + cover)


def test_vocab_subset_full_equals_none_and_tv_is_acceptance(setup):
    _, tgt, head = setup
    torch.manual_seed(4)
    ids = torch.randint(0, 97, (40,))
    feats = tgt.features(ids)
    with torch.no_grad():
        _, a = chain_pass(head, tgt, feats, ids, [(10, 30)], steps=2)
        _, b = chain_pass(head, tgt, feats, ids, [(10, 30)], steps=2, vocab_ids=torch.arange(97))
        _, c = chain_pass(head, tgt, feats, ids, [(10, 30)], steps=2, vocab_ids=torch.arange(0, 97, 2))
    for j in a:
        for k in ("kl", "p_acc", "p_prob", "p_max"):
            torch.testing.assert_close(a[j][k], b[j][k], atol=1e-5, rtol=1e-5)
        assert torch.all(a[j]["p_prob"] <= 1 + 1e-6) and torch.all(c[j]["p_prob"] <= 1 + 1e-6)
        assert torch.all(c[j]["p_max"] <= a[j]["p_max"] + 1e-6)  # subset ceiling <= full ceiling


def test_compaction_turns_trained_before_compaction(tmp_path):
    # body 3 adds a compaction user message: turns a0..a2 lose their reasoning there, so cover mode must train them
    # in the last pre-compaction body and only the post-compaction turn in the final body
    rows = [_body("C", n) for n in range(4)]
    compacted = _body("C", 3)
    compacted["messages"].append({"role": "user", "content": "<context_compaction_done>\nsummary"})
    compacted["messages"] += [{"role": "assistant", "content": "a3", "reasoning_content": "r3"}]
    rows.append(compacted)
    p = tmp_path / "rec.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    cover = select_bodies(str(p), "cover")
    assert [len(s.train_turns) for s in cover] == [3, 1]


def tiny_block_cfg():
    return {"hidden_size": 32, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "intermediate_size": 48, "num_hidden_layers": 2, "rms_norm_eps": 1e-6, "sliding_window": 3,
            "layer_types": ["sliding_attention", "full_attention"], "rope_parameters": {"rope_theta": 10000.0},
            "vocab_size": 97, "dflash_config": {"target_layer_ids": [0, 1], "mask_token_id": 5, "block_size": 4}}


def block_reference(dr, aux, pos, ids, a, N, embed):
    """Literal per-anchor DFlash forward: context K/V at positions < a, block [x_a, mask*N] at a..a+N."""
    from block_drafter import apply_rope, rope_cos_sin

    B, D = 1 + N, dr.cfg["head_dim"]
    c = dr.context(aux[:a])
    tok = torch.full((B,), dr.mask_token_id)
    tok[0] = ids[a]
    x = embed(tok)
    qpos = pos[a - 1] + 1 + torch.arange(B)
    for layer in dr.layers:
        cos, sin = rope_cos_sin(pos[:a], D, dr.theta)
        K, V = layer.ctx_kv(c, cos, sin)
        sa = layer.self_attn
        h = layer.input_layernorm(x)
        q = sa.q_norm(sa.q_proj(h).view(B, -1, D)).transpose(0, 1)
        k = sa.k_norm(sa.k_proj(h).view(B, -1, D)).transpose(0, 1)
        v = sa.v_proj(h).view(B, -1, D).transpose(0, 1)
        qc, qs = rope_cos_sin(qpos, D, dr.theta)
        q, k = apply_rope(q, qc, qs), apply_rope(k, qc, qs)
        Kall, Vall = torch.cat([K, k], 1), torch.cat([V, v], 1)
        kpos = torch.cat([pos[:a], qpos])
        is_blk = torch.arange(kpos.numel()) >= a
        allow = torch.ones(B, kpos.numel(), dtype=torch.bool)
        if layer.sliding is not None:
            allow &= (qpos[:, None] - kpos[None]) < layer.sliding
        if layer.causal:
            allow &= ~is_blk[None] | (kpos[None] <= qpos[:, None])
        g = q.shape[0] // K.shape[0]
        Kr, Vr = Kall.repeat_interleave(g, 0), Vall.repeat_interleave(g, 0)
        s = (q @ Kr.transpose(1, 2)) * dr.scale
        s = s.masked_fill(~allow[None], float("-inf"))
        o = torch.softmax(s, -1) @ Vr
        x = x + sa.o_proj(o.transpose(0, 1).reshape(B, -1))
        hn = layer.post_attention_layernorm(x)
        m = layer.mlp
        x = x + m.down_proj(torch.nn.functional.silu(m.gate_proj(hn)) * m.up_proj(hn))
    return dr.norm(x)


def test_block_drafter_matches_literal_reference():
    from block_drafter import BlockDrafter

    torch.manual_seed(5)
    dr = BlockDrafter(tiny_block_cfg(), dspark=True, markov_rank=4, target_vocab=97)
    for p in dr.parameters():
        p.data.normal_(0, 0.3)
    embed = torch.nn.Embedding(97, 32)
    L, N = 30, 3
    aux = torch.randn(L, 64)
    ids = torch.randint(0, 97, (L,))
    pos = torch.arange(L)
    anchors = torch.tensor([4, 9, 17, 25])
    with torch.no_grad():
        got = dr.blocks(aux, pos, ids, anchors, N, embed, kv_chunk=7, a_chunk=3)
        for bi, a in enumerate(anchors.tolist()):
            ref = block_reference(dr, aux, pos, ids, a, N, embed)
            torch.testing.assert_close(got[bi], ref, atol=1e-4, rtol=1e-4)


def test_block_pass_trains_and_saves(tmp_path):
    from block_core import block_pass
    from block_drafter import BlockDrafter, save_draft_dir

    class Tgt:
        pass

    torch.manual_seed(6)
    t = Tgt()
    t.embed_tokens = torch.nn.Embedding(97, 32).requires_grad_(False)
    t.lm_head = torch.nn.Linear(32, 97, bias=False).requires_grad_(False)
    dr = BlockDrafter(tiny_block_cfg(), dspark=True, markov_rank=4, target_vocab=97)
    for p in dr.parameters():
        p.data.normal_(0, 0.2)
    vocab = torch.arange(0, 97, 2)
    dr.init_dspark_heads(vocab.numel())
    L = 40
    ids = torch.randint(0, 97, (L,))
    feats = {"h": torch.randn(L, 32), "aux": torch.randn(L, 64)}
    spans = [(10, 25), (30, 39)]
    opt = torch.optim.AdamW(dr.parameters(), lr=3e-3)
    losses = []
    for _ in range(15):
        loss, st = block_pass(dr, t, feats, ids, spans, 3, vocab_ids=vocab, row_chunk=5, a_chunk=4)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], losses
    assert set(st) == {0, 1, 2} and st[0]["S"].tolist() == [a - 1 for a in list(range(10, 25)) + list(range(30, 39))]
    out = tmp_path / "d"
    save_draft_dir(dr, str(out), tiny_block_cfg(), vocab, t.lm_head.weight, 4)
    back = BlockDrafter.from_dir(str(out))
    assert back.dspark and back.markov_head.markov_w2.weight.shape[0] == vocab.numel()
    cfg = json.load(open(out / "config.json"))
    assert cfg["architectures"] == ["Qwen3DSparkModel"] and cfg["draft_vocab_size"] == vocab.numel()

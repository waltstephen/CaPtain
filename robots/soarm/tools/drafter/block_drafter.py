"""DFlash / DSpark block drafter for the Qwen3.5 target, mirroring vLLM 0.29's qwen3_dflash.py / qwen3_dspark.py.

Layout (DFlash "1 + N" fill-in, vLLM v1/spec_decode/dflash.py and the V2 DSpark speculator with
sample_from_anchor=false): after the target verified up to position t, the query block is
[x_{t+1} (bonus/anchor a = t+1), mask * N] at positions a .. a+N (M-RoPE row 0 of the target). Mask slot i (1..N)
predicts the token AT its own position, x_{a+i}. Every draft layer attends
  * context K/V: k_norm(k_proj(C)), v_proj(C) at positions < a, with C = hidden_norm(fc(concat target aux states))
    (no input_layernorm; computed once from the target's residual stream after the target_layer_ids layers),
  * the block's own K/V (q/k/v of input_layernorm(h), q_norm/k_norm, neox RoPE),
  sliding layers: causal, keys within (q_pos - window, q_pos]; full layers: non-causal (whole block + all context).
Embeddings and lm_head are the target's (frozen); with a draft vocabulary subset (16k ids) the lm_head rows of the
subset are used (served as draft_vocab_size + d2t).
DSpark adds, per mask slot i, a Markov bias markov_w2 @ markov_w1[prev] (prev = x_{a+i-1}; the anchor for slot 1) on
the draft-vocab logits and a confidence head sigmoid(proj(cat[hidden, markov_w1[prev]])) (vllm qwen3_dspark.py).
Trainable: fc, hidden_norm, the draft layers, norm (+ Markov / confidence heads).
"""
from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):  # Qwen3 / vLLM RMSNorm: x * rsqrt(mean(x^2) + eps) * w (fp32 math)
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()).to(x.dtype)


def rope_cos_sin(pos: torch.Tensor, head_dim: int, theta: float):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=pos.device, dtype=torch.float32) / head_dim))
    f = pos.float()[:, None] * inv[None]
    emb = torch.cat([f, f], dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):  # x [..., n, D], neox style
    d = x.shape[-1] // 2
    rot = torch.cat([-x[..., d:], x[..., :d]], dim=-1)
    return (x.float() * cos + rot.float() * sin).to(x.dtype)


class Layer(nn.Module):
    def __init__(self, cfg, idx):
        super().__init__()
        H, nq, nkv, D = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        self.nq, self.nkv, self.D = nq, nkv, D
        eps = cfg["rms_norm_eps"]
        lt = (cfg.get("layer_types") or ["full_attention"] * cfg["num_hidden_layers"])[idx]
        self.sliding = cfg["sliding_window"] if lt == "sliding_attention" else None
        self.causal = lt == "sliding_attention"  # vllm _dflash_layer_causal (no explicit is_causal/causal override)
        sa = nn.Module()
        sa.q_proj = nn.Linear(H, nq * D, bias=False)
        sa.k_proj = nn.Linear(H, nkv * D, bias=False)
        sa.v_proj = nn.Linear(H, nkv * D, bias=False)
        sa.o_proj = nn.Linear(nq * D, H, bias=False)
        sa.q_norm = RMSNorm(D, eps)
        sa.k_norm = RMSNorm(D, eps)
        self.self_attn = sa
        mlp = nn.Module()
        mlp.gate_proj = nn.Linear(H, cfg["intermediate_size"], bias=False)
        mlp.up_proj = nn.Linear(H, cfg["intermediate_size"], bias=False)
        mlp.down_proj = nn.Linear(cfg["intermediate_size"], H, bias=False)
        self.mlp = mlp
        self.input_layernorm = RMSNorm(H, eps)
        self.post_attention_layernorm = RMSNorm(H, eps)

    def ctx_kv(self, c, cos, sin):
        """c = hidden_norm(fc(aux)) [n, H] -> K, V [nkv, n, D] (K normed + roped)."""
        sa, n = self.self_attn, c.shape[0]
        k = sa.k_norm(sa.k_proj(c).view(n, self.nkv, self.D)).transpose(0, 1)
        v = sa.v_proj(c).view(n, self.nkv, self.D).transpose(0, 1)
        return apply_rope(k, cos, sin), v

    def forward(self, x, qpos, cos, sin, anchors, K, V, ctx_pos, B, scale, a_chunk):
        """x [nb*B, H] block inputs (nb anchors, B = 1+N slots), qpos [nb*B]; K/V [nkv, Lc, D] context."""
        sa = self.self_attn
        n = x.shape[0]
        h = self.input_layernorm(x)
        q = sa.q_norm(sa.q_proj(h).view(n, self.nq, self.D)).transpose(0, 1)
        k = sa.k_norm(sa.k_proj(h).view(n, self.nkv, self.D)).transpose(0, 1)
        v = sa.v_proj(h).view(n, self.nkv, self.D).transpose(0, 1)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        nb = anchors.numel()
        outs = []
        for a0 in range(0, nb, a_chunk):
            a1 = min(a0 + a_chunk, nb)
            sl = slice(a0 * B, a1 * B)
            an = anchors[a0:a1]
            hi = int(an.max())  # context indices this chunk can see: [lo, hi)
            lo = 0
            if self.sliding is not None:  # keys within the window of the chunk's earliest query only
                lo = int(torch.searchsorted(ctx_pos[:hi], qpos[sl].min() - self.sliding, right=True))
            args = (q[:, sl], k[:, sl], v[:, sl], qpos[sl], an, K[:, lo:hi], V[:, lo:hi], ctx_pos[lo:hi], lo, B,
                    scale, self.sliding, self.causal)
            outs.append(checkpoint(_block_attention, *args, use_reentrant=False) if torch.is_grad_enabled()
                        else _block_attention(*args))
        o = torch.cat(outs, dim=1).transpose(0, 1).reshape(n, -1)
        x = x + sa.o_proj(o)
        m = self.mlp
        hn = self.post_attention_layernorm(x)
        return x + m.down_proj(F.silu(m.gate_proj(hn)) * m.up_proj(hn))


def _block_attention(q, k, v, qpos, anchors, K, V, ctx_pos, lo, B, scale, sliding, causal):
    """q [nq, nb*B, D]; k/v [nkv, nb*B, D] of the blocks; K/V [nkv, m, D] = context indices [lo, lo+m) at ctx_pos
    [m] (the slice this chunk can see). Block b (anchor index a_b) attends context indices < a_b and its own block."""
    nq, nqt, D = q.shape
    nkv = K.shape[0]
    g = nq // nkv
    nb = anchors.numel()
    m = K.shape[1]
    qg = q.view(nkv, g, nb, B, D)
    s_ctx = torch.einsum("hgbqd,hmd->hgbqm", qg, K).float() * scale
    ci = lo + torch.arange(m, device=q.device)
    keep = ci[None, None, :] < anchors[:, None, None]  # [nb, 1, m]
    qp = qpos.view(nb, B)
    if sliding is not None:
        keep = keep & ((qp[:, :, None] - ctx_pos[None, None, :]) < sliding)
    s_ctx = s_ctx.masked_fill(~keep[None, None], float("-inf"))
    kb, vb = k.view(nkv, nb, B, D), v.view(nkv, nb, B, D)
    s_blk = torch.einsum("hgbqd,hbkd->hgbqk", qg, kb).float() * scale
    if causal or sliding is not None:
        bk = torch.ones(B, B, dtype=torch.bool, device=q.device)
        if causal:
            bk = torch.tril(bk)
        if sliding is not None:
            d = torch.arange(B, device=q.device)
            bk = bk & ((d[:, None] - d[None, :]) < sliding)
        s_blk = s_blk.masked_fill(~bk, float("-inf"))
    p = torch.softmax(torch.cat([s_ctx, s_blk], dim=-1), dim=-1).to(q.dtype)
    out = torch.einsum("hgbqm,hmd->hgbqd", p[..., :m], V) + torch.einsum("hgbqk,hbkd->hgbqd", p[..., m:], vb)
    return out.reshape(nq, nqt, D)


class BlockDrafter(nn.Module):
    def __init__(self, cfg: dict, dspark: bool = False, markov_rank: int = 256, target_vocab: int = 248320):
        super().__init__()
        self.cfg = cfg
        H = cfg["hidden_size"]
        dc = cfg.get("dflash_config", {})
        self.layer_ids = dc["target_layer_ids"]
        self.mask_token_id = dc["mask_token_id"]
        self.theta = cfg["rope_parameters"]["rope_theta"]
        self.fc = nn.Linear(H * len(self.layer_ids), H, bias=False)
        self.hidden_norm = RMSNorm(H, cfg["rms_norm_eps"])
        self.layers = nn.ModuleList([Layer(cfg, i) for i in range(cfg["num_hidden_layers"])])
        self.norm = RMSNorm(H, cfg["rms_norm_eps"])
        self.scale = cfg["head_dim"] ** -0.5
        self.dspark = dspark
        if dspark:
            mh = nn.Module()
            mh.markov_w1 = nn.Embedding(target_vocab, markov_rank)
            mh.markov_w2 = nn.Linear(markov_rank, 1, bias=False)  # resized to the draft vocab in set_vocab
            self.markov_head = mh
            ch = nn.Module()
            ch.proj = nn.Linear(H + markov_rank, 1, bias=True)
            self.confidence_head = ch
            self.markov_rank = markov_rank

    # ---------------- weights ----------------
    @classmethod
    def from_dir(cls, path: str, dspark: bool | None = None, **kw) -> "BlockDrafter":
        from safetensors.torch import load_file

        cfg = json.load(open(os.path.join(path, "config.json")))
        is_dspark = bool(cfg.get("markov_rank")) if dspark is None else dspark
        model = cls(cfg, dspark=is_dspark, markov_rank=cfg.get("markov_rank") or kw.get("markov_rank", 256),
                    target_vocab=cfg.get("vocab_size", 248320))
        sd = load_file(os.path.join(path, "model.safetensors"))
        sd = {k: v for k, v in sd.items() if k not in ("d2t", "t2d", "lm_head.weight", "embed_tokens.weight")}
        if is_dspark and "markov_head.markov_w2.weight" in sd:
            model.markov_head.markov_w2 = nn.Linear(model.markov_rank, sd["markov_head.markov_w2.weight"].shape[0],
                                                    bias=False)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        new_ok = ("markov_head.", "confidence_head.")
        missing = [k for k in missing if not k.startswith(new_ok)]
        if missing or unexpected:
            raise ValueError(f"draft weights mismatch: missing={missing} unexpected={unexpected}")
        return model

    def init_dspark_heads(self, draft_vocab: int) -> None:
        """Fresh Markov head (w1 ~ N(0, 0.02), w2 = 0 so the start equals the DFlash backbone) + confidence head."""
        nn.init.normal_(self.markov_head.markov_w1.weight, std=0.02)
        self.markov_head.markov_w2 = nn.Linear(self.markov_rank, draft_vocab, bias=False)
        nn.init.zeros_(self.markov_head.markov_w2.weight)
        nn.init.zeros_(self.confidence_head.proj.weight)
        nn.init.zeros_(self.confidence_head.proj.bias)

    # ---------------- forward ----------------
    def context(self, aux: torch.Tensor) -> torch.Tensor:
        return self.hidden_norm(self.fc(aux))

    def blocks(self, aux, pos0, ids, anchors, N, embed, kv_chunk=8192, a_chunk=16):
        """aux [L, nA*H] target aux states, pos0 [L] draft positions (token index: the V2 runner's 1-D positions;
        the V1 DFlash proposer would hand over M-RoPE row 0 for this multimodal target), ids [L], anchors [nb]
        (sorted token indices of the bonus/anchor tokens). Returns block hidden [nb, 1+N, H] after the final norm."""
        B = 1 + N
        dev = aux.device
        D = self.cfg["head_dim"]
        Lc = int(anchors.max())  # context indices < max anchor
        Ks, Vs = [[] for _ in self.layers], [[] for _ in self.layers]
        for a in range(0, Lc, kv_chunk):
            b = min(a + kv_chunk, Lc)
            cos, sin = rope_cos_sin(pos0[a:b], D, self.theta)

            def kv(x, cos=cos, sin=sin):
                c = self.context(x)
                return tuple(t for layer in self.layers for t in layer.ctx_kv(c, cos, sin))

            out = checkpoint(kv, aux[a:b], use_reentrant=False) if torch.is_grad_enabled() else kv(aux[a:b])
            for i in range(len(self.layers)):
                Ks[i].append(out[2 * i])
                Vs[i].append(out[2 * i + 1])
        Ks = [torch.cat(k, dim=1) for k in Ks]
        Vs = [torch.cat(v, dim=1) for v in Vs]
        ctx_pos = pos0[:Lc]
        # block inputs: [x_a, mask, ..., mask] at positions pos0[a-1]+1 .. +B (vLLM: last context position + 1 + off)
        last = pos0[anchors - 1]
        qpos = (last[:, None] + 1 + torch.arange(B, device=dev)[None]).reshape(-1)
        tok = torch.full((anchors.numel(), B), self.mask_token_id, dtype=torch.long, device=dev)
        tok[:, 0] = ids[anchors]
        x = embed(tok.reshape(-1)).to(aux.dtype)
        cos, sin = rope_cos_sin(qpos, D, self.theta)
        for i, layer in enumerate(self.layers):
            args = (x, qpos, cos, sin, anchors, Ks[i], Vs[i], ctx_pos, B, self.scale, a_chunk)
            x = checkpoint(layer, *args, use_reentrant=False) if torch.is_grad_enabled() else layer(*args)
        return self.norm(x).view(anchors.numel(), B, -1)

    def markov(self, prev_ids: torch.Tensor):
        e = self.markov_head.markov_w1(prev_ids)
        return e, self.markov_head.markov_w2(e)

    def confidence(self, hidden, markov_e):
        return self.confidence_head.proj(torch.cat([hidden, markov_e.to(hidden.dtype)], -1).float()).squeeze(-1)


def save_draft_dir(model: BlockDrafter, out: str, base_cfg: dict, vocab_ids: torch.Tensor | None,
                   lm_head_w: torch.Tensor, block_size: int, extra_meta: dict | None = None) -> None:
    """vLLM-loadable draft dir: config.json + model.safetensors (bf16) incl. the draft-vocab lm_head rows and d2t
    (vLLM DFlash/DSpark: draft_vocab_size + d2t[i] = target_id(i) - i). DSpark: Qwen3DSparkModel config fields."""
    from safetensors.torch import save_file

    os.makedirs(out, exist_ok=True)
    cfg = json.loads(json.dumps(base_cfg))
    cfg["dflash_config"] = dict(cfg.get("dflash_config", {}), block_size=block_size)
    sd = {k: v.detach().to(torch.bfloat16).contiguous().cpu() for k, v in model.state_dict().items()
          if not k.startswith("confidence_head.")}
    if model.dspark:
        sd["confidence_head.proj.weight"] = model.confidence_head.proj.weight.detach().float().contiguous().cpu()
        sd["confidence_head.proj.bias"] = model.confidence_head.proj.bias.detach().float().contiguous().cpu()
        cfg.update(architectures=["Qwen3DSparkModel"], markov_rank=model.markov_rank, markov_head_type="vanilla",
                   enable_confidence_head=True, confidence_head_with_markov=True, sample_from_anchor=False)
        cfg.pop("auto_map", None)
    if vocab_ids is not None:
        ids = vocab_ids.cpu().long()
        sd["lm_head.weight"] = lm_head_w[vocab_ids].detach().to(torch.bfloat16).contiguous().cpu()
        sd["d2t"] = ids - torch.arange(ids.numel())
        cfg["draft_vocab_size"] = int(ids.numel())
        cfg["tie_word_embeddings"] = False  # own (subset) lm_head; embed_tokens still shared from the target
    meta = {"format": "pt", **{k: str(v) for k, v in (extra_meta or {}).items()}}
    save_file(sd, os.path.join(out, "model.safetensors"), metadata=meta)
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)


def count_params(m: nn.Module) -> float:
    return sum(p.numel() for p in m.parameters()) / 1e6


__all__ = ["BlockDrafter", "save_draft_dir", "rope_cos_sin", "count_params"]

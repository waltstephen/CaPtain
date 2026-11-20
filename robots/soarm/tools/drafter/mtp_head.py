"""Qwen3.5 MTP head (vLLM qwen3_5_mtp.py semantics) with unrolled multi-step training.

vLLM, per draft step j (method "mtp", k = num_speculative_tokens, one MTP layer reused for every step):
  step 0 at target position s:  z = fc(cat[pre_fc_norm_embedding(E(x_{s+1})), pre_fc_norm_hidden(h_s)])
                                o_0 = norm(layer(z, pos s))      -> lm_head(o_0) drafts x_{s+2}
  h_s = the target's last hidden state AFTER its final norm (Qwen3NextModel.forward returns norm(h + residual), which
  gpu_model_runner passes as target_hidden_states for method mtp); E(x) = the target's input embedding (vision
  embeddings at image positions). The step-0 pass runs over every scheduled token (prefill too), so its KV cache covers
  the whole context.
  step j >= 1:                  input (E(d_j), o_{j-1}) at position s+j (all three M-RoPE rows = row0(s)+j); attends
                                the step-0 KV at positions <= s plus the chain's own step-1..j KV.
Training is teacher-forced on the recorded tokens (a later draft only matters when the earlier ones were accepted, i.e.
equal to the sampled tokens). Chain s attends K0[0..s] + {K1[s], .., Kj[s]}: implemented as chunked exact attention.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    apply_rotary_pos_emb,
)


class MTPHead(nn.Module):
    def __init__(self, text_config):
        super().__init__()
        H = text_config.hidden_size
        full_idx = text_config.layer_types.index("full_attention")
        self.fc = nn.Linear(2 * H, H, bias=False)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(text_config, full_idx)])
        self.norm = Qwen3_5RMSNorm(H, eps=text_config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(H, eps=text_config.rms_norm_eps)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(H, eps=text_config.rms_norm_eps)
        attn = self.layers[0].self_attn
        self.n_q = text_config.num_attention_heads
        self.n_kv = text_config.num_key_value_heads
        self.head_dim = attn.head_dim
        self.scaling = attn.scaling

    # ---- weights in the vLLM/HF checkpoint layout: mtp.<name> ----
    def load_mtp(self, path: str) -> None:
        from safetensors.torch import load_file

        sd = {k[len("mtp."):]: v for k, v in load_file(path).items() if k.startswith("mtp.")}
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise ValueError(f"MTP weights mismatch: missing={missing} unexpected={unexpected}")

    def save_mtp(self, path: str, dtype=torch.bfloat16, metadata: dict | None = None) -> None:
        from safetensors.torch import save_file

        sd = {f"mtp.{k}": v.detach().to(dtype).contiguous().cpu() for k, v in self.state_dict().items()}
        save_file(sd, path, metadata={"format": "pt", **(metadata or {})})

    # ---- pieces ----
    def _project(self, e: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([self.pre_fc_norm_embedding(e), self.pre_fc_norm_hidden(h)], dim=-1))

    def _q(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """x [n, H] (already input_layernorm'ed); cos/sin [n, rot]. -> q [Hq,n,D] (q_norm + rope), gate [n,Hq*D]."""
        a = self.layers[0].self_attn
        n, D = x.shape[0], self.head_dim
        q, gate = torch.chunk(a.q_proj(x).view(n, -1, 2 * D), 2, dim=-1)
        q = a.q_norm(q).transpose(0, 1)
        q = apply_rotary_pos_emb(q[None], q[None], cos[None], sin[None])[0][0]
        return q, gate.reshape(n, -1)

    def _kv(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """x [n, H] (already input_layernorm'ed) -> k (k_norm + rope), v: [Hkv,n,D]."""
        a = self.layers[0].self_attn
        n, D = x.shape[0], self.head_dim
        k = a.k_norm(a.k_proj(x).view(n, -1, D)).transpose(0, 1)
        v = a.v_proj(x).view(n, -1, D).transpose(0, 1)
        k = apply_rotary_pos_emb(k[None], k[None], cos[None], sin[None])[1][0]
        return k, v

    def _qkv(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        q, gate = self._q(x, cos, sin)
        k, v = self._kv(x, cos, sin)
        return q, gate, k, v

    def _kv_from_inputs(self, e, h, cos, sin):
        return self._kv(self.layers[0].input_layernorm(self._project(e, h)), cos, sin)

    def _finish(self, resid: torch.Tensor, attn: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """attn [Hq, n, D] -> normed MTP output [n, H] (vLLM returns norm(h + residual))."""
        layer = self.layers[0]
        n = resid.shape[0]
        o = attn.transpose(0, 1).reshape(n, -1) * torch.sigmoid(gate)
        x = resid + layer.self_attn.o_proj(o)
        x = x + layer.mlp(layer.post_attention_layernorm(x))
        return self.norm(x)

    def _attend(self, q, s_idx, k0, v0, dk, dv, q_chunk):
        outs = []
        for a in range(0, q.shape[1], q_chunk):
            b = min(a + q_chunk, q.shape[1])
            args = (q[:, a:b], s_idx[a:b], k0, v0, None if dk is None else dk[:, a:b],
                    None if dv is None else dv[:, a:b], self.scaling)
            outs.append(checkpoint(_chain_attention, *args, use_reentrant=False) if torch.is_grad_enabled()
                        else _chain_attention(*args))
        return torch.cat(outs, dim=1)

    def step0(self, e, h, cos, sin, rows, kv_chunk: int = 8192, q_chunk: int = 128):
        """Step-0 pass. e = E(x_{s+1}), h = h_s, cos/sin at s, all [L, ...]. The drafter's KV cache spans the whole
        context, so K/V are computed for every position (in checkpointed chunks: only K/V [Hkv,L,D] stay alive, about
        0.25 GB at 64 k tokens); queries, o_proj, MLP and norm run only at `rows` (the drafted chains), with exact
        causal attention over K/V[0..s]. Returns (o0 [len(rows),H], k0, v0 [Hkv,L,D])."""
        ks, vs = [], []
        for a in range(0, e.shape[0], kv_chunk):
            args = (e[a:a + kv_chunk], h[a:a + kv_chunk], cos[a:a + kv_chunk], sin[a:a + kv_chunk])
            k, v = (checkpoint(self._kv_from_inputs, *args, use_reentrant=False) if torch.is_grad_enabled()
                    else self._kv_from_inputs(*args))
            ks.append(k), vs.append(v)
        k, v = torch.cat(ks, dim=1), torch.cat(vs, dim=1)
        zr = self._project(e[rows], h[rows])
        q, gate = self._q(self.layers[0].input_layernorm(zr), cos[rows], sin[rows])
        attn = self._attend(q, rows, k, v, None, None, q_chunk)
        return self._finish(zr, attn, gate), k, v

    def step_j(self, e, h_prev, cos, sin, s_idx, k0, v0, diag_k, diag_v, q_chunk: int = 128):
        """Draft step j >= 1 for chains s_idx [n] (sorted). e = E(x_{s+1+j}), h_prev = o_{j-1}[s] ([n, H]);
        cos/sin at position s+j; diag_k/diag_v: lists of step-1..j-1 keys/values of these chains [Hkv, n, D].
        Returns (o_j [n,H], k_j, v_j [Hkv,n,D])."""
        z = self._project(e, h_prev)
        q, gate, k, v = self._qkv(self.layers[0].input_layernorm(z), cos, sin)
        dk = torch.stack(diag_k + [k], dim=2)  # [Hkv, n, j, D]
        dv = torch.stack(diag_v + [v], dim=2)
        attn = self._attend(q, s_idx, k0, v0, dk, dv, q_chunk)
        return self._finish(z, attn, gate), k, v


def _chain_attention(q, s_idx, k0, v0, dk, dv, scale):
    """q [Hq,n,D]; chain i attends k0[:, :s_idx[i]+1] and (if given) its own diag keys dk[:, i] ([Hkv,n,j,D]).
    Exact softmax, fp32 scores."""
    Hq, n, D = q.shape
    Hkv = k0.shape[0]
    g = Hq // Hkv
    m = int(s_idx.max()) + 1
    qg = q.view(Hkv, g, n, D)
    s_main = torch.einsum("hgnd,hmd->hgnm", qg, k0[:, :m]).float() * scale
    pos = torch.arange(m, device=q.device)
    s_main = s_main.masked_fill(pos[None, None, None, :] > s_idx[None, None, :, None], float("-inf"))
    if dk is None:
        p = torch.softmax(s_main, dim=-1).to(q.dtype)
        return torch.einsum("hgnm,hmd->hgnd", p, v0[:, :m]).reshape(Hq, n, D)
    s_diag = torch.einsum("hgnd,hnjd->hgnj", qg, dk).float() * scale
    p = torch.softmax(torch.cat([s_main, s_diag], dim=-1), dim=-1).to(q.dtype)
    out = torch.einsum("hgnm,hmd->hgnd", p[..., :m], v0[:, :m]) + torch.einsum("hgnj,hnjd->hgnd", p[..., m:], dv)
    return out.reshape(Hq, n, D)

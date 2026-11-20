"""Unrolled MTP chains over one rendered sequence: loss (training) and draft/target agreement (offline acceptance).

Chain s, step j drafts token x_{s+2+j}. Its target distribution is the target's softmax(logits(h_{s+1+j}) / T): the
target logits at the position whose next token is x_{s+2+j}, in the recorded context (teacher forcing).
loss = forward KL(p_T || q_T) + tv_weight * TV(p_T, q_T) per drafted position, averaged per step, weighted by
step_weights. 1 - TV = sum_v min(p_T, q_T) is exactly the acceptance of vLLM's probabilistic drafts.
Metrics per step: hit = argmax q == recorded token (the recorded tokens are the policy's T=0.7 samples, so with vLLM's
greedy drafts and rejection sampling P(accept) = p(argmax q) = E[hit]); p_acc = p_T(argmax q) (bf16 target, lower
variance).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from bodies import chain_starts


def _kl_rows(lm_w, w_sub, ids, o, ht, tgt_tok, T):
    """o, ht [n,H]; returns kl [n], tv [n] (fp32, differentiable), hit [n] bool, p_acc [n], p_max [n].
    w_sub/ids: the deploy draft-vocabulary subset (SOARM_DRAFT_VOCAB: the drafter scores only these lm_head rows), or
    None for the full vocabulary. With a subset q lives on the subset; KL is taken against p renormalized on it."""
    lt = F.linear(ht, lm_w).float() / T
    logp = torch.log_softmax(lt, -1)
    if ids is None:
        logq = torch.log_softmax(F.linear(o, lm_w).float() / T, -1)
        logp_on = logp
        kl = (logp.exp() * (logp - logq)).sum(-1)
    else:
        logq = torch.log_softmax(F.linear(o, w_sub).float() / T, -1)
        logp_on = logp[:, ids]
        logp_sub = torch.log_softmax(lt[:, ids], -1)
        kl = (logp_sub.exp() * (logp_sub - logq)).sum(-1)
    tv = 1.0 - torch.minimum(logp_on, logq).exp().sum(-1)  # draft ~ q, accept min(1, p/q): P(accept) = 1 - TV
    with torch.no_grad():
        d = logq.argmax(-1)
        p_acc = logp_on.gather(-1, d[:, None])[:, 0].exp()  # greedy draft: P(accept) = p(argmax q)
        hit = (d if ids is None else ids[d]) == tgt_tok
        p_max = logp_on.max(-1).values.exp()  # acceptance of a perfect argmax drafter (ceiling for greedy drafts)
    return kl, tv, hit, p_acc, p_max


def chain_pass(head, target, feats, input_ids, gen_spans, steps=3, T=0.7, row_chunk=256, step_weights=None,
               tv_weight=0.0, q_chunk=128, vocab_ids=None):
    """Returns (loss tensor or None if no chains, stats dict with per-step tensors over chain origins)."""
    h, emb, pos = feats["h"], feats["emb"], feats["pos"]
    dev = h.device
    ids = input_ids.to(dev)
    L = ids.shape[0]
    lm_w = target.lm_head.weight
    w_sub = None if vocab_ids is None else lm_w[vocab_ids]
    weights = step_weights or [1.0] * steps
    cos, sin = target.rope(pos[:, : L - 1], h.dtype)
    S0 = torch.tensor(chain_starts(gen_spans, 0), device=dev, dtype=torch.long)
    if S0.numel() == 0:
        return None, {}
    o_prev, k0, v0 = head.step0(emb[1:], h[:-1], cos, sin, S0, q_chunk=q_chunk)  # chain s: (E(x_{s+1}), h_s) at s
    prev_idx = S0
    diag_k, diag_v = [], []
    loss = None
    stats = {}
    for j in range(steps):
        S = torch.tensor(chain_starts(gen_spans, j), device=dev, dtype=torch.long)
        if S.numel() == 0:
            break
        rows = torch.searchsorted(prev_idx, S)
        if j == 0:
            o = o_prev
        else:
            p3 = (pos[0, S] + j)[None].expand(3, -1)
            c, s_ = target.rope(p3, h.dtype)
            o, kj, vj = head.step_j(
                target.embed_tokens(ids[S + 1 + j]), o_prev[rows], c, s_, S, k0, v0,
                [k[:, rows] for k in diag_k], [v[:, rows] for v in diag_v], q_chunk=q_chunk,
            )
            diag_k, diag_v = [k[:, rows] for k in diag_k] + [kj], [v[:, rows] for v in diag_v] + [vj]
        ht = h[S + 1 + j]
        tok = ids[S + 2 + j]
        kls, tvs, hits, paccs, pmaxs = [], [], [], [], []
        for a in range(0, S.numel(), row_chunk):
            b = min(a + row_chunk, S.numel())
            args = (lm_w, w_sub, vocab_ids, o[a:b], ht[a:b], tok[a:b], T)
            kl, tv, hit, pa, pm = (checkpoint(_kl_rows, *args, use_reentrant=False) if torch.is_grad_enabled()
                                   else _kl_rows(*args))
            kls.append(kl), tvs.append(tv), hits.append(hit), paccs.append(pa), pmaxs.append(pm)
        kl, tv = torch.cat(kls), torch.cat(tvs)
        step_loss = weights[j] * (kl.mean() + tv_weight * tv.mean())
        loss = step_loss if loss is None else loss + step_loss
        stats[j] = {"S": S, "kl": kl.detach(), "hit": torch.cat(hits), "p_acc": torch.cat(paccs),
                    "p_max": torch.cat(pmaxs), "p_prob": 1.0 - tv.detach()}
        o_prev, prev_idx = o, S
    return loss, stats


def simulate_acceptance(gen_spans, stats, steps) -> tuple[int, int]:
    """Replay vLLM decoding over each generation with k=steps greedy drafts and the recorded samples as the target's
    draws: per decode step emit 1 + number of leading draft hits. Returns (decode steps, emitted tokens)."""
    hit = {}
    for j, st in stats.items():
        for s, ok in zip(st["S"].tolist(), st["hit"].tolist()):
            hit[(j, s)] = ok
    n_steps = n_tok = 0
    for g, end in gen_spans:
        s = g - 1  # x_g sampled by the prefill step; first draft chain starts at h_{g-1}, x_g
        while s + 1 < end:  # x_{s+1} == <|im_end|> ends the request
            acc = 0
            while acc < steps and hit.get((acc, s), False):
                acc += 1
            n_steps += 1
            n_tok += 1 + acc
            s += 1 + acc
    return n_steps, n_tok


def expected_replay(gen_spans, stats, key, steps, reps=20, seed=0) -> tuple[int, int]:
    """Monte-Carlo version of simulate_acceptance: draft j of chain s is accepted with probability stats[j][key][s]
    (p_acc: this head; p_max: perfect argmax drafter). Positions are teacher-forced on the recorded tokens, so after an
    accepted draft that differs from the recorded token the next probabilities are approximate."""
    import random

    prob = {}
    for j, st in stats.items():
        if j < steps:
            for s, p in zip(st["S"].tolist(), st[key].tolist()):
                prob[(j, s)] = p
    rng = random.Random(seed)
    n_steps = n_tok = 0
    for _ in range(reps):
        for g, end in gen_spans:
            s = g - 1
            while s + 1 < end:
                acc = 0
                while acc < steps and rng.random() < prob.get((acc, s), 0.0):
                    acc += 1
                n_steps += 1
                n_tok += 1 + acc
                s += 1 + acc
    return n_steps, n_tok

"""Loss and acceptance stats of a DFlash / DSpark block drafter over one rendered sequence (teacher forcing).

Anchor a (the bonus token index) for every decode step inside a generation span (g, end): a in [g, end-1]. Mask slot
i = 1..N predicts x_{a+i} (valid while a+i <= end); its target distribution is softmax(target logits(h_{a+i-1}) / T).
Stats use the MTP layout of core.py so eval_accept.Meter / replays apply unchanged: draft step j = i-1 of chain
s = a-1. q is scored on the draft vocabulary subset (vocab_ids) like the served drafter (draft_vocab_size + d2t).
loss per row = decay_i * (TV(p_T, q_T) + kl_weight * KL(p_T|subset || q_T)) [+ conf_weight * BCE(confidence,
1 - TV) for DSpark], decay_i = exp(-(i-1)/gamma) (DeepSpec loss_decay_gamma), averaged over rows.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def valid_anchors(gen_spans) -> list[int]:
    out = []
    for g, end in gen_spans:
        out.extend(range(g, end))  # a in [g, end-1]
    return out


def _rows(lm_w, w_sub, ids, hq, bias, ht, tok, T, conf_logit):
    lt = F.linear(ht, lm_w).float() / T
    logp = torch.log_softmax(lt, -1)
    lq = F.linear(hq, w_sub if ids is not None else lm_w).float()
    if bias is not None:
        lq = lq + bias.float()
    logq = torch.log_softmax(lq / T, -1)
    if ids is None:
        logp_on, logp_sub = logp, logp
    else:
        logp_on = logp[:, ids]
        logp_sub = torch.log_softmax(lt[:, ids], -1)
    kl = (logp_sub.exp() * (logp_sub - logq)).sum(-1)
    tv = 1.0 - torch.minimum(logp_on, logq).exp().sum(-1)
    bce = None
    if conf_logit is not None:
        bce = F.binary_cross_entropy_with_logits(conf_logit, (1.0 - tv).detach().clamp(0, 1), reduction="none")
    with torch.no_grad():
        d = logq.argmax(-1)
        p_acc = logp_on.gather(-1, d[:, None])[:, 0].exp()
        hit = (d if ids is None else ids[d]) == tok
        p_max = logp_on.max(-1).values.exp()
    return kl, tv, bce, hit, p_acc, p_max


def block_pass(drafter, target, feats, input_ids, gen_spans, N, T=0.7, vocab_ids=None, max_anchors=None,
               gamma=4.0, kl_weight=0.1, conf_weight=1.0, row_chunk=256, a_chunk=16, generator=None):
    """Returns (loss or None, stats {j: {S, kl, hit, p_acc, p_max, p_prob[, conf]}})."""
    h, aux = feats["h"], feats["aux"]
    dev = h.device
    ids = input_ids.to(dev)
    L = ids.shape[0]
    A = valid_anchors(gen_spans)
    if not A:
        return None, {}
    anchors = torch.tensor(A, device=dev, dtype=torch.long)
    if max_anchors is not None and anchors.numel() > max_anchors:
        pick = torch.randperm(anchors.numel(), generator=generator)[:max_anchors].to(dev)
        anchors = anchors[pick].sort().values
    end_of = torch.empty(L, dtype=torch.long, device=dev)
    for g, end in gen_spans:
        end_of[g:end + 1] = end
    pos0 = torch.arange(L, device=dev)
    hid = drafter.blocks(aux, pos0, ids, anchors, N, target.embed_tokens, a_chunk=a_chunk)  # [nb, 1+N, H]
    lm_w = target.lm_head.weight
    w_sub = None if vocab_ids is None else lm_w[vocab_ids]
    loss_num = torch.zeros((), device=dev)
    loss_den = 0.0
    stats = {}
    for i in range(1, N + 1):
        ok = anchors + i <= end_of[anchors]
        if not bool(ok.any()):
            break
        an = anchors[ok]
        hq = hid[ok, i]
        tok = ids[an + i]
        ht = h[an + i - 1]
        bias = mk = None
        if drafter.dspark:
            mk, bias = drafter.markov(ids[an + i - 1])
        rows = {k: [] for k in ("kl", "tv", "bce", "hit", "p_acc", "p_max", "conf")}
        for r0 in range(0, an.numel(), row_chunk):
            r1 = min(r0 + row_chunk, an.numel())
            conf = drafter.confidence(hq[r0:r1], mk[r0:r1]) if drafter.dspark else None
            args = (lm_w, w_sub, vocab_ids, hq[r0:r1], None if bias is None else bias[r0:r1], ht[r0:r1],
                    tok[r0:r1], T, conf)
            kl, tv, bce, hit, pa, pm = (checkpoint(_rows, *args, use_reentrant=False) if torch.is_grad_enabled()
                                        else _rows(*args))
            for k, v in (("kl", kl), ("tv", tv), ("bce", bce), ("hit", hit), ("p_acc", pa), ("p_max", pm),
                         ("conf", None if conf is None else torch.sigmoid(conf.detach()))):
                if v is not None:
                    rows[k].append(v)
        kl, tv = torch.cat(rows["kl"]), torch.cat(rows["tv"])
        w = float(torch.exp(torch.tensor(-(i - 1) / gamma))) if gamma else 1.0
        term = tv + kl_weight * kl
        if rows["bce"]:
            term = term + conf_weight * torch.cat(rows["bce"])
        loss_num = loss_num + w * term.sum()
        loss_den += w * term.numel()
        st = {"S": an - 1, "kl": kl.detach(), "hit": torch.cat(rows["hit"]), "p_acc": torch.cat(rows["p_acc"]),
              "p_max": torch.cat(rows["p_max"]), "p_prob": 1.0 - tv.detach()}
        if rows["conf"]:
            st["conf"] = torch.cat(rows["conf"])
        stats[i - 1] = st
    return loss_num / max(loss_den, 1.0), stats

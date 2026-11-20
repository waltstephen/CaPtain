"""Continue-train the grafted Qwen3.5 MTP head on the policy's own outputs (self-distillation, one GPU).

    python train_mtp.py --data '$SOARM_ROOT/speed/runs/replay/rec_drafter_*.jsonl' \
        --heldout $SOARM_ROOT/speed/runs/replay/rec_sB.jsonl --out $SOARM_ROOT/speed/drafter/run1

Frozen bf16 target (sftv0-ck300) computes final-norm hidden states + merged input embeddings on the fly (no feature
cache). Trainable: the MTP head only (fc, one full-attention layer, 3 norms; fp32 master weights, bf16 autocast);
embeddings / lm_head are the target's, frozen. Loss: forward KL(p_T || q_T) + tv_weight * TV(p_T, q_T) at the deploy
temperature (default 0.7) to the target's next-next-token distribution, at every generated assistant token, for draft
steps 0..k-1 unrolled exactly like vLLM's k-step MTP proposer (core.py); q is scored on the deploy draft-vocabulary
subset (--draft-vocab, SOARM_DRAFT_VOCAB). 1 - TV is exactly the acceptance of vLLM's probabilistic drafts.
Memory: the target prefill is chunked (KV cache only); the head keeps only its K/V over the context and runs
queries/MLP/logits at the drafted rows, so ~64 k-token bodies fit a 24 GB GPU. Writes <out>/mtp_ep*.safetensors
(vLLM key layout mtp.*) per epoch and <out>/mtp.safetensors = last epoch.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import time

import torch

from bodies import Renderer, select_bodies
from core import chain_pass
from eval_accept import DEPLOY_DRAFT_VOCAB, evaluate, load_head, load_vocab_ids
from target import Target


def span_groups(spans: list[tuple[int, int]], max_rows: int) -> list[list[tuple[int, int]]]:
    """Pack consecutive generation spans into groups of <= max_rows tokens (a longer single span stays alone)."""
    groups, cur, n = [], [], 0
    for a, b in spans:
        if cur and n + b - a + 1 > max_rows:
            groups.append(cur)
            cur, n = [], 0
        cur.append((a, b))
        n += b - a + 1
    return groups + [cur] if cur else groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.path.expandvars("$SOARM_ROOT/sft_runs/sftv0-ck300/final"))
    ap.add_argument("--init", default=os.path.expandvars("$SOARM_ROOT/speed/weights/qwen35-4b-mtp/mtp.safetensors"))
    ap.add_argument("--data", required=True, help="glob of recorded request JSONL files (training seeds only)")
    ap.add_argument("--heldout", help="JSONL for the acceptance proxy (rec_sB eval seeds; never trained on)")
    ap.add_argument("--heldout-bodies", type=int, default=40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="cover", choices=["cover", "exact"])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5, help="Adam moves every weight ~lr per update: keep it small "
                    "for a pretrained head (1e-4 without warmup regressed hits in the CPU smoke)")
    ap.add_argument("--warmup", type=int, default=20, help="optimizer updates")
    ap.add_argument("--accum", type=int, default=4, help="bodies per optimizer update")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=3, help="unrolled draft steps (= vLLM num_speculative_tokens)")
    ap.add_argument("--step-weights", default="0.51,0.31,0.18", help="FastMTP beta=0.6 decay (speculators default)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--chunk", type=int, default=8192, help="target prefill chunk (tokens)")
    ap.add_argument("--max-len", type=int, default=0, help="skip longer bodies (0 = no limit; step 0 keeps only K/V "
                    "of the context, so 64 k-token bodies fit 24 GB)")
    ap.add_argument("--q-chunk", type=int, default=128, help="drafted rows per exact-attention chunk")
    ap.add_argument("--max-rows", type=int, default=6000,
                    help="generated tokens per backward; bodies with more are split into span groups (same loss)")
    ap.add_argument("--tv-weight", type=float, default=1.0,
                    help="loss = KL + w * TV; 1 - TV = acceptance of vLLM probabilistic drafts")
    ap.add_argument("--draft-vocab", default=DEPLOY_DRAFT_VOCAB,
                    help="SOARM_DRAFT_VOCAB subset the deployed drafter scores (train/eval q on it), or 'none'")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-before", action="store_true")
    ap.add_argument("--max-bodies", type=int, default=None, help="debug: cap training bodies")
    ap.add_argument("--body-fn", choices=["none", "cpu-smoke"], default="none")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    log_fh = open(os.path.join(a.out, "log.jsonl"), "a")

    def log(msg, **kv):
        print(msg, flush=True)
        if kv:
            log_fh.write(json.dumps({"t": time.time(), **kv}) + "\n")
            log_fh.flush()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    cuda = a.device.startswith("cuda")
    target = Target(a.model_dir, a.device, torch.bfloat16 if cuda else torch.float32)
    head = load_head(target, a.init)
    renderer = Renderer(target.processor)
    body_fn = None
    if a.body_fn == "cpu-smoke":
        from eval_accept import shrink_body

        body_fn = shrink_body
    weights = [float(w) for w in a.step_weights.split(",")][: a.steps]
    vocab_ids = load_vocab_ids(a.draft_vocab, target.device)

    files = sorted(glob.glob(a.data))
    if not files:
        raise SystemExit(f"no files match {a.data}")
    if a.heldout and os.path.abspath(a.heldout) in map(os.path.abspath, files):
        raise SystemExit("held-out file is in the training set")
    samples = [s for f in files for s in select_bodies(f, a.mode)][: a.max_bodies]
    held = select_bodies(a.heldout, a.mode)[: a.heldout_bodies] if a.heldout else []
    log(f"train bodies={len(samples)} from {len(files)} files; heldout bodies={len(held)}")

    def run_eval(tag):
        if held:
            res = evaluate(target, head, renderer, held, a.steps, a.temperature, a.chunk, body_fn, log=lambda *_: None,
                           vocab_ids=vocab_ids, q_chunk=a.q_chunk)
            log(f"[{tag}] heldout {json.dumps(res)}", kind="eval", tag=tag, **res)
            return res

    if a.eval_before:
        run_eval("before")

    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    total = a.epochs * math.ceil(len(samples) / a.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda u: min(1.0, (u + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(u, total) / max(total, 1)))
    )
    update = 0
    for epoch in range(a.epochs):
        random.shuffle(samples)
        head.train()
        t0, tok, micro = time.time(), 0, 0
        opt.zero_grad(set_to_none=True)
        for i, sample in enumerate(samples):
            body = sample.body()
            enc = renderer.encode(sample, body_fn(body) if body_fn else body)
            if not enc.gen_spans:
                continue
            if a.max_len and enc.input_ids.numel() > a.max_len:   # 24 GB: the head's activations over the body
                log(f"skip body {i + 1}: L={enc.input_ids.numel()} > --max-len {a.max_len}")
                continue
            groups = span_groups(enc.gen_spans, a.max_rows)
            total_rows = sum(b - a_ + 1 for a_, b in enc.gen_spans)
            ntok, stats, loss = 0, None, None
            try:
                feats = target.features(enc.input_ids, enc.mm, chunk=a.chunk)
                for spans in groups:  # rows-proportional activations: bound them per backward
                    with torch.autocast(target.device.type, dtype=torch.bfloat16, enabled=cuda):
                        lg, st = chain_pass(head, target, feats, enc.input_ids, spans, steps=a.steps,
                                            T=a.temperature, step_weights=weights, tv_weight=a.tv_weight,
                                            q_chunk=a.q_chunk, vocab_ids=vocab_ids)
                    if lg is None:
                        continue
                    rows = sum(b - a_ + 1 for a_, b in spans)
                    (lg * (rows / total_rows) / a.accum).backward()
                    loss, stats, ntok = lg, st, ntok + st[0]["S"].numel()
            except torch.OutOfMemoryError:  # drop this body and the partial update (grads may be half-accumulated)
                feats = loss = stats = None
                opt.zero_grad(set_to_none=True)
                micro = 0
                torch.cuda.empty_cache()
                log(f"OOM, skipped body {i + 1}: L={enc.input_ids.numel()}", kind="oom", L=enc.input_ids.numel())
                continue
            del feats
            if stats is None:
                continue
            micro += 1
            tok += ntok
            last = i == len(samples) - 1
            if micro % a.accum == 0 or last:
                gn = torch.nn.utils.clip_grad_norm_(head.parameters(), a.clip).item()
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                update += 1
                hits = {f"hit{j}": stats[j]["hit"].float().mean().item() for j in stats}
                hits.update({f"pprob{j}": stats[j]["p_prob"].float().mean().item() for j in stats})
                mem = torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0
                log(f"ep {epoch} upd {update}/{total} body {i + 1}/{len(samples)} L={enc.input_ids.numel()} "
                    f"loss={loss.item():.4f} gn={gn:.2f} {hits} tok/s={tok / (time.time() - t0):.0f} "
                    f"maxmem={mem:.1f}G", kind="train", epoch=epoch, update=update, loss=loss.item(), gn=gn,
                    L=enc.input_ids.numel(), lr=sched.get_last_lr()[0], maxmem_gb=mem, **hits)
        path = os.path.join(a.out, f"mtp_ep{epoch}.safetensors")
        head.save_mtp(path, metadata={"init": a.init, "epoch": str(epoch), "data": a.data})
        if epoch == a.epochs - 1:
            final = os.path.join(a.out, "mtp.safetensors")
            if os.path.exists(final):
                os.remove(final)
            os.link(path, final)
        log(f"saved epoch {epoch} ({time.time() - t0:.0f}s, {tok} trained positions)")
        run_eval(f"epoch{epoch}")


if __name__ == "__main__":
    main()

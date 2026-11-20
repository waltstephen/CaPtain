"""Domain fine-tuning of a DFlash drafter (z-lab/Qwen3.5-4B-DFlash) or a DSpark drafter (same backbone + Markov head
+ confidence head) on the policy's own outputs, and their offline acceptance (same held-out protocol as MTP).

    python train_block.py --kind dflash --init weights/Qwen3.5-4B-DFlash --data 'runs/replay/rec_drafter_[12].jsonl' \
        --heldout runs/replay/rec_sB.jsonl --out drafter/dflash1 --block 16
    python train_block.py --kind dspark --block 8 ...          # 7 drafts (DSpark "block 7", fill-in 1+N layout)
    python train_block.py --eval-only --init <draft dir> --kind dflash --block 16 --heldout ...

Target features on the fly (bf16 sftv0-ck300): residual stream after the draft's target_layer_ids (aux) + final
hidden (for p_T). The target's decoder layers / vision tower are offloaded to CPU while the drafter trains on the
body (--offload), so the 634 M-parameter drafter (8-bit AdamW) fits 24 GB next to it. Loss / stats: block_core.py.
Writes vLLM-loadable draft dirs (<out>/draft_ep*/): config.json + model.safetensors with the draft-vocab lm_head rows
and d2t (see block_drafter.save_draft_dir).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time

import torch

from block_core import block_pass
from block_drafter import BlockDrafter, count_params, save_draft_dir
from bodies import Renderer, select_bodies
from eval_accept import DEPLOY_DRAFT_VOCAB, Meter, load_vocab_ids
from target import Target

OVERLAY = os.path.expandvars("$SOARM_ROOT/speed/drafter/pylib")  # bitsandbytes symlink (8-bit AdamW)


@torch.no_grad()
def evaluate(target, drafter, renderer, samples, N, T, vocab_ids, offload, chunk=8192, log=print,
             body_fn=None) -> dict:
    meter = Meter(N)
    drafter.eval()
    t0 = time.time()
    for i, s in enumerate(samples):
        enc = renderer.encode(s, body_fn(s.body()) if body_fn else None)
        if not enc.gen_spans:
            continue
        feats = target.features(enc.input_ids, enc.mm, chunk=chunk, aux_layers=drafter.layer_ids)
        if offload:
            target.offload(True)
        with torch.autocast(target.device.type, dtype=torch.bfloat16, enabled=target.device.type == "cuda"):
            _, stats = block_pass(drafter, target, feats, enc.input_ids, enc.gen_spans, N, T, vocab_ids, a_chunk=8)
        del feats
        if offload:
            target.offload(False)
        meter.add(stats, enc.gen_spans)
        log(f"[eval {i + 1}/{len(samples)}] L={enc.input_ids.numel()} tok/step={meter.summary()['tok_per_step']:.3f} "
            f"{time.time() - t0:.0f}s")
    return meter.summary()


def short(res: dict, N: int) -> str:
    ks = [k for k in (3, 4, 5, 7, N) if k <= N]
    rp = {k: {m: round(res["replay"][k][m], 3) for m in ("hit", "p_acc", "p_prob")} for k in dict.fromkeys(ks)}
    steps = [(round(res[f"step{j}"]["p_acc"], 3), round(res[f"step{j}"]["p_prob"], 3)) for j in range(min(N, 8))]
    return f"replay {rp} steps(greedy,prob) {steps}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["dflash", "dspark"], required=True)
    ap.add_argument("--init", required=True, help="draft dir (config.json + model.safetensors)")
    ap.add_argument("--model-dir", default=os.path.expandvars("$SOARM_ROOT/sft_runs/sftv0-ck300/final"))
    ap.add_argument("--data", help="glob of recorded request JSONL files (training seeds only)")
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--heldout-bodies", type=int, default=40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=int, default=16, help="query block = 1 + N drafts (num_speculative_tokens N)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3, help="Markov / confidence heads (fresh)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--max-anchors", type=int, default=512, help="anchors (decode steps) sampled per body")
    ap.add_argument("--a-chunk", type=int, default=8, help="anchors per exact-attention chunk")
    ap.add_argument("--gamma", type=float, default=4.0)
    ap.add_argument("--kl-weight", type=float, default=0.1)
    ap.add_argument("--conf-weight", type=float, default=1.0)
    ap.add_argument("--markov-rank", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--draft-vocab", default=DEPLOY_DRAFT_VOCAB)
    ap.add_argument("--chunk", type=int, default=4096, help="target prefill chunk (4096: lower peak next to the drafter)")
    ap.add_argument("--no-offload", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cpu-smoke", action="store_true", help="shrunk bodies (eval_accept.shrink_body), CPU debug")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--max-bodies", type=int, default=None)
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
    N = a.block - 1
    cuda = a.device.startswith("cuda")

    def sync():
        if cuda:
            torch.cuda.synchronize()
    offload = cuda and not a.no_offload
    target = Target(a.model_dir, a.device, torch.bfloat16 if cuda else torch.float32)
    body_fn = None
    if a.cpu_smoke:
        from eval_accept import shrink_body

        body_fn = shrink_body
    vocab_ids = load_vocab_ids(a.draft_vocab, target.device)
    cfg_path = os.path.join(a.init, "config.json")
    base_cfg = json.load(open(cfg_path))
    drafter = BlockDrafter.from_dir(a.init, dspark=a.kind == "dspark", markov_rank=a.markov_rank)
    if a.kind == "dspark" and "markov_rank" not in base_cfg:
        drafter.init_dspark_heads(vocab_ids.numel() if vocab_ids is not None else target.lm_head.weight.shape[0])
    drafter = drafter.to(target.device, torch.float32)
    log(f"{a.kind}: {count_params(drafter):.0f} M params, N={N}, draft vocab "
        f"{'full' if vocab_ids is None else vocab_ids.numel()}")
    renderer = Renderer(target.processor)
    held = select_bodies(a.heldout, "cover")[: a.heldout_bodies]

    def run_eval(tag):
        res = evaluate(target, drafter, renderer, held, N, a.temperature, vocab_ids, offload, a.chunk,
                       log=lambda *_: None, body_fn=body_fn)
        log(f"[{tag}] heldout {short(res, N)}", kind="eval", tag=tag, **res)
        return res

    if a.eval_only:
        res = run_eval("eval")
        with open(os.path.join(a.out, "eval.json"), "w") as fh:
            json.dump(res, fh, indent=1)
        return

    files = sorted(glob.glob(a.data))
    if os.path.abspath(a.heldout) in map(os.path.abspath, files):
        raise SystemExit("held-out file is in the training set")
    samples = [s for f in files for s in select_bodies(f, "cover")][: a.max_bodies]
    log(f"train bodies={len(samples)} from {len(files)} files; heldout bodies={len(held)}")

    heads = [p for n, p in drafter.named_parameters() if n.startswith(("markov_head.", "confidence_head."))]
    body = [p for n, p in drafter.named_parameters() if not n.startswith(("markov_head.", "confidence_head."))]
    groups = [{"params": body, "lr": a.lr}] + ([{"params": heads, "lr": a.head_lr}] if heads else [])
    try:
        if not cuda:
            raise ImportError("8-bit AdamW is CUDA-only")
        sys.path.insert(0, OVERLAY)
        import bitsandbytes as bnb

        opt = bnb.optim.AdamW8bit(groups, betas=(0.9, 0.95), weight_decay=0.0)
        log("optimizer: bitsandbytes AdamW8bit")
    except ImportError:
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)
        log("optimizer: torch AdamW (fp32 states)")
    total = a.epochs * math.ceil(len(samples) / a.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda u: min(1.0, (u + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(u, total) / max(total, 1))))
    gen = torch.Generator().manual_seed(a.seed)
    update = 0
    for epoch in range(a.epochs):
        random.shuffle(samples)
        drafter.train()
        t0, micro, rows = time.time(), 0, 0
        opt.zero_grad(set_to_none=True)
        for i, s in enumerate(samples):
            enc = renderer.encode(s, body_fn(s.body()) if body_fn else None)
            if not enc.gen_spans:
                continue
            backward_started = False
            try:
                tm = [time.time()]
                feats = target.features(enc.input_ids, enc.mm, chunk=a.chunk, aux_layers=drafter.layer_ids)
                sync(), tm.append(time.time())
                if offload:
                    target.offload(True)
                sync(), tm.append(time.time())
                with torch.autocast(target.device.type, dtype=torch.bfloat16, enabled=target.device.type == "cuda"):
                    loss, stats = block_pass(drafter, target, feats, enc.input_ids, enc.gen_spans, N, a.temperature,
                                             vocab_ids, a.max_anchors, a.gamma, a.kl_weight, a.conf_weight,
                                             a_chunk=a.a_chunk, generator=gen)
                if loss is not None:
                    backward_started = True
                    (loss / a.accum).backward()
                sync(), tm.append(time.time())
                times = [tm[k + 1] - tm[k] for k in range(3)]
            except torch.OutOfMemoryError:  # grads of earlier bodies stay unless backward had started
                feats = loss = stats = None
                if backward_started:
                    opt.zero_grad(set_to_none=True)
                    micro = 0
                if cuda:
                    torch.cuda.empty_cache()
                log(f"OOM, skipped body {i + 1}: L={enc.input_ids.numel()}", kind="oom", L=enc.input_ids.numel())
                if offload:
                    target.offload(False)
                continue
            feats = None
            if offload:
                target.offload(False)
            sync()
            times.append(time.time() - tm[-1])
            if loss is None:
                continue
            micro += 1
            rows += stats[0]["S"].numel()
            if micro % a.accum == 0 or i == len(samples) - 1:
                gn = torch.nn.utils.clip_grad_norm_(drafter.parameters(), a.clip).item()
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                update += 1
                kv = {f"pprob{j}": stats[j]["p_prob"].float().mean().item() for j in list(stats)[:N:2]}
                mem = torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0
                log(f"ep {epoch} upd {update}/{total} body {i + 1}/{len(samples)} L={enc.input_ids.numel()} "
                    f"loss={loss.item():.4f} gn={gn:.2f} {kv} anchors/s={rows / (time.time() - t0):.0f} "
                    f"maxmem={mem:.1f}G s(feat,off,step,on)={[round(x, 1) for x in times]}", kind="train",
                    epoch=epoch, update=update, loss=loss.item(), gn=gn, maxmem_gb=mem, times=times, **kv)
        out_dir = os.path.join(a.out, f"draft_ep{epoch}")
        save_draft_dir(drafter, out_dir, base_cfg, vocab_ids, target.lm_head.weight, a.block,
                       {"init": a.init, "epoch": epoch, "kind": a.kind})
        log(f"saved {out_dir} ({time.time() - t0:.0f}s)")
        run_eval(f"epoch{epoch}")


if __name__ == "__main__":
    main()

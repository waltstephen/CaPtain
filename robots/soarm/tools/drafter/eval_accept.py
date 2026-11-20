"""Offline acceptance proxy of an MTP head on recorded policy requests (held-out: rec_sB = eval seeds).

    python eval_accept.py --mtp <mtp.safetensors> --data runs/replay/rec_sB.jsonl [--max-bodies 40] [--out x.json]

Per draft step j (k = --steps): hit_j = P(argmax draft == recorded sampled token | earlier drafts hit) (the recorded
tokens are the policy's T=0.7 samples, so this estimates vLLM's per-position acceptance with greedy drafts),
p_acc_j = mean bf16-target p_T(argmax draft), p_max_j = mean max_v p_T(v) (a perfect argmax drafter's acceptance: the
ceiling for greedy drafts), kl_j = KL(p_T || q_T). tok_per_step = replay of vLLM decoding with k greedy drafts
(1 + leading hits per decode step) - comparable to vLLM's mean acceptance length (live stack B: 2.68).
replay[k] = the same replay for every k <= --steps: "hit" (realized), "p_acc" / "p_max" / "p_prob" (Monte-Carlo with
those acceptance probabilities; "p_max" is the tok/step ceiling of any greedy drafter; p_prob_j = sum_v min(p_T, q_T)
is vLLM draft_sample_method="probabilistic": draft ~ q_T, accept with min(1, p/q)).
--cpu-smoke shrinks bodies (short system prompt, <= 1 image, first turns) for a CPU run of the full pipeline.
"""
from __future__ import annotations
import os

import argparse
import copy
import json
import time

import torch

from bodies import Renderer, select_bodies
from core import chain_pass, expected_replay, simulate_acceptance
from mtp_head import MTPHead
from target import Target


def shrink_body(body: dict, sys_chars: int = 1500, n_assistant: int = 2, max_images: int = 1) -> dict:
    body = copy.deepcopy(body)
    msgs, out, n_img, n_asst = body["messages"], [], 0, 0
    for m in msgs:
        if m["role"] == "system" and isinstance(m["content"], str):
            m["content"] = m["content"][:sys_chars]
        if isinstance(m.get("content"), list):
            keep = []
            for p in m["content"]:
                if p.get("type") == "image_url":
                    if n_img >= max_images:
                        continue
                    n_img += 1
                keep.append(p)
            m["content"] = keep or [{"type": "text", "text": ""}]
        if m["role"] == "assistant":
            if n_asst >= n_assistant:
                break
            n_asst += 1
        out.append(m)
    body["messages"] = out
    return body


class Meter:
    def __init__(self, steps: int):
        self.steps = steps
        self.n = [0] * steps
        self.hit = [0.0] * steps
        self.cum = [0.0] * steps  # chains whose drafts 0..j all hit
        self.p_acc = [0.0] * steps
        self.p_max = [0.0] * steps
        self.p_prob = [0.0] * steps
        self.kl = [0.0] * steps
        self.replay = {k: {m: [0, 0] for m in ("hit", "p_acc", "p_max", "p_prob")} for k in range(1, steps + 1)}

    def add(self, stats: dict, gen_spans) -> None:
        all_hit = None
        for j in range(self.steps):
            if j not in stats:
                break
            st = stats[j]
            self.n[j] += st["S"].numel()
            self.hit[j] += st["hit"].float().sum().item()
            self.p_acc[j] += st["p_acc"].float().sum().item()
            self.p_max[j] += st["p_max"].float().sum().item()
            self.p_prob[j] += st["p_prob"].float().sum().item()
            self.kl[j] += st["kl"].float().sum().item()
            h = dict(zip(st["S"].tolist(), st["hit"].tolist()))
            all_hit = h if all_hit is None else {s: all_hit.get(s, False) and ok for s, ok in h.items()}
            self.cum[j] += sum(all_hit.values())
        for k, r in self.replay.items():
            for m, (a, b) in (("hit", simulate_acceptance(gen_spans, stats, k)),
                              *((m, expected_replay(gen_spans, stats, m, k)) for m in ("p_acc", "p_max", "p_prob"))):
                r[m][0] += a
                r[m][1] += b

    def summary(self) -> dict:
        top = self.replay[self.steps]["hit"]
        out = {"tok_per_step": top[1] / max(top[0], 1), "decode_steps": top[0],
               "replay": {k: {m: b / max(a, 1) for m, (a, b) in r.items()} for k, r in self.replay.items()}}
        for j in range(self.steps):
            n = max(self.n[j], 1)
            prev = self.cum[j - 1] if j else n
            out[f"step{j}"] = {"n": self.n[j], "hit": self.hit[j] / n, "p_acc": self.p_acc[j] / n,
                               "p_max": self.p_max[j] / n, "p_prob": self.p_prob[j] / n, "kl": self.kl[j] / n,
                               "hit_given_prev": self.cum[j] / max(prev, 1)}
        return out


DEPLOY_DRAFT_VOCAB = os.path.expandvars("$SOARM_ROOT/speed/models/draft_vocab16384.safetensors")


def load_vocab_ids(path: str | None, device) -> torch.Tensor | None:
    """ids of the SOARM_DRAFT_VOCAB file (deploy drafter vocabulary subset); None/'none' = full vocabulary."""
    if not path or path == "none":
        return None
    from safetensors import safe_open

    with safe_open(path, "pt") as f:
        return f.get_tensor("ids").to(device=device, dtype=torch.long)


@torch.no_grad()
def evaluate(target, head, renderer, samples, steps=3, T=0.7, chunk=8192, body_fn=None, log=print,
             vocab_ids=None, q_chunk=128) -> dict:
    meter = Meter(steps)
    head.eval()
    t0 = time.time()
    for i, sample in enumerate(samples):
        body = sample.body()
        enc = renderer.encode(sample, body_fn(body) if body_fn else body)
        if not enc.gen_spans:
            continue
        feats = target.features(enc.input_ids, enc.mm, chunk=chunk)
        with torch.autocast(target.device.type, dtype=torch.bfloat16, enabled=target.device.type == "cuda"):
            _, stats = chain_pass(head, target, feats, enc.input_ids, enc.gen_spans, steps=steps, T=T,
                                  vocab_ids=vocab_ids, q_chunk=q_chunk)
        meter.add(stats, enc.gen_spans)
        log(f"[eval {i + 1}/{len(samples)}] L={enc.input_ids.numel()} gen={sum(b - a + 1 for a, b in enc.gen_spans)} "
            f"tok/step={meter.summary()['tok_per_step']:.3f} {time.time() - t0:.0f}s")
    return meter.summary()


def load_head(target: Target, mtp_path: str) -> MTPHead:
    head = MTPHead(target.text_config)
    head.load_mtp(mtp_path)
    return head.to(target.device, torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.path.expandvars("$SOARM_ROOT/sft_runs/sftv0-ck300/final"))
    ap.add_argument("--mtp", default=os.path.expandvars("$SOARM_ROOT/speed/weights/qwen35-4b-mtp/mtp.safetensors"))
    ap.add_argument("--data", required=True)
    ap.add_argument("--mode", default="cover", choices=["cover", "exact"])
    ap.add_argument("--max-bodies", type=int, default=None)
    ap.add_argument("--limit-lines", type=int, default=None)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cpu-smoke", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--draft-vocab", default=DEPLOY_DRAFT_VOCAB, help="SOARM_DRAFT_VOCAB subset file, or 'none'")
    a = ap.parse_args()
    target = Target(a.model_dir, a.device, torch.bfloat16 if a.device.startswith("cuda") else torch.float32)
    head = load_head(target, a.mtp)
    samples = select_bodies(a.data, a.mode, a.limit_lines)[: a.max_bodies]
    res = evaluate(target, head, Renderer(target.processor), samples, a.steps, a.temperature, a.chunk,
                   shrink_body if a.cpu_smoke else None, vocab_ids=load_vocab_ids(a.draft_vocab, target.device))
    res.update(mtp=a.mtp, data=a.data, bodies=len(samples), mode=a.mode, cpu_smoke=a.cpu_smoke,
               draft_vocab=a.draft_vocab)
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()

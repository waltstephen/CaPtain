"""Build the SOARM_DRAFT_VOCAB file for sitecustomize.py: the K lm_head rows the MTP drafter scores.

    $VLLM_VENV/bin/python tools/vllm_patches/build_draft_vocab.py <model dir> <out.safetensors> \
        --train rec_a.jsonl ... [--heldout rec_b.jsonl ...] [--k 32768]

Subset = the tokenizer's added/special tokens, every token the policy emitted in the --train recorded request bodies
(assistant text, reasoning with its <think> tags, tool calls in their qwen3_xml rendering), then the lowest BPE ids
(most frequent merges) up to K. --heldout reports the share of the policy's output
tokens outside the subset (only those can never be drafted). Rows are copied from the bf16 lm_head of <model dir>.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer


def xml_tool_call(fn: dict) -> str:
    """The qwen3_xml rendering the policy actually emits for a tool call (chat template of Qwen3.5)."""
    args = fn.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"input": args}
    body = "".join(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n" for k, v in args.items())
    return f"<tool_call>\n<function={fn.get('name')}>\n{body}</function>\n</tool_call><|im_end|>"


def output_tokens(tok, paths: list[str]) -> collections.Counter:
    cnt: collections.Counter = collections.Counter()
    seen: set[str] = set()
    for path in paths:
        with open(path) as fh:
            for line in fh:
                for m in json.loads(line)["messages"][-6:]:   # each turn appears in the next bodies: dedupe
                    if m.get("role") != "assistant":
                        continue
                    parts = [m["content"] if isinstance(m.get("content"), str) else "",
                             f"<think>\n{m.get('reasoning_content') or ''}\n</think>\n\n"]
                    parts += [xml_tool_call(t.get("function", t)) for t in m.get("tool_calls") or []]
                    key = json.dumps(parts)
                    if key not in seen:
                        seen.add(key)
                        for p in parts:
                            if p:
                                cnt.update(tok.encode(p))
    return cnt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--heldout", nargs="*", default=[])
    ap.add_argument("--k", type=int, default=32768)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    weight = None
    for fn in sorted(glob.glob(f"{a.model}/*.safetensors")):
        with safe_open(fn, "pt") as fh:
            name = next((k for k in fh.keys() if k.endswith("lm_head.weight")), None)
            if name:
                weight = fh.get_tensor(name)
                break
    if weight is None:
        raise SystemExit(f"no bf16 lm_head.weight in {a.model}")
    seen = sorted(tok.added_tokens_decoder) + [t for t, _ in output_tokens(tok, a.train).most_common()]
    ids = list(dict.fromkeys(seen + list(range(weight.shape[0]))))[:a.k]
    keep = set(ids)
    report = {"k": a.k, "train_distinct_tokens": len(seen)}
    if a.heldout:
        held = output_tokens(tok, a.heldout)
        total = sum(held.values())
        report.update(heldout_tokens=total, heldout_outside=sum(c for t, c in held.items() if t not in keep))
    idx = torch.tensor(sorted(ids), dtype=torch.int64)
    save_file({"weight": weight[idx].contiguous().to(torch.bfloat16), "ids": idx}, a.out)
    print(json.dumps(report))


if __name__ == "__main__":
    main()

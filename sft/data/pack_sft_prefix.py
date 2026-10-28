#!/usr/bin/env python3
"""Prefix-pack per-request SFT samples: within one episode, consecutive requests whose rendered chat texts (same tools,
processor chat template) and image lists are exact prefixes of the next one become ONE row - the longest view - with
``meta.supervise_turns`` listing the assistant-turn ordinals of every member's own target (train/train_sft.py supervises
exactly those spans). The loss is the same token sum as training the members separately; the forward cost is the longest
member's instead of the sum (about 3.5x fewer processed tokens on our teacher data: compactions and the image
cap break chains).

    python data/pack_sft_prefix.py --in DATA/sft.jsonl --out OUT/sft.jsonl --tokenizer MODEL_DIR
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def chains(rows: list[dict], render) -> list[list[dict]]:
    """Greedy chains over request order: member i+1 extends member i iff text_i is a prefix of text_{i+1}, the tool
    schemas are equal and the images of i are a prefix of the images of i+1."""
    by = collections.defaultdict(list)
    for r in rows:
        by[r["meta"]["episode"]].append(r)
    out: list[list[dict]] = []
    for ep in sorted(by):
        rs = sorted(by[ep], key=lambda r: r["meta"]["request_index"])
        texts = [render(r) for r in rs]
        cur = [0]
        for i in range(1, len(rs)):
            a, b = rs[cur[-1]], rs[i]
            if (texts[i].startswith(texts[cur[-1]]) and a.get("tools") == b.get("tools")
                    and list(b.get("images") or [])[:len(a.get("images") or [])] == list(a.get("images") or [])):
                cur.append(i)
            else:
                out.append([rs[j] for j in cur])
                cur = [i]
        out.append([rs[j] for j in cur])
    return out


def packed_row(chain: list[dict]) -> dict:
    last = chain[-1]
    turns = [sum(1 for m in r["messages"] if m.get("role") == "assistant") - 1 for r in chain]
    if len(chain) == 1:
        return last
    meta = dict(last["meta"], supervise_turns=turns, packed_ids=[r["id"] for r in chain], packed_n=len(chain),
                packed_member_tokens=sum(int(r["meta"].get("total_tokens") or 0) for r in chain))
    return dict(last, id=last["id"] + f"#pack{len(chain)}", meta=meta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    args = ap.parse_args()
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(str(args.tokenizer))
    render = lambda r: proc.apply_chat_template(r["messages"], tools=r.get("tools") or None, tokenize=False, add_generation_prompt=False)  # noqa: E731
    rows = [json.loads(l) for l in open(args.inp)]
    cs = chains(rows, render)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_tok_in = sum(int(r["meta"].get("total_tokens") or 0) for r in rows)
    n_tok_out = 0
    with open(args.out, "w") as f:
        for c in cs:
            r = packed_row(c)
            n_tok_out += int(r["meta"].get("total_tokens") or 0)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"rows_in": len(rows), "rows_out": len(cs), "targets": sum(len(c) for c in cs), "tokens_in": n_tok_in,
             "tokens_out": n_tok_out, "speedup": round(n_tok_in / max(1, n_tok_out), 2),
             "chain_len_hist": dict(collections.Counter(len(c) for c in cs))}
    (args.out.parent / "pack.stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()

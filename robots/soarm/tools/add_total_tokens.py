"""Set meta.total_tokens (text tokens of the rendered chat + image tokens) on SFT rows; mix_order/bucket_order need it.

    venv-sft/bin/python tools/add_total_tokens.py --in packed.jsonl --out packed_tok.jsonl --model <model dir> --image-root <dir>
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

P = None
ROOT = None
_SIZES: dict = {}


def init(model: str, root: str) -> None:
    global P, ROOT
    from transformers import AutoProcessor
    P = AutoProcessor.from_pretrained(model)
    ROOT = Path(root)


def image_tokens(path: str) -> int:
    from PIL import Image
    img = Image.open(path if Path(path).is_absolute() else ROOT / path).convert("RGB")
    if img.size not in _SIZES:
        grid = P.image_processor(images=[img], return_tensors="pt")["image_grid_thw"][0]
        merge = getattr(P.image_processor, "merge_size", 2)
        _SIZES[img.size] = int(grid.prod()) // (merge * merge)
    return _SIZES[img.size]


def count(line: str) -> str:
    r = json.loads(line)
    text = P.apply_chat_template(r["messages"], tools=r.get("tools") or None, tokenize=False, add_generation_prompt=False)
    n = len(P.tokenizer(text, add_special_tokens=False)["input_ids"]) + sum(image_tokens(p) for p in r.get("images") or [])
    r.setdefault("meta", {})["total_tokens"] = n
    return json.dumps(r, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    lines = [l for l in Path(a.inp).read_text().splitlines() if l.strip()]
    with ProcessPoolExecutor(a.workers, initializer=init, initargs=(a.model, a.image_root)) as ex:
        out = list(ex.map(count, lines, chunksize=8))
    Path(a.out).write_text("\n".join(out) + "\n")
    toks = sorted(json.loads(l)["meta"]["total_tokens"] for l in out)
    print(json.dumps({"rows": len(toks), "tokens": sum(toks), "max": toks[-1], "median": toks[len(toks) // 2]}))


if __name__ == "__main__":
    main()

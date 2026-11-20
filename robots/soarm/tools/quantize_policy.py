"""Quantize a CaPtain-4B (Qwen3.5) checkpoint for vLLM serving on a 4090 (soarm-speed, eval/deploy only).

    PYTHONPATH=<llmcompressor overlay> $VLLM_VENV/bin/python tools/quantize_policy.py <model dir> <out dir> \
        --scheme W4A16 [--lm-head bf16|W4A16|W8A16] [--method rtn|gptq --calib rows.jsonl --samples 64]

Language-model Linear layers only. Kept in bf16: the vision tower (model.visual.*), the gated-delta-net in_proj_a /
in_proj_b (output dim = 32 heads, too small for the Marlin 4-bit kernels) and, by default, lm_head (248320 x 2560,
1.27 GB in bf16 = ~40 % of the bytes a W4 decode step reads; --lm-head quantizes it). Output: compressed-tensors
(pack-quantized for W4A16), loaded by vLLM without --quantization.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

IGNORE = ["re:.*visual.*", "re:.*linear_attn\\.in_proj_a$", "re:.*linear_attn\\.in_proj_b$"]


def calib_dataset(path: Path, tokenizer, n: int, max_len: int):
    """Text-only calibration rows from SFT rows (messages -> chat template, images dropped)."""
    from datasets import Dataset
    texts = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        msgs = []
        for m in row.get("messages", []):
            content = m.get("content")
            if isinstance(content, list):
                content = "\n".join(c.get("text", "") for c in content if c.get("type") == "text")
            msgs.append({"role": m["role"], "content": content or ""})
        texts.append(tokenizer.apply_chat_template(msgs, tokenize=False))
        if len(texts) >= n:
            break
    enc = tokenizer(texts, max_length=max_len, truncation=True, padding=False)
    return Dataset.from_dict({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--scheme", default="W4A16")
    ap.add_argument("--method", choices=("rtn", "gptq"), default="rtn")
    ap.add_argument("--calib", default=None)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--lm-head", default="bf16", help="bf16 | a preset scheme for lm_head (W4A16, W8A16)")
    a = ap.parse_args()
    import torch
    from llmcompressor import oneshot
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    src, out = Path(a.model), Path(a.out)
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForImageTextToText.from_pretrained(src, dtype=torch.bfloat16,
                                                        device_map="cuda" if torch.cuda.is_available() else "cpu")
    ignore = IGNORE + (["lm_head"] if a.lm_head == "bf16" else [])
    # Linear (class) for the decoder, an exact-name regex for lm_head: vLLM matches lm_head by its own prefix
    # (language_model.lm_head), so a bare "lm_head" target is silently ignored at load time
    scheme = a.scheme if a.lm_head in ("bf16", a.scheme) else {a.scheme: ["Linear"], a.lm_head: ["re:.*lm_head$"]}
    if a.method == "rtn":
        from llmcompressor.modifiers.quantization import QuantizationModifier
        oneshot(model=model, recipe=QuantizationModifier(targets="Linear", scheme=scheme, ignore=ignore))
    else:
        from llmcompressor.modifiers.quantization import GPTQModifier
        ds = calib_dataset(Path(a.calib), tok, a.samples, a.max_len)
        oneshot(model=model, dataset=ds, recipe=GPTQModifier(targets="Linear", scheme=scheme, ignore=ignore),
                max_seq_length=a.max_len, num_calibration_samples=len(ds))
    model.save_pretrained(out, save_compressed=True)
    tok.save_pretrained(out)
    for name in ("chat_template.jinja", "processor_config.json", "preprocessor_config.json",
                 "video_preprocessor_config.json", "generation_config.json"):
        if (src / name).exists():
            shutil.copy(src / name, out / name)
    print("QUANT_DONE", out, sum(f.stat().st_size for f in out.iterdir()) / 2**30, "GiB")


if __name__ == "__main__":
    main()

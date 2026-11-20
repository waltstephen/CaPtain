"""Extract a PEFT LoRA adapter = truncated SVD of (finetuned - base) on the language-model Linear layers (soarm-speed).

    python tools/extract_lora.py <base dir> <finetuned dir> <out dir> --rank 64

Used to serve a full-finetuned student as base (4-bit, shared) + adapter (bf16) in vLLM (--enable-lora). Deltas outside
the Linear layers (embeddings, lm_head, norms, conv1d) are dropped; the printed energy fraction says how much of the
Linear delta the rank keeps.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TARGET = re.compile(r"^model\.language_model\.layers\.\d+\.(mlp\.(gate|up|down)_proj|self_attn\.[qkvo]_proj|"
                    r"linear_attn\.(in_proj_qkv|in_proj_z|out_proj))\.weight$")


def main() -> None:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("tuned")
    ap.add_argument("out")
    ap.add_argument("--rank", type=int, default=64)
    a = ap.parse_args()

    def opener(d: Path):
        idx = d / "model.safetensors.index.json"
        files = ({k: d / f for k, f in json.loads(idx.read_text())["weight_map"].items()} if idx.exists()
                 else {k: d / "model.safetensors" for k in safe_open(d / "model.safetensors", "pt").keys()})
        handles = {f: safe_open(f, "pt") for f in set(files.values())}
        return lambda k: handles[files[k]].get_tensor(k), files

    base, _ = opener(Path(a.base))
    tuned, keys = opener(Path(a.tuned))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tensors, modules = {}, set()
    kept = total = 0.0
    for k in sorted(keys):
        if not TARGET.match(k):
            continue
        d = (tuned(k).to(dev, torch.float32) - base(k).to(dev, torch.float32))
        u, s, v = torch.linalg.svd(d, full_matrices=False)
        r = a.rank
        kept += float((s[:r] ** 2).sum())
        total += float((s ** 2).sum())
        sq = s[:r].sqrt()
        name = k.removesuffix(".weight")
        tensors[f"base_model.model.{name}.lora_A.weight"] = (sq[:, None] * v[:r]).to(torch.bfloat16).cpu().contiguous()
        tensors[f"base_model.model.{name}.lora_B.weight"] = (u[:, :r] * sq[None]).to(torch.bfloat16).cpu().contiguous()
        modules.add(name.split(".")[-1])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, out / "adapter_model.safetensors")
    (out / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": a.rank, "lora_alpha": a.rank, "lora_dropout": 0.0,
        "bias": "none", "target_modules": sorted(modules), "base_model_name_or_path": str(a.base),
        "fan_in_fan_out": False, "use_rslora": False, "modules_to_save": None}, indent=1))
    print(json.dumps({"rank": a.rank, "delta_energy_kept": round(kept / total, 4), "modules": sorted(modules),
                      "tensors": len(tensors)}))


if __name__ == "__main__":
    main()

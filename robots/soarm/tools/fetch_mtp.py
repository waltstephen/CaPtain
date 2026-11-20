"""Fetch only the mtp.* tensors of a HF safetensors checkpoint with HTTP range reads (soarm-speed lever 2).

    python tools/fetch_mtp.py Qwen/Qwen3.5-4B <out.safetensors> [--endpoint https://hf-mirror.com] [--prefix mtp.]

The CaPtain-4B students have no MTP weights (config mtp_num_hidden_layers=1); the public Qwen3.5-4B has the head.
Reads each shard's safetensors header, then downloads just the byte ranges of the wanted tensors (~0.2 GB, not 9 GB).
"""
from __future__ import annotations

import argparse
import json
import struct
import urllib.request

DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}


def get(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    for attempt in range(20):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception:  # noqa: BLE001
            if attempt == 19:
                raise
    raise RuntimeError("unreachable")


def main() -> None:
    import torch
    from safetensors.torch import save_file
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("out")
    ap.add_argument("--endpoint", default="https://hf-mirror.com")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--prefix", default="mtp.")
    a = ap.parse_args()
    base = f"{a.endpoint}/{a.repo}/resolve/{a.revision}"
    index = json.loads(urllib.request.urlopen(f"{base}/model.safetensors.index.json", timeout=120).read())["weight_map"]
    wanted = sorted(k for k in index if k.startswith(a.prefix))
    tensors = {}
    for shard in sorted({index[k] for k in wanted}):
        url = f"{base}/{shard}"
        n = struct.unpack("<Q", get(url, 0, 7))[0]
        header = json.loads(get(url, 8, 8 + n - 1))
        for k in (k for k in wanted if index[k] == shard):
            meta = header[k]
            lo, hi = meta["data_offsets"]
            raw = get(url, 8 + n + lo, 8 + n + hi - 1)
            t = torch.frombuffer(bytearray(raw), dtype=getattr(torch, DTYPES[meta["dtype"]])).reshape(meta["shape"])
            tensors[k] = t.clone()
            print(k, meta["dtype"], meta["shape"], flush=True)
    save_file(tensors, a.out, metadata={"format": "pt", "source": f"{a.repo}@{a.revision}"})
    print("MTP_DONE", len(tensors), sum(t.numel() for t in tensors.values()) / 1e6, "M params")


if __name__ == "__main__":
    main()

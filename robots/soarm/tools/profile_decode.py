"""Profile single-stream decode of the running policy server (soarm-speed lever 1).

    # server started with POLICY_ARGS="--profiler-config {\"profiler\":\"torch\",\"torch_profiler_dir\":\"/abs/dir\",
    #                                   \"torch_profiler_with_stack\":false}"
    python tools/profile_decode.py http://127.0.0.1:8700 requests.jsonl --index 30 --tokens 128 --dir /abs/dir
    python tools/profile_decode.py ... --index 30 --prefill   # append prefill of request 30 on request 29's cache

Warms the prefix with request --index (T=0, 1 token), then profiles a greedy --tokens generation of the same request
and summarizes the newest trace: wall per output token, GPU busy fraction, GPU time by kernel class
(gemm/marlin, gated-delta-net/fla, attention, lm_head/logits, sampling, norm/elementwise, other), and the top kernels.
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import re
import time
import urllib.request

CLASSES = [("marlin/gemm", r"marlin|gemm|gemv|cutlass|sm89|ampere|cublas|matmul"),
           ("gdn/fla", r"fla|chunk|gated_delta|fused_recurrent|delta_rule|conv1d|causal_conv"),
           ("attention", r"flash|attn|attention|paged"),
           ("sampling/logits", r"sample|softmax|topk|top_k|argmax|logit|gumbel|exponential"),
           ("norm/elementwise", r"norm|silu|act|elementwise|vectorized|add|mul|copy|cat|index|fill|reduce"),
           ]


def post(url: str, body: dict | None = None) -> bytes:
    req = urllib.request.Request(url, data=json.dumps(body or {}).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600).read()


def classify(name: str) -> str:
    low = name.lower()
    for cls, pat in CLASSES:
        if re.search(pat, low):
            return cls
    return "other"


def summarize(path: str, out_tokens: int, wall_s: float) -> dict:
    with gzip.open(path) if path.endswith(".gz") else open(path) as fh:
        trace = json.load(fh)
    ev = [e for e in trace.get("traceEvents", []) if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy",
                                                                                            "gpu_memset")]
    if not ev:
        return {"trace": path, "error": "no kernel events"}
    t0 = min(e["ts"] for e in ev)
    t1 = max(e["ts"] + e["dur"] for e in ev)
    busy = sum(e["dur"] for e in ev)
    by = collections.Counter()
    top = collections.Counter()
    for e in ev:
        by[classify(e["name"])] += e["dur"]
        top[e["name"][:90]] += e["dur"]
    span_ms = (t1 - t0) / 1e3
    return {"trace": os.path.basename(path), "out_tokens": out_tokens, "wall_s": round(wall_s, 3),
            "ms_per_token_wall": round(wall_s * 1e3 / max(out_tokens, 1), 2),
            "gpu_span_ms": round(span_ms, 1), "gpu_busy_frac": round(busy / (t1 - t0), 3),
            "gpu_ms_per_token": round(busy / 1e3 / max(out_tokens, 1), 3),
            "by_class_ms_per_token": {k: round(v / 1e3 / max(out_tokens, 1), 3) for k, v in by.most_common()},
            "top_kernels_ms_per_token": [(k, round(v / 1e3 / max(out_tokens, 1), 3)) for k, v in top.most_common(12)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("server")
    ap.add_argument("record")
    ap.add_argument("--index", type=int, default=30)
    ap.add_argument("--tokens", type=int, default=None, help="output tokens to profile (default 128; 1 with --prefill)")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--prefill", action="store_true",
                    help="profile the APPEND prefill of request --index (warm request --index - 1), then --tokens tokens")
    a = ap.parse_args()
    lines = open(a.record).read().splitlines()
    body = json.loads(lines[a.index])
    warm = json.loads(lines[a.index - 1]) if a.prefill else dict(body)
    for b in (body, warm):
        if a.model:
            b["model"] = a.model
        b.update(stream=False, temperature=0.0, top_p=1.0, seed=0)
    if a.tokens is None:
        a.tokens = 1 if a.prefill else 128
    post(a.server + "/v1/chat/completions", dict(warm, max_tokens=1))          # warm prefix + compile
    before = set(glob.glob(os.path.join(a.dir, "**", "*.json*"), recursive=True))
    post(a.server + "/start_profile")
    t = time.perf_counter()
    res = json.loads(post(a.server + "/v1/chat/completions", dict(body, max_tokens=a.tokens, ignore_eos=True)))
    wall = time.perf_counter() - t
    post(a.server + "/stop_profile")
    n = res["usage"]["completion_tokens"]
    for _ in range(120):
        new = [p for p in glob.glob(os.path.join(a.dir, "**", "*.json*"), recursive=True) if p not in before]
        if new:
            time.sleep(5)
            break
        time.sleep(2)
    gpu = [p for p in new if "rank" in os.path.basename(p) or "worker" in os.path.basename(p).lower()] or new
    print(json.dumps(summarize(max(gpu, key=os.path.getsize), n, wall), indent=1))


if __name__ == "__main__":
    main()

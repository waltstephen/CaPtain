"""Replay recorded policy requests (tools/policy_proxy.py --record) against a server config (soarm-speed levers 1-3).

    python tools/policy_replay.py requests.jsonl http://127.0.0.1:8700/v1 [--limit 120] [--tag w4]

Sends the recorded chat/completions bodies in order (same prefixes -> same prefix-cache pattern as the episode),
streamed, one at a time. Reports s/request, TTFT (prefill), decode tok/s, output tokens, and the speculative-decoding
acceptance (vLLM /metrics deltas). Sampling settings are the recorded ones (T=1), so outputs differ run to run;
compare rates, not texts. --t0 N: greedy (temperature 0, max_tokens N) and --texts FILE saves every output, for
token-identity checks of lossless levers (spec decoding x prefix caching x gated-delta-net state) between configs.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from speed_eval import scrape  # noqa: E402

SPEC = {"drafts": "vllm:spec_decode_num_drafts_total", "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
        "accepted": "vllm:spec_decode_num_accepted_tokens_total"}


def spec_metrics(base: str) -> dict:
    url = base.rstrip("/").removesuffix("/v1") + "/metrics"
    try:
        text = urllib.request.urlopen(url, timeout=10).read().decode()
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for k, name in SPEC.items():
        out[k] = sum(float(line.split()[-1]) for line in text.splitlines() if line.startswith(name + "{")
                     or line.startswith(name + " "))
    return out


def run(url: str, body: dict, model: str | None, t0_tokens: int = 0, override: dict | None = None) -> dict:
    body = dict(body, stream=True, stream_options={"include_usage": True})
    for key, value in (override or {}).items():   # a config's sampling knobs; None removes the field
        if value is None:
            body.pop(key, None)
        else:
            body[key] = value
    if t0_tokens:
        body.update(temperature=0.0, top_p=1.0, max_tokens=t0_tokens, seed=0)
    if model:
        body["model"] = model
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    usage = {}
    text = []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("choices"):
                delta = d["choices"][0].get("delta") or {}
                piece = "".join(str(delta.get(k) or "") for k in ("reasoning_content", "reasoning", "content"))
                for tc in delta.get("tool_calls") or []:   # chunking-invariant: names + concatenated arguments
                    fn = tc.get("function") or {}
                    piece += (f"<call {fn['name']}>" if fn.get("name") else "") + str(fn.get("arguments") or "")
                text.append(piece)
                if first is None and piece:
                    first = time.perf_counter()
            if d.get("usage"):
                usage = d["usage"]
    end = time.perf_counter()
    first = first or end
    n = int(usage.get("completion_tokens") or 0)
    return {"total_s": end - t0, "ttft_s": first - t0, "decode_s": end - first, "out_tokens": n,
            "prompt_tokens": int(usage.get("prompt_tokens") or 0), "text": "".join(text)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("record")
    ap.add_argument("url")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=None, help="override the recorded model name")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--t0", type=int, default=0, help="greedy with this max_tokens (identity checks)")
    ap.add_argument("--texts", default=None, help="save outputs (jsonl) here")
    ap.add_argument("--override", default="{}", help="JSON of request fields to set (null = remove), e.g. a config's "
                                                     "temperature / max_tokens / thinking_token_budget")
    a = ap.parse_args()
    override = json.loads(a.override)
    bodies = [json.loads(line) for line in Path(a.record).read_text().splitlines() if line.strip()]
    if a.limit:
        bodies = bodies[:a.limit]
    s0 = spec_metrics(a.url)
    m0 = scrape(a.url)
    rows = [run(a.url, b, a.model, a.t0, override) for b in bodies]
    if a.texts:
        Path(a.texts).write_text("".join(json.dumps(r["text"]) + "\n" for r in rows))
    s1 = spec_metrics(a.url)
    m1 = scrape(a.url)
    out_tok = sum(r["out_tokens"] for r in rows)
    dec = sum(r["decode_s"] for r in rows)
    res = {"tag": a.tag, "url": a.url, "requests": len(rows), "override": override,
           "model_s": round(sum(r["total_s"] for r in rows), 2),
           "s_per_request": round(sum(r["total_s"] for r in rows) / len(rows), 3),
           "ttft_med_s": round(st.median(r["ttft_s"] for r in rows), 3),
           "decode_tok_s": round((out_tok - len(rows)) / dec, 1) if dec else None,
           "out_tok_per_request": round(out_tok / len(rows), 1),
           "prefix_hit": round((m1.get("prefix_hits", 0) - m0.get("prefix_hits", 0))
                               / max(m1.get("prefix_queries", 0) - m0.get("prefix_queries", 0), 1), 3)}
    if s1.get("drafts", 0) > s0.get("drafts", 0):
        drafts = s1["drafts"] - s0["drafts"]
        res["accepted_per_step"] = round(1 + (s1["accepted"] - s0["accepted"]) / drafts, 3)
        res["draft_acceptance"] = round((s1["accepted"] - s0["accepted"]) / max(s1["draft_tokens"] - s0["draft_tokens"], 1), 3)
    print(json.dumps(res), flush=True)
    if a.out:
        with open(a.out, "a") as fh:
            fh.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()

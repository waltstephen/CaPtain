"""Single-stream policy server microbenchmark (soarm-speed): cold prefill, warm (prefix-cached) prefill, decode rate.

    python tools/policy_bench.py http://127.0.0.1:8700/v1 [--images 2] [--prompt-tokens 30000] [--gen 512]

The prompt = a real episode image (agentview 512x512) repeated --images times + filler text; each trial first sends
the prompt cold (random salt at the start, so no prefix hit), then the same prompt again + a short suffix (warm:
what a ReAct step looks like with prefix caching). Decode: --gen tokens with ignore_eos, streamed; rate after TTFT.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import random
import statistics
import time
import urllib.request


def image_b64(seed: int = 0) -> str:
    """512x512 PNG; a different seed = different pixels (vLLM caches encoder outputs by image hash)."""
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(seed)
    arr = (rng.random((512, 512, 3)) * 40 + np.linspace(80, 200, 512)[None, :, None]).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


TEMPERATURE = 1.0


def stream(url: str, model: str, messages: list, gen: int) -> tuple[float, float, int]:
    body = {"model": model, "messages": messages, "max_tokens": gen, "temperature": TEMPERATURE, "stream": True,
            "ignore_eos": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": True}}
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    n = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("choices") and first is None:
                delta = d["choices"][0].get("delta", {})
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    first = time.perf_counter()
            if d.get("usage"):
                n = d["usage"]["completion_tokens"]
    end = time.perf_counter()
    first = first or end
    return first - t0, end - first, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--model", default="captain-4b")
    ap.add_argument("--images", type=int, default=2)
    ap.add_argument("--prompt-tokens", type=int, default=30000)
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--append-images", type=int, default=0,
                    help="K2: also time one ReAct-like append on the cached prompt: N new images (fresh pixels) ...")
    ap.add_argument("--append-text", type=int, default=0, help="... plus ~N new text tokens; also each part alone")
    ap.add_argument("--distinct-images", action="store_true",
                    help="the prompt's --images are all different (an episode history) instead of one image repeated")
    ap.add_argument("--temperature", type=float, default=1.0)
    a = ap.parse_args()
    global TEMPERATURE
    TEMPERATURE = a.temperature
    img = image_b64()
    words = "the arm moves the red cube to the mat and measures its height again ".split()

    def text(n: int) -> str:
        return " ".join(words[i % len(words)] for i in range(int(n * 0.95)))

    def prompt_tokens(msgs: list) -> int:
        body = {"model": a.model, "messages": msgs, "chat_template_kwargs": {"enable_thinking": True}}
        req = urllib.request.Request(a.url.removesuffix("/v1") + "/tokenize", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["count"]

    def new_images(seed: int) -> list:  # fresh pixels: no encoder-cache or prefix hit
        return [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64(seed * 100 + k + 1)}"}}
                for k in range(a.append_images)]

    variants = {}
    if a.append_images:
        variants["append_img"] = new_images
    if a.append_text:
        variants["append_txt"] = lambda s: [{"type": "text", "text": f"{s} " + text(a.append_text)}]
    if a.append_images and a.append_text:
        variants["append_both"] = lambda s: new_images(s) + [{"type": "text", "text": f"{s} " + text(a.append_text)}]
    rows = []
    for t in range(a.trials):
        salt = f"trial {random.random()} "
        content = ([{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64(10**7 + k)}"}}
                    for k in range(a.images)] if a.distinct_images else
                   [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}] * a.images)
        msgs = [{"role": "user", "content": content + [{"type": "text", "text": salt + text(a.prompt_tokens)}]}]
        cold_ttft, _, _ = stream(a.url, a.model, msgs, 1)
        base = msgs + [{"role": "assistant", "content": "ok"}]
        warm = base + [{"role": "user", "content": "next step " * 300}]
        warm_ttft, dec_s, n = stream(a.url, a.model, warm, a.gen)
        row = {"cold_prefill_s": cold_ttft, "warm_prefill_s": warm_ttft, "decode_tok_s": (n - 1) / dec_s}
        if variants:
            n0 = prompt_tokens(base)
            for name, make in variants.items():
                m = base + [{"role": "user", "content": make(random.randrange(10**6))}]
                ttft, _, _ = stream(a.url, a.model, m, 1)
                new = prompt_tokens(m) - n0
                row.update({f"{name}_s": ttft, f"{name}_tokens": new, f"{name}_tok_s": new / ttft})
        rows.append(row)
    med = {k: round(statistics.median(r[k] for r in rows), 3) for k in rows[0]}
    print(json.dumps({"url": a.url, "prompt_tokens": a.prompt_tokens, "images": a.images, "gen": a.gen, **med}))


if __name__ == "__main__":
    main()

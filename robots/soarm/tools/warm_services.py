"""Warm the policy prefix cache and the perception services before the first episode (soarm-speed J2/J4).

    python tools/warm_services.py --body <recorded request-1 body .jsonl> [--policy URL] [--sam3 URL] [--molmo URL]
                                  [--depth URL]

J2: the policy's opening prompt (system prompt + tool schema + robot intro, ~18 k tokens) is the same for every task;
sending one recorded first request with max_tokens=1 puts it in vLLM's prefix cache, so the first real request only
prefills the task-specific tail (measured cold: 1.8 s for request 1 on a fresh server).
J4: one SAM3 /segment + /segment_point, one Molmo point and one MoGe depth call on a real twin frame, so the first
calls of an episode do not pay kernel / allocator warm-up (first mark_point 1.24 s vs ~0.5 s later).
The warm-up calls never touch an episode; results are discarded.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import time
import urllib.request


def post(url: str, body: dict, timeout: float = 300) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def first_image(body: dict) -> str | None:
    for m in body.get("messages", []):
        if isinstance(m.get("content"), list):
            for p in m["content"]:
                if isinstance(p, dict) and p.get("type") == "image_url":
                    return p["image_url"]["url"].split(",", 1)[1]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True, help="jsonl whose first line is a recorded first request")
    ap.add_argument("--policy", default="http://127.0.0.1:8700/v1")
    ap.add_argument("--sam3", default="http://127.0.0.1:8114")
    ap.add_argument("--molmo", default="http://127.0.0.1:8122/v1")
    ap.add_argument("--depth", default="http://127.0.0.1:8130")
    a = ap.parse_args()
    body = json.loads(open(a.body).readline())
    out = {}
    t = time.perf_counter()
    post(a.policy + "/chat/completions", dict(body, max_tokens=1, stream=False))
    out["policy_prefix_s"] = round(time.perf_counter() - t, 2)
    img = first_image(body)
    if img:
        from PIL import Image
        import numpy as np
        rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(img))).convert("RGB"))
        h, w = rgb.shape[:2]
        for name, fn in (
                ("sam3_segment_s", lambda: post(a.sam3 + "/segment", {"image_base64": img, "text_prompt": "block"})),
                ("sam3_point_s", lambda: post(a.sam3 + "/segment_point",
                                              {"image_base64": img, "point_coords": [w / 2, h / 2]})),
                ("molmo_point_s", lambda: post(a.molmo + "/chat/completions", {
                    "model": "allenai/Molmo2-8B", "max_tokens": 64, "temperature": 0.0, "messages": [{"role": "user",
                    "content": [{"type": "text", "text": "Point at the block"},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}]}]})),
                ("depth_s", lambda: post(a.depth + "/estimate_depth", {"rgb_png_base64": img, "model": "moge2"}))):
            t = time.perf_counter()
            try:
                fn()
                out[name] = round(time.perf_counter() - t, 2)
            except Exception as exc:  # noqa: BLE001  (a missing service is reported, not fatal)
                out[name] = f"{type(exc).__name__}: {exc}"[:120]
    print(json.dumps(out))


if __name__ == "__main__":
    main()

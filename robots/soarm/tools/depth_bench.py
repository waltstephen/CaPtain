"""MoGe depth-service micro-benchmark on recorded SO-101 frames (soarm-speed hop-by-hop: observe -> MoGe).

    python tools/depth_bench.py <episode dir> --url http://127.0.0.1:8130 [--ref-url http://127.0.0.1:8131] [--n 20]

Per request (client side, the same code path as adapters/hardware/mono_depth.py MonoDepthSource._request):
  encode_ms  RGB -> PNG -> base64 (+ JSON) on the client
  http_ms    POST until the full response body is read
  server_ms  the service's own elapsed_s (PNG decode + MoGe inference + float32 npy encode)
  decode_ms  JSON + base64 + np.load + float64 cast on the client
With --ref-url (a second server, e.g. at the release resolution level), each depth is compared with the reference after
a median scale fit (mono_depth.py fits one scale per frame to the support plane): median and p90 relative error.
Frames: <episode>/workspace/images/obs_*_{head,wrist}.png, intrinsics from the episode's first observe call.
"""
from __future__ import annotations

import argparse
import ast
import base64
import io
import json
import statistics
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image


def intrinsics(ep: Path) -> dict[str, list[float]]:
    for line in (ep / "calls.jsonl").read_text().splitlines():
        call = json.loads(line)
        if call.get("name") == "observe" and call.get("success"):
            value = call["value"]
            value = ast.literal_eval(value) if isinstance(value, str) else value
            return {c["camera_name"]: c["intrinsics_3x3"] for c in value["cameras"]}
    raise SystemExit("no observe call with camera parameters")


def request(url: str, rgb: np.ndarray, intr: list[float]) -> tuple[np.ndarray, dict[str, float]]:
    t0 = time.perf_counter()
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    data = json.dumps({"rgb_png_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
                       "intrinsics_3x3": [float(v) for v in intr]}).encode()
    t1 = time.perf_counter()
    req = urllib.request.Request(f"{url}/estimate_depth", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
    t2 = time.perf_counter()
    body = json.loads(raw)
    depth = np.load(io.BytesIO(base64.b64decode(body["depth_npy_base64"])), allow_pickle=False).astype(np.float64)
    t3 = time.perf_counter()
    return depth, {"encode_ms": 1000 * (t1 - t0), "http_ms": 1000 * (t2 - t1), "server_ms": 1000 * body["elapsed_s"],
                   "decode_ms": 1000 * (t3 - t2), "total_ms": 1000 * (t3 - t0), "request_kb": len(data) / 1024,
                   "response_kb": len(raw) / 1024}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("episode")
    ap.add_argument("--url", required=True)
    ap.add_argument("--ref-url", default=None)
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    ep = Path(a.episode)
    intr = intrinsics(ep)
    frames = []
    for path in sorted((ep / "workspace" / "images").glob("obs_*_*.png"))[: a.n]:
        role = path.stem.rsplit("_", 1)[-1]
        cam = "agentview" if role == "head" else "robot0_eye_in_hand"
        frames.append((path.name, np.asarray(Image.open(path).convert("RGB")), intr[cam]))
    request(a.url, frames[0][1], frames[0][2])                     # warm-up
    rows, errs = [], []
    for name, rgb, k in frames:
        depth, t = request(a.url, rgb, k)
        rows.append(t)
        if a.ref_url:
            ref, _ = request(a.ref_url, rgb, k)
            ok = np.isfinite(depth) & np.isfinite(ref) & (depth > 0) & (ref > 0)
            scale = np.median(ref[ok] / depth[ok])
            rel = np.abs(depth[ok] * scale - ref[ok]) / ref[ok]
            errs.append((float(np.median(rel)), float(np.percentile(rel, 90))))
    out = {k: round(statistics.median(r[k] for r in rows), 1) for k in rows[0]}
    out["frames"] = len(rows)
    if errs:
        out["rel_err_vs_ref_median"] = round(statistics.median(e[0] for e in errs), 4)
        out["rel_err_vs_ref_p90"] = round(statistics.median(e[1] for e in errs), 4)
    print(json.dumps(out))


if __name__ == "__main__":
    main()

"""Summaries for reports: per-family success of batches, tokens and wall per episode, gateway reliability.

    python tools/summarize.py runs/pilot_terra runs/pilot_ds [--usage logs/gateway_usage_terra.jsonl ...] [--json out.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path


def load(batch_dir: Path) -> list[dict]:
    path = batch_dir / "batch.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    latest = {}
    for r in rows:
        latest[(r["family"], r["seed"])] = r
    return list(latest.values())


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def batch_summary(rows: list[dict]) -> dict:
    fam = collections.defaultdict(lambda: [0, 0, 0])
    for r in rows:
        f = fam[r["family"]]
        f[0] += 1
        f[1] += bool(r.get("success"))
        f[2] += bool(r.get("success")) and bool(r.get("finished")) and not r.get("false_finish")
    ok = [r for r in rows if "error" not in r]
    term = collections.Counter(r.get("termination") or r.get("error", "?")[:40] for r in rows)
    return {
        "episodes": len(rows), "success": sum(bool(r.get("success")) for r in rows),
        "strict_success": sum(v[2] for v in fam.values()),
        "false_finish": sum(bool(r.get("false_finish")) for r in rows),
        "per_family": {k: {"n": v[0], "success": v[1], "strict": v[2]} for k, v in sorted(fam.items())},
        "terminations": dict(term),
        "requests_median": statistics.median([r.get("requests", 0) for r in ok]) if ok else None,
        "wall_s_median": statistics.median([r["wall_s"] for r in ok]) if ok else None,
        "wall_s_p90": pct([r["wall_s"] for r in ok], 0.9),
        "input_tokens_median": statistics.median([r.get("input_tokens", 0) for r in ok]) if ok else None,
        "cached_share": (sum(r.get("cached_tokens", 0) for r in ok) / max(1, sum(r.get("input_tokens", 0) for r in ok))),
        "output_tokens_median": statistics.median([r.get("output_tokens", 0) for r in ok]) if ok else None,
    }


def usage_summary(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ok = [r for r in rows if r.get("event") == "ok"]
    bad = [r for r in rows if r.get("event") != "ok"]
    lat = [r["latency_s"] for r in ok]
    return {"requests_ok": len(ok), "errors": len(bad), "error_kinds": dict(collections.Counter(
        str(r.get("code") or r.get("error", ""))[:60] for r in bad)), "retried_ok": sum(r.get("attempt", 0) > 0 for r in ok),
        "latency_p50_s": pct(lat, 0.5), "latency_p90_s": pct(lat, 0.9), "latency_max_s": max(lat) if lat else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("batches", nargs="+")
    ap.add_argument("--usage", nargs="*", default=[])
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out = {Path(b).name: batch_summary(load(Path(b))) for b in a.batches}
    out.update({Path(u).name: usage_summary(Path(u)) for u in a.usage})
    text = json.dumps(out, indent=1)
    if a.json:
        Path(a.json).write_text(text)
    print(text)


if __name__ == "__main__":
    main()

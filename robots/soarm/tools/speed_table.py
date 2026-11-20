"""Summarize tools/speed_eval.py runs: python tools/speed_table.py <run dir> [...]  (medians per run, success counts)."""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path


def med(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(st.median(vals), 1) if vals else None


def main() -> None:
    cols = ("run n succ strict s/req req/ep wall_s real_wall_s p90_real model_s decode_s prefill_s out_tok/req "
            "perc_s depth_s motion_sim_s prefix_hit").split()
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for d in sys.argv[1:]:
        rows = [json.loads(line) for line in (Path(d) / "speed.jsonl").read_text().splitlines() if line.strip()]
        if not rows:
            continue
        succ = sum(bool(r.get("success")) for r in rows)
        strict = sum(bool(r.get("success")) and bool(r.get("finished")) for r in rows)
        req = sum(r.get("requests", 0) for r in rows)
        sreq = round(sum(r.get("model_s", 0) for r in rows) / max(req, 1), 2)
        real = sorted(r.get("real_wall_s", 0) for r in rows)
        p90 = real[min(len(real) - 1, int(0.9 * len(real)))]
        hits = sum(r.get("prefix_hits", 0) for r in rows) / max(sum(r.get("prefix_queries", 0) for r in rows), 1)
        otok = round(sum(r.get("out_tokens", 0) for r in rows) / max(req, 1))
        vals = [Path(d).name, len(rows), succ, strict, sreq, med(rows, "requests"), med(rows, "wall_s"),
                med(rows, "real_wall_s"), p90, med(rows, "model_s"), med(rows, "decode_s"), med(rows, "prefill_s"), otok,
                med(rows, "tools_perception_s"), med(rows, "depth_s"), med(rows, "sim_time_s"), f"{hits:.2f}"]
        print("| " + " | ".join(str(v) for v in vals) + " |")


if __name__ == "__main__":
    main()

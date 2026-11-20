"""Primary metric: real-arm seconds of STRICT-SUCCESS episodes, clock from the first policy request.

    python tools/success_stats.py [--with-ext] <run dir> [...]

Default: only the run dir itself (episodes run one at a time = uncontended model latency). --with-ext also reads the
<run>_ext sibling (3 parallel lanes on one policy server: model latency inflated by contention, use for counts only).

Per strict success (host success + finish()): real-arm s = (episode end - first model request start)
  - sim physics/IK wall - sim rendering wall - video-recorder rendering + simulated motion time (real servo schedule);
start-up before the first request is excluded (pre-warmed on the real arm). Prints n, median/min/max and the list.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def episode_real_s(ep: Path) -> dict | None:
    res = json.loads((ep / "episode_result.json").read_text())
    if not (res.get("host_final_success") and res.get("agent_finished")):
        return None
    events = [e for e in jsonl(ep / "driver" / "provider_events.jsonl") if e.get("event") == "response"]
    if not events:
        return None
    first = float(events[0]["t"]) - float(events[0]["latency_s"])
    end = float(res["started_wall_time"]) + float(res["wall_time_s"])
    t = json.loads((ep / "soarm" / "timing.json").read_text()) if (ep / "soarm" / "timing.json").exists() else {}
    frames = jsonl(ep / "soarm" / "frames" / "frames.jsonl")
    rec = sum(f["render_s"] for f in frames if f["t_wall"] >= first)
    real = (end - first) - t.get("command_wall_s", 0.0) - t.get("render_wall_s", 0.0) - rec + t.get("sim_time_s", 0.0)
    return {"episode": f"{ep.parent.name}/{ep.name}", "real_s": round(real, 1), "requests": len(events),
            "model_s": round(sum(float(e["latency_s"]) for e in events), 1), "motion_s": round(t.get("sim_time_s", 0.0), 1)}


def main() -> None:
    args = sys.argv[1:]
    with_ext = "--with-ext" in args
    for run in (a for a in args if a != "--with-ext"):
        rows = []
        for d in (Path(run),) + ((Path(run.rstrip("/") + "_ext"),) if with_ext else ()):
            for ep in sorted(d.glob("soarm_sim_*")):
                try:
                    r = episode_real_s(ep)
                except (FileNotFoundError, KeyError, json.JSONDecodeError):
                    r = None
                if r:
                    rows.append(r)
        vals = [r["real_s"] for r in rows]
        print(json.dumps({"run": Path(run).name, "strict_successes": len(rows),
                          "median_s": round(st.median(vals), 1) if vals else None,
                          "min_s": min(vals) if vals else None, "max_s": max(vals) if vals else None,
                          "median_requests": st.median(r["requests"] for r in rows) if rows else None,
                          "episodes": rows}))


if __name__ == "__main__":
    main()

"""Upper bounds for turn-level real-time chunking (lever D) from finished realistic-path episodes (soarm-speed).

    python tools/overlap_estimate.py <run dir> [...]

D = speculative next turn: while turn i's motion executes on the real arm, the next request is already sent with the
receipt predicted by the twin (the sim runs the same motion much faster than real time). If the real receipt agrees,
the next turn's model latency is hidden behind the motion: saving_i = min(motion_i, latency_{i+1}). Turns whose cell
returns NEW IMAGES after the motion cannot be predicted (camera content changes) and are excluded. Reported per
episode: motion turns, predictable ones, real-arm motion, and the D saving at 100 % receipt agreement (the real-arm
agreement rate is unknown here: in the twin the prediction equals the outcome by construction).
Turn mapping: hidden_request_marks.jsonl gives the env-server call index at the start of every model turn.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

CONTROL_HZ = 20.0


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def episode(ep: Path) -> dict | None:
    marks = jsonl(ep / "hidden_request_marks.jsonl")
    events = [e for e in jsonl(ep / "driver" / "provider_events.jsonl") if e.get("event") == "response"]
    calls = jsonl(ep / "calls.jsonl")
    if not marks or not events:
        return None
    starts = [int(m["call_index"]) for m in marks]          # call index at the start of model turn k
    turns = []
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else 10**9
        cs = [c for c in calls if start < int(c.get("call_index", 0)) <= end]
        motion = sum(float((c.get("value") or {}).get("control_steps") or 0) for c in cs
                     if isinstance(c.get("value"), dict)) / CONTROL_HZ
        images = any(c.get("name") == "observe" for c in cs if c.get("success"))
        turns.append({"motion_s": motion, "images": images})
    # turn k's tool cell runs after response k (events[k]); the next model call is events[k+1]
    saving = 0.0
    n_motion = n_pred = 0
    for k in range(min(len(turns), len(events) - 1)):
        t = turns[k]
        if t["motion_s"] <= 0:
            continue
        n_motion += 1
        if t["images"]:
            continue
        n_pred += 1
        saving += min(t["motion_s"], float(events[k + 1].get("latency_s") or 0.0))
    return {"turns": len(turns), "motion_turns": n_motion, "predictable": n_pred,
            "motion_s": round(sum(t["motion_s"] for t in turns), 1), "d_saving_s": round(saving, 1)}


def main() -> None:
    for run in sys.argv[1:]:
        rows = [r for r in (episode(ep) for ep in sorted(Path(run).glob("soarm_sim_*"))) if r]
        if not rows:
            continue
        def m(k):
            return round(st.mean(r[k] for r in rows), 1)
        print(json.dumps({"run": Path(run).name, "episodes": len(rows), **{k: m(k) for k in rows[0]}}))


if __name__ == "__main__":
    main()

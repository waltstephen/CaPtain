"""Sequential LIBERO episodes with the soarm-speed wall breakdown (Franka transfer check, eval only).

    python tools/libero_speed.py --out <dir> --run $SOARM_ROOT/speed/libero_run.sh \
        --policy-url http://127.0.0.1:8700/v1 --jobs libero_spatial:0-4,libero_object:0-4 --seed 11

Rows (speed.jsonl) carry the speed_eval.py fields; simulated motion time = sum of control_steps of all tool calls at the
LIBERO 20 Hz control rate (motion at the robot's real control schedule); real_wall_s = wall - motion tool wall + that.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch import parse_range, summarize  # noqa: E402
from speed_eval import breakdown, jsonl, scrape  # noqa: E402

CONTROL_HZ = 20.0


def control_steps(ep: Path) -> int:
    total = 0
    for c in jsonl(ep / "calls.jsonl"):
        v = c.get("value")
        if isinstance(v, dict) and isinstance(v.get("control_steps"), (int, float)):
            total += int(v["control_steps"])
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--policy-url", required=True)
    ap.add_argument("--model", default="captain-4b")
    ap.add_argument("--jobs", required=True, help="suite:ids,suite:ids")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, POLICY_URL=a.policy_url, POLICY_MODEL=a.model)
    for job in a.jobs.split(","):
        suite, ids = job.split(":")
        for tid in parse_range(ids):
            ep = out / f"{suite}_{tid}_seed-{a.seed}"
            if (ep / "episode_result.json").exists():
                continue
            before = scrape(a.policy_url)
            t0 = time.time()
            with (out / f"run_{suite}_{tid}.log").open("w") as fh:
                subprocess.run(["bash", a.run, str(out), suite, str(tid), str(a.seed)], env=env, stdout=fh,
                               stderr=subprocess.STDOUT)
            wall = time.time() - t0
            after = scrape(a.policy_url)
            row = {"family": suite, "seed": tid, "wall_s": round(wall, 1)}
            try:
                row.update(summarize(ep))
                row.update(breakdown(ep))
                row["sim_time_s"] = round(control_steps(ep) / CONTROL_HZ, 1)
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"
            row.update({k: round(after[k] - before.get(k, 0.0), 2) for k in after})
            row["real_wall_s"] = round(wall - row.get("tools_motion_s", 0) + row.get("sim_time_s", 0), 1)
            with (out / "speed.jsonl").open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()

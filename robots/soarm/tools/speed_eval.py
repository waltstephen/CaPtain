"""Sequential (1 lane = one arm) eval with a per-episode wall-time breakdown (soarm-speed, eval only).

    python tools/speed_eval.py --out $RUNS_DIR/speed_<cfg> --model captain-4b --policy-url http://127.0.0.1:8600/v1 \
        --families pick_lift,pick_place_mat,put_in_container --seeds 9000-9004 [--perception real]

Student protocol (run_episode.sh: T = 1, 60 requests, 1080 s, intro v1). One episode at a time, so the vLLM /metrics
deltas around an episode belong to it. Row fields (speed.jsonl):
  wall_s           driver process wall
  model_s          sum of request latencies (driver/provider_events.jsonl); prefill_s / decode_s / queue_s from vLLM
  tools_s          sum of tool-call wall (calls.jsonl) split into motion / perception / other by tool name
  command_s        physics + IK wall inside motion (soarm/timing.json), render_s camera rendering, depth_s MoGe calls
  sim_time_s       simulated time = how long the motions take on the real servos at the same control rate
  real_wall_s      wall_s - command_s - render_s + sim_time_s: the same episode with motions at real-arm speed and
                   camera frames for free (a real camera streams); everything else (model, perception, harness) as measured
Already finished episodes are skipped (resumable). Extra env for run_episode.sh passes through (SOARM_PERCEPTION, ...).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from batch import parse_range, prune, summarize  # noqa: E402

PERCEPTION_TOOLS = re.compile(r"^(segment|point|mark_point|measure_|inspect_|compare_|select_detection|track_target)")
MOTION_TOOLS = re.compile(r"^(move_to|release|hold|recover|grasp|gripper_command|open_gripper|close_gripper)")
SERVICE_ERROR = re.compile(r"HTTP Error 5\d\d|Internal Server Error|CUDA out of memory")
METRICS = {
    "prefill_s": "vllm:request_prefill_time_seconds_sum",
    "decode_s": "vllm:request_decode_time_seconds_sum",
    "queue_s": "vllm:request_queue_time_seconds_sum",
    "prompt_tokens": "vllm:prompt_tokens_total",
    "gen_tokens": "vllm:generation_tokens_total",
    "prefix_hits": "vllm:prefix_cache_hits_total",
    "prefix_queries": "vllm:prefix_cache_queries_total",
    "mm_hits": "vllm:mm_cache_hits_total",
    "mm_queries": "vllm:mm_cache_queries_total",
}


def scrape(base: str) -> dict[str, float]:
    url = base.rstrip("/").removesuffix("/v1") + "/metrics"
    try:
        text = urllib.request.urlopen(url, timeout=10).read().decode()
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, float] = {}
    for key, name in METRICS.items():
        vals = [float(m.group(1)) for m in re.finditer(rf"^{re.escape(name)}(?:{{[^}}]*}})? ([0-9.eE+-]+)$", text, re.M)]
        if vals:
            out[key] = sum(vals)
    return out


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def breakdown(ep: Path) -> dict:
    events = [e for e in jsonl(ep / "driver" / "provider_events.jsonl") if e.get("event") == "response"]
    calls = jsonl(ep / "calls.jsonl")
    tools = {"motion": 0.0, "perception": 0.0, "observe": 0.0, "other": 0.0}
    n_perc = n_perc_err = 0
    for c in calls:
        name = str(c.get("name", ""))
        kind = ("perception" if PERCEPTION_TOOLS.search(name) else "motion" if MOTION_TOOLS.search(name)
                else "observe" if name == "observe" else "other")
        tools[kind] += float(c.get("elapsed_s") or 0.0)
        n_perc += kind == "perception"
        # a perception service that answers 5xx (e.g. SAM3 CUDA OOM) turns the episode into a no-perception run
        n_perc_err += kind == "perception" and bool(SERVICE_ERROR.search(json.dumps(c)))
    timing = json.loads((ep / "soarm" / "timing.json").read_text()) if (ep / "soarm" / "timing.json").exists() else {}
    out = {"model_s": round(sum(float(e.get("latency_s") or 0) for e in events), 1),
           "out_tokens": sum((e.get("usage") or {}).get("output_tokens", 0) or 0 for e in events),
           "tool_calls": len(calls), "perception_calls": n_perc, "perception_service_errors": n_perc_err,
           **{f"tools_{k}_s": round(v, 1) for k, v in tools.items()},
           "command_s": round(timing.get("command_wall_s", 0.0), 1), "render_s": round(timing.get("render_wall_s", 0.0), 1),
           "depth_s": round(timing.get("depth_wall_s", 0.0), 1), "depth_calls": timing.get("depth_calls", 0),
           "sim_time_s": round(timing.get("sim_time_s", 0.0), 1)}
    # J1: start-up = episode start -> first model request (pre-warmable on the real arm: robot server, sandbox,
    # perception services, per-session calibration); the frame recorder's own rendering is not counted
    res = ep / "episode_result.json"
    if events and res.exists():
        start = float(json.loads(res.read_text())["started_wall_time"])
        first = float(events[0]["t"]) - float(events[0]["latency_s"])
        frames = jsonl(ep / "soarm" / "frames" / "frames.jsonl")
        rec = sum(f["render_s"] for f in frames if f["t_wall"] < first)
        out["startup_s"] = round(first - start - rec, 1)
        motion = [f for f in frames if f["kind"] == "command" and f["t_wall"] > first]
        if motion:   # wall to the first control tick after the first request, minus recorder cost before it
            rec2 = sum(f["render_s"] for f in frames if first <= f["t_wall"] < motion[0]["t_wall"])
            out["first_motion_after_go_s"] = round(motion[0]["t_wall"] - first - rec2, 1)
    return out


def main() -> None:
    from soarm import taskspec
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="captain-4b")
    ap.add_argument("--policy-url", required=True)
    ap.add_argument("--families", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--wall-time", type=int, default=1080)
    ap.add_argument("--fixed-episode-dir", default=None,
                    help="run every episode at this one path, then move it to <out>/soarm_sim_<tid>_seed-<seed>: the "
                         "prompt's workspace paths are identical across episodes, so a repeated instruction's first "
                         "request hits the prefix cache (deployment: the robot always runs in the same directory)")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, POLICY_URL=a.policy_url, POLICY_MODEL=a.model, WALL_TIME=str(a.wall_time))
    env.setdefault("MAX_REQUESTS", "60")
    for seed in parse_range(a.seeds):
        for family in a.families.split(","):
            tid = taskspec.get(family)["tid"]
            ep = out / f"soarm_sim_{tid}_seed-{seed}"
            res = ep / "episode_result.json"
            if res.exists() and json.loads(res.read_text()).get("termination_reason") != "infrastructure_error":
                continue
            if ep.exists():
                shutil.rmtree(ep)
            run_env = env
            if a.fixed_episode_dir:
                fixed = Path(a.fixed_episode_dir).resolve()
                if fixed.exists():
                    shutil.rmtree(fixed)
                run_env = dict(env, SOARM_EPISODE_DIR=str(fixed))
            before = scrape(a.policy_url)
            t0 = time.time()
            with (out / f"driver_{tid}_{seed}.log").open("w") as fh:
                subprocess.run(["bash", str(HERE / "run_episode.sh"), str(tid), str(seed), str(out)], env=run_env,
                               stdout=fh, stderr=subprocess.STDOUT)
            wall = time.time() - t0
            if a.fixed_episode_dir:
                shutil.move(str(fixed), str(ep))
            after = scrape(a.policy_url)
            row = {"family": family, "seed": seed, "model": a.model, "perception": env.get("SOARM_PERCEPTION", "sim"),
                   "wall_s": round(wall, 1)}
            try:
                row.update(summarize(ep))
                row.update(breakdown(ep))
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"
            row.update({k: round(after[k] - before.get(k, 0.0), 2) for k in after})
            rec_s = sum(f["render_s"] for f in jsonl(ep / "soarm" / "frames" / "frames.jsonl"))   # video recorder
            row["real_wall_s"] = round(wall - row.get("command_s", 0) - row.get("render_s", 0) - rec_s
                                       + row.get("sim_time_s", 0), 1)
            if "startup_s" in row:   # J1: the same episode with start-up pre-warmed before the user's "go"
                row["real_wall_prewarmed_s"] = round(row["real_wall_s"] - row["startup_s"], 1)
            prune(ep)
            with (out / "speed.jsonl").open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()

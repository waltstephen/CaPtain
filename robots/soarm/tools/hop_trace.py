"""Hop-by-hop wall-clock trace of one recorded SO-101 episode (soarm-speed).

    python tools/hop_trace.py <episode dir> [--turns]

Sources (all written by an ordinary run, no extra instrumentation): driver/provider_events.jsonl (end time and latency
of every model request), driver/items.jsonl (python cells and their outputs), calls.jsonl (every robot tool call:
env-server start offset t_s, elapsed_s, perception-service HTTP times in diagnostics.service_requests, control steps
in diagnostics.physical_execution), soarm/timing.json (physics/IK command wall, rendering, MoGe calls, simulated time).

Hops per model turn:
  model         request latency seen by the driver (driver -> vLLM -> driver, incl. HTTP + tokenize/template/images)
  exec          response end -> next request start: python cell in the sandbox + driver work around it
    tools       sum of robot tool calls run by the cell (env-server side elapsed)
      service   perception HTTP time inside tool calls (SAM3 / Molmo / MoGe requests as the env server saw them)
      local     the rest of tool time (sim/IK/control, image decode/encode, geometry, receipts)
    cell_other  exec - tools: sandbox RPC to the env server, python itself, printing, driver bookkeeping
Real-arm view: motion tools at their simulated duration (control steps / 20 Hz) instead of their sim wall, and the
sim-only camera rendering removed (soarm/timing.json render_wall_s), as in tools/speed_eval.py.
Calls are mapped to turns by time: the env-server clock is anchored so that the first call starts when the first
response ends (dispatch of the first cell is therefore counted as 0); a call belongs to the last response that ended
before it (50 ms tolerance).
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

CONTROL_HZ = 20.0
SERVICE_TOOLS = re.compile(r"^(segment|point|mark_point|measure_|inspect_|compare_|select_detection|track_target)")
MOTION_TOOLS = re.compile(r"^(move_to|release|hold|recover|grasp|gripper_command|open_gripper|close_gripper)")


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def simulated_s(call: dict) -> float:
    """Simulated (= real-servo) duration of a motion call: its value's control_steps at the control rate."""
    value = call.get("value")
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            value = None
    steps = (value or {}).get("control_steps") if isinstance(value, dict) else None
    if steps is None:
        steps = ((call.get("diagnostics") or {}).get("physical_execution") or {}).get("control_steps") or 0
    return int(steps) / CONTROL_HZ


def service_s(call: dict) -> float:
    reqs = (call.get("diagnostics") or {}).get("service_requests") or []
    return sum(float(r.get("elapsed_s") or 0.0) for r in reqs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("episode")
    ap.add_argument("--turns", action="store_true", help="print one line per model turn")
    a = ap.parse_args()
    ep = Path(a.episode)
    responses = [e for e in jsonl(ep / "driver" / "provider_events.jsonl") if e.get("event") == "response"]
    calls = jsonl(ep / "calls.jsonl")
    items = [r.get("item", r) for r in jsonl(ep / "driver" / "items.jsonl")]
    outputs = [i for i in items if i.get("type") == "custom_tool_call_output"]
    timing = json.loads((ep / "soarm" / "timing.json").read_text()) if (ep / "soarm" / "timing.json").exists() else {}

    starts = [r["t"] - r["latency_s"] for r in responses]
    ends = [r["t"] for r in responses]
    anchor = ends[0] - float(calls[0]["t_s"]) if calls else 0.0
    by_turn: dict[int, list[dict]] = {}
    for c in calls:   # the cell of the last response that ended before the call started (50 ms dispatch tolerance)
        t = anchor + float(c["t_s"])
        i = max((k for k, e in enumerate(ends) if e <= t + 0.05), default=0)
        by_turn.setdefault(i, []).append(c)
    turns = []
    for i, end in enumerate(ends):
        nxt = starts[i + 1] if i + 1 < len(starts) else None
        in_turn = by_turn.get(i, [])
        tools = sum(float(c.get("elapsed_s") or 0) for c in in_turn)
        svc = sum(service_s(c) for c in in_turn)
        motion_wall = sum(float(c.get("elapsed_s") or 0) for c in in_turn if MOTION_TOOLS.search(c["name"]))
        motion_real = sum(simulated_s(c) for c in in_turn if MOTION_TOOLS.search(c["name"]))
        out_text = "".join(p.get("text", "") for p in outputs[i]["output"]) if i < len(outputs) else ""
        exec_s = (nxt - end) if nxt is not None else max(tools, 0.0)
        turns.append({"turn": i + 1, "model_s": responses[i]["latency_s"],
                      "in_tok": responses[i]["usage"]["input_tokens"], "out_tok": responses[i]["usage"]["output_tokens"],
                      "exec_s": exec_s, "tools_s": tools, "service_s": svc, "local_s": tools - svc - motion_wall,
                      "motion_wall_s": motion_wall, "motion_real_s": motion_real, "cell_other_s": exec_s - tools,
                      "calls": [c["name"] for c in in_turn], "output_chars": len(out_text)})
        if a.turns:
            t = turns[-1]
            print(f"turn {t['turn']:3d} model {t['model_s']:.2f}s in {t['in_tok']} out {t['out_tok']} | exec "
                  f"{t['exec_s']:.2f}s = tools {t['tools_s']:.2f} (service {t['service_s']:.2f}, motion "
                  f"{t['motion_wall_s']:.2f} wall / {t['motion_real_s']:.2f} real, local {t['local_s']:.2f}) + other "
                  f"{t['cell_other_s']:.2f} | out {t['output_chars']} ch | {','.join(t['calls'])}")
    tot = {k: round(sum(t[k] for t in turns), 2) for k in
           ("model_s", "exec_s", "tools_s", "service_s", "local_s", "motion_wall_s", "motion_real_s", "cell_other_s")}
    render = float(timing.get("render_wall_s", 0.0))
    per_tool: dict[str, list[float]] = {}
    for c in calls:
        per_tool.setdefault(c["name"], []).append(float(c.get("elapsed_s") or 0))
    summary = {
        "turns": len(turns), "calls": len(calls), **tot,
        "render_wall_s": round(render, 2), "depth_wall_s": round(float(timing.get("depth_wall_s", 0.0)), 2),
        "command_wall_s": round(float(timing.get("command_wall_s", 0.0)), 2),
        "real_arm_s": round(tot["model_s"] + tot["exec_s"] - tot["motion_wall_s"] + tot["motion_real_s"] - render, 2),
        "tool_output_chars": sum(t["output_chars"] for t in turns),
        "per_tool": {k: {"n": len(v), "sum_s": round(sum(v), 2), "mean_ms": round(1000 * sum(v) / len(v))}
                     for k, v in sorted(per_tool.items(), key=lambda kv: -sum(kv[1]))},
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

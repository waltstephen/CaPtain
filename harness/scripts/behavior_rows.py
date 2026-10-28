#!/usr/bin/env python3
"""Behaviour statistics for a batch (calls.jsonl + episode_result.json): re-observe rate, verification
tools, images viewed, finish discipline, time spent. Works for driver and Codex episodes.

    python3 scripts/behavior_rows.py <batch_dir> [<batch_dir> ...]
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_rows import agent_block  # noqa: E402

PHYSICAL = {"move_to", "hold", "recover", "release"}
VERIFY_TOOLS = {"compare_object_observations", "inspect_release_geometry", "inspect_contact_geometry", "check_motion", "preview_motion", "track_target", "measure_object_geometry"}


def episode_stats(ep: Path) -> dict | None:
    calls_path = ep / "calls.jsonl"
    result_path = ep / "episode_result.json"
    if not calls_path.is_file() or not result_path.is_file():
        return None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    calls = [json.loads(l) for l in calls_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    names = [c["name"] for c in calls]
    counter = Counter(names)
    physical_idx = [i for i, c in enumerate(calls) if c["consumes_action"]]
    executed_physical = [i for i in physical_idx if calls[i].get("executed")]
    reobserved = sum(1 for i in executed_physical if any(n == "observe" for n in names[i + 1:i + 3]))
    last_physical = executed_physical[-1] if executed_physical else None
    observe_after_last = bool(last_physical is not None and "observe" in names[last_physical + 1:])
    measured_after_last = bool(last_physical is not None and "measure_object_geometry" in names[last_physical + 1:])
    block = agent_block(result)
    rollout = block.get("rollout") or {}
    return {
        "suite": result.get("suite"), "task_id": result.get("task_id"), "success": result.get("host_final_success"),
        "agent_finished": result.get("agent_finished"), "false_finish": result.get("false_finish"), "gate_rejections": result.get("gate_rejections"),
        "calls": len(calls), "physical": len(physical_idx), "executed_physical": len(executed_physical),
        "failed_physical": sum(1 for i in physical_idx if not calls[i].get("success")),
        "reobserve_rate": (reobserved / len(executed_physical)) if executed_physical else None,
        "observe_after_last_action": observe_after_last, "measure_after_last_action": measured_after_last,
        "observes": counter.get("observe", 0), "segments": counter.get("segment", 0), "points": counter.get("point", 0) + counter.get("mark_point", 0),
        "select_detection": counter.get("select_detection", 0), "measure_object_geometry": counter.get("measure_object_geometry", 0),
        "verify_tools": sum(counter.get(t, 0) for t in VERIFY_TOOLS), "preview_motion": counter.get("preview_motion", 0),
        "view_image_calls": rollout.get("view_image_calls"), "reasoning_items": rollout.get("reasoning_items"),
        "shell_commands": len(block.get("commands") or []), "requests": block.get("requests"),
        "wall_s": result.get("wall_time_s"), "tool_time_s": round(sum(c.get("elapsed_s", 0) for c in calls), 1),
        "tokens": (block.get("usage") or {}).get("total_tokens"), "output_tokens": (block.get("usage") or {}).get("output_tokens"),
    }


def _mean(v):
    v = [x for x in v if isinstance(x, (int, float))]
    return statistics.mean(v) if v else None


def batch_summary(rows: list[dict]) -> dict:
    keys = ["calls", "physical", "failed_physical", "reobserve_rate", "observes", "segments", "points", "select_detection", "measure_object_geometry",
            "verify_tools", "preview_motion", "view_image_calls", "reasoning_items", "shell_commands", "requests", "wall_s", "tool_time_s", "tokens", "output_tokens"]
    out = {k: _mean([r[k] for r in rows]) for k in keys}
    out["n"] = len(rows)
    out["observe_after_last_action_rate"] = _mean([1.0 if r["observe_after_last_action"] else 0.0 for r in rows])
    out["measure_after_last_action_rate"] = _mean([1.0 if r["measure_after_last_action"] else 0.0 for r in rows])
    for label, subset in (("success", [r for r in rows if r["success"]]), ("false_finish", [r for r in rows if r["false_finish"]])):
        out[label] = {"n": len(subset), **{k: _mean([r[k] for r in subset]) for k in ("physical", "observes", "verify_tools", "view_image_calls", "wall_s", "tokens", "requests")}}
    return out


def main() -> int:
    for arg in sys.argv[1:]:
        batch = Path(arg)
        rows = [s for ep in sorted((batch / "episodes").iterdir()) if ".infra-" not in ep.name and ".partial-" not in ep.name for s in [episode_stats(ep)] if s]
        summary = batch_summary(rows)
        (batch / "behavior.json").write_text(json.dumps({"summary": summary, "episodes": rows}, indent=1, sort_keys=True), encoding="utf-8")
        print(f"== {batch.name} (n={summary['n']})")
        for k, v in summary.items():
            if isinstance(v, dict):
                print(f"  {k}: {json.dumps({kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in v.items()})}")
            elif isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

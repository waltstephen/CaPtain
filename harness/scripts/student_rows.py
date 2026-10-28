#!/usr/bin/env python3
"""Student rows: success / false finish / requests / tool-call error rate per batch directory.

Reads episodes/*/episode_result.json and driver/transcript.jsonl of V1.1 batches (any --api) and
prints a markdown table. A tool call counts as an error when its output is a refusal
("unsupported call", "failed to parse function arguments", "<tool> failed:"), an exec envelope with
a non-zero exit code, or a code-mode result with a "Script error" / "Script failed" part.

    python3 scripts/student_rows.py "sft=outputs/<batch-a>" "rl=outputs/<batch-b>" ...
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

EXIT_RE = re.compile(r"^Process exited with code (\d+)$", re.M)
REFUSAL_PREFIXES = ("unsupported call", "failed to parse function arguments")


def _text(output) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return "\n".join(str(p.get("text", "")) for p in output if isinstance(p, dict) and p.get("type") == "input_text")
    return json.dumps(output)


def classify(call: dict, output: dict) -> str:
    text = _text(output.get("output"))
    if text.startswith(REFUSAL_PREFIXES):
        return "refused"
    name = str(call.get("name") or "")
    if text.startswith(f"{name} failed:") or text.startswith("write_stdin failed") or text.startswith("exec_command failed"):
        return "failed"
    if call.get("type") == "custom_tool_call":
        if text.startswith("Script failed") or "\nScript error:" in text or text.startswith("Script error"):
            return "script_error"
        return "ok"
    match = EXIT_RE.search(text)
    if match and int(match.group(1)) != 0:
        return "nonzero_exit"
    return "ok"


def episode_stats(episode_dir: Path) -> dict | None:
    result_path = episode_dir / "episode_result.json"
    if not result_path.is_file():
        return None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    items = []
    transcript = episode_dir / "driver" / "transcript.jsonl"
    if transcript.is_file():
        for line in transcript.read_text(encoding="utf-8").splitlines():
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    calls = {i.get("call_id"): i for i in items if i.get("type") in ("function_call", "custom_tool_call")}
    kinds = {"ok": 0, "refused": 0, "failed": 0, "script_error": 0, "nonzero_exit": 0}
    for item in items:
        if item.get("type") in ("function_call_output", "custom_tool_call_output"):
            call = calls.get(item.get("call_id")) or {}
            kinds[classify(call, item)] += 1
    total = sum(kinds.values())
    driver = result.get("driver") or {}
    return {
        "success": bool(result.get("host_final_success")),
        "false_finish": bool(result.get("false_finish")),
        "agent_finished": bool(result.get("agent_finished")),
        "termination": result.get("termination_reason"),
        "requests": driver.get("requests"),
        "tool_calls": total,
        "errors": total - kinds["ok"],
        "refused": kinds["refused"],
        "script_errors": kinds["script_error"],
        "actions": result.get("action_requests"),
        "wall_min": (result.get("wall_time_s") or 0) / 60.0,
        "infra": result.get("termination_reason") == "infrastructure_error",
    }


def summarize(batch_dir: Path) -> dict:
    rows = []
    for path in sorted((batch_dir / "episodes").glob("*")):
        if ".infra-" in path.name or ".partial-" in path.name or not path.is_dir():
            continue
        stats = episode_stats(path)
        if stats is not None:
            rows.append(stats)
    scored = [r for r in rows if not r["infra"]]
    calls = sum(r["tool_calls"] for r in scored)
    errors = sum(r["errors"] for r in scored)
    terminations: dict[str, int] = {}
    for r in scored:
        terminations[str(r["termination"])] = terminations.get(str(r["termination"]), 0) + 1

    def med(key):
        values = [r[key] for r in scored if isinstance(r.get(key), (int, float))]
        return statistics.median(values) if values else None

    return {
        "n": len(scored), "infra": len(rows) - len(scored), "success": sum(r["success"] for r in scored),
        "false_finish": sum(r["false_finish"] for r in scored), "finished": sum(r["agent_finished"] for r in scored),
        "requests_med": med("requests"), "actions_med": med("actions"), "wall_med": med("wall_min"),
        "tool_calls": calls, "tool_error_rate": (errors / calls) if calls else None,
        "refused": sum(r["refused"] for r in scored), "script_errors": sum(r["script_errors"] for r in scored),
        "terminations": terminations,
    }


def main() -> int:
    print("| row | success / scored (infra) | false finish | finish() | requests med | actions med | wall med (min) | tool calls | tool-call error rate | refused | terminations |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for arg in sys.argv[1:]:
        label, _, path = arg.partition("=")
        if not path:
            label, path = Path(arg).name, arg
        s = summarize(Path(path))
        rate = "-" if s["tool_error_rate"] is None else f"{100 * s['tool_error_rate']:.1f}%"
        terms = ", ".join(f"{k} {v}" for k, v in sorted(s["terminations"].items()))
        print(f"| {label} | **{s['success']}**/{s['n']} ({s['infra']}) | {s['false_finish']} | {s['finished']} | {s['requests_med']} | {s['actions_med']} | "
              f"{s['wall_med']:.1f} | {s['tool_calls']} | {rate} | {s['refused']} | {terms} |" if s["wall_med"] is not None else
              f"| {label} | 0/0 | | | | | | | | | |")
    return 0


if __name__ == "__main__":
    sys.exit(main())

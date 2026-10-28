#!/usr/bin/env python3
"""Summarise a batch directory (episodes/*/episode_result.json) into results.jsonl + summary.json/md.

Accepts episodes produced by the V1.1 driver (`driver` block) and by the Codex CLI reference
driver (`codex` block).

    python3 scripts/summarize_rows.py <batch_dir> [<batch_dir> ...]
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")


def agent_block(row: dict) -> dict:
    return row.get("driver") or row.get("codex") or {}


def load_rows(batch_dir: Path) -> list[dict]:
    rows = []
    for path in sorted((batch_dir / "episodes").glob("*/episode_result.json")):
        if ".infra-" in path.parent.name or ".partial-" in path.parent.name:
            continue
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return rows


def _mean(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.mean(values) if values else None


def _median(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(values) if values else None


def summarize(rows: list[dict]) -> dict:
    done = [r for r in rows if r.get("termination_reason") != "dry_run"]
    infra = [r for r in done if r.get("termination_reason") == "infrastructure_error"]
    scored = [r for r in done if r.get("host_final_success") is not None]
    success = [r for r in scored if r.get("host_final_success")]
    finished = [r for r in scored if r.get("agent_finished")]
    false_finish = [r for r in finished if not r.get("host_final_success")]
    missed = [r for r in scored if r.get("missed_finish")]
    per_suite = {}
    for suite in list(SUITES) + sorted({str(r.get("suite")) for r in scored} - set(SUITES)):
        s_rows = [r for r in scored if r.get("suite") == suite]
        if s_rows:
            per_suite[suite] = {"done": len(s_rows), "success": sum(1 for r in s_rows if r.get("host_final_success")),
                                "false_finish": sum(1 for r in s_rows if r.get("agent_finished") and not r.get("host_final_success"))}
    terminations: dict[str, int] = {}
    for r in done:
        terminations[r.get("termination_reason", "?")] = terminations.get(r.get("termination_reason", "?"), 0) + 1
    usage = [agent_block(r).get("usage") or {} for r in scored]
    requests = [agent_block(r).get("requests") for r in scored]
    commands = [len(agent_block(r).get("commands") or []) for r in scored]
    views = [(agent_block(r).get("rollout") or {}).get("view_image_calls") for r in scored]
    with_rollout = [v for v in views if v is not None]
    return {
        "episodes_done": len(done), "scored": len(scored), "success": len(success),
        "success_rate": (len(success) / len(scored)) if scored else None,
        "agent_finished": len(finished), "false_finish": len(false_finish), "missed_finish": len(missed),
        "host_success_ever": sum(1 for r in scored if r.get("host_success_ever")),
        "gate_rejections_total": sum(int(r.get("gate_rejections") or 0) for r in scored),
        "episodes_with_gate_rejection": sum(1 for r in scored if r.get("gate_rejections")),
        "infrastructure_errors": len(infra), "per_suite": per_suite, "terminations": terminations,
        "actions_mean": _mean([r.get("action_requests") for r in scored]),
        "actions_median": _median([r.get("action_requests") for r in scored]),
        "tool_calls_mean": _mean([r.get("tool_requests") for r in scored]),
        "observations_mean": _mean([r.get("observation_exports") for r in scored]),
        "wall_mean_s": _mean([r.get("wall_time_s") for r in scored]),
        "wall_median_s": _median([r.get("wall_time_s") for r in scored]),
        "requests_median": _median(requests),
        "tokens_mean": _mean([u.get("total_tokens") for u in usage]),
        "tokens_total": sum(u.get("total_tokens") for u in usage if isinstance(u.get("total_tokens"), (int, float))),
        "input_tokens_mean": _mean([u.get("input_tokens") for u in usage]),
        "cached_input_tokens_mean": _mean([u.get("cached_input_tokens") for u in usage]),
        "output_tokens_mean": _mean([u.get("output_tokens") for u in usage]),
        "output_tokens_median": _median([u.get("output_tokens") for u in usage]),
        "reasoning_tokens_mean": _mean([u.get("reasoning_output_tokens") for u in usage]),
        "commands_mean": _mean(commands), "commands_median": _median(commands),
        "view_image_mean": _mean(with_rollout), "view_image_episodes": sum(1 for v in with_rollout if v),
        "episodes_with_rollout": len(with_rollout),
    }


def table(rows: list[dict]) -> str:
    header = ("| suite | task | success | ever | finished | gate rej | term | actions | tools | obs | wall s | requests | out tok | tokens | cmds | views |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    order = {s: i for i, s in enumerate(SUITES)}
    for r in sorted(rows, key=lambda r: (order.get(r.get("suite"), 9), r.get("task_id", 0))):
        block = agent_block(r)
        usage = block.get("usage") or {}
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r.get("suite"), r.get("task_id"),
            {True: "YES", False: "no", None: "-"}[r.get("host_final_success")],
            {True: "y", False: "", None: "-"}[r.get("host_success_ever")],
            "finish()" if r.get("agent_finished") else "",
            r.get("gate_rejections", "-"), r.get("termination_reason"), r.get("action_requests"), r.get("tool_requests"),
            r.get("observation_exports"), r.get("wall_time_s"), block.get("requests", "-"), usage.get("output_tokens", "-"),
            usage.get("total_tokens"), len(block.get("commands") or []), (block.get("rollout") or {}).get("view_image_calls", "-")))
    return header + "\n".join(lines) + "\n"


def write_summary(batch_dir: Path) -> dict:
    rows = load_rows(batch_dir)
    with (batch_dir / "results.jsonl").open("w", encoding="utf-8") as stream:
        for r in rows:
            stream.write(json.dumps(r, sort_keys=True) + "\n")
    summary = summarize(rows)
    (batch_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    md = (f"# {batch_dir.name}\n\n```\n{json.dumps({k: v for k, v in summary.items() if k != 'per_suite'}, indent=1, sort_keys=True)}\n```\n\n"
          f"per suite: {json.dumps(summary['per_suite'])}\n\n" + table(rows))
    (batch_dir / "summary.md").write_text(md, encoding="utf-8")
    return summary


def main() -> int:
    for arg in sys.argv[1:]:
        batch = Path(arg)
        s = write_summary(batch)
        print(f"{batch.name}: {s['scored']} scored, success {s['success']} ({(s['success_rate'] or 0) * 100:.0f}%), finish() {s['agent_finished']}, "
              f"false_finish {s['false_finish']}, gate rejections {s['gate_rejections_total']} in {s['episodes_with_gate_rejection']} eps, infra {s['infrastructure_errors']}; "
              f"wall median {s['wall_median_s']}, requests median {s['requests_median']}, output tokens median {s['output_tokens_median']}")
        print("  per suite:", json.dumps(s["per_suite"]))
        print("  terminations:", json.dumps(s["terminations"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

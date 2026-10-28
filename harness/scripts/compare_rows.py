#!/usr/bin/env python3
"""Side-by-side table of batch directories (driver rows vs Codex rows).

    python3 scripts/compare_rows.py "V1.1 driver / gpt-5.5=outputs/eval-react-gpt-5.5-..." "Codex CLI / gpt-5.5=outputs/eval-codex-gpt-5.5-..."
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_rows import SUITES, agent_block, load_rows  # noqa: E402


def _mean(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.mean(values) if values else None


def _median(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(values) if values else None


def stats(rows: list[dict]) -> dict:
    scored = [r for r in rows if r.get("host_final_success") is not None and r.get("termination_reason") != "infrastructure_error"]
    blocks = [agent_block(r) for r in scored]
    return {
        "n": len(rows), "scored": len(scored), "infra": sum(1 for r in rows if r.get("termination_reason") == "infrastructure_error"),
        "success": sum(1 for r in scored if r["host_final_success"]),
        "finished": sum(1 for r in scored if r.get("agent_finished")),
        "false_finish": sum(1 for r in scored if r.get("false_finish")),
        "gate_rejections": sum(int(r.get("gate_rejections") or 0) for r in scored),
        "actions_median": _median([r.get("action_requests") for r in scored]),
        "requests_median": _median([b.get("requests") for b in blocks]),
        "output_tokens_median": _median([(b.get("usage") or {}).get("output_tokens") for b in blocks]),
        "wall_median_min": (_median([r.get("wall_time_s") for r in scored]) or 0) / 60 if scored else None,
        "tokens_mean_m": (_mean([(b.get("usage") or {}).get("total_tokens") for b in blocks]) or 0) / 1e6 if scored else None,
        "per_suite": {s: (sum(1 for r in scored if r.get("suite") == s and r["host_final_success"]), sum(1 for r in scored if r.get("suite") == s)) for s in SUITES},
    }


def fmt(value, spec):
    return "-" if value is None else format(value, spec)


def markdown(table_rows) -> str:
    suites = [s for s in SUITES if any(st["per_suite"][s][1] for _, st in table_rows)]
    lines = ["| row | success / scored (infra) | " + " | ".join(s.replace("libero_", "") for s in suites) +
             " | finish() | false finish | gate rej | actions med | requests med | out tok med | wall med (min) | tokens mean (M) |",
             "|---|---|" + "---|" * len(suites) + "---|---|---|---|---|---|---|---|"]
    for label, s in table_rows:
        per = " | ".join(f"{a}/{b}" for a, b in (s["per_suite"][x] for x in suites))
        lines.append(f"| {label} | **{s['success']}**/{s['scored']} ({s['infra']}) | {per} | {s['finished']} | {s['false_finish']} | {s['gate_rejections']} | "
                     f"{fmt(s['actions_median'], '.0f')} | {fmt(s['requests_median'], '.0f')} | {fmt(s['output_tokens_median'], '.0f')} | "
                     f"{fmt(s['wall_median_min'], '.1f')} | {fmt(s['tokens_mean_m'], '.2f')} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    table = []
    for arg in sys.argv[1:]:
        label, _, path = arg.partition("=")
        rows = load_rows(Path(path))
        table.append((label, stats(rows)))
    print(markdown(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

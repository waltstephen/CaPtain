#!/usr/bin/env python3
"""Per-task pairing of two batches on the same task list (e.g. Codex CLI vs the V1.1 driver).

    python3 scripts/pair_rows.py --a outputs/<codex batch> --b outputs/<driver batch> [--label-a "Codex CLI" --label-b "V1.1 driver"]
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_rows import SUITES, agent_block, load_rows  # noqa: E402


def index(rows):
    return {(r.get("suite"), r.get("task_id")): r for r in rows if r.get("termination_reason") != "infrastructure_error"}


def med(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(values) if values else None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    args = p.parse_args()
    a, b = index(load_rows(args.a)), index(load_rows(args.b))
    keys = sorted(set(a) & set(b), key=lambda k: (SUITES.index(k[0]) if k[0] in SUITES else 9, k[1]))
    both = a_only = b_only = neither = 0
    lines = [f"| suite | task | {args.label_a} | {args.label_b} | gate rej A/B | actions A/B | requests A/B | wall min A/B |", "|---|---|---|---|---|---|---|---|"]
    for k in keys:
        ra, rb = a[k], b[k]
        sa, sb = bool(ra.get("host_final_success")), bool(rb.get("host_final_success"))
        both += sa and sb; a_only += sa and not sb; b_only += sb and not sa; neither += not sa and not sb
        fmt = lambda r, s: ("YES" if s else ("FF" if r.get("agent_finished") else "no"))  # noqa: E731
        lines.append(f"| {k[0].replace('libero_', '')} | {k[1]} | {fmt(ra, sa)} | {fmt(rb, sb)} | {ra.get('gate_rejections', 0)}/{rb.get('gate_rejections', 0)} | "
                     f"{ra.get('action_requests')}/{rb.get('action_requests')} | {agent_block(ra).get('requests', '-')}/{agent_block(rb).get('requests', '-')} | "
                     f"{(ra.get('wall_time_s') or 0) / 60:.1f}/{(rb.get('wall_time_s') or 0) / 60:.1f} |")
    print(f"paired tasks: {len(keys)}; both succeed {both}, only {args.label_a} {a_only}, only {args.label_b} {b_only}, neither {neither}")
    for label, rows in ((args.label_a, [a[k] for k in keys]), (args.label_b, [b[k] for k in keys])):
        print(f"{label}: success {sum(1 for r in rows if r.get('host_final_success'))}/{len(rows)}, false finish {sum(1 for r in rows if r.get('false_finish'))}, "
              f"gate rejections {sum(int(r.get('gate_rejections') or 0) for r in rows)}, median actions {med([r.get('action_requests') for r in rows])}, "
              f"median requests {med([agent_block(r).get('requests') for r in rows])}, median output tokens {med([(agent_block(r).get('usage') or {}).get('output_tokens') for r in rows])}, "
              f"median wall {(med([r.get('wall_time_s') for r in rows]) or 0) / 60:.1f} min")
    print()
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

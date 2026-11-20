"""Summary of one model's sftopd eval (tools/sftopd_eval.sh): suites div3_seen + cube_heldout_families.

    python tools/sftopd_summary.py <eval dir> [--name NAME]   -> <eval dir>/summary.{json,md}

<eval dir>/<suite>/batch.jsonl rows (tools/batch.py). strict = host success + finish() + no false finish; host = the
simulator's goal atoms at the end. Families: div3_seen by its mix group (div3.json "group"), cube_heldout_families by spec.
Cap hits = termination request_cap_exhausted (60 requests) or the wall budget; infra = <suite>/infra_discarded.jsonl rows
(episodes the batch discarded as infrastructure_error, rerun on resume) plus jobs still missing.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from soarm import taskspec  # noqa: E402

SUITES = ("div3_seen", "cube_heldout_families")


def strict(r: dict) -> bool:
    return bool(r.get("success") and r.get("finished") and not r.get("false_finish"))


def rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir", type=Path)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    groups = {t["id"]: t["group"] for t in json.loads((taskspec.SPEC_DIR / "mixes" / "div3.json").read_text())["tasks"]}
    out = {"name": a.name or a.eval_dir.name, "suites": {}}
    md = [f"# sftopd eval: {out['name']}", "",
          "| suite | family | n | strict | host | finished | cap hits (req / wall) |", "|---|---|--:|--:|--:|--:|--:|"]
    for suite in SUITES:
        jobs = taskspec.suite(suite)["jobs"]
        got = {(r["family"], int(r["seed"])): r for r in rows(a.eval_dir / suite / "batch.jsonl")}
        done = [got[j] for j in jobs if j in got]
        infra = rows(a.eval_dir / suite / "infra_discarded.jsonl")
        fam = collections.defaultdict(list)
        for r in done:
            fam[groups.get(r["family"], r["family"]) if suite == "div3_seen" else r["family"]].append(r)
        term = collections.Counter(str(r.get("termination")) for r in done)

        def stats(rs):
            return {"n": len(rs), "strict": sum(map(strict, rs)), "host": sum(bool(r.get("success")) for r in rs),
                    "finished": sum(bool(r.get("finished")) for r in rs),
                    "req_cap": sum(r.get("termination") == "request_cap_exhausted" for r in rs),
                    "wall_cap": sum("wall" in str(r.get("termination")) or "timeout" in str(r.get("termination"))
                                    for r in rs)}
        s = {"jobs": len(jobs), "missing": len(jobs) - len(done), "infra_discarded": len(infra),
             "terminations": dict(term), **stats(done), "families": {k: stats(v) for k, v in sorted(fam.items())}}
        out["suites"][suite] = s
        for k, v in [("ALL", s)] + sorted(s["families"].items()):
            md.append(f"| {suite} | {k} | {v['n']} | {v['strict']} | {v['host']} | {v['finished']} | "
                      f"{v['req_cap']} / {v['wall_cap']} |")
    md += ["", "| suite | jobs | missing | infra discarded | terminations |", "|---|--:|--:|--:|---|"]
    for suite, s in out["suites"].items():
        md.append(f"| {suite} | {s['jobs']} | {s['missing']} | {s['infra_discarded']} | "
                  + ", ".join(f"{k} {v}" for k, v in sorted(s["terminations"].items())) + " |")
    (a.eval_dir / "summary.json").write_text(json.dumps(out, indent=1))
    (a.eval_dir / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

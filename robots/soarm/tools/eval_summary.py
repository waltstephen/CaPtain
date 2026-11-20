"""Fixed eval suites (CONTRACT 6): one table per model over the 4 suites, strict = success + finish() + no false finish.

    python tools/eval_summary.py runs/eval_sftv1 [runs/eval_base ...] [--json out.json]

Each model dir holds one batch per suite: <model dir>/<suite id>/batch.jsonl (tools/batch.py --suite <suite id>).
Jobs of the suite without a row count as missing (not as failures); infrastructure errors are listed separately.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from soarm import taskspec  # noqa: E402
from summarize import load  # noqa: E402


def strict(r: dict) -> bool:
    return bool(r.get("success") and r.get("finished") and not r.get("false_finish"))


def suite_stats(model_dir: Path, name: str) -> dict:
    jobs = taskspec.suite(name)["jobs"]
    rows = {(r["family"], int(r["seed"])): r for r in load(model_dir / name)}
    done = [rows[j] for j in jobs if j in rows and rows[j].get("termination") != "infrastructure_error"]
    per_spec: dict[str, list[int]] = {}
    for r in done:
        s = per_spec.setdefault(r["family"], [0, 0])
        s[0] += strict(r)
        s[1] += 1
    return {"jobs": len(jobs), "done": len(done), "missing": len(jobs) - len(done),
            "infra": sum(1 for j in jobs if rows.get(j, {}).get("termination") == "infrastructure_error"),
            "host_success": sum(bool(r.get("success")) for r in done), "strict": sum(strict(r) for r in done),
            "per_spec": per_spec}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", type=Path)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    names = taskspec.suites()
    out = {str(m): {n: suite_stats(m, n) for n in names} for m in a.models}
    print("| model | " + " | ".join(f"{n} strict / done (missing)" for n in names) + " |")
    print("|---|" + "---|" * len(names))
    for m, stats in out.items():
        cells = [f"{s['strict']}/{s['done']} ({s['missing']})" for s in stats.values()]
        print(f"| {Path(m).name} | " + " | ".join(cells) + " |")
    for m, stats in out.items():
        for n, s in stats.items():
            if s["per_spec"]:
                print(f"{Path(m).name} {n}: " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(s["per_spec"].items())))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

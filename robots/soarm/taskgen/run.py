"""taskgen end to end: propose -> schema -> layouts/reach/no-initial-success -> scripted physics solvability -> difficulty
-> dedupe, then paraphrases + resolver test for the accepted families and for the existing hand/legacy families.

    python -m taskgen.run --out DIR [--cube-target 18] [--budget-usd 3] [--rounds 8] [--broad-rounds 3]

Outputs in DIR: proposals.jsonl (every proposal with its stage verdict), specs/gen/<id>.json (accepted, CONTRACT 2,
tids from 1000), paraphrases/<id>.json (existing families: kept templates + resolver results), stats.json.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from soarm import taskspec
from soarm.tasks import SceneConfig

from . import dedupe, paraphrase, propose, verify
from .llm import LLM, BudgetExceeded

AXIS_PLAN = [["spatial", "ordinal"], ["counting", "functional"], ["multi_step", "spatial"], ["ordinal", "counting"],
             ["functional", "multi_step"], ["conditional", "counting"], ["spatial", "functional"],
             ["multi_step", "ordinal"], ["conditional", "spatial"], ["counting", "multi_step"]]
TID0 = 1000


class Run:
    def __init__(self, a) -> None:
        self.a = a
        self.out = Path(a.out)
        (self.out / "specs" / "gen").mkdir(parents=True, exist_ok=True)
        (self.out / "paraphrases").mkdir(exist_ok=True)
        self.llm = LLM(url=a.url, budget_usd=a.budget_usd, ledger=str(self.out / "llm_usage.jsonl"),
                       **({"model": a.model} if a.model else {}))
        self.config = SceneConfig.from_json(json.loads(a.scene_config)) if a.scene_config else None
        self.index = dedupe.Index()
        for s in taskspec.registry().values():
            self.index.add(s)
        self.accepted: list[dict] = []
        self.rejected_msgs: list[str] = []
        self.stats = {"proposed": 0, "parsed": 0, "schema": 0, "layouts": 0, "physics": 0, "dedupe": 0,
                      "accepted": 0, "by_domain": {}, "reasons": {}}
        self.next_tid = a.tid0
        self.t0 = time.time()

    def _log(self, row: dict) -> None:
        with open(self.out / "proposals.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")

    def _reason(self, key: str) -> None:
        self.stats["reasons"][key] = self.stats["reasons"].get(key, 0) + 1

    # ------------------------------------------------------------------ families
    def round(self, domain: str, axes: list[str], n: int) -> None:
        specs = propose.propose(self.llm, domain, axes, n, self.accepted, self.rejected_msgs)
        self.stats["proposed"] += n
        self.stats["parsed"] += len(specs)
        dom = self.stats["by_domain"].setdefault(domain, {"proposed": 0, "schema": 0, "layouts": 0, "physics": 0,
                                                            "dedupe": 0, "accepted": 0})
        dom["proposed"] += len(specs)
        for s in specs:
            s["id"] = "gen_" + re.sub(r"^gen_", "", re.sub(r"[^a-z0-9_]", "_", str(s.get("id", "x")).lower()))[:40]
            s.update(tid=self.next_tid, provenance="llm", builder="declarative", version=1, domain=domain,
                     split="heldout" if domain == "broad" else "train")
            s.setdefault("difficulty", 2)
            if not isinstance(s.get("difficulty"), int) or not 1 <= s["difficulty"] <= 5:
                s["difficulty"] = 2
            self.next_tid += 1
            row = {"id": s["id"], "tid": s["tid"], "domain": domain, "axes": axes, "spec": s}
            if domain == "cube":
                from soarm.mix import train_safe
                try:
                    why = train_safe(s)
                except (KeyError, TypeError, AttributeError):
                    why = None          # malformed goal: the schema stage reports it
                if why:
                    self._reason("heldout_structure")
                    self.rejected_msgs.append(f"- {s['id']}: {why}: reserved for held-out evaluation")
                    self._log({**row, "verify": {"stage": "heldout_structure", "errors": [why]}})
                    continue
            res = verify.verify([s], n_layouts=self.a.layouts, max_draws=self.a.max_draws,
                                min_solved=self.a.min_solved, workers=self.a.workers, config=self.config)[0]
            row["verify"] = {k: v for k, v in res.items() if k not in ("languages",)}
            stage = res["stage"]
            if stage == "schema":
                self._reason("schema")
                self.rejected_msgs.append(f"- {s['id']}: schema: {'; '.join(res['errors'])[:200]}")
            elif stage == "layouts":
                key = "layouts:needs_teacher" if res.get("needs_teacher") else "layouts"
                self._reason(key)
                reasons = res["layouts"]["reasons"]
                top = max(reasons, key=reasons.get) if reasons else "?"
                self.rejected_msgs.append(f"- {s['id']}: no valid layouts ({top}: plan points outside the arm's reach, "
                                          f"objects not fitting, or goal already true at start)")
            elif stage == "physics":
                self._reason("physics")
                self.rejected_msgs.append(f"- {s['id']}: scripted execution failed in physics "
                                          f"({res['physics_summary']['solved']}/{res['physics_summary']['layouts']})")
            if stage in ("layouts", "physics", "accepted"):
                dom["schema"] += 1
                self.stats["schema"] += 1
            if stage in ("physics", "accepted"):
                dom["layouts"] += 1
                self.stats["layouts"] += 1
            if stage == "accepted":
                dom["physics"] += 1
                self.stats["physics"] += 1
                ok, why = self.index.check(s)
                row["dedupe"] = why
                if not ok:
                    self._reason("duplicate")
                    self.rejected_msgs.append(f"- {s['id']}: duplicate ({why})")
                else:
                    s["difficulty"] = res["difficulty"]
                    dom["dedupe"] += 1
                    dom["accepted"] += 1
                    self.stats["dedupe"] += 1
                    self.stats["accepted"] += 1
                    self.index.add(s)
                    self.accepted.append(s)
                    row["accepted"] = True
            self._log(row)

    # ------------------------------------------------------------------ language
    def paraphrase_family(self, spec: dict, n: int, renders: int = 2, batch: int = 8) -> dict:
        new = paraphrase.generate(self.llm, spec, n)
        cands = [(t, "own") for t in spec["instructions"]] + [(t, "para") for t in new]
        ok = {t: 0 for t, _ in cands}
        tried = {t: 0 for t, _ in cands}
        kinds: dict[str, int] = {}
        for r in range(renders):
            seed = 9_000_000 + 101 * r            # resolver layouts: outside every data/eval seed range
            try:
                task = taskspec.sample(spec, seed)
            except RuntimeError:
                continue
            rng = np.random.default_rng([seed, 5])
            rendered = []
            for t, _ in cands:
                try:
                    text, used = paraphrase.render(spec, task, t, rng)
                except (KeyError, LookupError, ValueError):
                    continue
                for k in used.values():
                    kinds[k] = kinds.get(k, 0) + 1
                rendered.append((t, text))
            truth = paraphrase.goal_key(task.atoms)
            for i in range(0, len(rendered), batch):
                chunk = rendered[i:i + batch]
                got = paraphrase.resolve(self.llm, task, [x for _, x in chunk], tag=f"resolve:{spec['id']}")
                for (t, _), g in zip(chunk, got):
                    tried[t] += 1
                    ok[t] += int(g == truth)
        keep = [t for t, src in cands if src == "para" and tried[t] and ok[t] == tried[t]]
        own_pass = [t for t, src in cands if src == "own" and tried[t] and ok[t] == tried[t]]
        return {"id": spec["id"], "tid": spec.get("tid"), "generated": len(new), "kept": keep,
                "own_templates": len(spec["instructions"]), "own_pass": own_pass,
                "resolver": {t: [ok[t], tried[t]] for t, _ in cands}, "ref_kinds": kinds}

    def language(self, families: list[dict], n: int) -> None:
        from concurrent.futures import ThreadPoolExecutor

        def one(spec):
            try:
                return self.paraphrase_family(spec, n)
            except BudgetExceeded as exc:
                return {"id": spec["id"], "error": f"budget: {exc}"}
            except Exception as exc:          # one family's failure must not stop the others
                return {"id": spec["id"], "error": repr(exc)[:300]}

        rows = []
        with ThreadPoolExecutor(self.a.lang_threads) as ex:
            for spec, row in zip(families, ex.map(one, families)):
                rows.append(row)
                (self.out / "paraphrases" / f"{spec['id']}.json").write_text(json.dumps(row, indent=1))
                print(f"[lang] {spec['id']}: {len(row.get('kept', []))}/{row.get('generated', 0)} kept, own "
                      f"{len(row.get('own_pass', []))}/{row.get('own_templates', 0)}, ${self.llm.spent:.3f} "
                      f"{row.get('error', '')}", flush=True)
        self.stats["language"] = {
            "families": len(rows), "generated": sum(r.get("generated", 0) for r in rows),
            "kept": sum(len(r.get("kept", [])) for r in rows),
            "own_templates": sum(r.get("own_templates", 0) for r in rows),
            "own_pass": sum(len(r.get("own_pass", [])) for r in rows)}

    def save(self) -> None:
        self.stats.update(llm_usd=round(self.llm.spent, 4), llm_calls=self.llm.calls, llm_tokens=self.llm.tokens,
                          wall_s=round(time.time() - self.t0))
        (self.out / "stats.json").write_text(json.dumps(self.stats, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8661/v1/chat/completions")
    ap.add_argument("--model", default=None, help="chat model behind --url (default taskgen.llm.MODEL)")
    ap.add_argument("--scene-config", default=None, help="SceneConfig JSON the verifier samples and solves in")
    ap.add_argument("--tid0", type=int, default=TID0, help="first tid of this run's accepted specs (1000-9999)")
    ap.add_argument("--budget-usd", type=float, default=3.0)
    ap.add_argument("--cube-target", type=int, default=18)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--per-round", type=int, default=6)
    ap.add_argument("--broad-rounds", type=int, default=3)
    ap.add_argument("--layouts", type=int, default=5)
    ap.add_argument("--max-draws", type=int, default=60)
    ap.add_argument("--min-solved", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--paraphrases", type=int, default=36)
    ap.add_argument("--lang-threads", type=int, default=4)
    ap.add_argument("--skip-families", action="store_true", help="language only, for specs already in --out")
    ap.add_argument("--accepted-language-only", action="store_true",
                    help="paraphrase only this run's accepted families, not every existing cube/legacy family")
    ap.add_argument("--axis-plan", default=None,
                    help="comma-separated axis pairs per round, e.g. tolerant+multi_step,tolerant+counting "
                         "(default AXIS_PLAN; axes: propose.AXES)")
    a = ap.parse_args()
    axis_plan = [p.split("+") for p in a.axis_plan.split(",")] if a.axis_plan else AXIS_PLAN
    bad = {x for p in axis_plan for x in p} - set(propose.AXES)
    if bad:
        raise SystemExit(f"unknown axes {sorted(bad)}")
    run = Run(a)
    try:
        if not a.skip_families:
            for k in range(a.rounds):
                if sum(s["domain"] == "cube" for s in run.accepted) >= a.cube_target:
                    break
                run.round("cube", axis_plan[k % len(axis_plan)], a.per_round)
                run.save()
                print(f"[cube round {k}] accepted {len(run.accepted)}  ${run.llm.spent:.3f}", flush=True)
            for k in range(a.broad_rounds):
                run.round("broad", axis_plan[(k + 3) % len(axis_plan)], a.per_round)
                run.save()
                print(f"[broad round {k}] accepted {len(run.accepted)}  ${run.llm.spent:.3f}", flush=True)
            for s in run.accepted:
                (run.out / "specs" / "gen" / f"{s['id']}.json").write_text(json.dumps(s, indent=1))
        else:
            run.accepted = [json.loads(p.read_text()) for p in sorted((run.out / "specs" / "gen").glob("*.json"))]
        existing = [] if a.accepted_language_only else \
            [s for s in taskspec.registry().values() if s["domain"] in ("cube", "legacy")]
        run.language(run.accepted + existing, a.paraphrases)
        for s in run.accepted:      # accepted families carry their resolver-checked paraphrases (CONTRACT 2.4)
            row = json.loads((run.out / "paraphrases" / f"{s['id']}.json").read_text())
            own = row.get("own_pass") or s["instructions"]
            s["instructions"] = own + row.get("kept", [])
            (run.out / "specs" / "gen" / f"{s['id']}.json").write_text(json.dumps(s, indent=1))
    except BudgetExceeded as exc:
        print(f"stopped: {exc}", flush=True)
    finally:
        run.save()


if __name__ == "__main__":
    main()

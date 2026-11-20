"""Verifier pipeline for task-family specs (CONTRACT 2 JSON):

  1 schema      soarm.taskspec.validate + cube-domain rules (1-5 cubes, <= 1 container, <= 3 mats/bins, >= 5 templates)
  2 layouts     draw layouts with soarm.taskspec.sample until N pass: every object starts inside top-down reach, no goal
                atom true at the (ideal) start, a scripted plan exists whose every pick/carry/place point is reachable
  3 physics     settle in MuJoCo (nothing moved/tipped, no atom true: SoarmSim rule) + execute the plan with scripted
                top-down grasps through SoarmDriver.command; solvable = >= `min_solved` of N layouts reach the goal
  4 difficulty  from plan length, objects, references and the scripted success rate
Dedupe (taskgen.dedupe) runs over the accepted set afterwards.

    python -m taskgen.verify SPECS.jsonl|DIR --out results.jsonl [--layouts 5] [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from soarm import taskspec
from soarm.tasks import Reject, SceneConfig, evaluate_goal

from .planner import NoPlan, check_start_reach, plan, state_from

CUBE_KINDS = {"cube", "bowl", "box", "pad", "bin", "slot", "tray", "big_bowl", "big_mat"}
CONTAINERS = {"bowl", "box", "slot", "tray", "big_bowl"}


def _kinds(o):
    return o["kind"] if isinstance(o["kind"], list) else [o["kind"]]


def _max(c):
    return 1 if c is None else c if isinstance(c, int) else c[1]


def check_schema(spec: dict, min_templates: int = 5) -> list[str]:
    errs = []
    try:
        taskspec.validate(spec)
    except (ValueError, KeyError) as exc:
        errs.append(str(exc))
        return errs
    if spec["domain"] == "cube":
        objs = spec["objects"]
        bad = {k for o in objs for k in _kinds(o)} - CUBE_KINDS
        if bad:
            errs.append(f"kinds outside the cube domain: {sorted(bad)}")
        n_cubes = sum(_max(o.get("count")) for o in objs if _kinds(o) == ["cube"])
        if not 1 <= n_cubes <= 5:
            errs.append(f"cube domain allows 1-5 cubes, spec allows up to {n_cubes}")
        n_cont = sum(_max(o.get("count")) for o in objs if set(_kinds(o)) & CONTAINERS)
        if n_cont > 1:
            errs.append(f"at most one container, spec allows {n_cont}")
        n_mats = sum(_max(o.get("count")) for o in objs if set(_kinds(o)) & {"pad", "bin", "big_mat"})
        if n_mats > 3:
            errs.append(f"at most 3 mats/bins, spec allows {n_mats}")
    if len(spec["instructions"]) < min_templates:
        errs.append(f"{len(spec['instructions'])} instruction templates < {min_templates}")
    return errs


def draw_layouts(spec: dict, n: int, max_draws: int, seed0: int = 0,
                 config: SceneConfig | None = None) -> tuple[list, dict]:
    """Accepted (seed, task, n_steps) and rejection counts per reason (stage 2)."""
    ok, reasons, draws = [], {}, 0
    seed = seed0
    while len(ok) < n and draws < max_draws:
        draws += 1
        try:
            task = taskspec.sample(spec, seed * 1000, config, max_tries=1)
            check_start_reach(task)
            if any(evaluate_goal(state_from(task).world(), task.atoms)):
                raise Reject("goal atom holds at the start")
            steps = plan(task)
            ok.append((seed, task, len(steps)))
        except NoPlan as exc:
            key = "no_plan:" + str(exc).split(":")[0][:60] if "no scripted plan" in str(exc) else "unreachable_or_blocked"
            reasons[key] = reasons.get(key, 0) + 1
        except (Reject, RuntimeError) as exc:
            key = "sample:" + str(exc).split("(")[-1].rstrip(")")[:50]
            reasons[key] = reasons.get(key, 0) + 1
        seed += 1
    return ok, {"draws": draws, "accepted": len(ok), "reasons": reasons}


def _physics(job):
    import pickle
    spec_id, seed, task_json, config = job
    from .solve import run_instance
    task = _task_from_json(task_json)
    try:
        return spec_id, seed, run_instance(task, seed, pickle.loads(config))
    except Exception as exc:          # a crash in physics is a verifier failure of this layout, not of the pipeline
        return spec_id, seed, {"stage": "error", "ok": False, "detail": repr(exc)[:300]}


def _task_from_json(d):
    import pickle
    return pickle.loads(d)


def difficulty(spec: dict, steps: list[int], solve_rate: float) -> int:
    """1 (one pick/place) .. 5 (long multi-step): mean plan length, plus one for a low scripted success rate, plus one
    for conditional/count/ordinal structure."""
    m = float(np.mean(steps)) if steps else 1.0
    d = 1 if m <= 1.2 else 2 if m <= 2.2 else 3 if m <= 3.5 else 4
    if solve_rate < 0.7:
        d += 1
    if any("count" in g for g in spec["goal"]) or any(g.get("params", {}).get("order") for g in spec["goal"]):
        d += 1
    return int(min(5, d))


def verify(specs: list[dict], n_layouts: int = 5, max_draws: int = 60, min_solved: int = 3, workers: int = 8,
           seed0: int = 0, min_templates: int = 5, config: SceneConfig | None = None) -> list[dict]:
    import pickle
    rows = {}
    jobs = []
    for spec in specs:
        sid = spec.get("id", "?")
        row = {"id": sid, "tid": spec.get("tid"), "stage": "schema", "ok": False}
        rows[sid] = row
        errs = check_schema(spec, min_templates)
        if errs:
            row["errors"] = errs
            continue
        row["stage"] = "layouts"
        ok, stats = draw_layouts(spec, n_layouts, max_draws, seed0, config)
        row["layouts"] = stats
        if len(ok) < n_layouts:
            row["errors"] = [f"only {len(ok)}/{n_layouts} valid layouts in {stats['draws']} draws"]
            if any(k.startswith("no_plan") for k in stats["reasons"]):
                row["needs_teacher"] = True
            continue
        row["stage"] = "physics"
        row["plan_steps"] = [s for _, _, s in ok]
        row["languages"] = [t.language for _, t, _ in ok]
        for seed, task, _ in ok:
            jobs.append((sid, seed, pickle.dumps(task), pickle.dumps(config)))
    if jobs:
        with mp.get_context("fork").Pool(workers) as pool:
            for sid, seed, res in pool.imap_unordered(_physics, jobs):
                rows[sid].setdefault("physics", []).append({"seed": seed, **{k: v for k, v in res.items() if k != "detail"},
                                                            "detail": str(res.get("detail"))[:300]})
    by_id = {s.get("id"): s for s in specs}
    for sid, row in rows.items():
        if row["stage"] != "physics":
            continue
        ph = row["physics"]
        starts_ok = [p for p in ph if p["stage"] != "start"]
        solved = sum(p["ok"] for p in ph)
        row["physics_summary"] = {"layouts": len(ph), "start_rejected": len(ph) - len(starts_ok), "solved": solved}
        if solved < min_solved:
            row["errors"] = [f"scripted solve {solved}/{len(ph)} < {min_solved}"]
            continue
        rate = solved / max(1, len(starts_ok))
        row["difficulty"] = difficulty(by_id[sid], row["plan_steps"], rate)
        row["stage"], row["ok"] = "accepted", True
    return list(rows.values())


def load_specs(path: str) -> list[dict]:
    p = Path(path)
    if p.is_dir():
        return [json.loads(f.read_text()) for f in sorted(p.glob("*.json"))]
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layouts", type=int, default=5)
    ap.add_argument("--max-draws", type=int, default=60)
    ap.add_argument("--min-solved", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--min-templates", type=int, default=5, help="generated specs: 5; hand specs: 1")
    ap.add_argument("--scene-config", default=None, help="SceneConfig JSON (default: the default scene)")
    a = ap.parse_args()
    t0 = time.time()
    config = SceneConfig.from_json(json.loads(a.scene_config)) if a.scene_config else None
    rows = verify(load_specs(a.specs), a.layouts, a.max_draws, a.min_solved, a.workers, a.seed0, a.min_templates,
                  config=config)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for r in rows:
        print(f"{r['id']:<34} {r['stage']:<9} {'OK ' if r['ok'] else 'NO '} "
              f"{r.get('layouts', {}).get('accepted', '-')}/{r.get('layouts', {}).get('draws', '-')} "
              f"{r.get('physics_summary', '')} d={r.get('difficulty', '-')} {'; '.join(r.get('errors', []))[:150]}")
    print(f"{sum(r['ok'] for r in rows)}/{len(rows)} accepted, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

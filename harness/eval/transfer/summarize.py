#!/usr/bin/env python3
"""Score a transfer-benchmark run directory (written by eval/transfer/run_jobs.py) with each benchmark's own rules.

    python3 eval/transfer/summarize.py robotwin2  runs/captain-4b/robotwin2  [--json out.json]
    python3 eval/transfer/summarize.py simplerenv runs/captain-4b/simplerenv [--json out.json]
    python3 eval/transfer/summarize.py robodojo   runs/captain-4b/robodojo   [--json out.json]
    python3 eval/transfer/summarize.py capbench   runs/captain-4b/capbench   [--json out.json]
    python3 eval/transfer/summarize.py robotwin2  --rows rows.json            # re-aggregate saved per-episode rows

Metrics (all host-side; the model never sees them):
- robotwin2: ever = RoboTwin's check_success true after some physical action (latched, like the official eval loop that
  stops at the first success); host = final state; strict = host and the agent's own finish(). Restricted to the official
  Hard seed list (robotwin2/official_hard_seeds.tsv) when the run used it; "single-arm" = the 33 tasks whose expert
  program moves one arm at a time and holds each object with one arm (robotwin2/task_arm_classes.json).
- simplerenv: host = SimplerEnv's evaluate() at the end of the episode; horizon = the verdict at the official
  max_episode_steps of simulated time; ever = success at any official control period; grasp = is_src_obj_grasped at any
  step (Bridge partial metric); aggregation as SimplerEnv's calc_metrics_evaluation_videos.py.
- robodojo: official = RoboDojo's own verdict and Score at the task's native action budget; unlimited = the same episode
  continued up to 20x the budget. Task mean, dimension mean over tasks, overall = mean of the 5 dimensions; single-arm =
  the 32 tasks that need no simultaneous two-arm action, handover or navigation.
- capbench: host = the robosuite env's _check_success() on the final state; capped = the predicate at CaP-Gym's step
  horizon (what CaP-Gym scores); strict = host and finish().
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, 100 * (c - h)), 1), round(min(100.0, 100 * (c + h)), 1))


def episodes(run: Path):
    for d in sorted(p for p in run.iterdir() if p.is_dir() and not p.name.startswith("_")):
        f = d / "episode_result.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        if r.get("termination_reason") == "infrastructure_error":
            continue
        task_file = run / "_tasks" / f"{d.name}.json"
        task = json.loads(task_file.read_text(encoding="utf-8")) if task_file.exists() else {}
        yield d, r, task


def pct(rows, key) -> float:
    return round(100.0 * sum(bool(x[key]) for x in rows) / len(rows), 1) if rows else float("nan")


# ----------------------------------------------------------------------------------------------------------- RoboTwin 2.0
def robotwin2_rows(run: Path) -> list[dict]:
    rows = []
    for d, r, _ in episodes(run):
        reqs = [json.loads(line) for line in (d / "driver" / "requests.jsonl").read_text().splitlines() if line.strip()] \
            if (d / "driver" / "requests.jsonl").exists() else []
        host = bool(r.get("host_final_success"))
        rows.append({"instance": d.name, "task": d.name.rsplit("_s", 1)[0], "ever": bool(r.get("host_success_ever")),
                     "host": host, "strict": host and bool(r.get("agent_finished")) and not r.get("false_finish"),
                     "finished": bool(r.get("agent_finished")), "false_finish": bool(r.get("false_finish")),
                     "term": r.get("termination_reason"), "requests": len(reqs), "wall_s": r.get("wall_time_s")})
    return rows


def robotwin2_report(rows: list[dict]) -> dict:
    classes = {t: v["class"] for t, v in json.loads((HERE / "robotwin2" / "task_arm_classes.json").read_text()).items()}
    official = set()
    for line in (HERE / "robotwin2" / "official_hard_seeds.tsv").read_text().splitlines():
        if line and not line.startswith("#"):
            task, seed, _ = line.split("\t")
            official.add(f"{task}_s{seed}")
    inside = [x for x in rows if x["instance"] in official]
    out = {"episodes_outside_official_list": len(rows) - len(inside) if inside else None}
    if inside:   # an official Hard run: score exactly the official seed list
        rows = inside
    for label, sel in (("all", lambda x: True), ("single_arm", lambda x: classes.get(x["task"]) == "single")):
        sub = [x for x in rows if sel(x)]
        out[label] = {"n": len(sub), "tasks": len({x["task"] for x in sub}), "ever": pct(sub, "ever"),
                      "ever_ci95": wilson(sum(x["ever"] for x in sub), len(sub)), "host": pct(sub, "host"),
                      "strict": pct(sub, "strict"), "finish_called": sum(x["finished"] for x in sub),
                      "false_finish": sum(x["false_finish"] for x in sub)}
    per_task = defaultdict(list)
    for x in rows:
        per_task[x["task"]].append(x)
    out["per_task_ever"] = {t: pct(v, "ever") for t, v in sorted(per_task.items())}
    return out


# ---------------------------------------------------------------------------------------------------------- SimplerEnv
def simplerenv_report(run: Path) -> dict:
    import numpy as np
    rows = []
    for d, r, t in episodes(run):
        ef = d / "simplerenv" / "evaluation.json"
        if not ef.exists() or not t:
            continue
        e = json.loads(ef.read_text())
        ev = e["eval"]
        st = ev.get("episode_stats", {})
        host = bool(ev.get("success"))
        hz = e.get("horizon_eval")
        rows.append(dict(key=d.name, setting=t["setting"], group=t["group"], variant=t["variant"], env=t["env_name"],
                         kw=t["simplerenv"]["additional_env_build_kwargs"], official=t["official_aggregate"],
                         host=host, ever=bool(e["ever_success"]), strict=host and bool(r.get("agent_finished")),
                         horizon=bool(hz["success"]) if hz else host, finished=bool(r.get("agent_finished")),
                         grasp=bool(st.get("is_src_obj_grasped") or st.get("grasped"))))
    metrics = ("host", "ever", "strict", "horizon", "grasp")

    def rate(rs, m):
        return float(np.mean([x[m] for x in rs])) if rs else float("nan")

    def strip(x, drop):
        return "|".join(p for p in x["variant"].split("|") if not any(p.startswith(k) for k in drop))

    def orient(x):
        kw = x["kw"]
        return "horizontal" if kw.get("lr_switch") else "vertical" if kw.get("laid_vertically") else "standing" if kw.get("upright") else "?"

    def variant_mean(rs, m, drop):
        vs = defaultdict(list)
        for x in rs:
            vs[strip(x, drop)].append(x)
        return float(np.mean([rate(v, m) for v in vs.values()])) if vs else float("nan")

    out: dict = {}
    by = defaultdict(list)
    for x in rows:
        if x["official"]:
            by[(x["setting"], x["group"])].append(x)
    for (setting, group), rs in sorted(by.items()):
        res: dict = {"n": len(rs)}
        for m in metrics:
            if setting == "bridge_vm":
                res[m] = rate(rs, m)
            elif group == "pick_coke_can":   # mean over the 3 orientations
                per = {o: (variant_mean([x for x in rs if orient(x) == o], m, ("lr_switch", "laid_vertically", "upright"))
                           if setting == "google_va" else rate([x for x in rs if orient(x) == o], m))
                       for o in ("horizontal", "vertical", "standing")}
                res[m] = float(np.nanmean(list(per.values())))
            elif group == "drawer":          # mean of open and close
                per = {k: (variant_mean([x for x in rs if x["env"].startswith(k)], m, ("Open", "Close"))
                           if setting == "google_va" else rate([x for x in rs if x["env"].startswith(k)], m))
                       for k in ("Open", "Close")}
                res[m] = float(np.nanmean(list(per.values())))
            else:
                res[m] = variant_mean(rs, m, ()) if setting == "google_va" else rate(rs, m)
        out[f"{setting}/{group}"] = {k: (round(100 * v, 1) if isinstance(v, float) else v) for k, v in res.items()}
    for setting in ("bridge_vm", "google_vm", "google_va"):
        gs = [v for k, v in out.items() if k.startswith(setting + "/")]
        if gs:
            out[f"{setting}/AVERAGE"] = {m: round(float(np.mean([g[m] for g in gs])), 1) for m in metrics} | {"n": sum(g["n"] for g in gs)}
    out["_counts"] = {"episodes": len(rows), "finished": sum(x["finished"] for x in rows),
                      "false_finish": sum(x["finished"] and not x["host"] for x in rows)}
    return out


# ------------------------------------------------------------------------------------------------------------ RoboDojo
DIMS = ["Generalization", "Precision", "Long-Horizon", "Memory", "Open"]
SINGLE = set("""stack_bowls push_T pack_objects_into_box fold_clothes hang_mugs pour_liquid_into_cup make_toast
arrange_largest_number sort_nesting_dolls_by_size store_laptop_and_headphones stack_blocks cover_blocks
match_and_pick_from_conveyor swap_blocks press_by_number plug_in_charger insert_tubes pour_balls_into_vase play_Xylophone
deposit_coin build_tower classify_objects fill_egg_holder organize_table play_stacking_toy align_blocks general_pickup
solve_equation stack_blocks_by_language classify_objects_by_language store_tools_in_toolbox pour_by_language""".split())


def robodojo_report(run: Path) -> dict:
    per, dim_of = defaultdict(list), {}
    for d, r, t in episodes(run):
        hp = d / "robodojo" / "host_status.json"
        if not hp.exists() or not t:
            continue
        h = json.loads(hp.read_text())
        task = t["leaderboard_task"]
        dim_of[task] = t["dimension"]
        nv = h.get("native_verdict") or {}
        u_succ = bool(h.get("official_success"))
        n_succ = bool(nv.get("success"))
        per[task].append((n_succ, 1.0 if n_succ else float(nv.get("score") or 0.0),
                          u_succ, 1.0 if u_succ else float(h.get("official_score") or 0.0)))
    cols = ("official_sr", "official_score", "unlimited_sr", "unlimited_score")

    def agg(rs):
        return [100 * sum(x[i] for x in rs) / len(rs) for i in range(4)]

    def mean_over(tasks):
        return {c: round(sum(agg(per[t])[i] for t in tasks) / len(tasks), 1) for i, c in enumerate(cols)} if tasks else None

    out = {"per_task": {t: {"dimension": dim_of[t], "n": len(v), **dict(zip(cols, (round(a, 1) for a in agg(v))))}
                        for t, v in sorted(per.items())}}
    dims = {dm: mean_over([t for t in per if dim_of[t] == dm]) for dm in DIMS}
    out["per_dimension"] = {k: v for k, v in dims.items() if v}
    if out["per_dimension"]:
        vals = list(out["per_dimension"].values())
        out["overall_mean_of_dimensions"] = {c: round(sum(v[c] for v in vals) / len(vals), 1) for c in cols}
    out["all_tasks_mean"] = mean_over(list(per))
    out["single_arm_mean"] = mean_over([t for t in per if t in SINGLE])
    out["_counts"] = {"tasks": len(per), "episodes": sum(len(v) for v in per.values())}
    return out


# ----------------------------------------------------------------------------------------------------------- CaP-Bench
def capbench_report(run: Path) -> dict:
    rows = defaultdict(list)
    for d, r, _ in episodes(run):
        task = d.name.rsplit("_", 2)[0]
        host = bool(r.get("host_final_success"))
        prog = d / "capgym_progress.jsonl"
        caps = [x["capped"] for x in (json.loads(line) for line in prog.read_text().splitlines() if line.strip())
                if x.get("capped")] if prog.exists() else []
        capped = bool(caps[0]["success"]) if caps else host   # never reached the horizon: the final state counts
        rows[task].append({"host": host, "capped": capped, "finished": bool(r.get("agent_finished"))})
    out = {}
    for task, rs in sorted(rows.items()):
        n = len(rs)
        h, c, s = (sum(x["host"] for x in rs), sum(x["capped"] for x in rs), sum(x["host"] and x["finished"] for x in rs))
        out[task] = {"n": n, "host": round(100 * h / n, 1), "host_ci95": wilson(h, n), "capped": round(100 * c / n, 1),
                     "capped_ci95": wilson(c, n), "strict": round(100 * s / n, 1),
                     "false_finish": sum(x["finished"] and not x["host"] for x in rs)}
    lift_stack = [out[t] for t in ("capgym_lift", "capgym_stack") if t in out]
    if lift_stack:
        out["lift_stack_mean"] = {m: round(sum(v[m] for v in lift_stack) / len(lift_stack), 1) for m in ("host", "capped", "strict")}
    if len([t for t in out if t.startswith("capgym_")]) == 5:
        five = [out[t] for t in out if t.startswith("capgym_")]
        out["five_task_mean"] = {m: round(sum(v[m] for v in five) / 5, 1) for m in ("host", "capped", "strict")}
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("benchmark", choices=("robotwin2", "simplerenv", "robodojo", "capbench"))
    p.add_argument("run", type=Path, nargs="?")
    p.add_argument("--rows", type=Path, default=None, help="robotwin2: aggregate a saved per-episode rows JSON instead of a run dir")
    p.add_argument("--json", type=Path, default=None, help="also write the report (and per-episode rows) here")
    args = p.parse_args()
    if args.benchmark == "robotwin2":
        rows = json.loads(args.rows.read_text()) if args.rows else robotwin2_rows(args.run)
        report = {"summary": robotwin2_report(rows), "rows": rows}
        print(json.dumps(report["summary"], indent=1))
    else:
        if args.run is None:
            p.error("run directory required")
        report = {"simplerenv": simplerenv_report, "robodojo": robodojo_report, "capbench": capbench_report}[args.benchmark](args.run)
        print(json.dumps(report, indent=1))
    if args.json:
        args.json.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

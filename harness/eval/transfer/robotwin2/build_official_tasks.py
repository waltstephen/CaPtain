#!/usr/bin/env python3
"""Task jsons of the official RoboTwin 2.0 eval: per task, the first 100 seeds (from 100000, ascending) that pass RoboTwin's
own expert check, with RoboTwin's own eval instruction. usage: build_official_tasks.py SEEDS_DIR TASK_CONFIG OUT_DIR [N]"""
import json, sys, glob, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from seedrows import load
sd, cfg, out = sys.argv[1:4]; N = int(sys.argv[4]) if len(sys.argv) > 4 else 100
os.makedirs(out, exist_ok=True); summ = {}
for f in sorted(glob.glob(f"{sd}/*.{cfg}.jsonl")):
    task = os.path.basename(f).split(".")[0]
    rows = load(f)
    seeds = sorted(rows); valid = [s for s in seeds if rows[s]["valid"]]
    # contiguous coverage: only count valid seeds below the first unchecked seed
    lo = 100000; checked = set(seeds); hi = lo
    while hi in checked: hi += 1
    take = [s for s in valid if s < hi][:N]
    summ[task] = {"checked_contiguous_to": hi, "valid": len([s for s in valid if s < hi]), "taken": len(take),
                  "reasons": {r: sum(1 for s in seeds if rows[s]["reason"] == r) for r in ("expert_failed", "unstable", "expert_error")}}
    if len(take) < N and not os.environ.get("PARTIAL"): continue
    for s in take:
        r = rows[s]
        json.dump({"robotwin_task": task, "robotwin_seed": s, "language": r["instruction"], "task_config": cfg,
                   "layout_embodiment": "aloha-agilex", "allow_unstable": True, "instruction_type": r["instruction_type"],
                   "instance_source": "official RoboTwin 2.0 eval protocol (scripts/eval_policy_xpolicylab.py): first 100 seeds from 100000 passing RoboTwin's expert check (aloha-agilex, cuRobo), RoboTwin's build_instruction"},
                  open(f"{out}/{task}_s{s}.json", "w"), indent=1)
json.dump(summ, open(f"{out}/_summary.json", "w"), indent=1)
print(sum(1 for v in summ.values() if v["taken"] >= N), "tasks complete;", {k: (v["valid"], v["checked_contiguous_to"]) for k, v in summ.items() if v["taken"] < N})

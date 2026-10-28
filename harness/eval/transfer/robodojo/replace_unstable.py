#!/usr/bin/env python3
"""Replace layouts that RoboDojo rejects as unstable, as RoboDojo's eval_client does.

eval_client takes, per task, the layouts in index order and skips a layout whose scene fails RoboDojo's own stability
check (UnStableError) without counting it, until eval_nums layouts have been evaluated. In a CaPtain run such an episode
ends as an infrastructure error whose server_error.json names UnStableError. This script lists those episodes and writes
replacement tasks (the task's next unused layout index of RoboDojo's Eval_Layout set) that can be run with run_jobs.py.

    python3 eval/transfer/robodojo/replace_unstable.py RUN_DIR $ROBODOJO_ROOT/Assets/Eval_Layout/RoboDojo/arx_x5/0 > extra.jsonl
"""
import glob
import json
import os
import sys


def main(run: str, layouts: str) -> int:
    tasks = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(run, "_tasks", "*.json"))}
    used = {}
    for name, t in tasks.items():
        used.setdefault(t["robodojo_task"], set()).add(int(t["layout_id"]))
    for name, t in sorted(tasks.items()):
        candidates = [os.path.join(run, name)] + sorted(glob.glob(os.path.join(run, f"_{name}.infra-*")))
        unstable = False
        for d in candidates:
            err = os.path.join(d, "server_error.json")
            if os.path.exists(err) and "UnStable" in open(err).read():
                unstable = True
        result = os.path.join(run, name, "episode_result.json")
        if not unstable or (os.path.exists(result) and json.load(open(result)).get("termination_reason") != "infrastructure_error"):
            continue
        task = t["robodojo_task"]
        avail = sorted(int(p.rsplit("_", 1)[1][:-5]) for p in glob.glob(os.path.join(layouts, f"{task}_*.json"))
                       if p.rsplit("_", 1)[1][:-5].isdigit())
        nxt = next((i for i in avail if i not in used[task]), None)
        if nxt is None:
            print(f"{name}: unstable, no spare layout left", file=sys.stderr)
            continue
        used[task].add(nxt)
        new = dict(t, layout_id=nxt, name=f"{task}_L{nxt}")
        new["layout_note"] = t.get("layout_note", "") + f"; replaces layout {t['layout_id']} (RoboDojo UnStableError), as eval_client does"
        print(json.dumps(new))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))

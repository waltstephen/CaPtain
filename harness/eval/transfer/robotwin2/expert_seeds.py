#!/usr/bin/env python3
"""Official RoboTwin 2.0 eval seed check, reproduced with RoboTwin's OWN code (scripts/eval_policy_xpolicylab.py):
for each seed, setup_demo(seed, is_test=True) in the task config's embodiment (aloha-agilex), run the task's expert
play_once with RoboTwin's cuRobo planner, keep the seed iff plan_success and check_success (UnStableError / expert
error -> skipped), and build the eval instruction exactly as the eval script does (build_instruction: unseen for
demo_randomized, seen for demo_clean). One JSON line per seed.
usage (cwd = RoboTwin root): expert_seeds.py TASK TASK_CONFIG SEED_FROM SEED_TO OUT.jsonl"""
import json, sys, time, traceback
sys.path.insert(0, "scripts")
import eval_policy_xpolicylab as E

task, cfg, s0, s1, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
usr = {"task_name": task, "task_config": cfg, "policy_name": "captain_harness", "seed": 0}
args, emb = E.load_task_args(usr)
itype = E.resolve_eval_instruction(usr, args)
args["eval_instruction"] = itype
args["render_freq"] = 0
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seedrows import load
done = set(load(out))   # infra-error rows do not count as checked
f = open(out, "a")
for seed in range(s0, s1):
    if seed in done:
        continue
    t0 = time.time(); row = {"task": task, "task_config": cfg, "seed": seed, "embodiment": emb, "instruction_type": itype}
    env = E.class_decorator(task)
    try:
        env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
        info = env.play_once()
        env.close_env()
        ok = bool(env.plan_success and env.check_success())
        row.update(valid=ok, reason=None if ok else "expert_failed", info=(info or {}).get("info", {}))
        if ok:
            row["instruction"] = E.build_instruction(args, info or {"info": {}}, itype, 100, seed)
    except E.UnStableError:
        row.update(valid=False, reason="unstable")
        try: env.close_env()
        except Exception: pass
    except Exception as e:
        row.update(valid=False, reason="expert_error", error=f"{type(e).__name__}: {e}"[:300], tb=traceback.format_exc()[-1500:])
        try: env.close_env()
        except Exception: pass
    row["s"] = round(time.time() - t0, 1)
    f.write(json.dumps(row, default=str) + "\n"); f.flush()

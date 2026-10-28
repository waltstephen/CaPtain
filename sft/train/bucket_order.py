#!/usr/bin/env python3
"""Length-bucketed order for SFT_SAMPLING=sequential: sort rows by exact length (meta.total_tokens), cut into consecutive groups of
--world rows (one optimizer step = one row per rank, micro 1 / accum 1), shuffle the GROUPS (seeded) and write them out. Every step
then holds rows of similar length (ZeRO-3 waits for the slowest rank); the step order over the epoch stays random."""
import argparse, json, random
ap = argparse.ArgumentParser(); ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--world", type=int, required=True); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
rows = open(a.inp).read().splitlines()
L = [json.loads(r)["meta"]["total_tokens"] for r in rows]
order = sorted(range(len(rows)), key=lambda i: L[i])
groups = [order[i:i + a.world] for i in range(0, len(order), a.world)]
random.Random(a.seed).shuffle(groups)
with open(a.out, "w") as f:
    for g in groups:
        for i in g: f.write(rows[i] + "\n")
mx = [max(L[i] for i in g) for g in groups]; mean = [sum(L[i] for i in g) / len(g) for g in groups]
print(json.dumps({"rows": len(rows), "steps": len(groups), "sum_step_max_tokens": sum(mx), "sum_step_mean_tokens": round(sum(mean)),
                  "random_order_expected_step_max_ratio": "see log"}))

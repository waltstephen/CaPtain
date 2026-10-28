#!/usr/bin/env python3
"""Mixed-step order for SFT_SAMPLING=sequential: every optimizer step (one row per rank, --world rows) holds both agent rows and
QA/general packed rows in the global ratio (e.g. 11-12 agent + 4-5 QA of 16), so no step is all short answers. Within each type
rows are length-sorted and dealt to steps in order (similar lengths per step -> little ZeRO-3 straggler wait); step order shuffled."""
import argparse, json, random
ap = argparse.ArgumentParser(); ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--world", type=int, required=True); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
rows = open(a.inp).read().splitlines()
meta = [json.loads(r)["meta"] for r in rows]
qa = sorted([i for i, m in enumerate(meta) if m.get("source") == "qapack"], key=lambda i: meta[i]["total_tokens"])
ag = sorted([i for i, m in enumerate(meta) if m.get("source") != "qapack"], key=lambda i: meta[i]["total_tokens"])
S = (len(rows) + a.world - 1) // a.world
steps, qi, ai = [], 0, 0
for s in range(S):
    nq = (s + 1) * len(qa) // S - s * len(qa) // S
    na = min(a.world - nq, len(ag) - ai)
    steps.append(qa[qi:qi + nq] + ag[ai:ai + na]); qi += nq; ai += na
assert qi == len(qa) and ai == len(ag), (qi, ai)
random.Random(a.seed).shuffle(steps)
with open(a.out, "w") as f:
    for st in steps:
        for i in st: f.write(rows[i] + "\n")
mix = [sum(meta[i].get("source") == "qapack" for i in st) for st in steps]
print(json.dumps({"rows": len(rows), "steps": S, "qa_per_step_min_max": [min(mix), max(mix)], "agent_rows": len(ag), "qa_rows": len(qa),
                  "sum_step_max_tokens": sum(max(meta[i]["total_tokens"] for i in st) for st in steps)}))

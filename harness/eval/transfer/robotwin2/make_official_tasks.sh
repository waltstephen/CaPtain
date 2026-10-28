#!/usr/bin/env bash
# Build the task files of the official RoboTwin 2.0 Hard evaluation (5,000 episodes) with RoboTwin's own code.
#
#   ROBOTWIN_ROOT=... ROBOTWIN_PYTHON=... eval/transfer/robotwin2/make_official_tasks.sh <work_dir> [task_config=demo_randomized]
#
# For every task, expert_seeds.py runs RoboTwin's expert (aloha-agilex, cuRobo planner; needs a GPU) on each seed from
# 100000 up to the largest seed in official_hard_seeds.tsv, keeps a seed iff the expert succeeds (unstable / expert-error
# seeds are skipped, as in RoboTwin's eval script) and records RoboTwin's eval instruction for it. build_official_tasks.py
# then takes the first 100 valid seeds per task and writes <work_dir>/tasks/<task>_s<seed>.json. The script finally checks
# that the selected seeds equal official_hard_seeds.tsv (the list the paper's numbers were measured on).
set -euo pipefail
W=${1:?work_dir}; CFG=${2:-demo_randomized}
HERE=$(cd "$(dirname "$0")" && pwd)
: "${ROBOTWIN_ROOT:?}" "${ROBOTWIN_PYTHON:?}"
mkdir -p "$W/seeds" "$W/tasks"
python3 - "$HERE/official_hard_seeds.tsv" > "$W/ranges.txt" <<'PY'
import sys, collections
hi = collections.defaultdict(int)
for line in open(sys.argv[1]):
    if line.strip() and not line.startswith("#"):
        t, s, _ = line.split("\t"); hi[t] = max(hi[t], int(s))
for t in sorted(hi): print(t, hi[t] + 1)
PY
while read -r TASK END; do
  (cd "$ROBOTWIN_ROOT" && "$ROBOTWIN_PYTHON" "$HERE/expert_seeds.py" "$TASK" "$CFG" 100000 "$END" "$W/seeds/$TASK.$CFG.jsonl")
done < "$W/ranges.txt"
python3 "$HERE/build_official_tasks.py" "$W/seeds" "$CFG" "$W/tasks" 100
python3 - "$HERE/official_hard_seeds.tsv" "$W/tasks" <<'PY'
import glob, json, os, sys
want = {tuple(l.split("\t")[:2]) for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")}
have = {(json.load(open(p))["robotwin_task"], str(json.load(open(p))["robotwin_seed"])) for p in glob.glob(os.path.join(sys.argv[2], "*.json")) if not os.path.basename(p).startswith("_")}
print(f"official list {len(want)}, built {len(have)}, missing {len(want - have)}, extra {len(have - want)}")
sys.exit(0 if want == have else 1)
PY

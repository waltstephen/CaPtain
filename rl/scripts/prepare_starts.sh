#!/usr/bin/env bash
# Build the training start pool + segment bank from the harness's fresh initial states (seeds >= 1000; the official
# evaluation states 11-16 can never be drawn). Writes $CAPTAIN_WORKSPACE/configs/start-bank.json and start-pool.json, the paths
# the captain-4b configs expect. Run with the harness venv (LIBERO importable):
#   bash rl/scripts/prepare_starts.sh [fresh_dir=$CAPTAIN_ROOT/harness/data/libero_fresh_init] [min_per_task=50]
# For an exact reproduction use the reference start states (7,360 starts, 184 per task; see rl/README.md) and check that the
# printed bank_sha256 equals rl/configs/start-pool.reference.json["bank_sha256"].
set -euo pipefail
source "$(dirname "$0")/env.sh"
FRESH=${1:-$CAPTAIN_ROOT/harness/data/libero_fresh_init}; MIN=${2:-50}
mkdir -p "$CAPTAIN_WORKSPACE/configs"
cd "$CAPTAIN_ROOT"
PY=${HARNESS_PY:-$CAPTAIN_ROOT/harness/.venv-libero/bin/python}
LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-$CAPTAIN_ROOT/harness/.libero-config} PYTHONPATH=rl "$PY" rl/fresh_start_bank.py --fresh-dir "$FRESH" \
  --min-per-task "$MIN" --out-bank "$CAPTAIN_WORKSPACE/configs/start-bank.json" --out-manifest "$CAPTAIN_WORKSPACE/configs/start-pool.json"

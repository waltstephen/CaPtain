#!/bin/bash
# Realistic-path eval of a deploy stack (soarm-speed): 15 timing episodes one at a time (seeds 9000-9004) + 30 success
# episodes on 3 lanes (seeds 9005-9014) through tools/policy_proxy.py, which only records every request body (replay /
# drafter data). Request knobs and motion options come from the speed profile (SOARM_PROFILE, configs/speed_<name>.json).
#   tools/stack_eval.sh <name> <proxy port> [extra env assignments for the episodes ...]
#   e.g. SOARM_PROFILE=fast tools/stack_eval.sh fast1 8714
# FAMILIES: comma list (default pick_lift,pick_place_mat,put_in_container).
# Needs: policy server on :8700, SAM3 :8114, Molmo :8122, MoGe :8130 (tools/serve_perception.sh).
source "$(cd "$(dirname "$0")/.." && pwd)/env.sh"
NAME=${1:?name}; PORT=${2:?proxy port}; shift 2
cd "$SOARM_CODE"
mkdir -p "$RUNS_DIR/replay"
python tools/policy_proxy.py --port "$PORT" --upstream http://127.0.0.1:8700 \
  --record "$RUNS_DIR/replay/rec_$NAME.jsonl" > "$LOG_DIR/proxy_$NAME.log" 2>&1 &
PROXY=$!
trap 'kill $PROXY 2>/dev/null' EXIT
sleep 2
export SOARM_PERCEPTION=real MAX_REQUESTS=${MAX_REQUESTS:-60} "$@"
python tools/speed_eval.py --out "$RUNS_DIR/${NAME}_real" --policy-url "http://127.0.0.1:$PORT/v1" \
  --families "${FAMILIES:-pick_lift,pick_place_mat,put_in_container}" --seeds 9000-9004
python tools/batch.py --out "$RUNS_DIR/${NAME}_real_ext" --model captain-4b --policy-url "http://127.0.0.1:$PORT/v1" \
  --families "${FAMILIES:-pick_lift,pick_place_mat,put_in_container}" --seeds 9005-9014 --seed-kind eval --lanes 3 --wall-time 1080 \
  --quota-flag "$LOG_DIR/QUOTA_EXCEEDED"
echo "STACK_DONE $NAME"

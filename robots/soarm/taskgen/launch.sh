#!/bin/bash
# Start the taskgen gateway proxy (its spend lands in $LOG_DIR/gateway_cost_<Beijing day>.json, the file the teacher
# proxy caps at $30/day) and run taskgen end to end. Usage (in the code dir): bash taskgen/launch.sh OUT_DIR [run args]
set -e
cd "$(dirname "$0")/.."
source ./env.sh
OUT=${1:?out dir}; shift
mkdir -p "$OUT"
bash tools/proxy.sh start taskgen ${TG_PORT:-8661} --env-file "$GATEWAY_ENV_FILE" --concurrency 4 \
  --daily-cap-usd ${DAILY_CAP_USD:-30} --reasoning '{"effort":"medium"}'
sleep 3
nice -n 15 python -m taskgen.run --out "$OUT" --url "http://127.0.0.1:${TG_PORT:-8661}/v1/chat/completions" "$@" \
  > "$OUT/run.log" 2>&1 || true
bash tools/proxy.sh stop taskgen

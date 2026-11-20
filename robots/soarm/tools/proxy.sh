#!/bin/bash
# Teacher-model proxies (pidfiles in $LOG_DIR): tools/proxy.sh start|stop <name> <port> [extra gateway_proxy args]
source "$(cd "$(dirname "$0")/.." && pwd)/env.sh"
ACT=${1:?start|stop}; NAME=${2:?name}; PORT=${3:-}; shift 3 2>/dev/null || shift $#
PID=$LOG_DIR/proxy_$NAME.pid
if [ -f "$PID" ]; then kill "$(cat "$PID")" 2>/dev/null; rm -f "$PID"; fi
[ "$ACT" = stop ] && exit 0
cd "$SOARM_CODE"
DL=()
if [ "${SOARM_DATA_LEARNING:-off}" != off ]; then
  DL+=(--data-learning "$SOARM_DATA_LEARNING" --data-learning-rounds "${SOARM_DATA_LEARNING_ROUNDS:-3}")
fi
[ -n "${SOARM_RESPONSE_IMAGE_CAP:-}" ] && DL+=(--response-image-cap "$SOARM_RESPONSE_IMAGE_CAP")
setsid nohup python -m soarm.gateway_proxy --port "$PORT" --log "$LOG_DIR/gateway_usage_$NAME.jsonl" "${DL[@]}" "$@" \
  > "$LOG_DIR/proxy_$NAME.log" 2>&1 < /dev/null &
echo $! > "$PID"
echo "started $NAME pid $(cat "$PID") port $PORT"

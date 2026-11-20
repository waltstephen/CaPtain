#!/bin/bash
# tools/start_batch.sh <name> [batch.py args without --out/--stop-file] -> $RUNS_DIR/<name>, $LOG_DIR/batch_<name>.{log,pid}
# Optional WAIT_URL: wait until it answers (e.g. a vLLM server still loading). Env vars (SOARM_INTRO, CONTEXT_LIMIT, ...)
# are passed through to every episode.
source "$(cd "$(dirname "$0")/.." && pwd)/env.sh"
NAME=${1:?name}; shift
cd "$SOARM_CODE"
setsid nohup bash -c "if [ -n \"${WAIT_URL:-}\" ]; then until curl -s -m 5 ${WAIT_URL:-x} >/dev/null 2>&1; do sleep 10; done; fi; \
  exec python tools/batch.py --out $RUNS_DIR/$NAME --stop-file $SOARM_ROOT/STOP_$NAME --quota-flag $LOG_DIR/QUOTA_EXCEEDED $*" \
  > "$LOG_DIR/batch_$NAME.log" 2>&1 < /dev/null &
echo $! > "$LOG_DIR/batch_$NAME.pid"
echo "batch $NAME pid $(cat "$LOG_DIR/batch_$NAME.pid") -> $RUNS_DIR/$NAME"

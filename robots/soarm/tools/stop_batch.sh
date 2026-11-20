#!/bin/bash
# tools/stop_batch.sh <name> [now]: no new episodes; with "now" also kill the running episodes of $RUNS_DIR/<name>.
source "$(cd "$(dirname "$0")/.." && pwd)/env.sh"
NAME=${1:?name}
touch "$SOARM_ROOT/STOP_$NAME"
if [ "${2:-}" = now ]; then
  kill "$(cat "$LOG_DIR/batch_$NAME.pid")" 2>/dev/null
  bash "$SOARM_CODE/tools/kill_episodes.sh" "$NAME"
fi
echo "stop requested for $NAME (remove $SOARM_ROOT/STOP_$NAME before resuming)"

#!/usr/bin/env bash
# (Re)start the real-arm web control server (tools/real_arm_web.py) in the background; pid in $RUNS/web_server.pid.
set -uo pipefail
cd "$(dirname "$0")/.." && source env.sh
RUNS=${SOARM_REAL_RUNS:-${SOARM_ROOT}/speed/real_runs}; mkdir -p "$RUNS"
[ -f "$RUNS/web_server.pid" ] && kill "$(cat "$RUNS/web_server.pid")" 2>/dev/null && sleep 1
setsid nohup python tools/real_arm_web.py "${1:-8095}" > "$RUNS/web_server.log" 2>&1 < /dev/null &
echo $! > "$RUNS/web_server.pid"
sleep 2; cat "$RUNS/web_server.log"

#!/usr/bin/env bash
# Stop a real-arm episode: kill it, then park the arm at the standard rest pose (torque off).
set -uo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/env.sh"
pkill -f "react_episode.py --suite soarm_real"; pkill -f "soarm.env_server_main .*soarm_real"
sleep 1
python "$SOARM_CODE/tools/real_arm_park.py"

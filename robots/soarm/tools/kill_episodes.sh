#!/bin/bash
# tools/kill_episodes.sh <run name>...: kill the running episodes (driver + env server) of $RUNS_DIR/<name> only.
# Run as a script file so the patterns are not on the caller's own command line.
source "$(cd "$(dirname "$0")/.." && pwd)/env.sh"
[ $# -gt 0 ] || { echo "usage: kill_episodes.sh <run name>..." >&2; exit 2; }
for name in "$@"; do
  for p in $(pgrep -f "scripts/react_episode.py --suite soarm_sim .*--output-dir $RUNS_DIR/$name/"); do kill "$p"; done
  sleep 3
  for p in $(pgrep -f "soarm.env_server_main .*--output-dir $RUNS_DIR/$name/"); do kill "$p"; done
done
echo "drivers left (all runs): $(pgrep -fc 'scripts/react_episode.py --suite soarm_sim')"

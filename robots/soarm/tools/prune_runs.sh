#!/bin/bash
# Prune bulky artifacts of FINISHED episodes under the given run dirs (see batch.prune).
for root in "$@"; do
  for ep in "$root"/soarm_sim_*; do
    [ -f "$ep/episode_result.json" ] || continue
    rm -rf "$ep/public_sensor_artifacts" "$ep/observations" "$ep/public_measurements"
  done
done
du -sh "$@"

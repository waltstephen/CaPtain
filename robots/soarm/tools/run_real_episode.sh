#!/usr/bin/env bash
# One real-arm episode (suite soarm_real, soarm/real_episode.py) with the twin eval protocol of tools/run_episode.sh.
#
#   tools/run_real_episode.sh "<instruction>" <out_dir> [extra react_episode.py args ...]
#
# Env: POLICY_URL [http://127.0.0.1:8700/v1], POLICY_MODEL [captain-4b], SOARM_PROFILE [fast_v5_real],
#      SOARM_REAL_CONFIG [configs/hardware/so101_armhost_real.json], WALL_TIME [1080].
# Needs the arm at rest on /dev/ttyACM0, tools/camstream.py (armcal) on :8090, the policy and perception services.
# Stop: tools/real_arm_stop.sh (kills the episode, parks the arm, torque off).
set -uo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/env.sh"
H=$HARNESS_DIR
TASK=${1:?instruction}; OUT=${2:?out_dir}; shift 2
export CAPTAIN_IK_POSITION_TOL=${CAPTAIN_IK_POSITION_TOL:-0.008} CAPTAIN_IK_ORIENTATION_TOL=${CAPTAIN_IK_ORIENTATION_TOL:-0.1}
export CAPTAIN_COMMAND_LEAD_MIN_M=${CAPTAIN_COMMAND_LEAD_MIN_M:-0.012} CAPTAIN_COMMAND_LEAD_MIN_RAD=${CAPTAIN_COMMAND_LEAD_MIN_RAD:-0.12}
export SOARM_REAL_TASK=$TASK SOARM_INTRO=v5 MAX_REQUESTS=${MAX_REQUESTS:-100}
export SOARM_PROFILE=${SOARM_PROFILE:-fast_v5_real}
export SOARM_REAL_CONFIG=${SOARM_REAL_CONFIG:-$SOARM_CODE/configs/hardware/so101_armhost_real.json}
mkdir -p "$(dirname "$OUT")"
mapfile -t PROFILE_ARGS < <(python -m soarm.speed_profile driver-args "$SOARM_PROFILE")
exec "$H/.venv-libero/bin/python" "$H/scripts/react_episode.py" --suite soarm_real --task-id 0 --seed 0 \
  --model "${POLICY_MODEL:-captain-4b}" --api chat --auth none --endpoint "${POLICY_URL:-http://127.0.0.1:8700/v1}" --profile captain \
  --output-dir "$OUT" --root "$H" --mujoco-gl "$SOARM_GL" \
  --small-context-window 65536 --max-output-tokens 4096 --max-images 12 --wall-time "${WALL_TIME:-1080}" --grace-s 60 --max-actions 150 \
  --max-tool-calls 600 --max-requests "$MAX_REQUESTS" \
  --runtime-tool-contract python-runtime/v1 --inline-task --enable-thinking on --context-strategy window --context-limit 60000 \
  --keep-recent-tokens 16000 --tool-output-tokens 4000 --spill-tool-output --context-hard-limit 65536 \
  --student-python "$STUDENT_PYTHON" --sandbox-hide "$(dirname "$OUT")" --temperature 1.0 --top-p 1.0 --mark-requests \
  "${PROFILE_ARGS[@]}" "$@"

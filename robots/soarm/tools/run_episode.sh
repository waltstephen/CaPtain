#!/usr/bin/env bash
# One SO-101 twin episode with the CaPtain eval protocol of harness/eval/run_episode.sh (same driver flags), rendering
# per $SOARM_GL (osmesa = CPU), with host-only request marks for RL/critic use.
#
#   tools/run_episode.sh <task spec id|tid> <seed> <out_root> [extra react_episode.py args ...]
#
# Env: POLICY_URL [http://127.0.0.1:8650/v1, a teacher proxy or a vLLM server], POLICY_MODEL [deepseek-v4.1-flash-1017],
#      TEMP [1.0], WALL_TIME [1080 = student protocol; teachers 3600], MAX_OUTPUT_TOKENS [4096 = student protocol],
#      CONTEXT_LIMIT [60000 = student protocol; deepseek teachers 42000: the driver budgets context in the provider's
#      tokens, and deepseek views came out up to 100 k Qwen tokens], SOARM_INTRO [v1; teacher collection v2],
#      MAX_REQUESTS [100; the teacher budget per episode; LIBERO's 60 is not a constraint here],
#      STUDENT_PYTHON / FRESH_ENV_PREFLIGHT_PATH (env.sh), SOARM_NO_SANDBOX [0; 1 = run without bwrap on hosts that
#      forbid unprivileged user namespaces],
#      SOARM_PROFILE [unset = the protocol above; fast = configs/speed_fast.json, eval/deploy only]: its driver flags
#      (soarm/speed_profile.py) come after the flags above, so they override TEMP / MAX_OUTPUT_TOKENS; extra args
#      override the profile. The env server reads the same variable for its motion/backend options.
#      SOARM_EPISODE_DIR [unset = <out_root>/soarm_sim_<tid>_seed-<seed>]: write the episode to this fixed path instead
#      (tools/speed_eval.py --fixed-episode-dir: the prompt's paths are then identical across episodes, so the
#      policy server's prefix cache reuses the first request of a repeated instruction); one episode at a time.
set -uo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/env.sh"
H=$HARNESS_DIR
FAM=${1:?family or task id}; SEED=${2:?seed}; ROOT=${3:?out_root}; shift 3
if [[ "$FAM" =~ ^[0-9]+$ ]]; then TID=$FAM; else TID=$(python -c "from soarm.taskspec import get; print(get('$FAM')['tid'])"); fi
OUT=${SOARM_EPISODE_DIR:-$ROOT/soarm_sim_${TID}_seed-${SEED}}
mkdir -p "$ROOT"
SANDBOX_ARGS=()
[ "${SOARM_NO_SANDBOX:-0}" = 1 ] && SANDBOX_ARGS+=(--no-sandbox)
[ -n "${SOARM_EPISODE_DIR:-}" ] && SANDBOX_ARGS+=(--sandbox-hide "$(dirname "$OUT")")   # siblings of the fixed dir
PROFILE_ARGS=()
if [ -n "${SOARM_PROFILE:-}" ]; then
  PROFILE_OUT=$(python -m soarm.speed_profile driver-args "$SOARM_PROFILE") || exit 2
  [ -n "$PROFILE_OUT" ] && mapfile -t PROFILE_ARGS <<< "$PROFILE_OUT"
fi
exec "$H/.venv-libero/bin/python" "$H/scripts/react_episode.py" --suite soarm_sim --task-id "$TID" --seed "$SEED" \
  --model "${POLICY_MODEL:-deepseek-v4.1-flash-1017}" --api chat --auth none --endpoint "${POLICY_URL:-http://127.0.0.1:8650/v1}" --profile captain \
  --output-dir "$OUT" --root "$H" --mujoco-gl "$SOARM_GL" \
  --small-context-window 65536 --max-output-tokens "${MAX_OUTPUT_TOKENS:-4096}" --max-images 12 --wall-time "${WALL_TIME:-1080}" --grace-s 60 --max-actions 150 \
  --max-tool-calls 600 --max-requests "${MAX_REQUESTS:-100}" \
  --runtime-tool-contract python-runtime/v1 --inline-task --enable-thinking on --context-strategy window --context-limit "${CONTEXT_LIMIT:-60000}" \
  --keep-recent-tokens 16000 --tool-output-tokens 4000 --spill-tool-output --context-hard-limit 65536 \
  --student-python "$STUDENT_PYTHON" --sandbox-hide "$ROOT" --temperature "${TEMP:-1.0}" --top-p 1.0 --mark-requests \
  "${SANDBOX_ARGS[@]}" "${PROFILE_ARGS[@]}" "$@"

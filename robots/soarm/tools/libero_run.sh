#!/usr/bin/env bash
# Run ONE LIBERO episode with the release CaPtain student protocol (harness eval/run_episode.sh flags).
#   libero_run.sh <out_dir> <suite> <task_id> <seed> [extra react_episode.py args ...]
#   e.g. libero_run.sh ${SOARM_ROOT}/speed/runs/libero_t0 libero_spatial 0 11
# Episode lands in <out_dir>/<suite>_<task_id>_seed-<seed>/ (episode_result.json, driver/, server logs).
# Env (defaults): POLICY_URL [http://127.0.0.1:8700/v1]  POLICY_MODEL [captain-4b]
#   SAM3_URL [http://127.0.0.1:8114]  MOLMO_URL [http://127.0.0.1:8122/v1]  TEMP [1.0]
#   LIBERO_GL [osmesa = CPU rendering, as the SO-101 runs; egl = NVIDIA EGL as in the paper, EGL_DEVICE [0]]
set -euo pipefail
OUT=${1:?out_dir}; SUITE=${2:?suite}; TID=${3:?task_id}; SEED=${4:?seed}; shift 4
SPEED=${SPEED_DIR:-${SOARM_ROOT}/speed}
export CAPTAIN_HARNESS=$SPEED/harness_libero
export CAPTAIN_PYTHON=$CAPTAIN_HARNESS/.venv-libero/bin/python
export STUDENT_PYTHON=${STUDENT_PYTHON:-${SOARM_ROOT}/student/bin/python}
export FRESH_ENV_PREFLIGHT_PATH=${FRESH_ENV_PREFLIGHT_PATH:-$CAPTAIN_HARNESS/preflight.json}
export POLICY_URL=${POLICY_URL:-http://127.0.0.1:8700/v1} POLICY_MODEL=${POLICY_MODEL:-captain-4b}
export SAM3_URL=${SAM3_URL:-http://127.0.0.1:8114} MOLMO_URL=${MOLMO_URL:-http://127.0.0.1:8122/v1}
export LIBERO_CONFIG_PATH=$CAPTAIN_HARNESS/.libero-config
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
GL=${LIBERO_GL:-osmesa}
EXTRA=()
if [ "$GL" = osmesa ]; then   # run_episode.sh hard-codes --mujoco-gl egl; the later flag wins (argparse)
  export LD_LIBRARY_PATH=${SOARM_ROOT}/mesa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  EXTRA=(--mujoco-gl osmesa)
elif [ "$GL" != egl ]; then echo "LIBERO_GL must be osmesa or egl" >&2; exit 2; fi
mkdir -p "$OUT"
exec "$CAPTAIN_HARNESS/eval/run_episode.sh" "$SUITE" "$TID" "$SEED" "$OUT" "${EXTRA[@]}" "$@"

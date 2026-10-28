#!/usr/bin/env bash
# One CaPtain-4B evaluation episode with the paper protocol (T=1, top_p 1, 60-request cap, 1080 s wall, 150 physical actions,
# 64k-token context window with compaction at 60k, python-runtime/v1 tool contract, thinking on).
#
#   eval/run_episode.sh <suite> <task_id> <seed> <out_root> [extra react_episode.py args ...]
#
# Environment (defaults in brackets):
#   CAPTAIN_HARNESS   harness checkout                       [the directory containing this script's parent]
#   CAPTAIN_PYTHON    harness interpreter (.venv-libero)     [$CAPTAIN_HARNESS/.venv-libero/bin/python]
#   STUDENT_PYTHON    sandboxed Python 3.11 for the agent    [required; see README "Student Python"]
#   POLICY_URL        OpenAI-compatible endpoint (vLLM)      [http://127.0.0.1:8600/v1]
#   POLICY_MODEL      served model name                      [captain-4b]
#   SAM3_URL / MOLMO_URL perception services                 [http://127.0.0.1:8114 / http://127.0.0.1:8122/v1]
#   EGL_DEVICE        MuJoCo EGL device index                [0]
#   TEMP              sampling temperature                   [1.0]
#   FRESH_ENV_PREFLIGHT_PATH  optional toolchain preflight receipt (see README)
# The episode is written to <out_root>/<suite>_<task_id>_seed-<seed>/ (episode_result.json + driver/ transcript + server logs).
set -uo pipefail
SUITE=${1:?suite}; TID=${2:?task_id}; SEED=${3:?seed}; ROOT=${4:?out_root}; shift 4
H=${CAPTAIN_HARNESS:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${CAPTAIN_PYTHON:-$H/.venv-libero/bin/python}
: "${STUDENT_PYTHON:?set STUDENT_PYTHON to the sandboxed Python 3.11 interpreter (numpy 1.26.4, scipy 1.13.1, Pillow 10.4.0)}"
OUT=$ROOT/${SUITE}_${TID}_seed-${SEED}
mkdir -p "$ROOT"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=$H/src LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-$H/.libero-config}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
exec "$PY" "$H/scripts/react_episode.py" --suite "$SUITE" --task-id "$TID" --seed "$SEED" \
  --model "${POLICY_MODEL:-captain-4b}" --api chat --auth none --endpoint "${POLICY_URL:-http://127.0.0.1:8600/v1}" --profile captain \
  --output-dir "$OUT" --root "$H" --mujoco-gl egl --egl-device "${EGL_DEVICE:-0}" \
  --small-context-window 65536 --max-output-tokens 4096 --max-images 12 --wall-time 1080 --grace-s 60 --max-actions 150 \
  --max-tool-calls 600 --max-requests 60 --sam3-url "${SAM3_URL:-http://127.0.0.1:8114}" --molmo-url "${MOLMO_URL:-http://127.0.0.1:8122/v1}" \
  --runtime-tool-contract python-runtime/v1 --inline-task --enable-thinking on --context-strategy window --context-limit 60000 \
  --keep-recent-tokens 16000 --tool-output-tokens 4000 --spill-tool-output --context-hard-limit 65536 \
  --student-python "$STUDENT_PYTHON" --sandbox-hide "$ROOT" --temperature "${TEMP:-1.0}" --top-p 1.0 "$@"

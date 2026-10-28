#!/usr/bin/env bash
# One zero-shot transfer episode of a CaPtain model with the paper protocol.
#
#   eval/transfer/run_episode.sh <robotwin2|simplerenv|robodojo|capbench> <task.json> <out_root> [extra react_episode.py args ...]
#
# The model-side protocol is the LIBERO one (eval/run_episode.sh): T=1, top_p 1, thinking on, 64k-token window with
# compaction at 60k, python-runtime/v1 tool contract, 150 physical actions, 600 tool calls. Budgets per benchmark:
#   robotwin2, simplerenv, capbench: 60 model requests, 1080 s wall;   robodojo: 120 model requests, 3600 s wall.
#
# Environment (defaults in brackets):
#   CAPTAIN_TRANSFER_PYTHON harness interpreter for robotwin2/simplerenv/robodojo (Python 3.11, requirements-transfer.txt) [.venv-transfer]
#   CAPTAIN_HARNESS / CAPTAIN_PYTHON (capbench) / STUDENT_PYTHON / POLICY_URL / POLICY_MODEL / SAM3_URL / MOLMO_URL / TEMP: as eval/run_episode.sh
#   SIM_GPU          GPU index for the simulator process (SAPIEN / Isaac Sim / MuJoCo EGL)            [0]
#   robotwin2:  ROBOTWIN_ROOT (RoboTwin checkout, patched), ROBOTWIN_PYTHON (its interpreter)
#   simplerenv: SIMPLERENV_ROOT (SimplerEnv checkout), SIMPLERENV_PYTHON, [SIMPLERENV_EGL_VENDOR_FILE]
#   robodojo:   ROBODOJO_ROOT (RoboDojo checkout), ROBODOJO_ENV_SH (script that activates RoboDojo's Isaac Sim env)
# The episode is written to <out_root>/<task name>/ (episode_result.json, driver/ transcript, server logs, host-only files).
set -uo pipefail
BENCH=${1:?benchmark: robotwin2|simplerenv|robodojo|capbench}; TASK=$(readlink -f "${2:?task.json}"); ROOT=${3:?out_root}; shift 3
H=${CAPTAIN_HARNESS:-$(cd "$(dirname "$0")/../.." && pwd)}
# harness interpreter: CaP-Bench runs in the LIBERO venv (robosuite 1.4); the other three need MuJoCo 3 for the kinematic twins
if [ "$BENCH" = capbench ]; then PY=${CAPTAIN_PYTHON:-$H/.venv-libero/bin/python}; else PY=${CAPTAIN_TRANSFER_PYTHON:-$H/.venv-transfer/bin/python}; fi
: "${STUDENT_PYTHON:?set STUDENT_PYTHON to the sandboxed Python 3.11 interpreter (see harness/README.md)}"
NAME=$(basename "$TASK" .json)
mkdir -p "$ROOT"
GPU=${SIM_GPU:-0}
CAP=60; WALL=1080; SUITE=; TASK_ID=0; SEED=0; EGL=(--egl-device 0); SERVER=()
render() {  # render() <template> <name>: benchmark config with this machine's paths, once per out_root
  local dst="$ROOT/_config/$2.json"
  [ -s "$dst" ] || python3 "$H/eval/transfer/render_config.py" "$H/configs/benchmarks/$1" "$dst" >/dev/null || exit 2
  echo "$dst"
}
case "$BENCH" in
  robotwin2)
    SUITE=robotwin2_sim; CFG=$(render robotwin2_franka_dual.json robotwin2) || exit 2
    export CUDA_VISIBLE_DEVICES=$GPU   # SAPIEN renders on this GPU
    SERVER=(--server-arg=--robotwin-config "--server-arg=$CFG" --server-arg=--robotwin-task "--server-arg=$TASK") ;;
  simplerenv)
    SUITE=simplerenv_sim; CFG=$(render simplerenv.json simplerenv) || exit 2
    export CUDA_VISIBLE_DEVICES=$GPU
    SERVER=(--server-arg=--simplerenv-config "--server-arg=$CFG" --server-arg=--simplerenv-task "--server-arg=$TASK") ;;
  robodojo)
    SUITE=robodojo_sim; CAP=120; WALL=3600; EGL=(); CFG=$(render robodojo_x5.json robodojo) || exit 2
    # official layouts; episodes continue up to 20x the native action budget, the official verdict is taken at the budget
    export RD_GPU=$GPU RD_LAYOUT_VARIANT=official RD_STEP_LIMIT_FACTOR=${RD_STEP_LIMIT_FACTOR:-20}
    SERVER=(--server-arg=--robodojo-config "--server-arg=$CFG" --server-arg=--robodojo-task "--server-arg=$TASK" --server-arg=--idle-exit-s=600) ;;
  capbench)
    SUITE=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["suite"])' "$TASK")
    SEED=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["seed"])' "$TASK")
    EGL=(--egl-device "$GPU")
    export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-$H/.libero-config} ;;
  *) echo "unknown benchmark $BENCH" >&2; exit 2 ;;
esac
OUT=$ROOT/$NAME
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=$H/src
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1} NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
exec "$PY" "$H/scripts/react_episode.py" --suite "$SUITE" --task-id "$TASK_ID" --seed "$SEED" \
  --model "${POLICY_MODEL:-captain-4b}" --api chat --auth none --endpoint "${POLICY_URL:-http://127.0.0.1:8600/v1}" --profile captain \
  --output-dir "$OUT" --root "$H" --mujoco-gl egl "${EGL[@]}" \
  --small-context-window 65536 --max-output-tokens 4096 --max-images 12 --wall-time "$WALL" --grace-s 60 --max-actions 150 \
  --max-tool-calls 600 --max-requests "$CAP" --sam3-url "${SAM3_URL:-http://127.0.0.1:8114}" --molmo-url "${MOLMO_URL:-http://127.0.0.1:8122/v1}" \
  --runtime-tool-contract python-runtime/v1 --inline-task --enable-thinking on --context-strategy window --context-limit 60000 \
  --keep-recent-tokens 16000 --tool-output-tokens 4000 --spill-tool-output --context-hard-limit 65536 \
  --student-python "$STUDENT_PYTHON" --sandbox-hide "$ROOT" --temperature "${TEMP:-1.0}" --top-p 1.0 \
  "${SERVER[@]}" --server-arg=--contact-detector --server-arg=residual --server-arg=--contact-calibration "$@"

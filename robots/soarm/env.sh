# Runtime environment of the SO-101 twin pipeline: `source robots/soarm/env.sh` (every tool script does it).
# Settings are read from the first existing file of: $SOARM_ENV_FILE, robots/soarm/soarm.env, ~/.soarm.env
# (copy env.example; never commit a filled-in file). Every variable has a default, see env.example.
SOARM_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
for _f in "${SOARM_ENV_FILE:-}" "$SOARM_CODE/soarm.env" "$HOME/.soarm.env"; do
  if [ -n "$_f" ] && [ -f "$_f" ]; then source "$_f"; break; fi
done
unset _f
export SOARM_CODE
export SOARM_ROOT=${SOARM_ROOT:-$HOME/soarm_work}
export HARNESS_DIR=${HARNESS_DIR:-$(cd "$SOARM_CODE/../../harness" 2>/dev/null && pwd)}
export SFT_DIR=${SFT_DIR:-$(cd "$SOARM_CODE/../../sft" 2>/dev/null && pwd)}
export SOARM_MENAGERIE=${SOARM_MENAGERIE:-$SOARM_ROOT/menagerie/robotstudio_so101}
export RUNS_DIR=${RUNS_DIR:-$SOARM_ROOT/runs} LOG_DIR=${LOG_DIR:-$SOARM_ROOT/logs} DATA_DIR=${DATA_DIR:-$SOARM_ROOT/data}
export SOARM_VENV=${SOARM_VENV:-$SOARM_ROOT/venv-srv}
export STUDENT_PYTHON=${STUDENT_PYTHON:-$SOARM_ROOT/student/bin/python}
export SFT_VENV=${SFT_VENV:-$SOARM_ROOT/venv-sft} VLLM_VENV=${VLLM_VENV:-$SOARM_ROOT/venv-vllm}
export FRESH_ENV_PREFLIGHT_PATH=${FRESH_ENV_PREFLIGHT_PATH:-$SOARM_ROOT/preflight.json}
export GATEWAY_ENV_FILE=${GATEWAY_ENV_FILE:-$HOME/.soarm_gateway.env}
export DEEPSEEK_ENV_FILE=${DEEPSEEK_ENV_FILE:-$HOME/.soarm_deepseek.env}
export SOARM_DATA_LEARNING=${SOARM_DATA_LEARNING:-off} SOARM_DATA_LEARNING_ROUNDS=${SOARM_DATA_LEARNING_ROUNDS:-3}
export SOARM_RESPONSE_IMAGE_CAP=${SOARM_RESPONSE_IMAGE_CAP:-}
export SOARM_MACHINE=${SOARM_MACHINE:-}     # seed-range owner (CONTRACT 1): armhost | datahost
# perception (soarm/perception.py; ROLLOUT §3.1): real services by default; replica URL lists (empty = SAM3_URL/MOLMO_URL)
export SOARM_PERCEPTION=${SOARM_PERCEPTION:-real} SOARM_SAM3_URLS=${SOARM_SAM3_URLS:-} SOARM_MOLMO_URLS=${SOARM_MOLMO_URLS:-}
# depth: MoGe-2 from the RGB frame (the real arm's path; default) | sensor (tests/ablation only); MoGe-2 replica URLs
export SOARM_DEPTH=${SOARM_DEPTH:-moge2} SOARM_DEPTH_URLS=${SOARM_DEPTH_URLS:-}
export SOARM_DEPTH_NOISE=${SOARM_DEPTH_NOISE:-off}
export SOARM_GL=${SOARM_GL:-osmesa} MESA_DIR=${MESA_DIR:-$SOARM_ROOT/mesa}
mkdir -p "$RUNS_DIR" "$LOG_DIR" "$DATA_DIR" 2>/dev/null
export PATH=$SOARM_VENV/bin:$PATH
if [ "$SOARM_GL" = osmesa ]; then     # CPU rendering (llvmpipe); GPUs are never touched by the simulator
  export LD_LIBRARY_PATH=$MESA_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
else
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
fi
export PYTHONPATH=$SOARM_CODE:$HARNESS_DIR/src${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

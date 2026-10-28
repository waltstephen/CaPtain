#!/usr/bin/env bash
# Start the two perception services used by the capability API (detached; logs under $LOG_DIR):
#   SAM3      on :8114  (CaP-X SAM3 FastAPI server: /segment, /segment_point)       -> services/sam3_server.py
#   Molmo2-8B on :8122  (vLLM, OpenAI-compatible; "Point at <text>" pointing prompts)
# Usage: services/start_perception.sh [sam3|molmo|all]
# Environment (defaults in brackets):
#   TOOLS_PYTHON     interpreter with cap-x (pinned in third_party/manifest.json) and its sam3 submodule installed [python3]
#   VLLM_BIN         vLLM executable for Molmo2-8B                                      [vllm]
#   SAM3_CHECKPOINT  local sam3.pt (facebook/sam3, gated on HF); empty = download with $HF_TOKEN  []
#   MOLMO_MODEL      local dir or HF id of allenai/Molmo2-8B (the paper used revision e28fa28)       [allenai/Molmo2-8B]
#   SAM3_GPU / MOLMO_GPU  CUDA device indices                                           [0 / 0]
#   MOLMO_GPU_UTIL   vLLM --gpu-memory-utilization for Molmo                             [0.45]
#   HOST             bind address (keep 127.0.0.1 unless env servers run on other hosts)  [127.0.0.1]
#   LOG_DIR          [./logs/services]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LOG_DIR=${LOG_DIR:-$PWD/logs/services}; mkdir -p "$LOG_DIR"
what=${1:-all}
if [ "$what" = sam3 ] || [ "$what" = all ]; then
  CUDA_VISIBLE_DEVICES=${SAM3_GPU:-0} setsid nohup "${TOOLS_PYTHON:-python3}" "$HERE/sam3_server.py" --port 8114 --host "${HOST:-127.0.0.1}" \
    ${SAM3_CHECKPOINT:+--checkpoint "$SAM3_CHECKPOINT"} > "$LOG_DIR/sam3.log" 2>&1 < /dev/null &
  echo "sam3 pid=$!  (log $LOG_DIR/sam3.log)"
fi
if [ "$what" = molmo ] || [ "$what" = all ]; then
  CUDA_VISIBLE_DEVICES=${MOLMO_GPU:-0} setsid nohup "${VLLM_BIN:-vllm}" serve "${MOLMO_MODEL:-allenai/Molmo2-8B}" \
    --served-model-name allenai/Molmo2-8B Molmo2-8B molmo --host "${HOST:-127.0.0.1}" --port 8122 --trust-remote-code \
    --max-model-len 16384 --max-num-batched-tokens 16384 --gpu-memory-utilization "${MOLMO_GPU_UTIL:-0.45}" \
    --limit-mm-per-prompt '{"image":2}' > "$LOG_DIR/molmo_vllm.log" 2>&1 < /dev/null &
  echo "molmo pid=$!  (log $LOG_DIR/molmo_vllm.log)"
fi

#!/usr/bin/env bash
# Serve CaPtain-4B (or CaPtain-4B-SFT) with vLLM exactly as in the paper's evaluations (OpenAI-compatible, port 8600).
# Usage: services/serve_policy.sh <model dir or HF repo id> [port]
# Environment: VLLM_BIN [vllm], POLICY_GPU [0], POLICY_GPU_UTIL [0.60], SERVED_NAME [captain-4b], HOST [127.0.0.1], LOG_DIR [./logs/services]
# Tested with vLLM 0.29.0 (+cu129, torch 2.13) and transformers 5.17. One 80 GB GPU serves ~8 concurrent episodes comfortably;
# the paper ran <= 8 episodes per policy server. Add VLLM_USE_FLASHINFER_SAMPLER=0 if FlashInfer JIT is unavailable on your host.
set -euo pipefail
MODEL=${1:?model dir or HF id (e.g. <hf-user>/CaPtain-4B)}; PORT=${2:-8600}
LOG_DIR=${LOG_DIR:-$PWD/logs/services}; mkdir -p "$LOG_DIR"
CUDA_VISIBLE_DEVICES=${POLICY_GPU:-0} setsid nohup "${VLLM_BIN:-vllm}" serve "$MODEL" --served-model-name "${SERVED_NAME:-captain-4b}" \
  --host "${HOST:-127.0.0.1}" --port "$PORT" --dtype bfloat16 --gpu-memory-utilization "${POLICY_GPU_UTIL:-0.60}" \
  --max-model-len 81920 --enable-prefix-caching --limit-mm-per-prompt '{"image":12}' --max-num-seqs 16 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml > "$LOG_DIR/policy_$PORT.log" 2>&1 < /dev/null &
echo "policy pid=$!  (log $LOG_DIR/policy_$PORT.log); wait for: curl -s http://127.0.0.1:$PORT/v1/models"

#!/bin/bash
# vLLM policy server for student checkpoints (harness/services/serve_policy.sh flags, 64k context), one GPU per server.
#   tools/serve_policy.sh <model dir> <port> <gpu> [served name]       ($VLLM_VENV: vllm 0.29 for Qwen3.5)
#   POLICY_MEM [0.88], MAX_SEQS [8], MAX_MODEL_LEN [65536], MAX_MM_IMAGES [12]; POLICY_ARGS = extra vllm serve flags (word-split), e.g. --quantization fp8,
#   --enable-lora --lora-modules name=dir, --speculative-config '{...}'
#   SPEC_VERIFY_PATCH=1: load tools/vllm_patches/sitecustomize.py into vLLM (use with --attention-backend TRITON_ATTN):
#   split-KV attention for speculative verify steps, FA2 for prefill steps (SOARM_VLLM_PATCH_FA_PREFILL, default on),
#   and with SOARM_DRAFT_VOCAB=<file> the MTP drafter scores only that lm_head subset (build_draft_vocab.py). All
#   lossless; unset = stock vLLM. Deployment recipe: README.md "Fast version (SO-101)".
source "$(cd "$(dirname "$0")/.." && pwd)/env.sh"
[ "${SPEC_VERIFY_PATCH:-0}" = 1 ] && export PYTHONPATH=$SOARM_CODE/tools/vllm_patches:$PYTHONPATH
M=${1:?model dir}; P=${2:?port}; G=${3:?gpu}; N=${4:-captain-4b}
VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=$G setsid nohup "$VLLM_VENV/bin/vllm" serve "$M" --served-model-name "$N" \
  --host 127.0.0.1 --port "$P" --dtype bfloat16 --gpu-memory-utilization ${POLICY_MEM:-0.88} --max-model-len ${MAX_MODEL_LEN:-65536} \
  --enable-prefix-caching --limit-mm-per-prompt "{\"image\":${MAX_MM_IMAGES:-12}}" --max-num-seqs ${MAX_SEQS:-8} --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml ${POLICY_ARGS:-} > "$LOG_DIR/policy_$P.log" 2>&1 < /dev/null &
echo $! > "$LOG_DIR/policy_$P.pid"
echo "policy pid=$(cat "$LOG_DIR/policy_$P.pid") port=$P gpu=$G model=$M"

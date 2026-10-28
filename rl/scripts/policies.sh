#!/usr/bin/env bash
# Policy vLLM servers of a rollout node, with the NCCL weight-transfer engine (the trainer pushes new weights straight into
# them, rl/nccl_refit.py) and the rl/vllm_worker_ext.py empty-cache hook.
#   bash rl/scripts/policies.sh up "<gpus>" [mem=0.65] [base_port=8500]   # one server per GPU, ports base_port, base_port+1, ...
#   bash rl/scripts/policies.sh down | health
# env: INIT = initial policy weights (CaPtain-4B-SFT dir, or the run's latest export when resuming), MAXLEN [81920], SEQS [8]
set -u
source "$(dirname "$0")/env.sh"
ST=$CAPTAIN_WORKSPACE/state/$(hostname -s); LOGS=$CAPTAIN_LOGS/$(hostname -s); mkdir -p "$ST" "$LOGS"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
case "${1:-}" in
  up)
    i=0; mem=${3:-0.65}; base=${4:-8500}; model=${INIT:?INIT=<policy weights dir>}
    for g in $2; do
      port=$((base + i)); i=$((i + 1))
      CUDA_VISIBLE_DEVICES=$g VLLM_SERVER_DEV_MODE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONPATH="$CAPTAIN_ROOT/rl" \
      setsid nohup "$CAPTAIN_VLLM_PY" -m vllm.entrypoints.cli.main serve "$model" --served-model-name rl-policy \
        --host 0.0.0.0 --port $port --gpu-memory-utilization "$mem" --max-model-len "${MAXLEN:-81920}" --enable-prefix-caching \
        --limit-mm-per-prompt '{"image":12}' --max-num-seqs "${SEQS:-8}" --reasoning-parser qwen3 --enable-auto-tool-choice \
        --tool-call-parser qwen3_xml --dtype bfloat16 --weight-transfer-config '{"backend":"nccl"}' \
        --worker-extension-cls vllm_worker_ext.RefitMemExt > "$LOGS/policy-$port.log" 2>&1 < /dev/null &
      echo $! > "$ST/policy-$port.pid"; echo "policy :$port gpu $g pid $! log $LOGS/policy-$port.log"
    done ;;
  down)
    for f in "$ST"/policy-*.pid; do [ -f "$f" ] || continue; pid=$(cat "$f"); kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null; rm -f "$f"; echo "stopped $(basename "$f" .pid)"; done ;;
  health)
    for f in "$ST"/policy-*.pid; do [ -f "$f" ] || continue; p=$(basename "$f" .pid); p=${p#policy-}
      printf "%s:%s " "$p" "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "localhost:$p/health")"; done; echo ;;
  *) sed -n 2,7p "$0"; exit 2 ;;
esac

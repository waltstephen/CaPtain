#!/bin/bash
# Fixed SO-101 twin eval of one CaPtain-4B checkpoint (original CaPtain-4B / SO-101 SFT) on one 8-GPU node.
#   SFTOPD_DIR=/path/to/work bash robots/soarm/tools/sftopd_eval.sh <model_dir> <name>      (non-root, 8 GPUs)
# Set: div3_seen (169 = every div3 mix task, seed 9000) + cube_heldout_families (40 = 4 held-out cube specs x 9000-9009),
# the teacher-data harness (T=1, top_p 1, 100 requests, 3600 s wall, 4096 out, 64k window with compaction at 49,536
# (= hard limit 65,536 - one turn) keeping 16k, 12 images, thinking on, intro $SOARM_INTRO [v5 = compact_v1, as the data]),
# real SAM3 + Molmo2-8B, MoGe-2 depth, OSMesa CPU rendering.
# GPUs: 0 and 1 = SAM3 + Molmo2-8B (0.55) + MoGe-2 (one perception set each, episodes split by seed); 2-7 = vLLM policy
# replicas (8 seqs each; batch.py sends each episode to the least-loaded replica). 2-6 serve div3_seen (40 lanes), 7 serves
# cube_heldout_families (8 lanes). Episode starts are staggered (STAGGER_S, default 4 s).
# Infra-error episodes are discarded by batch.py and rerun (up to PASSES=3 passes).
# Episodes: ${EVAL_SCRATCH:-$SFTOPD_DIR/eval_scratch}/<name>; summary (summary.md/json + batch/infra jsonl): $SFTOPD_DIR/eval/<name>.
# Env (besides env.sh / soarm.env): SAM3_PYTHON (python with the harness SAM3 server deps), SAM3_CHECKPOINT (sam3.pt),
# MOLMO_MODEL (Molmo2-8B dir), MOGE_CHECKPOINT (MoGe-2 model.pt), optional SANDBOX_SHIM_DIR (a bwrap stand-in put first
# on PATH where containers have no user namespaces).
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(cd "$HERE/../.." && pwd)

M=${1:?model_dir}; NAME=${2:?name}
[ "$(id -u)" = 0 ] && { echo "run as a non-root user" >&2; exit 2; }
W=${SFTOPD_DIR:?set SFTOPD_DIR}
S=${EVAL_SCRATCH:-$W/eval_scratch}/$NAME; D=$W/eval/$NAME; L=$S/logs
mkdir -p "$L" "$D" "$HOME"
export LOG_DIR=$L RUNS_DIR=$S
source "$HERE/env.sh"
: "${SAM3_PYTHON:?}" "${SAM3_CHECKPOINT:?}" "${MOLMO_MODEL:?}" "${MOGE_CHECKPOINT:?}"
log() { echo "$(date -u +%FT%TZ) [eval $NAME] $*" | tee -a "$L/supervisor.log"; }
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -n "${SANDBOX_SHIM_DIR:-}" ] && export PATH=$SANDBOX_SHIM_DIR:$PATH
export VLLM_CACHE_ROOT=$HOME/.cache/vllm TRITON_CACHE_DIR=$HOME/.cache/triton TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/inductor
export HF_HOME=$HOME/.cache/huggingface HF_MODULES_CACHE=$HOME/.cache/huggingface/modules
mkdir -p "$HF_MODULES_CACHE"
getent passwd "$(id -u)" > /dev/null || export USER=${USER:-captain} LOGNAME=${LOGNAME:-captain}
export FRESH_ENV_PREFLIGHT_PATH=$S/preflight.json     # receipts are hostname-bound: write one here
log "model=$M repo=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null) harness=$(git -C "$HARNESS_DIR" rev-parse --short HEAD 2>/dev/null) intro=${SOARM_INTRO:-v5}"
[ -f "$M/config.json" ] || { log "no model at $M"; exit 2; }
"$SOARM_VENV/bin/python" "$REPO/rl/toolchain_preflight.py" --harness "$HARNESS_DIR" --contract python-runtime/v1 \
  --student-python "$STUDENT_PYTHON" > "$FRESH_ENV_PREFLIGHT_PATH" 2> "$L/preflight.err"
python -c "import json,os,sys; sys.exit(0 if json.load(open(os.environ['FRESH_ENV_PREFLIGHT_PATH']))['ok'] else 1)" \
  || { log "toolchain preflight failed ($FRESH_ENV_PREFLIGHT_PATH)"; exit 2; }
for i in $(seq 36); do MAXU=$(timeout 300 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "${MAXU:-99999}" -le 4096 ] && break; sleep 5; done
[ "${MAXU:-99999}" -le 4096 ] || { log "GPUs busy: max memory.used ${MAXU} MiB"; exit 3; }

PIDS=""
teardown() { log "teardown"; for p in $PIDS; do kill -- -"$p" 2>/dev/null || kill "$p" 2>/dev/null; done; sleep 10; }
trap teardown EXIT; trap 'log "signal"; exit 143' INT TERM
svc() { local name=$1; shift; setsid nohup "$@" > "$L/$name.log" 2>&1 < /dev/null & PIDS="$PIDS $!"; }
for k in 0 1; do
  CUDA_VISIBLE_DEVICES=$k svc sam3-$k "$SAM3_PYTHON" "$HARNESS_DIR/services/sam3_server.py" --port $((8114 + k)) \
    --host 127.0.0.1 --checkpoint "$SAM3_CHECKPOINT"
  CUDA_VISIBLE_DEVICES=$k PATH=$VLLM_VENV/bin:$PATH svc molmo-$k "$VLLM_VENV/bin/python" -m vllm.entrypoints.cli.main serve \
    "$MOLMO_MODEL" --served-model-name allenai/Molmo2-8B Molmo2-8B molmo --host 127.0.0.1 --port $((8122 + k)) \
    --trust-remote-code --max-model-len 16384 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.55 \
    --limit-mm-per-prompt '{"image":2}'
  CUDA_VISIBLE_DEVICES=$k svc moge-$k "$SOARM_ROOT/venv-depth/bin/python" "$HARNESS_DIR/services/depth_server.py" \
    --checkpoint "$MOGE_CHECKPOINT" --port $((8130 + k)) --device cuda:0
  sleep 20
done
POL=""
for g in 2 3 4 5 6 7; do
  P=$((8598 + g))
  VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=$g svc policy-$P "$VLLM_VENV/bin/vllm" serve "$M" --served-model-name captain-4b \
    --host 127.0.0.1 --port $P --dtype bfloat16 --gpu-memory-utilization 0.88 --max-model-len 65536 --enable-prefix-caching \
    --limit-mm-per-prompt '{"image":12}' --max-num-seqs 8 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml
  POL="$POL $P"; sleep 5
done
up() { curl -sf -m 5 "$1" > /dev/null; }
for i in $(seq 120); do ok=1
  for k in 0 1; do up 127.0.0.1:$((8114 + k))/docs && up 127.0.0.1:$((8122 + k))/health && up 127.0.0.1:$((8130 + k))/health || ok=0; done
  for P in $POL; do up 127.0.0.1:$P/health || ok=0; done
  [ $ok = 1 ] && break; sleep 15; done
[ $ok = 1 ] || { log "services not up after 30 min (logs $L)"; exit 3; }
log "services up: perception GPUs 0-1, policies$POL"
export SOARM_SAM3_URLS=http://127.0.0.1:8114,http://127.0.0.1:8115
export SOARM_MOLMO_URLS=http://127.0.0.1:8122/v1,http://127.0.0.1:8123/v1
export SOARM_DEPTH=moge2 SOARM_DEPTH_URLS=http://127.0.0.1:8130,http://127.0.0.1:8131
export MAX_REQUESTS=100 TEMP=1.0 MAX_OUTPUT_TOKENS=4096 CONTEXT_LIMIT=60000   # = the teacher data (context_limit 49,536 via the hard limit)
url() { local o="" p; for p in "$@"; do o="$o${o:+,}http://127.0.0.1:$p/v1"; done; echo "$o"; }
run_suite() {   # suite lanes ports...
  local su=$1 lanes=$2; shift 2
  for pass in $(seq ${PASSES:-3}); do
    python "$HERE/tools/batch.py" --out "$S/$su" --suite "$su" --model captain-4b --policy-url "$(url "$@")" --lanes "$lanes" \
      --wall-time 3600 --intro "${SOARM_INTRO:-v5}" --stagger-s "${STAGGER_S:-4}" --max-consecutive-errors 12 \
      --quota-flag "$L/NO_QUOTA_FLAG" >> "$L/batch_$su.log" 2>&1
    n=$(cat "$S/$su/batch.jsonl" 2>/dev/null | wc -l)
    log "$su pass $pass: $n rows"
    python - "$su" "$n" <<'EOF' && break
import sys
from soarm import taskspec
sys.exit(0 if int(sys.argv[2]) >= len(taskspec.suite(sys.argv[1])["jobs"]) else 1)
EOF
  done
}
cd "$HERE"
run_suite div3_seen 40 8600 8601 8602 8603 8604 & A=$!
sleep 30
run_suite cube_heldout_families 8 8605 & B=$!
wait $A $B
python "$HERE/tools/sftopd_summary.py" "$S" --name "$NAME" > /dev/null
for su in div3_seen cube_heldout_families; do
  mkdir -p "$D/$su"; cp "$S/$su"/*.jsonl "$D/$su/" 2>/dev/null
done
cp "$S/summary.md" "$S/summary.json" "$D/"
echo "model=$M episodes=$S" > "$D/SOURCE"
log "done: $D/summary.md"
cat "$D/summary.md"

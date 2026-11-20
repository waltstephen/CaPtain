#!/bin/bash
# CaPtain SFT on 24 GB GPUs (RTX 4090) with the unchanged sft/train/launch_sft.sh recipe: ZeRO-3 with parameters AND
# optimizer offloaded to CPU, checkpointed activations offloaded, chunked lm_head+CE, no intermediate checkpoints.
# Needs ~160 GB free host RAM per run (one run per host).
#   GPUS=1 tools/train_sft.sh <run name> <init model dir> <sft dir (from build_sft.sh)> [epochs] [extra train_sft args]
set -uo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/env.sh"
W=$SOARM_ROOT
RUN=${1:?run}; MODEL=${2:?model}; D=$(realpath -m "${3:?sft dir}"); E=${4:-2}; shift 4 2>/dev/null || shift $#
G=${GPUS:-0,1}
NG=$(echo "$G" | tr ',' '\n' | wc -l)
# Host RAM guard: a full-offload run pins ~60-95 GB (params + fp32 master + Adam states). Two runs on armhost (188 GB)
# made the box thrash and drop off the network (2026-11-09 07:45). Refuse to start below MIN_AVAIL_GB available.
AVAIL=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
if [ "$AVAIL" -lt "${MIN_AVAIL_GB:-110}" ]; then
  echo "refusing to start: ${AVAIL} GB available < ${MIN_AVAIL_GB:-110} GB (another offloaded SFT running?)" >&2; exit 3
fi
if pgrep -f "train/train_sft.py" > /dev/null; then echo "refusing to start: another train_sft.py is running" >&2; exit 3; fi
# E epochs are written as E copies of the step-ordered file and trained as ONE epoch: at an epoch boundary the Trainer
# re-forks its dataloader workers, which tipped the 188 GB host into the OOM killer (2026-11-09 09:59, step 155/310).
DATA=$D/sft_mix2_x$E.jsonl
[ -f "$DATA" ] || for i in $(seq 1 "$E"); do cat "$D/sft_mix2.jsonl"; done > "$DATA"
# C = mean supervised tokens per optimizer step at this world size (sup_tokens.py was run for world 2)
C=$(python3 -c "import json;d=json.load(open('$D/sup_tokens.json'));print(round(d['sup_tokens']/((d['rows']+$NG-1)//$NG),1))")
export PATH=$SFT_VENV/bin:$PATH CUDA_VISIBLE_DEVICES=$G GPUS_PER_NODE=$NG DS_SKIP_CUDA_CHECK=1 SFT_GC_OFFLOAD=${SFT_GC_OFFLOAD:-1} \
  SFT_SAVE_STEPS=0 SFT_CE_CHUNK=${SFT_CE_CHUNK:-2048} SFT_NUM_WORKERS=${SFT_NUM_WORKERS:-1} \
  OUT_ROOT=$W/sft_runs DS_CONFIG=$SOARM_CODE/configs/sft/ds_zero3_offload_all.json LR=${LR:-1e-5} WARMUP=${WARMUP:-5} \
  EPOCHS=1 MAX_LEN=66048 MASK_P=${MASK_P:-0.2} QA_WEIGHT=0.5 MASTER_PORT=$((29700 + ${G%%,*})) OMP_NUM_THREADS=12
cd "$SFT_DIR"
exec env MODEL="$MODEL" DATA="$DATA" IMAGE_ROOT="$D" C="$C" RUN="$RUN" bash train/launch_sft.sh "$@"

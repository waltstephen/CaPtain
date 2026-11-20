#!/bin/bash
# SFT of CaPtain-4B on the native SO-101 teacher episodes (the recipe of the SO-101 fast policy ck300-r1113c):
# one 8-GPU node (8x H100 80 GB), 2 epochs, LR 1e-5 cosine, warmup 20, frozen vision tower, ZeRO-3 (sft/train).
#   SFTOPD_DIR=/path/to/work BASE=/path/to/CaPtain-4B bash robots/soarm/tools/sftopd_sft.sh <run>
# Data (one of):
#   - tools/build_sft.sh output (world 8) in $SFTOPD_DIR/sft: sft_mix8.jsonl, sup_tokens.json, images/;
#   - the released dataset (huggingface.co/datasets/YijiaFan/CaPtain-SO101-SFT-Data): SFT_DATA_DIR=<download dir>
#     -> data/sft_packed_mix8.jsonl, metadata/sup_tokens.json, image paths relative to the dataset root.
# Output: $OUT_ROOT/<run>/final, copied to $SFTOPD_DIR/sft_models/<run>. SFT_PYTHON_BIN: bin dir of the SFT environment
# (sft/README.md), put first on PATH.
set -euo pipefail
RUN=${1:?run name}
D=${SFTOPD_DIR:?set SFTOPD_DIR (work dir with sft/sft_mix8.jsonl)}
CODE=$(cd "$(dirname "$0")/../../.." && pwd)          # repository root (sft/ is next to robots/)
BASE=${BASE:?set BASE to the CaPtain-4B model dir (huggingface.co/YijiaFan/CaPtain-4B)}
OUT_ROOT=${OUT_ROOT:-$D/runs}
if [ -n "${SFT_DATA_DIR:-}" ]; then
  DATA=$SFT_DATA_DIR/data/sft_packed_mix8.jsonl SUP=$SFT_DATA_DIR/metadata/sup_tokens.json IMAGE_ROOT=$SFT_DATA_DIR
else
  DATA=$D/sft/sft_mix8.jsonl SUP=$D/sft/sup_tokens.json IMAGE_ROOT=$D/sft
fi
[ "$(id -u)" = 0 ] && { echo "run as a non-root user" >&2; exit 2; }
[ -n "${SFT_PYTHON_BIN:-}" ] && export PATH=$SFT_PYTHON_BIN:$PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
mkdir -p "$D/logs" "$OUT_ROOT"
exec >> "$D/logs/sft-$RUN.log" 2>&1
echo "$(date -u +%FT%TZ) start $RUN base=$BASE"
cd "$CODE/sft"
C=$(python -c "import json;print(json.load(open('$SUP'))['C_weighted_per_step'])")
# No SFT_CE_CHUNK: its checkpointed lm_head chunks re-gather ZeRO-3 params in backward a data-dependent number of times
# per rank, which deadlocks multi-rank runs (a run hung in backward on all 8 ranks at step 4). 80 GB fits the
# unchunked supervised logits.
NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 GPUS_PER_NODE=8 MODEL=$BASE DATA=$DATA IMAGE_ROOT=$IMAGE_ROOT C=$C \
  RUN=$RUN OUT_ROOT=$OUT_ROOT EPOCHS=${EPOCHS:-2} LR=${LR:-1e-5} WARMUP=${WARMUP:-20} MASK_P=0.2 QA_WEIGHT=0.5 \
  MAX_LEN=66048 SFT_SAVE_STEPS=${SFT_SAVE_STEPS:-100} SFT_SAMPLING=sequential bash train/launch_sft.sh
F=$OUT_ROOT/$RUN/final
for f in processor_config.json preprocessor_config.json video_preprocessor_config.json generation_config.json chat_template.jinja; do
  [ -e "$F/$f" ] || [ ! -e "$BASE/$f" ] || cp "$BASE/$f" "$F/"
done
mkdir -p "$D/sft_models" && cp -r "$F" "$D/sft_models/$RUN"
echo "$(date -u +%FT%TZ) done $RUN -> $D/sft_models/$RUN"

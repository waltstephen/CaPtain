#!/usr/bin/env bash
# Multi-node CaPtain SFT launcher (torchrun + DeepSpeed ZeRO-3). Run the SAME command on every node; node 0 is the master.
#
#   NNODES=2 NODE_RANK=<0|1> MASTER_ADDR=<node-0 address> \
#   MODEL=<init model dir> DATA=<sft_mix16.jsonl> IMAGE_ROOT=<dir the row image paths are relative to> \
#   C=<C_weighted_per_step from train/sup_tokens.py> RUN=<run name> \
#   bash train/launch_sft.sh [extra train_sft.py args]
#
# Defaults reproduce the CaPtain-4B-SFT stage-2 configuration (configs/captain-4b-sft.yaml). Environment knobs:
#   NNODES (1) NODE_RANK (0) MASTER_ADDR (127.0.0.1) MASTER_PORT (29577) GPUS_PER_NODE (= visible GPUs)
#   OUT_ROOT (./runs)  LR (1e-5)  WARMUP (30)  EPOCHS (1.5)  MAX_LEN (66048)  DS_CONFIG (train/ds_zero3.json)
#   QA_WEIGHT (0.5)  MASK_P (0.2)  SFT_SAVE_STEPS (200)  SFT_RESUME (<checkpoint dir> or 1)
#   SFT_GC_OFFLOAD (0; set 1 on 40 GB GPUs)  SFT_MEMORY_FRACTION (unset; cap per-process GPU memory share)
#   SFT_SAMPLING (sequential: keep the step order written by mix_order.py / bucket_order.py)
#   C: the constant global per-token loss normaliser (--loss-tokens-per-step). It depends on the data AND the world size:
#      recompute it with `QA_W=0.5 python train/sup_tokens.py <data> <model dir> <world size>` (C_weighted_per_step).
#   W&B logging is enabled only if WANDB_API_KEY is set in the environment.
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
: "${MODEL:?MODEL (init model dir)}" "${DATA:?DATA (training jsonl)}" "${IMAGE_ROOT:?IMAGE_ROOT}" "${C:?C (C_weighted_per_step)}"
RUN=${RUN:-captain-4b-sft}; OUT_ROOT=${OUT_ROOT:-$PWD/runs}; OUT=$OUT_ROOT/$RUN; LOGDIR=$OUT_ROOT/logs
NN=${NNODES:-1}; NR=${NODE_RANK:-0}; MA=${MASTER_ADDR:-127.0.0.1}
mkdir -p "$OUT" "$LOGDIR"
export NCCL_DEBUG=${NCCL_DEBUG:-WARN} TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export SFT_GC_OFFLOAD=${SFT_GC_OFFLOAD:-0} SFT_SAMPLING=${SFT_SAMPLING:-sequential} SFT_SAVE_STEPS=${SFT_SAVE_STEPS:-200}
NPROC=${GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}
if ls -d "$OUT"/checkpoint-* "$OUT"/final > /dev/null 2>&1 && [ -z "${SFT_RESUME:-}" ]; then
  echo "run $RUN already has outputs in $OUT (set SFT_RESUME=1 to resume)" >&2; exit 2; fi
cd "$HERE"
echo "$(date -u +%FT%TZ) SFT_START run=$RUN node_rank=$NR/$NN nproc=$NPROC master=$MA data=$DATA model=$MODEL" | tee -a "$LOGDIR/sft_runs.log"
python -m torch.distributed.run --nnodes "$NN" --node_rank "$NR" --nproc_per_node "$NPROC" --master_addr "$MA" \
  --master_port "${MASTER_PORT:-29577}" train/train_sft.py --model "$MODEL" --data "$DATA" --image-root "$IMAGE_ROOT" \
  --output-dir "$OUT" --deepspeed "${DS_CONFIG:-train/ds_zero3.json}" --max-length "${MAX_LEN:-66048}" --epochs "${EPOCHS:-1.5}" \
  --lr "${LR:-1e-5}" --per-device-batch 1 --grad-accum 1 --warmup-steps "${WARMUP:-30}" \
  --freeze-vision --mask-text-hints "${MASK_P:-0.2}" --qa-loss-weight "${QA_WEIGHT:-0.5}" --loss-tokens-per-step "$C" \
  "$@" 2>&1 | tee -a "$LOGDIR/$RUN.r$NR.log"
rc=${PIPESTATUS[0]}
echo "$(date -u +%FT%TZ) SFT_END run=$RUN node_rank=$NR rc=$rc" | tee -a "$LOGDIR/sft_runs.log"
exit $rc

#!/bin/bash
# Short SFT run on ONE GPU to measure peak memory and s/step (ZeRO-3 + CPU-offloaded Adam).
#   tools/sft_probe.sh <gpu> <model dir> <packed rows> <image root> <out dir> [epochs] [extra train_sft args]
source "$(cd "$(dirname "$0")/.." && pwd)/env.sh"
G=${1:?gpu}; M=${2:?model}; D=${3:?data}; IR=${4:?image root}; O=${5:?out}; E=${6:-0.3}; shift 6 2>/dev/null || shift $#
cd "$SFT_DIR"
export CUDA_VISIBLE_DEVICES=$G SFT_SAVE_STEPS=0 SFT_SAMPLING=random PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS=8 DS_SKIP_CUDA_CHECK=1 SFT_GC_OFFLOAD=${SFT_GC_OFFLOAD:-0} TOKENIZERS_PARALLELISM=false
exec "$SFT_VENV/bin/python" -m torch.distributed.run --nproc_per_node 1 --master_port $((29600 + G)) train/train_sft.py \
  --model "$M" --data "$D" --image-root "$IR" --output-dir "$O" --deepspeed "${DS_CONFIG:-train/ds_zero3_offload.json}" --max-length 66048 \
  --epochs "$E" --lr 1e-6 --per-device-batch 1 --grad-accum 1 --warmup-steps 1 --freeze-vision "$@"
